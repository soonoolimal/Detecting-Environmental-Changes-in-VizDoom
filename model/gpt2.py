"""
PyTorch OpenAI GPT-2 model as causal sequence model, not language model.
"""


import torch
import torch.nn as nn

from einops import rearrange

from transformers.activations import ACT2FN
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import Conv1D, PreTrainedModel


class CausalSelfAttention(nn.Module):
    """Causal Multi Head Self Attention"""
    def __init__(self, config, scale=True):
        """
        Args:
            config: GPT-2 hyperparameters:
                - n_embd: dimension of attention block, a.k.a. d_model
                - n_ctx: 3T <= n_ctx = n_positions
                    where T is length of single input sequence
                    i.e., 3T is length of each input stacked sequence
                - n_head: number of heads
            scale: whether scale attention weights
        """
        super().__init__()
        
        n_embd = config.n_embd
        n_ctx = config.n_ctx
        n_head = config.n_head
        
        # n_embd = n_head * d_head
        if n_embd % n_head != 0:
            raise ValueError(
                f"Expected d_model % n_head = 0 for multiple heads, got d_model={n_embd} and n_head={n_head}."
            )
        
        # causal mask (lower triangular matrix)
        self.bias: torch.Tensor
        self.register_buffer(
            name="bias", tensor=torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.bool)).view(1, 1, n_ctx, n_ctx)
        )
        self.masked_bias: torch.Tensor  # get ready for softmax
        self.register_buffer(
            name="masked_bias", tensor=torch.tensor(-1e4)
        )
        
        # layers
        self.c_attn = Conv1D(3 * n_embd, n_embd)  # for copying input into q, k and v
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.c_proj = Conv1D(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # configs
        self.n_embd = n_embd
        self.n_ctx = n_ctx
        self.n_head = n_head
        
        self.scale = scale
    
    def forward(self, stacked, mask=None, layer_past=None, use_cache=False):
        """
        Args:
            stacked (B,3T,n_embd)
            layers_past: cached past keys and values
            use_cache: whether to cache past key/value states
        
        Returns:
            list: [attn_z,present]
                - attn_z  (B,3T,H): hidden states computed with causal multi-head self attention
                - present (2,B,nh,past+1(cur),dh): current key and value
        """
        query, key, value = self.c_attn(stacked).split(self.n_embd, dim=2)  # (B,3T,3H) -> (B,3T,H) * 3
        
        # split heads: n_embd = n_head * d_head
        query = rearrange(query, "B tT (nh dh) -> B nh tT dh", nh=self.n_head)
        key = rearrange(key, "B tT (nh dh) -> B nh dh tT", nh=self.n_head)
        value = rearrange(value, "B tT (nh dh) -> B nh tT dh", nh=self.n_head)
        
        # in standard autoregressive inference,
        # tokens are generated one at a time and caching avoids recomputing prior states
        # but here in our research, training, evaluation and inference use offline batch data
        # so the full sequence is always available upfront
        # that is, use_cache=False is appropriate in all cases
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # (B,nh,dh,past), (B,nh,past,dh)
            key = torch.cat((past_key, key), dim=-1)                               # (B,nh,dh,past+1)
            value = torch.cat((past_value, value), dim=-2)                         # (B,nh,past+1,dh)
        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))                  # (2,B,nh,past+1,dh)
        else:
            present = (None,)
        
        # attention
        attn_output = self._attn(query, key, value, mask)  # (B,nh,3T,dh)
        
        # merge heads
        attn_output = rearrange(attn_output, "B nh tT dh -> B tT (nh dh)")  # (B,3T,H)
        
        # projection
        attn_z = self.c_proj(attn_output)  # (B,3T,H)
        
        return [self.resid_dropout(attn_z), present]  # for upcomming residual connection
    
    def _attn(self, q, k, v, mask=None):
        """MHA(Q, K, V) = softmax(Q @ K' / sqrt{d_head} + M) @ V
        
        Args:
            q    (B,nh,3T,dh): query
            k    (B,nh,dh,3T): key
            v    (B,nh,3T,dh): value
            mask (B,1,1,3T): padding mask; (1,1) for broadcasting
                padding matters if
                    - length of full trajectories are different
                    - batch is consist of trajectories shorter than window size
                that is, no padding needed in all cases
        
        Returns:
            attn_output (B,nh,3T,dh): hidden states
        
        Variables:
            nd: number of query tokens
            ns: number of key tokens
        """
        # get attention weights: Q @ K'
        w = torch.matmul(q, k)  # (B,nh,3T,3T)
        
        # scale: ÷ sqrt{d_head}
        if self.scale:
            w /= (float(q.size(-1)) ** 0.5)

        # apply causal mask: + M
        nd, ns = w.size(-2), w.size(-1)                                  # query/key is row/column of attention weight matrix
        c_mask = self.bias[:, :, ns - nd:ns, :ns]                        # causal masking
        w = torch.where(c_mask.bool(), w, self.masked_bias.to(w.dtype))  # causal masked attention weights
        
        # apply padding mask: + M
        if mask is not None:
            w += mask  # broadcasting: (B,1,1,3T) -> (B,nh,3T,3T)

        w = nn.Softmax(dim=-1)(w)    
        w = self.attn_dropout(w)

        # weight values: @ V
        attn_output = torch.matmul(w, v)  # (B,nh,3T,dh)
        
        return attn_output


class FeedForward(nn.Module):
    """Feed Forward Layer of GPT-2"""
    def __init__(self, config):
        super().__init__()
        
        d_model = config.n_embd
        d_ff = 4 * d_model if config.n_inner is None else config.n_inner  # d_ff = 4 * d_model
        
        self.c_fc = Conv1D(d_ff, d_model)
        self.actvn = ACT2FN[config.activation_function]
        self.c_proj = Conv1D(d_model, d_ff)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x):           
        x = self.c_fc(x)        # (B,3T,d_ff)
        h = self.actvn(x)       # (B,3T,d_ff)
        h = self.c_proj(h)      # (B,3T,H)
        return self.dropout(h)  # (B,3T,H)


class GPTBlock(nn.Module):
    def __init__(self, config, scale=True):
        super().__init__()
        
        hidden_size = config.n_embd
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ff = FeedForward(config)
        
    def forward(self, stacked, mask=None, layer_past=None, use_cache=False):
        """
        Stacked Embeddings
                ↓
        Layer Normalization
                ↓
        Causal Multi Head Self Attention
                ↓
        Residual Connection
                ↓
        Layer Normalization
                ↓
        Feed Forwarding
                ↓
        Residual Connection
        """
        # causal multi-head self attention
        attn_outputs = self.attn(
            stacked=self.ln_1(stacked), mask=mask, layer_past=layer_past, use_cache=use_cache
        )  # [attn_z,present]
        attn_z = attn_outputs[0]  # (B,3T,H)
        
        # residual connection
        hidden_states = attn_z + stacked  # (B,3T,H)
        
        # feed forwarding
        ff_hidden_states = self.ff(self.ln_2(hidden_states))  # (B,3T,H)
        
        # residual connection
        hidden_states = ff_hidden_states + hidden_states  # (B,3T,H)
        
        return [hidden_states, attn_outputs[1]]


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """
    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module):
        # Initialize the weights.
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPTBlock(config, scale=True) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        self.init_weights()
    
    def forward(self, stacked, mask=None, past_key_values=None, use_cache=None, return_dict=True):
        """
        Args:
            stacked (B,3T,n_embd): stacked embeddings
            mask    (B,3T): duplicated-stacked padding mask

        Returns:
            last_hidden_state (B,3T,H)
            past_key_values   (2,B,nh,past+1(cur),dh): past-current key and values
        """
        if past_key_values is None:
            past_key_values = [None] * len(self.h)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        presents = () if use_cache else None
        
        if mask is not None:
            mask = mask[:, None, None, :]  # (B,1,1,3T)
            mask = mask.to(dtype=self.dtype)
            mask = (1.0 - mask) * -1e4
        
        hidden_states = self.drop(stacked)
        for block, layer_past in zip(self.h, past_key_values):
            outputs = block(
                stacked=hidden_states,
                mask=mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            hidden_states, present = outputs
            
            if use_cache is True:
                assert presents is not None
                presents = presents + (present,)
        
        hidden_states = self.ln_f(hidden_states)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, presents] if v is not None)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
        )