"""
PyTorch OpenAI GPT-2 model:
    Stacked Embedded Input
            ↓
    Layer Normalization
            ↓
    Causal Multi-Head Self Attention
            ↓
    Residual Connection
            ↓
    Layer Normalization
            ↓
    Feed Forwarding
            ↓
    Residual Connection
as a causal sequence model, not a language model.
"""

import torch
import torch.nn as nn

from einops import rearrange

from transformers.activations import ACT2FN
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import Conv1D, PreTrainedModel


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self Attention"""
    def __init__(self, config, scale=True):
        """
        config: GPT-2 hyperparameters
            - n_embd: dimension of attention block (d_model)
            - n_ctx: 3T <= n_ctx = n_positions
                where T is length(timesteps) of input sequence(s),
                i.e., 3T is length of each stacked input sequence
        """
        super().__init__()
        
        n_embd = config.n_embd
        n_ctx = config.n_ctx
        n_head = config.n_head
        
        assert n_embd % n_head == 0  # n_embd = n_head * d_head
        
        # causal mask (lower triangular matrix)
        self.bias: torch.Tensor
        self.register_buffer(
            name="bias", tensor=torch.tril(
                torch.ones((n_ctx, n_ctx), dtype=torch.bool)
            ).view(1, 1, n_ctx, n_ctx)
        )
        self.masked_bias: torch.Tensor  # get ready for mask bias via softmax
        self.register_buffer(
            name="masked_bias", tensor=torch.tensor(-1e4)
        )
        
        self.scale = scale
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_ctx = n_ctx

        # layers
        self.c_attn = Conv1D(3 * n_embd, n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
    
    def attn(self, q, k, v, mask=None):
        """MHA(Q, K, V) = softamx(Q @ K' / sqrt{d_head} + M) @ V
        
        Args:
            q: query  (B,nh,3T,dh)
            k: key    (B,nh,dh,3T)
            v: value  (B,nh,3T,dh)
            mask: padding mask, (B,1,1,3T) for broadcasting
                padding matters if
                    - length of full trajectories are different
                    - batch is consist of trajectories shorter than context window size
        
        Variables:
            nd: number of query tokens (number of destination positions)
            ns: number of key tokens (number of source positions)
        """
        w = torch.matmul (q, k)  # (B,nh,3T,3T)
        
        if self.scale:
            w /= (float(v.size(-1)) ** 0.5)
        
        # apply causal mask
        nd, ns = w.size(-2), w.size(-1)                                  # query/key is row/column of qv matrix
        c_mask = self.bias[:, :, ns - nd:ns, :ns]                        # causal masking
        w = torch.where(c_mask.bool(), w, self.masked_bias.to(w.dtype))  # causal masked attention scores
        
        # apply padding mask
        if mask is not None:
            w += mask  # broadcasting: (B,1,1,3T) -> (B,nh,3T,3T)
        
        w = nn.Softmax(dim=-1)(w)    
        w = self.attn_dropout(w)

        return torch.matmul(w, v)  # (B,nh,3T,dh)
    
    def forward(
        self,
        stacked,
        mask=None,
        layer_past=None,
        use_cache=False,
    ):
        """
        Args:
            stacked: (B,3T,n_embd)
            layers_past: cached past keys and values
        
        Returns:
            list: [attn_z,present]
                - attn_z: hidden states computed with causal multi-head self attention, (B,3T,H)
                - present: current key and value, (2,B,nh,past+1(cur),dh)
        """
        query, key, value = self.c_attn(stacked).split(self.n_embd, dim=2)  # (B,3T,3H) -> (B,3T,H) * 3
        
        # split heads: n_embd = n_head * d_head
        query = rearrange(query, "B tT (nh dh) -> B nh tT dh", nh=self.n_head)
        key = rearrange(key, "B tT (nh dh) -> B nh dh tT", nh=self.n_head)
        value = rearrange(value, "B tT (nh dh) -> B nh tT dh", nh=self.n_head)
        
        # for inference: use cached past keys and values
        # since keys and values are not computed in parallel (unlike training, where input is provided as batch)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # (B,nh,dh,past), (B,nh,past,dh)
            key = torch.cat((past_key, key), dim=-1)                               # (B,nh,dh,past+1)
            value = torch.cat((past_value, value), dim=-2)                         # (B,nh,past+1,dh)
        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))                  # (2,B,nh,past+1,dh)
        else:
            present = (None,)
        
        # attention  
        attn_output = self.attn(query, key, value, mask)  # (B,nh,3T,dh)
        
        # merge heads
        attn_output = rearrange(attn_output, "B nh tT dh -> B tT (nh dh)")  # (B,3T,H)
        
        # projection
        attn_z = self.c_proj(attn_output)  # (B,3T,H)
        
        return [self.resid_dropout(attn_z), present]  # for upcomming residual connection


class FeedForward(nn.Module):
    """Feed Forward Layer of GPT-2"""
    def __init__(self, config):
        super().__init__()
        
        d_model = config.n_embd
        d_ff = 4 * d_model if config.n_inner is None else config.n_inner  # d_ff = 4 * d_model
        
        self.c_fc = Conv1D(d_ff, d_model)
        self.act = ACT2FN[config.activation_function]
        self.c_proj = Conv1D(d_model, d_ff)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x):           # (B,3T,H)
        h = self.act(self.c_fc(x))  # (B,3T,d_ff)
        h = self.c_proj(h)          # (B,3T,H)
        return self.dropout(h)      # (B,3T,H)


class GPTBlock(nn.Module):
    def __init__(self, config, scale: bool = True):
        super().__init__()
        
        hidden_size = config.n_embd
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)
        
    def forward(
        self,
        stacked,
        mask=None,
        layer_past=None,
        use_cache=False,
    ):
        """
        Stacked Embedded Input
                ↓
        Layer Normalization
                ↓
        Causal Multi-Head Self Attention
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
            stacked=self.ln_1(stacked),
            mask=mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )  # [attn_z,present]
        attn_z = attn_outputs[0]  # (B,3T,H)
        
        # residual connection
        hidden_states = attn_z + stacked  # (B,3T,H)
        
        # feed forwarding
        ff_hidden_states = self.mlp(self.ln_2(hidden_states))  # (B,3T,H)
        
        # residual connection
        hidden_states = ff_hidden_states + hidden_states  # (B,3T,H)
        
        return [hidden_states, attn_outputs[1]]


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization
    and a simple interface for downloading and loading pretrained models.
    """
    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module):
        """
        Initialize the weights.
        """
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
        
    def forward(
        self,
        stacked,
        mask=None,
        past_key_values=None,
        use_cache=None,
        return_dict=True,
    ):
        """
        Args:
            stacked: stacked embedded input, (B,3T,n_embd)
            mask: stacked(duplicated) attention mask, (B,3T)
            
        Returns:
            last_hidden_state: (B,3T,H)
            past_key_values: past-current key and values, (2,B,nh,past+1(cur),dh)
        """
        batch_size = stacked.shape[0]
        assert batch_size > 0, "batch_size has to be defined and > 0"
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if past_key_values is None:
            past_key_values = [None] * len(self.h)
        presents = () if use_cache else None
            
        # Attention Mask
        if mask is not None:
            mask = mask[:, None, None, :]     # (B,1,1,3T)
            mask = mask.to(dtype=self.dtype)  # fp16 compatibility
            mask = (1.0 - mask) * -1e4
            
        hidden_states = self.drop(stacked)
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
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