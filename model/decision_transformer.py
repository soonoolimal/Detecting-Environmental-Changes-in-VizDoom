import torch
import torch.nn as nn

from einops import repeat, rearrange

import transformers

from model.base import TrajectoryModel
from model.gpt2 import GPT2Model
from transformers.modeling_utils import Conv1D


class DecisionTransformer(TrajectoryModel):
    def __init__(
        self,
        encoder,
        n_actions,
        hidden_size,
        seq_len,
        max_ep_len=4096,
        **kwargs,
    ):
        """
        Decision Transformer for image observations and discrete actions.
        
        Args:
            encoder: ObsEncoder instance, (B,T,C,H,W) -> (B,T,hidden_size)
            n_actions: number of discrete actions, for embedding and prediction head
            hidden_size H: embedding dimension & d_model
            seq_len: input sequence length T, sets n_ctx = n_positions = 3 * seq_len
            max_ep_len: maximum length of single full trajectory
        """
        super().__init__(ac_dim=n_actions, max_len=None)

        # (B,T,C=3,H,W) -> (B,T,H)
        self.encoder: nn.Module = encoder  # enc_dim must equal hidden_size

        # GPT-2 backbone
        kwargs["n_ctx"] = 3 * seq_len
        kwargs["n_positions"] = 3 * seq_len
        kwargs["n_embd"] = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # unused
            **kwargs,
        )
        self.gpt2 = GPT2Model(config)
        self.config = config
        
        # timestep embedding (index within environment step)
        self.embed_tstep = nn.Embedding(max_ep_len, hidden_size)
        
        # token-level positional embedding (index within token sequence of length 3T)
        # length is upper bounded by config.n_ctx = config.n_positions
        self.pos_emb = nn.Parameter(torch.zeros(1, config.n_ctx, hidden_size))
        
        # modality-specific embeddings
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_ob = nn.Linear(hidden_size, hidden_size)
        self.embed_ac = nn.Embedding(n_actions, hidden_size)
        
        # token-level LN
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # prediction heads
        self.pred_rtg = nn.Linear(hidden_size, 1)
        self.pred_ac = nn.Linear(hidden_size, n_actions)
        
        nn.init.normal_(self.embed_ac.weight, mean=0.0, std=0.02)
    
    def forward(self, observations, actions, rewards, returns_to_go, timesteps, mask=None):
        """
        Args: components from DTDataset.__getitem__(), will be stacked into batch (B,3T,H)
            observations:  (B,T,C=3,H,W)
            actions:       (B,T,)
            returns_to_go: (B,T,1)
            timesteps:     (B,T,)
            mask:          (B,T), padding mask (for each input sequence, valid: 1, pad: 0)
        
        Returns:
            rtg_preds: (B,T,1), predicted returns given action tokens for each timestep
            ac_logits: (B,T,ac_dim), predicted action logits given observation tokens for each timestep
            ob_enc:    (B,T,H), encoded observations (reused by TaskDetector to avoid duplicate encoder call)
        """
        # encode observations
        ob_enc = self.encoder(observations)  # (B,T,H)
        
        # GPT padding mask: 1 if can be attended to, 0 if not
        B, T, _ = ob_enc.size()
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.long, device=ob_enc.device)
            
        # only covers discrete action
        if actions.dtype != torch.long:
            actions = actions.long()
        
        # embed each modality with different heads
        rtg_emb = self.embed_rtg(returns_to_go)  # (B,T,H)
        ob_emb = self.embed_ob(ob_enc)            # (B,T,H)
        ac_emb = self.embed_ac(actions)           # (B,T,H)
        time_emb = self.embed_tstep(timesteps)    # (B,T,H)
        
        # time embeddings are treated similar to positional embeddings
        rtg_emb = rtg_emb + time_emb
        ob_emb = ob_emb + time_emb
        ac_emb = ac_emb + time_emb
        
        # stack each embeddings into sequence (R_t,o_t,a_t,...)
        # which works nice in an autoregressive sense since states predict actions
        stacked = rearrange(
            torch.stack((rtg_emb, ob_emb, ac_emb), dim=2),
            "B T M H -> B (T M) H"  # M: number of modalities
        )  # (B,T,M=3,H) -> (B,3T,H)
        
        # add token positional embeddings
        seq_len = stacked.size(1)  # 3T
        if seq_len > self.pos_emb.size(1):
            raise ValueError(
                f"Stacked sequence length 3T={seq_len} exceeds config.n_ctx={self.pos_emb.size(1)}. "
                "Increase GPT2Config(n_ctx/n_positions)."
            )
        stacked = stacked + self.pos_emb[:, :seq_len, :]
        stacked = self.embed_ln(stacked)
        
        # make attention mask to (B,3T), same as stacked input
        # i.e., repeat 3 times for each modality
        stacked_mask = repeat(mask, "B T -> B (T M)", M=3)  # (B,3T)

        # feed stacked embeddings to GPT-2 block
        out = self.gpt2(stacked, stacked_mask)
        h = out["last_hidden_state"]  # (B,3T,H)
        
        # recover (0: returns, 1: observations, 2: actions)
        h = rearrange(h, "B (T M) H -> B M T H", M=3)  # (B,3,T,H)
        
        # predict R_{t+1} given h(a_t)
        rtg_preds = self.pred_rtg(h[:, 2])  # (B,T,1)
        
        # predict a_t given h(o_t)
        ac_logits = self.pred_ac(h[:, 1])   # (B,T,ac_dim)
        
        return rtg_preds, ac_logits, ob_enc
    
    def configure_optimizers(self, lr, weight_decay, betas=(0.9, 0.95)) -> torch.optim.Optimizer:
        """AdamW with GPT-style Parameter Grouping (Decay vs No-Decay)"""
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, Conv1D)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters(recurse=False):
                if not _p.requires_grad:
                    continue
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # positional parameter should not be decayed
        no_decay.add("pos_emb")

        param_dict = {k: v for k, v in self.named_parameters() if v.requires_grad}

        inter_params = decay & no_decay
        if len(inter_params) != 0:
            raise ValueError(f"Parameters present in both decay and no_decay: {sorted(inter_params)}")

        union_params = decay | no_decay
        missing = set(param_dict.keys()) - union_params
        if len(missing) != 0:
            raise ValueError(f"Parameters were not separated into decay/no_decay sets: {sorted(missing)}")

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, weight_decay=weight_decay)