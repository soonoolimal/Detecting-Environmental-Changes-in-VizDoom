import torch
import torch.nn as nn

from einops import repeat, rearrange

import transformers
from transformers.modeling_utils import Conv1D

from model.base import TrajectoryModel
from model.gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):
    def __init__(
        self,
        encoder: nn.Module,
        n_actions: int,
        hidden_size: int,
        seq_len: int,
        last_k: int,
        max_ep_len: int = 4096,
        **kwargs,
    ):
        """Decision Transformer as Dynamics Encoder

        Assume:
            - partial observability as image
            - discrete action space

        Args:
            encoder: observation encoder
            n_actions: number of discrete actions, for embedding and prediction head
            hidden_size H: dimension of embedding and attention block
            seq_len T: input sequence length, sets n_ctx = n_positions = 3T
            max_ep_len: maximum length of a single full trajectory
            last_k: number of last timesteps used as TD input
        """
        super().__init__(ac_dim=n_actions, max_len=max_ep_len)

        self.encoder = encoder
        self.last_k = last_k

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

        # (timestep-level) time embedding within a trajectory
        self.embed_tstep = nn.Embedding(max_ep_len, hidden_size)

        # modality-specific input embeddings
        self.embed_ac = nn.Embedding(n_actions, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_ob = nn.Linear(hidden_size, hidden_size)

        # token-level positional embedding over the stacked sequence of length 3T
        self.pos_emb = nn.Parameter(torch.zeros(1, config.n_ctx, hidden_size))

        # token-level layer normalization
        self.embed_ln = nn.LayerNorm(hidden_size)

        # prediction heads
        self.pred_ac = nn.Linear(hidden_size, n_actions)    # h(o_t) -> a_t
        self.pred_rtg = nn.Linear(hidden_size, 1)           # h(a_t) -> R_{t+1}
        self.pred_ob = nn.Linear(hidden_size, hidden_size)  # h(a_t) -> enc(o_{t+1})

        nn.init.normal_(self.embed_ac.weight, mean=0.0, std=0.02)

    def forward(self, observations, actions, rewards, returns_to_go, timesteps, mask=None):
        """
        Args: returns from E2EDataset.__getitem__(), will be stacked into batch (B,3T,H)
            observations  (B,T,C=3,H,W) in [0,1]
            actions       (B,T)
            rewards       (B,T,1)
            returns_to_go (B,T,1)
            timesteps     (B,T)
            mask          (B,T): padding pask; 1=valid, 0=pad
                all ones since all trajectories share the same fixed length

        Returns:
            for loss:
                - rtg_preds (B,T,1)
                - ob_preds  (B,T,H)
                - ac_logits (B,T,n_actions)
            feed to TD input:
                - h_k      (B,last_k,H): RTG token hidden states before pred_rtg head
                - ob_enc_k (B,last_k,H): encoded observations
        """
        B, T, C, H, W = observations.size()

        # encode observations
        ob_enc = self.encoder(
            observations.reshape(B * T, C, H, W)
        ).reshape(B, T, -1)  # (B,T,H)

        # GPT-2 padding mask
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.long, device=ob_enc.device)

        # modality embeddings
        rtg_emb = self.embed_rtg(returns_to_go)  # (B,T,H)
        ob_emb = self.embed_ob(ob_enc)           # (B,T,H)
        ac_emb = self.embed_ac(actions)          # (B,T,H)
        time_emb = self.embed_tstep(timesteps)   # (B,T,H)

        # add time embeddings to each modality (treated as positional signals)
        rtg_emb = rtg_emb + time_emb
        ob_emb = ob_emb + time_emb
        ac_emb = ac_emb + time_emb

        # stack each embeddings into sequence [R_t,o_t,a_t,...]
        # which works nice in an autoregressive sense since states predict actions
        stacked = rearrange(
            torch.stack((rtg_emb, ob_emb, ac_emb), dim=2),
            "B T M H -> B (T M) H",  # M: number of modalities
        )  # (B,T,M=3,H) -> (B,3T,H)

        # token-level positional embedding
        stacked_len = stacked.size(1)  # 3T
        if stacked_len > self.pos_emb.size(1):
            raise ValueError(
                f"Expected stacked sequence length 3T < config.n_ctx, "
                f"got 3T={stacked_len} and n_ctx={self.pos_emb.size(1)}."
                f"Increase GPT2Config(n_ctx/n_positions)."
            )
        stacked = stacked + self.pos_emb[:, :stacked_len, :]

        # layer normalization
        stacked = self.embed_ln(stacked)

        # expand padding mask same with input
        # i.e., repeat 3 times for each modality: 
        mask_3t = repeat(mask, "B T -> B (T M)", M=3)  # (B,T) -> (B,3T)

        # GPT-2 forward
        h = self.gpt2(stacked, mask_3t)["last_hidden_state"]  # (B,3T,H)

        # recover
        # modality index: rtgs=0, observations=1, actions=2
        h = rearrange(h, "B (T M) H -> B M T H", M=3)  # (B,3T,H) -> (B,M=3,T,H)

        # prediction heads
        ac_logits = self.pred_ac(h[:, 1])   # h(o_t) -> a_t          (B,T,n_actions)
        rtg_preds = self.pred_rtg(h[:, 2])  # h(a_t) -> R_{t+1}      (B,T,1)
        ob_preds  = self.pred_ob(h[:, 2])   # h(a_t) -> enc(o_{t+1}) (B,T,H)

        # TD inputs
        h_k = h[:, 0, -self.last_k:, :]         # (B,last_k,H)
        ob_enc_k = ob_enc[:, -self.last_k:, :]  # (B,last_k,H)

        return rtg_preds, ob_preds, ac_logits, h_k, ob_enc_k

    def configure_optimizers(self, lr, weight_decay, betas=(0.9, 0.95)) -> torch.optim.Optimizer:
        """AdamW with GPT-style Parameter Grouping (Decay vs No-Decay)"""
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, Conv1D)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)

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