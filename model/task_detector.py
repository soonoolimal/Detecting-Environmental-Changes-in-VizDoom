from typing import Optional

import torch
import torch.nn as nn


class TaskDetector(nn.Module):
    """
    Transfer learning wrapper on top of pretrained DT.
    
    1. Runs frozen DT to get:
        - ob_preds:  (B,T,H)
        - rtg_preds: (B,T,1)
    2. Projects ob_preds and rtg_preds.
    3. Predicts 3-class task logits e_t:
        e_t = Classifier(z_rtg_preds, z_ob_preds)
            where class 0: vanilla, 1: observation-shifted, 2: reward-shifted
    """
    def __init__(
        self,
        dt: nn.Module,
        ob_pred_dim: int,
        proj_dim: int = 128,
        num_classes: int = 3,
    ):
        """
        Args:
            dt: pretrained DT
            ob_pred_dim: embedding dimension of DT (hidden_size H)
            proj_dim P: projection dimension
        """
        super().__init__()

        self.dt = dt

        for p in self.dt.parameters():
            p.requires_grad = False
        self.dt.eval()

        # projection layers
        self.proj_ob = nn.Sequential(
            nn.LayerNorm(ob_pred_dim),
            nn.Linear(ob_pred_dim, proj_dim),
            nn.GELU(),
        )
        self.proj_rtg = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, proj_dim),
            nn.GELU(),
        )

        # task classifier: outputs num_classes logits per timestep
        self.task_classifier = nn.Sequential(
            nn.LayerNorm(2 * proj_dim),
            nn.Linear(2 * proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, num_classes),
        )
        
        self.ob_pred_dim = ob_pred_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes

    def train(self, mode: bool = True):
        """
        Override train() to keep backbone permanently in eval mode.
        """
        super().train(mode)
        self.dt.eval()
        return self

    def forward(
        self,
        observations,
        actions,
        returns_to_go,
        timesteps,
        mask=None,
        **dt_kwargs,
    ):
        """
        Args:
            observations:  (B,T,C=3,H,W)
            actions:       (B,T)
            returns_to_go: (B,T,1)
            timesteps:     (B,T)
            mask:          (B,T)

        Returns:
            logits: (B,T,num_classes)
        """
        # self.dt is frozen (requires_grad=False) and permanently kept in eval()
        # via train() override, so torch.no_grad() is redundant here
        # with torch.no_grad():
        rtg_preds, ob_preds, _ = self.dt(
            observations=observations,
            actions=actions,
            rewards=torch.zeros_like(returns_to_go),  # dummies
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            mask=mask,
            **dt_kwargs,
        )

        if ob_preds.size(-1) != self.ob_pred_dim:
            raise ValueError(
                f"Expected `ob_preds` last dim {self.ob_pred_dim}, "
                f"got {ob_preds.size(-1)}"
            )
        if rtg_preds.size(-1) != 1:
            raise ValueError(
                f"Expected `rtg_preds` last dim 1, got {rtg_preds.size(-1)}"
            )

        z_ob = self.proj_ob(ob_preds)     # (B,T,P)
        z_rtg = self.proj_rtg(rtg_preds)  # (B,T,P)
        
        logits = self.task_classifier(
            torch.cat([z_rtg, z_ob], dim=-1)  # (B,T,2P)
        )

        return logits  # (B,T,num_classes)

    def predict_shift(
        self,
        batch: dict,
        device,
        omega: float = 0.5,
    ) -> bool:
        """
        Run forward on a single batch and return shift flag.
        """
        self.eval()
        
        with torch.no_grad():
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device).long()
            returns_to_go = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device).long()

            logits = self(
                observations=observations,
                actions=actions,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                mask=mask,
            )  # (B,T,num_classes)

        return self.detect_shift(
            logits=logits,
            mask=mask,
            omega=omega,
        )

    @staticmethod
    def detect_shift(
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        omega: float = 0.5,
    ) -> bool:
        """
        Compute a single shift flag over the whole batch:
            1. pred_t = argmax(logits_t) (per timestep)
            2. e_t = 1[pred_t != 0] (non-zero classes mean task is shifted)
            3. frac = mean(e_t) over valid timesteps
            4. flag = 1[frac >= omega]

        Args:
            logits: raw logits from forward()
            omega: fraction threshold to declare shift

        Returns:
            bool: True if task shift is detected
        """
        preds = logits.argmax(dim=-1)             # (B,T,num_classes) -> (B,T)
        e = (preds != 0).to(dtype=torch.float32)  # (B,T)

        if mask is not None:
            if mask.shape != e.shape:
                raise ValueError(
                    f"Attention mask must have shape (B,T)={tuple(e.shape)}, got {tuple(mask.shape)}."
                )
            mask = mask.to(dtype=torch.float32)
        else:
            mask = torch.ones_like(e)

        valid_timesteps = mask.sum()
        if valid_timesteps.item() == 0:
            return False

        frac = (e * mask).sum() / valid_timesteps
        
        return bool((frac >= omega).item())