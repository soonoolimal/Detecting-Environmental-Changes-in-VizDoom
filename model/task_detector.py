import torch
import torch.nn as nn


class TaskDetector(nn.Module):
    """
    3-class task detector on top of frozen dynamics encoder; DT.

    Inputs (from DT forward, last k timesteps):
        - h_k: RTG token hidden states before prediction head
        - ob_enc_k: encoded observations
    
    Inputs are concatenated and flattened,
    then passed through an MLP to produce 3-class logits.
    
    Classes:
        0: vanilla
        1: observation-shifted
        2: reward-shifted
    """

    def __init__(
        self,
        hidden_size: int,  # must match hidden_size H of DT
        last_k: int,
        proj_dim: int = 256,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()

        in_dim = last_k * 2 * hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, n_classes),
        )

    def forward(self, h_k, ob_enc_k):
        """
        Args:
            h_k      (B,last_k,H)
            ob_enc_k (B,last_k,H)

        Returns:
            logits (B,n_classes)
        """
        x = torch.cat([h_k, ob_enc_k], dim=-1)  # (B,last_k,2H)
        x = x.flatten(start_dim=1)              # (B,last_k*2H)
        return self.mlp(x)                      # (B,n_classes)