import torch.nn as nn
import torch.nn.functional as F


class ObsEncBackBone(nn.Module):
    """Conv Backbone: (B,C=3,H,W) -> (B,out_channels,H',W')
    
    This backbone is shared by:
        - ObsAutoEncoder (pretraining)
        - ObsEncoder     (encoding)
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        
        mid_channels = out_channels // 2
        
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        if (x.ndim != 4) or (x.size(1) != self.in_channels):
            raise ValueError(f"Expected shape of input (B,C={self.in_channels},H,W), got {tuple(x.shape)}.")
        return self.backbone(x)


class ObsEncoder(nn.Module):
    """Observation Encoder for DT Input: (B,T,C=3,H,W) -> (B,T,enc_dim)

    This Encoder project (resized) RGB image observation into a vector of dimension enc_dim,
    uses ObsEncBackBone + GAP + MLP, so it supports arbitrary (H,W).
    
    For pretraining, copy backbone weights of pretrained ObsAutoEncoder.
    """
    def __init__(self, enc_dim: int, in_channels=3, out_channels=64):
        super().__init__()
        
        self.backbone = ObsEncBackBone(in_channels, out_channels)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        
        # projection (MLP)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, enc_dim),
        )
        
        self.in_channels = in_channels
        self.enc_dim = enc_dim
        self.hidden_size = enc_dim  # convenience alias
    
    def load_backbone_from(self, backbone: nn.Module):
        """
        Copy pretrained ObsBackBone weights into this encoder backbone.

        Expected usage:
            enc.load_backbone_from(ae.backbone)
        """
        self.backbone.load_state_dict(backbone.state_dict(), strict=True)
    
    def freeze(self, only_backbone=True):
        # freeze only conv backbone (keep projection trainable)
        if only_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # freeze entire encoder (backbone + projection)
        else:
            for p in self.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        if (x.ndim != 5) or (x.size(2) != self.in_channels):
            raise ValueError(f"Expected shape of input (B,T,C={self.in_channels},H,W), got {tuple(x.shape)}.")

        B, T, C, H, W = x.shape
        
        x = x.reshape(B*T, C, H, W)

        z = self.backbone(x)  # (BT,out_channels,H',W')
        z = self.pool(z)      # (BT,out_channels,1,1)
        z = self.proj(z)      # (BT,enc_dim)

        return z.reshape(B, T, -1)  # (B,T,enc_dim)


class ObsAutoEncoder(nn.Module):
    """AutoEncoder for Pretraining: (B,C=3,H,W) -> (B,C=3,H,W)
    
    This AE uses the same backbone as DT encoder, i.e., keeps spatial feature map of DT encoder,
    decodes with interploate + conv, so output spatial size matches input exactly.
    """
    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        
        self.backbone = ObsEncBackBone(in_channels, feat_channels)
        
        mid_channels = feat_channels // 2
        
        self.dec_conv1 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(feat_channels, mid_channels, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        
        self.in_channels = in_channels
        self.feat_channels = feat_channels

    def forward(self, x):
        if x.ndim != 4 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected (B,C={self.in_channels},H,W), got {tuple(x.shape)}")
        
        H, W = x.shape[-2], x.shape[-1]
        
        z = self.backbone(x)  # (B,feat_channels,H',W')
        
        # progressively upsample towards (H,W)
        y = F.interpolate(z, size=(max(1, H // 4), max(1, W // 4)), mode="nearest")
        y = F.relu(self.dec_conv1(y), inplace=True)

        y = F.interpolate(y, size=(max(1, H // 2), max(1, W // 2)), mode="nearest")
        y = F.relu(self.dec_conv2(y), inplace=True)

        y = F.interpolate(y, size=(H, W), mode="nearest")
        y = self.dec_conv3(y)
        
        return y
