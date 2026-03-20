from torch import nn


class ObsEncoder(nn.Module):
    """
    Encode regularized RGB image observation into the vector of enc_dim.
    """
    def __init__(
        self,
        enc_dim: int,
        in_channels: int = 3,
        out_channels: int = 256,
        num_layers: int = 4,
    ):
        """
        Args:
            enc_dim: projection dimension
                should match the hidden_size of DT,
                so the encoded observation can be directly used as the token embedding
        """
        
        super().__init__()

        if out_channels % (2 ** (num_layers - 1)) != 0:
            raise ValueError(
                f"Expected out_channels divisible by 2^(num_layers-1), "
                f"got out_channels={out_channels} and 2^(num_layers-1)={2 ** (num_layers - 1)}."
            )

        self.enc_dim = enc_dim

        channels = [out_channels // (2 ** (num_layers - 1 - i)) for i in range(num_layers)]
        self.cnn = nn.Sequential(
            *[_make_conv_block(in_ch, out_ch) for in_ch, out_ch in zip([in_channels] + channels[:-1], channels)]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(out_channels, enc_dim)

    def forward(self, observations):
        """
        Args:
            observations (N,C=3,H=84,W=84) in [0,1]
        
        Returns:
            projections (N,enc_dim)
        """
        z = self.cnn(observations)  # (N,out_channels,H',W')
        z = self.pool(z)            # (N,out_channels,1,1)
        z = z.flatten(start_dim=1)  # (N,out_channels)
        z = self.proj(z)            # (N,enc_dim)
        
        return z


def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )