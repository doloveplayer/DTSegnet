import torch
import torch.nn as nn
from math import sqrt, exp
from Differential_Transformer import DifferentialTransformerBlock,SimpleRMSNorm
class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size  = (patch_size, patch_size)
        self.proj   = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm   = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B,H*W,C--> B,128*128,768
        x = self.norm(x)

        return x, H, W

class DTBlock(nn.Module):
    """
    This class implements a Differential Transformer Block.
    """

    def __init__(
            self,
            dim: int = 512,
            heads: int = 12,
            dropout: float = 0.1,
            λinit: float = None,
            depth: int = 24,
    ):
        """
        Initializes the Differential Transformer Block.
        """
        super(DTBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth
        self.λinit = λinit if λinit is not None else (0.8 - 0.6 * exp(-0.3 * (depth - 1)))
        self.norm = SimpleRMSNorm(dim)
        self.patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=dim)

        self.layers = nn.ModuleList(
            [
                DifferentialTransformerBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    λinit=λinit,
                ) for _ in range(depth)
            ]
        )

    def forward(self, x):
        # [b,3,h,w]
        B = x.shape[0]
        # [b,h/4 * w/4, 3]
        x, H, W = self.patch_embed.forward(x)
        # norm
        x = self.norm(x)

        # Post embed norm
        for layer in self.layers:
            x = layer(x)
        # [b,h/4 * w/4, 3]
        return x

