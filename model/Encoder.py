import math
from math import exp
import torch.nn as nn
from utils import trunc_normal_
from .Differential_Transformer import DifferentialTransformerBlock, SimpleRMSNorm


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=512):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

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
            depth: int = 24,
            λinit: float = None,

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

        self.layers = nn.ModuleList(
            [
                DifferentialTransformerBlock(
                    dim=self.dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    λinit=self.λinit,
                ) for _ in range(self.depth)
            ]
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # norm
        x = self.norm(x)

        # Post embed norm
        for layer in self.layers:
            x = layer(x)
        return x


class DiffTransformerEncoder(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8], depths=[3, 4, 6, 3], drop_rate=0.1):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        self.blocks = nn.ModuleList(
            [
                DTBlock(
                    dim=embed_dims[i], heads=num_heads[i],
                    dropout=drop_rate, depth=depths[i]
                )
                for i in range(4)
            ]
        )

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]),
            OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]),
            OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]),
            OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]),
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        feature_maps = []
        B = x.shape[0]

        feature_maps.append(x)

        for i, (block, patch_embed) in enumerate(zip(self.blocks, self.patch_embeds)):
            x, H, W = patch_embed(x)
            x = block(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(x)

        return feature_maps
