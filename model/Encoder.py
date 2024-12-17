import math
from math import exp
import torch.nn as nn
from .attention import AttentionBlock
from .Differential_Transformer import DifferentialTransformerBlock, SimpleRMSNorm


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=512):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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
            d: int,
            dim: int,
            heads: int,
            dropout: float,
            depth: int,
    ):
        """
        Initializes the Differential Transformer Block.
        """
        super(DTBlock, self).__init__()
        self.d = d
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth
        self.lambda_init = 0.8 - 0.6 * exp(-0.3 * (depth - 1))

        self.layers = nn.ModuleList(
            [
                DifferentialTransformerBlock(
                    d=d,
                    embedding_dim=self.dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    lambda_init=self.lambda_init,
                ) for _ in range(self.depth)
            ]
        )

    def forward(self, x):
        # Post embed norm
        for layer in self.layers:
            x = layer(x)
        return x

class DiffTransformerEncoder(nn.Module):
    def __init__(self, in_chans, embed_dims, num_heads, depths, drop_rate):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        self.blocks = nn.ModuleList(
            [
                DTBlock(
                    d=embed_dims[i],
                    dim=512, heads=num_heads[i],
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



class AttentionBlockLayer(nn.Module):
    """
    This class implements a Transformer Block with standard multi-head attention.
    """
    def __init__(self, dim, heads, dropout, depth):
        """
        Initializes the Attention Block Layer.
        """
        super(AttentionBlockLayer, self).__init__()
        self.depth = depth

        self.layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout
                ) for _ in range(depth)
            ]
        )

    def forward(self, x):
        """
        Forward pass through all attention layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, in_chans, embed_dims, num_heads, depths, drop_rate):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        self.blocks = nn.ModuleList(
            [
                AttentionBlockLayer(
                    dim=embed_dims[i],
                    heads=num_heads[i],
                    dropout=drop_rate,
                    depth=depths[i]
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
