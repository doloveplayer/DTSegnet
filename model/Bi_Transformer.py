import torch
import torch.nn as nn
import math
from utils import trunc_normal_
from .Differential_Transformer import SimpleRMSNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        # Multi-head attention forward pass
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class BiDirectionalAttentionModule(nn.Module):
    def __init__(self, c1, c4, num_heads, embed_dim, depth, dropout=0.1):
        super(BiDirectionalAttentionModule, self).__init__()
        self.depth = depth

        # Token projections
        self.patch_pos_embed = nn.Linear(c1, embed_dim)
        self.p4_embed = nn.Linear(c4, embed_dim)

        # Attention modules
        self.self_attention_p4 = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention_p4_to_p1 = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention_p1_to_p4 = MultiHeadAttention(embed_dim, num_heads)

        # MLP layers
        self.mlp1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Linear(embed_dim * 2, c1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)  # Dropout after MLP
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Linear(embed_dim * 2, c4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout)  # Dropout after MLP
        )

        # Normalization
        self.norm = SimpleRMSNorm(c4)

    def forward(self, P1, P4, pos_embed):
        b, c1, h1, w1 = P1.shape
        b, c4, h4, w4 = P4.shape

        P1 = P1 + pos_embed

        # Flatten and project
        P1_tokens = P1.flatten(2).permute(0, 2, 1)  # [b, H/4*W/4, C1]
        P4_tokens = P4.flatten(2).permute(0, 2, 1)  # [b, H/32*W/32, C4]

        P1_embedded = self.patch_pos_embed(P1_tokens)  # [b, H/4*W/4, embed_dim]
        P4_embedded = self.p4_embed(P4_tokens)  # [b, H/32*W/32, embed_dim]

        for _ in range(self.depth):
            # Self attention for P4
            P1_embedded_norm = self.norm(P1_embedded)
            P1_self_attn = self.self_attention_p4(P1_embedded_norm, P1_embedded_norm, P1_embedded_norm) + P1_embedded

            # Cross attention: P4 -> P1
            P1_to_P4_attn = self.cross_attention_p4_to_p1(
                P1_self_attn, P4_embedded, P4_embedded
            )

            # Cross attention: P1 -> P4
            P4_to_P1_attn = self.cross_attention_p1_to_p4(
                P4_embedded, P1_to_P4_attn, P1_to_P4_attn
            )

            P1_embedded = P1_to_P4_attn
            P4_embedded = P4_to_P1_attn

        # Final MLP and reshape
        P1_ = self.mlp1(P1_embedded).view(b, h1, w1, -1).permute(0, 3, 1, 2).contiguous()
        P4_ = self.mlp4(P4_embedded).view(b, h4, w4, -1).permute(0, 3, 1, 2).contiguous()

        return P1_, P4_


# 测试代码
if __name__ == "__main__":
    # 输入数据
    P1 = torch.rand(2, 64, 64, 64)  # [b=2, C1=64, H/4=64, W/4=64]
    P4 = torch.rand(2, 256, 8, 8)  # [b=2, C4=256, H/32=8, W/32=8]
    pos_embed = torch.rand(2, 64, 64, 64)  # [b=2, C1=64, H/4=64, W/4=64]

    # 初始化模块
    bi_attention = BiDirectionalAttentionModule(
        c1=64, c4=256, num_heads=4, mlp_dim=512, depth=3, dropout=0.1
    )

    # 前向传播
    p1_attention, P4_tokens = bi_attention(P1, P4, pos_embed)
    print("P1 Attention 形状:", p1_attention.shape)  # [b, H/4*W/4, C4]
    print("P4 Tokens 形状:", P4_tokens.shape)  # [b, H/32*W/32, C4]
