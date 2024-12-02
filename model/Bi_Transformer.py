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
    def __init__(self, c1, c4, num_heads, mlp_dim, depth, dropout=0.1):
        super(BiDirectionalAttentionModule, self).__init__()
        self.depth = depth

        # Token projections
        self.token_projection_p1 = nn.Linear(c1, c4)
        self.token_projection_pos = nn.Linear(c1, c4)

        # Attention modules
        self.self_attention_p4 = MultiHeadAttention(c4, num_heads)
        self.cross_attention_p4_to_p1 = MultiHeadAttention(c4, num_heads)
        self.cross_attention_p1_to_p4 = MultiHeadAttention(c4, num_heads)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(c4, mlp_dim),
            nn.SELU(),
            nn.Linear(mlp_dim, c4),
            nn.Dropout(dropout)  # Dropout after MLP
        )
        self.mlp_p1 = nn.Sequential(
            nn.Linear(c4, mlp_dim),
            nn.SELU(),
            nn.Linear(mlp_dim, c4),
            nn.Dropout(dropout)  # Dropout after MLP
        )
        self.mlp_p4 = nn.Sequential(
            nn.Linear(c4, mlp_dim),
            nn.SELU(),
            nn.Linear(mlp_dim, c4),
            nn.Dropout(dropout)  # Dropout after MLP
        )

        # Normalization
        self.norm = SimpleRMSNorm(c4)

        # Dropout after Attention output
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, P1, P4, pos_embed):
        b, c1, h1, w1 = P1.shape
        b, c4, h4, w4 = P4.shape

        # Flatten and project
        P1_tokens = P1.flatten(2).permute(0, 2, 1)  # [b, H/4*W/4, C1]
        P4_tokens = P4.flatten(2).permute(0, 2, 1)  # [b, H/32*W/32, C4]
        pos_tokens = pos_embed.flatten(2).permute(0, 2, 1)  # [b, H/4*W/4, C1]

        P1_embedded = self.token_projection_p1(P1_tokens)
        pos_embedded = self.token_projection_pos(pos_tokens)

        for _ in range(self.depth):
            # Self attention for P4
            P4_self_attn = self.norm(self.self_attention_p4(P4_tokens, P4_tokens, P4_tokens)) + P4_tokens
            P4_self_attn = self.attn_dropout(P4_self_attn)  # Dropout after attention

            # Cross attention: P4 -> P1
            P4_to_P1_attn = self.cross_attention_p4_to_p1(
                P4_self_attn, P1_embedded + pos_embedded, P1_embedded + pos_embedded
            )
            P4_to_P1_attn = self.mlp(P4_to_P1_attn) + P4_to_P1_attn

            # Cross attention: P1 -> P4
            P1_to_P4_attn = self.cross_attention_p1_to_p4(
                P1_embedded + pos_embedded, P4_to_P1_attn, P4_to_P1_attn
            )
            P1_to_P4_attn = self.norm(P1_to_P4_attn) + P1_embedded
            P1_to_P4_attn = self.attn_dropout(P1_to_P4_attn)  # Dropout after attention

            P4_tokens = P4_to_P1_attn

        # Final MLP and reshape
        p1_attention = self.mlp_p1(P1_to_P4_attn).view(b, h1, w1, -1).permute(0, 3, 1, 2).contiguous()
        P4_tokens = self.mlp_p4(P4_tokens).view(b, h4, w4, -1).permute(0, 3, 1, 2).contiguous()

        return p1_attention, P4_tokens


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
