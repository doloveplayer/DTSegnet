import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        # Multi-head attention forward pass
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class BiDirectionalAttentionModule(nn.Module):
    def __init__(self, c1, c4, num_heads, mlp_dim, depth):
        """
        双向注意力模块
        :param c1: P1 的通道数
        :param c4: P4 的通道数
        :param num_heads: 多头注意力的头数
        :param mlp_dim: MLP 的隐藏层维度
        :param depth: 双向注意力的循环次数
        """
        super(BiDirectionalAttentionModule, self).__init__()
        self.depth = depth

        # 投影 P1 和 pos_embed 到 P4 的通道数
        self.token_projection_p1 = nn.Linear(c1, c4)
        self.token_projection_pos = nn.Linear(c1, c4)

        # 自注意力（用于 P4）
        self.self_attention_p4 = MultiHeadAttention(c4, num_heads)

        # 交叉注意力
        self.cross_attention_p4_to_p1 = MultiHeadAttention(c4, num_heads)  # P4 -> P1
        self.cross_attention_p1_to_p4 = MultiHeadAttention(c4, num_heads)  # P1 -> P4

        # MLP 用于嵌入变换
        self.mlp = nn.Sequential(
            nn.Linear(c4, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, c4),
        )

    def forward(self, P1, P4, pos_embed):
        """
        前向传播
        :param P1: 特征图 P1，形状为 [b, C1, H/4, W/4]
        :param P4: 特征图 P4，形状为 [b, C4, H/32, W/32]
        :param pos_embed: 位置嵌入，形状为 [b, C1, H/4, W/4]
        :return: p1_attention 和 P4_tokens
        """
        # 获取形状
        b, c1, h1, w1 = P1.shape
        b, c4, h4, w4 = P4.shape

        # Token 化
        P1_tokens = P1.flatten(2).permute(0, 2, 1)  # [b, H/4*W/4, C1]
        P4_tokens = P4.flatten(2).permute(0, 2, 1)  # [b, H/32*W/32, C4]
        pos_tokens = pos_embed.flatten(2).permute(0, 2, 1)  # [b, H/4*W/4, C1]

        # 投影 P1 和 pos_embed 到与 P4 对齐的通道数
        P1_embedded = self.token_projection_p1(P1_tokens)  # [b, H/4*W/4, C4]
        pos_embedded = self.token_projection_pos(pos_tokens)  # [b, H/4*W/4, C4]

        # 初始化变量
        p1_attention = None

        # 双向注意力模块的循环
        for _ in range(self.depth):
            # Step 1: 对 P4 执行自注意力
            P4_self_attn = self.self_attention_p4(P4_tokens, P4_tokens, P4_tokens)

            # Step 2: 交叉注意力（P4 -> P1）
            cross_attn_p4_to_p1 = self.cross_attention_p4_to_p1(
                P4_self_attn, P1_embedded + pos_embedded, P1_embedded + pos_embedded
            )

            # 对 P4 -> P1 的交叉注意力结果通过 MLP 变换
            cross_attn_p4_transformed = self.mlp(cross_attn_p4_to_p1)

            # Step 3: 交叉注意力（P1 -> P4）
            cross_attn_p1_to_p4 = self.cross_attention_p1_to_p4(
                P1_embedded + pos_embedded, cross_attn_p4_transformed, cross_attn_p4_transformed
            )

            # 更新 P4 tokens
            P4_tokens = cross_attn_p4_transformed
            p1_attention = cross_attn_p1_to_p4  # 保存 P4 -> P1 的交叉注意力结果

        # [b, C4, H/4, W/4]、[b, C4, H/32, W/32]
        return p1_attention.reshape(b, h1, w1, -1).permute(0, 3, 1, 2).contiguous(),\
               P4_tokens.reshape(b, h4, w4, -1).permute(0, 3, 1, 2).contiguous()


# 测试代码
if __name__ == "__main__":
    # 输入数据
    P1 = torch.rand(2, 64, 64, 64)  # [b=2, C1=64, H/4=64, W/4=64]
    P4 = torch.rand(2, 256, 8, 8)   # [b=2, C4=256, H/32=8, W/32=8]
    pos_embed = torch.rand(2, 64, 64, 64)  # [b=2, C1=64, H/4=64, W/4=64]

    # 初始化模块
    bi_attention = BiDirectionalAttentionModule(
        c1=64, c4=256, num_heads=4, mlp_dim=512, depth=3
    )

    # 前向传播
    p1_attention, P4_tokens = bi_attention(P1, P4, pos_embed)
    print("P1 Attention 形状:", p1_attention.shape)  # [b, H/4*W/4, C4]
    print("P4 Tokens 形状:", P4_tokens.shape)       # [b, H/32*W/32, C4]
