import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    该类实现了一个标准的多头注意力模块，不使用官方库函数。
    """

    def __init__(self, dim, heads, dropout=0.1):
        """
        初始化多头注意力模块。

        参数：
        - dim: 嵌入维度（即输入的通道数）。
        - heads: 注意力头的数量。
        - dropout: Dropout比率。
        """
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.embed_dim = self.dim
        self.head_dim = self.embed_dim // heads  # 每个头的维度

        assert self.head_dim * heads == self.embed_dim, "嵌入维度必须能被头数整除"

        # 线性层：用于生成多头注意力的查询、键和值
        self.query = nn.Linear(dim, self.embed_dim)
        self.key = nn.Linear(dim, self.embed_dim)
        self.value = nn.Linear(dim, self.embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),  # 隐藏层，宽度是嵌入维度的4倍
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)  # 输出层，维度恢复为嵌入维度
        )

        # Layer normalization 和 dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        注意力模块的前向传播。

        参数：
        - x: 输入张量，形状为 (batch_size, seq_len, embed_dim)

        返回：
        - 经注意力计算后的输出张量，经过归一化和dropout处理
        """
        batch_size, seq_len, _ = x.size()

        # 第一步：通过线性变换生成 Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, dim)
        K = self.key(x)  # (batch_size, seq_len, dim)
        V = self.value(x)  # (batch_size, seq_len, dim)

        # 第二步：将 Q, K, V 重塑为 (batch_size, heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)  # (batch_size, heads, seq_len, head_dim)

        # 第三步：计算注意力得分（缩放点积注意力）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5  # (batch_size, heads, seq_len, seq_len)

        # 第四步：应用 Softmax 获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, heads, seq_len, seq_len)

        # 第五步：应用 Dropout
        attn_weights = self.dropout(attn_weights)

        # 第六步：计算输出（batch_size, heads, seq_len, head_dim） * V（batch_size, heads, seq_len, head_dim）
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, heads, seq_len, head_dim)

        # 第七步：重塑并通过 MLP
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, dim)
        output = self.mlp(attn_output)

        # 第八步：应用层归一化和跳跃连接
        x = self.norm(x + self.dropout(output))  # 添加残差连接并进行归一化

        return x
