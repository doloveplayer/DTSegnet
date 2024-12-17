import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    This class implements a standard Multihead Attention Block without using official library functions.
    """

    def __init__(self, dim, heads, dropout=0.1):
        """
        Initializes the Multihead Attention Block.

        Args:
        - dim: The embedding dimension (C in your input).
        - heads: The number of attention heads.
        - dropout: Dropout rate.
        """
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.embed_dim = 512
        self.head_dim = self.embed_dim // heads  # Dimension of each head

        assert self.head_dim * heads == self.embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear layers to generate queries, keys, and values for multi-head attention
        self.query = nn.Linear(dim, self.embed_dim)
        self.key = nn.Linear(dim, self.embed_dim)
        self.value = nn.Linear(dim, self.embed_dim)

        # Output projection
        self.out_proj = nn.Linear(self.embed_dim, dim)

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the Attention Block.

        Args:
        - x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
        - Output tensor after attention, normalization, and dropout
        """
        batch_size, seq_len, _ = x.size()

        # Step 1: Apply linear transformations to get Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, dim)
        K = self.key(x)  # (batch_size, seq_len, dim)
        V = self.value(x)  # (batch_size, seq_len, dim)

        # Step 2: Reshape to (batch_size, seq_len, heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1,
                                                                             2)  # (batch_size, heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1,
                                                                             2)  # (batch_size, heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1,
                                                                             2)  # (batch_size, heads, seq_len, head_dim)

        # Step 3: Compute attention scores (Scaled Dot Product Attention)
        # Q (batch_size, heads, seq_len, head_dim) * K^T (batch_size, heads, head_dim, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5  # (batch_size, heads, seq_len, seq_len)

        # Step 4: Apply Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, heads, seq_len, seq_len)

        # Step 5: Apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # Step 6: Compute the output (batch_size, heads, seq_len, head_dim) * V (batch_size, heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, heads, seq_len, head_dim)

        # Step 7: Reshape and apply the output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                    self.embed_dim)  # (batch_size, seq_len, dim)
        output = self.out_proj(attn_output)

        # Step 8: Apply layer normalization and skip connection
        x = self.norm(x + self.dropout(output))  # Add residual connection and normalize

        return x
