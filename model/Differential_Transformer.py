import torch
import torch.nn.functional as F
from torch import nn, Tensor
from math import sqrt, exp


class SimpleRMSNorm(nn.Module):
    """
    SimpleRMSNorm

    Args:
        dim (int): dimension of the embedding
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class DiffAttn(nn.Module):
    """
    Differential Attention module.

    This module computes attention weights based on the difference between two sets of queries and keys.

    Attributes:
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - W_q (nn.Linear): Linear layer for transforming queries.
    - W_k (nn.Linear): Linear layer for transforming keys.
    - W_v (nn.Linear): Linear layer for transforming values.
    """

    def __init__(self, d: int, embedding_dim: int):
        super(DiffAttn, self).__init__()
        self.d = d
        self.W_q = nn.Linear(d, 2 * embedding_dim)
        self.W_k = nn.Linear(d, 2 * embedding_dim)
        self.W_v = nn.Linear(d, embedding_dim)  # Changed to output d dimensions

    def forward(self, X: Tensor, lambda_: float) -> Tensor:
        """
        Forward pass of the Differential Attention module.

        Args:
        - X (Tensor): Input tensor. [b,N,embedding_dim]
        - λ (float): Scaling factor for the difference.

        Returns:
        - Tensor: Output tensor.
        """

        Q = self.W_q(X)  # [b,N,d]->[b,N,2embedding_dim]
        K = self.W_k(X)  # [b,N,d]->[b,N,2embedding_dim]
        V = self.W_v(X)  # [b,N,d]->[b,N,embedding_dim]

        Q1, Q2 = self.split(Q)  # [b,N,2embedding_dim]->[b,N,embedding_dim]
        K1, K2 = self.split(K)  # [b,N,2embedding_dim]->[b,N,embedding_dim]

        s = 1 / sqrt(self.d)

        # [b,N,N]
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s

        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)

        result = (A1_softmax - lambda_ * A2_softmax) @ V
        return result

    @staticmethod
    def split(X: Tensor) -> (Tensor, Tensor):
        """
        Splits the input tensor into two halves along the last dimension.

        Args:
        - X (Tensor): Input tensor.

        Returns:
        - Tuple[Tensor, Tensor]: Two tensors, each containing half of the input dimensions.
        """
        half_dim = X.shape[-1] // 2
        return X[..., :half_dim], X[..., half_dim:]


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention module.

    This module applies the Differential Attention mechanism multiple times in parallel.

    Attributes:
    - h (int): The number of attention heads.
    - d (int): The dimensionality of the attention weights.
    - embedding_dim (int): The dimensionality of the input embeddings.
    - λinit (float): The initial scaling factor for the difference.
    - diff_attn_heads (nn.ModuleList): List of Differential Attention modules.
    - W_o (nn.Linear): Linear layer for output transformation.
    - norm (nn.LayerNorm): Layer normalization module.
    """

    def __init__(self, h: int, d: int, embedding_dim: int, lambda_init: float):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d = d
        self.lambda_init = lambda_init
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * embedding_dim, d)  # Changed to h * d
        # Initialize λ parameters for each head
        self.lambda_q1 = nn.Parameter(torch.zeros(h, embedding_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(h, embedding_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(h, embedding_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(h, embedding_dim).normal_(mean=0, std=0.1))

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass of the Multi-Head Differential Attention module.

        Args:
        - X (Tensor): Input tensor. [b,H*W,d]-->[b,H*W,2embedding_dim]-->[b,H*W,embedding_dim]
        - λ (float): Scaling factor for the difference.

        Returns:
        - Tensor: Output tensor.
        """

        O_list = []

        for i, head in enumerate(self.diff_attn_heads):
            # Calculate the dynamic λ for each head
            lambda_1 = torch.exp(torch.sum(self.lambda_q1[i] * self.lambda_k1[i], dim=-1).float()).type_as(X)
            lambda_2 = torch.exp(torch.sum(self.lambda_q2[i] * self.lambda_k2[i], dim=-1).float()).type_as(X)
            lambda_full = lambda_1 - lambda_2 + self.lambda_init

            # Perform attention for the current head with the calculated λ_full
            O = head(X, lambda_full)
            # [b,N,d]
            O_list.append(O)

        O_concat = torch.cat(O_list, dim=-1)

        # Apply the output transformation
        result = self.W_o(O_concat) * (1 - self.lambda_init)

        return result


class DifferentialTransformerBlock(nn.Module):
    """
    This class implements a Differential Transformer Block.
    """

    def __init__(
            self,
            d: int,
            embedding_dim: int,
            heads: int,
            dropout: float,
            lambda_init: float
    ):
        """
        Initializes the Differential Transformer Block.
        """
        super(DifferentialTransformerBlock, self).__init__()

        # Differential
        self.attn = MultiHeadDifferentialAttention(
            h=heads, d=d, embedding_dim=embedding_dim, lambda_init=lambda_init
        )

        # FFN
        self.ffn = FeedForward(
            d,
            d * 4,
            dropout=dropout
        )

        self.norm = SimpleRMSNorm(embedding_dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the Differential Transformer Block.
        """
        # Norm
        residual = x

        attended = self.attn(self.norm(x)) + residual

        residual2 = attended

        attended = self.ffn(self.norm(attended)) + residual2

        return attended
