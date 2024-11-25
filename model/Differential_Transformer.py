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


class OutputHead(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None):
        """
        Output head implementation.
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features (e.g., number of classes).
            activation (callable or None): Optional activation function (e.g., softmax).
        """
        super(OutputHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        """
        Forward pass for OutputHead.
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


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
        self.W_q = nn.Linear(embedding_dim, 2 * d)
        self.W_k = nn.Linear(embedding_dim, 2 * d)
        self.W_v = nn.Linear(embedding_dim, d)  # Changed to output d dimensions

    def forward(self, X: Tensor, λ: float) -> Tensor:
        """
        Forward pass of the Differential Attention module.

        Args:
        - X (Tensor): Input tensor. [b,N,embedding_dim]
        - λ (float): Scaling factor for the difference.

        Returns:
        - Tensor: Output tensor.
        """

        Q = self.W_q(X)  # [b,N,embedding_dim]->[b,N,2d]
        K = self.W_k(X)  # [b,N,embedding_dim]->[b,N,2d]
        V = self.W_v(X)  # [b,N,embedding_dim]->[b,N,d]

        Q1, Q2 = self.split(Q)  # [b,N,2d]->[b,N,d]
        K1, K2 = self.split(K)  # [b,N,2d]->[b,N,d]

        s = 1 / sqrt(self.d)

        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s

        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)

        result = (A1_softmax - λ * A2_softmax) @ V
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

    def __init__(self, h: int, d: int, embedding_dim: int, λinit: float):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.h = h
        self.d = d
        self.λinit = λinit
        self.embedding_dim = embedding_dim
        self.diff_attn_heads = nn.ModuleList([DiffAttn(d, embedding_dim) for _ in range(h)])
        self.W_o = nn.Linear(h * d, embedding_dim)  # Changed to h * d
        self.norm = nn.LayerNorm(embedding_dim)
        self.λ = nn.Parameter(torch.full((1,), λinit))  # Learnable λ initialized to λinit

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass of the Multi-Head Differential Attention module.

        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.

        Returns:
        - Tensor: Output tensor.
        """

        O_list = [head(X, self.λ) for head in self.diff_attn_heads]

        O_concat = torch.cat(O_list, dim=-1)

        # Apply the output transformation
        result = self.W_o(O_concat)

        # Apply LayerNorm
        result = self.norm(result)

        # Scale by λinit
        result = result * (1 - self.λinit)

        return result


class DifferentialTransformerBlock(nn.Module):
    """
    This class implements a Differential Transformer Block.
    """

    def __init__(
            self,
            dim: int,
            heads: int = 12,
            dropout: float = 0.1,
            λinit: float = 0.8
    ):
        """
        Initializes the Differential Transformer Block.
        """
        super(DifferentialTransformerBlock, self).__init__()

        # Differential
        self.attn = MultiHeadDifferentialAttention(
            h=heads, d=dim, embedding_dim=dim, λinit=λinit
        )

        # FFN
        self.ffn = FeedForward(
            dim,
            dim * 4,
            dropout=dropout
        )

        self.norm = SimpleRMSNorm(dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the Differential Transformer Block.
        """
        # Norm
        residual = x

        attended = self.attn(self.norm(x)) + residual

        residual2 = attended

        attended = self.ffn(attended) + residual2

        return attended


class DifferentialTransformer(nn.Module):
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
            output_dim: int = 512,
    ):
        """
        Initializes the Differential Transformer Block.
        """
        super(DifferentialTransformer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth
        self.λinit = λinit if λinit is not None else (0.8 - 0.6 * exp(-0.3 * (depth - 1)))
        self.output_dim = output_dim

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

        # Embedding
        self.embed = nn.Embedding(num_embeddings=output_dim, embedding_dim=dim)

        # Norm
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x):
        # Embed
        x = self.norm(self.embed(x))

        # Post embed norm
        for layer in self.layers:
            x = layer(x)

        return OutputHead(self.dim, output_dim=self.output_dim)(x)
