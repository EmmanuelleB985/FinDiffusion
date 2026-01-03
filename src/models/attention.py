"""Attention mechanisms for the diffusion model."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask of shape (B, T) or (B, T, T)
        
        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float("-inf"))

        # Apply optional padding mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on external information."""

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_cond, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D)
            cond: Condition tensor of shape (B, C, D_cond) or (B, D_cond)
        
        Returns:
            Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # Handle 2D condition (expand to sequence)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, D_cond)

        # Compute Q from input, K/V from condition
        q = self.q(x)
        kv = self.kv(cond)
        k, v = rearrange(kv, "b c (two h d) -> two b h c d", two=2, h=self.n_heads)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention, optional cross-attention, and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_cond: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, causal)
        self.norm1 = nn.LayerNorm(d_model)

        self.has_cross_attn = d_cond is not None
        if self.has_cross_attn:
            self.cross_attn = CrossAttention(d_model, d_cond, n_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), mask)

        # Cross-attention (if conditioning)
        if self.has_cross_attn and cond is not None:
            x = x + self.cross_attn(self.norm2(x), cond)

        # Feed-forward
        x = x + self.ff(self.norm3(x))

        return x


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (B,) with values in [0, T]
        
        Returns:
            Embedding tensor of shape (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class TimeEmbedding(nn.Module):
    """Time embedding with MLP projection."""

    def __init__(self, d_model: int, d_time: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(d_time)
        self.mlp = nn.Sequential(
            nn.Linear(d_time, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal(t)
        return self.mlp(emb)
