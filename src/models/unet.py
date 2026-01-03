"""Transformer-based denoiser for time series diffusion."""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .attention import TransformerBlock, TimeEmbedding


class TransformerDenoiser(nn.Module):
    """
    Transformer-based denoiser for financial time series.
    
    Takes noisy returns, timestep, and conditions as input,
    predicts the noise (or clean data) for the denoising step.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        d_cond: int = 128,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Time embedding
        self.time_embed = TimeEmbedding(d_model)

        # Time injection layers (add time embedding to each layer)
        self.time_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, d_model * 2),
            )
            for _ in range(n_layers)
        ])

        # Transformer blocks with cross-attention for conditioning
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                d_cond=d_cond,
                dropout=dropout,
                causal=False,  # Bidirectional attention for denoising
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Zero-initialize output projection for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the denoiser.
        
        Args:
            x: Noisy input of shape (B, T, input_dim) or (B, T)
            t: Timestep of shape (B,)
            cond: Condition embedding of shape (B, d_cond)
        
        Returns:
            Predicted noise of shape (B, T, input_dim)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        B, T, _ = x.shape

        # Project input to model dimension
        h = self.input_proj(x)

        # Add positional embedding
        h = h + self.pos_embedding[:, :T, :]

        # Get time embedding
        time_emb = self.time_embed(t)

        # Process through transformer blocks
        for block, time_layer in zip(self.blocks, self.time_layers):
            # Inject time embedding (scale and shift)
            time_params = time_layer(time_emb)
            scale, shift = time_params.chunk(2, dim=-1)
            h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

            # Transformer block with optional conditioning
            h = block(h, cond=cond)

        # Output projection
        h = self.output_norm(h)
        out = self.output_proj(h)

        return out


class ConvDenoiser(nn.Module):
    """
    1D Convolutional denoiser as a simpler/faster alternative.
    Uses dilated convolutions for large receptive field.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        n_layers: int = 8,
        kernel_size: int = 3,
        d_cond: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, d_model, 1)

        # Time embedding
        self.time_embed = TimeEmbedding(d_model)

        # Condition projection
        self.cond_proj = nn.Linear(d_cond, d_model) if d_cond > 0 else None

        # Dilated conv blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # Cycle dilation: 1, 2, 4, 8, 1, 2, 4, 8
            self.blocks.append(
                ConvBlock(
                    d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # Time injection
        self.time_layers = nn.ModuleList([
            nn.Linear(d_model, d_model * 2) for _ in range(n_layers)
        ])

        # Output
        self.output_norm = nn.GroupNorm(8, d_model)
        self.output_proj = nn.Conv1d(d_model, input_dim, 1)

        # Zero-init output
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) or (B, T)
            t: (B,)
            cond: (B, d_cond)
        
        Returns:
            (B, T, input_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # (B, T, C) -> (B, C, T)
        h = x.permute(0, 2, 1)
        h = self.input_proj(h)

        # Time embedding
        time_emb = self.time_embed(t)

        # Add condition if provided
        if cond is not None and self.cond_proj is not None:
            cond_emb = self.cond_proj(cond)
            time_emb = time_emb + cond_emb

        # Process through blocks
        for block, time_layer in zip(self.blocks, self.time_layers):
            # Time injection
            time_params = time_layer(time_emb)
            scale, shift = time_params.chunk(2, dim=-1)
            h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

            h = block(h)

        # Output
        h = self.output_norm(h)
        h = self.output_proj(h)

        # (B, C, T) -> (B, T, C)
        return h.permute(0, 2, 1)


class ConvBlock(nn.Module):
    """Residual dilated conv block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h
