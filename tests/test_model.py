"""Tests for model components."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    FinancialDiffusion,
    GaussianDiffusion,
    TransformerDenoiser,
    ConditionEncoder,
)
from src.models.attention import MultiHeadAttention, CrossAttention, TransformerBlock


class TestAttention:
    """Test attention modules."""

    def test_multihead_attention_shape(self):
        batch, seq_len, d_model = 4, 64, 128
        n_heads = 8

        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch, seq_len, d_model)

        out = attn(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_cross_attention_shape(self):
        batch, seq_len, d_model, d_cond = 4, 64, 128, 64
        n_heads = 8

        attn = CrossAttention(d_model, d_cond, n_heads)
        x = torch.randn(batch, seq_len, d_model)
        cond = torch.randn(batch, d_cond)

        out = attn(x, cond)
        assert out.shape == (batch, seq_len, d_model)

    def test_transformer_block(self):
        batch, seq_len, d_model, d_cond = 4, 64, 128, 64

        block = TransformerBlock(d_model, n_heads=8, d_ff=256, d_cond=d_cond)
        x = torch.randn(batch, seq_len, d_model)
        cond = torch.randn(batch, d_cond)

        out = block(x, cond)
        assert out.shape == (batch, seq_len, d_model)


class TestConditionEncoder:
    """Test condition encoder."""

    def test_encode_all_conditions(self):
        batch, d_cond = 8, 128
        encoder = ConditionEncoder(d_cond=d_cond, n_regimes=3)

        trend = torch.randn(batch, 1)
        volatility = torch.rand(batch, 1)
        regime = torch.randint(0, 3, (batch,))

        out = encoder(trend, volatility, regime)
        assert out.shape == (batch, d_cond)

    def test_encode_partial_conditions(self):
        batch, d_cond = 8, 128
        encoder = ConditionEncoder(d_cond=d_cond)

        trend = torch.randn(batch, 1)
        out = encoder(trend=trend)
        assert out.shape == (batch, d_cond)

    def test_encode_from_dict(self):
        batch, d_cond = 8, 128
        encoder = ConditionEncoder(d_cond=d_cond)

        conditions = {
            "trend": 0.1,
            "volatility": 0.2,
            "regime": "bull",
        }
        out = encoder.encode_from_dict(conditions, batch, torch.device("cpu"))
        assert out.shape == (batch, d_cond)


class TestDenoiser:
    """Test denoiser models."""

    def test_transformer_denoiser_shape(self):
        batch, seq_len, d_model, d_cond = 4, 64, 128, 64

        denoiser = TransformerDenoiser(
            input_dim=1,
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            d_cond=d_cond,
            max_seq_len=128,
        )

        x = torch.randn(batch, seq_len, 1)
        t = torch.randint(0, 1000, (batch,))
        cond = torch.randn(batch, d_cond)

        out = denoiser(x, t, cond)
        assert out.shape == (batch, seq_len, 1)

    def test_denoiser_2d_input(self):
        batch, seq_len, d_model = 4, 64, 128

        denoiser = TransformerDenoiser(
            input_dim=1,
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            d_cond=64,
        )

        x = torch.randn(batch, seq_len)  # 2D input
        t = torch.randint(0, 1000, (batch,))
        cond = torch.randn(batch, 64)

        out = denoiser(x, t, cond)
        assert out.shape == (batch, seq_len, 1)


class TestGaussianDiffusion:
    """Test diffusion process."""

    @pytest.fixture
    def diffusion(self):
        denoiser = TransformerDenoiser(
            input_dim=1,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            d_cond=32,
        )
        return GaussianDiffusion(
            denoiser=denoiser,
            timesteps=100,
            beta_schedule="linear",
        )

    def test_q_sample(self, diffusion):
        batch, seq_len = 4, 64
        x_0 = torch.randn(batch, seq_len, 1)
        t = torch.randint(0, 100, (batch,))

        x_t, noise = diffusion.q_sample(x_0, t)

        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape

    def test_training_loss(self, diffusion):
        batch, seq_len = 4, 64
        x_0 = torch.randn(batch, seq_len, 1)
        cond = torch.randn(batch, 32)

        output = diffusion.training_loss(x_0, cond)

        assert "loss" in output
        assert output["loss"].shape == ()
        assert output["loss"].requires_grad

    def test_sample_shape(self, diffusion):
        batch, seq_len = 2, 32
        cond = torch.randn(batch, 32)

        # Use fewer timesteps for speed
        diffusion.timesteps = 10
        samples = diffusion.sample(
            shape=(batch, seq_len, 1),
            cond=cond,
            progress=False,
        )

        assert samples.shape == (batch, seq_len, 1)


class TestFinancialDiffusion:
    """Test complete financial diffusion model."""

    @pytest.fixture
    def model(self):
        return FinancialDiffusion(
            seq_len=64,
            input_dim=1,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            d_cond=32,
            n_regimes=3,
            timesteps=100,
        )

    def test_forward_pass(self, model):
        batch, seq_len = 4, 64
        x = torch.randn(batch, seq_len)

        output = model(x)

        assert "loss" in output
        assert output["loss"].requires_grad

    def test_forward_with_conditions(self, model):
        batch, seq_len = 4, 64
        x = torch.randn(batch, seq_len)
        trend = torch.randn(batch)
        vol = torch.rand(batch)
        regime = torch.randint(0, 3, (batch,))

        output = model(x, trend=trend, volatility=vol, regime=regime)

        assert "loss" in output

    def test_generate(self, model):
        n_samples = 2

        # Reduce timesteps for speed
        model.diffusion.timesteps = 10

        samples = model.generate(
            n_samples=n_samples,
            conditions={"trend": 0.1, "volatility": 0.2, "regime": "bull"},
            progress=False,
        )

        assert samples.shape == (n_samples, 64)


class TestEndToEnd:
    """End-to-end tests."""

    def test_training_step(self):
        model = FinancialDiffusion(
            seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            d_cond=16,
            timesteps=10,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training step
        x = torch.randn(4, 32)
        output = model(x)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_gradient_flow(self):
        model = FinancialDiffusion(
            seq_len=32,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            d_cond=16,
            timesteps=10,
        )

        x = torch.randn(4, 32)
        output = model(x)
        output["loss"].backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
