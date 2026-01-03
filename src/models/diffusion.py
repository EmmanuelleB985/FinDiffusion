"""Gaussian Diffusion implementation for financial time series."""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .unet import TransformerDenoiser, ConvDenoiser
from .condition import ConditionEncoder, ConditionExtractor


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """Cosine noise schedule from Improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Quadratic noise schedule."""
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process (DDPM).
    
    Implements the forward and reverse diffusion processes for
    generating synthetic financial time series.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",  # "epsilon" or "x0"
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        super().__init__()

        self.denoiser = denoiser
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range

        # Build noise schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-compute diffusion coefficients
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Pre-compute coefficients for q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Pre-compute coefficients for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).
        
        Args:
            x_0: Clean data of shape (B, T, D) or (B, T)
            t: Timesteps of shape (B,)
            noise: Optional pre-generated noise
        
        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    def predict_eps_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from x_t and predicted x_0."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alpha * x_0) / sqrt_one_minus_alpha

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        clip_denoised: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        
        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            cond: Optional condition embedding
            clip_denoised: Whether to clip predicted x_0
        
        Returns:
            Dict with 'mean', 'variance', 'log_variance', 'pred_x0'
        """
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        # Get model prediction
        model_output = self.denoiser(x_t, t, cond)

        # Convert to x_0 prediction
        if self.prediction_type == "epsilon":
            pred_x0 = self.predict_x0_from_eps(x_t, t, model_output)
        else:  # x0 prediction
            pred_x0 = model_output

        # Optionally clip predicted x_0
        if clip_denoised:
            pred_x0 = pred_x0.clamp(*self.clip_range)

        # Compute posterior mean and variance
        mean, variance, log_variance = self.q_posterior_mean_variance(pred_x0, x_t, t)

        return {
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance,
            "pred_x0": pred_x0,
        }

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single denoising step: sample x_{t-1} from p(x_{t-1} | x_t)."""
        out = self.p_mean_variance(x_t, t, cond)
        noise = torch.randn_like(x_t)

        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))

        x_prev = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples via full reverse diffusion.
        
        Args:
            shape: Shape of samples to generate (B, T, D) or (B, T)
            cond: Optional condition embedding
            device: Device to generate on
            progress: Whether to show progress bar
        
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion
        timesteps = list(reversed(range(self.timesteps)))
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling", leave=False)

        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, cond)

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using DDIM (faster sampling).
        
        Args:
            shape: Shape of samples to generate
            cond: Optional condition embedding
            n_steps: Number of DDIM steps (< timesteps for speedup)
            eta: DDIM stochasticity (0 = deterministic)
            device: Device to generate on
            progress: Whether to show progress bar
        
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device

        # Compute DDIM timestep sequence
        step_size = self.timesteps // n_steps
        timesteps = list(reversed(range(0, self.timesteps, step_size)))

        x = torch.randn(shape, device=device)

        if progress:
            timesteps = tqdm(timesteps, desc="DDIM Sampling", leave=False)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Get model prediction
            model_output = self.denoiser(x, t_batch, cond)

            # Predict x_0
            if self.prediction_type == "epsilon":
                pred_x0 = self.predict_x0_from_eps(x, t_batch, model_output)
            else:
                pred_x0 = model_output

            if self.clip_denoised:
                pred_x0 = pred_x0.clamp(*self.clip_range)

            # Get alpha values
            alpha_t = self._extract(self.alphas_cumprod, t_batch, x.shape)
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # DDIM update
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))

            pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * (
                (x - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t)
            )
            noise = torch.randn_like(x) if t > 0 else 0

            x = torch.sqrt(alpha_t_prev) * pred_x0 + pred_dir + sigma_t * noise

        return x

    def training_loss(
        self,
        x_0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x_0: Clean data
            cond: Optional condition embedding
            noise: Optional pre-generated noise
        
        Returns:
            Dict with 'loss' and optionally other metrics
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        # Add noise
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        # Get model prediction
        model_output = self.denoiser(x_t, t, cond)

        # Compute loss based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        else:
            target = x_0

        loss = F.mse_loss(model_output, target)

        return {"loss": loss}


class FinancialDiffusion(nn.Module):
    """
    Complete model for conditional financial time series generation.
    Combines denoiser, condition encoder, and diffusion process.
    """

    def __init__(
        self,
        seq_len: int = 252,
        input_dim: int = 1,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        d_cond: int = 128,
        n_regimes: int = 3,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        dropout: float = 0.1,
        denoiser_type: str = "transformer",  # "transformer" or "conv"
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim

        # Condition encoder
        self.condition_encoder = ConditionEncoder(
            d_cond=d_cond,
            n_regimes=n_regimes,
            dropout=dropout,
        )

        # Condition extractor (for training)
        self.condition_extractor = ConditionExtractor()

        # Denoiser backbone
        if denoiser_type == "transformer":
            denoiser = TransformerDenoiser(
                input_dim=input_dim,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                d_cond=d_cond,
                max_seq_len=seq_len,
                dropout=dropout,
            )
        else:
            denoiser = ConvDenoiser(
                input_dim=input_dim,
                d_model=d_model,
                n_layers=n_layers,
                d_cond=d_cond,
                dropout=dropout,
            )

        # Diffusion process
        self.diffusion = GaussianDiffusion(
            denoiser=denoiser,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        trend: Optional[torch.Tensor] = None,
        volatility: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
        drop_cond_prob: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            x: Clean returns of shape (B, T) or (B, T, 1)
            trend: Optional trend condition
            volatility: Optional volatility condition
            regime: Optional regime condition
            drop_cond_prob: Probability of dropping conditions (CFG)
        
        Returns:
            Dict with 'loss'
        """
        # Extract conditions from data if not provided
        if trend is None or volatility is None or regime is None:
            x_2d = x.squeeze(-1) if x.dim() == 3 else x
            extracted = self.condition_extractor.extract(x_2d)
            if trend is None:
                trend = extracted["trend"]
            if volatility is None:
                volatility = extracted["volatility"]
            if regime is None:
                regime = extracted["regime"]

        # Encode conditions
        cond = self.condition_encoder(
            trend=trend,
            volatility=volatility,
            regime=regime,
            drop_cond_prob=drop_cond_prob,
        )

        # Compute diffusion loss
        return self.diffusion.training_loss(x, cond)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        seq_len: Optional[int] = None,
        conditions: Optional[Dict] = None,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        device: Optional[torch.device] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate synthetic financial time series.
        
        Args:
            n_samples: Number of samples to generate
            seq_len: Sequence length (default: self.seq_len)
            conditions: Dict with 'trend', 'volatility', 'regime'
            use_ddim: Whether to use DDIM for faster sampling
            ddim_steps: Number of DDIM steps
            device: Device to generate on
            progress: Whether to show progress bar
        
        Returns:
            Generated returns of shape (n_samples, seq_len)
        """
        if device is None:
            device = next(self.parameters()).device
        if seq_len is None:
            seq_len = self.seq_len

        # Encode conditions
        if conditions is not None:
            cond = self.condition_encoder.encode_from_dict(conditions, n_samples, device)
        else:
            cond = None

        # Generate
        shape = (n_samples, seq_len, self.input_dim)

        if use_ddim:
            samples = self.diffusion.ddim_sample(
                shape, cond, n_steps=ddim_steps, device=device, progress=progress
            )
        else:
            samples = self.diffusion.sample(shape, cond, device=device, progress=progress)

        # Remove last dimension if univariate
        if self.input_dim == 1:
            samples = samples.squeeze(-1)

        return samples

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "seq_len": self.seq_len,
                "input_dim": self.input_dim,
            }
        }, path)

    @classmethod
    def load_from_checkpoint(cls, path: str, **kwargs) -> "FinancialDiffusion":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint.get("config", {})
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        return model
