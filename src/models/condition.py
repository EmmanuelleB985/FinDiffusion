"""Condition encoders for conditional generation."""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Encode market conditions (trend, volatility, regime) for conditional generation.
    
    Conditions:
        - trend: Expected annualized return (scalar)
        - volatility: Expected annualized volatility (scalar)
        - regime: Market regime (categorical: 0=bear, 1=sideways, 2=bull)
    """

    def __init__(
        self,
        d_cond: int = 128,
        n_regimes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_cond = d_cond

        # Trend encoder (continuous)
        self.trend_encoder = nn.Sequential(
            nn.Linear(1, d_cond // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cond // 2, d_cond // 3),
        )

        # Volatility encoder (continuous)
        self.vol_encoder = nn.Sequential(
            nn.Linear(1, d_cond // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cond // 2, d_cond // 3),
        )

        # Regime encoder (categorical)
        self.regime_embedding = nn.Embedding(n_regimes, d_cond // 3)

        # Projection to final dimension
        self.proj = nn.Sequential(
            nn.Linear(d_cond // 3 * 3, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        # For unconditional generation (classifier-free guidance)
        self.null_cond = nn.Parameter(torch.randn(d_cond))

    def forward(
        self,
        trend: Optional[torch.Tensor] = None,
        volatility: Optional[torch.Tensor] = None,
        regime: Optional[torch.Tensor] = None,
        drop_cond_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Encode conditions into a single embedding.
        
        Args:
            trend: Annualized return expectation, shape (B, 1) or (B,)
            volatility: Annualized vol expectation, shape (B, 1) or (B,)
            regime: Market regime index, shape (B,)
            drop_cond_prob: Probability of dropping conditions (for CFG training)
        
        Returns:
            Condition embedding of shape (B, d_cond)
        """
        # Determine batch size
        if trend is not None:
            B = trend.shape[0]
            device = trend.device
        elif volatility is not None:
            B = volatility.shape[0]
            device = volatility.device
        elif regime is not None:
            B = regime.shape[0]
            device = regime.device
        else:
            raise ValueError("At least one condition must be provided")

        # Handle unconditional generation
        if drop_cond_prob > 0 and self.training:
            drop_mask = torch.rand(B, device=device) < drop_cond_prob
            if drop_mask.all():
                return self.null_cond.expand(B, -1)

        # Process each condition
        embeddings = []

        # Trend
        if trend is not None:
            if trend.dim() == 1:
                trend = trend.unsqueeze(-1)
            trend_emb = self.trend_encoder(trend)
        else:
            trend_emb = torch.zeros(B, self.d_cond // 3, device=device)
        embeddings.append(trend_emb)

        # Volatility
        if volatility is not None:
            if volatility.dim() == 1:
                volatility = volatility.unsqueeze(-1)
            vol_emb = self.vol_encoder(volatility)
        else:
            vol_emb = torch.zeros(B, self.d_cond // 3, device=device)
        embeddings.append(vol_emb)

        # Regime
        if regime is not None:
            regime_emb = self.regime_embedding(regime)
        else:
            # Default to sideways (regime=1) if not provided
            regime_emb = self.regime_embedding(torch.ones(B, dtype=torch.long, device=device))
        embeddings.append(regime_emb)

        # Combine and project
        combined = torch.cat(embeddings, dim=-1)
        cond_emb = self.proj(combined)

        # Apply dropout mask for classifier-free guidance
        if drop_cond_prob > 0 and self.training:
            drop_mask = drop_mask.unsqueeze(-1)
            cond_emb = torch.where(drop_mask, self.null_cond.expand(B, -1), cond_emb)

        return cond_emb

    def encode_from_dict(
        self,
        conditions: Dict[str, Union[float, str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Encode conditions from a dictionary (convenience method).
        
        Args:
            conditions: Dict with keys 'trend', 'volatility', 'regime'
            batch_size: Number of samples to generate
            device: Target device
        
        Returns:
            Condition embedding of shape (batch_size, d_cond)
        """
        trend = None
        volatility = None
        regime = None

        if "trend" in conditions:
            val = conditions["trend"]
            if isinstance(val, (int, float)):
                trend = torch.full((batch_size, 1), val, device=device)
            else:
                trend = val.to(device)

        if "volatility" in conditions:
            val = conditions["volatility"]
            if isinstance(val, (int, float)):
                volatility = torch.full((batch_size, 1), val, device=device)
            else:
                volatility = val.to(device)

        if "regime" in conditions:
            val = conditions["regime"]
            if isinstance(val, str):
                regime_map = {"bear": 0, "sideways": 1, "bull": 2}
                regime = torch.full((batch_size,), regime_map[val], dtype=torch.long, device=device)
            elif isinstance(val, int):
                regime = torch.full((batch_size,), val, dtype=torch.long, device=device)
            else:
                regime = val.to(device)

        return self.forward(trend, volatility, regime)


class ConditionExtractor:
    """Extract conditions from real financial data for training."""

    def __init__(self, annualize_factor: int = 252):
        self.annualize_factor = annualize_factor

    def extract(self, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract trend, volatility, and regime from return sequences.
        
        Args:
            returns: Return tensor of shape (B, T)
        
        Returns:
            Dict with 'trend', 'volatility', 'regime' tensors
        """
        # Annualized return (trend)
        cumulative_return = (1 + returns).prod(dim=-1) - 1
        T = returns.shape[-1]
        annualized_return = (1 + cumulative_return) ** (self.annualize_factor / T) - 1

        # Annualized volatility
        daily_vol = returns.std(dim=-1)
        annualized_vol = daily_vol * (self.annualize_factor ** 0.5)

        # Regime classification based on Sharpe-like metric
        sharpe_proxy = annualized_return / (annualized_vol + 1e-8)
        regime = torch.zeros_like(sharpe_proxy, dtype=torch.long)
        regime[sharpe_proxy < -0.5] = 0  # Bear
        regime[(sharpe_proxy >= -0.5) & (sharpe_proxy <= 0.5)] = 1  # Sideways
        regime[sharpe_proxy > 0.5] = 2  # Bull

        return {
            "trend": annualized_return,
            "volatility": annualized_vol,
            "regime": regime,
        }
