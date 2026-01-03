#!/usr/bin/env python
"""Generate synthetic financial data using trained model."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FinancialDiffusion
from src.data import FinancialDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic financial data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length (default: from config)")
    parser.add_argument("--trend", type=float, default=None, help="Target annualized return")
    parser.add_argument("--volatility", type=float, default=None, help="Target annualized volatility")
    parser.add_argument("--regime", type=str, default=None, choices=["bear", "sideways", "bull"])
    parser.add_argument("--output", type=str, default="outputs/synthetic_returns.csv", help="Output path")
    parser.add_argument("--ddim", action="store_true", help="Use DDIM for faster sampling")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = FinancialDiffusion(
        seq_len=config["data"]["seq_len"],
        input_dim=config["model"]["input_dim"],
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        d_cond=config["model"]["condition_dim"],
        n_regimes=config["model"]["n_regimes"],
        timesteps=config["model"]["timesteps"],
        beta_schedule=config["model"]["beta_schedule"],
        beta_start=config["model"]["beta_start"],
        beta_end=config["model"]["beta_end"],
        prediction_type=config["model"]["prediction_type"],
        dropout=0.0,  # No dropout at inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Build conditions
    conditions = {}
    if args.trend is not None:
        conditions["trend"] = args.trend
    if args.volatility is not None:
        conditions["volatility"] = args.volatility
    if args.regime is not None:
        conditions["regime"] = args.regime

    if not conditions:
        conditions = None
        logger.info("Generating unconditional samples")
    else:
        logger.info(f"Generating with conditions: {conditions}")

    # Generate
    seq_len = args.seq_len or config["data"]["seq_len"]
    
    logger.info(f"Generating {args.n_samples} samples of length {seq_len}...")
    
    with torch.no_grad():
        samples = model.generate(
            n_samples=args.n_samples,
            seq_len=seq_len,
            conditions=conditions,
            use_ddim=args.ddim,
            ddim_steps=args.ddim_steps if args.ddim else None,
            device=device,
            progress=True,
        )

    samples = samples.cpu().numpy()

    # Load normalizer to denormalize
    data_state_path = Path(args.checkpoint).parent / "data_state.pt"
    if data_state_path.exists():
        data_state = torch.load(data_state_path, map_location="cpu")
        if "data_module" in data_state and data_state["data_module"].get("normalizer"):
            from src.data.preprocessing import ReturnNormalizer
            normalizer = ReturnNormalizer()
            normalizer.load_state_dict(data_state["data_module"]["normalizer"])
            samples = normalizer.inverse_transform(samples)
            logger.info("Denormalized samples using saved normalizer")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(samples)
    df.columns = [f"t_{i}" for i in range(samples.shape[1])]
    df.index.name = "sample_id"
    df.to_csv(output_path)

    logger.info(f"Saved {args.n_samples} samples to {output_path}")

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Mean daily return: {samples.mean():.6f}")
    logger.info(f"  Std daily return: {samples.std():.6f}")
    logger.info(f"  Annualized return: {samples.mean() * 252:.2%}")
    logger.info(f"  Annualized volatility: {samples.std() * np.sqrt(252):.2%}")


if __name__ == "__main__":
    main()
