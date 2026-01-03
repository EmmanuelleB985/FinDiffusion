#!/usr/bin/env python
"""Training script for FinDiffusion."""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FinancialDiffusion
from src.data import FinancialDataModule
from src.training import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train FinDiffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (small dataset, few epochs)",
    )
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override with command line args
    seed = args.seed if args.seed is not None else config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Debug mode overrides
    if args.debug:
        config["training"]["epochs"] = 2
        config["data"]["tickers"] = config["data"]["tickers"][:5]
        config["logging"]["log_every"] = 10
        logger.info("Debug mode enabled")

    # Setup data
    logger.info("Setting up data module...")
    data_module = FinancialDataModule(
        tickers=config["data"]["tickers"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        seq_len=config["data"]["seq_len"],
        stride=config["data"]["stride"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        batch_size=config["training"]["batch_size"],
        data_dir=config["paths"]["data_dir"],
    )
    data_module.setup()

    # Create model
    logger.info("Creating model...")
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
        dropout=config["model"]["dropout"],
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Estimate memory usage
    param_memory_mb = n_params * 4 / 1024 / 1024  # float32
    logger.info(f"Estimated parameter memory: {param_memory_mb:.1f} MB")

    # Setup trainer
    training_config = TrainingConfig(
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        betas=tuple(config["training"]["betas"]),
        clip_grad_norm=config["training"]["clip_grad_norm"],
        warmup_epochs=config["training"]["warmup_epochs"],
        min_lr=config["training"]["min_lr"],
        use_amp=config["training"]["use_amp"],
        log_every=config["logging"]["log_every"],
        sample_every=config["logging"]["sample_every"],
        save_every=config["training"]["save_every"],
        drop_cond_prob=0.1,
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        use_wandb=args.wandb or config["logging"]["use_wandb"],
        project=config["logging"]["project"],
    )

    trainer = Trainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        config=training_config,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save data module state
    state = {
        "data_module": data_module.state_dict(),
        "config": config,
    }
    torch.save(state, Path(config["paths"]["checkpoint_dir"]) / "data_state.pt")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
