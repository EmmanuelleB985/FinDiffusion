#!/usr/bin/env python
"""Evaluation script for FinDiffusion."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FinancialDiffusion
from src.data import FinancialDataModule
from src.evaluation import (
    StylizedFactsValidator,
    validate_stylized_facts,
    compute_all_metrics,
    print_metrics_report,
)
from src.evaluation.stylized_facts import compare_distributions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_data(checkpoint_path: str, config_path: str, device: torch.device):
    """Load model and data module."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load data module
    data_module = FinancialDataModule(
        tickers=config["data"]["tickers"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        seq_len=config["data"]["seq_len"],
        stride=config["data"]["stride"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        batch_size=config["evaluation"]["batch_size"],
        data_dir=config["paths"]["data_dir"],
    )
    data_module.setup()

    return model, data_module, config


@torch.no_grad()
def generate_samples(
    model: FinancialDiffusion,
    n_samples: int,
    conditions: dict = None,
    device: torch.device = None,
    use_ddim: bool = False,
) -> np.ndarray:
    """Generate synthetic samples."""
    logger.info(f"Generating {n_samples} synthetic samples...")
    
    samples = model.generate(
        n_samples=n_samples,
        conditions=conditions,
        use_ddim=use_ddim,
        ddim_steps=50 if use_ddim else None,
        device=device,
        progress=True,
    )
    
    return samples.cpu().numpy()


def get_real_samples(data_module: FinancialDataModule, n_samples: int) -> np.ndarray:
    """Get real samples from test set."""
    test_dataset = data_module.test_dataset
    
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)
    samples = [test_dataset[i].numpy() for i in indices]
    
    # Denormalize
    samples = np.array(samples)
    samples = data_module.denormalize(samples)
    
    return samples


def evaluate_stylized_facts(real: np.ndarray, synthetic: np.ndarray, output_dir: Path):
    """Evaluate stylized facts for both real and synthetic data."""
    logger.info("Evaluating stylized facts...")
    
    # Validate real data
    real_results = validate_stylized_facts(real)
    logger.info("Real data stylized facts:")
    for test_name, result in real_results.items():
        if test_name != "summary":
            logger.info(f"  {test_name}: {'PASS' if result['passed'] else 'FAIL'}")
    
    # Validate synthetic data
    syn_results = validate_stylized_facts(synthetic)
    logger.info("Synthetic data stylized facts:")
    for test_name, result in syn_results.items():
        if test_name != "summary":
            logger.info(f"  {test_name}: {'PASS' if result['passed'] else 'FAIL'}")
    
    # Compare distributions
    comparison = compare_distributions(real, synthetic)
    
    return {
        "real": real_results,
        "synthetic": syn_results,
        "comparison": comparison,
    }


def evaluate_conditional_generation(
    model: FinancialDiffusion,
    data_module: FinancialDataModule,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Evaluate conditional generation accuracy."""
    logger.info("Evaluating conditional generation...")
    
    results = {}
    
    # Test different trend conditions
    trend_targets = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    trend_results = []
    
    for target_trend in trend_targets:
        samples = generate_samples(
            model,
            n_samples=500,
            conditions={"trend": target_trend, "volatility": 0.2, "regime": "sideways"},
            device=device,
        )
        samples = data_module.denormalize(samples)
        
        # Calculate realized trends
        realized = np.sum(samples, axis=1) * (252 / samples.shape[1])
        mean_realized = np.mean(realized)
        std_realized = np.std(realized)
        
        trend_results.append({
            "target": target_trend,
            "realized_mean": float(mean_realized),
            "realized_std": float(std_realized),
            "error": float(abs(mean_realized - target_trend)),
        })
        
        logger.info(f"  Trend {target_trend:.1%}: realized {mean_realized:.1%} ± {std_realized:.1%}")
    
    results["trend_alignment"] = trend_results
    
    # Test different volatility conditions
    vol_targets = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    vol_results = []
    
    for target_vol in vol_targets:
        samples = generate_samples(
            model,
            n_samples=500,
            conditions={"trend": 0.0, "volatility": target_vol, "regime": "sideways"},
            device=device,
        )
        samples = data_module.denormalize(samples)
        
        # Calculate realized volatilities
        realized = np.std(samples, axis=1) * np.sqrt(252)
        mean_realized = np.mean(realized)
        std_realized = np.std(realized)
        
        vol_results.append({
            "target": target_vol,
            "realized_mean": float(mean_realized),
            "realized_std": float(std_realized),
            "error": float(abs(mean_realized - target_vol)),
        })
        
        logger.info(f"  Vol {target_vol:.0%}: realized {mean_realized:.0%} ± {std_realized:.0%}")
    
    results["volatility_alignment"] = vol_results
    
    return results


def create_visualizations(
    real: np.ndarray,
    synthetic: np.ndarray,
    output_dir: Path,
):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Return distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(real.flatten(), bins=100, alpha=0.7, label="Real", density=True)
    axes[0].hist(synthetic.flatten(), bins=100, alpha=0.7, label="Synthetic", density=True)
    axes[0].set_xlabel("Returns")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Return Distribution")
    axes[0].legend()
    axes[0].set_xlim(-0.1, 0.1)
    
    # Q-Q plot
    from scipy import stats
    real_sorted = np.sort(real.flatten())
    syn_sorted = np.sort(synthetic.flatten())
    n = min(len(real_sorted), len(syn_sorted))
    axes[1].scatter(real_sorted[:n:max(1, n//1000)], syn_sorted[:n:max(1, n//1000)], alpha=0.5, s=5)
    lims = [min(real_sorted.min(), syn_sorted.min()), max(real_sorted.max(), syn_sorted.max())]
    axes[1].plot(lims, lims, 'r--', alpha=0.8)
    axes[1].set_xlabel("Real Quantiles")
    axes[1].set_ylabel("Synthetic Quantiles")
    axes[1].set_title("Q-Q Plot")
    
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_comparison.png", dpi=150)
    plt.close()
    
    # 2. Sample paths
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i in range(2):
        # Cumulative returns
        cum_real = np.cumprod(1 + real[i])
        cum_syn = np.cumprod(1 + synthetic[i])
        
        axes[i, 0].plot(cum_real, label="Real", alpha=0.8)
        axes[i, 0].plot(cum_syn, label="Synthetic", alpha=0.8)
        axes[i, 0].set_title(f"Sample Path {i+1}")
        axes[i, 0].set_xlabel("Time")
        axes[i, 0].set_ylabel("Cumulative Return")
        axes[i, 0].legend()
        
        # Rolling volatility
        window = 21
        vol_real = np.convolve(np.abs(real[i]), np.ones(window)/window, mode='valid')
        vol_syn = np.convolve(np.abs(synthetic[i]), np.ones(window)/window, mode='valid')
        
        axes[i, 1].plot(vol_real * np.sqrt(252), label="Real", alpha=0.8)
        axes[i, 1].plot(vol_syn * np.sqrt(252), label="Synthetic", alpha=0.8)
        axes[i, 1].set_title(f"Rolling Volatility {i+1}")
        axes[i, 1].set_xlabel("Time")
        axes[i, 1].set_ylabel("Annualized Vol")
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_paths.png", dpi=150)
    plt.close()
    
    # 3. Autocorrelation comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    max_lag = 20
    
    # ACF of returns
    acf_real = [np.corrcoef(real.flatten()[:-lag], real.flatten()[lag:])[0,1] 
                for lag in range(1, max_lag+1)]
    acf_syn = [np.corrcoef(synthetic.flatten()[:-lag], synthetic.flatten()[lag:])[0,1] 
               for lag in range(1, max_lag+1)]
    
    axes[0].bar(np.arange(1, max_lag+1) - 0.2, acf_real, width=0.4, label="Real", alpha=0.7)
    axes[0].bar(np.arange(1, max_lag+1) + 0.2, acf_syn, width=0.4, label="Synthetic", alpha=0.7)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")
    axes[0].set_title("Autocorrelation of Returns")
    axes[0].legend()
    
    # ACF of squared returns
    real_sq = real.flatten() ** 2
    syn_sq = synthetic.flatten() ** 2
    acf_sq_real = [np.corrcoef(real_sq[:-lag], real_sq[lag:])[0,1] 
                   for lag in range(1, max_lag+1)]
    acf_sq_syn = [np.corrcoef(syn_sq[:-lag], syn_sq[lag:])[0,1] 
                  for lag in range(1, max_lag+1)]
    
    axes[1].bar(np.arange(1, max_lag+1) - 0.2, acf_sq_real, width=0.4, label="Real", alpha=0.7)
    axes[1].bar(np.arange(1, max_lag+1) + 0.2, acf_sq_syn, width=0.4, label="Synthetic", alpha=0.7)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF")
    axes[1].set_title("Autocorrelation of Squared Returns (Vol Clustering)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "autocorrelation.png", dpi=150)
    plt.close()
    
    logger.info(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinDiffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of samples to generate for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for output files",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="Use DDIM for faster sampling",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, data_module, config = load_model_and_data(args.checkpoint, args.config, device)

    # Get real samples
    real_samples = get_real_samples(data_module, args.n_samples)
    logger.info(f"Got {len(real_samples)} real samples")

    # Generate synthetic samples (unconditional)
    synthetic_samples = generate_samples(
        model,
        n_samples=args.n_samples,
        device=device,
        use_ddim=args.ddim,
    )
    synthetic_samples = data_module.denormalize(synthetic_samples)
    logger.info(f"Generated {len(synthetic_samples)} synthetic samples")

    # Evaluate stylized facts
    sf_results = evaluate_stylized_facts(real_samples, synthetic_samples, output_dir)

    # Compute all metrics
    metrics = compute_all_metrics(real_samples, synthetic_samples)
    logger.info("\n" + print_metrics_report(metrics))

    # Evaluate conditional generation
    cond_results = evaluate_conditional_generation(model, data_module, device, output_dir)

    # Create visualizations
    create_visualizations(real_samples, synthetic_samples, output_dir)

    # Save results
    all_results = {
        "stylized_facts": sf_results,
        "metrics": metrics,
        "conditional": cond_results,
        "config": {
            "n_samples": args.n_samples,
            "checkpoint": args.checkpoint,
        }
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Saved results to {output_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
