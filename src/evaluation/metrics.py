"""Evaluation metrics for synthetic financial data."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


def distribution_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
) -> Dict[str, float]:
    """
    Compute distribution similarity metrics.
    
    Args:
        real: Real data samples
        synthetic: Synthetic data samples
    
    Returns:
        Dict of metric names to values
    """
    real = real.flatten()
    synthetic = synthetic.flatten()
    
    # Remove NaN
    real = real[~np.isnan(real)]
    synthetic = synthetic[~np.isnan(synthetic)]
    
    metrics = {}
    
    # Wasserstein distance (Earth Mover's Distance)
    metrics["wasserstein"] = stats.wasserstein_distance(real, synthetic)
    
    # Kolmogorov-Smirnov statistic
    ks_stat, _ = stats.ks_2samp(real, synthetic)
    metrics["ks_statistic"] = ks_stat
    
    # Jensen-Shannon divergence (via histograms)
    bins = np.linspace(
        min(real.min(), synthetic.min()),
        max(real.max(), synthetic.max()),
        100,
    )
    hist_real, _ = np.histogram(real, bins=bins, density=True)
    hist_syn, _ = np.histogram(synthetic, bins=bins, density=True)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_real = hist_real + eps
    hist_syn = hist_syn + eps
    hist_real /= hist_real.sum()
    hist_syn /= hist_syn.sum()
    
    m = 0.5 * (hist_real + hist_syn)
    js_div = 0.5 * (stats.entropy(hist_real, m) + stats.entropy(hist_syn, m))
    metrics["js_divergence"] = js_div
    
    # Moment differences
    metrics["mean_diff"] = abs(np.mean(synthetic) - np.mean(real))
    metrics["std_diff"] = abs(np.std(synthetic) - np.std(real))
    metrics["skew_diff"] = abs(stats.skew(synthetic) - stats.skew(real))
    metrics["kurtosis_diff"] = abs(stats.kurtosis(synthetic) - stats.kurtosis(real))
    
    return metrics


def temporal_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
    max_lag: int = 20,
) -> Dict[str, float]:
    """
    Compute temporal structure metrics.
    
    Args:
        real: Real data (T,) or (N, T)
        synthetic: Synthetic data (T,) or (N, T)
        max_lag: Maximum lag for autocorrelation
    
    Returns:
        Dict of metric names to values
    """
    # Handle 2D arrays by averaging
    if real.ndim > 1:
        real = real.mean(axis=0)
    if synthetic.ndim > 1:
        synthetic = synthetic.mean(axis=0)
    
    metrics = {}
    
    # Autocorrelation comparison (raw returns)
    acf_real = _compute_acf(real, max_lag)
    acf_syn = _compute_acf(synthetic, max_lag)
    metrics["acf_mae"] = np.mean(np.abs(acf_syn - acf_real))
    
    # Autocorrelation of squared returns (volatility clustering)
    acf_sq_real = _compute_acf(real ** 2, max_lag)
    acf_sq_syn = _compute_acf(synthetic ** 2, max_lag)
    metrics["acf_squared_mae"] = np.mean(np.abs(acf_sq_syn - acf_sq_real))
    
    # Volatility clustering strength
    metrics["vol_cluster_real"] = acf_sq_real[1] if len(acf_sq_real) > 1 else 0
    metrics["vol_cluster_syn"] = acf_sq_syn[1] if len(acf_sq_syn) > 1 else 0
    
    return metrics


def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(x)
    x = x - np.mean(x)
    
    acf = np.zeros(max_lag)
    var = np.var(x)
    
    if var < 1e-10:
        return acf
    
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        acf[lag - 1] = np.mean(x[:-lag] * x[lag:]) / var
    
    return acf


def conditional_metrics(
    synthetic: np.ndarray,
    conditions: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate how well synthetic data matches specified conditions.
    
    Args:
        synthetic: Generated returns (N, T)
        conditions: Dict with 'trend', 'volatility', etc.
    
    Returns:
        Dict of condition alignment metrics
    """
    metrics = {}
    
    # Trend alignment
    if "trend" in conditions:
        target_trend = conditions["trend"]
        if isinstance(target_trend, np.ndarray):
            target_trend = target_trend.flatten()
        
        # Compute realized trends
        realized_trends = []
        for i in range(len(synthetic)):
            cum_ret = np.sum(synthetic[i])
            ann_ret = cum_ret * (252 / len(synthetic[i]))
            realized_trends.append(ann_ret)
        
        realized_trends = np.array(realized_trends)
        
        if isinstance(target_trend, np.ndarray) and len(target_trend) == len(realized_trends):
            trend_error = np.mean(np.abs(realized_trends - target_trend))
        else:
            trend_error = np.mean(np.abs(realized_trends - target_trend))
        
        metrics["trend_mae"] = trend_error
        metrics["trend_corr"] = np.corrcoef(realized_trends[:len(target_trend)], target_trend[:len(realized_trends)])[0, 1] if isinstance(target_trend, np.ndarray) else 0
    
    # Volatility alignment
    if "volatility" in conditions:
        target_vol = conditions["volatility"]
        if isinstance(target_vol, np.ndarray):
            target_vol = target_vol.flatten()
        
        realized_vols = []
        for i in range(len(synthetic)):
            daily_vol = np.std(synthetic[i])
            ann_vol = daily_vol * np.sqrt(252)
            realized_vols.append(ann_vol)
        
        realized_vols = np.array(realized_vols)
        
        if isinstance(target_vol, np.ndarray) and len(target_vol) == len(realized_vols):
            vol_error = np.mean(np.abs(realized_vols - target_vol))
        else:
            vol_error = np.mean(np.abs(realized_vols - target_vol))
        
        metrics["volatility_mae"] = vol_error
    
    return metrics


def diversity_metrics(synthetic: np.ndarray) -> Dict[str, float]:
    """
    Evaluate diversity of generated samples.
    
    Args:
        synthetic: Generated returns (N, T)
    
    Returns:
        Dict of diversity metrics
    """
    metrics = {}
    
    # Pairwise distances
    n_samples = min(len(synthetic), 1000)  # Limit for computational efficiency
    indices = np.random.choice(len(synthetic), n_samples, replace=False)
    samples = synthetic[indices]
    
    # Compute pairwise correlations
    corr_matrix = np.corrcoef(samples)
    upper_tri = corr_matrix[np.triu_indices(n_samples, k=1)]
    
    metrics["mean_pairwise_corr"] = np.mean(upper_tri)
    metrics["std_pairwise_corr"] = np.std(upper_tri)
    
    # Sample variance (diversity measure)
    sample_stds = np.std(samples, axis=1)
    metrics["mean_sample_std"] = np.mean(sample_stds)
    metrics["std_sample_std"] = np.std(sample_stds)
    
    # Unique samples (approximate via binning)
    binned = np.round(samples * 100).astype(int)
    unique_ratio = len(np.unique(binned, axis=0)) / len(binned)
    metrics["unique_ratio"] = unique_ratio
    
    return metrics


def compute_all_metrics(
    real: np.ndarray,
    synthetic: np.ndarray,
    conditions: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict]:
    """
    Compute all evaluation metrics.
    
    Args:
        real: Real data
        synthetic: Synthetic data
        conditions: Optional conditioning information
    
    Returns:
        Dict with all metrics organized by category
    """
    results = {}
    
    # Distribution metrics
    results["distribution"] = distribution_metrics(real, synthetic)
    
    # Temporal metrics
    results["temporal"] = temporal_metrics(real, synthetic)
    
    # Diversity metrics
    results["diversity"] = diversity_metrics(synthetic)
    
    # Conditional metrics (if conditions provided)
    if conditions is not None:
        results["conditional"] = conditional_metrics(synthetic, conditions)
    
    # Compute summary score
    dist_score = 1.0 / (1.0 + results["distribution"]["wasserstein"])
    temp_score = 1.0 / (1.0 + results["temporal"]["acf_squared_mae"])
    div_score = results["diversity"]["unique_ratio"]
    
    results["summary"] = {
        "distribution_score": dist_score,
        "temporal_score": temp_score,
        "diversity_score": div_score,
        "overall_score": (dist_score + temp_score + div_score) / 3,
    }
    
    return results


def print_metrics_report(metrics: Dict) -> str:
    """Format metrics as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 60)
    
    for category, values in metrics.items():
        lines.append(f"\n{category.upper()}")
        lines.append("-" * 40)
        
        if isinstance(values, dict):
            for name, value in values.items():
                if isinstance(value, float):
                    lines.append(f"  {name}: {value:.6f}")
                else:
                    lines.append(f"  {name}: {value}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
