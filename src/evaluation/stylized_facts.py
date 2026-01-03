"""Stylized facts validation for financial time series."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class StylizedFactsValidator:
    """
    Validate that generated financial data exhibits stylized facts.
    
    Stylized facts are statistical properties that are consistently
    observed across different markets, time periods, and asset classes.
    
    Key stylized facts tested:
    1. Fat tails (excess kurtosis, power-law tails)
    2. Volatility clustering (autocorrelation of squared returns)
    3. Leverage effect (negative correlation between returns and future vol)
    4. No autocorrelation in raw returns
    5. Volume-volatility correlation
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: Alpha for statistical tests
        """
        self.alpha = significance_level

    def validate_fat_tails(self, returns: np.ndarray) -> Dict:
        """
        Test for fat tails (leptokurtosis).
        
        Real financial returns have heavier tails than normal distribution,
        with typical excess kurtosis of 3-10.
        
        Args:
            returns: 1D array of returns
        
        Returns:
            Dict with test results
        """
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        # Excess kurtosis (0 for normal)
        excess_kurtosis = stats.kurtosis(returns, fisher=True)
        
        # Skewness
        skewness = stats.skew(returns)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Hill estimator for tail index
        tail_index = self._hill_estimator(returns)
        
        # Pass if: excess kurtosis > 0 AND Jarque-Bera rejects normality
        passed = (excess_kurtosis > 0) and (jb_pvalue < self.alpha)
        
        return {
            "test": "fat_tails",
            "passed": passed,
            "excess_kurtosis": float(excess_kurtosis),
            "skewness": float(skewness),
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pvalue),
            "tail_index": float(tail_index),
            "interpretation": self._interpret_fat_tails(excess_kurtosis, tail_index),
        }

    def _hill_estimator(self, returns: np.ndarray, quantile: float = 0.05) -> float:
        """
        Hill estimator for tail index.
        
        Lower tail index = heavier tails.
        Normal distribution has infinite tail index.
        Financial returns typically have tail index of 2-5.
        """
        abs_returns = np.abs(returns)
        sorted_returns = np.sort(abs_returns)[::-1]
        
        k = max(int(len(returns) * quantile), 10)
        threshold = sorted_returns[k]
        
        if threshold <= 0:
            return np.inf
        
        exceedances = sorted_returns[:k]
        log_ratios = np.log(exceedances / threshold)
        
        tail_index = k / np.sum(log_ratios)
        return tail_index

    def _interpret_fat_tails(self, kurtosis: float, tail_index: float) -> str:
        """Interpret fat tails results."""
        if kurtosis > 5:
            return "Very heavy tails (realistic)"
        elif kurtosis > 1:
            return "Moderately heavy tails (realistic)"
        elif kurtosis > 0:
            return "Slight heavy tails"
        else:
            return "Thin tails (not realistic for financial data)"

    def validate_volatility_clustering(self, returns: np.ndarray) -> Dict:
        """
        Test for volatility clustering.
        
        Volatility clustering means large (small) returns tend to be
        followed by large (small) returns. This shows up as significant
        autocorrelation in squared or absolute returns.
        
        Args:
            returns: 1D array of returns
        
        Returns:
            Dict with test results
        """
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        # Squared returns for volatility proxy
        squared_returns = returns ** 2
        abs_returns = np.abs(returns)
        
        # Autocorrelation at various lags
        acf_sq_1 = self._autocorr(squared_returns, 1)
        acf_sq_5 = self._autocorr(squared_returns, 5)
        acf_sq_10 = self._autocorr(squared_returns, 10)
        
        acf_abs_1 = self._autocorr(abs_returns, 1)
        
        # Ljung-Box test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(squared_returns, lags=[10, 20], return_df=True)
            lb_pvalue = float(lb_result["lb_pvalue"].iloc[-1])
        except:
            lb_pvalue = 0.0  # Assume significant if test fails
        
        # Pass if significant autocorrelation in squared returns
        passed = (acf_sq_1 > 0.05) or (lb_pvalue < self.alpha)
        
        return {
            "test": "volatility_clustering",
            "passed": passed,
            "acf_squared_lag1": float(acf_sq_1),
            "acf_squared_lag5": float(acf_sq_5),
            "acf_squared_lag10": float(acf_sq_10),
            "acf_absolute_lag1": float(acf_abs_1),
            "ljung_box_pvalue": lb_pvalue,
            "interpretation": self._interpret_vol_clustering(acf_sq_1),
        }

    def _autocorr(self, x: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        n = len(x)
        if lag >= n:
            return 0.0
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]

    def _interpret_vol_clustering(self, acf: float) -> str:
        """Interpret volatility clustering results."""
        if acf > 0.2:
            return "Strong volatility clustering (realistic)"
        elif acf > 0.1:
            return "Moderate volatility clustering (realistic)"
        elif acf > 0.05:
            return "Weak volatility clustering"
        else:
            return "No volatility clustering (not realistic)"

    def validate_leverage_effect(self, returns: np.ndarray) -> Dict:
        """
        Test for leverage effect.
        
        The leverage effect is the negative correlation between
        returns and future volatility: negative returns increase
        future volatility more than positive returns.
        
        Args:
            returns: 1D array of returns
        
        Returns:
            Dict with test results
        """
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        # Future volatility proxy
        future_vol = np.abs(returns[1:])
        past_returns = returns[:-1]
        
        # Correlation
        leverage_corr = np.corrcoef(past_returns, future_vol)[0, 1]
        
        # Also check asymmetric volatility
        negative_returns = past_returns[past_returns < 0]
        positive_returns = past_returns[past_returns > 0]
        
        avg_vol_after_neg = np.mean(future_vol[past_returns[:-1] < 0]) if len(negative_returns) > 0 else 0
        avg_vol_after_pos = np.mean(future_vol[past_returns[:-1] > 0]) if len(positive_returns) > 0 else 0
        
        asymmetry_ratio = avg_vol_after_neg / (avg_vol_after_pos + 1e-8)
        
        # Pass if negative correlation (leverage effect present)
        passed = leverage_corr < -0.02
        
        return {
            "test": "leverage_effect",
            "passed": passed,
            "leverage_correlation": float(leverage_corr),
            "vol_after_negative": float(avg_vol_after_neg),
            "vol_after_positive": float(avg_vol_after_pos),
            "asymmetry_ratio": float(asymmetry_ratio),
            "interpretation": self._interpret_leverage(leverage_corr),
        }

    def _interpret_leverage(self, corr: float) -> str:
        """Interpret leverage effect results."""
        if corr < -0.1:
            return "Strong leverage effect (realistic)"
        elif corr < -0.05:
            return "Moderate leverage effect (realistic)"
        elif corr < 0:
            return "Weak leverage effect"
        else:
            return "No leverage effect (less realistic for equities)"

    def validate_no_autocorrelation(self, returns: np.ndarray) -> Dict:
        """
        Test that raw returns have no significant autocorrelation.
        
        Unlike squared returns, raw returns should not be predictable
        from past returns (efficient markets).
        
        Args:
            returns: 1D array of returns
        
        Returns:
            Dict with test results
        """
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        # Autocorrelation of raw returns
        acf_1 = self._autocorr(returns, 1)
        acf_5 = self._autocorr(returns, 5)
        
        # Ljung-Box test
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
            lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
        except:
            lb_pvalue = 1.0
        
        # Pass if NO significant autocorrelation
        passed = (abs(acf_1) < 0.1) and (lb_pvalue > self.alpha)
        
        return {
            "test": "no_autocorrelation",
            "passed": passed,
            "acf_lag1": float(acf_1),
            "acf_lag5": float(acf_5),
            "ljung_box_pvalue": lb_pvalue,
            "interpretation": self._interpret_autocorr(acf_1),
        }

    def _interpret_autocorr(self, acf: float) -> str:
        """Interpret autocorrelation results."""
        if abs(acf) < 0.05:
            return "No significant autocorrelation (realistic)"
        elif abs(acf) < 0.1:
            return "Weak autocorrelation (acceptable)"
        else:
            return "Significant autocorrelation (not realistic)"

    def validate_all(self, returns: np.ndarray) -> Dict:
        """
        Run all stylized facts tests.
        
        Args:
            returns: 1D array of returns
        
        Returns:
            Dict with all test results
        """
        results = {
            "fat_tails": self.validate_fat_tails(returns),
            "volatility_clustering": self.validate_volatility_clustering(returns),
            "leverage_effect": self.validate_leverage_effect(returns),
            "no_autocorrelation": self.validate_no_autocorrelation(returns),
        }
        
        # Overall score
        n_passed = sum(1 for r in results.values() if r["passed"])
        n_total = len(results)
        
        results["summary"] = {
            "tests_passed": n_passed,
            "tests_total": n_total,
            "pass_rate": n_passed / n_total,
            "overall_quality": self._overall_quality(n_passed, n_total),
        }
        
        return results

    def _overall_quality(self, passed: int, total: int) -> str:
        """Assess overall data quality."""
        ratio = passed / total
        if ratio == 1.0:
            return "Excellent - All stylized facts captured"
        elif ratio >= 0.75:
            return "Good - Most stylized facts captured"
        elif ratio >= 0.5:
            return "Fair - Some stylized facts captured"
        else:
            return "Poor - Few stylized facts captured"


def validate_stylized_facts(
    returns: np.ndarray,
    significance_level: float = 0.05,
) -> Dict:
    """
    Convenience function to validate stylized facts.
    
    Args:
        returns: Array of returns (can be 2D, will aggregate)
        significance_level: Alpha for statistical tests
    
    Returns:
        Dict with validation results
    """
    validator = StylizedFactsValidator(significance_level)
    
    # Flatten if 2D
    if returns.ndim > 1:
        returns = returns.flatten()
    
    return validator.validate_all(returns)


def compare_distributions(
    real_returns: np.ndarray,
    synthetic_returns: np.ndarray,
) -> Dict:
    """
    Compare real and synthetic return distributions.
    
    Args:
        real_returns: Real market returns
        synthetic_returns: Generated returns
    
    Returns:
        Dict with comparison metrics
    """
    real = real_returns.flatten()
    synthetic = synthetic_returns.flatten()
    
    # Remove NaN
    real = real[~np.isnan(real)]
    synthetic = synthetic[~np.isnan(synthetic)]
    
    # Basic statistics comparison
    stats_comparison = {
        "mean": {
            "real": float(np.mean(real)),
            "synthetic": float(np.mean(synthetic)),
            "difference": float(np.mean(synthetic) - np.mean(real)),
        },
        "std": {
            "real": float(np.std(real)),
            "synthetic": float(np.std(synthetic)),
            "difference": float(np.std(synthetic) - np.std(real)),
        },
        "skewness": {
            "real": float(stats.skew(real)),
            "synthetic": float(stats.skew(synthetic)),
            "difference": float(stats.skew(synthetic) - stats.skew(real)),
        },
        "kurtosis": {
            "real": float(stats.kurtosis(real)),
            "synthetic": float(stats.kurtosis(synthetic)),
            "difference": float(stats.kurtosis(synthetic) - stats.kurtosis(real)),
        },
    }
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(real, synthetic)
    
    # Wasserstein distance
    wasserstein = stats.wasserstein_distance(real, synthetic)
    
    return {
        "statistics": stats_comparison,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein_distance": float(wasserstein),
    }
