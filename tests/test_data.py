"""Tests for data utilities."""

import pytest
import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    ReturnNormalizer,
    compute_returns,
    create_windows,
    remove_outliers,
)
from src.data.dataset import FinancialDataset


class TestReturnNormalizer:
    """Test return normalizer."""

    def test_fit_transform(self):
        data = np.random.randn(1000) * 0.02  # Typical daily returns scale
        normalizer = ReturnNormalizer(method="robust")

        normalized = normalizer.fit_transform(data)

        assert normalized.shape == data.shape
        assert np.abs(np.median(normalized)) < 0.5  # Should be centered

    def test_inverse_transform(self):
        data = np.random.randn(1000) * 0.02
        normalizer = ReturnNormalizer(method="robust")

        normalized = normalizer.fit_transform(data)
        recovered = normalizer.inverse_transform(normalized)

        np.testing.assert_allclose(recovered, data, rtol=1e-5)

    def test_clipping(self):
        data = np.array([-10, -1, 0, 1, 10])
        normalizer = ReturnNormalizer(method="standard", clip_range=(-2, 2))

        normalized = normalizer.fit_transform(data)

        assert normalized.min() >= -2
        assert normalized.max() <= 2

    def test_state_dict(self):
        data = np.random.randn(1000)
        normalizer1 = ReturnNormalizer(method="robust")
        normalizer1.fit(data)

        state = normalizer1.state_dict()

        normalizer2 = ReturnNormalizer()
        normalizer2.load_state_dict(state)

        assert normalizer2.center_ == normalizer1.center_
        assert normalizer2.scale_ == normalizer1.scale_


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_compute_log_returns(self):
        prices = pd.DataFrame({
            "A": [100, 101, 102, 103],
            "B": [50, 51, 50.5, 52],
        })

        returns = compute_returns(prices, method="log")

        assert len(returns) == 3  # One less due to differencing
        assert returns.shape[1] == 2

    def test_compute_simple_returns(self):
        prices = pd.DataFrame({"A": [100, 110, 99]})

        returns = compute_returns(prices, method="simple")

        expected = np.array([0.1, -0.1])
        np.testing.assert_allclose(returns["A"].values, expected, rtol=1e-5)

    def test_create_windows(self):
        data = np.arange(100).astype(float)
        window_size = 10
        stride = 5

        windows = create_windows(data, window_size, stride)

        expected_n_windows = (100 - 10) // 5 + 1
        assert windows.shape == (expected_n_windows, window_size)
        
        # Check first window
        np.testing.assert_array_equal(windows[0], np.arange(10))
        
        # Check second window (with stride)
        np.testing.assert_array_equal(windows[1], np.arange(5, 15))

    def test_create_windows_2d(self):
        data = np.random.randn(100, 3)
        windows = create_windows(data, window_size=10, stride=5)

        assert windows.shape[1] == 10
        assert windows.shape[2] == 3

    def test_remove_outliers_clip(self):
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.5, -0.5, 0.01]})  # 0.5 are outliers

        cleaned = remove_outliers(returns, threshold=2.0, method="clip")

        assert cleaned["A"].max() < 0.5
        assert cleaned["A"].min() > -0.5


class TestFinancialDataset:
    """Test financial dataset."""

    @pytest.fixture
    def sample_returns(self):
        # Create synthetic return data
        np.random.seed(42)
        return np.random.randn(500) * 0.02

    def test_dataset_creation(self, sample_returns):
        dataset = FinancialDataset(
            returns=sample_returns,
            seq_len=50,
            stride=10,
        )

        assert len(dataset) > 0

    def test_dataset_getitem(self, sample_returns):
        dataset = FinancialDataset(
            returns=sample_returns,
            seq_len=50,
            stride=10,
        )

        sample = dataset[0]

        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (50,)
        assert sample.dtype == torch.float32

    def test_dataset_normalization(self, sample_returns):
        dataset = FinancialDataset(
            returns=sample_returns,
            seq_len=50,
            stride=10,
        )

        # Check that data is normalized
        sample = dataset[0]
        assert sample.abs().mean() < 5  # Should be normalized

        # Check that raw data is different
        raw = dataset.get_raw(0)
        assert not np.allclose(sample.numpy(), raw)

    def test_dataset_multiasset(self):
        returns = np.random.randn(500, 5) * 0.02  # 5 assets

        dataset = FinancialDataset(
            returns=returns,
            seq_len=50,
            stride=10,
        )

        # Should have more samples due to multiple assets
        assert len(dataset) > (500 - 50) // 10


class TestIntegration:
    """Integration tests for data pipeline."""

    def test_full_pipeline(self):
        # Simulate downloaded price data
        np.random.seed(42)
        n_days = 500
        n_assets = 3

        prices = pd.DataFrame(
            np.cumprod(1 + np.random.randn(n_days, n_assets) * 0.02, axis=0) * 100,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

        # Compute returns
        returns = compute_returns(prices, method="log")

        # Clean
        returns = remove_outliers(returns, threshold=3.0)

        # Create dataset
        dataset = FinancialDataset(
            returns=returns.values,
            seq_len=50,
            stride=10,
        )

        # Get a batch
        batch = torch.stack([dataset[i] for i in range(4)])

        assert batch.shape == (4, 50)
        assert not torch.isnan(batch).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
