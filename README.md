# FinDiffusion: Conditional Diffusion Models for Synthetic Financial Time Series

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of conditional diffusion models for generating realistic synthetic financial time series. 

## Key Features

- **Conditional Generation**: Control trend, volatility, and market regime of generated data
- **Stylized Facts Preservation**: Captures fat tails, volatility clustering, and leverage effects
- **Multi-Asset Support**: Generate correlated multi-asset price paths
- **Memory Efficient**: Optimized for single GPU (20GB VRAM)

### Installation

```bash
git clone https://github.com/EmmanuelleB985/FinDiffusion.git
cd FinDiffusion
pip install -e .
```

### Generate Synthetic Data (5 minutes)

```python
from src.models import FinancialDiffusion
from src.data import FinancialDataModule

# Load pretrained model
model = FinancialDiffusion.load_from_checkpoint("checkpoints/best.ckpt")

# Generate 1000 synthetic daily return paths (252 days each)
conditions = {
    "trend": 0.10,      # 10% annual return
    "volatility": 0.20,  # 20% annual vol
    "regime": "bull"     # Market regime
}
synthetic_returns = model.generate(n_samples=1000, seq_len=252, conditions=conditions)
```

### Train 

```bash
# Download and prepare data
python scripts/download_data.py --tickers SP500 --start 2010-01-01 --end 2024-01-01

# Train model
python scripts/train.py --config configs/default.yaml --gpus 1

# Evaluate on stylized facts
python scripts/evaluate.py --checkpoint checkpoints/best.ckpt
```

## Results

### Stylized Facts Comparison (S&P 500 Components)

| Metric | Real Data | FinDiffusion | GAN Baseline | VAE Baseline |
|--------|-----------|--------------|--------------|--------------|
| Excess Kurtosis | 4.82 | **4.51** | 2.13 | 1.87 |
| Vol Clustering (ACF₁) | 0.23 | **0.21** | 0.08 | 0.05 |
| Leverage Effect | -0.12 | **-0.09** | -0.02 | 0.01 |
| Tail Index (α) | 3.1 | **3.3** | 5.2 | 6.1 |

### Conditional Generation Accuracy

| Condition | Target | Generated (Mean ± Std) |
|-----------|--------|------------------------|
| Trend = +20% | +20.0% | +19.2% ± 2.1% |
| Trend = -10% | -10.0% | -9.8% ± 1.8% |
| Vol = 15% | 15.0% | 14.7% ± 0.9% |
| Vol = 30% | 30.0% | 29.4% ± 1.2% |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FinDiffusion Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: x_t (noisy returns) + t (timestep) + c (conditions)    │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Condition  │    │    Time     │    │   Input     │        │
│  │  Encoder    │    │  Embedding  │    │  Projection │        │
│  │  (MLP)      │    │  (Sinusoid) │    │  (Conv1D)   │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         └────────┬─────────┴─────────┬────────┘                │
│                  │                   │                         │
│                  ▼                   ▼                         │
│         ┌─────────────────────────────────────┐               │
│         │      Transformer Encoder            │               │
│         │  (Self-Attention + Cross-Attention) │               │
│         │         × N layers                  │               │
│         └─────────────────┬───────────────────┘               │
│                           │                                    │
│                           ▼                                    │
│         ┌─────────────────────────────────────┐               │
│         │       Output Projection             │               │
│         │   (Predict noise ε or x_0)          │               │
│         └─────────────────────────────────────┘               │
│                                                                │
│  Output: Predicted noise for denoising step                   │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
fin-diffusion/
├── configs/
│   ├── default.yaml          # Default training config
│   └── large.yaml            # Config for larger model
├── src/
│   ├── models/
│   │   ├── attention.py      # Multi-head & cross attention
│   │   ├── unet.py           # U-Net / Transformer backbone
│   │   ├── diffusion.py      # DDPM implementation
│   │   └── condition.py      # Condition encoders
│   ├── data/
│   │   ├── dataset.py        # PyTorch datasets
│   │   ├── preprocessing.py  # Data normalization
│   │   └── download.py       # Data fetching utilities
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   └── scheduler.py      # LR schedulers
│   └── evaluation/
│       ├── stylized_facts.py # Statistical tests
│       └── metrics.py        # Evaluation metrics
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── generate.py           # Generation script
│   └── download_data.py      # Data download script
├── notebooks/
│   └── demo.ipynb            # Interactive demo
└── tests/
    ├── test_model.py
    └── test_data.py
```

## Methodology

### Diffusion Process

We use Denoising Diffusion Probabilistic Models (DDPM) with:

1. **Forward Process**: Gradually add Gaussian noise to financial returns
   ```
   q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
   ```

2. **Reverse Process**: Learn to denoise, conditioned on market conditions
   ```
   p_θ(x_{t-1} | x_t, c) = N(x_{t-1}; μ_θ(x_t, t, c), Σ_θ(x_t, t, c))
   ```

### Conditioning Mechanism

Conditions are injected via cross-attention:
- **Trend**: Expected annualized return [-50%, +50%]
- **Volatility**: Expected annualized volatility [5%, 80%]  
- **Regime**: Categorical (bull/bear/sideways)

### Stylized Facts

We validate generated data captures:
- **Fat Tails**: Excess kurtosis > 0, power-law tail behavior
- **Volatility Clustering**: Significant ACF of squared returns
- **Leverage Effect**: Negative correlation between returns and future volatility
- **No Autocorrelation**: Returns themselves show no significant ACF

## Training

### Hardware Requirements

- **Minimum**: 1× GPU with 16GB VRAM (RTX 4080, A4000)
- **Recommended**: 1× GPU with 20GB+ VRAM (RTX 4090, A5000, A100)
- **Training Time**: ~4 hours on A100 for default config

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 252 | Sequence length (1 trading year) |
| `d_model` | 256 | Model dimension |
| `n_layers` | 6 | Number of transformer layers |
| `n_heads` | 8 | Attention heads |
| `timesteps` | 1000 | Diffusion timesteps |
| `batch_size` | 64 | Batch size |
| `lr` | 1e-4 | Learning rate |
| `epochs` | 100 | Training epochs |

### Monitoring

Training logs to Weights & Biases:
```bash
wandb login
python scripts/train.py --config configs/default.yaml --wandb
```

## Evaluation

### Stylized Facts Tests

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.ckpt --n_samples 10000
```

This runs:
1. **Jarque-Bera Test**: Non-normality of returns
2. **Ljung-Box Test**: Autocorrelation of squared returns
3. **Hill Estimator**: Tail index estimation
4. **Leverage Correlation**: Returns vs future volatility

### Downstream Tasks

```bash
# Deep hedging evaluation
python scripts/evaluate_hedging.py --checkpoint checkpoints/best.ckpt

# Portfolio optimization backtesting
python scripts/evaluate_portfolio.py --checkpoint checkpoints/best.ckpt
```

## References

```bibtex
@inproceedings{tanaka2025cofindiff,
  title={CoFinDiff: Controllable Financial Diffusion Model for Time Series Generation},
  author={Tanaka, Yuki and Hashimoto, Ryuji and others},
  booktitle={IJCAI},
  year={2025}
}

@inproceedings{sattarov2023findiff,
  title={FinDiff: Diffusion Models for Financial Tabular Data Generation},
  author={Sattarov, Timur and Schreyer, Marco and Borth, Damian},
  booktitle={ICAIF},
  year={2023}
}

@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
