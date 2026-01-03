#!/usr/bin/env python
"""Download financial data for training."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import download_stock_data, get_sp500_tickers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download stock data")
    parser.add_argument(
        "--tickers",
        type=str,
        default="SP500",
        help="Comma-separated tickers or 'SP500' for S&P 500 stocks",
    )
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date")
    parser.add_argument("--output", type=str, default="data/prices.csv", help="Output path")
    parser.add_argument("--n_tickers", type=int, default=30, help="Number of tickers if using SP500")
    args = parser.parse_args()

    if args.tickers.upper() == "SP500":
        tickers = get_sp500_tickers()[:args.n_tickers]
    else:
        tickers = [t.strip() for t in args.tickers.split(",")]

    logger.info(f"Downloading {len(tickers)} tickers from {args.start} to {args.end}")

    prices = download_stock_data(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        save_path=args.output,
    )

    logger.info(f"Downloaded {len(prices)} trading days, {len(prices.columns)} tickers")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
