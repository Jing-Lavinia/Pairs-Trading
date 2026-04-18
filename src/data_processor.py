"""
data_processor.py
-----------------
Market data assembly and mid-price computation.

Reads per-ticker CSV files from RAW_DATA_DIR and joins them into a
single MultiIndex DataFrame (ticker, field) aligned on a common timestamp
index.  Missing price bars are forward-filled (up to the data boundary);
missing volume bars are filled with zero to avoid overstating liquidity.
"""
import logging
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def build_market_data(active_tickers: list) -> pd.DataFrame:
    """
    Load and join per-ticker CSV files into a unified MultiIndex DataFrame.

    Price columns (BidPrice, AskPrice) are forward-filled to handle trading
    halts.  Volume columns are filled with 0 — never forward-filled — to
    avoid overstating available liquidity during inactive periods.
    """
    logger.info(f"Assembling market data matrix for {len(active_tickers)} active tickers...")

    df_dict = {}
    missing_tickers = []

    for ticker in active_tickers:
        file_path = RAW_DATA_DIR / f"{ticker}.csv"
        if not file_path.exists():
            missing_tickers.append(ticker)
            continue
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        df_dict[ticker] = df

    if missing_tickers:
        logger.error(f"Missing data for: {missing_tickers}. Run fetch_crypto_data.py first.")
        raise FileNotFoundError(f"Missing raw data for {len(missing_tickers)} tickers.")

    combined = pd.concat(df_dict.values(), axis=1)

    price_cols = [c for c in combined.columns if 'Price'  in c[1]]
    vol_cols   = [c for c in combined.columns if 'Volume' in c[1]]

    combined[price_cols] = combined[price_cols].ffill()
    combined[vol_cols]   = combined[vol_cols].fillna(0.0)

    combined.dropna(inplace=True)

    logger.info(f"Matrix built successfully. Shape: {combined.shape}")
    logger.info(f"Common backtest horizon: {combined.index[0]} to {combined.index[-1]}")

    return combined


def calculate_mid_prices(market_data: pd.DataFrame, stock_names: list) -> pd.DataFrame:
    """Add a MidPrice column for each ticker: (BidPrice + AskPrice) / 2."""
    for stock in stock_names:
        market_data[stock, "MidPrice"] = (
            market_data[stock, "BidPrice"] + market_data[stock, "AskPrice"]
        ) / 2
    market_data = market_data.sort_index(axis=1)
    return market_data
