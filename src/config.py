"""
config.py
---------
Central configuration for the Crypto Statistical Arbitrage Pipeline.

Defines all directory paths, the trading universe (top-20 liquid altcoins
on Binance), walk-forward scheduling parameters, cointegration filters,
half-life bounds, position sizing, and ML pipeline settings.

All other modules import constants from here. Adjust values here to
reconfigure the entire pipeline without touching strategy logic.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "results" / "figures"
LOGS_DIR = BASE_DIR / "logs"

CACHE_FILE = DATA_DIR / "walk_forward_cache.pkl"

# ---------------------------------------------------------------------------
# Time resolution: 1-Hour bars  (1 day = 24 bars)
# ---------------------------------------------------------------------------
BARS_PER_DAY = 24

DAYS_TO_FETCH = 180   # Historical depth: last 180 days (~4320 1H bars)
OOS_START = "2026-03-01"  # Everything after this date is strictly out-of-sample

# ---------------------------------------------------------------------------
# Universe: top-20 liquid altcoins on Binance
# ---------------------------------------------------------------------------
UNIVERSE_TICKERS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "DOT", "POL", "LINK", "UNI", "LTC", "NEAR", "BCH",
    "ATOM", "XLM", "FIL", "TRX", "ETC"
]
TICKERS = UNIVERSE_TICKERS

# ---------------------------------------------------------------------------
# Walk-forward scheduling
# ---------------------------------------------------------------------------
WARMUP_BARS  = 30 * BARS_PER_DAY   # Training window: 30 days (720 bars)
RETRAIN_BARS = 7  * BARS_PER_DAY   # Re-cluster every 7 days (168 bars)

# ---------------------------------------------------------------------------
# Cointegration & spread filters
# ---------------------------------------------------------------------------
P_THRESHOLD   = 0.05
ZSCORE_WINDOW = 3 * BARS_PER_DAY   # Rolling Z-score lookback: 3 days
HURST_WINDOW  = 3 * BARS_PER_DAY   # Hurst exponent lookback: 3 days
HURST_THRESHOLD = 0.60              # Allow moderate trending (crypto-adapted)

HALF_LIFE_MULTIPLIER = 1.5
MIN_HALF_LIFE = 4                   # Minimum mean-reversion time: 4 hours
MAX_HALF_LIFE = 2 * BARS_PER_DAY   # Maximum: 48 hours — beyond this, spread is non-stationary

# ---------------------------------------------------------------------------
# Portfolio & execution parameters
# ---------------------------------------------------------------------------
CAPITAL = 100_000.0
COMMISSION_RATE      = 0.00045    # 4.5 bps round-trip (Binance taker + slippage)
NOTIONAL_TRADE_AMOUNT = 30_000.0  # Fixed notional per leg
STOP_LOSS_MULTIPLIER  = 2         # Stop triggered at 2x entry threshold

# ---------------------------------------------------------------------------
# Grid search & concentration limits
# ---------------------------------------------------------------------------
GRID_SEARCH_STEPS   = 20
MAX_PAIRS_PER_TICKER = 3          # Hub-stock constraint: one coin in at most 3 pairs
