"""
fetch_crypto_data.py
--------------------
Binance 1H OHLCV data fetcher with local cache management.

Pulls historical kline data from the Binance REST API and saves each
ticker as a CSV in RAW_DATA_DIR.  On subsequent runs, files that are
already up-to-date (last bar < 1.5 hours old) are skipped to avoid
unnecessary API calls.

Output format per ticker: timestamp index, columns AskPrice / BidPrice /
AskVolume / BidVolume (derived from close price and volume with a 1 bp
synthetic spread).

Note: If you encounter HTTP 451 (geo-restriction), switch the base URL
to "https://api.binance.us/api/v3/klines".
"""
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from src.config import RAW_DATA_DIR, UNIVERSE_TICKERS, DAYS_TO_FETCH


def fetch_binance_klines_historical(symbol, interval='1h', days=180):
    """Fetch historical klines from Binance for a single symbol."""
    url = "https://api.binance.com/api/v3/klines"

    end_time   = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    limit = 1000

    print(f"[INFO] Fetching {symbol} (last {days} days, interval: {interval})...")

    while start_time < end_time:
        params = {
            "symbol":    f"{symbol}USDT",
            "interval":  interval,
            "limit":     limit,
            "startTime": start_time,
            "endTime":   end_time
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[ERROR] {symbol} request failed: {e}. Retrying in 2s...")
            time.sleep(2)
            continue

        if not data:
            break

        all_klines.extend(data)
        # Advance the start pointer past the last received bar
        start_time = data[-1][6] + 1

        current_sync_date = datetime.fromtimestamp(data[-1][0] / 1000.0).strftime('%Y-%m-%d')
        print(f"   [SYNC] {current_sync_date} | accumulated: {len(all_klines)} bars", end='\r')

        time.sleep(0.1)  # Respect Binance rate limits

    print(f"\n[DONE] {symbol} fetch complete.")

    if not all_klines:
        return pd.DataFrame()

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(all_klines, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['close', 'volume']].astype(float)
    return df


def fetch_and_audit_vault() -> None:
    """
    Sync the local data vault for all universe tickers.
    Skips any ticker whose cached file is already fresh (< 1.5 hours stale).
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    spread_bps = 0.0001  # Synthetic 1 bp half-spread to model bid/ask

    print("=" * 75)
    print(f"CRYPTO 1H DATA SYNC ENGINE (target range: {DAYS_TO_FETCH} days)")
    print("=" * 75)

    for i, ticker in enumerate(UNIVERSE_TICKERS, 1):
        file_path = RAW_DATA_DIR / f"{ticker}.csv"

        # Skip if local data is still fresh (< 1.5 hours since last bar)
        if file_path.exists():
            try:
                existing_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not existing_df.empty:
                    last_timestamp = existing_df.index[-1]
                    time_diff = datetime.now() - last_timestamp
                    if time_diff.total_seconds() < 5400:
                        print(f"[{i:03d}/{len(UNIVERSE_TICKERS):03d}] [SKIP] {ticker} | already up-to-date (last: {last_timestamp})")
                        continue
            except Exception:
                print(f"[WARN] Failed to read cached {ticker}. Re-fetching.")

        df = fetch_binance_klines_historical(ticker, interval='1h', days=DAYS_TO_FETCH)

        if df.empty:
            print(f"[{i:03d}/{len(UNIVERSE_TICKERS):03d}] [ERROR] {ticker} | no data returned.")
            continue

        # Convert close/volume to synthetic bid/ask format for strategy compatibility
        temp_df = pd.DataFrame(index=df.index)
        close_price = df['close']
        vol = df['volume']

        temp_df['AskPrice'] = close_price * (1 + spread_bps / 2)
        temp_df['BidPrice'] = close_price * (1 - spread_bps / 2)
        temp_df['AskVolume'] = vol / 2
        temp_df['BidVolume'] = vol / 2

        temp_df.to_csv(file_path)
        print(f"[{i:03d}/{len(UNIVERSE_TICKERS):03d}] [SAVED] {ticker} | {len(temp_df)} bars")

    print("=" * 75)
    print("[SUCCESS] Data vault sync complete.")


if __name__ == "__main__":
    fetch_and_audit_vault()
