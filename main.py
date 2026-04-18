"""
main.py
-------
End-to-end entry point for the Crypto Statistical Arbitrage Pipeline.

Execution phases:
  PHASE 0 -- Local data vault check / Binance fetch (skipped if cache is fresh)
  PHASE 1 -- Market data assembly and mid-price computation
  PHASE 2 -- Walk-forward PCA+DBSCAN clustering + Engle-Granger cointegration
             (cached to disk after the first run for reproducibility)
  PHASE 3 -- In-sample grid search over entry thresholds (1.5-3.5 sigma)
  PHASE 4 -- Out-of-sample PnL settlement and KPI reporting
  PHASE 5 -- Chart generation (all charts use strictly OOS data)

CLI usage:
  python main.py              # Normal run (loads cached walk-forward if available)
  python main.py --refresh    # Force re-run of walk-forward analysis
"""

import logging
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.config import (
    CAPITAL, COMMISSION_RATE, CACHE_FILE, FIGURES_DIR,
    GRID_SEARCH_STEPS, LOGS_DIR, NOTIONAL_TRADE_AMOUNT, P_THRESHOLD,
    RESULTS_DIR, RETRAIN_BARS, STOP_LOSS_MULTIPLIER, TICKERS,
    WARMUP_BARS, ZSCORE_WINDOW, OOS_START, MAX_PAIRS_PER_TICKER, DATA_DIR,
)
from src.utils import setup_global_logger
from src.fetch_crypto_data import fetch_and_audit_vault
from src.backtest_strategy import (
    calculate_pnl, calculate_positions, calculate_threshold_pnls,
    calculate_strategy_kpis,
)
from src.data_processor import calculate_mid_prices, build_market_data
from src.stats_analysis import run_walk_forward_analysis
from src.visualization import (
    plot_cointegration_summary, plot_correlation_heatmap, plot_drawdown_profile,
    plot_total_cumulative_pnl, plot_trade_pnl_distribution,
    plot_pnl_attribution, plot_capital_utilization, plot_gross_vs_net_pnl,
)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    setup_global_logger(str(LOGS_DIR / "strategy_execution.log"))
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # PHASE 0: Data vault check — skip Binance fetch if cache is fresh
    # ------------------------------------------------------------------
    logger.info("PHASE 0: CHECKING LOCAL DATA VAULT (OFFLINE MODE)")
    raw_data_dir = DATA_DIR / "raw"

    if raw_data_dir.exists() and len(list(raw_data_dir.glob("*.csv"))) >= len(TICKERS):
        logger.info("[!] Local data found. Skipping Binance fetch to preserve reproducibility.")
    else:
        logger.info("Local data incomplete or missing. Syncing from Binance...")
        fetch_and_audit_vault()

    # ------------------------------------------------------------------
    # PHASE 1: Build unified market data matrix
    # ------------------------------------------------------------------
    logger.info("PHASE 1: DYNAMIC DATA ASSEMBLY FROM LOCAL VAULT")
    try:
        market_data = build_market_data(TICKERS)
    except FileNotFoundError as e:
        logger.error(e)
        return

    stock_names = list(market_data.columns.get_level_values(0).unique())
    market_data = calculate_mid_prices(market_data, stock_names)

    # ------------------------------------------------------------------
    # PHASE 2: Walk-forward cointegration (load cache if available)
    # ------------------------------------------------------------------
    if CACHE_FILE.exists():
        logger.info(f"Loading cached walk-forward analysis from {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            data_zvalues, data_gammas, data_hurst, stock_pairs, final_analysis = pickle.load(f)
    else:
        logger.info("Running intensive walk-forward analysis with ML clustering...")
        data_zvalues, data_gammas, data_hurst, stock_pairs, final_analysis = run_walk_forward_analysis(
            market_data, stock_names, WARMUP_BARS, RETRAIN_BARS, P_THRESHOLD, ZSCORE_WINDOW
        )
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump((data_zvalues, data_gammas, data_hurst, stock_pairs, final_analysis), f)

    if not stock_pairs:
        logger.warning("No cointegrated pairs found. Terminating.")
        return

    # Prepare the backtest data slice (post-warmup, with Z-values joined)
    backtest_start = market_data.index[WARMUP_BARS]
    market_bt      = market_data.loc[backtest_start:]
    unique_stocks  = list(set([s[0] for s in stock_pairs] + [s[1] for s in stock_pairs]))
    tradable_pairs_data = market_bt[unique_stocks].copy()

    # Attach Kalman Z-scores as additional columns
    new_z_columns = {}
    for pair in stock_pairs:
        new_z_columns[(pair[0] + pair[1], "Z-Value")] = data_zvalues[pair].loc[backtest_start:]
    if new_z_columns:
        z_df = pd.DataFrame(new_z_columns)
        tradable_pairs_data = pd.concat([tradable_pairs_data, z_df], axis=1)

    half_life_series = final_analysis["HalfLife"]

    # ------------------------------------------------------------------
    # PHASE 3: In-sample grid search over entry thresholds
    # ------------------------------------------------------------------
    logger.info("PHASE 3: IN-SAMPLE GRID SEARCH OPTIMISATION")
    is_mask          = tradable_pairs_data.index < pd.to_datetime(OOS_START).tz_localize(None)
    tradable_pairs_is = tradable_pairs_data.loc[is_mask]

    pnl_threshold = calculate_threshold_pnls(
        tradable_pairs_is, stock_pairs, data_gammas, data_hurst, half_life_series,
        notional_trade_amount=NOTIONAL_TRADE_AMOUNT, commission_rate=COMMISSION_RATE,
        stop_loss_multiplier=STOP_LOSS_MULTIPLIER, steps=GRID_SEARCH_STEPS
    )

    pnl_reset = pnl_threshold.reset_index()

    # Summarise grid-search results and rank all candidates
    all_candidates     = []
    zero_trade_count   = 0
    negative_pnl_count = 0

    for pair in stock_pairs:
        pair_data = pnl_reset[pnl_reset["Pairs"] == pair]
        if pair_data.empty:
            continue
        best_idx = pair_data["PnLs"].idxmax()
        best_row = pair_data.loc[best_idx]
        max_pnl  = best_row["PnLs"]

        all_candidates.append({
            "pair":           pair,
            "best_threshold": best_row["Thresholds"],
            "max_pnl":        max_pnl,
            "min_pnl":        pair_data["PnLs"].min()
        })

        if max_pnl == 0:
            zero_trade_count += 1
        elif max_pnl < 0:
            negative_pnl_count += 1

    all_candidates.sort(key=lambda x: x["max_pnl"], reverse=True)

    logger.info(f"--- IN-SAMPLE GRID SEARCH DIAGNOSTICS ---")
    logger.info(f"Total pairs evaluated: {len(all_candidates)}")
    logger.info(f"Pairs with $0 PnL (threshold too high, no trades):  {zero_trade_count}")
    logger.info(f"Pairs with negative best PnL (eaten by friction):    {negative_pnl_count}")
    logger.info(f"Pairs with positive PnL (eligible for OOS):          "
                f"{len(all_candidates) - zero_trade_count - negative_pnl_count}")

    logger.info("--- TOP 10 PAIRS BY MAX IN-SAMPLE PNL ---")
    for i, cand in enumerate(all_candidates[:10]):
        status = "PASS" if cand["max_pnl"] > 0 else ("NO TRADES" if cand["max_pnl"] == 0 else "LOSE")
        logger.info(
            f"  {i + 1}. {cand['pair']} | "
            f"Best Thresh: {cand['best_threshold']:.2f}sigma | "
            f"Max IS PnL: ${cand['max_pnl']:.2f} | [{status}]"
        )

    # Final pair selection: profitable IS pairs that pass hub-stock concentration check
    profitable_candidates = [c for c in all_candidates if c["max_pnl"] > 0]
    stock_pairs_final  = []
    threshold_dict     = {}
    final_ticker_counts = defaultdict(int)

    for candidate in profitable_candidates:
        pair  = candidate["pair"]
        s1, s2 = pair[0], pair[1]
        if (final_ticker_counts[s1] < MAX_PAIRS_PER_TICKER and
                final_ticker_counts[s2] < MAX_PAIRS_PER_TICKER):
            stock_pairs_final.append(pair)
            threshold_dict[pair] = candidate["best_threshold"]
            final_ticker_counts[s1] += 1
            final_ticker_counts[s2] += 1
            logger.info(f"[+] Selected for OOS: {pair} | Threshold: {candidate['best_threshold']:.2f}sigma")

    if not stock_pairs_final:
        logger.error("FATAL: No profitable pairs survived in-sample training. Terminating.")
        return

    # ------------------------------------------------------------------
    # PHASE 4: Out-of-sample evaluation
    # ------------------------------------------------------------------
    logger.info("PHASE 4: OUT-OF-SAMPLE PNL SETTLEMENT & EVALUATION")
    pos_df, pos_diff_df = calculate_positions(
        tradable_pairs_data, stock_pairs_final, data_gammas, data_hurst,
        half_life_series, threshold_dict,
        notional_trade_amount=NOTIONAL_TRADE_AMOUNT,
        stop_loss_multiplier=STOP_LOSS_MULTIPLIER
    )

    pnl_results = calculate_pnl(
        pos_diff_df, tradable_pairs_data, stock_pairs_final,
        commission_rate=COMMISSION_RATE
    )

    # Slice strictly to OOS period and reset cumulative PnL from zero
    oos_mask = pnl_results.index >= pd.to_datetime(OOS_START).tz_localize(None)
    pnl_oos  = pnl_results.loc[oos_mask].copy()
    pnl_oos['Cum PnL'] = pnl_oos['PnL'].cumsum()

    for pair in stock_pairs_final:
        s1, s2   = pair[0], pair[1]
        pnl_col  = f"{s1}{s2} PnL"
        cum_col  = f"{s1}{s2} Cum PnL"
        if pnl_col in pnl_oos.columns:
            pnl_oos[cum_col] = pnl_oos[pnl_col]

    calculate_strategy_kpis(pnl_oos, capital=CAPITAL)

    # ------------------------------------------------------------------
    # PHASE 5: Chart generation (strictly OOS data)
    # ------------------------------------------------------------------
    logger.info("PHASE 5: GENERATING CHARTS (strictly out-of-sample)")

    plot_total_cumulative_pnl(stock_pairs_final, pnl_oos)
    plot_drawdown_profile(pnl_oos, capital=CAPITAL)

    unique_stocks = list(set([s[0] for s in stock_pairs] + [s[1] for s in stock_pairs]))
    plot_correlation_heatmap(market_data, unique_stocks[:30])

    plot_pnl_attribution(stock_pairs_final, pnl_oos)
    plot_capital_utilization(pos_df, tradable_pairs_data, stock_pairs_final, CAPITAL)
    plot_gross_vs_net_pnl(pnl_oos)

    logger.info("[SUCCESS] Pipeline complete.")


if __name__ == "__main__":
    main()
