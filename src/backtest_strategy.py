"""
backtest_strategy.py
--------------------
Vectorised pair-trading backtest engine with Numba JIT acceleration.
Configured for 1-Hour (1H) crypto statistical arbitrage.

Core simulation logic (_simulate_pair_trade) runs as a compiled Numba
@njit function for C-level throughput across thousands of bar iterations.

Entry/exit rules:
  - ENTRY:  |Z-score| > threshold AND Hurst < threshold (mean-reverting regime)
  - EXIT 1: |Z-score| <= 0.5 * threshold  (early take-profit)
  - EXIT 2: |Z-score| >= stop_loss_mult * threshold  (stop-loss)
  - EXIT 3: bars_held >= max_hold  (time-based stop, set to 1.5x half-life)

Position sizing: fixed notional per leg, quantity derived from mid-price.
"""
import pandas as pd
import numpy as np
import logging
from numba import njit

from src.config import HURST_THRESHOLD, HALF_LIFE_MULTIPLIER, MAX_HALF_LIFE

logger = logging.getLogger(__name__)


@njit
def _simulate_pair_trade(
        n_periods, z_arr, m1_arr, m2_arr,
        bv1_arr, av1_arr, bv2_arr, av2_arr,
        gamma_arr, hurst_arr, can_open_arr,
        threshold, notional, stop_loss_mult, max_hold,
        hurst_threshold
):
    """
    Numba-JIT single-pair simulation loop.

    Parameters
    ----------
    n_periods      : total number of 1H bars
    z_arr          : Kalman Z-score series
    m1/m2_arr      : mid-price series for leg 1 and leg 2
    bv/av arrays   : bid/ask volume series (reserved for future liquidity checks)
    gamma_arr      : time-varying Kalman hedge ratio
    hurst_arr      : rolling Hurst exponent
    can_open_arr   : boolean gate (all True in current implementation)
    threshold      : entry Z-score level (sigma)
    notional       : USD notional per leg
    stop_loss_mult : stop triggered at stop_loss_mult * threshold
    max_hold       : maximum bars to hold (time stop), in 1H bar units
    hurst_threshold: maximum Hurst value to allow a new entry

    Returns
    -------
    pos1_hist, pos2_hist : per-bar position (in coin units) for each leg
    """
    pos1_hist = np.zeros(n_periods)
    pos2_hist = np.zeros(n_periods)

    curr1 = 0.0
    curr2 = 0.0
    bars_held = 0  # counts 1H bars since entry

    for t in range(n_periods - 1):
        z = z_arr[t]

        if t > 0:
            prev_m1 = m1_arr[t - 1]
            prev_m2 = m2_arr[t - 1]

            # Skip bar if prices are invalid (zero or missing)
            if prev_m1 <= 1e-8 or prev_m2 <= 1e-8:
                curr1, curr2, bars_held = 0.0, 0.0, 0
                pos1_hist[t + 1] = curr1
                pos2_hist[t + 1] = curr2
                continue

            if curr1 != 0.0:
                bars_held += 1

            # Exit 1: stop-loss or time stop
            if curr1 != 0.0 and (abs(z) >= stop_loss_mult * threshold or bars_held >= max_hold):
                curr1, curr2, bars_held = 0.0, 0.0, 0

            # Exit 2: early take-profit — close when spread reverts halfway
            elif abs(z) <= (0.5 * threshold) and curr1 != 0.0:
                curr1, curr2, bars_held = 0.0, 0.0, 0

        # Entry: only in mean-reverting regime (Hurst < threshold)
        valid_regime = hurst_arr[t] < hurst_threshold
        can_open     = can_open_arr[t]

        if curr1 == 0.0 and valid_regime and can_open:
            gamma = gamma_arr[t]
            if z < -threshold:
                # Spread too low -> long S1, short S2 (scaled by gamma)
                curr1 =  notional / m1_arr[t]
                curr2 = -(notional * gamma) / m2_arr[t]
                bars_held = 1
            elif z > threshold:
                # Spread too high -> short S1, long S2
                curr1 = -notional / m1_arr[t]
                curr2 =  (notional * gamma) / m2_arr[t]
                bars_held = 1

        pos1_hist[t + 1] = curr1
        pos2_hist[t + 1] = curr2

    return pos1_hist, pos2_hist


def calculate_threshold_pnls(tradable_pairs_data, stock_pairs, gamma_dictionary,
                              hurst_dictionary, half_life_series,
                              notional_trade_amount=30000.0, commission_rate=0.00045,
                              stop_loss_multiplier=2.5, steps=20):
    """
    In-sample grid search over entry thresholds for each pair.
    Tests `steps` evenly-spaced thresholds in [1.5, 3.5] sigma.
    Returns a DataFrame indexed by pair with columns Thresholds and PnLs.
    """
    logger.info("Initializing JIT-compiled grid search engine...")
    positions  = {}
    timestamps = tradable_pairs_data.index
    n_periods  = len(timestamps)

    can_open_arr = np.ones(n_periods, dtype=np.bool_)

    for pair in stock_pairs:
        s1, s2 = pair[0], pair[1]

        search_space = np.linspace(1.5, 3.5, steps)

        hl = half_life_series.get(pair, MAX_HALF_LIFE)
        if np.isnan(hl) or np.isinf(hl) or hl <= 0:
            hl = MAX_HALF_LIFE

        # Cap max holding period at 2x the half-life but never exceed 2 days
        max_hold_bars = int(min(hl * HALF_LIFE_MULTIPLIER, MAX_HALF_LIFE * 2))

        bp1_arr = tradable_pairs_data[s1, 'BidPrice'].values
        ap1_arr = tradable_pairs_data[s1, 'AskPrice'].values
        bp2_arr = tradable_pairs_data[s2, 'BidPrice'].values
        ap2_arr = tradable_pairs_data[s2, 'AskPrice'].values

        m1_arr = (bp1_arr + ap1_arr) / 2.0
        m2_arr = (bp2_arr + ap2_arr) / 2.0

        bv1_arr = tradable_pairs_data[s1, 'BidVolume'].values
        av1_arr = tradable_pairs_data[s1, 'AskVolume'].values
        bv2_arr = tradable_pairs_data[s2, 'BidVolume'].values
        av2_arr = tradable_pairs_data[s2, 'AskVolume'].values

        z_arr     = tradable_pairs_data[s1 + s2, 'Z-Value'].values
        gamma_arr = gamma_dictionary[s1, s2].values
        hurst_arr = hurst_dictionary[s1, s2].values

        for i in search_space:
            p1_hist, p2_hist = _simulate_pair_trade(
                n_periods, z_arr, m1_arr, m2_arr,
                bv1_arr, av1_arr, bv2_arr, av2_arr,
                gamma_arr, hurst_arr, can_open_arr,
                i, notional_trade_amount, stop_loss_multiplier, max_hold_bars,
                HURST_THRESHOLD
            )
            positions[f"{s1}_{s2}_{s1}_Thres_{i}"] = p1_hist
            positions[f"{s1}_{s2}_{s2}_Thres_{i}"] = p2_hist

    positions_df   = pd.DataFrame(positions, index=timestamps)
    positions_diff = positions_df.diff().fillna(0)
    pnl_results    = {'Pairs': [], 'Thresholds': [], 'PnLs': []}

    for pair in stock_pairs:
        s1, s2 = pair[0], pair[1]
        m1 = (tradable_pairs_data[s1, 'BidPrice'] + tradable_pairs_data[s1, 'AskPrice']) / 2
        m2 = (tradable_pairs_data[s2, 'BidPrice'] + tradable_pairs_data[s2, 'AskPrice']) / 2

        for i in search_space:
            thresh = i
            c1 = f"{s1}_{s2}_{s1}_Thres_{thresh}"
            c2 = f"{s1}_{s2}_{s2}_Thres_{thresh}"

            # Net cash flow = mark-to-market change minus commission on traded notional
            cf1 = (positions_diff[c1] * -m1) - (positions_diff[c1].abs() * m1 * commission_rate)
            cf2 = (positions_diff[c2] * -m2) - (positions_diff[c2].abs() * m2 * commission_rate)

            nav = (cf1.cumsum() + positions_df[c1] * m1) + (cf2.cumsum() + positions_df[c2] * m2)

            pnl_results['Pairs'].append((s1, s2))
            pnl_results['Thresholds'].append(thresh)
            pnl_results['PnLs'].append(nav.iloc[-1] if len(nav) > 0 else 0.0)

    return pd.DataFrame(pnl_results).set_index('Pairs')


def calculate_positions(market_data, active_pairs, data_gammas, data_hurst,
                        half_life_series, threshold_dict,
                        notional_trade_amount=30000.0, stop_loss_multiplier=2.5):
    """
    Generate final position matrix for all active pairs using their
    in-sample-optimised thresholds. Aggregates across pairs sharing a ticker.
    """
    logger.info("Generating final multi-asset portfolio matrix (JIT accelerated)...")
    positions  = {}
    timestamps = market_data.index
    n_periods  = len(timestamps)

    can_open_arr = np.ones(n_periods, dtype=np.bool_)

    for pair in active_pairs:
        s1, s2 = pair[0], pair[1]
        thresh = threshold_dict[pair]
        hl     = half_life_series.get(pair, MAX_HALF_LIFE)
        if np.isnan(hl) or np.isinf(hl) or hl <= 0:
            hl = MAX_HALF_LIFE

        max_hold_bars = int(min(hl * HALF_LIFE_MULTIPLIER, MAX_HALF_LIFE * 2))

        bp1_arr = market_data[s1, 'BidPrice'].values
        ap1_arr = market_data[s1, 'AskPrice'].values
        bp2_arr = market_data[s2, 'BidPrice'].values
        ap2_arr = market_data[s2, 'AskPrice'].values

        m1_arr = (bp1_arr + ap1_arr) / 2.0
        m2_arr = (bp2_arr + ap2_arr) / 2.0

        bv1_arr = market_data[s1, 'BidVolume'].values
        av1_arr = market_data[s1, 'AskVolume'].values
        bv2_arr = market_data[s2, 'BidVolume'].values
        av2_arr = market_data[s2, 'AskVolume'].values

        z_arr     = market_data[s1 + s2, 'Z-Value'].values
        gamma_arr = data_gammas[pair].values
        hurst_arr = data_hurst[pair].values

        p1_hist, p2_hist = _simulate_pair_trade(
            n_periods, z_arr, m1_arr, m2_arr,
            bv1_arr, av1_arr, bv2_arr, av2_arr,
            gamma_arr, hurst_arr, can_open_arr,
            thresh, notional_trade_amount, stop_loss_multiplier, max_hold_bars,
            HURST_THRESHOLD
        )

        # Accumulate positions across pairs that share a ticker
        positions[s1] = positions.get(s1, np.zeros(n_periods)) + p1_hist
        positions[s2] = positions.get(s2, np.zeros(n_periods)) + p2_hist

    pos_df = pd.DataFrame(positions, index=timestamps)
    return pos_df, pos_df.diff().fillna(0)


def calculate_pnl(pos_diff_df, market_data, active_pairs, commission_rate=0.00045):
    """
    Mark-to-market portfolio PnL settlement.
    Computes both net PnL (post-commission) and gross PnL (pre-commission)
    for each pair and the total portfolio.
    """
    logger.info("Settling mark-to-market portfolio PnL...")
    pnl_df          = pd.DataFrame(index=market_data.index)
    total_net_pnl   = pd.Series(0.0, index=market_data.index)
    total_gross_pnl = pd.Series(0.0, index=market_data.index)

    for pair in active_pairs:
        s1, s2 = pair[0], pair[1]
        if s1 not in pos_diff_df.columns or s2 not in pos_diff_df.columns:
            continue

        m1 = (market_data[s1, 'BidPrice'] + market_data[s1, 'AskPrice']) / 2
        m2 = (market_data[s2, 'BidPrice'] + market_data[s2, 'AskPrice']) / 2

        diff1 = pos_diff_df[s1]
        diff2 = pos_diff_df[s2]
        pos1  = diff1.cumsum()
        pos2  = diff2.cumsum()

        # Net: deduct commission on every traded notional unit
        cf1_net = (diff1 * -m1) - (diff1.abs() * m1 * commission_rate)
        cf2_net = (diff2 * -m2) - (diff2.abs() * m2 * commission_rate)

        # Gross: no commission (used for friction analysis)
        cf1_gross = diff1 * -m1
        cf2_gross = diff2 * -m2

        nav_net   = cf1_net.cumsum()   + pos1 * m1 + cf2_net.cumsum()   + pos2 * m2
        nav_gross = cf1_gross.cumsum() + pos1 * m1 + cf2_gross.cumsum() + pos2 * m2

        pnl_df[f"{s1}{s2} PnL"] = nav_net
        total_net_pnl   += nav_net.diff().fillna(0)
        total_gross_pnl += nav_gross.diff().fillna(0)

    pnl_df['PnL']       = total_net_pnl
    pnl_df['Gross PnL'] = total_gross_pnl
    return pnl_df


def calculate_strategy_kpis(pnl_results: pd.DataFrame, capital: float):
    """
    Compute and log institutional performance metrics.
    Annualisation constant: 365 days * 24 hours = 8760 bars/year (1H crypto).
    """
    start_time = pnl_results.index[0]
    end_time   = pnl_results.index[-1]

    total_profit  = pnl_results['PnL'].sum()
    total_return  = total_profit / capital

    # Per-bar (hourly) return series
    period_returns = pnl_results['PnL'].diff().fillna(0) / capital

    periods_per_year = 8760  # 365 * 24 for 1H crypto
    years = len(pnl_results) / periods_per_year

    annualized_return  = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    annual_volatility  = period_returns.std() * np.sqrt(periods_per_year)

    risk_free_rate = 0.02
    sharpe_ratio   = (annualized_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0.0

    downside_returns  = period_returns[period_returns < 0]
    downside_vol      = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio     = (annualized_return - risk_free_rate) / downside_vol if downside_vol != 0 else np.nan

    cumulative_returns = (1 + period_returns).cumprod()
    rolling_max        = cumulative_returns.cummax()
    drawdown           = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown       = drawdown.min()

    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    gross_profit  = pnl_results['PnL'][pnl_results['PnL'] > 0].sum()
    gross_loss    = abs(pnl_results['PnL'][pnl_results['PnL'] < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    active_periods  = pnl_results['PnL'][pnl_results['PnL'] != 0]
    winning_periods = active_periods[active_periods > 0]
    win_rate        = len(winning_periods) / len(active_periods) if len(active_periods) > 0 else 0

    logger.info("\n" + "=" * 60)
    logger.info("INSTITUTIONAL STRATEGY PERFORMANCE REPORT (1H)")
    logger.info("=" * 60)
    logger.info(f"Test Period:                 {start_time} to {end_time}")
    logger.info(f"Initial Capital:             $ {capital:,.2f}")
    logger.info(f"Total Profit:                $ {total_profit:,.2f}")
    logger.info(f"Total Return:                {total_return * 100:.2f}%")
    logger.info(f"Annualized Return (CAGR):    {annualized_return * 100:.2f}%")
    logger.info(f"Annualized Volatility:       {annual_volatility * 100:.2f}%")
    logger.info("-" * 60)
    logger.info(f"Sharpe Ratio:                {sharpe_ratio:.2f}")
    logger.info(f"Sortino Ratio:               {sortino_ratio:.2f}")
    logger.info(f"Calmar Ratio:                {calmar_ratio:.2f}")
    logger.info("-" * 60)
    logger.info(f"Max Drawdown:                {max_drawdown * 100:.2f}%")
    logger.info(f"Profit Factor:               {profit_factor:.2f}")
    logger.info(f"Win Rate (Trade Bars):       {win_rate * 100:.2f}%")
    logger.info(f"Total Active Trading Bars:   {len(active_periods)}")
    logger.info("=" * 60 + "\n")

    return {
        "Total Return":         total_return,
        "Annualized Return":    annualized_return,
        "Annualized Volatility": annual_volatility,
        "Sharpe Ratio":         sharpe_ratio,
        "Max Drawdown":         max_drawdown
    }
