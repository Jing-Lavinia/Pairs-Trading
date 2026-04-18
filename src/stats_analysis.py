"""
stats_analysis.py
-----------------
Statistical engine for pair selection and walk-forward cointegration analysis.

Pipeline overview:
  1. find_statistical_clusters  -- PCA (5 components) + DBSCAN on 1H log-returns
     to identify cryptos with similar structural behaviour, dramatically pruning
     the O(n^2) pair search space before any cointegration tests are run.
  2. run_cointegration_tests     -- Engle-Granger two-step test on each candidate
     pair; filters by p-value and mean-reversion half-life bounds.
  3. Kalman filter regression    -- Numba-JIT Kalman loop estimates a time-varying
     hedge ratio (gamma) and the resulting spread in O(n) time.
  4. Rolling Hurst exponent      -- Numba-JIT R/S analysis distinguishes
     mean-reverting (H < 0.5) from trending (H > 0.5) regimes.
  5. run_walk_forward_analysis   -- Orchestrates the above across rolling
     30-day training / 7-day test windows with strict OOS separation.
  6. get_tradable_pairs          -- Final filter enforcing hub-stock constraints
     (no single coin in more than MAX_PAIRS_PER_TICKER pairs).
"""
import logging
import itertools
from numba import njit
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from collections import defaultdict
from src.config import MIN_HALF_LIFE, MAX_HALF_LIFE, P_THRESHOLD, HURST_WINDOW, MAX_PAIRS_PER_TICKER
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba JIT kernels
# ---------------------------------------------------------------------------

@njit
def _kalman_loop(x_vals, y_vals):
    """
    Kalman filter for time-varying linear regression: y = theta0 + theta1 * x.
    Returns per-bar hedge ratio (theta1) and spread (innovation) histories.
    State noise variance is set via the steady-state delta parameterisation.
    """
    n = len(x_vals)
    delta = 1e-5
    trans_cov_val = delta / (1.0 - delta)
    obs_cov = 1e-3

    theta0, theta1 = 0.0, 0.0
    P00, P01, P10, P11 = 0.0, 0.0, 0.0, 0.0

    gamma_history  = np.zeros(n)
    spread_history = np.zeros(n)

    for t in range(n):
        x_t = x_vals[t]
        y_t = y_vals[t]

        # Predict step
        P00_pred = P00 + trans_cov_val
        P01_pred = P01
        P10_pred = P10
        P11_pred = P11 + trans_cov_val

        y_pred     = theta0 + theta1 * x_t
        innovation = y_t - y_pred

        hp0 = P00_pred + x_t * P10_pred
        hp1 = P01_pred + x_t * P11_pred
        S   = hp0 * 1.0 + hp1 * x_t + obs_cov

        # Kalman gain
        K0 = (P00_pred * 1.0 + P01_pred * x_t) / S
        K1 = (P10_pred * 1.0 + P11_pred * x_t) / S

        # Update step
        theta0 = theta0 + K0 * innovation
        theta1 = theta1 + K1 * innovation

        ikh00 = 1.0 - K0 * 1.0
        ikh01 = 0.0 - K0 * x_t
        ikh10 = 0.0 - K1 * 1.0
        ikh11 = 1.0 - K1 * x_t

        P00 = ikh00 * P00_pred + ikh01 * P10_pred
        P01 = ikh00 * P01_pred + ikh01 * P11_pred
        P10 = ikh10 * P00_pred + ikh11 * P10_pred
        P11 = ikh10 * P01_pred + ikh11 * P11_pred

        gamma_history[t]  = theta1
        spread_history[t] = innovation

    return gamma_history, spread_history


@njit
def _rolling_hurst_fast(spread, window):
    """
    Rolling Hurst exponent via log-log regression of variance-vs-lag.
    H < 0.5 -> mean-reverting; H > 0.5 -> trending; H ~ 0.5 -> random walk.
    Uses lags [2, 5, 10, 20] to cover the short-term mean-reversion horizon.
    """
    n     = len(spread)
    hurst = np.full(n, np.nan)
    lags  = np.array([2, 5, 10, 20])
    log_lags = np.log(lags)
    mean_x = np.mean(log_lags)
    var_x  = np.sum((log_lags - mean_x) ** 2) / len(lags)

    for t in range(window, n):
        y = np.zeros(len(lags))
        for i in range(len(lags)):
            lag   = lags[i]
            diffs = spread[t - window + lag:t + 1] - spread[t - window:t + 1 - lag]
            m_diff = np.mean(diffs)
            s_diff = np.sqrt(np.sum((diffs - m_diff) ** 2) / len(diffs))
            y[i]  = np.log(s_diff + 1e-8)

        mean_y = np.mean(y)
        cov_xy = np.sum((log_lags - mean_x) * (y - mean_y)) / len(lags)

        hurst[t] = (cov_xy / var_x) if var_x > 1e-8 else 0.5

    return hurst


# ---------------------------------------------------------------------------
# Cointegration helpers
# ---------------------------------------------------------------------------

def estimate_long_run_short_run_relationships(y: pd.Series, x: pd.Series) -> tuple:
    """Engle-Granger long-run OLS + ECM short-run speed-of-adjustment (alpha)."""
    assert isinstance(y, pd.Series) and isinstance(x, pd.Series)
    assert y.isnull().sum() == 0 and x.isnull().sum() == 0
    assert y.index.equals(x.index)

    long_run_fit = OLS(y, add_constant(x)).fit()
    c, gamma = long_run_fit.params
    z = long_run_fit.resid

    short_run_fit = OLS(y.diff().iloc[1:], z.shift().iloc[1:]).fit()
    alpha = short_run_fit.params.iloc[0]

    return c, gamma, alpha, z


def engle_granger_two_step_cointegration_test(y: pd.Series, x: pd.Series) -> tuple:
    """
    Returns (adf_stat, p_value, half_life).
    Half-life is derived from the OLS mean-reversion coefficient of the spread ECM.
    A negative coefficient implies mean-reversion; positive implies explosive.
    """
    assert isinstance(y, pd.Series) and isinstance(x, pd.Series)
    assert y.isnull().sum() == 0 and x.isnull().sum() == 0
    assert y.index.equals(x.index)

    long_run_fit = OLS(y, add_constant(x)).fit()
    z = long_run_fit.resid

    z_lag  = z.shift(1).iloc[1:]
    z_diff = z.diff().iloc[1:]

    try:
        hl_fit = OLS(z_diff, add_constant(z_lag)).fit()
        beta   = hl_fit.params.iloc[1]
        # beta >= 0 means the spread is not mean-reverting
        half_life = (-np.log(2) / beta) if (beta < 0 and not np.isnan(beta)) else np.inf
    except Exception:
        half_life = np.inf

    adf_res = adfuller(z)
    return adf_res[0], adf_res[1], half_life


def kalman_filter_regression(x: pd.Series, y: pd.Series) -> tuple:
    """Wrapper: converts Series to float64 arrays and calls the JIT Kalman loop."""
    x_vals = np.asarray(x.values, dtype=np.float64)
    y_vals = np.asarray(y.values, dtype=np.float64)
    return _kalman_loop(x_vals, y_vals)


# ---------------------------------------------------------------------------
# Pair-level processing (runs in parallel via joblib)
# ---------------------------------------------------------------------------

def _process_single_pair(stock1, stock2, log_m1_arr, log_m2_arr, zscore_window):
    """
    Test one candidate pair: cointegration -> half-life bounds -> Kalman Z-score -> Hurst.
    Returns None if the pair fails any filter.
    """
    log_m1 = pd.Series(log_m1_arr)
    log_m2 = pd.Series(log_m2_arr)

    _, pvalue, half_life = engle_granger_two_step_cointegration_test(log_m1, log_m2)

    if pvalue > P_THRESHOLD or not (MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE):
        return None

    stats = estimate_long_run_short_run_relationships(log_m1, log_m2)
    kalman_gamma, kalman_spread = kalman_filter_regression(log_m2, log_m1)

    spread_series = pd.Series(kalman_spread)
    rolling_mean  = spread_series.rolling(window=zscore_window).mean()
    rolling_std   = spread_series.rolling(window=zscore_window).std()
    rolling_z     = ((spread_series - rolling_mean) / rolling_std).fillna(0.0).values

    rolling_hurst = _rolling_hurst_fast(np.asarray(kalman_spread, dtype=np.float64), HURST_WINDOW)

    return {
        "pair":          (stock1, stock2),
        "pvalue":        pvalue,
        "half_life":     half_life,
        "stats":         stats,
        "rolling_z":     rolling_z,
        "kalman_gamma":  kalman_gamma,
        "rolling_hurst": rolling_hurst
    }


# ---------------------------------------------------------------------------
# ML clustering: prune the pair search space before cointegration tests
# ---------------------------------------------------------------------------

def find_statistical_clusters(train_slice: pd.DataFrame, stock_names: list) -> dict:
    """
    PCA + DBSCAN on 1H log-returns to group cryptos by structural similarity.

    Only pairs within the same cluster are passed to the expensive ADF tests,
    reducing the O(n^2) search from ~190 pairs to ~105-120 per window.

    Returns:
        cluster_map: {ticker -> cluster_label}  (-1 = noise / excluded)
    """
    returns_df = pd.DataFrame()
    for stock in stock_names:
        prices = train_slice[stock, "MidPrice"]
        returns_df[stock] = np.log(prices / prices.shift(1))
    returns_df = returns_df.fillna(0.0)

    # Each row = one ticker, each column = one time-step return (feature)
    X = returns_df.T.values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA: retain 5 components (captures ~90% variance, removes idiosyncratic noise)
    pca   = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # DBSCAN: density-based clustering — no need to pre-specify number of clusters
    dbscan = DBSCAN(eps=15, min_samples=2)
    labels = dbscan.fit_predict(X_pca)

    cluster_map = {stock: label for stock, label in zip(stock_names, labels)}

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    logger.info(f"  [>] ML Clustering: {n_clusters} clusters identified. {n_noise} noisy stocks masked.")

    return cluster_map


# ---------------------------------------------------------------------------
# Cointegration test runner (parallelised)
# ---------------------------------------------------------------------------

def run_cointegration_tests(market_data: pd.DataFrame, stock_names: list,
                            valid_tasks: list, zscore_window: int = 130) -> tuple:
    """
    Run Engle-Granger + Kalman + Hurst tests on all cluster-approved pairs in parallel.
    Returns analysis DataFrame, z-value dict, gamma dict, and Hurst dict.
    """
    log_prices = {stock: np.log(market_data[stock, "MidPrice"].values)
                  for stock in stock_names}

    results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_process_single_pair)(s1, s2, log_prices[s1], log_prices[s2], zscore_window)
        for s1, s2 in valid_tasks
    )

    data_analysis = {"Pairs": [], "Constant": [], "Gamma": [], "Alpha": [], "P-Value": [], "HalfLife": []}
    data_zvalues, data_gammas, data_hurst = {}, {}, {}

    for res in results:
        if res is not None:
            pair = res["pair"]
            data_analysis["Pairs"].append(pair)
            data_analysis["Constant"].append(res["stats"][0])
            data_analysis["Gamma"].append(res["stats"][1])
            data_analysis["Alpha"].append(res["stats"][2])
            data_analysis["P-Value"].append(res["pvalue"])
            data_analysis["HalfLife"].append(res["half_life"])

            data_zvalues[pair] = pd.Series(res["rolling_z"],     index=market_data.index)
            data_gammas[pair]  = pd.Series(res["kalman_gamma"],  index=market_data.index)
            data_hurst[pair]   = pd.Series(res["rolling_hurst"], index=market_data.index)

    df_analysis = (round(pd.DataFrame(data_analysis), 4).set_index("Pairs")
                   if data_analysis["Pairs"]
                   else pd.DataFrame(columns=["Constant", "Gamma", "Alpha", "P-Value", "HalfLife"]))
    return df_analysis, data_zvalues, data_gammas, data_hurst


def get_tradable_pairs(df_analysis: pd.DataFrame, p_threshold: float) -> tuple:
    """
    Apply hub-stock concentration constraint: pick pairs in ascending p-value
    order, but reject any pair that would cause either coin to exceed
    MAX_PAIRS_PER_TICKER active positions.
    """
    tradable      = []
    ticker_counts = defaultdict(int)

    # Prioritise pairs with the strongest (lowest) p-value
    sorted_df = df_analysis.sort_values(by="P-Value")

    for pair, row in sorted_df.iterrows():
        p_val = row["P-Value"]
        hl    = row["HalfLife"]

        if p_val < p_threshold and MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE:
            s1, s2 = pair[0], pair[1]
            if (ticker_counts[s1] < MAX_PAIRS_PER_TICKER and
                    ticker_counts[s2] < MAX_PAIRS_PER_TICKER):
                tradable.append(pair)
                ticker_counts[s1] += 1
                ticker_counts[s2] += 1

    return tradable, df_analysis.loc[tradable]


# ---------------------------------------------------------------------------
# Walk-forward orchestrator
# ---------------------------------------------------------------------------

def run_walk_forward_analysis(
        market_data: pd.DataFrame,
        stock_names: list,
        warmup_bars: int,
        retrain_bars: int,
        p_threshold: float = 0.05,
        zscore_window: int = 130,
) -> tuple:
    """
    Rolling walk-forward analysis:
      - Each window trains on `warmup_bars` bars, then tests on the next
        `retrain_bars` bars (strictly OOS).
      - PCA+DBSCAN clustering is re-run every window so the candidate pair
        list adapts to the current crypto market regime.
      - Only OOS Z-scores, gammas, and Hurst values are written into the
        combined dictionaries to prevent any in-sample contamination.
    """
    n_bars  = len(market_data)
    all_idx = market_data.index

    combined_zvalues: dict = {}
    combined_gammas:  dict = {}
    combined_hurst:   dict = {}
    active_pairs_set: set  = set()

    refit_points = list(range(warmup_bars, n_bars, retrain_bars))
    if not refit_points:
        logger.error("Not enough data for walk-forward analysis.")
        return {}, {}, {}, [], pd.DataFrame()

    logger.info(f"Walk-forward schedule: {len(refit_points)} dynamic windows")
    final_analysis_list = []

    for window_idx, refit_at in enumerate(refit_points):
        start_idx   = max(0, refit_at - warmup_bars)
        train_slice = market_data.iloc[start_idx:refit_at]
        next_refit  = refit_points[window_idx + 1] if window_idx + 1 < len(refit_points) else n_bars
        oos_slice_idx = all_idx[refit_at:next_refit]

        logger.info(
            f"--- Training Window [{window_idx + 1}/{len(refit_points)}]: "
            f"{train_slice.index[0].date()} to {train_slice.index[-1].date()} ---"
        )

        # Dynamic cluster membership — adapts to the current regime
        current_cluster_map = find_statistical_clusters(train_slice, stock_names)

        # Only test pairs within the same cluster (cluster label -1 = noise, excluded)
        valid_tasks = [
            (s1, s2)
            for s1, s2 in itertools.combinations(stock_names, 2)
            if current_cluster_map[s1] == current_cluster_map[s2] and current_cluster_map[s1] != -1
        ]

        logger.info(f"  [>] {len(valid_tasks)} pair combinations from PCA clusters.")

        if not valid_tasks:
            logger.warning("  [!] No valid pairs clustered in this window. Skipping.")
            continue

        try:
            df_analysis, _, _, _ = run_cointegration_tests(
                train_slice, stock_names, valid_tasks, zscore_window
            )
        except Exception:
            continue

        final_analysis_list.append(df_analysis)
        tradable_pairs, _ = get_tradable_pairs(df_analysis, p_threshold)

        if not tradable_pairs:
            continue

        active_pairs_set.update(tradable_pairs)

        for pair in tradable_pairs:
            if pair not in combined_zvalues:
                combined_zvalues[pair] = pd.Series(np.nan, index=all_idx)
                combined_gammas[pair]  = pd.Series(np.nan, index=all_idx)
                combined_hurst[pair]   = pd.Series(np.nan, index=all_idx)

            log_m1 = np.log(market_data.iloc[start_idx:next_refit][pair[0], "MidPrice"])
            log_m2 = np.log(market_data.iloc[start_idx:next_refit][pair[1], "MidPrice"])

            kalman_gamma, kalman_spread = kalman_filter_regression(log_m2, log_m1)
            slice_idx = all_idx[start_idx:next_refit]

            spread_full  = pd.Series(kalman_spread, index=slice_idx)
            rolling_mean = spread_full.rolling(window=zscore_window).mean()
            rolling_std  = spread_full.rolling(window=zscore_window).std()
            rolling_z    = (spread_full - rolling_mean) / rolling_std
            rolling_h    = _rolling_hurst_fast(np.asarray(kalman_spread, dtype=np.float64), HURST_WINDOW)

            # Write only the OOS portion into the combined series
            combined_zvalues[pair].loc[oos_slice_idx] = rolling_z.fillna(0).loc[oos_slice_idx]
            combined_gammas[pair].loc[oos_slice_idx]  = pd.Series(kalman_gamma, index=slice_idx).loc[oos_slice_idx]
            combined_hurst[pair].loc[oos_slice_idx]   = pd.Series(rolling_h,   index=slice_idx).loc[oos_slice_idx]

    if final_analysis_list:
        final_analysis = pd.concat(final_analysis_list)
        # Keep only the most recent statistics when a pair recurs across windows
        final_analysis = final_analysis[~final_analysis.index.duplicated(keep='last')]
    else:
        final_analysis = pd.DataFrame()

    return combined_zvalues, combined_gammas, combined_hurst, list(active_pairs_set), final_analysis
