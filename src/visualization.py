"""
visualization.py
================
Institutional-grade charting suite for pairs-trading strategy analysis.

Chart inventory
---------------
01  plot_bid_ask_development          Bid/ask prices + volumes for a pair over time
02  plot_mid_price_check              Ask / Mid / Bid price microstructure sample
03  plot_pvalue_distribution          Histogram of Engle-Granger p-values across pairs
04  plot_tradable_z                   Rolling Kalman Z-score with entry thresholds
05  plot_pair_bid_ask_prices          Dual-axis bid/ask price detail per pair
06  plot_pair_bid_ask_volumes         Dual-axis bid/ask volume detail per pair
07  plot_thresholds                   Threshold sensitivity overlay on Z-score
08  plot_profitability_curve          PnL vs entry-threshold grid-search curve
09  plot_positions_over_time          Long/short position history per pair
10  plot_individual_cumulative_pnl    Single-pair cumulative equity curve
11  plot_total_cumulative_pnl         Portfolio-level cumulative equity curve (all pairs)
12  plot_spread_distribution          Spread / Z-score distribution with normal overlay
13  plot_rolling_metrics              Rolling Sharpe, volatility, and drawdown panel
14  plot_correlation_heatmap          Pairwise mid-price return correlation matrix
15  plot_cointegration_summary        Bar chart of p-values across all screened pairs
16  plot_drawdown_profile             Underwater equity curve (drawdown over time)
17  plot_trade_pnl_distribution       Histogram of per-bar PnL with key quantiles
18  plot_pnl_heatmap                  Monthly PnL heatmap (calendar view)
19  plot_hedge_ratio_evolution        Kalman dynamic hedge-ratio path per pair
20  plot_zscore_heatmap               Z-score magnitude heatmap across all pairs over time
21  plot_pnl_attribution              OOS net profit bar chart ranked by pair contribution
22  plot_capital_utilization          Gross exposure % of capital over time
23  plot_gross_vs_net_pnl             Friction analysis: gross vs net cumulative PnL
"""

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.autolayout": True,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

SAVE_DIR = "results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

_PALETTE = {
    "primary": "#2C7BB6",
    "secondary": "#D7191C",
    "accent": "#1A9641",
    "neutral": "#636363",
    "up": "#27AE60",
    "down": "#E74C3C",
    "mid": "#2980B9",
    "band_pos": "#2ECC71",
    "band_neg": "#E74C3C",
    "alpha": 0.72,
}


def _save(name: str, dpi: int = 150) -> None:
    plt.savefig(os.path.join(SAVE_DIR, name), dpi=dpi, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 01  Bid / Ask Development
# ---------------------------------------------------------------------------

def plot_bid_ask_development(
    stock1: str, stock2: str, data: pd.DataFrame
) -> None:
    """
    Plot bid and ask price evolution alongside bid/ask volumes for two stocks.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(6, 1, figure=fig, hspace=0.08)
    ax1 = fig.add_subplot(gs[:5, 0])
    ax2 = fig.add_subplot(gs[5, 0], sharex=ax1)

    ax1.set_title(f"Bid & Ask Price Development: {stock1} vs {stock2}", fontweight="bold")
    for ticker, color_bid, color_ask in [
        (stock1, _PALETTE["primary"], _PALETTE["secondary"]),
        (stock2, _PALETTE["accent"], _PALETTE["neutral"]),
    ]:
        ax1.plot(data.index, data[ticker, "BidPrice"], color=color_bid, lw=0.9, label=f"{ticker} Bid")
        ax1.plot(data.index, data[ticker, "AskPrice"], color=color_ask, lw=0.9, alpha=0.8, label=f"{ticker} Ask")
    ax1.set_ylabel("Price ($)")
    ax1.legend(ncol=4, loc="upper right")
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.set_title("Bid & Ask Volumes")
    for ticker, color in [(stock1, _PALETTE["primary"]), (stock2, _PALETTE["accent"])]:
        ax2.fill_between(data.index, data[ticker, "BidVolume"], alpha=0.45, color=color, label=f"{ticker} Bid Vol")
        ax2.fill_between(data.index, data[ticker, "AskVolume"], alpha=0.25, color=color, label=f"{ticker} Ask Vol")
    ax2.set_ylabel("Volume")
    ax2.legend(ncol=4, loc="upper right")

    _save(f"01_Bid_Ask_Dev_{stock1}_{stock2}.png")


# ---------------------------------------------------------------------------
# 02  Mid Price Check
# ---------------------------------------------------------------------------

def plot_mid_price_check(stock: str, data: pd.DataFrame) -> None:
    """
    Verify mid-price calculation against raw bid/ask for the first 100 bars.
    """
    sample = data[stock].iloc[:100]
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.fill_between(sample.index, sample["BidPrice"], sample["AskPrice"],
                    alpha=0.18, color=_PALETTE["neutral"], label="Bid-Ask Spread")
    ax.plot(sample.index, sample["AskPrice"], color=_PALETTE["secondary"], lw=1.2, alpha=0.85, label="Ask Price")
    ax.plot(sample.index, sample["MidPrice"], color=_PALETTE["mid"], lw=2, label="Mid Price")
    ax.plot(sample.index, sample["BidPrice"], color=_PALETTE["primary"], lw=1.2, alpha=0.85, label="Bid Price")
    ax.set_title(f"Bid / Mid / Ask Price Microstructure: {stock}", fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.set_xticks([])
    ax.legend(loc="lower left")
    _save(f"02_Mid_Price_Check_{stock}.png")


# ---------------------------------------------------------------------------
# 03  P-Value Distribution
# ---------------------------------------------------------------------------

def plot_pvalue_distribution(df: pd.DataFrame) -> None:
    """
    Histogram of Engle-Granger cointegration p-values with significance marker.
    """
    if "P-Value" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    n, bins, patches = ax.hist(
        df["P-Value"], bins=80, color=_PALETTE["primary"],
        edgecolor="white", lw=0.5, alpha=0.85, density=False
    )
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.05:
            patch.set_facecolor(_PALETTE["down"])

    ax.axvline(x=0.05, color=_PALETTE["down"], ls=":", lw=2, label="Significance Threshold (0.05)")
    pct_sig = (df["P-Value"] < 0.05).mean() * 100
    ax.text(0.055, ax.get_ylim()[1] * 0.88, f"{pct_sig:.1f}% pairs significant",
            fontsize=10, color=_PALETTE["down"])

    ax.set_title("Distribution of Engle-Granger Cointegration P-Values", fontweight="bold")
    ax.set_xlabel("P-Value (Statistical Significance)")
    ax.set_ylabel("Frequency (Number of Pairs)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.legend()
    _save("03_PValue_Distribution.png")


# ---------------------------------------------------------------------------
# 04  Tradable Z-Score
# ---------------------------------------------------------------------------

def plot_tradable_z(
    stock_pairs: list, data_zvalues: dict
) -> None:
    """
    Rolling Kalman Z-score with ±1σ threshold bands and shaded entry zones.
    """
    for pair in stock_pairs:
        zvalue = data_zvalues[pair]
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(zvalue.index, zvalue, color=_PALETTE["primary"], lw=0.9, label="Rolling Z-Score")
        ax.fill_between(zvalue.index, 1, zvalue.where(zvalue > 1), alpha=0.15, color=_PALETTE["down"], label="Short Signal Zone")
        ax.fill_between(zvalue.index, -1, zvalue.where(zvalue < -1), alpha=0.15, color=_PALETTE["up"], label="Long Signal Zone")
        ax.axhline(0.0, color=_PALETTE["neutral"], lw=1.0, alpha=0.5, label="Mean (0)")
        ax.axhline(1.0, color=_PALETTE["band_pos"], ls="--", lw=1.5, label="+1σ Threshold")
        ax.axhline(-1.0, color=_PALETTE["band_neg"], ls="--", lw=1.5, label="-1σ Threshold")
        ax.set_title(f"Rolling Kalman Z-Score: {pair[0]} / {pair[1]}", fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Z-Score")
        ax.legend(loc="upper right", ncol=3)
        _save(f"04_Tradable_Z_{pair[0]}_{pair[1]}.png")


# ---------------------------------------------------------------------------
# 05  Pair Bid / Ask Prices
# ---------------------------------------------------------------------------

def plot_pair_bid_ask_prices(
    stock1: str, stock2: str, tradable_pairs_data: pd.DataFrame
) -> None:
    """
    Dual-axis bid/ask price comparison for the first 100 bars of a traded pair.
    """
    sample = tradable_pairs_data.iloc[:100]
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(sample.index, sample[stock1, "AskPrice"], color=_PALETTE["secondary"], lw=1.1, label=f"{stock1} Ask")
    ax1.plot(sample.index, sample[stock1, "BidPrice"], color=_PALETTE["primary"], lw=1.1, label=f"{stock1} Bid")
    ax1.set_title(f"Bid & Ask Prices: {stock1} / {stock2}", fontweight="bold")
    ax1.set_ylabel(f"{stock1} Price ($)")
    ax1.legend(loc="lower left")
    ax1.set_xticks([])

    ax2 = ax1.twinx()
    ax2.plot(sample.index, sample[stock2, "AskPrice"], color=_PALETTE["accent"], lw=1.1, alpha=0.85, label=f"{stock2} Ask")
    ax2.plot(sample.index, sample[stock2, "BidPrice"], color=_PALETTE["neutral"], lw=1.1, alpha=0.85, label=f"{stock2} Bid")
    ax2.set_ylabel(f"{stock2} Price ($)")
    ax2.legend(loc="upper right")
    _save(f"05_Prices_Detail_{stock1}_{stock2}.png")


# ---------------------------------------------------------------------------
# 06  Pair Bid / Ask Volumes
# ---------------------------------------------------------------------------

def plot_pair_bid_ask_volumes(
    stock1: str, stock2: str, tradable_pairs_data: pd.DataFrame
) -> None:
    """
    Dual-axis bid/ask volume comparison for the first 100 bars of a traded pair.
    """
    sample = tradable_pairs_data.iloc[:100]
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.bar(range(len(sample)), sample[stock1, "BidVolume"].values, color=_PALETTE["primary"], alpha=0.65, label=f"{stock1} Bid Vol")
    ax1.bar(range(len(sample)), sample[stock1, "AskVolume"].values, color=_PALETTE["secondary"], alpha=0.5, label=f"{stock1} Ask Vol")
    ax1.set_title(f"Bid & Ask Volumes: {stock1} / {stock2}", fontweight="bold")
    ax1.set_ylabel(f"{stock1} Volume")
    ax1.legend(loc="lower left")
    ax1.set_xticks([])

    ax2 = ax1.twinx()
    ax2.plot(range(len(sample)), sample[stock2, "BidVolume"].values, color=_PALETTE["accent"], lw=1.2, label=f"{stock2} Bid Vol")
    ax2.plot(range(len(sample)), sample[stock2, "AskVolume"].values, color=_PALETTE["neutral"], lw=1.2, alpha=0.8, label=f"{stock2} Ask Vol")
    ax2.set_ylabel(f"{stock2} Volume")
    ax2.legend(loc="upper right")
    _save(f"06_Volumes_Detail_{stock1}_{stock2}.png")


# ---------------------------------------------------------------------------
# 07  Threshold Sensitivity Overlay
# ---------------------------------------------------------------------------

def plot_thresholds(
    stock_pairs: list, tradable_pairs_data: pd.DataFrame, pnl_reset: pd.DataFrame
) -> None:
    """
    Z-score time series overlaid with the full grid of entry-threshold levels tested.
    """
    for pair in stock_pairs:
        stock1, stock2 = pair
        zvalue = tradable_pairs_data[stock1 + stock2, "Z-Value"]
        pair_thresholds = pnl_reset.loc[pnl_reset["Pairs"] == pair, "Thresholds"].values

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(zvalue.index, zvalue, color=_PALETTE["primary"], lw=0.8, alpha=0.6, label="Z-Score")
        ax.set_title(f"Threshold Sensitivity — {stock1} / {stock2}", fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Z-Score Magnitude")

        for th in pair_thresholds:
            ax.axhline(y=th, color=_PALETTE["up"], ls=":", alpha=0.35)
            ax.axhline(y=-th, color=_PALETTE["down"], ls=":", alpha=0.35)

        custom_legend = [
            Line2D([0], [0], color=_PALETTE["primary"], alpha=0.6),
            Line2D([0], [0], color=_PALETTE["up"], ls=":"),
            Line2D([0], [0], color=_PALETTE["down"], ls=":"),
        ]
        ax.legend(custom_legend, ["Z-Score", "Positive Thresholds", "Negative Thresholds"])
        _save(f"07_Threshold_Lines_{stock1}_{stock2}.png")


# ---------------------------------------------------------------------------
# 08  Profitability Curve
# ---------------------------------------------------------------------------

def plot_profitability_curve(
    stock_pairs: list, pnl_reset: pd.DataFrame
) -> None:
    """
    PnL vs entry-threshold curve with optimal threshold annotated.
    """
    for pair in stock_pairs:
        stock1, stock2 = pair
        pair_data = pnl_reset[pnl_reset["Pairs"] == pair].copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pair_data["Thresholds"], pair_data["PnLs"],
                color=_PALETTE["primary"], lw=2, marker="o", ms=4, label="PnL per Threshold")
        ax.axhline(0, color=_PALETTE["neutral"], lw=0.8, ls="--")
        ax.fill_between(pair_data["Thresholds"], 0, pair_data["PnLs"],
                        where=(pair_data["PnLs"] > 0), alpha=0.15, color=_PALETTE["up"])
        ax.fill_between(pair_data["Thresholds"], 0, pair_data["PnLs"],
                        where=(pair_data["PnLs"] < 0), alpha=0.15, color=_PALETTE["down"])

        if not pair_data.empty:
            best_idx = pair_data["PnLs"].idxmax()
            best_th = pair_data.loc[best_idx, "Thresholds"]
            best_pnl = pair_data.loc[best_idx, "PnLs"]
            ax.axvline(best_th, color=_PALETTE["accent"], lw=1.5, ls="--", label=f"Optimal: {best_th:.3f}σ")
            ax.scatter([best_th], [best_pnl], color=_PALETTE["accent"], s=80, zorder=5)

        ax.set_title(f"Threshold PnL Sensitivity: {stock1} / {stock2}", fontweight="bold")
        ax.set_xlabel("Entry Threshold (σ from Mean)")
        ax.set_ylabel("Final PnL ($)")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.5)
        _save(f"08_Profitability_Curve_{stock1}_{stock2}.png")


# ---------------------------------------------------------------------------
# 09  Positions Over Time
# ---------------------------------------------------------------------------

def plot_positions_over_time(
    stock_pairs_final: list, positions_df: pd.DataFrame
) -> None:
    """
    Signed position time series for each leg of a traded pair.
    """
    for pair in stock_pairs_final:
        stock1, stock2 = pair
        col1 = f"{stock1}_{stock2}_{stock1}"
        col2 = f"{stock1}_{stock2}_{stock2}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
        fig.suptitle(f"Position History: {stock1} / {stock2}", fontweight="bold")

        ax1.fill_between(positions_df.index, 0, positions_df[col1],
                         where=(positions_df[col1] > 0), color=_PALETTE["up"], alpha=0.6, label="Long")
        ax1.fill_between(positions_df.index, 0, positions_df[col1],
                         where=(positions_df[col1] < 0), color=_PALETTE["down"], alpha=0.6, label="Short")
        ax1.axhline(0, color=_PALETTE["neutral"], lw=0.8)
        ax1.set_ylabel(f"{stock1} Position (shares)")
        ax1.legend(loc="upper right")

        ax2.fill_between(positions_df.index, 0, positions_df[col2],
                         where=(positions_df[col2] > 0), color=_PALETTE["up"], alpha=0.6, label="Long")
        ax2.fill_between(positions_df.index, 0, positions_df[col2],
                         where=(positions_df[col2] < 0), color=_PALETTE["down"], alpha=0.6, label="Short")
        ax2.axhline(0, color=_PALETTE["neutral"], lw=0.8)
        ax2.set_ylabel(f"{stock2} Position (shares)")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        _save(f"09_Positions_{stock1}_{stock2}.png")


# ---------------------------------------------------------------------------
# 10  Individual Cumulative PnL
# ---------------------------------------------------------------------------

def plot_individual_cumulative_pnl(
    stock_pairs_final: list, pnl_df: pd.DataFrame
) -> None:
    """
    Single-pair equity curve with running maximum and drawdown bands.
    """
    for pair in stock_pairs_final:
        stock1, stock2 = pair
        cum_col = f"{stock1}{stock2} Cum PnL"
        series = pnl_df[cum_col]

        running_max = series.cummax()
        drawdown = series - running_max

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"Cumulative PnL: {stock1} / {stock2}", fontweight="bold")

        ax1.plot(series.index, series, color=_PALETTE["primary"], lw=1.5, label="Cum PnL")
        ax1.plot(running_max.index, running_max, color=_PALETTE["neutral"], lw=0.8, ls="--", alpha=0.7, label="Running Maximum")
        ax1.fill_between(series.index, series, running_max, alpha=0.15, color=_PALETTE["down"])
        ax1.axhline(0, color=_PALETTE["neutral"], lw=0.8, ls=":")
        ax1.set_ylabel("Profit & Loss ($)")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax1.legend(loc="upper left")

        ax2.fill_between(drawdown.index, drawdown, 0, color=_PALETTE["down"], alpha=0.55, label="Drawdown")
        ax2.set_ylabel("Drawdown ($)")
        ax2.set_xlabel("Date")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax2.legend(loc="lower right")
        plt.tight_layout()
        _save(f"10_CumPnL_{stock1}_{stock2}.png")


# ---------------------------------------------------------------------------
# 11  Total Cumulative PnL
# ---------------------------------------------------------------------------

def plot_total_cumulative_pnl(
    stock_pairs_final: list, pnl_df: pd.DataFrame
) -> None:
    """
    Portfolio-level equity curve overlaid with individual pair contributions.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title("Portfolio Cumulative PnL — All Pairs", fontweight="bold")

    for pair in stock_pairs_final:
        stock1, stock2 = pair
        ax.plot(pnl_df.index, pnl_df[f"{stock1}{stock2} Cum PnL"],
                lw=0.9, alpha=0.55, label=f"{stock1}/{stock2}")

    ax.plot(pnl_df.index, pnl_df["Cum PnL"], color="black", lw=2.5, label="Total Portfolio")
    ax.axhline(0, color=_PALETTE["neutral"], lw=0.8, ls=":")
    ax.set_ylabel("Profit & Loss ($)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    _save("11_Total_Cumulative_PnL.png", dpi=200)


# ---------------------------------------------------------------------------
# 12  Spread Distribution
# ---------------------------------------------------------------------------

def plot_spread_distribution(
    stock_pairs: list, data_zvalues: dict
) -> None:
    """
    Empirical Z-score distribution compared to a fitted normal, with tail shading.
    """
    for pair in stock_pairs:
        zvalue = data_zvalues[pair].dropna()
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(zvalue, bins=100, color=_PALETTE["primary"], alpha=0.65,
                edgecolor="white", lw=0.3, density=True, label="Empirical Distribution")

        mu, sigma = scipy_stats.norm.fit(zvalue)
        x_range = np.linspace(zvalue.min(), zvalue.max(), 400)
        ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, sigma),
                color=_PALETTE["secondary"], lw=2, label=f"Normal Fit (μ={mu:.3f}, σ={sigma:.3f})")

        for thr, color in [(1, _PALETTE["accent"]), (2, _PALETTE["down"])]:
            ax.axvline(thr, color=color, ls="--", lw=1.2, alpha=0.8)
            ax.axvline(-thr, color=color, ls="--", lw=1.2, alpha=0.8)

        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                         zvalue.min(), -2, alpha=0.08, color=_PALETTE["down"])
        ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1],
                         2, zvalue.max(), alpha=0.08, color=_PALETTE["down"])

        kurt = scipy_stats.kurtosis(zvalue)
        skew = scipy_stats.skew(zvalue)
        ax.text(0.97, 0.95, f"Kurtosis: {kurt:.2f}\nSkewness: {skew:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax.set_title(f"Z-Score Distribution: {pair[0]} / {pair[1]}", fontweight="bold")
        ax.set_xlabel("Z-Score")
        ax.set_ylabel("Density")
        ax.legend()
        _save(f"12_Spread_Distribution_{pair[0]}_{pair[1]}.png")


# ---------------------------------------------------------------------------
# 13  Rolling Performance Metrics
# ---------------------------------------------------------------------------

def plot_rolling_metrics(
    pnl_df: pd.DataFrame,
    capital: float = 100_000.0,
    window: int = 78 * 5,
) -> None:
    """
    Three-panel rolling performance dashboard: Sharpe ratio, annualised volatility,
    and underwater drawdown curve.

    Args:
        pnl_df: DataFrame with 'PnL' and 'Cum PnL' columns.
        capital: Initial capital for return normalisation.
        window: Rolling window length in bars (default = 5 trading days of 5-min bars).
    """
    returns = pnl_df["PnL"] / capital
    periods_per_year = 252 * 78

    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(periods_per_year)
    roll_vol = roll_std * np.sqrt(periods_per_year) * 100

    cum = pnl_df["Cum PnL"]
    running_max = cum.cummax()
    drawdown_pct = ((cum - running_max) / (capital + running_max)) * 100

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Rolling Performance Metrics", fontweight="bold", fontsize=14)

    axes[0].plot(roll_sharpe.index, roll_sharpe, color=_PALETTE["primary"], lw=1.1)
    axes[0].axhline(0, color=_PALETTE["neutral"], lw=0.8, ls="--")
    axes[0].axhline(1, color=_PALETTE["accent"], lw=0.8, ls=":", alpha=0.8)
    axes[0].set_ylabel("Rolling Sharpe Ratio")
    axes[0].fill_between(roll_sharpe.index, roll_sharpe, 0,
                          where=(roll_sharpe > 0), alpha=0.12, color=_PALETTE["up"])
    axes[0].fill_between(roll_sharpe.index, roll_sharpe, 0,
                          where=(roll_sharpe < 0), alpha=0.12, color=_PALETTE["down"])

    axes[1].plot(roll_vol.index, roll_vol, color=_PALETTE["secondary"], lw=1.1)
    axes[1].set_ylabel("Rolling Annualised Volatility (%)")
    axes[1].fill_between(roll_vol.index, roll_vol, alpha=0.1, color=_PALETTE["secondary"])

    axes[2].fill_between(drawdown_pct.index, drawdown_pct, 0, color=_PALETTE["down"], alpha=0.55)
    axes[2].plot(drawdown_pct.index, drawdown_pct, color=_PALETTE["down"], lw=0.8)
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    _save("13_Rolling_Metrics.png")


# ---------------------------------------------------------------------------
# 14  Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    market_data: pd.DataFrame, stock_names: list
) -> None:
    """
    Pearson correlation heatmap of log-returns across all tickers.
    """
    returns = pd.DataFrame(
        {s: np.log(market_data[s, "MidPrice"]).diff().dropna() for s in stock_names}
    )
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(stock_names)), max(6, len(stock_names) - 1)))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson Correlation")

    ax.set_xticks(range(len(stock_names)))
    ax.set_yticks(range(len(stock_names)))
    ax.set_xticklabels(stock_names, rotation=45, ha="right")
    ax.set_yticklabels(stock_names)

    for i in range(len(stock_names)):
        for j in range(len(stock_names)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(corr.values[i, j]) < 0.7 else "white")

    ax.set_title("Log-Return Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    _save("14_Correlation_Heatmap.png")


# ---------------------------------------------------------------------------
# 15  Cointegration Summary Bar Chart
# ---------------------------------------------------------------------------

def plot_cointegration_summary(df_analysis: pd.DataFrame) -> None:
    """
    Horizontal bar chart of Engle-Granger p-values for all screened pairs,
    sorted from most to least significant.
    """
    if "P-Value" not in df_analysis.columns:
        return

    sorted_df = df_analysis["P-Value"].sort_values()
    labels = [f"{p[0]}/{p[1]}" for p in sorted_df.index]
    colors = [_PALETTE["up"] if v < 0.05 else _PALETTE["down"] for v in sorted_df.values]

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)))
    bars = ax.barh(labels, sorted_df.values, color=colors, edgecolor="white", lw=0.4)
    ax.axvline(0.05, color=_PALETTE["secondary"], lw=1.5, ls="--", label="5% Significance")
    ax.set_xlabel("Engle-Granger P-Value")
    ax.set_title("Cointegration Significance — All Pair Combinations", fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    _save("15_Cointegration_Summary.png")


# ---------------------------------------------------------------------------
# 16  Drawdown Profile
# ---------------------------------------------------------------------------

def plot_drawdown_profile(
    pnl_df: pd.DataFrame, capital: float = 100_000.0
) -> None:
    """
    Underwater equity curve showing portfolio drawdown depth over time.
    """
    cum = pnl_df["Cum PnL"]
    running_max = cum.cummax()
    drawdown_pct = (cum - running_max) / (capital + running_max) * 100

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(drawdown_pct.index, drawdown_pct, 0, color=_PALETTE["down"], alpha=0.55, label="Drawdown")
    ax.plot(drawdown_pct.index, drawdown_pct, color=_PALETTE["down"], lw=0.9)
    ax.set_title("Portfolio Drawdown Profile (Underwater Equity Curve)", fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.legend()
    _save("16_Drawdown_Profile.png")


# ---------------------------------------------------------------------------
# 17  Trade PnL Distribution
# ---------------------------------------------------------------------------

def plot_trade_pnl_distribution(pnl_df: pd.DataFrame) -> None:
    """
    Histogram of per-bar PnL for active trading periods with percentile markers.
    """
    active_pnl = pnl_df["PnL"][pnl_df["PnL"] != 0].dropna()
    if active_pnl.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(active_pnl, bins=120, color=_PALETTE["primary"], edgecolor="white",
            lw=0.3, alpha=0.75, density=True, label="PnL Distribution")

    mu, sigma = scipy_stats.norm.fit(active_pnl)
    x_range = np.linspace(active_pnl.min(), active_pnl.max(), 400)
    ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, sigma),
            color=_PALETTE["secondary"], lw=2, label="Normal Fit")

    for pct, label, color in [
        (5, "5th Pct", _PALETTE["down"]),
        (25, "25th Pct", _PALETTE["neutral"]),
        (75, "75th Pct", _PALETTE["neutral"]),
        (95, "95th Pct", _PALETTE["up"]),
    ]:
        val = np.percentile(active_pnl, pct)
        ax.axvline(val, color=color, lw=1.2, ls="--", alpha=0.8, label=f"{label}: ${val:,.0f}")

    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_title("Per-Bar Active PnL Distribution", fontweight="bold")
    ax.set_xlabel("PnL per Bar ($)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2)
    _save("17_Trade_PnL_Distribution.png")


# ---------------------------------------------------------------------------
# 18  Monthly PnL Heatmap
# ---------------------------------------------------------------------------

def plot_pnl_heatmap(pnl_df: pd.DataFrame) -> None:
    """
    Calendar heatmap of monthly aggregated PnL, coloured green/red by sign.
    """
    monthly = pnl_df["PnL"].resample("ME").sum()
    if monthly.empty:
        return

    monthly_df = monthly.reset_index()
    monthly_df.columns = ["Date", "PnL"]
    monthly_df["Year"] = monthly_df["Date"].dt.year
    monthly_df["Month"] = monthly_df["Date"].dt.month

    pivot = monthly_df.pivot(index="Year", columns="Month", values="PnL")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.1), max(4, len(pivot) * 0.9)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Monthly PnL ($)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=7.5,
                        color="black" if abs(val) < vmax * 0.6 else "white")

    ax.set_title("Monthly PnL Heatmap", fontweight="bold")
    plt.tight_layout()
    _save("18_PnL_Heatmap.png")


# ---------------------------------------------------------------------------
# 19  Hedge Ratio Evolution
# ---------------------------------------------------------------------------

def plot_hedge_ratio_evolution(
    stock_pairs: list, data_gammas: dict
) -> None:
    """
    Kalman-filter dynamic hedge ratio (gamma) over time with smoothed trend overlay.
    """
    for pair in stock_pairs:
        gamma = data_gammas[pair].dropna()
        smoothed = gamma.rolling(window=78 * 5, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(18, 5))
        ax.plot(gamma.index, gamma, color=_PALETTE["primary"], lw=0.7, alpha=0.55, label="Kalman Gamma (raw)")
        ax.plot(smoothed.index, smoothed, color=_PALETTE["secondary"], lw=1.8, label="5-Day Smoothed Gamma")
        ax.fill_between(gamma.index, gamma, smoothed, alpha=0.08, color=_PALETTE["neutral"])
        ax.set_title(f"Dynamic Hedge Ratio (Kalman Filter): {pair[0]} / {pair[1]}", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Hedge Ratio (γ)")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        _save(f"19_Hedge_Ratio_{pair[0]}_{pair[1]}.png")


# ---------------------------------------------------------------------------
# 20  Z-Score Heatmap across Pairs
# ---------------------------------------------------------------------------

def plot_zscore_heatmap(stock_pairs: list, data_zvalues: dict) -> None:
    """
    Time-series heatmap showing Z-score magnitude across all tradable pairs
    simultaneously, enabling cross-pair signal concentration analysis.
    """
    if not stock_pairs:
        return

    pair_labels = [f"{p[0]}/{p[1]}" for p in stock_pairs]
    first_z = data_zvalues[stock_pairs[0]]

    resample_n = max(1, len(first_z) // 500)
    index_resampled = first_z.iloc[::resample_n].index

    matrix = np.zeros((len(stock_pairs), len(index_resampled)))
    for i, pair in enumerate(stock_pairs):
        z = data_zvalues[pair].reindex(first_z.index).fillna(0).iloc[::resample_n]
        matrix[i, : len(z)] = z.values

    vmax = np.percentile(np.abs(matrix), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(18, max(5, len(stock_pairs) * 0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Z-Score")

    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.set_xlabel("Time (resampled)")
    ax.set_title("Z-Score Heatmap Across All Tradable Pairs", fontweight="bold")
    plt.tight_layout()
    _save("20_ZScore_Heatmap.png")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def plot_pnl_attribution(stock_pairs_final: list, pnl_df: pd.DataFrame) -> None:
    pair_pnls = {}
    for pair in stock_pairs_final:
        col_name = f"{pair[0]}{pair[1]} Cum PnL"
        if col_name in pnl_df.columns:
            pair_pnls[f"{pair[0]}/{pair[1]}"] = pnl_df[col_name].iloc[-1]

    if not pair_pnls: return

    sorted_pnls = dict(sorted(pair_pnls.items(), key=lambda item: item[1]))
    labels = list(sorted_pnls.keys())
    values = list(sorted_pnls.values())
    colors = [_PALETTE["up"] if v > 0 else _PALETTE["down"] for v in values]

    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.5)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", lw=0.5)
    ax.axvline(0, color=_PALETTE["neutral"], lw=1.2, ls="--")
    ax.set_title("OOS PnL Attribution by Pair (Net Profit)", fontweight="bold")
    ax.set_xlabel("Cumulative PnL ($)")

    for bar, val in zip(bars, values):
        x_offset = 100 if val > 0 else -100
        ha = 'left' if val > 0 else 'right'
        ax.text(val + x_offset, bar.get_y() + bar.get_height() / 2, f"${val:,.0f}",
                va='center', ha=ha, fontsize=9, fontweight='bold')
    plt.tight_layout()
    _save("21_PnL_Attribution.png")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def plot_capital_utilization(pos_df: pd.DataFrame, market_data: pd.DataFrame, active_pairs: list,
                             capital: float) -> None:
    gross_exposure = pd.Series(0.0, index=pos_df.index)
    for pair in active_pairs:
        s1, s2 = pair[0], pair[1]
        if s1 in pos_df.columns and s2 in pos_df.columns:
            m1 = (market_data[s1, 'BidPrice'] + market_data[s1, 'AskPrice']) / 2
            m2 = (market_data[s2, 'BidPrice'] + market_data[s2, 'AskPrice']) / 2
            pair_exposure = pos_df[s1].abs() * m1 + pos_df[s2].abs() * m2
            gross_exposure += pair_exposure

    utilization_pct = (gross_exposure / capital) * 100
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(utilization_pct.index, 0, utilization_pct, color=_PALETTE["primary"], alpha=0.3)
    ax.plot(utilization_pct.index, utilization_pct, color=_PALETTE["primary"], lw=1.2)
    avg_utilization = utilization_pct.mean()
    ax.axhline(avg_utilization, color=_PALETTE["accent"], ls="--", lw=1.5,
               label=f"Avg Utilization: {avg_utilization:.1f}%")
    ax.set_title("Capital Utilization (Gross Exposure % of Total Capital)", fontweight="bold")
    ax.set_ylabel("Gross Exposure (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper right")
    plt.tight_layout()
    _save("22_Capital_Utilization.png")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def plot_gross_vs_net_pnl(pnl_df: pd.DataFrame) -> None:
    if "Gross PnL" not in pnl_df.columns: return
    cum_gross = pnl_df["Gross PnL"].cumsum()
    cum_net = pnl_df["Cum PnL"]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(cum_gross.index, cum_gross, color=_PALETTE["accent"], lw=1.5, label="Gross PnL (Pre-Friction)")
    ax.plot(cum_net.index, cum_net, color=_PALETTE["primary"], lw=2, label="Net PnL (Post-Friction)")
    ax.fill_between(cum_net.index, cum_net, cum_gross, color=_PALETTE["down"], alpha=0.15,
                    label="Friction (Commissions & Slippage)")
    ax.axhline(0, color=_PALETTE["neutral"], lw=0.8, ls=":")
    ax.set_title("Gross vs Net Cumulative PnL (Friction Analysis)", fontweight="bold")
    ax.set_ylabel("Profit & Loss ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(loc="upper left")
    plt.tight_layout()
    _save("23_Gross_Vs_Net_PnL.png")