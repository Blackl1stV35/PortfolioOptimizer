"""
engine/fx_timing.py  --  FX Timing Module
==========================================
Tracks USD/THB rate vs rolling history, computes z-score,
and signals the statistically optimal window for THB -> USD conversion.

Logic:
  - High z-score (USD strong vs history) = BAD time to convert THB -> USD
    (you get fewer USD for your THB)
  - Low z-score (USD weak vs history)    = GOOD time to convert THB -> USD
    (you get more USD for your THB -- i.e. USD is cheap)

Output injected into Excel report as a dedicated "FX Timing" sheet.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


# Signal thresholds (z-score of rolling 90-day window)
SIGNAL_STRONG_BUY  = -1.0   # USD unusually weak  -> convert now
SIGNAL_BUY         = -0.5
SIGNAL_NEUTRAL     = 0.5
SIGNAL_AVOID       = 1.0    # USD unusually strong -> wait


def download_fx(lookback_days: int = 365) -> pd.Series:
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        raw = yf.download("THBUSD=X", start=start, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"].squeeze()
        else:
            raw = raw["Close"]
        # THB/USD -> invert to get USD/THB (how many THB per 1 USD)
        fx = (1.0 / raw).dropna()
        fx.name = "USD_THB"
        log.info("  FX: %d days of USD/THB loaded (latest: %.4f)", len(fx), float(fx.iloc[-1]))
        return fx
    except Exception as e:
        log.warning("  FX download failed: %s", e)
        return pd.Series(dtype=float)


def compute_fx_signal(fx: pd.Series, window: int = 90) -> dict:
    if fx.empty or len(fx) < window:
        return {"signal": "unknown", "zscore": np.nan, "current": np.nan,
                "mean_90d": np.nan, "std_90d": np.nan, "percentile": np.nan}

    current   = float(fx.iloc[-1])
    roll_90   = fx.rolling(window)
    mean_90   = float(roll_90.mean().iloc[-1])
    std_90    = float(roll_90.std().iloc[-1])
    zscore    = (current - mean_90) / std_90 if std_90 > 0 else 0.0
    pct       = float((fx <= current).mean() * 100)   # percentile in 1-yr range

    if zscore <= SIGNAL_STRONG_BUY:
        signal = "strong_buy"
        advice = "USD at 90-day low -- excellent time to convert THB -> USD"
    elif zscore <= SIGNAL_BUY:
        signal = "buy"
        advice = "USD slightly weak -- good conversion opportunity"
    elif zscore <= SIGNAL_NEUTRAL:
        signal = "neutral"
        advice = "USD near 90-day average -- conversion timing is normal"
    elif zscore <= SIGNAL_AVOID:
        signal = "caution"
        advice = "USD slightly elevated -- consider waiting a few days"
    else:
        signal = "avoid"
        advice = "USD at 90-day high -- defer THB conversion if possible"

    return {
        "signal":    signal,
        "advice":    advice,
        "zscore":    round(zscore, 3),
        "current":   round(current, 4),
        "mean_90d":  round(mean_90, 4),
        "std_90d":   round(std_90, 4),
        "percentile": round(pct, 1),
        "52w_low":   round(float(fx.min()), 4),
        "52w_high":  round(float(fx.max()), 4),
    }


def dca_budget_usd(thb_amount: float, current_rate: float,
                   signal: dict, commission_usd: float = 13.39) -> dict:
    usd_gross  = thb_amount / current_rate
    usd_net    = usd_gross - commission_usd
    bkln_price = 21.03   # updated by caller ideally, fallback here
    shares_est = int(usd_net / bkln_price) if usd_net > 0 else 0

    # How much would you get at the 90d mean rate?
    mean_rate = signal.get("mean_90d", current_rate)
    usd_at_avg = thb_amount / mean_rate - commission_usd
    fx_impact  = usd_net - usd_at_avg

    return {
        "thb_in":      thb_amount,
        "rate_used":   current_rate,
        "usd_gross":   round(usd_gross, 2),
        "usd_net":     round(usd_net, 2),
        "shares_bkln": shares_est,
        "usd_vs_avg":  round(fx_impact, 2),
    }


def plot_fx_signal(fx: pd.Series, signal: dict, window: int = 90) -> str:
    if fx.empty:
        return ""
    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), facecolor="#FAFBFD")
        roll = fx.rolling(window)
        mean = roll.mean(); std = roll.std()
        upper2 = mean + 2*std; lower2 = mean - 2*std

        ax1 = axes[0]; ax1.set_facecolor("#FAFBFD")
        ax1.plot(fx.index, fx.values, color="#3B8BD4", lw=1.5, label="USD/THB")
        ax1.plot(mean.index, mean.values, color="#888", lw=1, ls="--", label="90d mean")
        ax1.fill_between(fx.index, lower2, upper2, alpha=0.12, color="#3B8BD4")
        ax1.axhline(float(fx.iloc[-1]), color="#1D9E75", lw=1, ls=":")
        ax1.set_ylabel("THB per USD")
        ax1.legend(fontsize=9); ax1.grid(color="#E4E8F0", lw=0.6)
        ax1.set_title("USD/THB rate vs 90-day band", fontsize=11, fontweight="bold")
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

        zscores = (fx - mean) / std
        colors_z = ["#A32D2D" if z > SIGNAL_AVOID else
                    "#BA7517" if z > SIGNAL_NEUTRAL else
                    "#1D9E75" if z < SIGNAL_BUY else "#888"
                    for z in zscores.fillna(0)]
        ax2 = axes[1]; ax2.set_facecolor("#FAFBFD")
        ax2.bar(zscores.index, zscores.values, color=colors_z, width=1.2, alpha=0.8)
        ax2.axhline(0, color="#888", lw=0.8)
        ax2.axhline(SIGNAL_BUY,    color="#1D9E75", lw=0.8, ls="--", label="buy zone")
        ax2.axhline(SIGNAL_AVOID,  color="#A32D2D", lw=0.8, ls="--", label="avoid zone")
        ax2.set_ylabel("Z-score")
        ax2.set_title("USD strength z-score  (negative = cheap USD = buy THB→USD)",
                      fontsize=10, fontweight="bold")
        ax2.legend(fontsize=9); ax2.grid(color="#E4E8F0", lw=0.6)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        path = str(PLOT_DIR / "fx_signal.png")
        fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
        log.info("  FX signal chart saved")
        return path
    except Exception as e:
        log.warning("  FX chart failed: %s", e)
        return ""


def run_fx_analysis(cfg: dict, bkln_price: float = 21.03) -> dict:
    fx     = download_fx(lookback_days=365)
    signal = compute_fx_signal(fx)
    budget = dca_budget_usd(
        cfg.get("analysis", {}).get("dca_monthly_budget_thb", 51000),
        signal.get("current", cfg.get("meta", {}).get("fx_usd_thb", 32.68)),
        signal,
    )
    # update bkln price in budget
    budget["shares_bkln"] = int(budget["usd_net"] / bkln_price) if budget["usd_net"] > 0 else 0
    chart = plot_fx_signal(fx, signal)

    result = {"signal": signal, "budget": budget, "chart": chart, "history": fx}
    log.info("  FX signal: %s  z=%.2f  rate=%.4f  advice: %s",
             signal["signal"], signal["zscore"],
             signal["current"], signal.get("advice",""))
    return result
