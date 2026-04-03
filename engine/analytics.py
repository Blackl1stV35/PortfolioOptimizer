"""
engine/analytics.py  --  Python-only Core Analytics
====================================================
Server-compatible replacement for julia_bridge.py.
Uses sklearn LedoitWolf for covariance and numpy for Monte Carlo.
Identical outputs -- just without Julia speed (fine for 2-10 assets).
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

log = logging.getLogger(__name__)


def compute_cov(returns_df: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (sklearn)."""
    cov = LedoitWolf().fit(returns_df.values).covariance_
    log.info("  Covariance: LedoitWolf (%dx%d)", *cov.shape)
    return cov


def compute_risk_table(returns: pd.DataFrame, rf: float) -> pd.DataFrame:
    rows = []
    for t in returns.columns:
        r = returns[t].dropna()
        if len(r) < 6:
            continue
        ann = r.mean() * 12
        vol = r.std() * np.sqrt(12)
        sr  = (r.mean() - rf) / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
        neg = r[r < 0]
        semi = neg.std() * np.sqrt(12) if len(neg) > 1 else np.nan
        sortino = ((r.mean() - rf) / neg.std() * np.sqrt(12)
                   if len(neg) > 1 and neg.std() > 0 else np.nan)
        cum  = (1 + r).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        mdd  = float(dd.min())
        v95  = float(np.percentile(r, 5))
        mask = r <= v95
        cv95 = float(r[mask].mean()) if mask.any() else v95
        g = r[r > 0].sum(); l = abs(r[r < 0].sum())
        rows.append({
            "Ticker": t, "Ann. Return": ann, "Ann. Volatility": vol,
            "Sharpe Ratio": sr, "Sortino Ratio": sortino,
            "Max Drawdown": mdd, "Calmar Ratio": ann / abs(mdd) if mdd != 0 else np.nan,
            "VaR 95%": v95, "CVaR 95%": cv95,
            "Omega Ratio": g / l if l > 0 else np.inf, "Semi-Volatility": semi,
        })
    return pd.DataFrame(rows).set_index("Ticker")


def monte_carlo(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    income_rate: float,
    initial_value: float,
    monthly_add: float = 0.0,
    n_paths: int = 5_000,
    n_months: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised numpy bootstrap Monte Carlo."""
    t0       = time.monotonic()
    port_ret = returns_df.values @ weights
    T        = len(port_ret)
    idxs     = np.random.randint(0, T, (n_paths, n_months))
    sampled  = port_ret[idxs]

    vp  = np.zeros((n_paths, n_months))
    ip  = np.zeros((n_paths, n_months))
    val = np.full(n_paths, float(initial_value))
    cum = np.zeros(n_paths)

    for m in range(n_months):
        val  = val * (1.0 + sampled[:, m]) + monthly_add
        inc  = val * income_rate
        cum += inc
        vp[:, m] = val
        ip[:, m] = cum

    log.info("  Monte Carlo: %d paths x %d months in %.2fs",
             n_paths, n_months, time.monotonic() - t0)
    return vp, ip
