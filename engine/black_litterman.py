"""
engine/black_litterman.py  --  Black-Litterman View Integration
===============================================================
Encodes forward-looking investment views from config/views.yaml,
blends them with the market equilibrium using the BL model,
and returns adjusted expected returns for Riskfolio optimisation.

The adjusted mu replaces port.mu before optimisation, making
weight recommendations forward-looking rather than purely historical.

Views format in views.yaml:
  views:
    - ticker: BKLN
      type: absolute          # "absolute" or "relative"
      expected_return: 0.08   # annualised
      confidence: 0.70        # 0-1, your confidence in this view
      rationale: "Floating rate holds up in higher-for-longer environment"

    - ticker_long: BKLN
      ticker_short: ARCC      # relative: BKLN outperforms ARCC by X%
      type: relative
      expected_outperformance: 0.02
      confidence: 0.50
      rationale: "BKLN has lower duration risk vs ARCC if credit spreads widen"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)
VIEWS_FILE = Path("config") / "views.yaml"


def load_views() -> list[dict]:
    if not VIEWS_FILE.exists():
        log.info("  views.yaml not found -- skipping Black-Litterman")
        return []
    with open(VIEWS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    views = data.get("views", [])
    log.info("  Loaded %d BL views from views.yaml", len(views))
    return views


def build_bl_matrices(
    views: list[dict],
    tickers: list[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (P, Q, omega_diag) for Black-Litterman:
      P     : k × n view matrix
      Q     : k × 1 view returns vector
      omega  : k × 1 uncertainty vector (diagonal of Omega)
    Returns (None, None, None) if no valid views.
    """
    n = len(tickers)
    t_idx = {t: i for i, t in enumerate(tickers)}
    P_rows, Q_vals, omega_vals = [], [], []

    for v in views:
        vtype = v.get("type", "absolute")
        conf  = float(v.get("confidence", 0.5))
        # Convert confidence -> uncertainty (tau * sigma^2 proxy, simplified)
        tau   = 1.0 / (conf + 1e-9)   # low conf -> high uncertainty

        if vtype == "absolute":
            ticker = v.get("ticker")
            if ticker not in t_idx:
                log.warning("  BL view: ticker %s not in portfolio -- skipping", ticker)
                continue
            row = np.zeros(n)
            row[t_idx[ticker]] = 1.0
            ret = float(v.get("expected_return", 0.0)) / 12   # annualised -> monthly
            P_rows.append(row)
            Q_vals.append(ret)
            omega_vals.append(tau * 0.001)   # scaled uncertainty

        elif vtype == "relative":
            long_t  = v.get("ticker_long")
            short_t = v.get("ticker_short")
            if long_t not in t_idx or short_t not in t_idx:
                log.warning("  BL relative view: missing ticker -- skipping")
                continue
            row = np.zeros(n)
            row[t_idx[long_t]]  =  1.0
            row[t_idx[short_t]] = -1.0
            ret = float(v.get("expected_outperformance", 0.0)) / 12
            P_rows.append(row)
            Q_vals.append(ret)
            omega_vals.append(tau * 0.001)

    if not P_rows:
        return None, None, None

    P     = np.array(P_rows)
    Q     = np.array(Q_vals).reshape(-1, 1)
    omega = np.array(omega_vals)
    return P, Q, omega


def apply_black_litterman(
    mu_hist: pd.DataFrame,
    cov:     pd.DataFrame,
    views:   list[dict],
    tau:     float = 0.05,
) -> pd.DataFrame:
    """
    mu_hist : port.mu (1 x n DataFrame, monthly)
    cov     : port.cov (n x n DataFrame, monthly)
    tau     : scalar weight on prior uncertainty (typically 1/T)

    Returns adjusted mu as a DataFrame in same format as port.mu.
    Falls back to mu_hist if no valid views.
    """
    tickers = list(cov.columns)
    P, Q, omega = build_bl_matrices(views, tickers)

    if P is None:
        log.info("  BL: no valid views -- using historical mu")
        return mu_hist

    mu_vec = mu_hist.values.flatten().reshape(-1, 1)   # n × 1
    Sigma  = cov.values                                 # n × n

    # Black-Litterman posterior
    tau_S    = tau * Sigma
    Omega    = np.diag(omega)

    # BL formula: mu_bl = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
    #                      [(tau*Sigma)^-1 mu + P' Omega^-1 Q]
    try:
        tS_inv   = np.linalg.inv(tau_S)
        Om_inv   = np.linalg.inv(Omega)
        M        = tS_inv + P.T @ Om_inv @ P
        b        = tS_inv @ mu_vec + P.T @ Om_inv @ Q
        mu_bl    = np.linalg.solve(M, b)
        mu_bl_df = pd.DataFrame(mu_bl.T, columns=tickers, index=mu_hist.index)
        log.info("  BL: posterior mu computed with %d views", len(P))
        for t in tickers:
            delta = float(mu_bl_df[t].iloc[0] - mu_hist[t].iloc[0])
            log.info("    %s: hist_mu=%.4f  bl_mu=%.4f  delta=%+.4f",
                     t, float(mu_hist[t].iloc[0]),
                     float(mu_bl_df[t].iloc[0]), delta)
        return mu_bl_df
    except np.linalg.LinAlgError as e:
        log.warning("  BL matrix inversion failed (%s) -- using historical mu", e)
        return mu_hist


def run_black_litterman(port, cfg: dict) -> bool:
    """
    Mutates port.mu in-place with BL-adjusted expected returns.
    Returns True if BL was applied, False if fallback to historical.
    """
    views = load_views()
    if not views:
        return False

    bl_mu = apply_black_litterman(
        mu_hist=port.mu,
        cov=port.cov,
        views=views,
        tau=0.05,
    )
    port.mu = bl_mu
    return True
