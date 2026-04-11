"""
engine/scenario_analyzer.py  --  What-If Optimizer (Priority 3)
================================================================
Lets the user simulate adding a new ticker to the portfolio and see
the impact on risk, return, income, and Black-Litterman weights
before committing the trade.
"""

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
COMMISSION_USD = 5.34


# ── Helpers ───────────────────────────────────────────────────────────────────
def _derive_holdings(cfg: dict) -> dict:
    shares = defaultdict(int); cost = defaultdict(float)
    for tx in cfg.get("transactions", []):
        if tx["type"] == "BUY":
            t = tx["ticker"]
            shares[t] += tx["shares"]; cost[t] += tx["total_usd"]
        elif tx["type"] == "SELL":
            shares[tx["ticker"]] -= tx["shares"]
    return {t: {"shares": s, "avg_cost": cost[t]/s, "total_cost": cost[t]}
            for t, s in shares.items() if s > 0}


def _fetch_price(ticker: str) -> Optional[float]:
    try:
        raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        close = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        return float(close.dropna().iloc[-1])
    except Exception as exc:
        log.warning("Price fetch %s: %s", ticker, exc)
        return None


def _fetch_returns(tickers: list, start: str = "2022-01-01") -> pd.DataFrame:
    try:
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        return prices.resample("ME").last().pct_change().dropna()
    except Exception as exc:
        log.warning("Returns fetch: %s", exc)
        return pd.DataFrame()


def _risk_metrics(returns: pd.DataFrame, weights: np.ndarray, rf: float = 0.045/12) -> dict:
    if returns.empty:
        return {}
    port_ret = returns.values @ weights
    ann_ret  = port_ret.mean() * 12
    ann_vol  = port_ret.std() * np.sqrt(12)
    sharpe   = (port_ret.mean() - rf) / port_ret.std() * np.sqrt(12) if port_ret.std() > 0 else 0
    v95      = float(np.percentile(port_ret, 5))
    cvar     = float(port_ret[port_ret <= v95].mean()) if (port_ret <= v95).any() else v95
    cum      = (1 + port_ret).cumprod()
    pk       = np.maximum.accumulate(cum)
    mdd      = float(((cum - pk) / pk).min())
    return {
        "ann_return": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "cvar_95": cvar, "max_drawdown": mdd,
    }


def _optimize_weights(returns: pd.DataFrame, rf: float = 0.045/12) -> np.ndarray:
    """Max-Sharpe via Riskfolio, fallback to equal weight."""
    n = len(returns.columns)
    try:
        import riskfolio as rp
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov="ledoit")
        port.sht = False; port.upperlng = 1.0
        w = port.optimization(model="Classic", rm="MV", obj="Sharpe",
                               rf=rf, l=2, hist=True)
        if w is not None and not w.isnull().values.any():
            return w["weights"].values
    except Exception as e:
        log.warning("Riskfolio optimize: %s", e)
    return np.ones(n) / n


# ── Main scenario function ────────────────────────────────────────────────────
def run_addition_scenario(
    cfg: dict,
    candidates: list[dict],          # [{"ticker":"PDI","usd_amount":2000}, ...]
    optimization_strategy: str = "Max Sharpe",
    max_weight: Optional[float] = None,
    rf: float = 0.045/12,
    mc_paths: int = 3000,
    mc_months: int = 60,
) -> dict:
    """
    Simulate adding candidate ticker(s) to the current portfolio.

    Returns
    -------
    dict with keys:
        before, after: risk metrics dicts
        delta: signed differences
        before_weights, after_weights: weight dicts
        income_before, income_after: monthly USD income estimates
        transactions_preview: list of proposed tx dicts (for Apply)
        mc_before, mc_after: (value_paths, income_paths) tuples
        error: str if something failed
    """
    if not candidates:
        return {"error": "No candidates provided"}

    # ── Build current state ───────────────────────────────────────────────────
    holdings  = _derive_holdings(cfg)
    current_tickers = list(holdings.keys())
    blended_yield   = cfg.get("analysis", {}).get(
        "blended_yield_estimate", 0.0844)  # fallback

    # ── Proposed new transactions ─────────────────────────────────────────────
    tx_preview = []
    new_tickers = []
    new_shares_map = {}

    for cand in candidates:
        ticker = cand.get("ticker", "").upper().strip()
        usd    = float(cand.get("usd_amount", 0))
        shares_override = int(cand.get("shares", 0))

        if not ticker:
            continue

        price = _fetch_price(ticker)
        if price is None:
            log.warning("Skipping %s — price unavailable", ticker)
            continue

        if shares_override > 0:
            shares = shares_override
        elif usd > COMMISSION_USD:
            shares = int((usd - COMMISSION_USD) / price)
        else:
            log.warning("Skipping %s — insufficient USD amount", ticker)
            continue

        if shares <= 0:
            continue

        gross = shares * price
        total = gross + COMMISSION_USD
        tx_preview.append({
            "id":            f"T{len(cfg.get('transactions', []))+len(tx_preview)+1:03d}",
            "date":          datetime.today().strftime("%Y-%m-%d"),
            "type":          "BUY",
            "ticker":        ticker,
            "exchange":      "ARCX",
            "currency":      "USD",
            "shares":        shares,
            "price_usd":     round(price, 4),
            "gross_usd":     round(gross, 2),
            "commission_usd":COMMISSION_USD,
            "total_usd":     round(total, 2),
            "note":          f"What-If scenario — {datetime.today().strftime('%Y-%m-%d')}",
        })
        new_tickers.append(ticker)
        new_shares_map[ticker] = shares

    if not tx_preview:
        return {"error": "No valid candidates — check tickers and amounts"}

    # ── Fetch returns for BEFORE universe ─────────────────────────────────────
    all_tickers_before = current_tickers
    returns_before     = _fetch_returns(all_tickers_before)

    # ── Fetch returns for AFTER universe ─────────────────────────────────────
    all_tickers_after = list(dict.fromkeys(current_tickers + new_tickers))
    returns_after     = _fetch_returns(all_tickers_after)

    # Drop tickers with missing data
    returns_before = returns_before.dropna(axis=1, how="all")
    returns_after  = returns_after.dropna(axis=1, how="all")

    if returns_before.empty or returns_after.empty:
        return {"error": "Insufficient price history for optimization"}

    # ── Compute weights ───────────────────────────────────────────────────────
    w_before = _optimize_weights(returns_before, rf)
    w_after  = _optimize_weights(returns_after,  rf)

    # Apply max_weight constraint if set
    if max_weight and 0 < max_weight < 1:
        w_after = np.minimum(w_after, max_weight)
        s = w_after.sum()
        w_after = w_after / s if s > 0 else w_after

    # ── Risk metrics ──────────────────────────────────────────────────────────
    metrics_before = _risk_metrics(returns_before, w_before, rf)
    metrics_after  = _risk_metrics(returns_after,  w_after,  rf)

    delta = {
        k: round(metrics_after.get(k, 0) - metrics_before.get(k, 0), 4)
        for k in ["ann_return", "ann_vol", "sharpe", "cvar_95", "max_drawdown"]
    }

    # ── Income estimate ───────────────────────────────────────────────────────
    # Sum portfolio market value × blended yield / 12
    yields = {
        "BKLN": 0.0707, "ARCC": 0.1066, "PDI": 0.152,
        "PFLT": 0.149, "MAIN": 0.076, "HTGC": 0.12,
    }
    def _income(hld: dict, new_map: dict = None) -> float:
        total = 0.0
        all_h = dict(hld)
        if new_map:
            for t, s in new_map.items():
                p = _fetch_price(t) or 0
                all_h[t] = {"shares": s, "avg_cost": p, "total_cost": s * p}
        for t, h in all_h.items():
            p   = _fetch_price(t) or h["avg_cost"]
            mkt = h["shares"] * p
            y   = yields.get(t, blended_yield)
            total += mkt * y / 12
        return round(total, 2)

    income_before = _income(holdings)
    income_after  = _income(holdings, new_shares_map)

    # ── Monte Carlo (reuse analytics.monte_carlo) ─────────────────────────────
    mc_before_result = mc_after_result = None
    try:
        from engine.analytics import monte_carlo
        fx   = cfg.get("meta", {}).get("fx_usd_thb", 32.68)
        snap = cfg.get("ks_app_snapshot_20260327", {})
        pv   = float(snap.get("market_value_thb", 6338 * fx)) / fx
        ir   = blended_yield / 12

        mc_before_result = monte_carlo(
            returns_before, w_before, ir, pv, 0.0, mc_paths, mc_months)

        pv_after = pv + sum(t["total_usd"] for t in tx_preview)
        ir_after = income_after / pv_after if pv_after > 0 else ir
        mc_after_result  = monte_carlo(
            returns_after,  w_after,  ir_after, pv_after, 0.0, mc_paths, mc_months)
    except Exception as e:
        log.warning("MC failed in scenario: %s", e)

    return {
        "before":               metrics_before,
        "after":                metrics_after,
        "delta":                delta,
        "before_weights":       dict(zip(returns_before.columns, w_before.round(4))),
        "after_weights":        dict(zip(returns_after.columns,  w_after.round(4))),
        "income_before_usd":    income_before,
        "income_after_usd":     income_after,
        "income_delta_usd":     round(income_after - income_before, 2),
        "transactions_preview": tx_preview,
        "mc_before":            mc_before_result,
        "mc_after":             mc_after_result,
        "tickers_before":       list(returns_before.columns),
        "tickers_after":        list(returns_after.columns),
        "error":                None,
    }


def apply_scenario_to_config(cfg: dict, transactions: list[dict]) -> dict:
    """Append confirmed transactions to portfolio config dict (in-memory)."""
    cfg_new = deepcopy(cfg)
    cfg_new.setdefault("transactions", []).extend(transactions)
    cfg_new.setdefault("meta", {})["data_as_of"] = datetime.today().strftime("%Y-%m-%d")
    return cfg_new
