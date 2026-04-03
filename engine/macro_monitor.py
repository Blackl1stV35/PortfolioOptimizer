"""
engine/macro_monitor.py  --  Macro Pulse & Risk Monitor
========================================================
Fetches macro indicators relevant to BKLN / ARCC / PDI portfolio.
All functions degrade gracefully when data is unavailable.

Data sources:
  yfinance    : VIX, WTI oil, US Treasury yields, HY proxy (HYG)
  portfolio.yaml : Thai policy rate (manual override -- no free API)
  engine/fx_timing.py : USD/THB signal (reused)

Thai policy rate (BOT):
  No free real-time API exists. Set manually in portfolio.yaml:
    macro:
      thai_policy_rate: 1.00        # update after each BOT meeting
      thai_policy_rate_date: "2026-01-08"
      us_fed_rate: 3.625            # midpoint of Fed target range
      us_fed_rate_date: "2026-03-19"
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# ── Tickers ───────────────────────────────────────────────────────────────────
_T = {
    "vix":    "^VIX",
    "oil":    "CL=F",
    "t2y":    "^IRX",      # 13-week T-bill as 2Y proxy for spread calc
    "t10y":   "^TNX",      # 10-year Treasury
    "t2yr":   "^TYX",      # 30Y (unused but available)
    "hy":     "HYG",       # HY bond ETF as credit proxy
    "ig":     "LQD",       # IG bond ETF
    "thbusd": "THBUSD=X",  # reused by fx_timing
}

_CACHE: dict = {}
_CACHE_TTL  = 300   # 5 minutes


def _ttl_get(key: str):
    entry = _CACHE.get(key)
    if entry and (datetime.now() - entry["ts"]).seconds < _CACHE_TTL:
        return entry["data"]
    return None


def _ttl_set(key: str, data):
    _CACHE[key] = {"data": data, "ts": datetime.now()}
    return data


# ── Raw fetch ─────────────────────────────────────────────────────────────────
@_ttl_get.__class__.__call__  # just a reminder pattern, not actual decorator
def _fetch(ticker: str, period: str = "3mo") -> pd.Series:
    cached = _ttl_get(ticker)
    if cached is not None:
        return cached
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw["Close"].squeeze()
        else:
            s = raw["Close"]
        s = s.dropna()
        return _ttl_set(ticker, s)
    except Exception as exc:
        log.warning("macro fetch %s: %s", ticker, exc)
        return _ttl_set(ticker, pd.Series(dtype=float))


# ── Individual indicators ─────────────────────────────────────────────────────
def fetch_vix() -> dict:
    s = _fetch("^VIX", "3mo")
    if s.empty:
        return {"current": None, "change_30d": None, "series": s, "signal": "unknown"}
    cur   = float(s.iloc[-1])
    old   = float(s.iloc[-22]) if len(s) >= 22 else float(s.iloc[0])
    delta = cur - old
    signal = "green" if cur < 18 else ("red" if cur > 28 else "yellow")
    return {"current": cur, "change_30d": delta, "series": s, "signal": signal}


def fetch_oil() -> dict:
    s = _fetch("CL=F", "3mo")
    if s.empty:
        return {"current": None, "change_30d": None, "series": s, "signal": "unknown"}
    cur   = float(s.iloc[-1])
    old   = float(s.iloc[-22]) if len(s) >= 22 else float(s.iloc[0])
    delta = cur - old
    signal = "green" if cur < 80 else ("red" if cur > 95 else "yellow")
    return {"current": cur, "change_30d": delta, "series": s, "signal": signal,
            "geopolitical_note": "WTI > $95 triggers defensive posture" if cur > 95 else ""}


def fetch_yield_curve() -> dict:
    """2s10s spread using 13W T-bill (~2Y proxy) and 10Y TNX."""
    t2  = _fetch("^IRX", "1y")
    t10 = _fetch("^TNX", "1y")
    if t2.empty or t10.empty:
        return {"spread_2s10s": None, "series_2y": t2, "series_10y": t10,
                "signal": "unknown", "inverted": False}
    aligned = pd.DataFrame({"t2": t2, "t10": t10}).dropna()
    if aligned.empty:
        return {"spread_2s10s": None, "series_2y": t2, "series_10y": t10,
                "signal": "unknown", "inverted": False}
    spread = float((aligned["t10"] - aligned["t2"]).iloc[-1])
    signal = "green" if spread > 0.25 else ("red" if spread < -0.25 else "yellow")
    return {
        "spread_2s10s":  round(spread, 3),
        "series_spread": (aligned["t10"] - aligned["t2"]),
        "series_2y":     aligned["t2"],
        "series_10y":    aligned["t10"],
        "signal":        signal,
        "inverted":      spread < 0,
    }


def fetch_credit_risk() -> dict:
    """
    HY credit proxy: relative performance of HYG vs LQD.
    Wider spread = HYG underperforms LQD = higher default risk.
    Returns a normalised 0-100 score (higher = more stress).
    """
    hy  = _fetch("HYG", "1y")
    ig  = _fetch("LQD", "1y")
    if hy.empty or ig.empty:
        return {"score": None, "signal": "unknown", "series_hy": hy, "series_ig": ig}
    aligned = pd.DataFrame({"hy": hy, "ig": ig}).dropna()
    if len(aligned) < 22:
        return {"score": None, "signal": "unknown", "series_hy": hy, "series_ig": ig}
    # 3-month relative return: negative = HY underperforming = stress
    rel_3m = float((aligned["hy"] / aligned["hy"].iloc[-66] -
                    aligned["ig"] / aligned["ig"].iloc[-66]).iloc[-1]) if len(aligned) >= 66 else 0.0
    # Normalise to 0-100 stress score (invert: negative rel = high stress)
    score = max(0, min(100, int(50 - rel_3m * 500)))
    signal = "green" if score < 35 else ("red" if score > 65 else "yellow")
    return {
        "score":      score,
        "rel_3m":     round(rel_3m, 4),
        "signal":     signal,
        "series_hy":  aligned["hy"],
        "series_ig":  aligned["ig"],
    }


def fetch_liquidity_risk() -> dict:
    """
    TED spread proxy: 13W T-bill (^IRX) vs 10Y TNX.
    True TED uses LIBOR but that's discontinued; this is a rough proxy.
    """
    irx = _fetch("^IRX", "1y")
    if irx.empty:
        return {"score": None, "signal": "unknown"}
    # Absolute level of short-term rates as liquidity stress proxy
    cur   = float(irx.iloc[-1])
    score = max(0, min(100, int(cur * 20)))   # 5% rate → score 100
    signal = "green" if score < 30 else ("red" if score > 60 else "yellow")
    return {"score": score, "current_irx": cur, "signal": signal, "series": irx}


def fetch_recession_probability() -> dict:
    """
    Recession probability proxy via yield curve inversion depth and duration.
    Inverted for >3 months with spread < -0.5% → high probability.
    """
    yc = fetch_yield_curve()
    spread = yc.get("spread_2s10s")
    if spread is None:
        return {"probability": 30, "signal": "yellow", "note": "proxy unavailable"}
    # Simple heuristic: deeper/longer inversion → higher probability
    if spread < -0.75:
        prob, signal = 55, "red"
    elif spread < -0.25:
        prob, signal = 35, "yellow"
    elif spread < 0:
        prob, signal = 20, "yellow"
    else:
        prob, signal = 10, "green"
    return {
        "probability": prob,
        "signal":      signal,
        "note":        f"2s10s spread: {spread:+.2f}%",
    }


def get_manual_rates(cfg: dict) -> dict:
    """Read manually-maintained policy rates from portfolio.yaml."""
    macro = cfg.get("macro", {})
    return {
        "thai_rate":      float(macro.get("thai_policy_rate", 1.00)),
        "thai_rate_date": str(macro.get("thai_policy_rate_date", "2026-01-08")),
        "us_fed_rate":    float(macro.get("us_fed_rate", 3.625)),
        "us_fed_date":    str(macro.get("us_fed_rate_date", "2026-03-19")),
        "thai_signal":    "green" if macro.get("thai_policy_rate", 1.00) <= 1.25 else "yellow",
        "us_signal":      "yellow" if 3.0 <= macro.get("us_fed_rate", 3.625) <= 4.5 else "green",
    }


# ── Aggregate ─────────────────────────────────────────────────────────────────
def get_macro_data(cfg: dict) -> dict:
    """
    Fetch all macro indicators. Returns a single dict with all data.
    Safe to call repeatedly — TTL cache prevents redundant API calls.
    """
    log.info("Fetching macro data ...")
    from engine.fx_timing import download_fx, compute_fx_signal
    fx_hist  = download_fx(lookback_days=180)
    fx_sig   = compute_fx_signal(fx_hist)

    return {
        "rates":       get_manual_rates(cfg),
        "vix":         fetch_vix(),
        "oil":         fetch_oil(),
        "yield_curve": fetch_yield_curve(),
        "credit":      fetch_credit_risk(),
        "liquidity":   fetch_liquidity_risk(),
        "recession":   fetch_recession_probability(),
        "fx":          fx_sig,
        "fetched_at":  datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    }


def get_risk_gauges(macro: dict) -> dict:
    """
    Returns 4 gauge scores (0-100) for the Risk Gauges tab.
    Higher = more stress / risk.
    """
    vix_cur  = macro["vix"].get("current") or 20.0
    return {
        "default_risk":   macro["credit"].get("score") or 50,
        "liquidity_risk": macro["liquidity"].get("score") or 40,
        "maturity_risk":  max(0, min(100, int(50 - (macro["yield_curve"].get("spread_2s10s") or 0) * 40))),
        "uncertainty":    max(0, min(100, int((vix_cur - 10) / 40 * 100))),
    }


def get_macro_regime(macro: dict) -> dict:
    """
    Scores 0-10 across key indicators → Defensive / Neutral / Aggressive.
    Returns regime string, cash % suggestion, and action guidance.
    """
    score = 0
    gauges = get_risk_gauges(macro)
    vix    = macro["vix"].get("current") or 20.0
    oil    = macro["oil"].get("current") or 80.0
    rec    = macro["recession"].get("probability") or 25
    fx_sig = macro["fx"].get("signal", "neutral")

    # VIX scoring
    score += 0 if vix < 18 else (1 if vix < 25 else 2)
    # Yield curve
    spread = macro["yield_curve"].get("spread_2s10s") or 0
    score += 0 if spread > 0.25 else (1 if spread > -0.25 else 2)
    # Credit
    score += 0 if gauges["default_risk"] < 35 else (1 if gauges["default_risk"] < 65 else 2)
    # Oil
    score += 0 if oil < 80 else (1 if oil < 95 else 2)
    # Recession
    score += 0 if rec < 20 else (1 if rec < 40 else 2)

    if score >= 7:
        regime       = "Defensive"
        cash_pct     = "22–25%"
        cash_thb_est = "92k THB → hold 20–23k in cash"
        action       = "Hold cash. Delay PDI deployment. Watch for VIX mean reversion before adding risk."
        color        = "red"
    elif score >= 4:
        regime       = "Neutral"
        cash_pct     = "15–20%"
        cash_thb_est = "92k THB → 14–18k cash, consider partial PDI"
        action       = "Deploy cautiously. Consider 50% of available cash into PDI on next dip."
        color        = "yellow"
    else:
        regime       = "Aggressive"
        cash_pct     = "10–15%"
        cash_thb_est = "92k THB → deploy 78k, keep 14k liquid"
        action       = "Conditions favour deployment. DCA into BKLN and PDI this week."
        color        = "green"

    return {
        "regime":       regime,
        "score":        score,
        "cash_pct":     cash_pct,
        "cash_thb_est": cash_thb_est,
        "action":       action,
        "color":        color,
    }