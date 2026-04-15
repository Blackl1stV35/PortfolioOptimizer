"""
utils/finnomena.py  —  KAsset / Finnomena NAV Fetcher
======================================================
Fetches daily NAV for Thai mutual funds.

Primary  : Finnomena public page scrape (Cloudflare-gated in CI, works in browser)
Fallback : Manual entry stored in portfolio YAML nav_history
Cache    : st.session_state, TTL 6 hours

Fund codes:
  K-FIXED-A  →  KFIXEDA-A  (Kasikorn fixed-income, ตราสารหนี้)

Usage:
    from utils.finnomena import get_nav, get_nav_history
    nav = get_nav("KFIXEDA-A")   # returns {"nav": 13.2345, "date": "2026-04-15", "source": ...}
"""

from __future__ import annotations
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

_CACHE: dict = {}
_TTL_SECONDS = 21600   # 6 hours

# Known fund → Finnomena API code mappings
FUND_MAP = {
    "KFIXEDA":   "KFIXEDA-A",
    "KFIXEDA-A": "KFIXEDA-A",
    "K-FIXED-A": "KFIXEDA-A",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/html",
    "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
    "Referer":         "https://www.finnomena.com/",
}


def _cache_get(key: str) -> dict | None:
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < _TTL_SECONDS:
        return entry["data"]
    return None


def _cache_set(key: str, data: dict) -> dict:
    _CACHE[key] = {"data": data, "ts": time.time()}
    return data


def get_nav(fund_code: str, cfg: dict | None = None) -> dict:
    """
    Return latest NAV for a Thai mutual fund.

    Returns dict:
        nav       : float | None
        date      : str (YYYY-MM-DD)
        fund_code : str
        source    : "finnomena" | "manual" | "unavailable"
        stale     : bool (True if data is >1 business day old)
    """
    code = FUND_MAP.get(fund_code.upper(), fund_code)
    cached = _cache_get(code)
    if cached:
        return cached

    # ── Attempt 1: Finnomena REST API ─────────────────────────────────────────
    result = _fetch_finnomena_api(code)

    # ── Attempt 2: Finnomena graph data endpoint ───────────────────────────────
    if not result:
        result = _fetch_finnomena_graph(code)

    # ── Fallback: manual entry from YAML nav_history ───────────────────────────
    if not result and cfg:
        result = _from_yaml_history(cfg, fund_code)

    if not result:
        result = {
            "nav":       None,
            "date":      str(date.today()),
            "fund_code": code,
            "source":    "unavailable",
            "stale":     True,
        }

    result["stale"] = _is_stale(result.get("date"))
    return _cache_set(code, result)


def _fetch_finnomena_api(code: str) -> dict | None:
    """Try Finnomena's internal REST endpoint."""
    urls = [
        f"https://finnomena.com/fn3/api/fund/nav/latest/?fund={code}",
        f"https://finnomena.com/fn3/api/fund/public/nav?fund={code}",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=_HEADERS, timeout=6)
            if r.status_code != 200:
                continue
            data = r.json()
            # Finnomena API shape: {"data": {"nav": ..., "date": ...}}
            inner = data.get("data") or data
            if isinstance(inner, list) and inner:
                inner = inner[-1]
            nav  = float(inner.get("nav") or inner.get("value") or 0)
            dt   = str(inner.get("date") or inner.get("nav_date") or date.today())[:10]
            if nav > 0:
                log.info("Finnomena API: %s NAV=%.4f date=%s", code, nav, dt)
                return {"nav": nav, "date": dt, "fund_code": code, "source": "finnomena"}
        except Exception as exc:
            log.debug("Finnomena API attempt failed (%s): %s", url, exc)
    return None


def _fetch_finnomena_graph(code: str) -> dict | None:
    """Try Finnomena's graph/chart data endpoint (used by their web charts)."""
    url = f"https://finnomena.com/fn3/api/fund/nav/q?fund={code}&range=1M"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=6)
        if r.status_code != 200:
            return None
        data = r.json()
        series = data.get("data") or []
        if series:
            last = series[-1]
            nav = float(last.get("value") or last.get("nav") or 0)
            dt  = str(last.get("date") or "")[:10]
            if nav > 0:
                return {"nav": nav, "date": dt, "fund_code": code, "source": "finnomena_graph"}
    except Exception as exc:
        log.debug("Finnomena graph failed: %s", exc)
    return None


def _from_yaml_history(cfg: dict, fund_code: str) -> dict | None:
    """Extract the latest NAV from the manual nav_history in portfolio YAML."""
    history = cfg.get("nav_history", [])
    if not history:
        # Try investments list
        for inv in cfg.get("investments", []):
            if inv.get("fund_code", "").upper() in (fund_code.upper(), "KFIXEDA"):
                mv = inv.get("market_value_thb")
                dt = inv.get("last_manual_update", str(date.today()))
                if mv:
                    return {
                        "nav":              None,
                        "market_value_thb": float(mv),
                        "date":             dt,
                        "fund_code":        fund_code,
                        "source":           "manual",
                    }
        return None

    last = sorted(history, key=lambda x: x.get("date", ""), reverse=True)[0]
    nav = last.get("nav") or None
    mv  = last.get("market_value_thb")
    return {
        "nav":              float(nav) if nav else None,
        "market_value_thb": float(mv) if mv else None,
        "date":             str(last.get("date", date.today())),
        "fund_code":        fund_code,
        "source":           "manual",
    }


def _is_stale(dt_str: str | None) -> bool:
    if not dt_str:
        return True
    try:
        nav_date = datetime.strptime(dt_str[:10], "%Y-%m-%d").date()
        # Allow 3 calendar days (weekends + holidays)
        return (date.today() - nav_date).days > 3
    except Exception:
        return True


def update_nav_in_yaml(cfg: dict, fund_code: str, nav: float,
                        market_value_thb: float | None = None) -> dict:
    """
    Append a new NAV entry to cfg['nav_history'] and return the updated cfg.
    Call save_cfg() afterwards to persist.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    today = str(date.today())
    cfg.setdefault("nav_history", [])
    cfg["nav_history"].append({
        "date":             today,
        "nav":              nav,
        "market_value_thb": market_value_thb,
        "source":           "manual_update",
    })
    # Also update investments list
    for inv in cfg.get("investments", []):
        if inv.get("fund_code", "").upper() == fund_code.upper().replace("-", ""):
            if market_value_thb:
                inv["market_value_thb"]   = market_value_thb
            inv["nav_per_unit"]           = nav
            inv["last_manual_update"]     = today
    cfg["meta"]["data_as_of"] = today
    return cfg


def get_kfixed_market_value(cfg: dict) -> float:
    """Convenience — return total THB market value for this KAsset account."""
    total = 0.0
    for inv in cfg.get("investments", []):
        mv = inv.get("market_value_thb", 0)
        total += float(mv) if mv else 0.0
    # If no investments list, check cash
    if total == 0.0:
        total = float(cfg.get("cash", {}).get("thb", 0))
    return total
