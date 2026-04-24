"""
core.py  —  PortfolioOptimizer shared runtime
===============================================
Single source of truth for:
  - Account loading / YAML I/O
  - Holdings derivation
  - Price / FX caching
  - GitHub push helper
  - Supabase dual-write (Phase 2+)

Import in every page:
    from core import load_cfg, save_cfg, derive_holdings, load_prices,
                     load_accounts, get_active_account, _latest_snapshot,
                     ROOT, t, get_lang, _groq_available
"""

from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
import yfinance as yf

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent

# ── i18n ──────────────────────────────────────────────────────────────────────
try:
    from utils.i18n import t, get_lang, set_lang, lang_toggle_button
except Exception:
    def t(k, **kw): return k        # noqa
    def get_lang(): return "en"     # noqa
    def set_lang(l): pass           # noqa
    def lang_toggle_button(): pass  # noqa

# ── Finnomena (KAsset) ────────────────────────────────────────────────────────
try:
    from utils.finnomena import get_nav, get_kfixed_market_value
    _FINNOMENA = True
except Exception:
    _FINNOMENA = False
    def get_nav(*a, **kw): return {}  # noqa
    def get_kfixed_market_value(*a): return 0.0  # noqa

# ── Groq availability ─────────────────────────────────────────────────────────
def _groq_available() -> bool:
    try:
        if st.secrets.get("groq", {}).get("api_key"):
            return True
    except Exception:
        pass
    return bool(os.environ.get("GROQ_API_KEY"))

# ── Julia availability ────────────────────────────────────────────────────────
@st.cache_resource
def _julia_runtime():
    """Initialise Julia ONCE per server process via cache_resource."""
    try:
        from engine.julia_bridge import _init_julia, backend_info
        ok = _init_julia()
        return {"available": ok, "info": backend_info()}
    except Exception as exc:
        return {"available": False, "info": {"backend": f"Python/NumPy ({exc})"}}

def _julia_available() -> bool:
    return _julia_runtime().get("available", False)

# ═════════════════════════════════════════════════════════════════════════════
# ACCOUNT MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════
ACCOUNTS_FILE = ROOT / "config" / "accounts.yaml"

@st.cache_data(ttl=300)
def load_accounts() -> list[dict]:
    try:
        with open(ACCOUNTS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return [a for a in data.get("accounts", []) if a.get("active", True)]
    except FileNotFoundError:
        return [{"id": "397543-7", "display_name": "Kanokphan S. (USD)",
                 "broker": "K CYBER TRADE", "color": "#1A2E5C", "accent": "#2E5BA8",
                 "yaml_file": "portfolio_397543-7.yaml", "strategy": "Aggressive Income Builder",
                 "base_currency": "USD", "reporting_currency": "THB",
                 "account_type": "US income portfolio", "active": True}]


def _account_yaml_path(account: dict) -> Path:
    fname = account.get("yaml_file", f"portfolio_{account['id']}.yaml")
    return ROOT / "config" / fname


def get_active_account() -> dict:
    accounts = load_accounts()
    aid = st.session_state.get("active_account_id", accounts[0]["id"])
    for a in accounts:
        if a["id"] == aid:
            return a
    return accounts[0]


# ═════════════════════════════════════════════════════════════════════════════
# YAML I/O  (Supabase dual-write ready)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_cfg(account_id: str = "") -> dict:
    accounts = load_accounts()
    if not account_id:
        account_id = st.session_state.get("active_account_id", accounts[0]["id"])
    for a in accounts:
        if a["id"] == account_id:
            path = _account_yaml_path(a)
            try:
                with open(path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except FileNotFoundError:
                legacy = ROOT / "config" / "portfolio.yaml"
                if legacy.exists():
                    with open(legacy, encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
    return {}


def save_cfg(cfg: dict, account_id: str = "") -> None:
    accounts = load_accounts()
    if not account_id:
        account_id = st.session_state.get("active_account_id", accounts[0]["id"])
    for a in accounts:
        if a["id"] == account_id:
            path = _account_yaml_path(a)
            cfg.setdefault("meta", {})["data_as_of"] = str(date.today())
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            # ── Supabase dual-write (Phase 2 hook) ────────────────────────────
            try:
                from utils.db import get_db
                db = get_db()
                if db.available:
                    _supabase_sync(cfg, account_id, db)
            except Exception as exc:
                log.debug("Supabase sync skipped: %s", exc)
            st.cache_data.clear()
            return
    with open(ROOT / "config" / "portfolio.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    st.cache_data.clear()


def _supabase_sync(cfg: dict, account_id: str, db) -> None:
    """Phase 2: sync latest snapshot to Supabase portfolio row."""
    snap   = _latest_snapshot(cfg)
    mkt    = snap.get("market_value_usd") or snap.get("market_value_thb", 0)
    cash_u = cfg.get("cash", {}).get("usd", 0)
    try:
        db.upsert("portfolios", {
            "portfolio_id":  account_id,
            "name":          cfg.get("meta", {}).get("account_id", account_id),
            "fx_usd_thb":    cfg.get("meta", {}).get("fx_usd_thb", 32.68),
            "wht_rate":      cfg.get("settings", {}).get("wht_active", 0.15),
            "updated_at":    datetime.now().isoformat(),
        }, on_conflict="portfolio_id")
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _latest_snapshot(cfg: dict) -> dict:
    keys = sorted([k for k in cfg if k.startswith("ks_app_snapshot_")], reverse=True)
    return cfg.get(keys[0], {}) if keys else {}


def derive_holdings(cfg: dict) -> dict:
    shares: dict[str, int]   = defaultdict(int)
    cost:   dict[str, float] = defaultdict(float)
    for tx in cfg.get("transactions", []):
        ticker = tx.get("ticker", "")
        if tx["type"] in ("BUY", "DRIP"):
            shares[ticker] += int(tx.get("shares", 0))
            cost[ticker]   += float(tx.get("total_usd", 0))
        elif tx["type"] == "SELL":
            shares[ticker] -= int(tx.get("shares", 0))
    return {
        t: {"shares": s, "avg_cost": round(cost[t]/s, 4),
            "total_cost": round(cost[t], 2)}
        for t, s in shares.items() if s > 0
    }


@st.cache_data(ttl=300, show_spinner="Fetching prices…")
def load_prices(tickers: tuple, period: str = "6mo") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
    prices = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.ffill()


@st.cache_data(ttl=300)
def load_fx_series() -> pd.Series:
    raw   = yf.download("THBUSD=X", period="1y", auto_adjust=True, progress=False)
    close = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    return (1.0 / close).dropna()


@st.cache_data(ttl=600)
def load_returns(tickers: tuple) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    prices = load_prices(tickers, period="3y")
    return prices.resample("ME").last().pct_change().dropna().dropna(axis=1, how="all")


# ═════════════════════════════════════════════════════════════════════════════
# GITHUB PUSH  (silent)
# ═════════════════════════════════════════════════════════════════════════════
def github_push(cfg: dict, action: str, extra: list | None = None) -> dict:
    try:
        from utils.github_commit import auto_commit_portfolio
        return auto_commit_portfolio(str(ROOT), cfg, action, extra)
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR  (called from every page)
# ═════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> tuple[dict, dict, dict, float, float, float, tuple]:
    """
    Render the persistent sidebar and return current context.

    Returns
    -------
    (active_acct, cfg, holdings, fx_r, wht, rf_ann, tickers)
    """
    all_accounts = load_accounts()
    if "active_account_id" not in st.session_state:
        st.session_state["active_account_id"] = all_accounts[0]["id"]

    active = get_active_account()
    cfg    = load_cfg(active["id"])
    hold   = derive_holdings(cfg)
    fx_r   = float(cfg.get("meta", {}).get("fx_usd_thb", 32.68))
    wht    = float(cfg.get("settings", {}).get("wht_active", 0.15))
    rf_ann = float(cfg.get("settings", {}).get("risk_free_rate_annual", 0.045))
    ticks  = tuple(sorted(hold.keys()))

    with st.sidebar:
        st.markdown("## 📊 PortfolioOptimizer")

        # ── Account badge ─────────────────────────────────────────────────────
        color = active.get("color", "#1A2E5C")
        meta  = cfg.get("meta", {})
        st.markdown(f"""
<div style='background:{color};border-radius:8px;padding:10px 14px;margin:4px 0 8px'>
  <div style='color:#aab8d8;font-size:10px;font-weight:700;letter-spacing:.08em'>
    {t("account")}</div>
  <div style='color:#fff;font-size:14px;font-weight:700'>{active["id"]}</div>
  <div style='color:#b0c4e0;font-size:11px'>{active.get("display_name","")}</div>
  <div style='color:#7a9ccf;font-size:10px'>{active.get("broker","")}</div>
</div>""", unsafe_allow_html=True)

        st.caption(f"FX {fx_r:.4f}  |  WHT {wht*100:.0f}%  |  {meta.get('data_as_of','—')}")

        # ── Account switcher ──────────────────────────────────────────────────
        if len(all_accounts) > 1:
            st.markdown("**Switch account**")
            cols = st.columns(len(all_accounts))
            for i, acct in enumerate(all_accounts):
                _is_active = acct["id"] == active["id"]
                _btn_style = "primary" if _is_active else "secondary"
                if cols[i].button(
                    acct.get("display_name", acct["id"])[:14],
                    key=f"acct_btn_{acct['id']}",
                    type=_btn_style,
                    use_container_width=True,
                    disabled=_is_active,
                ):
                    st.session_state["active_account_id"] = acct["id"]
                    st.cache_data.clear()
                    st.rerun()

        st.divider()

        # ── Status ────────────────────────────────────────────────────────────
        _gh  = "✅" if _has_pat(cfg) else "⚠️"
        _ai  = "✅ AI" if _groq_available() else "⚠️ AI"
        _db  = "✅ DB" if _db_available() else "⚡ YAML"
        st.caption(f"{_gh} GitHub  {_ai}  {_db}")

        st.divider()
        if st.button(t("refresh"), use_container_width=True):
            st.cache_data.clear(); st.rerun()
        lang_toggle_button()

    return active, cfg, hold, fx_r, wht, rf_ann, ticks


def _has_pat(cfg: dict) -> bool:
    try:
        if st.secrets.get("github", {}).get("pat"):
            return True
    except Exception:
        pass
    return bool(os.environ.get("GITHUB_PAT")) or bool(cfg.get("github", {}).get("pat"))


def _db_available() -> bool:
    try:
        from utils.db import get_db
        return get_db().available
    except Exception:
        return False
