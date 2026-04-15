"""
app.py  --  PortfolioOptimizer  —  3-Page Streamlit Dashboard
=============================================================
Priority 2: 3-page sidebar (Dashboard | Intelligence Hub | Analytics Engine)
Priority 3: What-If Optimizer tab inside Analytics Engine
Priority 4: GitHub auto-commit after dividends, trades, and What-If Apply

Deploy: streamlit run app.py
"""

import sys
import io
import warnings
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime, date

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
import yfinance as yf

# ── Multi-account + Groq + Julia + i18n + new features ──────────────────────────
from pathlib import Path as _Path

# ====================== I18N SETUP ======================
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Import translation with protection
try:
    from utils.i18n import t as translate_func, lang_toggle_button
    t = translate_func                     
except Exception as e:
    st.warning("i18n module failed to load")
    def t(key, **kwargs): 
        return key
    def lang_toggle_button(): 
        pass
    
# Finnomena NAV (KAsset mutual fund)
try:
    from utils.finnomena import get_nav, get_kfixed_market_value
    _FINNOMENA = True
except Exception:
    _FINNOMENA = False


def _groq_available() -> bool:
    try:
        import streamlit as st
        if st.secrets.get("groq", {}).get("api_key"): return True
    except Exception: pass
    import os
    return bool(os.environ.get("GROQ_API_KEY"))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PortfolioOptimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's auto-discovered multi-page sidebar entries.
# All navigation is handled by the custom 3-page st.sidebar.radio below.
# pages/_*.py files are excluded by naming convention; this CSS hides the
# residual [data-testid="stSidebarNav"] block that may appear on older versions.
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ACCOUNTS_FILE = ROOT / "config" / "accounts.yaml"
VIEWS_FILE    = ROOT / "config" / "views.yaml"


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-ACCOUNT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_accounts() -> list[dict]:
    """Load account registry from config/accounts.yaml."""
    try:
        with open(ACCOUNTS_FILE, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return [a for a in data.get("accounts", []) if a.get("active", True)]
    except FileNotFoundError:
        # Fallback: single account from legacy portfolio.yaml
        return [{
            "id": "397543-7",
            "display_name": "Kanokphan S.",
            "broker": "K CYBER TRADE (Kasikorn Securities)",
            "color": "#1A2E5C",
            "accent": "#2E5BA8",
            "yaml_file": "portfolio_397543-7.yaml",
            "strategy": "Aggressive Income Builder",
        }]


def _account_yaml_path(account: dict) -> Path:
    fname = account.get("yaml_file", f"portfolio_{account['id']}.yaml")
    return ROOT / "config" / fname


def get_active_account() -> dict:
    """Return the currently selected account dict from session state."""
    accounts = load_accounts()
    active_id = st.session_state.get("active_account_id", accounts[0]["id"])
    for a in accounts:
        if a["id"] == active_id:
            return a
    return accounts[0]


@st.cache_data(ttl=60)
def load_cfg(account_id: str = "") -> dict:
    """Load portfolio YAML for the given account (or active account)."""
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
                # Legacy fallback
                legacy = ROOT / "config" / "portfolio.yaml"
                with open(legacy, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
    return {}


def save_cfg(cfg: dict, account_id: str = ""):
    """Save portfolio YAML for the given account."""
    accounts = load_accounts()
    if not account_id:
        account_id = st.session_state.get("active_account_id", accounts[0]["id"])
    for a in accounts:
        if a["id"] == account_id:
            path = _account_yaml_path(a)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            st.cache_data.clear()
            return
    # Legacy fallback
    with open(ROOT / "config" / "portfolio.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    st.cache_data.clear()


def _latest_snapshot(cfg: dict) -> dict:
    """
    Return the most-recent ks_app_snapshot_* dict regardless of date suffix.
    Falls back to empty dict so callers never crash on a new account.
    """
    snap_keys = sorted(
        [k for k in cfg if k.startswith("ks_app_snapshot_")],
        reverse=True      # lexicographic DESC = newest date first
    )
    if snap_keys:
        return cfg.get(snap_keys[0], {})
    return {}


def derive_holdings(cfg: dict) -> dict:
    shares = defaultdict(int); cost = defaultdict(float); comm = defaultdict(float)
    for tx in cfg.get("transactions", []):
        if tx["type"] == "BUY":
            t = tx["ticker"]
            shares[t] += tx["shares"]; cost[t] += tx["total_usd"]; comm[t] += tx["commission_usd"]
        elif tx["type"] == "SELL":
            shares[tx["ticker"]] -= tx["shares"]
    return {t: {"shares": s, "avg_cost": cost[t]/s, "total_cost": round(cost[t],2)}
            for t, s in shares.items() if s > 0}


@st.cache_data(ttl=300, show_spinner="Fetching prices...")
def load_prices(tickers: tuple) -> pd.DataFrame:
    raw = yf.download(list(tickers), period="6mo", auto_adjust=True, progress=False)
    return (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill()


@st.cache_data(ttl=300, show_spinner="Fetching FX data...")
def load_fx_series() -> pd.Series:
    raw = yf.download("THBUSD=X", period="1y", auto_adjust=True, progress=False)
    close = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    return (1.0 / close).dropna()


def _github_push(cfg: dict, action: str, extra: list = None):
    """Silent GitHub push — never breaks the UI."""
    try:
        from utils.github_commit import auto_commit_portfolio
        result = auto_commit_portfolio(str(ROOT), cfg, action, extra)
        return result
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT SIDEBAR  (multi-account)
# ══════════════════════════════════════════════════════════════════════════════

# ── Initialise session state ──────────────────────────────────────────────────
_all_accounts = load_accounts()
if "active_account_id" not in st.session_state:
    st.session_state["active_account_id"] = _all_accounts[0]["id"]

_active_acct = get_active_account()
cfg      = load_cfg(_active_acct["id"])
holdings = derive_holdings(cfg)
fx_r     = cfg.get("meta", {}).get("fx_usd_thb", 32.68)
wht      = cfg.get("settings", {}).get("wht_active", 0.30)
rf_ann   = cfg.get("settings", {}).get("risk_free_rate_annual", 0.045)
tickers  = tuple(sorted(holdings.keys()))

with st.sidebar:
    st.markdown("## 📊 PortfolioOptimizer")
    st.markdown("---")

    # ── Account badge + switcher ──────────────────────────────────────────────
    _color  = _active_acct.get("color", "#1A2E5C")
    _meta   = cfg.get("meta", {})
    st.markdown(f"""
<div style='background:{_color};border-radius:8px;padding:10px 14px;margin-bottom:8px'>
  <div style='color:#aab8d8;font-size:11px;font-weight:600;letter-spacing:.06em'>ACCOUNT</div>
  <div style='color:#fff;font-size:15px;font-weight:700'>{_active_acct["id"]}</div>
  <div style='color:#b0c4e0;font-size:11px'>{_active_acct.get("display_name","")}</div>
  <div style='color:#7a9ccf;font-size:11px'>{_active_acct.get("broker","")}</div>
</div>
""", unsafe_allow_html=True)

    st.caption(f"Data as of: {_meta.get('data_as_of','—')}")
    st.caption(f"FX: {fx_r:.4f} THB/USD  |  WHT: {wht*100:.0f}%")

    # Account switcher (only if multiple accounts exist)
    if len(_all_accounts) > 1:
        with st.expander("🔄 Switch Account", expanded=False):
            _switch_to = st.selectbox(
                "Account",
                options=[a["id"] for a in _all_accounts],
                index=next(i for i,a in enumerate(_all_accounts) if a["id"] == _active_acct["id"]),
                format_func=lambda aid: next(
                    (f"{a['id']} — {a['display_name']}" for a in _all_accounts if a["id"] == aid),
                    aid
                ),
                key="account_switcher",
                label_visibility="collapsed",
            )
            if _switch_to != _active_acct["id"]:
                if st.button("Switch to this account", key="switch_btn", type="primary"):
                    st.session_state["active_account_id"] = _switch_to
                    st.cache_data.clear()
                    st.rerun()

    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    _nav_options = [t("dashboard"), t("intelligence_hub"), t("analytics_engine")]
    if len(_all_accounts) > 1:
        _nav_options.append(t("family_overview"))

    page = st.radio(
        "Navigation",
        options=_nav_options,
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── Status indicators ─────────────────────────────────────────────────────
    def _has_pat() -> bool:
        try:
            if st.secrets.get("github", {}).get("pat"): return True
        except Exception: pass
        import os
        return bool(os.environ.get("GITHUB_PAT")) or bool(cfg.get("github", {}).get("pat"))
    has_gh = _has_pat()
    st.caption(f"GitHub: {'✅' if has_gh else '⚠️ not set'}  |  "
               f"AI: {'✅ Groq' if _groq_available() else '⚠️ no key'}")

    # Language toggle
    st.markdown("---")
    lang_toggle_button()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == t("dashboard"):

    tab_ov, tab_ph, tab_dc, tab_ta, tab_tx = st.tabs([
        "Overview",
        "Price & History",
        "Dividend Calendar",
        "Tax & Reconciliation",
        "Transactions & Activity",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab_ov:
        st.subheader(t("portfolio_overview"))
        _acct_type = _active_acct.get("account_type", "")

        # ── KAsset / Thai mutual fund account ─────────────────────────────────
        if "mutual fund" in _acct_type.lower():
            _kfixed_mkt = 0.0
            _nav_info   = {}
            if _FINNOMENA:
                _nav_info   = get_nav("KFIXEDA", cfg)
                _kfixed_mkt = get_kfixed_market_value(cfg)
            if not _kfixed_mkt:
                _kfixed_mkt = sum(
                    float(inv.get("market_value_thb", 0))
                    for inv in cfg.get("investments", []))

            if _nav_info.get("stale"):
                st.warning(t("nav_stale") + f" — last: {_nav_info.get('date','?')}")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric(t("total_fund_value"), f"฿{_kfixed_mkt:,.2f}")
            c2.metric("NAV / unit",
                      f"฿{_nav_info['nav']:.4f}" if _nav_info.get("nav") else "Manual entry")
            c3.metric("Source", _nav_info.get("source","manual").replace("_"," ").title())
            c4.metric(t("cash_thb"), f"฿{cfg.get('cash',{}).get('thb',0):,.2f}")

            st.divider()
            for inv in cfg.get("investments", []):
                st.markdown(f"**{inv.get('fund_code','')}** — {inv.get('description','')}")
                _ic1,_ic2,_ic3 = st.columns(3)
                _ic1.metric("Market value", f"฿{inv.get('market_value_thb',0):,.2f}")
                _ic2.metric("Units held", str(inv.get("units_held") or "—"))
                _ic3.metric("Last updated", str(inv.get("last_manual_update","—")))

            st.divider()
            with st.expander(t("manual_nav_entry"), expanded=False):
                with st.form("nav_update_form"):
                    _nu_nav = st.number_input("NAV per unit (฿)", min_value=0.0001,
                                               format="%.4f", value=10.0)
                    _nu_mv  = st.number_input("Total market value (฿)", min_value=0.0,
                                               value=float(_kfixed_mkt))
                    if st.form_submit_button(t("update_nav"), type="primary"):
                        from utils.finnomena import update_nav_in_yaml
                        _cfg_u = update_nav_in_yaml(cfg, "KFIXEDA", _nu_nav, _nu_mv)
                        save_cfg(_cfg_u, _active_acct["id"])
                        st.success(f"NAV updated: ฿{_nu_nav:.4f}  |  market value ฿{_nu_mv:,.2f}")
                        st.rerun()

        else:
            # ── Normal account (USD income or THB cash) ────────────────────────
            snap    = _latest_snapshot(cfg)
            mkt_usd = snap.get("market_value_usd", 0)
            mkt_thb = snap.get("market_value_thb", mkt_usd * fx_r if mkt_usd else 0)
            unr_thb = snap.get("unrealized_thb", snap.get("unrealized_usd", 0) * fx_r)
            div_thb = snap.get("total_dividends_thb", 0)
            cash_u  = cfg.get("cash", {}).get("usd", 0)
            cash_b  = cfg.get("cash", {}).get("thb", 0)

            c1,c2,c3,c4 = st.columns(4)
            if cfg.get("meta",{}).get("base_currency") == "THB" and mkt_usd == 0:
                c1.metric(t("cash_thb"), f"฿{cash_b:,.2f}")
                c2.metric(t("market_value_thb"), f"฿{mkt_thb:,.0f}")
                c3.metric(t("dividends_received"), f"฿{div_thb:,.2f}")
                c4.metric(t("market_value_usd"), f"${mkt_thb/fx_r:,.0f}" if fx_r else "—")
            else:
                c1.metric(t("market_value_thb"), f"฿{mkt_thb:,.0f}",
                           f"฿{unr_thb:+,.0f} {t('unrealised')}")
                c2.metric(t("market_value_usd"), f"${mkt_usd:,.0f}")
                c3.metric(t("dividends_received"), f"฿{div_thb:,.2f}")
                c4.metric(t("cash_usd"), f"${cash_u:.2f}")

            st.divider()
            if tickers:
                prices = load_prices(tickers)
                rows = []
                for tkr, h in holdings.items():
                    curr = float(prices[tkr].iloc[-1]) if tkr in prices.columns else 0
                    mkt  = h["shares"] * curr; unr = mkt - h["total_cost"]
                    rows.append({
                        t("ticker"):   tkr,
                        t("shares"):   h["shares"],
                        t("avg_cost"): f"${h['avg_cost']:.2f}",
                        t("current"):  f"${curr:.2f}",
                        t("mkt_value"):f"${mkt:,.2f}",
                        "P&L":         f"${unr:+,.2f}",
                        "P&L THB":     f"฿{unr*fx_r:+,.0f}",
                        "P&L %":       f"{unr/h['total_cost']*100:+.2f}%" if h["total_cost"] else "—",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

                # ARCC missed dividend alert
                if "ARCC" in holdings:
                    st.warning(
                        "⚠️  **ARCC Q1 2026 dividend MISSED** — bought 2026-03-25, "
                        "ex-date was 2026-03-12.  "
                        "Next: Q2 2026 est. ex ~2026-06-12 · 133 shares × $0.48 = **$63.84 gross**"
                    )

    # ── Tab 2: Price & History ────────────────────────────────────────────────
    with tab_ph:
        st.subheader("Price History (6 months)")
        if tickers:
            prices = load_prices(tickers)
            fig = go.Figure()
            colors = {"BKLN": "#2E5BA8", "ARCC": "#C0392B"}
            for tkr in tickers:
                if tkr in prices.columns:
                    p = prices[tkr].dropna()
                    fig.add_trace(go.Scatter(
                        x=p.index, y=p.values, name=tkr,
                        line_color=colors.get(tkr, "#888"),
                    ))
            fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                               legend=dict(orientation="h"), yaxis_title="Price (USD)")
            st.plotly_chart(fig, width="stretch")

            # Monthly returns
            monthly = prices.resample("ME").last().pct_change().dropna()
            if not monthly.empty:
                st.subheader("Monthly Returns (%)")
                fig2 = go.Figure()
                for tkr in monthly.columns:
                    fig2.add_trace(go.Bar(x=monthly.index, y=monthly[tkr]*100,
                                          name=tkr, marker_color=colors.get(tkr,"#888")))
                fig2.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0),
                                   barmode="group", yaxis_title="%")
                st.plotly_chart(fig2, width="stretch")

    # ── Tab 3: Dividend Calendar (smart checklist) ────────────────────────────
    with tab_dc:
        st.subheader("Dividend Calendar — Smart Checklist")
        st.caption("Check off dividends when received. Hit 'Process' to auto-update portfolio.yaml.")

        # Build upcoming dividend table
        up_rows = []
        instruments = cfg.get("instruments", {})
        for tkr, inst in instruments.items():
            sh = holdings.get(tkr, {}).get("shares", 0)
            if not sh: continue
            for u in inst.get("dividend_policy", {}).get("estimated_upcoming", []):
                amt    = u.get("amount", u.get("amount_per_share_usd", 0.0))
                elig   = u.get("eligible_for_our_shares", True)
                gross  = sh * amt if elig else 0.0
                net_30 = gross * 0.70
                net_15 = gross * 0.85
                up_rows.append({
                    "Confirmed":  False,
                    "Ticker":     tkr,
                    "Period":     u.get("period", u.get("ex", "")),
                    "Ex-date":    u.get("ex", ""),
                    "Pay-date":   u.get("pay", ""),
                    "Shares":     sh if elig else 0,
                    "$/share":    amt,
                    "Gross $":    round(gross, 2),
                    "Net @30%":   round(net_30, 2),
                    "Net @15%":   round(net_15, 2),
                    "KS THB":     "",
                    "Status":     "projected" if elig else "MISSED",
                })

        if up_rows:
            div_df = pd.DataFrame(up_rows)
            col_conf = st.data_editor(
                div_df,
                column_config={
                    "Confirmed": st.column_config.CheckboxColumn(
                        "✅ Confirmed", help="Tick when dividend hits your account"),
                    "KS THB": st.column_config.TextColumn(
                        "KS App THB", help="Paste actual THB received from KS app"),
                },
                disabled=["Ticker","Period","Ex-date","Pay-date","Shares",
                           "$/share","Gross $","Net @30%","Net @15%","Status"],
                width="stretch", hide_index=True,
                key="div_editor",
            )

            if st.button("✅ Process Confirmed Dividends", type="primary"):
                confirmed = col_conf[col_conf["Confirmed"] == True]
                if confirmed.empty:
                    st.warning("No dividends ticked as confirmed.")
                else:
                    cfg_new = dict(cfg)
                    for _, row in confirmed.iterrows():
                        ks_thb = None
                        try:
                            ks_thb = float(row["KS THB"]) if row["KS THB"] else None
                        except Exception:
                            pass
                        new_div = {
                            "period":               str(row["Period"]),
                            "ticker":               str(row["Ticker"]),
                            "shares_eligible":      int(row["Shares"]),
                            "ex_date":              str(row["Ex-date"]),
                            "pay_date":             str(row["Pay-date"]),
                            "amount_per_share_usd": float(row["$/share"]),
                            "gross_usd_estimated":  float(row["Gross $"]),
                            "wht_rate_assumed":     wht,
                            "net_usd_estimated":    float(row["Net @30%"]),
                            "thb_ks_app":           ks_thb,
                            "source":               "Dividend Calendar checklist",
                            "status":               "received",
                        }
                        cfg_new.setdefault("dividends_received", []).append(new_div)

                    cfg_new["meta"]["data_as_of"] = str(date.today())
                    save_cfg(cfg_new, _active_acct["id"])

                    # GitHub auto-commit
                    gh_result = _github_push(cfg_new, "dividend confirmed")
                    gh_msg = "✅ Pushed to GitHub" if gh_result["success"] else f"⚠️ GitHub: {gh_result.get('error','')}"

                    st.success(f"✅ {len(confirmed)} dividend(s) added to portfolio.yaml.  {gh_msg}")
                    st.rerun()
        else:
            st.info("No upcoming dividends found in portfolio.yaml instruments section.")

        # Historical received
        st.divider()
        st.subheader("Received History")
        hist = cfg.get("dividends_received", [])
        if hist:
            hdf = pd.DataFrame(hist)[["period","ticker","shares_eligible","amount_per_share_usd",
                                       "gross_usd_estimated","thb_ks_app","status"]]
            hdf.columns = ["Period","Ticker","Shares","$/share","Gross $","KS THB","Status"]
            # Bug 1 fix: replace None / em-dash with np.nan so PyArrow doesn't crash
            hdf["KS THB"] = (
                hdf["KS THB"]
                .replace({None: np.nan, "—": np.nan, "null": np.nan, "": np.nan})
                .pipe(pd.to_numeric, errors="coerce")
            )
            st.dataframe(hdf, width="stretch", hide_index=True)
        else:
            st.info("No dividends recorded yet.")

        # ── 📅 Export Dividend Calendar (.ics) ───────────────────────────────
        st.divider()
        st.subheader("📅 Export to Calendar")
        st.caption(
            "Download the confirmed dividend history + all upcoming projected dividends "
            "as an .ics file — importable into Google Calendar, Apple Calendar, and Outlook."
        )

        _ics_col1, _ics_col2 = st.columns([1, 3])
        _ics_filter = _ics_col1.radio(
            "Include",
            ["Confirmed + Upcoming", "Upcoming only", "Confirmed only"],
            label_visibility="collapsed",
            horizontal=False,
            key="ics_filter",
        )

        if _ics_col2.button("📥 Generate .ics file", type="primary", key="gen_ics"):
            try:
                from engine.dividend_calendar import build_events
                import io, tempfile
                from pathlib import Path as _ICSPath

                # Build all events
                _all_events = build_events(cfg, holdings)

                # Filter by user selection
                if _ics_filter == "Upcoming only":
                    _events = [e for e in _all_events if "PROJECTED" in e.get("description","")
                               or "proj" in e.get("uid","")]
                elif _ics_filter == "Confirmed only":
                    _events = [e for e in _all_events if "proj" not in e.get("uid","")]
                else:
                    _events = _all_events

                if not _events:
                    st.warning("No events to export with the current filter.")
                else:
                    # Generate ICS bytes without writing to disk
                    try:
                        from icalendar import Calendar as _iCal, Event as _iEvt, Alarm as _iAlarm
                        from datetime import datetime as _idt, timedelta as _itd
                        _cal = _iCal()
                        _cal.add("prodid", "-//PortfolioOptimizer//dividend_calendar//EN")
                        _cal.add("version", "2.0")
                        _cal.add("calscale", "GREGORIAN")
                        _cal.add("x-wr-calname", f"Dividends — {_active_acct['id']}")
                        _cal.add("x-wr-timezone", "Asia/Bangkok")
                        for _ev in _events:
                            _e = _iEvt()
                            _e.add("uid",         _ev["uid"] + "@portfoliooptimizer")
                            _e.add("summary",      _ev["summary"])
                            _e.add("dtstart",      _ev["date"])
                            _e.add("dtend",        _ev["date"] + _itd(days=1))
                            _e.add("description",  _ev["description"])
                            _e.add("dtstamp",      _idt.now())
                            _ad = _ev.get("alarm_days", 0)
                            if _ad > 0:
                                _a = _iAlarm()
                                _a.add("action",      "DISPLAY")
                                _a.add("description", _ev["summary"])
                                _a.add("trigger",     _itd(days=-_ad))
                                _e.add_component(_a)
                            _cal.add_component(_e)
                        _ics_bytes = _cal.to_ical()
                    except ImportError:
                        # icalendar not installed — generate RFC 5545 manually
                        _lines = [
                            "BEGIN:VCALENDAR", "VERSION:2.0",
                            "PRODID:-//PortfolioOptimizer//EN",
                            f"X-WR-CALNAME:Dividends {_active_acct['id']}",
                        ]
                        from datetime import timedelta as _itd
                        for _ev in _events:
                            _ds  = _ev["date"].strftime("%Y%m%d")
                            _de  = (_ev["date"] + _itd(days=1)).strftime("%Y%m%d")
                            _desc = _ev["description"].replace("\n","\\n").replace(",","\\,")
                            _sum  = _ev["summary"].replace(",","\\,")
                            _lines += [
                                "BEGIN:VEVENT",
                                f"UID:{_ev['uid']}@portfoliooptimizer",
                                f"DTSTART;VALUE=DATE:{_ds}",
                                f"DTEND;VALUE=DATE:{_de}",
                                f"SUMMARY:{_sum}",
                                f"DESCRIPTION:{_desc}",
                            ]
                            _ad = _ev.get("alarm_days", 0)
                            if _ad > 0:
                                _lines += [
                                    "BEGIN:VALARM", "ACTION:DISPLAY",
                                    f"DESCRIPTION:{_sum}",
                                    f"TRIGGER:-P{_ad}D", "END:VALARM",
                                ]
                            _lines.append("END:VEVENT")
                        _lines.append("END:VCALENDAR")
                        _ics_bytes = "\r\n".join(_lines).encode("utf-8")

                    from datetime import datetime as _idt2
                    _fname = f"dividends_{_active_acct['id']}_{_idt2.today().strftime('%Y%m%d')}.ics"
                    st.download_button(
                        label=f"⬇️ Download {_fname}  ({len(_events)} events)",
                        data=_ics_bytes,
                        file_name=_fname,
                        mime="text/calendar",
                        key="ics_download_btn",
                    )
                    st.success(
                        f"✅ Calendar ready — {len(_events)} event(s) "
                        f"({_ics_filter}).  "
                        "Import the .ics into Google Calendar / Apple Calendar / Outlook."
                    )

            except Exception as _ics_e:
                st.error(f"ICS generation failed: {_ics_e}")
                import traceback; st.code(traceback.format_exc())

    # ── Tab 4: Tax & Reconciliation ───────────────────────────────────────────
    with tab_tx:
        st.subheader("Withholding Tax Reconciliation")
        try:
            from engine.wht_reconciliation import build_reconciliation, summarise_wht
            records = build_reconciliation(cfg, fx_r)
            summary = summarise_wht(records, cfg)
            verdict_map = {"treaty_15":"success","default_30":"error",
                           "partial":"warning","no_data":"info","overpaid":"error"}
            primary = (max(set([r.verdict for r in records]),
                           key=[r.verdict for r in records].count)
                       if records else "no_data")
            getattr(st, verdict_map.get(primary,"info"))(summary)
            if records:
                rows = [{"Period":r.period,"Ticker":r.ticker,"Gross $":f"${r.gross_usd:.4f}",
                         "Net @30%":f"${r.net_30pct:.4f}","Net @15%":f"${r.net_15pct:.4f}",
                         # Bug 1 fix: use np.nan not "—" so PyArrow can infer float column
                         "KS THB": float(r.ks_thb) if r.ks_thb is not None else np.nan,
                         "Implied WHT":f"{r.implied_wht*100:.1f}%" if r.implied_wht else "n/a",
                         "Verdict":r.verdict} for r in records]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"WHT module error: {e}")

    # ── Tab 5: Transactions & Activity ────────────────────────────────────────
    with st.tabs(["Transactions & Activity"])[0] if False else (tab_ta,)[0]:
        pass

    with tab_ta:
        st.subheader("Transaction Ledger")
        txns = cfg.get("transactions", [])
        if txns:
            st.dataframe(pd.DataFrame(txns), width="stretch", hide_index=True)
        else:
            st.info("No transactions recorded.")

        st.divider()
        st.subheader("Log New Trade")
        with st.form("trade_form"):
            c1,c2,c3 = st.columns(3)
            tx_date   = c1.date_input("Date", value=date.today())
            tx_type   = c2.selectbox("Type", ["BUY","SELL"])
            tx_ticker = c3.text_input("Ticker", "BKLN").upper()
            c4,c5,c6 = st.columns(3)
            tx_shares = c4.number_input("Shares", min_value=1, step=1, value=50)
            tx_price  = c5.number_input("Price (USD)", min_value=0.01, value=21.00, format="%.2f")
            tx_comm   = c6.number_input("Commission", min_value=0.0, value=5.34, format="%.2f")
            tx_exch   = st.selectbox("Exchange", ["ARCX","XNAS","NYSE"])
            tx_note   = st.text_input("Note")
            gross = tx_shares * tx_price
            total = gross + tx_comm if tx_type == "BUY" else gross - tx_comm
            st.caption(f"Gross: ${gross:,.2f}  |  Total: ${total:,.2f}")
            submitted = st.form_submit_button("💾 Save Trade")

        if submitted:
            nid    = f"T{len(txns)+1:03d}"
            new_tx = {
                "id": nid, "date": str(tx_date), "type": tx_type,
                "ticker": tx_ticker, "exchange": tx_exch, "currency": "USD",
                "shares": int(tx_shares), "price_usd": float(tx_price),
                "gross_usd": round(float(gross),2), "commission_usd": float(tx_comm),
                "total_usd": round(float(total),2), "note": tx_note,
            }
            cfg_new = dict(cfg)
            cfg_new.setdefault("transactions", []).append(new_tx)
            cfg_new["meta"]["data_as_of"] = str(date.today())
            save_cfg(cfg_new, _active_acct["id"])

            gh_result = _github_push(cfg_new, f"new trade {nid} {tx_type} {tx_ticker}")
            gh_msg = "✅ GitHub" if gh_result["success"] else f"⚠️ {gh_result.get('error','')[:60]}"
            st.success(f"✅ Saved {nid}: {tx_type} {tx_shares} {tx_ticker} @ ${tx_price:.2f}  |  {gh_msg}")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: INTELLIGENCE HUB
# ══════════════════════════════════════════════════════════════════════════════
elif page == t("intelligence_hub"):

    tab_mp, tab_sec, tab_fx = st.tabs([
        "📡 Macro Pulse", "📋 SEC Intelligence", "💱 FX Timing"
    ])

    # ── Macro Pulse ───────────────────────────────────────────────────────────
    with tab_mp:
        st.subheader("Macro Pulse & Risk Dashboard")
        try:
            from engine.macro_monitor import get_macro_data, get_macro_regime, get_risk_gauges
            with st.spinner("Fetching macro indicators..."):
                macro  = get_macro_data(cfg)
                regime = get_macro_regime(macro)
                gauges = get_risk_gauges(macro)

            reg_map = {"Defensive":"error","Neutral":"warning","Aggressive":"success"}
            getattr(st, reg_map.get(regime["regime"],"info"))(
                f"**Macro Regime: {regime['regime']}** (score {regime['score']}/10)  |  "
                f"Suggested cash: **{regime['cash_pct']}**  |  {regime['action']}"
            )
            rates = macro["rates"]; vix = macro["vix"]; oil = macro["oil"]; fx_sig = macro["fx"]
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Thai Rate",    f"{rates['thai_rate']:.2f}%")
            c2.metric("US Fed Rate",  f"{rates['us_fed_rate']:.2f}%")
            c3.metric("VIX",          f"{vix.get('current',0):.1f}" if vix.get("current") else "n/a",
                       delta=f"{vix.get('change_30d',0):+.1f} 30d" if vix.get("change_30d") else None)
            c4.metric("WTI Oil",      f"${oil.get('current',0):.2f}" if oil.get("current") else "n/a")
            c5.metric("USD/THB",      f"{fx_sig.get('current',0):.4f}",
                       delta=f"z={fx_sig.get('zscore',0):+.2f}" if fx_sig.get("zscore") else None)
            c6.metric("Recession Prob", f"~{macro['recession'].get('probability',0)}%")

            # Gauge bars
            st.subheader("Risk Gauges")
            for label, key in [("Default Risk","default_risk"),("Liquidity Risk","liquidity_risk"),
                                ("Maturity Risk","maturity_risk"),("Uncertainty","uncertainty")]:
                score = gauges.get(key, 0)
                color = "🟢" if score<35 else ("🟡" if score<65 else "🔴")
                st.write(f"{color} **{label}**: {score}/100")
                st.progress(score/100)

        except Exception as e:
            st.error(f"Macro module error: {e}")

    # ── SEC Intelligence (includes fixed BDC screener) ─────────────────────
    with tab_sec:
        st.subheader("SEC EDGAR Intelligence")
        identity = cfg.get("edgar", {}).get("identity", "")

        if not identity:
            st.error("Add `edgar: identity: your@email.com` to `config/portfolio.yaml`")
        else:
            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                "Dividend Declarations", "NAV & NII", "Insider Trades", "BDC Screener"
            ])

            with sub_tab1:
                st.caption("8-K filings — authoritative dividend data")
                try:
                    from engine.edgar_monitor import get_arcc_dividend_declarations
                    with st.spinner("Fetching ARCC 8-K filings..."):
                        divs = get_arcc_dividend_declarations(identity)
                    if divs:
                        st.dataframe(pd.DataFrame(divs), width="stretch", hide_index=True)
                    else:
                        st.info("No recent ARCC dividend declarations found (last 90 days).")
                except Exception as e:
                    st.error(f"Error: {e}")

            with sub_tab2:
                try:
                    from engine.edgar_monitor import get_arcc_fundamentals
                    with st.spinner("Fetching ARCC 10-Q XBRL..."):
                        fund = get_arcc_fundamentals(identity)
                    if fund:
                        nav = fund.get("nav_latest"); nii = fund.get("nii_latest")
                        cov = fund.get("coverage_ratio")
                        c1,c2,c3 = st.columns(3)
                        c1.metric("NAV/share", f"${nav:.2f}" if nav else "n/a",
                                   delta=f"{fund.get('nav_trend_4q',0):+.3f} (4Q)" if fund.get("nav_trend_4q") else None)
                        c2.metric("NII/share (qtr)", f"${nii:.3f}" if nii else "n/a")
                        c3.metric("Dividend Coverage",
                                   f"{cov:.2f}×" if cov else "n/a",
                                   delta="covered" if cov and cov>=1.0 else "⚠️ at risk")
                        if cov:
                            (st.success if cov>=1.15 else st.warning if cov>=1.0 else st.error)(
                                f"Coverage {cov:.2f}× — {'well covered' if cov>=1.15 else 'covered but thin' if cov>=1.0 else 'DIVIDEND AT RISK'}")
                    else:
                        st.info("Fundamentals unavailable.")
                except Exception as e:
                    st.error(f"Error: {e}")

            with sub_tab3:
                try:
                    from engine.edgar_monitor import get_arcc_insider_trades
                    with st.spinner("Fetching Form 4..."):
                        trades = get_arcc_insider_trades(identity)
                    if trades:
                        df = pd.DataFrame(trades)
                        df["value"] = df["value"].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(df, width="stretch", hide_index=True)
                        buys = [t for t in trades if t["type"]=="BUY"]
                        if buys: st.success(f"{len(buys)} open-market insider purchase(s) — management conviction signal")
                    else:
                        st.info("No Form 4 trades in last 90 days.")
                except Exception as e:
                    st.error(f"Error: {e}")

            # ── FIXED BDC Screener (Priority 1) ───────────────────────────────
            with sub_tab4:
                st.caption("Screen BDCs before adding to portfolio. Fixed: no more Series-to-float errors.")
                candidates = st.multiselect(
                    "Select tickers to screen",
                    ["PFLT","MAIN","HTGC","GBDC","ARCC","OBDC","SLRC"],
                    default=["PFLT","MAIN"],
                )
                if st.button("🔍 Run Screen", key="bdc_screen"):
                    from engine.edgar_monitor import screen_bdc_candidate
                    for tkr in candidates:
                        with st.spinner(f"Screening {tkr}..."):
                            res = screen_bdc_candidate(tkr, identity)
                        sig = {"green":"✅","yellow":"🟡","red":"🔴","unknown":"⚪"}.get(res.get("signal",""), "⚪")
                        with st.expander(f"{sig} {tkr}", expanded=True):
                            if res.get("error"):
                                st.error(res["error"])
                            else:
                                c1,c2 = st.columns(2)
                                if res.get("nav_latest"):
                                    c1.metric("NAV/share", f"${res['nav_latest']:.2f}",
                                               delta=f"{res['nav_1yr_chg']*100:+.1f}% (1yr)" if res.get("nav_1yr_chg") else None)
                                if res.get("nii_latest"):
                                    c2.metric("NII/share (qtr)", f"${res['nii_latest']:.3f}")
                                for w in res.get("warnings", []):
                                    st.warning(w)
                                if not res.get("warnings"):
                                    st.success("No red flags in XBRL data")

    # ── FX Timing ─────────────────────────────────────────────────────────────
    with tab_fx:
        st.subheader("FX Timing — USD/THB Conversion Signal")
        try:
            from engine.fx_timing import compute_fx_signal
            fx_hist   = load_fx_series()
            signal    = compute_fx_signal(fx_hist)
            sig_icon  = {"strong_buy":"🟢","buy":"🟢","neutral":"🟡","caution":"🟠","avoid":"🔴"}
            icon      = sig_icon.get(signal.get("signal",""), "⚪")

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("USD/THB Now",  f"{signal['current']:.4f}")
            c2.metric("90-day Mean",  f"{signal['mean_90d']:.4f}")
            c3.metric("Z-score",      f"{signal['zscore']:+.3f}")
            c4.metric("Signal",       f"{icon} {signal.get('signal','').replace('_',' ').title()}")
            st.info(f"**Advice:** {signal.get('advice','')}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fx_hist.index, y=fx_hist.values, name="USD/THB", line_color="#3B8BD4"))
            roll = fx_hist.rolling(90)
            fig.add_trace(go.Scatter(x=roll.mean().index, y=roll.mean().values,
                                      name="90d mean", line=dict(color="#888",dash="dash")))
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, width="stretch")

            budget_thb = st.slider("THB to convert", 10000, 200000, 51000, 1000)
            usd_gross  = budget_thb / signal["current"]
            usd_net    = usd_gross - 5.34
            c1,c2 = st.columns(2)
            c1.metric("USD gross", f"${usd_gross:,.2f}")
            c2.metric("USD net (after commission)", f"${usd_net:,.2f}")
        except Exception as e:
            st.error(f"FX module error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == t("analytics_engine"):

    tab_ro, tab_bt, tab_wi, tab_mc, tab_gp, tab_ch, tab_ex, tab_ai, tab_rpt = st.tabs([
        t("risk_optimisation"),
        t("backtest"),
        t("whatif"),
        t("monte_carlo"),
        t("gen_plan"),
        t("advanced_charts"),
        t("exit_simulator"),
        t("ai_research"),
        t("download_report"),
    ])

    # ── Shared price/returns load ─────────────────────────────────────────────
    @st.cache_data(ttl=600)
    def _get_returns(tickers_t):
        prices = load_prices(tickers_t)
        return prices.resample("ME").last().pct_change().dropna().dropna(axis=1, how="all")

    # ── Tab 1: Risk & Optimisation ────────────────────────────────────────────
    with tab_ro:
        st.subheader("Portfolio Risk & Optimisation")
        if not tickers:
            st.warning("No holdings found."); st.stop()
        with st.spinner("Running optimisation..."):
            try:
                import riskfolio as rp
                from engine.analytics import compute_risk_table
                returns = _get_returns(tickers)
                risk_tbl = compute_risk_table(returns, rf_ann/12)

                # Display risk table
                st.subheader("Risk Metrics")
                fmt_map = {"Ann. Return":"{:.2%}","Ann. Volatility":"{:.2%}",
                           "Sharpe Ratio":"{:.3f}","Sortino Ratio":"{:.3f}",
                           "Max Drawdown":"{:.2%}"}
                disp = risk_tbl.copy()
                for col, fmt in fmt_map.items():
                    if col in disp.columns:
                        disp[col] = disp[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "n/a")
                st.dataframe(disp, width="stretch")

                # Optimal weights
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="ledoit")
                w_dict = {}
                for label, rm, obj in [("Max Sharpe","MV","Sharpe"),("Min Variance","MV","MinRisk"),
                                         ("Min CVaR","CVaR","MinRisk"),("HRP",None,None)]:
                    try:
                        if label == "HRP":
                            hp = rp.HCPortfolio(returns=returns)
                            w  = hp.optimization(model="HRP", codependence="pearson",
                                                  rm="MV", rf=rf_ann/12, linkage="ward", leaf_order=True)
                        else:
                            w = port.optimization(model="Classic", rm=rm, obj=obj,
                                                   rf=rf_ann/12, l=2, hist=True)
                        if w is not None and not w.isnull().values.any():
                            w_dict[label] = w.to_dict()["weights"]
                    except Exception:
                        pass

                if w_dict:
                    st.subheader("Optimal Weights")
                    w_rows = []
                    for strat, ws in w_dict.items():
                        row = {"Strategy": strat}
                        for t, v in ws.items(): row[t] = f"{v:.1%}"
                        w_rows.append(row)
                    st.dataframe(pd.DataFrame(w_rows), width="stretch", hide_index=True)
                    try:
                        from utils.llm_summarizer import summarise_risk, render_summary
                        _risk_text = summarise_risk(
                            risk_tbl.to_dict(orient="index"),
                            w_dict.get("Max Sharpe", {}))
                        render_summary(_risk_text, "risk_opt")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Optimisation error: {e}")

    # ── Tab 2: Backtest ───────────────────────────────────────────────────────
    with tab_bt:
        st.subheader("Walk-Forward Backtest")
        train_w = st.slider("Training window (months)", 12, 36, 24)
        if st.button("▶ Run Backtest"):
            try:
                from engine.backtest import run_walkforward
                returns = _get_returns(tickers)
                with st.spinner("Running..."):
                    bt = run_walkforward(returns, train_months=train_w)
                if not bt:
                    st.warning("Not enough data.")
                else:
                    rows = [{"Strategy": s.replace("_"," ").title(),
                             "Ann. Return": f"{m['ann_return']*100:.2f}%",
                             "Ann. Vol": f"{m['ann_vol']*100:.2f}%",
                             "Sharpe": f"{m['sharpe']:.3f}",
                             "Max DD": f"{m['max_drawdown']*100:.2f}%",
                             "Final $1": f"${m['final_equity']:.3f}"}
                            for s,m in bt["metrics"].items()]
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    if bt.get("equity_chart"):
                        st.image(bt["equity_chart"], use_container_width=True)
                    try:
                        from utils.llm_summarizer import summarise_backtest, render_summary
                        _bt_text = summarise_backtest(bt["metrics"])
                        render_summary(_bt_text, "backtest")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Backtest error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── Tab 3: What-If Optimizer (NEW — Priority 3) ───────────────────────────
    with tab_wi:
        st.subheader("🔬 What-If Optimizer")
        st.caption(
            "Simulate adding a new ticker before committing capital. "
            "See exact impact on Sharpe, CVaR, monthly income, and Monte Carlo projections."
        )

        with st.form("whatif_form"):
            st.subheader("Candidate Positions (up to 3 tickers)")
            # Row headers
            _h1, _h2, _h3 = st.columns([2,2,2])
            _h1.caption("Ticker"); _h2.caption("USD to deploy"); _h3.caption("Shares (0 = use USD)")
            # Ticker rows
            _r1c1, _r1c2, _r1c3 = st.columns([2,2,2])
            wi_t1 = _r1c1.text_input("T1", "PDI", label_visibility="collapsed").upper().strip()
            wi_u1 = _r1c2.number_input("U1", min_value=0.0, value=2000.0, step=100.0, label_visibility="collapsed")
            wi_s1 = _r1c3.number_input("S1", min_value=0, value=0, label_visibility="collapsed")
            _r2c1, _r2c2, _r2c3 = st.columns([2,2,2])
            wi_t2 = _r2c1.text_input("T2", "", placeholder="optional", label_visibility="collapsed").upper().strip()
            wi_u2 = _r2c2.number_input("U2", min_value=0.0, value=0.0, step=100.0, label_visibility="collapsed")
            wi_s2 = _r2c3.number_input("S2", min_value=0, value=0, label_visibility="collapsed")
            _r3c1, _r3c2, _r3c3 = st.columns([2,2,2])
            wi_t3 = _r3c1.text_input("T3", "", placeholder="optional", label_visibility="collapsed").upper().strip()
            wi_u3 = _r3c2.number_input("U3", min_value=0.0, value=0.0, step=100.0, label_visibility="collapsed")
            wi_s3 = _r3c3.number_input("S3", min_value=0, value=0, label_visibility="collapsed")

            st.divider()
            _fa, _fb, _fc = st.columns(3)
            wi_max_wt     = _fa.slider("Max weight per ticker (0=no limit)", 0.0, 1.0, 0.0, 0.05)
            wi_sensitivity = _fb.checkbox("±5% entry price sensitivity", value=False,
                                           help="Run 3 sub-scenarios: base price, +5%, −5%")
            wi_label      = _fc.text_input("Scenario label (optional)", "", placeholder="e.g. PDI 30%")
            wi_submit     = st.form_submit_button("🔬 Run What-If Analysis", type="primary")

        # Build the primary ticker name for confirm dialog
        wi_ticker = wi_t1 if wi_t1 else ""

        if wi_submit and wi_t1:
            from engine.scenario_analyzer import run_addition_scenario, apply_scenario_to_config

            # Build multi-asset candidates list
            _raw = [(wi_t1,wi_u1,wi_s1),(wi_t2,wi_u2,wi_s2),(wi_t3,wi_u3,wi_s3)]
            candidates = [{"ticker":t,"usd_amount":float(u),"shares":int(s)}
                           for t,u,s in _raw if t]

            # Run base + optional ±5% sensitivity sub-scenarios
            _price_mults = {"Base":1.0}
            if wi_sensitivity:
                _price_mults.update({"+5% entry":1.05, "−5% entry":0.95})

            _all_results = {}
            for _slabel, _mult in _price_mults.items():
                _adj = [{"ticker":c["ticker"],
                          "usd_amount": c["usd_amount"]/_mult if c["usd_amount"]>0 else 0,
                          "shares": c["shares"]} for c in candidates]
                _spinner = f"Analysing {', '.join(c['ticker'] for c in candidates)} ({_slabel})..."
                with st.spinner(_spinner):
                    _all_results[_slabel] = run_addition_scenario(
                        cfg=cfg, candidates=_adj,
                        max_weight=wi_max_wt if wi_max_wt > 0 else None,
                        mc_paths=3000, mc_months=60,
                    )

            result = _all_results["Base"]

            # ── Sensitivity comparison table (if enabled) ─────────────────────
            if wi_sensitivity and len(_all_results) > 1:
                st.subheader("Sensitivity Analysis — ±5% Entry Price")
                _sens_rows = []
                for _sl, _sr in _all_results.items():
                    _d = _sr.get("delta", {})
                    _sens_rows.append({
                        "Scenario":     _sl,
                        "ΔSharpe":      f"{_d.get('sharpe',0):+.3f}",
                        "ΔAnn. Return": f"{_d.get('ann_return',0)*100:+.2f}%",
                        "ΔCVaR":        f"{_d.get('cvar_95',0)*100:+.2f}%",
                        "ΔIncome/mo":   f"${_sr.get('income_delta_usd',0):+.2f}",
                        "Income after": f"${_sr.get('income_after_usd',0):.2f}",
                    })
                st.dataframe(pd.DataFrame(_sens_rows), width="stretch", hide_index=True)
                st.divider()

            if result.get("error"):
                st.error(result["error"])
            else:
                # ── Delta metrics ─────────────────────────────────────────
                st.subheader("Impact Summary")
                delta = result["delta"]

                c1,c2,c3,c4,c5 = st.columns(5)
                def _delta_metric(col, label, val, pct=True, good_positive=True):
                    suffix = "%" if pct else ""
                    v_disp = f"{val*100:+.2f}{suffix}" if pct else f"{val:+.3f}"
                    color  = "normal" if (val>0)==good_positive else "inverse"
                    col.metric(label, v_disp, delta_color=color)

                c1.metric("Δ Sharpe",     f"{delta.get('sharpe',0):+.3f}")
                c2.metric("Δ Ann. Return", f"{delta.get('ann_return',0)*100:+.2f}%")
                c3.metric("Δ Volatility",  f"{delta.get('ann_vol',0)*100:+.2f}%")
                c4.metric("Δ CVaR 95%",    f"{delta.get('cvar_95',0)*100:+.2f}%")
                c5.metric("Δ Monthly Income",
                           f"${result['income_delta_usd']:+.2f}/mo",
                           help="Estimated additional monthly dividend income")

                # ── Groq AI executive summary ─────────────────────────────
                try:
                    from utils.llm_summarizer import summarise_whatif, render_summary
                    _wi_text = summarise_whatif(result)
                    render_summary(_wi_text, "whatif",
                        regenerate_fn=lambda force=True: summarise_whatif(result, force=force))
                except Exception:
                    pass

                # ── Before vs After weights ───────────────────────────────
                st.subheader("Portfolio Weights: Before vs After")
                bw = result["before_weights"]; aw = result["after_weights"]
                all_t = sorted(set(list(bw.keys()) + list(aw.keys())))
                wdf = pd.DataFrame({
                    "Ticker":  all_t,
                    "Before":  [f"{bw.get(t,0):.1%}" for t in all_t],
                    "After":   [f"{aw.get(t,0):.1%}" for t in all_t],
                    "Δ Weight":[f"{(aw.get(t,0)-bw.get(t,0))*100:+.1f}%" for t in all_t],
                })
                st.dataframe(wdf, width="stretch", hide_index=True)

                # ── Transaction preview ───────────────────────────────────
                st.subheader("Proposed Transactions")
                tx_prev = result["transactions_preview"]
                st.dataframe(pd.DataFrame(tx_prev), width="stretch", hide_index=True)

                # ── Monte Carlo fan chart ─────────────────────────────────
                if result.get("mc_before") and result.get("mc_after"):
                    st.subheader("Monte Carlo: Before vs After (5-Year, p10/p50/p90)")
                    vp_b, _ = result["mc_before"]
                    vp_a, _ = result["mc_after"]
                    x = list(range(1, 61))
                    fig = go.Figure()
                    # Pre-defined solid hex colours + matching rgba fill colours
                    _mc_palette = {
                        "Before": {"line": "#2E5BA8", "fill": "rgba(46,91,168,0.12)"},
                        "After":  {"line": "#1D9E75", "fill": "rgba(29,158,117,0.12)"},
                    }
                    for vp, name in [(vp_b, "Before"), (vp_a, "After")]:
                        color      = _mc_palette[name]["line"]
                        fill_color = _mc_palette[name]["fill"]
                        p50 = np.percentile(vp, 50, axis=0)
                        p10 = np.percentile(vp, 10, axis=0)
                        p90 = np.percentile(vp, 90, axis=0)
                        fig.add_trace(go.Scatter(x=x, y=p50.tolist(), name=f"{name} p50",
                                                  line=dict(color=color, width=2.5)))
                        fig.add_trace(go.Scatter(x=x, y=p90.tolist(), name=f"{name} p90",
                                                  line=dict(color=color, width=0.8, dash="dot")))
                        fig.add_trace(go.Scatter(x=x, y=p10.tolist(), name=f"{name} p10",
                                                  fill="tonexty",
                                                  fillcolor=fill_color,
                                                  line=dict(color=color, width=0.8, dash="dot")))
                    fig.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0),
                                      xaxis_title="Month", yaxis_title="Portfolio Value ($)",
                                      legend=dict(orientation="h"))
                    st.plotly_chart(fig, width="stretch")

                # ── Apply button ──────────────────────────────────────────
                st.divider()
                st.subheader("Apply to Portfolio")
                _tickers_str = "+".join(c["ticker"] for c in candidates).upper()
                _confirm_key = f"{_tickers_str} CONFIRMED"
                confirm_txt = st.text_input(
                    f'Type "{_confirm_key}" to apply',
                    placeholder=_confirm_key,
                )
                # ── Sandbox vs Live apply toggle ──────────────────────────
                _sandbox = st.checkbox(
                    "🔬 Sandbox mode — preview only, never saves to portfolio.yaml",
                    value=True, key="whatif_sandbox",
                    help="In sandbox mode you can freely add/remove any asset with zero risk."
                )
                if _sandbox:
                    st.info(
                        "**Sandbox active** — results are preview-only. "
                        "Uncheck to apply the transaction to your live portfolio."
                    )
                else:
                    if st.button("✅ Apply What-If to portfolio.yaml", type="primary"):
                        if confirm_txt.strip().upper() == _confirm_key:
                            cfg_new = apply_scenario_to_config(cfg, tx_prev)
                            save_cfg(cfg_new, _active_acct["id"])
                            gh_result = _github_push(cfg_new, f"What-If applied: {_tickers_str}")
                            gh_msg = "✅ GitHub sync" if gh_result["success"] else f"⚠️ {gh_result.get('error','')[:60]}"
                            st.success(f"✅ Transaction(s) applied.  {gh_msg}")
                            st.rerun()
                        else:
                            st.error("Confirmation text doesn't match — not applied.")

    # ── Tab 4: Monte Carlo ────────────────────────────────────────────────────
    with tab_mc:
        st.subheader("Monte Carlo Income Projection")
        c1,c2,c3 = st.columns(3)
        n_paths = c1.select_slider("Paths", [1000,3000,5000,10000], 3000)
        n_years = c2.slider("Horizon (years)", 1, 10, 5)
        mo_add  = c3.number_input("Monthly DCA add (USD)", 0, 5000, 200, 50)

        if st.button("▶ Run Monte Carlo"):
            try:
                from engine.analytics import monte_carlo
                import riskfolio as rp
                returns = _get_returns(tickers)
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="ledoit")
                # Fix 3: guard solver failure with EW fallback
                try:
                    w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe",
                                               rf=rf_ann/12, l=2, hist=True)
                    if w_opt is None or w_opt.isnull().values.any():
                        raise ValueError("Solver returned null weights")
                    w = w_opt["weights"].values
                except Exception as _opt_exc:
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "Riskfolio solver failed (%s) — falling back to Equal Weight", _opt_exc)
                    w = np.ones(len(returns.columns)) / len(returns.columns)
                snap = _latest_snapshot(cfg)
                pv = float(snap.get("market_value_thb", 6338*fx_r)) / fx_r
                try:
                    from engine.julia_bridge import monte_carlo as _jl_mc
                    vp, ip = _jl_mc(returns, w, 0.0707/12, pv, float(mo_add), n_paths, n_years*12)
                except Exception:
                    from engine.analytics import monte_carlo
                    vp, ip = monte_carlo(returns, w, 0.0707/12, pv, float(mo_add), n_paths, n_years*12)
                x = list(range(1, n_years*12+1))
                p50 = np.percentile(vp,50,axis=0); p10 = np.percentile(vp,10,axis=0); p90 = np.percentile(vp,90,axis=0)
                fig = go.Figure([
                    go.Scatter(x=x,y=p90.tolist(),name="p90",line_color="rgba(29,158,117,0.3)",showlegend=False),
                    go.Scatter(x=x,y=p10.tolist(),name="p10–p90",fill="tonexty",
                               fillcolor="rgba(29,158,117,0.15)",line_color="rgba(29,158,117,0.3)"),
                    go.Scatter(x=x,y=p50.tolist(),name="Median",line=dict(color="#1D9E75",width=2.5)),
                ])
                fig.update_layout(title="Portfolio value",height=340,
                                  xaxis_title="Month",yaxis_title="USD",margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig, width="stretch")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("p50 Final", f"${np.percentile(vp[:,-1],50):,.0f}")
                c2.metric("p10 Final", f"${np.percentile(vp[:,-1],10):,.0f}")
                c3.metric("p90 Final", f"${np.percentile(vp[:,-1],90):,.0f}")
                c4.metric("p50 Income/mo", f"${np.percentile(vp[:,-1],50)*0.0707/12:,.2f}")
                try:
                    from utils.llm_summarizer import summarise_monte_carlo, render_summary
                    _mc_text = summarise_monte_carlo(
                        {"p50":float(np.percentile(vp[:,-1],50)),
                         "p10":float(np.percentile(vp[:,-1],10)),
                         "p90":float(np.percentile(vp[:,-1],90))},
                        n_years, 1000.0)
                    render_summary(_mc_text, "mc")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"MC error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── Tab 5: Generational Plan ──────────────────────────────────────────────
    with tab_gp:
        st.subheader("Generational Wealth Plan — 30-Year Horizon")
        c1,c2 = st.columns(2)
        target = c1.number_input("Target monthly income (USD)", 100, 10000, 1000, 100)
        mo_dca = c2.number_input("Monthly DCA contribution (USD)", 0, 5000, 200, 50)
        if st.button("▶ Run 30-Year Plan"):
            try:
                from engine.generational_planner import run_generational_plan, GenPlanConfig
                import riskfolio as rp
                returns = _get_returns(tickers)
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="ledoit")
                # Fix 3: guard solver failure with EW fallback
                try:
                    w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe",
                                               rf=rf_ann/12, l=2, hist=True)
                    if w_opt is None or w_opt.isnull().values.any():
                        raise ValueError("Solver returned null weights")
                    w = w_opt["weights"].values
                except Exception as _opt_exc:
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "Riskfolio solver failed (%s) — falling back to Equal Weight", _opt_exc)
                    w = np.ones(len(returns.columns)) / len(returns.columns)
                snap = _latest_snapshot(cfg)
                pv = float(snap.get("market_value_thb", 6338*fx_r)) / fx_r
                plan = GenPlanConfig(n_paths=5000, horizon_years=30,
                                      monthly_add_usd=float(mo_dca),
                                      target_income_m=float(target), initial_value=pv)
                with st.spinner("Running 5,000 paths × 360 months (Julia-accelerated)..."):
                    try:
                        from engine.julia_bridge import generational_plan as _jl_gp
                        _pr = returns.values @ w
                        _gp = _jl_gp(float(_pr.mean()), float(_pr.std()),
                                      float(pv), float(mo_dca), 5000, 0.0707/12, float(target))
                        # Normalise Julia output to match Python planner format
                        _ms_raw = _gp.get("milestones", {})
                        _ms = {}
                        for k, v in _ms_raw.items():
                            yr = int(k) if str(k).isdigit() else int(str(k).replace("Year ","").strip())
                            _ms[f"Year {yr}"] = {
                                "p10_value":       float(v.get("p10_value",0)),
                                "p50_value":       float(v.get("p50_value",0)),
                                "p90_value":       float(v.get("p90_value",0)),
                                "p50_income_m":    float(v.get("p50_income_m",0)),
                                "prob_above_target":float(v.get("prob_above_target",0)),
                            }
                        result = {
                            "milestones": _ms,
                            "months_to_target": {
                                "years":       int(_gp.get("years_to_target",0)),
                                "extra_months":int(_gp.get("extra_months",0)),
                            },
                            "chart_bytes": None,
                        }
                    except Exception:
                        result = run_generational_plan(returns, w, cfg, plan)
                ms_rows = [{"Year": yr,
                             "p10 Value": f"${ms['p10_value']:,.0f}",
                             "p50 Value": f"${ms['p50_value']:,.0f}",
                             "p90 Value": f"${ms['p90_value']:,.0f}",
                             "p50 Income/mo": f"${ms['p50_income_m']:,.2f}",
                             f"P(>${target:,.0f}/mo)": f"{ms['prob_above_target']:.1f}%"}
                            for yr, ms in result["milestones"].items()]
                st.dataframe(pd.DataFrame(ms_rows), width="stretch", hide_index=True)
                mtt = result.get("months_to_target", {})
                if mtt.get("years"):
                    st.success(f"Median time to ${target:,}/mo: {mtt['years']} years {mtt['extra_months']} months")
                if result.get("chart_bytes"):
                    st.image(result["chart_bytes"], use_container_width=True)
                try:
                    from utils.llm_summarizer import summarise_generational, render_summary
                    _gen_text = summarise_generational(
                        result.get("milestones",{}),
                        result.get("months_to_target",{}),
                        float(target))
                    render_summary(_gen_text, "gen_plan")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Generational plan error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── Tab 6: Advanced Interactive Charts ─────────────────────────────────────
    with tab_ch:
        st.subheader(t("advanced_charts"))        # Protected usage
        
        try:
            from engine.charts import build_chart, TIMEFRAMES, INDICATOR_DEFAULTS
            
            col1, col2 = st.columns([3, 1])
            
            all_tickers = list(holdings.keys()) + ["SPY", "QQQ", "AGG"]
            ch_ticker = col1.selectbox(
                t("ticker"), all_tickers, index=0, key="ch_ticker"
            )
            
            ch_tf = col2.selectbox(
                "Timeframe", 
                list(TIMEFRAMES.keys()),
                index=list(TIMEFRAMES.keys()).index("6M"), 
                key="ch_tf"
            )
    
            with st.expander("⚙️ Indicators & Comparison", expanded=False):
                i_cols = st.columns(4)
                indicators = {}
                for i, (ind_name, default) in enumerate(INDICATOR_DEFAULTS.items()):
                    indicators[ind_name] = i_cols[i % 4].checkbox(
                        ind_name, value=default, key=f"ind_{ind_name}"
                    )
                
                cmp_str = st.text_input(
                    "Compare tickers (comma-separated, e.g. ARCC,PDI)", 
                    key="ch_cmp"
                )
                benchmark = st.selectbox(
                    "Benchmark", ["None", "SPY", "QQQ", "AGG", "^TNX"], 
                    key="ch_bm"
                )
                benchmark = None if benchmark == "None" else benchmark
    
            cmp_list = [x.strip().upper() for x in cmp_str.split(",") if x.strip()]
    
            with st.spinner(f"Loading {ch_ticker} ({ch_tf})..."):
                fig = build_chart(
                    ticker=ch_ticker,
                    timeframe=ch_tf,
                    indicators=indicators,
                    benchmark=benchmark,
                    compare_tickers=cmp_list,
                )
            
            st.plotly_chart(fig, use_container_width=True)
    
        except Exception as e:
            st.error(f"Chart engine error: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    # ── Tab 7: Exit Position Simulator ─────────────────────────────────────────
    with tab_ex:
        st.subheader(t("exit_sim_title"))
        st.caption("Sandbox mode — computes full portfolio impact of selling a position. Never modifies YAML.")

        if not holdings:
            st.warning("No holdings found in current account.")
        else:
            _ex_col1, _ex_col2, _ex_col3 = st.columns(3)
            _ex_ticker = _ex_col1.selectbox(t("select_position"), list(holdings.keys()), key="ex_ticker")
            _ex_held   = holdings[_ex_ticker]["shares"]
            _ex_shares = _ex_col2.slider(t("shares_to_sell"), 1, _ex_held, min(_ex_held, 50), key="ex_shares")
            _ex_price  = _ex_col3.number_input(t("exit_price"), min_value=0.01,
                                                 value=float(holdings[_ex_ticker]["avg_cost"]),
                                                 format="%.4f", key="ex_price")

            _ex_pct = _ex_shares / _ex_held
            st.caption(f"Selling **{_ex_shares}/{_ex_held} shares** ({_ex_pct:.0%} of position) "
                       f"at **${_ex_price:.4f}** — gross proceeds ${_ex_shares*_ex_price:,.2f}")

            if st.button(t("compute_exit"), type="primary", key="compute_exit_btn"):
                from engine.exit_simulator import simulate_exit
                _ex_result = simulate_exit(
                    cfg=cfg, ticker=_ex_ticker,
                    shares_to_sell=_ex_shares, exit_price_usd=_ex_price,
                    fx_rate=fx_r, wht_rate=wht,
                )

                if _ex_result.get("error"):
                    st.error(_ex_result["error"])
                else:
                    # ── KPI cards ─────────────────────────────────────────────
                    _xc1,_xc2,_xc3,_xc4 = st.columns(4)
                    _pnl = _ex_result["pnl"]
                    _inc = _ex_result["income"]
                    _xc1.metric(t("capital_gain"),
                                f"${_pnl['capital_gain_usd']:+,.2f}",
                                f"฿{_pnl['capital_gain_thb']:+,.0f}")
                    _xc2.metric(t("lost_income"),
                                f"${_inc['lost_net_mo']:.2f}/mo",
                                f"${_inc['income_drop_pct']:+.1f}% portfolio")
                    _xc3.metric("Proceeds (USD)", f"${_ex_result['trade']['net_proceeds_usd']:,.2f}")
                    _xc4.metric("Proceeds (THB)", f"฿{_ex_result['trade']['net_proceeds_thb']:,.0f}")

                    # ── Details ───────────────────────────────────────────────
                    _xt1, _xt2 = st.tabs(["💰 P&L + Income", "🌐 FX + Generational"])
                    with _xt1:
                        _ex_rows = [
                            {"Metric": "Cost basis", "Value": f"${_pnl['cost_basis_usd']:,.2f}"},
                            {"Metric": "Avg cost/share", "Value": f"${_pnl['avg_cost_usd']:.4f}"},
                            {"Metric": "Exit price", "Value": f"${_ex_price:.4f}"},
                            {"Metric": "Capital gain/loss", "Value": f"${_pnl['capital_gain_usd']:+,.2f}  ({_pnl['gain_pct']:+.2f}%)"},
                            {"Metric": "Capital gains tax (Thai)", "Value": _ex_result['tax']['capital_gains_tax_usd']},
                            {"Metric": "Lost income/mo (net)", "Value": f"${_inc['lost_net_mo']:.2f}"},
                            {"Metric": "Remaining income/mo", "Value": f"${_inc['remaining_net_mo']:.2f}"},
                            {"Metric": "Portfolio income drop", "Value": f"{_inc['income_drop_pct']:.1f}%"},
                        ]
                        st.dataframe(pd.DataFrame(_ex_rows), width="stretch", hide_index=True)

                        # Concentration after
                        _conc = _ex_result["portfolio"]["concentration_after"]
                        if _conc:
                            st.subheader("Concentration after exit")
                            _conc_df = pd.DataFrame([
                                {"Ticker": k, "Weight": f"{v:.1%}",
                                 "Status": "⚠️ High" if v > 0.5 else "✅ OK"}
                                for k,v in sorted(_conc.items(), key=lambda x: -x[1])
                            ])
                            st.dataframe(_conc_df, width="stretch", hide_index=True)

                    with _xt2:
                        _gen = _ex_result["generational"]
                        st.metric("Future income lost (30yr compounded)",
                                  f"${_gen['future_income_lost_30yr']:,.0f}",
                                  help=_gen["note"])
                        st.metric("Proceeds → reinvested at FX",
                                  f"฿{_ex_result['fx']['proceeds_thb']:,.0f}")
                        st.caption(_ex_result["tax"]["note"])

                    # ── AI recommendation ──────────────────────────────────────
                    st.divider()
                    st.subheader("📋 Analysis")
                    st.info(_ex_result["recommendation"])

    # ── Tab 8: AI Portfolio Research ────────────────────────────────────────────
    with tab_ai:
        st.subheader(t("ai_research_title"))
        st.caption(
            "Ask ANY question about your portfolio, macro, tax, strategy, or exits. "
            "Responds in English or Thai based on your language setting."
        )

        if not _groq_available():
            st.error("Groq API key not configured. Add `[groq] api_key = 'gsk_...'` to .streamlit/secrets.toml.")
        else:
            # Chat history in session state
            if "ai_history" not in st.session_state:
                st.session_state["ai_history"] = []

            # Starter questions
            from utils.research_agent import STARTER_QUESTIONS, ask_stream
            _lang = get_lang()
            _starters = STARTER_QUESTIONS.get(_lang, STARTER_QUESTIONS["en"])

            with st.expander("💡 Starter questions — click to ask", expanded=len(st.session_state["ai_history"]) == 0):
                for _sq in _starters:
                    if st.button(_sq, key=f"sq_{hash(_sq)}", use_container_width=True):
                        st.session_state["ai_history"].append({"role":"user","content":_sq})
                        st.rerun()

            # Chat display
            for _msg in st.session_state["ai_history"]:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

            # Input
            _user_input = st.chat_input(t("ask_anything"))
            if _user_input:
                st.session_state["ai_history"].append({"role":"user","content":_user_input})
                with st.chat_message("user"):
                    st.markdown(_user_input)
                with st.chat_message("assistant"):
                    _response = st.write_stream(ask_stream(
                        question=_user_input,
                        history=st.session_state["ai_history"][:-1],
                        cfg=cfg, lang=_lang, fx=fx_r, wht=wht,
                    ))
                st.session_state["ai_history"].append({"role":"assistant","content":_response})

            if st.session_state["ai_history"]:
                if st.button("🗑 Clear conversation", key="clear_ai"):
                    st.session_state["ai_history"] = []
                    st.rerun()

    # ── Tab 9: Download Full Report ───────────────────────────────────────────
    with tab_rpt:
        st.subheader(t("download_report"))
        st.info("Generates the complete report with all analytics, charts, and projections.")
        if st.button("📊 Generate Report (30–60s)", type="primary"):
            try:
                from engine.report_builder import build_report
                from engine.analytics import compute_risk_table
                from engine.wht_reconciliation import build_reconciliation
                from engine.generational_planner import run_generational_plan, GenPlanConfig
                from engine.backtest import run_walkforward
                from engine.fx_timing import compute_fx_signal
                from engine.dividend_calendar import build_events
                import riskfolio as rp

                with st.spinner("Running full analysis..."):
                    prices  = load_prices(tickers)
                    returns = prices.resample("ME").last().pct_change().dropna().dropna(axis=1,how="all")
                    risk_tbl= compute_risk_table(returns, rf_ann/12)

                    port = rp.Portfolio(returns=returns)
                    port.assets_stats(method_mu="hist", method_cov="ledoit")
                    # Fix 3: guard solver failure with EW fallback
                    try:
                        w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe",
                                                   rf=rf_ann/12, l=2, hist=True)
                        if w_opt is None or w_opt.isnull().values.any():
                            raise ValueError("Solver returned null weights")
                        w_main = w_opt["weights"].values
                        w_dict = {"Max Sharpe": w_opt.to_dict()["weights"]}
                    except Exception as _opt_exc:
                        import logging as _lg
                        _lg.getLogger(__name__).warning(
                            "Riskfolio solver failed (%s) — falling back to Equal Weight", _opt_exc)
                        _n = len(tickers)
                        w_main = np.ones(_n) / _n
                        w_dict = {"Equal Weight": {t: 1/_n for t in tickers}}

                    snap   = _latest_snapshot(cfg)
                    pv     = float(snap.get("market_value_thb", 6338*fx_r)) / fx_r

                    pnl_rows = []
                    for tkr, h in holdings.items():
                        curr = float(prices[tkr].iloc[-1]) if tkr in prices.columns else 0
                        mkt  = h["shares"]*curr; unr = mkt-h["total_cost"]; pct = unr/h["total_cost"] if h["total_cost"] else 0
                        pnl_rows.append({"Ticker":tkr,"Shares":h["shares"],"Avg Cost $":h["avg_cost"],
                                          "Current $":round(curr,2),"Cost Basis $":h["total_cost"],
                                          "Market Value $":round(mkt,2),"Unrealised $":round(unr,2),
                                          "Unrealised THB":round(unr*fx_r,0),"Net P&L $":round(unr,2),"Net P&L %":round(pct,4)})
                    pnl_df = pd.DataFrame(pnl_rows).set_index("Ticker")

                    div_rows = []
                    for d in cfg.get("dividends_received",[]):
                        sh=d.get("shares_eligible",0); pps=d.get("amount_per_share_usd",0.0); gross=sh*pps; net=gross*(1-wht)
                        div_rows.append({"Period":d.get("period",""),"Ticker":d.get("ticker",""),
                                          "Ex-date":d.get("ex_date",""),"Pay-date":d.get("pay_date",""),
                                          "Shares":sh,"$/share":pps,"Gross $":round(gross,2),
                                          "WHT":f"{wht*100:.0f}%","Net $":round(net,2),
                                          "Net THB est":round(net*fx_r,0),
                                          # Bug 1 fix: np.nan not "n/a" for numeric column
                                          "KS App THB": float(d["thb_ks_app"]) if d.get("thb_ks_app") is not None else np.nan,
                                          "Status":d.get("status","received")})

                    wht_records = build_reconciliation(cfg, fx_r)
                    bt          = run_walkforward(returns, train_months=24)
                    bt_metrics  = bt.get("metrics",{}) if bt else {}
                    fx_hist     = load_fx_series(); fx_signal = compute_fx_signal(fx_hist)
                    cal_events  = build_events(cfg, holdings)

                    plan = GenPlanConfig(n_paths=2000,horizon_years=30,
                                         monthly_add_usd=200,target_income_m=1000,initial_value=pv)
                    gen  = run_generational_plan(returns, w_main, cfg, plan)

                    conf = {"rf":rf_ann,"wht":wht,"fx":fx_r,
                            "cash_usd":cfg.get("cash",{}).get("usd",0),
                            "deposited":cfg.get("cash",{}).get("total_deposited_usd",0)}

                    xlsx_bytes = build_report(
                        pnl_df=pnl_df, risk_tbl=risk_tbl, w_dict=w_dict,
                        div_rows=div_rows, wht_records=wht_records,
                        bt_metrics=bt_metrics, milestones=gen.get("milestones",{}),
                        fx_signal=fx_signal, cal_events=cal_events,
                        cfg=cfg, conf=conf, plots={}, bl_applied=False,
                    )

                fname = f"portfolio_report_{datetime.today().strftime('%Y%m%d')}.xlsx"
                st.download_button(
                    "⬇️ Download Excel Report",
                    xlsx_bytes, fname,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.success("Report ready!")
            except Exception as e:
                st.error(f"Report error: {e}")
                import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# KAsset account: special Overview for Thai mutual fund accounts
# ══════════════════════════════════════════════════════════════════════════════
# Injected into Dashboard > Overview tab when account_type == "Thai mutual fund"
# (Handled inline above via _active_acct check — see Dashboard tab_ov)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: FAMILY OVERVIEW  (multi-account consolidated view)
# ══════════════════════════════════════════════════════════════════════════════
elif page == t("family_overview"):

    st.subheader("\U0001f468\u200d\U0001f469\u200d\U0001f467 Family Portfolio Overview")
    st.caption("Consolidated view across all accounts. Add accounts in config/accounts.yaml.")

    _family_rows = []
    _total_mkt_usd = 0.0; _total_income_mo = 0.0
    _YIELDS = {"BKLN":0.0707,"ARCC":0.1066,"PDI":0.152,"MAIN":0.076,"HTGC":0.12}

    for _acct in _all_accounts:
        try:
            _acfg  = load_cfg(_acct["id"])
            _ahold = derive_holdings(_acfg)
            _afx   = _acfg.get("meta",{}).get("fx_usd_thb", 32.68)
            _awht  = _acfg.get("settings",{}).get("wht_active", 0.30)
            if not _ahold:
                continue
            _atickers = tuple(sorted(_ahold.keys()))
            try:
                _aprices = load_prices(_atickers)
                _amkt = sum(_ahold[t]["shares"] * float(_aprices[t].iloc[-1])
                            for t in _atickers if t in _aprices.columns)
            except Exception:
                _amkt = sum(_ahold[t]["shares"] * _ahold[t]["avg_cost"] for t in _atickers)
            _aincome_net = sum(
                _ahold[t]["shares"] * _ahold[t]["avg_cost"] * _YIELDS.get(t,0.08) / 12
                for t in _atickers) * (1 - _awht)
            _snap = _latest_snapshot(_acfg)
            _total_mkt_usd += _amkt; _total_income_mo += _aincome_net
            _family_rows.append({
                "Account":       _acct["id"],
                "Holder":        _acct.get("display_name",""),
                "Holdings":      ", ".join(_atickers),
                "Mkt Value ($)": f"${_amkt:,.0f}",
                "Mkt Value (฿)": f"\u0e3f{_amkt*_afx:,.0f}",
                "Income/mo (net)":f"${_aincome_net:.2f}",
                "Dividends (฿)": f"\u0e3f{_snap.get('total_dividends_thb',0):,.2f}",
            })
        except Exception as _e:
            _family_rows.append({"Account":_acct["id"],"Holder":"—","Holdings":f"Error: {_e}",
                                  "Mkt Value ($)":"—","Mkt Value (฿)":"—",
                                  "Income/mo (net)":"—","Dividends (฿)":"—"})

    if _family_rows:
        _c1,_c2,_c3,_c4 = st.columns(4)
        _c1.metric("Total Family NAV", f"${_total_mkt_usd:,.0f}", f"\u0e3f{_total_mkt_usd*fx_r:,.0f}")
        _c2.metric("Combined Income/mo", f"${_total_income_mo:.2f}", "net after WHT")
        _c3.metric("Accounts", str(len(_all_accounts)))
        _c4.metric("Annual income est.", f"${_total_income_mo*12:,.0f}")
        st.divider()
        st.dataframe(pd.DataFrame(_family_rows), width="stretch", hide_index=True)
        if len(_family_rows) > 1:
            _vals = [float(r["Income/mo (net)"].replace("$","")) for r in _family_rows]
            _fig = go.Figure(go.Bar(
                x=[r["Account"] for r in _family_rows], y=_vals,
                text=[f"${v:.2f}/mo" for v in _vals], textposition="auto",
                marker_color=[a.get("accent","#2E5BA8") for a in _all_accounts[:len(_family_rows)]]))
            _fig.update_layout(title="Monthly income by account", height=260,
                               margin=dict(l=0,r=0,t=30,b=0), yaxis_title="USD/month")
            st.plotly_chart(_fig, width="stretch")
    else:
        st.info("No accounts with holdings found. Check config/accounts.yaml.")
