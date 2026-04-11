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
CONFIG_FILE = ROOT / "config" / "portfolio.yaml"
VIEWS_FILE  = ROOT / "config" / "views.yaml"


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_cfg() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_cfg(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    st.cache_data.clear()


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
# PERSISTENT SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
cfg      = load_cfg()
holdings = derive_holdings(cfg)
fx_r     = cfg.get("meta", {}).get("fx_usd_thb", 32.68)
wht      = cfg.get("settings", {}).get("wht_active", 0.30)
rf_ann   = cfg.get("settings", {}).get("risk_free_rate_annual", 0.045)
tickers  = tuple(sorted(holdings.keys()))

with st.sidebar:
    st.markdown("## 📊 PortfolioOptimizer")
    st.markdown("---")

    # Account badge
    meta = cfg.get("meta", {})
    st.markdown(f"""
<div style='background:#1A2E5C;border-radius:8px;padding:10px 14px;margin-bottom:8px'>
  <div style='color:#aab8d8;font-size:11px;font-weight:600;letter-spacing:.06em'>ACCOUNT</div>
  <div style='color:#fff;font-size:15px;font-weight:700'>{meta.get('account_id','397543-7')}</div>
  <div style='color:#7a9ccf;font-size:11px'>K CYBER TRADE (Kasikorn Securities)</div>
</div>
""", unsafe_allow_html=True)

    st.caption(f"Data as of: {meta.get('data_as_of','—')}")
    st.caption(f"FX: {fx_r:.4f} THB/USD  |  WHT: {wht*100:.0f}%")
    st.markdown("---")

    # 3-page navigation (Priority 2)
    page = st.radio(
        "Navigation",
        options=["📊 Dashboard", "🔍 Intelligence Hub", "🧪 Analytics Engine"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # GitHub PAT status indicator — check secrets.toml, env var, and yaml
    def _has_pat() -> bool:
        try:
            if st.secrets.get("github", {}).get("pat"):
                return True
        except Exception:
            pass
        import os
        if os.environ.get("GITHUB_PAT"):
            return True
        return bool(cfg.get("github", {}).get("pat"))
    has_gh = _has_pat()
    st.caption(f"GitHub sync: {'✅ configured' if has_gh else '⚠️ not set'}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    tab_ov, tab_ph, tab_dc, tab_ta, tab_tx = st.tabs([
        "Overview",
        "Price & History",
        "Dividend Calendar",
        "Tax & Reconciliation",
        "Transactions & Activity",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab_ov:
        st.subheader("Portfolio Overview")
        snap = cfg.get("ks_app_snapshot_20260327", {})

        c1,c2,c3,c4 = st.columns(4)
        mkt_thb = snap.get("market_value_thb", 0)
        unr_thb = snap.get("unrealized_thb", 0)
        div_thb = snap.get("total_dividends_thb", 0)
        cash_u  = cfg.get("cash", {}).get("usd", 0)

        c1.metric("Market Value (THB)",  f"฿{mkt_thb:,.0f}",  f"฿{unr_thb:+,.0f} unrealised")
        c2.metric("Market Value (USD)",  f"${mkt_thb/fx_r:,.0f}")
        c3.metric("Dividends Received",  f"฿{div_thb:,.2f}")
        c4.metric("Cash (USD)",          f"${cash_u:.2f}")

        st.divider()
        if tickers:
            prices = load_prices(tickers)
            rows = []
            for tkr, h in holdings.items():
                curr = float(prices[tkr].iloc[-1]) if tkr in prices.columns else 0
                mkt  = h["shares"] * curr; unr = mkt - h["total_cost"]
                rows.append({
                    "Ticker":   tkr,
                    "Shares":   h["shares"],
                    "Avg Cost": f"${h['avg_cost']:.2f}",
                    "Current":  f"${curr:.2f}",
                    "Mkt Value":f"${mkt:,.2f}",
                    "P&L":      f"${unr:+,.2f}",
                    "P&L THB":  f"฿{unr*fx_r:+,.0f}",
                    "P&L %":    f"{unr/h['total_cost']*100:+.2f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ARCC missed dividend alert
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
            st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig2, use_container_width=True)

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
                use_container_width=True, hide_index=True,
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
                    save_cfg(cfg_new)

                    # GitHub auto-commit (Priority 4)
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
            st.dataframe(hdf, use_container_width=True, hide_index=True)
        else:
            st.info("No dividends recorded yet.")

    # ── Tab 4: Tax & Reconciliation ───────────────────────────────────────────
    with tab_tx:
        st.subheader("Withholding Tax Reconciliation")
        try:
            from engine.wht_reconciliation import build_reconciliation, summarise_wht
            records = build_reconciliation(cfg, fx_r)
            summary = summarise_wht(records)
            verdict_map = {"treaty_15":"success","default_30":"error",
                           "partial":"warning","no_data":"info","overpaid":"error"}
            primary = (max(set([r.verdict for r in records]),
                           key=[r.verdict for r in records].count)
                       if records else "no_data")
            getattr(st, verdict_map.get(primary,"info"))(summary)
            if records:
                rows = [{"Period":r.period,"Ticker":r.ticker,"Gross $":f"${r.gross_usd:.4f}",
                         "Net @30%":f"${r.net_30pct:.4f}","Net @15%":f"${r.net_15pct:.4f}",
                         "KS THB":r.ks_thb or "—",
                         "Implied WHT":f"{r.implied_wht*100:.1f}%" if r.implied_wht else "n/a",
                         "Verdict":r.verdict} for r in records]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"WHT module error: {e}")

    # ── Tab 5: Transactions & Activity ────────────────────────────────────────
    with st.tabs(["Transactions & Activity"])[0] if False else (tab_ta,)[0]:
        pass

    with tab_ta:
        st.subheader("Transaction Ledger")
        txns = cfg.get("transactions", [])
        if txns:
            st.dataframe(pd.DataFrame(txns), use_container_width=True, hide_index=True)
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
            save_cfg(cfg_new)

            gh_result = _github_push(cfg_new, f"new trade {nid} {tx_type} {tx_ticker}")
            gh_msg = "✅ GitHub" if gh_result["success"] else f"⚠️ {gh_result.get('error','')[:60]}"
            st.success(f"✅ Saved {nid}: {tx_type} {tx_shares} {tx_ticker} @ ${tx_price:.2f}  |  {gh_msg}")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: INTELLIGENCE HUB
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Intelligence Hub":

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
                        st.dataframe(pd.DataFrame(divs), use_container_width=True, hide_index=True)
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
                        st.dataframe(df, use_container_width=True, hide_index=True)
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
            st.plotly_chart(fig, use_container_width=True)

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
elif page == "🧪 Analytics Engine":

    tab_ro, tab_bt, tab_wi, tab_mc, tab_gp, tab_rpt = st.tabs([
        "⚡ Risk & Optimisation",
        "⏱ Backtest",
        "🔬 What-If Optimizer",
        "📈 Monte Carlo",
        "🏛 Generational Plan",
        "📥 Download Report",
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
                st.dataframe(disp, use_container_width=True)

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
                    st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)
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
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    if bt.get("equity_chart"):
                        st.image(bt["equity_chart"], use_container_width=True)
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
            st.subheader("Candidate Positions")
            c1,c2,c3 = st.columns(3)
            wi_ticker  = c1.text_input("Ticker", "PDI").upper()
            wi_usd     = c2.number_input("USD to deploy", min_value=0.0, value=2000.0, step=100.0)
            wi_shares  = c3.number_input("OR specific shares (0 = use USD amount)",
                                          min_value=0, value=0, step=1)
            wi_max_wt  = st.slider("Max weight per ticker (0 = no limit)", 0.0, 1.0, 0.0, 0.05)
            wi_submit  = st.form_submit_button("🔬 Run What-If Analysis", type="primary")

        if wi_submit and wi_ticker:
            from engine.scenario_analyzer import run_addition_scenario, apply_scenario_to_config

            candidates = [{"ticker": wi_ticker,
                            "usd_amount": wi_usd,
                            "shares": int(wi_shares)}]

            with st.spinner(f"Analysing adding {wi_ticker}..."):
                result = run_addition_scenario(
                    cfg=cfg,
                    candidates=candidates,
                    max_weight=wi_max_wt if wi_max_wt > 0 else None,
                    mc_paths=3000, mc_months=60,
                )

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
                st.dataframe(wdf, use_container_width=True, hide_index=True)

                # ── Transaction preview ───────────────────────────────────
                st.subheader("Proposed Transactions")
                tx_prev = result["transactions_preview"]
                st.dataframe(pd.DataFrame(tx_prev), use_container_width=True, hide_index=True)

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
                    st.plotly_chart(fig, use_container_width=True)

                # ── Apply button ──────────────────────────────────────────
                st.divider()
                st.subheader("Apply to Portfolio")
                confirm_txt = st.text_input(
                    f'Type "{wi_ticker} CONFIRMED" to apply',
                    placeholder=f"{wi_ticker} CONFIRMED",
                )
                if st.button("✅ Apply What-If to portfolio.yaml", type="primary"):
                    if confirm_txt.strip().upper() == f"{wi_ticker} CONFIRMED":
                        cfg_new = apply_scenario_to_config(cfg, tx_prev)
                        save_cfg(cfg_new)
                        gh_result = _github_push(cfg_new, f"What-If applied: {wi_ticker}")
                        gh_msg = "✅ GitHub sync" if gh_result["success"] else f"⚠️ {gh_result.get('error','')[:60]}"
                        st.success(f"✅ Transaction(s) appended to portfolio.yaml.  {gh_msg}")
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
                w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe",
                                           rf=rf_ann/12, l=2, hist=True)
                w = w_opt["weights"].values if w_opt is not None else np.ones(len(tickers))/len(tickers)
                snap = cfg.get("ks_app_snapshot_20260327", {})
                pv = float(snap.get("market_value_thb", 6338*fx_r)) / fx_r
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
                st.plotly_chart(fig, use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("p50 Final", f"${np.percentile(vp[:,-1],50):,.0f}")
                c2.metric("p10 Final", f"${np.percentile(vp[:,-1],10):,.0f}")
                c3.metric("p90 Final", f"${np.percentile(vp[:,-1],90):,.0f}")
                c4.metric("p50 Income/mo", f"${np.percentile(vp[:,-1],50)*0.0707/12:,.2f}")
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
                w_opt = port.optimization(model="Classic",rm="MV",obj="Sharpe",rf=rf_ann/12,l=2,hist=True)
                w = w_opt["weights"].values if w_opt is not None else np.ones(len(tickers))/len(tickers)
                snap = cfg.get("ks_app_snapshot_20260327", {})
                pv = float(snap.get("market_value_thb", 6338*fx_r)) / fx_r
                plan = GenPlanConfig(n_paths=5000, horizon_years=30,
                                      monthly_add_usd=float(mo_dca),
                                      target_income_m=float(target), initial_value=pv)
                with st.spinner("Running 5,000 paths × 360 months..."):
                    result = run_generational_plan(returns, w, cfg, plan)
                ms_rows = [{"Year": yr,
                             "p10 Value": f"${ms['p10_value']:,.0f}",
                             "p50 Value": f"${ms['p50_value']:,.0f}",
                             "p90 Value": f"${ms['p90_value']:,.0f}",
                             "p50 Income/mo": f"${ms['p50_income_m']:,.2f}",
                             f"P(>${target:,.0f}/mo)": f"{ms['prob_above_target']:.1f}%"}
                            for yr, ms in result["milestones"].items()]
                st.dataframe(pd.DataFrame(ms_rows), use_container_width=True, hide_index=True)
                mtt = result.get("months_to_target", {})
                if mtt.get("years"):
                    st.success(f"Median time to ${target:,}/mo: {mtt['years']} years {mtt['extra_months']} months")
                if result.get("chart_bytes"):
                    st.image(result["chart_bytes"], use_container_width=True)
            except Exception as e:
                st.error(f"Generational plan error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── Tab 6: Download Full Report ───────────────────────────────────────────
    with tab_rpt:
        st.subheader("Download Full Excel Report (12 sheets)")
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
                    w_opt = port.optimization(model="Classic",rm="MV",obj="Sharpe",rf=rf_ann/12,l=2,hist=True)
                    w_main = w_opt["weights"].values if w_opt is not None else np.ones(len(tickers))/len(tickers)
                    w_dict = {"Max Sharpe": w_opt.to_dict()["weights"] if w_opt is not None else {t:1/len(tickers) for t in tickers}}

                    snap   = cfg.get("ks_app_snapshot_20260327", {})
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
                                          "Net THB est":round(net*fx_r,0),"KS App THB":d.get("thb_ks_app","n/a"),
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
