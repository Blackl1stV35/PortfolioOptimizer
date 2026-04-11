"""
pages/04_SEC_Intelligence.py  --  SEC EDGAR Intelligence (standalone page)
Priority 1 fix is inside engine/edgar_monitor.py via _scalar() and _scalar_series().
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings("ignore")
import yaml
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.edgar_monitor import (
    get_edgar_intelligence, screen_bdc_candidate,
    get_arcc_fundamentals, get_arcc_insider_trades,
    get_arcc_dividend_declarations,
)

st.set_page_config(page_title="SEC Intelligence", page_icon="📋", layout="wide")

CONFIG_FILE = Path(__file__).parent.parent / "config" / "portfolio.yaml"


@st.cache_data(ttl=60)
def _cfg() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_data(ttl=1800, show_spinner="Fetching SEC filings ...")
def _intel(_sig: str) -> dict:
    return get_edgar_intelligence(_cfg())


cfg      = _cfg()
_sig     = str(cfg.get("meta", {}).get("data_as_of", ""))
identity = cfg.get("edgar", {}).get("identity", "")

st.title("📋 SEC Intelligence — EDGAR Filings Monitor")
st.caption("Direct from SEC EDGAR  ·  No API key  ·  No cost  ·  No rate limit")

if not identity:
    st.error("Add `edgar: identity: your@email.com` to `config/portfolio.yaml` to enable this panel.")
    st.code("""
# config/portfolio.yaml
edgar:
  identity: "your.name@email.com"   # SEC courtesy user-agent header (not auth)
""")
    st.stop()

col_r, col_t = st.columns([1, 6])
if col_r.button("🔄 Refresh (30 min cache)"):
    st.cache_data.clear(); st.rerun()

intel = _intel(_sig)

if not intel.get("available"):
    st.warning(intel.get("reason", "edgartools unavailable"))
    st.code("pip install edgartools")
    st.stop()

col_t.caption(f"Last fetched: {intel.get('fetched_at', '')}")

st.subheader("ARCC — Ares Capital Corporation")

tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Dividend Declarations",
    "📊 NAV & NII Fundamentals",
    "👔 Insider Trades",
    "🔍 BDC Candidate Screener",
])

# ── Tab 1: Dividend Declarations ──────────────────────────────────────────────
with tab1:
    st.caption("8-K filings — authoritative data, typically 1–3 days before aggregators.")
    divs = intel.get("arcc_dividends", [])
    if divs:
        df = pd.DataFrame(divs)
        cols = [c for c in ["declared","period","amount","ex_date","record_date","pay_date","source"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("Cross-check vs portfolio.yaml")
        yaml_rows = []
        for tkr, inst in cfg.get("instruments", {}).items():
            if tkr != "ARCC": continue
            for u in inst.get("dividend_policy", {}).get("estimated_upcoming", []):
                yaml_rows.append({
                    "Source":   "portfolio.yaml (estimate)",
                    "Period":   u.get("period", ""),
                    "Amount":   u.get("amount", u.get("amount_per_share_usd", "")),
                    "Ex-date":  u.get("ex", ""),
                    "Pay-date": u.get("pay", ""),
                })
        if yaml_rows:
            st.dataframe(pd.DataFrame(yaml_rows), use_container_width=True, hide_index=True)
            st.caption("If SEC data differs from YAML estimate, update portfolio.yaml with confirmed dates.")
        else:
            st.info("No upcoming estimates in portfolio.yaml to compare.")
    else:
        st.info("No ARCC dividend declarations in last 90 days.")
        st.caption("ARCC typically files ~6 weeks before ex-date. Q2 2026 expected ~late April/May 2026.")


# ── Tab 2: NAV & NII ──────────────────────────────────────────────────────────
with tab2:
    fund = intel.get("arcc_fundamentals", {})
    if not fund:
        st.warning("Fundamentals unavailable — check edgar.identity config or edgartools installation.")
    else:
        nav  = fund.get("nav_latest");  nii = fund.get("nii_latest")
        cov  = fund.get("coverage_ratio"); dep = fund.get("depr_latest_m")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NAV per share", f"${nav:.2f}" if nav else "n/a",
                   delta=f"{fund.get('nav_trend_4q',0):+.3f} (4Q)" if fund.get("nav_trend_4q") else None)
        c2.metric("NII per share (qtr)", f"${nii:.3f}" if nii else "n/a")
        c3.metric("Dividend coverage", f"{cov:.2f}×" if cov else "n/a",
                   delta="covered" if cov and cov >= 1.0 else "at risk" if cov else None,
                   delta_color="normal" if cov and cov >= 1.0 else "inverse")
        c4.metric("Unrealised depreciation", f"${dep:.0f}M" if dep else "n/a",
                   help="Key BDC early-warning signal. Rising depreciation precedes NAV declines.")

        if cov:
            (st.success if cov >= 1.15 else st.warning if cov >= 1.0 else st.error)(
                f"Coverage {cov:.2f}× — {'dividend well covered' if cov>=1.15 else 'covered but thin margin' if cov>=1.0 else 'NII BELOW DIVIDEND — sustainability at risk'}"
            )

        st.divider()
        c_l, c_r = st.columns(2)
        nav_s = fund.get("nav_series")
        nii_s = fund.get("nii_series")

        if nav_s is not None and len(nav_s) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nav_s.index, y=nav_s.values, mode="lines+markers",
                                      name="NAV/share", line=dict(color="#1D9E75", width=2)))
            fig.add_hline(y=18.00, line_dash="dot", line_color="#BA7517",
                          annotation_text="Your avg cost $18.00")
            fig.update_layout(title="ARCC NAV per share (quarterly)", height=260,
                               margin=dict(l=0, r=0, t=30, b=0), yaxis_title="USD")
            c_l.plotly_chart(fig, use_container_width=True)

        if nii_s is not None and len(nii_s) > 0:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=nii_s.index, y=nii_s.values, name="NII/share",
                                   marker_color="#3B8BD4"))
            fig2.add_hline(y=0.48, line_dash="dot", line_color="#E24B4A",
                           annotation_text="Dividend $0.48")
            fig2.update_layout(title="NII per share vs dividend", height=260,
                                margin=dict(l=0, r=0, t=30, b=0), yaxis_title="USD/share")
            c_r.plotly_chart(fig2, use_container_width=True)

        depr_s = fund.get("depr_series")
        if depr_s is not None and len(depr_s) > 0:
            st.subheader("Unrealised depreciation — BDC early warning")
            depr_m = depr_s / 1e6
            fig3 = go.Figure(go.Scatter(x=depr_m.index, y=depr_m.values, mode="lines",
                                         line=dict(color="#E24B4A", width=1.5),
                                         fill="tozeroy", fillcolor="rgba(226,75,74,0.1)"))
            fig3.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="$M")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Rising depreciation → NAV declines → dividend cuts. This caught FSK/TCPC failure years early.")


# ── Tab 3: Insider Trades ─────────────────────────────────────────────────────
with tab3:
    st.caption("Form 4 open-market purchases/sales in the last 90 days.")
    trades = intel.get("arcc_insider", [])
    if trades:
        df = pd.DataFrame(trades)
        df["value"] = df["value"].apply(lambda x: f"${x:,.0f}")
        df["price"] = df["price"].apply(lambda x: f"${x:.2f}")
        df["shares"] = df["shares"].apply(lambda x: f"{x:,}")
        df.columns = [c.title() for c in df.columns]
        st.dataframe(df, use_container_width=True, hide_index=True)
        buys  = [t for t in trades if t["type"] == "BUY"]
        sells = [t for t in trades if t["type"] == "SELL"]
        c1, c2 = st.columns(2)
        c1.metric("Open-market buys",  len(buys))
        c2.metric("Open-market sells", len(sells))
        if buys:
            st.success(f"{len(buys)} insider purchase(s) — management has skin in the game.")
        if sells and not buys:
            st.warning("Only insider selling detected — monitor, but planned sales are common.")
    else:
        st.info("No Form 4 open-market trades in last 90 days.")


# ── Tab 4: BDC Candidate Screener (Priority 1 FIX) ───────────────────────────
with tab4:
    st.caption(
        "Screen a BDC candidate using XBRL data before adding to portfolio.  "
        "**Fix applied:** all XBRL values now use `_scalar()` — no more "
        "`float() argument must be a string or a real number, not 'Series'` errors."
    )
    CANDIDATES = ["PFLT", "MAIN", "HTGC", "GBDC", "ARCC", "OBDC", "SLRC", "CSWC", "TPVG"]
    selected   = st.multiselect("Select tickers to screen", CANDIDATES, default=["PFLT", "MAIN"])

    if st.button("🔍 Run Screen", key="bdc_screen_page"):
        for tkr in selected:
            with st.spinner(f"Screening {tkr} via XBRL ..."):
                res = screen_bdc_candidate(tkr, identity)

            sig_icon = {"green": "✅", "yellow": "🟡", "red": "🔴", "unknown": "⚪"}.get(
                res.get("signal", ""), "⚪")

            with st.expander(f"{sig_icon} {tkr}", expanded=True):
                if res.get("error"):
                    st.error(res["error"])
                    st.caption("EDGAR may not have XBRL data for this ticker, or it's not a BDC.")
                else:
                    c1, c2, c3 = st.columns([1, 2, 3])
                    if res.get("nav_latest"):
                        nav_delta = f"{res['nav_1yr_chg']*100:+.1f}% (1yr)" if res.get("nav_1yr_chg") else None
                        c1.metric("NAV/share",  f"${res['nav_latest']:.2f}", delta=nav_delta)
                    if res.get("nii_latest"):
                        c2.metric("NII/share (latest qtr)", f"${res['nii_latest']:.3f}")
                    if res.get("depr_m"):
                        c3.metric("Unrealised depr.", f"${res['depr_m']:.0f}M")

                    warnings_list = res.get("warnings", [])
                    if warnings_list:
                        for w in warnings_list:
                            st.warning(w)
                    else:
                        st.success("No red flags detected in XBRL data")
