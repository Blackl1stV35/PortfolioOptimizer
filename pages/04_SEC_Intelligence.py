"""
pages/04_SEC_Intelligence.py  --  SEC EDGAR Intelligence Dashboard
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings("ignore")
import yaml
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.edgar_monitor import get_edgar_intelligence, screen_bdc_candidate

st.set_page_config(page_title="SEC Intelligence", page_icon="📋", layout="wide")

CONFIG_FILE = Path(__file__).parent.parent / "config" / "portfolio.yaml"

@st.cache_data(ttl=60)
def _cfg():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@st.cache_data(ttl=1800, show_spinner="Fetching SEC filings...")
def _edgar(cfg_sig: str):
    return get_edgar_intelligence(_cfg())

cfg     = _cfg()
cfg_sig = cfg.get("meta", {}).get("data_as_of", "")
identity = cfg.get("edgar", {}).get("identity", "")

st.title("SEC Intelligence — EDGAR Filings Monitor")
st.caption("Authoritative data direct from SEC EDGAR. No API key. No cost.")

if not identity:
    st.error("Add `edgar: identity: your.name@email.com` to `config/portfolio.yaml` to enable this panel.")
    st.code("""
# Add to config/portfolio.yaml:
edgar:
  identity: "your.name@email.com"    # SEC courtesy header (not authentication)
""")
    st.stop()

col_r, col_t = st.columns([1, 6])
if col_r.button("Refresh (30 min cache)"):
    st.cache_data.clear(); st.rerun()

intel = _edgar(cfg_sig)

if not intel.get("available"):
    st.warning(intel.get("reason", "edgartools unavailable"))
    st.code("pip install edgartools")
    st.stop()

col_t.caption(f"Last fetched: {intel.get('fetched_at','')}")

# ── ARCC tabs ─────────────────────────────────────────────────────────────────
st.subheader("ARCC — Ares Capital Corporation")
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Dividend Declarations",
    "📊 NAV & NII Fundamentals",
    "👔 Insider Trades",
    "🔍 BDC Candidate Screener",
])

# ── Tab 1: Dividend Declarations ──────────────────────────────────────────────
with tab1:
    st.caption("8-K filings — authoritative dividend data, typically 1-3 days before aggregators.")
    divs = intel.get("arcc_dividends", [])
    if divs:
        df = pd.DataFrame(divs)[["declared","period","amount","ex_date","record_date","pay_date","source"]]
        df.columns = ["Filed","Period","Amount","Ex-date","Record","Pay-date","Source"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Compare with YAML
        st.divider()
        st.subheader("Cross-check vs portfolio.yaml")
        yaml_upcoming = []
        for tkr, inst in cfg.get("instruments",{}).items():
            if tkr != "ARCC": continue
            for u in inst.get("dividend_policy",{}).get("estimated_upcoming",[]):
                yaml_upcoming.append({
                    "Source":   "portfolio.yaml (estimated)",
                    "Period":   u.get("period",""),
                    "Amount":   u.get("amount", u.get("amount_per_share_usd","")),
                    "Ex-date":  u.get("ex",""),
                    "Pay-date": u.get("pay",""),
                })
        if yaml_upcoming:
            st.dataframe(pd.DataFrame(yaml_upcoming), use_container_width=True, hide_index=True)
            st.caption("If SEC 8-K data differs from YAML estimates, update portfolio.yaml with the confirmed dates.")
        else:
            st.info("No upcoming dividends in portfolio.yaml to compare.")
    else:
        st.info("No recent ARCC dividend declarations found in the last 90 days.")
        st.caption("ARCC typically files dividend 8-K ~6 weeks before ex-date. "
                   "Q2 2026 expected ~late April/early May 2026.")


# ── Tab 2: NAV & NII Fundamentals ────────────────────────────────────────────
with tab2:
    fund = intel.get("arcc_fundamentals", {})
    if not fund:
        st.warning("Fundamentals unavailable — check edgartools installation.")
    else:
        # Metric cards
        c1,c2,c3,c4 = st.columns(4)
        nav = fund.get("nav_latest")
        nii = fund.get("nii_latest")
        cov = fund.get("coverage_ratio")
        dep = fund.get("depr_latest_m")

        cov_colors = {"green":"normal","yellow":"off","red":"inverse"}
        c1.metric("NAV per share",
                  f"${nav:.2f}" if nav else "n/a",
                  delta=f"{fund.get('nav_trend_4q',0):+.3f} (4Q)" if fund.get("nav_trend_4q") else None)
        c2.metric("NII per share (quarterly)",
                  f"${nii:.3f}" if nii else "n/a")
        c3.metric("Dividend coverage",
                  f"{cov:.2f}x" if cov else "n/a",
                  delta="covered" if cov and cov >= 1.0 else ("at risk" if cov else None),
                  delta_color=cov_colors.get(fund.get("coverage_signal","yellow"),"off"))
        c4.metric("Unrealised depreciation",
                  f"${dep:.0f}M" if dep else "n/a",
                  help="Rising depreciation is the key early-warning signal for BDC stress. See FSK/TCPC case study.")

        # Coverage context
        if cov:
            if cov >= 1.15:
                st.success(f"Coverage {cov:.2f}x — dividend well-covered. NII comfortably exceeds $0.48/quarter.")
            elif cov >= 1.0:
                st.warning(f"Coverage {cov:.2f}x — dividend covered but thin margin. Monitor closely.")
            else:
                st.error(f"Coverage {cov:.2f}x — NII below dividend. Dividend sustainability at risk. "
                          "Review ARCC position sizing.")

        st.divider()

        # NAV chart
        nav_s = fund.get("nav_series")
        nii_s = fund.get("nii_series")
        col_l, col_r = st.columns(2)

        with col_l:
            if nav_s is not None and len(nav_s) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=nav_s.index, y=nav_s.values,
                                          mode="lines+markers", name="NAV/share",
                                          line=dict(color="#1D9E75", width=2)))
                if nav:
                    fig.add_hline(y=18.00, line_dash="dot", line_color="#BA7517",
                                  annotation_text="Your avg cost $18.00")
                fig.update_layout(title="ARCC NAV per share (quarterly)",
                                  height=280, margin=dict(l=0,r=0,t=30,b=0),
                                  yaxis_title="USD")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("NAV series unavailable")

        with col_r:
            if nii_s is not None and len(nii_s) > 0:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=nii_s.index, y=nii_s.values,
                                       name="NII/share", marker_color="#3B8BD4"))
                fig2.add_hline(y=0.48, line_dash="dot", line_color="#E24B4A",
                               annotation_text="Dividend $0.48")
                fig2.update_layout(title="ARCC NII per share vs dividend",
                                   height=280, margin=dict(l=0,r=0,t=30,b=0),
                                   yaxis_title="USD per share")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("NII series unavailable")

        # Depreciation chart
        depr_s = fund.get("depr_series")
        if depr_s is not None and len(depr_s) > 0:
            depr_m = depr_s / 1e6
            st.subheader("Unrealised portfolio depreciation — BDC early warning signal")
            fig3 = go.Figure()
            fig3.add_trace(go.Area(x=depr_m.index, y=depr_m.values,
                                    name="Gross unrealised depreciation ($M)",
                                    line=dict(color="#E24B4A", width=1.5),
                                    fillcolor="rgba(226,75,74,0.1)"))
            fig3.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                               yaxis_title="$M")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Rising depreciation precedes NAV declines and dividend cuts. "
                       "This metric caught the FSK and TCPC failures years before the market did.")


# ── Tab 3: Insider Trades ─────────────────────────────────────────────────────
with tab3:
    st.caption("Form 4 filings — executive open-market purchases and sales in the last 90 days.")
    trades = intel.get("arcc_insider", [])
    if trades:
        df = pd.DataFrame(trades)[["date","insider","type","shares","price","value"]]
        df["value"] = df["value"].apply(lambda x: f"${x:,.0f}")
        df["price"] = df["price"].apply(lambda x: f"${x:.2f}")
        df["shares"] = df["shares"].apply(lambda x: f"{x:,}")
        df.columns = ["Date","Insider","Type","Shares","Price","Value"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        buys  = [t for t in trades if t["type"] == "BUY"]
        sells = [t for t in trades if t["type"] == "SELL"]
        c1,c2 = st.columns(2)
        c1.metric("Open-market buys", len(buys),
                  help="Executives buying with their own money = strong conviction signal")
        c2.metric("Open-market sells", len(sells))
        if buys:
            st.success(f"{len(buys)} insider purchase(s) detected — management has skin in the game.")
        if sells and not buys:
            st.warning("Only insider selling detected — worth monitoring, though planned sales are common.")
    else:
        st.info("No Form 4 open-market trades in the last 90 days.")


# ── Tab 4: BDC Candidate Screener ────────────────────────────────────────────
with tab4:
    st.caption("Screen a BDC before adding it to your portfolio. Checks NAV trend, NII coverage, and unrealised depreciation.")

    candidates = ["PFLT","MAIN","HTGC","GBDC","ARCC","OBDC","SLRC"]
    selected   = st.multiselect("Select tickers to screen", candidates,
                                 default=["PFLT","MAIN"])
    if st.button("Run screen"):
        for tkr in selected:
            with st.spinner(f"Screening {tkr}..."):
                res = screen_bdc_candidate(tkr, identity)
            c1,c2,c3 = st.columns([1,3,3])
            signal_icon = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(res.get("signal",""), "⚪")
            c1.markdown(f"### {signal_icon} {tkr}")
            with c2:
                if res.get("nav_latest"):
                    st.metric("NAV/share",  f"${res['nav_latest']:.2f}",
                              delta=f"{res['nav_1yr_chg']*100:+.1f}% (1yr)" if res.get("nav_1yr_chg") else None)
                if res.get("nii_latest"):
                    st.metric("NII/share (latest qtr)", f"${res['nii_latest']:.3f}")
            with c3:
                warnings = res.get("warnings", [])
                if warnings:
                    for w in warnings:
                        st.warning(w)
                elif res.get("error"):
                    st.error(res["error"])
                else:
                    st.success("No red flags detected in XBRL data")
            st.divider()