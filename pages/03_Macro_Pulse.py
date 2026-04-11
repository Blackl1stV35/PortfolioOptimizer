"""
pages/03_Macro_Pulse.py  --  Macro Pulse & Risk Dashboard (standalone page)
Accessible via Streamlit multi-page routing AND embedded inside Intelligence Hub tab.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings("ignore")
import yaml
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.macro_monitor import (
    get_macro_data, get_macro_regime, get_risk_gauges
)

st.set_page_config(page_title="Macro Pulse", page_icon="📡", layout="wide")

CONFIG_FILE = Path(__file__).parent.parent / "config" / "portfolio.yaml"


@st.cache_data(ttl=60)
def _cfg() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_data(ttl=300, show_spinner="Fetching macro data ...")
def _macro(_sig: str) -> dict:
    return get_macro_data(_cfg())


cfg     = _cfg()
_sig    = str(cfg.get("meta", {}).get("data_as_of", ""))
macro   = _macro(_sig)
regime  = get_macro_regime(macro)
gauges  = get_risk_gauges(macro)
rates   = macro["rates"]
vix     = macro["vix"]
oil     = macro["oil"]
yc      = macro["yield_curve"]
rec     = macro["recession"]
fx      = macro["fx"]

_SIG = {"green": "🟢", "yellow": "🟡", "red": "🔴", "unknown": "⚪"}


def _icon(s): return _SIG.get(s, "⚪")


def _spark(series: pd.Series, color: str = "#1D9E75") -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines",
                               line=dict(color=color, width=1.5)))
    fig.update_layout(height=55, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis_visible=False, yaxis_visible=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _gauge(label: str, value: int, signal: str) -> go.Figure:
    color = {"green": "#1D9E75", "yellow": "#BA7517", "red": "#E24B4A"}.get(signal, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": label, "font": {"size": 12}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 35], "color": "#E1F5EE"},
                          {"range": [35, 65], "color": "#FAEEDA"},
                          {"range": [65, 100], "color": "#FCEBEB"}]},
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig


st.title("📡 Macro Pulse & Risk Dashboard")
st.caption("Key indicators affecting BKLN · ARCC · PDI · Cash decisions")

col_r, col_t = st.columns([1, 6])
if col_r.button("Refresh data"):
    st.cache_data.clear(); st.rerun()

col_t.caption(f"Last fetched: {macro['fetched_at']}  |  "
              f"Thai rate as of: {rates['thai_rate_date']}  |  "
              f"Fed rate as of: {rates['us_fed_date']}")

# Regime banner
regime_colors = {"Defensive": "error", "Neutral": "warning", "Aggressive": "success"}
getattr(st, regime_colors.get(regime["regime"], "info"))(
    f"**{_icon(regime['color'])} Macro Regime: {regime['regime']}**  |  "
    f"Score {regime['score']}/10  |  "
    f"Suggested cash: **{regime['cash_pct']}** ({regime['cash_thb_est']})  |  "
    f"{regime['action']}"
)
st.divider()

# 6 Metric cards
r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

with r1c1:
    st.metric(f"{_icon(rates['thai_signal'])} Thai Policy Rate (BOT)",
              f"{rates['thai_rate']:.2f}%")
    st.caption(f"As of {rates['thai_rate_date']} — update in portfolio.yaml → macro")

with r1c2:
    st.metric(f"{_icon(rates['us_signal'])} US Fed Funds Rate",
              f"{rates['us_fed_rate']:.2f}%")
    st.caption(f"As of {rates['us_fed_date']} — BKLN income ↑ when rates stay elevated")

with r1c3:
    cur_vix = vix.get("current"); d_vix = vix.get("change_30d")
    st.metric(f"{_icon(vix['signal'])} VIX (Uncertainty)",
              f"{cur_vix:.1f}" if cur_vix else "n/a",
              delta=f"{d_vix:+.1f} (30d)" if d_vix else None)
    if not vix["series"].empty:
        st.plotly_chart(_spark(vix["series"], "#E24B4A" if vix["signal"] == "red" else "#BA7517"),
                        use_container_width=True, config={"displayModeBar": False})

with r2c1:
    cur_oil = oil.get("current"); d_oil = oil.get("change_30d")
    st.metric(f"{_icon(oil['signal'])} WTI Oil ($/bbl)",
              f"${cur_oil:.2f}" if cur_oil else "n/a",
              delta=f"{d_oil:+.2f} (30d)" if d_oil else None)
    if oil.get("geopolitical_note"):
        st.caption(f"⚠ {oil['geopolitical_note']}")
    if not oil["series"].empty:
        st.plotly_chart(_spark(oil["series"], "#BA7517"),
                        use_container_width=True, config={"displayModeBar": False})

with r2c2:
    z = fx.get("zscore"); fx_sig_str = fx.get("signal", "unknown")
    fx_icon = "🟢" if fx_sig_str in ("buy", "strong_buy") else ("🔴" if fx_sig_str == "avoid" else "🟡")
    st.metric(f"{fx_icon} USD/THB Signal",
              f"{fx.get('current', 0):.4f}",
              delta=f"z={z:+.2f}" if z else None)
    st.caption(fx.get("advice", ""))

with r2c3:
    prob = rec.get("probability", 0); sig = rec.get("signal", "yellow")
    st.metric(f"{_icon(sig)} US Recession Probability", f"~{prob}%")
    st.caption(rec.get("note", ""))

st.divider()

# 4 Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Policy Rates", "⚡ Risk Gauges", "🌐 Economic Snapshot", "💼 Portfolio Implications"
])

with tab1:
    st.subheader("Central bank policy rates")
    irx = macro["liquidity"].get("series", pd.Series(dtype=float))
    if not irx.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=irx.index, y=irx.values,
                                  name="US 13W T-bill (Fed proxy)", line=dict(color="#3B8BD4", width=2)))
        fig.add_hline(y=rates["us_fed_rate"], line_dash="dot", line_color="#888",
                      annotation_text=f"Fed {rates['us_fed_rate']:.2f}%")
        fig.update_layout(title="US short-term rate", height=260, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Thai BOT rate: **{rates['thai_rate']:.2f}%** as of {rates['thai_rate_date']} — update manually after each MPC meeting")

    sp = yc.get("series_spread", pd.Series(dtype=float))
    if not sp.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sp.index, y=sp.values, mode="lines",
                                   name="2s10s spread",
                                   line=dict(color="#1D9E75" if not yc["inverted"] else "#E24B4A", width=2),
                                   fill="tozeroy",
                                   fillcolor="rgba(29,158,117,0.1)" if not yc["inverted"] else "rgba(226,75,74,0.1)"))
        fig2.add_hline(y=0, line_color="#888", line_dash="dash")
        fig2.update_layout(title="2s10s yield curve spread", height=200,
                           margin=dict(l=0, r=0, t=30, b=0), yaxis_title="% spread")
        st.plotly_chart(fig2, use_container_width=True)
        inv = "⚠ Yield curve inverted — historically precedes recession 12–18 months out" if yc["inverted"] else "Yield curve not inverted — neutral signal"
        st.caption(inv)

with tab2:
    st.subheader("Risk gauges (0 = low stress · 100 = high stress)")
    g1, g2, g3, g4 = st.columns(4)
    labels = {"default_risk": "Default / Credit", "liquidity_risk": "Liquidity",
              "maturity_risk": "Maturity / Curve", "uncertainty": "Uncertainty (VIX)"}
    sigs   = {"default_risk": macro["credit"].get("signal", "yellow"),
              "liquidity_risk": macro["liquidity"].get("signal", "yellow"),
              "maturity_risk": yc.get("signal", "yellow"),
              "uncertainty": vix.get("signal", "yellow")}
    for col, key in zip([g1, g2, g3, g4], gauges):
        with col:
            st.plotly_chart(_gauge(labels[key], gauges[key], sigs[key]),
                            use_container_width=True, config={"displayModeBar": False})
            st.caption(f"{gauges[key]}/100")
    composite = round(sum(gauges.values()) / 4)
    st.subheader(f"Composite stress: {composite}/100")
    st.progress(composite / 100)

with tab3:
    c1, c2 = st.columns(2)
    t10 = yc.get("series_10y", pd.Series(dtype=float))
    if not t10.empty:
        fig = go.Figure(go.Scatter(x=t10.index, y=t10.values, mode="lines",
                                    line=dict(color="#3B8BD4", width=2), fill="tozeroy",
                                    fillcolor="rgba(59,139,212,0.1)"))
        fig.update_layout(title="US 10Y Treasury yield (%)", height=220, margin=dict(l=0,r=0,t=30,b=0))
        c1.plotly_chart(fig, use_container_width=True)
        c1.caption("Rising 10Y → higher BKLN floating income; pressure on long-duration assets")
    oil_s = oil.get("series", pd.Series(dtype=float))
    if not oil_s.empty:
        cur_o = float(oil_s.iloc[-1])
        color_o = "#E24B4A" if cur_o > 95 else ("#BA7517" if cur_o > 80 else "#1D9E75")
        fig2 = go.Figure(go.Scatter(x=oil_s.index, y=oil_s.values, mode="lines",
                                     line=dict(color=color_o, width=2), fill="tozeroy",
                                     fillcolor="rgba(186,117,23,0.1)"))
        fig2.add_hline(y=95, line_dash="dot", line_color="#E24B4A", annotation_text="Defensive threshold $95")
        fig2.update_layout(title="WTI Oil ($/bbl)", height=220, margin=dict(l=0,r=0,t=30,b=0))
        c2.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("Cash deployment recommendation")
    reg_map = {"Defensive": "error", "Neutral": "warning", "Aggressive": "success"}
    getattr(st, reg_map.get(regime["regime"], "info"))(
        f"**Regime: {regime['regime']}** (score {regime['score']}/10)\n\n"
        f"Suggested cash: **{regime['cash_pct']}** — {regime['cash_thb_est']}\n\n"
        f"**Action:** {regime['action']}"
    )
    st.subheader("PDI deployment")
    cash_thb = cfg.get("analysis", {}).get("cash_reserve_thb", 92000)
    fx_cur   = fx.get("current", 32.68)
    cash_usd = cash_thb / fx_cur
    c1, c2   = st.columns(2)
    c1.metric("Available cash (THB)", f"฿{cash_thb:,.0f}")
    c1.metric("Available cash (USD)", f"${cash_usd:,.2f}")
    pct_map  = {"Aggressive": 0.80, "Neutral": 0.50, "Defensive": 0.20}
    deploy   = pct_map.get(regime["regime"], 0.50)
    c2.metric("Suggested deployment", f"${cash_usd * deploy:,.2f}",
              delta=f"{deploy*100:.0f}% of available")
    c2.caption({"Aggressive": "Deploy 80% → PDI + BKLN DCA",
                "Neutral":    "Deploy 50% cautiously → split PDI / BKLN",
                "Defensive":  "Deploy 20% max — preserve cash"}.get(regime["regime"], ""))
