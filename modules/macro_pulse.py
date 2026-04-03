"""
pages/03_Macro_Pulse.py  --  Macro Pulse & Risk Dashboard
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings; warnings.filterwarnings("ignore")
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.macro_monitor import (
    get_macro_data, get_macro_regime, get_risk_gauges
)

st.set_page_config(page_title="Macro Pulse", page_icon="📡", layout="wide")

CONFIG_FILE = Path(__file__).parent.parent / "config" / "portfolio.yaml"

# ── Load cfg ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _cfg() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# ── Fetch macro data (cached 5 min via module TTL) ────────────────────────────
@st.cache_data(ttl=300, show_spinner="Fetching macro data ...")
def _macro(cfg_hash: str) -> dict:
    return get_macro_data(_cfg())

cfg = _cfg()
cfg_hash = str(cfg.get("meta", {}).get("data_as_of", ""))


# ── Colour helpers ────────────────────────────────────────────────────────────
_SIG = {"green": "🟢", "yellow": "🟡", "red": "🔴", "unknown": "⚪"}

def _icon(signal: str) -> str:
    return _SIG.get(signal, "⚪")

def _delta_color(v):
    if v is None: return "off"
    return "normal" if v == 0 else ("normal" if v > 0 else "inverse")


# ── Gauge chart helper ────────────────────────────────────────────────────────
def _gauge(label: str, value: int, signal: str) -> go.Figure:
    color = {"green": "#1D9E75", "yellow": "#BA7517", "red": "#E24B4A"}.get(signal, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"size": 13}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0,  35], "color": "#E1F5EE"},
                {"range": [35, 65], "color": "#FAEEDA"},
                {"range": [65,100], "color": "#FCEBEB"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ── Sparkline helper ──────────────────────────────────────────────────────────
def _spark(series: pd.Series, color: str = "#1D9E75") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
    ))
    fig.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PAGE START
# ═════════════════════════════════════════════════════════════════════════════
st.title("Macro Pulse & Risk Dashboard")
st.caption("Macro indicators relevant to BKLN · ARCC · PDI · Cash decisions")

col_refresh, col_ts = st.columns([1, 5])
if col_refresh.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

macro  = _macro(cfg_hash)
regime = get_macro_regime(macro)
gauges = get_risk_gauges(macro)
rates  = macro["rates"]
vix    = macro["vix"]
oil    = macro["oil"]
yc     = macro["yield_curve"]
rec    = macro["recession"]
fx     = macro["fx"]

col_ts.caption(f"Last fetched: {macro['fetched_at']}  |  "
               f"Thai rate as of: {rates['thai_rate_date']}  |  "
               f"Fed rate as of: {rates['us_fed_date']}")

# ── Traffic light regime banner ───────────────────────────────────────────────
regime_colors = {"Defensive": "error", "Neutral": "warning", "Aggressive": "success"}
getattr(st, regime_colors.get(regime["regime"], "info"))(
    f"**{_icon(regime['color'])} Macro Regime: {regime['regime']}**  |  "
    f"Score {regime['score']}/10  |  "
    f"Suggested cash: **{regime['cash_pct']}** ({regime['cash_thb_est']})  |  "
    f"{regime['action']}"
)

st.divider()

# ── 6 Metric cards (2 rows × 3 cols) ─────────────────────────────────────────
r1c1, r1c2, r1c3 = st.columns(3)
r2c1, r2c2, r2c3 = st.columns(3)

with r1c1:
    st.metric(f"{_icon(rates['thai_signal'])} Thai Policy Rate (BOT)",
              f"{rates['thai_rate']:.2f}%",
              help="Bank of Thailand 1-day repo rate. Update manually in portfolio.yaml → macro → thai_policy_rate")
    st.caption(f"As of {rates['thai_rate_date']} · Next BOT meeting: check bot.or.th")

with r1c2:
    st.metric(f"{_icon(rates['us_signal'])} US Fed Funds Rate",
              f"{rates['us_fed_rate']:.2f}%",
              help="Federal Reserve target rate midpoint. Update in portfolio.yaml → macro → us_fed_rate")
    st.caption(f"As of {rates['us_fed_date']} · BKLN income ↑ when rates stay elevated")

with r1c3:
    cur_vix = vix.get("current"); d_vix = vix.get("change_30d")
    st.metric(f"{_icon(vix['signal'])} VIX (Uncertainty)",
              f"{cur_vix:.1f}" if cur_vix else "n/a",
              delta=f"{d_vix:+.1f} (30d)" if d_vix else None)
    if not vix["series"].empty:
        st.plotly_chart(_spark(vix["series"], "#E24B4A" if vix["signal"]=="red" else "#BA7517"),
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
    z = fx.get("zscore"); fx_sig = fx.get("signal","unknown")
    st.metric(f"{_icon('green' if fx_sig in ['buy','strong_buy'] else ('red' if fx_sig=='avoid' else 'yellow'))} USD/THB Signal",
              f"{fx.get('current',0):.4f}",
              delta=f"z={z:+.2f}" if z else None,
              help="z-score < -0.5: good time to convert THB → USD. Reuses engine/fx_timing.py")
    st.caption(fx.get("advice", ""))

with r2c3:
    prob = rec.get("probability", 0)
    sig  = rec.get("signal","yellow")
    st.metric(f"{_icon(sig)} US Recession Probability",
              f"~{prob}%",
              help="Proxy via 2s10s yield curve inversion depth and duration.")
    st.caption(rec.get("note",""))

st.divider()

# ── 4 Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Policy Rates", "⚡ Risk Gauges",
    "🌐 Economic Snapshot", "💼 Portfolio Implications"
])

# ── Tab 1: Policy Rates ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Central bank policy rates")
    st.caption("Thai BOT rate is updated manually. US Fed proxied via 13W T-bill yield.")

    c1, c2 = st.columns(2)
    with c1:
        irx = macro["liquidity"].get("series", pd.Series(dtype=float))
        if not irx.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=irx.index, y=irx.values,
                                     name="US 13W T-bill (Fed proxy)",
                                     line=dict(color="#3B8BD4", width=2)))
            fig.add_hline(y=rates["us_fed_rate"], line_dash="dot",
                          line_color="#888", annotation_text=f"Fed {rates['us_fed_rate']:.2f}%")
            fig.update_layout(title="US short-term rate (13W T-bill)",
                              height=280, margin=dict(l=0,r=0,t=30,b=0),
                              xaxis_title=None, yaxis_title="%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("US rate data unavailable")

    with c2:
        # Thai rate — step chart from manual config (just a single point + context)
        fig2 = go.Figure()
        fig2.add_trace(go.Indicator(
            mode="number+delta",
            value=rates["thai_rate"],
            delta={"reference": 1.50, "relative": False,
                   "suffix": "% vs 2024 high"},
            title={"text": f"Thai BOT Rate<br><span style='font-size:12px'>As of {rates['thai_rate_date']}</span>"},
            number={"suffix": "%"},
        ))
        fig2.update_layout(height=280, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Update `portfolio.yaml → macro → thai_policy_rate` after each BOT meeting. "
                   "Next BOT MPC dates: [bot.or.th/monetary-policy/monetary-policy-committee-mpc](https://www.bot.or.th)")

    # Yield curve
    st.subheader("US yield curve (2Y vs 10Y)")
    if not yc.get("series_spread", pd.Series(dtype=float)).empty:
        sp = yc["series_spread"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sp.index, y=sp.values, mode="lines",
                                   name="2s10s spread",
                                   line=dict(color="#1D9E75" if not yc["inverted"] else "#E24B4A", width=2),
                                   fill="tozeroy",
                                   fillcolor="rgba(29,158,117,0.1)" if not yc["inverted"] else "rgba(226,75,74,0.1)"))
        fig3.add_hline(y=0, line_color="#888", line_dash="dash")
        fig3.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_title="% spread", xaxis_title=None)
        st.plotly_chart(fig3, use_container_width=True)
        inv_msg = "⚠ Yield curve inverted — historically precedes recession by 12–18 months" if yc["inverted"] else "Yield curve not inverted — neutral signal"
        st.caption(inv_msg)
    else:
        st.info("Yield curve data unavailable")


# ── Tab 2: Risk Gauges ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Risk gauge overview")
    st.caption("Higher score = more stress. Threshold: <35 green · 35–65 yellow · >65 red")

    g1, g2, g3, g4 = st.columns(4)
    sigs = {
        "default_risk":   macro["credit"].get("signal","yellow"),
        "liquidity_risk": macro["liquidity"].get("signal","yellow"),
        "maturity_risk":  yc.get("signal","yellow"),
        "uncertainty":    vix.get("signal","yellow"),
    }
    labels = {
        "default_risk":   "Default / Credit Risk",
        "liquidity_risk": "Liquidity Risk",
        "maturity_risk":  "Maturity / Curve Risk",
        "uncertainty":    "Uncertainty (VIX)",
    }
    for col, key in zip([g1,g2,g3,g4], gauges):
        with col:
            st.plotly_chart(
                _gauge(labels[key], gauges[key], sigs[key]),
                use_container_width=True, config={"displayModeBar": False}
            )
            st.caption(f"Score: {gauges[key]}/100")

    # Composite score bar
    composite = round(sum(gauges.values()) / 4)
    st.subheader(f"Composite stress score: {composite}/100")
    st.progress(composite / 100)

    st.divider()
    with st.expander("What each gauge measures"):
        st.markdown("""
**Default / Credit Risk** — Relative performance of HYG (high yield) vs LQD (investment grade).
HY underperformance signals rising default expectations. Relevant to ARCC credit exposure.

**Liquidity Risk** — 13-week T-bill yield level as a proxy for short-term funding stress.
Elevated short rates tighten credit conditions for BDCs like ARCC.

**Maturity / Curve Risk** — 2s10s yield spread. Deeply inverted curve = maturity mismatch risk
for floating-rate instruments. BKLN resets quarterly so it is partially insulated.

**Uncertainty (VIX)** — CBOE Volatility Index. Elevated VIX raises discount rates across all
risk assets including CEFs like PDI. VIX > 28 typically compresses CEF premiums.
        """)


# ── Tab 3: Economic Snapshot ──────────────────────────────────────────────────
with tab3:
    st.subheader("Market-based economic indicators")
    st.caption("Using market-traded proxies. For authoritative inflation data check BOT / BLS.")

    c1, c2 = st.columns(2)

    with c1:
        # 10Y yield as inflation / growth proxy
        t10 = yc.get("series_10y", pd.Series(dtype=float.iloc[-1]))
        if not t10.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t10.index, y=t10.values, mode="lines",
                                     name="US 10Y yield", line=dict(color="#3B8BD4", width=2),
                                     fill="tozeroy", fillcolor="rgba(59,139,212,0.1)"))
            fig.update_layout(title="US 10Y Treasury yield (%)", height=240,
                              margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Rising 10Y yield → higher BKLN floating income; pressure on longer-duration assets")

    with c2:
        oil_s = macro["oil"].get("series", pd.Series(dtype=float.iloc[-1]))
        if not oil_s.empty:
            cur_o = float(oil_s.iloc[-1])
            color_o = "#E24B4A" if cur_o > 95 else ("#BA7517" if cur_o > 80 else "#1D9E75")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=oil_s.index, y=oil_s.values, mode="lines",
                                      name="WTI Oil", line=dict(color=color_o, width=2),
                                      fill="tozeroy", fillcolor="rgba(150, 150, 150, 0.1)"))
            fig2.add_hline(y=95, line_dash="dot", line_color="#E24B4A",
                           annotation_text="Defensive threshold $95")
            fig2.update_layout(title="WTI Oil ($/bbl)", height=240,
                               margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    # HYG vs LQD
    hy_s = macro["credit"].get("series_hy", pd.Series(dtype=float.iloc[-1]))
    ig_s = macro["credit"].get("series_ig", pd.Series(dtype=float.iloc[-1]))
    if not hy_s.empty and not ig_s.empty:
        aligned = pd.DataFrame({"HYG (High Yield)": hy_s, "LQD (Inv Grade)": ig_s}).dropna()
        # Normalise to 100 at start
        norm = aligned / aligned.iloc[0] * 100
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=norm.index, y=norm["HYG (High Yield)"],
                                   name="HYG", line=dict(color="#E24B4A", width=1.5)))
        fig3.add_trace(go.Scatter(x=norm.index, y=norm["LQD (Inv Grade)"],
                                   name="LQD", line=dict(color="#3B8BD4", width=1.5)))
        fig3.update_layout(title="HYG vs LQD (rebased 100) — credit stress proxy",
                           height=220, margin=dict(l=0,r=0,t=30,b=0),
                           legend=dict(orientation="h"))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("HYG lagging LQD = rising credit stress = ARCC / PDI headwind")


# ── Tab 4: Portfolio Implications ─────────────────────────────────────────────
with tab4:
    st.subheader("Portfolio implications & cash deployment")

    # Correlation heatmap placeholder (requires returns data)
    st.info("Correlation heatmap: loads when existing portfolio price data is available. "
            "Navigate to Analytics page first to prime the cache, then return here.")

    st.divider()

    # Cash recommendation card
    st.subheader("Cash deployment recommendation")
    rec_color = {"Defensive":"error","Neutral":"warning","Aggressive":"success"}
    getattr(st, rec_color.get(regime["regime"],"info"))(
        f"**Regime: {regime['regime']}** (score {regime['score']}/10)\n\n"
        f"Suggested cash allocation: **{regime['cash_pct']}**\n\n"
        f"{regime['cash_thb_est']}\n\n"
        f"**Action:** {regime['action']}"
    )

    # PDI deployment suggestion
    st.subheader("PDI deployment decision")
    cash_thb = cfg.get("analysis", {}).get("cash_reserve_thb", 92000)
    fx_cur   = fx.get("current", 32.68)
    cash_usd = cash_thb / fx_cur

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Available cash (THB)", f"฿{cash_thb:,.0f}")
        st.metric("Available cash (USD)", f"${cash_usd:,.2f}")
    with col_b:
        if regime["regime"] == "Aggressive":
            pdi_deploy_pct = 0.80
            pdi_msg = "Deploy up to 80% → PDI + BKLN DCA"
        elif regime["regime"] == "Neutral":
            pdi_deploy_pct = 0.50
            pdi_msg = "Deploy 50% cautiously → split PDI / BKLN"
        else:
            pdi_deploy_pct = 0.20
            pdi_msg = "Deploy 20% max. Preserve cash for entry."
        st.metric("Suggested deployment", f"${cash_usd * pdi_deploy_pct:,.2f}",
                  delta=f"{pdi_deploy_pct*100:.0f}% of available")
        st.caption(pdi_msg)

    # BKLN income impact at current regime
    st.divider()
    st.subheader("Income sensitivity to macro regime")

    bkln_shares = 192   # from your portfolio — could read from cfg
    monthly_per_share = 0.100
    base_income = bkln_shares * monthly_per_share

    scenario_data = {
        "Scenario": ["Rates cut 100bps", "Rates unchanged (current)", "Rates rise 50bps"],
        "BKLN income change": ["-12 to -15%", "Baseline", "+5 to +7%"],
        "Est. monthly income": [
            f"${base_income * 0.87:,.2f}",
            f"${base_income:,.2f}",
            f"${base_income * 1.06:,.2f}",
        ],
        "Action": ["DCA to offset yield decline", "Hold and compound", "Accelerate DCA"],
    }
    st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, hide_index=True)
    st.caption("BKLN holds floating-rate senior loans that reprice quarterly. "
               "Rate sensitivity is lower than fixed-rate bonds but still present.")