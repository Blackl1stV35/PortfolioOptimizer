"""pages/p2_intelligence.py — Intelligence Hub."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import load_fx_series, t, get_lang


def render(*, active, cfg, holdings, fx_r, wht, rf_ann, tickers):
    tab_mp, tab_sec, tab_fx = st.tabs([
        "📡 " + t("ai_label") + " Macro", "📋 SEC EDGAR", "💱 FX Timing",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab_mp:
        st.subheader("Macro Pulse & Risk Dashboard")
        try:
            from engine.macro_monitor import get_macro_data, get_macro_regime, get_risk_gauges
            with st.spinner("Fetching macro indicators…"):
                macro  = get_macro_data(cfg)
                regime = get_macro_regime(macro)
                gauges = get_risk_gauges(macro)

            _reg_color = {"Defensive":"error","Neutral":"warning","Aggressive":"success"}
            getattr(st, _reg_color.get(regime["regime"],"info"))(
                f"**Macro Regime: {regime['regime']}** (score {regime['score']}/10)  |  "
                f"Cash suggestion: **{regime['cash_pct']}**  |  {regime['action']}"
            )
            rates = macro["rates"]; vix = macro["vix"]; oil = macro["oil"]; fx_sig = macro["fx"]
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Thai Rate",   f"{rates['thai_rate']:.2f}%")
            c2.metric("US Fed",      f"{rates['us_fed_rate']:.2f}%")
            c3.metric("VIX",         f"{vix.get('current',0):.1f}" if vix.get("current") else "n/a",
                      delta=f"{vix.get('change_30d',0):+.1f}" if vix.get("change_30d") else None)
            c4.metric("WTI Oil",     f"${oil.get('current',0):.1f}" if oil.get("current") else "n/a")
            c5.metric("USD/THB",     f"{fx_sig.get('current',0):.4f}")
            c6.metric("Recession %", f"~{macro['recession'].get('probability',0)}%")

            st.subheader("Risk gauges (0=calm → 100=stressed)")
            for label, key in [("Default/Credit","default_risk"),("Liquidity","liquidity_risk"),
                                ("Curve/Maturity","maturity_risk"),("Uncertainty (VIX)","uncertainty")]:
                score = gauges.get(key, 0)
                icon  = "🟢" if score < 35 else ("🟡" if score < 65 else "🔴")
                st.write(f"{icon} **{label}**: {score}/100")
                st.progress(score / 100)
        except Exception as e:
            st.error(f"Macro module: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    with tab_sec:
        st.subheader("SEC EDGAR Intelligence")
        identity = cfg.get("edgar", {}).get("identity", "")
        if not identity:
            st.error("Add `edgar.identity` to portfolio YAML to enable.")
        else:
            sub1, sub2, sub3, sub4 = st.tabs(
                ["Dividend Declarations","NAV & NII","Insider Trades","BDC Screener"])

            with sub1:
                try:
                    from engine.edgar_monitor import get_arcc_dividend_declarations
                    with st.spinner("Fetching ARCC 8-K…"):
                        divs = get_arcc_dividend_declarations(identity)
                    if divs:
                        st.dataframe(pd.DataFrame(divs), width="stretch", hide_index=True)
                    else:
                        st.info("No recent declarations (last 90 days).")
                except Exception as e:
                    st.error(str(e))

            with sub2:
                try:
                    from engine.edgar_monitor import get_arcc_fundamentals
                    with st.spinner("Fetching ARCC XBRL…"):
                        fund = get_arcc_fundamentals(identity)
                    if fund:
                        nav=fund.get("nav_latest"); nii=fund.get("nii_latest")
                        cov=fund.get("coverage_ratio")
                        c1,c2,c3 = st.columns(3)
                        c1.metric("NAV/share", f"${nav:.2f}" if nav else "n/a",
                                  delta=f"{fund.get('nav_trend_4q',0):+.3f} (4Q)")
                        c2.metric("NII/share (qtr)", f"${nii:.3f}" if nii else "n/a")
                        c3.metric("Coverage", f"{cov:.2f}×" if cov else "n/a")
                except Exception as e:
                    st.error(str(e))

            with sub3:
                try:
                    from engine.edgar_monitor import get_arcc_insider_trades
                    with st.spinner("Fetching Form 4…"):
                        trades = get_arcc_insider_trades(identity)
                    if trades:
                        st.dataframe(pd.DataFrame(trades), width="stretch", hide_index=True)
                    else:
                        st.info("No Form 4 trades (last 90 days).")
                except Exception as e:
                    st.error(str(e))

            with sub4:
                from engine.edgar_monitor import screen_bdc_candidate
                candidates = st.multiselect("Tickers to screen",
                    ["PFLT","MAIN","HTGC","GBDC","ARCC","OBDC","SLRC"], default=["PFLT"])
                if st.button("🔍 Run Screen", type="primary"):
                    for tkr in candidates:
                        with st.spinner(f"Screening {tkr}…"):
                            res = screen_bdc_candidate(tkr, identity)
                        sig = {"green":"✅","yellow":"🟡","red":"🔴"}.get(res.get("signal",""),"⚪")
                        with st.expander(f"{sig} {tkr}", expanded=True):
                            if res.get("error"):
                                st.error(res["error"])
                            else:
                                c1,c2 = st.columns(2)
                                if res.get("nav_latest"):
                                    c1.metric("NAV/share", f"${res['nav_latest']:.2f}",
                                              delta=f"{res.get('nav_1yr_chg',0)*100:+.1f}%")
                                if res.get("nii_latest"):
                                    c2.metric("NII/share", f"${res['nii_latest']:.3f}")
                                for w in res.get("warnings",[]):
                                    st.warning(w)
                                if not res.get("warnings"):
                                    st.success("No red flags.")

    # ══════════════════════════════════════════════════════════════════════════
    with tab_fx:
        st.subheader("FX Timing — USD/THB Conversion Signal")
        try:
            from engine.fx_timing import compute_fx_signal
            fx_hist  = load_fx_series()
            signal   = compute_fx_signal(fx_hist)
            sig_icon = {"strong_buy":"🟢","buy":"🟢","neutral":"🟡",
                        "caution":"🟠","avoid":"🔴"}.get(signal.get("signal",""),"⚪")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("USD/THB", f"{signal['current']:.4f}")
            c2.metric("90d Mean", f"{signal['mean_90d']:.4f}")
            c3.metric("Z-score",  f"{signal['zscore']:+.3f}")
            c4.metric("Signal",   f"{sig_icon} {signal.get('signal','').replace('_',' ').title()}")
            st.info(f"**Advice:** {signal.get('advice','')}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fx_hist.index, y=fx_hist.values,
                          name="USD/THB", line_color="#3B8BD4"))
            roll = fx_hist.rolling(90)
            fig.add_trace(go.Scatter(x=roll.mean().index, y=roll.mean().values,
                          name="90d MA", line=dict(color="#888", dash="dash")))
            fig.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0),
                              plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                              font_color="#FFFFFF")
            st.plotly_chart(fig, width="stretch")

            budget = st.slider("THB to convert", 10000, 300000, 51000, 1000)
            st.info(f"USD gross: **${budget/signal['current']:,.2f}**  |  "
                    f"After $5.34 commission: **${budget/signal['current']-5.34:,.2f}**")
        except Exception as e:
            st.error(f"FX module: {e}")
