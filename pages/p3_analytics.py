"""pages/p3_analytics.py — Analytics Engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import load_returns, load_prices, t, _julia_runtime
from utils.llm_summarizer import (summarise_risk, summarise_backtest,
                                   summarise_monte_carlo, summarise_generational,
                                   render_summary)

_MC_PALETTE = {
    "Base":   {"line": "#2E5BA8", "fill": "rgba(46,91,168,0.12)"},
    "After":  {"line": "#1D9E75", "fill": "rgba(29,158,117,0.12)"},
}


def render(*, active, cfg, holdings, fx_r, wht, rf_ann, tickers):
    if not tickers:
        st.warning("No holdings in this account."); return

    tab_ro, tab_bt, tab_mc, tab_gp, tab_ch = st.tabs([
        t("risk_optimisation"), t("backtest"),
        t("monte_carlo"), t("gen_plan"), t("advanced_charts"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # RISK & OPTIMISATION
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ro:
        st.subheader(t("risk_optimisation"))
        try:
            import riskfolio as rp
            returns = load_returns(tickers)
            risk_tbl = _compute_risk_table(returns, rf_ann/12)
            st.dataframe(_format_risk_table(risk_tbl), width="stretch")

            port = rp.Portfolio(returns=returns)
            port.assets_stats(method_mu="hist", method_cov="ledoit")
            w_dict = {}
            for label, rm, obj in [("Max Sharpe","MV","Sharpe"),
                                    ("Min Variance","MV","MinRisk"),
                                    ("Min CVaR","CVaR","MinRisk")]:
                try:
                    w = port.optimization(model="Classic", rm=rm, obj=obj,
                                           rf=rf_ann/12, l=2, hist=True)
                    if w is not None and not w.isnull().values.any():
                        w_dict[label] = w.to_dict()["weights"]
                except Exception:
                    pass
            if w_dict:
                st.subheader("Optimal weights")
                st.dataframe(
                    pd.DataFrame([{"Strategy":s, **{k:f"{v:.1%}" for k,v in ws.items()}}
                                  for s,ws in w_dict.items()]),
                    width="stretch", hide_index=True
                )
                try:
                    summary = summarise_risk(risk_tbl.to_dict(orient="index"),
                                            w_dict.get("Max Sharpe",{}))
                    render_summary(summary, "risk_opt")
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Optimisation error: {e}")
            import traceback; st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════════
    # BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    with tab_bt:
        st.subheader(t("backtest"))
        train_w = st.slider("Training window (months)", 12, 36, 24, key="bt_tw")
        if st.button(t("run"), type="primary", key="run_bt"):
            try:
                from engine.backtest import run_walkforward
                returns = load_returns(tickers)
                with st.spinner("Running walk-forward…"):
                    bt = run_walkforward(returns, train_months=train_w)
                if not bt:
                    st.warning("Insufficient data.")
                else:
                    rows = [{"Strategy": s.replace("_"," ").title(),
                             "Ann. Return": f"{m['ann_return']*100:.2f}%",
                             "Sharpe": f"{m['sharpe']:.3f}",
                             "Max DD": f"{m['max_drawdown']*100:.2f}%",
                             "Final $1": f"${m['final_equity']:.3f}"}
                            for s,m in bt["metrics"].items()]
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    if bt.get("equity_chart"):
                        st.image(bt["equity_chart"], width="stretch")
                    try:
                        summary = summarise_backtest(bt["metrics"])
                        render_summary(summary, "backtest")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Backtest error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════════
    with tab_mc:
        st.subheader(t("monte_carlo"))
        c1,c2,c3 = st.columns(3)
        n_paths  = c1.select_slider("Paths", [1000,3000,5000,10000], 3000, key="mc_paths")
        n_years  = c2.slider("Horizon (years)", 1, 10, 5, key="mc_years")
        mo_add   = c3.number_input("Monthly DCA (USD)", 0, 5000, 200, 50, key="mc_dca")
        target   = st.number_input("Target income/mo (USD)", 100, 10000, 1000, 100, key="mc_target")

        if st.button(t("run"), type="primary", key="run_mc"):
            try:
                returns = load_returns(tickers)
                w       = _get_weights(returns, rf_ann/12)
                snap    = _latest_snapshot(cfg)
                pv      = float(snap.get("market_value_usd", 0) or
                                snap.get("market_value_thb",0)/fx_r)

                with st.spinner("Running Monte Carlo…"):
                    vp, ip = _run_mc(returns, w, 0.0707/12, pv,
                                     float(mo_add), n_paths, n_years*12)

                p50 = np.percentile(vp,50,axis=0)
                p10 = np.percentile(vp,10,axis=0)
                p90 = np.percentile(vp,90,axis=0)
                x   = list(range(1, n_years*12+1))

                fig = go.Figure([
                    go.Scatter(x=x, y=p90.tolist(), name="p90",
                               line=dict(color="rgba(29,158,117,0.3)"), showlegend=False),
                    go.Scatter(x=x, y=p10.tolist(), name="p10–p90",
                               fill="tonexty", fillcolor="rgba(29,158,117,0.12)",
                               line=dict(color="rgba(29,158,117,0.3)")),
                    go.Scatter(x=x, y=p50.tolist(), name="Median",
                               line=dict(color="#1D9E75", width=2.5)),
                ])
                fig.add_hline(y=target/(0.0707/12)/12, line=dict(color="#E24B4A",width=1,dash="dot"),
                              annotation_text=f"Target portfolio for ${target}/mo")
                fig.update_layout(height=340, margin=dict(l=0,r=0,t=20,b=0),
                                  xaxis_title="Month", yaxis_title="USD",
                                  plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                                  font_color="#FFFFFF")
                st.plotly_chart(fig, width="stretch")

                c1,c2,c3,c4 = st.columns(4)
                p50f = float(np.percentile(vp[:,-1],50))
                p10f = float(np.percentile(vp[:,-1],10))
                p90f = float(np.percentile(vp[:,-1],90))
                c1.metric("p50 Final", f"${p50f:,.0f}")
                c2.metric("p10 Final", f"${p10f:,.0f}")
                c3.metric("p90 Final", f"${p90f:,.0f}")
                c4.metric("p50 Income/mo", f"${p50f*0.0707/12:,.2f}")

                prob = float(np.mean(vp[:,-1]*0.0707/12 > target)*100)
                st.metric(f"P(income > ${target}/mo at year {n_years})", f"{prob:.1f}%")

                try:
                    summary = summarise_monte_carlo(
                        {"p50":p50f,"p10":p10f,"p90":p90f}, n_years, float(target))
                    render_summary(summary, "mc")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"MC error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATIONAL PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with tab_gp:
        st.subheader(t("gen_plan"))
        c1,c2 = st.columns(2)
        target  = c1.number_input("Target income/mo (USD)", 100, 10000, 1000, 100, key="gp_target")
        mo_dca  = c2.number_input("Monthly DCA (USD)", 0, 5000, 200, 50, key="gp_dca")

        if st.button(t("run"), type="primary", key="run_gp"):
            try:
                returns = load_returns(tickers)
                w       = _get_weights(returns, rf_ann/12)
                snap    = _latest_snapshot(cfg)
                pv      = float(snap.get("market_value_usd",0) or
                                snap.get("market_value_thb",0)/fx_r)
                port_ret = returns.values @ w

                with st.spinner("Running 5,000 × 360-month generational plan…"):
                    from engine.julia_bridge import generational_plan as jl_gp
                    result = jl_gp(float(port_ret.mean()), float(port_ret.std()),
                                   float(pv), float(mo_dca), 5000,
                                   0.0707/12, float(target))

                milestones = {}
                for k, ms in result.get("milestones",{}).items():
                    yr = int(k) if str(k).isdigit() else int(str(k).replace("Year","").strip())
                    milestones[yr] = {ki: float(v) for ki,v in dict(ms).items()}

                rows = [{
                    "Year":           yr,
                    "p10 Value":      f"${ms.get('p10_value',0):,.0f}",
                    "p50 Value":      f"${ms.get('p50_value',0):,.0f}",
                    "p90 Value":      f"${ms.get('p90_value',0):,.0f}",
                    "p50 Income/mo":  f"${ms.get('p50_income_m',0):.2f}",
                    f"P(>${target}/mo)": f"{ms.get('prob_above_target',0):.1f}%",
                } for yr, ms in sorted(milestones.items())]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

                # Fan chart
                x_yr = sorted(milestones.keys())
                fig = go.Figure([
                    go.Scatter(x=x_yr, y=[milestones[y]["p90_value"] for y in x_yr],
                               name="p90", line=dict(color="rgba(29,158,117,0.3)"), showlegend=False),
                    go.Scatter(x=x_yr, y=[milestones[y]["p10_value"] for y in x_yr],
                               name="p10–p90", fill="tonexty",
                               fillcolor="rgba(29,158,117,0.12)",
                               line=dict(color="rgba(29,158,117,0.3)")),
                    go.Scatter(x=x_yr, y=[milestones[y]["p50_value"] for y in x_yr],
                               name="Median", line=dict(color="#1D9E75",width=2.5)),
                ])
                fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                                  xaxis_title="Year", yaxis_title="Portfolio (USD)",
                                  plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                                  font_color="#FFFFFF")
                st.plotly_chart(fig, width="stretch")

                mtt = {"years": int(result.get("years_to_target",0)),
                       "extra_months": int(result.get("extra_months",0))}
                if mtt["years"]:
                    st.success(f"Median time to ${target:,}/mo: "
                               f"**{mtt['years']}yr {mtt['extra_months']}mo**")
                try:
                    summary = summarise_generational(milestones, mtt, float(target))
                    render_summary(summary, "gen_plan")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Gen plan error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════════════════════
    # ADVANCED CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ch:
        st.subheader(t("advanced_charts"))
        try:
            from engine.charts import build_chart, TIMEFRAMES, INDICATOR_DEFAULTS

            all_tickers = list(tickers) + ["SPY","QQQ","AGG","^TNX"]
            col1,col2 = st.columns([3,1])
            ch_ticker = col1.selectbox("Primary ticker", all_tickers, key="ch_tk")
            ch_tf     = col2.selectbox("Timeframe", list(TIMEFRAMES.keys()),
                                       index=list(TIMEFRAMES.keys()).index("6M"), key="ch_tf")

            with st.expander("⚙️ Indicators & overlay", expanded=False):
                _icols = st.columns(4)
                ind = {n: _icols[i%4].checkbox(n, value=d, key=f"ind_{n}")
                       for i,(n,d) in enumerate(INDICATOR_DEFAULTS.items())}
                cmp_raw  = st.text_input("Compare tickers (comma-separated)", key="ch_cmp")
                bm       = st.selectbox("Benchmark", ["None","SPY","QQQ","AGG"], key="ch_bm")
                bm_val   = None if bm == "None" else bm
                cmp_list = [x.strip().upper() for x in cmp_raw.split(",") if x.strip()]

            with st.spinner(f"Loading {ch_ticker}…"):
                fig = build_chart(ch_ticker, ch_tf, ind, bm_val, cmp_list)
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Chart error: {e}")
            import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _compute_risk_table(returns, rf):
    from engine.julia_bridge import risk_metrics as jl_risk
    import numpy as np
    rows = {}
    for col in returns.columns:
        r = returns[[col]].dropna()
        w = np.array([1.0])
        try:
            m = jl_risk(r, w, rf)
        except Exception:
            from engine.julia_bridge import _py_risk_metrics
            m = _py_risk_metrics(r.values, w, rf)
        rows[col] = m
    return pd.DataFrame(rows).T.rename(columns={
        "ann_return":"Ann. Return","ann_vol":"Ann. Volatility",
        "sharpe":"Sharpe Ratio","cvar_95":"CVaR 95%","max_drawdown":"Max Drawdown"
    })


def _format_risk_table(df):
    out = df.copy()
    for col in ["Ann. Return","Ann. Volatility","CVaR 95%","Max Drawdown"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2%}")
    if "Sharpe Ratio" in out.columns:
        out["Sharpe Ratio"] = out["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    return out


def _get_weights(returns, rf):
    n = len(returns.columns)
    try:
        import riskfolio as rp
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov="ledoit")
        w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe", rf=rf, l=2, hist=True)
        if w_opt is not None and not w_opt.isnull().values.any():
            return w_opt["weights"].values
    except Exception:
        pass
    import numpy as np
    return np.ones(n) / n


def _run_mc(returns, w, income_yield, pv, monthly_add, n_paths, n_months):
    try:
        from engine.julia_bridge import monte_carlo as jl_mc
        return jl_mc(returns, w, income_yield, pv, monthly_add, n_paths, n_months)
    except Exception:
        from engine.julia_bridge import _py_monte_carlo
        return _py_monte_carlo(returns.values, w, income_yield, pv, monthly_add, n_paths, n_months)


def _latest_snapshot(cfg):
    keys = sorted([k for k in cfg if k.startswith("ks_app_snapshot_")], reverse=True)
    return cfg.get(keys[0], {}) if keys else {}
