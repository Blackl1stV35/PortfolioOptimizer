"""
ui/app.py  --  Streamlit Portfolio Dashboard
=============================================
Run: streamlit run ui/app.py
Opens http://localhost:8501

Features:
  - Live portfolio overview (P&L, dividends, FX)
  - Trade entry form -- no YAML editing needed
  - FX timing signal with z-score chart
  - Dividend tracker + calendar download
  - Monte Carlo income projection
  - WHT reconciliation
  - Backtest equity curves
  - Generational wealth plan

The YAML is read on every page load so changes made elsewhere
are reflected immediately.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Add project root to path so engine imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import streamlit as st
from datetime import datetime, date
from collections import defaultdict

st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG_FILE = ROOT / "config" / "portfolio.yaml"
OUTPUT_DIR  = ROOT / "output"

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_prices(tickers: list) -> pd.DataFrame:
    raw = yf.download(tickers, period="6mo", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"].ffill()
    return raw[["Close"]].ffill()

# ── dataframe cleaning helper ────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Arrow compatibility by fixing mixed-type object columns."""
    for col in df.columns:
        if df[col].dtype == "object":
            # Try numeric conversion
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # If all NaN, fallback to string
            if df[col].isna().all():
                df[col] = df[col].astype(str)
    return df

def load_cfg() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_cfg(cfg: dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)


def derive_holdings(cfg: dict) -> dict:
    shares = defaultdict(int); cost = defaultdict(float); comm = defaultdict(float)
    for tx in cfg.get("transactions", []):
        if tx["type"] == "BUY":
            t = tx["ticker"]
            shares[t] += tx["shares"]
            cost[t]   += tx["total_usd"]
            comm[t]   += tx["commission_usd"]
        elif tx["type"] == "SELL":
            shares[tx["ticker"]] -= tx["shares"]
    return {t: {"shares": shares[t],
                "avg_cost": cost[t]/shares[t],
                "total_cost": cost[t]}
            for t, s in shares.items() if s > 0}


def color_delta(v: float, good_positive: bool = True) -> str:
    if v > 0: return "🟢" if good_positive else "🔴"
    if v < 0: return "🔴" if good_positive else "🟢"
    return "⚪"


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Portfolio Analytics")
    st.markdown("*Aggressive Income Builder*")
    cfg       = load_cfg()
    holdings  = derive_holdings(cfg)
    fx        = cfg.get("meta", {}).get("fx_usd_thb", 32.68)
    wht       = cfg.get("settings", {}).get("wht_active", 0.30)
    tickers   = list(holdings.keys())

    st.markdown(f"**Account:** {cfg.get('meta',{}).get('account_id','')}")
    st.markdown(f"**As of:** {cfg.get('meta',{}).get('data_as_of','')}")
    st.markdown(f"**FX rate:** {fx:.4f} THB/USD")
    st.markdown(f"**WHT rate:** {wht*100:.0f}%")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Dashboard",
        "📝 Trade Entry",
        "💱 FX Timing",
        "💰 Dividend Tracker",
        "📈 Monte Carlo",
        "🏛 Generational Plan",
        "🔍 WHT Reconciliation",
        "⏱ Backtest",
        "⚙️ Settings",
    ])


# ── Page: Dashboard ───────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("Portfolio Dashboard")

    if tickers:
        prices = load_prices(tickers)
        snap = cfg.get("ks_app_snapshot_20260327", {})
        pos  = snap.get("positions", {})

        # Summary metrics
        cols = st.columns(4)
        mkt_thb  = snap.get("market_value_thb", 0)
        cost_thb = snap.get("total_cost_thb", 0)
        unr_thb  = snap.get("unrealized_thb", 0)
        div_thb  = snap.get("total_dividends_thb", 0)
        cols[0].metric("Market Value (THB)", f"฿{mkt_thb:,.0f}", f"฿{unr_thb:,.0f}")
        cols[1].metric("Total Cost (THB)", f"฿{cost_thb:,.0f}")
        cols[2].metric("Dividends Received", f"฿{div_thb:,.2f}")
        cols[3].metric("Cash (USD)", f"${cfg.get('cash',{}).get('usd',0):.2f}")

        st.divider()

        # Holdings table
        st.subheader("Current Holdings")
        rows = []
        total_mkt = 0
        for tkr, h in holdings.items():
            curr = float(prices[tkr].iloc[-1]) if tkr in prices.columns else 0
            mkt  = h["shares"] * curr
            unr  = mkt - h["total_cost"]
            total_mkt += mkt
            rows.append({
                "Ticker":          tkr,
                "Shares":          h["shares"],
                "Avg Cost":        f"${h['avg_cost']:.2f}",
                "Current":         f"${curr:.2f}",
                "Market Value":    f"${mkt:,.2f}",
                "Unrealised":      f"${unr:+,.2f}",
                "Unrealised THB":  f"฿{unr*fx:+,.0f}",
                "P&L %":           f"{unr/h['total_cost']*100:+.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Mini price chart
        st.subheader("Price History (6 months)")
        fig = go.Figure()
        for tkr in tickers:
            if tkr in prices.columns:
                p = prices[tkr].dropna()
                fig.add_trace(go.Scatter(x=p.index, y=p, name=tkr, mode="lines"))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("No holdings found. Add transactions in portfolio.yaml.")


# ── Page: Trade Entry ─────────────────────────────────────────────────────────
elif page == "📝 Trade Entry":
    st.title("Log a Trade")
    st.info("Fill in the form and click Save. The YAML is updated automatically.")

    txns = cfg.get("transactions", [])
    next_id = f"T{len(txns)+1:03d}"

    with st.form("trade_form"):
        c1, c2, c3 = st.columns(3)
        tx_date   = c1.date_input("Date", value=date.today())
        tx_type   = c2.selectbox("Type", ["BUY", "SELL"])
        tx_ticker = c3.text_input("Ticker", value="BKLN")

        c4, c5, c6 = st.columns(3)
        tx_shares = c4.number_input("Shares", min_value=1, step=1, value=50)
        tx_price  = c5.number_input("Price (USD)", min_value=0.01, value=21.00, format="%.2f")
        tx_comm   = c6.number_input("Commission (USD)", min_value=0.0, value=5.34, format="%.2f")

        tx_exch = st.selectbox("Exchange", ["ARCX", "XNAS", "NYSE"])
        tx_note = st.text_input("Note")

        gross   = tx_shares * tx_price
        total   = gross + tx_comm if tx_type == "BUY" else gross - tx_comm
        st.markdown(f"**Gross:** ${gross:,.2f}  |  **Total:** ${total:,.2f}")

        submitted = st.form_submit_button("Save Transaction")
        if submitted:
            new_tx = {
                "id":            next_id,
                "date":          str(tx_date),
                "type":          tx_type,
                "ticker":        tx_ticker.upper(),
                "exchange":      tx_exch,
                "currency":      "USD",
                "shares":        int(tx_shares),
                "price_usd":     float(tx_price),
                "gross_usd":     round(float(gross), 2),
                "commission_usd":float(tx_comm),
                "total_usd":     round(float(total), 2),
                "note":          tx_note,
            }
            cfg.setdefault("transactions", []).append(new_tx)
            cfg["meta"]["data_as_of"] = str(date.today())
            save_cfg(cfg)
            st.success(f"Saved {next_id}: {tx_type} {tx_shares} {tx_ticker} @ ${tx_price:.2f}")
            st.rerun()

    st.subheader("Transaction History")
    tx_df = pd.DataFrame(cfg.get("transactions", []))
    if not tx_df.empty:
        st.dataframe(tx_df, width="stretch", hide_index=True)


# ── Page: FX Timing ───────────────────────────────────────────────────────────
elif page == "💱 FX Timing":
    st.title("FX Timing  --  THB → USD Conversion Signal")

    try:
        from engine.fx_timing import download_fx, compute_fx_signal, dca_budget_usd
        with st.spinner("Loading FX data..."):
            fx_hist = download_fx(lookback_days=365)
            signal  = compute_fx_signal(fx_hist)

        sig_name = signal.get("signal", "unknown")
        sig_color = {"strong_buy":"🟢","buy":"🟢","neutral":"🟡",
                     "caution":"🟠","avoid":"🔴"}.get(sig_name, "⚪")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("USD/THB Now",     f"{signal['current']:.4f}")
        c2.metric("90-day Mean",     f"{signal['mean_90d']:.4f}")
        c3.metric("Z-score",         f"{signal['zscore']:+.3f}")
        c4.metric("Signal", f"{sig_color} {sig_name.replace('_',' ').title()}")

        st.info(f"**Advice:** {signal.get('advice','')}")
        st.caption(f"52-week range: {signal['52w_low']:.4f} – {signal['52w_high']:.4f}  |  "
                   f"Percentile: {signal['percentile']:.1f}th")

        # FX history chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fx_hist.index, y=fx_hist.values, name="USD/THB", line_color="#3B8BD4"))
        roll_mean = fx_hist.rolling(90).mean()
        roll_std  = fx_hist.rolling(90).std()
        fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean.values,
                                 name="90d mean", line=dict(color="#888", dash="dash")))
        fig.add_trace(go.Scatter(x=fx_hist.index, y=(roll_mean+roll_std).values,
                                 name="+1σ", line=dict(color="#BA7517", width=0.5, dash="dot"), showlegend=False))
        fig.add_trace(go.Scatter(x=fx_hist.index, y=(roll_mean-roll_std).values,
                                 name="-1σ", line=dict(color="#1D9E75", width=0.5, dash="dot"), showlegend=False))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

        # Budget calculator
        st.subheader("DCA Budget Calculator")
        budget_thb = st.slider("THB amount to convert", 10000, 200000, 51000, 1000)
        budget     = dca_budget_usd(budget_thb, signal["current"], signal)
        c1, c2, c3 = st.columns(3)
        c1.metric("USD gross", f"${budget['usd_gross']:,.2f}")
        c2.metric("USD net (after commission)", f"${budget['usd_net']:,.2f}")
        c3.metric("Est. BKLN shares", f"{budget['shares_bkln']}")
        if budget["usd_vs_avg"] != 0:
            delta_dir = "more" if budget["usd_vs_avg"] > 0 else "less"
            st.caption(f"vs 90-day average rate: ${abs(budget['usd_vs_avg']):.2f} {delta_dir} USD than at average rate")

    except Exception as e:
        st.error(f"FX module error: {e}")


# ── Page: Dividend Tracker ────────────────────────────────────────────────────
elif page == "💰 Dividend Tracker":
    st.title("Dividend Tracker")

    divs = cfg.get("dividends_received", [])
    if divs:
        rows = []
        for d in divs:
            sh    = d.get("shares_eligible", 0)
            pps   = d.get("amount_per_share_usd", 0.0)
            gross = sh * pps
            rows.append({
                "Period":       d.get("period",""),
                "Ticker":       d.get("ticker",""),
                "Shares":       sh,
                "$/share":      pps,
                "Gross $":      f"${gross:.2f}",
                "Net @30% WHT": f"${gross*0.70:.2f}",
                "Net @15% WHT": f"${gross*0.85:.2f}",
                "KS THB":       d.get("thb_ks_app","n/a"),
                "Status":       d.get("status","received"),
            })
            df = pd.DataFrame(rows)
            df = clean_dataframe(df)
            st.dataframe(df, width="stretch")
    else:
        st.info("No dividend records in portfolio.yaml yet.")

    st.subheader("Upcoming projected dividends")
    upcoming_rows = []
    for tkr, inst in cfg.get("instruments", {}).items():
        sh = holdings.get(tkr, {}).get("shares", 0)
        if sh == 0: continue
        for u in inst.get("dividend_policy", {}).get("estimated_upcoming", []):
            amt      = u.get("amount", u.get("amount_per_share_usd", 0.0))
            eligible = u.get("eligible_for_our_shares", True)
            gross    = sh * amt if eligible else 0.0
            upcoming_rows.append({
                "Period":  u.get("period", u.get("ex", "")),
                "Ticker":  tkr,
                "Ex-date": u.get("ex",""),
                "Pay-date":u.get("pay",""),
                "Shares":  sh if eligible else 0,
                "Gross $": f"${gross:.2f}",
                "Net @WHT":f"${gross*(1-wht):.2f}",
                "Status":  "projected" if eligible else "MISSED",
            })
    if upcoming_rows:
        up_df = pd.DataFrame(upcoming_rows)
        up_df = clean_dataframe(up_df)
        st.dataframe(up_df, width="stretch", hide_index=True)

    # Download calendar
    st.divider()
    st.subheader("Download Dividend Calendar")
    if st.button("Generate .ics calendar file"):
        try:
            from engine.dividend_calendar import run_calendar
            events, cal_path = run_calendar(cfg, holdings)
            with open(cal_path, "rb") as f:
                st.download_button("Download portfolio_dividends.ics", f.read(),
                                   file_name="portfolio_dividends.ics",
                                   mime="text/calendar")
            st.success(f"{len(events)} events generated")
        except Exception as e:
            st.error(f"Calendar error: {e}")


# ── Page: Monte Carlo ─────────────────────────────────────────────────────────
elif page == "📈 Monte Carlo":
    st.title("Monte Carlo Income Projection")
    st.caption("Bootstrap simulation of portfolio value and dividend income. Uses Julia if installed.")

    c1, c2, c3 = st.columns(3)
    n_paths   = c1.select_slider("Paths", [1000, 5000, 10000, 20000], value=5000)
    n_years   = c2.slider("Horizon (years)", 1, 10, 5)
    monthly_a = c3.number_input("Monthly DCA add (USD)", 0, 5000, 200, 50)

    if st.button("Run simulation"):
        try:
            from engine.julia_bridge import julia_monte_carlo, init_julia
            import riskfolio as rp

            with st.spinner("Downloading price data..."):
                prices_h = load_prices(tickers)
                monthly  = prices_h.resample("ME").last()
                returns  = monthly.pct_change().dropna()

            port  = rp.Portfolio(returns=returns)
            port.assets_stats(method_mu="hist", method_cov="ledoit")
            w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe",
                                      rf=0.045/12, l=2, hist=True)
            if w_opt is None:
                w = np.ones(len(tickers)) / len(tickers)
            else:
                w = w_opt["weights"].values

            port_val = sum(holdings[t]["shares"] *
                          float(prices_h[t].iloc[-1])
                          for t in tickers if t in prices_h.columns)
            income_r = 0.0707 / 12

            with st.spinner(f"Running {n_paths:,} paths × {n_years*12} months..."):
                vp, ip = julia_monte_carlo(returns, w, income_r, port_val,
                                           n_paths, n_years * 12)

            # Fan chart
            x = np.arange(1, n_years*12+1)
            p10 = np.percentile(vp, 10, axis=0)
            p50 = np.percentile(vp, 50, axis=0)
            p90 = np.percentile(vp, 90, axis=0)
            fig = go.Figure([
                go.Scatter(x=x, y=p90, name="p90", line_color="rgba(29,158,117,0.3)", showlegend=False),
                go.Scatter(x=x, y=p10, name="p10", fill="tonexty",
                           fillcolor="rgba(29,158,117,0.15)", line_color="rgba(29,158,117,0.3)"),
                go.Scatter(x=x, y=p50, name="Median", line=dict(color="#1D9E75", width=2.5)),
            ])
            fig.update_layout(title="Portfolio value paths", height=350,
                              xaxis_title="Month", yaxis_title="USD",
                              margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, width="stretch")

            # Summary stats
            final_p50 = np.percentile(vp[:,-1], 50)
            final_inc = final_p50 * income_r
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("p50 Final Value",   f"${np.percentile(vp[:,-1],50):,.0f}")
            c2.metric("p10 Final Value",   f"${np.percentile(vp[:,-1],10):,.0f}")
            c3.metric("p90 Final Value",   f"${np.percentile(vp[:,-1],90):,.0f}")
            c4.metric("p50 Monthly Income",f"${final_inc:,.2f}")

        except Exception as e:
            st.error(f"Monte Carlo error: {e}")
            import traceback; st.code(traceback.format_exc())


# ── Page: Generational Plan ───────────────────────────────────────────────────
elif page == "🏛 Generational Plan":
    st.title("Generational Wealth Plan  --  30-Year Horizon")
    st.caption("Models portfolio growth, income, and transfer value for indefinite holding period.")

    from engine.generational_planner import GenPlanConfig, run_generational_plan

    c1, c2 = st.columns(2)
    target_income = c1.number_input("Target monthly income (USD)", 100, 10000, 1000, 100)
    monthly_dca   = c2.number_input("Monthly DCA contribution (USD)", 0, 5000, 200, 50)

    if st.button("Run 30-year plan"):
        try:
            with st.spinner("Running simulation..."):
                prices_h = load_prices(tickers)
                monthly  = prices_h.resample("ME").last()
                returns  = monthly.pct_change().dropna()

                import riskfolio as rp
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="ledoit")
                w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe",
                                          rf=0.045/12, l=2, hist=True)
                w = w_opt["weights"].values if w_opt is not None else np.ones(len(tickers))/len(tickers)

                port_val = sum(holdings[t]["shares"] *
                               float(prices_h[t].iloc[-1])
                               for t in tickers if t in prices_h.columns)

                plan = GenPlanConfig(
                    n_paths=5000, horizon_years=30,
                    monthly_add_usd=monthly_dca,
                    income_rate_m=0.0707/12,
                    target_income_m=target_income,
                    initial_value=port_val,
                )
                result = run_generational_plan(returns, w, cfg, plan)

            st.subheader("Milestone Summary")
            ms_rows = []
            for yr, ms in result["milestones"].items():
                ms_rows.append({
                    "Year":            yr,
                    "p50 Value":       f"${ms['p50_value']:,.0f}",
                    "p10 Value":       f"${ms['p10_value']:,.0f}",
                    "p90 Value":       f"${ms['p90_value']:,.0f}",
                    "p50 Real (infl)": f"${ms['p50_real']:,.0f}",
                    "p50 Income/mo":   f"${ms['p50_income_m']:,.2f}",
                    f"P(>$target)":    f"{ms['prob_above_target']:.1f}%",
                    "Cum. Dividends":  f"${ms['p50_cum_div']:,.0f}",
                })
            st.dataframe(pd.DataFrame(ms_rows), width="stretch", hide_index=True)

            mtt = result.get("months_to_target", {})
            if mtt.get("years"):
                st.success(f"Median time to reach ${target_income:,}/mo income: "
                           f"~{mtt['years']} years {mtt['extra_months']} months")
            else:
                st.warning(f"Target ${target_income:,}/mo not reached within 30 years at current trajectory.")

            if result.get("chart"):
                st.image(result["chart"], width="stretch")

        except Exception as e:
            st.error(f"Generational plan error: {e}")
            import traceback; st.code(traceback.format_exc())


# ── Page: WHT Reconciliation ──────────────────────────────────────────────────
elif page == "🔍 WHT Reconciliation":
    st.title("Withholding Tax Reconciliation")
    st.caption("Compares your estimated gross dividends vs KS actual received amounts to determine effective WHT rate.")

    try:
        from engine.wht_reconciliation import run_wht_reconciliation, summarise_wht, build_reconciliation
        records = build_reconciliation(cfg, fx)
        summary = summarise_wht(records)

        verdict_map = {"treaty_15":"success","default_30":"error",
                       "partial":"warning","no_data":"info","overpaid":"error"}

        # Summary box
        primary_verdict = max(set([r.verdict for r in records]), key=[r.verdict for r in records].count) if records else "no_data"
        getattr(st, verdict_map.get(primary_verdict, "info"))(summary)

        # Table
        rows = []
        for r in records:
            rows.append({
                "Period":       r.period,
                "Ticker":       r.ticker,
                "Shares":       r.shares,
                "Gross $":      f"${r.gross_usd:.4f}",
                "Net @30%":     f"${r.net_30pct:.4f}",
                "Net @15%":     f"${r.net_15pct:.4f}",
                "KS THB":       r.ks_thb or "not recorded",
                "KS USD est.":  f"${r.ks_usd_est:.4f}" if r.ks_usd_est else "n/a",
                "Implied WHT":  f"{r.implied_wht*100:.1f}%" if r.implied_wht else "n/a",
                "Verdict":      r.verdict,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.divider()
        st.subheader("Record KS actual amount")
        st.caption("After receiving a dividend, paste the KS app amount below to enable reconciliation.")
        with st.form("div_entry"):
            d_period = st.text_input("Period (e.g. 2026-04)")
            d_ticker = st.selectbox("Ticker", tickers)
            d_thb    = st.number_input("KS app THB amount", min_value=0.0, format="%.2f")
            if st.form_submit_button("Save"):
                for div in cfg.get("dividends_received", []):
                    if div.get("period") == d_period and div.get("ticker") == d_ticker:
                        div["thb_ks_app"] = float(d_thb)
                        div["status"]     = "received"
                        save_cfg(cfg)
                        st.success(f"Saved ฿{d_thb:.2f} for {d_ticker} {d_period}")
                        st.rerun()

    except Exception as e:
        st.error(f"WHT module error: {e}")


# ── Page: Backtest ────────────────────────────────────────────────────────────
elif page == "⏱ Backtest":
    st.title("Walk-Forward Backtest")
    st.caption("Tests optimisation strategies on historical data using a rolling training window.")

    train_w = st.slider("Training window (months)", 12, 36, 24)

    if st.button("Run backtest"):
        try:
            from engine.backtest import run_walkforward
            with st.spinner("Running walk-forward backtest..."):
                prices_h = load_prices(tickers)
                monthly  = prices_h.resample("ME").last()
                returns  = monthly.pct_change().dropna()
                bt       = run_walkforward(returns, train_months=train_w)

            if not bt:
                st.warning("Not enough data for backtest with this window.")
            else:
                # Metrics table
                st.subheader("Strategy metrics")
                rows = []
                for s, m in bt["metrics"].items():
                    rows.append({
                        "Strategy":     s.replace("_"," ").title(),
                        "Ann. Return":  f"{m['ann_return']*100:.2f}%",
                        "Ann. Vol":     f"{m['ann_vol']*100:.2f}%",
                        "Sharpe":       f"{m['sharpe']:.3f}",
                        "Max Drawdown": f"{m['max_drawdown']*100:.2f}%",
                        "Calmar":       f"{m['calmar']:.3f}",
                        "Final $1":     f"${m['final_equity']:.3f}",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

                # Equity chart
                eq = bt["equity"]
                fig = go.Figure()
                colors = {"max_sharpe":"#1D9E75","min_variance":"#3B8BD4",
                          "equal_weight":"#888","hrp":"#BA7517"}
                for s in eq.columns:
                    fig.add_trace(go.Scatter(x=eq.index, y=eq[s],
                                            name=s.replace("_"," ").title(),
                                            line_color=colors.get(s,"#333")))
                fig.add_hline(y=1.0, line_color="#ccc", line_dash="dot")
                fig.update_layout(title="Normalised equity curves", height=400,
                                  margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Backtest error: {e}")
            import traceback; st.code(traceback.format_exc())


# ── Page: Settings ────────────────────────────────────────────────────────────
elif page == "⚙️ Settings":
    st.title("Settings")

    with st.form("settings_form"):
        st.subheader("FX & Tax")
        new_fx  = st.number_input("USD/THB rate", value=float(fx), format="%.4f")
        new_wht = st.selectbox("WHT rate", [0.30, 0.15],
                               index=0 if wht == 0.30 else 1,
                               format_func=lambda x: f"{x*100:.0f}% (default)" if x==0.30 else f"{x*100:.0f}% (W-8BEN treaty)")
        st.subheader("Analysis")
        new_rf    = st.number_input("Risk-free rate (annual)", value=float(cfg.get("settings",{}).get("risk_free_rate_annual",0.045)), format="%.4f")
        new_dca   = st.number_input("DCA monthly budget (THB)", value=int(cfg.get("analysis",{}).get("dca_monthly_budget_thb",51000)))
        new_tgt   = st.number_input("Target monthly income (USD)", value=float(cfg.get("analysis",{}).get("target_monthly_income_usd",1000.0)))

        if st.form_submit_button("Save settings"):
            cfg.setdefault("meta",{})["fx_usd_thb"] = float(new_fx)
            cfg.setdefault("settings",{})["wht_active"] = float(new_wht)
            cfg["settings"]["risk_free_rate_annual"] = float(new_rf)
            cfg.setdefault("analysis",{})["dca_monthly_budget_thb"] = int(new_dca)
            cfg["analysis"]["target_monthly_income_usd"] = float(new_tgt)
            save_cfg(cfg)
            st.success("Settings saved to portfolio.yaml")
            st.rerun()
