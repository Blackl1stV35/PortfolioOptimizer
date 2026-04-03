"""
app.py  --  Portfolio Analytics Server
=======================================
Single entry point for Streamlit Community Cloud / Render / any server.

DEPLOY (free):
  Streamlit Community Cloud  -- https://share.streamlit.io
    1. Push this folder to a GitHub private repo
    2. Log in to share.streamlit.io with GitHub
    3. New app -> select repo -> Main file: app.py
    4. Deploy (free, always-on)

  Render.com (alternative)
    1. Push to GitHub
    2. New Web Service -> connect repo
    3. Build command: pip install -r requirements.txt
    4. Start command:  streamlit run app.py --server.port $PORT
    5. Free tier (spins down after inactivity, back up in ~30s)

DATA PERSISTENCE
  portfolio.yaml and views.yaml live in config/ in the GitHub repo.
  To add a trade: use the Trade Entry page -> Download updated YAML
  -> commit the downloaded file to GitHub -> Streamlit auto-redeploys.
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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
import yfinance as yf

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data helpers ──────────────────────────────────────────────────────────────
CONFIG_FILE = ROOT / "config" / "portfolio.yaml"
VIEWS_FILE  = ROOT / "config" / "views.yaml"


@st.cache_data(ttl=60)
def _load_cfg_raw() -> str:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return f.read()


def load_cfg() -> dict:
    return yaml.safe_load(_load_cfg_raw()) or {}


def derive_holdings(cfg: dict) -> dict:
    shares = defaultdict(int); cost = defaultdict(float); comm = defaultdict(float)
    for tx in cfg.get("transactions", []):
        if tx["type"] == "BUY":
            t = tx["ticker"]
            shares[t] += tx["shares"]; cost[t] += tx["total_usd"]; comm[t] += tx["commission_usd"]
        elif tx["type"] == "SELL":
            shares[tx["ticker"]] -= tx["shares"]
    return {t: {"shares": s, "avg_cost": cost[t]/s, "total_cost": round(cost[t],2),
                "total_comm": round(comm[t],2)}
            for t, s in shares.items() if s > 0}


@st.cache_data(ttl=300, show_spinner="Fetching prices...")
def load_prices(tickers: tuple) -> pd.DataFrame:
    raw = yf.download(list(tickers), period="6mo", auto_adjust=True, progress=False)
    return (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill()


@st.cache_data(ttl=300, show_spinner="Fetching FX data...")
def load_fx() -> pd.Series:
    raw = yf.download("THBUSD=X", period="1y", auto_adjust=True, progress=False)
    close = raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    return (1.0 / close).dropna()


# ── Riskfolio helpers (cached so they only run once per session) ──────────────
@st.cache_data(ttl=600, show_spinner="Running portfolio optimisation...")
def run_riskfolio(returns_tuple: tuple, rf: float) -> tuple:
    """Returns (w_dict, frontier_df, risk_tbl) all serialised as JSON-safe."""
    import riskfolio as rp
    from engine.analytics import compute_cov, compute_risk_table

    returns = pd.DataFrame(list(returns_tuple[1]),
                           index=returns_tuple[0], columns=returns_tuple[2])
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    cov = compute_cov(returns)
    if cov is not None:
        port.cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    port.sht = False; port.upperlng = 1.0

    # BL views
    try:
        from engine.black_litterman import run_black_litterman
        run_black_litterman(port, {})
    except Exception:
        pass

    strategies = [
        ("Max Sharpe (MV)",   "Classic","MV",  "Sharpe"),
        ("Min Variance",      "Classic","MV",  "MinRisk"),
        ("Max Sharpe (CVaR)", "Classic","CVaR","Sharpe"),
        ("Min CVaR",          "Classic","CVaR","MinRisk"),
        ("Max Sharpe (Sort)", "Classic","SLPM","Sharpe"),
    ]
    rf_m = rf / 12; w_dict = {}
    for label, model, rm, obj in strategies:
        try:
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf_m, l=2, hist=True)
            if w is not None and not w.isnull().values.any():
                w_dict[label] = w.to_dict()["weights"]
        except Exception:
            pass

    n = len(returns.columns)
    w_dict["Equal Weight"] = {t: 1/n for t in returns.columns}
    try:
        hp = rp.HCPortfolio(returns=returns)
        wh = hp.optimization(model="HRP", codependence="pearson", rm="MV",
                              rf=rf_m, linkage="ward", leaf_order=True)
        if wh is not None: w_dict["HRP"] = wh.to_dict()["weights"]
    except Exception:
        pass

    # Efficient frontier
    frontier_dict = {}
    if len(returns.columns) >= 3:
        try:
            fr = port.efficient_frontier(model="Classic", rm="MV", points=50,
                                          rf=rf_m, hist=True)
            frontier_dict = {c: fr[c].tolist() for c in fr.columns}
        except Exception:
            pass

    risk_tbl = compute_risk_table(returns, rf_m)
    return (w_dict, frontier_dict, risk_tbl.to_dict())


def _returns_tuple(returns: pd.DataFrame):
    return (list(returns.index), returns.values.tolist(), list(returns.columns))


# ── Plot helpers (return bytes for st.image or download) ──────────────────────
def _riskfolio_plots(returns: pd.DataFrame, w_dict: dict,
                     frontier_dict: dict, rf: float) -> dict:
    import riskfolio as rp
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = {}
    # Rebuild objects for plotting
    w_main_ser = pd.Series(list(w_dict.values())[0])
    w_main = pd.DataFrame(w_main_ser, columns=["weights"])

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")
    port.sht = False; port.upperlng = 1.0
    rf_m = rf / 12

    def _save(fig) -> bytes:
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig); buf.seek(0); return buf.read()

    for name, fn in [
        ("pie",      lambda: (plt.subplots(figsize=(7,5))[0],
                              rp.plot_pie(w=w_main, title="Allocation",
                                          others=0.05, nrow=25, cmap="tab20",
                                          ax=plt.subplots(figsize=(7,5))[1]))[0]),
        ("hist",     lambda: (f:=plt.subplots(figsize=(9,5))[0],
                              rp.plot_hist(returns=returns, w=w_main, alpha=0.05,
                                           bins=50, height=5, width=9,
                                           ax=plt.subplots(figsize=(9,5))[1]))[0]),
        ("risk_con", lambda: (f:=plt.subplots(figsize=(7,4))[0],
                              rp.plot_risk_con(w=w_main, cov=port.cov,
                                               returns=returns, rm="MV",
                                               rf=rf_m, alpha=0.05,
                                               height=4, width=7,
                                               ax=plt.subplots(figsize=(7,4))[1]))[0]),
        ("drawdown", lambda: (f:=plt.subplots(2,1,figsize=(10,7))[0],
                              rp.plot_drawdown(w=w_main, returns=returns,
                                               alpha=0.05, height=7, width=10,
                                               ax=list(plt.subplots(2,1,figsize=(10,7))[1])))[0]),
    ]:
        try:
            fig, ax = plt.subplots(figsize=(8,5))
            if name == "pie":
                rp.plot_pie(w=w_main,title="Max Sharpe Allocation",others=0.05,nrow=25,cmap="tab20",ax=ax)
            elif name == "hist":
                rp.plot_hist(returns=returns,w=w_main,alpha=0.05,bins=50,height=5,width=8,ax=ax)
            elif name == "risk_con":
                rp.plot_risk_con(w=w_main,cov=port.cov,returns=returns,rm="MV",rf=rf_m,alpha=0.05,height=5,width=8,ax=ax)
            elif name == "drawdown":
                plt.close(fig)
                fig, axes = plt.subplots(2,1,figsize=(10,7))
                rp.plot_drawdown(w=w_main,returns=returns,alpha=0.05,height=7,width=10,ax=list(axes))
            plots[name] = _save(fig)
        except Exception as e:
            plt.close("all")

    # Frontier
    if frontier_dict and len(returns.columns) >= 3:
        try:
            fr = pd.DataFrame(frontier_dict)
            fig, ax = plt.subplots(figsize=(9,6))
            rp.plot_frontier(w_frontier=fr, mu=port.mu, cov=port.cov,
                             returns=returns, rm="MV", rf=rf_m, alpha=0.05,
                             cmap="viridis", w=w_main, label="Max Sharpe", ax=ax)
            plots["frontier"] = _save(fig)
        except Exception:
            plt.close("all")

    return plots


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
cfg      = load_cfg()
holdings = derive_holdings(cfg)
fx_r     = cfg.get("meta", {}).get("fx_usd_thb", 32.68)
wht      = cfg.get("settings", {}).get("wht_active", 0.30)
rf_ann   = cfg.get("settings", {}).get("risk_free_rate_annual", 0.045)
tickers  = tuple(sorted(holdings.keys()))

with st.sidebar:
    st.markdown("### Portfolio Analytics")
    st.caption("Aggressive Income Builder · Generational Hold")
    st.divider()
    st.markdown(f"**Account:** {cfg.get('meta',{}).get('account_id','')}")
    st.markdown(f"**Holder:** {cfg.get('meta',{}).get('account_holder','')}")
    st.markdown(f"**As of:** {cfg.get('meta',{}).get('data_as_of','')}")
    st.markdown(f"**FX:** {fx_r:.4f} THB/USD  |  **WHT:** {wht*100:.0f}%")
    st.divider()
    page = st.radio("", [
        "📊 Dashboard",
        "📝 Trade Entry",
        "🔬 Analytics & Optimisation",
        "💰 Dividend Tracker",
        "💱 FX Timing",
        "📈 Monte Carlo",
        "🏛 Generational Plan",
        "🔍 WHT Reconciliation",
        "⏱ Backtest",
        "📥 Download Report",
    ], label_visibility="collapsed")

    st.divider()
    if st.button("Clear cache"):
        st.cache_data.clear(); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("Portfolio Dashboard")
    snap = cfg.get("ks_app_snapshot_20260327", {})

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Market Value", f"฿{snap.get('market_value_thb',0):,.0f}",
              f"฿{snap.get('unrealized_thb',0):,.0f}")
    c2.metric("Total Cost",   f"฿{snap.get('total_cost_thb',0):,.2f}")
    c3.metric("Dividends",    f"฿{snap.get('total_dividends_thb',0):,.2f}")
    c4.metric("Cash (USD)",   f"${cfg.get('cash',{}).get('usd',0):.2f}")

    st.divider()
    st.subheader("Holdings")
    if tickers:
        prices = load_prices(tickers)
        rows = []
        for tkr, h in holdings.items():
            curr = float(prices[tkr].iloc[-1]) if tkr in prices.columns else 0
            mkt  = h["shares"] * curr; unr = mkt - h["total_cost"]
            rows.append({"Ticker":tkr,"Shares":h["shares"],
                         "Avg Cost":f"${h['avg_cost']:.2f}","Current":f"${curr:.2f}",
                         "Market Value":f"${mkt:,.2f}","Unrealised":f"${unr:+,.2f}",
                         "THB Unrealised":f"฿{unr*fx_r:+,.0f}",
                         "P&L %":f"{unr/h['total_cost']*100:+.2f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Price chart
        st.subheader("Price History (6 months)")
        fig = go.Figure()
        for tkr in tickers:
            if tkr in prices.columns:
                p = prices[tkr].dropna()
                fig.add_trace(go.Scatter(x=p.index, y=p, name=tkr))
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        # ARCC missed dividend alert
        st.warning("ARCC Q1 2026 dividend MISSED (bought 2026-03-25, ex-date was 2026-03-12).  "
                   "Next: Q2 2026 est. ex ~2026-06-12 · 133 shares x $0.48 = **$63.84 gross**")
    else:
        st.info("No holdings. Add transactions via Trade Entry.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRADE ENTRY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Trade Entry":
    st.title("Log a Trade")
    st.info("Fill the form and download the updated YAML. Commit it to GitHub to persist.")

    txns = cfg.get("transactions", [])
    nid  = f"T{len(txns)+1:03d}"

    with st.form("trade"):
        c1,c2,c3 = st.columns(3)
        tx_date   = c1.date_input("Date", value=date.today())
        tx_type   = c2.selectbox("Type", ["BUY","SELL"])
        tx_ticker = c3.text_input("Ticker", "BKLN").upper()

        c4,c5,c6 = st.columns(3)
        tx_shares = c4.number_input("Shares", min_value=1, step=1, value=50)
        tx_price  = c5.number_input("Price USD", min_value=0.01, value=21.00, format="%.2f")
        tx_comm   = c6.number_input("Commission USD", min_value=0.0, value=5.34, format="%.2f")

        tx_exch = st.selectbox("Exchange", ["ARCX","XNAS","NYSE"])
        tx_note = st.text_input("Note (optional)")

        gross = tx_shares * tx_price
        total = gross + tx_comm if tx_type == "BUY" else gross - tx_comm
        st.caption(f"Gross: ${gross:,.2f}  |  Total: ${total:,.2f}")

        submitted = st.form_submit_button("Preview updated YAML")

    if submitted:
        new_tx = {
            "id": nid, "date": str(tx_date), "type": tx_type,
            "ticker": tx_ticker, "exchange": tx_exch, "currency": "USD",
            "shares": int(tx_shares), "price_usd": float(tx_price),
            "gross_usd": round(float(gross),2), "commission_usd": float(tx_comm),
            "total_usd": round(float(total),2), "note": tx_note,
        }
        cfg_new = dict(cfg); cfg_new.setdefault("transactions",[]).append(new_tx)
        cfg_new.setdefault("meta",{})["data_as_of"] = str(date.today())
        yaml_str = yaml.dump(cfg_new, allow_unicode=True, sort_keys=False)
        st.success(f"Trade {nid} added: {tx_type} {tx_shares} {tx_ticker} @ ${tx_price:.2f}")
        st.download_button("Download updated portfolio.yaml",
                           data=yaml_str.encode("utf-8"),
                           file_name="portfolio.yaml", mime="text/yaml")
        st.caption("Commit this file to config/portfolio.yaml in your GitHub repo to persist.")

    st.subheader("Transaction History")
    tx_df = pd.DataFrame(cfg.get("transactions",[]))
    if not tx_df.empty:
        st.dataframe(tx_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS & OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Analytics & Optimisation":
    st.title("Portfolio Analytics & Optimisation")

    if not tickers:
        st.warning("No holdings found."); st.stop()

    with st.spinner("Loading data..."):
        prices   = load_prices(tickers)
        monthly  = prices.resample("ME").last().pct_change().dropna()
        monthly  = monthly.dropna(axis=1, how="all")

    if monthly.empty or len(monthly) < 6:
        st.error("Not enough price history."); st.stop()

    rt   = _returns_tuple(monthly)
    w_dict_raw, frontier_dict, risk_tbl_dict = run_riskfolio(rt, rf_ann)

    risk_tbl = pd.DataFrame(risk_tbl_dict).T

    # Risk metrics table
    st.subheader("Risk Metrics")
    fmt_map = {"Ann. Return":"{:.2%}","Ann. Volatility":"{:.2%}","Sharpe Ratio":"{:.3f}",
               "Sortino Ratio":"{:.3f}","Max Drawdown":"{:.2%}","Calmar Ratio":"{:.3f}",
               "VaR 95%":"{:.2%}","CVaR 95%":"{:.2%}","Omega Ratio":"{:.3f}","Semi-Volatility":"{:.2%}"}
    display_tbl = risk_tbl.copy()
    for col, fmt in fmt_map.items():
        if col in display_tbl.columns:
            display_tbl[col] = display_tbl[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "n/a")
    st.dataframe(display_tbl, use_container_width=True)

    # Weights
    st.subheader("Optimal Weights")
    w_rows = []
    for strat, ws in w_dict_raw.items():
        row = {"Strategy": strat}
        for t, v in ws.items(): row[t] = f"{v:.1%}"
        w_rows.append(row)
    st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

    # Riskfolio plots
    st.subheader("Riskfolio-Lib Charts")
    with st.spinner("Generating charts..."):
        plots = _riskfolio_plots(monthly, w_dict_raw, frontier_dict, rf_ann)

    cols = st.columns(2)
    for i, (name, label) in enumerate([("pie","Allocation"),("hist","Return Distribution"),
                                        ("risk_con","Risk Contribution"),("drawdown","Drawdown")]):
        if name in plots:
            cols[i%2].image(plots[name], caption=label, use_container_width=True)
    if "frontier" in plots:
        st.image(plots["frontier"], caption="Efficient Frontier", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIVIDEND TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Dividend Tracker":
    st.title("Dividend Tracker")

    divs = cfg.get("dividends_received", [])
    if divs:
        rows = []
        for d in divs:
            sh = d.get("shares_eligible",0); pps = d.get("amount_per_share_usd",0.0)
            gross = sh*pps
            rows.append({"Period":d.get("period",""),"Ticker":d.get("ticker",""),
                         "Shares":sh,"$/share":pps,"Gross $":f"${gross:.2f}",
                         "Net @30%":f"${gross*0.70:.2f}","Net @15%":f"${gross*0.85:.2f}",
                         "KS THB":d.get("thb_ks_app","n/a"),"Status":d.get("status","")})
        st.subheader("Received dividends")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Upcoming projected")
    up_rows = []
    for tkr, inst in cfg.get("instruments",{}).items():
        sh = holdings.get(tkr,{}).get("shares",0)
        if not sh: continue
        for u in inst.get("dividend_policy",{}).get("estimated_upcoming",[]):
            amt = u.get("amount",u.get("amount_per_share_usd",0.0))
            elig = u.get("eligible_for_our_shares",True)
            gross = sh*amt if elig else 0.0
            up_rows.append({"Period":u.get("period",u.get("ex","")),"Ticker":tkr,
                             "Ex-date":u.get("ex",""),"Pay-date":u.get("pay",""),
                             "Eligible":elig,"Gross $":f"${gross:.2f}",
                             "Net @WHT":f"${gross*(1-wht):.2f}",
                             "Status":"projected" if elig else "MISSED"})
    if up_rows:
        st.dataframe(pd.DataFrame(up_rows), use_container_width=True, hide_index=True)

    # Calendar download
    st.divider()
    st.subheader("Dividend Calendar (.ics)")
    if st.button("Generate calendar file"):
        try:
            from engine.dividend_calendar import run_calendar
            events, _ = run_calendar(cfg, holdings)
            # Build ics in memory
            lines = ["BEGIN:VCALENDAR","VERSION:2.0","PRODID:-//Portfolio//EN"]
            from datetime import timedelta
            for ev in events:
                dt = ev["date"]; dt1 = dt + timedelta(days=1)
                lines += ["BEGIN:VEVENT",
                          f"UID:{ev['uid']}@portfolio",
                          f"DTSTART;VALUE=DATE:{dt.strftime('%Y%m%d')}",
                          f"DTEND;VALUE=DATE:{dt1.strftime('%Y%m%d')}",
                          f"SUMMARY:{ev['summary'].replace(',','\\,')}",
                          f"DESCRIPTION:{ev['description'].replace(chr(10),'\\n')}",
                          "END:VEVENT"]
            lines.append("END:VCALENDAR")
            ics_bytes = "\r\n".join(lines).encode("utf-8")
            st.download_button("Download portfolio_dividends.ics", ics_bytes,
                               "portfolio_dividends.ics", "text/calendar")
            st.success(f"{len(events)} calendar events ready")
        except Exception as e:
            st.error(f"Calendar error: {e}")

    # Log actual KS amount
    st.divider()
    st.subheader("Record KS actual received amount")
    with st.form("div_log"):
        dc1,dc2 = st.columns(2)
        d_period = dc1.text_input("Period (e.g. 2026-04)")
        d_ticker = dc2.selectbox("Ticker", list(tickers))
        d_thb    = st.number_input("KS app THB amount", min_value=0.0, format="%.2f")
        if st.form_submit_button("Preview updated YAML"):
            cfg_new = dict(cfg)
            for div in cfg_new.get("dividends_received",[]):
                if div.get("period")==d_period and div.get("ticker")==d_ticker:
                    div["thb_ks_app"] = float(d_thb); div["status"] = "received"
            yaml_str = yaml.dump(cfg_new, allow_unicode=True, sort_keys=False)
            st.download_button("Download updated portfolio.yaml",
                               yaml_str.encode("utf-8"), "portfolio.yaml", "text/yaml")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FX TIMING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💱 FX Timing":
    st.title("FX Timing  --  THB → USD Conversion Signal")
    try:
        from engine.fx_timing import compute_fx_signal
        fx_hist  = load_fx()
        signal   = compute_fx_signal(fx_hist)
        sig_icon = {"strong_buy":"🟢","buy":"🟢","neutral":"🟡","caution":"🟠","avoid":"🔴"}
        icon     = sig_icon.get(signal.get("signal",""), "⚪")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("USD/THB Now",  f"{signal['current']:.4f}")
        c2.metric("90-day Mean",  f"{signal['mean_90d']:.4f}")
        c3.metric("Z-score",      f"{signal['zscore']:+.3f}")
        c4.metric("Signal", f"{icon} {signal.get('signal','').replace('_',' ').title()}")
        st.info(f"**Advice:** {signal.get('advice','')}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fx_hist.index, y=fx_hist.values,
                                  name="USD/THB", line_color="#3B8BD4"))
        roll = fx_hist.rolling(90)
        fig.add_trace(go.Scatter(x=roll.mean().index, y=roll.mean().values,
                                  name="90d mean", line=dict(color="#888",dash="dash")))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("DCA Budget Calculator")
        budget_thb = st.slider("THB to convert", 10000, 200000, 51000, 1000)
        usd_gross  = budget_thb / signal["current"]
        usd_net    = usd_gross - 5.34   # KS commission
        bkln_curr  = load_prices(tickers)[tickers[0]].iloc[-1] if tickers else 21.03
        sh_est     = int(usd_net / float(bkln_curr)) if usd_net > 0 else 0
        c1,c2,c3 = st.columns(3)
        c1.metric("USD gross",            f"${usd_gross:,.2f}")
        c2.metric("USD net (after comm)", f"${usd_net:,.2f}")
        c3.metric("Est. BKLN shares",     str(sh_est))

    except Exception as e:
        st.error(f"FX module error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Monte Carlo":
    st.title("Monte Carlo Income Projection")

    c1,c2,c3 = st.columns(3)
    n_paths  = c1.select_slider("Paths", [1000,3000,5000,10000], 3000)
    n_years  = c2.slider("Horizon (years)", 1, 10, 5)
    mo_add   = c3.number_input("Monthly DCA add (USD)", 0, 5000, 200, 50)

    if st.button("Run simulation"):
        from engine.analytics import monte_carlo, compute_cov
        import riskfolio as rp

        with st.spinner("Running..."):
            prices   = load_prices(tickers)
            monthly  = prices.resample("ME").last().pct_change().dropna().dropna(axis=1,how="all")
            port     = rp.Portfolio(returns=monthly)
            port.assets_stats(method_mu="hist", method_cov="ledoit")
            w_opt = port.optimization(model="Classic",rm="MV",obj="Sharpe",
                                       rf=rf_ann/12,l=2,hist=True)
            w = w_opt["weights"].values if w_opt is not None else np.ones(len(tickers))/len(tickers)
            port_val = sum(holdings[t]["shares"]*float(prices[t].iloc[-1])
                           for t in tickers if t in prices.columns)
            vp, ip = monte_carlo(monthly, w, 0.0707/12, port_val, float(mo_add), n_paths, n_years*12)

        x = list(range(1, n_years*12+1))
        p10 = np.percentile(vp,10,axis=0); p50 = np.percentile(vp,50,axis=0)
        p90 = np.percentile(vp,90,axis=0)
        fig = go.Figure([
            go.Scatter(x=x,y=p90.tolist(),name="p90",line_color="rgba(29,158,117,0.3)",showlegend=False),
            go.Scatter(x=x,y=p10.tolist(),name="p10–p90",fill="tonexty",
                       fillcolor="rgba(29,158,117,0.15)",line_color="rgba(29,158,117,0.3)"),
            go.Scatter(x=x,y=p50.tolist(),name="Median",line=dict(color="#1D9E75",width=2.5)),
        ])
        fig.update_layout(title="Portfolio value (monthly)",height=350,
                          xaxis_title="Month",yaxis_title="USD",
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("p50 Final",   f"${np.percentile(vp[:,-1],50):,.0f}")
        c2.metric("p10 Final",   f"${np.percentile(vp[:,-1],10):,.0f}")
        c3.metric("p90 Final",   f"${np.percentile(vp[:,-1],90):,.0f}")
        c4.metric("p50 Income/mo", f"${np.percentile(vp[:,-1],50)*0.0707/12:,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GENERATIONAL PLAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏛 Generational Plan":
    st.title("Generational Wealth Plan  --  30-Year Horizon")

    c1,c2 = st.columns(2)
    target = c1.number_input("Target monthly income (USD)", 100, 10000, 1000, 100)
    mo_dca = c2.number_input("Monthly DCA contribution (USD)", 0, 5000, 200, 50)

    if st.button("Run 30-year plan"):
        from engine.generational_planner import run_generational_plan, GenPlanConfig
        import riskfolio as rp

        with st.spinner("Running 5,000 paths x 360 months..."):
            prices  = load_prices(tickers)
            monthly = prices.resample("ME").last().pct_change().dropna().dropna(axis=1,how="all")
            port    = rp.Portfolio(returns=monthly)
            port.assets_stats(method_mu="hist", method_cov="ledoit")
            w_opt   = port.optimization(model="Classic",rm="MV",obj="Sharpe",rf=rf_ann/12,l=2,hist=True)
            w = w_opt["weights"].values if w_opt is not None else np.ones(len(tickers))/len(tickers)
            port_val = sum(holdings[t]["shares"]*float(prices[t].iloc[-1])
                           for t in tickers if t in prices.columns)
            plan = GenPlanConfig(n_paths=5000, horizon_years=30,
                                  monthly_add_usd=float(mo_dca),
                                  target_income_m=float(target),
                                  initial_value=port_val)
            result = run_generational_plan(monthly, w, cfg, plan)

        st.subheader("Milestone Summary")
        ms_rows = []
        for yr, ms in result["milestones"].items():
            ms_rows.append({"Year":yr,
                             "p10 Value":f"${ms['p10_value']:,.0f}",
                             "p50 Value":f"${ms['p50_value']:,.0f}",
                             "p90 Value":f"${ms['p90_value']:,.0f}",
                             "Real p50":f"${ms['p50_real']:,.0f}",
                             "Income p50/mo":f"${ms['p50_income_m']:,.2f}",
                             f"P(>${target:,.0f}/mo)":f"{ms['prob_above_target']:.1f}%",
                             "Cum Dividends":f"${ms['p50_cum_div']:,.0f}"})
        st.dataframe(pd.DataFrame(ms_rows), use_container_width=True, hide_index=True)

        mtt = result.get("months_to_target",{})
        if mtt.get("years"):
            st.success(f"Median time to ${target:,}/mo: {mtt['years']} years {mtt['extra_months']} months")
        else:
            st.warning(f"Target ${target:,}/mo not reached within 30 years at current trajectory.")

        if result.get("chart_bytes"):
            st.image(result["chart_bytes"], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WHT RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 WHT Reconciliation":
    st.title("Withholding Tax Reconciliation")
    st.caption("Determines whether KS applies 30% (default) or 15% (W-8BEN treaty) by back-calculating from your KS amounts.")

    try:
        from engine.wht_reconciliation import build_reconciliation, summarise_wht
        records = build_reconciliation(cfg, fx_r)
        summary = summarise_wht(records)
        verdict_map = {"treaty_15":"success","default_30":"error","partial":"warning",
                       "no_data":"info","overpaid":"error"}
        primary = max(set([r.verdict for r in records]),
                      key=[r.verdict for r in records].count) if records else "no_data"
        getattr(st, verdict_map.get(primary,"info"))(summary)

        rows = [{"Period":r.period,"Ticker":r.ticker,"Shares":r.shares,
                 "Gross $":f"${r.gross_usd:.4f}","Net @30%":f"${r.net_30pct:.4f}",
                 "Net @15%":f"${r.net_15pct:.4f}",
                 "KS THB":r.ks_thb or "not recorded",
                 "KS USD":f"${r.ks_usd_est:.4f}" if r.ks_usd_est else "n/a",
                 "Implied WHT":f"{r.implied_wht*100:.1f}%" if r.implied_wht else "n/a",
                 "Verdict":r.verdict} for r in records]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Record actual KS amount")
        with st.form("wht_form"):
            wc1,wc2 = st.columns(2)
            w_period = wc1.text_input("Period (e.g. 2026-04)")
            w_ticker = wc2.selectbox("Ticker", list(tickers))
            w_thb    = st.number_input("KS THB received", min_value=0.0, format="%.2f")
            if st.form_submit_button("Preview updated YAML"):
                cfg_new = dict(cfg)
                for div in cfg_new.get("dividends_received",[]):
                    if div.get("period")==w_period and div.get("ticker")==w_ticker:
                        div["thb_ks_app"] = float(w_thb); div["status"] = "received"
                yaml_str = yaml.dump(cfg_new, allow_unicode=True, sort_keys=False)
                st.download_button("Download updated portfolio.yaml",
                                   yaml_str.encode("utf-8"), "portfolio.yaml", "text/yaml")
    except Exception as e:
        st.error(f"WHT error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏱ Backtest":
    st.title("Walk-Forward Backtest")
    train_w = st.slider("Training window (months)", 12, 36, 24)

    if st.button("Run backtest"):
        from engine.backtest import run_walkforward
        with st.spinner("Running walk-forward..."):
            prices  = load_prices(tickers)
            monthly = prices.resample("ME").last().pct_change().dropna().dropna(axis=1,how="all")
            bt      = run_walkforward(monthly, train_months=train_w)

        if not bt:
            st.warning("Not enough data.")
        else:
            rows = [{"Strategy":s.replace("_"," ").title(),
                     "Ann. Return":f"{m['ann_return']*100:.2f}%",
                     "Ann. Vol":f"{m['ann_vol']*100:.2f}%",
                     "Sharpe":f"{m['sharpe']:.3f}",
                     "Max DD":f"{m['max_drawdown']*100:.2f}%",
                     "Final $1":f"${m['final_equity']:.3f}"}
                    for s, m in bt["metrics"].items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if bt.get("equity_chart"):
                st.image(bt["equity_chart"], use_container_width=True)
            if bt.get("sharpe_chart"):
                st.image(bt["sharpe_chart"], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DOWNLOAD REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Download Report":
    st.title("Download Full Excel Report")
    st.info("Generates a 10-sheet report with all analytics, charts, and projections.")

    if st.button("Generate report (takes 30-60s)"):
        from engine.report_builder import build_report
        from engine.analytics import compute_risk_table, monte_carlo
        from engine.wht_reconciliation import build_reconciliation
        from engine.generational_planner import run_generational_plan, GenPlanConfig
        from engine.backtest import run_walkforward
        from engine.fx_timing import compute_fx_signal
        from engine.dividend_calendar import build_events
        import riskfolio as rp

        with st.spinner("Running full analysis..."):
            # Market data
            prices  = load_prices(tickers)
            monthly = prices.resample("ME").last().pct_change().dropna().dropna(axis=1,how="all")
            rt      = _returns_tuple(monthly)
            rf_m    = rf_ann / 12

            # Optimisation
            w_dict_raw, frontier_dict, risk_dict = run_riskfolio(rt, rf_ann)
            risk_tbl = pd.DataFrame(risk_dict).T

            # P&L
            from collections import defaultdict
            pnl_rows = []
            for tkr, h in holdings.items():
                curr = float(prices[tkr].iloc[-1]) if tkr in prices.columns else 0
                mkt  = h["shares"]*curr; unr = mkt-h["total_cost"]; net = mkt-h["total_cost"]
                pct  = net/h["total_cost"] if h["total_cost"]>0 else 0
                pnl_rows.append({"Ticker":tkr,"Shares":h["shares"],"Avg Cost $":h["avg_cost"],
                                  "Current $":round(curr,2),"Cost Basis $":h["total_cost"],
                                  "Market Value $":round(mkt,2),"Unrealised $":round(unr,2),
                                  "Unrealised THB":round(unr*fx_r,0),"Net P&L $":round(net,2),"Net P&L %":round(pct,4)})
            pnl_df = pd.DataFrame(pnl_rows).set_index("Ticker")

            # Dividends
            div_rows = []
            for d in cfg.get("dividends_received",[]):
                sh=d.get("shares_eligible",0); pps=d.get("amount_per_share_usd",0.0)
                gross=sh*pps; net=gross*(1-wht)
                div_rows.append({"Period":d.get("period",""),"Ticker":d.get("ticker",""),
                                  "Ex-date":d.get("ex_date",""),"Pay-date":d.get("pay_date",""),
                                  "Shares":sh,"$/share":pps,"Gross $":round(gross,2),
                                  "WHT":f"{wht*100:.0f}%","Net $":round(net,2),
                                  "Net THB est":round(net*fx_r,0),
                                  "KS App THB":d.get("thb_ks_app","n/a"),
                                  "Status":d.get("status","received")})

            # WHT
            wht_records = build_reconciliation(cfg, fx_r)

            # Backtest
            bt = run_walkforward(monthly, train_months=24)
            bt_metrics  = bt.get("metrics",{}) if bt else {}
            bt_eq_bytes = bt.get("equity_chart", b"") if bt else b""

            # Generational
            w_main = np.array(list(list(w_dict_raw.values())[0].values())) if w_dict_raw else np.ones(len(tickers))/len(tickers)
            port_val = sum(holdings[t]["shares"]*float(prices[t].iloc[-1]) for t in tickers if t in prices.columns)
            plan = GenPlanConfig(n_paths=3000, horizon_years=30, monthly_add_usd=200,
                                  target_income_m=cfg.get("analysis",{}).get("target_monthly_income_usd",1000),
                                  initial_value=port_val)
            gen  = run_generational_plan(monthly, w_main, cfg, plan)

            # FX
            fx_hist   = load_fx()
            fx_signal = compute_fx_signal(fx_hist)

            # Calendar
            cal_events = build_events(cfg, holdings)

            # Riskfolio charts (as bytes for Excel embedding)
            rpl_plots = _riskfolio_plots(monthly, w_dict_raw, frontier_dict, rf_ann)
            plots = {k: io.BytesIO(v) for k, v in rpl_plots.items()}
            if gen.get("chart_bytes"):
                plots["generational"] = io.BytesIO(gen["chart_bytes"])
            if bt_eq_bytes:
                plots["backtest_equity"] = io.BytesIO(bt_eq_bytes)

            conf = {"rf":rf_ann,"wht":wht,"fx":fx_r,
                    "cash_usd":cfg.get("cash",{}).get("usd",0),
                    "deposited":cfg.get("cash",{}).get("total_deposited_usd",0)}

            # Build report
            xlsx_bytes = build_report(
                pnl_df=pnl_df, risk_tbl=risk_tbl, w_dict=w_dict_raw,
                div_rows=div_rows, wht_records=wht_records,
                bt_metrics=bt_metrics, milestones=gen.get("milestones",{}),
                fx_signal=fx_signal, cal_events=cal_events,
                cfg=cfg, conf=conf, plots=plots, bl_applied=False,
            )

        fname = f"portfolio_report_{datetime.today().strftime('%Y%m%d')}.xlsx"
        st.download_button("Download Excel Report", xlsx_bytes, fname,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("Report ready!")
