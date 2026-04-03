"""
riskfolio_autonomous.py  --  Portfolio Analytics Engine (Full)
==============================================================
Integrates:
  - Riskfolio-Lib v7 (portfolio optimisation, all plots)
  - Julia bridge (nonlinear shrinkage covariance, fast Monte Carlo)
  - Black-Litterman views from config/views.yaml
  - Walk-forward backtest engine
  - Generational wealth planner (30-year Monte Carlo)
  - FX timing signal (USD/THB z-score)
  - DCA price alerts (Windows toast + email)
  - Dividend calendar (.ics)
  - WHT reconciliation
  - Full 12-sheet Excel report

SETUP
-----
    pip install riskfolio-lib yfinance openpyxl xlsxwriter pandas numpy
        pyyaml watchdog juliacall win10toast icalendar plotly streamlit

RUN
---
    python riskfolio_autonomous.py
    python watchdog_runner.py          # auto-trigger on YAML save
    streamlit run ui/app.py            # browser dashboard
"""

import os
import sys
import warnings
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

try:
    import yaml
except ImportError:
    sys.exit("pyyaml not installed. Run: pip install pyyaml")

# Portfolio optimisation: Riskfolio-Lib (Cajas, 2026) — github.com/dcajasn/Riskfolio-Lib
# Cite: @misc{riskfolio, author={Dany Cajas}, title={Riskfolio-Lib (7.2.1)}, year={2026}}
try:
    import riskfolio as rp
except ImportError:
    sys.exit("riskfolio-lib not installed. Run: pip install riskfolio-lib")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
CONFIG_FILE = ROOT / "config" / "portfolio.yaml"
OUTPUT_DIR  = ROOT / "output"
PLOT_DIR    = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

TODAY       = datetime.today().strftime("%Y%m%d")
REPORT_PATH = OUTPUT_DIR / f"portfolio_report_{TODAY}.xlsx"


# ══════════════════════════════════════════════════════════════════════════════
# 1. YAML LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_yaml() -> dict:
    if not CONFIG_FILE.exists():
        log.error("portfolio.yaml not found: %s", CONFIG_FILE)
        sys.exit(1)
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log.info("Loaded portfolio.yaml  (as_of: %s)", cfg.get("meta",{}).get("data_as_of",""))
    return cfg


def derive_holdings(cfg: dict) -> dict:
    shares = defaultdict(int); cost = defaultdict(float); comm = defaultdict(float)
    for tx in cfg.get("transactions", []):
        if tx["type"] == "BUY":
            t = tx["ticker"]
            shares[t] += tx["shares"]; cost[t] += tx["total_usd"]; comm[t] += tx["commission_usd"]
        elif tx["type"] == "SELL":
            shares[tx["ticker"]] -= tx["shares"]
    return {t: {"shares": s, "avg_cost": cost[t]/s, "total_cost": round(cost[t],2), "total_comm": round(comm[t],2)}
            for t, s in shares.items() if s > 0}


def build_conf(cfg: dict) -> dict:
    s = cfg.get("settings", {}); m = cfg.get("meta", {})
    return {
        "rf":        s.get("risk_free_rate_annual", 0.045),
        "wht":       s.get("wht_active", 0.30),
        "fx":        m.get("fx_usd_thb", 32.68),
        "start":     s.get("data_history_start", "2022-01-01"),
        "end":       datetime.today().strftime("%Y-%m-%d"),
        "cash_usd":  cfg.get("cash",{}).get("usd", 0.0),
        "deposited": cfg.get("cash",{}).get("total_deposited_usd", 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════
def download_data(tickers: list, start: str, end: str):
    log.info("Downloading %s ...", tickers)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices  = prices.ffill().dropna(how="all")
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna().dropna(axis=1, how="all")
    log.info("  %d months x %d assets loaded.", len(returns), len(returns.columns))
    return monthly, returns


# ══════════════════════════════════════════════════════════════════════════════
# 3. PORTFOLIO OBJECT  (Julia nonlinear shrinkage cov injected here)
# ══════════════════════════════════════════════════════════════════════════════
def build_portfolio(returns: pd.DataFrame) -> rp.Portfolio:
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit")   # baseline

    # Upgrade covariance matrix with Julia Analytical Nonlinear Shrinkage
    try:
        from engine.julia_bridge import julia_cov
        jcov = julia_cov(returns)
        if jcov is not None:
            port.cov = pd.DataFrame(jcov, index=returns.columns, columns=returns.columns)
            log.info("  [Julia] Nonlinear shrinkage cov injected into port.cov")
    except Exception as e:
        log.warning("  Julia cov unavailable (%s) -- using sklearn Ledoit-Wolf", e)

    port.sht = False; port.upperlng = 1.0
    return port


# ══════════════════════════════════════════════════════════════════════════════
# 4. BLACK-LITTERMAN
# ══════════════════════════════════════════════════════════════════════════════
def apply_bl(port: rp.Portfolio, cfg: dict) -> bool:
    try:
        from engine.black_litterman import run_black_litterman
        applied = run_black_litterman(port, cfg)
        if applied:
            log.info("  Black-Litterman mu applied")
        return applied
    except Exception as e:
        log.warning("  BL failed (%s) -- historical mu retained", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPTIMISATIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_optimisations(port: rp.Portfolio, rf_m: float) -> dict:
    results = {}
    strategies = [
        ("Max Sharpe (MV)",      "Classic", "MV",   "Sharpe"),
        ("Min Variance",         "Classic", "MV",   "MinRisk"),
        ("Max Sharpe (CVaR)",    "Classic", "CVaR", "Sharpe"),
        ("Min CVaR",             "Classic", "CVaR", "MinRisk"),
        ("Min Max Drawdown",     "Classic", "MDD",  "MinRisk"),
        ("Max Sharpe (Sortino)", "Classic", "SLPM", "Sharpe"),
    ]
    log.info("Running optimisations ...")
    for label, model, rm, obj in strategies:
        try:
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf_m, l=2, hist=True)
            if w is not None and not w.isnull().values.any():
                results[label] = w; log.info("  [OK]   %s", label)
        except Exception as exc:
            log.warning("  [FAIL] %s -- %s", label, exc)

    n = len(port.returns.columns)
    results["Equal Weight"] = pd.DataFrame([1/n]*n, index=port.returns.columns, columns=["weights"])
    log.info("  [OK]   Equal Weight")

    try:
        hport = rp.HCPortfolio(returns=port.returns)
        try:    w_hrp = hport.optimization(model="HRP", codependence="pearson", rm="MV", rf=rf_m, linkage="ward", leaf_order=True)
        except TypeError: w_hrp = hport.optimization(model="HRP", codependence="pearson", rm="MV", rf=rf_m, linkage="ward", max_k=10, leaf_order=True)
        if w_hrp is not None:
            results["HRP (Risk Parity)"] = w_hrp; log.info("  [OK]   HRP")
    except Exception as e:
        log.warning("  [FAIL] HRP -- %s", e)

    log.info("  %d strategies ready.", len(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════════════════════
def build_frontier(port: rp.Portfolio, rf_m: float) -> pd.DataFrame:
    if len(port.returns.columns) < 3:
        log.warning("  Frontier skipped: need 3+ assets, have %d", len(port.returns.columns))
        return pd.DataFrame()
    try:
        f = port.efficient_frontier(model="Classic", rm="MV", points=50, rf=rf_m, hist=True)
        log.info("  %d frontier portfolios.", f.shape[1]); return f
    except Exception as e:
        log.warning("  Frontier failed: %s", e); return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 7. RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_risk_table(returns: pd.DataFrame, rf: float) -> pd.DataFrame:
    rows = []
    for t in returns.columns:
        r = returns[t].dropna()
        if len(r) < 6: continue
        ann = r.mean()*12; vol = r.std()*np.sqrt(12)
        sr  = (r.mean()-rf)/r.std()*np.sqrt(12) if r.std()>0 else np.nan
        neg = r[r<0]
        semi = neg.std()*np.sqrt(12) if len(neg)>1 else np.nan
        sortino = ((r.mean()-rf)/neg.std()*np.sqrt(12) if len(neg)>1 and neg.std()>0 else np.nan)
        cum  = (1+r).cumprod(); peak = cum.cummax(); dd = (cum-peak)/peak
        mdd  = float(dd.min())
        v95  = float(np.percentile(r,5)); mask = r<=v95
        cv95 = float(r[mask].mean()) if mask.any() else v95
        g    = r[r>0].sum(); l = abs(r[r<0].sum())
        rows.append({"Ticker":t,"Ann. Return":ann,"Ann. Volatility":vol,
                     "Sharpe Ratio":sr,"Sortino Ratio":sortino,"Max Drawdown":mdd,
                     "Calmar Ratio":ann/abs(mdd) if mdd!=0 else np.nan,
                     "VaR 95%":v95,"CVaR 95%":cv95,"Omega Ratio":g/l if l>0 else np.inf,
                     "Semi-Volatility":semi})
    return pd.DataFrame(rows).set_index("Ticker")


# ══════════════════════════════════════════════════════════════════════════════
# 8. POSITION P&L
# ══════════════════════════════════════════════════════════════════════════════
def position_pnl(prices: pd.DataFrame, holdings: dict, fx: float) -> pd.DataFrame:
    rows = []
    for ticker, h in holdings.items():
        curr = float(prices[ticker].iloc[-1]) if ticker in prices.columns else 0.0
        sh, avg, tc = h["shares"], h["avg_cost"], h["total_cost"]
        mkt = sh*curr; unr = mkt-sh*avg; net = mkt-tc
        rows.append({"Ticker":ticker,"Shares":sh,"Avg Cost $":avg,"Current $":round(curr,2),
                     "Cost Basis $":round(tc,2),"Market Value $":round(mkt,2),
                     "Unrealised $":round(unr,2),"Unrealised THB":round(unr*fx,0),
                     "Net P&L $":round(net,2),"Net P&L %":round(net/tc if tc>0 else 0,4)})
    return pd.DataFrame(rows).set_index("Ticker")


# ══════════════════════════════════════════════════════════════════════════════
# 9. RISKFOLIO PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, name):
    p = str(PLOT_DIR/f"{name}.png"); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); return p

def generate_plots(port: rp.Portfolio, w_dict: dict, frontier: pd.DataFrame, rf_m: float) -> dict:
    plots = {}; w_main = list(w_dict.values())[0]; log.info("Generating Riskfolio plots ...")


    def _try(label, fn):
        try: plots[label] = fn(); log.info("  [OK] plot_%s", label)
        except Exception as e: log.warning("  plot_%s: %s", label, e)

    def _pie():
        f,ax = plt.subplots(figsize=(8,6)); rp.plot_pie(w=w_main,title="Max Sharpe Allocation",others=0.05,nrow=25,cmap="tab20",ax=ax); return _save(f,"pie")

    def _frontier():
        if frontier.empty or len(port.returns.columns)<3: raise ValueError("Need 3+ assets")
        f,ax = plt.subplots(figsize=(10,7)); rp.plot_frontier(w_frontier=frontier,mu=port.mu,cov=port.cov,returns=port.returns,rm="MV",rf=rf_m,alpha=0.05,cmap="viridis",w=w_main,label="Max Sharpe",ax=ax); return _save(f,"frontier")

    def _risk_con():
        f,ax = plt.subplots(figsize=(8,5)); rp.plot_risk_con(w=w_main,cov=port.cov,returns=port.returns,rm="MV",rf=rf_m,alpha=0.05,color="tab:blue",height=6,width=10,ax=ax); return _save(f,"risk_con")

    def _hist():
        f,ax = plt.subplots(figsize=(9,5)); rp.plot_hist(returns=port.returns,w=w_main,alpha=0.05,bins=50,height=6,width=10,ax=ax); return _save(f,"hist")

    def _drawdown():
        f, axes = plt.subplots(2,1,figsize=(10,8))
        rp.plot_drawdown(w=w_main,returns=port.returns,alpha=0.05,height=8,width=10,ax=list(axes))
        return _save(f,"drawdown")

    def _clusters():
        if len(port.returns.columns)<3: raise ValueError(f"plot_clusters needs 3+ assets, got {len(port.returns.columns)} -- skipping")
        f,ax = plt.subplots(figsize=(9,5)); rp.plot_clusters(returns=port.returns,codependence="pearson",linkage="ward",k=None,max_k=10,leaf_order=True,ax=ax); return _save(f,"clusters")

    for lbl, fn in [("pie",_pie),("frontier",_frontier),("risk_con",_risk_con),("hist",_hist),("drawdown",_drawdown),("clusters",_clusters)]:
        _try(lbl, fn)

    log.info("  %d / 6 Riskfolio plots saved.", len(plots))
    return plots


# ══════════════════════════════════════════════════════════════════════════════
# 10. DIVIDEND PROJECTION
# ══════════════════════════════════════════════════════════════════════════════
def build_dividend_projection(cfg: dict, holdings: dict, wht: float, fx: float) -> list:
    rows = []
    for d in cfg.get("dividends_received", []):
        ticker = d.get("ticker",""); sh = d.get("shares_eligible",0)
        gross  = sh * d.get("amount_per_share_usd",0.0); net = gross*(1-wht)
        rows.append({"Period":d.get("period",""),"Ticker":ticker,"Ex-date":d.get("ex_date",""),
                     "Pay-date":d.get("pay_date",""),"Shares":sh,"$/share":d.get("amount_per_share_usd",0.0),
                     "Gross $":round(gross,2),"WHT":f"{wht*100:.0f}%","Net $":round(net,2),
                     "Net THB est":round(net*fx,0),"KS App THB":d.get("thb_ks_app","n/a"),
                     "Status":d.get("status","received")})
    for ticker, inst in cfg.get("instruments",{}).items():
        sh = holdings.get(ticker,{}).get("shares",0)
        if sh==0: continue
        for u in inst.get("dividend_policy",{}).get("estimated_upcoming",[]):
            eligible = u.get("eligible_for_our_shares",True)
            amt      = u.get("amount",u.get("amount_per_share_usd",0.0))
            gross    = sh*amt if eligible else 0.0; net = gross*(1-wht)
            rows.append({"Period":u.get("period",u.get("ex","")),"Ticker":ticker,
                         "Ex-date":u.get("ex",""),"Pay-date":u.get("pay",""),
                         "Shares":sh if eligible else 0,"$/share":amt,
                         "Gross $":round(gross,2),"WHT":f"{wht*100:.0f}%","Net $":round(net,2),
                         "Net THB est":round(net*fx,0),"KS App THB":"n/a",
                         "Status":"projected" if eligible else "MISSED"})
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 11. EXCEL REPORT  (12 sheets)
# ══════════════════════════════════════════════════════════════════════════════
def write_excel_report(prices, returns, risk_tbl, w_dict, pnl_df, frontier,
                       plots, div_rows, cfg, holdings, conf,
                       bt_result, gen_result, fx_result, wht_records,
                       cal_events, alerts_fired, bl_applied):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage

    wb = Workbook(); wb.remove(wb.active)

    def fill(h):    return PatternFill("solid", fgColor=h)
    def fnt(bold=False, color="1A1A2E", size=10): return Font(bold=bold, color=color, size=size, name="Arial")
    def aln(h="center"): return Alignment(horizontal=h, vertical="center")
    def bdr():
        s = Side(style="thin", color="CCCCDD"); return Border(left=s, right=s, top=s, bottom=s)
    def cw(ws, col, w): ws.column_dimensions[get_column_letter(col)].width = w
    def rh(ws, row, h): ws.row_dimensions[row].height = h

    def wcell(ws, r, c, v, bold=False, color="1A1A2E", bg=None, fmt=None, size=10):
        x = ws.cell(r, c, v); x.font = fnt(bold, color, size); x.alignment = aln(); x.border = bdr()
        if bg:  x.fill          = fill(bg)
        if fmt: x.number_format = fmt
        return x

    def titrow(ws, row, text, ncols=9, bg="1A1A2E", h=30):
        ws.merge_cells(f"A{row}:{get_column_letter(ncols)}{row}")
        x = ws[f"A{row}"]; x.value = text; x.font = Font(bold=True, color="FFFFFF", size=13, name="Arial")
        x.fill = fill(bg); x.alignment = aln(); rh(ws, row, h)

    def hdrrow(ws, row, labels, bg="0F3460", h=22):
        rh(ws, row, h)
        for c, lbl in enumerate(labels, 1):
            x = ws.cell(row, c, lbl); x.font = Font(bold=True, color="FFFFFF", size=10, name="Arial")
            x.fill = fill(bg); x.alignment = aln(); x.border = bdr()

    def addimg(ws, path, anchor, w=650, h=380):
        if path and Path(path).exists():
            im = XLImage(path); im.width = w; im.height = h; ws.add_image(im, anchor)

    BG1="F2F4F8"; BG2="FFFFFF"; G="1D9E75"; A="BA7517"; R="A32D2D"

    # ── Sheet 1: Dashboard ───────────────────────────────────────────────────
    ws = wb.create_sheet("Dashboard"); ws.sheet_view.showGridLines=False; ws.sheet_properties.tabColor=G
    for i,w in enumerate([10,18,14,14,14,14,14,14,14],1): cw(ws,i,w)
    meta = cfg.get("meta",{})
    titrow(ws,1,f"PORTFOLIO ANALYTICS  |  {meta.get('account_holder','')}  |  "
               f"Account {meta.get('account_id','')}  |  {datetime.today().strftime('%Y-%m-%d')}")
    snap = cfg.get("ks_app_snapshot_20260327",{})
    kv = [
        ("Market Value (THB)", f"฿{snap.get('market_value_thb',0):,.2f}"),
        ("Total Cost (THB)",   f"฿{snap.get('total_cost_thb',0):,.2f}"),
        ("Unrealised (THB)",   f"฿{snap.get('unrealized_thb',0):,.2f}"),
        ("Unrealised %",       f"{snap.get('unrealized_pct',0)*100:.2f}%"),
        ("Dividends (THB)",    f"฿{snap.get('total_dividends_thb',0):,.2f}"),
        ("Cash (USD)",         f"${conf['cash_usd']:.2f}"),
        ("FX Rate",            f"{conf['fx']:.4f} THB/USD"),
        ("WHT Rate",           f"{conf['wht']*100:.0f}% {'(treaty)' if conf['wht']<0.20 else '(default)'}"),
        ("BL Views Applied",   "YES" if bl_applied else "NO (historical mu)"),
        ("Julia Engine",       "ACTIVE" if _julia_active() else "FALLBACK (Python)"),
    ]
    for i,(k,v) in enumerate(kv):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws,r,19)
        wcell(ws,r,1,k,bold=True,bg=bg)
        ws.merge_cells(f"B{r}:I{r}")
        x=ws.cell(r,2,v); x.font=fnt(bold=True,color=(G if any(c in v for c in ["฿","$","YES","ACTIVE"]) else R if "-" in v else "1A1A2E")); x.fill=fill(bg); x.alignment=aln(); x.border=bdr()

    if alerts_fired:
        rh(ws,13+len(kv),28); ws.merge_cells(f"A{13+len(kv)}:I{13+len(kv)}")
        alert_txt = "  |  ".join([f"{a['ticker']} ${a['price']:.2f} → {a['zone']}" for a in alerts_fired])
        x=ws.cell(13+len(kv),1,f"DCA ALERTS: {alert_txt}"); x.fill=fill("FFF3CD"); x.font=Font(bold=True,color="633806",size=10,name="Arial"); x.alignment=aln(); x.border=bdr()

    # ── Sheet 2: P&L ─────────────────────────────────────────────────────────
    ws2=wb.create_sheet("Position P&L"); ws2.sheet_view.showGridLines=False; ws2.sheet_properties.tabColor=G
    for i,w in enumerate([6,14,10,14,14,16,16,14,14,12],1): cw(ws2,i,w)
    titrow(ws2,1,f"POSITION P&L  --  Computed from transaction ledger  --  {datetime.today().strftime('%Y-%m-%d')}")
    hdrrow(ws2,2,["#","Ticker","Shares","Avg Cost $","Current $","Cost Basis $","Market Value $","Unrealised $","Unrealised THB","Net P&L %"])
    for i,(tkr,row) in enumerate(pnl_df.iterrows()):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws2,r,19); pc=G if row["Net P&L $"]>=0 else R
        wcell(ws2,r,1,i+1,bg=bg); wcell(ws2,r,2,tkr,bold=True,bg=bg,color="0F3460")
        wcell(ws2,r,3,row["Shares"],bg=bg,fmt="#,##0"); wcell(ws2,r,4,row["Avg Cost $"],bg=bg,fmt="$#,##0.00")
        wcell(ws2,r,5,row["Current $"],bg=bg,fmt="$#,##0.00"); wcell(ws2,r,6,row["Cost Basis $"],bg=bg,fmt="$#,##0.00")
        wcell(ws2,r,7,row["Market Value $"],bg=bg,fmt="$#,##0.00")
        wcell(ws2,r,8,row["Unrealised $"],bg=bg,color=pc,fmt="$#,##0.00")
        wcell(ws2,r,9,row["Unrealised THB"],bg=bg,color=pc,fmt="#,##0")
        wcell(ws2,r,10,row["Net P&L %"],bg=bg,color=pc,fmt="0.00%")

    # ── Sheet 3: Transactions ────────────────────────────────────────────────
    ws3=wb.create_sheet("Transactions"); ws3.sheet_view.showGridLines=False; ws3.sheet_properties.tabColor="534AB7"
    for i,w in enumerate([6,12,8,10,10,10,12,12,12,12,28],1): cw(ws3,i,w)
    titrow(ws3,1,"TRANSACTION LEDGER  --  Source: portfolio.yaml",ncols=11)
    hdrrow(ws3,2,["ID","Date","Type","Ticker","Exchange","Shares","Price $","Gross $","Commission $","Total $","Note"])
    for i,tx in enumerate(cfg.get("transactions",[])):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws3,r,19); tc=G if tx["type"]=="BUY" else R
        wcell(ws3,r,1,tx.get("id",""),bg=bg); wcell(ws3,r,2,tx.get("date",""),bg=bg)
        wcell(ws3,r,3,tx.get("type",""),bold=True,bg=bg,color=tc)
        wcell(ws3,r,4,tx.get("ticker",""),bold=True,bg=bg,color="0F3460")
        wcell(ws3,r,5,tx.get("exchange",""),bg=bg); wcell(ws3,r,6,tx.get("shares",0),bg=bg,fmt="#,##0")
        wcell(ws3,r,7,tx.get("price_usd",0),bg=bg,fmt="$#,##0.00"); wcell(ws3,r,8,tx.get("gross_usd",0),bg=bg,fmt="$#,##0.00")
        wcell(ws3,r,9,tx.get("commission_usd",0),bg=bg,color=R,fmt="$#,##0.00")
        wcell(ws3,r,10,tx.get("total_usd",0),bold=True,bg=bg,fmt="$#,##0.00"); wcell(ws3,r,11,tx.get("note",""),bg=bg)

    # ── Sheet 4: Dividends ───────────────────────────────────────────────────
    ws4=wb.create_sheet("Dividend Tracker"); ws4.sheet_view.showGridLines=False; ws4.sheet_properties.tabColor=G
    for i,w in enumerate([12,8,12,12,8,10,10,8,10,12,12,14],1): cw(ws4,i,w)
    titrow(ws4,1,"DIVIDEND TRACKER  --  History + Projected  (WHT applied)",ncols=12)
    hdrrow(ws4,2,["Period","Ticker","Ex-date","Pay-date","Shares","$/share","Gross $","WHT","Net $","Net THB est","KS App THB","Status"])
    sc_map={"received":G,"pending":A,"projected":"3B8BD4","MISSED":R}
    for i,d in enumerate(div_rows):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws4,r,19); sc=sc_map.get(d.get("Status",""),"1A1A2E")
        cols=["Period","Ticker","Ex-date","Pay-date","Shares","$/share","Gross $","WHT","Net $","Net THB est","KS App THB","Status"]
        fmts=[None,None,None,None,"#,##0","$#,##0.000","$#,##0.00",None,"$#,##0.00","#,##0",None,None]
        for j,(col,fmt) in enumerate(zip(cols,fmts)):
            xc=wcell(ws4,r,j+1,d.get(col,""),bg=bg,fmt=fmt,color=(sc if col in ["Status","Ticker"] else "1A1A2E"))
            if col=="Status" and d.get(col): xc.font=Font(bold=True,color="FFFFFF",size=10,name="Arial"); xc.fill=fill(sc)

    # ── Sheet 5: WHT Reconciliation ──────────────────────────────────────────
    ws5=wb.create_sheet("WHT Reconciliation"); ws5.sheet_view.showGridLines=False; ws5.sheet_properties.tabColor=A
    for i,w in enumerate([10,8,8,12,12,12,12,12,14,14,20],1): cw(ws5,i,w)
    titrow(ws5,1,"WITHHOLDING TAX RECONCILIATION  --  Verifies 30% vs 15% treaty rate",ncols=11)
    hdrrow(ws5,2,["Period","Ticker","Shares","Gross $","Net @30%","Net @15%","KS THB","KS USD est","Implied WHT %","Verdict","Note"])
    for i,rec in enumerate(wht_records):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws5,r,19)
        vc={"treaty_15":G,"default_30":R,"partial":A,"overpaid":R,"no_data":"888888"}.get(rec.verdict,A)
        vals=[(1,rec.period,None),(2,rec.ticker,None),(3,rec.shares,"#,##0"),
              (4,rec.gross_usd,"$#,##0.0000"),(5,rec.net_30pct,"$#,##0.0000"),(6,rec.net_15pct,"$#,##0.0000"),
              (7,rec.ks_thb if rec.ks_thb else "missing",None),(8,rec.ks_usd_est if rec.ks_usd_est else "n/a",None),
              (9,f"{rec.implied_wht*100:.1f}%" if rec.implied_wht else "n/a",None),
              (10,rec.verdict,None),(11,rec.note,None)]
        for c,v,fmt in vals:
            xc=wcell(ws5,r,c,v,bg=bg,fmt=fmt,color=(vc if c in [9,10] else "1A1A2E"))
            if c==10: xc.font=Font(bold=True,color="FFFFFF",size=10,name="Arial"); xc.fill=fill(vc)

    # ── Sheet 6: Risk Metrics ────────────────────────────────────────────────
    ws6=wb.create_sheet("Risk Metrics"); ws6.sheet_view.showGridLines=False; ws6.sheet_properties.tabColor="3B8BD4"
    nc6=len(risk_tbl.index)+1
    for i in range(1,nc6+1): cw(ws6,i,20)
    titrow(ws6,1,"INDIVIDUAL ASSET RISK METRICS  (Annualised, monthly total returns)",ncols=nc6)
    hdrrow(ws6,2,["Metric"]+list(risk_tbl.index))
    fmt_map={"Ann. Return":"0.00%","Ann. Volatility":"0.00%","Sharpe Ratio":"0.000","Sortino Ratio":"0.000","Max Drawdown":"0.00%","Calmar Ratio":"0.000","VaR 95%":"0.00%","CVaR 95%":"0.00%","Omega Ratio":"0.000","Semi-Volatility":"0.00%"}
    lower_better={"Ann. Volatility","Max Drawdown","CVaR 95%","VaR 95%","Semi-Volatility"}
    for i,metric in enumerate(risk_tbl.columns):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws6,r,19); wcell(ws6,r,1,metric,bold=True,bg=bg)
        vals=risk_tbl[metric].dropna()
        best=vals.idxmin() if metric in lower_better else (vals.idxmax() if not vals.empty else None)
        for j,tkr in enumerate(risk_tbl.index):
            v=risk_tbl.loc[tkr,metric]; ib=tkr==best and pd.notna(v)
            wcell(ws6,r,2+j,v,bg=f"{G}33" if ib else bg,color=G if ib else "1A1A2E",fmt=fmt_map.get(metric,"0.000"))

    # ── Sheet 7: Optimal Weights ─────────────────────────────────────────────
    ws7=wb.create_sheet("Optimal Weights"); ws7.sheet_view.showGridLines=False; ws7.sheet_properties.tabColor="7F77DD"
    nc7=len(w_dict)+1; cw(ws7,1,22)
    for i in range(2,nc7+1): cw(ws7,i,22)
    titrow(ws7,1,"OPTIMAL PORTFOLIO WEIGHTS  --  All Strategies",ncols=nc7)
    hdrrow(ws7,2,["Asset"]+list(w_dict.keys()))
    for i,tkr in enumerate(returns.columns):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws7,r,19); wcell(ws7,r,1,tkr,bold=True,bg=bg)
        for j,wdf in enumerate(w_dict.values()):
            v=float(wdf.loc[tkr,"weights"]) if tkr in wdf.index else 0.0
            wcell(ws7,r,2+j,v,bg=bg,color=G if v>0.35 else "1A1A2E",fmt="0.0%")

    # ── Sheet 8: Riskfolio Charts ────────────────────────────────────────────
    ws8=wb.create_sheet("Riskfolio Charts"); ws8.sheet_view.showGridLines=False; ws8.sheet_properties.tabColor="0F3460"
    titrow(ws8,1,"RISKFOLIO-LIB VISUAL ANALYTICS")
    for key,anchor,pw,ph in [("pie","A3",600,400),("frontier","A26",700,460),("risk_con","A54",640,380),("hist","A77",680,380),("drawdown","A100",700,480),("clusters","A128",680,380)]:
        addimg(ws8,plots.get(key,""),anchor,pw,ph)

    # ── Sheet 9: Backtest ────────────────────────────────────────────────────
    ws9=wb.create_sheet("Backtest"); ws9.sheet_view.showGridLines=False; ws9.sheet_properties.tabColor="534AB7"
    titrow(ws9,1,"WALK-FORWARD BACKTEST RESULTS")
    if bt_result and bt_result.get("metrics"):
        hdrrow(ws9,2,["Strategy","Ann. Return","Ann. Vol","Sharpe","Max Drawdown","Calmar","Final $1"])
        for i,(s,m) in enumerate(bt_result["metrics"].items()):
            r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws9,r,19)
            wcell(ws9,r,1,s.replace("_"," ").title(),bold=True,bg=bg)
            for j,(val,fmt) in enumerate([(m["ann_return"],"0.00%"),(m["ann_vol"],"0.00%"),(m["sharpe"],"0.000"),(m["max_drawdown"],"0.00%"),(m.get("calmar",np.nan),"0.000"),(m["final_equity"],"$0.000")],1):
                wcell(ws9,r,j+1,val if not np.isnan(float(val)) else "-",bg=bg,fmt=fmt)
        addimg(ws9,bt_result.get("equity_chart",""),"A15",700,380)
        addimg(ws9,bt_result.get("sharpe_chart",""),"A37",700,300)
    else:
        ws9.cell(3,1,"Backtest requires 3+ assets for meaningful results. Add PFLT or similar to portfolio.yaml.")

    # ── Sheet 10: Generational Plan ──────────────────────────────────────────
    ws10=wb.create_sheet("Generational Plan"); ws10.sheet_view.showGridLines=False; ws10.sheet_properties.tabColor=G
    titrow(ws10,1,"GENERATIONAL WEALTH PLAN  --  30-Year Monte Carlo")
    if gen_result and gen_result.get("milestones"):
        plan=gen_result["plan"]
        hdrrow(ws10,2,["Year","p10 Value","p50 Value","p90 Value","p50 Real (infl-adj)","p50 Income/mo","p10 Income/mo","p90 Income/mo","P(>target)","Cum Div p50"])
        for i,(yr,ms) in enumerate(gen_result["milestones"].items()):
            r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws10,r,19)
            wcell(ws10,r,1,f"Year {yr}",bold=True,bg=bg)
            for j,(v,fmt) in enumerate([(ms["p10_value"],"$#,##0"),(ms["p50_value"],"$#,##0"),(ms["p90_value"],"$#,##0"),(ms["p50_real"],"$#,##0"),(ms["p50_income_m"],"$#,##0.00"),(ms["p10_income_m"],"$#,##0.00"),(ms["p90_income_m"],"$#,##0.00"),(ms["prob_above_target"],"0.0\"%\""),(ms["p50_cum_div"],"$#,##0")],1):
                wcell(ws10,r,j+2,v,bg=bg,fmt=fmt,color=(G if j==1 else "1A1A2E"))
        mtt=gen_result.get("months_to_target",{})
        if mtt.get("years"):
            rh(ws10,3+len(gen_result["milestones"])+1,24)
            ws10.merge_cells(f"A{3+len(gen_result['milestones'])+1}:J{3+len(gen_result['milestones'])+1}")
            x=ws10.cell(3+len(gen_result["milestones"])+1,1,f"Median time to ${plan.target_income_m:,.0f}/mo target: {mtt['years']} years {mtt['extra_months']} months")
            x.font=Font(bold=True,color="085041",size=11,name="Arial"); x.fill=fill("E1F5EE"); x.alignment=aln(); x.border=bdr()
        addimg(ws10,gen_result.get("chart",""),"A18",700,480)
    else:
        ws10.cell(3,1,"Generational plan data not available.")

    # ── Sheet 11: FX Timing ──────────────────────────────────────────────────
    ws11=wb.create_sheet("FX Timing"); ws11.sheet_view.showGridLines=False; ws11.sheet_properties.tabColor=A
    titrow(ws11,1,"FX TIMING SIGNAL  --  USD/THB Conversion Optimiser")
    if fx_result:
        sig=fx_result.get("signal",{}); bud=fx_result.get("budget",{})
        kv=[("Signal",sig.get("signal","").replace("_"," ").title()),("Z-score",f"{sig.get('zscore',0):+.3f}"),("Current USD/THB",f"{sig.get('current',0):.4f}"),("90-day Mean",f"{sig.get('mean_90d',0):.4f}"),("52-week Low",f"{sig.get('52w_low',0):.4f}"),("52-week High",f"{sig.get('52w_high',0):.4f}"),("Advice",sig.get("advice","")),("DCA Budget (THB)",f"฿{bud.get('thb_in',0):,.0f}"),("USD Gross",f"${bud.get('usd_gross',0):,.2f}"),("USD Net (after comm)",f"${bud.get('usd_net',0):,.2f}"),("Est. BKLN shares",str(bud.get("shares_bkln",0)))]
        for i,(k,v) in enumerate(kv):
            r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws11,r,19)
            wcell(ws11,r,1,k,bold=True,bg=bg); ws11.merge_cells(f"B{r}:G{r}")
            xc=ws11.cell(r,2,v); xc.fill=fill(bg); xc.font=fnt(bold=True,color=("1D9E75" if sig.get("signal","") in ["buy","strong_buy"] else "A32D2D" if sig.get("signal","")=="avoid" else "1A1A2E")); xc.alignment=aln(); xc.border=bdr()
        addimg(ws11,fx_result.get("chart",""),"A17",700,380)

    # ── Sheet 12: Dividend Calendar ──────────────────────────────────────────
    ws12=wb.create_sheet("Dividend Calendar"); ws12.sheet_view.showGridLines=False; ws12.sheet_properties.tabColor=G
    for i,w in enumerate([14,10,12,12,40,8],1): cw(ws12,i,w)
    titrow(ws12,1,f"DIVIDEND CALENDAR  --  {len(cal_events)} events  --  Import portfolio_dividends.ics into Calendar app")
    hdrrow(ws12,2,["Event date","Ticker","Type","Period","Summary","Alarm (days)"])
    for i,ev in enumerate(cal_events):
        r=3+i; bg=BG1 if i%2==0 else BG2; rh(ws12,r,19)
        is_ex = "EX-DIV" in ev.get("summary","")
        ec = A if is_ex else G
        wcell(ws12,r,1,str(ev.get("date","")),bg=bg); wcell(ws12,r,2,ev.get("uid","").split("_")[1] if "_" in ev.get("uid","") else "",bg=bg,color="0F3460",bold=True)
        wcell(ws12,r,3,"Ex-date" if is_ex else "Pay-date",bg=bg,color=ec); wcell(ws12,r,4,ev.get("uid","").split("_")[2] if ev.get("uid","").count("_")>=2 else "",bg=bg)
        wcell(ws12,r,5,ev.get("summary",""),bg=bg); wcell(ws12,r,6,ev.get("alarm_days",0),bg=bg)

    wb.save(str(REPORT_PATH)); log.info("Report saved --> %s", REPORT_PATH)
    return str(REPORT_PATH)


def _julia_active() -> bool:
    try:
        from engine.julia_bridge import _ready; return _ready
    except Exception: return False


# ══════════════════════════════════════════════════════════════════════════════
# 12. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 62)
    log.info("  PORTFOLIO ANALYTICS ENGINE  --  FULL  |  riskfolio %s", rp.__version__)
    log.info("=" * 62)

    cfg      = load_yaml()
    conf     = build_conf(cfg)
    holdings = derive_holdings(cfg)
    rf_m     = conf["rf"] / 12

    log.info("Holdings:")
    for tkr, h in holdings.items():
        log.info("  %s: %d shares @ avg $%.2f  (total $%.2f)", tkr, h["shares"], h["avg_cost"], h["total_cost"])

    tickers = list(holdings.keys())
    prices, returns = download_data(tickers, conf["start"], conf["end"])

    # Core analytics
    port     = build_portfolio(returns)
    bl_ok    = apply_bl(port, cfg)
    w_dict   = run_optimisations(port, rf_m)
    frontier = build_frontier(port, rf_m)
    risk_tbl = compute_risk_table(returns, rf_m)
    pnl_df   = position_pnl(prices, holdings, conf["fx"])
    div_rows = build_dividend_projection(cfg, holdings, conf["wht"], conf["fx"])
    plots    = generate_plots(port, w_dict, frontier, rf_m)

    # Price alerts
    prices_latest = {t: float(prices[t].iloc[-1]) for t in tickers if t in prices.columns}
    try:
        from engine.alerts import check_alerts
        alerts_fired = check_alerts(prices_latest, cfg)
    except Exception as e:
        log.warning("Alerts failed: %s", e); alerts_fired = []

    # FX timing
    try:
        from engine.fx_timing import run_fx_analysis
        bkln_p = prices_latest.get("BKLN", 21.03)
        fx_result = run_fx_analysis(cfg, bkln_p)
    except Exception as e:
        log.warning("FX timing failed: %s", e); fx_result = {}

    # Dividend calendar
    try:
        from engine.dividend_calendar import run_calendar
        cal_events, cal_path = run_calendar(cfg, holdings)
    except Exception as e:
        log.warning("Calendar failed: %s", e); cal_events = []; cal_path = ""

    # WHT reconciliation
    try:
        from engine.wht_reconciliation import run_wht_reconciliation
        wht_records = run_wht_reconciliation(cfg, conf["fx"])
    except Exception as e:
        log.warning("WHT reconciliation failed: %s", e); wht_records = []

    # Walk-forward backtest
    try:
        from engine.backtest import run_walkforward, plot_backtest
        bt_result = run_walkforward(returns, train_months=24)
        if bt_result:
            bt_charts = plot_backtest(bt_result)
            bt_result["equity_chart"] = bt_charts[0]
            bt_result["sharpe_chart"] = bt_charts[1]
    except Exception as e:
        log.warning("Backtest failed: %s", e); bt_result = {}

    # Generational plan (Monte Carlo)
    try:
        from engine.generational_planner import run_generational_plan
        w_main = list(w_dict.values())[0]["weights"].values
        gen_result = run_generational_plan(returns, w_main, cfg)
    except Exception as e:
        log.warning("Generational plan failed: %s", e); gen_result = {}

    # Excel report
    path = write_excel_report(
        prices, returns, risk_tbl, w_dict, pnl_df, frontier,
        plots, div_rows, cfg, holdings, conf,
        bt_result, gen_result, fx_result, wht_records,
        cal_events, alerts_fired, bl_ok
    )

    log.info("=" * 62)
    log.info("  DONE  -->  %s", path)
    log.info("=" * 62)

    total_mkt = pnl_df["Market Value $"].sum()
    total_unr = pnl_df["Unrealised $"].sum()
    div_thb   = cfg.get("ks_app_snapshot_20260327",{}).get("total_dividends_thb",0)
    log.info("  Portfolio value  : $%s  (THB %s)", f"{total_mkt:,.2f}", f"{total_mkt*conf['fx']:,.0f}")
    log.info("  Unrealised P&L   : $%s  (THB %s)", f"{float(total_unr):,.2f}", f"{float(pnl_df['Unrealised THB'].sum()):,.0f}")
    log.info("  Dividends (KS)   : THB %s", f"{div_thb:,.2f}")
    log.info("  Cash (USD)       : $%.2f", conf["cash_usd"])

    if gen_result.get("milestones"):
        ms10 = gen_result["milestones"].get(10,{})
        log.info("  Generational Y10 : p50 value=$%s  income=$%s/mo",
                 f"{ms10.get('p50_value',0):,.0f}", f"{ms10.get('p50_income_m',0):,.2f}")

    net_arcc = 63.84 * (1 - conf["wht"])
    log.info("  ARCC Q2 2026 est : $63.84 gross  $%.2f net  THB %s  (ex ~2026-06-12)",
             net_arcc, f"{net_arcc*conf['fx']:,.0f}")


if __name__ == "__main__":
    main()
