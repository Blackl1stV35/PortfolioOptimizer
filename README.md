# Portfolio Analytics System

**Aggressive Income Builder** — BKLN · ARCC · expandable  
**Account:** 397543-7 · K CYBER TRADE (Kasikorn Securities)  
**Strategy:** Maximum monthly income · Indefinite / generational hold

---

## Project structure

```
portfolio_analytics/
├── config/
│   ├── portfolio.yaml          ← EDIT THIS to update positions, settings, views
│   └── views.yaml              ← Black-Litterman forward-looking views
├── engine/
│   ├── __init__.py
│   ├── julia_bridge.py         ← Python↔Julia bridge (auto-fallback if Julia absent)
│   ├── julia_engine.jl         ← Julia PortfolioEngine module
│   ├── alerts.py               ← DCA price alerts (Windows toast + email)
│   ├── fx_timing.py            ← USD/THB conversion timing signal
│   ├── dividend_calendar.py    ← .ics calendar generator
│   ├── wht_reconciliation.py   ← Withholding tax audit (30% vs 15%)
│   ├── black_litterman.py      ← BL view integration
│   ├── backtest.py             ← Walk-forward backtest engine
│   └── generational_planner.py ← 30-year Monte Carlo wealth planner
├── ui/
│   └── app.py                  ← Streamlit 9-page browser dashboard
├── scheduler/
│   └── run_weekly.py           ← Time-based autonomous scheduler
├── output/                     ← Generated reports (auto-created)
│   ├── portfolio_report_YYYYMMDD.xlsx
│   ├── portfolio_dividends.ics
│   └── plots/
├── riskfolio_autonomous.py     ← Main engine (run this)
├── watchdog_runner.py          ← Auto-trigger on YAML save
└── requirements.txt
```

---

## One-time setup

```powershell
# 1. Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Verify
python -c "import riskfolio, yfinance, streamlit; print('All OK')"
```

### Julia setup (optional — unlocks 30-80x faster Monte Carlo)

```powershell
# juliacall downloads Julia 1.10 LTS automatically (~200MB, one-time)
# CovarianceEstimation.jl installs on first bridge init (~30s)
# No manual steps needed beyond pip install juliacall (in requirements.txt)

# To set Julia thread count (recommended):
# Windows: add JULIA_NUM_THREADS=auto to System Environment Variables
# Then restart your terminal
```

---

## Running the system

### Option A — Manual run (generates one report)
```powershell
python riskfolio_autonomous.py
```

### Option B — Watchdog (auto-trigger on YAML save)
```powershell
python watchdog_runner.py
# Keep this running. Edit config/portfolio.yaml and save -> report fires automatically
```

### Option C — Browser dashboard (no YAML editing needed)
```powershell
streamlit run ui/app.py
# Opens http://localhost:8501
# Use the Trade Entry page to log trades without touching YAML
```

### Option D — Scheduled (every Monday 18:00 + daily alerts)
```powershell
pip install schedule
python scheduler/run_weekly.py
```

---

## Updating your portfolio

### When you make a new trade

**Method 1 — YAML (recommended for audit trail)**

Open `config/portfolio.yaml`, scroll to `transactions:`, add:

```yaml
- id: "T004"
  date: "2026-04-15"
  type: "BUY"
  ticker: "BKLN"
  exchange: "ARCX"
  currency: "USD"
  shares: 50
  price_usd: 20.75
  gross_usd: 1037.50
  commission_usd: 5.34
  total_usd: 1042.84
  note: "DCA tranche -- Q2 2026"
```

**Method 2 — Streamlit form**  
Open the browser dashboard → Trade Entry page → fill form → Save.

### When a dividend is received

Open `config/portfolio.yaml`, scroll to `dividends_received:`, add:

```yaml
- period: "2026-04"
  ticker: "BKLN"
  shares_eligible: 185
  ex_date: "2026-04-23"
  pay_date: "2026-04-27"
  amount_per_share_usd: 0.100
  gross_usd_estimated: 18.50
  wht_rate_assumed: 0.30
  net_usd_estimated: 12.95
  thb_ks_app: 435.00        # <-- paste from KS app after receiving
  source: "Actual"
  status: "received"
```

The WHT Reconciliation sheet will automatically calculate whether KS is applying 15% or 30%.

### When you update your market views

Open `config/views.yaml` and edit the confidence / expected_return values. The Black-Litterman model blends these with historical data to produce forward-looking optimisation weights.

---

## Excel report — 12 sheets

| Sheet | Contents |
|-------|----------|
| Dashboard | Portfolio summary, KS snapshot, alert status, Julia/BL active |
| Position P&L | Per-asset P&L computed from transaction ledger |
| Transactions | Full trade history from portfolio.yaml |
| Dividend Tracker | History + projected dividends with WHT breakdown |
| WHT Reconciliation | 30% vs 15% WHT audit per dividend payment |
| Risk Metrics | Sharpe, Sortino, CVaR, Max Drawdown per asset |
| Optimal Weights | Weights for all 8 strategies (BL-adjusted if views.yaml active) |
| Riskfolio Charts | Pie, Frontier, Risk Contribution, Histogram, Drawdown, Clusters |
| Backtest | Walk-forward equity curves + rolling Sharpe (all 4 strategies) |
| Generational Plan | 30-year p10/p50/p90 wealth and income projections |
| FX Timing | USD/THB z-score signal + DCA budget calculator |
| Dividend Calendar | All events with ex-dates, pay-dates, amounts |

---

## Key configuration values (config/portfolio.yaml)

| Setting | Where | Default | Change when |
|---------|-------|---------|-------------|
| `fx_usd_thb` | `meta` | 32.68 | FX rate moves >1 THB |
| `wht_active` | `settings` | 0.30 | KS confirms W-8BEN/treaty (→ 0.15) |
| `risk_free_rate_annual` | `settings` | 0.045 | US T-bill rate changes significantly |
| `dca_monthly_budget_thb` | `analysis` | 51000 | You change your monthly investment amount |
| `target_monthly_income_usd` | `analysis` | 1000 | Your income target changes |

---

## Common issues

**`plot_clusters` skipped** — Normal with 2 assets (BKLN + ARCC). Will activate when you add a 3rd ticker.

**Efficient frontier shows 2 dots** — Same root cause. Add a 3rd asset.

**Julia disabled** — System still works fully in Python mode. To enable: `pip install juliacall` and set `JULIA_NUM_THREADS=auto`.

**WHT shows 30%** — Contact KS and request W-8BEN form filing to claim Thailand-US treaty rate of 15%.

**Alert toast not showing** — Windows only. Install: `pip install win10toast`.

---

## Git workflow (recommended for trade history)

```powershell
git init
git add .
git commit -m "Initial portfolio setup"

# After each trade:
git commit -am "BUY BKLN 50 shares @ $20.75 -- T004"

# Full history of every position change, forever
```
