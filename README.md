# Portfolio Analytics System

> Personal quantitative income portfolio research tool —  
> built on Riskfolio-Lib, Python, and optional Julia acceleration.

**Strategy:** Aggressive Income Builder · Indefinite / Generational Hold  
**Holdings:** BKLN (Invesco Senior Loan ETF) · ARCC (Ares Capital Corp)  
**Broker:** K CYBER TRADE — Kasikorn Securities · Account 397543-7

---

# Portfolio Optimizer Dashboard

## Backtest Results
<p float="left">
  <img src="PortfolioOptimizer/output/plots/backtest_equity.png" width="400"/>
  <img src="PortfolioOptimizer/output/plots/backtest_sharpe.png" width="400"/>
</p>

## FX Signal & Generational Plan
<p float="left">
  <img src="PortfolioOptimizer/output/plots/fx_signal.png" width="400"/>
  <img src="PortfolioOptimizer/output/plots/generational_plan.png" width="400"/>
</p>

## Distribution & Composition
<p float="left">
  <img src="PortfolioOptimizer/output/plots/hist.png" width="400"/>
  <img src="PortfolioOptimizer/output/plots/pie.png" width="400"/>
</p>

## Risk Contribution
<p float="left">
  <img src="PortfolioOptimizer/output/plots/risk_con.png" width="400"/>
</p>

---

## What this does

This system replaces the K CYBER TRADE app's analytics with a full quantitative
research environment. It reads your holdings and transaction history from a
single YAML file, fetches live prices from Yahoo Finance, runs portfolio
optimisation using Riskfolio-Lib, and outputs a 12-sheet Excel report plus a
live browser dashboard.

Every feature degrades gracefully — Julia is optional, the Streamlit UI is
optional, email alerts are optional. At minimum, the core engine runs with only
`riskfolio-lib`, `yfinance`, `pyyaml`, and `openpyxl`.

---

## Features

**Portfolio optimisation**
Eight strategies via Riskfolio-Lib v7: Maximum Sharpe (MV, CVaR, Sortino),
Minimum Variance, Minimum CVaR, Minimum Max Drawdown, Hierarchical Risk Parity,
and Equal Weight. Black-Litterman forward views can be added in `config/views.yaml`
to make weight recommendations forward-looking rather than purely historical.

**Julia acceleration**
An optional Julia bridge provides Analytical Nonlinear Shrinkage covariance
estimation (Ledoit-Wolf 2018, QIS) and multi-threaded Monte Carlo simulation.
The system falls back transparently to Python/sklearn when Julia is absent.
Julia becomes worthwhile at 10+ assets; with the current 2-asset portfolio the
Python fallback is fast enough.

**Walk-forward backtest**
Rolling 24-month training window with out-of-sample performance evaluation.
Outputs normalised equity curves and rolling Sharpe ratios for all strategies.

**30-year generational planner**
10,000-path bootstrap Monte Carlo producing p10/p50/p90 portfolio value bands,
monthly income projections, inflation-adjusted real values, and the median time
to reach a target passive income level. Designed specifically for the indefinite
hold thesis.

**FX timing signal**
Computes a 90-day z-score of the USD/THB rate. A negative z-score indicates the
USD is historically cheap — the optimal window to convert THB savings to USD for
the next DCA tranche.

**DCA price alerts**
Zone-based triggers for each ticker (strong buy / buy / hold / reassess).
Fires a Windows desktop notification and optional email when a ticker enters an
actionable zone.

**Withholding tax reconciliation**
Back-calculates the implied WHT rate from KS app THB amounts and compares it
against 30% (default US rate) and 15% (Thailand-US treaty rate). Flags whether
filing a W-8BEN form with KS would recover meaningful income.

**Dividend calendar**
Generates an iCalendar `.ics` file from confirmed and projected dividend dates.
Importable into Google Calendar, Apple Calendar, and Outlook. Includes ex-date
reminders (5 days prior) and pay-date notifications.

**12-sheet Excel report**
Produced on every run with all analytics, charts, and projections embedded.

**Streamlit dashboard**
10-page browser interface covering all features. Deployable for free on
Streamlit Community Cloud.

**Autonomous operation**
Save `config/portfolio.yaml` and the watchdog fires the full analysis within
2.5 seconds. A time-based scheduler runs the full report every Monday at 18:00
and price alerts on weekdays at 16:45.

---

## Project structure

```
portfolio_analytics/
├── app.py                        ← Streamlit dashboard (server entry point)
├── riskfolio_autonomous.py       ← Local engine entry point
├── watchdog_runner.py            ← File-change trigger
├── requirements.txt
├── config/
│   ├── portfolio.yaml            ← All holdings, transactions, dividends
│   └── views.yaml                ← Black-Litterman forward views
├── engine/
│   ├── analytics.py              ← Covariance, Monte Carlo, risk metrics
│   ├── julia_bridge.py           ← Python-Julia bridge (graceful fallback)
│   ├── julia_engine.jl           ← Julia PortfolioEngine module
│   ├── alerts.py                 ← DCA price alerts (toast + email)
│   ├── fx_timing.py              ← USD/THB z-score conversion signal
│   ├── dividend_calendar.py      ← .ics calendar generator
│   ├── wht_reconciliation.py     ← Withholding tax audit
│   ├── black_litterman.py        ← Black-Litterman view integration
│   ├── backtest.py               ← Walk-forward backtest engine
│   ├── generational_planner.py   ← 30-year Monte Carlo wealth planner
│   └── report_builder.py         ← Excel report (server build)
├── scheduler/
│   └── run_weekly.py             ← Time-based autonomous scheduler
├── ui/
│   └── app.py                    ← Local Streamlit app
└── output/                       ← Generated reports (auto-created)
    ├── portfolio_report_YYYYMMDD.xlsx
    ├── portfolio_dividends.ics
    └── plots/
```

---

## Installation

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Julia (optional — 30-80x faster Monte Carlo)

`juliacall` downloads Julia 1.10 LTS automatically on first import (one-time,
~200 MB). `CovarianceEstimation.jl` installs automatically on first engine
initialisation (~30 s).

```bash
# Set thread count before starting (Windows — System Environment Variables):
JULIA_NUM_THREADS=auto

# macOS / Linux:
export JULIA_NUM_THREADS=auto
```

---

## Running

### One-off report

```bash
python riskfolio_autonomous.py
```

Output: `output/portfolio_report_YYYYMMDD.xlsx`

### Auto-trigger on YAML save

```bash
python watchdog_runner.py
```

Edit and save `config/portfolio.yaml` in any text editor — the full analysis
fires automatically within 2.5 seconds.

### Browser dashboard

```bash
streamlit run app.py
```

Opens `http://localhost:8501`

### Scheduled (Monday 18:00 + weekday alerts)

```bash
pip install schedule
python scheduler/run_weekly.py
```

---

## Updating your portfolio

### Adding a trade

Open `config/portfolio.yaml`, scroll to the `transactions:` section, and append
an entry following the template below. The engine derives current holdings,
average cost, and total cost basis from this ledger automatically — do not edit
the `ks_app_snapshot` section manually.

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
  note: "Q2 2026 DCA tranche"
```

Alternatively, use the Streamlit **Trade Entry** page — no YAML editing needed.

### Logging a received dividend

Append to `dividends_received:` in `config/portfolio.yaml`:

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
  thb_ks_app: 435.00
  source: "Actual"
  status: "received"
```

The `thb_ks_app` field is the amount shown in the K CYBER TRADE app. Once
entered, the WHT Reconciliation module back-calculates the implied rate and
determines whether KS is applying 30% or 15%.

### Updating market views

Edit `config/views.yaml`. The Black-Litterman model re-runs on the next
analysis cycle, blending your forward-looking views with historical returns to
produce adjusted expected returns for each asset.

### Tracking version history

```bash
git init
git add .
git commit -m "Initial portfolio setup"

# After each trade:
git commit -am "BUY BKLN 50sh @ $20.75 — T004"
```

Git provides a complete timestamped trade history at no cost.

---

## Configuration reference

All key parameters live in `config/portfolio.yaml`.

| Parameter | Location | Default | Update when |
|-----------|----------|---------|-------------|
| `fx_usd_thb` | `meta` | `32.68` | FX rate moves more than 1 THB |
| `wht_active` | `settings` | `0.30` | KS confirms W-8BEN treaty rate → set to `0.15` |
| `risk_free_rate_annual` | `settings` | `0.045` | US T-bill rate changes significantly |
| `dca_monthly_budget_thb` | `analysis` | `51000` | Monthly investment amount changes |
| `target_monthly_income_usd` | `analysis` | `1000.0` | Income target changes |

---

## Excel report — sheet guide

| Sheet | Contents |
|-------|----------|
| Dashboard | KPI summary, holdings snapshot, active alerts |
| Position P&L | Per-asset P&L computed from the transaction ledger |
| Transactions | Full trade history |
| Dividend Tracker | Received and projected dividends with WHT breakdown |
| WHT Reconciliation | 30% vs 15% audit per dividend payment period |
| Risk Metrics | Sharpe, Sortino, CVaR, Max Drawdown, Omega per asset |
| Optimal Weights | All 8 strategies (BL-adjusted when views are active) |
| Riskfolio Charts | Allocation pie, Frontier, Risk Contribution, Histogram, Drawdown |
| Backtest | Walk-forward equity curves and rolling Sharpe ratios |
| Generational Plan | 30-year p10/p50/p90 wealth and income fan charts |
| FX Timing | USD/THB z-score signal and DCA budget breakdown |
| Dividend Calendar | All ex-dates and pay-dates with per-period amounts |

---

## Server deployment (free)

### Streamlit Community Cloud — recommended

Free, always-on, no credit card required.

1. Push this repository to a private GitHub repository
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub
3. Click **New app** → select your repository → Main file: `app.py` → Deploy
4. App is live at `https://your-app-name.streamlit.app`

To restrict access to yourself: Settings → Sharing → Viewers must be
logged in → add your email address.

**Updating data from the server:** The Trade Entry and Dividend Tracker pages
include a **Download updated portfolio.yaml** button. Download the file,
navigate to `config/portfolio.yaml` in your GitHub repository, click the pencil
edit icon, paste the new content, and commit. Streamlit redeploys in ~60 seconds.

### Render.com — alternative

Free tier spins down after 15 minutes of inactivity and restarts in ~30 seconds
on the next request.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

A `Procfile` is included in the server deployment package for Render.

---

## Citation

This project uses **Riskfolio-Lib** for all portfolio optimisation and risk
analytics. Please cite the library in any published or shared research that uses
outputs from this system.

```bibtex
@misc{riskfolio,
      author = {Dany Cajas},
      title  = {Riskfolio-Lib (7.2.1)},
      year   = {2026},
      url    = {https://github.com/dcajasn/Riskfolio-Lib},
}
```

Documentation: [riskfolio-lib.readthedocs.io](https://riskfolio-lib.readthedocs.io)

---

## Disclaimer

This tool is for personal investment research only. Nothing produced by this
system constitutes financial advice. All investments carry risk, including the
possible loss of principal. Dividend projections are estimates and not
guarantees. Withholding tax rates should be verified directly with
K CYBER TRADE. The author is not a licensed financial advisor.

---

## License

MIT — see `LICENSE`
