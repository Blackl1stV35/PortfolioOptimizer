# Portfolio Analytics System

> Personal quantitative income portfolio research tool —  
> built on Riskfolio-Lib, Python, Julia acceleration, SEC EDGAR intelligence, and macro monitoring.

---

## Dashboard Preview

### Backtest Results

<div style="display: flex; justify-content: space-around;">
  <figure style="flex:1; text-align:center; margin:10px;">
    <img src="/output/plots/backtest_equity.png" style="width:100%; height:auto;"/>
    <figcaption>Walk-Forward Equity Curves</figcaption>
  </figure>
  <figure style="flex:1; text-align:center; margin:10px;">
    <img src="/output/plots/backtest_sharpe.png" style="width:100%; height:auto;"/>
    <figcaption>Rolling Sharpe Ratio</figcaption>
  </figure>
</div>

### FX Signal & Generational Plan

<div style="display: flex; justify-content: space-around;">
  <figure style="flex:1; text-align:center; margin:10px;">
    <img src="/output/plots/fx_signal.png" style="width:100%; height:auto;"/>
    <figcaption>USD/THB FX Signal</figcaption>
  </figure>
  <figure style="flex:1; text-align:center; margin:10px;">
    <img src="/output/plots/generational_plan.png" style="width:100%; height:auto;"/>
    <figcaption>Generational Wealth Plan</figcaption>
  </figure>
</div>

### Distribution & Composition

<div style="display: flex; justify-content: space-around;">
  <figure style="flex:1; text-align:center; margin:10px;">
    <img src="/output/plots/hist.png" style="width:100%; height:auto;"/>
    <figcaption>Portfolio Returns Histogram</figcaption>
  </figure>
  <figure style="flex:1; text-align:center; margin:10px;">
    <img src="/output/plots/pie.png" style="width:100%; height:auto;"/>
    <figcaption>Max Sharpe Allocation</figcaption>
  </figure>
</div>

### Risk Contribution

<div style="display: flex; justify-content: center;">
  <figure style="flex:1; text-align:center; margin:10px; max-width:60%;">
    <img src="/output/plots/risk_con.png" style="width:100%; height:auto;"/>
    <figcaption>Risk Contribution per Asset</figcaption>
  </figure>
</div>

---

## What this does

This system replaces the K CYBER TRADE app's analytics with a full quantitative
research environment. It reads your holdings and transaction history from a single
YAML file, fetches live prices from Yahoo Finance, pulls authoritative data directly
from SEC EDGAR filings, monitors macro conditions affecting the portfolio, runs
portfolio optimisation using Riskfolio-Lib, and outputs a 12-sheet Excel report
plus a live 12-page browser dashboard.

Every feature degrades gracefully — Julia is optional, the Streamlit UI is optional,
email alerts are optional, and edgartools falls back cleanly when unavailable. At
minimum, the core engine runs with only `riskfolio-lib`, `yfinance`, `pyyaml`,
and `openpyxl`.

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

**Walk-forward backtest**  
Rolling 24-month training window with out-of-sample performance evaluation.
Outputs normalised equity curves and rolling Sharpe ratios for all strategies.

**30-year generational planner**  
10,000-path bootstrap Monte Carlo producing p10/p50/p90 portfolio value bands,
monthly income projections, inflation-adjusted real values, and the median time
to reach a target passive income level. Designed for the indefinite hold thesis.

**Macro Pulse & Risk Dashboard**  
Single-screen macro monitor covering the six indicators that directly drive BKLN,
ARCC, and PDI performance: Thai BOT policy rate, US Fed funds rate, VIX, WTI oil,
USD/THB FX signal, and US recession probability. Four analysis tabs cover policy
rates, risk gauges (default, liquidity, maturity, uncertainty), an economic snapshot,
and a dynamic cash deployment recommendation. The regime engine (Defensive / Neutral /
Aggressive) adjusts suggested cash allocation and PDI deployment percentage in
real time based on the composite macro score.

**SEC EDGAR Intelligence (edgartools)**  
Direct integration with SEC EDGAR filings — no API key, no cost, no rate limits.
Pulls ARCC dividend declarations from 8-K filings 1-3 days before aggregators
publish them. Extracts NAV per share and Net Investment Income per share from
10-Q XBRL data to compute the dividend coverage ratio — the single most important
metric for assessing whether ARCC's $0.48 quarterly dividend is sustainable.
Monitors unrealised portfolio depreciation as an early-warning signal for BDC stress.
Tracks executive Form 4 insider trades. Includes a one-click BDC candidate screener
for evaluating PFLT, MAIN, HTGC, and others before adding them to the portfolio.

**FX timing signal**  
Computes a 90-day z-score of the USD/THB rate. A negative z-score indicates the
USD is historically cheap — the optimal window to convert THB savings to USD for
the next DCA tranche.

**DCA price alerts**  
Zone-based triggers for each ticker (strong buy / buy / hold / reassess). Fires a
Windows desktop notification and optional email when a ticker enters an actionable zone.

**Withholding tax reconciliation**  
Back-calculates the implied WHT rate from KS app THB amounts and compares it against
30% (default US rate) and 15% (Thailand-US treaty rate). Flags whether filing a
W-8BEN form with KS would recover meaningful income.

**Dividend calendar**  
Generates an iCalendar `.ics` file from confirmed and projected dividend dates.
Importable into Google Calendar, Apple Calendar, and Outlook.

**12-sheet Excel report**  
Produced on every run with all analytics, charts, projections, and a dedicated
Macro Pulse sheet.

**Streamlit dashboard**  
12-page browser interface covering all features. Deployable free on Streamlit
Community Cloud.

**Autonomous operation**  
Save `config/portfolio.yaml` and the watchdog fires the full analysis within 2.5
seconds. A time-based scheduler runs the full report every Monday at 18:00 and
price alerts on weekdays at 16:45.

---

## Project structure

```
portfolio_analytics/
├── app.py                          ← Streamlit dashboard (server entry point)
├── riskfolio_autonomous.py         ← Local engine entry point
├── watchdog_runner.py              ← File-change auto-trigger
├── requirements.txt
├── config/
│   ├── portfolio.yaml              ← All holdings, transactions, dividends, macro rates
│   └── views.yaml                  ← Black-Litterman forward views
├── engine/
│   ├── analytics.py                ← Covariance, Monte Carlo, risk metrics (Python-only)
│   ├── julia_bridge.py             ← Python-Julia bridge (graceful fallback)
│   ├── julia_engine.jl             ← Julia PortfolioEngine module
│   ├── alerts.py                   ← DCA price alerts (toast + email)
│   ├── fx_timing.py                ← USD/THB z-score conversion signal
│   ├── dividend_calendar.py        ← .ics calendar generator
│   ├── wht_reconciliation.py       ← Withholding tax audit
│   ├── black_litterman.py          ← Black-Litterman view integration
│   ├── backtest.py                 ← Walk-forward backtest engine
│   ├── generational_planner.py     ← 30-year Monte Carlo wealth planner
│   ├── report_builder.py           ← Excel report (server build)
│   ├── macro_monitor.py            ← Macro Pulse & Risk Dashboard engine
│   └── edgar_monitor.py            ← SEC EDGAR intelligence via edgartools
├── pages/
│   ├── 03_Macro_Pulse.py           ← Macro dashboard Streamlit page
│   └── 04_SEC_Intelligence.py      ← SEC EDGAR Streamlit page
├── scheduler/
│   └── run_weekly.py               ← Time-based autonomous scheduler
├── ui/
│   └── app.py                      ← Local Streamlit app
└── output/                         ← Generated reports (auto-created)
    ├── portfolio_report_YYYYMMDD.xlsx
    ├── portfolio_dividends.ics
    └── plots/
        ├── backtest_equity.png
        ├── backtest_sharpe.png
        ├── fx_signal.png
        ├── generational_plan.png
        ├── hist.png
        ├── pie.png
        ├── risk_con.png
        ├── frontier.png
        └── drawdown.png
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

# Install all dependencies
pip install -r requirements.txt
```

### Julia (optional — 30-80x faster Monte Carlo)

`juliacall` downloads Julia 1.10 LTS automatically on first import (one-time, ~200 MB).
`CovarianceEstimation.jl` installs automatically on first bridge initialisation (~30 s).

```bash
# Windows — add to System Environment Variables:
JULIA_NUM_THREADS=auto

# macOS / Linux:
export JULIA_NUM_THREADS=auto
```

### SEC EDGAR identity (required for edgartools)

The SEC requires a courtesy user-agent header for all programmatic access.
Add your email to `config/portfolio.yaml` — this is not authentication.

```yaml
edgar:
  identity: "your.name@email.com"
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

Edit and save `config/portfolio.yaml` — the full analysis fires automatically
within 2.5 seconds.

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

Open `config/portfolio.yaml`, scroll to `transactions:`, and append:

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
  shares_eligible: 192
  ex_date: "2026-04-23"
  pay_date: "2026-04-27"
  amount_per_share_usd: 0.100
  gross_usd_estimated: 19.20
  wht_rate_assumed: 0.30
  net_usd_estimated: 13.44
  thb_ks_app: 460.00
  source: "Actual"
  status: "received"
```

The `thb_ks_app` field triggers the WHT Reconciliation module to back-calculate
the effective rate and determine whether 15% or 30% is being applied.

### Updating macro rates

After each BOT or Fed meeting, update `config/portfolio.yaml`:

```yaml
macro:
  thai_policy_rate:      1.00
  thai_policy_rate_date: "2026-04-09"
  us_fed_rate:           3.625
  us_fed_rate_date:      "2026-03-19"
```

### Updating market views

Edit `config/views.yaml`. The Black-Litterman model re-runs on the next cycle,
blending your forward-looking views with historical returns to produce adjusted
expected returns.

### Version control

```bash
git init
git add .
git commit -m "Initial portfolio setup"

# After each trade:
git commit -am "BUY BKLN 50sh @ $20.75 — T004"
```

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
| `cash_reserve_thb` | `analysis` | `92000` | Available cash for deployment decisions |
| `thai_policy_rate` | `macro` | `1.00` | After each BOT MPC meeting |
| `us_fed_rate` | `macro` | `3.625` | After each FOMC meeting |
| `identity` | `edgar` | — | One-time setup (your email for SEC courtesy header) |

---

## Dashboard pages

| Page | Contents |
|------|----------|
| Dashboard | Live holdings, KPI cards, 6-month price chart, ARCC dividend alert |
| Trade Entry | Form → download updated YAML to commit |
| Analytics & Optimisation | Risk metrics, 8 strategy weights, all Riskfolio charts |
| Dividend Tracker | History + projected + .ics calendar download + KS amount logger |
| FX Timing | USD/THB z-score signal + DCA budget calculator |
| Monte Carlo | Fan chart, p10/p50/p90 income projection |
| Generational Plan | 30-year milestone table + fan chart |
| WHT Reconciliation | 30% vs 15% back-calculation from KS app amounts |
| Backtest | Walk-forward equity curves + rolling Sharpe by strategy |
| Macro Pulse | Policy rates, risk gauges, economic snapshot, cash deployment recommendation |
| SEC Intelligence | ARCC 8-K dividend declarations, NAV/NII XBRL, insider trades, BDC screener |
| Download Report | Generates and downloads the full 12-sheet Excel report |

---

## Excel report — sheet guide

| Sheet | Contents |
|-------|----------|
| Dashboard | KPI summary, holdings snapshot, active alert status |
| Position P&L | Per-asset P&L computed from transaction ledger |
| Transactions | Full trade history |
| Dividend Tracker | Received and projected dividends with WHT breakdown |
| WHT Reconciliation | 30% vs 15% audit per dividend payment period |
| Risk Metrics | Sharpe, Sortino, CVaR, Max Drawdown, Omega per asset |
| Optimal Weights | All 8 strategies (BL-adjusted when views are active) |
| Riskfolio Charts | Allocation pie, Frontier, Risk Contribution, Histogram, Drawdown |
| Backtest | Walk-forward equity curves and rolling Sharpe ratios |
| Generational Plan | 30-year p10/p50/p90 wealth and income fan charts |
| FX Timing | USD/THB z-score signal and DCA budget breakdown |
| Macro Pulse | Regime summary, all 6 macro indicators with signals |

---

## Macro Pulse — indicator guide

| Indicator | Source | Why it matters to this portfolio |
|-----------|--------|----------------------------------|
| Thai BOT policy rate | Manual (portfolio.yaml) | Cost of THB savings; affects FX conversion timing |
| US Fed funds rate | Manual (portfolio.yaml) | BKLN floating coupons reprice against SOFR quarterly |
| VIX | yfinance `^VIX` | High VIX compresses BDC/CEF valuations; signals entry opportunities |
| WTI oil | yfinance `CL=F` | Oil > $95 triggers Defensive posture |
| USD/THB FX signal | engine/fx_timing.py | Negative z-score = cheap USD = optimal conversion window |
| Recession probability | 2s10s curve proxy | Deep inversion → elevated probability → reduce ARCC sizing |
| HYG vs LQD (credit) | yfinance | HY underperformance = rising default risk = ARCC headwind |
| 2s10s spread | yfinance `^IRX` / `^TNX` | Inversion precedes recession by 12-18 months |

**Regime thresholds**

| Score | Regime | Cash suggestion | Action |
|-------|--------|-----------------|--------|
| 7–10 | Defensive | 22–25% | Hold cash; delay PDI deployment |
| 4–6 | Neutral | 15–20% | Deploy 50% cautiously; split PDI / BKLN |
| 0–3 | Aggressive | 10–15% | DCA into BKLN and PDI this week |

---

## SEC EDGAR Intelligence — what edgartools provides

| Data | Filing | Frequency | Value |
|------|--------|-----------|-------|
| Dividend declarations | 8-K | Quarterly | Confirms amount and dates 1-3 days before aggregators |
| NAV per share | 10-Q XBRL | Quarterly | True asset value; rising NAV = healthy BDC |
| NII per share | 10-Q XBRL | Quarterly | Must exceed $0.48 for coverage ≥ 1.0x |
| Dividend coverage ratio | Computed | Quarterly | Key sustainability metric; < 1.0x = red flag |
| Unrealised depreciation | 10-Q XBRL | Quarterly | BDC early-warning signal |
| Executive insider trades | Form 4 | As filed | Management buying = conviction signal |
| BDC candidate screen | XBRL multi-ticker | On demand | NAV trend + coverage for PFLT, MAIN, HTGC |

---

## Server deployment (free)

### Streamlit Community Cloud — recommended

Free, always-on, no credit card required.

1. Push this repository to a private GitHub repository
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub
3. Click **New app** → select repository → Main file: `app.py` → Deploy
4. App is live at `https://your-app-name.streamlit.app`

To restrict access: Settings → Sharing → Viewers must be logged in → add your email.

**Updating data:** The Trade Entry and Dividend Tracker pages include a
**Download updated portfolio.yaml** button. Download it, navigate to
`config/portfolio.yaml` in GitHub, click edit, paste the content, and commit.
Streamlit redeploys in ~60 seconds.

### Render.com — alternative

Free tier spins down after 15 minutes of inactivity and restarts in ~30 seconds.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## Citation

This project uses **Riskfolio-Lib** for all portfolio optimisation and risk analytics.
Please cite the library in any published or shared research that uses outputs from
this system.

```bibtex
@misc{riskfolio,
      author = {Dany Cajas},
      title  = {Riskfolio-Lib (7.2.1)},
      year   = {2026},
      url    = {https://github.com/dcajasn/Riskfolio-Lib},
}
```

This project also uses **edgartools** for SEC EDGAR data access.

```bibtex
@software{edgartools,
  author  = {Dany Gunning},
  title   = {edgartools: Python library for SEC EDGAR filings},
  url     = {https://github.com/dgunning/edgartools},
  license = {MIT},
}
```

Documentation:
- Riskfolio-Lib: [riskfolio-lib.readthedocs.io](https://riskfolio-lib.readthedocs.io)
- edgartools: [edgartools.readthedocs.io](https://edgartools.readthedocs.io)

---

## Disclaimer

This tool is for personal investment research only. Nothing produced by this system
constitutes financial advice. All investments carry risk, including the possible loss
of principal. Dividend projections are estimates and not guarantees. Withholding tax
rates should be verified directly with K CYBER TRADE. SEC filing data is sourced
directly from EDGAR and is subject to the SEC's terms of use. The author is not a
licensed financial advisor.

---

## License

MIT — see `LICENSE`
