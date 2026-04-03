# Portfolio Analytics System

> Quantitative income portfolio research tool for personal investment —  
> built on Riskfolio-Lib, Python, and optional Julia acceleration.

**Strategy:** Aggressive Income Builder · Indefinite / Generational Hold  
**Portfolio:** BKLN (Invesco Senior Loan ETF) · ARCC (Ares Capital Corp)  
**Account:** K CYBER TRADE (Kasikorn Securities) · Account 397543-7

---

## Features

- **Portfolio optimisation** — 8 strategies via Riskfolio-Lib v7: Max Sharpe
  (MV, CVaR, Sortino), Min Variance, Min CVaR, Min Max Drawdown, HRP, Equal
  Weight
- **Julia acceleration** — Analytical Nonlinear Shrinkage covariance
  (Ledoit-Wolf 2018) and multi-threaded Monte Carlo via `juliacall`; graceful
  Python fallback when Julia is absent
- **Black-Litterman views** — encode forward-looking market views in
  `config/views.yaml`; posterior mu injected into Riskfolio before optimisation
- **Walk-forward backtest** — rolling 24-month training window, out-of-sample
  equity curves and Sharpe ratios for all strategies
- **30-year generational planner** — 10,000-path Monte Carlo with p10/p50/p90
  wealth and income bands, inflation-adjusted projections, time-to-income-target
- **FX timing signal** — USD/THB 90-day z-score flags optimal conversion windows
  for THB → USD DCA purchases
- **DCA price alerts** — zone-based triggers (strong buy / buy / hold /
  reassess) with Windows toast notification and optional email
- **WHT reconciliation** — back-calculates implied withholding tax rate from KS
  app amounts; confirms whether 30% (default) or 15% (W-8BEN treaty) applies
- **Dividend calendar** — generates `.ics` file importable into Google Calendar,
  Apple Calendar, and Outlook
- **12-sheet Excel report** — full report with embedded Riskfolio charts,
  generated on every run
- **Streamlit dashboard** — 10-page browser UI; deployable free on Streamlit
  Community Cloud
- **Watchdog auto-trigger** — saves `portfolio.yaml` → analysis fires in 2.5s
- **Scheduled runs** — Monday 18:00 full analysis + weekday 16:45 price alerts

---

## Project structure
