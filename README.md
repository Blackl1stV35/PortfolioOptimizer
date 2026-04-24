# PortfolioOptimizer  

**Private family income portfolio management system.**  
Thai retail investor focus · USD income assets · KAsset THB mutual fund · Multi-account · AI-assisted

---

## What this is

A multi-page Streamlit application that tracks, analyses, and optimises a generational income portfolio across three brokerage accounts. It is not a general-purpose tool — it is built around the specific needs of a Thai investor holding USD income assets (BKLN, ARCC, PDI) and a Thai fixed-income mutual fund (K-FIXED-A), with a 30-year wealth-building mandate.

---

## Accounts managed

| ID | Account | Currency | Contents |
|----|---------|----------|----------|
| `397543-7` | K CYBER TRADE (Kasikorn Securities) | USD | BKLN 192sh · ARCC 133sh · PDI 105sh |
| `722379-7` | K CYBER TRADE (Kasikorn Securities) | THB | Cash only — Thai domestic account |
| `005-8-95518-3` | KAsset Wisdom (Finnomena) | THB | K-FIXED-A ฿1,396,284.70 |

---

## Project structure

```
PortfolioOptimizer/
├── app.py                        Entry point — 85 lines of pure routing
├── core.py                       Shared runtime: accounts, YAML I/O, cache, sidebar
├── requirements.txt
│
├── pages/
│   ├── p1_dashboard.py           Dashboard: overview, prices, dividend calendar, WHT, transactions
│   ├── p2_intelligence.py        Intelligence Hub: macro, SEC EDGAR, FX timing
│   ├── p3_analytics.py           Analytics: risk, backtest, Monte Carlo, generational plan, charts
│   ├── p4_sandbox.py             Sandbox: What-If, 3D frontier (Plotly + Three.js), exit simulator
│   ├── p5_accounts.py            Account manager: consolidated view, settings, cash transfers
│   ├── p6_research.py            AI research agent (Groq streaming)
│   └── p7_family.py              Family overview (shown when 2+ accounts)
│
├── engine/
│   ├── julia_engine.jl           Julia worker: LW covariance, MC, gen plan, max-Sharpe, HRP
│   ├── julia_bridge.py           Python↔Julia bridge (juliacall, Python fallback)
│   ├── charts.py                 Advanced charts via pandas_ta (SMA/EMA/BB/RSI/MACD)
│   ├── exit_simulator.py         Exit position impact: P&L, tax, income loss, 30yr generational
│   ├── scenario_analyzer.py      What-If multi-asset optimizer
│   ├── wht_reconciliation.py     WHT reconciliation with treaty band 12–22%
│   ├── edgar_monitor.py          SEC EDGAR: ARCC 8-K, XBRL NAV/NII, BDC screener
│   ├── analytics.py              Portfolio risk metrics
│   ├── backtest.py               Walk-forward backtest
│   ├── black_litterman.py        Black-Litterman model
│   ├── dividend_calendar.py      Upcoming dividend event builder
│   ├── fx_timing.py              USD/THB z-score signal
│   ├── generational_planner.py   30-year generational plan
│   ├── macro_monitor.py          VIX, yield curve, oil, FX macro indicators
│   └── report_builder.py        (Legacy — kept for reference, Excel export removed)
│
├── utils/
│   ├── i18n.py                   EN/TH language switching — t() helper
│   ├── finnomena.py              KAsset/Finnomena NAV fetcher + YAML fallback
│   ├── research_agent.py         Groq llama-3.3-70b-versatile streaming agent
│   ├── llm_summarizer.py         Post-analysis AI summaries with session-state cache
│   └── github_commit.py          GitHub PAT auto-commit after data mutations
│
└── config/
    ├── accounts.yaml             Account registry (IDs, colours, YAML filenames)
    ├── portfolio_397543-7.yaml   USD income account (BKLN/ARCC/PDI)
    ├── portfolio_722379-7.yaml   THB cash account
    └── portfolio_005895518-3.yaml KAsset Wisdom mutual fund account
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

Copy `.streamlit/secrets.toml.template` → `.streamlit/secrets.toml` and fill in:

```toml
[github]
pat      = "ghp_..."            # GitHub Personal Access Token (for auto-commit)
repo_url = "https://github.com/Blackl1stV35/PortfolioOptimizer"

[groq]
api_key = "gsk_..."             # From console.groq.com — free tier

[account_pins]
# Not required — PIN gate removed. Accounts switch by button click.
```

> **Never commit `secrets.toml`.** It is already in `.gitignore`.

### 3. Run

```bash
streamlit run app.py
```

---

## Julia acceleration (optional)

Julia provides sub-3-second Monte Carlo (5,000 × 360 months), Ledoit-Wolf covariance, Max-Sharpe, Min-CVaR, and HRP optimisation. Without Julia, Python/NumPy fallbacks are used automatically.

```bash
# Install Julia (https://julialang.org/downloads/)
# Then:
pip install juliacall
# On first run, Julia initialises once via @st.cache_resource
```

Julia is **not required** for the app to function.

---

## Pages

### 📊 Dashboard
- **Overview** — live market values in USD and THB, unrealised P&L per holding (live prices via yfinance)
- **Price & History** — 6-timeframe interactive price chart (1M → MAX) with monthly return bars
- **Dividend Calendar** — smart checklist for upcoming dividends with ✅ confirm flow; auto-updates portfolio YAML; ICS export for Google/Apple/Outlook Calendar
- **Tax & Reconciliation** — WHT implied rate back-calculation; treaty band 12–22% (W-8BEN validated); auto-detects 15% treaty vs 30% default
- **Transactions** — editable `st.data_editor` ledger + step-by-step add-transaction wizard with live price autofill, impact preview, and optional GitHub push

### 🔍 Intelligence Hub
- **Macro Pulse** — Thai policy rate, US Fed rate, VIX, oil, yield curve, recession probability, macro regime (Aggressive / Neutral / Defensive) with cash deployment recommendation
- **SEC EDGAR** — ARCC 8-K dividend declarations, XBRL NAV/NII fundamentals, insider Form 4 trades, BDC candidate screener
- **FX Timing** — USD/THB 90-day z-score signal with deployment recommendation

### 🧪 Analytics Engine
- **Risk & Optimisation** — per-asset risk metrics (Sharpe, CVaR, Max Drawdown) via Julia; Max-Sharpe, Min-Variance, Min-CVaR weight tables; Groq AI summary
- **Backtest** — walk-forward backtest across multiple strategies; Groq AI summary
- **Monte Carlo** — 1,000–10,000 path simulation (Julia-accelerated); p10/p50/p90 fan chart; target income probability; Groq AI summary
- **Generational Plan** — 30-year, 5,000-path plan; milestone table (Year 5/10/15/20/25/30); median time-to-target; Groq AI summary
- **Advanced Charts** — pandas_ta technical indicators: SMA20/50, EMA20, Bollinger Bands, RSI(14), MACD(12,26,9); multi-asset overlay with benchmark comparison; 8 timeframes

### 🔬 What-If & 3D Sandbox
- **What-If Simulator** — add up to 3 tickers simultaneously; ±5% entry price sensitivity sub-scenarios; ΔSharpe / ΔCVaR / Δincome KPIs; before/after weight table; 5-year MC fan chart; sandbox checkbox (never writes to YAML until explicitly confirmed)
- **3D Frontier** — Plotly WebGL 3D scatter: X=volatility, Y=return, Z=monthly income; coloured by Sharpe ratio; 2,000 random portfolios
- **Three.js Bubble Chart** — interactive WebGL asset bubbles; drag to rotate, scroll to zoom, hover for details; size=weight, colour=yield
- **Exit Simulator** — full exit impact: capital gain/loss, Thai CGT exemption note, lost monthly income, 30-year compounded income loss, portfolio concentration after exit

### 👥 Account Manager
- Consolidated NAV, income, and cash across all 3 accounts
- Income bar chart by account
- Per-account settings editor (FX rate, WHT rate, risk-free rate, income target)
- Cash transfer recorder (bookkeeping only — no broker API)

### 🤖 AI Research
- Groq `llama-3.3-70b-versatile` streaming agent
- Full portfolio context injected into system prompt (holdings, WHT, FX, strategy)
- Responds in English or Thai based on language setting
- 7 curated starter questions in both languages

---

## KAsset / Finnomena NAV

The Finnomena API is Cloudflare-blocked from server environments. NAV is managed manually:

1. Open **Dashboard → KAsset Wisdom account → Overview**
2. Expand **"Update NAV manually"**
3. Enter NAV per unit + total market value → **Update NAV**
4. The app stores the entry in `portfolio_005895518-3.yaml → nav_history`
5. A ⚠️ stale warning appears if data is more than 3 days old

---

## Language switching

Click the **🌐 ภาษาไทย / English** button at the bottom of the sidebar. All UI labels, navigation items, and metric names switch. AI Research responses also switch language based on this setting.

---

## Data flow

```
YAML files (primary) ──→ core.load_cfg()
                              │
                              ├── render_sidebar()   (once per re-run)
                              │
                              └── pages/pN.render()  (only active page re-runs)
                                        │
                                        ├── yfinance  (prices, FX — cached 5 min)
                                        ├── Julia     (heavy compute — cached resource)
                                        ├── Groq      (summaries — session-state cache)
                                        └── Finnomena (NAV — YAML fallback)
                                        │
                                    save_cfg()
                                        │
                                        ├── YAML write
                                        ├── Supabase dual-write (Phase 2 hook, optional)
                                        └── GitHub auto-commit (if PAT configured)
```

---

## WHT reconciliation logic

| Implied WHT | Verdict | Meaning |
|-------------|---------|---------|
| < 5% | no_data | KS data missing or zero |
| 12–22% | **treaty_15** ✅ | W-8BEN applied (18% band accounts for KS FX rounding) |
| 27–33% | default_30 ⚠️ | Standard US withholding — file W-8BEN with KS |
| > 35% | overpaid ❌ | Contact KS |
| Other | partial | Unusual — verify KS statement |

W-8BEN treaty rate (15%) validated on account 397543-7 as of 2026-04-14.

---

## Dividend calendar — upcoming (as of 2026-04-14)

| Ticker | Ex-date | Pay-date | Eligible shares | Est. gross | Est. net (15% WHT) |
|--------|---------|----------|-----------------|------------|---------------------|
| PDI | 2026-04-13 | 2026-05-01 | 105 | $23.15 | $19.68 |
| BKLN | 2026-04-23 | 2026-04-27 | 192 | $19.20 | $16.32 |
| PDI | 2026-05-11 | 2026-06-01 | 105 | $23.15 | $19.68 |
| BKLN | 2026-05-28 | 2026-06-01 | 192 | $19.20 | $16.32 |
| **ARCC** | **2026-06-12** | **2026-06-30** | **133** | **$63.84** | **$54.26** |
| BKLN | 2026-06-25 | 2026-06-27 | 192 | $19.20 | $16.32 |
| PDI | 2026-06-12 | 2026-07-01 | 105 | $23.15 | $19.68 |

> ⚠️ ARCC Q1 2026 dividend **missed** — shares purchased 2026-03-25, ex-date was 2026-03-12.

---

## Hosting

### Current: Streamlit Community Cloud
Free, 1 app, ~1 GB RAM. Julia not available (Python fallback active). Suitable for current portfolio size.

### Recommended upgrade: Hugging Face Spaces
Free, always-on, 16 GB RAM, Julia installable via Dockerfile. Zero cold starts.

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://install.julialang.org | sh
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN echo 'juliacall>=0.9.0' >> requirements.txt && pip install juliacall
CMD ["streamlit", "run", "app.py", "--server.port=7860"]
```

---

## Git commit convention

Auto-commits are triggered after: dividend confirmation, trade logging, What-If apply, NAV update.

```
feat: multi-account + Groq LLM + Julia + 3D sandbox
fix: WHT treaty band 12-22%, snapshot key resolver, PyArrow KS THB
data: portfolio_397543-7 — WHT 15% validated, mkt value $8,201.80
data: KAsset Wisdom account 005-8-95518-3 — K-FIXED-A ฿1,396,284.70
```

---

## Licence

Private — not for redistribution.
