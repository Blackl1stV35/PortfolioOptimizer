# Portfolio Analytics — Free Server Deployment Guide

## Option A — Streamlit Community Cloud (Recommended, free forever)

**Prerequisites:** GitHub account (free)

1. Create a **private** GitHub repository (e.g. `my-portfolio-analytics`)
2. Upload the entire contents of this zip into the repo root
3. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
4. Click **New app** → select your repo → Main file: `app.py` → Deploy
5. Your dashboard is live at `https://your-app.streamlit.app`

**Access control:** Streamlit Community Cloud apps are public by default.  
To restrict access: Settings → Sharing → Viewers must be logged in → add your email.

**Updating portfolio data:**
- Trade Entry / Dividend pages give you a **Download updated portfolio.yaml** button
- Download it → replace `config/portfolio.yaml` in GitHub → Streamlit auto-redeploys in ~60s

---

## Option B — Render.com (Free tier, spins down after 15min inactivity)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan:** Free
5. Deploy

Note: Free tier sleeps after 15 minutes of inactivity. First visit takes ~30s to wake up.

---

## File structure

```
/
├── app.py                  ← Streamlit app (single entry point)
├── requirements.txt        ← Dependencies (server-safe, no Julia)
├── Procfile                ← For Render.com
├── .streamlit/
│   └── config.toml         ← Theme and server settings
├── config/
│   ├── portfolio.yaml      ← YOUR PORTFOLIO DATA (commit changes here)
│   └── views.yaml          ← Black-Litterman forward views
└── engine/
    ├── analytics.py        ← Core: covariance, Monte Carlo, risk metrics
    ├── report_builder.py   ← Excel report generation
    ├── backtest.py         ← Walk-forward backtest
    ├── generational_planner.py  ← 30-year Monte Carlo
    ├── fx_timing.py        ← USD/THB z-score signal
    ├── dividend_calendar.py ← .ics calendar generator
    ├── wht_reconciliation.py ← 30% vs 15% WHT audit
    └── black_litterman.py  ← BL view integration
```

---

## Adding a trade (workflow)

1. Open your deployed app → **Trade Entry** page
2. Fill in the form → **Preview updated YAML**
3. Click **Download updated portfolio.yaml**
4. In GitHub: navigate to `config/portfolio.yaml` → edit (pencil icon) → paste content → Commit
5. Streamlit redeploys automatically in ~60 seconds

Or: edit `config/portfolio.yaml` directly in GitHub's web editor.

---

## Environment differences vs local version

| Feature | Server | Local |
|---------|--------|-------|
| Julia acceleration | Not available (Python fallback, fast enough for 2-10 assets) | Available via juliacall |
| Auto-trigger on YAML save | Not applicable | watchdog_runner.py |
| Windows toast alerts | Not applicable | alerts.py + win10toast |
| Data persistence | GitHub repo (commit updated YAML) | Local filesystem |
| Excel report | Download button in app | Saved to output/ folder |
| Scheduler | Not applicable | scheduler/run_weekly.py |

---

## Performance on free tier

| Operation | Time |
|-----------|------|
| Dashboard load | ~3s (prices cached 5min) |
| Monte Carlo 3k paths × 5yr | ~1s |
| Monte Carlo 5k paths × 30yr | ~5s |
| Full backtest | ~8s |
| Excel report generation | ~30s |
| Riskfolio charts | ~15s |

All computations use Python/sklearn — no Julia required on the server.
