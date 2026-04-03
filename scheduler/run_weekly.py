"""
scheduler/run_weekly.py  --  Time-Based Autonomous Scheduler
============================================================
Runs the full portfolio analysis every Monday at 18:00 (configurable).
Also checks DCA price alerts every market day at 16:30 (US market close).

Run this instead of cron if you prefer a pure-Python solution:
    python scheduler/run_weekly.py

Runs silently in the background. All output goes to output/scheduler.log.

DEPENDENCIES
    pip install schedule

ALTERNATIVES (if you prefer OS-level scheduling)
    Linux/macOS cron:
        0 18 * * 1 python /path/to/riskfolio_autonomous.py

    Windows Task Scheduler:
        Program  : python.exe
        Arguments: D:\\portfolio_package\\riskfolio_autonomous.py
        Trigger  : Weekly, Monday, 18:00
"""

import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import schedule
except ImportError:
    print("ERROR: schedule not installed.")
    print("       Run: pip install schedule")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
MAIN_SCRIPT = ROOT / "riskfolio_autonomous.py"
ALERT_SCRIPT = ROOT / "engine" / "alerts.py"
LOG_FILE    = ROOT / "output" / "scheduler.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# ── Schedule configuration ────────────────────────────────────────────────────
WEEKLY_DAY  = "monday"     # day for full analysis
WEEKLY_TIME = "18:00"      # local time for full analysis
ALERT_TIME  = "16:45"      # local time for daily price alert check
TIMEZONE    = "Asia/Bangkok"  # informational only -- schedule uses local clock

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("scheduler")


def _run(label: str, script: Path):
    log.info("=" * 50)
    log.info("Starting: %s", label)
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=False,
            timeout=600,          # 10 min max
        )
        if result.returncode == 0:
            log.info("Completed: %s", label)
        else:
            log.error("FAILED (exit %d): %s", result.returncode, label)
    except subprocess.TimeoutExpired:
        log.error("TIMEOUT (>10 min): %s", label)
    except Exception as exc:
        log.error("ERROR running %s: %s", label, exc)
    log.info("=" * 50)


def run_full_analysis():
    _run("Full Portfolio Analysis", MAIN_SCRIPT)


def run_price_alerts():
    """Quick alert check using yfinance -- no full report generated."""
    try:
        import sys; sys.path.insert(0, str(ROOT))
        import yaml
        from collections import defaultdict
        from engine.alerts import check_alerts, summarise_alerts
        import yfinance as yf

        with open(ROOT / "config" / "portfolio.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        holdings = defaultdict(int)
        for tx in cfg.get("transactions", []):
            if tx["type"] == "BUY":
                holdings[tx["ticker"]] += tx["shares"]

        tickers = [t for t, s in holdings.items() if s > 0]
        if not tickers:
            return

        raw = yf.download(tickers, period="1d", auto_adjust=True, progress=False)
        if hasattr(raw.columns, "levels"):
            prices_latest = {t: float(raw["Close"][t].iloc[-1]) for t in tickers}
        else:
            prices_latest = {tickers[0]: float(raw["Close"].iloc[-1])}

        fired = check_alerts(prices_latest, cfg)
        if fired:
            log.info("Alert check: %d zone(s) fired\n%s",
                     len(fired), summarise_alerts(fired))
        else:
            log.info("Alert check: all tickers within normal zones")

    except Exception as exc:
        log.warning("Alert check failed: %s", exc)


def main():
    log.info("Portfolio Scheduler started")
    log.info("  Full analysis: every %s at %s", WEEKLY_DAY.title(), WEEKLY_TIME)
    log.info("  Alert checks : weekdays at %s", ALERT_TIME)
    log.info("  Log file     : %s", LOG_FILE)

    # Schedule weekly full analysis
    getattr(schedule.every(), WEEKLY_DAY).at(WEEKLY_TIME).do(run_full_analysis)

    # Schedule daily alert checks (Mon-Fri)
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        getattr(schedule.every(), day).at(ALERT_TIME).do(run_price_alerts)

    # Run immediately on start
    log.info("Running initial analysis on scheduler start ...")
    run_full_analysis()

    log.info("Scheduler running. Next runs:")
    for job in schedule.jobs:
        log.info("  %s", job)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
