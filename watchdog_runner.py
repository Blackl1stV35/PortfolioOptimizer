"""
watchdog_runner.py  --  Portfolio File Watcher
===============================================
Watches config/portfolio.yaml (and any .csv in config/) for changes.
When you save the file, automatically re-runs riskfolio_autonomous.py
and generates a fresh report in output/.

SETUP
-----
    pip install watchdog

RUN
---
    python watchdog_runner.py
    (keep this terminal open while you work)

STOP
----
    Ctrl+C

WINDOWS AUTOSTART AT LOGIN (optional)
--------------------------------------
Create a file named start_watchdog.bat:

    @echo off
    cd /d D:\\portfolio_package
    call .venv\\Scripts\\activate
    python watchdog_runner.py >> output\\watchdog.log 2>&1

Then add it to Windows Task Scheduler:
  - Trigger: At log on
  - Action: Start program -> start_watchdog.bat
  - Start in: D:\\portfolio_package
"""

import sys
import time
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
except ImportError:
    print("ERROR: watchdog not installed.")
    print("       Run: pip install watchdog")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
WATCH_DIR        = Path("config")
WATCH_EXTENSIONS = {".yaml", ".yml", ".csv"}
MAIN_SCRIPT      = Path("riskfolio_autonomous.py")
DEBOUNCE_SECS    = 2.5
LOG_FILE         = Path("output") / "watchdog.log"
LOG_FILE.parent.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(LOG_FILE, encoding="utf-8"),
]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
)
log = logging.getLogger("watchdog")


def _c(code, text):
    try:   return f"\033[{code}m{text}\033[0m"
    except: return text

GRN = lambda t: _c("32", t)
YEL = lambda t: _c("33", t)
CYN = lambda t: _c("36", t)
BLD = lambda t: _c("1",  t)
RED = lambda t: _c("31", t)


# ── Debounced runner ──────────────────────────────────────────────────────────
class DebouncedRunner:
    def __init__(self, delay: float):
        self._delay   = delay
        self._timer   = None
        self._lock    = threading.Lock()
        self._running = False

    def schedule(self, path: str):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._run, args=[path])
            self._timer.start()

    def _run(self, path: str):
        with self._lock:
            if self._running:
                log.warning("Analysis already running -- skipping duplicate trigger")
                return
            self._running = True
        try:
            self._execute(path)
        finally:
            with self._lock:
                self._running = False

    def _execute(self, path: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print()
        log.info(BLD("=" * 56))
        log.info(CYN(f"  Change detected  :  {path}"))
        log.info(CYN(f"  Triggering run   :  {ts}"))
        log.info(BLD("=" * 56))

        if not MAIN_SCRIPT.exists():
            log.error(RED(f"  Script not found: {MAIN_SCRIPT.resolve()}"))
            return

        start = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, str(MAIN_SCRIPT)],
                capture_output=False,
            )
            elapsed = time.monotonic() - start
            if result.returncode == 0:
                log.info(GRN(f"  Completed in {elapsed:.1f}s  ->  check output/ folder"))
            else:
                log.error(RED(f"  FAILED (exit {result.returncode}) after {elapsed:.1f}s"))
        except Exception as exc:
            log.error(RED(f"  Error: {exc}"))

        log.info(BLD("=" * 56))
        print()


# ── Event handler ─────────────────────────────────────────────────────────────
class PortfolioHandler(FileSystemEventHandler):
    def __init__(self, runner: DebouncedRunner):
        self._runner = runner
        super().__init__()

    def _relevant(self, event) -> bool:
        return (not event.is_directory and
                Path(event.src_path).suffix.lower() in WATCH_EXTENSIONS)

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and self._relevant(event):
            log.info(YEL(f"Modified: {event.src_path}"))
            self._runner.schedule(event.src_path)

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent) and self._relevant(event):
            log.info(YEL(f"Created: {event.src_path}"))
            self._runner.schedule(event.src_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not WATCH_DIR.exists():
        log.error(RED(f"Watch directory not found: {WATCH_DIR.resolve()}"))
        log.error(RED("Run from the portfolio_analytics/ root folder."))
        sys.exit(1)

    runner   = DebouncedRunner(delay=DEBOUNCE_SECS)
    handler  = PortfolioHandler(runner)
    observer = Observer()
    observer.schedule(handler, path=str(WATCH_DIR), recursive=False)
    observer.start()

    print()
    log.info(BLD("Portfolio Watchdog  --  Started"))
    log.info(f"  Watching : {WATCH_DIR.resolve()}")
    log.info(f"  Triggers : {', '.join(sorted(WATCH_EXTENSIONS))}")
    log.info(f"  Script   : {MAIN_SCRIPT.resolve()}")
    log.info(f"  Debounce : {DEBOUNCE_SECS}s")
    log.info(f"  Log file : {LOG_FILE.resolve()}")
    log.info(CYN("  Edit and save config/portfolio.yaml to trigger analysis."))
    log.info(CYN("  Press Ctrl+C to stop."))
    print()

    # Run once on startup for a fresh report immediately
    log.info(YEL("Running initial analysis on startup ..."))
    runner._execute("startup")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info(YEL("Stopping ..."))
        observer.stop()
    observer.join()
    log.info("Watchdog stopped.")


if __name__ == "__main__":
    main()
