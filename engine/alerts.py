"""
engine/alerts.py  --  DCA Price Alert System
=============================================
Checks live prices against YAML-defined DCA zones on every engine run.
Sends Windows toast notification + optional email when triggers fire.

Dependencies (optional -- alerts degrade gracefully if missing):
    pip install win10toast-click requests
    pip install secure-smtplib   (or use standard smtplib + SSL)
"""

import smtplib
import logging
import platform
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

# ── Alert zone definitions (read from portfolio.yaml settings) ────────────────
# Fallback defaults if not in YAML
DEFAULT_ZONES = {
    "BKLN": [
        {"label": "Strong Buy",  "min": 20.00, "max": 20.70, "action": "buy_aggressive"},
        {"label": "Buy",         "min": 20.70, "max": 21.10, "action": "buy"},
        {"label": "Hold",        "min": 21.10, "max": 21.50, "action": "hold"},
        {"label": "Reassess",    "min": 21.50, "max": 99.99, "action": "reassess"},
    ],
    "ARCC": [
        {"label": "Target Entry","min": 18.00, "max": 19.50, "action": "buy"},
        {"label": "Acceptable",  "min": 19.50, "max": 20.50, "action": "buy_small"},
        {"label": "Avoid",       "min": 20.50, "max": 99.99, "action": "avoid"},
    ],
}

ALERT_HISTORY_FILE = Path("output") / "alert_history.yaml"


# ── Notification channels ─────────────────────────────────────────────────────
def _windows_toast(title: str, message: str) -> bool:
    if platform.system() != "Windows":
        return False
    try:
        from win10toast import ToastNotifier
        ToastNotifier().show_toast(title, message, duration=8, threaded=True)
        return True
    except ImportError:
        log.debug("win10toast not installed -- skipping toast")
    except Exception as e:
        log.debug("Toast failed: %s", e)
    return False


def _send_email(subject: str, body: str, cfg: dict) -> bool:
    email_cfg = cfg.get("alerts", {}).get("email", {})
    if not email_cfg.get("enabled", False):
        return False
    try:
        sender  = email_cfg["sender"]
        to      = email_cfg["recipient"]
        pw      = email_cfg["app_password"]   # Gmail App Password
        host    = email_cfg.get("smtp_host", "smtp.gmail.com")
        port    = int(email_cfg.get("smtp_port", 587))

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = to
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(sender, pw)
            s.sendmail(sender, to, msg.as_string())
        return True
    except Exception as e:
        log.warning("Email alert failed: %s", e)
        return False


# ── Alert history dedup ───────────────────────────────────────────────────────
def _load_history() -> dict:
    if ALERT_HISTORY_FILE.exists():
        with open(ALERT_HISTORY_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_history(history: dict):
    ALERT_HISTORY_FILE.parent.mkdir(exist_ok=True)
    with open(ALERT_HISTORY_FILE, "w") as f:
        yaml.dump(history, f)


def _already_sent(history: dict, ticker: str, zone_label: str) -> bool:
    key = f"{ticker}_{zone_label}"
    last = history.get(key)
    if not last:
        return False
    # Re-alert after 24h even for same zone
    delta = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
    return delta < 86400


def _mark_sent(history: dict, ticker: str, zone_label: str):
    history[f"{ticker}_{zone_label}"] = datetime.now().isoformat()


# ── Main entry point ──────────────────────────────────────────────────────────
def check_alerts(prices_latest: dict, cfg: dict) -> list[dict]:
    """
    prices_latest : {ticker: current_price_float}
    cfg           : full portfolio YAML dict
    Returns list of fired alert dicts.
    """
    zones_cfg = cfg.get("dca_zones", DEFAULT_ZONES)
    alert_cfg = cfg.get("alerts", {})
    history   = _load_history()
    fired     = []

    for ticker, price in prices_latest.items():
        zones = zones_cfg.get(ticker, DEFAULT_ZONES.get(ticker, []))
        for zone in zones:
            if zone["min"] <= price < zone["max"]:
                label  = zone["label"]
                action = zone["action"]

                alert = {
                    "ticker":  ticker,
                    "price":   price,
                    "zone":    label,
                    "action":  action,
                    "ts":      datetime.now().isoformat(),
                }
                fired.append(alert)

                # Log regardless
                if action in ("buy_aggressive", "buy"):
                    log.info("  ALERT [%s] $%.2f --> %s zone (%s)",
                             ticker, price, label, action)
                else:
                    log.info("  ZONE  [%s] $%.2f --> %s", ticker, price, label)

                # Only notify for actionable zones
                if action in ("buy_aggressive", "buy", "reassess") \
                        and not _already_sent(history, ticker, label):

                    title = f"Portfolio Alert: {ticker} {label}"
                    body  = (
                        f"{ticker} is at ${price:.2f} -- {label} zone.\n"
                        f"Recommended action: {action.replace('_',' ').title()}\n"
                        f"DCA zone: ${zone['min']:.2f} -- ${zone['max']:.2f}\n"
                        f"Triggered: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    )

                    toast_ok = _windows_toast(title, body)
                    email_ok = _send_email(f"[Portfolio] {title}", body, cfg)

                    if toast_ok or email_ok:
                        _mark_sent(history, ticker, label)
                        log.info("    Notification sent (toast=%s email=%s)",
                                 toast_ok, email_ok)
                break  # price is in exactly one zone

    _save_history(history)
    return fired


def summarise_alerts(fired: list[dict]) -> str:
    if not fired:
        return "All tickers within normal hold zones."
    lines = []
    for a in fired:
        lines.append(f"  {a['ticker']:6s}  ${a['price']:.2f}  -->  {a['zone']}  ({a['action']})")
    return "\n".join(lines)
