"""
engine/dividend_calendar.py  --  Dividend Calendar Generator
=============================================================
Generates an iCalendar (.ics) file from portfolio.yaml dividend schedule.
Import the output into Google Calendar, Apple Calendar, Outlook, or iPhone.

Events created:
  - Ex-dividend date  (reminder: must own shares BEFORE this date)
  - Pay date          (when cash hits your account)
  - Projected future dates based on instrument dividend_policy

Dependencies:
    pip install icalendar
"""

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
OUTPUT_DIR = Path("output")


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except ValueError:
            continue
    return None


def _make_event(uid: str, summary: str, dt: date,
                description: str, alarm_days: int = 3) -> dict:
    return {
        "uid":         uid,
        "summary":     summary,
        "date":        dt,
        "description": description,
        "alarm_days":  alarm_days,
    }


def build_events(cfg: dict, holdings: dict) -> list[dict]:
    events = []
    instruments = cfg.get("instruments", {})

    # Historical / confirmed dividends received
    for div in cfg.get("dividends_received", []):
        ticker = div.get("ticker", "")
        if ticker not in holdings:
            continue
        period = div.get("period", "")
        amt    = div.get("amount_per_share_usd", 0.0)
        sh     = div.get("shares_eligible", holdings.get(ticker, {}).get("shares", 0))
        gross  = sh * amt
        note   = div.get("source", "")
        status = div.get("status", "received")

        ex_dt  = _parse_date(div.get("ex_date"))
        pay_dt = _parse_date(div.get("pay_date"))

        if ex_dt:
            events.append(_make_event(
                uid=f"ex_{ticker}_{period}",
                summary=f"[EX-DIV] {ticker} {period} -- own shares before this date",
                dt=ex_dt,
                description=(
                    f"Ex-dividend date for {ticker} ({period})\n"
                    f"Amount: ${amt:.3f}/share\n"
                    f"Eligible shares: {sh}\n"
                    f"Gross: ${gross:.2f}\n"
                    f"Status: {status}\n{note}"
                ),
                alarm_days=5,
            ))
        if pay_dt:
            events.append(_make_event(
                uid=f"pay_{ticker}_{period}",
                summary=f"[DIVIDEND] {ticker} {period} ~${gross:.2f} gross",
                dt=pay_dt,
                description=(
                    f"Dividend payment for {ticker} ({period})\n"
                    f"Amount: ${amt:.3f}/share x {sh} shares\n"
                    f"Gross USD: ${gross:.2f}\n"
                    f"Net ~30% WHT: ${gross*0.70:.2f}\n"
                    f"Net ~15% WHT: ${gross*0.85:.2f}\n"
                    f"Source: {note}"
                ),
                alarm_days=1,
            ))

    # Upcoming projected dividends from instrument definitions
    for ticker, inst in instruments.items():
        if ticker not in holdings:
            continue
        sh = holdings[ticker]["shares"]
        for upcoming in inst.get("dividend_policy", {}).get("estimated_upcoming", []):
            period = upcoming.get("period", upcoming.get("ex", ""))
            amt    = upcoming.get("amount", upcoming.get("amount_per_share_usd", 0.0))
            eligible = upcoming.get("eligible_for_our_shares", True)
            gross    = sh * amt if eligible else 0.0
            missed   = not eligible

            ex_dt  = _parse_date(upcoming.get("ex"))
            pay_dt = _parse_date(upcoming.get("pay"))

            if ex_dt and ex_dt >= date.today():
                events.append(_make_event(
                    uid=f"ex_{ticker}_{period}_proj",
                    summary=(
                        f"{'[MISSED] ' if missed else '[EX-DIV] '}"
                        f"{ticker} {period} -- "
                        f"{'BUY BEFORE THIS DATE' if not missed else 'already missed'}"
                    ),
                    dt=ex_dt,
                    description=(
                        f"{'PROJECTED' if not missed else 'MISSED'} ex-div: {ticker}\n"
                        f"Amount: ${amt:.3f}/share\n"
                        f"Shares: {sh}\n"
                        f"Gross: ${gross:.2f}  {'(MISSED)' if missed else ''}\n"
                        f"Note: {upcoming.get('note','')}"
                    ),
                    alarm_days=7 if not missed else 0,
                ))

            if pay_dt and pay_dt >= date.today() and eligible:
                events.append(_make_event(
                    uid=f"pay_{ticker}_{period}_proj",
                    summary=f"[DIVIDEND PROJECTED] {ticker} {period} ~${gross:.2f} gross",
                    dt=pay_dt,
                    description=(
                        f"Projected payment: {ticker} ({period})\n"
                        f"${amt:.3f}/share x {sh} shares = ${gross:.2f} gross\n"
                        f"Net ~30% WHT: ${gross*0.70:.2f}\n"
                        f"Net ~15% WHT: ${gross*0.85:.2f}"
                    ),
                    alarm_days=1,
                ))

    events.sort(key=lambda e: e["date"])
    return events


def write_ics(events: list[dict], path: Path) -> str:
    try:
        from icalendar import Calendar, Event, Alarm
        from icalendar import vText, vDatetime
    except ImportError:
        log.warning("icalendar not installed -- writing plain .ics manually")
        return _write_ics_manual(events, path)

    cal = Calendar()
    cal.add("prodid", "-//Portfolio Analytics//dividend_calendar//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-calname", "Portfolio Dividends")
    cal.add("x-wr-timezone", "Asia/Bangkok")

    for ev in events:
        e = Event()
        e.add("uid",     ev["uid"] + "@portfolio")
        e.add("summary", ev["summary"])
        e.add("dtstart", ev["date"])
        e.add("dtend",   ev["date"] + timedelta(days=1))
        e.add("description", ev["description"])
        e.add("dtstamp", datetime.now())

        if ev.get("alarm_days", 0) > 0:
            a = Alarm()
            a.add("action",  "DISPLAY")
            a.add("description", ev["summary"])
            a.add("trigger", timedelta(days=-ev["alarm_days"]))
            e.add_component(a)

        cal.add_component(e)

    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        f.write(cal.to_ical())
    log.info("  Calendar written: %s (%d events)", path, len(events))
    return str(path)


def _write_ics_manual(events: list[dict], path: Path) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Portfolio Analytics//EN",
        "X-WR-CALNAME:Portfolio Dividends",
    ]
    for ev in events:
        dt_str   = ev["date"].strftime("%Y%m%d")
        dt_end   = (ev["date"] + timedelta(days=1)).strftime("%Y%m%d")
        desc     = ev["description"].replace("\n", "\\n")
        summary  = ev["summary"].replace(",", "\\,")
        alarm_d  = ev.get("alarm_days", 0)
        lines += [
            "BEGIN:VEVENT",
            f"UID:{ev['uid']}@portfolio",
            f"DTSTART;VALUE=DATE:{dt_str}",
            f"DTEND;VALUE=DATE:{dt_end}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{desc}",
        ]
        if alarm_d > 0:
            lines += [
                "BEGIN:VALARM",
                "ACTION:DISPLAY",
                f"DESCRIPTION:{summary}",
                f"TRIGGER:-P{alarm_d}D",
                "END:VALARM",
            ]
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")

    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines))
    log.info("  Calendar written (manual): %s (%d events)", path, len(events))
    return str(path)


def run_calendar(cfg: dict, holdings: dict) -> tuple[list[dict], str]:
    events   = build_events(cfg, holdings)
    cal_path = OUTPUT_DIR / "portfolio_dividends.ics"
    written  = write_ics(events, cal_path)
    log.info("  %d dividend calendar events generated", len(events))
    return events, written
