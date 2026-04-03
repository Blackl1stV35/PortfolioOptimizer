"""
engine/edgar_monitor.py  --  SEC EDGAR Intelligence via edgartools
===================================================================
Pulls authoritative data directly from SEC filings for portfolio holdings.
No API key required. No rate limits. Free forever.

SETUP (once):
    pip install edgartools
    Set EDGAR_IDENTITY in portfolio.yaml:
      edgar:
        identity: "your.name@email.com"   # SEC courtesy header, not auth

DATA PROVIDED (ARCC-focused, most useful for BDC holdings):
  - Dividend declarations from 8-K filings (earlier than yfinance)
  - NAV per share from 10-Q XBRL
  - Net Investment Income (NII) per share -- dividend coverage ratio
  - Unrealised portfolio depreciation -- BDC early warning
  - Insider Form 4 trades (executives buying/selling)

BKLN (ETF): Limited value -- passive ETF, no earnings, no NAV in the BDC sense.
            Useful only for expense ratio confirmation from annual report.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── edgartools availability check ────────────────────────────────────────────
def _edgar_available() -> bool:
    try:
        import edgar  # noqa: F401
        return True
    except ImportError:
        log.warning("edgartools not installed. Run: pip install edgartools")
        return False


def _init_edgar(identity: str):
    from edgar import set_identity
    set_identity(identity)


# ══════════════════════════════════════════════════════════════════════════════
# ARCC: DIVIDEND MONITORING VIA 8-K
# ══════════════════════════════════════════════════════════════════════════════
def get_arcc_dividend_declarations(identity: str, lookback_days: int = 90) -> list[dict]:
    """
    Parses ARCC 8-K filings to extract dividend declarations.
    Returns authoritative data 1-3 days earlier than yfinance/aggregators.

    Relevant 8-K items:
      Item 8.01 -- Other Events (dividend announcement)
      Item 9.01 -- Financial Statements and Exhibits

    Example returned dict:
      {"declared": "2026-02-04", "amount": 0.48, "ex_date": "2026-03-12",
       "record_date": "2026-03-13", "pay_date": "2026-03-31",
       "period": "Q1 2026", "source": "8-K"}
    """
    if not _edgar_available():
        return []
    try:
        _init_edgar(identity)
        from edgar import Company

        arcc       = Company("ARCC")
        since_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        filings_8k = arcc.get_filings(form="8-K", date=since_date)

        results = []
        for filing in filings_8k:
            try:
                doc  = filing.obj()
                text = str(doc)
                # Look for dividend keyword in 8-K
                if "dividend" not in text.lower():
                    continue
                # Extract key fields heuristically from text
                result = {
                    "declared":    filing.filing_date,
                    "amount":      _extract_dividend_amount(text),
                    "ex_date":     _extract_date_pattern(text, ["ex-dividend", "ex dividend"]),
                    "record_date": _extract_date_pattern(text, ["record date", "stockholders of record"]),
                    "pay_date":    _extract_date_pattern(text, ["payable", "payment date"]),
                    "period":      _extract_quarter(text),
                    "source":      f"8-K filed {filing.filing_date}",
                    "accession":   filing.accession_no,
                }
                if result["amount"]:   # Only include if we found an amount
                    results.append(result)
            except Exception as e:
                log.debug("8-K parse error: %s", e)
                continue

        log.info("ARCC: %d dividend declarations found in last %d days",
                 len(results), lookback_days)
        return results

    except Exception as exc:
        log.warning("ARCC dividend 8-K fetch failed: %s", exc)
        return []


def _extract_dividend_amount(text: str) -> Optional[float]:
    """Extract dollar amount per share from 8-K text."""
    import re
    # Matches patterns like "$0.48 per share" or "0.48 per share"
    patterns = [
        r"\$(\d+\.\d{2})\s+per\s+share",
        r"(\d+\.\d{2})\s+per\s+share",
        r"dividend\s+of\s+\$(\d+\.\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _extract_date_pattern(text: str, keywords: list[str]) -> Optional[str]:
    """Extract a date near a keyword in filing text."""
    import re
    date_pat = r"(\w+ \d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})"
    for kw in keywords:
        idx = text.lower().find(kw)
        if idx == -1:
            continue
        snippet = text[idx: idx + 120]
        m = re.search(date_pat, snippet)
        if m:
            raw = m.group(1)
            for fmt in ("%B %d, %Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
    return None


def _extract_quarter(text: str) -> Optional[str]:
    import re
    m = re.search(r"(first|second|third|fourth|Q[1-4])\s+quarter\s+(\d{4})",
                  text, re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} Quarter {m.group(2)}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ARCC: NAV + NII FROM 10-Q XBRL
# ══════════════════════════════════════════════════════════════════════════════
def get_arcc_fundamentals(identity: str) -> dict:
    """
    Extracts key BDC metrics from ARCC's most recent 10-Q via XBRL.

    Key metrics for BDC health:
      - NAV per share: should be stable or rising; sharp decline = portfolio stress
      - NII per share: must exceed dividend ($0.48) for coverage > 1.0x
      - Coverage ratio: NII / dividend -- < 1.0x is a red flag
      - Unrealised depreciation: rising depreciation precedes NAV declines

    ARCC XBRL tags (us-gaap standard):
      NetAssetValuePerShare
      InvestmentIncomeNetOfFederalIncomeTaxExpenseBenefitPerShare (NII)
      TaxBasisOfInvestmentsGrossUnrealizedDepreciation
    """
    if not _edgar_available():
        return {}
    try:
        _init_edgar(identity)
        from edgar import Company

        arcc  = Company("ARCC")
        facts = arcc.get_facts()

        # NAV per share history
        nav_ts = _safe_time_series(facts, "us-gaap:NetAssetValuePerShare")

        # NII per share (most direct coverage metric)
        nii_ts = _safe_time_series(
            facts,
            "us-gaap:InvestmentIncomeNetOfFederalIncomeTaxExpenseBenefitPerShare"
        )

        # Unrealised depreciation (BDC early warning signal)
        depr_ts = _safe_time_series(
            facts,
            "us-gaap:TaxBasisOfInvestmentsGrossUnrealizedDepreciation"
        )

        # Compute latest values
        nav_latest  = float(nav_ts.iloc[-1])  if nav_ts  is not None and len(nav_ts)  > 0 else None
        nii_latest  = float(nii_ts.iloc[-1])  if nii_ts  is not None and len(nii_ts)  > 0 else None
        depr_latest = float(depr_ts.iloc[-1]) if depr_ts is not None and len(depr_ts) > 0 else None

        # Dividend coverage
        dividend_qtr = 0.48  # current quarterly dividend
        coverage     = (nii_latest / dividend_qtr) if (nii_latest and dividend_qtr > 0) else None

        # NAV trend (last 4 quarters)
        nav_trend = None
        if nav_ts is not None and len(nav_ts) >= 4:
            nav_trend = float(nav_ts.iloc[-1] - nav_ts.iloc[-5]) if len(nav_ts) >= 5 else float(nav_ts.iloc[-1] - nav_ts.iloc[0])

        # Coverage signal
        if coverage is None:      cov_signal = "unknown"
        elif coverage >= 1.15:    cov_signal = "green"
        elif coverage >= 1.0:     cov_signal = "yellow"
        else:                     cov_signal = "red"

        result = {
            "nav_latest":      nav_latest,
            "nii_latest":      nii_latest,
            "coverage_ratio":  round(coverage, 3) if coverage else None,
            "coverage_signal": cov_signal,
            "depr_latest_m":   round(depr_latest / 1e6, 1) if depr_latest else None,
            "nav_trend_4q":    round(nav_trend, 3) if nav_trend else None,
            "nav_series":      nav_ts,
            "nii_series":      nii_ts,
            "depr_series":     depr_ts,
            "dividend_qtr":    dividend_qtr,
            "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        log.info("ARCC fundamentals: NAV=$%.2f  NII=$%.3f  coverage=%.2fx  depr=$%sM",
                 nav_latest or 0, nii_latest or 0,
                 coverage or 0, f"{depr_latest/1e6:.0f}" if depr_latest else "n/a")
        return result

    except Exception as exc:
        log.warning("ARCC fundamentals fetch failed: %s", exc)
        return {}


def _safe_time_series(facts, tag: str) -> Optional[pd.Series]:
    try:
        ts = facts.time_series(tag)
        if ts is None or len(ts) == 0:
            return None
        return ts
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ARCC: INSIDER FORM 4 TRADES
# ══════════════════════════════════════════════════════════════════════════════
def get_arcc_insider_trades(identity: str, lookback_days: int = 90) -> list[dict]:
    """
    Fetches recent Form 4 insider trades for ARCC.
    Executive buying = strong confidence signal.
    Large insider selling = potential concern.
    """
    if not _edgar_available():
        return []
    try:
        _init_edgar(identity)
        from edgar import Company

        arcc      = Company("ARCC")
        since     = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        form4s    = arcc.get_filings(form="4", date=since)
        trades    = []

        for f in form4s:
            try:
                obj = f.obj()
                df  = obj.transactions
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    trade_type = str(row.get("transaction_code", "")).upper()
                    # P = open market purchase, S = sale
                    if trade_type not in ("P", "S"):
                        continue
                    trades.append({
                        "date":       str(f.filing_date),
                        "insider":    str(row.get("reporting_owner_name", "Unknown")),
                        "type":       "BUY" if trade_type == "P" else "SELL",
                        "shares":     int(row.get("shares", 0)),
                        "price":      float(row.get("price_per_share", 0)),
                        "value":      int(row.get("shares", 0)) * float(row.get("price_per_share", 0)),
                        "accession":  f.accession_no,
                    })
            except Exception:
                continue

        log.info("ARCC: %d insider trades in last %d days", len(trades), lookback_days)
        return sorted(trades, key=lambda x: x["date"], reverse=True)

    except Exception as exc:
        log.warning("ARCC insider trades fetch failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE SCREENER (for evaluating new BDCs before adding)
# ══════════════════════════════════════════════════════════════════════════════
def screen_bdc_candidate(ticker: str, identity: str) -> dict:
    """
    Quick BDC health check for a candidate ticker (e.g. PFLT, MAIN, HTGC).
    Returns NAV trend, coverage ratio, and a go/no-go signal.
    """
    if not _edgar_available():
        return {"error": "edgartools not installed"}
    try:
        _init_edgar(identity)
        from edgar import Company

        co    = Company(ticker)
        facts = co.get_facts()

        nav_ts  = _safe_time_series(facts, "us-gaap:NetAssetValuePerShare")
        nii_ts  = _safe_time_series(facts, "us-gaap:InvestmentIncomeNetOfFederalIncomeTaxExpenseBenefitPerShare")
        depr_ts = _safe_time_series(facts, "us-gaap:TaxBasisOfInvestmentsGrossUnrealizedDepreciation")

        nav_now  = float(nav_ts.iloc[-1])  if nav_ts  is not None else None
        nii_now  = float(nii_ts.iloc[-1])  if nii_ts  is not None else None
        depr_now = float(depr_ts.iloc[-1]) if depr_ts is not None else None

        # NAV decline over past year
        nav_1yr_chg = None
        if nav_ts is not None and len(nav_ts) >= 5:
            nav_1yr_chg = float(nav_ts.iloc[-1] - nav_ts.iloc[-5]) / float(abs(nav_ts.iloc[-5])) if nav_ts.iloc[-5] != 0 else 0

        # Simple go/no-go
        warnings = []
        if nav_1yr_chg is not None and nav_1yr_chg < -0.05:
            warnings.append(f"NAV declined {nav_1yr_chg*100:.1f}% over 4 quarters")
        if depr_now is not None and nav_now is not None:
            depr_pct = depr_now / (nav_now * 1e6) if nav_now > 0 else 0
            if depr_pct > 0.15:
                warnings.append(f"Unrealised depreciation {depr_pct*100:.0f}% of NAV -- elevated")
        if nii_now is not None:
            # Rough quarterly check (NII reported quarterly)
            div_proxy = nav_now * 0.10 / 4 if nav_now else 0.40
            coverage  = nii_now / div_proxy if div_proxy > 0 else None
            if coverage and coverage < 1.0:
                warnings.append(f"NII coverage {coverage:.2f}x -- dividend at risk")

        return {
            "ticker":       ticker,
            "nav_latest":   nav_now,
            "nii_latest":   nii_now,
            "nav_1yr_chg":  nav_1yr_chg,
            "depr_m":       round(depr_now / 1e6, 1) if depr_now else None,
            "warnings":     warnings,
            "signal":       "red" if warnings else "green",
        }
    except Exception as exc:
        log.warning("BDC screen %s failed: %s", ticker, exc)
        return {"ticker": ticker, "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def get_edgar_intelligence(cfg: dict) -> dict:
    """
    Main entry point. Returns all EDGAR data for the portfolio.
    Gracefully returns empty dicts if edgartools is unavailable.
    """
    identity = cfg.get("edgar", {}).get("identity", "")
    if not identity:
        log.warning("edgar.identity not set in portfolio.yaml -- EDGAR features disabled")
        return {"available": False, "reason": "Set edgar.identity in portfolio.yaml"}

    holdings = cfg.get("instruments", {})
    result   = {"available": True, "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")}

    if "ARCC" in holdings:
        result["arcc_dividends"]  = get_arcc_dividend_declarations(identity)
        result["arcc_fundamentals"] = get_arcc_fundamentals(identity)
        result["arcc_insider"]    = get_arcc_insider_trades(identity)

    # BKLN is a passive ETF -- no meaningful BDC metrics to extract
    result["bkln_note"] = "BKLN is a passive ETF. NAV tracks loan index. No XBRL earnings to parse."

    return result