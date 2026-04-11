"""
engine/edgar_monitor.py  --  SEC EDGAR Intelligence (edgartools)
================================================================
BUG FIX (Priority 1):
  All XBRL time-series values are now extracted with _scalar() which
  handles Series, DataFrame, ndarray, and bare scalars safely.
  The previous bug was passing a pandas Series directly to float(),
  which raises: "float() argument must be a string or a real number, not 'Series'"
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ── Safe scalar extractor (THE FIX) ──────────────────────────────────────────
def _scalar(obj) -> Optional[float]:
    """
    Robustly extract a single float from whatever XBRL returns.
    Handles: float, int, str, pd.Series, pd.DataFrame, np.ndarray, None.
    Returns None instead of raising on bad input.
    """
    if obj is None:
        return None
    try:
        # Already a scalar
        if isinstance(obj, (int, float)):
            return float(obj)
        # String (some XBRL values come back as numeric strings)
        if isinstance(obj, str):
            return float(obj.replace(",", "").strip())
        # pandas Series — take last non-null value
        if isinstance(obj, pd.Series):
            clean = obj.dropna()
            if clean.empty:
                return None
            return float(clean.iloc[-1])
        # pandas DataFrame — take last row, first column
        if isinstance(obj, pd.DataFrame):
            clean = obj.dropna(how="all")
            if clean.empty:
                return None
            return float(clean.iloc[-1, 0])
        # numpy scalar or array
        import numpy as np
        if isinstance(obj, np.ndarray):
            flat = obj.flatten()
            non_nan = flat[~np.isnan(flat.astype(float, casting="safe"))]
            if len(non_nan) == 0:
                return None
            return float(non_nan[-1])
        # Last resort
        return float(obj)
    except Exception as exc:
        log.debug("_scalar conversion failed for %s: %s", type(obj).__name__, exc)
        return None


def _scalar_series(obj) -> Optional[pd.Series]:
    """Return a clean pd.Series from XBRL output, or None."""
    if obj is None:
        return None
    try:
        if isinstance(obj, pd.Series):
            s = obj.dropna()
            return s if not s.empty else None
        if isinstance(obj, pd.DataFrame):
            s = obj.iloc[:, 0].dropna()
            return s if not s.empty else None
        return None
    except Exception:
        return None


# ── edgartools availability ───────────────────────────────────────────────────
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


# ── ARCC: Dividend 8-K ────────────────────────────────────────────────────────
def get_arcc_dividend_declarations(identity: str, lookback_days: int = 90) -> list[dict]:
    if not _edgar_available():
        return []
    try:
        _init_edgar(identity)
        from edgar import Company
        arcc      = Company("ARCC")
        since     = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        filings   = arcc.get_filings(form="8-K", date=since)
        results   = []
        for filing in filings:
            try:
                text = str(filing.obj())
                if "dividend" not in text.lower():
                    continue
                result = {
                    "declared":    filing.filing_date,
                    "amount":      _extract_amount(text),
                    "ex_date":     _extract_date(text, ["ex-dividend", "ex dividend"]),
                    "record_date": _extract_date(text, ["record date", "stockholders of record"]),
                    "pay_date":    _extract_date(text, ["payable", "payment date"]),
                    "period":      _extract_quarter(text),
                    "source":      f"8-K filed {filing.filing_date}",
                }
                if result["amount"]:
                    results.append(result)
            except Exception as e:
                log.debug("8-K parse: %s", e)
        log.info("ARCC: %d dividend declarations found", len(results))
        return results
    except Exception as exc:
        log.warning("ARCC 8-K fetch failed: %s", exc)
        return []


def _extract_amount(text: str) -> Optional[float]:
    for pat in [r"\$(\d+\.\d{2})\s+per\s+share",
                r"(\d+\.\d{2})\s+per\s+share",
                r"dividend\s+of\s+\$(\d+\.\d{2})"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return _scalar(m.group(1))
    return None


def _extract_date(text: str, keywords: list) -> Optional[str]:
    date_pat = r"(\w+ \d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})"
    for kw in keywords:
        idx = text.lower().find(kw)
        if idx == -1:
            continue
        m = re.search(date_pat, text[idx: idx + 120])
        if m:
            raw = m.group(1)
            for fmt in ("%B %d, %Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
    return None


def _extract_quarter(text: str) -> Optional[str]:
    m = re.search(r"(first|second|third|fourth|Q[1-4])\s+quarter\s+(\d{4})",
                  text, re.IGNORECASE)
    return f"{m.group(1).title()} Quarter {m.group(2)}" if m else None


# ── ARCC: Fundamentals (NAV + NII) ───────────────────────────────────────────
def get_arcc_fundamentals(identity: str) -> dict:
    if not _edgar_available():
        return {}
    try:
        _init_edgar(identity)
        from edgar import Company
        arcc  = Company("ARCC")
        facts = arcc.get_facts()

        # ── FIXED: use _scalar() and _scalar_series() everywhere ──
        nav_ts   = _scalar_series(facts.time_series("us-gaap:NetAssetValuePerShare"))
        nii_ts   = _scalar_series(facts.time_series(
            "us-gaap:InvestmentIncomeNetOfFederalIncomeTaxExpenseBenefitPerShare"))
        depr_ts  = _scalar_series(facts.time_series(
            "us-gaap:TaxBasisOfInvestmentsGrossUnrealizedDepreciation"))

        nav_latest  = _scalar(nav_ts.iloc[-1])  if nav_ts  is not None else None
        nii_latest  = _scalar(nii_ts.iloc[-1])  if nii_ts  is not None else None
        depr_latest = _scalar(depr_ts.iloc[-1]) if depr_ts is not None else None

        div_qtr  = 0.48
        coverage = (nii_latest / div_qtr) if (nii_latest and div_qtr > 0) else None

        nav_trend = None
        if nav_ts is not None and len(nav_ts) >= 5:
            nav_trend = _scalar(nav_ts.iloc[-1]) - _scalar(nav_ts.iloc[-5])

        cov_signal = ("green"   if coverage and coverage >= 1.15 else
                      "yellow"  if coverage and coverage >= 1.0  else
                      "red"     if coverage else "unknown")

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
            "dividend_qtr":    div_qtr,
            "fetched_at":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        log.info("ARCC fundamentals: NAV=$%.2f  NII=$%.3f  coverage=%.2fx",
                 nav_latest or 0, nii_latest or 0, coverage or 0)
        return result
    except Exception as exc:
        log.warning("ARCC fundamentals failed: %s", exc)
        return {}


# ── ARCC: Insider Form 4 ──────────────────────────────────────────────────────
def get_arcc_insider_trades(identity: str, lookback_days: int = 90) -> list[dict]:
    if not _edgar_available():
        return []
    try:
        _init_edgar(identity)
        from edgar import Company
        arcc  = Company("ARCC")
        since = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        trades = []
        for f in arcc.get_filings(form="4", date=since):
            try:
                df = f.obj().transactions
                if df is None or df.empty:
                    continue
                for _, row in df.iterrows():
                    code = str(row.get("transaction_code", "")).upper()
                    if code not in ("P", "S"):
                        continue
                    shares = _scalar(row.get("shares", 0)) or 0
                    price  = _scalar(row.get("price_per_share", 0)) or 0
                    trades.append({
                        "date":     str(f.filing_date),
                        "insider":  str(row.get("reporting_owner_name", "Unknown")),
                        "type":     "BUY" if code == "P" else "SELL",
                        "shares":   int(shares),
                        "price":    price,
                        "value":    int(shares * price),
                    })
            except Exception:
                continue
        return sorted(trades, key=lambda x: x["date"], reverse=True)
    except Exception as exc:
        log.warning("ARCC insider trades failed: %s", exc)
        return []


# ── BDC Candidate Screener (FIXED) ───────────────────────────────────────────
def screen_bdc_candidate(ticker: str, identity: str) -> dict:
    """
    Screen a BDC for NAV trend, NII coverage, and depreciation warning.
    All XBRL values go through _scalar() — no more Series-to-float errors.
    """
    if not _edgar_available():
        return {"error": "edgartools not installed"}
    try:
        _init_edgar(identity)
        from edgar import Company
        co    = Company(ticker)
        facts = co.get_facts()

        # ── FIXED: _scalar_series then _scalar for each value ──
        nav_ts  = _scalar_series(facts.time_series("us-gaap:NetAssetValuePerShare"))
        nii_ts  = _scalar_series(facts.time_series(
            "us-gaap:InvestmentIncomeNetOfFederalIncomeTaxExpenseBenefitPerShare"))
        depr_ts = _scalar_series(facts.time_series(
            "us-gaap:TaxBasisOfInvestmentsGrossUnrealizedDepreciation"))

        nav_now  = _scalar(nav_ts.iloc[-1])  if nav_ts  is not None else None
        nii_now  = _scalar(nii_ts.iloc[-1])  if nii_ts  is not None else None
        depr_now = _scalar(depr_ts.iloc[-1]) if depr_ts is not None else None

        # NAV change over past year (last 5 quarterly readings)
        nav_1yr_chg = None
        if nav_ts is not None and len(nav_ts) >= 5:
            nav_old = _scalar(nav_ts.iloc[-5])
            nav_cur = _scalar(nav_ts.iloc[-1])
            if nav_old and nav_old != 0 and nav_cur is not None:
                nav_1yr_chg = (nav_cur - nav_old) / abs(nav_old)

        warnings = []
        if nav_1yr_chg is not None and nav_1yr_chg < -0.05:
            warnings.append(f"NAV declined {nav_1yr_chg*100:.1f}% over 4 quarters")

        if depr_now is not None and nav_now is not None and nav_now > 0:
            depr_pct = depr_now / (nav_now * 1e6)
            if depr_pct > 0.15:
                warnings.append(f"Unrealised depreciation {depr_pct*100:.0f}% of NAV — elevated")

        if nii_now is not None and nav_now:
            div_proxy  = nav_now * 0.10 / 4
            coverage   = nii_now / div_proxy if div_proxy > 0 else None
            if coverage and coverage < 1.0:
                warnings.append(f"NII coverage {coverage:.2f}× — dividend at risk")

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
        return {"ticker": ticker, "error": str(exc), "warnings": [], "signal": "unknown"}


# ── Aggregate entry point ─────────────────────────────────────────────────────
def get_edgar_intelligence(cfg: dict) -> dict:
    identity = cfg.get("edgar", {}).get("identity", "")
    if not identity:
        return {"available": False, "reason": "Set edgar.identity in portfolio.yaml"}

    holdings = cfg.get("instruments", {})
    result   = {"available": True, "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")}

    if "ARCC" in holdings:
        result["arcc_dividends"]    = get_arcc_dividend_declarations(identity)
        result["arcc_fundamentals"] = get_arcc_fundamentals(identity)
        result["arcc_insider"]      = get_arcc_insider_trades(identity)

    result["bkln_note"] = "BKLN is a passive ETF — no BDC-specific XBRL metrics apply."
    return result
