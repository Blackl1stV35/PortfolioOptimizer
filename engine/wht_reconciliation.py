"""
engine/wht_reconciliation.py  --  Withholding Tax Reconciliation
=================================================================
Builds a comparison table:
  - Your estimated gross dividend (shares x $/share)
  - Net at 30% WHT (default US rate)
  - Net at 15% WHT (Thailand-US treaty rate, if W-8BEN on file)
  - KS app actual received (from portfolio.yaml dividends_received)
  - Implied WHT rate (back-calculated from KS actual)

This answers definitively whether KS is applying 30% or 15%.
When the implied WHT matches 15% -> W-8BEN/treaty is working.
When it matches 30% -> contact KS to file W-8BEN.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class DivRecord:
    period:         str
    ticker:         str
    shares:         int
    per_share:      float
    gross_usd:      float
    net_30pct:      float
    net_15pct:      float
    ks_thb:         Optional[float]
    fx_rate:        float
    ks_usd_est:     Optional[float]
    implied_wht:    Optional[float]
    verdict:        str
    note:           str


VERDICT_COLORS = {
    "treaty_15":  "green",
    "default_30": "red",
    "partial":    "amber",
    "no_data":    "gray",
    "overpaid":   "red",
}


def _back_calc_wht(gross: float, net: float) -> Optional[float]:
    if not net or not gross or gross <= 0 or net > gross:
        return None
    wht = 1.0 - (net / gross)
    return round(wht, 4)


def _classify(implied: Optional[float]) -> tuple[str, str]:
    if implied is None:
        return "no_data", "No KS data -- paste actual amount into portfolio.yaml"
    if implied < 0.05:
        return "no_data", "Implied WHT near zero -- check KS data accuracy"
    if implied > 0.35:
        return "overpaid", f"WHT {implied*100:.1f}% -- higher than expected, contact KS"
    if 0.12 <= implied <= 0.17:
        return "treaty_15", f"WHT ~{implied*100:.1f}% -- treaty rate applied (W-8BEN active)"
    if 0.27 <= implied <= 0.33:
        return "default_30", f"WHT ~{implied*100:.1f}% -- default 30% applied, contact KS to file W-8BEN"
    return "partial", f"WHT ~{implied*100:.1f}% -- unusual rate, verify with KS statement"


def build_reconciliation(cfg: dict, fx_rate: float = 32.68) -> list[DivRecord]:
    records  = []
    received = cfg.get("dividends_received", [])

    for div in received:
        ticker   = div.get("ticker", "")
        period   = div.get("period", "")
        shares   = div.get("shares_eligible", 0)
        per_sh   = div.get("amount_per_share_usd", 0.0)
        gross    = shares * per_sh
        net_30   = gross * 0.70
        net_15   = gross * 0.85
        ks_thb   = div.get("thb_ks_app")

        # Convert KS THB amount back to USD using current fx
        ks_usd   = (float(ks_thb) / fx_rate) if ks_thb else None
        implied  = _back_calc_wht(gross, ks_usd)
        verdict, note = _classify(implied)

        records.append(DivRecord(
            period=period, ticker=ticker, shares=shares,
            per_share=per_sh, gross_usd=round(gross, 4),
            net_30pct=round(net_30, 4), net_15pct=round(net_15, 4),
            ks_thb=float(ks_thb) if ks_thb else None,
            fx_rate=fx_rate,
            ks_usd_est=round(ks_usd, 4) if ks_usd else None,
            implied_wht=implied,
            verdict=verdict, note=note,
        ))

    return records


def summarise_wht(records: list[DivRecord]) -> str:
    if not records:
        return "No dividend records in portfolio.yaml yet."

    verdicts = [r.verdict for r in records if r.verdict != "no_data"]
    if not verdicts:
        return ("No KS actual amounts recorded yet.\n"
                "Action: after each dividend, paste the KS app THB amount "
                "into portfolio.yaml -> dividends_received -> thb_ks_app")

    treaty_count  = verdicts.count("treaty_15")
    default_count = verdicts.count("default_30")
    total = len(verdicts)

    if treaty_count == total:
        conclusion = "CONFIRMED: W-8BEN/treaty (15%) is being applied. No action needed."
    elif default_count == total:
        conclusion = ("WARNING: 30% WHT detected on all records.\n"
                      "Action: Contact KS and request W-8BEN form filing "
                      "to claim Thailand-US tax treaty 15% rate.\n"
                      "Potential annual saving: ~$" +
                      f"{sum(r.gross_usd for r in records if r.verdict == 'default_30') * 0.15:.2f}")
    else:
        conclusion = (f"MIXED: {treaty_count}/{total} at ~15%, {default_count}/{total} at ~30%.\n"
                      "Verify KS statement for each period.")

    lines = [conclusion, ""]
    for r in records:
        wht_str = f"{r.implied_wht*100:.1f}%" if r.implied_wht else "n/a"
        lines.append(f"  {r.period} {r.ticker}: gross=${r.gross_usd:.2f}  "
                     f"KS_THB={r.ks_thb or 'missing'}  WHT={wht_str}  [{r.verdict}]")
    return "\n".join(lines)


def run_wht_reconciliation(cfg: dict, fx_rate: float = 32.68) -> list[DivRecord]:
    records = build_reconciliation(cfg, fx_rate)
    summary = summarise_wht(records)
    log.info("  WHT Reconciliation:\n%s", summary)
    return records
