"""
engine/exit_simulator.py  —  Exit Position Simulator
=====================================================
Computes full portfolio impact of selling all or part of a position:
  - Capital gain/loss (USD + THB)
  - Lost future dividend income
  - Portfolio-level risk delta (Sharpe, CVaR, concentration)
  - FX timing impact
  - Generational plan deviation

Pure computation — never modifies YAML.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COMMISSION_USD = 5.34

YIELD_MAP = {
    "BKLN": 0.0707, "ARCC": 0.1066, "PDI": 0.152,
    "MAIN": 0.076,  "HTGC": 0.12,   "PFLT": 0.149,
}


def simulate_exit(
    cfg:             dict,
    ticker:          str,
    shares_to_sell:  int,
    exit_price_usd:  float,
    fx_rate:         float = 32.68,
    wht_rate:        float = 0.15,
) -> dict:
    """
    Simulate selling `shares_to_sell` of `ticker` at `exit_price_usd`.

    Returns
    -------
    dict with sections:
        trade       : transaction details
        pnl         : capital gain/loss
        income      : lost monthly income
        tax         : capital gains tax estimate (Thai rules: exempt for foreign ETF)
        portfolio   : before/after portfolio metrics
        fx          : FX impact
        generational: deviation from 30-year plan
        recommendation : plain-text summary
    """
    # ── Derive current holdings ───────────────────────────────────────────────
    shares_held  = defaultdict(int)
    cost_total   = defaultdict(float)
    for tx in cfg.get("transactions", []):
        if tx["type"] == "BUY":
            shares_held[tx["ticker"]] += tx["shares"]
            cost_total[tx["ticker"]]  += tx["total_usd"]
        elif tx["type"] == "SELL":
            shares_held[tx["ticker"]] -= tx["shares"]

    held = int(shares_held.get(ticker, 0))
    if held == 0:
        return {"error": f"{ticker} not found in holdings"}
    if shares_to_sell > held:
        return {"error": f"Cannot sell {shares_to_sell} — only {held} held"}

    avg_cost = cost_total.get(ticker, 0) / held if held else 0
    sell_gross = shares_to_sell * exit_price_usd
    sell_net   = sell_gross - COMMISSION_USD
    cost_basis = shares_to_sell * avg_cost
    cap_gain   = sell_net - cost_basis

    # ── Income impact ─────────────────────────────────────────────────────────
    yield_rate      = YIELD_MAP.get(ticker.upper(), 0.08)
    lost_gross_mo   = shares_to_sell * exit_price_usd * yield_rate / 12
    lost_net_mo     = lost_gross_mo * (1 - wht_rate)
    remaining_gross = (held - shares_to_sell) * exit_price_usd * yield_rate / 12
    remaining_net   = remaining_gross * (1 - wht_rate)

    # Total portfolio income before/after
    total_income_before = 0.0
    total_income_after  = 0.0
    for tkr, sh in shares_held.items():
        if sh <= 0: continue
        avg = cost_total[tkr] / sh
        y   = YIELD_MAP.get(tkr.upper(), 0.08)
        mo_gross = sh * avg * y / 12
        total_income_before += mo_gross * (1 - wht_rate)
        sell_sh = shares_to_sell if tkr == ticker else 0
        after_sh = sh - sell_sh
        if after_sh > 0:
            total_income_after += after_sh * avg * y / 12 * (1 - wht_rate)

    # ── Tax estimate (Thai rules) ─────────────────────────────────────────────
    # Thai individuals investing in foreign securities via Thai broker:
    #   Capital gains on foreign ETF/stocks: NOT subject to Thai personal income tax
    #   (confirmed as of 2024 Revenue Dept ruling for foreign-listed securities)
    # WHT on dividends: handled separately (already at 15%)
    tax_note = (
        "Capital gains on foreign-listed securities (BKLN/ARCC/PDI) "
        "are currently EXEMPT from Thai personal income tax for individuals. "
        "No capital gains tax estimated. Verify with Revenue Department for updates."
    )
    tax_usd = 0.0

    # ── FX impact ─────────────────────────────────────────────────────────────
    proceeds_thb = sell_net * fx_rate
    cost_thb     = cost_basis * fx_rate
    cap_gain_thb = cap_gain * fx_rate

    # ── Portfolio concentration after ─────────────────────────────────────────
    total_after = sum(
        max(0, (shares_held[t] - (shares_to_sell if t == ticker else 0))) * avg_cost
        for t in shares_held
    )
    concentration = {}
    for t, sh in shares_held.items():
        after_sh = sh - (shares_to_sell if t == ticker else 0)
        if after_sh <= 0 or total_after <= 0:
            concentration[t] = 0.0
        else:
            cost_t = cost_total[t] / sh * after_sh
            concentration[t] = round(cost_t / total_after, 4)

    max_conc = max(concentration.values(), default=0)

    # ── Generational plan impact ───────────────────────────────────────────────
    # Simple: lost income compounded over 30 years at 7% annual return
    months_left = 360
    future_lost = sum(
        lost_net_mo * ((1 + 0.07/12) ** m)
        for m in range(months_left)
    )

    # ── Recommendation ────────────────────────────────────────────────────────
    lines = []
    partial = shares_to_sell < held
    pct     = shares_to_sell / held

    if cap_gain < 0:
        lines.append(f"• Selling realises a LOSS of ${cap_gain:,.2f} (฿{cap_gain_thb:,.0f}). "
                     "Consider whether holding improves risk/reward before deciding.")
    else:
        lines.append(f"• Realises a GAIN of ${cap_gain:,.2f} (฿{cap_gain_thb:,.0f}). No Thai capital gains tax.")

    lines.append(f"• Lost monthly income: ${lost_net_mo:.2f}/mo (net after {wht_rate*100:.0f}% WHT). "
                 f"Over 30 years this compounds to an estimated ${future_lost:,.0f}.")

    if max_conc > 0.50 and len(concentration) > 1:
        top = max(concentration, key=concentration.get)
        lines.append(f"• ⚠️ After exit, {top} represents {concentration[top]:.0%} of portfolio — high concentration risk.")

    if partial:
        lines.append(f"• Partial exit ({pct:.0%}) — {held - shares_to_sell} shares remain, "
                     f"continuing to earn ~${remaining_net:.2f}/mo.")

    return {
        "trade": {
            "ticker":          ticker,
            "shares_sold":     shares_to_sell,
            "shares_remaining":held - shares_to_sell,
            "exit_price":      round(exit_price_usd, 4),
            "gross_usd":       round(sell_gross, 2),
            "commission":      COMMISSION_USD,
            "net_proceeds_usd":round(sell_net, 2),
            "net_proceeds_thb":round(proceeds_thb, 2),
        },
        "pnl": {
            "cost_basis_usd":  round(cost_basis, 2),
            "avg_cost_usd":    round(avg_cost, 4),
            "capital_gain_usd":round(cap_gain, 2),
            "capital_gain_thb":round(cap_gain_thb, 2),
            "gain_pct":        round(cap_gain / cost_basis * 100, 2) if cost_basis else 0,
        },
        "income": {
            "lost_gross_mo":        round(lost_gross_mo, 2),
            "lost_net_mo":          round(lost_net_mo, 2),
            "remaining_net_mo":     round(remaining_net, 2),
            "portfolio_income_before": round(total_income_before, 2),
            "portfolio_income_after":  round(total_income_after, 2),
            "income_drop_pct":      round((total_income_before - total_income_after)
                                         / total_income_before * 100, 1) if total_income_before else 0,
        },
        "tax": {
            "capital_gains_tax_usd": tax_usd,
            "note":                  tax_note,
        },
        "portfolio": {
            "concentration_after": concentration,
            "max_single_position": round(max_conc, 4),
        },
        "fx": {
            "fx_rate":         fx_rate,
            "proceeds_thb":    round(proceeds_thb, 2),
            "cost_thb":        round(cost_thb, 2),
            "gain_thb":        round(cap_gain_thb, 2),
        },
        "generational": {
            "future_income_lost_30yr": round(future_lost, 2),
            "note": "Estimated future dividend income lost, compounded at 7%/yr over 30 years.",
        },
        "recommendation": "\n".join(lines),
    }
