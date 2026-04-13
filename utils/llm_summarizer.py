"""
utils/llm_summarizer.py  —  Groq LLM Executive Summary Layer
=============================================================
Generates concise, finance-specific plain-English summaries after every
major analysis in the Analytics Engine tabs.

Model : llama-3.3-70b-versatile  (best free Groq tier for quant finance)
Cache : in-memory (st.session_state) keyed by SHA-256 of input data.
        Prevents redundant API calls within the same session.

Config (.streamlit/secrets.toml):
    [groq]
    api_key = "gsk_..."

Fallback: if Groq is unreachable or unconfigured, returns a plain-text
          summary built deterministically from the numbers — the UI never
          shows an error to the user.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS  = 480
TEMPERATURE = 0.15   # Low = consistent, reproducible financial reasoning

_SYSTEM = (
    "You are a senior quantitative portfolio analyst specialising in income-focused "
    "US-listed instruments (BDCs, CLO ETFs, CEFs) held by a Thai retail investor. "
    "Write in plain English. Be direct. Use 4–6 bullet points maximum. "
    "Always include: (1) the key positive or negative delta, (2) the main risk, "
    "(3) one concrete actionable recommendation. No markdown headers. No JSON."
)


# ── API client ────────────────────────────────────────────────────────────────
def _client():
    key = ""
    try:
        import streamlit as st
        key = st.secrets.get("groq", {}).get("api_key", "")
    except Exception:
        pass
    key = key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception as exc:
        log.debug("Groq init failed: %s", exc)
        return None


def available() -> bool:
    return _client() is not None


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_get(key: str) -> str | None:
    try:
        import streamlit as st
        return st.session_state.get(f"_groq_{key}")
    except Exception:
        return None


def _cache_set(key: str, text: str):
    try:
        import streamlit as st
        st.session_state[f"_groq_{key}"] = text
    except Exception:
        pass


# ── Core API call ─────────────────────────────────────────────────────────────
def _call(user_prompt: str) -> str | None:
    c = _client()
    if not c:
        return None
    try:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("Groq call failed: %s", exc)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Public summary functions — one per analysis type
# ═════════════════════════════════════════════════════════════════════════════

def summarise_whatif(result: dict, force: bool = False) -> str:
    """Executive summary for a What-If scenario result dict."""
    cache_key = _hash({
        "t": "whatif",
        "delta": result.get("delta", {}),
        "income_delta": result.get("income_delta_usd"),
        "tickers_after": sorted(result.get("tickers_after", [])),
    })
    if not force:
        cached = _cache_get(cache_key)
        if cached:
            return cached

    delta = result.get("delta", {})
    wt_a  = result.get("after_weights", {})
    max_w = max(wt_a.values(), default=0)

    prompt = (
        f"What-If portfolio scenario — adding {result.get('tickers_after',[])} "
        f"to existing {result.get('tickers_before',[])}:\n"
        f"• ΔSharpe: {delta.get('sharpe',0):+.3f}\n"
        f"• ΔAnn. Return: {delta.get('ann_return',0)*100:+.2f}%\n"
        f"• ΔVolatility: {delta.get('ann_vol',0)*100:+.2f}%\n"
        f"• ΔCVaR 95%: {delta.get('cvar_95',0)*100:+.2f}%\n"
        f"• ΔMonthly dividend income: ${result.get('income_delta_usd',0):+.2f}/month\n"
        f"• New monthly income: ${result.get('income_after_usd',0):.2f}/month\n"
        f"• Largest single position after: {max_w:.1%}\n"
        f"• New weights: {json.dumps({k: f'{v:.1%}' for k,v in wt_a.items()})}\n\n"
        f"Summarise in 4–6 bullets: income impact, risk/return trade-off, "
        f"concentration risk, macro fit for a Thai income investor, and recommendation."
    )
    text = _call(prompt) or _fallback_whatif(result)
    _cache_set(cache_key, text)
    return text


def summarise_risk(risk_dict: dict, weights: dict) -> str:
    """Summary for Risk & Optimisation tab."""
    cache_key = _hash({"t": "risk", "w": weights, "sharpes": {k: v.get("Sharpe Ratio") for k,v in risk_dict.items()}})
    cached = _cache_get(cache_key)
    if cached:
        return cached

    rows = "\n".join(
        f"  {t}: Sharpe={m.get('Sharpe Ratio',0):.3f}  "
        f"Ann.Ret={m.get('Ann. Return',0)*100:.2f}%  "
        f"CVaR={m.get('CVaR 95%',0)*100:.2f}%  "
        f"MaxDD={m.get('Max Drawdown',0)*100:.2f}%"
        for t, m in risk_dict.items()
    )
    wt_str = json.dumps({k: f"{v:.1%}" for k,v in weights.items()})
    prompt = (
        f"Portfolio risk metrics:\n{rows}\n"
        f"Optimal weights (Max Sharpe): {wt_str}\n\n"
        f"Summarise in 4–6 bullets: risk/return quality, diversification, "
        f"whether concentration is appropriate for a Thai income investor, "
        f"and any single-position concerns."
    )
    text = _call(prompt) or "• Risk metrics computed — review Sharpe and CVaR columns for detail."
    _cache_set(cache_key, text)
    return text


def summarise_backtest(bt_metrics: dict) -> str:
    """Summary for Backtest tab."""
    cache_key = _hash({"t": "bt", "m": bt_metrics})
    cached = _cache_get(cache_key)
    if cached:
        return cached

    rows = "\n".join(
        f"  {s}: Sharpe={m.get('sharpe',0):.3f}  Ann.Ret={m.get('ann_return',0)*100:.2f}%  "
        f"MaxDD={m.get('max_drawdown',0)*100:.2f}%  Final $1=${m.get('final_equity',1):.3f}"
        for s, m in bt_metrics.items()
    )
    prompt = (
        f"Walk-forward backtest (out-of-sample) results:\n{rows}\n\n"
        f"Summarise: which strategy performed best, whether outperformance looks "
        f"consistent or luck-driven, and whether it's suitable for a long-term "
        f"Thai income portfolio."
    )
    text = _call(prompt) or "• Backtest complete — compare Sharpe ratios to identify the best risk-adjusted strategy."
    _cache_set(cache_key, text)
    return text


def summarise_monte_carlo(vp_last: dict, n_years: int, target: float) -> str:
    """Summary for Monte Carlo tab."""
    cache_key = _hash({"t": "mc", "vp": vp_last, "yrs": n_years, "tgt": target})
    cached = _cache_get(cache_key)
    if cached:
        return cached

    prompt = (
        f"Monte Carlo projection — {n_years}-year horizon:\n"
        f"• Median (p50) final portfolio: ${vp_last.get('p50',0):,.0f}\n"
        f"• Pessimistic (p10): ${vp_last.get('p10',0):,.0f}\n"
        f"• Optimistic (p90): ${vp_last.get('p90',0):,.0f}\n"
        f"• Implied median income/mo: ${vp_last.get('p50',0)*0.0707/12:,.2f}\n"
        f"• Target income/mo: ${target:.0f}\n\n"
        f"Summarise: likelihood of meeting the target, downside risk implications, "
        f"and one key recommendation for the Thai income investor."
    )
    text = _call(prompt) or _fallback_mc(vp_last, n_years, target)
    _cache_set(cache_key, text)
    return text


def summarise_generational(milestones: dict, mtt: dict, target: float) -> str:
    """Summary for 30-Year Generational Plan tab."""
    cache_key = _hash({"t": "gen", "mtt": mtt, "tgt": target, "ms_keys": list(milestones.keys())})
    cached = _cache_get(cache_key)
    if cached:
        return cached

    ms_str = "\n".join(
        f"  Year {yr}: p50 ฿{ms.get('p50_value',0):,.0f}  "
        f"income ${ms.get('p50_income_m',0):.2f}/mo  "
        f"P(>target)={ms.get('prob_above_target',0):.1f}%"
        for yr, ms in milestones.items()
    )
    prompt = (
        f"30-year generational plan for Thai income investor:\n"
        f"Target: ${target:.0f}/month passive income\n"
        f"Median time to target: {mtt.get('years','?')} years {mtt.get('extra_months',0)} months\n"
        f"Milestones:\n{ms_str}\n\n"
        f"Summarise: whether the plan is on track, the biggest risk to the thesis, "
        f"and one key action to accelerate the outcome."
    )
    text = _call(prompt) or f"• Generational plan computed. Median time to ${target:.0f}/mo: {mtt.get('years','?')} years."
    _cache_set(cache_key, text)
    return text


# ── Deterministic fallbacks ───────────────────────────────────────────────────
def _fallback_whatif(r: dict) -> str:
    d  = r.get("delta", {})
    sh = d.get("sharpe", 0)
    inc= r.get("income_delta_usd", 0)
    return (
        f"• ΔSharpe {sh:+.3f}: risk-adjusted return {'improved' if sh > 0 else 'decreased'}.\n"
        f"• Monthly income change: ${inc:+.2f}/month.\n"
        f"• CVaR impact: {d.get('cvar_95',0)*100:+.2f}% (higher = more tail risk).\n"
        f"• Check max single-position weight does not exceed 35%.\n"
        f"• Proceed if income delta is positive and Sharpe improves; otherwise reconsider entry timing.\n"
        f"  [AI summary unavailable — Groq API key not configured]"
    )


def _fallback_mc(vp: dict, n_years: int, target: float) -> str:
    p50 = vp.get("p50", 0)
    return (
        f"• Median portfolio after {n_years} years: ${p50:,.0f}.\n"
        f"• Implied median income: ${p50*0.0707/12:,.2f}/month.\n"
        f"• Target ${target:.0f}/month {'likely' if p50*0.0707/12 >= target else 'not yet'} reachable at this horizon.\n"
        f"  [AI summary unavailable — Groq API key not configured]"
    )


# ── Streamlit render helper ───────────────────────────────────────────────────
def render_summary(text: str, context_key: str, regenerate_fn=None):
    """
    Render a summary inside a Streamlit expander with optional regenerate button.
    Call this after any analysis display block.
    """
    import streamlit as st
    with st.expander("🤖 AI Executive Summary", expanded=True):
        if text:
            st.info(text)
        else:
            st.caption("Add Groq API key to .streamlit/secrets.toml to enable AI summaries.")
        if regenerate_fn and available():
            if st.button("♻ Regenerate", key=f"regen_{context_key}"):
                with st.spinner("Calling Groq LLM..."):
                    new_text = regenerate_fn(force=True)
                st.rerun()
