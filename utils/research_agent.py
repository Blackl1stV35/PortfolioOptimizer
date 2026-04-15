"""
utils/research_agent.py  —  AI Portfolio Research Agent
========================================================
Full conversational agent powered by Groq llama-3.3-70b-versatile.
Answers any portfolio, macro, tax, or strategy question with full context.
Responds in the user's chosen language (EN or TH).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Generator

log = logging.getLogger(__name__)

MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024


def _groq_client():
    key = ""
    try:
        import streamlit as st
        key = st.secrets.get("groq", {}).get("api_key", "")
    except Exception:
        pass
    key = key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    from groq import Groq
    return Groq(api_key=key)


def _build_system_prompt(cfg: dict, lang: str, fx: float, wht: float) -> str:
    """Build a rich system prompt from the live portfolio context."""
    from collections import defaultdict
    shares = defaultdict(int); cost = defaultdict(float)
    for tx in cfg.get("transactions", []):
        if tx["type"] == "BUY":
            shares[tx["ticker"]] += tx["shares"]
            cost[tx["ticker"]]   += tx["total_usd"]

    holdings_str = "\n".join(
        f"  {t}: {s} shares @ avg ${cost[t]/s:.2f}" for t, s in shares.items() if s > 0
    )
    cash      = cfg.get("cash", {}).get("usd", 0)
    strategy  = cfg.get("analysis", {}).get("strategy", "Aggressive Income Builder")
    target    = cfg.get("analysis", {}).get("target_monthly_income_usd", 200)
    meta      = cfg.get("meta", {})

    lang_instruction = (
        "Always respond in Thai (ภาษาไทย) unless the user writes in English."
        if lang == "th" else
        "Always respond in English unless the user writes in Thai."
    )

    return f"""You are an expert quantitative portfolio analyst and financial advisor \
specialising in income-focused portfolios for Thai retail investors.

PORTFOLIO CONTEXT (live data):
  Account: {meta.get('account_id', '397543-7')} — {meta.get('account_holder', '')}
  Broker: {meta.get('broker', 'K CYBER TRADE')}
  Strategy: {strategy}
  Target monthly income: ${target}/month (net after WHT)

Current Holdings:
{holdings_str}
  Cash (USD): ${cash:.2f}
  FX rate: {fx:.4f} THB/USD
  WHT: {wht*100:.0f}% (validated treaty rate)

ANALYTICAL FRAMEWORK:
  • Always reference the 30-year Monte Carlo generational plan when discussing long-term strategy
  • FX timing: mention USD/THB signal when discussing conversions
  • WHT: treaty 15% rate is validated — never suggest 30% unless specifically asked
  • Thai capital gains on foreign securities: currently EXEMPT from personal income tax
  • Income assets held: BKLN (monthly, ~7.07%), ARCC (quarterly, ~10.99%), PDI (monthly, ~15.2%)
  • ARCC Q1 2026 dividend was MISSED (bought after ex-date). Q2 2026 ex ~2026-06-12.

RESPONSE STYLE:
  • {lang_instruction}
  • Use bullet points and tables for comparisons
  • Be specific with numbers — always show USD and THB equivalents
  • For What-If questions, show ΔSharpe, ΔMonthly income, and FX impact
  • For exit questions, include tax implications, lost income, and generational impact
  • For macro questions, reference VIX, USD/THB z-score, yield curve, and regime
  • Never hallucinate NAV, prices, or filings — say "I'll need live data" if uncertain
  • End every major analysis with one clear "Recommendation" bullet
"""


def ask(
    question: str,
    history:  list[dict],
    cfg:      dict,
    lang:     str = "en",
    fx:       float = 32.68,
    wht:      float = 0.15,
) -> str:
    """
    Non-streaming ask. Returns full response string.
    history: list of {"role": "user"|"assistant", "content": str}
    """
    client = _groq_client()
    if not client:
        return (
            "❌ Groq API key not configured. "
            "Add `[groq] api_key = 'gsk_...'` to .streamlit/secrets.toml."
        )

    system = _build_system_prompt(cfg, lang, fx, wht)
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-10:])   # last 10 turns for context
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            stream=False,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        log.warning("Groq ask failed: %s", exc)
        return f"⚠️ Groq error: {exc}"


def ask_stream(
    question: str,
    history:  list[dict],
    cfg:      dict,
    lang:     str = "en",
    fx:       float = 32.68,
    wht:      float = 0.15,
) -> Generator[str, None, None]:
    """
    Streaming ask — yields text chunks for st.write_stream().
    """
    client = _groq_client()
    if not client:
        yield "❌ Groq API key not configured."
        return

    system = _build_system_prompt(cfg, lang, fx, wht)
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": question})

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as exc:
        yield f"\n⚠️ Groq error: {exc}"


# ── Suggested starter questions ───────────────────────────────────────────────
STARTER_QUESTIONS = {
    "en": [
        "📊 What is my current monthly income and how long until I reach $1,000/month?",
        "🔬 Should I add PDI or MAIN to my portfolio? Compare risk and income.",
        "🚪 What happens to my generational plan if I exit all BKLN today?",
        "🌐 Is the USD/THB rate good for converting THB to invest now?",
        "💰 How do I minimise withholding tax legally as a Thai investor?",
        "📈 Run a 30-year Monte Carlo — what's my p10/p50/p90 outcome?",
        "🏦 What macro risks should I watch for my income portfolio this quarter?",
    ],
    "th": [
        "📊 รายได้ปันผลต่อเดือนของฉันตอนนี้เท่าไหร่ และนานแค่ไหนถึงจะถึง $1,000/เดือน?",
        "🔬 ควรเพิ่ม PDI หรือ MAIN ในพอร์ต? เปรียบเทียบความเสี่ยงและรายได้",
        "🚪 ถ้าขาย BKLN ทั้งหมดวันนี้ แผน 30 ปีจะได้รับผลกระทบอย่างไร?",
        "🌐 ค่าเงิน USD/THB ตอนนี้เหมาะสมที่จะแลกซื้อหุ้นสหรัฐไหม?",
        "💰 ลดภาษีหัก ณ ที่จ่ายอย่างถูกกฎหมายในฐานะนักลงทุนไทยได้อย่างไร?",
        "📈 รัน Monte Carlo 30 ปี — ผลลัพธ์ p10/p50/p90 เป็นอย่างไร?",
        "🏦 ความเสี่ยงมหภาคอะไรที่ควรติดตามสำหรับพอร์ตรายได้ในไตรมาสนี้?",
    ],
}
