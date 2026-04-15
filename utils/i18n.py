"""
utils/i18n.py — Thai / English Language Switching
=====================================================
Usage:
    from utils.i18n import t, set_lang, get_lang, lang_toggle_button
    st.subheader(t("portfolio_overview"))
    
Language is stored in st.session_state["lang"] — "en" or "th".
"""

from __future__ import annotations

# ── String table ──────────────────────────────────────────────────────────────
_STRINGS: dict[str, dict[str, str]] = {
    # ── Navigation ──────────────────────────────────────────────────────────
    "dashboard": {"en": "📊 Dashboard", "th": "📊 แดชบอร์ด"},
    "intelligence_hub": {"en": "🔍 Intelligence Hub", "th": "🔍 ศูนย์ข้อมูล"},
    "analytics_engine": {"en": "🧪 Analytics Engine", "th": "🧪 เครื่องมือวิเคราะห์"},
    "family_overview": {"en": "👨‍👩‍👧 Family Overview", "th": "👨‍👩‍👧 ภาพรวมครอบครัว"},

    # ── Dashboard tabs ───────────────────────────────────────────────────────
    "overview": {"en": "Overview", "th": "ภาพรวม"},
    "price_history": {"en": "Price & History", "th": "ราคา & ประวัติ"},
    "dividend_calendar": {"en": "Dividend Calendar", "th": "ตารางปันผล"},
    "tax_recon": {"en": "Tax & Reconciliation", "th": "ภาษี & การตรวจสอบ"},
    "transactions": {"en": "Transactions & Activity", "th": "รายการ & กิจกรรม"},

    # ── Overview metrics ─────────────────────────────────────────────────────
    "portfolio_overview": {"en": "Portfolio Overview", "th": "ภาพรวมพอร์ต"},
    "market_value_thb": {"en": "Market Value (THB)", "th": "มูลค่าตลาด (บาท)"},
    "market_value_usd": {"en": "Market Value (USD)", "th": "มูลค่าตลาด (USD)"},
    "dividends_received": {"en": "Dividends Received", "th": "ปันผลที่ได้รับ"},
    "cash_usd": {"en": "Cash (USD)", "th": "เงินสด (USD)"},
    "cash_thb": {"en": "Cash (THB)", "th": "เงินสด (บาท)"},
    "unrealised": {"en": "Unrealised P&L", "th": "กำไร/ขาดทุนที่ยังไม่รับรู้"},

    # ── Common terms ────────────────────────────────────────────────────────
    "ticker": {"en": "Ticker", "th": "หลักทรัพย์"},
    "shares": {"en": "Shares", "th": "จำนวนหุ้น"},
    "avg_cost": {"en": "Avg Cost", "th": "ต้นทุนเฉลี่ย"},
    "current": {"en": "Current", "th": "ราคาปัจจุบัน"},
    "mkt_value": {"en": "Market Value", "th": "มูลค่าตลาด"},

    # ── Dividend calendar ────────────────────────────────────────────────────
    "smart_checklist": {"en": "Dividend Calendar — Smart Checklist", "th": "ตารางปันผล — รายการตรวจสอบอัจฉริยะ"},
    "confirmed": {"en": "✅ Confirmed", "th": "✅ ยืนยันแล้ว"},
    "process_confirmed": {"en": "✅ Process Confirmed Dividends", "th": "✅ ประมวลผลปันผลที่ยืนยัน"},
    "export_ics": {"en": "📅 Export to Calendar", "th": "📅 ส่งออกไปยังปฏิทิน"},
    "received_history": {"en": "Received History", "th": "ประวัติการรับปันผล"},

    # ── Analytics tabs ───────────────────────────────────────────────────────
    "risk_optimisation": {"en": "⚡ Risk & Optimisation", "th": "⚡ ความเสี่ยง & การปรับพอร์ต"},
    "backtest": {"en": "⏱ Backtest", "th": "⏱ ทดสอบย้อนหลัง"},
    "whatif": {"en": "🔬 What-If Optimizer", "th": "🔬 เครื่องมือ What-If"},
    "monte_carlo": {"en": "📈 Monte Carlo", "th": "📈 มอนติคาร์โล"},
    "gen_plan": {"en": "🏛 Generational Plan", "th": "🏛 แผนความมั่งคั่ง 30 ปี"},
    "advanced_charts": {"en": "📉 Advanced Charts", "th": "📉 กราฟขั้นสูง"},
    "exit_simulator": {"en": "🚪 Exit Simulator", "th": "🚪 จำลองการขาย"},
    "ai_research": {"en": "🤖 AI Research", "th": "🤖 วิจัย AI"},
    "download_report": {"en": "📥 Download Report", "th": "📥 ดาวน์โหลดรายงาน"},

    # ── Buttons / actions ────────────────────────────────────────────────────
    "run": {"en": "▶ Run", "th": "▶ เรียกใช้"},
    "refresh": {"en": "🔄 Refresh All Data", "th": "🔄 รีเฟรชข้อมูล"},
    "save_trade": {"en": "💾 Save Trade", "th": "💾 บันทึกรายการ"},
    "switch_account": {"en": "Switch to this account", "th": "เปลี่ยนบัญชีนี้"},
    "log_new_trade": {"en": "Log New Trade", "th": "บันทึกรายการซื้อขาย"},
    "apply_whatif": {"en": "✅ Apply What-If", "th": "✅ ใช้ What-If"},
    "generate_report": {"en": "📊 Generate Report", "th": "📊 สร้างรายงาน"},
    "update_nav": {"en": "Update NAV", "th": "อัปเดต NAV"},

    # ── WHT reconciliation ───────────────────────────────────────────────────
    "wht_recon": {"en": "Withholding Tax Reconciliation", "th": "การตรวจสอบภาษีหัก ณ ที่จ่าย"},

    # ── Account switcher ─────────────────────────────────────────────────────
    "account": {"en": "ACCOUNT", "th": "บัญชี"},
    "data_as_of": {"en": "Data as of", "th": "ข้อมูล ณ วันที่"},
    "github_sync": {"en": "GitHub sync", "th": "ซิงค์ GitHub"},
    "ai_label": {"en": "AI", "th": "AI"},

    # ── Exit simulator ───────────────────────────────────────────────────────
    "exit_sim_title": {"en": "🚪 Exit Position Simulator", "th": "🚪 จำลองการขายหลักทรัพย์"},
    "select_position": {"en": "Select position to exit", "th": "เลือกหลักทรัพย์ที่ต้องการขาย"},
    "shares_to_sell": {"en": "Shares to sell", "th": "จำนวนหุ้นที่ขาย"},
    "exit_price": {"en": "Exit price (USD)", "th": "ราคาขาย (USD)"},
    "compute_exit": {"en": "Compute Exit Impact", "th": "คำนวณผลกระทบจากการขาย"},
    "capital_gain": {"en": "Capital Gain / Loss", "th": "กำไร/ขาดทุนจากการขาย"},
    "lost_income": {"en": "Lost Monthly Income", "th": "รายได้ปันผลที่หายไป/เดือน"},
    "portfolio_impact": {"en": "Portfolio Impact", "th": "ผลกระทบต่อพอร์ต"},

    # ── AI Research ──────────────────────────────────────────────────────────
    "ai_research_title": {"en": "🤖 AI Portfolio Research Agent", "th": "🤖 ผู้ช่วย AI วิเคราะห์พอร์ต"},
    "ask_anything": {"en": "Ask anything about your portfolio...", "th": "ถามอะไรก็ได้เกี่ยวกับพอร์ตของคุณ..."},

    # ── Language toggle ──────────────────────────────────────────────────────
    "language_toggle": {"en": "🌐 ภาษาไทย", "th": "🌐 English"},

    # ── KAsset / Finnomena ───────────────────────────────────────────────────
    "kasset_overview": {"en": "KAsset Wisdom — K-FIXED-A", "th": "กองทุน KAsset Wisdom — K-FIXED-A"},
    "nav_stale": {"en": "⚠️ NAV data may be outdated", "th": "⚠️ ข้อมูล NAV อาจไม่เป็นปัจจุบัน"},
    "manual_nav_entry": {"en": "Update NAV manually", "th": "อัปเดต NAV ด้วยตนเอง"},
    "total_fund_value": {"en": "Total Fund Value (THB)", "th": "มูลค่ากองทุนรวม (บาท)"},
}

def t(key: str, **kwargs) -> str:
    """Translate key to current language. Supports {var} interpolation."""
    try:
        import streamlit as st
        lang = st.session_state.get("lang", "en")
    except Exception:
        lang = "en"

    row = _STRINGS.get(key, {})
    text = row.get(lang) or row.get("en") or key

    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text


def get_lang() -> str:
    try:
        import streamlit as st
        return st.session_state.get("lang", "en")
    except Exception:
        return "en"


def set_lang(lang: str):
    try:
        import streamlit as st
        st.session_state["lang"] = lang
    except Exception:
        pass


def lang_toggle_button():
    """Render a compact language toggle button in Streamlit sidebar."""
    try:
        import streamlit as st
        if st.sidebar.button(t("language_toggle"), key="lang_toggle", use_container_width=True):
            current = get_lang()
            set_lang("th" if current == "en" else "en")
            st.rerun()
    except Exception:
        pass
