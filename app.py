"""
app.py  —  PortfolioOptimizer  v4
==================================
Entry point only. All logic lives in pages/*.py
Deploy: streamlit run app.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from core import render_sidebar, t

st.set_page_config(
    page_title="PortfolioOptimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's auto-nav (we use our own)
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar renders once, returns context ─────────────────────────────────────
active, cfg, holdings, fx_r, wht, rf_ann, tickers = render_sidebar()

# ── Page routing ──────────────────────────────────────────────────────────────
from core import load_accounts
_all = load_accounts()

PAGES = {
    "📊 " + t("dashboard"):        "pages.p1_dashboard",
    "🔍 " + t("intelligence_hub"): "pages.p2_intelligence",
    "🧪 " + t("analytics_engine"): "pages.p3_analytics",
    "🔬 " + t("whatif"):           "pages.p4_sandbox",
    "👥 " + t("account"):          "pages.p5_accounts",
    "🤖 " + t("ai_research"):      "pages.p6_research",
}
if len(_all) > 1:
    PAGES["👨‍👩‍👧 " + t("family_overview")] = "pages.p7_family"

with st.sidebar:
    st.markdown("---")
    page = st.radio(
        "nav",
        list(PAGES.keys()),
        label_visibility="collapsed",
        key="main_nav",
    )

# ── Dynamic page import ───────────────────────────────────────────────────────
import importlib
mod = importlib.import_module(PAGES[page])
mod.render(active=active, cfg=cfg, holdings=holdings,
           fx_r=fx_r, wht=wht, rf_ann=rf_ann, tickers=tickers)
