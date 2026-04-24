"""pages/p6_research.py — AI Portfolio Research Agent."""
from __future__ import annotations

import streamlit as st

from core import t, get_lang, _groq_available


def render(*, active, cfg, holdings, fx_r, wht, rf_ann, tickers):
    st.subheader("🤖 " + t("ai_research_title"))
    st.caption(
        "Ask anything about your portfolio, macro, tax, exits, or strategy. "
        "Responds in English or Thai based on your language setting."
    )

    if not _groq_available():
        st.error("Groq API key not configured. "
                 "Add `[groq] api_key = 'gsk_...'` to .streamlit/secrets.toml.")
        return

    from utils.research_agent import STARTER_QUESTIONS, ask_stream

    lang = get_lang()

    # Session chat history
    if "ai_history" not in st.session_state:
        st.session_state["ai_history"] = []

    # Starter questions
    with st.expander("💡 Quick start questions — click to ask",
                     expanded=len(st.session_state["ai_history"]) == 0):
        starters = STARTER_QUESTIONS.get(lang, STARTER_QUESTIONS["en"])
        for sq in starters:
            if st.button(sq, key=f"sq_{hash(sq)}", use_container_width=True):
                st.session_state["ai_history"].append({"role":"user","content":sq})
                st.rerun()

    # Render chat history
    for msg in st.session_state["ai_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    user_input = st.chat_input(t("ask_anything"))
    if user_input:
        st.session_state["ai_history"].append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            response = st.write_stream(ask_stream(
                question=user_input,
                history=st.session_state["ai_history"][:-1],
                cfg=cfg, lang=lang, fx=fx_r, wht=wht,
            ))
        st.session_state["ai_history"].append({"role":"assistant","content":response})

    # Controls
    col1, col2 = st.columns(2)
    if st.session_state["ai_history"]:
        if col1.button("🗑 Clear conversation", key="clear_ai"):
            st.session_state["ai_history"] = []
            st.rerun()
