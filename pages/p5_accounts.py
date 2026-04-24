"""pages/p5_accounts.py — Cross-account manager (Phase 2+)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import (load_accounts, load_cfg, save_cfg, derive_holdings,
                  _latest_snapshot, t, get_lang, _FINNOMENA, get_kfixed_market_value)

YIELDS = {"BKLN":0.0707,"ARCC":0.1066,"PDI":0.152,"MAIN":0.076,"HTGC":0.12}


def render(*, active, cfg, holdings, fx_r, wht, rf_ann, tickers):
    st.subheader("👥 Account Manager")
    st.caption("Consolidated view and settings across all accounts.")

    all_accounts = load_accounts()

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []; total_thb = 0.0; total_inc = 0.0
    for acct in all_accounts:
        acfg  = load_cfg(acct["id"])
        ahold = derive_holdings(acfg)
        afx   = float(acfg.get("meta",{}).get("fx_usd_thb", 32.68))
        awht  = float(acfg.get("settings",{}).get("wht_active", 0.15))
        atype = acct.get("account_type","")

        # Market value
        if "mutual fund" in atype.lower():
            amkt_thb = get_kfixed_market_value(acfg) if _FINNOMENA else 0
            amkt_thb = amkt_thb or sum(float(i.get("market_value_thb",0))
                                       for i in acfg.get("investments",[]))
            acash_thb = float(acfg.get("cash",{}).get("thb",0))
            asnap_thb = amkt_thb
        else:
            snap      = _latest_snapshot(acfg)
            asnap_thb = snap.get("market_value_thb",
                                 snap.get("market_value_usd",0)*afx)
            acash_thb = float(acfg.get("cash",{}).get("thb",0)) + \
                        float(acfg.get("cash",{}).get("usd",0))*afx

        # Income
        ainc = 0.0
        for tkr, h in ahold.items():
            y = YIELDS.get(tkr,0.08)
            ainc += h["shares"] * h["avg_cost"] * y / 12 * (1-awht)

        total_thb += asnap_thb; total_inc += ainc
        rows.append({
            "ID":             acct["id"],
            "Name":           acct.get("display_name",""),
            "Type":           atype,
            "Strategy":       acct.get("strategy",""),
            "Holdings":       ", ".join(ahold.keys()) if ahold else "—",
            "Value (฿)":      asnap_thb,
            "Cash (฿)":       acash_thb,
            "Income/mo ($)":  round(ainc, 2),
            "WHT %":          awht*100,
            "Base CCY":       acct.get("base_currency","USD"),
        })

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Family NAV (฿)",  f"฿{total_thb:,.0f}")
    c2.metric("Combined Income/mo ($)", f"${total_inc:.2f}")
    c3.metric("Annual income est.",     f"${total_inc*12:,.0f}")
    c4.metric("Accounts",               str(len(all_accounts)))
    st.divider()

    # Account table
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        column_config={
            "Value (฿)":     st.column_config.NumberColumn(format="฿%.0f"),
            "Cash (฿)":      st.column_config.NumberColumn(format="฿%.0f"),
            "Income/mo ($)": st.column_config.NumberColumn(format="$%.2f"),
            "WHT %":         st.column_config.NumberColumn(format="%.0f%%"),
        },
        width="stretch", hide_index=True,
    )

    # Income bar chart
    if len(rows) > 1:
        fig = go.Figure(go.Bar(
            x=[r["ID"] for r in rows],
            y=[r["Income/mo ($)"] for r in rows],
            marker_color=[a.get("accent","#2E5BA8") for a in all_accounts],
            text=[f"${r['Income/mo ($)']:.2f}" for r in rows],
            textposition="auto",
        ))
        fig.update_layout(title="Monthly income by account", height=240,
                          margin=dict(l=0,r=0,t=30,b=0), yaxis_title="USD/mo",
                          plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                          font_color="#FFFFFF")
        st.plotly_chart(fig, width="stretch")

    # ── Account settings editor ────────────────────────────────────────────────
    st.divider()
    st.subheader("Account settings")
    sel_id = st.selectbox("Edit account", [a["id"] for a in all_accounts],
                           key="acct_settings_sel")
    sel_cfg = load_cfg(sel_id)
    with st.expander(f"Settings — {sel_id}", expanded=True):
        col1, col2 = st.columns(2)
        new_fx  = col1.number_input("FX rate (THB/USD)", 20.0, 50.0,
                                     float(sel_cfg.get("meta",{}).get("fx_usd_thb",32.68)),
                                     0.01, format="%.4f", key="s_fx")
        new_wht = col2.number_input("WHT rate (%)", 0.0, 30.0,
                                     float(sel_cfg.get("settings",{}).get("wht_active",0.15))*100,
                                     0.5, format="%.1f", key="s_wht")
        new_rf  = col1.number_input("Risk-free rate (%)", 0.0, 10.0,
                                     float(sel_cfg.get("settings",{}).get("risk_free_rate_annual",0.045))*100,
                                     0.1, format="%.2f", key="s_rf")
        new_target = col2.number_input("Target income/mo (USD)", 0, 10000,
                                        int(sel_cfg.get("analysis",{}).get("target_monthly_income_usd",200)),
                                        50, key="s_tgt")

        if st.button("💾 Save settings", key="save_settings"):
            sel_cfg.setdefault("meta",{})["fx_usd_thb"] = round(new_fx, 4)
            sel_cfg.setdefault("settings",{})["wht_active"]             = round(new_wht/100, 4)
            sel_cfg["settings"]["wht_default"]                           = round(new_wht/100, 4)
            sel_cfg["settings"]["risk_free_rate_annual"]                 = round(new_rf/100, 4)
            sel_cfg.setdefault("analysis",{})["target_monthly_income_usd"] = float(new_target)
            save_cfg(sel_cfg, sel_id)
            st.success("Settings saved."); st.rerun()

    # ── Cash transfer recorder ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Cash transfer (record only — no broker API)")
    st.caption("Record a transfer between accounts for your own bookkeeping.")
    t1,t2 = st.columns(2)
    from_id = t1.selectbox("From account", [a["id"] for a in all_accounts], key="tr_from")
    to_id   = t2.selectbox("To account",   [a["id"] for a in all_accounts], key="tr_to")
    tr_amt  = st.number_input("Amount (฿ if THB account, $ if USD)", 0.0, 1e8, 0.0, 100.0, key="tr_amt")
    tr_note = st.text_input("Note", key="tr_note")
    if st.button("📋 Record transfer", key="save_transfer") and tr_amt > 0 and from_id != to_id:
        from datetime import date as _date
        for aid, sign in [(from_id,-1),(to_id,+1)]:
            _c = load_cfg(aid)
            _c.setdefault("transfer_log",[]).append({
                "date": str(_date.today()), "amount": tr_amt*sign,
                "counterpart": to_id if sign==-1 else from_id,
                "note": tr_note,
            })
            save_cfg(_c, aid)
        st.success(f"Transfer of {tr_amt:,.2f} recorded.")
        st.rerun()
