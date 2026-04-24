"""pages/p1_dashboard.py — Dashboard page (re-runs ONLY when user is here)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from core import (load_prices, load_fx_series, save_cfg, github_push,
                  derive_holdings, _latest_snapshot, load_cfg, load_accounts,
                  t, get_lang, _FINNOMENA, get_nav, get_kfixed_market_value)


def render(*, active, cfg, holdings, fx_r, wht, rf_ann, tickers):
    acct_type = active.get("account_type", "")
    lang      = get_lang()

    tab_ov, tab_ph, tab_dc, tab_wht, tab_tx = st.tabs([
        t("overview"), t("price_history"), t("dividend_calendar"),
        t("tax_recon"), t("transactions"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ov:
        st.subheader(t("portfolio_overview"))

        if "mutual fund" in acct_type.lower():
            _render_kasset_overview(cfg, fx_r, active)
            return

        snap    = _latest_snapshot(cfg)
        mkt_usd = snap.get("market_value_usd", 0)
        mkt_thb = snap.get("market_value_thb", mkt_usd * fx_r)
        unr_thb = snap.get("unrealized_thb", snap.get("unrealized_usd", 0) * fx_r)
        div_thb = snap.get("total_dividends_thb", 0)
        cash_u  = cfg.get("cash", {}).get("usd", 0)
        cash_b  = cfg.get("cash", {}).get("thb", 0)

        c1, c2, c3, c4 = st.columns(4)
        if cfg.get("meta", {}).get("base_currency") == "THB" and mkt_usd == 0:
            c1.metric(t("cash_thb"),        f"฿{cash_b:,.2f}")
            c2.metric(t("market_value_thb"), f"฿{mkt_thb:,.0f}")
            c3.metric(t("dividends_received"), f"฿{div_thb:,.2f}")
            c4.metric(t("market_value_usd"), f"${mkt_thb/fx_r:,.0f}" if fx_r else "—")
        else:
            c1.metric(t("market_value_thb"), f"฿{mkt_thb:,.0f}",
                      f"฿{unr_thb:+,.0f} {t('unrealised')}")
            c2.metric(t("market_value_usd"), f"${mkt_usd:,.0f}")
            c3.metric(t("dividends_received"), f"฿{div_thb:,.2f}")
            c4.metric(t("cash_usd"),          f"${cash_u:.2f}")

        if tickers:
            prices = load_prices(tickers)
            rows = []
            for tkr, h in holdings.items():
                curr = float(prices[tkr].iloc[-1]) if tkr in prices.columns else 0
                mkt  = h["shares"] * curr
                unr  = mkt - h["total_cost"]
                pct  = unr / h["total_cost"] * 100 if h["total_cost"] else 0
                rows.append({
                    t("ticker"):    tkr,
                    t("shares"):    h["shares"],
                    t("avg_cost"):  round(h["avg_cost"], 2),
                    t("current"):   round(curr, 2),
                    t("mkt_value"): round(mkt, 2),
                    "P&L $":        round(unr, 2),
                    "P&L ฿":        round(unr * fx_r, 0),
                    "P&L %":        round(pct, 2),
                })
            st.dataframe(
                pd.DataFrame(rows),
                column_config={
                    "P&L $":  st.column_config.NumberColumn(format="$%.2f"),
                    "P&L ฿":  st.column_config.NumberColumn(format="฿%.0f"),
                    "P&L %":  st.column_config.NumberColumn(format="%.2f%%"),
                    t("avg_cost"): st.column_config.NumberColumn(format="$%.2f"),
                    t("current"):  st.column_config.NumberColumn(format="$%.2f"),
                    t("mkt_value"):st.column_config.NumberColumn(format="$%.2f"),
                },
                width="stretch", hide_index=True, use_container_width=False,
            )
            if "ARCC" in holdings:
                st.warning(
                    "⚠️ ARCC Q1 2026 dividend **MISSED** (bought 2026-03-25, "
                    "ex-date 2026-03-12).  "
                    "Next: Q2 2026 est. ex ~2026-06-12 · 133sh × $0.48 = **$63.84 gross**"
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — PRICE & HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ph:
        if not tickers:
            st.info("No holdings to chart."); return

        prices = load_prices(tickers)
        _tf = st.selectbox("Timeframe", ["1M","3M","6M","1Y","3Y","MAX"],
                           index=2, key="dash_tf")
        _tf_map = {"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","3Y":"3y","MAX":"max"}

        prices_tf = load_prices(tickers, period=_tf_map[_tf])
        _accent   = {t: active.get("accent","#2E5BA8") for t in tickers}
        _colors   = {"BKLN":"#2E5BA8","ARCC":"#C0392B","PDI":"#7B2FBE"}

        fig = go.Figure()
        for tkr in tickers:
            if tkr in prices_tf.columns:
                p = prices_tf[tkr].dropna()
                fig.add_trace(go.Scatter(
                    x=p.index, y=p.values, name=tkr,
                    line_color=_colors.get(tkr, _accent.get(tkr,"#888")),
                    line_width=2,
                ))
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation="h"), yaxis_title="Price (USD)",
                          plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                          font_color="#FFFFFF")
        st.plotly_chart(fig, width="stretch")

        monthly = prices.resample("ME").last().pct_change().dropna()
        if not monthly.empty:
            st.markdown("**Monthly Returns (%)**")
            fig2 = go.Figure()
            for tkr in monthly.columns:
                fig2.add_trace(go.Bar(
                    x=monthly.index, y=monthly[tkr]*100, name=tkr,
                    marker_color=_colors.get(tkr,"#888"),
                ))
            fig2.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
                               barmode="group", yaxis_title="%",
                               plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                               font_color="#FFFFFF")
            st.plotly_chart(fig2, width="stretch")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — DIVIDEND CALENDAR
    # ══════════════════════════════════════════════════════════════════════════
    with tab_dc:
        st.subheader(t("smart_checklist"))
        st.caption("Tick dividends when received, click Process to auto-update portfolio.yaml")

        instruments = cfg.get("instruments", {})
        up_rows = []
        for tkr, inst in instruments.items():
            sh = holdings.get(tkr, {}).get("shares", 0)
            if not sh:
                continue
            for u in inst.get("dividend_policy", {}).get("estimated_upcoming", []):
                elig  = u.get("eligible_for_our_shares", True)
                amt   = float(u.get("amount", 0))
                gross = sh * amt if elig else 0.0
                up_rows.append({
                    "✅":         False,
                    "Ticker":     tkr,
                    "Period":     str(u.get("period", u.get("ex",""))),
                    "Ex-date":    str(u.get("ex","")),
                    "Pay-date":   str(u.get("pay","")),
                    "Shares":     sh if elig else 0,
                    "$/share":    amt,
                    "Gross $":    round(gross, 2),
                    f"Net @{wht*100:.0f}%": round(gross*(1-wht), 2),
                    "KS THB":     np.nan,
                    "Status":     "projected" if elig else "MISSED",
                })

        if up_rows:
            edited = st.data_editor(
                pd.DataFrame(up_rows),
                column_config={
                    "✅":      st.column_config.CheckboxColumn("✅ Confirmed"),
                    "KS THB": st.column_config.NumberColumn("KS THB (actual)", format="฿%.2f"),
                    "Gross $":st.column_config.NumberColumn(format="$%.2f"),
                    f"Net @{wht*100:.0f}%": st.column_config.NumberColumn(format="$%.2f"),
                },
                disabled=["Ticker","Period","Ex-date","Pay-date","Shares","$/share","Status"],
                width="stretch", hide_index=True, key="div_editor",
            )

            if st.button(t("process_confirmed"), type="primary"):
                confirmed = edited[edited["✅"] == True]
                if confirmed.empty:
                    st.warning("No dividends ticked.")
                else:
                    new_cfg = dict(cfg)
                    for _, row in confirmed.iterrows():
                        new_cfg.setdefault("dividends_received", []).append({
                            "period":               str(row["Period"]),
                            "ticker":               str(row["Ticker"]),
                            "shares_eligible":      int(row["Shares"]),
                            "ex_date":              str(row["Ex-date"]),
                            "pay_date":             str(row["Pay-date"]),
                            "amount_per_share_usd": float(row["$/share"]),
                            "gross_usd_estimated":  float(row["Gross $"]),
                            "wht_rate_assumed":     float(wht),
                            "net_usd_estimated":    float(row[f"Net @{wht*100:.0f}%"]),
                            "thb_ks_app":           None if pd.isna(row["KS THB"]) else float(row["KS THB"]),
                            "status":               "received",
                        })
                    save_cfg(new_cfg, active["id"])
                    gh = github_push(new_cfg, "dividend confirmed")
                    gh_msg = "✅ GitHub" if gh["success"] else f"⚠️ {gh.get('error','')[:60]}"
                    st.success(f"✅ {len(confirmed)} dividend(s) saved.  {gh_msg}")
                    st.rerun()

        # ── ICS export ────────────────────────────────────────────────────────
        st.divider()
        st.markdown(f"**{t('export_ics')}**")
        _ics_filter = st.radio("Include", ["Confirmed + Upcoming", "Upcoming only"],
                               horizontal=True, key="ics_filter")
        if st.button("📥 Generate .ics", type="secondary"):
            _bytes = _build_ics(cfg, holdings, wht, _ics_filter, active["id"])
            fname  = f"dividends_{active['id']}_{date.today().strftime('%Y%m%d')}.ics"
            st.download_button(f"⬇️ Download {fname}", _bytes, fname, "text/calendar")

        st.divider()
        st.markdown(f"**{t('received_history')}**")
        hist = cfg.get("dividends_received", [])
        if hist:
            hdf = pd.DataFrame(hist)
            # Clean KS THB column for PyArrow
            if "thb_ks_app" in hdf.columns:
                hdf["thb_ks_app"] = pd.to_numeric(hdf["thb_ks_app"], errors="coerce")
            st.dataframe(hdf, width="stretch", hide_index=True)
        else:
            st.info("No dividends recorded yet.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — WHT RECONCILIATION
    # ══════════════════════════════════════════════════════════════════════════
    with tab_wht:
        st.subheader(t("wht_recon"))
        try:
            from engine.wht_reconciliation import build_reconciliation, summarise_wht
            records = build_reconciliation(cfg, fx_r)
            summary = summarise_wht(records, cfg)
            verdict_map = {"treaty_15":"success","default_30":"error",
                           "partial":"warning","no_data":"info","overpaid":"error"}
            primary = (max(set([r.verdict for r in records]),
                          key=[r.verdict for r in records].count)
                       if records else "no_data")
            getattr(st, verdict_map.get(primary,"info"))(summary)

            if records:
                rows = [{
                    "Period":      r.period,
                    "Ticker":      r.ticker,
                    "Gross $":     round(r.gross_usd, 4),
                    "Net @30%":    round(r.net_30pct, 4),
                    "Net @15%":    round(r.net_15pct, 4),
                    "KS THB":      float(r.ks_thb) if r.ks_thb is not None else np.nan,
                    "Implied WHT": f"{r.implied_wht*100:.1f}%" if r.implied_wht else "n/a",
                    "Verdict":     r.verdict,
                } for r in records]
                df_wht = pd.DataFrame(rows)
                df_wht["KS THB"] = pd.to_numeric(df_wht["KS THB"], errors="coerce")
                st.dataframe(df_wht, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"WHT module error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — TRANSACTIONS (structured wizard)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_tx:
        _render_transaction_wizard(cfg, active, holdings, fx_r)


# ─────────────────────────────────────────────────────────────────────────────
# SUB-RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def _render_kasset_overview(cfg, fx_r, active):
    kfixed_mkt = 0.0
    nav_info   = {}
    if _FINNOMENA:
        nav_info   = get_nav("KFIXEDA", cfg)
        kfixed_mkt = get_kfixed_market_value(cfg)
    if not kfixed_mkt:
        kfixed_mkt = sum(float(inv.get("market_value_thb",0))
                         for inv in cfg.get("investments",[]))
    if nav_info.get("stale"):
        st.warning(t("nav_stale") + f" — last: {nav_info.get('date','?')}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(t("total_fund_value"), f"฿{kfixed_mkt:,.2f}")
    c2.metric("NAV/unit", f"฿{nav_info['nav']:.4f}" if nav_info.get("nav") else "Manual")
    c3.metric("Source", nav_info.get("source","manual").replace("_"," ").title())
    c4.metric(t("cash_thb"), f"฿{cfg.get('cash',{}).get('thb',0):,.2f}")

    for inv in cfg.get("investments",[]):
        st.markdown(f"**{inv.get('fund_code','')}** — {inv.get('description','')}")
        i1,i2,i3 = st.columns(3)
        i1.metric("Market value", f"฿{inv.get('market_value_thb',0):,.2f}")
        i2.metric("Units held", str(inv.get("units_held") or "—"))
        i3.metric("Updated", str(inv.get("last_manual_update","—")))

    st.divider()
    with st.expander(t("manual_nav_entry")):
        with st.form("nav_form"):
            nu_nav = st.number_input("NAV/unit (฿)", min_value=0.0001, format="%.4f", value=10.0)
            nu_mv  = st.number_input("Market value (฿)", min_value=0.0, value=float(kfixed_mkt))
            if st.form_submit_button(t("update_nav"), type="primary"):
                from utils.finnomena import update_nav_in_yaml
                cfg2 = update_nav_in_yaml(cfg, "KFIXEDA", nu_nav, nu_mv)
                save_cfg(cfg2, active["id"])
                st.success(f"Updated: ฿{nu_nav:.4f}  |  ฿{nu_mv:,.2f}")
                st.rerun()


def _render_transaction_wizard(cfg, active, holdings, fx_r):
    st.subheader(t("log_new_trade"))

    # ── Ledger ────────────────────────────────────────────────────────────────
    txns = cfg.get("transactions", [])
    if txns:
        st.markdown("**Transaction history** (editable)")
        txn_df = pd.DataFrame(txns)
        # Convert date strings → datetime so DateColumn works;
        # keep all other string cols as-is for compatibility.
        if "date" in txn_df.columns:
            txn_df["date"] = pd.to_datetime(txn_df["date"], errors="coerce")
        edited_txns = st.data_editor(
            txn_df,
            column_config={
                "date":          st.column_config.DateColumn("Date"),
                "type":          st.column_config.SelectboxColumn("Type", options=["BUY","SELL","DRIP"]),
                "ticker":        st.column_config.TextColumn("Ticker"),
                "shares":        st.column_config.NumberColumn("Shares", format="%d"),
                "price_usd":     st.column_config.NumberColumn("Price $", format="$%.4f"),
                "commission_usd":st.column_config.NumberColumn("Commission $", format="$%.2f"),
                "total_usd":     st.column_config.NumberColumn("Total $", format="$%.2f"),
            },
            num_rows="dynamic",
            width="stretch", hide_index=True, key="txn_editor",
        )
        if st.button("💾 Save transaction changes", key="save_txns"):
            cfg2 = dict(cfg)
            rows = edited_txns.copy()
            # Convert datetime back to ISO date strings for YAML storage
            if "date" in rows.columns:
                rows["date"] = rows["date"].astype(str).str[:10]
            cfg2["transactions"] = rows.to_dict("records")
            save_cfg(cfg2, active["id"])
            github_push(cfg2, "transactions edited")
            st.success("Transactions saved."); st.rerun()

    st.divider()

    # ── Step-by-step wizard ───────────────────────────────────────────────────
    st.markdown("**Add new transaction**")
    instruments = cfg.get("instruments", {})
    known_tickers = list(instruments.keys()) + ["BKLN","ARCC","PDI","MAIN","HTGC","PFLT"]

    col1, col2 = st.columns([1,2])
    with col1:
        tx_type   = st.selectbox("Type", ["BUY","SELL","DRIP"], key="tx_type")
        tx_ticker = st.selectbox("Ticker", known_tickers, key="tx_ticker")
        tx_date   = st.date_input("Date", value=date.today(), key="tx_date")

    with col2:
        tx_shares = st.number_input("Shares", min_value=1, value=50, step=1, key="tx_shares")
        # Auto-fetch live price
        try:
            import yfinance as yf
            _live = yf.download(tx_ticker, period="1d", progress=False)
            _live_price = float(_live["Close"].iloc[-1]) if not _live.empty else 0.0
        except Exception:
            _live_price = 0.0
        tx_price  = st.number_input("Price (USD)", min_value=0.0001,
                                     value=round(_live_price or 20.0, 4),
                                     format="%.4f", key="tx_price")
        tx_comm   = st.number_input("Commission (USD)", min_value=0.0,
                                     value=5.34, format="%.2f", key="tx_comm")

    # Live preview
    gross = tx_shares * tx_price
    total = gross + tx_comm if tx_type == "BUY" else gross - tx_comm
    st.info(f"**Preview:** {tx_type} {tx_shares} × {tx_ticker} @ ${tx_price:.4f}  "
            f"— Gross **${gross:,.2f}**  |  Total **${total:,.2f}**  "
            f"|  THB equiv. **฿{total*fx_r:,.0f}**")

    tx_note = st.text_input("Note (optional)", key="tx_note")

    if st.button("✅ Confirm & Save", type="primary", key="save_tx"):
        nid    = f"T{len(txns)+1:03d}"
        new_tx = {
            "id": nid, "date": str(tx_date), "type": tx_type,
            "ticker": tx_ticker, "exchange": instruments.get(tx_ticker,{}).get("exchange","ARCX"),
            "currency": "USD", "shares": int(tx_shares), "price_usd": round(tx_price, 4),
            "gross_usd": round(gross, 2), "commission_usd": round(tx_comm, 2),
            "total_usd": round(total, 2), "note": tx_note,
        }
        cfg2 = dict(cfg)
        cfg2.setdefault("transactions", []).append(new_tx)
        save_cfg(cfg2, active["id"])
        gh = github_push(cfg2, f"trade {nid} {tx_type} {tx_ticker}")
        st.success(f"✅ {nid} saved.  {'GitHub ✅' if gh['success'] else ''}")
        st.rerun()


def _build_ics(cfg, holdings, wht, filter_mode, account_id) -> bytes:
    """Build RFC 5545 ICS bytes for all dividend events."""
    from datetime import timedelta
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0",
        f"PRODID:-//PortfolioOptimizer//{account_id}//EN",
        f"X-WR-CALNAME:Dividends {account_id}",
    ]
    instruments = cfg.get("instruments", {})
    for tkr, inst in instruments.items():
        sh = holdings.get(tkr, {}).get("shares", 0)
        if not sh:
            continue
        dp = inst.get("dividend_policy", {})
        upcoming = dp.get("estimated_upcoming", [])
        for u in upcoming:
            elig = u.get("eligible_for_our_shares", True)
            if not elig:
                continue
            amt  = float(u.get("amount", 0))
            net  = sh * amt * (1 - wht)
            ex_d = str(u.get("ex","")).replace("-","")
            uid  = f"{tkr}_{ex_d}@portfoliooptimizer"
            desc = (f"Ticker: {tkr} | Shares: {sh} | "
                    f"Gross: ${sh*amt:.2f} | Net ({wht*100:.0f}% WHT): ${net:.2f}")
            lines += [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTART;VALUE=DATE:{ex_d}",
                f"DTEND;VALUE=DATE:{ex_d}",
                f"SUMMARY:{tkr} Ex-Dividend — ${net:.2f} net",
                f"DESCRIPTION:{desc}",
                "BEGIN:VALARM", "ACTION:DISPLAY",
                f"DESCRIPTION:{tkr} dividend tomorrow",
                "TRIGGER:-P1D", "END:VALARM",
                "END:VEVENT",
            ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines).encode("utf-8")
