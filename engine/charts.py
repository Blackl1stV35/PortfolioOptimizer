"""
engine/charts.py  —  Advanced Interactive Charts (pandas_ta)
=============================================================
Provides Plotly figures with full technical indicators:
  SMA / EMA / BBANDS / RSI / MACD / Volume / Stochastic

Call from app.py Analytics Engine → Advanced Charts tab.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

log = logging.getLogger(__name__)

# Timeframe → yfinance period + interval
TIMEFRAMES = {
    "1D":  {"period": "1d",   "interval": "5m"},
    "5D":  {"period": "5d",   "interval": "30m"},
    "1M":  {"period": "1mo",  "interval": "1d"},
    "3M":  {"period": "3mo",  "interval": "1d"},
    "6M":  {"period": "6mo",  "interval": "1d"},
    "1Y":  {"period": "1y",   "interval": "1wk"},
    "3Y":  {"period": "3y",   "interval": "1wk"},
    "MAX": {"period": "max",  "interval": "1mo"},
}

INDICATOR_DEFAULTS = {
    "SMA20": True, "SMA50": True, "EMA20": False,
    "BB": True, "RSI": True, "MACD": True, "Volume": False,
}

COLORS = {
    "price":   "#3B8BD4",
    "sma20":   "#F4A261",
    "sma50":   "#E76F51",
    "ema20":   "#2A9D8F",
    "bb_upper":"rgba(120,120,220,0.4)",
    "bb_lower":"rgba(120,120,220,0.4)",
    "bb_fill": "rgba(120,120,220,0.08)",
    "rsi_line":"#7B2FBE",
    "rsi_ob":  "rgba(220,50,50,0.15)",
    "rsi_os":  "rgba(50,180,50,0.15)",
    "macd":    "#1D9E75",
    "signal":  "#E24B4A",
    "hist_pos":"rgba(29,158,117,0.6)",
    "hist_neg":"rgba(226,75,74,0.6)",
}


@staticmethod
def _fetch(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                           auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        return raw.dropna(how="all")
    except Exception as exc:
        log.warning("Chart fetch %s: %s", ticker, exc)
        return pd.DataFrame()


def build_chart(
    ticker: str,
    timeframe: str = "6M",
    indicators: dict | None = None,
    benchmark: str | None = None,
    compare_tickers: list[str] | None = None,
    dark: bool = True,
) -> go.Figure:
    """
    Build a full interactive candlestick chart with selected indicators.

    Parameters
    ----------
    ticker      : primary ticker (e.g. "BKLN")
    timeframe   : one of TIMEFRAMES keys
    indicators  : dict of {indicator_name: bool}, defaults to INDICATOR_DEFAULTS
    benchmark   : optional benchmark ticker (e.g. "SPY")
    compare_tickers : list of additional tickers to overlay (normalised)
    dark        : use dark theme

    Returns
    -------
    Plotly Figure
    """
    ind = {**INDICATOR_DEFAULTS, **(indicators or {})}
    tf  = TIMEFRAMES.get(timeframe, TIMEFRAMES["6M"])

    df = _fetch(ticker, tf["period"], tf["interval"])
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {ticker}", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False, font=dict(size=16))
        return fig

    # ── Compute indicators via pandas_ta ──────────────────────────────────────
    try:
        import pandas_ta as ta
        if ind.get("SMA20"):
            df["SMA20"] = ta.sma(df["Close"], length=20)
        if ind.get("SMA50"):
            df["SMA50"] = ta.sma(df["Close"], length=50)
        if ind.get("EMA20"):
            df["EMA20"] = ta.ema(df["Close"], length=20)
        if ind.get("BB"):
            bb = ta.bbands(df["Close"], length=20, std=2)
            if bb is not None:
                df["BB_Upper"] = bb.iloc[:, 0]
                df["BB_Mid"]   = bb.iloc[:, 1]
                df["BB_Lower"] = bb.iloc[:, 2]
        if ind.get("RSI"):
            df["RSI"] = ta.rsi(df["Close"], length=14)
        if ind.get("MACD"):
            macd = ta.macd(df["Close"])
            if macd is not None:
                df["MACD"]       = macd.iloc[:, 0]
                df["MACD_Signal"]= macd.iloc[:, 1]
                df["MACD_Hist"]  = macd.iloc[:, 2]
    except ImportError:
        log.warning("pandas_ta not installed — using simple rolling for indicators")
        if ind.get("SMA20"): df["SMA20"] = df["Close"].rolling(20).mean()
        if ind.get("SMA50"): df["SMA50"] = df["Close"].rolling(50).mean()
        if ind.get("EMA20"): df["EMA20"] = df["Close"].ewm(span=20).mean()
        if ind.get("BB"):
            roll = df["Close"].rolling(20)
            df["BB_Upper"] = roll.mean() + 2*roll.std()
            df["BB_Mid"]   = roll.mean()
            df["BB_Lower"] = roll.mean() - 2*roll.std()
        if ind.get("RSI"):
            delta = df["Close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            df["RSI"] = 100 - 100/(1+rs)

    # ── Subplot layout ────────────────────────────────────────────────────────
    has_rsi  = ind.get("RSI")  and "RSI"  in df.columns
    has_macd = ind.get("MACD") and "MACD" in df.columns
    has_vol  = ind.get("Volume") and "Volume" in df.columns

    rows    = 1 + int(has_rsi) + int(has_macd) + int(has_vol)
    heights = [0.55] + [0.15 if rows > 2 else 0.25] * (rows - 1)

    subplot_titles = [ticker]
    if has_vol:  subplot_titles.append("Volume")
    if has_rsi:  subplot_titles.append("RSI (14)")
    if has_macd: subplot_titles.append("MACD")

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    # ── Row 1: Price + overlays ───────────────────────────────────────────────
    has_ohlc = all(c in df.columns for c in ["Open","High","Low","Close"])
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name=ticker,
            increasing_line_color="#2ECC71",
            decreasing_line_color="#E24B4A",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name=ticker,
            line=dict(color=COLORS["price"], width=2),
        ), row=1, col=1)

    # Bollinger Bands
    if ind.get("BB") and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
            line=dict(color=COLORS["bb_upper"], width=1, dash="dot"),
            name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
            fill="tonexty", fillcolor=COLORS["bb_fill"],
            line=dict(color=COLORS["bb_lower"], width=1, dash="dot"),
            name="BB Bands"), row=1, col=1)

    if ind.get("SMA20") and "SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"],
            line=dict(color=COLORS["sma20"], width=1.2),
            name="SMA20"), row=1, col=1)
    if ind.get("SMA50") and "SMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"],
            line=dict(color=COLORS["sma50"], width=1.2),
            name="SMA50"), row=1, col=1)
    if ind.get("EMA20") and "EMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"],
            line=dict(color=COLORS["ema20"], width=1.2, dash="dash"),
            name="EMA20"), row=1, col=1)

    # Benchmark / comparison overlays (normalised to 100)
    for cmp_ticker, cmp_color in [
        (benchmark, "#888888"),
        *[(ct, ["#F9C74F","#90BE6D","#F3722C"][i % 3])
          for i, ct in enumerate(compare_tickers or [])],
    ]:
        if not cmp_ticker:
            continue
        cmp_df = _fetch(cmp_ticker, tf["period"], tf["interval"])
        if cmp_df.empty:
            continue
        aligned = cmp_df["Close"].reindex(df.index, method="ffill")
        base    = aligned.dropna().iloc[0] if not aligned.dropna().empty else 1
        norm    = (aligned / base) * (df["Close"].dropna().iloc[0] if not df["Close"].dropna().empty else 1)
        fig.add_trace(go.Scatter(x=aligned.index, y=norm.values,
            line=dict(color=cmp_color, width=1.2, dash="dash"),
            name=f"{cmp_ticker} (norm)", opacity=0.75), row=1, col=1)

    # ── Additional rows ───────────────────────────────────────────────────────
    cur_row = 2

    if has_vol:
        colors_vol = ["#2ECC71" if c >= o else "#E24B4A"
                      for c, o in zip(df["Close"], df["Open"] if "Open" in df.columns else df["Close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
            marker_color=colors_vol, name="Volume", showlegend=False), row=cur_row, col=1)
        cur_row += 1

    if has_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
            line=dict(color=COLORS["rsi_line"], width=1.5),
            name="RSI"), row=cur_row, col=1)
        fig.add_hrect(y0=70, y1=100, row=cur_row, col=1, fillcolor=COLORS["rsi_ob"], line_width=0)
        fig.add_hrect(y0=0,  y1=30,  row=cur_row, col=1, fillcolor=COLORS["rsi_os"], line_width=0)
        fig.add_hline(y=70, row=cur_row, col=1, line=dict(color="#E24B4A", width=0.8, dash="dot"))
        fig.add_hline(y=30, row=cur_row, col=1, line=dict(color="#2ECC71", width=0.8, dash="dot"))
        cur_row += 1

    if has_macd:
        hist = df["MACD_Hist"].fillna(0)
        fig.add_trace(go.Bar(x=df.index, y=hist,
            marker_color=[COLORS["hist_pos"] if v >= 0 else COLORS["hist_neg"] for v in hist],
            name="MACD Hist", showlegend=False), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
            line=dict(color=COLORS["macd"], width=1.5), name="MACD"), row=cur_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
            line=dict(color=COLORS["signal"], width=1.2, dash="dot"),
            name="Signal"), row=cur_row, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    bg   = "#0E1117" if dark else "#FFFFFF"
    grid = "#1E2A3B" if dark else "#E0E0E0"
    txt  = "#FFFFFF"  if dark else "#000000"

    fig.update_layout(
        height=560 + 120*(rows-1),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=txt, size=11),
        legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, rows+1):
        fig.update_xaxes(gridcolor=grid, row=i, col=1, showgrid=True)
        fig.update_yaxes(gridcolor=grid, row=i, col=1, showgrid=True)

    return fig
