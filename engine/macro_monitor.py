"""engine/macro_monitor.py -- Macro Pulse monitor (see full version in conversation history)"""
import logging, time
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
_CACHE: dict = {}
_TTL = 300

def _ttl_get(k): e=_CACHE.get(k); return e["data"] if e and (datetime.now()-e["ts"]).seconds<_TTL else None
def _ttl_set(k,d): _CACHE[k]={"data":d,"ts":datetime.now()}; return d

def _fetch(ticker, period="3mo"):
    c = _ttl_get(ticker)
    if c is not None: return c
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        s = (raw["Close"].squeeze() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]).dropna()
        return _ttl_set(ticker, s)
    except: return _ttl_set(ticker, pd.Series(dtype=float))

def fetch_vix():
    s=_fetch("^VIX","3mo"); cur=float(s.iloc[-1]) if not s.empty else None
    d=cur-float(s.iloc[-22]) if s.empty is False and len(s)>=22 else None
    sig="green" if cur and cur<18 else ("red" if cur and cur>28 else "yellow")
    return {"current":cur,"change_30d":d,"series":s,"signal":sig}

def fetch_oil():
    s=_fetch("CL=F","3mo"); cur=float(s.iloc[-1]) if not s.empty else None
    d=cur-float(s.iloc[-22]) if not s.empty and len(s)>=22 else None
    sig="green" if cur and cur<80 else ("red" if cur and cur>95 else "yellow")
    note="WTI > $95 triggers defensive posture" if cur and cur>95 else ""
    return {"current":cur,"change_30d":d,"series":s,"signal":sig,"geopolitical_note":note}

def fetch_yield_curve():
    t2=_fetch("^IRX","1y"); t10=_fetch("^TNX","1y")
    if t2.empty or t10.empty: return {"spread_2s10s":None,"signal":"unknown","inverted":False}
    aligned=pd.DataFrame({"t2":t2,"t10":t10}).dropna()
    spread=float((aligned["t10"]-aligned["t2"]).iloc[-1]) if not aligned.empty else None
    sig="green" if spread and spread>0.25 else ("red" if spread and spread<-0.25 else "yellow")
    return {"spread_2s10s":round(spread,3) if spread else None,"series_spread":(aligned["t10"]-aligned["t2"]) if not aligned.empty else pd.Series(dtype=float),"series_2y":aligned.get("t2",pd.Series(dtype=float)),"series_10y":aligned.get("t10",pd.Series(dtype=float)),"signal":sig,"inverted":bool(spread and spread<0)}

def fetch_credit_risk():
    hy=_fetch("HYG","1y"); ig=_fetch("LQD","1y")
    if hy.empty or ig.empty: return {"score":50,"signal":"yellow","series_hy":hy,"series_ig":ig}
    aligned=pd.DataFrame({"hy":hy,"ig":ig}).dropna()
    rel_3m=float((aligned["hy"]/aligned["hy"].iloc[-66]-aligned["ig"]/aligned["ig"].iloc[-66]).iloc[-1]) if len(aligned)>=66 else 0.0
    score=max(0,min(100,int(50-rel_3m*500)))
    return {"score":score,"signal":"green" if score<35 else ("red" if score>65 else "yellow"),"series_hy":aligned.get("hy",hy),"series_ig":aligned.get("ig",ig)}

def fetch_liquidity_risk():
    irx=_fetch("^IRX","1y")
    if irx.empty: return {"score":None,"signal":"unknown"}
    cur=float(irx.iloc[-1]); score=max(0,min(100,int(cur*20)))
    return {"score":score,"current_irx":cur,"signal":"green" if score<30 else ("red" if score>60 else "yellow"),"series":irx}

def fetch_recession_probability():
    yc=fetch_yield_curve(); spread=yc.get("spread_2s10s")
    if spread is None: return {"probability":30,"signal":"yellow","note":"proxy unavailable"}
    prob,sig=(55,"red") if spread<-0.75 else (35,"yellow") if spread<-0.25 else (20,"yellow") if spread<0 else (10,"green")
    return {"probability":prob,"signal":sig,"note":f"2s10s spread: {spread:+.2f}%"}

def get_manual_rates(cfg):
    m=cfg.get("macro",{}); tr=float(m.get("thai_policy_rate",1.00)); ur=float(m.get("us_fed_rate",3.625))
    return {"thai_rate":tr,"thai_rate_date":str(m.get("thai_policy_rate_date","2026-01-08")),"us_fed_rate":ur,"us_fed_date":str(m.get("us_fed_rate_date","2026-03-19")),"thai_signal":"green" if tr<=1.25 else "yellow","us_signal":"yellow" if 3.0<=ur<=4.5 else "green"}

def get_macro_data(cfg):
    try:
        from engine.fx_timing import download_fx, compute_fx_signal
        fx_h=download_fx(lookback_days=180); fx_sig=compute_fx_signal(fx_h)
    except: fx_sig={"signal":"unknown","zscore":0,"current":32.68,"advice":"FX data unavailable"}
    return {"rates":get_manual_rates(cfg),"vix":fetch_vix(),"oil":fetch_oil(),"yield_curve":fetch_yield_curve(),"credit":fetch_credit_risk(),"liquidity":fetch_liquidity_risk(),"recession":fetch_recession_probability(),"fx":fx_sig,"fetched_at":datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

def get_risk_gauges(macro):
    vix_cur=macro["vix"].get("current") or 20.0
    return {"default_risk":macro["credit"].get("score") or 50,"liquidity_risk":macro["liquidity"].get("score") or 40,"maturity_risk":max(0,min(100,int(50-(macro["yield_curve"].get("spread_2s10s") or 0)*40))),"uncertainty":max(0,min(100,int((vix_cur-10)/40*100)))}

def get_macro_regime(macro):
    score=0; gauges=get_risk_gauges(macro); vix=macro["vix"].get("current") or 20.0; oil=macro["oil"].get("current") or 80.0
    rec=macro["recession"].get("probability") or 25; spread=macro["yield_curve"].get("spread_2s10s") or 0
    score+=0 if vix<18 else (1 if vix<25 else 2); score+=0 if spread>0.25 else (1 if spread>-0.25 else 2)
    score+=0 if gauges["default_risk"]<35 else (1 if gauges["default_risk"]<65 else 2)
    score+=0 if oil<80 else (1 if oil<95 else 2); score+=0 if rec<20 else (1 if rec<40 else 2)
    if score>=7: return {"regime":"Defensive","score":score,"cash_pct":"22–25%","cash_thb_est":"92k THB → hold 20–23k","action":"Hold cash. Delay PDI deployment.","color":"red"}
    if score>=4: return {"regime":"Neutral","score":score,"cash_pct":"15–20%","cash_thb_est":"92k THB → 14–18k cash","action":"Deploy cautiously. Consider partial PDI.","color":"yellow"}
    return {"regime":"Aggressive","score":score,"cash_pct":"10–15%","cash_thb_est":"92k THB → deploy 78k","action":"DCA into BKLN and PDI this week.","color":"green"}
