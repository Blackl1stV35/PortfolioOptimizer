"""engine/backtest.py  --  Walk-Forward Backtest Engine (server version)"""

import io
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

COLORS = {"max_sharpe":"#1D9E75","min_variance":"#3B8BD4",
          "equal_weight":"#888780","hrp":"#BA7517"}


def run_walkforward(returns: pd.DataFrame, train_months: int = 24,
                    strategies=None) -> dict:
    if strategies is None:
        strategies = ["max_sharpe", "min_variance", "equal_weight", "hrp"]
    T, N = returns.shape
    n_test = T - train_months
    if n_test < 3:
        log.warning("Backtest: insufficient data (%d months)", T)
        return {}
    log.info("Backtest: %d OOS months, %d strategies", n_test, len(strategies))
    test_dates = returns.index[train_months:]
    equity     = {s: np.ones(n_test + 1) for s in strategies}
    ret_hist   = {s: [] for s in strategies}
    for i in range(n_test):
        train    = returns.iloc[i: i + train_months]
        test_ret = returns.iloc[i + train_months]
        for strat in strategies:
            w = _weights(train, strat)
            r = float((test_ret.values * w).sum())
            equity[strat][i+1] = equity[strat][i] * (1 + r)
            ret_hist[strat].append(r)
    equity_df = pd.DataFrame(equity,
                              index=[returns.index[train_months-1]] + list(test_dates))
    rets_df   = pd.DataFrame(ret_hist, index=test_dates)
    metrics   = _metrics(rets_df, equity_df)
    log.info("Backtest done. Sharpes: %s",
             {s: f"{m['sharpe']:.3f}" for s, m in metrics.items()})
    return {"equity": equity_df, "returns": rets_df, "metrics": metrics,
            "equity_chart": _plot_equity(equity_df),
            "sharpe_chart": _plot_sharpe(rets_df)}


def _weights(train, strategy):
    N = train.shape[1]
    try:
        from engine.analytics import compute_cov
        cov = compute_cov(train)
    except Exception:
        cov = np.cov(train.values.T)
    mu = train.mean().values
    if strategy == "equal_weight": return np.ones(N) / N
    if strategy == "hrp":
        vol = np.sqrt(np.diag(cov)); w = 1/np.where(vol>0,vol,1e-9); return w/w.sum()
    try:
        import riskfolio as rp
        port = rp.Portfolio(returns=train)
        port.mu  = pd.DataFrame(mu.reshape(1,-1), columns=train.columns)
        port.cov = pd.DataFrame(cov, index=train.columns, columns=train.columns)
        port.sht = False; port.upperlng = 1.0
        obj = "Sharpe" if strategy == "max_sharpe" else "MinRisk"
        w   = port.optimization(model="Classic", rm="MV", obj=obj,
                                 rf=0.045/12, l=2, hist=False)
        if w is not None and not w.isnull().values.any():
            return w["weights"].values
    except Exception:
        pass
    # Analytical fallback
    try:
        Si = np.linalg.inv(cov); e = np.ones(N)
        if strategy == "min_variance":
            w = Si@e / (e@Si@e)
        else:
            exc = mu - 0.045/12; z = Si@exc
            w = z/z.sum() if z.sum()>0 else Si@e/(e@Si@e)
        w = np.maximum(w, 0); s = w.sum(); return w/s if s>0 else e/N
    except Exception:
        return np.ones(N)/N


def _metrics(rets_df, equity_df):
    rf_m = 0.045/12; result = {}
    for s in rets_df.columns:
        r  = rets_df[s].values; eq = equity_df[s].values
        ann = r.mean()*12; vol = r.std()*np.sqrt(12)
        sr  = (r.mean()-rf_m)/r.std()*np.sqrt(12) if r.std()>0 else np.nan
        pk  = np.maximum.accumulate(eq); dd = (eq-pk)/pk; mdd = float(dd.min())
        result[s] = {"ann_return":ann,"ann_vol":vol,"sharpe":sr,"max_drawdown":mdd,
                     "final_equity":float(eq[-1]),
                     "calmar":ann/abs(mdd) if mdd!=0 else np.nan}
    return result


def _plot_equity(equity: pd.DataFrame) -> bytes:
    try:
        fig, ax = plt.subplots(figsize=(10,5), facecolor="#FAFBFD")
        ax.set_facecolor("#FAFBFD")
        for s in equity.columns:
            ax.plot(equity.index, equity[s], color=COLORS.get(s,"#333"),
                    lw=1.8, label=s.replace("_"," ").title())
        ax.axhline(1.0, color="#ccc", lw=0.8, ls=":")
        ax.set_ylabel("Normalised value"); ax.set_title("Walk-Forward Equity Curves", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(color="#E4E8F0", lw=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig); buf.seek(0); return buf.read()
    except Exception as e:
        log.warning("Equity chart: %s", e); return b""


def _plot_sharpe(rets_df: pd.DataFrame, window: int = 12) -> bytes:
    try:
        fig, ax = plt.subplots(figsize=(10,4), facecolor="#FAFBFD")
        ax.set_facecolor("#FAFBFD"); rf_m = 0.045/12
        for s in rets_df.columns:
            roll = rets_df[s].rolling(window)
            rs   = (roll.mean()-rf_m)/roll.std()*np.sqrt(12)
            ax.plot(rs.index, rs, color=COLORS.get(s,"#333"),
                    lw=1.5, label=s.replace("_"," ").title())
        ax.axhline(0, color="#888", lw=0.8); ax.axhline(1, color="#1D9E75", lw=0.8, ls="--")
        ax.set_ylabel(f"Rolling {window}-month Sharpe")
        ax.set_title("Rolling Sharpe Ratio", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(color="#E4E8F0", lw=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig); buf.seek(0); return buf.read()
    except Exception as e:
        log.warning("Sharpe chart: %s", e); return b""
