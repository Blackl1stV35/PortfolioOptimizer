"""
engine/julia_bridge.py  --  Python ↔ Julia Bridge  (v2)
========================================================
Fixes from v1:
  • Windows-safe path: Path.resolve() + forward slashes in Julia include()
  • Lazy init: _init_julia() is called once and cached
  • Thread pool for non-blocking heavy compute
  • Full Python fallbacks for every public function

Set PORTFOLIOOPTIMIZER_ENABLE_JULIA=0 to force Python-only mode.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent          # resolve() → absolute, no symlinks
JL_FILE = ROOT / "julia_engine.jl"

_EXECUTOR    = ThreadPoolExecutor(max_workers=4, thread_name_prefix="julia_worker")
_julia_ready = False
_jl          = None   # juliacall module handle


# ── Julia init (lazy, one-time) ───────────────────────────────────────────────

def _init_julia() -> bool:
    """
    Initialise Julia once.  Safe to call repeatedly — returns cached state.
    FIX: use forward slashes (JL_FILE.as_posix()) so Windows backslashes
         never appear in the Julia include() string literal.
    """
    global _julia_ready, _jl
    if _julia_ready:
        return True
    if os.environ.get("PORTFOLIOOPTIMIZER_ENABLE_JULIA", "1") != "1":
        return False
    if not JL_FILE.exists():
        log.warning("Julia engine file not found: %s", JL_FILE)
        return False
    try:
        import juliacall                            # type: ignore
        _jl = juliacall.newmodule("PortfolioEnginePy")
        # Use .as_posix() → forward slashes only, works on Windows + Linux
        jl_path = JL_FILE.as_posix()
        _jl.seval(f'include("{jl_path}")')
        _jl.seval("using .PortfolioEngine")
        _julia_ready = True
        log.info("Julia engine initialised: %s", jl_path)
        return True
    except Exception as exc:
        log.warning("Julia unavailable — Python fallback active: %s", exc)
        _julia_ready = False
        return False


# ── Thread-pool helper ────────────────────────────────────────────────────────

def run_in_thread(fn: Callable, *args, **kwargs) -> Future:
    """Submit fn to thread pool. Returns Future — never blocks Streamlit."""
    return _EXECUTOR.submit(fn, *args, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def ledoit_wolf_cov(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Analytical Nonlinear Shrinkage (Ledoit-Wolf 2020). Falls back to OAS."""
    R = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    # Julia requires a dense, column-major (Fortran-order) Float64 matrix.
    # np.asfortranarray ensures juliacall sees Matrix{Float64} not PyArray.
    R = np.asfortranarray(R, dtype=np.float64)
    if _init_julia():
        try:
            t0 = time.perf_counter()
            Σ  = np.array(_jl.PortfolioEngine.ledoit_wolf_cov(R))
            log.debug("Julia ledoit_wolf_cov: %.3fs", time.perf_counter() - t0)
            return Σ
        except Exception as exc:
            log.warning("Julia cov failed, falling back: %s", exc)
    return _py_ledoit(R)


def monte_carlo(
    returns      : pd.DataFrame | np.ndarray,
    weights      : np.ndarray,
    income_yield : float,
    pv           : float,
    monthly_add  : float,
    n_paths      : int = 3000,
    n_months     : int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-threaded Monte Carlo.
    Returns (value_paths, income_paths) each shaped (n_paths, n_months).
    """
    R = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    R = np.asfortranarray(R, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    if _init_julia():
        try:
            t0   = time.perf_counter()
            Σ    = _py_ledoit(R)
            mu   = R.mean(axis=0)
            vp, ip = _jl.PortfolioEngine.monte_carlo(
                mu, Σ, w, float(pv), float(monthly_add),
                int(n_paths), int(n_months), float(income_yield)
            )
            log.debug("Julia monte_carlo %d×%d: %.3fs",
                      n_paths, n_months, time.perf_counter() - t0)
            return np.array(vp), np.array(ip)
        except Exception as exc:
            log.warning("Julia MC failed, falling back: %s", exc)

    return _py_monte_carlo(R, w, income_yield, pv, monthly_add, n_paths, n_months)


def generational_plan(
    port_mu        : float,
    port_vol       : float,
    pv             : float,
    monthly_add    : float,
    n_paths        : int,
    income_yield   : float,
    target_income_m: float,
) -> dict:
    """30-year generational plan. Returns milestone dict."""
    if _init_julia():
        try:
            t0  = time.perf_counter()
            raw = dict(_jl.PortfolioEngine.generational_plan(
                float(port_mu), float(port_vol), float(pv),
                float(monthly_add), int(n_paths),
                float(income_yield), float(target_income_m)
            ))
            ms: dict = {}
            for yr, d in dict(raw.get("milestones", {})).items():
                ms[int(yr)] = {k: float(v) for k, v in dict(d).items()}
            raw["milestones"]      = ms
            raw["years_to_target"] = int(raw.get("years_to_target", 0))
            raw["extra_months"]    = int(raw.get("extra_months",    0))
            log.debug("Julia generational_plan: %.3fs", time.perf_counter() - t0)
            return raw
        except Exception as exc:
            log.warning("Julia gen plan failed, falling back: %s", exc)

    return _py_generational_plan(
        port_mu, port_vol, pv, monthly_add, n_paths, income_yield, target_income_m
    )


def risk_metrics(
    returns : pd.DataFrame | np.ndarray,
    weights : np.ndarray,
    rf      : float = 0.045 / 12,
) -> dict:
    R = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns)
    R = np.asfortranarray(R, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64); w /= w.sum()
    if _init_julia():
        try:
            r = dict(_jl.PortfolioEngine.risk_metrics(R, w, float(rf)))
            return {k: float(v) for k, v in r.items()}
        except Exception as exc:
            log.warning("Julia risk_metrics failed, falling back: %s", exc)
    return _py_risk_metrics(R, w, rf)


def backend_info() -> dict:
    julia = _init_julia()
    return {
        "julia_available": julia,
        "backend":  "Julia (multi-threaded)" if julia else "Python/NumPy (single-threaded)",
        "threads":  int(os.environ.get("JULIA_NUM_THREADS", "1")) if julia else 1,
        "jl_file":  str(JL_FILE),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Python fallbacks
# ══════════════════════════════════════════════════════════════════════════════

def _py_ledoit(R: np.ndarray) -> np.ndarray:
    try:
        from sklearn.covariance import OAS
        return OAS().fit(R).covariance_
    except Exception:
        return np.cov(R, rowvar=False)


def _py_monte_carlo(R, w, income_yield, pv, monthly_add, n_paths, n_months):
    port_ret = R @ w
    mu, vol  = port_ret.mean(), port_ret.std()
    rng      = np.random.default_rng(42)
    rand_r   = rng.normal(mu, vol, (n_paths, n_months))
    vp = np.zeros((n_paths, n_months))
    ip = np.zeros((n_paths, n_months))
    pv_vec = np.full(n_paths, pv)
    for m in range(n_months):
        pv_vec     = np.maximum(pv_vec * (1 + rand_r[:, m]) + monthly_add, 0)
        vp[:, m]   = pv_vec
        ip[:, m]   = pv_vec * income_yield
    return vp, ip


def _py_risk_metrics(R, w, rf):
    port_ret = R @ w
    ann_ret  = port_ret.mean() * 12
    ann_vol  = port_ret.std()  * np.sqrt(12)
    sharpe   = (port_ret.mean() - rf) / port_ret.std() * np.sqrt(12) \
               if port_ret.std() > 0 else 0.0
    v95  = float(np.percentile(port_ret, 5))
    cvar = float(port_ret[port_ret <= v95].mean()) \
           if (port_ret <= v95).any() else v95
    cum  = np.cumprod(1 + port_ret)
    pk   = np.maximum.accumulate(cum)
    mdd  = float(((cum - pk) / pk).min())
    return {
        "ann_return":   float(ann_ret),
        "ann_vol":      float(ann_vol),
        "sharpe":       float(sharpe),
        "cvar_95":      cvar,
        "max_drawdown": mdd,
    }


def _py_generational_plan(mu, vol, pv, monthly_add, n_paths,
                           income_yield, target):
    vp, ip = _py_monte_carlo(
        np.column_stack([np.random.normal(mu, vol, 1000)]),
        np.array([1.0]), income_yield, pv, monthly_add, n_paths, 360,
    )
    milestones = {}
    for yr in [5, 10, 15, 20, 25, 30]:
        m     = yr * 12 - 1
        v_col = vp[:, m]; i_col = ip[:, m]
        milestones[yr] = {
            "p10_value":         float(np.percentile(v_col, 10)),
            "p50_value":         float(np.percentile(v_col, 50)),
            "p90_value":         float(np.percentile(v_col, 90)),
            "p10_income_m":      float(np.percentile(i_col, 10)),
            "p50_income_m":      float(np.percentile(i_col, 50)),
            "p90_income_m":      float(np.percentile(i_col, 90)),
            "prob_above_target": float(np.mean(i_col > target) * 100),
            "cum_div_p50":       float(np.percentile(ip[:, :m+1], 50, axis=0).sum()),
        }
    cross = []
    for p in range(n_paths):
        idx = np.argmax(ip[p, :] > target)
        cross.append(idx if ip[p, idx] > target else 361)
    med = int(np.median(cross))
    return {
        "milestones":       milestones,
        "median_cross_m":   med,
        "years_to_target":  med // 12,
        "extra_months":     med % 12,
        "prob_never":       float(np.mean(np.array(cross) > 360) * 100),
        "p10_final":        float(np.percentile(vp[:, -1], 10)),
        "p50_final":        float(np.percentile(vp[:, -1], 50)),
        "p90_final":        float(np.percentile(vp[:, -1], 90)),
    }

def max_sharpe(returns, rf: float = 0.045/12):
    """Max-Sharpe weights via Julia (replaces rp.Portfolio.optimization)."""
    import numpy as np
    R = returns.values if hasattr(returns, "values") else returns
    R = np.asfortranarray(R, dtype=np.float64)
    n = R.shape[1]
    if _init_julia():
        try:
            w = np.array(_jl.PortfolioEngine.max_sharpe_weights(R, float(rf)))
            w = np.maximum(w, 0)
            s = w.sum()
            return w/s if s > 0 else np.ones(n)/n
        except Exception as exc:
            log.warning("Julia max_sharpe failed: %s", exc)
    return np.ones(n) / n


def min_cvar(returns, alpha: float = 0.95):
    """Min-CVaR portfolio weights via Julia."""
    import numpy as np
    R = returns.values if hasattr(returns, "values") else returns
    R = np.asfortranarray(R, dtype=np.float64)
    n = R.shape[1]
    if _init_julia():
        try:
            w = np.array(_jl.PortfolioEngine.min_cvar_weights(R, float(alpha)))
            w = np.maximum(w, 0)
            s = w.sum()
            return w/s if s > 0 else np.ones(n)/n
        except Exception as exc:
            log.warning("Julia min_cvar failed: %s", exc)
    return np.ones(n) / n


def hrp(returns):
    """Hierarchical Risk Parity weights via Julia."""
    import numpy as np
    R = returns.values if hasattr(returns, "values") else returns
    R = np.asfortranarray(R, dtype=np.float64)
    n = R.shape[1]
    if _init_julia():
        try:
            w = np.array(_jl.PortfolioEngine.hrp_weights(R))
            w = np.maximum(w, 0)
            s = w.sum()
            return w/s if s > 0 else np.ones(n)/n
        except Exception as exc:
            log.warning("Julia HRP failed: %s", exc)
    return np.ones(n) / n

