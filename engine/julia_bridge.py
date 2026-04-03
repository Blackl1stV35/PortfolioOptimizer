"""
engine/julia_bridge.py  --  Python-Julia Bridge
================================================
Provides two Julia-accelerated functions:
  julia_cov(returns_df)          -- Analytical Nonlinear Shrinkage covariance
  julia_monte_carlo(...)         -- Multi-threaded bootstrap Monte Carlo

Each function degrades gracefully to a Python/sklearn fallback when:
  - Julia is not installed
  - juliacall is not installed
  - First-run JIT compilation fails
  - Any runtime error

SETUP (once):
    pip install juliacall
    # juliacall auto-downloads Julia 1.10 LTS on first import (~200MB, one-time)
    # CovarianceEstimation.jl installs automatically on first bridge init (~30s)

PERFORMANCE vs Python fallback:
    Covariance  : ~10-50x faster (mainly relevant at 10+ assets)
    Monte Carlo : ~30-80x faster (critical for 10k paths × 360 months)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Module state ──────────────────────────────────────────────────────────────
_jl    = None
_eng   = None
_ready = False

JL_ENGINE = Path(__file__).parent / "julia_engine.jl"


def init_julia() -> bool:
    """
    Initialise Julia runtime and load PortfolioEngine module.
    Safe to call multiple times -- returns cached state after first call.
    """
    global _jl, _eng, _ready
    if _ready:
        return True

    # Quick availability check
    try:
        import juliacall  # noqa: F401
    except ImportError:
        log.info("  juliacall not installed -- Julia features disabled")
        log.info("    To enable: pip install juliacall")
        return False

    try:
        log.info("  Initialising Julia (first run: 30-90s JIT compile) ...")
        t0 = time.monotonic()

        from juliacall import Main as jl
        _jl = jl

        # Install CovarianceEstimation.jl if needed (silent, one-time)
        jl.seval('import Pkg; Pkg.add(name="CovarianceEstimation"); using CovarianceEstimation')

        # Load our engine module
        if not JL_ENGINE.exists():
            log.warning("  julia_engine.jl not found at %s -- Julia disabled", JL_ENGINE)
            return False

        jl_path = str(JL_ENGINE).replace("\\", "/")
        jl.seval(f'include("{jl_path}")')
        jl.seval("using .PortfolioEngine")
        _eng   = jl.PortfolioEngine
        _ready = True

        elapsed = time.monotonic() - t0
        n_threads = int(jl.seval("Threads.nthreads()"))
        log.info("  Julia ready in %.1fs  |  %d threads active", elapsed, n_threads)
        return True

    except Exception as exc:
        log.warning("  Julia init failed: %s", exc)
        log.warning("  Falling back to Python for all computations")
        _ready = False
        return False


# ══════════════════════════════════════════════════════════════════════════════
# COVARIANCE
# ══════════════════════════════════════════════════════════════════════════════
def julia_cov(returns_df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Analytical Nonlinear Shrinkage covariance estimator (Ledoit-Wolf 2018).
    Strictly optimal when T (observations) is close to N (assets).

    Falls back to sklearn LedoitWolf if Julia unavailable.
    Returns numpy array (N x N) or None on complete failure.
    """
    if not _ready and not init_julia():
        return _py_cov(returns_df)
    try:
        X   = np.asfortranarray(returns_df.values, dtype=np.float64)
        cov = np.array(_eng.ledoit_wolf_cov(_jl.Matrix(X)))
        log.info("  [Julia] Analytical Nonlinear Shrinkage cov (%dx%d)", *cov.shape)
        return cov
    except Exception as exc:
        log.warning("  Julia cov failed (%s) -- sklearn fallback", exc)
        return _py_cov(returns_df)


def _py_cov(returns_df: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(returns_df.values).covariance_
        log.info("  [Python] sklearn LedoitWolf cov (%dx%d)", *cov.shape)
        return cov
    except Exception as exc:
        log.warning("  sklearn cov failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
def julia_monte_carlo(
    returns_df:          pd.DataFrame,
    weights:             np.ndarray,
    monthly_income_rate: float,
    initial_value:       float,
    n_paths:             int   = 10_000,
    n_months:            int   = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap Monte Carlo simulation of portfolio value and cumulative income.

    Parameters
    ----------
    returns_df           : monthly returns DataFrame (T x N)
    weights              : portfolio weights vector (N,)
    monthly_income_rate  : fractional monthly dividend yield (e.g. 0.0707/12)
    initial_value        : starting portfolio value in USD
    n_paths              : number of simulation paths (10k recommended)
    n_months             : simulation horizon in months

    Returns
    -------
    value_paths  : ndarray (n_paths x n_months) -- portfolio value per month
    income_paths : ndarray (n_paths x n_months) -- cumulative income per month
    """
    if not _ready and not init_julia():
        return _py_mc(returns_df, weights, monthly_income_rate,
                       initial_value, n_paths, n_months)
    try:
        t0  = time.monotonic()
        R   = np.asfortranarray(returns_df.values, dtype=np.float64)
        w   = np.asarray(weights, dtype=np.float64)
        vp, ip = _eng.monte_carlo_income(
            _jl.Matrix(R),
            _jl.Vector(w),
            float(monthly_income_rate),
            float(initial_value),
            int(n_paths),
            int(n_months),
        )
        vp_np = np.array(vp)
        ip_np = np.array(ip)
        log.info("  [Julia] MC %d paths × %d months in %.2fs",
                 n_paths, n_months, time.monotonic() - t0)
        return vp_np, ip_np
    except Exception as exc:
        log.warning("  Julia MC failed (%s) -- numpy fallback", exc)
        return _py_mc(returns_df, weights, monthly_income_rate,
                       initial_value, n_paths, n_months)


def _py_mc(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    monthly_income_rate: float,
    initial_value: float,
    n_paths: int,
    n_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised numpy Monte Carlo (slower but always available)."""
    t0       = time.monotonic()
    port_ret = returns_df.values @ weights
    T        = len(port_ret)
    idxs     = np.random.randint(0, T, (n_paths, n_months))
    sampled  = port_ret[idxs]

    vp = np.zeros((n_paths, n_months))
    ip = np.zeros((n_paths, n_months))
    val = np.full(n_paths, float(initial_value))
    cum = np.zeros(n_paths)

    for m in range(n_months):
        val  = val * (1.0 + sampled[:, m])
        inc  = val * monthly_income_rate
        cum += inc
        vp[:, m] = val
        ip[:, m] = cum

    log.info("  [Python] MC %d paths × %d months in %.2fs",
             n_paths, n_months, time.monotonic() - t0)
    return vp, ip
