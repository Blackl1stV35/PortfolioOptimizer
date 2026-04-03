"""
engine/julia_engine.jl  --  Julia PortfolioEngine Module
=========================================================
Loaded by julia_bridge.py via:
    jl.seval('include("engine/julia_engine.jl")')
    jl.seval("using .PortfolioEngine")

Provides:
  ledoit_wolf_cov(X)          -- Analytical Nonlinear Shrinkage (Ledoit-Wolf 2018)
  risk_metrics(r, rf)         -- Full risk metrics vector for a return series
  monte_carlo_income(R, w, ..)-- Multi-threaded bootstrap Monte Carlo

Threading:
  Julia uses all available CPU threads automatically for monte_carlo_income.
  Set JULIA_NUM_THREADS=4 (or "auto") before starting Python to enable.
  On Windows: set JULIA_NUM_THREADS=auto in system environment variables.
"""

module PortfolioEngine

using Statistics
using LinearAlgebra
using CovarianceEstimation

# ══════════════════════════════════════════════════════════════════════════════
# COVARIANCE  --  Analytical Nonlinear Shrinkage (Ledoit-Wolf 2018, QIS)
# ══════════════════════════════════════════════════════════════════════════════
"""
    ledoit_wolf_cov(X::Matrix{Float64}) -> Matrix{Float64}

Estimates the covariance matrix of asset returns X using Analytical Nonlinear
Shrinkage (Ledoit & Wolf 2018 / Quadratic-Inverse Shrinkage).

This is strictly superior to linear shrinkage when T (rows) is close to N
(columns) -- the regime you are in with monthly data and 2-10 assets.

X should be (T x N): rows = time periods, columns = assets.
"""
function ledoit_wolf_cov(X::Matrix{Float64})::Matrix{Float64}
    cov(AnalyticalNonlinearShrinkage(), X)
end


# ══════════════════════════════════════════════════════════════════════════════
# RISK METRICS  --  full set for a single return series
# ══════════════════════════════════════════════════════════════════════════════
"""
    risk_metrics(r, rf=0.045/12) -> NamedTuple

Computes annualised risk metrics for monthly return vector r.
rf = monthly risk-free rate (default: 4.5% annual / 12).
"""
function risk_metrics(
    r::Vector{Float64},
    rf::Float64 = 0.045 / 12,
)
    μ   = mean(r)
    σ   = std(r)
    neg = filter(<(0.0), r)
    sσ  = length(neg) > 1 ? std(neg) : NaN

    # Cumulative returns and drawdown
    cum  = cumprod(1.0 .+ r)
    peak = accumulate(max, cum)
    dd   = @. (cum - peak) / peak
    mdd  = minimum(dd)

    # VaR / CVaR (95%)
    s   = sort(r)
    idx = max(1, floor(Int, 0.05 * length(s)))
    v95 = s[idx]
    cv95 = mean(filter(x -> x <= v95, r))

    # Omega ratio
    gains  = sum(filter(>(0.0), r))
    losses = abs(sum(filter(<(0.0), r)))
    omega  = losses > 0 ? gains / losses : Inf

    (
        ann_return    = μ * 12,
        ann_vol       = σ * sqrt(12),
        sharpe        = σ > 0 ? (μ - rf) / σ * sqrt(12) : NaN,
        sortino       = (!isnan(sσ) && sσ > 0) ? (μ - rf) / sσ * sqrt(12) : NaN,
        max_drawdown  = mdd,
        calmar        = mdd != 0.0 ? μ * 12 / abs(mdd) : NaN,
        var95         = v95,
        cvar95        = cv95,
        omega         = omega,
        semi_vol      = isnan(sσ) ? NaN : sσ * sqrt(12),
    )
end


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO  --  multi-threaded bootstrap
# ══════════════════════════════════════════════════════════════════════════════
"""
    monte_carlo_income(R, w, income_rate, v0, n_paths, n_months)

Bootstrap Monte Carlo simulation of portfolio value and cumulative income.

Parameters
----------
R            : (T x N) matrix of historical monthly returns
w            : (N,)   portfolio weights (must sum to 1)
income_rate  : fractional monthly dividend/income rate (e.g. 0.0707 / 12)
v0           : initial portfolio value (USD)
n_paths      : number of simulation paths
n_months     : simulation horizon in months

Returns
-------
value_paths  : (n_paths x n_months) matrix  -- portfolio value per month
income_paths : (n_paths x n_months) matrix  -- cumulative income per month

Threading: uses Julia's built-in multi-threading (Threads.@threads).
Set JULIA_NUM_THREADS=auto for best performance.
"""
function monte_carlo_income(
    R::Matrix{Float64},
    w::Vector{Float64},
    income_rate::Float64,
    v0::Float64,
    n_paths::Int = 10_000,
    n_months::Int = 60,
)::Tuple{Matrix{Float64}, Matrix{Float64}}

    T, _ = size(R)
    port_ret = R * w          # pre-compute portfolio return series (T x 1)

    V = Matrix{Float64}(undef, n_paths, n_months)
    I = Matrix{Float64}(undef, n_paths, n_months)

    Threads.@threads for p in 1:n_paths
        val = v0
        cum = 0.0
        for m in 1:n_months
            idx  = rand(1:T)
            val  = val * (1.0 + port_ret[idx])
            inc  = val * income_rate
            cum += inc
            V[p, m] = val
            I[p, m] = cum
        end
    end

    return V, I
end


# ══════════════════════════════════════════════════════════════════════════════
# DCA SIMULATION  --  models monthly new-money contribution
# ══════════════════════════════════════════════════════════════════════════════
"""
    monte_carlo_dca(R, w, income_rate, v0, monthly_add, n_paths, n_months)

Like monte_carlo_income but adds `monthly_add` USD to the portfolio each month
before computing income. Models your DCA accumulation plan.
"""
function monte_carlo_dca(
    R::Matrix{Float64},
    w::Vector{Float64},
    income_rate::Float64,
    v0::Float64,
    monthly_add::Float64,
    n_paths::Int = 10_000,
    n_months::Int = 360,
)::Tuple{Matrix{Float64}, Matrix{Float64}}

    T, _ = size(R)
    port_ret = R * w

    V = Matrix{Float64}(undef, n_paths, n_months)
    I = Matrix{Float64}(undef, n_paths, n_months)

    Threads.@threads for p in 1:n_paths
        val = v0
        cum = 0.0
        for m in 1:n_months
            idx  = rand(1:T)
            val  = val * (1.0 + port_ret[idx]) + monthly_add
            inc  = val * income_rate
            cum += inc
            V[p, m] = val
            I[p, m] = cum
        end
    end

    return V, I
end


# ══════════════════════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER  --  analytical mean-variance (long-only)
# ══════════════════════════════════════════════════════════════════════════════
"""
    efficient_frontier_mv(mu, Sigma, n_points, rf) -> Matrix{Float64}

Computes the long-only mean-variance efficient frontier analytically.
Returns (n_assets x n_points) weight matrix -- each column is one portfolio.
Faster than CVXPY for small N, useful as a sanity check against Riskfolio.
"""
function efficient_frontier_mv(
    mu::Vector{Float64},
    Sigma::Matrix{Float64},
    n_points::Int = 50,
    rf::Float64   = 0.045 / 12,
)::Matrix{Float64}

    N = length(mu)
    W = Matrix{Float64}(undef, N, n_points)

    # Min-variance via analytical formula (no constraints handled via clipping)
    S_inv = inv(Sigma)
    e     = ones(N)
    w_mv  = S_inv * e / (e' * S_inv * e)
    w_mv  = max.(w_mv, 0.0); w_mv ./= sum(w_mv)

    # Max-return: 100% in highest mu asset
    w_max = zeros(N); w_max[argmax(mu)] = 1.0

    for i in 1:n_points
        t    = (i - 1) / (n_points - 1)
        w    = (1.0 - t) .* w_mv .+ t .* w_max
        w    = max.(w, 0.0)
        s    = sum(w)
        W[:, i] = s > 0 ? w ./ s : e ./ N
    end

    return W
end

end  # module PortfolioEngine
