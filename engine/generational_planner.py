"""engine/generational_planner.py  --  30-Year Generational Wealth Planner"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

log = logging.getLogger(__name__)


@dataclass
class GenPlanConfig:
    n_paths:          int   = 5_000
    horizon_years:    int   = 30
    monthly_add_usd:  float = 200.0
    income_rate_m:    float = 0.0707 / 12
    inflation_annual: float = 0.03
    target_income_m:  float = 1000.0
    initial_value:    float = 6082.0


def run_generational_plan(returns: pd.DataFrame, weights: np.ndarray,
                          cfg: dict, plan: GenPlanConfig = None) -> dict:
    if plan is None:
        a    = cfg.get("analysis", {})
        snap = cfg.get("ks_app_snapshot_20260327", {})
        fx   = cfg.get("meta", {}).get("fx_usd_thb", 32.68)
        plan = GenPlanConfig(
            monthly_add_usd = a.get("dca_monthly_budget_thb", 51000) / fx,
            income_rate_m   = 0.0707 / 12,
            target_income_m = float(a.get("target_monthly_income_usd", 1000.0)),
            initial_value   = float(snap.get("market_value_thb", 6082*fx)) / fx,
        )

    from engine.analytics import monte_carlo
    n_months = plan.horizon_years * 12
    vp, ip   = monte_carlo(returns, weights, plan.income_rate_m,
                            plan.initial_value, plan.monthly_add_usd,
                            plan.n_paths, n_months)

    milestones = {}
    for yr in [5, 10, 15, 20, 30]:
        if yr * 12 > n_months:
            continue
        m         = yr * 12 - 1
        val_s     = vp[:, m]
        mon_inc   = val_s * plan.income_rate_m
        defl      = (1 + plan.inflation_annual) ** yr
        prob      = float((mon_inc >= plan.target_income_m).mean() * 100)
        milestones[yr] = {
            "p10_value":    round(float(np.percentile(val_s, 10)), 0),
            "p50_value":    round(float(np.percentile(val_s, 50)), 0),
            "p90_value":    round(float(np.percentile(val_s, 90)), 0),
            "p50_real":     round(float(np.percentile(val_s, 50)) / defl, 0),
            "p50_income_m": round(float(np.percentile(mon_inc, 50)), 2),
            "p10_income_m": round(float(np.percentile(mon_inc, 10)), 2),
            "p90_income_m": round(float(np.percentile(mon_inc, 90)), 2),
            "p50_cum_div":  round(float(np.percentile(ip[:, m], 50)), 0),
            "prob_above_target": round(prob, 1),
        }
        # Use %s with f-strings to avoid %,d issue
        log.info("  Year %2d: p50=$%s  income=$%s/mo  P(>target)=%.1f%%",
                 yr,
                 f"{milestones[yr]['p50_value']:,.0f}",
                 f"{milestones[yr]['p50_income_m']:,.2f}",
                 prob)

    mtt = _time_to_target(vp, plan.income_rate_m, plan.target_income_m, n_months)
    buf = _plot(vp, ip, plan, milestones)
    return {"milestones": milestones, "months_to_target": mtt, "plan": plan,
            "chart_bytes": buf, "value_paths": vp, "income_paths": ip}


def _time_to_target(vp, rate, target, n_months):
    for m in range(n_months):
        if (vp[:, m] * rate >= target).mean() >= 0.50:
            yr = (m+1) // 12; mo = (m+1) % 12
            return {"months": m+1, "years": yr, "extra_months": mo}
    return {"months": None, "years": None, "extra_months": None}


def _plot(vp, ip, plan, milestones) -> bytes:
    try:
        n_months = vp.shape[1]; x = np.arange(1, n_months+1) / 12
        fig, axes = plt.subplots(2, 1, figsize=(11, 9), facecolor="#FAFBFD")
        fig.suptitle(f"Generational Wealth Plan  --  {plan.n_paths:,} paths",
                     fontsize=13, fontweight="bold")

        ax1 = axes[0]; ax1.set_facecolor("#FAFBFD")
        p10 = np.percentile(vp, 10, axis=0)
        p50 = np.percentile(vp, 50, axis=0)
        p90 = np.percentile(vp, 90, axis=0)
        ax1.fill_between(x, p10, p90, alpha=0.15, color="#1D9E75")
        ax1.plot(x, p50, color="#1D9E75", lw=2.2, label="Median (p50)")
        ax1.plot(x, p10, color="#1D9E75", lw=0.8, ls="--")
        ax1.plot(x, p90, color="#1D9E75", lw=0.8, ls="--")
        for yr, ms in milestones.items():
            ax1.annotate(f"Y{yr}\n${ms['p50_value']/1000:.0f}k",
                         xy=(yr, ms["p50_value"]), xytext=(yr+0.3, ms["p50_value"]*1.06),
                         fontsize=8, color="#085041",
                         arrowprops=dict(arrowstyle="->", color="#085041", lw=0.8))
        ax1.set_ylabel("Portfolio Value (USD)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}k"))
        ax1.set_title("Portfolio value over time", fontsize=10)
        ax1.legend(fontsize=9); ax1.grid(color="#E4E8F0", lw=0.6)
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

        ax2 = axes[1]; ax2.set_facecolor("#FAFBFD")
        mon_inc = vp * plan.income_rate_m
        i10 = np.percentile(mon_inc, 10, axis=0)
        i50 = np.percentile(mon_inc, 50, axis=0)
        i90 = np.percentile(mon_inc, 90, axis=0)
        ax2.fill_between(x, i10, i90, alpha=0.15, color="#3B8BD4")
        ax2.plot(x, i50, color="#3B8BD4", lw=2.2, label="Median monthly income")
        ax2.plot(x, i10, color="#3B8BD4", lw=0.8, ls="--")
        ax2.plot(x, i90, color="#3B8BD4", lw=0.8, ls="--")
        ax2.axhline(plan.target_income_m, color="#BA7517", lw=1.5, ls="--",
                    label=f"Target: ${plan.target_income_m:,.0f}/mo")
        ax2.set_xlabel("Years"); ax2.set_ylabel("Monthly Passive Income (USD)")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        ax2.set_title("Monthly dividend income projection", fontsize=10)
        ax2.legend(fontsize=9); ax2.grid(color="#E4E8F0", lw=0.6)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig); buf.seek(0); return buf.read()
    except Exception as e:
        log.warning("Generational chart failed: %s", e); return b""
