"""
pages/p4_sandbox.py  —  Interactive Portfolio Sandbox (Phase 2 / Phase 3)
=========================================================================
Phase 2: What-If multi-asset simulator (session-state only, never touches YAML)
          + ±5% entry price sensitivity table
Phase 3: 3D Efficient Frontier (Plotly WebGL) + Three.js bubble chart
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import load_returns, t, get_lang


def render(*, active, cfg, holdings, fx_r, wht, rf_ann, tickers):
    st.subheader("🔬 " + t("whatif") + " & 3D Sandbox")
    st.caption("Sandbox mode — session-state only. Nothing touches your live portfolio.")

    tab_wi, tab_3d, tab_exit = st.tabs([
        "🔬 What-If Simulator", "🌐 3D Frontier", "🚪 " + t("exit_simulator"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # WHAT-IF SIMULATOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab_wi:
        _render_whatif(cfg, holdings, fx_r, wht, rf_ann, tickers, active)

    # ══════════════════════════════════════════════════════════════════════════
    # 3D EFFICIENT FRONTIER  (Phase 3 — Plotly WebGL, zero extra deps)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_3d:
        _render_3d_frontier(cfg, holdings, fx_r, rf_ann, tickers)

    # ══════════════════════════════════════════════════════════════════════════
    # EXIT SIMULATOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab_exit:
        _render_exit_simulator(cfg, holdings, fx_r, wht, active)


# ─────────────────────────────────────────────────────────────────────────────
def _render_whatif(cfg, holdings, fx_r, wht, rf_ann, tickers, active):
    st.markdown("**Candidate positions** — up to 3 tickers in one scenario")

    with st.form("whatif_form"):
        h1,h2,h3 = st.columns([2,2,2])
        h1.caption("Ticker"); h2.caption("USD to deploy"); h3.caption("Shares (0=use USD)")

        wi_t1 = h1.text_input("T1","PDI",label_visibility="collapsed").upper().strip()
        wi_u1 = h2.number_input("U1",0.0,10000.0,2000.0,100.0,label_visibility="collapsed")
        wi_s1 = h3.number_input("S1",0,1000,0,label_visibility="collapsed")

        wi_t2 = h1.text_input("T2","",placeholder="optional",label_visibility="collapsed").upper().strip()
        wi_u2 = h2.number_input("U2",0.0,10000.0,0.0,100.0,label_visibility="collapsed")
        wi_s2 = h3.number_input("S2",0,1000,0,label_visibility="collapsed")

        wi_t3 = h1.text_input("T3","",placeholder="optional",label_visibility="collapsed").upper().strip()
        wi_u3 = h2.number_input("U3",0.0,10000.0,0.0,100.0,label_visibility="collapsed")
        wi_s3 = h3.number_input("S3",0,1000,0,label_visibility="collapsed")

        st.divider()
        fa,fb,fc = st.columns(3)
        max_wt      = fa.slider("Max weight/ticker (0=none)", 0.0, 1.0, 0.0, 0.05)
        sensitivity = fb.checkbox("±5% entry price sensitivity", value=False)
        label       = fc.text_input("Scenario label","", placeholder="e.g. PDI 30%")
        submit      = st.form_submit_button("🔬 Run What-If", type="primary")

    if not submit or not wi_t1:
        return

    from engine.scenario_analyzer import run_addition_scenario
    candidates = [
        {"ticker":t,"usd_amount":float(u),"shares":int(s)}
        for t,u,s in [(wi_t1,wi_u1,wi_s1),(wi_t2,wi_u2,wi_s2),(wi_t3,wi_u3,wi_s3)]
        if t
    ]

    price_mults = {"Base":1.0}
    if sensitivity:
        price_mults.update({"+5% entry":1.05, "−5% entry":0.95})

    all_results: dict = {}
    for slabel, mult in price_mults.items():
        adj = [{"ticker":c["ticker"],
                "usd_amount": c["usd_amount"]/mult if c["usd_amount"]>0 else 0,
                "shares":     c["shares"]} for c in candidates]
        with st.spinner(f"Analysing {', '.join(c['ticker'] for c in candidates)} ({slabel})…"):
            all_results[slabel] = run_addition_scenario(
                cfg=cfg, candidates=adj,
                max_weight=max_wt if max_wt > 0 else None,
                mc_paths=3000, mc_months=60,
            )

    result = all_results["Base"]
    if result.get("error"):
        st.error(result["error"]); return

    # ── Sensitivity table ──────────────────────────────────────────────────────
    if sensitivity:
        st.subheader("Sensitivity — ±5% entry price")
        sens_rows = []
        for sl, sr in all_results.items():
            d = sr.get("delta",{})
            sens_rows.append({"Scenario":sl,
                "ΔSharpe":f"{d.get('sharpe',0):+.3f}",
                "ΔReturn":f"{d.get('ann_return',0)*100:+.2f}%",
                "ΔCVaR":  f"{d.get('cvar_95',0)*100:+.2f}%",
                "ΔIncome":f"${sr.get('income_delta_usd',0):+.2f}/mo",
            })
        st.dataframe(pd.DataFrame(sens_rows), width="stretch", hide_index=True)

    # ── Delta KPIs ────────────────────────────────────────────────────────────
    st.subheader("Impact — Base scenario")
    delta = result["delta"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ΔSharpe",      f"{delta.get('sharpe',0):+.3f}")
    c2.metric("ΔReturn",      f"{delta.get('ann_return',0)*100:+.2f}%")
    c3.metric("ΔVolatility",  f"{delta.get('ann_vol',0)*100:+.2f}%")
    c4.metric("ΔCVaR 95%",   f"{delta.get('cvar_95',0)*100:+.2f}%")
    c5.metric("ΔIncome/mo",  f"${result.get('income_delta_usd',0):+.2f}")

    # ── Weight comparison ──────────────────────────────────────────────────────
    bw = result.get("before_weights",{}); aw = result.get("after_weights",{})
    all_t = sorted(set(list(bw)+list(aw)))
    wdf = pd.DataFrame({
        "Ticker": all_t,
        "Before":  [f"{bw.get(t,0):.1%}" for t in all_t],
        "After":   [f"{aw.get(t,0):.1%}" for t in all_t],
        "Δ":       [f"{(aw.get(t,0)-bw.get(t,0))*100:+.1f}pp" for t in all_t],
    })
    st.dataframe(wdf, width="stretch", hide_index=True)

    # ── Transaction preview ────────────────────────────────────────────────────
    tx_prev = result.get("transactions_preview",[])
    if tx_prev:
        st.subheader("Proposed transactions")
        st.dataframe(pd.DataFrame(tx_prev), width="stretch", hide_index=True)

    # ── MC fan chart ───────────────────────────────────────────────────────────
    if result.get("mc_before") and result.get("mc_after"):
        st.subheader("Monte Carlo — Before vs After (5-year)")
        x = list(range(1,61)); fig = go.Figure()
        for vp, name in [(result["mc_before"][0],"Base"),
                         (result["mc_after"][0],"After")]:
            c  = ("Before" if name=="Base" else "After")
            pal = _MC_PALETTE.get(c if c in _MC_PALETTE else "Base", _MC_PALETTE["Base"])
            p50 = np.percentile(vp,50,axis=0)
            p10 = np.percentile(vp,10,axis=0)
            p90 = np.percentile(vp,90,axis=0)
            fig.add_trace(go.Scatter(x=x,y=p50.tolist(),name=f"{name} p50",
                          line=dict(color=pal["line"],width=2.5)))
            fig.add_trace(go.Scatter(x=x,y=p90.tolist(),name=f"{name} p90",
                          line=dict(color=pal["line"],width=0.8,dash="dot"),showlegend=False))
            fig.add_trace(go.Scatter(x=x,y=p10.tolist(),name=f"{name} p10",
                          fill="tonexty", fillcolor=pal["fill"],
                          line=dict(color=pal["line"],width=0.8,dash="dot"),showlegend=False))
        fig.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Month", yaxis_title="Portfolio (USD)",
                          plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                          font_color="#FFFFFF", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    # ── Groq summary ───────────────────────────────────────────────────────────
    try:
        from utils.llm_summarizer import summarise_whatif, render_summary
        render_summary(summarise_whatif(result), "whatif")
    except Exception:
        pass

    # ── Apply (sandbox → live) ──────────────────────────────────────────────
    st.divider()
    _sandbox = st.checkbox("🔬 Stay in sandbox (no YAML write)", value=True, key="sandbox_chk")
    if not _sandbox and tx_prev:
        tickers_str = "+".join(c["ticker"] for c in candidates).upper()
        confirm = st.text_input(f'Type "{tickers_str} CONFIRMED" to apply')
        if st.button("✅ Apply to portfolio.yaml", type="primary"):
            if confirm.strip().upper() == f"{tickers_str} CONFIRMED":
                from engine.scenario_analyzer import apply_scenario_to_config
                from core import save_cfg, github_push
                cfg2 = apply_scenario_to_config(cfg, tx_prev)
                save_cfg(cfg2, active["id"])
                gh   = github_push(cfg2, f"What-If applied: {tickers_str}")
                st.success(f"✅ Applied.  {'GitHub ✅' if gh['success'] else ''}")
                st.rerun()
            else:
                st.error("Confirmation doesn't match.")


_MC_PALETTE = {
    "Base":  {"line":"#2E5BA8","fill":"rgba(46,91,168,0.12)"},
    "After": {"line":"#1D9E75","fill":"rgba(29,158,117,0.12)"},
}


# ─────────────────────────────────────────────────────────────────────────────
# 3D EFFICIENT FRONTIER  (Phase 3 — Plotly WebGL + Three.js bubble)
# ─────────────────────────────────────────────────────────────────────────────
def _render_3d_frontier(cfg, holdings, fx_r, rf_ann, tickers):
    st.markdown("### 3D Efficient Frontier — Risk × Return × Income")
    st.caption(
        "Each point is a random portfolio. Colour = Sharpe ratio. "
        "X = volatility, Y = expected return, Z = estimated monthly income."
    )
    if not tickers or len(tickers) < 2:
        st.info("Need ≥2 holdings to build a frontier.")
        return

    n_pts = st.slider("Portfolio samples", 500, 5000, 2000, 500, key="frontier_pts")
    YIELDS = {"BKLN":0.0707,"ARCC":0.1066,"PDI":0.152,"MAIN":0.076}

    if st.button("🌐 Build 3D Frontier", type="primary", key="build_3d"):
        returns = load_returns(tickers)
        if returns.empty:
            st.warning("Insufficient return data."); return

        with st.spinner(f"Sampling {n_pts:,} portfolios via Julia…"):
            try:
                from engine.julia_bridge import ledoit_wolf_cov
                Σ = ledoit_wolf_cov(returns) * 12
            except Exception:
                Σ = np.cov(returns.values, rowvar=False) * 12
            μ = returns.mean().values * 12

        rng     = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(len(tickers)), size=n_pts)

        vols    = np.sqrt(np.einsum("ni,ij,nj->n", weights, Σ, weights))
        rets    = weights @ μ
        sharpes = (rets - rf_ann) / np.maximum(vols, 1e-9)
        incomes = np.array([
            sum(holdings.get(t,{}).get("avg_cost",0) *
                holdings.get(t,{}).get("shares",0) *
                YIELDS.get(t,0.08) / 12 * w
                for t,w in zip(tickers,wi))
            for wi in weights
        ])

        fig = go.Figure(go.Scatter3d(
            x=vols*100, y=rets*100, z=incomes,
            mode="markers",
            marker=dict(
                size=3,
                color=sharpes,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                opacity=0.7,
            ),
            text=[f"Sharpe: {s:.3f}<br>Vol: {v*100:.1f}%<br>"
                  f"Return: {r*100:.1f}%<br>Income: ${i:.2f}/mo"
                  for s,v,r,i in zip(sharpes,vols,rets,incomes)],
            hoverinfo="text",
        ))

        # Mark current holdings weight
        if holdings:
            h_n  = len(tickers)
            h_w  = np.ones(h_n)/h_n
            h_v  = float(np.sqrt(h_w @ Σ @ h_w)) * 100
            h_r  = float(h_w @ μ) * 100
            h_i  = float(sum(holdings.get(t,{}).get("avg_cost",0)*
                             holdings.get(t,{}).get("shares",0)*
                             YIELDS.get(t,0.08)/12/h_n for t in tickers))
            fig.add_trace(go.Scatter3d(
                x=[h_v], y=[h_r], z=[h_i],
                mode="markers+text", text=["Current"],
                marker=dict(size=10, color="#E24B4A", symbol="diamond"),
                name="Current portfolio",
            ))

        fig.update_layout(
            height=560,
            scene=dict(
                xaxis_title="Volatility (%)",
                yaxis_title="Expected Return (%)",
                zaxis_title="Income (USD/mo)",
                bgcolor="#0E1117",
                xaxis=dict(gridcolor="#2A3A4A"),
                yaxis=dict(gridcolor="#2A3A4A"),
                zaxis=dict(gridcolor="#2A3A4A"),
            ),
            paper_bgcolor="#0E1117",
            font_color="#FFFFFF",
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("🔴 Diamond = current equal-weight position  |  "
                   "Colour = Sharpe ratio (purple→high)")

        # ── Three.js bubble chart (Phase 3 — interactive WebGL) ──────────────
        st.markdown("### 🌐 Interactive 3D Asset Bubble Chart (WebGL)")
        _render_threejs_bubbles(holdings, tickers, YIELDS, fx_r)


def _render_threejs_bubbles(holdings, tickers, YIELDS, fx_r):
    """
    Phase 3: Three.js bubble chart injected via st.components.v1.html.
    Each asset = sphere. Size = portfolio weight. Colour = yield.
    Drag to rotate, scroll to zoom.
    """
    assets = []
    total_val = sum(
        holdings[t]["shares"] * holdings[t]["avg_cost"]
        for t in tickers if t in holdings
    )
    for i, tkr in enumerate(tickers):
        if tkr not in holdings: continue
        h   = holdings[tkr]
        val = h["shares"] * h["avg_cost"]
        assets.append({
            "name":   tkr,
            "weight": round(val/total_val, 4) if total_val > 0 else 0,
            "yield":  YIELDS.get(tkr, 0.08),
            "value":  round(val, 2),
            "income": round(val * YIELDS.get(tkr,0.08) / 12, 2),
        })

    assets_json = json.dumps(assets)
    html = f"""
<!DOCTYPE html><html><head>
<style>
  body {{ margin:0; background:#0E1117; overflow:hidden; font-family:sans-serif; }}
  canvas {{ display:block; }}
  #info {{ position:absolute; top:10px; left:10px; color:#aaa; font-size:12px; }}
  #tooltip {{ position:absolute; display:none; background:rgba(0,0,0,0.8);
              color:#fff; padding:8px 12px; border-radius:6px; font-size:13px;
              pointer-events:none; }}
</style>
</head><body>
<div id="info">Drag to rotate · Scroll to zoom</div>
<div id="tooltip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const assets = {assets_json};
const W = window.innerWidth, H = 420;
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(W, H);
renderer.setClearColor(0x0E1117);
document.body.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 1000);
camera.position.set(0, 0, 4);

const light = new THREE.PointLight(0xffffff, 1.2, 100);
light.position.set(5,5,5); scene.add(light);
scene.add(new THREE.AmbientLight(0x404040));

const colorScale = (v) => {{
  const t = Math.max(0, Math.min(1, (v-0.05)/0.12));
  const r = Math.round(46  + t*(29-46));
  const g = Math.round(91  + t*(158-91));
  const b = Math.round(168 + t*(117-168));
  return new THREE.Color(`rgb(${{r}},${{g}},${{b}})`);
}};

const spheres = []; const meshes = [];
const angles = assets.map((_,i)=>i*2*Math.PI/assets.length);
assets.forEach((a,i)=>{{
  const radius = 0.15 + a.weight * 1.2;
  const geo    = new THREE.SphereGeometry(radius, 32, 32);
  const mat    = new THREE.MeshPhongMaterial({{
    color: colorScale(a.yield), transparent:true, opacity:0.85,
    shininess:80,
  }});
  const mesh   = new THREE.Mesh(geo, mat);
  const angle  = angles[i];
  mesh.position.set(Math.cos(angle)*1.5, Math.sin(angle)*0.5, 0);
  mesh.userData = a;
  scene.add(mesh); meshes.push(mesh);
}});

// Orbit controls (manual)
let isDragging=false, prevX=0, prevY=0;
const group = new THREE.Group();
meshes.forEach(m=>group.add(m)); scene.add(group);

renderer.domElement.addEventListener('mousedown',e=>{{isDragging=true;prevX=e.clientX;prevY=e.clientY;}});
renderer.domElement.addEventListener('mousemove',e=>{{
  if(!isDragging) return;
  group.rotation.y += (e.clientX-prevX)*0.01;
  group.rotation.x += (e.clientY-prevY)*0.01;
  prevX=e.clientX; prevY=e.clientY;
}});
renderer.domElement.addEventListener('mouseup',()=>{{isDragging=false;}});
renderer.domElement.addEventListener('wheel',e=>{{camera.position.z+=e.deltaY*0.005;}});

// Tooltip on hover
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tip = document.getElementById('tooltip');
renderer.domElement.addEventListener('mousemove',e=>{{
  mouse.x = (e.clientX/W)*2-1;
  mouse.y = -(e.clientY/H)*2+1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(meshes);
  if(hits.length>0) {{
    const a = hits[0].object.userData;
    tip.style.display='block';
    tip.style.left=(e.clientX+14)+'px';
    tip.style.top=(e.clientY-30)+'px';
    tip.innerHTML=`<b>${{a.name}}</b><br>Weight: ${{(a.weight*100).toFixed(1)}}%<br>Yield: ${{(a.yield*100).toFixed(2)}}%<br>Value: $${{a.value.toLocaleString()}}<br>Income: $${{a.income.toFixed(2)}}/mo`;
  }} else {{tip.style.display='none';}}
}});

function animate(){{requestAnimationFrame(animate);if(!isDragging)group.rotation.y+=0.003;renderer.render(scene,camera);}}
animate();
</script></body></html>"""

    # Render Three.js scene via st.html (Streamlit 1.35+).
    # st.components.v1.html is deprecated after 2026-06-01.
    try:
        st.html(html)
    except AttributeError:
        # Streamlit <1.35 fallback — upgrade Streamlit to fix this warning
        st.warning("Upgrade Streamlit to ≥1.35 for full Three.js support.")
        st.code(html[:200] + "…", language="html")


# ─────────────────────────────────────────────────────────────────────────────
# EXIT SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
def _render_exit_simulator(cfg, holdings, fx_r, wht, active):
    st.subheader(t("exit_sim_title"))
    st.caption("Compute full portfolio impact of exiting a position — sandbox, no YAML write.")

    if not holdings:
        st.info("No holdings."); return

    c1,c2,c3 = st.columns(3)
    ex_ticker = c1.selectbox(t("select_position"), list(holdings.keys()), key="ex_tk")
    ex_held   = holdings[ex_ticker]["shares"]
    ex_shares = c2.slider(t("shares_to_sell"), 1, ex_held, min(ex_held,50), key="ex_sh")
    ex_price  = c3.number_input(t("exit_price"), 0.0001,
                                 float(holdings[ex_ticker]["avg_cost"]*1.5),
                                 float(holdings[ex_ticker]["avg_cost"]),
                                 format="%.4f", key="ex_pr")

    st.info(
        f"Selling **{ex_shares}/{ex_held} shares** ({ex_shares/ex_held:.0%}) "
        f"@ **${ex_price:.4f}** — gross **${ex_shares*ex_price:,.2f}** "
        f"| ฿{ex_shares*ex_price*fx_r:,.0f}"
    )

    if st.button(t("compute_exit"), type="primary", key="compute_exit"):
        from engine.exit_simulator import simulate_exit
        res = simulate_exit(cfg, ex_ticker, ex_shares, ex_price, fx_r, wht)

        if res.get("error"):
            st.error(res["error"]); return

        c1,c2,c3,c4 = st.columns(4)
        c1.metric(t("capital_gain"),
                  f"${res['pnl']['capital_gain_usd']:+,.2f}",
                  f"฿{res['pnl']['capital_gain_thb']:+,.0f}")
        c2.metric(t("lost_income"),
                  f"${res['income']['lost_net_mo']:.2f}/mo",
                  f"{res['income']['income_drop_pct']:+.1f}% portfolio")
        c3.metric("Proceeds (USD)", f"${res['trade']['net_proceeds_usd']:,.2f}")
        c4.metric("30yr income lost",
                  f"${res['generational']['future_income_lost_30yr']:,.0f}")

        xt1,xt2 = st.tabs(["P&L Detail","FX + Tax + Generational"])
        with xt1:
            rows = [
                {"Metric":"Cost basis",    "Value":f"${res['pnl']['cost_basis_usd']:,.2f}"},
                {"Metric":"Avg cost/sh",   "Value":f"${res['pnl']['avg_cost_usd']:.4f}"},
                {"Metric":"Capital gain",  "Value":f"${res['pnl']['capital_gain_usd']:+,.2f} ({res['pnl']['gain_pct']:+.2f}%)"},
                {"Metric":"CGT (Thailand)","Value":"฿0 — foreign ETF exempt"},
                {"Metric":"Lost income/mo","Value":f"${res['income']['lost_net_mo']:.2f}"},
                {"Metric":"Remaining income","Value":f"${res['income']['remaining_net_mo']:.2f}/mo"},
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            conc = res["portfolio"]["concentration_after"]
            if conc:
                cdf = pd.DataFrame([{"Ticker":k,"Weight":f"{v:.1%}",
                                      "Flag":"⚠️ High" if v>0.5 else "✅"}
                                    for k,v in sorted(conc.items(),key=lambda x:-x[1])])
                st.dataframe(cdf, width="stretch", hide_index=True)
        with xt2:
            st.metric("Proceeds THB", f"฿{res['fx']['proceeds_thb']:,.0f}")
            st.metric("Cost THB",     f"฿{res['fx']['cost_thb']:,.0f}")
            st.metric("30yr loss",    f"${res['generational']['future_income_lost_30yr']:,.0f}")
            st.caption(res["tax"]["note"])

        st.divider()
        st.info(res["recommendation"])
