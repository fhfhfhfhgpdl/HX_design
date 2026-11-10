import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
\
import calculation_func as hxc

st.set_page_config(
    page_title="Heat Exchanger Parametric Dashboard", layout="wide"
)

st.title("Heat Exchanger Parametric Analysis Dashboard")


d_eo_ref = hxc.d_eo_ref
d_bi_ref = hxc.d_bi_ref
d_oi_ref = hxc.d_oi_ref
p_ref = hxc.p_ref
n_s_ref = hxc.n_s_ref

N_segments = 100
X_h = 60.0
X_c = 57.0

L_tot_ref, PD_t_ref, PD_a_ref, _, _, Tc_out_ref, _, _, _ = hxc.run_simulation(
    N_segments,
    X_h,
    X_c,
    d_bi_ref,
    hxc.d_ei_ref,
    d_eo_ref,
    hxc.d_bo_ref,
    n_s_ref,
    p_ref,
    d_oi_ref,
)

PD_t_star_limit = 50000 / PD_t_ref
PD_a_star_limit = 50000 / PD_a_ref


def find_intersections(x_arr, y_arr, limit_val):
  intersections = []
  for i in range(len(x_arr) - 1):
    y1, y2 = y_arr[i], y_arr[i + 1]
    if (y1 - limit_val) * (y2 - limit_val) <= 0 and y1 != y2:
      x1, x2 = x_arr[i], x_arr[i + 1]
      x_int = x1 + (limit_val - y1) * (x2 - x1) / (y2 - y1)
      intersections.append(x_int)
  return intersections


st.sidebar.header("⚙️ Fixed Parameters (무차원 수 입력)")

fix_d_eo_star = st.sidebar.number_input(
    "D_eo*",
    value=0.842,
    min_value=0.801,
    max_value=1.200,
    step=0.005,
    format="%.3f",
)
fix_d_bi_star = st.sidebar.number_input(
    "D_bi*",
    value=1.000,
    min_value=0.700,
    max_value=1.200,
    step=0.005,
    format="%.3f",
)
fix_d_oi_star = st.sidebar.number_input(
    "D_oi*",
    value=1.000,
    min_value=0.700,
    max_value=1.200,
    step=0.005,
    format="%.3f",
)
fix_p_star = st.sidebar.number_input(
    "Pitch p*",
    value=1.000,
    min_value=0.700,
    max_value=1.200,
    step=0.005,
    format="%.3f",
)
fix_n_s_star = st.sidebar.number_input(
    "Starts N_s*",
    value=1.000,
    min_value=0.700,
    max_value=1.200,
    step=0.005,
    format="%.3f",
)

fix_d_eo = fix_d_eo_star * d_eo_ref
fix_d_bi = fix_d_bi_star * d_bi_ref
fix_d_oi = fix_d_oi_star * d_oi_ref
fix_p = fix_p_star * p_ref
fix_n_s = fix_n_s_star * n_s_ref

d_vi_cur, d_vo_cur = hxc.hyd_dia(fix_d_bi, hxc.d_ei_ref, fix_n_s)
theta_cur, e_cur = hxc.parameters(
    d_vo_cur, fix_d_eo, hxc.d_bo_ref, fix_n_s, fix_p
)
d_hyd_cur = fix_d_oi - d_vo_cur

L_tot_cur, PD_t_cur, PD_a_cur, _, _, Tc_out_cur, _, _, _ = hxc.run_simulation(
    N_segments,
    X_h,
    X_c,
    fix_d_bi,
    hxc.d_ei_ref,
    fix_d_eo,
    hxc.d_bo_ref,
    fix_n_s,
    fix_p,
    fix_d_oi,
)

st.subheader("📌 Current Geometric Parameters (계산된 형상 치수)")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "D_eo (실제값)", f"{fix_d_eo*1000:.2f} mm", f"D_eo* = {fix_d_eo_star:.3f}"
)
col2.metric(
    "D_bi (실제값)", f"{fix_d_bi*1000:.2f} mm", f"D_bi* = {fix_d_bi_star:.3f}"
)
col3.metric(
    "D_oi (실제값)", f"{fix_d_oi*1000:.2f} mm", f"D_oi* = {fix_d_oi_star:.3f}"
)
col4.metric(
    "Pitch p (실제값)", f"{fix_p*1000:.2f} mm", f"p* = {fix_p_star:.3f}"
)
col5.metric(
    "Starts N_s (실제값)", f"{fix_n_s:.2f}", f"N_s* = {fix_n_s_star:.3f}"
)

col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
col_g1.metric("Tube Inner Dia (D_vi)", f"{d_vi_cur*1000:.2f} mm")
col_g2.metric("Tube Outer Dia (D_vo)", f"{d_vo_cur*1000:.2f} mm")
col_g3.metric("Hydraulic Dia (D_hyd)", f"{d_hyd_cur*1000:.2f} mm")
col_g4.metric("Rib Height (e)", f"{e_cur*1000:.2f} mm")
col_g5.metric("Helix Angle (θ)", f"{theta_cur:.2f}°")

st.markdown("---")

st.subheader("📊 Performance Results")
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric(
    "Total Length (L)",
    f"{L_tot_cur:.3f} m",
    f"L* = {L_tot_cur/L_tot_ref:.3f}",
)
col_m2.metric(
    "Tube PD (ΔP_t)",
    f"{PD_t_cur/1000:.1f} kPa",
    f"ΔP_t* = {PD_t_cur/PD_t_ref:.3f}",
)
col_m3.metric(
    "Annular PD (ΔP_a)",
    f"{PD_a_cur/1000:.1f} kPa",
    f"ΔP_a* = {PD_a_cur/PD_a_ref:.3f}",
)
col_m4.metric("Cold Outlet Temp (Tc_out)", f"{Tc_out_cur:.2f} ℃")
col_m5.metric(
    "Heat Duty (Q)", f"{(3.5 * hxc.calc_cp_libr(65, X_h) * 30 / 1000):.1f} kW"
)

st.markdown("---")

param_configs = {
    "d_eo": {
        "name": "D_eo",
        "ref": d_eo_ref,
        "val": fix_d_eo,
        "range": (0.801, 1.2),
    },
    "d_bi": {
        "name": "D_bi",
        "ref": d_bi_ref,
        "val": fix_d_bi,
        "range": (0.8, 1.2),
    },
    "d_oi": {
        "name": "D_oi",
        "ref": d_oi_ref,
        "val": fix_d_oi,
        "range": (0.7, 1.2),
    },
    "n_s": {
        "name": "N_s",
        "ref": n_s_ref,
        "val": fix_n_s,
        "range": (0.8, 1.2),
    },
    "p": {"name": "p", "ref": p_ref, "val": fix_p, "range": (0.7, 1.2)},
}

N_pts = 60
cols = st.columns(2)

for idx, (p_key, cfg) in enumerate(param_configs.items()):
  x_star = np.linspace(cfg["range"][0], cfg["range"][1], N_pts)
  L_star, PD_t_star, PD_a_star = [], [], []

  for s in x_star:
    v_act = s * cfg["ref"]
    kw = {
        "N_segments": N_segments,
        "X_h": X_h,
        "X_c": X_c,
        "d_bi": fix_d_bi,
        "d_ei": hxc.d_ei_ref,
        "d_eo": fix_d_eo,
        "d_bo": hxc.d_bo_ref,
        "n_s_val": fix_n_s,
        "p_val": fix_p,
        "d_oi": fix_d_oi,
    }
    if p_key == "d_eo":
      kw["d_eo"] = v_act
    elif p_key == "d_bi":
      kw["d_bi"] = v_act
    elif p_key == "d_oi":
      kw["d_oi"] = v_act
    elif p_key == "p":
      kw["p_val"] = v_act
    elif p_key == "n_s":
      kw["n_s_val"] = v_act

    L_t, pd_t, pd_a, _, _, _, _, _, _ = hxc.run_simulation(
        kw["N_segments"],
        kw["X_h"],
        kw["X_c"],
        kw["d_bi"],
        kw["d_ei"],
        kw["d_eo"],
        kw["d_bo"],
        kw["n_s_val"],
        kw["p_val"],
        kw["d_oi"],
    )
    L_star.append(L_t / L_tot_ref)
    PD_t_star.append(pd_t / PD_t_ref)
    PD_a_star.append(pd_a / PD_a_ref)

  x_arr = np.array(x_star)
  pdt_arr = np.array(PD_t_star)
  pda_arr = np.array(PD_a_star)

  t_int = find_intersections(x_arr, pdt_arr, PD_t_star_limit)
  a_int = find_intersections(x_arr, pda_arr, PD_a_star_limit)

  fig = make_subplots(specs=[[{"secondary_y": True}]])

  fig.add_trace(
      go.Scatter(
          x=x_arr, y=L_star, name="L*", line=dict(color="red", width=2)
      ),
      secondary_y=False,
  )
  fig.add_trace(
      go.Scatter(
          x=x_arr,
          y=pdt_arr,
          name="ΔP_t*",
          line=dict(color="blue", dash="dash", width=2),
      ),
      secondary_y=True,
  )
  fig.add_trace(
      go.Scatter(
          x=x_arr,
          y=pda_arr,
          name="ΔP_a*",
          line=dict(color="green", dash="dashdot", width=2),
      ),
      secondary_y=True,
  )

  fig.add_hline(
      y=PD_t_star_limit,
      line=dict(color="blue", dash="dot"),
      secondary_y=True,
  )
  fig.add_hline(
      y=PD_a_star_limit,
      line=dict(color="green", dash="dot"),
      secondary_y=True,
  )

  cur_star_val = cfg["val"] / cfg["ref"]
  fig.add_vline(
      x=cur_star_val, line=dict(color="gray", width=2, dash="solid")
  )

  for xi in t_int:
    fig.add_trace(
        go.Scatter(
            x=[xi],
            y=[PD_t_star_limit],
            mode="markers+text",
            name="Tube Int",
            text=[f"({xi:.3f}, {PD_t_star_limit:.3f})"],
            textposition="top left",
            marker=dict(color="red", size=8),
        ),
        secondary_y=True,
    )
  for xi in a_int:
    fig.add_trace(
        go.Scatter(
            x=[xi],
            y=[PD_a_star_limit],
            mode="markers+text",
            name="Annular Int",
            text=[f"({xi:.3f}, {PD_a_star_limit:.3f})"],
            textposition="bottom right",
            marker=dict(color="green", size=8),
        ),
        secondary_y=True,
    )

  fig.update_layout(
      title=f"Parametric Study: {cfg['name']}*",
      hovermode="x unified",
      height=380,
      margin=dict(l=20, r=20, t=40, b=20),
  )
  fig.update_xaxes(title_text=f"{cfg['name']}*")
  fig.update_yaxes(title_text="Length (L*)", secondary_y=False)
  fig.update_yaxes(title_text="Pressure Drop (ΔP*)", secondary_y=True)

  with cols[idx % 2]:
    st.plotly_chart(fig, use_container_width=True)