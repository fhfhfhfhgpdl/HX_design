import numpy as np
import calculation_func as hxc
import sum_length as ref

# ==========================================
# 1. 사용자 입력 파라미터 정의 (원하는 값으로 수정 가능)
# ==========================================
d_eo = 138.11 / 1000  # m
d_bi = 71.36/1000  # m
d_oi = 147.87/1000  # m
p    = 34.38/1000  # m
n_s  = 16
d_ei = d_eo - 2*hxc.tw
d_bo = d_bi + 2*hxc.tw

# ==========================================
# 2. 형상 치수 (Geometry Parameters) 계산
# ==========================================
d_vi, d_vo = hxc.hyd_dia(d_bi, d_eo, n_s)
theta, e = hxc.parameters(d_vo, d_eo, d_bi, n_s, p)
d_hyd = d_oi - d_vo

# ==========================================
# 3. 열교환기 시뮬레이션 실행
# ==========================================
L_tot, PD_t, PD_a, PD_t_sum, PD_a_sum, Tc_out, h_t, h_a, U = hxc.run_simulation(
    n_s, ref.X_h, ref.X_c,
    d_bi, d_eo, n_s, p, d_oi
)

# ==========================================
# 4. 터미널 출력
# ==========================================
print("=" * 50)
print("[ Input Parameters ]")
print(f"  - D_eo : {d_eo*1000:6.2f} mm")
print(f"  - D_bi : {d_bi*1000:6.2f} mm")
print(f"  - D_oi : {d_oi*1000:6.2f} mm")
print(f"  - p    : {p*1000:6.2f} mm")
print(f"  - N_s  : {n_s:6.2f}")
print("-" * 50)
print("[ Calculated Geometry Parameters ]")
print(f"  - Tube Equivalent Inner Dia (D_vi) : {d_vi*1000:6.3f} mm")
print(f"  - Tube Equivalent Outer Dia (D_vo) : {d_vo*1000:6.3f} mm")
print(f"  - Annular Hydraulic Dia (D_hyd)    : {d_hyd*1000:6.3f} mm")
print(f"  - Helix Angle (theta)              : {theta:6.3f} deg")
print(f"  - Rib Height (e)                   : {e*1000:6.3f} mm")
print("-" * 50)
print("[ Simulation Results ]")
print(f"  - Total Tube Length (L_tot)        : {L_tot:.4f} m")
print(f"  - Cold Fluid Outlet Temp (Tc_out)  : {Tc_out:.2f} ℃")
print(f"  - Tube Side Pressure Drop (PD_t)   : {PD_t/1000:.3f} kPa")
print(f"  - Annular Side Pressure Drop(PD_a) : {PD_a/1000:.3f} kPa")
print("=" * 50)