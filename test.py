import numpy as np
import calculation_func as hxc
import sum_length as ref

# ==========================================
# 1. 사용자 입력 파라미터 정의 (원하는 값으로 수정 가능)
# ==========================================
user_d_eo = 36.09 / 1000  # m
user_d_bi = 24.50/1000  # m
user_d_oi = 56.00/1000  # m
user_p    = 10/1000  # m
user_n_s  = 12

# ==========================================
# 2. 형상 치수 (Geometry Parameters) 계산
# ==========================================
d_vi, d_vo = hxc.hyd_dia(user_d_bi, hxc.d_ei_ref, user_n_s)
theta, e = hxc.parameters(d_vo, user_d_eo, hxc.d_bo_ref, user_n_s, user_p)
d_hyd = user_d_oi - d_vo

# ==========================================
# 3. 열교환기 시뮬레이션 실행
# ==========================================
L_tot, PD_t, PD_a, _, _, Tc_out, _, _, _ = hxc.run_simulation(
    ref.N_segments, ref.X_h, ref.X_c,
    user_d_bi, hxc.d_ei_ref, user_d_eo, hxc.d_bo_ref,
    user_n_s, user_p, user_d_oi
)

# ==========================================
# 4. 터미널 출력
# ==========================================
print("=" * 50)
print("[ Input Parameters ]")
print(f"  - D_eo : {user_d_eo*1000:6.2f} mm")
print(f"  - D_bi : {user_d_bi*1000:6.2f} mm")
print(f"  - D_oi : {user_d_oi*1000:6.2f} mm")
print(f"  - p    : {user_p*1000:6.2f} mm")
print(f"  - N_s  : {user_n_s:6.2f}")
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