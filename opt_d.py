import numpy as np
from scipy.optimize import minimize
import calculation_func as hxc
import sum_length as ref

# 고정 파라미터 세팅
d_eo_fixed = 37.89 / 1000  # D_eo = 37.89 mm 고정
fixed_p = hxc.p_ref
fixed_n_s = hxc.n_s_ref
N_segments = ref.N_segments
X_h = ref.X_h
X_c = ref.X_c

# 1. 목표 함수: L_tot 최소화
def objective(x):
    d_bi, d_oi = x
    try:
        L_tot, _, _, _, _, _, _, _, _ = hxc.run_simulation(
            N_segments, X_h, X_c,
            d_bi, hxc.d_ei_ref, d_eo_fixed, hxc.d_bo_ref,
            fixed_n_s, fixed_p, d_oi
        )
        return L_tot
    except:
        return 1e6  # 계산 불가 영역 회피용 페널티

# 2. 제약 조건 (COBYLA 알고리즘은 >= 0일 때 조건 만족)
def constraint_pd_t(x):
    try:
        _, PD_t, _, _, _, _, _, _, _ = hxc.run_simulation(
            N_segments, X_h, X_c, x[0], hxc.d_ei_ref, d_eo_fixed, hxc.d_bo_ref, fixed_n_s, fixed_p, x[1]
        )
        # 단위를 kPa로 줄여 스케일링 (50.0 kPa 이하)
        return 50.0 - (PD_t / 1000.0)
    except: return -1.0

def constraint_pd_a(x):
    try:
        _, _, PD_a, _, _, _, _, _, _ = hxc.run_simulation(
            N_segments, X_h, X_c, x[0], hxc.d_ei_ref, d_eo_fixed, hxc.d_bo_ref, fixed_n_s, fixed_p, x[1]
        )
        # 단위를 kPa로 줄여 스케일링 (50.0 kPa 이하)
        return 50.0 - (PD_a / 1000.0)
    except: return -1.0

# 탐색을 위한 초기 추정값 (안전한 유로 공간이 확보된 값)
x0 = [0.032, 0.075]  # D_bi = 32mm, D_oi = 75mm

# COBYLA 알고리즘용 모든 제약 조건 통합
constraints = [
    {'type': 'ineq', 'fun': constraint_pd_t},
    {'type': 'ineq', 'fun': constraint_pd_a},
    # 기하학적 제약 (서로 최소 1mm 두께/여유 확보)
    {'type': 'ineq', 'fun': lambda x: d_eo_fixed - x[0] - 0.001},  # D_eo > D_bi
    {'type': 'ineq', 'fun': lambda x: x[1] - d_eo_fixed - 0.001},  # D_oi > D_eo
    # 물리적 한계 바운더리
    {'type': 'ineq', 'fun': lambda x: x[0] - 0.025},  # D_bi 최소 25mm 이상
    {'type': 'ineq', 'fun': lambda x: 0.100 - x[1]}   # D_oi 최대 100mm 이하
]

print(f"최적화 탐색 중... (D_eo 고정: {d_eo_fixed*1000:.2f} mm)")

# COBYLA는 시뮬레이션 기반 최적화(Derivative-Free)에 매우 강력함
result = minimize(
    objective, 
    x0, 
    method='COBYLA', 
    constraints=constraints,
    options={'rhobeg': 0.005, 'maxiter': 1000, 'disp': False}
)

if result.success:
    opt_d_bi, opt_d_oi = result.x
    
    # 찾은 최적점으로 최종 결과 계산
    opt_L, opt_PD_t, opt_PD_a, _, _, _, _, _, _ = hxc.run_simulation(
        N_segments, X_h, X_c,
        opt_d_bi, hxc.d_ei_ref, d_eo_fixed, hxc.d_bo_ref,
        fixed_n_s, fixed_p, opt_d_oi
    )
    
    print("\n✅ 최적 설계 조건 찾기 성공!")
    print(f"고정 D_eo : {d_eo_fixed*1000:.2f} mm")
    print(f"최적 D_bi : {opt_d_bi*1000:.2f} mm")
    print(f"최적 D_oi : {opt_d_oi*1000:.2f} mm")
    print("-" * 35)
    print(f"결과 열교환기 길이 (L_tot) : {opt_L:.3f} m")
    print(f"결과 Tube 측 압력강하    : {opt_PD_t/1000:.1f} kPa")
    print(f"결과 Annular 측 압력강하 : {opt_PD_a/1000:.1f} kPa")
else:
    print("\n❌ 최적해를 찾지 못했습니다.")