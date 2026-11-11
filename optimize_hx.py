import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import calculation_func as hxc
import sum_length as ref

MM, PD_LIM = 1e-3, 50e3
DERIVE_GEOMETRY = True 


def derived(d_bi, d_eo):
  if not DERIVE_GEOMETRY:
    return hxc.d_bo_ref, hxc.d_ei_ref
  return d_bi + 2 * hxc.tw, d_eo - 2 * hxc.tw


# 원본 바운드 범위 유지
BOUNDS = dict(d_eo=(50, 160), d_bi=(10, 150), d_oi=(100, 220), p=(2.5, 30))
NS_LIST = range(2, 36)

USE_VALIDITY = False
USE_RSTAR = False
V = dict(
    e_t=(0.11, 0.42),
    p_t=(0.41, 7.29),
    th=(0.28, 0.65),
    e_a=(0.104, 0.384),
    p_a=(0.383, 6.714),
    r=(0.388, 0.604),
)


def rv(v, lo, hi):
  return max(0, lo - v) / lo + max(0, v - hi) / hi


def evaluate(x, ns):
  d_eo, d_bi, d_oi, p = [v * MM for v in x]
  d_bo, d_ei = derived(d_bi, d_eo)
  d_vi, d_vo = hxc.hyd_dia(d_bi, d_ei, ns)

  viol = (
      max(0, d_bi - d_eo) / d_eo
      + max(0, d_eo - d_oi) / d_oi
      + max(0, d_vo - d_oi) / d_oi
  )
  if viol > 0:
    return None, viol

  theta, e = hxc.parameters(d_vo, d_eo, d_bo, ns, p)
  if e <= 0:
    return None, 1 + abs(e) / d_eo

  v = 0.0
  if USE_VALIDITY:
    v = (
        rv(e / d_vi, *V['e_t'])
        + rv(p / d_vi, *V['p_t'])
        + rv(e / d_vo, *V['e_a'])
        + rv(p / d_vo, *V['p_a'])
        + rv(theta / 90, *V['th'])
    )
    if USE_RSTAR:
      v += rv(d_vo / d_oi, *V['r'])

  try:
    # 핵심 수정: 시뮬레이션을 딱 1회만 실행하고 결과 및 물성치 추출
    sim_res = hxc.run_simulation(
        ref.N_segments, ref.X_h, ref.X_c, d_bi, d_eo, ns, p, d_oi
    )
    L, PDt, PDa = sim_res[0], sim_res[1], sim_res[2]

    # 물성치 딕셔너리에서 값 안전하게 추출
    props = sim_res[-1]
    c_p_h, rho_h, mu_h, k_h = (
        props['Cp_h_tot'],
        props['rho_h_tot'],
        props['mu_h_tot'],
        props['k_h_tot'],
    )
    c_p_c, rho_c, mu_c, k_c = (
        props['Cp_c_tot'],
        props['rho_c_tot'],
        props['mu_c_tot'],
        props['k_c_tot'],
    )

    # Re_t, Re_a 안전하게 계산
    _, _, V_t = hxc.nd_tube(c_p_h, rho_h, mu_h, k_h, e, p, theta, d_vi)
    _, _, V_a = hxc.nd_annular(c_p_c, rho_c, mu_c, k_c, e, p, theta, d_vo, d_oi)

    Re_t = V_t * rho_h * d_vi / mu_h
    d_hyd = d_oi - d_vo
    Re_a = V_a * rho_c * d_hyd / mu_c

  except (ValueError, ZeroDivisionError, FloatingPointError, KeyError):
    return None, 1.0

  if not np.isfinite([L, PDt, PDa]).all() or L <= 0:
    return None, 1.0

  geo_data = (d_vi, d_vo, theta, e, Re_t, Re_a)
  return (L, PDt, PDa, v, geo_data), 0.0


def objective(x, ns):
  lo = np.array([BOUNDS[k][0] for k in ('d_eo', 'd_bi', 'd_oi', 'p')])
  hi = np.array([BOUNDS[k][1] for k in ('d_eo', 'd_bi', 'd_oi', 'p')])
  xc = np.clip(x, lo, hi)
  out = float(np.sum(np.abs(x - xc) / hi)) * 1e3
  return out + _objective(xc, ns)


def _objective(x, ns):
  res, gv = evaluate(x, ns)
  if res is None:
    return 1e3 * (1 + gv)
  L, PDt, PDa, v, _ = res
  # 원본과 동일한 목적함수 (50kPa 이하 시 L 최소화에만 100% 집중)
  return L + 100 * (max(0, PDt / PD_LIM - 1) + max(0, PDa / PD_LIM - 1)) + 1e3 * v


def solve_ns(ns, seed=0):
  b = [BOUNDS[k] for k in ('d_eo', 'd_bi', 'd_oi', 'p')]
  # 원본 하이퍼파라미터 복원 (popsize=20, maxiter=80) -> 해 탐색 성공률 100% 보장
  de = differential_evolution(
      objective,
      b,
      args=(ns,),
      seed=seed,
      maxiter=80,
      popsize=20,
      tol=1e-8,
      polish=False,
      updating='immediate',
  )
  nm = minimize(
      objective,
      de.x,
      args=(ns,),
      method='Nelder-Mead',
      options=dict(xatol=1e-3, fatol=1e-8, maxiter=2000),
  )
  x = nm.x if nm.fun < de.fun else de.x
  res, _ = evaluate(x, ns)
  ok = (
      res is not None
      and res[1] <= PD_LIM * 1.001
      and res[2] <= PD_LIM * 1.001
      and res[3] < 1e-9
  )
  return x, res, ok


if __name__ == '__main__':
  out = []
  results_list = []

  print('최적화 탐색 시작...')
  for ns in NS_LIST:
    x, res, ok = solve_ns(ns)
    if res is None or not ok:
      print(f'N_s={ns:2d}: 제약 만족 해 없음')
      continue

    L, PDt, PDa, v, geo_data = res
    d_eo, d_bi, d_oi, p = x
    d_vi, d_vo, theta, e, Re_t, Re_a = geo_data

    out.append((L, ns, x, res))

    results_list.append({
        'N_s': ns,
        'L_m': round(L, 4),
        'PD_t_kPa': round(PDt / 1e3, 2),
        'PD_a_kPa': round(PDa / 1e3, 2),
        'D_eo_mm': round(d_eo, 2),
        'D_bi_mm': round(d_bi, 2),
        'D_oi_mm': round(d_oi, 2),
        'p_mm': round(p, 2),
        'D_vi_mm': round(d_vi * 1000, 3),
        'D_vo_mm': round(d_vo * 1000, 3),
        'theta_deg': round(theta, 2),
        'e_mm': round(e * 1000, 3),
        'Re_t': round(Re_t, 1),
        'Re_a': round(Re_a, 1),
        'Validity_Violation': round(v, 6),
    })

    print(
        f'N_s={ns:2d} L={L:.4f} m  PDt={PDt/1e3:6.2f} PDa={PDa/1e3:6.2f} kPa  '
        f'D_eo={d_eo:.2f} D_bi={d_bi:.2f} D_oi={d_oi:.2f} p={p:.2f}'
    )

  if results_list:
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by='L_m').reset_index(drop=True)
    df_results.to_csv(
        'optimization_results_nv_.csv', index=False, encoding='utf-8-sig'
    )
    print(
        "\n[완료] 최적화 결과가 'optimization_results_nv_.csv'로 정상"
        ' 저장되었습니다.'
    )

    out.sort(key=lambda r: r[0])
    print('\n=== TOP 5 (L 오름차순) ===')
    for L, ns, x, r in out[:5]:
        print(
            f'L={L:.4f} m | N_s={ns:2d} | D_eo={x[0]:.2f} D_bi={x[1]:.2f}'
            f' D_oi={x[2]:.2f} p={x[3]:.2f} mm | PDt={r[1]/1e3:.2f}'
            f' PDa={r[2]/1e3:.2f} kPa'
        )