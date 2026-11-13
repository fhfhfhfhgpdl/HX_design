import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import calculation_func as hxc
import sum_length as ref

MM, PD_LIM = 1e-3, 50e3
DERIVE_GEOMETRY = True 

def derived(d_bi, d_eo):
    if not DERIVE_GEOMETRY: return hxc.d_bo_ref, hxc.d_ei_ref
    return d_bi + 2*hxc.tw, d_eo - 2*hxc.tw

BOUNDS = dict(d_eo=(50, 200), d_bi=(10, 180), d_oi=(80, 220), p=(2.0, 80))
NS_LIST = range(2, 41)

USE_VALIDITY = True
USE_RSTAR    = False
V = dict(e_t=(0.11, 0.42), p_t=(0.41, 7.29), th=(0.28, 0.65),
         e_a=(0.104, 0.384), p_a=(0.383, 6.714), r=(0.388, 0.604))
V2=dict(e_t=(0.001, 0.698), p_t=(0.01, 25.2))

def rv(v, lo, hi): return max(0, lo - v)/lo + max(0, v - hi)/hi
def evaluate(x, ns):
  d_eo, d_bi, d_oi, p = [v * MM for v in x]
  d_bo, d_ei = derived(d_bi, d_eo)
  d_vi, d_vo = hxc.hyd_dia(d_bi, d_ei, ns)

  # 1. 기하학적 간섭 체크
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
    v = rv(e/d_vi, *V2["e_t"]) + rv(p/d_vi, *V2["p_t"])
    '''v = (rv(e / d_vi, *V["e_t"])
        + rv(p / d_vi, *V["p_t"])
        + rv(e / d_vo, *V["e_a"])
        + rv(p / d_vo, *V["p_a"])
        + rv(theta / 90, *V["th"]))'''
    if USE_RSTAR:
      v += rv(d_vo / d_oi, *V["r"])

  try:
    res_sim = hxc.run_simulation(
        ref.N_segments, ref.X_h, ref.X_c, d_bi, d_eo, ns, p, d_oi
    )
    L, PDt, PDa = res_sim[0], res_sim[1], res_sim[2]

    Th_avg = 65.0
    rho_h = hxc.density_libr(Th_avg, ref.X_h)
    mu_h = hxc.viscosity_libr(Th_avg, ref.X_h)
    V_t = 4 * hxc.m_h / (np.pi * rho_h * (d_vi**2))
    Re_t = V_t * rho_h * d_vi / mu_h

    Tc_avg = 61.5
    rho_c = hxc.density_libr(Tc_avg, ref.X_c)
    mu_c = hxc.viscosity_libr(Tc_avg, ref.X_c)
    d_hyd = d_oi - d_vo
    A_eff = np.pi / 4 * (d_oi**2 - d_vo**2)
    V_a = hxc.m_c / rho_c / A_eff
    Re_a = V_a * rho_c * d_hyd / mu_c

  except Exception as err:
    return None, 1.0

  if not np.isfinite([L, PDt, PDa]).all() or L <= 0:
    return None, 1.0

  geo_data = (d_vi, d_vo, theta, e, Re_t, Re_a)
  return (L, PDt, PDa, v, geo_data), 0.0

def objective(x, ns):
    lo = np.array([BOUNDS[k][0] for k in ("d_eo","d_bi","d_oi","p")])
    hi = np.array([BOUNDS[k][1] for k in ("d_eo","d_bi","d_oi","p")])
    xc = np.clip(x, lo, hi)
    out = float(np.sum(np.abs(x-xc)/hi))*1e3
    return out + _objective(xc, ns)

def _objective(x, ns):
    res, gv = evaluate(x, ns)
    if res is None:
        return 1e4 + gv * 100  
    
    L, PDt, PDa, v, _ = res
    
    pd_penalty = 0.0
    if PDt > PD_LIM:
        pd_penalty += (PDt - PD_LIM) / 1000.0 * 50.0
    if PDa > PD_LIM:
        pd_penalty += (PDa - PD_LIM) / 1000.0 * 50.0
        
    return L + pd_penalty + 1e3 * v

def solve_ns(ns, seed=42):
    b = [BOUNDS[k] for k in ("d_eo", "d_bi", "d_oi", "p")]
    
    de = differential_evolution(objective, b, args=(ns,), seed=seed, maxiter=50, popsize=15,
                                 tol=1e-4, polish=False, updating="deferred")
    nm = minimize(objective, de.x, args=(ns,), method="Nelder-Mead",
                  options=dict(xatol=1e-2, fatol=1e-4, maxiter=500))
    x = nm.x if nm.fun < de.fun else de.x
    res, _ = evaluate(x, ns)
    
    ok = (res is not None) and (res[1] <= PD_LIM * 1.01) and (res[2] <= PD_LIM * 1.01) and (res[3] < 1e-9)
    return x, res, ok

if __name__ == "__main__":
    out = []
    results_list = []

    print("최적화 탐색을 시작합니다...")
    for ns in NS_LIST:
        x, res, ok = solve_ns(ns)
        if res is None or not ok:
            print(f"N_s={ns:2d}: 제약 만족 해 없음 (압력 초과 또는 유로 간섭)")
            continue
        
        L, PDt, PDa, v, geo_data = res
        d_eo, d_bi, d_oi, p = x
        d_vi, d_vo, theta, e, Re_t, Re_a = geo_data
        
        out.append((L, ns, x, res))
        
        results_list.append({
            "N_s": ns,
            "L_m": round(L, 4),
            "PD_t_kPa": round(PDt / 1e3, 2),
            "PD_a_kPa": round(PDa / 1e3, 2),
            "D_eo_mm": round(d_eo, 2),
            "D_bi_mm": round(d_bi, 2),
            "D_oi_mm": round(d_oi, 2),
            "p_mm": round(p, 2),
            "D_vi_mm": round(d_vi * 1000, 3),
            "D_vo_mm": round(d_vo * 1000, 3),
            "theta_deg": round(theta, 2),
            "e_mm": round(e * 1000, 3),
            "e_star_t": round(e / d_vi, 4),
            "p_star_t": round(p / d_vi, 4),
            "Re_t": round(Re_t, 1),
            "Re_a": round(Re_a, 1),
            "Validity_Violation": round(v, 6)
        })
        
        print(f"N_s={ns:2d} | L={L:.4f} m | PDt={PDt/1e3:5.1f} kPa | PDa={PDa/1e3:5.1f} kPa | "
              f"D_eo={d_eo:.1f} D_bi={d_bi:.1f} D_oi={d_oi:.1f} p={p:.1f} mm")

    if results_list:
        df_results = pd.DataFrame(results_list)
        df_results = df_results.sort_values(by="L_m").reset_index(drop=True)
        df_results.to_csv("optimization_results_pv_.csv", index=False, encoding="utf-8-sig")
        print("\n[완료] 최적화 결과가 'optimization_results_pv_.csv'로 정상 저장되었습니다.")

        out.sort(key=lambda r: r[0])
        print("\n=== TOP 5 (L 오름차순) ===")
        for L, ns, x, r in out[:5]:
            print(f"L={L:.4f} m | N_s={ns:2d} | D_eo={x[0]:.2f} D_bi={x[1]:.2f} D_oi={x[2]:.2f} p={x[3]:.2f} mm "
                  f"| PDt={r[1]/1e3:.2f} PDa={r[2]/1e3:.2f} kPa")
    else:
        print("\n[경고] 유효한 해를 전혀 찾지 못했습니다. BOUNDS 범위를 점검해 보세요.")