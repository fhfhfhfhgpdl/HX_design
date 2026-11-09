import numpy as np
import matplotlib.pyplot as plt
import calculation_func as hxc

N_segments = 100

X_h = 60.0 
X_c = 57.0

d_vi, d_vo = hxc.hyd_dia(hxc.d_bi_ref, hxc.d_ei_ref, hxc.n_s_ref)
theta, e = hxc.parameters(d_vo, hxc.d_eo_ref, hxc.d_bo_ref, hxc.n_s_ref, hxc.p_ref)

Th_nodes = np.linspace(50.0, 80.0, N_segments + 1)
Tc_current = 45.0  

h_tube_list = []
h_ann_list = []
U_list = []
L_total = 0.0


for i in range(N_segments):
    Th_current = Th_nodes[i]
    Th_next = Th_nodes[i+1]
    Th_avg = (Th_current + Th_next) / 2.0
    
    Cp_h = hxc.calc_cp_libr(Th_avg, X_h)
    rho_h = hxc.density_libr(Th_avg, X_h)
    mu_h = hxc.viscosity_libr(Th_avg, X_h)
    k_h = hxc.k_libr(Th_avg, X_h)
    
    dQ = hxc.m_h * Cp_h * (Th_next - Th_current) 
    
    h_t = hxc.nd_tube(Cp_h, rho_h, mu_h, k_h, e, hxc.p_ref, theta, d_vi)

    
    Cp_c_in = hxc.calc_cp_libr(Tc_current, X_c)
    dT_c = dQ / (hxc.m_c * Cp_c_in)
    Tc_next = Tc_current + dT_c
    Tc_avg = (Tc_current + Tc_next) / 2.0
    
    Cp_c = hxc.calc_cp_libr(Tc_avg, X_c)
    rho_c = hxc.density_libr(Tc_avg, X_c)
    mu_c = hxc.viscosity_libr(Tc_avg, X_c)
    k_c = hxc.k_libr(Tc_avg, X_c)
    
    h_a = hxc.nd_annular(Cp_c, rho_c, mu_c, k_c, e, hxc.p_ref, theta, d_vo, hxc.d_oi_ref)

    
    R_conv_i = (d_vo / d_vi) * (1.0 / h_t)
    R_cond   = (d_vo / (2.0 * hxc.k_w)) * np.log(d_vo / d_vi)
    R_conv_o = 1.0 / h_a
    U_o = 1.0 / (R_conv_i + R_cond + R_conv_o)
    
    h_tube_list.append(h_t)
    h_ann_list.append(h_a)
    U_list.append(U_o)
    
    dT1 = Th_next - Tc_next
    dT2 = Th_current - Tc_current
    LMTD_local = (dT1 - dT2) / np.log(dT1 / dT2) if abs(dT1 - dT2) > 1e-5 else dT1
        
    dA = dQ / (U_o * LMTD_local)
    dL = dA / (np.pi * d_vo)
    
    L_total += dL
    Tc_current = Tc_next 

print(f"최종 계산된 튜브 길이 (L_tot) : {L_total:.4f} m")
print(f"차가운 유체 최종 출구 온도 (Tc_out) : {Tc_current:.4f} ℃")

h_tube_list.reverse()
h_ann_list.reverse()
U_list.reverse()
Th_nodes_plot = np.linspace(80.0, 50.0, N_segments)

plt.figure(figsize=(10, 6))
x_axis = np.arange(1, N_segments + 1)

plt.plot(x_axis, h_tube_list, label='Tube Side ($h_t$)', color='red', linewidth=2)
plt.plot(x_axis, h_ann_list, label='Annular Side ($h_a$)', color='blue', linewidth=2)
plt.plot(x_axis, U_list, label='Overall HTC($U$)', color='black', linestyle='--', linewidth=2.5)

plt.title('Heat Transfer Coefficients along the Heat Exchanger (100 Segments)', fontsize=14, fontweight='bold')
plt.xlabel('Segment Index (From Cold End 50℃ -> Hot End 80℃)', fontsize=12)
plt.ylabel('Heat Transfer Coefficient (W/m²·K)', fontsize=12)
plt.xlim(0, 100)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

# 그래프를 띄우기 전에 그림 파일로 먼저 저장!
plt.savefig('plot_result.png', dpi=300)
plt.show()