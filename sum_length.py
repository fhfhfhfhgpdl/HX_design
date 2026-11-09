import numpy as np
import matplotlib.pyplot as plt
import calculation_func as hxc

N_segments = 100

X_h = 60.0 
X_c = 57.0
L_tot_ref, PD_t_ref, PD_a_ref, PD_t_sum_ref, PD_a_sum_ref,  Tc_out, h_tube_list, h_ann_list, U_list = hxc.run_simulation(
    N_segments, X_h, X_c,
    hxc.d_bi_ref, hxc.d_ei_ref, hxc.d_eo_ref, hxc.d_bo_ref, 
    hxc.n_s_ref, hxc.p_ref, hxc.d_oi_ref
)

print("-" * 40)
print("[ Reference Design Results ]")
print(f"Total Tube Length (L_tot_ref) : {L_tot_ref:.4f} m")
print(f"Cold Fluid Outlet (Tc_out)    : {Tc_out:.4f} ℃")
print(f"Tube Side Pressure Drop(AVG)       : {PD_t_ref:.4f} Pa")
print(f"Annular Side Pressure Drop(AVG)    : {PD_a_ref:.4f} Pa")
print(f"Tube Side Pressure Drop(Sum)       : {PD_t_sum_ref:.4f} Pa")
print(f"Annular Side Pressure Drop(Sum)    : {PD_a_sum_ref:.4f} Pa")
print("-" * 40)

h_tube_list.reverse()
h_ann_list.reverse()
U_list.reverse()

plt.figure(figsize=(10, 6))
x_axis = np.arange(0, N_segments) 

plt.plot(x_axis, h_tube_list, label='Tube Side ($h_t$)', color='red', linewidth=2)
plt.plot(x_axis, h_ann_list, label='Annular Side ($h_a$)', color='blue', linewidth=2)
plt.plot(x_axis, U_list, label='Overall ($U_o$)', color='black', linestyle='--', linewidth=2.5)

plt.title('Heat Transfer Coefficients (Reference Design)', fontsize=14, fontweight='bold')
plt.xlabel('Segment Index $i$ (Hot fluid inlet 80℃ $\\rightarrow$ outlet 50℃)', fontsize=12)
plt.ylabel('Heat Transfer Coefficient (W/m²·K)', fontsize=12)
plt.xlim(0, 100)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig('plot_result_ref.png', dpi=300)
plt.show()