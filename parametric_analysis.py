import numpy as np
import matplotlib.pyplot as plt
import calculation_func as hxc
import sum_length as ref

d_eo_ref = hxc.d_eo_ref
L_tot_ref = ref.L_tot_ref
PD_t_ref = ref.PD_t_ref
PD_a_ref = ref.PD_a_ref

PD_t_star_limit = 50000 / PD_t_ref
PD_a_star_limit = 50000 / PD_a_ref

N_points = 100
d_eo_star_list = np.linspace(0.801, 1.2, N_points)

L_star_list = []
PD_t_star_list = []
PD_a_star_list = []

for d_eo_star in d_eo_star_list:
    d_eo_val = d_eo_star * d_eo_ref
    
    L_tot, PD_t, PD_a, _, _, _, _, _, _ = hxc.run_simulation(
        ref.N_segments, ref.X_h, ref.X_c,
        hxc.d_bi_ref, hxc.d_ei_ref, d_eo_val, hxc.d_bo_ref,
        hxc.n_s_ref, hxc.p_ref, hxc.d_oi_ref
    )
    
    # Calculate non-dimensional values
    L_star_list.append(L_tot / L_tot_ref)
    PD_t_star_list.append(PD_t / PD_t_ref)
    PD_a_star_list.append(PD_a / PD_a_ref)

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:red'
ax1.set_xlabel('Non-dimensional Outer Diameter ($D_{eo}^*$)', fontsize=12)
ax1.set_ylabel('Non-dimensional Length ($L^*$)', color=color1, fontsize=12)
line1 = ax1.plot(d_eo_star_list, L_star_list, label='$L^*$', color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()
color2 = 'tab:blue'
color3 = 'tab:green'
ax2.set_ylabel('Non-dimensional Pressure Drop ($∆P^*$)', fontsize=12)
line2 = ax2.plot(d_eo_star_list, PD_t_star_list, label='$∆P_t^*$', color=color2, linewidth=2, linestyle='--')
line3 = ax2.plot(d_eo_star_list, PD_a_star_list, label='$∆P_a^*$', color=color3, linewidth=2, linestyle='-.')

# Add limit lines for pressure drop 50 kPa
line_lim_t = ax2.axhline(y=PD_t_star_limit, color=color2, linestyle=':', linewidth=1.5, label='$∆P_{t,limit}^*$ (50kPa)')
line_lim_a = ax2.axhline(y=PD_a_star_limit, color=color3, linestyle=':', linewidth=1.5, label='$∆P_{a,limit}^*$ (50kPa)')

lines = line1 + line2 + line3 + [line_lim_t, line_lim_a]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=10)

plt.title('Parametric Analysis: Effects of $D_{eo}^*$ ', fontsize=13, fontweight='bold')
plt.tight_layout()

plt.savefig('plot_param_D_epng', dpi=300)
plt.show()