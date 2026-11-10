import numpy as np
import matplotlib.pyplot as plt
import calculation_func as hxc
import sum_length as ref

L_tot_ref = ref.L_tot_ref
PD_t_ref = ref.PD_t_ref
PD_a_ref = ref.PD_a_ref

PD_t_star_limit = 50000 / PD_t_ref
PD_a_star_limit = 50000 / PD_a_ref

def find_intersections(x_arr, y_arr, limit_val):
    intersections = []
    for i in range(len(x_arr) - 1):
        y1, y2 = y_arr[i], y_arr[i+1]
        if (y1 - limit_val) * (y2 - limit_val) <= 0 and y1 != y2:
            x1, x2 = x_arr[i], x_arr[i+1]
            x_int = x1 + (limit_val - y1) * (x2 - x1) / (y2 - y1)
            intersections.append(x_int)
    return intersections


d_eo_fixed = 37.89 / 1000  # m

d_vi_fixed, d_vo_fixed = hxc.hyd_dia(hxc.d_bi_ref, hxc.d_ei_ref, hxc.n_s_ref)
theta_fixed, e_fixed = hxc.parameters(d_vo_fixed, d_eo_fixed, hxc.d_bo_ref, hxc.n_s_ref, hxc.p_ref)
d_hyd_fixed = hxc.d_oi_ref - d_vo_fixed

L_tot_fixed, PD_t_fixed, PD_a_fixed, _, _, Tc_out_fixed, _, _, _ = hxc.run_simulation(
    ref.N_segments, ref.X_h, ref.X_c,
    hxc.d_bi_ref, hxc.d_ei_ref, d_eo_fixed, hxc.d_bo_ref,
    hxc.n_s_ref, hxc.p_ref, hxc.d_oi_ref
)

print("=" * 50)
print(f"[ Fixed Condition: D_eo = {d_eo_fixed*1000:.2f} mm ]")
print("-" * 50)
print("1. Calculated Geometry Parameters:")
print(f"   - Tube Equivalent Inner Dia (D_vi) : {d_vi_fixed*1000:.3f} mm")
print(f"   - Tube Equivalent Outer Dia (D_vo) : {d_vo_fixed*1000:.3f} mm")
print(f"   - Annular Hydraulic Dia (D_hyd)    : {d_hyd_fixed*1000:.3f} mm")
print(f"   - Helix Angle (theta)              : {theta_fixed:.3f} deg")
print(f"   - Rib Height (e)                   : {e_fixed*1000:.3f} mm")
print("-" * 50)
print("2. Simulation Results (Base State):")
print(f"   - Total Tube Length (L_tot)        : {L_tot_fixed:.4f} m  (L* = {L_tot_fixed/L_tot_ref:.4f})")
print(f"   - Cold Fluid Outlet Temp (Tc_out)  : {Tc_out_fixed:.2f} ℃")
print(f"   - Tube Side Pressure Drop (PD_t)   : {PD_t_fixed/1000:.3f} kPa (ΔP_t* = {PD_t_fixed/PD_t_ref:.4f})")
print(f"   - Annular Side Pressure Drop(PD_a) : {PD_a_fixed/1000:.3f} kPa (ΔP_a* = {PD_a_fixed/PD_a_ref:.4f})")
print("=" * 50)
print()

param_configs = {
    'd_bi': {'name': 'D_{bi}', 'ref': hxc.d_bi_ref, 'range': (0.8, 1.2)},
    'd_oi': {'name': 'D_{oi}', 'ref': hxc.d_oi_ref, 'range': (0.7, 1.2)},
    'p':    {'name': 'p',        'ref': hxc.p_ref,    'range': (0.7, 1.2)},
    'n_s':  {'name': 'N_s',      'ref': hxc.n_s_ref,  'range': (0.8, 1.2)}
}

N_points = 100


for param_key, cfg in param_configs.items():
    param_star_list = np.linspace(cfg['range'][0], cfg['range'][1], N_points)
    
    L_star_list = []
    PD_t_star_list = []
    PD_a_star_list = []
    
    for val_star in param_star_list:
        val_actual = val_star * cfg['ref']
        
        kwargs = {
            'N_segments': ref.N_segments, 'X_h': ref.X_h, 'X_c': ref.X_c,
            'd_bi': hxc.d_bi_ref, 'd_ei': hxc.d_ei_ref, 'd_eo': d_eo_fixed, 
            'd_bo': hxc.d_bo_ref, 'n_s_val': hxc.n_s_ref, 'p_val': hxc.p_ref, 
            'd_oi': hxc.d_oi_ref
        }
        
        if param_key == 'd_bi': kwargs['d_bi'] = val_actual
        elif param_key == 'd_oi': kwargs['d_oi'] = val_actual
        elif param_key == 'p':    kwargs['p_val'] = val_actual
        elif param_key == 'n_s':  kwargs['n_s_val'] = val_actual
        
        L_tot, PD_t, PD_a, _, _, _, _, _, _ = hxc.run_simulation(
            kwargs['N_segments'], kwargs['X_h'], kwargs['X_c'],
            kwargs['d_bi'], kwargs['d_ei'], kwargs['d_eo'], kwargs['d_bo'],
            kwargs['n_s_val'], kwargs['p_val'], kwargs['d_oi']
        )
        
        L_star_list.append(L_tot / L_tot_ref)
        PD_t_star_list.append(PD_t / PD_t_ref)
        PD_a_star_list.append(PD_a / PD_a_ref)

    param_star_arr = np.array(param_star_list)
    PD_t_star_arr = np.array(PD_t_star_list)
    PD_a_star_arr = np.array(PD_a_star_list)

    t_intersections = find_intersections(param_star_arr, PD_t_star_arr, PD_t_star_limit)
    a_intersections = find_intersections(param_star_arr, PD_a_star_arr, PD_a_star_limit)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:red'
    ax1.set_xlabel(f"Non-dimensional Outer Parameter (${cfg['name']}^*$)", fontsize=12)
    ax1.set_ylabel('Non-dimensional Length ($L^*$)', color=color1, fontsize=12)
    line1 = ax1.plot(param_star_list, L_star_list, label='$L^*$', color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    color3 = 'tab:green'
    ax2.set_ylabel('Non-dimensional Pressure Drop ($ΔP^*$)', fontsize=12)
    line2 = ax2.plot(param_star_list, PD_t_star_list, label='$ΔP_t^*$', color=color2, linewidth=2, linestyle='--')
    line3 = ax2.plot(param_star_list, PD_a_star_list, label='$ΔP_a^*$', color=color3, linewidth=2, linestyle='-.')

    line_lim_t = ax2.axhline(y=PD_t_star_limit, color='tab:blue', linestyle=':', linewidth=1.5, label='$ΔP_{t,limit}^*$ (50kPa)')
    line_lim_a = ax2.axhline(y=PD_a_star_limit, color='tab:green', linestyle=':', linewidth=1.5, label='$ΔP_{a,limit}^*$ (50kPa)')
    
    for x_int in t_intersections:
        ax2.plot(x_int, PD_t_star_limit, 'ro', markersize=6)
        ax2.annotate(f'({x_int:.3f}, {PD_t_star_limit:.3f})',
                     xy=(x_int, PD_t_star_limit),
                     xytext=(x_int + 0.015, PD_t_star_limit + 0.05),
                     fontsize=9, fontweight='bold', color=color2,
                     arrowprops=dict(arrowstyle='->', color=color2, lw=1))

    for idx, x_int in enumerate(a_intersections):
        ax2.plot(x_int, PD_a_star_limit, 'go', markersize=6)
        
        x_off = 0.04 if idx % 2 == 0 else -0.04
        y_off = -0.08 if idx % 2 == 0 else -0.14
        ha_align = 'left' if idx % 2 == 0 else 'right'
        
        ax2.annotate(f'({x_int:.3f}, {PD_a_star_limit:.3f})',
                     xy=(x_int, PD_a_star_limit),
                     xytext=(x_int + x_off, PD_a_star_limit + y_off),
                     fontsize=9, fontweight='bold', color=color3,
                     ha=ha_align, va='top',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color3),
                     arrowprops=dict(arrowstyle='->', color=color3, lw=1))
    lines = line1 + line2 + line3 + [line_lim_t, line_lim_a]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    plt.title(f"Parametric Analysis: Effects of ${cfg['name']}^*$ (Fixed $D_{{eo}} = 37.89\\text{{ mm}}$)", fontsize=13, fontweight='bold')
    plt.tight_layout()

    plt.savefig(f"plot_param_{param_key}_set_deo37.png", dpi=300)
    plt.show()