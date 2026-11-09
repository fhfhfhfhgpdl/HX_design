import numpy as np
import math

# reference values, [m]
tw = 0.5 / 1000
d_bi_ref = 35 / 1000
d_bo_ref = 36 / 1000
d_ei_ref = 44 / 1000
d_eo_ref = 45 / 1000
d_oi_ref = 45 / 1000
d_can = 450 / 1000
p_ref = 25 / 1000
k_w = 401 #W/m*K
n_s_ref = 4  # Number of starts

# kg/s
m_h = 3.5
m_c = 3.0

def calc_cp_libr(T, X):
    # J/kg*K
    x = X / 100.0
    term1 = -0.26642 + 10.97101 * x - 12.7661 * (x**2)
    term2 = 0.02502 - 0.08334 * x + 0.07541 * (x**2)
    return (term1 + term2 * T) * 1000

def k_libr(t, x):
    # W/m*K 
    T = t + 273.15
    a1 = -1407.53 + 11.0513*T - 1.46741*(10**-2)*(T**2)   
    a2 = 38.9855 - 0.240475*T + 3.48073*(10**-4)*(T**2)
    a3 = -0.265025 + 1.51915*(10**-3)*T - 2.32262*(10**-6)*(T**2)
    return (a1 + a2*x + a3*(x**2)) * 0.001

def viscosity_libr(t, x):
    # kg/m*s 
    T = t + 273.15
    a1 = -494.122 + 16.3967*x - 0.14511*(x**2)
    a2 = 28606.4 - 934.568*x + 8.52755*(x**2)
    a3 = 70.3848 - 2.35014*x + 0.0207809*(x**2)
    return (math.exp(a1 + a2/T + a3*math.log(T))) * 0.001

def density_libr(t, x):
    # kg/m^3 
    T = t + 273.15
    a1 = (10976.3 + 0.71244*x + 2.21446*(x**2)) / 10
    a2 = (6796.2 - 148.247*x - 0.89696*(x**2)) / (10**4)
    a3 = (-350.97 - 324.312*x + 4.97020*(x**2)) / (10**7)
    return (a1 + a2*T + a3*(T**2))

def hyd_dia(d_bi, d_ei, n_s_val):
    R = 0.6
    FB = (1-R) * math.pi * d_bi / n_s_val
    d_vi = math.sqrt(d_bi**2 + n_s_val*(d_ei - d_bi)*FB/math.pi)
    d_vo = d_vi + 2*tw
    return d_vi, d_vo

def parameters(d_vo, d_eo, d_bo, n_s_val, p):
    theta = math.atan(math.pi * d_vo / (n_s_val * p)) * 180 / math.pi
    e = (d_eo - d_bo) / 2
    return theta, e

def nd_tube(c_p, rho, mu, k, e, p, theta, d_vi):
    Pr_t = c_p * mu / k
    e_star_t = e / d_vi
    p_star_t = p / d_vi
    theta_star_t = theta / 90.0
    V_t = 4 * m_h / (math.pi * rho * (d_vi**2))
    Re_t = V_t * rho * d_vi / mu

    if 500 <= Re_t <= 5000:
        Nu_t = 0.014 * (Re_t**0.842) * (e_star_t**-0.067) * (p_star_t**-0.293) * (theta_star_t**-0.705) * (Pr_t**0.4)
    else:
        Nu_t = 0.064 * (Re_t**0.773) * (e_star_t**-0.242) * (p_star_t**-0.108) * (theta_star_t**-0.599) * (Pr_t**0.4)

    h_t = Nu_t * k / d_vi
    return h_t

def nd_annular(c_p, rho, mu, k, e, p, theta, d_vo, d_oi):
    Pr_a = c_p * mu / k
    e_star_a = e / d_vo
    p_star_a = p / d_vo
    theta_star_a = theta / 90.0
    r_star = d_vo / d_oi
    d_hyd = d_oi - d_vo
    A_eff = math.pi / 4 * (d_oi**2 - d_vo**2)
    V_a = m_c / rho / A_eff
    Re_a = V_a * rho * d_hyd / mu

    if Re_a <= 800:
        f_a = (96 * (r_star**0.035) / Re_a) * (1 + 101.7 * (Re_a**0.52) * (e_star_a**(1.65 + 2*theta_star_a)) * (r_star**5.77))
    else:
        e_f = 1 + 222 * (Re_a**0.09) * (e_star_a**2.4) * (p_star_a**-0.49) * (theta_star_a**-0.38) * (r_star**2.22)
        f_a = 4 * ((1.7372 * math.log(Re_a / (1.964 * math.log(Re_a) - 3.8215)))**-2) * (1 + 0.0925 * r_star) * e_f

    Nu_a = ( (f_a/8) * Re_a * Pr_a / (1 + 9.77 * math.sqrt(f_a/8) * (Pr_a**(2/3) - 1)) ) * (Re_a**-0.2) * (e_star_a**-0.32) * (p_star_a**-0.28) * (r_star**-1.64)

    h_a = Nu_a * k / d_hyd
    return h_a