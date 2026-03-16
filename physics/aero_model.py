"""
physics/aero_model.py
---------------------
Parametric aero model for WEC / ELMS / GTWC car classes.

Physics:
  q   = ½ρv²                          dynamic pressure
  F_L = CL(h_f, h_r, ψ) · q · A_ref  downforce
  F_D = CD(h_f, h_r, ψ) · q · A_ref  drag
  CL  = CL_base · GH(h_f, h_r) · YD(ψ)
  CD  = (CD_base + k_i·CL²) · GH_drag · YD_drag(ψ)

Ground effect:  Gaussian centred on optimal ride heights
Yaw:            Quadratic penalty
"""

import numpy as np

AIR_DENSITY = 1.225   # kg/m³ ISA sea level

CAR_SPECS = {
    "LMH": {
        "name": "LMH Hypercar",
        "series": "WEC",
        "CL_base": 2.55,
        "CD_base": 0.78,
        "k_induced": 0.025,
        "A_ref": 1.92,
        "mass_kg": 1030,
        "rh_front_default": 35,
        "rh_rear_default": 75,
        "rh_front_range": (18, 80),
        "rh_rear_range": (38, 120),
        "rh_optimal_front": 32,
        "rh_optimal_rear": 65,
        "gh_sigma_front": 13,
        "gh_sigma_rear": 17,
        "gh_base": 0.66,
        "yaw_df_k": 0.0065,
        "yaw_drag_k": 0.014,
        "aero_balance": 0.465,
        "speed_default": 240,
        "speed_range": (80, 330),
        "accent": "#ff4400",
        "variant": "lmh",
    },
    "LMDh": {
        "name": "LMDh Hypercar",
        "series": "WEC / IMSA",
        "CL_base": 2.38,
        "CD_base": 0.74,
        "k_induced": 0.024,
        "A_ref": 1.88,
        "mass_kg": 1030,
        "rh_front_default": 38,
        "rh_rear_default": 78,
        "rh_front_range": (20, 85),
        "rh_rear_range": (42, 125),
        "rh_optimal_front": 35,
        "rh_optimal_rear": 70,
        "gh_sigma_front": 13,
        "gh_sigma_rear": 17,
        "gh_base": 0.65,
        "yaw_df_k": 0.007,
        "yaw_drag_k": 0.013,
        "aero_balance": 0.455,
        "speed_default": 230,
        "speed_range": (80, 320),
        "accent": "#00ccff",
        "variant": "lmdh",
    },
    "LMP2": {
        "name": "ORECA 07 LMP2",
        "series": "ELMS / WEC",
        "CL_base": 2.10,
        "CD_base": 0.68,
        "k_induced": 0.022,
        "A_ref": 1.75,
        "mass_kg": 930,
        "rh_front_default": 40,
        "rh_rear_default": 82,
        "rh_front_range": (24, 90),
        "rh_rear_range": (48, 130),
        "rh_optimal_front": 38,
        "rh_optimal_rear": 78,
        "gh_sigma_front": 12,
        "gh_sigma_rear": 16,
        "gh_base": 0.64,
        "yaw_df_k": 0.0075,
        "yaw_drag_k": 0.012,
        "aero_balance": 0.445,
        "speed_default": 220,
        "speed_range": (80, 300),
        "accent": "#4488ff",
        "variant": "lmp2",
    },
    "LMP3": {
        "name": "Ligier JS P320 LMP3",
        "series": "ELMS",
        "CL_base": 1.55,
        "CD_base": 0.56,
        "k_induced": 0.020,
        "A_ref": 1.62,
        "mass_kg": 930,
        "rh_front_default": 46,
        "rh_rear_default": 90,
        "rh_front_range": (30, 96),
        "rh_rear_range": (54, 136),
        "rh_optimal_front": 46,
        "rh_optimal_rear": 88,
        "gh_sigma_front": 12,
        "gh_sigma_rear": 15,
        "gh_base": 0.62,
        "yaw_df_k": 0.0085,
        "yaw_drag_k": 0.012,
        "aero_balance": 0.438,
        "speed_default": 205,
        "speed_range": (80, 280),
        "accent": "#44dd66",
        "variant": "lmp3",
    },
    "GT3": {
        "name": "GT3 (GTWC)",
        "series": "GTWC Europe",
        "CL_base": 1.02,
        "CD_base": 0.46,
        "k_induced": 0.018,
        "A_ref": 2.12,
        "mass_kg": 1300,
        "rh_front_default": 56,
        "rh_rear_default": 76,
        "rh_front_range": (38, 102),
        "rh_rear_range": (52, 120),
        "rh_optimal_front": 58,
        "rh_optimal_rear": 74,
        "gh_sigma_front": 16,
        "gh_sigma_rear": 18,
        "gh_base": 0.75,
        "yaw_df_k": 0.0095,
        "yaw_drag_k": 0.010,
        "aero_balance": 0.420,
        "speed_default": 195,
        "speed_range": (80, 270),
        "accent": "#ffcc00",
        "variant": "gt3",
    },
}


def _dynamic_pressure(speed_kmh: float) -> float:
    v = speed_kmh / 3.6
    return 0.5 * AIR_DENSITY * v * v


def ground_effect_factor(front_rh, rear_rh, spec: dict):
    opt_f = spec["rh_optimal_front"]
    opt_r = spec["rh_optimal_rear"]
    s_f   = spec["gh_sigma_front"]
    s_r   = spec["gh_sigma_rear"]
    base  = spec["gh_base"]
    f_fac = np.exp(-0.5 * ((front_rh - opt_f) / s_f) ** 2)
    r_fac = np.exp(-0.5 * ((rear_rh  - opt_r) / s_r) ** 2)
    return base + (1.0 - base) * f_fac * r_fac


def _yaw_factors(yaw, spec):
    return (1.0 - spec["yaw_df_k"]   * yaw**2,
            1.0 + spec["yaw_drag_k"] * yaw**2)


def compute_downforce(car_class, front_rh, rear_rh, yaw, speed_kmh):
    spec  = CAR_SPECS[car_class]
    q     = _dynamic_pressure(speed_kmh)
    gh    = ground_effect_factor(front_rh, rear_rh, spec)
    df_y, _ = _yaw_factors(yaw, spec)
    return float(spec["CL_base"] * gh * df_y * q * spec["A_ref"])


def compute_drag(car_class, front_rh, rear_rh, yaw, speed_kmh):
    spec    = CAR_SPECS[car_class]
    q       = _dynamic_pressure(speed_kmh)
    gh      = ground_effect_factor(front_rh, rear_rh, spec)
    _, dr_y = _yaw_factors(yaw, spec)
    CL_eff  = spec["CL_base"] * gh
    CD_tot  = spec["CD_base"] + spec["k_induced"] * CL_eff**2
    return float(CD_tot * dr_y * q * spec["A_ref"])


def compute_surfaces(car_class, front_rh, speed_kmh, n=28):
    spec         = CAR_SPECS[car_class]
    rh_lo, rh_hi = spec["rh_rear_range"]
    yaw_arr = np.linspace(-10, 10, n)
    rh_arr  = np.linspace(rh_lo, rh_hi, n)
    YAW, RH = np.meshgrid(yaw_arr, rh_arr)

    q      = _dynamic_pressure(speed_kmh)
    gh     = ground_effect_factor(front_rh, RH, spec)
    df_y   = 1.0 - spec["yaw_df_k"]   * YAW**2
    drag_y = 1.0 + spec["yaw_drag_k"] * YAW**2

    df_grid   = spec["CL_base"] * gh * df_y * q * spec["A_ref"]
    CL_eff    = spec["CL_base"] * gh
    CD_grid   = spec["CD_base"] + spec["k_induced"] * CL_eff**2
    drag_grid = CD_grid * drag_y * q * spec["A_ref"]

    return yaw_arr, rh_arr, df_grid, drag_grid


def get_readouts(car_class, front_rh, rear_rh, yaw, speed_kmh):
    spec  = CAR_SPECS[car_class]
    df    = compute_downforce(car_class, front_rh, rear_rh, yaw, speed_kmh)
    drag  = compute_drag(car_class, front_rh, rear_rh, yaw, speed_kmh)
    ld    = df / max(drag, 1.0)
    q     = _dynamic_pressure(speed_kmh)
    gh    = ground_effect_factor(front_rh, rear_rh, spec)

    rake       = rear_rh - front_rh
    rake_ref   = spec["rh_optimal_rear"] - spec["rh_optimal_front"]
    bal_shift  = (rake - rake_ref) * 0.0008
    aero_bal   = float(np.clip(spec["aero_balance"] + bal_shift, 0.35, 0.56))

    return {
        "downforce_N":      df,
        "drag_N":           drag,
        "ld_ratio":         ld,
        "dynamic_q":        q,
        "aero_balance_pct": aero_bal * 100,
        "rake_mm":          rake,
        "df_front_N":       df * aero_bal,
        "df_rear_N":        df * (1.0 - aero_bal),
        "gh_factor":        float(gh),
        "speed_ms":         speed_kmh / 3.6,
    }