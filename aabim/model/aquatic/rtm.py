import numpy as np

def iop_from_oac_spm_np(wl, a_w, a0, a1, bb_w,
                        chl, a_g_440, spm,
                        a_g_s, a_nap_star, a_nap_s,
                        bb_p_star, bb_p_gamma):
    """
    Returns a, bb (both shape [nwl]) following iop_from_oac_spm_core.
    """
    wl = wl.astype(float)
    a_w = a_w.astype(float); a0 = a0.astype(float); a1 = a1.astype(float); bb_w = bb_w.astype(float)

    aph_440 = 0.06 * (chl ** 0.65)

    # a_phy
    a_phy = (a0 + a1 * np.log(aph_440)) * aph_440
    a_phy = np.maximum(a_phy, 0.0)

    # a_g
    a_g = a_g_440 * np.exp(-a_g_s * (wl - 440.0))

    # nap and particles from SPM
    a_nap_440 = a_nap_star * spm
    bb_p_550  = bb_p_star  * spm

    a_nap = a_nap_440 * np.exp(-a_nap_s * (wl - 440.0))
    bb_p  = bb_p_550  * (wl / 550.0) ** (-bb_p_gamma)

    a  = a_w  + a_phy + a_g + a_nap
    bb = bb_w + bb_p
    return a, bb


def solve_rb_am03_np(wl, a, bb, water_type,
                     theta_sun_deg, theta_view_deg,
                     h_w, rrs_obs):
    """
    Returns r_b (shape [nwl]) following solve_rb_am03_core.
    """
    deg2rad = np.pi / 180.0
    n_air, n_water = 1.0, 1.34

    tv = theta_view_deg * deg2rad
    ts = theta_sun_deg  * deg2rad

    sin_tv_w = (n_air / n_water) * np.sin(tv)
    sin_ts_w = (n_air / n_water) * np.sin(ts)

    view_w = np.arcsin(np.clip(sin_tv_w, -1.0, 1.0))
    sun_w  = np.arcsin(np.clip(sin_ts_w, -1.0, 1.0))

    cos_sun  = np.cos(sun_w)
    cos_view = np.cos(view_w)
    if cos_sun <= 0 or cos_view <= 0:
        return np.full_like(a, 0.0, dtype=float)

    ext = a + bb
    rb = np.zeros_like(ext, dtype=float)

    ok = np.isfinite(ext) & (ext > 0) & np.isfinite(rrs_obs)
    if not np.any(ok):
        return rb

    omega_b = np.zeros_like(ext, dtype=float)
    omega_b[ok] = bb[ok] / ext[ok]

    if water_type == 1:
        f_rs = np.full_like(ext, 0.095, dtype=float)
        k0 = 1.0395
    elif water_type == 2:
        f_rs = 0.0512 * (
            1.0 + 4.6659*omega_b - 7.8387*omega_b**2 + 5.4571*omega_b**3
        ) * (1.0 + 0.1098/cos_sun) * (1.0 + 0.4021/cos_view)
        k0 = 1.0546
    else:
        raise ValueError("water_type must be 1 or 2")

    rrs_deep = f_rs * omega_b

    Kd  = k0 * (ext / cos_sun)
    kuW = (ext / cos_view) * (1.0 + omega_b) ** 3.5421 * (1.0 - 0.2786 / cos_sun)
    kuB = (ext / cos_view) * (1.0 + omega_b) ** 2.2658 * (1.0 - 0.0577 / cos_sun)

    Ars1, Ars2 = 1.1576, 1.0389

    termW = np.exp(-h_w * (Kd + kuW))
    termB = np.exp(-h_w * (Kd + kuB))

    rrs_wc = rrs_deep * (1.0 - Ars1 * termW)
    delta  = rrs_obs - rrs_wc

    tiny = 1e-30
    goodB = ok & np.isfinite(termB) & (termB > tiny)

    rb[goodB] = (np.pi / Ars2) * (delta[goodB] / termB[goodB])

    rb[~np.isfinite(rb)] = 0.0
    return rb
