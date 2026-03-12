from math import cos, sin, pow, exp
import numpy as np

from reverie.correction.surface.water_refractive_index import get_water_refractive_index

# function ray_tau
# computes Rayleigh optical thickness for wl in microns
# Hansen and Travis 1974
# QV 2016-12-14

def ray_tau_ht74(wl, Patm=1013.25):
    # Wavelength in um
    wl = wl * 1e-3
    tau_ray = Patm/1013.25*(0.008569*np.power(wl,-4)*(1.+0.0113*np.power(wl,-2)+0.00013*np.power(wl,-4)))
    return tau_ray

def ray_tau_b99(wl_nm, p_mb=1013.25, co2_ppm=400., lat_deg=45., z_m=0):
    """
    Rayleigh optical depth following Bodhaine et al. (1999)
    Wavelength-dependent depolarization and refractivity included.
    Parameters
    ----------
    wl_nm : scalar or array
        Wavelength in nanometers
    p_mb : float
        Pressure in millibars
    co2_ppm : float
        CO₂ concentration (ppm)
    lat_deg : float
        Latitude in degrees
    z_m : float
        Altitude in meters
    """

    wl_nm  = np.asarray(wl_nm, dtype=float)
    wl_um  = wl_nm * 1e-3         # micrometers
    wl_cm  = wl_nm * 1e-7         # cm

    # -----------------------------
    # 1. Refractive index n(λ)
    #    Peck & Reeder (1972) used in Bodhaine
    # -----------------------------
    n_300_minus1 = 1e-8 * (
        8060.51 +
        2480990.0 / (132.274 - 1.0/wl_um**2) +
        17455.7  / (39.32957 - 1.0/wl_um**2)
    )

    # CO₂ correction (Bodhaine)
    dn_co2 = (co2_ppm - 300.0) * 1.15e-8

    n_air = 1.0 + n_300_minus1 + dn_co2

    # -----------------------------
    # 2. Depolarization (King factor)
    #    Wavelength-dependent N₂ and O₂ terms from Bodhaine
    # -----------------------------
    F_N2 = 1.034 + 3.17e-4 / wl_um**2
    F_O2 = 1.096 + 1.385e-3 / wl_um**2 + 1.448e-4 / wl_um**4

    # volume fractions (dry air)
    f_N2  = 0.78084
    f_O2  = 0.20946
    f_Ar  = 0.00934
    f_CO2 = co2_ppm * 1e-6

    # Depolarization ratio of the mixture
    rho = (
        f_N2  * (F_N2 - 1) +
        f_O2  * (F_O2 - 1) +
        f_Ar  * (1.00 - 1) +
        f_CO2 * (1.15 - 1)
    )

    # King factor K(λ) = (6 + 3ρ) / (6 - 7ρ)
    K = (6.0 + 3.0*rho) / (6.0 - 7.0*rho)

    # -----------------------------
    # 3. Rayleigh scattering cross-section
    # -----------------------------
    Ns = 2.546899e19  # molecule cm⁻³ at STP (used in B99)

    term = ((n_air**2 - 1) / (n_air**2 + 2))**2

    sigma = (24.0 * np.pi**3 / (Ns**2 * wl_cm**4)) * term * K

    # -----------------------------
    # 4. Optical depth τ = σ * column density
    # -----------------------------
    p_dyn = p_mb * 1000.0  # mb → dyn cm⁻²

    # Mean molecular mass (g/mol)
    M = 28.9595 + 15.0556 * f_CO2

    NA = 6.0221367e23

    lat = np.deg2rad(lat_deg)

    # Gravity (cm/s²) from Bodhaine eqn.
    g0 = 980.6160 * (1 - 0.0026373*np.cos(2*lat) + 5.9e-6*np.cos(2*lat)**2)
    # Altitude corrections usually negligible, skipped for clarity
    g = g0

    tau = sigma * (p_dyn * NA) / (M * g)

    return tau


def fresnel_reflectance(theta, n_w):
    n_air = 1

    if np.allclose(theta, 0):
        r = ((n_air - n_w) / (n_air + n_w)) ** 2
        return r

    theta_t = np.arcsin(n_air / n_w * np.sin(theta))
    r = 0.5 * ((np.power(np.sin(theta - theta_t), 2) / np.power(np.sin(theta + theta_t), 2)) +
                (np.power(np.tan(theta - theta_t), 2) / np.power(np.tan(theta + theta_t), 2)))
    return r

# function ray_phase
# computes Rayleigh phase function for given geometry
# QV 2016-12-14
# MR 2025-11-21 https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/rayleigh_phase.htm

def ray_phase(theta_s, theta_v, phi_s, phi_v):
    delta_phi = abs(phi_s - phi_v)

    cos_theta_plus = np.cos(theta_s) * np.cos(theta_v) - np.sin(theta_s) * np.sin(theta_v) * np.cos(delta_phi)
    # theta_plus = np.arccos(cos_theta_plus)
    phase_r = (0.75 * (1. + np.power(cos_theta_plus, 2.)))

    return phase_r


# function ray_tr
# computes Rayleigh transmittance for given geometry
# QV 2016-12-14

def ray_tr(wl, theta_s, theta_v, Patm=1013.25):
    tau_ray = ray_tau_ht74(wl, Patm=Patm)
    ray_tr = (1.+exp(-1.*tau_ray/cos(theta_v))) * (1.+exp(-1.*tau_ray/cos(theta_s))) / 4.
    return ray_tr


# function sky_refl
# computes diffuse sky reflectance
# QV 2016-12-14

def sky_refl(theta, n_w=1.34):
    # angle of transmittance theta_t for air incident rays (Mobley, 1994 p156)
    theta_t = np.arcsin( 1. / n_w * np.sin(theta))
    r_int=0.5 * (np.power(np.sin(theta - theta_t) / np.sin(theta + theta_t ), 2) +
              np.power(np.tan(theta - theta_t) / np.tan(theta + theta_t), 2))
    return r_int

# function ray_refl
# computes Rayleigh reflectance for given geometry
# QV 2016-12-14
# MR 2025-11-21

def get_sky_glint(wl, theta_s, theta_v, phi_s, phi_v, Patm):

    wl = np.asarray(wl)
    theta_s = np.asarray(theta_s)
    theta_v = np.asarray(theta_v)
    phi_s = np.asarray(phi_s)
    phi_v = np.asarray(phi_v)

    # theta_s, theta_v, phi_s, phi_v = np.broadcast_arrays(
    #     np.asarray(theta_s), np.asarray(theta_v),
    #     np.asarray(phi_s), np.asarray(phi_v)
    # )

    if (wl.ndim == 1) & (theta_s.ndim == 2):
        wl_ = wl[:, None, None]  # shape (len_wl, 1, 1)
        theta_s_ = theta_s[None, :, :]  # shape (1, y, x)
        theta_v_ = theta_v[None, :, :]
        phi_s_ = phi_s[None, :, :]
        phi_v_ = phi_v[None, :, :]
    else:
        wl_ = wl
        theta_s_ = theta_s
        theta_v_ = theta_v
        phi_s_ = phi_s
        phi_v_ = phi_v

    n_w_ = get_water_refractive_index(
        salinity = 30,
        temperature = 12,
        wavelength = wl_
    )

    tau_ray = ray_tau_ht74(wl_, Patm=Patm)
    phase_r = ray_phase(theta_s_, theta_v_, phi_s_, phi_v_)

    fr_theta_s = fresnel_reflectance(theta_s, n_w_)
    fr_theta_v = fresnel_reflectance(theta_v, n_w_)

    phase_ray_fr = (fr_theta_s + fr_theta_v) * phase_r

    sky_glint = (tau_ray * phase_ray_fr) / (4. * np.cos(theta_s) * np.cos(theta_v))

    return sky_glint
