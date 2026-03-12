import numpy as np
import pandas as pd

from reverie.correction.surface.rayleigh import get_sky_glint, ray_tau_ht74, ray_tau_b99, ray_phase
from reverie.correction.surface.water_refractive_index import get_water_refractive_index
from reverie.correction.surface import z17
#/D/Documents/phd/thesis/3_chapter/data/wise/sky_glint/

if __name__ == "__main__":
    wl = np.arange(350, 1000, 10)

    # theta_s = np.deg2rad(np.arange(10, 81, 10))
    # theta_v = np.deg2rad(np.arange(1, 31, 10))
    # phi_s = np.deg2rad(np.arange(1, 360, 10))
    # phi_v = np.deg2rad(np.arange(1, 360, 10))

    theta_0 = np.deg2rad(32)
    theta_v = np.deg2rad(11)
    phi_0 = np.deg2rad(180)
    phi_v = np.deg2rad(270)

    raa = abs(phi_0 - phi_v)

    ### Quentin 2018 sky glint

    W, T0, TV, P0, PV = np.meshgrid(wl, theta_0, theta_v, phi_0, phi_v, indexing='ij')

    # Flatten grids for vectorized computation
    W_flat = W.ravel()
    T0_flat = T0.ravel()
    TV_flat = TV.ravel()
    P0_flat = P0.ravel()
    PV_flat = PV.ravel()

    tau_r_h74 = ray_tau_ht74(wl, Patm=1007)
    tau_r_b99 = ray_tau_b99(wl, p_mb=1007)
    n_w = get_water_refractive_index(30, 12, wl)
    phase_r = ray_phase(theta_0, theta_v, phi_0, phi_v)
    sky_glint_q18 = get_sky_glint(W_flat, T0_flat, TV_flat, P0_flat, PV_flat, 1007)

    df_q18 = pd.DataFrame({
        # 'model': "quen18",
        'wavelength': W_flat,
        'tau_r_h74': tau_r_h74,
        'tau_r_b99': tau_r_b99,
        'phase_r': phase_r,
        'theta_s': np.rad2deg(T0_flat),
        'theta_v': np.rad2deg(TV_flat),
        'phi_s': np.rad2deg(P0_flat),
        'phi_v': np.rad2deg(PV_flat),
        'sky_glint_q18': sky_glint_q18
    })
    df_q18.to_csv('/D/Documents/phd/thesis/3_chapter/data/wise/sky_glint/sky_glint_quen18.csv', index=False)

    ### ZHANG glint

    sky_glint_z17 = z17.get_sky_sun_rho(
        aot_550=0.055,
        sun_zen=theta_0,
        view_ang=[theta_v,
                  raa],
        water_salinity=30,
        water_temperature=17,
        wavelength=wl,
        wind_speed=0)

    df_z17 = pd.DataFrame({
        # 'model': "z17",
        'wavelength': W_flat,
        'theta_s': np.rad2deg(T0_flat),
        'theta_v': np.rad2deg(TV_flat),
        'phi_s': np.rad2deg(P0_flat),
        'phi_v': np.rad2deg(PV_flat),
        'sky_glint_z17_rho': sky_glint_z17["rho"],
        'sky_glint_z17_sky': sky_glint_z17["sky"],
        'sky_glint_z17_sun': sky_glint_z17["sun"],
    })
    df_z17.to_csv('/D/Documents/phd/thesis/3_chapter/data/wise/sky_glint/sky_glint_z17.csv', index=False)
