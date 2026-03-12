import logging
import math

from scipy import stats
import xarray as xr
import numpy as np
import scipy.interpolate as sp
from tqdm import tqdm
import pandas as pd

import reverie.correction.atmospheric.get_atmosphere as atm
from reverie.correction.surface.rayleigh import get_sky_glint

import lut

def compute_rho_dark(args, method = "percentile"):

    i, wl, rho_t, sol_zen, view_zen, sol_azi, view_azi, relative_azimuth, mask_water = args

    rho_t = rho_t.sel(wavelength=wl).values

    if method == "min":
        geometry = "resolved"
        i_dark_flat = np.nanargmin(rho_t)
        i_dark = np.unravel_index(i_dark_flat, rho_t.shape)
        rho_dark = rho_t.flatten()[i_dark]


    if method == "percentile":
        geometry = "resolved"
        percentile = 1
        rho_dark = np.nanpercentile(rho_t, percentile)
        i_dark = np.argwhere(rho_t == rho_dark)
        i_dark = tuple(i_dark[0]) if i_dark.size > 0 else None

    if method == "ols":
        geometry = "mean"
        rho_t_sort = np.sort(rho_t, axis=None)
        n_pixels = 1000
        slope, intercept, r, p, std_err = stats.linregress(
            list(range(n_pixels + 1)), rho_t_sort[0:n_pixels + 1]
        )
        rho_dark = intercept


    if geometry == "resolved":
        sol_zen_dark = sol_zen[i_dark] if i_dark is not None else np.nan
        view_zen_dark = view_zen[i_dark] if i_dark is not None else np.nan
        sol_azi_dark = sol_azi[i_dark] if i_dark is not None else np.nan
        view_azi_dark = view_azi[i_dark] if i_dark is not None else np.nan
        relative_azimuth_dark = relative_azimuth[i_dark] if i_dark is not None else np.nan
        mask_water_dark = mask_water[i_dark] if i_dark is not None else np.nan
    else:
        sol_zen_dark = np.nanmean(sol_zen)
        view_zen_dark = np.nanmean(view_zen)
        sol_azi_dark = np.nanmean(sol_azi)
        view_azi_dark = np.nanmean(view_azi)
        relative_azimuth_dark = np.nanmean(relative_azimuth)
        mask_water_dark = True

    return i, sol_zen_dark, view_zen_dark, sol_azi_dark, view_azi_dark, relative_azimuth_dark, rho_dark, mask_water_dark

def aot555_dsf(wavelength, rho_dark, sol_zen, view_zen, sol_azi, view_azi, relative_azimuth, target_pressure, sensor_altitude, water, ozone, mask_water_dark, dev = True, name = None):

    lut_aer = lut.load_aer()

    lut_points = (
        lut_aer.sol_zen.values,
        lut_aer.view_zen.values,
        lut_aer.relative_azimuth.values,
        lut_aer.aot550.values,
        lut_aer.target_pressure.values,
        lut_aer.sensor_altitude.values,
        lut_aer.wavelength.values,
    )

    rho_path_values = lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
    t_ra_values = lut_aer["total_scattering_trans_total"][:, :, :, :, :, :, :].values
    s_ra_values = lut_aer["spherical_albedo_total"][:, :, :, :, :, :, :].values

    aot550_values = lut_aer.aot550.values
    n_aot = len(aot550_values)

    rho_path_save = np.full((len(wavelength), n_aot), np.nan)
    rho_path_gas_cor = np.full((len(wavelength), n_aot), np.nan)
    sky_glint = np.full((len(wavelength), n_aot), np.nan)
    sky_glint_at_sensor = np.full((len(wavelength), n_aot), np.nan)
    rho_path_corrected = np.full((len(wavelength), n_aot), np.nan)
    aot550_candidate = np.full_like(wavelength, np.nan)
    for i, wl in tqdm(enumerate(wavelength)):
        # TODO: apply gas transmittance "correction" to rho_path
        # Innacurate because doesn't account for true interaction between scattering and absorption
        # 6S rho_path(Rayleigh + aerosol + gas) !=
        # (rho_path(rayleigh+aerosol) * t_gas) / (1 - rho_path(Rayleigh+aerosol) * Spherical albedo)
        # Viccarious calibration "remove" this innacuracy


        xi = np.hstack([
            np.full((n_aot, 1), sol_zen[i]),
            np.full((n_aot, 1), view_zen[i]),
            np.full((n_aot, 1), relative_azimuth[i]),
            aot550_values.reshape(-1, 1),
            np.full((n_aot, 1), target_pressure),
            np.full((n_aot, 1), sensor_altitude),
            np.full((n_aot, 1), wl),
        ])

        rho_path = sp.interpn(
            points=lut_points,
            # values=lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
            values=rho_path_values,
            xi=xi,
        )

        t_ra = sp.interpn(
            points=lut_points,
            # values=lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
            values=t_ra_values,
            xi=xi,
        )

        s_ra = sp.interpn(
            points=lut_points,
            # values=lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
            values=s_ra_values,
            xi=xi,
        )

        t_gas = lut.get_t_gas(wl, sol_zen[i], view_zen[i], relative_azimuth[i], water, ozone, target_pressure,
                              sensor_altitude)
        rho_path_save[i] = rho_path
        # rho_path_gas_cor[i] = (rho_path * t_gas) / (1 - rho_path * s_ra)

        if mask_water_dark[i]:
            theta_0 = np.deg2rad(sol_zen[i])
            theta_v = np.deg2rad(view_zen[i])
            phi_0 = np.deg2rad(sol_azi[i])
            phi_v = np.deg2rad(view_azi[i])

            sky_glint[i] = get_sky_glint(wl, theta_0, theta_v, phi_0, phi_v, 1007)

            sky_glint_at_sensor[i] = ((sky_glint[i] * t_ra * t_gas) / (1 - sky_glint[i] * s_ra))

            rho_path_corrected[i] = rho_path + sky_glint_at_sensor[i]
        else:
            rho_path_corrected[i] = rho_path

        f_interp = sp.interp1d(rho_path_corrected[i], aot550_values, bounds_error=False, fill_value=np.nan)
        aot550_candidate[i] = f_interp(rho_dark[i])

    if dev:
        data = []
        for i, wl in enumerate(wavelength):
            for aot_idx, aot_val in enumerate(lut_aer.aot550.values):
                data.append({
                    "image_name": name,
                    "wavelength": wl,
                    "rho_dark": rho_dark[i],
                    "sol_zen": sol_zen[i],
                    "view_zen": view_zen[i],
                    "sol_azi": sol_azi[i],
                    "view_azi": view_azi[i],
                    "relative_azimuth": relative_azimuth[i],
                    "target_pressure": target_pressure,
                    "sensor_altitude": sensor_altitude,
                    "water": water,
                    "ozone": ozone,
                    "mask_water_dark": mask_water_dark[i],
                    "aot550": aot_val,
                    "rho_path_save": rho_path_save[i, aot_idx],
                    # "rho_path_gas_cor": rho_path_gas_cor[i, aot_idx],
                    "sky_glint": sky_glint[i, aot_idx],
                    "sky_glint_at_sensor": sky_glint_at_sensor[i, aot_idx],
                    "rho_path_corrected": rho_path_corrected[i, aot_idx],
                    "aot550_candidate": aot550_candidate[i]
                })

        df = pd.DataFrame(data)
        df.to_csv("/D/Documents/phd/thesis/3_chapter/data/wise/dsf/aot555_dsf_analysis"+name+".csv", index=False)


    ### DEV

    # import matplotlib.pyplot as plt
    #
    # aot_index = 0
    # plt.plot(wavelength, rho_dark)
    # # plt.plot(wavelength, rho_path_corrected[:, aot_index])
    # # plt.plot(wavelength, sky_glint_at_sensor[:, aot_index])
    # # plt.plot(wavelength, sky_glint[:, aot_index])
    # plt.plot(wavelength, rho_path_save[:, aot_index])
    # plt.show()
    #
    # plt.plot(aot550_candidate)
    # plt.show()

    ### DEV

    # percentile = 10  # Change to desired percentile
    # aot550_retrieved = np.nanpercentile(aot550_candidate, percentile)

    aot550_retrieved = np.nanmedian(aot550_candidate)

    # aot550_retrieved = np.nanmin(aot550_candidate)

    return aot550_retrieved