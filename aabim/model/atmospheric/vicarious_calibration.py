import os

import pandas
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

from reverie import ReveCube
import reverie.correction.atmospheric.get_atmosphere as atm
from reverie.correction.atmospheric import lut
from reverie.correction.surface.rayleigh import get_sky_glint


import logging

def run_vicarious_cal(l1: ReveCube, in_situ: pandas.DataFrame, window_size, land: bool):
    """
    Function to compute vicarious calibration gain from a ReveCube L1 image and in situ Rrs.

    Parameters
    ----------
    image
    in_situ
    window_size

    Returns
    -------

    Description
    -----------
    The gains are computed on apparent reflectance (rho_t^\star = Lt / F0).
    Path reflectance from 6S used in the calculation is also apparent reflectance.
    Therfore the underestimation of reflectance due to the overestimation of irradiance at the plane level cancels out.

    """

    window_dict = l1.extract_pixel(
        insitu_path,
        max_time_diff=12,
        window_size=window_size,
        output_box = None #"/D/Documents/phd/thesis/3_chapter/data/wise/viccal/"
    )[1]

    if window_dict is None:
        raise Exception("No matchup found with in situ data")
    else:
        logging.info(f"Extracted window {list(window_dict.keys())}")

    aero_interps, aero_points = atm.build_aer_interpolators(lut.load_aer())
    aero_interps = list(aero_interps.values())

    gas_interp, gas_points = atm.build_gas_interpolator(lut.load_gas())

    aod555 = l1.in_ds.aerosol_optical_thickness_at_555_nm.values
    p_mb = l1.in_ds.surface_air_pressure.values

    l1.mask_wavelength(np.unique(in_situ["wavelength"]))
    l1.mask_wavelength(lut.load_aer().wavelength.values)

    wavelength = l1.wavelength

    viccal_df = pd.DataFrame()

    for uuid, window in tqdm(window_dict.items()):

        y_slice = slice(int(window.row_off), int(window.row_off + window.height))
        x_slice = slice(int(window.col_off), int(window.col_off + window.width))

        image_sub = l1.in_ds.isel(x=x_slice, y=y_slice)
        rho_t = image_sub["rho_at_sensor"].values

        ra_components = atm.get_ra(l1, window, wavelength, aod555, *aero_interps)
        gas_component = atm.get_gas(l1, window, wavelength, gas_interp)

        rho_path_ra = ra_components["rho_path"].values
        t_ra = ra_components["trans_ra"].values
        s_ra = ra_components["spherical_albedo_ra"].values
        t_g = gas_component["t_gas"].values

        # theta_0 = np.deg2rad(image_sub["sun_zenith"])
        # theta_v = np.deg2rad(image_sub["view_zenith"])
        # phi_0 = np.deg2rad(image_sub["sun_azimuth"])
        # phi_v = np.deg2rad(image_sub["view_azimuth"])

        # sky_glint = get_sky_glint(l1.wavelength, theta_0, theta_v, phi_0, phi_v, p_mb)
        sky_glint = ra_components["sky_glint_ra"].values

        if l1.valid_mask is None:
            l1.get_valid_mask(window=window)

        x = image_sub.x.values
        y = image_sub.y.values

        rho_w = (in_situ[in_situ["uuid"] == uuid]["rho"]).values
        in_situ_wl = in_situ[in_situ["uuid"] == uuid]["wavelength"].values

        rho_w = np.interp(x=wavelength, xp=in_situ_wl, fp=rho_w)

        # Add two new dims to temp
        rho_w_3d = rho_w[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dims
        rho_w_3d = np.repeat(rho_w_3d, len(y), axis=1)
        rho_w = np.repeat(rho_w_3d, len(x), axis=2)

        # rho_path_gas_cor = (rho_path_ra * t_g) / (1 - rho_path_ra * s_ra)

        rho_s = rho_w + sky_glint if not land else rho_w

        # rho_t_target = rho_path_ra + (rho_s * t_ra * t_g / (1 - rho_s * s_ra))

        # From atbd mod08
        # rho_t_target = t_g_o3_02_co2 * (rho_path_r + (rho_path_ra - rho_path_r) * t_g_h2o +
        #                                 t_tot * (rho_s / (1 - s * rho_s)) * t_g_h2o)

        rho_t_target = t_g * (rho_path_ra + t_ra * (rho_s / (1- s_ra * rho_s)))

        # From acolite
        # rho_s = (rho_t / t_g - rho_path_ra) / (t_ra + s_ra * (rho_t / t_g - rho_path_ra))
        # rho_t_target = t_g * ((rho_path_ra * (rho_s * s_ra - 1) - rho_s * t_ra) / (rho_s * s_ra - 1))

        gain = rho_t_target / rho_t

        ### DEV

        # import matplotlib.pyplot as plt
        #
        # image_sub = l1.in_ds.isel(x=slice(window.row_off+400, window.row_off+407), y=slice(window.col_off+400, window.col_off+407))
        # rho_t_test = image_sub["rho_at_sensor"].values * gain
        #
        # # rho_w_retrieved = (rho_t * gain - rho_path_ra) / (t_ra * t_g + s_ra * (rho_t * gain - rho_path_ra)) - sky_glint
        # rho_w_retrieved = ((rho_t_test / t_g) - rho_path_ra) / (t_ra + s_ra * ((rho_t_test / t_g) - rho_path_ra)) - sky_glint
        #
        # # Plot retrieved water reflectance for the center pixel (y=1, x=1)
        #
        # spectrum = rho_w_retrieved[:, 1, 1]
        # plt.figure()
        # plt.plot(wavelength, spectrum, marker="o")
        # plt.plot(wavelength, rho_w[:, 1, 1], marker="x")
        # plt.xlabel("wavelength (nm)")
        # plt.ylabel("rho_w")
        # plt.show()
        #
        # plt.figure()
        # plt.plot(wavelength, gain[:, 1, 1], marker="o")
        # plt.xlabel("wavelength (nm)")
        # plt.ylabel("gain")
        # plt.show()

        ###

        ds_out = xr.Dataset(
            {
                "rho_w": xr.DataArray(
                    rho_w,
                    dims=["wavelength", "y", "x"],
                    coords={
                        "wavelength": wavelength,
                        "y": y,
                        "x": x,
                    },
                ),
                "sky_glint": xr.DataArray(
                    sky_glint,
                    dims=["wavelength", "y", "x"],
                    coords={
                        "wavelength": wavelength,
                        "y": y,
                        "x": x,
                    },
                ),
                "rho_s": xr.DataArray(
                    rho_s,
                    dims=["wavelength", "y", "x"],
                    coords={
                        "wavelength": wavelength,
                        "y": y,
                        "x": x,
                    },
                ),
                # "rho_path_gas_cor": xr.DataArray(
                #     rho_path_gas_cor,
                #     dims=["wavelength", "y", "x"],
                #     coords={
                #         "wavelength": wavelength,
                #         "y": y,
                #         "x": x,
                #     },
                # ),
                "rho_t_target": xr.DataArray(
                    rho_t_target,
                    dims=["wavelength", "y", "x"],
                    coords={
                        "wavelength": wavelength,
                        "y": y,
                        "x": x,
                    },
                ),
                "rho_t": xr.DataArray(
                    rho_t,
                    dims=["wavelength", "y", "x"],
                    coords={
                        "wavelength": wavelength,
                        "y": y,
                        "x": x,
                    },
                ),
                "gain": xr.DataArray(
                    gain,
                    dims=["wavelength", "y", "x"],
                    coords={
                        "wavelength": wavelength,
                        "y": y,
                        "x": x,
                    },
                ),
            }
        )

        # Save the results to data frame
        temp = xr.merge(
            (image_sub, ra_components, gas_component, ds_out),
        )

        temp = temp.to_dataframe()

        temp["uuid"] = uuid
        temp["land"] = land

        viccal_df = pd.concat([viccal_df, temp], axis=0)

    return viccal_df


if __name__ == "__main__":
    # insitu data
    # insitu_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/sas_rho.csv"
    insitu_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/jetski_rho.csv"
    # insitu_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/svc_rho.csv"
    in_situ = pd.read_csv(insitu_path)

    image_dir = "/D/Data/WISE/"

    images = [
        "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l1r.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l1r.nc",
        "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-l1r.nc",
        "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-l1r.nc",
    ]

    for image in images:
        l1 = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        viccal_df = run_vicarious_cal(l1, in_situ, window_size = 7, land = False)

        viccal_df.to_csv(
            os.path.join(
                "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/",
                f"{l1.image_name}_gain_jetski.csv",
            )
        )
