import math
import os

import xarray as xr
import pandas as pd
import numpy as np
import scipy.interpolate as sp
from tqdm import tqdm

from reverie import ReveCube
from reverie.image.tile import Tile
from reverie.utils.helper import fwhm_2_rsr
from reverie.correction.surface import z17

import logging


def run_ac_point(l1: ReveCube, in_situ, window_size, gain):
    """
    Function to compensate for the atmospheric effect on an image with vicarious gain.

    Parameters
    ----------
    l1
    in_situ
    window_size
    gain a data frame with columns: wavelength, gain

    Returns
    -------

    Description
    -----------
    The gains are computed on apparent reflectance (rho_t^\star = Lt / F0).
    Path reflectance from 6S used in the calculation is also apparent reflectance.
    Therfore the underestimation of reflectance due to the overestimation of irradiance at the plane level cancels out.

    """

    image = l1.in_ds
    # filter image for bad bands
    bad_band_list = image["radiance_at_sensor"].bad_band_list
    if isinstance(bad_band_list, str):
        bad_band_list = str.split(bad_band_list, ", ")
    bad_band_list = np.array(bad_band_list)
    good_band_indices = np.where(bad_band_list == "1")[0]
    good_bands_slice = slice(min(good_band_indices), max(good_band_indices) + 1)
    image = image.isel(wavelength=good_bands_slice)

    # filter the image for the in situ wavelength range
    # min_wavelength_in_situ = in_situ["wavelength"].min()
    # max_wavelength_in_situ = in_situ["wavelength"].max()
    min_wavelength_in_situ = 380
    max_wavelength_in_situ = 800

    # Filter the image wavelengths based on these values
    # Create a boolean mask for the wavelengths within the range
    mask = (image.wavelength >= min_wavelength_in_situ) & (
        image.wavelength <= max_wavelength_in_situ
    )

    # Apply the mask to the image
    image = image.isel(wavelength=mask)
    wavelength = image.wavelength.values

    # Load F0 from coddington
    F0 = xr.open_dataset(
        "/home/raphael/PycharmProjects/reverie/reverie/data/solar_irradiance/hybrid_reference_spectrum_c2022-11-30_with_unc.nc"
    )

    # F0["SSI"][(F0["Vacuum Wavelength"] > 350) & (F0["Vacuum Wavelength"] < 1000)].plot()

    logging.info("Calculating equvilent F0")

    _waves = F0["Vacuum Wavelength"].values
    _f0 = F0["SSI"].values
    unit = F0["SSI"].units

    # calculate extraterrestrial solar irradiance on a specific day considering the sun-earth distance
    # (https://en.wikipedia.org/wiki/Sunlight#:~:text=If%20the%20extraterrestrial%20solar%20radiation,hitting%20the%20ground%20is%20around)
    doy = l1.acq_time_z.timetuple().tm_yday
    distance_factor = 1 + 0.033412 * np.cos(2 * np.pi * (doy - 3) / 365.0)
    _f0_cal = _f0 * distance_factor

    resolution, wrange = 0.01, (_waves[0], _waves[-1] + 0.0001)

    _f0_c_fine = sp.interp1d(_waves, _f0_cal)(
        np.arange(wrange[0], wrange[1], resolution)
    )

    # From https://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python
    # def precision_and_scale(x):
    #     max_digits = 14
    #     int_part = int(abs(x))
    #     magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    #     if magnitude >= max_digits:
    #         return (magnitude, 0)
    #     frac_part = abs(x) - int_part
    #     multiplier = 10 ** (max_digits - magnitude)
    #     frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    #     while frac_digits % 10 == 0:
    #         frac_digits /= 10
    #     scale = int(math.log10(frac_digits))
    #     return (magnitude + scale, scale)
    #
    # precision_and_scale(F0["Vacuum Wavelength"][155])

    # Resolution is tricky to determine programmatically, hard code it
    resolution = 0.001

    rsr, wave = fwhm_2_rsr(
        np.full(len(wavelength), 5.05),
        wavelength,
        wrange=wrange,
        resolution=resolution,
    )
    f0_wise = np.zeros_like(wavelength, dtype=float)
    for i, item in enumerate(rsr):
        f0_wise[i] = np.sum(_f0_cal * item) / np.sum(item)

    # Simple interpolation to central wavelength doesn't work
    # temp = np.interp(
    #     x=wavelength, xp=F0["Vacuum Wavelength"].values, fp=F0["SSI"].values
    # )

    # Convert F0 from W m-2 nm-1 to uW cm-2 nm-1
    temp = f0_wise * 1e2

    # Add two new dims to temp (at the end)
    F0_3d = temp[:, np.newaxis, np.newaxis]

    # find the window on which we do the cal/val
    window_dict = l1.extract_pixel(
        insitu_path,
        max_time_diff=12,
        window_size=3,
    )[1]

    logging.info(f"Extracted window {window_dict.keys}")

    ac_df = pd.DataFrame()

    for uuid, window in tqdm(window_dict.items()):
        tile = Tile.FromWindow(window)

        image_sub = image.isel(x=slice(tile[0], tile[1]), y=slice(tile[2], tile[3]))

        # image["radiance_at_sensor"].mean(["x", "y"]).plot()
        # plt.show()

        # Do gas correction with 6S output "global_gas_trans_total"
        # It integrates all gas transmittance in the downward and upward direction
        # Dimensions of the LUT are {wavelength, sun_zenith, view_zenith, relative_azimuth,  sensor_altitude, water_vapor, ozone, target_pressure}
        # Variables taken from the image {wavelength, sun_zenith, view_zenith, relative_azimuth, sensor_altitude}
        # Variables taken from the GMAO MERRA2 model {water_vapor, ozone, target_pressure}

        total_lut = xr.open_dataset(
            "/home/raphael/PycharmProjects/reverie/reverie/data/lut/output/test_total.nc"
        )

        if l1.valid_mask is None:
            l1.get_valid_mask(tile=tile)

        n_pixels = len(
            image_sub.variables["sun_zenith"]
            .values[l1.get_valid_mask(tile=tile)]
            .reshape(-1, 1)
        )

        # Get dimension of image and create placeholder dataset for F0, total_trans

        x = image_sub.x.values
        y = image_sub.y.values

        # Repeat temp along the new dims
        F0_image = np.repeat(F0_3d, len(y), axis=1)
        F0_image = np.repeat(F0_image, len(x), axis=2)

        F0_image = xr.Dataset(
            data_vars=dict(
                F0=(
                    ["wavelength", "y", "x"],
                    F0_image,
                    {
                        "long_name": "Solar Spectral Irradiance Reference Spectrum (uW cm-2 nm-1)",
                        "standard_name": "solar_irradiance_per_unit_wavelength",
                        "units": "uW cm-2 nm-1",
                    },
                ),
            ),
            coords=dict(
                wavelength=wavelength,
                y=("y", y),
                x=("x", x),
            ),
        )

        atmosphere_components = xr.Dataset(
            data_vars=dict(
                path_reflectance=(
                    ["wavelength", "y", "x"],
                    np.full((len(wavelength), len(y), len(x)), np.nan, float),
                ),
                gas_trans_total=(
                    ["wavelength", "y", "x"],
                    np.full((len(wavelength), len(y), len(x)), np.nan, float),
                ),
                rayleigh_trans_total=(
                    ["wavelength", "y", "x"],
                    np.full((len(wavelength), len(y), len(x)), np.nan, float),
                ),
                aerosol_trans_total=(
                    ["wavelength", "y", "x"],
                    np.full((len(wavelength), len(y), len(x)), np.nan, float),
                ),
                spherical_albedo_total=(
                    ["wavelength", "y", "x"],
                    np.full((len(wavelength), len(y), len(x)), np.nan, float),
                ),
            ),
            coords=dict(
                wavelength=wavelength,
                y=("y", y),
                x=("x", x),
            ),
        )

        # up_trans = xr.Dataset(
        #     data_vars=dict(
        #         atmospheric_reflectance_at_sensor=(
        #             ["wavelength", "y", "x"],
        #             np.full((len(wavelength), len(y), len(x)), np.nan, float),
        #         ),
        #         gas_trans_up=(
        #             ["wavelength", "y", "x"],
        #             np.full((len(wavelength), len(y), len(x)), np.nan, float),
        #         ),
        #         rayleigh_trans_up=(
        #             ["wavelength", "y", "x"],
        #             np.full((len(wavelength), len(y), len(x)), np.nan, float),
        #         ),
        #         aerosol_trans_up=(
        #             ["wavelength", "y", "x"],
        #             np.full((len(wavelength), len(y), len(x)), np.nan, float),
        #         ),
        #         spherical_albedo=(
        #             ["wavelength", "y", "x"],
        #             np.full((len(wavelength), len(y), len(x)), np.nan, float),
        #         ),
        #     ),
        #     coords=dict(
        #         wavelength=wavelength,
        #         y=("y", y),
        #         x=("x", x),
        #     ),
        # )

        # create placeholder for (x, y) gas_trans

        pr = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        gt = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        rt = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        at = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)
        sa = np.full(l1.get_valid_mask(tile=tile).shape, np.nan, float)

        points = (
            total_lut.sol_zen.values,
            total_lut.view_zen.values,
            total_lut.relative_azimuth.values,
            total_lut.water.values,
            total_lut.ozone.values,
            total_lut.aot550.values,
            total_lut.target_pressure.values,
            total_lut.sensor_altitude.values,
            total_lut.wavelength.values,
        )

        for index, band in enumerate(wavelength):
            xi = np.hstack(
                [
                    image_sub.variables["sun_zenith"]
                    .values[l1.get_valid_mask(tile=tile)]
                    .reshape(-1, 1),
                    image_sub.variables["view_zenith"]
                    .values[l1.get_valid_mask(tile=tile)]
                    .reshape(-1, 1),
                    image_sub.variables["relative_azimuth"]
                    .values[l1.get_valid_mask(tile=tile)]
                    .reshape(-1, 1),
                    np.repeat(
                        image_sub.variables[
                            "atmosphere_mass_content_of_water_vapor"
                        ].values,
                        n_pixels,
                    ).reshape(-1, 1),
                    np.repeat(
                        image_sub.variables[
                            "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
                        ].values,
                        n_pixels,
                    ).reshape(-1, 1),
                    np.repeat(
                        image_sub.variables[
                            "aerosol_optical_thickness_at_555_nm"
                        ].values,
                        n_pixels,
                    ).reshape(-1, 1),
                    np.repeat(
                        image_sub.variables["surface_air_pressure"].values, n_pixels
                    ).reshape(-1, 1),
                    np.repeat((-image_sub.z.values / 1000), n_pixels).reshape(-1, 1),
                    np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
                ]
            )

            # Interpolate the values of global gas total transmittance
            pr_values = sp.interpn(
                points=points,
                values=total_lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :, :, :].values,
                xi=xi,
            )

            pr[l1.get_valid_mask(tile=tile)] = pr_values

            atmosphere_components["path_reflectance"][index, :, :] = pr

            gt_values = sp.interpn(
                points=points,
                values=total_lut["global_gas_trans_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            gt[l1.get_valid_mask(tile=tile)] = gt_values

            atmosphere_components["gas_trans_total"][index, :, :] = gt

            rt_values = sp.interpn(
                points=points,
                values=total_lut["rayleigh_trans_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            rt[l1.get_valid_mask(tile=tile)] = rt_values

            atmosphere_components["rayleigh_trans_total"][index, :, :] = rt

            at_values = sp.interpn(
                points=points,
                values=total_lut["aerosol_trans_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            at[l1.get_valid_mask(tile=tile)] = rt_values

            atmosphere_components["aerosol_trans_total"][index, :, :] = at

            sa_values = sp.interpn(
                points=points,
                values=total_lut["spherical_albedo_total"][
                    :, :, :, :, :, :, :, :, :
                ].values,
                xi=xi,
            )

            sa[l1.get_valid_mask(tile=tile)] = sa_values

            atmosphere_components["spherical_albedo_total"][index, :, :] = sa

        # for index, band in tqdm(
        #     enumerate(wavelength), desc="Computing up transmittances"
        # ):
        #     xi = np.hstack(
        #         [
        #             image.variables["sun_zenith"]
        #             .values[image.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             image.variables["view_zenith"]
        #             .values[image.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             image.variables["relative_azimuth"]
        #             .values[image.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             np.repeat(
        #                 image.variables[
        #                     "atmosphere_mass_content_of_water_vapor"
        #                 ].values,
        #                 n_pixels,
        #             ).reshape(-1, 1),
        #             np.repeat(
        #                 image.variables[
        #                     "equivalent_thickness_at_stp_of_atmosphere_ozone_content"
        #                 ].values,
        #                 n_pixels,
        #             ).reshape(-1, 1),
        #             np.repeat(
        #                 image.variables[
        #                     "aerosol_optical_thickness_at_555_nm"
        #                 ].values,
        #                 n_pixels,
        #             ).reshape(-1, 1),
        #             np.repeat(
        #                 image.variables["surface_air_pressure"].values, n_pixels
        #             ).reshape(-1, 1),
        #             np.repeat((-image.z.values / 1000), n_pixels).reshape(-1, 1),
        #             np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
        #         ]
        #     )
        #
        #     # Interpolate the values of global gas total transmittance
        #     pr_values = sp.interpn(
        #         points=points,
        #         values=total_lut["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :, :, :].values,
        #         xi=xi,
        #     )
        #
        #     pr[image.get_valid_mask(tile=tile)] = pr_values
        #
        #     up_trans["atmospheric_reflectance_at_sensor"][index, :, :] = pr
        #
        #     gt_values = sp.interpn(
        #         points=points,
        #         values=total_lut["global_gas_trans_upward"][
        #             :, :, :, :, :, :, :, :, :
        #         ].values,
        #         xi=xi,
        #     )
        #
        #     gt[image.get_valid_mask(tile=tile)] = gt_values
        #
        #     up_trans["gas_trans_up"][index, :, :] = gt
        #
        #     rt_values = sp.interpn(
        #         points=points,
        #         values=total_lut["rayleigh_trans_upward"][
        #             :, :, :, :, :, :, :, :, :
        #         ].values,
        #         xi=xi,
        #     )
        #
        #     rt[image.get_valid_mask(tile=tile)] = rt_values
        #
        #     up_trans["rayleigh_trans_up"][index, :, :] = rt
        #
        #     at_values = sp.interpn(
        #         points=points,
        #         values=total_lut["aerosol_trans_upward"][
        #             :, :, :, :, :, :, :, :, :
        #         ].values,
        #         xi=xi,
        #     )
        #
        #     at[image.get_valid_mask(tile=tile)] = rt_values
        #
        #     up_trans["aerosol_trans_up"][index, :, :] = at
        #
        #     sa_values = sp.interpn(
        #         points=points,
        #         values=total_lut["spherical_albedo_total"][
        #             :, :, :, :, :, :, :, :, :
        #         ].values,
        #         xi=xi,
        #     )
        #
        #     sa[image.get_valid_mask(tile=tile)] = sa_values
        #
        #     up_trans["spherical_albedo"][index, :, :] = sa

        ######## Rho sky Zhang 17

        rho_z17 = z17.get_sky_sun_rho(
            aot_550=float(image_sub.variables["aerosol_optical_thickness_at_555_nm"].mean().values),
            sun_zen=float(image_sub.variables["sun_zenith"].mean().values),
            view_ang=[float(image_sub.variables["view_zenith"].mean().values), float(image_sub.variables["relative_azimuth"].mean().values)],
            water_salinity=30,
            water_temperature=17,
            wavelength=wavelength,
            wind_speed=float(image_sub.variables["wind_speed"].mean().values)
        )

        sky = rho_z17["sky"]
        # Add two new dims to temp
        sky_3d = sky[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dims
        sky_3d = np.repeat(sky_3d, len(y), axis=1)
        sky_3d = np.repeat(sky_3d, len(x), axis=2)

        sun = rho_z17["sun"]
        # Add two new dims to temp
        sun_3d = sun[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dims
        sun_3d = np.repeat(sun_3d, len(y), axis=1)
        sun_3d = np.repeat(sun_3d, len(x), axis=2)

        rho = rho_z17["rho"]
        # Add two new dims to temp
        rho_3d = rho[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dims
        rho_3d = np.repeat(rho_3d, len(y), axis=1)
        rho_3d = np.repeat(rho_3d, len(x), axis=2)

        rho_sky = xr.Dataset(
            data_vars=dict(
                rho_sky=(
                    ["wavelength", "y", "x"],
                    sky_3d,
                ),
                rho_sun=(
                    ["wavelength", "y", "x"],
                    sun_3d,
                ),
                rho=(
                    ["wavelength", "y", "x"],
                    rho_3d,
                ),
            ),
            coords=dict(
                wavelength=wavelength,
                y=("y", y),
                x=("x", x),
            ),
        )

########## Rho sky OSOAA

        # rho_sky_nc = nc.Dataset(
        #     "/D/Documents/phd/thesis/2_chapter/data/ACOLITE-RSKY-202102-82W-MOD2.nc"
        # )
        #
        # rho_sky = xr.Dataset(
        #     data_vars=dict(
        #         rho_sky=(
        #             ["wavelength", "y", "x"],
        #             np.full((len(wavelength), len(y), len(x)), np.nan, float),
        #         ),
        #     ),
        #     coords=dict(
        #         wavelength=wavelength,
        #         y=("y", y),
        #         x=("x", x),
        #     ),
        # )
        #
        # # Compute sea surface reflection from the ACOLITE Lut (OSOAA model)
        #
        # sr = np.full(image.get_valid_mask(tile=tile).shape, np.nan, float)
        #
        # for index, band in enumerate(wavelength):
        #     xi = np.hstack(
        #         [
        #             np.repeat(band * 1e-3, n_pixels).reshape(-1, 1),
        #             image.variables["relative_azimuth"]
        #             .values[image.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             image.variables["view_zenith"]
        #             .values[image.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             image.variables["sun_zenith"]
        #             .values[image.get_valid_mask(tile=tile)]
        #             .reshape(-1, 1),
        #             np.repeat(
        #                 image.variables["wind_speed"].values,
        #                 n_pixels,
        #             ).reshape(-1, 1),
        #             np.repeat(
        #                 image.variables[
        #                     "aerosol_optical_thickness_at_555_nm"
        #                 ].values,
        #                 n_pixels,
        #             ).reshape(-1, 1),
        #         ]
        #     )
        #
        #     sea_rho = sp.interpn(
        #         points=(
        #             rho_sky_nc.wave[()],
        #             rho_sky_nc.azi[()],
        #             rho_sky_nc.thv[()],
        #             rho_sky_nc.ths[()],
        #             rho_sky_nc.wind[()],
        #             rho_sky_nc.tau[()],
        #         ),
        #         values=rho_sky_nc.variables["surface"][:, :, :, :, :, :],
        #         xi=xi,
        #         method="cubic"
        #     )
        #
        #     sr[image.get_valid_mask(tile=tile)] = sea_rho
        #
        #     rho_sky["rho_sky"][index, :, :] = sr

        total_trans = (
            atmosphere_components["gas_trans_total"]
            * atmosphere_components["rayleigh_trans_total"]
            * atmosphere_components["aerosol_trans_total"]
        )

        spherical_albedo = atmosphere_components["spherical_albedo_total"]

        rho_path = atmosphere_components["path_reflectance"]

        gain_values = gain["gain_mean"]
        gain_wavelength = gain["wavelength"]

        temp = np.interp(x=wavelength, xp=gain_wavelength, fp=gain_values)

        # Add two new dims to temp (at the end)
        temp_3d = temp[:, np.newaxis, np.newaxis]

        # Repeat temp along the new dims
        temp_3d = np.repeat(temp_3d, len(y), axis=1)
        temp_3d = np.repeat(temp_3d, len(x), axis=2)

        gain_image = xr.Dataset(
            data_vars=dict(
                gain=(
                    ["wavelength", "y", "x"],
                    temp_3d,
                ),
            ),
            coords=dict(
                wavelength=wavelength,
                y=("y", y),
                x=("x", x),
            ),
        )

        # gain_mean.plot()
        #
        # plt.fill_between(
        #     gain_mean.wavelength,
        #     gain_mean - gain_std,
        #     gain_mean + gain_std,
        #     color="gray",
        #     alpha=0.2,
        # )
        #
        # plt.xlabel(r"Wavelength [nm]")
        # plt.ylabel(r"Gain = rho_t_target / rho_t")
        #
        # plt.show()

        rho_t = (math.pi * image_sub["radiance_at_sensor"]) / (F0_image["F0"] * np.cos(np.deg2rad(image_sub["sun_zenith"])))
        rho_t_cal = rho_t * gain_image["gain"]

        # rho_t.mean(["x", "y"]).plot()
        # rho_t_cal.mean(["x", "y"]).plot()
        # plt.show()

        rho_s_cal = (rho_t_cal - rho_path) / (total_trans + spherical_albedo * (rho_t_cal - rho_path))
        # rho_s_cal.mean(["x", "y"]).plot()
        # plt.show()

        rho_w_cal = rho_s_cal - rho_sky["rho_sky"]

        # rho_w_cal.mean(["x", "y"]).plot()
        # plt.show()

        rho_w_cal.name = "rho_w"

        # Save the results to data frame
        temp = xr.merge(
            (image_sub, F0_image, atmosphere_components, gain_image, rho_w_cal),
        )

        temp = temp.to_dataframe()

        temp["uuid"] = uuid

        ac_df = pd.concat([ac_df, temp], axis=0)

    return ac_df


if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        # "MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.nc",
        # "MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.nc",
        # "MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc",
    ]

    for image in images:
        l1 = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        # insitu data
        # insitu_path = "/D/Documents/phd/thesis/2_chapter/data/wise/match_bg_20220705.csv"
        insitu_path = "/D/Documents/phd/training/sls_2024/practicals/presentation/exctraction_points.csv"
        in_situ = pd.read_csv(insitu_path)

        gain = pd.read_csv(
            "/D/Documents/phd/thesis/2_chapter/data/wise/viccal/gain_final.csv"
        )

        ac_matchup = run_ac_point(l1, in_situ, 7, gain)

        ac_matchup.to_csv(
            os.path.join(
                # "/D/Documents/phd/thesis/2_chapter/data/wise/viccal/",
                "/D/Documents/phd/training/sls_2024/practicals/presentation/"
                f"{l1.image_name}_rho_w_L2.csv",
            )
        )

