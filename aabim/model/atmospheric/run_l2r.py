import logging
import math
import os
import shutil
import time

from scipy import stats
import xarray as xr
import numpy as np
import scipy.interpolate as sp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import re
import glob
from scipy.interpolate import RegularGridInterpolator
import multiprocessing
from functools import partial
import math
import pandas as pd


from rasterio.windows import Window

from reverie import ReveCube
import reverie.correction.atmospheric.get_atmosphere as atm
from reverie.correction.surface.rayleigh import get_sky_glint
import dsf
import lut
from reverie.correction.surface.rayleigh import fresnel_reflectance
from reverie.correction.surface.water_refractive_index import get_water_refractive_index

def process_window(window, l1_path, output_dir, aod555_dsf, aero_interps, gas_interp, mask_water):

    l1 = ReveCube.from_reve_nc(l1_path)

    y_slice = slice(int(window.row_off), int(window.row_off + window.height))
    x_slice = slice(int(window.col_off), int(window.col_off + window.width))

    image_sub = l1.in_ds.isel(x=x_slice, y=y_slice)
    rho_t = image_sub["rho_at_sensor"]

    if not np.isnan(rho_t).all():
        ra_components = atm.get_ra(l1, window, l1.wavelength, aod555_dsf, *aero_interps)
        gas_component = atm.get_gas(l1, window, l1.wavelength, gas_interp)

        rho_path_ra = ra_components["rho_path"].values
        t_ra = ra_components["trans_ra"].values
        s_ra = ra_components["spherical_albedo_ra"].values
        t_g = gas_component["t_gas"].values

        rho_path = (rho_path_ra * t_g) / (1 - rho_path_ra * s_ra)
        rho_s = (rho_t - rho_path) / (t_ra * t_g + s_ra * (rho_t - rho_path))

        # theta_0 = np.deg2rad(image_sub["sun_zenith"])
        # theta_v = np.deg2rad(image_sub["view_zenith"])
        # phi_0 = np.deg2rad(image_sub["sun_azimuth"])
        # phi_v = np.deg2rad(image_sub["view_azimuth"])

        # sky_glint = get_sky_glint(l1.wavelength, theta_0, theta_v, phi_0, phi_v, 1007)
        sky_glint = ra_components["sky_glint_ra"].values

        mask_water_window = mask_water[y_slice, x_slice]
        rho_w = np.where(mask_water_window, rho_s - sky_glint, np.nan)

        wavelength = l1.wavelength
        nir_i = np.abs(wavelength - 800).argmin()
        rho_nir = rho_w[nir_i, :, :]

        # Gao and li 2021
        n_w = get_water_refractive_index(30, 12, wavelength)
        rho_f = fresnel_reflectance(0, n_w)
        rho_f_nir = rho_f[nir_i]
        ref_ratio = rho_nir / rho_f_nir

        rho_w_g21 = rho_w - (rho_f * ref_ratio)

    else:
        rho_s = np.full_like(rho_t, np.nan)
        rho_w = np.full_like(rho_t, np.nan)
        rho_w_g21 = np.full_like(rho_t, np.nan)

    y_coords = l1.in_ds["y"].isel(y=y_slice)
    x_coords = l1.in_ds["x"].isel(x=x_slice)
    wl_coords = l1.wavelength

    out_path = os.path.join(
        output_dir,
        f"rho_surface_window_{window.row_off}-{window.row_off+window.height}_{window.col_off}-{window.col_off+window.width}.nc")

    ds_out = xr.Dataset(
        {
            "rho_surface": xr.DataArray(
                rho_s,
                dims=["wavelength", "y", "x"],
                coords={
                    "wavelength": wl_coords,
                    "y": y_coords,
                    "x": x_coords,
                },
            ),
            "rho_w": xr.DataArray(
                rho_w,
                dims=["wavelength", "y", "x"],
                coords={
                    "wavelength": wl_coords,
                    "y": y_coords,
                    "x": x_coords,
                },
            ),
            "rho_w_g21": xr.DataArray(
                rho_w_g21,
                dims=["wavelength", "y", "x"],
                coords={
                    "wavelength": wl_coords,
                    "y": y_coords,
                    "x": x_coords,
                },
            ),
        }
    )

    ds_out.attrs["row_off"] = int(window.row_off)
    ds_out.attrs["row_end"] = int(window.row_off + window.height)
    ds_out.attrs["col_off"] = int(window.col_off)
    ds_out.attrs["col_end"] = int(window.col_off + window.width)

    ds_out.to_netcdf(out_path)

    ### DEV

    # import matplotlib.pyplot as plt
    #
    # # Select indices for blue, green, red bands
    # blue_i = np.abs(wl_coords - 470).argmin()
    # green_i = np.abs(wl_coords - 550).argmin()
    # red_i = np.abs(wl_coords - 650).argmin()
    #
    # # Extract and normalize each band
    # rgb = np.stack([
    #     rho_w[red_i],  # Red
    #     rho_w[green_i],  # Green
    #     rho_w[blue_i],  # Blue
    # ], axis=-1)
    #
    # # Normalize to [0, 1] for display
    # rgb_min = np.nanmin(rgb)
    # rgb_max = np.nanmax(rgb)
    # rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)
    # rgb_norm = np.clip(rgb_norm, 0, 1)
    #
    # plt.imshow(rgb_norm)
    # plt.title("rho_w RGB composite")
    # plt.axis("off")
    # plt.show()

    ### DEV

    return out_path

def run_l2r(l1, aod_method, gain_df = None, wavelength_filter = None):

    log_dir = os.path.dirname(l1.in_path)
    log_file = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(l1.in_path))[0]}_l2r.log")

    # Get the root logger and remove all handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set up file and stream handlers
    file_handler = logging.FileHandler(log_file, mode='w')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    start_time = time.time()
    logging.info("Starting run_l2r processing")

    if "l1rg" in l1.in_path:
        out_name = re.sub(r'l1\w+', 'l2rg', l1.in_path)
    else:
        out_name = re.sub(r'l1\w+', 'l2r', l1.in_path)

    if wavelength_filter is not None:
        l1.mask_wavelength(wavelength_filter)

    if gain_df is not None:
        out_name = re.sub(r'l1\w+', 'l2rg', l1.in_path)

        wavelength_mask = gain_df["wavelength"].values
        l1.mask_wavelength(wavelength_mask)

    wavelength = l1.wavelength
    mask_water = l1.in_ds.mask_water.values.astype(bool)


    scale_factor = 1e-5 # NetCDF divide by scale when writing data and multiply when reading
    l1.create_reve_nc(out_name)

    if aod_method == "dsf":
        # Create rho_dark spectrum with corresponding pixel values
        rho_t = l1.in_ds.rho_at_sensor
        sol_zen = l1.in_ds.sun_zenith.values.astype(np.float32)
        view_zen = l1.in_ds.view_zenith.values.astype(np.float32)
        sol_azi = l1.in_ds.sun_azimuth.values.astype(np.float32)
        view_azi = l1.in_ds.view_azimuth.values.astype(np.float32)
        raa = l1.in_ds.relative_azimuth.values.astype(np.float32)

        rho_dark = np.full_like(wavelength, np.nan)
        sol_zen_dark = np.full_like(wavelength, np.nan)
        view_zen_dark = np.full_like(wavelength, np.nan)
        sol_azi_dark = np.full_like(wavelength, np.nan)
        view_azi_dark = np.full_like(wavelength, np.nan)
        relative_azimuth_dark = np.full_like(wavelength, np.nan)
        mask_water_dark = np.full_like(wavelength, np.nan)

        args_list = [(i, wl, rho_t, sol_zen, view_zen, sol_azi, view_azi, raa, mask_water) for i, wl in
                     enumerate(wavelength)]
        results = process_map(
            dsf.compute_rho_dark,
            args_list,
            max_workers=os.cpu_count() - 2,
            desc="Computing rho_dark in parallel"
        )

        for i, sol_zen_val, view_zen_val, sol_azi_val, view_azi_val, relative_azimuth_val, rho_dark_val, mask_water_val in results:
            rho_dark[i] = rho_dark_val
            sol_zen_dark[i] = sol_zen_val
            view_zen_dark[i] = view_zen_val
            sol_azi_dark[i] = sol_azi_val
            view_azi_dark[i] = view_azi_val
            relative_azimuth_dark[i] = relative_azimuth_val
            mask_water_dark[i] = mask_water_val

        if np.any(mask_water_dark):
            logging.debug("dark pixel found in water")

        # Interpolate rho_path to rho_dark and get the corresponding aod555 value
        target_pressure = l1.in_ds.surface_air_pressure.values
        sensor_altitude = l1.in_ds.z.values
        water = l1.in_ds.atmosphere_mass_content_of_water_vapor.values
        ozone = l1.in_ds.equivalent_thickness_at_stp_of_atmosphere_ozone_content.values

        aod555 = dsf.aot555_dsf(
            wavelength[~np.isnan(sol_zen_dark)],
            rho_dark[~np.isnan(sol_zen_dark)],
            sol_zen_dark[~np.isnan(sol_zen_dark)],
            view_zen_dark[~np.isnan(sol_zen_dark)],
            sol_azi_dark[~np.isnan(sol_zen_dark)],
            view_azi_dark[~np.isnan(sol_zen_dark)],
            relative_azimuth_dark[~np.isnan(sol_zen_dark)],
            target_pressure,
            sensor_altitude,
            water,
            ozone,
            mask_water_dark,
            name=l1.image_name
        )

        # aod555 = l1.in_ds.aerosol_optical_thickness_at_555_nm.values

        logging.info("Computed aod555 = {}".format(aod555))

    if aod_method == "merra2":
        aod555 = l1.in_ds.variables[
                        "aerosol_optical_thickness_at_555_nm"
                    ].values

    # Determine the available RAM size
    # ram_gb = helper.get_available_ram_gb() - 4

    # Estimate the corresponding window size (including wavelength)
    # window_size = estimate_window_size(l1, wavelength, ram_gb)
    window_size = 900

    # Divide the image in windows
    windows = l1.create_windows(window_size)

    logging.info(
        f"{len(windows)} windows to process at {window_size} pixels"
    )

    l1.create_var_nc(name="rho_surface", type="i4", dims=("wavelength", "y", "x"), comp="zlib", scale=scale_factor)
    l1.create_var_nc(name="rho_w", type="i4", dims=("wavelength", "y", "x"), comp="zlib", scale=scale_factor)
    l1.create_var_nc(name="rho_w_g21", type="i4", dims=("wavelength", "y", "x"), comp="zlib", scale=scale_factor)

    # non_empty_windows = []
    # for window in tqdm(windows, desc="Filtering windows"):
    #     y_slice = slice(window.row_off, window.row_off + window.height)
    #     x_slice = slice(window.col_off, window.col_off + window.width)
    #
    #     rho_t = l1.in_ds.isel(
    #         x=x_slice,
    #         y=y_slice
    #     )["rho_at_sensor"]
    #
    #     if not np.isnan(rho_t).all():
    #         non_empty_windows.append(window)
    #     else:
    #         l1.out_ds.variables["rho_surface"][:, y_slice, x_slice] = l1.no_data * scale_factor
    #         l1.out_ds.variables["rho_w"][:, y_slice, x_slice] = l1.no_data * scale_factor
    #
    # logging.info(
    #     f"{len(non_empty_windows)} non-empty windows to process"
    # )

    # output_dir = os.path.join(os.path.dirname(l1.in_path),"temp_windows_"+l1.image_name)  # Must exist
    # os.makedirs(output_dir, exist_ok=True)

    # aero_interps, aero_points = atm.build_aer_interpolators(lut.load_aer())
    # gas_interp, gas_points = atm.build_gas_interpolator(lut.load_gas())

    aer_lut = lut.load_aer()  # your WISE-resampled LUT file
    gas_lut = lut.load_gas()

    aer_lut = lut.slice_lut_to_wavelengths(aer_lut, wavelength)
    gas_lut = lut.slice_lut_to_wavelengths(gas_lut, wavelength)

    aero_interps, _ = atm.build_aer_interpolators_no_wl(aer_lut)
    gas_interp, _ = atm.build_gas_interpolator_no_wl(gas_lut)

    for window in tqdm(windows, desc="Processing windows"):
        # process_window(
        #     window,
        #     l1_path=l1.in_path,
        #     output_dir=output_dir,
        #     aod555_dsf=aod555,
        #     aero_interps=list(aero_interps.values()),
        #     gas_interp=gas_interp,
        #     mask_water=mask_water
        # )

        y_slice = slice(int(window.row_off), int(window.row_off + window.height))
        x_slice = slice(int(window.col_off), int(window.col_off + window.width))

        image_sub = l1.in_ds.isel(x=x_slice, y=y_slice)

        if gain_df is not None:
            # gain_1d_intercept = np.array(gain_df["intercept_estimate"][1:])
            gain_1d_coeff = np.array(gain_df["rho_t_estimate"][1:])

            # Add two new dims to temp
            # gain_3d_intercept = gain_1d_intercept[:, np.newaxis, np.newaxis]
            gain_3d_coeff = gain_1d_coeff[:, np.newaxis, np.newaxis]

            # Repeat temp along the new dims
            # gain_3d_intercept = np.repeat(gain_3d_intercept, window.height, axis=1)
            # gain_intercept = np.repeat(gain_3d_intercept, window.width, axis=2)

            gain_3d_coeff = np.repeat(gain_3d_coeff, window.height, axis=1)
            gain_coeff = np.repeat(gain_3d_coeff, window.width, axis=2)

            # rho_t = image_sub["rho_at_sensor"] * gain_coeff + gain_intercept
            rho_t = image_sub["rho_at_sensor"] * gain_coeff
        else:
            rho_t = image_sub["rho_at_sensor"]

        ### DEV

        # import matplotlib.pyplot as plt
        #
        # # Find mask where any wavelength is not NaN at (y, x)
        # mask_valid = ~np.isnan(rho_t)
        # mask_any_valid = np.any(mask_valid, axis=0)  # shape: (y, x)
        #
        # # Get the (y, x) indices
        # y_idx, x_idx = np.where(mask_any_valid)
        # # y_idx and x_idx are arrays of indices where at least one wavelength is not NaN
        #
        # plt.figure()
        # plt.plot(wavelength, rho_t[:, y_idx[200], x_idx[200]], marker="o")
        # plt.plot(wavelength, image_sub["rho_at_sensor"][:, y_idx[200], x_idx[200]], marker="x")
        # plt.xlabel("wavelength (nm)")
        # plt.ylabel("rho_w")
        # plt.show()

        ###

        if not np.isnan(rho_t).all():
            # ra_components = atm.get_ra(l1, window, l1.wavelength, aod555, *list(aero_interps.values()))
            # gas_component = atm.get_gas(l1, window, l1.wavelength, gas_interp)

            ra_components = atm.get_ra_no_wl(l1, window, l1.wavelength, aod555, aero_interps)
            gas_component = atm.get_gas_no_wl(l1, window, l1.wavelength, gas_interp)

            rho_path_ra = ra_components["rho_path"].values
            t_ra = ra_components["trans_ra"].values
            s_ra = ra_components["spherical_albedo_ra"].values
            t_g = gas_component["t_gas"].values

            # rho_path_gas_cor = (rho_path_ra * t_g) / (1 - rho_path_ra * s_ra)
            # rho_s = (rho_t - rho_path_ra) / (t_ra * t_g + s_ra * (rho_t - rho_path_ra))

            rho_s = (rho_t / t_g - rho_path_ra) / t_ra + s_ra * (rho_t / t_g - rho_path_ra)

            # theta_0 = np.deg2rad(image_sub["sun_zenith"])
            # theta_v = np.deg2rad(image_sub["view_zenith"])
            # phi_0 = np.deg2rad(image_sub["sun_azimuth"])
            # phi_v = np.deg2rad(image_sub["view_azimuth"])

            # sky_glint = get_sky_glint(l1.wavelength, theta_0, theta_v, phi_0, phi_v, 1007)
            sky_glint = ra_components["sky_glint_ra"].values

            mask_water_window = mask_water[y_slice, x_slice]
            rho_w = np.where(mask_water_window, rho_s - sky_glint, np.nan)

            wavelength = l1.wavelength
            nir_i = np.abs(wavelength - 800).argmin()
            rho_nir = rho_w[nir_i, :, :]

            # Gao and li 2021
            n_w = get_water_refractive_index(30, 12, wavelength)
            rho_f = fresnel_reflectance(0, n_w)
            rho_f_nir = rho_f[nir_i]
            ref_ratio = rho_nir / rho_f_nir

            rho_w_g21 = rho_w - rho_f[:, None, None] * ref_ratio[None, :, :]#(rho_f * ref_ratio)

            ### DEV

            # import matplotlib.pyplot as plt
            #
            # rho_w_valid = np.isnan(rho_w)
            #
            # plt.figure()
            # plt.plot(wavelength, rho_w[:, 700, 999], marker="x")
            # plt.xlabel("wavelength (nm)")
            # plt.ylabel("rho_w")
            # plt.show()

            ###

        else:
            rho_s = np.full_like(rho_t, np.nan)
            rho_w = np.full_like(rho_t, np.nan)
            rho_w_g21 = np.full_like(rho_t, np.nan)

        np.nan_to_num(rho_s, copy=False, nan=l1.no_data * scale_factor)
        np.nan_to_num(rho_w, copy=False, nan=l1.no_data * scale_factor)
        np.nan_to_num(rho_w_g21, copy=False, nan=l1.no_data * scale_factor)

        l1.out_ds["rho_surface"][:, y_slice, x_slice] = rho_s
        l1.out_ds["rho_w"][:, y_slice, x_slice] = rho_w
        l1.out_ds["rho_w_g21"][:, y_slice, x_slice] = rho_w_g21

    # process_func = partial(
    #     process_window,
    #     l1_path=l1.in_path,
    #     output_dir=output_dir,
    #     aod555_dsf=aod555,
    #     aero_interps=list(aero_interps.values()),
    #     gas_interp=gas_interp,
    #     mask_water=mask_water
    # )
    #
    # results = process_map(
    #     process_func,
    #     non_empty_windows,
    #     max_workers = 1 ,#os.cpu_count() - 2,
    #     desc="Processing windows in parallel"
    # )

    # output_dir = os.path.join(os.path.dirname(l1.in_path),"temp_windows_"+l1.image_name)
    # wavelength = l1.wavelength
    # scale_factor = 1e-5
    # aod555 = 0.03407641127705574
    # import netCDF4
    # l1.out_ds = netCDF4.Dataset("/D/Data/WISE/ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l2rg.nc", "a", format="NETCDF4")
    # l1.no_data = math.trunc(netCDF4.default_fillvals["i4"] * scale_factor)

    # logging.debug("Merging results")
    # all_files = sorted(glob.glob(f"{output_dir}/rho_surface_window_*.nc"))
    #
    # for file in tqdm(all_files, desc="Writing results"):
    #     ds = xr.open_dataset(file)
    #
    #     # row_off = ds.attrs["row_off"]
    #     # row_end = ds.attrs["row_end"]
    #     # col_off = ds.attrs["col_off"]
    #     # col_end = ds.attrs["col_end"]
    #     # y_slice = slice(row_off, row_end)
    #     # x_slice = slice(col_off, col_end)
    #
    #     basename = os.path.basename(file)
    #     # Remove extension and split by underscores
    #     parts = basename.replace(".nc", "").split("_")
    #     # Extract row and column ranges
    #     row_range = parts[3].split("-")
    #     col_range = parts[4].split("-")
    #     # Convert to integer slices
    #     y_slice = slice(int(row_range[0]), int(row_range[1]))
    #     x_slice = slice(int(col_range[0]), int(col_range[1]))
    #
    #     rho_s = ds["rho_surface"][:, :, :].values
    #     np.nan_to_num(rho_s, copy=False, nan=l1.no_data * scale_factor)
    #
    #     rho_w = ds["rho_w"][:, :, :].values
    #     np.nan_to_num(rho_w, copy=False, nan=l1.no_data * scale_factor)
    #
    #     l1.out_ds["rho_surface"][:, y_slice, x_slice] = rho_s
    #     l1.out_ds["rho_w"][:, y_slice, x_slice] = rho_w
    #
    #     ds.close()

    # ds = xr.open_mfdataset(all_files, combine="by_coords")
    # # ds.to_netcdf("final_rho_surface.nc")
    #
    # # If sorted ascending flip it back
    # if np.all(np.diff(ds.y.values) > 0):
    #     logging.debug("Y sorted ascending")
    #     ds = ds.sortby("y", ascending=False)
    #
    # l1.create_var_nc(name="rho_w_c", type="i4", dims=( "wavelength", "y", "x", ), comp="zlib", scale=scale_factor)
    #
    # nir_i = np.abs(wavelength - 900).argmin()
    #
    # rho_w_ref = ds["rho_w"][nir_i, :, :].values
    # rho_nir_min = np.nanmin(rho_w_ref)
    #
    # for i in tqdm(range(len(wavelength)), desc="Computing glint corr in final ds:"):
    #     rho_s = ds["rho_surface"][i, :, :].values
    #     np.nan_to_num(rho_s, copy=False, nan=l1.no_data * scale_factor)
    #
    #     rho_w = ds["rho_w"][i, :, :].values
    #     np.nan_to_num(rho_w, copy=False, nan=l1.no_data * scale_factor)
    #
    #     rho_w_c = rho_w - (rho_w_ref - rho_nir_min)
    #     np.nan_to_num(rho_w_c, copy=False, nan=l1.no_data * scale_factor)
    #
    #     l1.out_ds["rho_surface"][i, :, :] = rho_s
    #     l1.out_ds["rho_w"][i, :, :] = rho_w
    #     l1.out_ds.variables["rho_w_c"][i, :, :] = rho_w_c


    l1.out_ds["rho_surface"].aod555_dsf = aod555

    # ds = l1.out_ds

    # ds = xr.open_mfdataset(all_files, combine="by_coords")
    # # ds.to_netcdf("final_rho_surface.nc")
    #
    # # If sorted ascending flip it back
    # if np.all(np.diff(ds.y.values) > 0):
    #     logging.debug("Y sorted ascending")
    #     ds = ds.sortby("y", ascending=False)
    #
    # # aer_lut = lut.load_aer()
    # # gas_lut = lut.load_gas()
    # # aero_interps, aer_points = atm.build_aer_interpolators(aer_lut)
    # # gas_interp, gas_points = atm.build_gas_interpolator(gas_lut)
    # # interp_rho, interp_t, interp_s = aero_interps.values()
    # # interp_tgas = gas_interp
    # #
    # # for window in tqdm(non_empty_windows, desc="Processing windows"):
    # #     y_slice = slice(int(window.row_off), int(window.row_off + window.height))
    # #     x_slice = slice(int(window.col_off), int(window.col_off + window.width))
    # #
    # #     rho_t = l1.in_ds.isel(
    # #         x=x_slice,
    # #         y=y_slice
    # #     )["rho_at_sensor"]
    # #
    # #     ra_components = atm.get_ra(l1, window, wavelength, aod555, interp_rho, interp_t, interp_s)
    # #     gas_component = atm.get_gas(l1, window, wavelength, interp_tgas)
    # #     rho_path_ra = ra_components["rho_path"].values
    # #     t_ra = ra_components["trans_ra"].values
    # #     s_ra = ra_components["spherical_albedo_ra"].values
    # #     t_g = gas_component["t_gas"].values
    # #
    # #     # rho_path_ra, t_ra, s_ra = atm.get_ra_vectorized(l1, window, wavelength, aod555, interp_rho, interp_t, interp_s)
    # #     # t_g = atm.get_gas_vectorized(l1, window, wavelength, interp_tgas)
    # #
    # #     rho_path = (rho_path_ra * t_g) / (1 - rho_path_ra * s_ra)
    # #
    # #     rho_s = (rho_t - rho_path) / (t_ra * t_g + s_ra * (rho_t - rho_path))
    # #
    # #     np.nan_to_num(rho_s, copy=False, nan=l1.no_data * scale)
    # #     l1.out_ds.variables["rho_surface"][:, y_slice, x_slice] = rho_s
    #
    # # Create indices and mask
    # logging.debug("Computing masks from rho_s")
    #
    # blue_i = np.abs(wavelength - 400).argmin()
    # green_i = np.abs(wavelength - 550).argmin()
    # red_i = np.abs(wavelength - 600).argmin()
    # nir_i = np.abs(wavelength - 900).argmin()
    # swir_i = np.abs(wavelength - 2190).argmin()
    #
    # blue = ds.isel(wavelength=blue_i)["rho_surface"].values
    # green = ds.isel(wavelength=green_i)["rho_surface"].values
    # red = ds.isel(wavelength=red_i)["rho_surface"].values
    # nir = ds.isel(wavelength=nir_i)["rho_surface"].values
    # swir = ds.isel(wavelength=swir_i)["rho_surface"].values
    #
    # ndwi = (green - swir) / (green + swir)
    # ndvi = (nir - red) / (nir + red)
    # s_vis_nir = nir / (blue + green)
    #
    # # mask_water = ((ndwi > 0) & (ndvi < 0.1)).astype(int)
    # valid_mask = l1.get_valid_mask()
    # mask_water = np.full(valid_mask.shape, np.nan, dtype=float)
    # mask_water[valid_mask] = (ndwi > 0)[valid_mask]
    #
    # l1.create_var_nc(name="ndwi", type="i4", dims=("y", "x"), comp="zlib", scale=scale_factor)
    # l1.create_var_nc(name="ndvi", type="i4", dims=("y","x"), comp="zlib", scale=scale_factor)
    # l1.create_var_nc(name="s_vis_nir", type="i4", dims=("y", "x"), comp="zlib", scale=scale_factor)
    #
    # np.nan_to_num(ndwi, copy=False, nan=l1.no_data * scale_factor)
    # np.nan_to_num(ndvi, copy=False, nan=l1.no_data * scale_factor)
    # np.nan_to_num(s_vis_nir, copy=False, nan=l1.no_data * scale_factor)
    #
    # l1.out_ds["ndwi"][:, :] = ndwi
    # l1.out_ds["ndvi"][:, :] = ndvi
    # l1.out_ds["s_vis_nir"][:, :] = s_vis_nir
    #
    # l1.create_var_nc(
    #     name="mask_water",
    #     type="u1",
    #     dims=(
    #         "y",
    #         "x",
    #     ),
    #     comp="zlib",
    #     complevel=1,
    #     scale=1,
    # )
    #
    # mask_water_int = np.full(valid_mask.shape, np.nan, dtype=float)
    # mask_water_int[valid_mask] = mask_water[valid_mask].astype(int)
    # np.nan_to_num(mask_water_int, copy=False, nan=l1.no_data)
    # l1.out_ds["mask_water"][:, :] = mask_water_int

    logging.debug("Copying geometry")

    geom = {
        "sun_azimuth": l1.in_ds.sun_azimuth,
        "sun_zenith": l1.in_ds.sun_zenith,
        "view_azimuth": l1.in_ds.view_azimuth,
        "view_zenith": l1.in_ds.view_zenith,
        "raa": l1.in_ds.relative_azimuth,
    }

    for var in tqdm(geom, desc="Writing geometry"):
        l1.create_var_nc(
            name=var,
            type="i4",
            dims=(
                "y",
                "x",
            ),
            scale=scale_factor,
        )
        data = geom[var]

        np.nan_to_num(data, copy=False, nan=l1.no_data * scale_factor)

        l1.out_ds.variables[var][:, :] = data

    l1.out_ds.close()
    # shutil.rmtree(output_dir)

    elapsed = time.time() - start_time
    logging.info(f"Finished run_l2r in {elapsed:.2f} seconds")

    return 0

if __name__ == "__main__":

    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-l1rg.nc",
        "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l1r.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l1r.nc",
        "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-l1r.nc",
        "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-l1r.nc",
    ]

    gain_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/gain_ratio_sas.csv"
    gain = pd.read_csv(gain_path)

    wavelength_filter = (370, 800)

    for image in images:
        l1c = ReveCube.from_reve_nc(os.path.join(image_dir, image))
        # run_l2r(l1c, aod_method="merra2", gain_df=gain, wavelength_filter=wavelength_filter)
        run_l2r(l1c, aod_method="dsf", gain_df=None)

    # images = [
    #     # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-l2rg.nc",
    #     # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc",
    #     # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc",
    #     # "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-l2rg.nc",
    # ]
    #
    # from run_l2w import run_l2w
    #
    # for image in images:
    #
    #     l2r = ReveCube.from_reve_nc(os.path.join(image_dir, image))
    #     run_l2w(l2r)
