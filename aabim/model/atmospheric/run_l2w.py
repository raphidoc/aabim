import logging
import math
import os
import shutil
import time

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import re
import numpy as np
import xarray as xr
import glob
from functools import partial
import netCDF4

from reverie import ReveCube
from reverie.correction.surface.rayleigh import fresnel_reflectance
from reverie.correction.surface.water_refractive_index import get_water_refractive_index

# def process_window_l2w(window, l2s_path, output_dir, aod555):
#     # Load input cube inside the process (avoid sharing in memory)
#     l2s = ReveCube.from_reve_nc(l2s_path)
#
#     # Extract slices
#     y_slice = slice(int(window.row_off), int(window.row_off + window.height))
#     x_slice = slice(int(window.col_off), int(window.col_off + window.width))
#
#     # mask_water = l2r.in_ds.isel(
#     #     x=x_slice,
#     #     y=y_slice
#     # )["mask_water"].astype("bool")
#
#     image_sub = l2s.in_ds.isel(y=y_slice, x=x_slice)
#     wavelength = l2s.wavelength
#     rho_s = image_sub["rho_surface"]
#
#     # glint = get_glint_z17(image_sub, wavelength, aod555)
#     # rho_w = rho_s - glint["rho_surface_sky"]
#
#     theta_0 = np.deg2rad(image_sub["sun_zenith"])
#     theta_v = np.deg2rad(image_sub["view_zenith"])
#     phi_0 = np.deg2rad(image_sub["sun_azimuth"])
#     phi_v = np.deg2rad(image_sub["view_azimuth"])
#
#     sky_glint = get_sky_glint(wavelength, theta_0, theta_v, phi_0, phi_v, 1007)
#     rho_w = rho_s - sky_glint
#
#     # Save this window to a temporary NetCDF
#     y_coords = image_sub["y"]
#     x_coords = image_sub["x"]
#     wl_coords = image_sub.wavelength
#
#     out_path = os.path.join(
#         output_dir,
#         f"rho_w_window_{window.row_off}-{window.row_off+window.height}_{window.col_off}-{window.col_off+window.width}.nc")
#     da = xr.DataArray(
#         rho_w,
#         dims=["wavelength", "y", "x"],
#         coords={
#             "wavelength": wl_coords,
#             "y": y_coords,
#             "x": x_coords,
#         },
#         name = "rho_w"
#     )
#     da.to_netcdf(out_path)
#
#     return 0

def run_l2w(l2r):

    log_dir = os.path.dirname(l2r.in_path)
    log_file = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(l2r.in_path))[0]}_l2w.log")

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
    logging.info("Starting run_l2w processing")

    if "l2rg" in l2r.in_path:
        out_name = re.sub(r'l2rg', 'l2wg', l2r.in_path)
    else:
        out_name = re.sub(r'l2r', 'l2w', l2r.in_path)

    scale_factor = 1e-5 # NetCDF divide by scale when writing data and multiply when reading
    # l2r.create_reve_nc(out_name)


    wavelength = l2r.wavelength
    nir_i = np.abs(wavelength - 800).argmin()
    rho_nir = l2r.in_ds["rho_w"][nir_i, :, :].values

    # Hochberg et al 2003
    # rho_nir_min = np.nanmin(rho_nir)
    # # np.nanpercentile(rho_nir, 0.1)
    # # rho_nir_min = np.nanpercentile(rho_nir[rho_nir >= 0], 0.1)
    # ref_sub = rho_nir - rho_nir_min
    #
    # logging.info("Residual glint h03 rho_w({}) = {}".format(wavelength[nir_i], rho_nir_min))

    # Gao and li 2021
    n_w = get_water_refractive_index(30, 12, wavelength)
    rho_f = fresnel_reflectance(0, n_w)
    rho_f_nir = rho_f[nir_i]
    ref_ratio = rho_nir / rho_f_nir

    logging.info("Residual glint (ref_ratio) g21  rho_f({}) = {}".format(wavelength[nir_i], ref_ratio))

    l2r.in_ds.close()
    l2r.out_ds = netCDF4.Dataset(l2r.in_path, "a", format="NETCDF4")
    #  TODO: add check for existing var and remove if so

    # if "rho_w_h03" not in l2r.out_ds.variables:
    #     l2r.create_var_nc(
    #         name="rho_w_h03",
    #         type="i4",
    #         dims=(
    #             "wavelength",
    #             "y",
    #             "x",
    #         ),
    #         comp="zlib",
    #         complevel=1,
    #         scale=scale_factor,
    #     )

    if "rho_w_g21" not in l2r.out_ds.variables:
        l2r.create_var_nc(
            name="rho_w_g21",
            type="i4",
            dims=(
                "wavelength",
                "y",
                "x",
            ),
            comp="zlib",
            complevel=1,
            scale=scale_factor,
        )

    if l2r.no_data is None:
        l2r.no_data = math.trunc(netCDF4.default_fillvals["i4"] * scale_factor)

    for i, wl in tqdm(enumerate(l2r.wavelength), desc="Residual glint correction"):
        rho_w = l2r.out_ds.variables["rho_w"][i,:,:]

        # rho_w_h03 = rho_w - ref_sub
        rho_w_g21 = rho_w - (rho_f[i] * ref_ratio)

        # np.nan_to_num(rho_w_h03, copy=False, nan=l2r.no_data * scale_factor)
        np.nan_to_num(rho_w_g21, copy=False, nan=l2r.no_data * scale_factor)
        # rho_t = np.round(rho_t).astype("int32")

        # l2r.out_ds.variables["rho_w_h03"][i, :, :] = rho_w_h03
        l2r.out_ds.variables["rho_w_g21"][i, :, :] = rho_w_g21

    # for i, wl in tqdm(enumerate(l2r.wavelength), desc="Residual glint correction"):
    #     rho_w = l2r.in_ds.isel(wavelength=i)["rho_w"]
    #     rho_w_c = rho_w - (rho_nir - residual_glint)
    #
    #     np.nan_to_num(rho_w_c, copy=False, nan=l2r.no_data * scale_factor)
    #     # rho_t = np.round(rho_t).astype("int32")
    #
    #     l2r.out_ds.variables["rho_w_c"][i, :, :] = rho_w_c


    # ρρcc(λλ) = ρρ(λλ) − (ρρ(λλcc) − ρρmin (λλcc))

    # # y_start = int(window.row_off)
    # # y_stop = y_start + int(window.height)
    # # x_start = int(window.col_off)
    # # x_stop = x_start + int(window.width)
    # # # return self.valid_mask[tile.sline : tile.eline, tile.spixl : tile.epixl]
    # # water_mask[y_start: y_stop, x_start: x_stop]
    #
    # wavelength = l2r.wavelength
    #
    # # Apply sky and sun glint correction, write each band in the output dataset as l2w
    # # Divide the image in windows
    # window_size = 500
    #
    # windows = l2r.create_windows(window_size)
    #
    # logging.info(
    #     f"{len(windows)} windows to process at {window_size} pixels"
    # )
    #
    # non_empty_windows = []
    # for window in tqdm(windows, desc="Filtering windows"):
    #     y_slice = slice(int(window.row_off), int(window.row_off + window.height))
    #     x_slice = slice(int(window.col_off), int(window.col_off + window.width))
    #
    #     rho_s = l2r.in_ds.isel(
    #         x=x_slice,
    #         y=y_slice
    #     )["rho_surface"]
    #
    #     mask_water = l2r.in_ds.isel(
    #         x=x_slice,
    #         y=y_slice
    #     )["mask_water"].astype("bool")
    #
    #     if not np.isnan(rho_s).all() and mask_water.any():
    #         non_empty_windows.append(window)
    #     else:
    #         l2r.out_ds.variables["rho_w"][:, y_slice, x_slice] = l2r.no_data * scale_factor
    #
    # logging.info(
    #     f"{len(non_empty_windows)}  non empty windows to process"
    # )
    #
    # output_dir = "temp_windows_"+os.path.splitext(out_name)[0]  # Must exist
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # aod555 = l2r.in_ds["rho_surface"].attrs["aod555_dsf"]
    # aod555 = 0.055
    #
    # process_func = partial(
    #     process_window_l2w,
    #     l2s_path = l2r.in_path,
    #     output_dir = output_dir,
    #     aod555 = aod555,
    # )
    #
    # # Parallel dispatch for all non-empty windows
    # results = process_map(
    #     process_func,
    #     non_empty_windows,
    #     max_workers=4, #os.cpu_count() - 2,
    #     desc="Processing windows in parallel"
    # )
    #
    # # Merge all windows
    # all_files = sorted(glob.glob(f"{output_dir}/rho_w_window_*.nc"))
    # ds = xr.open_mfdataset(all_files, combine="by_coords")
    # # ds.to_netcdf("final_rho_surface.nc")
    #
    # rho_w = ds["rho_w"]
    #
    # mask_water = l2r.in_ds["mask_water"].astype("bool")
    # rho_w = rho_w.where(mask_water, np.nan)
    #
    # rho_w = rho_w.values
    #
    # np.nan_to_num(rho_w, copy=False, nan=l2r.no_data * scale_factor)
    # l2r.out_ds["rho_w"][:, :, :] = rho_w
    #
    l2r.out_ds.close()
    #
    # shutil.rmtree(output_dir)

    return 0

if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-l2rg.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l2r.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-l2r.nc",
        # "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-l2r.nc",
    ]

    for image in images:

        l2r = ReveCube.from_reve_nc(os.path.join(image_dir, image))
        run_l2w(l2r)