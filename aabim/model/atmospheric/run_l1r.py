import os
import math

import numpy as np
import re
# from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from pyproj import Transformer
import pandas as pd

from reverie import ReveCube
from reverie.utils.helper import get_f0
import reverie.utils.helper as helper
from reverie.ancillary.get_ancillary import add_ancillary
import lut

# def compute_rho_t(i, wl, l1c: ReveCube, f0, scale):
#     rho_t = (math.pi * l1c.in_ds.sel(wavelength=wl)["radiance_at_sensor"]) / (
#             f0[i] * np.cos(np.deg2rad(l1c.in_ds["sun_zenith"]))
#     )
#
#     rho_t = rho_t * scale
#
#     # Assign missing value
#     rho_t[rho_t == 0] = l1c.no_data * scale
#
#     l1c.out_ds.variables["radiance_at_sensor"][i, :, :] = i
#
#     return 0

def run_l1r(l1c: ReveCube, bbox = None, wavelength_filter = None, gain = None):
    l1c.filter_bad_band()

    if gain is not None:
        out_name = re.sub(r'l1\w+', 'l1rg', l1c.in_path)
        l1c.mask_wavelength(gain["wavelength"].values)
    else:
        out_name = re.sub(r'l1\w+', 'l1r', l1c.in_path)


    if wavelength_filter is not None:
        l1c.mask_wavelength(wavelength_filter)

    wavelength = l1c.in_ds.wavelength.values

    if bbox is not None:
        transformer = Transformer.from_crs("EPSG:4326", l1c.CRS, always_xy=True)
        x_min, y_min = transformer.transform(bbox["lon"][0], bbox["lat"][0])
        x_max, y_max = transformer.transform(bbox["lon"][1], bbox["lat"][1])

        # projected y is descending from north to south, so slice from max to min
        #l1c.y[0] > l1c.y[-1]

        l1c.in_ds = l1c.in_ds.sel(
            x=slice(x_min, x_max),
            y=slice(y_max, y_min)
        )
        # Necessary after crop to maintain consistency between class attributes and self.in_ds
        l1c.update_attributes()

    # Get extraterrestrial solar irradiance for WISE bands
    doy = l1c.acq_time_z.timetuple().tm_yday
    f0 = get_f0(doy, wavelength)

    scale_factor = 1e-5
    # scale = np.reciprocal(float(scale))

    l1c.create_reve_nc(out_name)
    l1c.create_var_nc(
        name="rho_at_sensor",
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

    if gain is not None:
        l1c.create_var_nc(
            name="rho_at_sensor_calibrated",
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

        l1c.create_var_nc(
            name="rho_at_sensor_delta",
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

    # args_list = [(i, wl, l1c, f0, scale) for i, wl in enumerate(wavelength)]
    # results = process_map(compute_rho_t, args_list, max_workers=os.cpu_count() - 2)

    for i, wl in tqdm(enumerate(wavelength)):

        rho_t = (math.pi * l1c.in_ds.sel(wavelength=wl)["radiance_at_sensor"]) / (
                f0[i] * np.cos(np.deg2rad(l1c.in_ds["sun_zenith"]))
        )

        if gain is not None:
            mask = np.int32(gain["wavelength"]) == np.int32(wl)
            if not mask.any():
                raise ValueError(f"Wavelength {wl} not found in gain table")
            gain_intercept = gain.loc[mask, "intercept_estimate"].values[0]
            gain_coeff = gain.loc[mask, "rho_t_estimate"].values[0]

            rho_t_cal = rho_t * gain_coeff + gain_intercept
            rho_t_delta = rho_t_cal - rho_t

            np.nan_to_num(rho_t_cal, copy=False, nan=l1c.no_data * scale_factor)
            np.nan_to_num(rho_t_delta, copy=False, nan=l1c.no_data * scale_factor)

            l1c.out_ds.variables["rho_at_sensor_calibrated"][i, :, :] = rho_t_cal
            l1c.out_ds.variables["rho_at_sensor_delta"][i, :, :] = rho_t_delta

        # Assign missing value
        np.nan_to_num(rho_t, copy=False, nan=l1c.no_data * scale_factor)
        # rho_t = np.round(rho_t).astype("int32")

        l1c.out_ds.variables["rho_at_sensor"][i, :, :] = rho_t


    # Compute water mask on rho_t to apply sky_glint correction to rho_path
    rho_t = l1c.out_ds["rho_at_sensor"]

    # blue_i = np.abs(wavelength - 400).argmin()
    green_i = np.abs(wavelength - 550).argmin()
    # red_i = np.abs(wavelength - 600).argmin()
    # nir_i = np.abs(wavelength - 850).argmin()
    swir_i = np.abs(wavelength - 2190).argmin()

    # blue = rho_t[blue_i, :, :]
    green = rho_t[green_i, :, :]
    # red = rho_t[red_i, :, :]
    # nir = rho_t[nir_i, :, :]
    swir = rho_t[swir_i, :, :]

    ndwi = (green - swir) / (green + swir)

    # mask_water = ((ndwi > 0) & (ndvi < 0.1)).astype(int)
    valid_mask = l1c.get_valid_mask()
    mask_water = np.full(valid_mask.shape, np.nan, dtype=float)
    mask_water[valid_mask] = (ndwi > 0)[valid_mask]

    l1c.create_var_nc(name="ndwi", type="i4", dims=("y", "x"), comp="zlib", scale=scale_factor)

    np.nan_to_num(ndwi, copy=False, nan=l1c.no_data * scale_factor)
    l1c.out_ds["ndwi"][:, :] = ndwi

    l1c.create_var_nc(name="mask_water", type="u1", dims=("y", "x"), comp="zlib", scale=1)

    mask_water_int = np.full(valid_mask.shape, np.nan, dtype=float)
    mask_water_int[valid_mask] = mask_water[valid_mask].astype(int)
    np.nan_to_num(mask_water_int, copy=False, nan=l1c.no_data)
    l1c.out_ds["mask_water"][:, :] = mask_water_int

    # Create geometric variables
    geom = {
        "sun_azimuth": l1c.in_ds.sun_azimuth,
        "sun_zenith": l1c.in_ds.sun_zenith,
        "view_azimuth": l1c.in_ds.view_azimuth,
        "view_zenith": l1c.in_ds.view_zenith,
        "relative_azimuth": l1c.in_ds.relative_azimuth,
    }

    for var in tqdm(geom, desc="Writing geometry"):
        l1c.create_var_nc(
            name=var,
            type="i4",
            dims=(
                "y",
                "x",
            ),
            scale=scale_factor,
        )
        data = geom[var]

        np.nan_to_num(data, copy=False, nan=l1c.no_data * scale_factor)

        l1c.out_ds.variables[var][:, :] = data

    l1c.out_ds.close()

    add_ancillary(l1c)

    return 0

if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-l1cg.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l1cg.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l1cg.nc",
        # "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-l1cg.nc",
        # "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-l1cg.nc",
        # "MC-11A/190820_MC-11A-WI-1x1x1_v02-l1cg.nc"
    ]

    lut_aer = lut.load_aer()
    wavelength_lut = lut_aer["wavelength"].values

    bbox = {"lon": ( -64.44237, -64.23789), "lat": (49.7126, 49.84078)}
    # bbox = None
    # Reduced bbox for dev ACI-13A
    # bbox = {"lon": (-64.36808, -64.35322), "lat": (49.80347, 49.81397)}

    gain_path = "/D/Documents/phd/thesis/3_chapter/data/wise/viccal/gain_el_sma.csv"
    gain = pd.read_csv(gain_path)

    wavelength_gain = gain["wavelength"].values

    for image in images:
        l1c = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        run_l1r(l1c, bbox, wavelength_gain, gain=gain)
        # run_l1r(l1c, bbox, wavelength_lut, gain=None)
