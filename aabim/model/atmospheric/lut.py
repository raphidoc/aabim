import xarray as xr
import numpy as np
import scipy.interpolate as sp

def load_aer():
    # lut_aer = xr.open_dataset(
    #     "/home/raphael/PycharmProjects/reverie/reverie/data/lut/lut_aerosol.nc"
    # )

    # lut_aer["wavelength"] = lut_aer["wavelength"].values * 1e3
    lut_aer = xr.open_dataset(
        "/home/raphael/PycharmProjects/reverie/reverie/data/lut/aer_lut_WISE.nc"
    )
    lut_aer["sensor_altitude"] = lut_aer["sensor_altitude"].values * -1e3

    return lut_aer


def load_gas():
    # lut_gas = xr.open_dataset(
    #     "/home/raphael/PycharmProjects/reverie/reverie/data/lut/lut_gas.nc"
    # )

    # lut_gas["wavelength"] = lut_gas["wavelength"].values * 1e3
    lut_gas = xr.open_dataset(
        "/home/raphael/PycharmProjects/reverie/reverie/data/lut/gas_lut_WISE.nc"
    )
    lut_gas["sensor_altitude"] = lut_gas["sensor_altitude"].values * -1e3

    return lut_gas

def slice_lut_to_wavelengths(ds, wl_target):
    wl_target = np.asarray(wl_target, dtype=np.float32)

    # nearest-index selection (safe if grids match closely)
    ds2 = ds.sel(wavelength=wl_target, method="nearest")

    # sanity: ensure we actually matched what you think
    max_err = np.max(np.abs(ds2.wavelength.values.astype(np.float32) - wl_target))
    if max_err > 0.5:   # adjust tolerance to your instrument band spacing
        raise ValueError(f"LUT wavelength mismatch: max |Δλ| = {max_err} nm")

    return ds2

def get_t_gas(wavelength, sol_zen, view_zen, relative_azimuth, water, ozone, target_pressure, sensor_altitude):
    lut = load_gas()

    t_gas = lut["global_gas_trans_total"]

    lut_points = (
        lut.sol_zen.values,
        lut.view_zen.values,
        lut.relative_azimuth.values,
        lut.water.values,
        lut.ozone.values,
        lut.target_pressure.values,
        lut.sensor_altitude.values,
        lut.wavelength.values,
    )

    # n_wavelength = len(wavelength)

    # xi = np.hstack([
    #     np.full((n_wavelength, 1), sol_zen),
    #     np.full((n_wavelength, 1), view_zen),
    #     np.full((n_wavelength, 1), relative_azimuth),
    #     np.full((n_wavelength, 1), water),
    #     np.full((n_wavelength, 1), ozone),
    #     np.full((n_wavelength, 1), target_pressure),
    #     np.full((n_wavelength, 1), sensor_altitude),
    #     wavelength.reshape(-1, 1),
    # ])
    #
    # t_gas = sp.interpn(
    #     points=lut_points,
    #     # values=lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
    #     values=lut["global_gas_trans_total"][:, :, :, :, :, :, :].values,
    #     xi=xi,
    # )

    xi = np.hstack([
        sol_zen,
        view_zen,
        relative_azimuth,
        water,
        ozone,
        target_pressure,
        sensor_altitude,
        wavelength,
    ])

    t_gas = sp.interpn(
        points=lut_points,
        # values=lut_aer["atmospheric_reflectance_at_sensor"][:, :, :, :, :, :, :].values
        values=lut["global_gas_trans_total"][:, :, :, :, :, :, :, :].values,
        xi=xi,
    )

    return t_gas

if __name__ == "__main__":
    get_t_gas(
        350,
    35,
    15,
    83,
    1,
    0.2,
    1007,
    3004)
