import numpy as np
import xarray as xr

from reverie import ReveCube

AER_VARS = [
    "atmospheric_reflectance_at_sensor",
    "total_scattering_trans_total",
    "spherical_albedo_total",
    "sky_glint_total",
]

GAS_VARS = [
    "global_gas_trans_total",
]

def _to_monotonic(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if np.any(np.diff(arr) < 0):
        arr = arr[::-1]
        flip = True
    else:
        flip = False
    return arr, flip

def resample_lut_to_wise(aer_lut: xr.Dataset,
                         gas_lut: xr.Dataset,
                         wise_wavelength_nm: np.ndarray,
                         lut_wavelength_unit: str = "nm"):
    """
    Returns (aer_lut_wise, gas_lut_wise) with wavelength resampled to WISE grid.
    lut_wavelength_unit: "nm" or "um" for the LUT wavelength coordinate.
    """

    wise_wl = np.asarray(wise_wavelength_nm, dtype=np.float32)

    # Convert WISE wl to LUT units if needed
    if lut_wavelength_unit == "um":
        wise_wl_in_lut_units = wise_wl * 1e-3
    elif lut_wavelength_unit == "nm":
        wise_wl_in_lut_units = wise_wl
    else:
        raise ValueError("lut_wavelength_unit must be 'nm' or 'um'")

    # Ensure LUT wavelength monotonic increasing (xarray interp assumes sorted)
    aer_wl, aer_flip = _to_monotonic(aer_lut["wavelength"].values)
    gas_wl, gas_flip = _to_monotonic(gas_lut["wavelength"].values)

    if aer_flip:
        aer_lut = aer_lut.isel(wavelength=slice(None, None, -1))
    if gas_flip:
        gas_lut = gas_lut.isel(wavelength=slice(None, None, -1))

    # Sanity: require WISE wl inside LUT range (or allow extrapolate explicitly)
    wmin_a, wmax_a = float(aer_lut.wavelength.min()), float(aer_lut.wavelength.max())
    wmin_g, wmax_g = float(gas_lut.wavelength.min()), float(gas_lut.wavelength.max())
    if wise_wl_in_lut_units.min() < wmin_a or wise_wl_in_lut_units.max() > wmax_a:
        raise ValueError(f"WISE wl outside aerosol LUT range [{wmin_a}, {wmax_a}] in LUT units")
    if wise_wl_in_lut_units.min() < wmin_g or wise_wl_in_lut_units.max() > wmax_g:
        raise ValueError(f"WISE wl outside gas LUT range [{wmin_g}, {wmax_g}] in LUT units")

    # Interpolate only wavelength
    aer_lut_wise = aer_lut[AER_VARS].interp(
        wavelength=wise_wl_in_lut_units,
        method="linear",
    )

    gas_lut_wise = gas_lut[GAS_VARS].interp(
        wavelength=wise_wl_in_lut_units,
        method="linear",
    )

    # Set coord back to WISE nm for downstream code clarity
    aer_lut_wise = aer_lut_wise.assign_coords(wavelength=("wavelength", wise_wl))
    gas_lut_wise = gas_lut_wise.assign_coords(wavelength=("wavelength", wise_wl))

    # Keep other coords/attrs if you want
    aer_lut_wise.attrs.update(aer_lut.attrs)
    gas_lut_wise.attrs.update(gas_lut.attrs)

    return aer_lut_wise, gas_lut_wise

if __name__ == "__main__":
    l1 = ReveCube.from_reve_nc("/D/Data/WISE/ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l1r.nc")
    aer_lut = xr.open_dataset("lut_aerosol.nc")
    gas_lut = xr.open_dataset("lut_gas.nc")

    wise_wl_nm = l1.wavelength.astype(np.float32)

    aer_wise, gas_wise = resample_lut_to_wise(aer_lut, gas_lut, wise_wl_nm, lut_wavelength_unit="um")

    aer_wise.to_netcdf("aer_lut_WISE.nc")
    gas_wise.to_netcdf("gas_lut_WISE.nc")