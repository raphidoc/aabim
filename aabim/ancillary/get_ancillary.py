"""
get_ancillary.py — Add MERRA-2 scalar ancillary variables to an aabim image.

Pipeline
--------
1. Download two hourly MERRA-2 MET + AER files bracketing the image
   acquisition time (files are cached locally in *anc_dir*).
2. Bilinearly interpolate each field to the image centre lat/lon and time.
3. Derive physical quantities and write them as CF-compliant **scalar**
   variables to the image NetCDF in append mode, then mirror the new
   variables into ``image.in_ds`` for immediate downstream use.

Ancillary variables written
---------------------------
Variable name                                           | MERRA-2 field | Conversion
aerosol_optical_depth_at_550_nm                         | TOTEXTTAU     | —
wind_speed                                              | U10M, V10M    | √(U²+V²)
wind_direction                                          | U10M, V10M    | atan2(U,V) → °
surface_air_pressure                                    | PS            | Pa → hPa
atmosphere_mass_content_of_water_vapor                  | TQV           | kg m⁻² → g cm⁻²
equivalent_thickness_at_stp_of_atmosphere_ozone_content | TO3           | DU → cm-atm

Design note
-----------
Storing ancillary data as scalar variables (one value per image) is the
right approach for airborne hyperspectral data: the MERRA-2 grid is coarse
(~0.5°×0.625°) relative to a typical flightline, so a single spatially
interpolated value per image is appropriate.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import netCDF4 as nc

from aabim.image import Image
from aabim.ancillary import obpg

log = logging.getLogger(__name__)

# Variables expected to be present after a successful add_ancillary() call.
ANCILLARY_VARS: list[str] = [
    "aerosol_optical_depth_at_550_nm",
    "wind_speed",
    "wind_direction",
    "surface_air_pressure",
    "atmosphere_mass_content_of_water_vapor",
    "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
]

# MERRA-2 fields required from the downloaded files.
_REQUIRED_FIELDS = ["U10M", "V10M", "PS", "TQV", "TO3", "TOTEXTTAU"]


def add_ancillary(image: Image, anc_dir: str | Path | None = None) -> None:
    """Download MERRA-2 and write scalar ancillary variables to *image*.

    Modifies ``image.in_path`` on disk (append mode) **and** updates
    ``image.in_ds`` in memory so the image is ready for immediate use
    without reloading.

    If all variables in :data:`ANCILLARY_VARS` are already present the
    function returns early without downloading anything.

    Parameters
    ----------
    image : Image
    anc_dir : path, optional
        Directory for caching downloaded MERRA-2 files.
        Defaults to ``<image_dir>/anc/``.
    """
    if all(v in image.in_ds for v in ANCILLARY_VARS):
        log.info("Ancillary data already present — skipping download.")
        return

    anc_dir = Path(anc_dir) if anc_dir else Path(image.in_path).parent / "anc"
    anc = obpg.get_gmao(image, anc_dir=anc_dir)
    _check_fields(anc)

    scalars = _build_scalars(anc)

    # HDF5 does not allow concurrent read+write access: close the xarray
    # dataset handle before opening the file in append mode, then reopen.
    import xarray as xr
    image.in_ds.close()
    _write_to_nc(image.in_path, scalars)
    image.in_ds = xr.open_dataset(image.in_path)
    log.info("Ancillary data written to %s", image.in_path)


# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #

def _check_fields(anc) -> None:
    """Raise ValueError if any required MERRA-2 field is NaN."""
    missing = [f for f in _REQUIRED_FIELDS if math.isnan(float(anc[f]))]
    if missing:
        raise ValueError(f"Missing or NaN MERRA-2 fields: {missing}")


def _build_scalars(anc) -> dict:
    """Derive physical quantities from raw MERRA-2 fields."""
    u10m = float(anc["U10M"])
    v10m = float(anc["V10M"])
    return {
        "aerosol_optical_depth_at_550_nm": {
            "value":         float(anc["TOTEXTTAU"]),
            "standard_name": "atmosphere_optical_thickness_due_to_aerosol",
            "units":         "1",
            "long_name":     "Aerosol optical depth at 550 nm",
        },
        "wind_speed": {
            "value":         math.sqrt(u10m**2 + v10m**2),
            "standard_name": "wind_speed",
            "units":         "m s-1",
            "long_name":     "Wind speed at 10 m",
        },
        "wind_direction": {
            "value":         math.degrees(math.atan2(u10m, v10m)),
            "standard_name": "wind_from_direction",
            "units":         "degree",
            "long_name":     "Wind direction at 10 m (from)",
        },
        "surface_air_pressure": {
            "value":         float(anc["PS"]) / 100.0,  # Pa → hPa
            "standard_name": "surface_air_pressure",
            "units":         "hPa",
            "long_name":     "Surface air pressure",
        },
        "atmosphere_mass_content_of_water_vapor": {
            "value":         float(anc["TQV"]) / 10.0,  # kg m-2 → g cm-2
            "standard_name": "atmosphere_mass_content_of_water_vapor",
            "units":         "g cm-2",
            "long_name":     "Atmosphere mass content of water vapor",
        },
        "equivalent_thickness_at_stp_of_atmosphere_ozone_content": {
            "value":         float(anc["TO3"]) / 1000.0,  # DU → cm-atm
            "standard_name": "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
            "units":         "cm-atm",
            "long_name":     "Equivalent thickness at STP of atmospheric ozone",
        },
    }


def _write_to_nc(nc_path: str | Path, scalars: dict) -> None:
    """Append scalar variables to an existing NetCDF file."""
    with nc.Dataset(str(nc_path), "a", format="NETCDF4") as ds:
        for name, meta in scalars.items():
            if name in ds.variables:
                log.debug("Variable %s already in file — skipping.", name)
                continue
            var                = ds.createVariable(name, "f4")
            var.standard_name  = meta["standard_name"]
            var.units          = meta["units"]
            var.long_name      = meta["long_name"]
            var.source         = "GMAO MERRA-2"
            var[:]             = meta["value"]
