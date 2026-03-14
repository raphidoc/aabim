"""Pytest fixtures shared across all test modules.

The image fixture (tests/data/ACI13_bbox_l1c.nc) and insitu_reflectance.csv
are generated once by running ``python tests/make_fixtures.py`` locally and
committed to the repo.

Generated test artefacts (CalibrationData, fitted models, etc.) are saved to
tests/data/tmp/ so they can be inspected manually after the test run.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aabim.calibration import CalibrationData, InSitu
from aabim.image import Image

DATA_DIR = Path(__file__).parent / "data"
TMP_DIR  = DATA_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Solar irradiance spectrum bundled with the package
_F0_PATH = Path(__file__).parent.parent / (
    "aabim/data/solar_irradiance/"
    "hybrid_reference_spectrum_c2022-11-30_with_unc.nc"
)

# AOD variable expected by CalibrationData.compute() / AerLUT
_AOD_VAR_CODE = "aerosol_optical_depth_at_550_nm"

# Synthetic ancillary values injected into aci13_l1r so unit tests run
# without network access.  Values are typical for this scene
# (Quebec coast, July 2022).
_SYNTHETIC_ANC = {
    _AOD_VAR_CODE: 0.10,          # aerosol optical depth at 550 nm
    "surface_air_pressure": 1013.0,                              # hPa
    "atmosphere_mass_content_of_water_vapor": 2.5,               # g cm-2
    "equivalent_thickness_at_stp_of_atmosphere_ozone_content": 0.35,  # cm-atm
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"Test fixture not found: {path}\n"
            "Run  python tests/make_fixtures.py  to generate it."
        )


def _compute_rho_at_sensor(img: Image) -> xr.DataArray:
    """Convert calibrated radiance to TOA reflectance.

    rho_t = π × L(λ)  /  [F0(λ) × cos(θ_sun)]

    Units
    -----
    L   : uW cm-2 nm-1 sr-1   (radiance_at_sensor)
    F0  : W m-2 nm-1 → *100 → uW cm-2 nm-1
    """
    f0_ds = xr.open_dataset(_F0_PATH)
    f0_wl  = f0_ds["Vacuum Wavelength"].values  # nm
    f0_ssi = f0_ds["SSI"].values                # W m-2 nm-1

    doy = img.acq_time_z.timetuple().tm_yday
    dist_factor = 1 + 0.034 * np.cos(2 * np.pi * (doy - 2) / 365.0)
    f0_corrected = f0_ssi * dist_factor  # W m-2 nm-1

    # Interpolate to sensor wavelengths; convert to uW cm-2 nm-1
    f0_sensor = np.interp(img.wavelength, f0_wl, f0_corrected) * 100.0

    f0_da  = xr.DataArray(f0_sensor, dims=["wavelength"],
                          coords={"wavelength": img.wavelength})
    cos_sz = np.cos(np.deg2rad(img.in_ds["sun_zenith"]))  # (y, x)

    rho_t = (math.pi * img.in_ds["radiance_at_sensor"]) / (f0_da * cos_sz)
    return rho_t.assign_attrs(
        units="1",
        long_name="top_of_atmosphere_reflectance",
    )


# --------------------------------------------------------------------------- #
# Image fixtures                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def aci13_l1c() -> Image:
    """Tiny cropped ACI13 L1C image (full wavelength range, raw radiance)."""
    _skip_if_missing(DATA_DIR / "ACI13_bbox_l1c.nc")
    return Image.from_aabim_nc(str(DATA_DIR / "ACI13_bbox_l1c.nc"))


@pytest.fixture(scope="session")
def aci13_lut() -> Image:
    """ACI13 image clipped to LUT wavelength range [361.51, 990] nm.

    Use for LUT tests to avoid out-of-range interpolation errors.
    The sensor extends to 991 nm while the bundled LUT tops out at 990 nm.
    """
    _skip_if_missing(DATA_DIR / "ACI13_bbox_l1c.nc")
    img = Image.from_aabim_nc(str(DATA_DIR / "ACI13_bbox_l1c.nc"))
    img.mask_wavelength([361.51, 990.0])
    return img


@pytest.fixture(scope="session")
def aci13_l1r(aci13_lut: Image) -> Image:
    """ACI13 image with rho_at_sensor and synthetic ancillary data.

    Derived from aci13_lut (LUT-compatible wavelengths).  Used for
    CalibrationData.compute() unit tests (no network required).

    Notes
    -----
    * rho_at_sensor is computed from radiance_at_sensor using the bundled
      solar irradiance spectrum (simple linear interpolation, no RSR convolution).
    * Ancillary variables are synthetic but physically plausible values
      representative of this scene.  Use aci13_l1r_anc for tests that require
      real MERRA-2 data.
    """
    rho_t = _compute_rho_at_sensor(aci13_lut)
    aci13_lut.in_ds = aci13_lut.in_ds.assign(
        rho_at_sensor=rho_t,
        **_SYNTHETIC_ANC,
    )
    return aci13_lut


# --------------------------------------------------------------------------- #
# Ancillary-enriched image fixtures                                            #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def aci13_l1c_anc() -> Image:
    """Copy of ACI13 fixture with MERRA-2 ancillary data added.

    The copy is written to tests/data/tmp/ so the original fixture is never
    modified.  If MERRA-2 files are already present from a previous run,
    no download is performed.  Skips if the download fails (no network).
    """
    _skip_if_missing(DATA_DIR / "ACI13_bbox_l1c.nc")
    from aabim.ancillary.get_ancillary import add_ancillary

    copy_path = TMP_DIR / "ACI13_bbox_l1c_anc.nc"
    shutil.copy2(DATA_DIR / "ACI13_bbox_l1c.nc", copy_path)
    img = Image.from_aabim_nc(str(copy_path))
    try:
        add_ancillary(img, anc_dir=TMP_DIR / "anc")
    except Exception as exc:
        pytest.skip(f"Could not fetch MERRA-2 ancillary data: {exc}")
    return img


@pytest.fixture(scope="session")
def aci13_l1r_anc(aci13_l1c_anc: Image) -> Image:
    """ACI13 with real ancillary + rho_at_sensor, clipped to LUT wavelengths.

    Rrs from in-situ is already converted to irradiance reflectance (rho = π × Rrs)
    in the insitu fixture.  Here we compute rho_at_sensor for the image side:
        rho_t = π × L(λ) / (F0(λ) × cos(θ_sun))
    """
    aci13_l1c_anc.mask_wavelength([361.51, 990.0])
    rho_t = _compute_rho_at_sensor(aci13_l1c_anc)
    aci13_l1c_anc.in_ds = aci13_l1c_anc.in_ds.assign(rho_at_sensor=rho_t)
    return aci13_l1c_anc


# --------------------------------------------------------------------------- #
# In-situ fixture                                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def insitu() -> InSitu:
    """Real in-situ ρ_w from tests/data/insitu_reflectance.csv.

    Column mapping applied:
      uuid_l2       → uuid
      rrs_estimate  → rho  (= π × Rrs)
    """
    _skip_if_missing(DATA_DIR / "insitu_reflectance.csv")
    df = pd.read_csv(DATA_DIR / "insitu_reflectance.csv")
    df = df.rename(columns={"uuid_l2": "uuid"})
    df["rho"] = df["rrs_estimate"] * math.pi
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    return InSitu(df[["uuid", "date_time", "lat", "lon", "wavelength", "rho"]])


# --------------------------------------------------------------------------- #
# CalibrationData fixture                                                      #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def cal_data(aci13_l1r: Image, insitu: InSitu) -> CalibrationData:
    """Real CalibrationData computed from ACI13 image + in-situ CSV.

    Saved to tests/data/tmp/cal_data.nc for manual inspection.
    """
    cd = CalibrationData.compute(aci13_l1r, insitu)
    cd.save(str(TMP_DIR / "cal_data.nc"))
    return cd
