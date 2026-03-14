"""Tests for CorrectMixin.to_l1r() and to_l2r().

to_l1r  — requires only the bundled ACI13 L1C fixture (no network).
to_l2r  — requires ancillary data (skipped when MERRA-2 unavailable).
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.conftest import TMP_DIR


# --------------------------------------------------------------------------- #
# to_l1r                                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def l1r_zarr(aci13_l1c, tmp_path_factory):
    """Run to_l1r once for the module; return (zarr_path, Image)."""
    out = str(tmp_path_factory.mktemp("l1r") / "scene_l1r.zarr")
    img = aci13_l1c.to_l1r(out)
    return out, img


def test_to_l1r_returns_image(l1r_zarr, aci13_l1c):
    """to_l1r returns an Image of the same concrete type."""
    _, img = l1r_zarr
    assert type(img) is type(aci13_l1c)


def test_to_l1r_processing_level(l1r_zarr):
    _, img = l1r_zarr
    assert img.level == "L1R"
    assert img.in_ds.attrs["processing_level"] == "L1R"


def test_to_l1r_rho_at_sensor_present(l1r_zarr):
    _, img = l1r_zarr
    assert "rho_at_sensor" in img.in_ds


def test_to_l1r_rho_at_sensor_finite(l1r_zarr):
    """Finite pixels must have positive reflectance."""
    _, img = l1r_zarr
    rho = img.in_ds["rho_at_sensor"].values
    finite = rho[np.isfinite(rho)]
    assert finite.size > 0
    assert (finite > 0).all()


def test_to_l1r_rho_at_sensor_shape(l1r_zarr, aci13_l1c):
    _, img = l1r_zarr
    assert img.in_ds["rho_at_sensor"].dims == ("wavelength", "y", "x")
    assert img.in_ds["rho_at_sensor"].shape == aci13_l1c.in_ds["radiance_at_sensor"].shape


def test_to_l1r_mask_water_dtype(l1r_zarr):
    _, img = l1r_zarr
    assert "mask_water" in img.in_ds
    mw = img.in_ds["mask_water"].values
    assert mw.dtype == np.uint8
    assert set(np.unique(mw)).issubset({0, 1})


def test_to_l1r_ndwi_range(l1r_zarr):
    _, img = l1r_zarr
    ndwi = img.in_ds["ndwi"].values
    finite = ndwi[np.isfinite(ndwi)]
    assert ((finite >= -1.0) & (finite <= 1.0)).all()


def test_to_l1r_zarr_written(l1r_zarr):
    """Zarr store directory exists and contains rho_at_sensor."""
    path, _ = l1r_zarr
    import os
    assert os.path.isdir(path)
    ds = xr.open_zarr(path)
    assert "rho_at_sensor" in ds
    assert "spatial_ref" in ds


def test_to_l1r_with_cal_model(aci13_l1c, tmp_path):
    """Applying a gain-1 RatioModel leaves rho_at_sensor unchanged."""
    wl   = aci13_l1c.wavelength
    ones = xr.Dataset(
        {"gain": ("wavelength", np.ones(len(wl), dtype=np.float32))},
        coords={"wavelength": wl},
    )

    class _FakeRatioModel:
        model_name = "ratio"
        coeffs = ones

        def to_dict(self):
            return {"model_name": "ratio"}

    out = str(tmp_path / "l1r_cal.zarr")
    img = aci13_l1c.to_l1r(out, cal_model=_FakeRatioModel())
    assert "rho_at_sensor" in img.in_ds
    assert img.in_ds.attrs.get("cal_model") is not None


# --------------------------------------------------------------------------- #
# to_l2r                                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def l2r_zarr(aci13_l1c_anc):
    """Run to_l2r once; save results to tests/data/tmp/ for manual inspection.

    Outputs:
        tests/data/tmp/ACI13_bbox_l2r.zarr   — primary Zarr store
        tests/data/tmp/ACI13_bbox_l2r.nc     — CF NetCDF copy

    Clips to LUT wavelength range [361.51, 990] nm before processing
    (the sensor extends to 991 nm but the bundled LUT tops out at 990 nm).
    """
    import shutil

    zarr_out = str(TMP_DIR / "ACI13_bbox_l2r.zarr")
    nc_out   = str(TMP_DIR / "ACI13_bbox_l2r.nc")

    # Remove stale outputs so a fresh run always overwrites them.
    if (TMP_DIR / "ACI13_bbox_l2r.zarr").exists():
        shutil.rmtree(zarr_out)
    if (TMP_DIR / "ACI13_bbox_l2r.nc").exists():
        (TMP_DIR / "ACI13_bbox_l2r.nc").unlink()

    # Clip to LUT range (idempotent if already clipped by aci13_l1r_anc)
    aci13_l1c_anc.mask_wavelength([361.51, 990.0])
    # n_workers=1 keeps the test suite from spawning many processes
    img = aci13_l1c_anc.to_l2r(zarr_out, n_workers=1, window_size=64)

    # Write the CF NetCDF copy for easy inspection in QGIS / Python
    img.zarr_to_nc(zarr_out, nc_out)

    return zarr_out, img


def test_to_l2r_returns_image(l2r_zarr, aci13_l1c_anc):
    _, img = l2r_zarr
    assert type(img) is type(aci13_l1c_anc)


def test_to_l2r_processing_level(l2r_zarr):
    _, img = l2r_zarr
    assert img.level == "L2R"
    assert img.in_ds.attrs["processing_level"] == "L2R"


def test_to_l2r_output_variables(l2r_zarr):
    """All expected output variables are present."""
    _, img = l2r_zarr
    for var in ("rho_at_sensor", "rho_s", "rho_w", "rho_w_gl21",
                "ndwi", "mask_water"):
        assert var in img.in_ds, f"Missing variable: {var}"


def test_to_l2r_rho_w_nan_on_land(l2r_zarr):
    """rho_w must be NaN wherever mask_water == 0."""
    _, img = l2r_zarr
    mw   = img.in_ds["mask_water"].values.astype(bool)   # (y, x)
    rho_w = img.in_ds["rho_w"].values                    # (n_wl, y, x)
    land  = ~mw                                           # (y, x)
    if land.any():
        assert np.isnan(rho_w[:, land]).all(), \
            "rho_w should be NaN on all non-water pixels"


def test_to_l2r_zarr_written(l2r_zarr):
    path, _ = l2r_zarr
    import os
    assert os.path.isdir(path)
    ds = xr.open_zarr(path)
    assert "spatial_ref" in ds
    assert ds.attrs.get("processing_level") == "L2R"


def test_l2r_artifacts_saved_to_tmp(l2r_zarr):
    """Zarr store and NetCDF copy are both present in tests/data/tmp/."""
    import os
    assert os.path.isdir(TMP_DIR / "ACI13_bbox_l2r.zarr"), \
        "Zarr store not found in tests/data/tmp/"
    assert os.path.isfile(TMP_DIR / "ACI13_bbox_l2r.nc"), \
        "NetCDF copy not found in tests/data/tmp/"

    ds = xr.open_dataset(TMP_DIR / "ACI13_bbox_l2r.nc")
    assert "rho_w" in ds
    assert ds.attrs.get("processing_level") == "L2R"
