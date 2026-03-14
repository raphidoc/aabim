"""Tests for IOMixin.to_zarr() — CF grid_mapping / GDAL compatibility."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.conftest import TMP_DIR


# --------------------------------------------------------------------------- #
# to_zarr                                                                      #
# --------------------------------------------------------------------------- #


def test_to_zarr_creates_store(aci13_l1c, tmp_path):
    """to_zarr() creates a directory store on disk."""
    out = str(tmp_path / "scene.zarr")
    aci13_l1c.to_zarr(out)
    import os
    assert os.path.isdir(out)


def test_to_zarr_spatial_ref_variable(aci13_l1c, tmp_path):
    """spatial_ref scalar variable is present with required GDAL attributes."""
    out = str(tmp_path / "scene.zarr")
    aci13_l1c.to_zarr(out)

    ds = xr.open_zarr(out)
    assert "spatial_ref" in ds, "spatial_ref variable missing"
    attrs = ds["spatial_ref"].attrs
    assert "crs_wkt"      in attrs, "crs_wkt attribute missing"
    assert "spatial_ref"  in attrs, "spatial_ref (GDAL alias) attribute missing"
    assert "GeoTransform" in attrs, "GeoTransform attribute missing"


def test_to_zarr_geotransform_format(aci13_l1c, tmp_path):
    """GeoTransform is a space-separated string of 6 numbers."""
    out = str(tmp_path / "scene.zarr")
    aci13_l1c.to_zarr(out)

    ds = xr.open_zarr(out)
    gt = ds["spatial_ref"].attrs["GeoTransform"]
    parts = gt.split()
    assert len(parts) == 6
    # All parts must be parseable as floats
    floats = [float(p) for p in parts]
    # pixel width (index 1) must be positive
    assert floats[1] > 0
    # pixel height (index 5) must be negative for a north-up image
    assert floats[5] < 0


def test_to_zarr_grid_mapping_attribute(aci13_l1c, tmp_path):
    """Spatially-explicit variables carry grid_mapping: 'spatial_ref'."""
    out = str(tmp_path / "scene.zarr")
    aci13_l1c.to_zarr(out)

    ds = xr.open_zarr(out)
    for name, var in ds.data_vars.items():
        if name == "spatial_ref":
            continue
        if "y" in var.dims and "x" in var.dims:
            assert var.attrs.get("grid_mapping") == "spatial_ref", (
                f"Variable '{name}' is missing grid_mapping attribute"
            )


def test_to_zarr_roundtrip_crs(aci13_l1c, tmp_path):
    """CRS survives a write-read roundtrip (WKT comparison)."""
    import pyproj

    out = str(tmp_path / "scene.zarr")
    aci13_l1c.to_zarr(out)

    ds = xr.open_zarr(out)
    crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]
    crs_rt = pyproj.CRS.from_wkt(crs_wkt)
    assert crs_rt.equals(aci13_l1c.CRS)


def test_to_zarr_cf_convention(aci13_l1c, tmp_path):
    """Global Conventions attribute is CF-1.0."""
    out = str(tmp_path / "scene.zarr")
    aci13_l1c.to_zarr(out)

    ds = xr.open_zarr(out)
    assert ds.attrs.get("Conventions") == "CF-1.0"
