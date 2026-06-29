"""Tests for aabim.data.atmospheric (AerLUT, GasLUT, gen_lut)."""

import shutil

import numpy as np
import pytest
import xarray as xr

from aabim.data.atmospheric import AerLUT, GasLUT


# ---------------------------------------------------------------------------
# LUT interpolation tests
# ---------------------------------------------------------------------------

def test_aer_lut_from_image(aci13_lut):
    lut = AerLUT.from_image(aci13_lut)
    assert len(lut.wavelength) == len(aci13_lut.wavelength)
    np.testing.assert_array_equal(lut.wavelength, aci13_lut.wavelength)


def test_gas_lut_from_image(aci13_lut):
    lut = GasLUT.from_image(aci13_lut)
    assert len(lut.wavelength) == len(aci13_lut.wavelength)
    np.testing.assert_array_equal(lut.wavelength, aci13_lut.wavelength)


def test_aer_lut_cache(aci13_lut, tmp_path):
    """Second call with same cache_dir should return identical wavelengths."""
    lut1 = AerLUT.from_image(aci13_lut, cache_dir=tmp_path)
    lut2 = AerLUT.from_image(aci13_lut, cache_dir=tmp_path)
    np.testing.assert_array_equal(lut1.wavelength, lut2.wavelength)


# ---------------------------------------------------------------------------
# gen_lut generation tests — require sixs_json binary
# ---------------------------------------------------------------------------

_SIXS = shutil.which("sixs_json")
_SKIP_SIXS = pytest.mark.skipif(
    _SIXS is None, reason="sixs_json binary not found in PATH"
)

# Minimal grid that exercises every axis with at least 2 points so that
# RegularGridInterpolator can be built and queried in the LUT test.
_MINI_AER = dict(
    sol_zen=[30.0, 40.0],
    view_zen=[0.0, 10.0],
    rel_azi=[0.0, 90.0, 180.0],
    aot550=[0.05, 0.20],
    pressure=[750.0, 1013.0],
    alt_m=[3000.0, 4000.0],
    wl_nm=[450.0, 550.0, 650.0],
)

_MINI_GAS = dict(
    sol_zen=[30.0, 40.0],
    view_zen=[0.0, 10.0],
    rel_azi=[0.0, 90.0, 180.0],
    water=[1.0, 2.5],
    ozone=[0.25, 0.35],
    pressure=[750.0, 1013.0],
    alt_m=[3000.0, 4000.0],
    wl_nm=[450.0, 550.0, 650.0],
)


@_SKIP_SIXS
def test_generate_aerosol_lut(tmp_path):
    out = AerLUT.generate(
        output_path=tmp_path / "lut_aerosol_mini.nc",
        n_workers=4,
        complevel=0,
        **_MINI_AER,
    )

    assert out.exists()
    ds = xr.open_dataset(out)

    # Shape check
    expected_vars = [
        "atmospheric_reflectance_at_sensor",
        "total_scattering_trans_total",
        "spherical_albedo_total",
        "sky_glint_total",
    ]
    for v in expected_vars:
        assert v in ds, f"missing variable {v}"

    assert ds.sizes["sol_zen"]          == len(_MINI_AER["sol_zen"])
    assert ds.sizes["relative_azimuth"] == len(_MINI_AER["rel_azi"])
    assert ds.sizes["wavelength"]       == len(_MINI_AER["wl_nm"])

    # No NaN values
    for v in expected_vars:
        assert not np.isnan(ds[v].values).any(), f"NaN in {v}"

    # Physical bounds
    assert (ds["total_scattering_trans_total"].values >= 0).all()
    assert (ds["total_scattering_trans_total"].values <= 1).all()
    assert (ds["atmospheric_reflectance_at_sensor"].values >= 0).all()
    # sky_glint is a small correction term — values > 0.05 indicate a bug
    # (wrong refractive index units, etc.)
    assert (ds["sky_glint_total"].values <= 0.05).all(), \
        f"sky_glint_total max={ds['sky_glint_total'].values.max():.4f} > 0.05 — check _water_refractive_index units"

    # Spot-check against the bundled LUT at a matching geometry point.
    # These values come from lut_aerosol.nc and must match to 3 significant figures.
    s = ds.sel(sol_zen=30, view_zen=0, relative_azimuth=90,
               aot550=0.05, target_pressure=1013, sensor_altitude=3000)
    rho  = float(s["atmospheric_reflectance_at_sensor"].sel(wavelength=450))
    t    = float(s["total_scattering_trans_total"].sel(wavelength=550))
    glint = float(s["sky_glint_total"].sel(wavelength=450))
    assert rho   == pytest.approx(0.02961, rel=0.01), f"rho_path@450nm={rho}"
    assert t     == pytest.approx(0.92518, rel=0.01), f"t_total@550nm={t}"
    assert glint == pytest.approx(0.00389, rel=0.05), f"sky_glint@450nm={glint}"

    # Coordinate coverage: relative_azimuth must span 0–180
    assert float(ds.relative_azimuth.min()) == pytest.approx(0.0)
    assert float(ds.relative_azimuth.max()) == pytest.approx(180.0)

    ds.close()


@_SKIP_SIXS
def test_generate_gas_lut(tmp_path):
    out = GasLUT.generate(
        output_path=tmp_path / "lut_gas_mini.nc",
        n_workers=4,
        complevel=0,
        **_MINI_GAS,
    )

    assert out.exists()
    ds = xr.open_dataset(out)

    assert "global_gas_trans_total" in ds
    assert ds.sizes["sol_zen"]          == len(_MINI_GAS["sol_zen"])
    assert ds.sizes["relative_azimuth"] == len(_MINI_GAS["rel_azi"])
    assert ds.sizes["wavelength"]       == len(_MINI_GAS["wl_nm"])

    tgas = ds["global_gas_trans_total"].values
    assert not np.isnan(tgas).any(), "NaN in global_gas_trans_total"
    assert (tgas >= 0).all() and (tgas <= 1).all(), "t_gas outside [0, 1]"

    assert float(ds.relative_azimuth.min()) == pytest.approx(0.0)
    assert float(ds.relative_azimuth.max()) == pytest.approx(180.0)

    ds.close()
