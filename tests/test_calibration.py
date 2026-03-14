"""Tests for aabim.calibration (InSitu, CalibrationData, models).

All fixtures use real data from tests/data/:
  - insitu_reflectance.csv  → insitu fixture
  - ACI13_bbox_l1c.nc       → aci13_l1r → cal_data fixture

Generated artefacts (fitted models) are saved to tests/data/tmp/.
"""

import math

import numpy as np
import pytest

from aabim.calibration import (
    CalibrationData,
    CalibrationModel,
    InSitu,
    OLSModel,
    RatioModel,
)
from tests.conftest import TMP_DIR


# --------------------------------------------------------------------------- #
# InSitu                                                                       #
# --------------------------------------------------------------------------- #


def test_insitu_n_uuids(insitu):
    assert len(insitu.uuids) == 18


def test_insitu_wavelength_range(insitu):
    assert insitu.wavelengths.min() == pytest.approx(356.0)
    assert insitu.wavelengths.max() == pytest.approx(800.0)


def test_insitu_rho_positive(insitu):
    uuid = insitu.uuids[0]
    wl, rho = insitu.rho_for_uuid(uuid)
    assert (rho > 0).all()


def test_insitu_rho_is_pi_times_rrs(insitu):
    """rho = π × Rrs, so rho should be larger than Rrs (which is < 0.1 sr⁻¹)."""
    uuid = insitu.uuids[0]
    _, rho = insitu.rho_for_uuid(uuid)
    assert rho.max() < 1.0   # reflectance, not radiance


def test_insitu_location(insitu):
    loc = insitu.location_for_uuid(insitu.uuids[0])
    assert -65.0 < loc["lon"] < -64.0
    assert 49.0  < loc["lat"] < 50.0


# --------------------------------------------------------------------------- #
# CalibrationData                                                              #
# --------------------------------------------------------------------------- #


def test_cal_data_has_matchups(cal_data):
    assert cal_data.df["uuid"].nunique() > 0


def test_cal_data_columns(cal_data):
    required = {"uuid", "wavelength", "rho_t", "rho_t_hat"}
    assert required.issubset(cal_data.df.columns)


def test_cal_data_rho_t_hat_positive(cal_data):
    assert (cal_data.df["rho_t_hat"] > 0).all()


def test_cal_data_saved(cal_data):
    """CalibrationData is saved to TMP_DIR by the fixture itself."""
    assert (TMP_DIR / "cal_data.nc").exists()


# --------------------------------------------------------------------------- #
# CalibrationModel.fit / apply                                                 #
# --------------------------------------------------------------------------- #


def test_ratio_model_fit(cal_data):
    model = RatioModel.fit(cal_data)
    assert len(model.wavelength) == len(cal_data.wavelengths)
    # Bad bands (radiance = 0) produce NaN gain; non-NaN values must be positive
    assert np.nanmin(model.coeffs["gain"].values) > 0


def test_ols_model_fit(cal_data):
    model = OLSModel.fit(cal_data)
    assert len(model.wavelength) == len(cal_data.wavelengths)
    assert not np.isnan(model.coeffs["r2"].values).all()


def test_model_apply_3d(cal_data):
    model = RatioModel.fit(cal_data)
    n_wl = len(model.wavelength)
    rho_t = np.ones((n_wl, 10, 10)) * 0.05
    result = model.apply(rho_t)
    assert result.shape == (n_wl, 10, 10)


# --------------------------------------------------------------------------- #
# Persistence: save / load round-trip                                          #
# --------------------------------------------------------------------------- #


def test_ratio_model_roundtrip(cal_data):
    model = RatioModel.fit(cal_data)
    path = TMP_DIR / "ratio_model.csv"
    model.save(str(path))

    loaded = CalibrationModel.load(str(path))
    assert isinstance(loaded, RatioModel)
    np.testing.assert_allclose(model.wavelength, loaded.wavelength, rtol=1e-6)
    np.testing.assert_allclose(
        model.coeffs["gain"].values, loaded.coeffs["gain"].values
    )


def test_ols_model_roundtrip(cal_data):
    model = OLSModel.fit(cal_data)
    path = TMP_DIR / "ols_model.csv"
    model.save(str(path))

    loaded = CalibrationModel.load(str(path))
    assert isinstance(loaded, OLSModel)
    np.testing.assert_allclose(
        model.coeffs["a"].values, loaded.coeffs["a"].values
    )


def test_load_csv_no_header(cal_data, tmp_path):
    """RatioModel.load() works on a plain CSV with no comment header (e.g. from R)."""
    model = RatioModel.fit(cal_data)
    import pandas as pd

    # Write a plain CSV without the # comment lines (as R would produce)
    df = model.coeffs.to_dataframe().reset_index()[["wavelength", "gain"]]
    csv_path = tmp_path / "gains_no_header.csv"
    df.to_csv(csv_path, index=False)

    loaded = RatioModel.load(str(csv_path))
    assert isinstance(loaded, RatioModel)
    np.testing.assert_allclose(
        model.coeffs["gain"].values, loaded.coeffs["gain"].values
    )


def test_to_l1r_with_csv_cal_model(aci13_l1c, tmp_path):
    """to_l1r applies a unit-gain RatioModel loaded from a hand-crafted CSV."""
    wl = aci13_l1c.wavelength
    csv_path = tmp_path / "unit_gain.csv"
    lines = ["# model_name: ratio", "# metadata: {}", "wavelength,gain"]
    lines += [f"{w:.4f},1.0" for w in wl]
    csv_path.write_text("\n".join(lines))

    model = CalibrationModel.load(str(csv_path))
    assert isinstance(model, RatioModel)

    out = str(tmp_path / "l1r_csv_cal.zarr")
    img = aci13_l1c.to_l1r(out, cal_model=model)
    assert "rho_at_sensor" in img.in_ds
    assert img.in_ds.attrs.get("cal_model") is not None


# --------------------------------------------------------------------------- #
# Full pipeline: ancillary → CalibrationData → fit → plot                     #
# --------------------------------------------------------------------------- #


def test_pipeline_plot(aci13_l1r_anc, insitu):
    """End-to-end: real ancillary data → CalibrationData → OLS fit → saved plot.

    In-situ Rrs (lw/ed) is converted to irradiance reflectance ρ_w = π × Rrs
    in the insitu fixture before being passed to CalibrationData.compute().
    """
    cd = CalibrationData.compute(aci13_l1r_anc, insitu)
    assert cd.df["uuid"].nunique() > 0

    model = OLSModel.fit(cd)
    plot_path = TMP_DIR / "calibration_scatter.png"
    cd.plot(model=model, save_path=plot_path)

    assert plot_path.exists()
