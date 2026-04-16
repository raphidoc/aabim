"""Minimal smoke tests for aabim.inversion.

These tests verify the data-preparation and result-output stages without
requiring Stan or CmdStan to be installed.  Model-fitting tests are skipped
when CmdStan is not available.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from aabim.inversion import (
    AquaModel,
    FelzenszwalbSegmenter,
    InversionData,
    InversionResult,
    NoSegmenter,
    SLICSegmenter,
)

TMP_DIR = Path(__file__).parent / "data" / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WL = np.linspace(400.0, 700.0, 20)   # 20 bands in IOP LUT range
_NR, _NC = 8, 8                         # tiny spatial grid


def _make_synthetic_image(rho_var: str = "rho_w") -> SimpleNamespace:
    """Return an Image-like namespace with the minimum attributes needed by
    InversionData.from_image()."""
    rng = np.random.default_rng(0)

    # Spectral data (n_wl, n_rows, n_cols)
    rho = rng.uniform(0.001, 0.05, size=(len(_WL), _NR, _NC)).astype(np.float32)
    mask_water = np.ones((_NR, _NC), dtype=bool)

    sun_zen  = np.full((_NR, _NC), 30.0, dtype=np.float32)
    view_zen = np.full((_NR, _NC), 10.0, dtype=np.float32)

    x = np.arange(_NC, dtype=np.float64)
    y = np.arange(_NR, dtype=np.float64)

    ds = xr.Dataset(
        data_vars={
            rho_var: (["wavelength", "y", "x"], rho,
                      {"units": "1", "long_name": rho_var}),
            "mask_water":  (["y", "x"], mask_water.astype(np.int8)),
            "sun_zenith":  (["y", "x"], sun_zen),
            "view_zenith": (["y", "x"], view_zen),
        },
        coords={"wavelength": _WL, "y": y, "x": x},
    )

    import pyproj
    crs = pyproj.CRS.from_epsg(32619)

    img = SimpleNamespace(
        wavelength=_WL,
        in_ds=ds,
        CRS=crs,
    )
    return img


# ---------------------------------------------------------------------------
# Segmenters
# ---------------------------------------------------------------------------

class TestSegmenters:
    def test_no_segmenter_all_water(self):
        rgb  = np.zeros((_NR, _NC, 3), dtype=np.float32)
        mask = np.ones((_NR, _NC), dtype=bool)
        labels = NoSegmenter().segment(rgb, mask)
        assert labels.shape == (_NR, _NC)
        assert (labels[mask] >= 0).all()
        assert (labels[~mask] == -1).all()

    def test_no_segmenter_partial_water(self):
        rgb  = np.zeros((_NR, _NC, 3), dtype=np.float32)
        mask = np.zeros((_NR, _NC), dtype=bool)
        mask[:4, :4] = True
        labels = NoSegmenter().segment(rgb, mask)
        assert (labels[~mask] == -1).all()
        assert labels[mask].min() >= 0

    def test_felzenszwalb_returns_labels(self):
        pytest.importorskip("skimage", reason="scikit-image not installed")
        rng = np.random.default_rng(1)
        rgb  = rng.random((_NR, _NC, 3)).astype(np.float32)
        mask = np.ones((_NR, _NC), dtype=bool)
        labels = FelzenszwalbSegmenter(scale=10, min_size=2).segment(rgb, mask)
        assert labels.shape == (_NR, _NC)
        assert labels.dtype == np.int32
        assert (labels >= -1).all()

    def test_slic_returns_labels(self):
        pytest.importorskip("skimage", reason="scikit-image not installed")
        rng = np.random.default_rng(2)
        rgb  = rng.random((_NR, _NC, 3)).astype(np.float32)
        mask = np.ones((_NR, _NC), dtype=bool)
        labels = SLICSegmenter(n_segments=4).segment(rgb, mask)
        assert labels.shape == (_NR, _NC)


# ---------------------------------------------------------------------------
# InversionData
# ---------------------------------------------------------------------------

class TestInversionData:
    def test_from_image_aqua_nosegmenter(self):
        img  = _make_synthetic_image("rho_w")
        idata = InversionData.from_image(img, segmenter=NoSegmenter(), mode="aqua")
        assert idata.mode == "aqua"
        assert len(idata) > 0
        assert idata.wavelength.shape == _WL.shape
        # Each segment dict must have the required Stan data keys
        d = idata.segments[0]
        for key in ("n_wl", "wavelength", "rho_obs", "rho_sigma",
                    "theta_sun_deg", "theta_view_deg",
                    "a_w", "a0", "a1", "bb_w", "rb_mu", "rb_sd"):
            assert key in d, f"Missing key: {key}"

    def test_from_image_rho_w_gl21(self):
        img = _make_synthetic_image("rho_w")
        # Add a second reflectance variable under a different name
        img.in_ds = img.in_ds.assign(
            rho_w_gl21=img.in_ds["rho_w"]
        )
        idata = InversionData.from_image(img, segmenter=NoSegmenter(),
                                         rho_var="rho_w_gl21", mode="aqua")
        assert len(idata) > 0

    def test_from_image_missing_var_raises(self):
        img = _make_synthetic_image("rho_w")
        with pytest.raises(KeyError):
            InversionData.from_image(img, segmenter=NoSegmenter(),
                                     rho_var="nonexistent", mode="aqua")

    def test_from_image_bad_bottom_class_raises(self):
        img = _make_synthetic_image("rho_w")
        with pytest.raises(ValueError, match="Bottom classes not in library"):
            InversionData.from_image(img, segmenter=NoSegmenter(),
                                     bottom_classes=["unicorn"], mode="aqua")

    def test_len(self):
        img = _make_synthetic_image()
        idata = InversionData.from_image(img)
        assert len(idata) == len(idata.segments)

    def test_gdf_segment_count(self):
        img = _make_synthetic_image()
        idata = InversionData.from_image(img)
        assert len(idata.gdf) == len(idata)


# ---------------------------------------------------------------------------
# InversionResult
# ---------------------------------------------------------------------------

def _make_result(n_wl: int = 10, n_rows: int = 4, n_cols: int = 4,
                 n_segs: int = 2, method: str = "sample") -> InversionResult:
    """Build a minimal InversionResult with synthetic posteriors."""
    rng = np.random.default_rng(42)
    wl  = np.linspace(400.0, 700.0, n_wl)

    labels = np.zeros((n_rows, n_cols), dtype=np.int32)
    labels[n_rows // 2:, :] = 1

    params = ["chl", "a_g_440", "spm", "bb_p_gamma", "a_g_slope", "h_w"]
    summaries = []
    for _ in range(n_segs):
        s: dict = {}
        for p in params:
            s[f"{p}_mean"] = float(rng.uniform(0, 1))
            s[f"{p}_std"]  = float(rng.uniform(0, 0.1))
            s[f"{p}_q05"]  = s[f"{p}_mean"] - 0.1
            s[f"{p}_q95"]  = s[f"{p}_mean"] + 0.1
        s["r_b_mean"] = rng.uniform(0.01, 0.5, size=n_wl).astype(np.float32)
        s["r_b_std"]  = rng.uniform(0.001, 0.05, size=n_wl).astype(np.float32)
        summaries.append(s)

    geoms = [box(0, 0, 2, 2), box(2, 2, 4, 4)]
    gdf   = gpd.GeoDataFrame(
        {"segment_id": [0, 1], "n_pixels": [8, 8], "geometry": geoms},
        crs="EPSG:32619",
    )
    return InversionResult(
        summaries=summaries, labels=labels, gdf=gdf,
        wavelength=wl, params=params, method=method,
    )


class TestInversionResult:
    def test_to_zarr_creates_store(self, tmp_path):
        res  = _make_result()
        out  = str(tmp_path / "result.zarr")
        res.to_zarr(out)
        ds = xr.open_zarr(out)
        assert "chl_mean" in ds
        assert "r_b_mean" in ds
        assert ds["r_b_mean"].dims == ("wavelength", "y", "x")

    def test_to_zarr_scalar_shape(self, tmp_path):
        res = _make_result(n_rows=4, n_cols=4)
        out = str(tmp_path / "result.zarr")
        res.to_zarr(out)
        ds = xr.open_zarr(out)
        assert ds["chl_mean"].dims == ("y", "x")
        assert ds["chl_mean"].shape == (4, 4)

    def test_to_zarr_optimize_no_std(self, tmp_path):
        res = _make_result(method="optimize")
        # Remove std/quantile keys from summaries to simulate optimize output
        for s in res.summaries:
            for p in res.params:
                s.pop(f"{p}_std",  None)
                s.pop(f"{p}_q05",  None)
                s.pop(f"{p}_q95",  None)
        out = str(tmp_path / "result_opt.zarr")
        res.to_zarr(out)  # should not raise
        ds = xr.open_zarr(out)
        assert "chl_mean" in ds

    def test_to_gpkg_creates_file(self, tmp_path):
        res  = _make_result()
        out  = str(tmp_path / "result.gpkg")
        res.to_gpkg(out)
        gdf = gpd.read_file(out)
        assert len(gdf) == 2
        assert "chl_mean" in gdf.columns

    def test_to_gpkg_rb_mean_string(self, tmp_path):
        res = _make_result()
        out = str(tmp_path / "rb.gpkg")
        res.to_gpkg(out)
        gdf = gpd.read_file(out)
        assert isinstance(gdf["r_b_mean"].iloc[0], str)


# ---------------------------------------------------------------------------
# AquaModel.fit — skipped if CmdStan not available
# ---------------------------------------------------------------------------

def _cmdstan_available() -> bool:
    try:
        import cmdstanpy
        cmdstanpy.CmdStanModel  # noqa: B018
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cmdstan_available(), reason="CmdStan not installed")
def test_aquamodel_fit_optimize():
    """End-to-end smoke test: fit AquaModel with optimize on a tiny image."""
    img   = _make_synthetic_image("rho_w")
    idata = InversionData.from_image(img, segmenter=NoSegmenter(), mode="aqua")
    result = AquaModel.fit(idata, method="optimize", n_workers=1)
    assert result is not None
    assert len(result.summaries) == len(idata)
    # At least some segments should have produced a result
    non_empty = [s for s in result.summaries if s]
    assert len(non_empty) > 0
