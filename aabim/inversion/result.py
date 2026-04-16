"""Inversion result storage.

:class:`InversionResult` maps per-segment posterior summaries back to the
pixel grid and writes outputs in two formats:

``to_zarr(output)``
    Pixel-grid parameter maps (same spatial structure as the L2R output).
    Scalar parameters are stored as ``(y, x)`` arrays with ``_mean``,
    ``_std``, ``_q05``, ``_q95`` suffixes.  Spectral bottom reflectance
    ``r_b`` is stored as ``(wavelength, y, x)``.  Uncertainty suffixes
    are NaN when ``method="optimize"``.

``to_gpkg(output)``
    Per-segment GeoPackage (only meaningful when a real
    :class:`~aabim.inversion.segment.Segmenter` was used).  Each row
    carries the posterior summaries as attributes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
import geopandas as gpd
import dask.array as da

log = logging.getLogger(__name__)

_SCALAR_SUFFIXES = ("_mean", "_std", "_q05", "_q95")


@dataclass
class InversionResult:
    """Posterior summaries for all segments.

    Attributes
    ----------
    summaries : list of dict
        One dict per segment; keys follow the pattern
        ``<param>_mean``, ``<param>_std``, ``<param>_q05``,
        ``<param>_q95``.  ``r_b_mean`` and ``r_b_std`` are 1-D arrays of
        length ``n_wl``.
    labels : np.ndarray, shape (n_rows, n_cols)
        Segment label array; ``-1`` = excluded pixel.
    gdf : GeoDataFrame
        Segment polygons.
    wavelength : np.ndarray
        Sensor wavelengths (nm).
    params : list of str
        Scalar parameter names (without suffix).
    method : {"sample", "optimize"}
    """

    summaries:  List[Optional[Dict[str, Any]]]
    labels:     np.ndarray
    gdf:        gpd.GeoDataFrame
    wavelength: np.ndarray
    params:     List[str]
    method:     str

    # ------------------------------------------------------------------
    # Pixel-grid Zarr output
    # ------------------------------------------------------------------

    def to_zarr(self, output: str) -> None:
        """Write pixel-grid parameter maps to a Zarr store.

        Parameters
        ----------
        output : str
            Path to the output Zarr directory store.
        """
        import shutil
        out = Path(output)
        if out.exists():
            shutil.rmtree(out)

        n_rows, n_cols = self.labels.shape
        n_wl           = len(self.wavelength)
        n_segs         = len(self.summaries)

        # Build lookup: segment_id → summary dict
        unique = np.unique(self.labels[self.labels >= 0])
        lookup = {int(uid): self.summaries[idx]
                  for idx, uid in enumerate(unique)
                  if idx < n_segs and self.summaries[idx]}

        nan2d = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        nan3d = np.full((n_wl, n_rows, n_cols), np.nan, dtype=np.float32)

        # Allocate output arrays
        arrays: Dict[str, np.ndarray] = {}
        for p in self.params:
            for suf in _SCALAR_SUFFIXES:
                arrays[f"{p}{suf}"] = nan2d.copy()
        arrays["r_b_mean"] = nan3d.copy()
        arrays["r_b_std"]  = nan3d.copy()

        # Fill pixel-by-pixel from segment summaries
        for seg_id, summ in lookup.items():
            if not summ:
                continue
            mask = self.labels == seg_id
            for p in self.params:
                for suf in _SCALAR_SUFFIXES:
                    key = f"{p}{suf}"
                    val = summ.get(key, np.nan)
                    arrays[key][mask] = float(val)
            if "r_b_mean" in summ:
                rb_m = np.asarray(summ["r_b_mean"], dtype=np.float32)
                arrays["r_b_mean"][:, mask] = rb_m[:, None]
            if "r_b_std" in summ:
                rb_s = np.asarray(summ["r_b_std"], dtype=np.float32)
                arrays["r_b_std"][:, mask] = rb_s[:, None]

        # Build xarray Dataset
        coords_2d = {"y": np.arange(n_rows), "x": np.arange(n_cols)}
        coords_3d = {
            "wavelength": self.wavelength,
            "y": np.arange(n_rows),
            "x": np.arange(n_cols),
        }

        dvars: Dict[str, xr.DataArray] = {}
        for p in self.params:
            for suf in _SCALAR_SUFFIXES:
                key = f"{p}{suf}"
                dvars[key] = xr.DataArray(
                    arrays[key], dims=["y", "x"], coords=coords_2d,
                    attrs={"units": "1", "long_name": key},
                )
        dvars["r_b_mean"] = xr.DataArray(
            arrays["r_b_mean"], dims=["wavelength", "y", "x"], coords=coords_3d,
            attrs={"units": "1", "long_name": "Bottom reflectance (posterior mean)"},
        )
        dvars["r_b_std"] = xr.DataArray(
            arrays["r_b_std"], dims=["wavelength", "y", "x"], coords=coords_3d,
            attrs={"units": "1", "long_name": "Bottom reflectance (posterior std)"},
        )

        ds = xr.Dataset(dvars)
        ds.attrs["inversion_method"] = self.method
        ds.to_zarr(str(output), mode="w", consolidated=True)
        log.info("InversionResult written → %s", output)

    # ------------------------------------------------------------------
    # Per-segment GeoPackage output
    # ------------------------------------------------------------------

    def to_gpkg(self, output: str) -> None:
        """Write per-segment posteriors to a GeoPackage.

        Parameters
        ----------
        output : str
            Path to the output ``.gpkg`` file.
        """
        rows = []
        for idx, row in self.gdf.iterrows():
            seg_id = int(row["segment_id"])
            summ   = self.summaries[idx] if idx < len(self.summaries) else {}
            d      = dict(row)
            if summ:
                for p in self.params:
                    for suf in _SCALAR_SUFFIXES:
                        key = f"{p}{suf}"
                        d[key] = float(summ.get(key, np.nan))
                # Store mean r_b as a comma-separated string (GeoPackage
                # does not support array columns)
                if "r_b_mean" in summ:
                    d["r_b_mean"] = ",".join(
                        f"{v:.6f}" for v in summ["r_b_mean"]
                    )
            rows.append(d)

        out_gdf = gpd.GeoDataFrame(rows, crs=self.gdf.crs)
        out_gdf.to_file(output, driver="GPKG")
        log.info("Segment GeoPackage written → %s", output)
