"""Per-segment Stan data preparation.

Design
------
:class:`InversionData` is the bridge between an :class:`~aabim.image.Image`
and a Stan inversion model.  It holds a list of per-segment data
dictionaries ready to be passed to CmdStanPy, together with the segment
label array and GeoDataFrame needed to map results back to the pixel grid.

Workflow
--------
.. code-block:: python

    seg      = FelzenszwalbSegmenter(scale=50, min_size=20)
    inv_data = InversionData.from_image(
        img_l2r,
        segmenter  = seg,
        bottom_classes = ["coral", "sand", "seagrass"],
        rho_var    = "rho_w",          # or "rho_w_gl21"
        mode       = "aqua",           # or "atm_aqua"
    )

Shipped data
------------
IOP vectors (``a_w``, ``a0``, ``a1``) and the bottom-reflectance library
(``r_b_gamache.csv``) are looked up from ``aabim/data/aquatic/`` by default.
Pass ``data_dir`` to override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import geopandas as gpd
from shapely.geometry import box

from aabim.inversion.segment import Segmenter, NoSegmenter

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "aquatic"

# Columns produced by the segment GeoDataFrame
_SEG_COLS = ["segment_id", "n_pixels", "geometry"]


def _interp1(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """Linear interpolation with flat extrapolation."""
    order = np.argsort(x)
    return np.interp(xout, x[order], y[order])


def _load_iop_luts(
    wavelength: np.ndarray,
    data_dir: Path,
) -> Dict[str, np.ndarray]:
    """Load and interpolate IOP LUT vectors to sensor wavelengths."""
    import pandas as pd

    a_w_df   = pd.read_csv(data_dir / "a_w.csv")
    a01_df   = pd.read_csv(data_dir / "a0_a1_phyto.csv")

    return {
        "a_w": _interp1(a_w_df["wavelength"].values,  a_w_df["a_w"].values,  wavelength),
        "a0":  _interp1(a01_df["wavelength"].values,  a01_df["a0"].values,   wavelength),
        "a1":  _interp1(a01_df["wavelength"].values,  a01_df["a1"].values,   wavelength),
        "bb_w": 0.00111 * (wavelength / 500.0) ** (-4.32),
    }


def _load_rb_library(
    wavelength: np.ndarray,
    data_dir: Path,
) -> Dict[str, Any]:
    """Load and interpolate bottom-reflectance library."""
    import pandas as pd

    rb_df = pd.read_csv(data_dir / "r_b_gamache_obs.csv")
    rb_df["class"] = rb_df["class"].astype(str)
    classes = sorted(rb_df["class"].unique().tolist())

    grp     = rb_df.groupby(["wavelength", "class"])["r_b"]
    mu_wide = grp.mean().unstack("class")
    sd_wide = grp.std(ddof=1).fillna(1e-4).unstack("class")
    wl_lib  = mu_wide.index.to_numpy(dtype=float)

    mu = np.column_stack([_interp1(wl_lib, mu_wide[c].values.astype(float), wavelength)
                          for c in classes])
    sd = np.column_stack([_interp1(wl_lib, sd_wide[c].values.astype(float), wavelength)
                          for c in classes])
    sd = np.maximum(sd, 1e-6)
    return {"classes": classes, "r_b_mu_lib": mu, "r_b_sd_lib": sd}


def _rgb_composite(image: Any, rho_var: str) -> np.ndarray:
    """Build a normalised (0–1) RGB composite for segmentation."""
    import xarray as xr

    wl = np.asarray(image.wavelength)
    ds = image.in_ds

    def _band(target: float) -> np.ndarray:
        i = int(np.argmin(np.abs(wl - target)))
        arr = ds[rho_var].isel(wavelength=i).values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        p2, p98 = np.nanpercentile(arr, [2, 98])
        if p98 > p2:
            arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
        return arr

    r = _band(665.0)
    g = _band(560.0)
    b = _band(490.0)
    return np.stack([r, g, b], axis=-1)


@dataclass
class InversionData:
    """Per-segment Stan data, ready for :class:`~aabim.inversion.model.InversionModel`.

    Attributes
    ----------
    segments : list of dict
        One Stan data dict per segment.
    labels : np.ndarray, shape (n_rows, n_cols)
        Integer label array; ``-1`` = excluded pixel.
    gdf : GeoDataFrame
        One row per segment with geometry and ``n_pixels``.
    wavelength : np.ndarray
        Sensor wavelengths (nm).
    mode : {"aqua", "atm_aqua"}
        Inversion mode.
    """

    segments:   List[Dict[str, Any]]
    labels:     np.ndarray
    gdf:        gpd.GeoDataFrame
    wavelength: np.ndarray
    mode:       str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_image(
        cls,
        image: Any,
        *,
        segmenter:     Optional[Segmenter] = None,
        bottom_classes: Sequence[str]      = (),
        rho_var:        str                = "rho_w",
        mode:           str                = "aqua",
        water_type:     int                = 2,
        shallow:        int                = 1,
        mad_floor:      float              = 1e-4,
        data_dir:       Optional[str]      = None,
    ) -> "InversionData":
        """Build InversionData from an aabim Image.

        Parameters
        ----------
        image :
            L2R :class:`~aabim.image.Image` (``mode="aqua"``) or L1C image
            (``mode="atm_aqua"``).
        segmenter :
            A :class:`~aabim.inversion.segment.Segmenter` instance.
            Defaults to :class:`~aabim.inversion.segment.NoSegmenter`.
        bottom_classes :
            Names of bottom-reflectance classes to use as prior from the
            library.  The prior is the mean across the selected classes.
            Pass an empty sequence to use the library-wide mean.
        rho_var :
            Which water-leaving reflectance variable to invert
            (``"rho_w"`` or ``"rho_w_gl21"``).  Ignored in
            ``mode="atm_aqua"``.
        mode :
            ``"aqua"`` for aquatic-only inversion from *rho_w*;
            ``"atm_aqua"`` for joint atmospheric + aquatic from
            *rho_at_sensor*.
        water_type :
            AM03 model type: 1 = turbid coastal, 2 = case-II (default).
        shallow :
            1 = shallow-water RTM (depth + bottom reflectance active),
            0 = deep-water RTM.
        mad_floor :
            Minimum per-band uncertainty (MAD) to avoid zero-variance
            likelihood.
        data_dir :
            Directory with ``a_w.csv``, ``a0_a1_phyto.csv``,
            ``r_b_gamache.csv``.  Defaults to the shipped
            ``aabim/data/aquatic/`` directory.
        """
        if mode not in ("aqua", "atm_aqua"):
            raise ValueError(f"mode must be 'aqua' or 'atm_aqua', got {mode!r}")

        segmenter = segmenter or NoSegmenter()
        ddir      = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        wl        = np.asarray(image.wavelength, dtype=np.float64)

        # ---------- observation variable --------------------------------
        obs_var = rho_var if mode == "aqua" else "rho_at_sensor"
        if obs_var not in image.in_ds:
            raise KeyError(
                f"Variable '{obs_var}' not found in image dataset. "
                f"Available: {list(image.in_ds.data_vars)}"
            )

        # ---------- segmentation ----------------------------------------
        water_mask = image.in_ds["mask_water"].values.astype(bool)
        rgb        = _rgb_composite(image, obs_var if mode == "aqua" else "rho_at_sensor")
        labels     = segmenter.segment(rgb, water_mask)

        # ---------- IOP LUTs -------------------------------------------
        iop  = _load_iop_luts(wl, ddir)
        rlib = _load_rb_library(wl, ddir)

        # Bottom prior: mean over selected classes (or all)
        classes = rlib["classes"]
        if bottom_classes:
            missing = [c for c in bottom_classes if c not in classes]
            if missing:
                raise ValueError(
                    f"Bottom classes not in library: {missing}. "
                    f"Available: {classes}"
                )
            idx = [classes.index(c) for c in bottom_classes]
        else:
            idx = list(range(len(classes)))

        rb_mu = rlib["r_b_mu_lib"][:, idx].mean(axis=1)
        rb_sd = rlib["r_b_sd_lib"][:, idx].mean(axis=1)
        rb_mu = np.clip(rb_mu, 1e-6, 1 - 1e-6)

        # ---------- atmospheric LUT slice (atm_aqua mode) ---------------
        atm_slice: Dict[str, Any] = {}
        if mode == "atm_aqua":
            from aabim.data.atmospheric.atmospheric import AerLUT
            aer_lut = AerLUT.from_image(image)
            sol_zen_med  = float(np.nanmedian(image.in_ds["sun_zenith"].values))
            view_zen_med = float(np.nanmedian(image.in_ds["view_zenith"].values))
            raa_med      = float(np.nanmedian(image.in_ds["relative_azimuth"].values))
            pressure     = float(image.in_ds["surface_air_pressure"].values)
            altitude     = float(image.z.values)
            atm_slice = aer_lut.slice_for_stan(
                sol_zen_med, view_zen_med, raa_med, pressure, altitude
            )

        # ---------- per-segment data dicts ------------------------------
        rho_data   = image.in_ds[obs_var].values   # (n_wl, n_rows, n_cols)
        sun_zen    = image.in_ds["sun_zenith"].values
        view_zen   = image.in_ds["view_zenith"].values

        unique_labels = np.unique(labels[labels >= 0])
        n_wl          = len(wl)
        segments: List[Dict[str, Any]] = []
        seg_rows: List[Dict[str, Any]] = []

        for seg_id in unique_labels:
            mask = labels == seg_id
            n_pix = int(mask.sum())

            # Spectral observations: median + MAD uncertainty
            pix_rho = rho_data[:, mask]                     # (n_wl, n_pix)
            rho_obs = np.nanmedian(pix_rho, axis=1)
            rho_sig = np.maximum(
                1.4826 * np.nanmedian(np.abs(pix_rho - rho_obs[:, None]), axis=1),
                mad_floor,
            )

            # Geometry: median over segment
            theta_sun  = float(np.nanmedian(sun_zen[mask]))
            theta_view = float(np.nanmedian(view_zen[mask]))

            d = {
                "n_wl":         n_wl,
                "wavelength":   wl.tolist(),
                "rho_obs":      rho_obs.tolist(),
                "rho_sigma":    rho_sig.tolist(),
                "theta_sun_deg":  theta_sun,
                "theta_view_deg": theta_view,
                "water_type":   int(water_type),
                "shallow":      int(shallow),
                "a_w":          iop["a_w"].tolist(),
                "a0":           iop["a0"].tolist(),
                "a1":           iop["a1"].tolist(),
                "bb_w":         iop["bb_w"].tolist(),
                "rb_mu":        rb_mu.tolist(),
                "rb_sd":        rb_sd.tolist(),
            }

            if mode == "atm_aqua":
                d.update(atm_slice)

            segments.append(d)

            # Bounding box geometry for GeoDataFrame
            rows, cols = np.where(mask)
            x_vals = image.in_ds.x.values[cols]
            y_vals = image.in_ds.y.values[rows]
            geom = box(x_vals.min(), y_vals.min(), x_vals.max(), y_vals.max())
            seg_rows.append({"segment_id": int(seg_id), "n_pixels": n_pix, "geometry": geom})

        gdf = gpd.GeoDataFrame(seg_rows, crs=image.CRS)

        return cls(
            segments   = segments,
            labels     = labels,
            gdf        = gdf,
            wavelength = wl,
            mode       = mode,
        )

    def __len__(self) -> int:
        return len(self.segments)
