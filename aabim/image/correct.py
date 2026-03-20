"""
correct.py — CorrectMixin: L1C → L1R and L1C → L2R processing.

L1R  (Level 1 Reflectance)
--------------------------
Converts raw at-sensor radiance to top-of-atmosphere (TOA) reflectance and
adds a water-pixel mask derived from NDWI.

    rho_at_sensor = π × L(λ) / [F0(λ, DOY) × cos(θ_sun)]

    F0  : Coddington et al. 2022 solar irradiance, Earth–Sun distance
          corrected: × (1 + 0.034 × cos(2π × (DOY − 2) / 365))
          Converted from W m⁻² nm⁻¹ → uW cm⁻² nm⁻¹ (× 100).

Output Zarr variables added by to_l1r
    rho_at_sensor  (wavelength, y, x)
    ndwi           (y, x)
    mask_water     (y, x)  — uint8, 1 = water

L2R  (Level 2 Reflectance)
--------------------------
Combines the L1R step with full atmospheric correction using AerLUT + GasLUT.
Both methods share the same rho_at_sensor computation; to_l2r does not require
a pre-existing L1R image on disk.

Inverse atmospheric model (per-window, shape (n_wl, ny, nx))
    rho_tg  = rho_at_sensor / t_g − rho_path
    rho_s   = rho_tg / t_ra + s_ra × rho_tg
    rho_w   = rho_s − sky_glint          (water pixels; NaN elsewhere)
    rho_w_gl21 = G21(rho_w, λ)

G21 residual glint correction  [Gao & Li 2021]
    n_w(λ)     = Quan & Fry 1995 (T = 12 °C, S = 30 PSU)
    rho_f(λ)   = Fresnel reflectance at nadir (θ = 0)
    ref_ratio  = rho_w[800 nm] / rho_f[800 nm]
    rho_w_gl21 = rho_w − rho_f(λ) × ref_ratio

Output Zarr variables added by to_l2r
    rho_at_sensor  (wavelength, y, x)   — with optional cal_model applied
    rho_s          (wavelength, y, x)
    rho_w          (wavelength, y, x)
    rho_w_gl21     (wavelength, y, x)
    ndwi / mask_water  (y, x)
    geometry + ancillary scalars carried from input

Parallel processing (CPU)
--------------------------
The image is divided into non-overlapping windows of size window_size × window_size.
A ProcessPoolExecutor dispatches windows across cores.  LUTs are loaded once per
worker process via an initializer.  Each worker writes its results directly to its
non-overlapping region of the pre-initialised output Zarr store — Zarr chunk writes
are atomic, so no locking is required as long as windows do not share chunks
(guaranteed when chunk_size ≤ window_size, which is enforced here).
"""
from __future__ import annotations

import copy
import datetime
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

from aabim.utils.optics import get_water_refractive_index

# Inlined here to avoid a circular import through aabim.ancillary → aabim.image.
_ANCILLARY_VARS = [
    "aerosol_optical_depth_at_550_nm",
    "wind_speed",
    "wind_direction",
    "surface_air_pressure",
    "atmosphere_mass_content_of_water_vapor",
    "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
]

if TYPE_CHECKING:
    from aabim.image.image import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundled solar irradiance spectrum
# ---------------------------------------------------------------------------
_F0_PATH = (
    Path(__file__).parent.parent
    / "data/solar_irradiance/hybrid_reference_spectrum_c2022-11-30_with_unc.nc"
)

# Ancillary variables the image must carry before to_l2r() is called.
# rho_at_sensor and mask_water are computed internally by to_l2r.
_L2R_REQUIRED_VARS = _ANCILLARY_VARS


# ---------------------------------------------------------------------------
# Module-level worker state (populated by _worker_init in each subprocess)
# ---------------------------------------------------------------------------
_worker_aer_lut = None
_worker_gas_lut = None


def _worker_init(aer_cache: str, gas_cache: str, backend: str) -> None:
    """Load LUT interpolators once per worker process."""
    global _worker_aer_lut, _worker_gas_lut
    from aabim.data.atmospheric.atmospheric import AerLUT, GasLUT  # local import — subprocess

    _worker_aer_lut = AerLUT.from_cache_file(aer_cache, backend)
    _worker_gas_lut = GasLUT.from_cache_file(gas_cache, backend)


def _correction_worker(args: dict) -> None:
    """Atmospheric correction for one spatial window.

    Reads rho_at_sensor + geometry from the input Zarr, runs the LUT queries,
    computes rho_s / rho_w / rho_w_gl21, and writes the results directly to
    the corresponding region of the output Zarr.  Uses the process-level
    LUT state loaded by :func:`_worker_init`.
    """
    global _worker_aer_lut, _worker_gas_lut
    from aabim.data.atmospheric.atmospheric import BaseLUT

    row_off = args["row_off"]
    col_off = args["col_off"]
    height  = args["height"]
    width   = args["width"]
    y_slice = slice(row_off, row_off + height)
    x_slice = slice(col_off, col_off + width)

    # --- Read window from input Zarr (via xarray for CF decoding) ---------
    ds_in   = xr.open_zarr(args["in_path"])
    isel    = {"y": y_slice, "x": x_slice}
    rho_t   = np.asarray(ds_in["rho_at_sensor"].isel(**isel).values, dtype=np.float32)

    if np.isnan(rho_t).all():
        return

    sun_zen  = np.asarray(ds_in["sun_zenith"].isel(**isel).values,      dtype=np.float32)
    view_zen = np.asarray(ds_in["view_zenith"].isel(**isel).values,     dtype=np.float32)
    raa      = np.asarray(ds_in["relative_azimuth"].isel(**isel).values,dtype=np.float32)
    ds_in.close()

    valid_mask = np.isfinite(rho_t).any(axis=0)   # (height, width)
    n_pix = int(valid_mask.sum())
    if n_pix == 0:
        return

    n_wl     = rho_t.shape[0]
    aod550   = float(args["aod550"])
    pressure = float(args["pressure"])
    water    = float(args["water"])
    ozone    = float(args["ozone"])
    z_alt    = float(args["z"])

    # --- Build query matrices (n_pix × n_features) ------------------------
    sun_v  = sun_zen[valid_mask].reshape(-1, 1)
    view_v = view_zen[valid_mask].reshape(-1, 1)
    raa_v  = raa[valid_mask].reshape(-1, 1)
    ones   = np.ones((n_pix, 1), dtype=np.float32)

    xi_aer = np.hstack([sun_v, view_v, raa_v,
                        ones * aod550, ones * pressure, ones * z_alt])
    xi_gas = np.hstack([sun_v, view_v, raa_v,
                        ones * water, ones * ozone, ones * pressure, ones * z_alt])

    rho_path_v = _worker_aer_lut.interps["rho"](xi_aer)       # (n_pix, n_wl)
    t_ra_v     = _worker_aer_lut.interps["t"](xi_aer)
    s_ra_v     = _worker_aer_lut.interps["s"](xi_aer)
    sky_v      = _worker_aer_lut.interps["sky_glint"](xi_aer)
    t_gas_v    = _worker_gas_lut.interps["t_gas"](xi_gas)

    unpack   = BaseLUT.unpack_to_spatial
    rho_path  = unpack(rho_path_v, valid_mask, n_wl)
    t_ra      = unpack(t_ra_v,     valid_mask, n_wl)
    s_ra      = unpack(s_ra_v,     valid_mask, n_wl)
    sky_glint = unpack(sky_v,      valid_mask, n_wl)
    t_g       = unpack(t_gas_v,    valid_mask, n_wl)

    # --- Inverse atmospheric model ----------------------------------------
    rho_tg = rho_t / t_g - rho_path
    rho_s  = rho_tg / t_ra + s_ra * rho_tg

    # --- Water-leaving reflectance (water pixels only) --------------------
    mask_water_win = args["mask_water_window"]                  # (height, width) bool
    rho_w = np.where(mask_water_win[np.newaxis], rho_s - sky_glint, np.nan)

    # --- G21 residual glint correction ------------------------------------
    rho_f  = args["rho_f"]                                      # (n_wl,)
    nir_i  = int(args["nir_i"])
    ref_ratio   = rho_w[nir_i] / rho_f[nir_i]                  # (height, width)
    rho_w_gl21  = rho_w - rho_f[:, None, None] * ref_ratio[None]

    # --- Write to output Zarr ---------------------------------------------
    z_out = zarr.open(args["out_path"], mode="r+")
    z_out["rho_s"][:, y_slice, x_slice]      = rho_s
    z_out["rho_w"][:, y_slice, x_slice]      = rho_w
    z_out["rho_w_gl21"][:, y_slice, x_slice] = rho_w_gl21


# ---------------------------------------------------------------------------
# Pure physics helpers
# ---------------------------------------------------------------------------

def _solar_irradiance(
    wavelength: np.ndarray,
    doy: int,
    fwhm: np.ndarray | None = None,
    srf_wl: np.ndarray | None = None,
    srf: np.ndarray | None = None,
) -> np.ndarray:
    """Coddington F0, Earth–Sun corrected, in uW cm⁻² nm⁻¹.

    When *fwhm* is provided, a per-band Gaussian SRF is evaluated at the
    Coddington native resolution (~0.001 nm) and used to compute the
    band-averaged F0, matching the approach of the original get_f0()
    implementation and correctly averaging out all Fraunhofer features:

        F0_band = Σ F0(λ) · G(λ; λ_c, σ) / Σ G(λ; λ_c, σ)

    When *srf_wl* / *srf* are provided instead (tabulated RSR for
    multispectral sensors), the spectrum is interpolated onto that grid
    before the weighted average.

    Parameters
    ----------
    wavelength : np.ndarray, shape (n_bands,)
        Band-centre wavelengths (nm).
    doy : int
        Day-of-year for the Earth–Sun distance correction.
    fwhm : np.ndarray, shape (n_bands,), optional
        Per-band FWHM (nm).  When provided, a Gaussian SRF is computed at
        the Coddington native wavelengths for each band.
    srf_wl : np.ndarray, shape (n_fine,), optional
        Fine wavelength axis of a tabulated SRF (nm).
    srf : np.ndarray, shape (n_fine, n_bands), optional
        Per-band tabulated SRF values (any normalization).
    """
    ds      = xr.open_dataset(_F0_PATH)
    f0_wl   = ds["Vacuum Wavelength"].values   # nm
    f0_ssi  = ds["SSI"].values                 # W m⁻² nm⁻¹
    ds.close()

    dist_factor = 1.0 + 0.034 * np.cos(2 * np.pi * (doy - 2) / 365.0)
    f0_corrected = f0_ssi * dist_factor * 100.0   # → uW cm⁻² nm⁻¹

    if fwhm is not None:
        # Evaluate Gaussian SRF at Coddington native resolution per band,
        # then compute the normalised weighted average.  Only the F0 values
        # within 4 σ of each band centre are used for efficiency.
        sigma    = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        f0_bands = np.empty(len(wavelength), dtype=np.float32)
        for i, (wl_c, sig) in enumerate(zip(wavelength, sigma)):
            mask    = np.abs(f0_wl - wl_c) <= 4.0 * sig
            w       = np.exp(-0.5 * ((f0_wl[mask] - wl_c) / sig) ** 2)
            f0_bands[i] = np.dot(f0_corrected[mask], w) / w.sum()
        return f0_bands

    if srf_wl is not None and srf is not None:
        # Tabulated RSR: interpolate Coddington onto the SRF fine grid.
        f0_on_srf = np.interp(srf_wl, f0_wl, f0_corrected)   # (n_fine,)
        # Weighted average: (n_fine,) @ (n_fine, n_bands) → (n_bands,)
        return (f0_on_srf @ srf / srf.sum(axis=0)).astype(np.float32)

    return np.interp(wavelength, f0_wl, f0_corrected).astype(np.float32)


def _fresnel_nadir(n_w: np.ndarray) -> np.ndarray:
    """Fresnel reflectance at nadir (θ = 0)."""
    return ((1.0 - n_w) / (1.0 + n_w)) ** 2


# ---------------------------------------------------------------------------
# CorrectMixin
# ---------------------------------------------------------------------------

class CorrectMixin:
    """Mixin that adds :meth:`to_l1r` and :meth:`to_l2r` to the Image class."""

    # ------------------------------------------------------------------ #
    # Shared internal: compute rho_at_sensor                              #
    # ------------------------------------------------------------------ #

    def _compute_rho_at_sensor(self, cal_model=None) -> xr.DataArray:
        """TOA reflectance from radiance_at_sensor, with optional calibration.

        Returns a lazy DataArray (wavelength, y, x).
        """
        wavelength = np.asarray(self.wavelength, dtype=np.float64)
        doy      = self.acq_time_z.timetuple().tm_yday
        fwhm     = self.fwhm
        if fwhm is not None:
            # Gaussian SRF evaluated at Coddington native resolution
            f0 = _solar_irradiance(wavelength, doy, fwhm=fwhm)
        else:
            srf_data = self.srf   # tabulated RSR for multispectral sensors
            if srf_data is not None:
                srf_wl, srf_matrix = srf_data
                f0 = _solar_irradiance(wavelength, doy,
                                       srf_wl=srf_wl, srf=srf_matrix)
            else:
                f0 = _solar_irradiance(wavelength, doy)

        f0_da   = xr.DataArray(f0, dims=["wavelength"],
                               coords={"wavelength": self.in_ds.wavelength})
        cos_sz  = np.cos(np.deg2rad(self.in_ds["sun_zenith"]))    # (y, x)

        rho_t = (math.pi * self.in_ds["radiance_at_sensor"]) / (f0_da * cos_sz)

        if cal_model is not None:
            image_wl = xr.DataArray(wavelength, dims=["wavelength"])
            coeffs   = cal_model.coeffs.interp(wavelength=image_wl, method="linear")
            if cal_model.model_name == "ratio":
                rho_t = rho_t * coeffs["gain"]
            else:
                rho_t = coeffs["a"] * rho_t + coeffs["b"]

        return rho_t.assign_attrs(
            standard_name="toa_bidirectional_reflectance",
            units="1",
            long_name="Top-of-atmosphere reflectance",
        )

    def _compute_water_mask(
        self, rho_t: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """NDWI and water mask derived from rho_at_sensor.

        Both are evaluated eagerly (2-D arrays, negligible memory).

        Returns
        -------
        ndwi_da, mask_water_da
        """
        wavelength = np.asarray(self.wavelength, dtype=np.float64)
        green_i = int(np.abs(wavelength - 550.0).argmin())
        nir_i   = int(np.abs(wavelength - 800.0).argmin())

        rho_green = rho_t.isel(wavelength=green_i).values
        rho_nir   = rho_t.isel(wavelength=nir_i).values
        denom     = rho_green + rho_nir
        ndwi      = np.where(denom != 0.0,
                             (rho_green - rho_nir) / denom,
                             np.nan).astype(np.float32)

        coords_2d = {"y": self.in_ds.y, "x": self.in_ds.x}
        ndwi_da = xr.DataArray(
            ndwi, dims=["y", "x"], coords=coords_2d,
            attrs={"long_name": "Normalised Difference Water Index", "units": "1"},
        )
        mask_water_da = xr.DataArray(
            (ndwi > 0).astype(np.uint8), dims=["y", "x"], coords=coords_2d,
            attrs={
                "long_name": "Water pixel mask",
                "flag_values": np.array([0, 1], dtype=np.uint8),
                "flag_meanings": "non_water water",
            },
        )
        return ndwi_da, mask_water_da

    # ------------------------------------------------------------------ #
    # to_l1r                                                               #
    # ------------------------------------------------------------------ #

    def to_l1r(self, output: str, cal_model=None) -> "Image":
        """Convert at-sensor radiance to TOA reflectance (L1C → L1R).

        Requires ``radiance_at_sensor`` in ``in_ds``.

        Parameters
        ----------
        output : str
            Path to the output Zarr directory store.
        cal_model : CalibrationModel, optional
            When provided, the per-wavelength calibration is applied to
            ``rho_at_sensor`` after the radiance → reflectance conversion.

        Returns
        -------
        Image
            New Image whose ``in_ds`` contains ``rho_at_sensor``,
            ``ndwi``, ``mask_water``, and all geometry from the input.
        """
        if "radiance_at_sensor" not in self.in_ds:
            raise KeyError(
                "'radiance_at_sensor' not found in image dataset. "
                f"Available: {list(self.in_ds.data_vars)}"
            )

        log.info("to_l1r: computing rho_at_sensor for '%s'", self.image_name)

        rho_t = self._compute_rho_at_sensor(cal_model)
        ndwi_da, mask_water_da = self._compute_water_mask(rho_t)

        ds_out = self.in_ds.assign(
            rho_at_sensor=rho_t,
            ndwi=ndwi_da,
            mask_water=mask_water_da,
        )
        ds_out.attrs["processing_level"] = "L1R"
        ds_out.attrs["l1r_applied_at"]   = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        if cal_model is not None:
            import json
            ds_out.attrs["cal_model"] = json.dumps(cal_model.to_dict())

        result = copy.copy(self)
        result.in_ds = ds_out
        result.level = "L1R"
        result.to_zarr(output)
        log.info("L1R image written → %s", output)
        return result

    # ------------------------------------------------------------------ #
    # to_l2r                                                               #
    # ------------------------------------------------------------------ #

    def to_l2r(
        self,
        output: str,
        cal_model=None,
        backend: str = "cpu",
        n_workers: int = -1,
        window_size: int = 256,
    ) -> "Image":
        """Atmospheric correction: L1C → L2R.

        Computes rho_at_sensor internally (does *not* require a pre-existing
        L1R file on disk).  Requires ancillary atmospheric variables; use
        :func:`aabim.ancillary.add_ancillary` to fetch and attach MERRA-2
        data first.

        Parameters
        ----------
        output : str
            Path to the output Zarr directory store.
        cal_model : CalibrationModel, optional
            Applied to ``rho_at_sensor`` before atmospheric correction.
        backend : {"cpu", "gpu"}
            "cpu" dispatches windows to a ProcessPoolExecutor.
            "gpu" processes tiles sequentially with CuPy.
        n_workers : int, default -1
            Number of CPU worker processes (-1 = all logical cores).
        window_size : int, default 256
            Processing tile size in pixels.  Also used as the Zarr chunk
            size in y/x so that each worker writes complete chunks.

        Returns
        -------
        Image
            New Image backed by the output Zarr store.
        """
        if "radiance_at_sensor" not in self.in_ds:
            raise KeyError("'radiance_at_sensor' not found — image must be L1C.")

        missing = [v for v in _L2R_REQUIRED_VARS if v not in self.in_ds]
        if missing:
            raise KeyError(
                f"Variables missing from image dataset: {missing}. "
                "Run add_ancillary() before to_l2r()."
            )

        import os
        if n_workers == -1:
            n_workers = os.cpu_count() or 1

        log.info(
            "to_l2r: '%s'  backend=%s  workers=%d  window=%d",
            self.image_name, backend, n_workers, window_size,
        )

        # --- Step 1: build L1R dataset (lazy) ----------------------------
        rho_t = self._compute_rho_at_sensor(cal_model)
        ndwi_da, mask_water_da = self._compute_water_mask(rho_t)

        # Materialise rho_at_sensor into a temporary in-memory dataset
        # so the input Zarr is self-contained for worker reads.
        ds_l1r = self.in_ds.assign(
            rho_at_sensor=rho_t,
            ndwi=ndwi_da,
            mask_water=mask_water_da,
        )
        ds_l1r.attrs["processing_level"] = "L1R"

        # --- Step 2: ensure LUT caches exist on disk --------------------
        from aabim.data.atmospheric.atmospheric import AerLUT, GasLUT

        aer_lut = AerLUT.from_image(self, backend=backend)
        gas_lut = GasLUT.from_image(self, backend=backend)
        aer_cache = str(aer_lut.cache_path)
        gas_cache = str(gas_lut.cache_path)

        # --- Step 3: pre-initialise the output Zarr store ---------------
        n_wl   = len(self.wavelength)
        n_rows = self.n_rows
        n_cols = self.n_cols
        wavelength = np.asarray(self.wavelength, dtype=np.float32)

        import dask.array as da
        nan_3d = da.full(
            (n_wl, n_rows, n_cols), np.nan, dtype=np.float32,
            chunks=(n_wl, window_size, window_size),
        )
        coords_3d = {
            "wavelength": self.in_ds.wavelength,
            "y": self.in_ds.y,
            "x": self.in_ds.x,
        }
        _attrs3 = {"units": "1"}

        ds_out = ds_l1r.assign(
            rho_s     =xr.DataArray(nan_3d, dims=["wavelength","y","x"],
                                    coords=coords_3d, attrs={**_attrs3,
                                    "long_name": "Surface reflectance"}),
            rho_w     =xr.DataArray(nan_3d, dims=["wavelength","y","x"],
                                    coords=coords_3d, attrs={**_attrs3,
                                    "long_name": "Water-leaving reflectance"}),
            rho_w_gl21=xr.DataArray(nan_3d, dims=["wavelength","y","x"],
                                    coords=coords_3d, attrs={**_attrs3,
                                    "long_name": "Water-leaving reflectance (Gao & Li 2021)"}),
        )
        ds_out.attrs["processing_level"] = "L2R"
        ds_out.attrs["backend"]          = backend
        ds_out.attrs["aod_source"]       = "merra2"
        ds_out.attrs["l2r_applied_at"]   = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        if cal_model is not None:
            import json
            ds_out.attrs["cal_model"] = json.dumps(cal_model.to_dict())

        # Write the full dataset (NaN-filled 3D arrays + everything else)
        result_tmp = copy.copy(self)
        result_tmp.in_ds = ds_out
        result_tmp.level = "L2R"
        result_tmp.to_zarr(output)
        log.info("Output Zarr initialised → %s", output)

        # --- Step 4: write L1R input to a temp Zarr for workers ----------
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp(prefix="aabim_l1r_")
        in_path = str(Path(tmp_dir) / "l1r_input.zarr")
        try:
            # Write rho_at_sensor + geometry to a temp Zarr for workers to read
            ds_worker_in = ds_l1r[
                ["rho_at_sensor", "sun_zenith", "view_zenith",
                 "sun_azimuth", "view_azimuth", "relative_azimuth"]
            ]
            ds_worker_in.to_zarr(in_path, mode="w", consolidated=True)

            # --- Step 5: build per-window arguments ----------------------
            aod550   = float(self.in_ds["aerosol_optical_depth_at_550_nm"].values)
            pressure = float(self.in_ds["surface_air_pressure"].values)
            water    = float(self.in_ds["atmosphere_mass_content_of_water_vapor"].values)
            ozone    = float(self.in_ds["equivalent_thickness_at_stp_of_atmosphere_ozone_content"].values)
            z_alt    = float(self.z.values)

            mask_water = mask_water_da.values.astype(bool)    # (n_rows, n_cols)
            n_w   = get_water_refractive_index(30, 12, wavelength)
            rho_f = _fresnel_nadir(n_w).astype(np.float32)    # (n_wl,)
            nir_i = int(np.abs(wavelength - 800.0).argmin())

            windows    = self.create_windows(window_size)
            window_args = [
                {
                    "in_path":          in_path,
                    "out_path":         output,
                    "row_off":          int(w.row_off),
                    "col_off":          int(w.col_off),
                    "height":           int(w.height),
                    "width":            int(w.width),
                    "aod550":           aod550,
                    "pressure":         pressure,
                    "water":            water,
                    "ozone":            ozone,
                    "z":                z_alt,
                    "mask_water_window": mask_water[
                        w.row_off : w.row_off + w.height,
                        w.col_off : w.col_off + w.width,
                    ],
                    "rho_f":            rho_f,
                    "nir_i":            nir_i,
                    "backend":          backend,
                }
                for w in windows
            ]

            # --- Step 6: dispatch -----------------------------------------
            if backend == "gpu":
                _worker_init(aer_cache, gas_cache, backend)
                for args in tqdm(window_args, desc="L2R correction (GPU)"):
                    _correction_worker(args)
            else:
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_worker_init,
                    initargs=(aer_cache, gas_cache, backend),
                ) as pool:
                    futures = {
                        pool.submit(_correction_worker, args): i
                        for i, args in enumerate(window_args)
                    }
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="L2R correction (CPU)",
                    ):
                        fut.result()   # re-raise any worker exception

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # --- Step 7: consolidate metadata and return Image ---------------
        zarr.consolidate_metadata(output)
        log.info("L2R image complete → %s", output)

        result = copy.copy(self)
        result.in_ds = xr.open_zarr(output)
        result.level = "L2R"
        return result
