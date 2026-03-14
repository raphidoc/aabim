"""
lut.py — Aerosol and gas radiative-transfer look-up tables.

Design
------
* BaseLUT  — abstract class that owns the load → slice → build → query pipeline.
* AerLUT   — 4-variable aerosol LUT (rho_path, T, S, sky_glint).
* GasLUT   — 1-variable gas transmittance LUT.

Backend
-------
Pass ``backend="cpu"`` or ``backend="gpu"`` at construction time (or via the
``from_image`` factory).  The backend is stored on the instance and governs
both array allocation and interpolator type.

LUT loading — resolution order
--------------------------------
1. Sensor-sliced cache on disk (keyed by sensor name + wavelength hash).
   Located under ``cache_dir`` (default: ``~/.cache/aabim/lut``).
   Skips the expensive xarray interp step on repeated runs.
2. Bundled base LUT in ``aabim/data/atmospheric/``.
   Used during development or when the package ships with data files.
3. Remote download via pooch (``_registry.fetch_*``).
   Hash-verified, cached in the OS cache dir.

Steps 2 and 3 are handled transparently by ``_registry.py``.
Step 1 is handled by ``BaseLUT.load_or_create_sensor_lut``.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import scipy.interpolate as sp
import xarray as xr

from aabim.data.registry import fetch_aer_lut, fetch_gas_lut

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional GPU support — fall back gracefully when CuPy is unavailable
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    import cupyx.scipy.interpolate as cpx

    _CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    cpx = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

Backend = Literal["cpu", "gpu"]
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aabim"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wl_hash(wavelength: np.ndarray) -> str:
    """Short deterministic hash of a wavelength array used as a cache key."""
    return hashlib.md5(
        np.ascontiguousarray(wavelength, dtype=np.float32)
    ).hexdigest()[:12]


def xp(backend: Backend):
    """Return the array namespace (numpy or cupy) for *backend*."""
    if backend == "gpu":
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not installed; cannot use backend='gpu'.")
        return cp
    return np


def interp_cls(backend: Backend):
    """Return the RegularGridInterpolator class for *backend*."""
    return (
        cpx.RegularGridInterpolator if backend == "gpu" else sp.RegularGridInterpolator
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLUT(ABC):
    """
    Common pipeline: load raw LUT → slice to sensor wavelength (+ cache) →
    build interpolators → query per image window.

    Subclasses must implement:
        * ``lut_stem``      : str        — prefix used for the cache filename
        * ``lut_variables`` : list[str]  — variables to keep after slicing
        * ``fetch_raw``     : classmethod(sensor_name) → path string
        * ``build_interps`` : (ds, backend) → (interps_dict, points)
    """

    # ---- subclass contracts ------------------------------------------------

    @property
    @abstractmethod
    def lut_stem(self) -> str: ...

    @property
    @abstractmethod
    def lut_variables(self) -> list[str]: ...

    @classmethod
    @abstractmethod
    def fetch_raw(cls) -> str:
        """Return local path to the base LUT (downloading if necessary)."""
        ...

    @abstractmethod
    def build_interps(
        self, ds: xr.Dataset, backend: Backend
    ) -> Tuple[Dict[str, object], tuple]: ...

    # ---- construction ------------------------------------------------------

    def __init__(
        self,
        interps: Dict[str, object],
        points: tuple,
        ds_sliced: xr.Dataset,
        wavelength: np.ndarray,
        backend: Backend,
    ) -> None:
        self.interps = interps
        self.points = points
        self.ds_sliced = ds_sliced
        self.wavelength = wavelength
        self.backend = backend

    @classmethod
    def load_raw(cls) -> str:
        """Return base LUT dataset."""
        return xr.open_dataset(cls.fetch_raw())

    @classmethod
    def from_image(
        cls,
        image,
        backend: Backend = "cpu",
        cache_dir: Path | None = None,
    ) -> "BaseLUT":
        """
        Canonical factory: reads sensor name + wavelength from *image*,
        loads (or builds + caches) the sensor-sliced LUT, builds interpolators.

        Resolution order
        ----------------
        1. Sensor-sliced NetCDF in *cache_dir* (fast path).
        2. Base LUT from ``aabim/data/atmospheric/`` (bundled).
        3. Base LUT downloaded via pooch (remote fallback).
        Steps 2 and 3 are handled by ``cls.fetch_raw`` → ``_registry``.
        """
        sensor_name = image.sensor.name
        wavelength = np.asarray(image.wavelength, dtype=np.float32)

        ds_sliced, cache_path = cls.load_or_create_sensor_lut(
            sensor_name=sensor_name,
            wavelength=wavelength,
            cache_dir=cache_dir or DEFAULT_CACHE_DIR,
        )

        # build_interps needs backend before __init__ sets self.backend,
        # so we pass it explicitly and initialise the instance afterwards.
        instance = cls.__new__(cls)
        interps, points = instance.build_interps(ds_sliced, backend)
        BaseLUT.__init__(instance, interps, points, ds_sliced, wavelength, backend)
        instance.cache_path = cache_path
        return instance

    @classmethod
    def from_cache_file(
        cls,
        path: str | Path,
        backend: Backend = "cpu",
    ) -> "BaseLUT":
        """Construct a LUT directly from an already-interpolated cache file.

        Used by parallel worker processes that receive the cache path as an
        argument and need to rebuild the interpolators without an Image object.

        Parameters
        ----------
        path : str or Path
            Path to a sensor-sliced LUT NetCDF written by
            :meth:`load_or_create_sensor_lut`.
        backend : {"cpu", "gpu"}
        """
        ds = xr.open_dataset(path)
        instance = cls.__new__(cls)
        interps, points = instance.build_interps(ds, backend)
        BaseLUT.__init__(
            instance, interps, points, ds, ds.wavelength.values, backend
        )
        instance.cache_path = Path(path)
        return instance

    # ---- sensor-interpolation with disk cache --------------------------------------

    @classmethod
    def load_or_create_sensor_lut(
        cls,
        sensor_name: str,
        wavelength: np.ndarray,
        cache_dir: Path,
    ) -> tuple:
        """
        Return a (dataset, cache_path) pair for *sensor_name*.

        Reads from *cache_dir* if the interpolated file already exists; otherwise
        fetches the base LUT (bundled or remote), interpolates it, writes it to
        *cache_dir*, and returns it.

        Cache filename: ``{lut_stem}_{sensor_name}_{wl_hash}.nc``
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = (
            cache_dir / f"{cls._stem()}_{sensor_name}_{wl_hash(wavelength)}.nc"
        )

        if cache_path.exists():
            log.debug("Sensor LUT cache hit: %s", cache_path)
            return xr.open_dataset(cache_path), cache_path

        log.debug("Sensor LUT cache miss for %s — fetching base LUT …", sensor_name)
        ds_raw = cls.load_raw()
        ds_sliced = cls.interp_to_wavelength(ds_raw, wavelength, cls._variables())

        ds_sliced.to_netcdf(cache_path)
        log.debug("Sensor LUT written → %s", cache_path)
        return ds_sliced, cache_path

    # ---- static helpers ----------------------------------------------------

    @staticmethod
    def interp_to_wavelength(
        ds: xr.Dataset,
        wavelength: np.ndarray,
        variables: list[str],
    ) -> xr.Dataset:
        """Interpolate *ds* to *wavelength* (nm), keeping only *variables*."""
        wmin = float(ds.wavelength.min())
        wmax = float(ds.wavelength.max())
        if wavelength.min() < wmin or wavelength.max() > wmax:
            raise ValueError(
                f"Sensor wavelength [{wavelength.min()}, {wavelength.max()}] nm "
                f"are outside LUT range [{wmin}, {wmax}] nm."
            )
        return ds[variables].interp(wavelength=wavelength, method="linear")

    # ---- classmethod shims to read abstract properties ---------------------
    # object.__new__ avoids calling __init__ (which needs interps etc.)
    # so the property getter can be called on a bare instance.

    @classmethod
    def _stem(cls) -> str:
        return cls.lut_stem.fget(object.__new__(cls))  # type: ignore[attr-defined]

    @classmethod
    def _variables(cls) -> list[str]:
        return cls.lut_variables.fget(object.__new__(cls))  # type: ignore[attr-defined]

    # ---- shared pixel-packing utilities ------------------------------------

    @staticmethod
    def unpack_to_spatial(
        vals: np.ndarray,       # (n_pixels, n_wl) — already on CPU
        valid_mask: np.ndarray, # (ny, nx) bool
        n_wl: int,
    ) -> np.ndarray:
        """Place interpolated pixel values back into a (n_wl, ny, nx) array."""
        ny, nx = valid_mask.shape
        out = np.full((n_wl, ny * nx), np.nan, dtype=np.float32)
        out[:, valid_mask.ravel()] = vals.T
        return out.reshape(n_wl, ny, nx)

    def to_cpu(self, arr) -> np.ndarray:
        """Move *arr* to CPU numpy regardless of backend."""
        if self.backend == "gpu":
            return arr.get()
        return np.asarray(arr)


# ---------------------------------------------------------------------------
# AerLUT
# ---------------------------------------------------------------------------

AER_VARIABLES = [
    "atmospheric_reflectance_at_sensor",
    "total_scattering_trans_total",
    "spherical_albedo_total",
    "sky_glint_total",
]


class AerLUT(BaseLUT):
    """Aerosol radiative-transfer LUT (rho_path, T_scat, S_sph, sky_glint)."""

    @property
    def lut_stem(self) -> str:
        return "aer"

    @property
    def lut_variables(self) -> list[str]:
        return AER_VARIABLES

    @classmethod
    def fetch_raw(cls) -> str:
        """Return local path to the base aerosol LUT (download if needed)."""
        return fetch_aer_lut()

    def build_interps(
        self, ds: xr.Dataset, backend: Backend
    ) -> Tuple[Dict[str, object], tuple]:
        _xp = xp(backend)
        ic = interp_cls(backend)

        def arr(v):
            return _xp.asarray(v, dtype=_xp.float32)

        points = (
            arr(ds.sol_zen.values),
            arr(ds.view_zen.values),
            arr(ds.relative_azimuth.values),
            arr(ds.aot550.values),
            arr(ds.target_pressure.values),
            arr(ds.sensor_altitude.values),
        )
        kw = dict(method="linear", bounds_error=False, fill_value=float("nan"))
        interps = {
            "rho":       ic(points, arr(ds["atmospheric_reflectance_at_sensor"].values), **kw),
            "t":         ic(points, arr(ds["total_scattering_trans_total"].values),       **kw),
            "s":         ic(points, arr(ds["spherical_albedo_total"].values),             **kw),
            "sky_glint": ic(points, arr(ds["sky_glint_total"].values),                    **kw),
        }
        return interps, points

    def get_ra(self, image, window, aod550: float) -> xr.Dataset:
        """
        Interpolate aerosol RT quantities for all valid pixels in *window*.

        Returns an xr.Dataset with variables
        ``rho_path``, ``trans_ra``, ``spherical_albedo_ra``, ``sky_glint_ra``
        on dims ``(wavelength, y, x)``.
        """
        _xp = xp(self.backend)
        x, y, x_slice, y_slice = spatial_slices(image, window)
        valid_mask = image.get_valid_mask(window)
        ny, nx = valid_mask.shape
        n_wl = len(self.wavelength)

        out = empty_spatial_dataset(
            ["rho_path", "trans_ra", "spherical_albedo_ra", "sky_glint_ra"],
            self.wavelength, y, x,
        )

        n_pixels = int(valid_mask.sum())
        if n_pixels == 0:
            return out

        sub = image.in_ds.isel(x=x_slice, y=y_slice)
        sun  = _xp.asarray(sub["sun_zenith"].values[valid_mask],       dtype=_xp.float32)
        view = _xp.asarray(sub["view_zenith"].values[valid_mask],      dtype=_xp.float32)
        raa  = _xp.asarray(sub["relative_azimuth"].values[valid_mask], dtype=_xp.float32)
        press = float(image.in_ds["surface_air_pressure"].values)
        z     = float(image.z.values)

        xi = _xp.hstack([
            sun.reshape(-1, 1),
            view.reshape(-1, 1),
            raa.reshape(-1, 1),
            _xp.full((n_pixels, 1), aod550, dtype=_xp.float32),
            _xp.full((n_pixels, 1), press,  dtype=_xp.float32),
            _xp.full((n_pixels, 1), z,      dtype=_xp.float32),
        ])

        rho_v = self.to_cpu(self.interps["rho"](xi))
        t_v   = self.to_cpu(self.interps["t"](xi))
        s_v   = self.to_cpu(self.interps["s"](xi))
        g_v   = self.to_cpu(self.interps["sky_glint"](xi))

        out["rho_path"].values[:]            = self.unpack_to_spatial(rho_v, valid_mask, n_wl)
        out["trans_ra"].values[:]            = self.unpack_to_spatial(t_v,   valid_mask, n_wl)
        out["spherical_albedo_ra"].values[:] = self.unpack_to_spatial(s_v,   valid_mask, n_wl)
        out["sky_glint_ra"].values[:]        = self.unpack_to_spatial(g_v,   valid_mask, n_wl)
        return out

    def slice_for_stan(
        self,
        sol_zen: float,
        view_zen: float,
        raa: float,
        pressure: float,
        altitude: float,
    ) -> dict:
        """
        Return per-AOD grid arrays suitable for Stan's data block.

        Sweeps the ``aot550`` axis at fixed geometry, returning arrays of
        shape ``(n_aod, n_wl)`` for spectral quantities.
        """
        aod_grid = self.points[3]
        if self.backend == "gpu":
            aod_grid = aod_grid.get()
        n_aod = len(aod_grid)

        xi = np.hstack([
            np.full((n_aod, 1), sol_zen,  dtype=np.float32),
            np.full((n_aod, 1), view_zen, dtype=np.float32),
            np.full((n_aod, 1), raa,      dtype=np.float32),
            aod_grid.reshape(-1, 1),
            np.full((n_aod, 1), pressure, dtype=np.float32),
            np.full((n_aod, 1), altitude, dtype=np.float32),
        ])

        if self.backend == "gpu":
            xi_xp = cp.asarray(xi)
            rho = self.to_cpu(self.interps["rho"](xi_xp))
            t   = self.to_cpu(self.interps["t"](xi_xp))
            s   = self.to_cpu(self.interps["s"](xi_xp))
            g   = self.to_cpu(self.interps["sky_glint"](xi_xp))
        else:
            rho = self.interps["rho"](xi)
            t   = self.interps["t"](xi)
            s   = self.interps["s"](xi)
            g   = self.interps["sky_glint"](xi)

        return {
            "aod_grid":     np.asarray(aod_grid, dtype=np.float32),
            "rho_path_ra":  rho,
            "t_ra":         t,
            "s_ra":         s,
            "sky_glint_ra": g,
        }


# ---------------------------------------------------------------------------
# GasLUT
# ---------------------------------------------------------------------------

GAS_VARIABLES = ["global_gas_trans_total"]


class GasLUT(BaseLUT):
    """Gas transmittance LUT."""

    @property
    def lut_stem(self) -> str:
        return "gas"

    @property
    def lut_variables(self) -> list[str]:
        return GAS_VARIABLES

    @classmethod
    def fetch_raw(cls) -> str:
        """Return local path to the base gas LUT (download if needed)."""
        return fetch_gas_lut()

    def build_interps(
        self, ds: xr.Dataset, backend: Backend
    ) -> Tuple[Dict[str, object], tuple]:
        _xp = xp(backend)
        ic = interp_cls(backend)

        def arr(v):
            return _xp.asarray(v, dtype=_xp.float32)

        points = (
            arr(ds.sol_zen.values),
            arr(ds.view_zen.values),
            arr(ds.relative_azimuth.values),
            arr(ds.water.values),
            arr(ds.ozone.values),
            arr(ds.target_pressure.values),
            arr(ds.sensor_altitude.values),
        )
        kw = dict(method="linear", bounds_error=False, fill_value=float("nan"))
        interps = {
            "t_gas": ic(points, arr(ds["global_gas_trans_total"].values), **kw),
        }
        return interps, points

    def get_gas(self, image, window) -> xr.Dataset:
        """
        Interpolate gas transmittance for all valid pixels in *window*.

        Returns an xr.Dataset with variable ``t_gas`` on dims
        ``(wavelength, y, x)``.
        """
        _xp = xp(self.backend)
        x, y, x_slice, y_slice = spatial_slices(image, window)
        valid_mask = image.get_valid_mask(window)
        ny, nx = valid_mask.shape
        n_wl = len(self.wavelength)

        out = empty_spatial_dataset(["t_gas"], self.wavelength, y, x)

        n_pixels = int(valid_mask.sum())
        if n_pixels == 0:
            return out

        sub = image.in_ds.isel(x=x_slice, y=y_slice)
        sun  = _xp.asarray(sub["sun_zenith"].values[valid_mask],       dtype=_xp.float32)
        view = _xp.asarray(sub["view_zenith"].values[valid_mask],      dtype=_xp.float32)
        raa  = _xp.asarray(sub["relative_azimuth"].values[valid_mask], dtype=_xp.float32)

        water = float(image.in_ds["atmosphere_mass_content_of_water_vapor"].values)
        ozone = float(image.in_ds["equivalent_thickness_at_stp_of_atmosphere_ozone_content"].values)
        press = float(image.in_ds["surface_air_pressure"].values)
        z     = float(image.z.values)

        xi = _xp.hstack([
            sun.reshape(-1, 1),
            view.reshape(-1, 1),
            raa.reshape(-1, 1),
            _xp.full((n_pixels, 1), water, dtype=_xp.float32),
            _xp.full((n_pixels, 1), ozone, dtype=_xp.float32),
            _xp.full((n_pixels, 1), press, dtype=_xp.float32),
            _xp.full((n_pixels, 1), z,     dtype=_xp.float32),
        ])

        tgas_v = self.to_cpu(self.interps["t_gas"](xi))
        out["t_gas"].values[:] = self.unpack_to_spatial(tgas_v, valid_mask, n_wl)
        return out


# ---------------------------------------------------------------------------
# Module-level spatial utilities
# ---------------------------------------------------------------------------


def spatial_slices(image, window):
    """Return (x, y, x_slice, y_slice) for a given window (or None → full image)."""
    if window is not None:
        row_off, col_off = int(window.row_off), int(window.col_off)
        y_slice = slice(row_off, row_off + int(window.height))
        x_slice = slice(col_off, col_off + int(window.width))
        x = np.asarray(image.x[col_off: col_off + int(window.width)])
        y = np.asarray(image.y[row_off: row_off + int(window.height)])
    else:
        x_slice = y_slice = slice(None)
        x = image.x.values
        y = image.y.values
    return x, y, x_slice, y_slice


def empty_spatial_dataset(
    var_names: list[str],
    wavelength: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> xr.Dataset:
    """Allocate an all-NaN spatial dataset for the given variable names."""
    n_wl, ny, nx = len(wavelength), len(y), len(x)
    return xr.Dataset(
        data_vars={
            name: (
                ["wavelength", "y", "x"],
                np.full((n_wl, ny, nx), np.nan, dtype=np.float32),
            )
            for name in var_names
        },
        coords=dict(wavelength=wavelength, y=("y", y), x=("x", x)),
    )

if __name__ == "__main__":
    from aabim.image.image import Image
    image = Image.from_aabim_nc("/D/Data/WISE/ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l1r.nc")
    aer_lut = AerLUT.from_image(image)