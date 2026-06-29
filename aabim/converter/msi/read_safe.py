"""
read_safe.py — Sentinel-2 MSI .SAFE to aabim NetCDF converter.

The ``Safe`` class reads a Sentinel-2 L1C or L2A .SAFE product directory and
converts it to the aabim CF-1.0 NetCDF format.

Design
------
- Parses ``GRANULE/*/MTD_TL.xml`` for sensing time, UTM CRS/geotransform, and
  23×23 sun/view angle grids (5 km step), bilinearly interpolated to image
  pixel resolution via :func:`scipy.ndimage.zoom`.
- Reads per-band JP2 files from ``GRANULE/*/IMG_DATA/R{res}m/`` via GDAL.
- L1C products → ``rho_at_sensor`` (TOA reflectance, DN / 10 000).
- L2A products → ``rho_surface``   (BOA reflectance, DN / 10 000).
- Band wavelengths and FWHM are hard-coded per satellite (S2A / S2B / S2C)
  from ESA's official MSI spectral response functions.
- View angles are averaged across all bands and detectors to produce a single
  per-pixel zenith / azimuth grid.

Backend
-------
XML parsing  : stdlib ``xml.etree.ElementTree``
Raster I/O   : GDAL (JP2OpenJPEG / OPENJPEG driver)
Interpolation: ``scipy.ndimage.zoom`` (angle grids → image resolution)

Resolution / band mapping
-------------------------
10 m : B02, B03, B04, B08  (4 bands)
20 m : B01–B07, B8A, B11, B12  (10 bands) — default; most complete set
60 m : B01–B12 including B09, B10  (12 bands)

Usage
-----
>>> safe = Safe("/path/to/S2A_MSIL2A_*.SAFE")
>>> safe.to_aabim_nc("/path/to/output.nc")

>>> # With spatial crop and non-default resolution
>>> safe = Safe("/path/to/S2A_MSIL2A_*.SAFE", resolution=10,
...             bbox={"lon": (-64.2, -63.9), "lat": (47.8, 48.1)})
>>> safe.to_aabim_nc()
"""
from __future__ import annotations

import datetime
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import affine as _affine
import numpy as np
import pyproj
import xarray as xr
from osgeo import gdal
from scipy import ndimage
from tqdm import tqdm

from aabim.image.image import Image
from aabim.image.sensor import Sensor

gdal.UseExceptions()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSI band spectral properties  (ESA official SRF values, nm)
# Key: bandId string "0"…"12"  →  (band_name, central_wavelength_nm, FWHM_nm)
# ---------------------------------------------------------------------------
_MSI_BANDS: dict[str, dict[str, tuple[str, float, float]]] = {
    "S2A": {
        "0":  ("B01",  442.7,  21),
        "1":  ("B02",  492.4,  66),
        "2":  ("B03",  559.8,  36),
        "3":  ("B04",  664.6,  31),
        "4":  ("B05",  704.1,  15),
        "5":  ("B06",  740.5,  15),
        "6":  ("B07",  782.8,  20),
        "7":  ("B08",  832.8, 106),
        "8":  ("B8A",  864.7,  21),
        "9":  ("B09",  945.1,  20),
        "10": ("B10", 1373.5,  31),
        "11": ("B11", 1613.7,  91),
        "12": ("B12", 2202.4, 175),
    },
    "S2B": {
        "0":  ("B01",  442.2,  21),
        "1":  ("B02",  492.1,  66),
        "2":  ("B03",  559.0,  36),
        "3":  ("B04",  664.9,  31),
        "4":  ("B05",  703.8,  16),
        "5":  ("B06",  739.1,  15),
        "6":  ("B07",  779.7,  20),
        "7":  ("B08",  832.9, 106),
        "8":  ("B8A",  864.0,  22),
        "9":  ("B09",  943.2,  21),
        "10": ("B10", 1376.9,  30),
        "11": ("B11", 1610.4,  94),
        "12": ("B12", 2185.7, 185),
    },
    # S2C: same focal plane design as S2A; update when ESA publishes official SRF
    "S2C": {
        "0":  ("B01",  442.7,  21),
        "1":  ("B02",  492.4,  66),
        "2":  ("B03",  559.8,  36),
        "3":  ("B04",  664.6,  31),
        "4":  ("B05",  704.1,  15),
        "5":  ("B06",  740.5,  15),
        "6":  ("B07",  782.8,  20),
        "7":  ("B08",  832.8, 106),
        "8":  ("B8A",  864.7,  21),
        "9":  ("B09",  945.1,  20),
        "10": ("B10", 1373.5,  31),
        "11": ("B11", 1613.7,  91),
        "12": ("B12", 2202.4, 175),
    },
}

# Band names available at each native resolution (L2A product layout)
_RES_BANDS: dict[int, list[str]] = {
    10: ["B02", "B03", "B04", "B08"],
    20: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"],
    60: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A",
         "B09", "B10", "B11", "B12"],
}

_S2_ALTITUDE_M = 786_000.0   # nominal Sentinel-2 orbit altitude (m)


class Safe(Image):
    """Sentinel-2 MSI image read from a .SAFE product directory.

    Extends :class:`~aabim.image.image.Image` with MSI-specific metadata
    parsing and a converter to the aabim CF-1.0 NetCDF format.

    Parameters
    ----------
    safe_dir : str or Path
        Path to the .SAFE product directory (the one containing
        ``MTD_MSI*.xml``).
    resolution : {10, 20, 60}, optional
        Spatial resolution in metres.  Default ``20`` gives the most complete
        band set (10 bands).
    bbox : dict, optional
        ``{"lon": (west, east), "lat": (south, north)}``.  Spatial crop
        applied *before* data are loaded to avoid allocating the full
        10 980 × 10 980 array for large tiles.
    """

    def __init__(
        self,
        safe_dir: str | Path,
        resolution: int = 20,
        bbox: dict | None = None,
    ):
        t0 = time.perf_counter()

        safe_dir = Path(safe_dir)
        if not safe_dir.is_dir():
            raise ValueError(f"Directory does not exist: {safe_dir}")
        if resolution not in (10, 20, 60):
            raise ValueError(f"resolution must be 10, 20, or 60; got {resolution}")

        # ------------------------------------------------------------------ #
        # Parse product name                                                  #
        # ------------------------------------------------------------------ #
        prod_name = safe_dir.name
        m = re.match(r"(S2[ABC])_MSI(L1C|L2A)", prod_name)
        if m is None:
            raise ValueError(
                f"Cannot parse satellite / level from directory name: {prod_name!r}"
            )
        satellite  = m.group(1)                    # S2A, S2B, or S2C
        level      = m.group(2)                    # L1C or L2A
        image_name = prod_name.removesuffix(".SAFE")
        log.debug("Satellite: %s  Level: %s", satellite, level)

        # ------------------------------------------------------------------ #
        # Granule paths                                                       #
        # ------------------------------------------------------------------ #
        granule_dir = next((safe_dir / "GRANULE").iterdir())
        tile_xml    = granule_dir / "MTD_TL.xml"
        img_dir     = granule_dir / "IMG_DATA" / f"R{resolution}m"
        log.debug("Granule: %s", granule_dir.name)

        # ------------------------------------------------------------------ #
        # Tile metadata (sensing time, CRS, geotransform, angle grids)       #
        # ------------------------------------------------------------------ #
        tm = _parse_tile_xml(tile_xml, resolution)

        acq_time_z  = tm["sensing_time"]
        crs         = pyproj.CRS.from_epsg(tm["epsg"])
        n_rows_full = tm["n_rows"]
        n_cols_full = tm["n_cols"]
        ulx, uly    = tm["ulx"], tm["uly"]
        xdim, ydim  = tm["xdim"], tm["ydim"]   # ydim < 0 (north-up)

        x = ulx + np.arange(n_cols_full) * xdim   # pixel centres, ascending
        y = uly + np.arange(n_rows_full) * ydim   # pixel centres, descending
        Aff = _affine.Affine(xdim, 0, ulx, 0, ydim, uly)
        log.debug("Grid: %d×%d  UL=(%.0f, %.0f)  step=(%+d, %+d)",
                  n_rows_full, n_cols_full, ulx, uly, xdim, ydim)

        # ------------------------------------------------------------------ #
        # Band list (sorted by wavelength)                                    #
        # ------------------------------------------------------------------ #
        sat_bands  = _MSI_BANDS.get(satellite, _MSI_BANDS["S2A"])
        name_to_wl = {v[0]: (v[1], v[2]) for v in sat_bands.values()}
        band_names = sorted(_RES_BANDS[resolution], key=lambda b: name_to_wl[b][0])
        wavelength = np.array([name_to_wl[b][0] for b in band_names])
        fwhm       = np.array([name_to_wl[b][1] for b in band_names])
        log.debug("Bands (%d): %s  [%.0f–%.0f nm]",
                  len(band_names), band_names, wavelength.min(), wavelength.max())

        # ------------------------------------------------------------------ #
        # JP2 file discovery                                                  #
        # ------------------------------------------------------------------ #
        jp2_files: list[str] = []
        for bn in band_names:
            matches = sorted(img_dir.glob(f"*_{bn}_{resolution}m.jp2"))
            if not matches:
                raise FileNotFoundError(f"Band {bn} not found in {img_dir}")
            jp2_files.append(str(matches[0]))
        self._jp2_files = jp2_files
        log.debug("JP2 files: %d found", len(jp2_files))

        # ------------------------------------------------------------------ #
        # Optional bbox crop (early, before geometry allocation)             #
        # ------------------------------------------------------------------ #
        # _gdal_window: (row_off, col_off, win_x, win_y) — GDAL convention
        self._gdal_window: tuple | None = None
        if bbox is not None:
            tf = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
            xw, ys = tf.transform(bbox["lon"][0], bbox["lat"][0])
            xe, yn = tf.transform(bbox["lon"][1], bbox["lat"][1])
            col_idx = np.where((x >= xw) & (x <= xe))[0]
            row_idx = np.where((y >= ys) & (y <= yn))[0]
            if col_idx.size == 0 or row_idx.size == 0:
                raise ValueError("bbox does not overlap the image extent.")
            c0, c1 = int(col_idx[0]),  int(col_idx[-1]) + 1
            r0, r1 = int(row_idx[0]),  int(row_idx[-1]) + 1
            self._gdal_window = (r0, c0, c1 - c0, r1 - r0)
            x = x[c0:c1]
            y = y[r0:r1]
            log.debug("Early bbox crop → %d rows × %d cols  window=%s",
                      len(y), len(x), self._gdal_window)

        n_rows, n_cols = len(y), len(x)

        # ------------------------------------------------------------------ #
        # Initialise base Image                                               #
        # ------------------------------------------------------------------ #
        super().__init__(
            in_path    = str(safe_dir),
            in_ds      = xr.Dataset(),
            image_name = image_name,
            wavelength = wavelength,
            acq_time_z = acq_time_z,
            z          = _S2_ALTITUDE_M,
            y          = y,
            x          = x,
            n_rows     = n_rows,
            n_cols     = n_cols,
            Affine     = Aff,
            crs        = crs,
            level      = level,
        )

        # ------------------------------------------------------------------ #
        # Instance state                                                      #
        # ------------------------------------------------------------------ #
        self.sensor        = Sensor(name=f"MSI_{satellite}", wavelengths=wavelength, fwhm=fwhm)
        self._band_names   = band_names
        self._level        = level
        self._scale_factor = 1.0 / 10_000.0
        self.no_data       = 0

        # ------------------------------------------------------------------ #
        # Angle grids → full image resolution  (Image.__init__ reset them)   #
        # ------------------------------------------------------------------ #
        self.sun_zenith  = _zoom_angle(tm["sun_zenith_grid"],  n_rows_full, n_cols_full, self._gdal_window)
        self.sun_azimuth = _zoom_angle(tm["sun_azimuth_grid"], n_rows_full, n_cols_full, self._gdal_window)

        view_z, view_a = _build_view_angles(
            tm["view_grids"], n_rows_full, n_cols_full, self._gdal_window
        )
        self.view_zenith  = view_z
        self.view_azimuth = view_a

        # Relative azimuth: compute directly to avoid the buggy base method
        diff = np.abs(self.view_azimuth - self.sun_azimuth)
        diff[diff > 180.0] = 360.0 - diff[diff > 180.0]
        self.relative_azimuth = diff

        log.debug("Sun zenith:  %.1f–%.1f °",
                  float(np.nanmin(self.sun_zenith)),  float(np.nanmax(self.sun_zenith)))
        log.debug("View zenith: %.1f–%.1f °",
                  float(np.nanmin(self.view_zenith)), float(np.nanmax(self.view_zenith)))

        # ------------------------------------------------------------------ #
        # Geographic coordinates (needed by downstream methods)              #
        # ------------------------------------------------------------------ #
        self.expand_coordinate()
        self.cal_time(self.center_lon, self.center_lat)

        log.info(
            "Safe initialised in %.2f s  (%s  %d bands  %d×%d px)",
            time.perf_counter() - t0, level, len(wavelength), n_rows, n_cols,
        )

    # ---------------------------------------------------------------------- #
    # Mandatory abstract override                                             #
    # ---------------------------------------------------------------------- #

    def cal_view_geom(self):
        """View angles are read from the XML angle grids; this is a no-op."""

    # ---------------------------------------------------------------------- #
    # Converter                                                               #
    # ---------------------------------------------------------------------- #

    def to_aabim_nc(self, out_path: str | None = None) -> None:
        """Write the image to the aabim CF-1.0 NetCDF format.

        Parameters
        ----------
        out_path : str, optional
            Output ``.nc`` path.  Defaults to ``<safe_dir_parent>/<image_name>.nc``.
        """
        t0 = time.perf_counter()

        if out_path is None:
            out_path = str(Path(self.in_path).with_suffix(".nc"))
        log.debug("Output: %s", out_path)
        log.debug("Shape: %d rows × %d cols × %d bands",
                  self.n_rows, self.n_cols, len(self.wavelength))

        self.create_reve_nc(out_path)

        # ------------------------------------------------------------------ #
        # Reflectance variable                                                #
        # ------------------------------------------------------------------ #
        var_name = "rho_surface" if self._level == "L2A" else "rho_at_sensor"
        self.create_var_nc(
            name  = var_name,
            type  = "i4",
            dims  = ("wavelength", "y", "x"),
            scale = self._scale_factor,
        )
        # Save no_data before geometry create_var_nc calls overwrite self.no_data
        refl_no_data = self.no_data
        self.out_ds.variables[var_name].bad_band_list = [1] * len(self.wavelength)

        fwhm_var = self.out_ds.createVariable("fwhm", "f4", ("wavelength",))
        fwhm_var.units = "nm"
        fwhm_var.long_name = "Sensor spectral bandwidth (full-width at half-maximum)"
        fwhm_var[:] = self.sensor.fwhm

        gdal_win = self._gdal_window
        for i, jp2_path in enumerate(tqdm(self._jp2_files, desc="Writing bands")):
            ds  = gdal.Open(jp2_path)
            bnd = ds.GetRasterBand(1)
            if gdal_win is not None:
                r0, c0, wc, wr = gdal_win
                dn = bnd.ReadAsArray(xoff=c0, yoff=r0, win_xsize=wc, win_ysize=wr)
            else:
                dn = bnd.ReadAsArray()
            ds = None   # close GDAL dataset

            data_f = dn.astype(np.float32) * self._scale_factor
            # DN == 0 is ESA's no-data sentinel; map to the netCDF fill value
            data_f[dn == 0] = refl_no_data * self._scale_factor
            self.out_ds.variables[var_name][i, :, :] = data_f

        log.debug("%s written (%d bands)", var_name, len(self.wavelength))

        # ------------------------------------------------------------------ #
        # Geometry variables                                                  #
        # ------------------------------------------------------------------ #
        geom_vars = {
            "sun_zenith":       self.sun_zenith,
            "sun_azimuth":      self.sun_azimuth,
            "view_zenith":      self.view_zenith,
            "view_azimuth":     self.view_azimuth,
            "relative_azimuth": self.relative_azimuth,
        }
        for var, data in tqdm(geom_vars.items(), desc="Writing geometry"):
            self.create_var_nc(name=var, type="f4", dims=("y", "x"), scale=1.0)
            self.out_ds.variables[var][:, :] = data.astype(np.float32)

        self.out_ds.close()
        log.info("Safe → aabim NetCDF in %.2f s: %s",
                 time.perf_counter() - t0, out_path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_tile_xml(tile_xml: Path, resolution: int) -> dict:
    """Extract sensing time, geocoding and angle grids from ``MTD_TL.xml``."""
    tree = ET.parse(tile_xml)
    root = tree.getroot()

    sensing_time = datetime.datetime.strptime(
        root.find(".//{*}SENSING_TIME").text.strip(),
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ).replace(tzinfo=datetime.timezone.utc)

    epsg = int(root.find(".//{*}HORIZONTAL_CS_CODE").text.strip().split(":")[-1])

    # Pick the Size / Geoposition elements for the requested resolution
    geo    = root.find(".//{*}Tile_Geocoding")
    sizes  = {int(s.get("resolution")): s for s in geo.findall("{*}Size")}
    geopos = {int(g.get("resolution")): g for g in geo.findall("{*}Geoposition")}
    sz = sizes[resolution]
    gp = geopos[resolution]
    n_rows = int(sz.find("{*}NROWS").text)
    n_cols = int(sz.find("{*}NCOLS").text)
    ulx    = float(gp.find("{*}ULX").text)
    uly    = float(gp.find("{*}ULY").text)
    xdim   = float(gp.find("{*}XDIM").text)
    ydim   = float(gp.find("{*}YDIM").text)

    angles = root.find(".//{*}Tile_Angles")

    sun   = angles.find("{*}Sun_Angles_Grid")
    sun_z = _parse_angle_grid(sun.find("{*}Zenith"))
    sun_a = _parse_angle_grid(sun.find("{*}Azimuth"))

    view_grids = [
        {
            "bandId":     vag.get("bandId"),
            "detectorId": vag.get("detectorId"),
            "zenith":     _parse_angle_grid(vag.find("{*}Zenith")),
            "azimuth":    _parse_angle_grid(vag.find("{*}Azimuth")),
        }
        for vag in angles.findall("{*}Viewing_Incidence_Angles_Grids")
    ]

    return {
        "sensing_time":     sensing_time,
        "epsg":             epsg,
        "n_rows":           n_rows,
        "n_cols":           n_cols,
        "ulx":              ulx,
        "uly":              uly,
        "xdim":             xdim,
        "ydim":             ydim,
        "sun_zenith_grid":  sun_z,
        "sun_azimuth_grid": sun_a,
        "view_grids":       view_grids,
    }


def _parse_angle_grid(el: ET.Element) -> np.ndarray:
    """Parse a ``<Zenith>`` or ``<Azimuth>`` element into a 2-D float32 array."""
    rows = el.find("{*}Values_List").findall("{*}VALUES")
    return np.array(
        [
            [np.nan if v == "NaN" else float(v) for v in row.text.split()]
            for row in rows
        ],
        dtype=np.float32,
    )


def _zoom_angle(
    sparse: np.ndarray,
    n_rows_full: int,
    n_cols_full: int,
    crop_window: tuple | None,
) -> np.ndarray:
    """Bilinearly zoom a sparse 23×23 angle grid to image pixel resolution.

    NaN cells (detector gaps) are filled with the grid nanmean before zooming
    so that scipy.ndimage.zoom receives a clean array.  After zooming, the
    result is optionally cropped to the bbox window.

    Parameters
    ----------
    sparse : ndarray, shape (n_ang_r, n_ang_c)
        Sparse angle values (may contain NaN).
    n_rows_full, n_cols_full : int
        Full-tile pixel dimensions (before any bbox crop).
    crop_window : (row_off, col_off, win_x, win_y) or None
        GDAL-style crop window applied after zooming to the full tile.
    """
    filled = sparse.copy()
    nan_mask = np.isnan(filled)
    if nan_mask.any():
        filled[nan_mask] = np.nanmean(filled)

    zoomed = ndimage.zoom(
        filled,
        zoom=(n_rows_full / filled.shape[0], n_cols_full / filled.shape[1]),
        order=1,
    )
    zoomed = zoomed[:n_rows_full, :n_cols_full]

    if crop_window is not None:
        r0, c0, wc, wr = crop_window
        zoomed = zoomed[r0:r0 + wr, c0:c0 + wc]

    return zoomed.astype(np.float32)


def _build_view_angles(
    view_grids: list[dict],
    n_rows_full: int,
    n_cols_full: int,
    crop_window: tuple | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-band/detector view angle grids and zoom to image resolution.

    For each ``bandId``, detector grids are nanmean-ed first (they are
    disjoint across the swath with NaN where a detector does not cover).
    The per-band means are then averaged across all bands to yield a single
    view zenith / azimuth grid.
    """
    by_band_z: dict[str, list[np.ndarray]] = defaultdict(list)
    by_band_a: dict[str, list[np.ndarray]] = defaultdict(list)
    for vg in view_grids:
        by_band_z[vg["bandId"]].append(vg["zenith"])
        by_band_a[vg["bandId"]].append(vg["azimuth"])

    # nanmean across detectors per band, then across bands
    band_z = np.nanmean(
        np.stack([np.nanmean(grids, axis=0) for grids in by_band_z.values()]),
        axis=0,
    )
    band_a = np.nanmean(
        np.stack([np.nanmean(grids, axis=0) for grids in by_band_a.values()]),
        axis=0,
    )

    return (
        _zoom_angle(band_z, n_rows_full, n_cols_full, crop_window),
        _zoom_angle(band_a, n_rows_full, n_cols_full, crop_window),
    )
