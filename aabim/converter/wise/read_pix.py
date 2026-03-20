"""
read_pix.py — WISE .pix to aabim NetCDF converter.

The ``Pix`` class reads a WISE hyperspectral image directory in PCIGeomatics
.pix format and converts it to the aabim CF-1.0 NetCDF format.

Usage
-----
>>> pix = Pix("/path/to/ACI-13A.dpix")
>>> pix.to_aabim_nc("/path/to/ACI13.nc")

Input directory structure
-------------------------
*.pix          — radiometric data (GDAL PCIDISK driver)
*.pix.hdr      — ENVI header
*.glu          — geolocation LUT (across-track pixel mapping)
*.glu.hdr      — ENVI header for the glu file
*Navcor_sum.log — navigation summary (heading, pitch, roll, altitude …)
"""
from __future__ import annotations

import datetime
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pyproj
from osgeo import gdal
from tqdm import tqdm

from aabim.image.image import Image
from aabim.image.sensor import Sensor
from aabim.converter.wise.flightline import FlightLine
from aabim.utils import helper

gdal.UseExceptions()
log = logging.getLogger(__name__)


class Pix(Image):
    """WISE hyperspectral image read from a .pix directory.

    Extends :class:`~aabim.image.image.Image` with WISE-specific geometry
    computation (GLU-based view angles) and a converter to the aabim NetCDF
    format.

    Parameters
    ----------
    pix_dir : str or Path
        Path to the ``.dpix`` directory containing the ``.pix``, ``.glu``
        and navigation files.
    """

    def __init__(self, pix_dir: str | Path, bbox: dict | None = None):
        """
        Parameters
        ----------
        pix_dir : str or Path
            Path to the ``.dpix`` directory.
        bbox : dict, optional
            ``{"lon": (west, east), "lat": (south, north)}``.  When provided,
            the spatial crop is applied *before* geometry arrays are allocated,
            which avoids out-of-memory errors on large images.  Equivalent to
            calling :meth:`crop` afterwards, but much more memory-efficient.
        """
        t0 = time.perf_counter()

        pix_dir = str(pix_dir)
        if not os.path.isdir(pix_dir):
            raise ValueError(f"Directory does not exist: {pix_dir}")

        files = os.listdir(pix_dir)

        # --- ENVI header and .pix file --------------------------------------- #
        hdr_f = next(f for f in files if re.match(r".*\.pix\.hdr", f))
        pix_f = next(f for f in files if re.match(r".*\.pix$", f))
        self.hdr_f = os.path.join(pix_dir, hdr_f)
        self.pix_f = os.path.join(pix_dir, pix_f)

        self.header = helper.read_envi_hdr(hdr_f=self.hdr_f)
        log.debug("ENVI header: %s", self.hdr_f)

        image_name = re.findall(r".*(?=-L|-N)", pix_f)[0]
        log.debug("Image name: %s", image_name)

        wavelength = np.array(
            [float(w) for w in self.header["wavelength"].split(",")]
        )
        log.debug(
            "Wavelength: %d bands  [%.1f – %.1f nm]",
            len(wavelength), wavelength.min(), wavelength.max(),
        )

        acq_time_z = datetime.datetime.strptime(
            self.header["acquisition time"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=datetime.timezone.utc)
        log.debug("Acquisition time (UTC): %s", acq_time_z)

        n_rows = int(self.header["lines"])
        n_cols = int(self.header["samples"])
        log.debug("Raster size: %d rows × %d cols", n_rows, n_cols)

        map_info = self.header["map info"]
        Affine, crs, _ = helper.parse_mapinfo(map_info)
        log.debug("CRS: EPSG:%s", crs.to_epsg())

        y, x = self._compute_yx(Affine, n_rows, n_cols)

        # --- Early bbox crop (before geometry arrays are allocated) ----------- #
        self._crop_offset = (0, 0)   # (row_offset, col_offset)
        self._gdal_window = None
        if bbox is not None:
            transformer = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
            x_west,  y_south = transformer.transform(bbox["lon"][0], bbox["lat"][0])
            x_east,  y_north = transformer.transform(bbox["lon"][1], bbox["lat"][1])
            col_idx = np.where((x >= x_west) & (x <= x_east))[0]
            row_idx = np.where((y >= y_south) & (y <= y_north))[0]
            if col_idx.size == 0 or row_idx.size == 0:
                raise ValueError("bbox does not overlap the image extent.")
            c0, c1 = int(col_idx[0]), int(col_idx[-1]) + 1
            r0, r1 = int(row_idx[0]), int(row_idx[-1]) + 1
            self._gdal_window = (r0, c0, c1 - c0, r1 - r0)
            self._crop_offset = (r0, c0)
            x = x[c0:c1];  y = y[r0:r1]
            n_cols = len(x);  n_rows = len(y)
            log.debug(
                "Early bbox crop → %d rows × %d cols  window=%s",
                n_rows, n_cols, self._gdal_window,
            )

        # --- Scale factor ---------------------------------------------------- #
        scale_raw = [v for k, v in self.header.items() if "scale" in k][0]
        try:
            scale_factor = np.reciprocal(float(int(scale_raw)))
        except ValueError:
            scale_factor = float(scale_raw)
        self.scale_factor = scale_factor
        log.debug("Scale factor: %g  (raw header value: %s)", scale_factor, scale_raw)

        # --- No-data value --------------------------------------------------- #
        ignore = self.header.get("data ignore value", "")
        self.no_data = int(ignore) if ignore.strip() else -99999
        log.debug("No-data value: %d", self.no_data)

        # --- Data type ------------------------------------------------------- #
        dtype_code = int(self.header["data type"])
        self.dtype = {1: np.dtype("uint8"), 2: np.dtype("int16")}.get(
            dtype_code, np.dtype("float32")
        )
        log.debug("Data type: %s  (ENVI code %d)", self.dtype, dtype_code)

        # --- GDAL dataset ---------------------------------------------------- #
        self.src_ds = gdal.Open(self.pix_f)
        log.info("GDAL driver: %s  |  %s", self.src_ds.GetDriver().ShortName, self.pix_f)

        # --- GLU + navigation ------------------------------------------------ #
        try:
            glu_hdr_f = next(f for f in files if re.match(r".*\.glu\.hdr$", f))
            glu_f     = next(f for f in files if re.match(r".*\.glu$", f))
            nav_f     = next(f for f in files if re.match(r".*Navcor_sum\.log$", f))
            self.glu_hdr_f = os.path.join(pix_dir, glu_hdr_f)
            self.glu_f     = os.path.join(pix_dir, glu_f)
            self.nav_f     = os.path.join(pix_dir, nav_f)
            log.debug("GLU file: %s", self.glu_f)
            log.debug("Navigation log: %s", self.nav_f)
            self.flightline = FlightLine.from_wise_file(
                nav_sum_log=self.nav_f, glu_hdr=self.glu_hdr_f
            )
            sensor_z = self.flightline.height
            log.debug(
                "Flight line: %d samples × %d lines  altitude=%.1f m",
                self.flightline.samples, self.flightline.lines, sensor_z,
            )
        except StopIteration:
            log.warning("GLU / navigation files not found — view geometry unavailable.")
            self.glu_f = self.glu_hdr_f = self.nav_f = None
            self.flightline = None
            sensor_z = 0.0

        # --- Initialise base Image ------------------------------------------- #
        import xarray as xr
        super().__init__(
            in_path=self.pix_f,
            in_ds=xr.Dataset(),          # populated later by to_aabim_nc
            image_name=image_name,
            wavelength=wavelength,
            acq_time_z=acq_time_z,
            z=sensor_z,
            y=y,
            x=x,
            n_rows=n_rows,
            n_cols=n_cols,
            Affine=Affine,
            crs=crs,
            level="L1",
        )
        # Image.__init__ resets no_data to None — restore the value we parsed above
        self.no_data = int(ignore) if ignore.strip() else -99999
        fwhm_raw = self.header.get("fwhm", "")
        fwhm = (
            np.array([float(w) for w in fwhm_raw.replace("{", "").replace("}", "").split(",")])
            if fwhm_raw.strip()
            else None
        )
        self.sensor = Sensor(name="WISE", wavelengths=wavelength, fwhm=fwhm)

        # --- Geometry -------------------------------------------------------- #
        log.debug("Computing sun geometry …")
        self.expand_coordinate()
        self.cal_time(self.center_lon, self.center_lat)
        self.cal_sun_geom()
        log.debug(
            "Sun zenith: %.1f – %.1f °",
            float(np.nanmin(self.sun_zenith)), float(np.nanmax(self.sun_zenith)),
        )
        self.cal_view_geom()
        self.cal_relative_azimuth()

        log.info("Pix initialised in %.2f s", time.perf_counter() - t0)

    # ---------------------------------------------------------------------- #
    # Geometry                                                                #
    # ---------------------------------------------------------------------- #

    def cal_valid_mask(self):
        """Compute valid mask from GDAL when in_ds is empty, else delegate to Image."""
        if self.valid_mask is not None:
            return self.valid_mask
        if self.in_ds.data_vars:
            super().cal_valid_mask()
            return self.valid_mask
        # in_ds not yet populated (called during __init__): read first band from GDAL
        gdal_win = getattr(self, "_gdal_window", None)
        band = self.src_ds.GetRasterBand(1)
        if gdal_win is not None:
            r0, c0, win_x, win_y = gdal_win
            data = band.ReadAsArray(xoff=c0, yoff=r0, win_xsize=win_x, win_ysize=win_y)
        else:
            data = band.ReadAsArray()
        self.valid_mask = data != self.no_data
        log.debug("Valid mask: %d / %d px", self.valid_mask.sum(), self.valid_mask.size)
        return self.valid_mask

    @staticmethod
    def _compute_yx(affine, n_rows: int, n_cols: int):
        """Return projected y and x coordinate arrays from an Affine transform."""
        cols = np.arange(n_cols)
        rows = np.arange(n_rows)
        x = affine.c + cols * affine.a
        y = affine.f + rows * affine.e
        return y, x

    def cal_view_geom(self):
        """Compute per-pixel view zenith/azimuth from the geolocation LUT."""
        if self.flightline is None:
            log.warning("No flight line — skipping view geometry.")
            self.view_zenith = np.full((self.n_rows, self.n_cols), np.nan)
            self.view_azimuth = np.full((self.n_rows, self.n_cols), np.nan)
            self.sample_index = np.full((self.n_rows, self.n_cols), np.nan)
            self.line_index   = np.full((self.n_rows, self.n_cols), np.nan)
            return

        glu_data = gdal.Open(self.glu_f)
        nsamples_glu = glu_data.RasterXSize
        nlines_glu   = glu_data.RasterYSize
        if nsamples_glu != self.flightline.samples or nlines_glu != self.flightline.lines:
            raise ValueError("GLU dimensions do not match flight line.")

        x_glu = glu_data.GetRasterBand(1).ReadAsArray()
        y_glu = glu_data.GetRasterBand(2).ReadAsArray()

        v_zenith_fl, v_azimuth_fl = self.flightline.cal_view_geom()

        v_zenith  = np.full((self.n_rows, self.n_cols), np.nan)
        v_azimuth = np.full((self.n_rows, self.n_cols), np.nan)
        v_sample  = np.full((self.n_rows, self.n_cols), np.nan)
        v_line    = np.full((self.n_rows, self.n_cols), np.nan)

        row_off, col_off = getattr(self, "_crop_offset", (0, 0))
        for row in tqdm(range(self.flightline.lines), desc="Processing GLU"):
            xs, ys = x_glu[row], y_glu[row]
            rows_c, cols_c = helper.transform_rowcol(
                self.Affine, xs=xs, ys=ys, precision=5
            )
            rows_c = rows_c - row_off
            cols_c = cols_c - col_off
            mask = (
                (rows_c >= 0) & (rows_c < self.n_rows) &
                (cols_c >= 0) & (cols_c < self.n_cols)
            )
            rows_c, cols_c = rows_c[mask], cols_c[mask]

            v_zenith[rows_c, cols_c]  = v_zenith_fl[row][mask]
            v_azimuth[rows_c, cols_c] = v_azimuth_fl[row][mask]
            v_sample[rows_c, cols_c]  = np.arange(self.flightline.samples)[mask]
            v_line[rows_c, cols_c]    = row

        self.view_zenith  = helper.fill_na_2d(v_zenith)
        self.view_azimuth = helper.fill_na_2d(v_azimuth)
        self.sample_index = helper.fill_na_2d(v_sample)
        self.line_index   = helper.fill_na_2d(v_line)

        valid = self.cal_valid_mask()
        for arr in [self.view_zenith, self.view_azimuth,
                    self.sample_index, self.line_index]:
            arr[~valid] = np.nan

    def read_band(self, bandindex: int, tile=None):
        """Read digital numbers for band *bandindex* (0-based)."""
        band = self.src_ds.GetRasterBand(bandindex + 1)
        if tile is not None:
            return band.ReadAsArray(
                xoff=tile[2], yoff=tile[0],
                win_xsize=tile.xsize, win_ysize=tile.ysize,
            )
        return band.ReadAsArray()

    # ---------------------------------------------------------------------- #
    # Crop                                                                    #
    # ---------------------------------------------------------------------- #

    def crop(self, bbox: dict) -> "Pix":
        """Crop the image in place to a lat/lon bounding box.

        Must be called **before** :meth:`to_aabim_nc`.  The crop window is
        stored internally and applied lazily when GDAL bands are read during
        conversion, so no full-image data is ever loaded into memory.

        Parameters
        ----------
        bbox : dict
            ``{"lon": (west, east), "lat": (south, north)}``

        Returns
        -------
        self
        """
        transformer = pyproj.Transformer.from_crs(4326, self.CRS, always_xy=True)
        x_west, y_south = transformer.transform(bbox["lon"][0], bbox["lat"][0])
        x_east, y_north = transformer.transform(bbox["lon"][1], bbox["lat"][1])

        # x ascending, y descending (north-up raster)
        col_idx = np.where((self.x >= x_west) & (self.x <= x_east))[0]
        row_idx = np.where((self.y >= y_south) & (self.y <= y_north))[0]

        if col_idx.size == 0 or row_idx.size == 0:
            raise ValueError("Bounding box does not overlap the image extent.")

        col_start, col_stop = int(col_idx[0]),  int(col_idx[-1]) + 1
        row_start, row_stop = int(row_idx[0]),  int(row_idx[-1]) + 1

        # Store window for GDAL reads in to_aabim_nc()
        self._gdal_window = (row_start, col_start,
                             col_stop - col_start, row_stop - row_start)
        log.debug(
            "GDAL window: row_off=%d  col_off=%d  win_x=%d  win_y=%d",
            *self._gdal_window,
        )

        # Slice geometry arrays (already in memory)
        for attr in ("sun_zenith", "sun_azimuth", "view_zenith", "view_azimuth",
                     "relative_azimuth", "sample_index", "line_index", "valid_mask"):
            arr = getattr(self, attr, None)
            if arr is not None:
                setattr(self, attr, arr[row_start:row_stop, col_start:col_stop])

        # Update spatial attributes
        self.x      = self.x[col_start:col_stop]
        self.y      = self.y[row_start:row_stop]
        self.n_cols = len(self.x)
        self.n_rows = len(self.y)

        log.info(
            "Cropped to bbox %s → %d rows × %d cols", bbox, self.n_rows, self.n_cols
        )
        return self

    # ---------------------------------------------------------------------- #
    # Converter                                                               #
    # ---------------------------------------------------------------------- #

    def to_aabim_nc(self, out_path: str | None = None) -> None:
        """Write the image to the aabim CF-1.0 NetCDF format.

        Parameters
        ----------
        out_path : str, optional
            Output file path.  Defaults to replacing the ``.dpix`` directory
            extension with ``.nc``.
        """
        t0 = time.perf_counter()

        if out_path is None:
            out_path = re.sub(r"\.dpix$", ".nc", self.pix_f)

        gdal_win = getattr(self, "_gdal_window", None)  # (row_off, col_off, win_x, win_y)
        if gdal_win is not None:
            log.debug(
                "Writing cropped output: %d rows × %d cols  window=%s",
                self.n_rows, self.n_cols, gdal_win,
            )
        else:
            log.debug("Writing full image: %d rows × %d cols", self.n_rows, self.n_cols)
        log.debug("Output path: %s", out_path)

        self.create_reve_nc(out_path)

        # Radiance
        self.create_var_nc(
            name="radiance_at_sensor",
            type="i4",
            dims=("wavelength", "y", "x"),
            scale=self.scale_factor,
        )
        self.out_ds.variables["radiance_at_sensor"].bad_band_list = (
            self.header["bbl"].split(",  ")
        )
        self.out_ds.variables["radiance_at_sensor"].units = "uW cm-2 nm-1 sr-1"
        self.out_ds.variables["radiance_at_sensor"].standard_name = (
            "upwelling_radiance_per_unit_wavelength_in_air"
        )

        if self.sensor.fwhm is not None:
            fwhm_var = self.out_ds.createVariable("fwhm", "f4", ("wavelength",))
            fwhm_var[:] = self.sensor.fwhm
            fwhm_var.units = "nm"
            fwhm_var.long_name = "Sensor spectral bandwidth (full-width at half-maximum)"

        for band in tqdm(range(len(self.wavelength)), desc="Writing radiance"):
            gdal_band = self.src_ds.GetRasterBand(band + 1)
            if gdal_win is not None:
                row_off, col_off, win_x, win_y = gdal_win
                data = gdal_band.ReadAsArray(
                    xoff=col_off, yoff=row_off, win_xsize=win_x, win_ysize=win_y
                ).astype(float)
            else:
                data = gdal_band.ReadAsArray().astype(float)
            data = data * self.scale_factor
            data[data == 0] = self.no_data * self.scale_factor
            self.out_ds.variables["radiance_at_sensor"][band, :, :] = data
        log.debug("Radiance written: %d bands", len(self.wavelength))

        # Geometry
        geom_vars = {
            "sun_azimuth":      self.sun_azimuth,
            "sun_zenith":       self.sun_zenith,
            "view_azimuth":     self.view_azimuth,
            "view_zenith":      self.view_zenith,
            "relative_azimuth": self.relative_azimuth,
            "sample_index":     self.sample_index,
            "line_index":       self.line_index,
        }
        for var, data in tqdm(geom_vars.items(), desc="Writing geometry"):
            self.create_var_nc(name=var, type="i4", dims=("y", "x"),
                               scale=self.scale_factor)
            np.nan_to_num(data, copy=False, nan=self.no_data * self.scale_factor)
            self.out_ds.variables[var][:, :] = data

        self.out_ds.close()
        log.info("Pix → aabim NetCDF in %.2f s: %s", time.perf_counter() - t0, out_path)
