# Standard library imports
import logging
from abc import ABC
import datetime

# Third party imports
import numpy as np
import pyproj
import xarray as xr
import affine
from rasterio.windows import Window

# aabim import
from aabim.utils import helper, astronomy
from aabim.image.tile import Tile
from aabim.image.sensor import Sensor

from .io import IOMixin
from .extract import ExtractMixin
from .calibrate import CalibrateMixin
from .correct import CorrectMixin


class Image(IOMixin, ExtractMixin, CalibrateMixin, CorrectMixin, ABC):
    """
    Image class is used as the base template for spectral imagery data structure in AABIM.
    This class is to be expanded by converter specific class as in ../converter/wise/read_pix.py

    The class is designed to be used with the xarray library to handle multi-dimensional arrays.

    Parameters
    ----------
    in_path: str
        Path to the input dataset
    in_ds: xarray.Dataset
        xarray Dataset object
    wavelength: np.ndarray
        Array of wavelength
    acq_time_z: datetime
        Image acquisition time in UTC
    z: float
        Altitude of the sensor
    y: np.ndarray
        Array of y coordinates
    x: np.ndarray
        Array of x coordinates
    lat: np.ndarray
        Array of latitude
    lon: np.ndarray
        Array of longitude
    n_rows: int
        Number of rows
    n_cols: int
        Number of columns
    Affine: Affine
        Affine transformation
    crs: pyproj.crs.CRS
        Coordinate Reference System

    Attributes
    ----------

    Methods
    -------
    from_zarr
    from_reve_nc
    decode_xr
    __init__
    __str__
    cal_coordinate
    cal_coordinate_grid
    cal_time
    cal_sun_geom
    get_valid_mask
    cal_valid_mask

    """

    def __init__(
        self,
        in_path: str,
        in_ds: xr.Dataset,
        image_name: str,
        wavelength: np.ndarray,
        acq_time_z: datetime.datetime,
        z: float,
        y: np.ndarray,
        x: np.ndarray,
        n_rows: int,
        n_cols: int,
        Affine,
        crs: pyproj.crs.CRS,
        level: str
    ):
        # Dataset accessor
        self.in_path = in_path
        self.in_ds = in_ds
        self.out_path = None
        self.out_ds = None

        self.image_name = image_name

        # Spectral attributes
        self.wavelength = wavelength

        # Spatiotemporal attributes
        self.acq_time_z = acq_time_z
        self.z = z
        self.y = y
        self.x = x
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.Affine = Affine
        self.CRS = crs

        self.level = level

        self.center_lon = None
        self.center_lat = None
        self.lon_grid = None
        self.lat_grid = None

        # Sensor attribute store metadata from the sensor in a dictionary
        self.sensor = Sensor(name="unknown", wavelengths=wavelength)

        # Optional attributes

        self.no_data = None
        self.valid_mask = None
        self.bad_band_list = None

        self.water_mask = None

        # Time attributes
        self.acq_time_local = None
        self.central_lon_local_timezone = None

        # Geometric attributes
        self.sun_zenith = None
        self.sun_azimuth = None
        self.view_zenith = None
        self.view_azimuth = None
        self.relative_azimuth = None

        # Pixel location on the sensor array
        self.sample_index = None
        # Scan line location on the image
        self.line_index = None

    def __str__(self):
        return f"""
        Image from sensor {self.sensor} acquired on {self.acq_time_z.strftime('%Y-%m-%d %H:%M:%SZ')}
        Central longitude: {self.center_lon:.3f}E
        Central latitude: {self.center_lat:.3f}N
        shape: x:{self.x.shape}, y:{self.y.shape}
        wavelength: {self.wavelength}
        """

    # def cal_coordinate(self, affine, n_rows, n_cols, crs):
    #     """
    #     Compute the pixel coordinates
    #
    #     Parameters
    #     ----------
    #     affine: Affine transformation
    #     n_rows: Number of rows
    #     n_cols: Number of column
    #     crs: pyproj CRS object
    #
    #     Returns
    #     -------
    #     x: projected pixel coordinates
    #     y: projected pixel coordinates
    #     longitude:
    #     latitude:
    #     center_longitude:
    #     center_latitude:
    #     """
    #     logging.debug("calculating pixel coordinates")
    #
    #     # Define image projected coordinates from Affine tranformation
    #     # TODO: it appear that x, y doesn't match lon lat ....
    #     x, y = helper.transform_xy(
    #         affine, rows=list(range(n_rows)), cols=list(range(n_cols))
    #     )
    #
    #     # compute longitude and latitude from the projected coordinates
    #     transformer = pyproj.Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
    #
    #     xv, yv = np.meshgrid(x, y[0])
    #     lon, _ = transformer.transform(xv, yv)
    #     lon = lon[0, :]
    #
    #     xv, yv = np.meshgrid(x[0], y)
    #     _, lat = transformer.transform(xv, yv)
    #     lat = lat[:, 0]
    #
    #     # print(f"lat shape: {lat.shape}, lon shape: {lon.shape}")
    #     logging.info(
    #         f"n_rows({n_rows}) = y({len(y)}) = lat({len(lat)}) and n_cols({n_cols}) = x({len(x)}) = lon({len(lon)})"
    #     )
    #
    #     self.x, self.y, self.lon, self.lat = (x, y, lon, lat)
    #
    def cal_coordinate(self, affine, n_rows, n_cols):
        logging.debug("calculating pixel coordinates")

        # Projected coordinates
        x, y = helper.transform_xy(
            affine, rows=list(range(n_rows)), cols=list(range(n_cols))
        )

        logging.info(
            f"n_rows({n_rows}) = y({len(y)}) and " f"n_cols({n_cols}) = x({len(x)})"
        )

        self.x, self.y = x, y

    def expand_coordinate(self):
        # Transformer to WGS84
        transformer = pyproj.Transformer.from_crs(
            self.CRS.to_epsg(), 4326, always_xy=True
        )

        # 2D meshgrid for all pixel positions
        xv, yv = np.meshgrid(self.x, self.y)
        lon_grid, lat_grid = transformer.transform(xv, yv)

        central_lon = lon_grid[len(lat_grid[:, 0]) // 2, len(lon_grid[0, :]) // 2]
        central_lat = lat_grid[len(lat_grid[:, 0]) // 2, len(lon_grid[0, :]) // 2]

        self.lon_grid, self.lat_grid = lon_grid, lat_grid
        self.center_lon, self.center_lat = central_lon, central_lat

    def cal_time(self, central_lon, central_lat):
        """
        Compute local time of image acquisition

        Parameters
        ----------
        central_lon: central longitude
        central_lat: central latitude

        Returns
        -------
        x: projected pixel coordinates
        y: projected pixel coordinates
        longitude:
        latitude:
        center_longitude:
        center_latitude:
        """
        # find the local timezone from the central longitude and latitude
        tz = helper.findLocalTimeZone(central_lon, central_lat)

        self.acq_time_local = tz.convert(self.acq_time_z)
        logging.info(
            f"acquired UTC time:{self.acq_time_z}, and local time：{self.acq_time_local}"
        )

        offset_hours = self.acq_time_local.utcoffset().total_seconds() / 3600

        self.central_lon_local_timezone = offset_hours * 15
        logging.info(f"central longitude of {tz}:{self.central_lon_local_timezone}")

    def cal_sun_geom(self):
        """calculate solar zenith and azimuth angle for each pixel in the scene
        Parameters
        ----------
        acq_time_local: local time of image acquisition
        lat_grid: latitude grid
        lon_grid: longitude grid

        Returns
        -------
        sun_zenith: spatially resolved solar zenith [degree]
        sun_azimuth: spatially resolved solar azimuth [degree]
        """

        logging.debug("calculating solar zenith and azimuth")

        utc_offset = self.acq_time_local.utcoffset().total_seconds() / 3600

        self.sun_zenith, self.sun_azimuth = astronomy.sun_geom_noaa(
            self.acq_time_local, utc_offset, self.lat_grid, self.lon_grid
        )

        self.sun_zenith[~self.get_valid_mask()] = np.nan
        self.sun_azimuth[~self.get_valid_mask()] = np.nan

    def get_valid_mask(self, window: Window = None):
        """
        Get the valid mask for the enire image or a specific tile
        :param window: an object of class Window
        :return:[]
        """
        # logging.debug("geting valid mask")
        if self.valid_mask is None:
            self.cal_valid_mask()
        if window is None:
            return self.valid_mask
        else:
            y_start = int(window.row_off)
            y_stop = y_start + int(window.height)
            x_start = int(window.col_off)
            x_stop = x_start + int(window.width)
            # return self.valid_mask[tile.sline : tile.eline, tile.spixl : tile.epixl]
            return self.valid_mask[y_start:y_stop, x_start:x_stop]

    def cal_valid_mask(self):
        """
        Calculate the mask of valid pixel for the entire image (!= nodata)
        :return: valid_mask
        """
        logging.debug("calculating valid mask")
        if self.valid_mask is None:
            # Select the first variable with dim (wavelength, y, x) to compute the valid mask
            for name, data in self.in_ds.data_vars.items():
                if data.dims == ("wavelength", "y", "x"):
                    # TODO: some bands might be full of NA, refer to attribute bad_band_list
                    self.valid_mask = data.isel(wavelength=60).notnull().data
                    break

    def filter_bad_band(self):
        # filter image for bad bands
        bad_band_list = self.in_ds["radiance_at_sensor"].bad_band_list
        bad_band_list = [int(b) for b in bad_band_list]
        # if isinstance(bad_band_list, str):
        #     bad_band_list = str.split(bad_band_list, ", ")

        bad_band_list = np.array(bad_band_list)
        good_band_indices = np.where(bad_band_list == 1)[0]
        good_bands_slice = slice(min(good_band_indices), max(good_band_indices) + 1)

        self.wavelength = self.wavelength[good_band_indices]
        self.in_ds = self.in_ds.isel(wavelength=good_bands_slice)

    def mask_wavelength(self, wavelength):
        wavelength_mask = (self.wavelength >= min(wavelength)) & (
            self.wavelength <= max(wavelength)
        )

        self.wavelength = self.wavelength[wavelength_mask]
        self.in_ds = self.in_ds.sel(wavelength=self.wavelength)

    def read_band(self, bandindex, tile: Tile = None):
        """
        read DN for a given band
        :param bandindex: bandindex starts with 0
        :param tile: an instance of the Tile class used for tiled processing
        :return: re
        """
        pass

    def cal_relative_azimuth(self):
        """
        calculate relative azimuth angle
        :return:
        The angle between the viewing and the sun azimuth retricted to 0 - 180 degree, 0 facing towards the sun and 180 facing away from the sun
        """
        # In ACOLTIE and 6S relative azimuth is defined as theta_s - theta_v in the range 0 - 180
        # if 0  then theta_s = theta_v, viewing is in the incidence direction of the sun ray, looking away from it
        # if 180 then viewing is in the opposite direction of the sun's ray, looking toward it
        relative_azimuth = np.abs(self.view_azimuth - self.sun_azimuth)

        if 0 <= relative_azimuth.all() <= 180:
            self.relative_azimuth = relative_azimuth
        elif 180 < relative_azimuth.all() <= 360:
            self.relative_azimuth = 360 - relative_azimuth
        else:
            raise ValueError("relative azimuth is not in a valid 0 - 360 range")

    # @abstractmethod
    def cal_view_geom(self):
        pass

    @property
    def fwhm(self) -> np.ndarray | None:
        """Per-band sensor FWHM (nm), or ``None`` if not stored in the dataset.

        Reads ``in_ds["fwhm"]`` when present (written by
        :meth:`~aabim.converter.wise.read_pix.Pix.to_aabim_nc`).  Returns
        ``None`` when the dataset pre-dates FWHM storage, in which case
        callers may fall back to estimating the bandwidth from the wavelength
        spacing.
        """
        if "fwhm" in self.in_ds:
            return np.asarray(self.in_ds["fwhm"].values, dtype=np.float64)
        return None

    @property
    def srf(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Spectral response function as a ``(srf_wl, srf_matrix)`` pair.

        * ``srf_wl``     — 1-D float64 array ``(n_fine,)`` — fine wavelength
          axis in nm.
        * ``srf_matrix`` — 2-D float64 array ``(n_fine, n_bands)`` — per-band
          SRF values (peak-normalized to 1).

        Resolution order:

        1. **Tabulated RSR** — if the dataset contains ``srf`` and
           ``wavelength_srf`` variables (multispectral sensors).
        2. **Gaussian from FWHM** — if ``fwhm`` is stored, a Gaussian SRF is
           computed on-the-fly at 0.1 nm resolution (hyperspectral sensors
           such as WISE).
        3. Returns ``None`` when neither is available.
        """
        if "srf" in self.in_ds and "wavelength_srf" in self.in_ds:
            srf_wl = np.asarray(self.in_ds["wavelength_srf"].values, dtype=np.float64)
            srf_matrix = np.asarray(self.in_ds["srf"].values, dtype=np.float64)
            return srf_wl, srf_matrix
        fwhm = self.fwhm
        if fwhm is not None:
            from aabim.image.sensor import Sensor
            sensor = Sensor(
                name="", wavelengths=np.asarray(self.wavelength, dtype=np.float64),
                fwhm=fwhm,
            )
            srf_wl, srf_matrix = sensor.to_gaussian_srf(resolution=0.1)
            return srf_wl, srf_matrix.astype(np.float64)
        return None

    def update_attributes(self):
        self.wavelength = self.in_ds.wavelength.values
        self.y = self.in_ds.y
        self.x = self.in_ds.x
        self.n_rows = len(self.y)
        self.n_cols = len(self.x)
        # New affine origin
        a, b, c, d, e, f, g, h, i = self.Affine
        # New origin (top-left corner)
        c = self.x[0].values
        f = self.y[0].values
        self.Affine = affine.Affine(a, b, c, d, e, f)

    def crop(self, bbox: dict) -> "Image":
        """Crop image in place to a lat/lon bounding box.

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

        # x is ascending; y is descending (north-up raster convention)
        cropped_ds = self.in_ds.sel(x=slice(x_west, x_east), y=slice(y_north, y_south))

        # Update origin in grid_mapping to reflect the new spatial extent
        new_x0 = float(cropped_ds.x[0])
        new_y0 = float(cropped_ds.y[0])
        cropped_ds["grid_mapping"].attrs["affine_transform"] = (
            self.Affine.a, self.Affine.b, new_x0,
            self.Affine.d, self.Affine.e, new_y0,
        )

        self.in_ds = cropped_ds
        self.update_attributes()
        return self

    def create_windows(self, window_size: int) -> list[Window]:
        n_y = len(self.y)
        n_x = len(self.x)
        windows = []
        for y_start in range(0, n_y, window_size):
            y_end = min(y_start + window_size, n_y)
            height = y_end - y_start
            for x_start in range(0, n_x, window_size):
                x_end = min(x_start + window_size, n_x)
                width = x_end - x_start
                windows.append(Window(x_start, y_start, width, height))
        return windows

    # def output_rgb(self):
    #     xr_ds = self.in_ds
    #
    #     import matplotlib.pyplot as plt
    #     from skimage import exposure
    #
    #     def adjust_gamma(img):
    #         corrected = exposure.adjust_gamma(img, 1)
    #         return corrected
    #
    #     def find_nearest(array, value):
    #         array = np.asarray(array)
    #         idx = (np.abs(array - value)).argmin()
    #         return idx
    #
    #     560 - xr_ds["radiance_at_sensor"].coords["wavelength"]
    #
    #     find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 650)
    #
    #     red_band = find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 650)
    #     green_band = find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 550)
    #     blue_band = find_nearest(xr_ds["radiance_at_sensor"].coords["wavelength"], 450)
    #
    #     red_array = xr_ds.isel(wavelength=red_band)["radiance_at_sensor"]
    #     green_array = xr_ds.isel(wavelength=green_band)["radiance_at_sensor"]
    #     blue_array = xr_ds.isel(wavelength=blue_band)["radiance_at_sensor"]
    #
    #     y_ticks, x_ticks = list(range(0, self.n_rows, 1000)), list(
    #         range(0, self.n_cols, 1000)
    #     )
    #     cor_x, cor_y = helper.transform_xy(self.Affine, rows=y_ticks, cols=x_ticks)
    #
    #     rgb_data = np.zeros((red_array.shape[0], red_array.shape[1], 3), dtype=float)
    #     rgb_data[:, :, 0] = red_array
    #     rgb_data[:, :, 1] = green_array
    #     rgb_data[:, :, 2] = blue_array
    #     dst = adjust_gamma(rgb_data)
    #     dst[~self.get_valid_mask()] = 1.0
    #     # dst = rgb_data
    #     plt.imshow(dst)
    #     plt.xticks(ticks=x_ticks, labels=cor_x)
    #     plt.yticks(ticks=y_ticks, labels=cor_y, rotation=90)
    #     plt.show()
