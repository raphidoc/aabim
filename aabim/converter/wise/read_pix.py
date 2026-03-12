# Standard library imports
import os
import time
import datetime
import math

# Third party imports
from osgeo import gdal
from tqdm import tqdm
import numpy as np
import re
import netCDF4
from p_tqdm import p_uimap
import xarray as xr
import zarr
import numcodecs
import logging


# REVERIE import
from reverie.image import ReveCube
from .flightline import FlightLine
from reverie.utils import helper
from reverie.image.tile import Tile
from reverie.utils.cf_aliases import get_cf_std_name

# from reverie.utils.tile import Tile

gdal.UseExceptions()


class Pix(ReveCube):
    """
    This class expand ReveCube() to read the WISE images in .pix format, compute the observation geometry and convert
     that data to the NetCDF CF reverie format.

    DateCube with extension .pix (PCIGeomatica) cand be read with the GDAL driver PCIDISK
    They are minimally are composed of two files:
    * .pix containing the image data
    * .pix.hdr ENVI header describing the .pix data

    Additionally, geo correction LUT might be present:
    * .glu contains the geolocation data of the pixels across the track of the plane
    * .glu.hdr ENVI header describing the .glu data

    Additionnally, navigation files might be present:
    * -Navcor_sum.log statistical summary of the NAVCOR.log
    * -NAVCOR.log: full navigation log from the GNSS IMU system
    """

    def __init__(self, pix_dir):
        t0 = time.perf_counter()

        if os.path.isdir(pix_dir):
            self.image_dir = pix_dir
        else:
            raise ValueError("pix_dir does not exist")

        self.image_name = os.path.basename(pix_dir).split(".")[0]

        # WISE radiometric data

        files = os.listdir(pix_dir)

        hdr_f = [file for file in files if re.match(r".*\.pix\.hdr", file)][0]
        self.hdr_f = os.path.join(pix_dir, hdr_f)

        pix_f = [file for file in files if re.match(r".*\.pix$", file)][0]
        self.pix_f = os.path.join(pix_dir, pix_f)

        if not os.path.isfile(self.hdr_f) or not os.path.isfile(self.pix_f):
            raise ValueError(f"{self.hdr_f} or {self.pix_f} does not exist")

        self.image_name = [re.findall(r".*(?=-L|-N)", file) for file in files][0][0]

        # Open the .pix file with GDAL
        self.src_ds = gdal.Open(self.pix_f)
        logging.info(
            f"Dataset open with GDAL driver: {self.src_ds.GetDriver().ShortName}"
        )

        # Parse ENVI header
        self.header = helper.read_envi_hdr(hdr_f=self.hdr_f)

        self.wavelength = np.array(
            [float(w) for w in self.header["wavelength"].split(",")]
        )

        # Define time for the image
        # datetime.strptime("21/11/06T16:30:00Z", "%d/%m/%yT%H:%M:%SZ")
        self.acq_time_z = datetime.datetime.strptime(
            self.header["acquisition time"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=datetime.timezone.utc)

        # Define image cube size
        # self.in_ds.RasterYSize
        # Vertical axis: lines = rows = height = y
        # self.in_ds.RasterXSize
        # Horizontal axis: samples = columns = width = x
        self.n_rows = int(self.header["lines"])
        self.n_cols = int(self.header["samples"])
        self.n_bands = int(self.header["bands"])

        # Define image coordinate from 'map info'
        map_info = self.header["map info"]
        self.Affine, self.CRS, self.Proj4String = helper.parse_mapinfo(map_info)
        # Compute projected and geographic coordinates
        self.cal_coordinate(self.Affine, self.n_rows, self.n_cols)

        # Define data type, unit, scale factor, offset, and ignore values
        dtype = int(self.header["data type"])
        if dtype == 1:
            self.dtype = np.dtype("uint8")
        if dtype == 2:
            self.dtype = np.dtype("int16")

        # TODO: how to deal with unit parsing ? Should we at all ? Just ask the user ?
        self.unit = "uW cm-2 nm-1 sr-1"
        unit_res = {
            key: val for key, val in self.header.items() if re.search(f"unit", key)
        }
        logging.info(f"Found unit: {unit_res}\nHard coded unit is: {self.unit}")
        #
        # if len(res) == 0:
        #     print("No unit found in the header you should probably: 1. yell of despair and 2. Try your best guess and contact someone :)")
        # if len(res) > 1:
        #     print("Multiple unit found")

        # either 'data scale factor' or 'radiance scale factor' can exist in ENVI hdr
        # self.scale = int(self.Header.metadata['radiance scale factor'])
        # scale_res = {key: val for key, val in self.header.items() if re.search(f"scale", key)}
        """
        scale is used by NetCDF CF in writing and reading
        Reading: multiply by the scale and add the add_offset
        Writing: subtract the add_offset and divide by the scale
        If the scale factor is integer, to properly apply the scale in the writing order we need the
        reciprocal of it.
        """
        scale_factor = [val for key, val in self.header.items() if "scale" in key][0]
        logging.info(f"Read scale factor as: {scale_factor}")
        try:
            scale_factor = int(scale_factor)
            self.scale_factor = np.reciprocal(float(scale_factor))
        except ValueError as e:
            print(e)
            self.scale_factor = float(scale_factor)
        logging.info(f"Converted scale factor as: {self.scale_factor}")
        ignore_value = self.header["data ignore value"]
        if ignore_value == "":
            self.no_data = -99999
        else:
            self.no_data = int(ignore_value)

        # Geocorrection Look Up tables
        # In case Algae-WISE, glu file are named L2C

        glu_hdr_f = [file for file in files if re.match(r".*\.glu\.hdr$", file)][0]
        self.glu_hdr_f = os.path.join(pix_dir, glu_hdr_f)
        glu_f = [file for file in files if re.match(r".*\.glu$", file)][0]
        self.glu_f = os.path.join(pix_dir, glu_f)

        # Navigation data: altitude, heading, pitch, roll, speed
        nav_f = [file for file in files if re.match(r".*Navcor_sum\.log$", file)][0]
        self.nav_f = os.path.join(pix_dir, nav_f)

        if (
            not os.path.isfile(self.glu_hdr_f)
            or not os.path.isfile(self.glu_f)
            or not os.path.isfile(self.nav_f)
        ):
            logging.warning(
                "Navigation data or geo correction missing, cannot compute viewing geometry."
            )
            self.flightline = None
            self.glu_f = None
            self.glu_hdr_f = None

        else:
            self.flightline = FlightLine.from_wise_file(
                nav_sum_log=self.nav_f, glu_hdr=self.glu_hdr_f
            )
            self.z = self.flightline.height

        super().__init__(
            in_path=self.pix_f,
            in_ds=self.src_ds,
            image_name=self.image_name,
            wavelength=self.wavelength,
            acq_time_z=self.acq_time_z,
            z=self.z,
            y=self.y,
            x=self.x,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            Affine=self.Affine,
            crs=self.CRS,
        )

        self.expand_coordinate()

        self.cal_time(self.center_lon, self.center_lat)

        # Other instance attribute that need to be instanced before population by methods
        # self.valid_mask = None

        self.cal_sun_geom()

        self.cal_view_geom()

        self.cal_relative_azimuth()
        t1 = time.perf_counter()

        logging.info(
            f"ReveCube initiated from class {self.__class__.__name__} in {t1-t0:.2f}s"
        )

    def cal_view_geom(self):
        """
        extract viewing zenith angle
        the orginal data from the flight line is not georeferenced, and the nrows and ncols are not the same as the georeferenced ones
        so, we need to transfer the original viewing geometry to the georefernce grid using the georeference LUT
        :return:
        """
        logging.debug("Calculating viewing geometry from glu files")

        if self.flightline is None:
            raise Exception("no flight line found")

        glu_data = None
        if self.glu_f is not None:
            glu_data = gdal.Open(self.glu_f)
            nchannels, nsamples_glu, nlines_glu = (
                glu_data.RasterCount,
                glu_data.RasterXSize,
                glu_data.RasterYSize,
            )
            # WISEMan glu file has three channels but not AlgaeWISE
            # However, the first two channels are the same
            # if nchannels != 3:
            #     raise Exception("the glu file does not have three channels")
            if (
                nsamples_glu != self.flightline.samples
                or nlines_glu != self.flightline.lines
            ):
                raise Exception("samples or lines of flightline and glu do not match")

        # data_glu =  glu_data.ReadAsArray()

        band_x, band_y = glu_data.GetRasterBand(1), glu_data.GetRasterBand(2)
        x_glu, y_glu = band_x.ReadAsArray(), band_y.ReadAsArray()

        v_zenith_fl, v_azimuth_fl = self.flightline.cal_view_geom()
        # v_zenith_fl, v_azimuth_fl = v_zenith_fl.flatten(), v_azimuth_fl.flatten()

        ## initialize viewing zenith and azimuth with default values
        v_zenith_level1 = np.full((self.n_rows, self.n_cols), np.nan)
        v_azimuth_level1 = np.full((self.n_rows, self.n_cols), np.nan)
        # Initialize the sample position on spatial dimension of the imaging spectrometer array
        v_sample_array = np.full((self.n_rows, self.n_cols), np.nan)
        v_line_array = np.full((self.n_rows, self.n_cols), np.nan)

        # TODO: use multiprocessing here, and probably change the name row for line, it is confusing
        for row in tqdm(range(self.flightline.lines), desc="Processing GLU"):
            xs, ys = x_glu[row], y_glu[row]
            # print(xs,ys)
            rows_c, cols_c = helper.transform_rowcol(
                self.Affine, xs=xs, ys=ys, precision=5
            )
            mask = (rows_c < self.n_rows) & (cols_c < self.n_cols)
            # print(np.max(rows_c),np.max(cols_c))
            rows_c = rows_c[mask]
            cols_c = cols_c[mask]

            v_zenith_level1[rows_c, cols_c] = v_zenith_fl[row][mask]
            v_azimuth_level1[rows_c, cols_c] = v_azimuth_fl[row][mask]

            # Across-track samples position on the spatial dimension of the imaging spectrometer array
            v_sample_array[rows_c, cols_c] = np.arange(0, self.flightline.samples)[mask]

            # Line index for across-track selection of the imaging spectrometer array spatial dimension
            v_line_array[rows_c, cols_c] = row

        self.view_zenith = helper.fill_na_2d(v_zenith_level1)
        self.view_azimuth = helper.fill_na_2d(v_azimuth_level1)
        self.sample_index = helper.fill_na_2d(v_sample_array)
        self.line_index = helper.fill_na_2d(v_line_array)

        self.view_zenith[~self.get_valid_mask()] = np.nan
        self.view_azimuth[~self.get_valid_mask()] = np.nan
        self.sample_index[~self.get_valid_mask()] = np.nan
        self.line_index[~self.get_valid_mask()] = np.nan

    def read_band(self, bandindex, tile: Tile = None):
        """
        read DN for a given band
        :param bandindex: bandindex starts with 0
        :param tile: an instance of the Tile class used for tiled processing
        :return: re
        """

        band_temp = self.src_ds.GetRasterBand(bandindex + 1)
        if tile:
            radiance_at_sensor = band_temp.ReadAsArray(
                xoff=tile[2], yoff=tile[0], win_xsize=tile.xsize, win_ysize=tile.ysize
            )
        else:
            radiance_at_sensor = band_temp.ReadAsArray()

        return radiance_at_sensor

    def get_valid_mask(self, tile: Tile = None):
        """
        Get the valid mask for the enire image or a specific tile
        :param tile: an object of class tile.Tile()
        :return:
        """
        if self.valid_mask is None:
            self.cal_valid_mask()
        if tile is None:
            return self.valid_mask
        else:
            return self.valid_mask[tile.sline: tile.eline, tile.spixl: tile.epixl]

    def cal_valid_mask(self):
        """
        Calculate the mask of valid pixel for the entire image (!= nodata)
        :return: valid_mask
        """
        if self.valid_mask is None:
            # TODO: Problem when the band read is flagged in bbl (bad band list),
            #   the mask contains only False. Should filter out the bbl bands.
            iband = 10
            radiance_at_sensor = self.read_band(iband)
            self.valid_mask = radiance_at_sensor > 0

    def to_reve_nc(self, out_file: str = None):
        """
        Convert the Pix() object to reve NetCDF CF format
        :return:
        """

        t0 = time.perf_counter()

        if out_file is None:
            out_file = self.image_dir.replace(".dpix", ".nc")

        # Create NetCDF file
        self.create_reve_nc(out_file)

        # Create radiometric variable
        self.create_var_nc(
            name="radiance_at_sensor",
            type="i4",
            dims=(
                "wavelength",
                "y",
                "x",
            ),
            scale=self.scale_factor,
        )

        # Add optional bad_band_list to the variable attributes
        self.out_ds.variables["radiance_at_sensor"].bad_band_list = self.header['bbl'].split(",  ")

        # self.n_bands = 1
        for band in tqdm(range(0, self.n_bands, 1), desc="Writing band: "):
            # GDAL use 1 base index
            # Reading from GDAL can be realy slow
            data = self.src_ds.GetRasterBand(band + 1)
            data = data.ReadAsArray()
            data = data * self.scale_factor

            # Assign missing value
            """
            scale is used by NetCDF CF in writing and reading
            Reading: multiply by the scale and add the add_offset
            Writing: subtract the add_offset and divide by the scale
            If the scale factor is integer, to properly apply the scale in the writing order we need the
            reciprocal of it.
            """
            data[data == 0] = self.no_data * self.scale_factor

            self.out_ds.variables["radiance_at_sensor"][band, :, :] = data

        # Create geometric variables
        geom = {
            "sun_azimuth": self.sun_azimuth,
            "sun_zenith": self.sun_zenith,
            "view_azimuth": self.view_azimuth,
            "view_zenith": self.view_zenith,
            "relative_azimuth": self.relative_azimuth,
            "sample_index": self.sample_index,
            "line_index": self.line_index,
        }

        for var in tqdm(geom, desc="Writing geometry"):
            self.create_var_nc(
                name=var,
                type="i4",
                dims=(
                    "y",
                    "x",
                ),
                scale=self.scale_factor,
            )
            data = geom[var]

            np.nan_to_num(data, copy=False, nan=self.no_data * self.scale_factor)

            self.out_ds.variables[var][:, :] = data

        self.out_ds.close()
        t1 = time.perf_counter()

        print(f"Exported {self.__class__.__name__} to REVE.nc in {t1-t0:.2f}s")
        return

    def to_zarr(self, out_store: str = None):
        t0 = time.perf_counter()

        if out_store is None:
            raise Exception("out_store not set, cannot create dataset")

        try:
            store = zarr.DirectoryStore(out_store)
        except Exception as e:
            print(e)
            return

        root_grp = zarr.group(store, overwrite=True)

        # grid_mapping
        crs = self.CRS
        logging.info("Detected EPSG:" + str(crs.to_epsg()))

        # For GDAL
        # https://gdal.org/drivers/raster/zarr.html
        root_grp.attrs["_CRS"] = {
            "url": "http://www.opengis.net/def/crs/EPSG/0/" + str(crs.to_epsg()),
            "wkt": crs.to_wkt(),
        }

        root_grp.attrs["Conventions"] = "CF-1.0"
        root_grp.attrs["title"] = "Remote sensing image written by REVERIE"
        root_grp.attrs[
            "history"
        ] = "File created on " + datetime.datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%SZ"
        )
        root_grp.attrs["institution"] = "AquaTel UQAR"
        root_grp.attrs["source"] = "Remote sensing imagery"
        root_grp.attrs["version"] = "0.1.0"
        root_grp.attrs["references"] = "https://github.com/raphidoc/reverie"
        root_grp.attrs[
            "comment"
        ] = "Reflectance Extraction and Validation for Environmental Remote Imaging Exploration"

        w = root_grp.create_dataset(
            "wavelength",
            shape=(len(self.wavelength)),
            chunks=(len(self.wavelength)),
            dtype="f4",
            compressor=zarr.Zlib(level=1),
        )
        # _ARRAY_DIMENSIONS is xarray had hoc attribute
        # to store the dimension following netCDF CF convention
        w.attrs["_ARRAY_DIMENSIONS"] = ["wavelength"]
        w.attrs["units"] = "nm"
        w.attrs["standard_name"] = "radiation_wavelength"
        w.attrs["long_name"] = "Central wavelengths of the sensor bands"
        w.attrs["axis"] = "wavelength"
        w[:] = self.wavelength

        t = root_grp.create_dataset(
            "time",
            shape=(len([self.acq_time_z])),
            chunks=(len([self.acq_time_z])),
            dtype="f4",
            compressor=zarr.Zlib(level=1),
        )
        t.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        t.attrs["units"] = "seconds since 1970-01-01 00:00:00 +00:00"
        t.attrs["standard_name"] = "time"
        t.attrs["long_name"] = "UTC acquisition time of remote sensing image"
        t[:] = self.acq_time_z.timestamp()

        z = root_grp.create_dataset(
            "z",
            shape=(len([self.z])),
            chunks=(len([self.z])),
            dtype="f4",
            compressor=zarr.Zlib(level=1),
        )
        z.attrs["_ARRAY_DIMENSIONS"] = ["z"]
        z.attrs["units"] = "m"
        z.attrs["standard_name"] = "altitude"
        z.attrs[
            "long_name"
        ] = "Altitude is the viewing height above the geoid, positive upward"
        z[:] = self.z

        y = root_grp.create_dataset(
            "y",
            shape=(len(self.y)),
            chunks=(len(self.y)),
            dtype="f4",
            compressor=zarr.Zlib(level=1),
        )
        y.attrs["_ARRAY_DIMENSIONS"] = ["y"]
        y.attrs["units"] = "m"
        y.attrs["standard_name"] = "projection_y_coordinate"
        y.attrs["long_name"] = "y-coordinate in projected coordinate system"
        y.attrs["axis"] = "y"
        y[:] = self.y

        lat = root_grp.create_dataset(
            "lat",
            shape=(len(self.y)),
            chunks=(len(self.y)),
            dtype="f4",
            compressor=zarr.Zlib(level=1),
        )
        lat.attrs["_ARRAY_DIMENSIONS"] = ["y"]
        lat.attrs["units"] = "degrees_north"
        lat.attrs["standard_name"] = "latitude"
        lat[:] = self.lat

        x = root_grp.create_dataset(
            "x",
            shape=(len(self.x)),
            chunks=(len(self.x)),
            dtype="f4",
            compressor=zarr.Zlib(level=1),
        )
        x.attrs["_ARRAY_DIMENSIONS"] = ["x"]
        x.attrs["units"] = "m"
        x.attrs["standard_name"] = "projection_x_coordinate"
        x.attrs["long_name"] = "x-coordinate in projected coordinate system"
        x.attrs["axis"] = "x"
        x[:] = self.x

        lon = root_grp.create_dataset(
            "lon",
            shape=(len(self.x)),
            chunks=(len(self.x)),
            dtype="f4",
            compressor=zarr.Zlib(level=1),
        )
        lon.attrs["_ARRAY_DIMENSIONS"] = ["x"]
        lon.attrs["units"] = "degrees_east"
        lon.attrs["standard_name"] = "longitude"
        lon[:] = self.lon

        grid_mapping = root_grp.create_dataset(
            "grid_mapping",
            shape=(),
            chunks=(),
            dtype="i4",
        )

        # For xarray:
        # the grid_mapping variable MUST have the _ARRAY_DIMENSIONS attribute even if it is empty
        grid_mapping.attrs["_ARRAY_DIMENSIONS"] = []

        # Maybe not necessary but it follows the CF convention
        cf_grid_mapping = crs.to_cf()

        grid_mapping.attrs["crs_wkt"] = cf_grid_mapping["crs_wkt"]
        grid_mapping.attrs["semi_major_axis"] = cf_grid_mapping["semi_major_axis"]
        grid_mapping.attrs["semi_minor_axis"] = cf_grid_mapping["semi_minor_axis"]
        grid_mapping.attrs["inverse_flattening"] = cf_grid_mapping["inverse_flattening"]
        grid_mapping.attrs["reference_ellipsoid_name"] = cf_grid_mapping[
            "reference_ellipsoid_name"
        ]
        grid_mapping.attrs["longitude_of_prime_meridian"] = cf_grid_mapping[
            "longitude_of_prime_meridian"
        ]
        grid_mapping.attrs["prime_meridian_name"] = cf_grid_mapping[
            "prime_meridian_name"
        ]
        grid_mapping.attrs["geographic_crs_name"] = cf_grid_mapping[
            "geographic_crs_name"
        ]
        grid_mapping.attrs["horizontal_datum_name"] = cf_grid_mapping[
            "horizontal_datum_name"
        ]
        grid_mapping.attrs["projected_crs_name"] = cf_grid_mapping["projected_crs_name"]
        grid_mapping.attrs["grid_mapping_name"] = cf_grid_mapping["grid_mapping_name"]
        grid_mapping.attrs["latitude_of_projection_origin"] = cf_grid_mapping[
            "latitude_of_projection_origin"
        ]
        grid_mapping.attrs["longitude_of_central_meridian"] = cf_grid_mapping[
            "longitude_of_central_meridian"
        ]
        grid_mapping.attrs["false_easting"] = cf_grid_mapping["false_easting"]
        grid_mapping.attrs["false_northing"] = cf_grid_mapping["false_northing"]
        grid_mapping.attrs["scale_factor_at_central_meridian"] = cf_grid_mapping[
            "scale_factor_at_central_meridian"
        ]

        # self.no_data = netCDF4.default_fillvals[type]
        # When scaling the default _FillValue, get somehow messed up when reading with GDAL
        self.no_data = math.trunc(netCDF4.default_fillvals["i4"] / 1000)

        # filters = [numcodecs.Delta(dtype="i4")]
        compressor = numcodecs.Blosc(
            cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE
        )
        # filters have to be an iterable, in case multiple filters are applied
        filters = [numcodecs.Quantize(digits=3, dtype="f4")]
        # TODO, cannot make filters work properly
        #  https://zarr.readthedocs.io/en/stable/spec/v2.html#filters
        filters = [zarr.codecs.FixedScaleOffset(0, 1000, "f4")]

        # Create radiometric variable
        # chunk size is set to 1 wavelength to allow for parallel computation and I/O
        # size along the spatial dims should be optimized for the available memory
        radiance_at_sensor = root_grp.create(
            "radiance_at_sensor",
            shape=(self.n_bands, self.n_rows, self.n_cols),
            chunks=(1, 500, 500),
            dtype="<f4",
            fill_value=self.no_data,
            # filters=filters,
            compressor=compressor,
        )

        radiance_at_sensor.attrs["_ARRAY_DIMENSIONS"] = ["wavelength", "y", "x"]

        radiance_at_sensor.attrs["grid_mapping"] = "grid_mapping"  # self.proj_var.name
        radiance_at_sensor.attrs["_CRS"] = {
            "url": "http://www.opengis.net/def/crs/EPSG/0/" + str(crs.to_epsg()),
            "wkt": crs.to_wkt(),
        }

        # Follow the standard name table CF convention
        std_name, std_unit = get_cf_std_name(alias="radiance_at_sensor")
        radiance_at_sensor.attrs["units"] = std_unit
        radiance_at_sensor.attrs["standard_name"] = std_name

        # self.n_bands = 1
        for band in tqdm(range(0, self.n_bands, 1), desc="Writing band: "):
            # GDAL use 1 base index
            # Reading from GDAL can be really slow, would be nice to be able to parallelize this
            data = self.src_ds.GetRasterBand(band + 1)
            data = data.ReadAsArray()
            data = data * self.scale_factor

            data[data == 0] = np.nan  # self.no_data

            # filters[0].encode(data).max()

            radiance_at_sensor[
                band, :, :
            ] = data  # filters[0].encode(data).reshape(self.n_rows, self.n_cols)

        # GDAL doesn't seem to be able to read the the spectral data (wavelength, y, y)
        # when just spatial data is also present (y, x)
        # Create geometric variables
        geom = {
            "sun_azimuth": self.sun_azimuth,
            "sun_zenith": self.sun_zenith,
            "view_azimuth": self.view_azimuth,
            "view_zenith": self.view_zenith,
            "relative_azimuth": self.relative_azimuth,
            "sample_index": self.sample_index,
            "line_index": self.line_index,
        }

        for var in tqdm(geom, desc="Writing geometry"):
            array_zarr = root_grp.create(
                var,
                shape=(self.n_rows, self.n_cols),
                chunks=(500, 500),
                dtype="f4",
                fill_value=self.no_data,
                # filters=filters,
                compressor=compressor,
            )

            array_zarr.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]

            array_zarr.attrs["_CRS"] = {
                "url": "http://www.opengis.net/def/crs/EPSG/0/" + str(crs.to_epsg()),
                "wkt": crs.to_wkt(),
            }

            data = geom[var]

            array_zarr[:, :] = data

        # Write consolidated metadata
        zarr.convenience.consolidate_metadata(store, metadata_key=".zmetadata")

        t1 = time.perf_counter()

        print(f"Exported {self.__class__.__name__} to Zarr in {t1-t0:.2f}s")
        # No need to close the store, it is never really opened
