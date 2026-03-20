import datetime
import math
import os


import xarray as xr

import pyproj
import affine
import netCDF4
import logging
import numpy as np

import zarr

from aabim.utils.cf_aliases import get_cf_std_name

log = logging.getLogger(__name__)


class IOMixin:

    @classmethod
    def from_zarr(cls, in_path: str):
        """Populate ReveCube object from zarr store

        Parameters
        ----------
        src_store: str
            zarr store to read from

        Returns
        -------
        """

        if os.path.isdir(in_path):
            in_ds = xr.open_zarr(in_path, consolidated=True, decode_times=False)
        else:
            raise Exception(f"Directory {in_path} does not exist")

        return cls.decode_xr(in_path, in_ds)

    @classmethod
    def from_aabim_nc(cls, in_path: str):
        """Populate Image object from aabim CF NetCDF dataset

        Parameters
        ----------
        in_path: str
            NetcCDF (.nc) CF-1.0 compliant file to read from

        Returns
        -------
        """

        if os.path.isfile(in_path):
            in_ds = xr.open_dataset(in_path, decode_times=False)

            # in_ds = netCDF4.Dataset(in_path, "r+", format="NETCDF4")
            # image_name = os.path.basename(in_path).split(".")[0]
            # in_ds.image_name = image_name
            # in_ds.close()
        else:
            raise Exception(f"File {in_path} does not exist")

        return cls.decode_xr(in_path, in_ds)

    @classmethod
    def decode_xr(cls, in_path, in_ds: xr.Dataset):
        # Image name
        image_name = in_ds.attrs["image_name"]

        # Spectral attributes
        wavelength = in_ds.variables["wavelength"].data

        # Spatiotemporal attributes
        time_var = in_ds.variables["time"]
        # if time is decoded by xarray as datetime64[ns], convert it to datetime.datetime
        # ts = (
        #     time_var.data[0] - np.datetime64("1970-01-01T00:00:00Z")
        # ) / np.timedelta64(1, "s")
        # TODO: better define acq_time_z as image center datetime,
        #  also should it be decoded by xarray or not ?
        # For some reason utcfromtimestamp doesn't carry tz info (acutally UTC is not a timezone, GMT is the timezone)
        # It create a problem if the data loaded by panda follow the iso8601 format.
        # In that case the datetime object will have a utc timezone.
        # We cannot make comutation on timezone aware and not aware.
        # It's best to force the timezone to be utc and to let the user actually deal with the data provided to the app.
        acq_time_z = datetime.datetime.utcfromtimestamp(int(time_var.data[0])).replace(
            tzinfo=datetime.timezone.utc
        )
        altitude_var = in_ds.variables["z"]
        z = altitude_var[:][0]
        y = in_ds.variables["y"][:].data
        x = in_ds.variables["x"][:].data
        n_rows = in_ds.sizes["y"]
        n_cols = in_ds.sizes["x"]

        grid_mapping = in_ds.variables["grid_mapping"]

        # Derive Affine from the actual x/y coordinates so it is always
        # consistent with the dataset extent (including bbox subsets).
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        Affine = affine.Affine(dx, 0.0, float(x[0]) - dx / 2.0,
                               0.0, dy,  float(y[0]) - dy / 2.0)
        crs = pyproj.CRS.from_wkt(grid_mapping.attrs["crs_wkt"])

        level = in_ds.attrs.get("processing_level", None)

        return cls(
            in_path,
            in_ds,
            image_name,
            wavelength,
            acq_time_z,
            z,
            y,
            x,
            n_rows,
            n_cols,
            Affine,
            crs,
            level,
        )

    def create_reve_nc(self, out_path: str = None):
        """Create CF-1.0 compliant NetCDF dataset from DateCube
        CF-1.0 (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.0/build/cf-conventions.html#dimensions)
        is chosen to ensure compatibility with the GDAL NetCDF driver:
        (https://gdal.org/drivers/raster/netcdf.html).

        This format can also be read by SNAP but the spectrum view tool does not find any spectral dimension.
        Quinten wrote something about that in ACOLITE.

        :return: None
        """

        if out_path is None:
            raise Exception("out_path file not set, cannot create dataset")

        try:
            nc_ds = netCDF4.Dataset(out_path, "w", format="NETCDF4")
        except Exception as e:
            print(e)
            return

        self.out_path = out_path

        # TODO validate that it follow the convention with cfdm / cf-python.
        #  For compatibility with GDAL NetCDF driver use CF-1.0
        #  https://cfconventions.org/conventions.html
        nc_ds.Conventions = "CF-1.0"
        nc_ds.title = self.image_name
        nc_ds.processing_level = self.level
        nc_ds.history = "File created on " + datetime.datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%SZ"
        )
        nc_ds.institution = "AquaTel UQAR"
        nc_ds.source = "Remote sensing imagery"
        nc_ds.version = "0.1.0"
        nc_ds.references = "https://github.com/raphidoc/aabim"
        nc_ds.comment = "Atmospheric Aquatic Bayesian Inversion Model"
        nc_ds.image_name = os.path.basename(self.in_path).split(".")[0]

        # Create Dimensions
        nc_ds.createDimension("wavelength", len(self.wavelength))
        nc_ds.createDimension("time", len([self.acq_time_z]))
        nc_ds.createDimension("z", len([self.z]))
        nc_ds.createDimension("y", len(self.y))
        nc_ds.createDimension("x", len(self.x))

        band_var = nc_ds.createVariable("wavelength", "f4", ("wavelength",))
        band_var.units = "nm"
        band_var.standard_name = "radiation_wavelength"
        band_var.long_name = "Central wavelengths of the sensor bands"
        band_var.axis = "wavelength"
        band_var[:] = self.wavelength

        # Create coordinate variables
        # We will store time as seconds since 1 january 1970 good luck people of 2038 :) !
        t_var = nc_ds.createVariable("time", "f8", ("time",))
        t_var.standard_name = "time"
        t_var.long_name = "UTC acquisition time of remote sensing image"
        # CF convention for time zone is UTC if ommited
        # xarray will convert it to a datetime64[ns] object considering it is local time
        t_var.units = "seconds since 1970-01-01 00:00:00 +00:00"
        t_var.calendar = "gregorian"
        t_var[:] = self.acq_time_z.timestamp()

        z_var = nc_ds.createVariable("z", "f8", ("z",))
        z_var.units = "m"
        z_var.standard_name = "altitude"
        z_var.long_name = (
            "Altitude is the viewing height above the geoid, positive upward"
        )
        z_var.axis = "z"
        z_var[:] = self.z

        y_var = nc_ds.createVariable("y", "f8", ("y",))
        y_var.units = "m"
        y_var.standard_name = "projection_y_coordinate"
        y_var.long_name = "y-coordinate in projected coordinate system"
        y_var.axis = "y"
        y_var[:] = self.y

        # lat_var = nc_ds.createVariable("lat", "f8", ("y",))
        # lat_var.standard_name = "latitude"
        # lat_var.units = "degrees_north"
        # # lat_var.long_name = 'latitude'
        # lat_var[:] = self.lat

        x_var = nc_ds.createVariable("x", "f8", ("x",))
        x_var.units = "m"
        x_var.standard_name = "projection_x_coordinate"
        x_var.long_name = "x-coordinate in projected coordinate system"
        x_var.axis = "x"
        x_var[:] = self.x

        # lon_var = nc_ds.createVariable("lon", "f8", ("x",))
        # lon_var.standard_name = "longitude"
        # lon_var.units = "degrees_east"
        # # lon_var.long_name = 'longitude'
        # lon_var[:] = self.lon

        # grid_mapping
        crs = self.CRS
        logging.info("Detected EPSG:" + str(crs.to_epsg()))
        cf_grid_mapping = crs.to_cf()

        grid_mapping = nc_ds.createVariable("grid_mapping", np.int32, ())

        # Fix GDAL projection issue by providing the attributes spatial_ref and GeoTranform
        # GeoTransform is an affine transform array defined by GDAL with parameters in the orders:
        # GT0 = 541107.0  # x coordinate of the upper-left corner of the upper-left pixel
        # GT1 = 1.5  # w-e pixel resolution / pixel width.
        # GT2 = 0  # row rotation (typically zero).
        # GT3 = (
        #     5438536.5  # y-coordinate of the upper-left corner of the upper-left pixel.
        # )
        # GT4 = 0  # column rotation (typically zero).
        # GT5 = 1.5  # n-s pixel resolution / pixel height (negative value for a north-up image).
        # grid_mapping.GeoTransform = f"{GT0} {GT1} {GT2} {GT3} {GT4} {GT5}"
        # raise "Implement this first !"

        grid_mapping.spatial_ref = cf_grid_mapping["crs_wkt"]

        grid_mapping.grid_mapping_name = cf_grid_mapping["grid_mapping_name"]
        grid_mapping.crs_wkt = cf_grid_mapping["crs_wkt"]
        grid_mapping.semi_major_axis = cf_grid_mapping["semi_major_axis"]
        grid_mapping.semi_minor_axis = cf_grid_mapping["semi_minor_axis"]
        grid_mapping.inverse_flattening = cf_grid_mapping["inverse_flattening"]
        grid_mapping.reference_ellipsoid_name = cf_grid_mapping[
            "reference_ellipsoid_name"
        ]
        grid_mapping.longitude_of_prime_meridian = cf_grid_mapping[
            "longitude_of_prime_meridian"
        ]
        grid_mapping.prime_meridian_name = cf_grid_mapping["prime_meridian_name"]
        grid_mapping.geographic_crs_name = cf_grid_mapping["geographic_crs_name"]
        grid_mapping.horizontal_datum_name = cf_grid_mapping["horizontal_datum_name"]
        grid_mapping.projected_crs_name = cf_grid_mapping["projected_crs_name"]
        grid_mapping.grid_mapping_name = cf_grid_mapping["grid_mapping_name"]
        grid_mapping.latitude_of_projection_origin = cf_grid_mapping[
            "latitude_of_projection_origin"
        ]
        grid_mapping.longitude_of_central_meridian = cf_grid_mapping[
            "longitude_of_central_meridian"
        ]
        grid_mapping.false_easting = cf_grid_mapping["false_easting"]
        grid_mapping.false_northing = cf_grid_mapping["false_northing"]
        grid_mapping.scale_factor_at_central_meridian = cf_grid_mapping[
            "scale_factor_at_central_meridian"
        ]

        grid_mapping.affine_transform = (
            self.Affine.a,
            self.Affine.b,
            self.Affine.c,
            self.Affine.d,
            self.Affine.e,
            self.Affine.f,
        )

        # self.proj_var = grid_mapping
        self.out_ds = nc_ds

    def create_var_nc(
        self,
        name: str = None,
        type=None,
        dims: tuple = None,
        scale: float = 1.0,
        comp="zlib",
        complevel=1,
    ):
        """Create a CF-1.0 variable in a NetCDF dataset

        Parameters
        ----------
        name : str
            Name of the variable to write to the NetCDF dataset
        type : str
            Data type of the variable to write to the NetCDF dataset
        dims : tuple
            Contain the dimension of the var with the form ('wavelength', 'y', 'x',).
        scale : float
            scale_factor is used by NetCDF CF in writing and reading for lossy compression.
        comp : str, default: 'zlib'
            Name of the compression algorithm to use. Use None to deactivate compression.
        complevel : int, default: 1
            Compression level.

        Returns
        -------

        See Also
        --------
        create_dataset_nc
        """

        ds = self.out_ds

        # Easier to leave missing_value as the default _FillValue,
        # but then GDAL doesn't recognize it ...
        # self.no_data = netCDF4.default_fillvals[type]
        # When scaling the default _FillValue, it get somehow messed up when reading with GDAL

        # Offset no_data to fit inside data type after scaling
        # self.no_data = math.trunc(netCDF4.default_fillvals[type] * 0.00001)
        self.no_data = math.trunc(netCDF4.default_fillvals[type] * scale)

        # if scale == 1.0:
        #     self.no_data = netCDF4.default_fillvals[type] - 50

        data_var = ds.createVariable(
            varname=name,
            datatype=type,
            dimensions=dims,
            fill_value=self.no_data,
            compression=comp,
            complevel=complevel,
        )

        data_var.grid_mapping = "grid_mapping"  # self.proj_var.name

        # Follow the standard name table CF convention
        # TODO: fill the units and standard name automatically from name ?
        #  Water leaving reflectance could be: reflectance_of_water_leaving_radiance_on_incident_irradiance
        #  "reﬂectance based on the water-leaving radiance and the incident irradiance" (Ocean Optics Web Book)
        std_name, std_unit = get_cf_std_name(alias=name)

        data_var.units = std_unit
        data_var.standard_name = std_name
        # data_var.long_name = ''

        # self.__dst.variables['rho_remote_sensing'].valid_min = 0
        # self.__dst.variables['rho_remote_sensing'].valid_max = 6000
        data_var.missing_value = self.no_data

        """
            scale is used by NetCDF CF in writing and reading
            Reading: multiply by the scale and add the add_offset
            Writing: subtract the add_offset and divide by the scale
            If the scale factor is integer, to properly apply the scale in the writing order we need the
            reciprocal of it.
            """
        data_var.scale_factor = scale
        data_var.add_offset = 0

    def to_netcdf(self, path: str) -> None:
        """Save the current image dataset to a CF-compliant NetCDF file.

        Parameters
        ----------
        path : str
            Output file path (``*.nc``).
        """
        self.in_ds.to_netcdf(path)

    def to_reve_nc(self):
        """
        TODO : Wrapper for create_dataset_nc, create_var_nc, write_var_nc, close_nc ?
        Returns
        -------
        """
        pass

    def to_zarr(self, out_store: str, consolidated: bool = True) -> None:
        """Write the image dataset to a CF-compliant, GDAL-readable Zarr store.

        CRS and georeferencing are encoded via the CF *grid_mapping* convention:
        a scalar variable ``spatial_ref`` carries ``crs_wkt``, ``spatial_ref``,
        and ``GeoTransform`` attributes.  GDAL ≥ 3.4 and QGIS read this
        natively through the Zarr driver.  The approach is forward-compatible
        with the emerging GeoZarr spec (which is a superset of CF grid_mapping).

        Parameters
        ----------
        out_store : str
            Path to the output Zarr directory store (e.g. ``"scene_l2r.zarr"``).
        consolidated : bool, default True
            Write a consolidated ``.zmetadata`` file for fast remote access.
        """
        ds = self.in_ds.copy()

        # --- Build spatial_ref scalar variable ---------------------------------
        # GDAL reads crs_wkt + GeoTransform from this variable to reconstruct
        # the full georeferencing.  spatial_ref duplicates crs_wkt as a GDAL
        # compatibility alias.
        crs_wkt = self.CRS.to_wkt()
        a = self.Affine

        # GDAL GeoTransform: (x_origin, pixel_width, x_rot,
        #                      y_origin,    y_rot,  pixel_height)
        # For a north-up image: x_rot == y_rot == 0, pixel_height < 0.
        geo_transform = f"{a.c} {a.a} {a.b} {a.f} {a.d} {a.e}"

        spatial_ref = xr.Variable(
            (),
            np.int32(0),
            attrs={
                "grid_mapping_name": self.CRS.to_cf().get("grid_mapping_name", ""),
                "crs_wkt":           crs_wkt,
                "spatial_ref":       crs_wkt,   # GDAL alias
                "GeoTransform":      geo_transform,
            },
        )
        ds = ds.assign({"spatial_ref": spatial_ref})

        # --- Tag every spatially-explicit variable with grid_mapping -----------
        # A variable is "spatially explicit" when both 'y' and 'x' appear in
        # its dimensions.
        updated_vars = {}
        for name, var in ds.data_vars.items():
            if name == "spatial_ref":
                continue
            if "y" in var.dims and "x" in var.dims:
                updated_vars[name] = var.assign_attrs(
                    {**var.attrs, "grid_mapping": "spatial_ref"}
                )
        if updated_vars:
            ds = ds.assign(updated_vars)

        # --- Global attributes -------------------------------------------------
        ds.attrs.setdefault("Conventions", "CF-1.0")
        ds.attrs["creation_time"] = datetime.datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # --- Write -------------------------------------------------------------
        ds.to_zarr(out_store, mode="w", consolidated=consolidated)
        log.info("Image written to Zarr store → %s", out_store)

    def zarr_to_nc(self, zarr_path: str, nc_path: str) -> None:
        """Convert a Zarr store produced by :meth:`to_zarr` to CF NetCDF.

        All CF metadata (``spatial_ref``, ``grid_mapping`` attributes,
        ``Conventions``) are already embedded in the Zarr store, so this
        is a straight pass-through via xarray.

        Parameters
        ----------
        zarr_path : str
            Path to the source Zarr directory store.
        nc_path : str
            Destination NetCDF file path (``*.nc``).
        """
        ds = xr.open_zarr(zarr_path)
        ds.to_netcdf(nc_path)
        log.info("Zarr → NetCDF written → %s", nc_path)
