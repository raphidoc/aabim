from __future__ import annotations

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm

from aabim.utils import helper


class ExtractMixin:
    def get_footprint(self):

        import rasterio.features
        from shapely.geometry import shape

        # Filter observation outside the image valid_mask
        valid_mask = self.get_valid_mask()

        # Convert the valid_mask to list of polygon(s)
        polygons = [
            shape(geom)
            for geom, value in rasterio.features.shapes(
                valid_mask.astype("int16"), mask=valid_mask, transform=self.Affine
            )
            if value == 1
        ]

        # Convert the list of polygons to a GeoDataFrame
        footprint_gdf = gpd.GeoDataFrame(geometry=polygons, crs=self.CRS.to_epsg())

        return footprint_gdf

    def extract_pixel(
        self,
        matchup_file: str = None,
        var_name: list = None,
        max_time_diff: float = None,
        window_size: int = 1,
        output_box: str = None,
    ):
        """
        Extract pixel value from a list of in-situ data

        Parameters
        ----------
        matchup_file : str
            Path to the in-situ data file
        var_name : list
            List of the variable name of image data to extract
        max_time_diff : float
            Maximum time difference between the in-situ data and the image acquisition time, in hours.
        window_size : int
            Size of the window used to extract pixel value from the image.
            Unit is pixel and must be an odd number (i.e 1,3,5,7,...).

        Returns
        -------
        pixex_df : pandas.DataFrame
            A dataframe containing the extracted pixel value
        """

        # def extractor(uuid, xr_ds, matchup_gdf=matchup_gdf):

        # iterator = p_uimap(extractor, matchup_gdf["UUID"])

        # load the nc dataset with xarray
        xr_ds = self.in_ds

        matchup_gdf = pd.read_csv(matchup_file)
        matchup_gdf.columns = matchup_gdf.columns.str.lower()
        matchup_gdf = matchup_gdf[["date_time", "lat", "lon", "uuid"]]

        # When matchup data is in long format (wavelength along the rows), keep unique observation
        matchup_gdf = matchup_gdf.drop_duplicates()

        # Filter observation outside requested time range
        if max_time_diff:
            matchup_gdf["date_time"] = pd.to_datetime(
                matchup_gdf["date_time"], utc=True
            )
            matchup_gdf = matchup_gdf[
                abs(matchup_gdf["date_time"] - self.acq_time_z)
                < pd.Timedelta(max_time_diff, unit="h")
            ]
            logging.info(
                "%s observations remaining after time filtering" % len(matchup_gdf)
            )

            if len(matchup_gdf) == 0:
                raise Exception("no matchup data remaining after time filter")

        # Project matchup data to image
        match_geometry = gpd.points_from_xy(
            matchup_gdf["lon"], matchup_gdf["lat"], crs="EPSG:4326"
        )

        matchup_gdf = gpd.GeoDataFrame(matchup_gdf, geometry=match_geometry)

        logging.info(f"Projecting in-situ data to EPSG: {self.CRS.to_epsg()}")
        matchup_gdf = matchup_gdf.to_crs(self.CRS.to_epsg())

        # import rasterio.features
        # from shapely.geometry import shape
        #
        # # Filter observation outside the image valid_mask
        # valid_mask = self.get_valid_mask()
        #
        # # Convert the valid_mask to list of polygon(s)
        # polygons = [
        #     shape(geom)
        #     for geom, value in rasterio.features.shapes(
        #         valid_mask.astype("int16"), mask=valid_mask, transform=self.Affine
        #     )
        #     if value == 1
        # ]
        #
        # # Convert the list of polygons to a GeoDataFrame
        # footprint_gdf = gpd.GeoDataFrame(geometry=polygons, crs=matchup_gdf.crs)
        #
        # # Write the GeoDataFrame to a GeoJSON file
        # # footprint_gdf.to_file("footprint.geojson", driver="GeoJSON")

        # Perform a spatial join between the matchup_gdf and the polygons_gdf
        footprint_gdf = self.get_footprint()

        filtered_gdf = gpd.sjoin(
            matchup_gdf, footprint_gdf, how="inner", predicate="intersects"
        )

        logging.info(
            "%s observation remaining after spatial filtering" % len(filtered_gdf)
        )

        if len(filtered_gdf) == 0:
            raise Exception("no matchup data remaining after spatial filter")

        pixex_df = pd.DataFrame()
        window_dict = {}
        window_data = {}

        for uuid in tqdm(filtered_gdf["uuid"]):
            temp_gdf = filtered_gdf[filtered_gdf["uuid"] == uuid].reset_index()

            # Make sure coordinates are in raster CRS
            x_coords, y_coords = xr_ds.x.values, xr_ds.y.values
            x_pt = temp_gdf.geometry.x.iloc[0]
            y_pt = temp_gdf.geometry.y.iloc[0]

            x_index = np.abs(x_coords - x_pt).argmin()
            y_index = np.abs(y_coords - y_pt).argmin()

            half_window = window_size // 2
            data_sel = xr_ds.isel(
                y=slice(max(0, y_index - half_window), y_index + half_window + 1),
                x=slice(max(0, x_index - half_window), x_index + half_window + 1),
            )

            # print("UTM20N x = " + str(data_sel.x[1].item()))
            # print("UTM20N y = " + str(data_sel.y[1].item()))
            #
            # # Get UTM20N x, y values
            # x = data_sel.x[1].item()
            # y = data_sel.y[1].item()
            #
            # # Define the transformer from UTM20N (EPSG:32620) to WGS84 (EPSG:4326)
            # transformer = pyproj.Transformer.from_crs("EPSG:32620", "EPSG:4326", always_xy=True)
            #
            # lon, lat = transformer.transform(x, y)
            #
            # print("lat = " + str(lat))
            # print("lon = " + str(lon))

            # Pick the exact pixel center in lat/lon
            # lat = xr_ds.lat.isel(y=y_index).item()
            # lon = xr_ds.lon.isel(x=x_index).item()

            # print("lat = " + str(lat))
            # print("lon = " + str(lon))
            #
            # import pyproj
            #
            # # Define the transformer from WGS84 (EPSG:4326) to UTM20N (EPSG:32620)
            # transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32620", always_xy=True)
            #
            # x, y = transformer.transform(lon, lat)
            #
            # print("UTM20N x = " + str(x))
            # print("UTM20N y = " + str(y))

            # Point geometry (same CRS as raster)
            # point_geom = gpd.GeoDataFrame(
            #     {'uuid': [uuid]},
            #     geometry=[Point(data_sel.x[half_window], data_sel.y[half_window])],
            #     crs=self.CRS.to_epsg()
            # )
            #
            # point_geom.to_file(os.path.join(output_box, f"{uuid}_point.geojson"), driver="GeoJSON")

            # Window bounds
            win = Window.from_slices(
                rows=slice(max(0, y_index - half_window), y_index + half_window + 1),
                cols=slice(max(0, x_index - half_window), x_index + half_window + 1),
            )
            window_dict[uuid] = win
            window_data[uuid] = data_sel

            # y_start, y_stop = int(win.row_off), int(win.row_off + win.height)
            # x_start, x_stop = int(win.col_off), int(win.col_off + win.width)
            #
            # x0, y0 = helper.transform_xy(self.Affine, rows=[y_start - 0.5], cols=[x_start - 0.5])
            # x1, y1 = helper.transform_xy(self.Affine, rows=[y_stop - 0.5], cols=[x_stop - 0.5])
            #
            # box_geom = gpd.GeoDataFrame(
            #     {'uuid': [uuid]},
            #     geometry=[box(x0[0], y0[0], x1[0], y1[0])],
            #     crs=self.CRS.to_epsg()
            # )
            # box_geom.to_file(os.path.join(output_box, f"{self.image_name}_windows.geojson"), driver="GeoJSON")

            # import holoviews as hv
            # from holoviews import opts
            # import panel as pn
            #
            # # setting bokeh as backend
            # hv.extension("bokeh")
            #
            # opts.defaults(
            #     opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
            #     opts.Image(cmap="viridis", width=400, height=400),
            #     opts.Labels(
            #         text_color="white",
            #         text_font_size="8pt",
            #         text_align="left",
            #         text_baseline="bottom",
            #     ),
            #     opts.Path(color="white"),
            #     opts.Spread(width=600),
            #     opts.Overlay(show_legend=False),
            # )
            #
            # ds = hv.Dataset(data_sel, vdims=["radiance_at_sensor"])
            #
            # plot = ds.to(hv.Image, kdims=["x", "y"], dynamic=True).hist()
            #
            # renderer = hv.renderer("bokeh")
            # renderer.save(
            #     plot, "/D/Documents/PhD/Thesis/Chapter2/Data/WISE/pixex/test.html"
            # )
            #
            # pn.serve(plot)

            temp_pixex_df = data_sel.to_dataframe()
            # temp_pixex_df = temp_pixex_df.rename_axis("Wavelength")
            # temp_pixex_df = temp_pixex_df.reset_index()
            # TODO output a wide format when wavelength and non wavelength data are mixed ?
            # temp_pixex_df = pd.pivot(temp_pixex_df, index=['x', 'y'], columns='Wavelength', values='radiance_at_sensor')
            temp_pixex_df["uuid"] = uuid
            temp_pixex_df["image_name"] = self.image_name
            # temp_pixex_df['Sensor'] = str(temp_pix_ex_array.Sensor.values)
            # temp_pixex_df['ImageDate'] = str(temp_pix_ex_array.coords['isodate'].values)
            # temp_pixex_df['AtCor'] = 'ac4icw'

            pixex_df = pd.concat([pixex_df, temp_pixex_df], axis=0)

        if output_box is not None and os.path.isdir(output_box):

            polygons = []
            uuids = []

            for uuid, window in window_dict.items():
                # Get pixel coordinates from window slices
                y_start = int(window.row_off)
                y_stop = y_start + int(window.height)
                x_start = int(window.col_off)
                x_stop = x_start + int(window.width)

                # Get projected coordinates using affine transform
                x0, y0 = helper.transform_xy(
                    self.Affine, rows=[y_start - 0.5], cols=[x_start - 0.5]
                )
                x1, y1 = helper.transform_xy(
                    self.Affine, rows=[y_stop - 0.5], cols=[x_stop - 0.5]
                )

                # Create polygon (bounding box)
                poly = box(x0[0], y0[0], x1[0], y1[0])
                polygons.append(poly)
                uuids.append(uuid)

            gdf = gpd.GeoDataFrame(
                {"uuid": uuids, "geometry": polygons}, crs=self.CRS.to_epsg()
            )
            geojson_path = os.path.join(
                output_box, self.image_name + "_windows.geojson"
            )
            gdf.to_file(geojson_path, driver="GeoJSON")

        return pixex_df, window_dict, window_data

    def extract_pixel_line(self, line_index: int, line_window: int):
        xr_ds = self.in_ds

        # Stack the 'y' and 'x' dims into a single dimension 'yx'
        stacked_ds = xr_ds.stack(yx=("y", "x"))

        # Assign 'LineIndex' as a coordinate to the stacked Dataset
        LineIndex = xr_ds["LineIndex"].values.flatten()
        stacked_ds = stacked_ds.assign_coords(LineIndex=("yx", LineIndex))

        # Swap the 'yx' dimension with 'LineIndex'
        xr_ds_swapped = stacked_ds.swap_dims({"yx": "LineIndex"})

        # Method of transforming LineIndex as a coordinate variable and reindexing the dataset onto it
        # Using .sel work but only for a single value of line_index as ther is
        # necessary duplicate in the coordinate variable.
        # This break the assumption that coordinate variables are monotic.
        # See: https://docs.xarray.dev/en/latest/user-guide/indexing.html
        # and: https://docs.unidata.ucar.edu/nug/current/netcdf_data_set_components.html#coordinate_variables
        # extracted_line = xr_ds_swapped.sel(LineIndex=line_index)

        # To select more than one line we can create a boolean mask for the line window
        # A bit slow
        mask = (xr_ds_swapped["LineIndex"] >= line_index) & (
            xr_ds_swapped["LineIndex"] <= line_index + line_window
        )
        extracted_line = xr_ds_swapped.where(mask, drop=True)

        extracted_line = extracted_line.to_dataframe()

        return extracted_line
        