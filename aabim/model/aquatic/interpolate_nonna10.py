import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
import netCDF4 as nc

from rasterio.fill import fillnodata
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from rasterio.transform import array_bounds


from reverie import ReveCube


def interpolate_nonna10(
    cube,
    raster_paths,
    band_name="Depth",
    out_var="depth",
    merge_method="first",      # "first" | "min" | "max"
    resampling="bilinear",     # "nearest" | "bilinear" | "cubic"
    valid_range=None,          # e.g. (0, 200) or (-200, 0)
    dtype="float32",
):
    """
    Merge rasters first, then reproject/resample once onto the cube grid.
    """

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    rs = resampling_map.get(resampling, Resampling.bilinear)

    srcs = [rasterio.open(p) for p in raster_paths]
    try:
        # band index by description (fallback to 1)
        band_idxs = []
        for s in srcs:
            desc = list(s.descriptions or [])
            band_idxs.append(desc.index(band_name) + 1 if band_name in desc else 1)

        # NOTE: rasterio.merge assumes same band index across datasets.
        # If your "Depth" band index differs between files, fix the rasters or standardize beforehand.
        if len(set(band_idxs)) != 1:
            raise ValueError(f"Depth band index differs across rasters: {band_idxs}. Standardize band order/descriptions first.")

        bidx = band_idxs[0]

        # Mosaic (in CRS of first raster)
        mosaic, mosaic_transform = merge(
            srcs,
            indexes=bidx,
            method=merge_method,
            nodata=np.nan,
        )
        mosaic = mosaic[0].astype("float32", copy=False)
        mosaic_crs = srcs[0].crs

    finally:
        for s in srcs:
            s.close()

    # positive with depth
    mosaic = abs(mosaic)

    # --- crop mosaic to cube extent (in mosaic CRS), then fill nodata gaps there ---
    ny, nx = cube.n_rows, cube.n_cols

    # cube bounds in cube CRS -> (west, south, east, north)
    cube_bounds = array_bounds(ny, nx, cube.Affine)

    # transform cube bounds to mosaic CRS
    mb = transform_bounds(cube.CRS, mosaic_crs, *cube_bounds, densify_pts=21)
    west, south, east, north = mb

    # build a window on the mosaic that covers the cube extent
    win = from_bounds(west, south, east, north, transform=mosaic_transform)

    # clip window to mosaic array bounds
    r0 = max(0, int(np.floor(win.row_off)))
    c0 = max(0, int(np.floor(win.col_off)))
    r1 = min(mosaic.shape[0], int(np.ceil(win.row_off + win.height)))
    c1 = min(mosaic.shape[1], int(np.ceil(win.col_off + win.width)))

    mosaic_sub = mosaic[r0:r1, c0:c1]
    sub_transform = mosaic_transform * rasterio.Affine.translation(c0, r0)

    # fill gaps (NaNs) on the subset
    valid_mask = np.isfinite(mosaic_sub)  # True where data is valid
    mosaic_sub_filled = fillnodata(
        mosaic_sub.astype("float32", copy=False),
        mask=valid_mask,
        max_search_distance=200,  # pixels; increase if gaps are large
        smoothing_iterations=0
    )

    # # optional range filter
    # if valid_range is not None:
    #     lo, hi = valid_range
    #     mosaic = np.where((mosaic >= lo) & (mosaic <= hi), mosaic, np.nan)

    # --- now reproject the filled subset to cube grid ---
    dst = np.full((ny, nx), np.nan, dtype=dtype)

    reproject(
        source=mosaic_sub_filled,
        destination=dst,
        src_transform=sub_transform,
        src_crs=mosaic_crs,
        dst_transform=cube.Affine,
        dst_crs=cube.CRS,
        dst_shape=(ny, nx),
        resampling=rs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # Attach to dataset
    # da = xr.DataArray(
    #     dst,
    #     dims=("y", "x"),
    #     coords={"y": cube.in_ds["y"], "x": cube.in_ds["x"]},
    #     name=out_var,
    #     attrs={
    #         "long_name": "Bathymetry / water depth",
    #         "units": "m",
    #         "grid_mapping": "grid_mapping",
    #         "merge_method": merge_method,
    #         "resampling": resampling,
    #     },
    # )

    if cube.out_path is None:
        cube.out_path = cube.in_path
        cube.in_ds.close()

    cube.out_ds = nc.Dataset(cube.out_path, "a", format="NETCDF4")

    # remove existing variable if present
    if "bathymetry_nonna10" in cube.out_ds.variables:
        del cube.out_ds.variables["bathymetry_nonna10"]

    bathy = cube.out_ds.createVariable(
        "bathymetry_nonna10",
        "f4",
        ("y", "x"),
        zlib=True,
        complevel=1,
        fill_value=np.float32(np.nan),
    )
    # cube.create_var_nc(
    #     name="rho_w_g21",
    #     type="f4",
    #     dims=(
    #         "y",
    #         "x",
    #     ),
    #     comp="zlib",
    #     complevel=1,
    #     scale=1,
    # )

    bathy.grid_mapping = "grid_mapping"
    bathy.standard_name = "bathymetry_nonna10"
    bathy.units = "meter"
    bathy.long_name = "bathymetry_nonna10"
    bathy.description = "NONNA10 bathymetry"
    bathy[:] = dst
    # image.in_ds["aerosol_optical_thickness_at_555_nm"][:] = anc.variables["TOTEXTTAU"]

    cube.in_ds.close()

    # cube.in_ds[out_var] = da
    return cube

if __name__ == "__main__":

    nonna10_path = "/D/Documents/phd/thesis/4_chapter/data/nonna10/products_nonna10/Bathymetry/"

    # cube = ReveCube.from_reve_nc("/D/Data/WISE/ACI-13A/el_jetski/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc")
    cube = ReveCube.from_reve_nc("/D/Data/WISE/ACI-12A/el_sma/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc")

    interpolate_nonna10(
        cube,
        raster_paths=[nonna10_path+"NONNA10_4970N06440W.tiff", nonna10_path+"NONNA10_4970N06450W.tiff", nonna10_path+"NONNA10_4980N06440W.tiff", nonna10_path+"NONNA10_4980N06450W.tiff"],
        band_name="Depth",
        merge_method="min",      # pick shallowest if depth is positive-down
        resampling="bilinear",
        valid_range=(0, 200),
    )