#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes

from shapely.geometry import shape as shp_shape
import geopandas as gpd

from pyproj import Transformer
from skimage.color import rgb2lab
from skimage.segmentation import felzenszwalb
from skimage.filters import gaussian


def get_crs_and_transform(ds: xr.Dataset, data_var: str = "rho_w") -> Tuple[rasterio.crs.CRS, rasterio.Affine]:
    gm_name = ds[data_var].attrs.get("grid_mapping")
    if not gm_name:
        raise ValueError(f"{data_var} has no grid_mapping attribute")

    gm = ds[gm_name]
    wkt = (
        gm.attrs.get("spatial_ref")
        or gm.attrs.get("crs_wkt")
        or gm.attrs.get("WKT")
        or gm.attrs.get("proj4text")
    )
    if not wkt:
        raise ValueError(f"Could not find CRS WKT in grid mapping variable '{gm_name}'")

    crs = rasterio.crs.CRS.from_wkt(wkt)

    x = ds["x"].values
    y = ds["y"].values
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    height = int(ds.dims["y"])
    width = int(ds.dims["x"])
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    return crs, transform

def spatial_lowpass_inplace(
    nc_path: str,
    rho_var: str,
    sigma_xy: float = 1.2,
    truncate: float = 4.0,
) -> None:
    """
    In-place spatial low-pass (Gaussian) of rho_var in a NetCDF file.
    Handles NaNs by normalized convolution: blur(data*mask)/blur(mask).
    """
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        if rho_var not in ds:
            raise KeyError(f"{rho_var} not found in {nc_path}")

        rho = ds[rho_var]  # (wl,y,x) typically
        if rho.ndim != 3:
            raise ValueError(f"{rho_var} must be 3D (wl,y,x). Got dims={rho.dims}")

        wl_dim, y_dim, x_dim = rho.dims
        R = rho.values.astype(np.float32)  # (nwl, ny, nx)

        # mask of finite values
        M = np.isfinite(R).astype(np.float32)
        R0 = np.where(np.isfinite(R), R, 0.0).astype(np.float32)

        Rout = np.empty_like(R0, dtype=np.float32)

        # Blur each wavelength band in 2D
        for k in range(R.shape[0]):
            num = gaussian(
                R0[k],
                sigma=sigma_xy,
                truncate=truncate,
                preserve_range=True,
                channel_axis=None,
            )
            den = gaussian(
                M[k],
                sigma=sigma_xy,
                truncate=truncate,
                preserve_range=True,
                channel_axis=None,
            )
            Rout[k] = num / (den + 1e-12)

        # restore NaNs where we had no support
        Rout[M == 0] = np.nan

        # preserve attrs/encoding as much as possible
        ds2 = ds.copy()
        ds2[rho_var] = (rho.dims, Rout, rho.attrs)

        tmp = nc_path + ".tmp_lp.nc"
        ds2.to_netcdf(tmp)

    os.replace(tmp, nc_path)

def crop_to_temp(
    nc_path: str,
    out_dir: str,
    bbox: Optional[dict],
    rho_var: str = "rho_w",
    out_name: str = "temp_crop.nc",
    wavelength_filter: tuple = None
) -> str:
    if bbox is None:
        return nc_path

    with xr.open_dataset(nc_path, engine="netcdf4") as ds:

        # --- simple wavelength range filter (lo, hi)
        if wavelength_filter is not None:
            lo, hi = wavelength_filter
            if lo > hi:
                lo, hi = hi, lo

            rho = ds[rho_var]
            wl_dim = rho.dims[0]  # spectral dimension
            ds = ds.sel({wl_dim: slice(lo, hi)})

        crs, _ = get_crs_and_transform(ds, data_var=rho_var)
        transformer = Transformer.from_crs("EPSG:4326", crs.to_string(), always_xy=True)

        x_min, y_min = transformer.transform(bbox["lon"][0], bbox["lat"][0])
        x_max, y_max = transformer.transform(bbox["lon"][1], bbox["lat"][1])

        y_desc = bool(ds["y"][0] > ds["y"][-1])
        y0, y1 = (y_max, y_min) if y_desc else (y_min, y_max)

        ds_crop = ds.sel(x=slice(x_min, x_max), y=slice(y0, y1))

        os.makedirs(out_dir, exist_ok=True)
        temp_crop = os.path.join(out_dir, out_name)
        ds_crop.to_netcdf(temp_crop)

    return temp_crop


def build_rgb_lab(
    ds: xr.Dataset,
    rho_var: str = "rho_w",
    rgb_wl: Tuple[float, float, float] = (665, 560, 490),
    clip: Tuple[float, float] = (1, 99),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rho = ds[rho_var]
    wl_dim = rho.dims[0]
    wl = rho[wl_dim].values.astype(float)

    def pick_band(target: float) -> int:
        i = int(np.argmin(np.abs(wl - target)))
        if np.abs(wl[i] - target) > 25:
            print(f"[warn] RGB target {target} nm not present; using {wl[i]:.1f} nm instead", file=sys.stderr)
        return i

    iR, iG, iB = map(pick_band, rgb_wl)

    R = rho.isel({wl_dim: iR}).values.astype(np.float32)
    G = rho.isel({wl_dim: iG}).values.astype(np.float32)
    B = rho.isel({wl_dim: iB}).values.astype(np.float32)

    valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
    valid &= (R >= 0) & (G >= 0) & (B >= 0)

    def stretch(ch: np.ndarray) -> np.ndarray:
        if not np.any(valid):
            return np.zeros_like(ch, dtype=np.float32)
        lo, hi = np.nanpercentile(ch[valid], clip)
        ch2 = np.clip(ch, lo, hi)
        return (ch2 - lo) / (hi - lo + 1e-12)

    rgb = np.stack([stretch(R), stretch(G), stretch(B)], axis=-1)
    lab = rgb2lab(rgb)
    return rgb, lab, valid, wl


def segment_on_lab(
    lab: np.ndarray,
    valid: np.ndarray,
    scale: float = 400,
    sigma: float = 1.0,
    min_size: int = 50,
) -> np.ndarray:
    lab2 = lab.copy()
    lab2[~valid] = 0.0

    labels = felzenszwalb(lab2, scale=scale, sigma=sigma, min_size=min_size).astype(np.int32)
    labels[~valid] = 0

    u = np.unique(labels)
    u = u[u != 0]
    remap = {int(old): i + 1 for i, old in enumerate(u)}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out

def segment_stats(
    ds: xr.Dataset,
    labels: np.ndarray,
    rho_var: str = "rho_w_g21",
    sun_var: str = "sun_zenith",
    view_var: str = "view_zenith",
    depth_var: str = "bathymetry_nonna10",
):

    rho = ds[rho_var]
    wl_dim = rho.dims[0]
    wl = rho[wl_dim].values.astype(float)

    R = rho.values.astype(np.float32)  # (wl,y,x)
    nwl, ny, nx = R.shape

    X = np.moveaxis(R, 0, -1).reshape(-1, nwl)
    lab = labels.reshape(-1)

    sun = ds[sun_var].values.reshape(-1).astype(np.float32) if sun_var in ds else None
    view = ds[view_var].values.reshape(-1).astype(np.float32) if view_var in ds else None
    h_w = ds[depth_var].values.reshape(-1).astype(np.float32) if depth_var in ds else None

    seg_ids = np.unique(lab)
    seg_ids = seg_ids[seg_ids != 0]

    med = np.full((len(seg_ids), nwl), np.nan, dtype=np.float32)
    mad = np.full((len(seg_ids), nwl), np.nan, dtype=np.float32)
    npx = np.zeros(len(seg_ids), dtype=np.int32)

    theta_sun = np.full(len(seg_ids), np.nan, dtype=np.float32)
    theta_view = np.full(len(seg_ids), np.nan, dtype=np.float32)
    h_w_mu = np.full(len(seg_ids), np.nan, dtype=np.float32)
    h_w_sd = np.full(len(seg_ids), np.nan, dtype=np.float32)

    for i, sid in enumerate(seg_ids.astype(int)):
        idx = np.where(lab == sid)[0]
        Xi = X[idx]
        ok = np.all(np.isfinite(Xi), axis=1)
        idx = idx[ok]
        Xi = Xi[ok]
        if Xi.shape[0] == 0:
            continue

        npx[i] = Xi.shape[0]
        m = np.median(Xi, axis=0)
        med[i] = m
        mad[i] = np.median(np.abs(Xi - m[None, :]), axis=0)

        if sun is not None:
            theta_sun[i] = np.nanmedian(sun[idx])
        if view is not None:
            theta_view[i] = np.nanmedian(view[idx])
        if h_w is not None:
            h_w_mu[i] = np.nanmedian(h_w[idx])
            h_w_sd[i] = np.nanstd(h_w[idx])

    return seg_ids, npx, wl, med, mad, theta_sun, theta_view, h_w_mu, h_w_sd


def write_seg_id_inplace(nc_path: str, labels: np.ndarray, var_name: str = "seg_id", fill_value: int = 0) -> None:
    with netCDF4.Dataset(nc_path, mode="a") as nc:
        if "y" not in nc.dimensions or "x" not in nc.dimensions:
            raise ValueError("NetCDF must have 'y' and 'x' dimensions")

        ny = len(nc.dimensions["y"])
        nx = len(nc.dimensions["x"])
        if labels.shape != (ny, nx):
            raise ValueError(f"labels shape {labels.shape} != (y,x)=({ny},{nx})")

        if var_name in nc.variables:
            v = nc.variables[var_name]
        else:
            v = nc.createVariable(var_name, "i4", ("y", "x"), zlib=True, complevel=1, fill_value=fill_value)
            v.long_name = "Segmentation label id (0 = invalid/masked)"

        v[:, :] = labels.astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# Write polygons layer (segments) into GeoPackage, INCLUDING scalar fields
# -----------------------------------------------------------------------------
def write_segments_layer(
    out_gpkg: str,
    layer: str,
    labels: np.ndarray,
    ds: xr.Dataset,
    seg_ids: np.ndarray,
    npx: np.ndarray,
    theta_sun: np.ndarray,
    theta_view: np.ndarray,
    h_w_mu: np.ndarray,
    h_w_sd: np.ndarray,
    rho_var_for_crs: str = "rho_w",
) -> gpd.GeoDataFrame:
    crs, transform = get_crs_and_transform(ds, data_var=rho_var_for_crs)

    mask = labels != 0
    geoms, vals = [], []
    for geom, val in shapes(labels.astype(np.int32), mask=mask, transform=transform):
        geoms.append(shp_shape(geom))
        vals.append(int(val))

    gdf = gpd.GeoDataFrame({"seg_id": vals, "geometry": geoms}, crs=crs)

    sid_to_i = {int(s): i for i, s in enumerate(seg_ids.astype(int))}

    def _get(arr: np.ndarray, sid: int) -> float:
        i = sid_to_i.get(int(sid), None)
        if i is None:
            return np.nan
        v = float(arr[i])
        return v if np.isfinite(v) else np.nan

    gdf["n_pix"] = [int(npx[sid_to_i[sid]]) if sid in sid_to_i else None for sid in gdf["seg_id"].astype(int)]
    gdf["theta_sun"] = [ _get(theta_sun, sid) for sid in gdf["seg_id"].astype(int) ]
    gdf["theta_view"] = [ _get(theta_view, sid) for sid in gdf["seg_id"].astype(int) ]
    gdf["h_w_mu"] = [ _get(h_w_mu, sid) for sid in gdf["seg_id"].astype(int) ]
    gdf["h_w_sd"] = [ _get(h_w_sd, sid) for sid in gdf["seg_id"].astype(int) ]

    # gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
    # return gdf

    # after gdf is built and scalar columns are added...

    # make sure types are consistent
    gdf["seg_id"] = gdf["seg_id"].astype(int)
    gdf["n_pix"] = pd.to_numeric(gdf["n_pix"], errors="coerce")

    # dissolve: geopandas handles geometry union internally
    agg = {
        "n_pix": "sum",
        "theta_sun": "first",
        "theta_view": "first",
        "h_w_mu": "first",
        "h_w_sd": "first",
    }

    gdf = gdf.dissolve(by="seg_id", as_index=False, aggfunc=agg)

    # optional: clean geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)

    gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
    return gdf


def write_spectra_layer(
    out_gpkg: str,
    layer: str,
    seg_ids: np.ndarray,
    wl: np.ndarray,
    rho_med: np.ndarray,
    rho_mad: Optional[np.ndarray] = None,
) -> None:
    wl = np.asarray(wl).astype(float)

    rows = []
    for i, sid in enumerate(seg_ids.astype(int)):
        for j, w in enumerate(wl):
            row = {"seg_id": int(sid), "wl_nm": float(w), "rho_med": float(rho_med[i, j])}
            if rho_mad is not None:
                row["rho_mad"] = float(rho_mad[i, j])
            rows.append(row)

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry=[None] * len(df), crs=None)
    gdf.to_file(out_gpkg, layer=layer, driver="GPKG", mode="a")


def segment_and_export(
    nc_path: str,
    out_dir: str,
    bbox: Optional[dict],
    rho_var: str = "rho_w",
    wavelength_filter: Optional[Tuple[float, float]] = (370, 730),
    rgb_wl: Tuple[float, float, float] = (665, 560, 490),
) -> Tuple[str, str]:
    temp_nc = crop_to_temp(nc_path, out_dir=out_dir, bbox=bbox, rho_var=rho_var, wavelength_filter=wavelength_filter)

    # Apply spatial low-pass to rho_var INSIDE the temp crop file
    # spatial_lowpass_inplace(temp_nc, rho_var=rho_var, sigma_xy=1.8)

    with xr.open_dataset(temp_nc, engine="netcdf4") as ds:
        rgb, lab, valid, _ = build_rgb_lab(ds, rho_var=rho_var, rgb_wl=rgb_wl)

        # --- depth mask: remove pixels deeper than 10 m
        depth = ds["bathymetry_nonna10"].values.astype(np.float32)
        theta_view = ds["view_zenith"].values.astype(np.float32)
        valid &= np.isfinite(depth) & (depth <= 20) & np.isfinite(theta_view)

        labels = segment_on_lab(lab, valid, scale=400, sigma=1.0, min_size=50)

    write_seg_id_inplace(temp_nc, labels, var_name="seg_id", fill_value=0)

    with xr.open_dataset(temp_nc, engine="netcdf4") as ds:
        seg_ids, npx, wl, rho_med, rho_mad, theta_sun, theta_view, h_w_mu, h_w_sd = segment_stats(
            ds,
            labels,
            rho_var=rho_var,
            sun_var="sun_zenith",
            view_var="view_zenith",
            depth_var="bathymetry_nonna10",
        )

        print("wl length:", len(wl))

        out_gpkg = os.path.join(out_dir, "segments.gpkg")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_gpkg):
            os.remove(out_gpkg)

        write_segments_layer(
            out_gpkg, "segments",
            labels, ds,
            seg_ids, npx,
            theta_sun, theta_view, h_w_mu, h_w_sd,
            rho_var_for_crs=rho_var,
        )
        write_spectra_layer(out_gpkg, "spectra", seg_ids, wl, rho_med, rho_mad)

    return temp_nc, out_gpkg


if __name__ == "__main__":
    nc_path = "/D/Data/WISE/ACI-12A/el_sma/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc"
    out_dir = os.path.join(os.path.dirname(nc_path), "segmentation_outputs")

    # reduced for optimization testing
    # bbox = {"lon": (-64.37024, -64.35578), "lat": (49.77804, 49.78492)}
    # Ultra reduce for sampling testing
    # bbox = {"lon": (-64.37031, -64.36776), "lat": (49.778227, 49.780921)}
    # For publication
    bbox = {"lon": (-64.3771, -64.35086), "lat": (49.77186, 49.79783)}

    temp_nc, gpkg = segment_and_export(
        nc_path=nc_path,
        out_dir=out_dir,
        bbox=bbox,
        rho_var="rho_w_g21",
        wavelength_filter=(370, 730),
        rgb_wl=(665, 560, 490),
    )

    print("Wrote cropped NetCDF:", temp_nc)
    print("Wrote GeoPackage:", gpkg)


# QGIS action code to map polygon selection -> spectrum table filtering
# from qgis.core import QgsProject, QgsVectorLayer
#
# seg_id = [% "seg_id" %]
#
# proj = QgsProject.instance()
#
# tbl = proj.mapLayersByName("segments — spectra")[0]  # <-- exact layer name
# tbl.removeSelection()
#
# # build expression depending on seg_id type
# if isinstance(seg_id, (int, float)):
#     expr = f"\"seg_id\" = {int(seg_id)}"
# else:
#     expr = f"\"seg_id\" = '{seg_id}'"
#
# tbl.selectByExpression(expr, QgsVectorLayer.SetSelection)


# Paste in QGIS python console
# from qgis.core import QgsProject, QgsVectorLayer
#
# proj = QgsProject.instance()
#
# POLY_LAYER_NAME = "segments — aux"        # <-- the polygon layer you click
# SPECTRA_LAYER_NAME = "segments — spectra" # <-- table to select rows in
#
# poly = proj.mapLayersByName(POLY_LAYER_NAME)[0]
# tbl  = proj.mapLayersByName(SPECTRA_LAYER_NAME)[0]
#
# def _sync_spectra_selection(selected, deselected, clear_and_select):
#     if not selected:
#         tbl.removeSelection()
#         return
#
#     f = next(poly.getFeatures(f"id={selected[0]}"), None)
#     if f is None:
#         return
#
#     seg_id = f["seg_id"]
#     tbl.removeSelection()
#
#     if isinstance(seg_id, (int, float)):
#         expr = f"\"seg_id\" = {int(seg_id)}"
#     else:
#         expr = f"\"seg_id\" = '{seg_id}'"
#
#     tbl.selectByExpression(expr, QgsVectorLayer.SetSelection)
#
# try:
#     poly.selectionChanged.disconnect(_sync_spectra_selection)
# except Exception:
#     pass
#
# poly.selectionChanged.connect(_sync_spectra_selection)
#
# print("OK: clicking polygons now syncs selection in the spectra table.")