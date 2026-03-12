#!/usr/bin/env python3
import os
import re
import json
import math
import logging
import numpy as np
import xarray as xr
import netCDF4

import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes
from shapely.geometry import shape as shp_shape
import geopandas as gpd

from sklearn.decomposition import IncrementalPCA
from skimage.segmentation import slic
from pyproj import Transformer


# -----------------------------
# CRS + Transform from grid_mapping
# -----------------------------
def get_crs_and_transform(ds: xr.Dataset, data_var="rho_w_g21"):
    gm_name = ds[data_var].attrs.get("grid_mapping", None)
    if gm_name is None:
        raise ValueError(f"{data_var} has no grid_mapping attribute")

    gm = ds[gm_name]
    wkt = gm.attrs.get("spatial_ref") or gm.attrs.get("crs_wkt") or gm.attrs.get("WKT") or gm.attrs.get("proj4text")
    if wkt is None:
        raise ValueError(f"Could not find CRS WKT in grid mapping variable '{gm_name}'")

    crs = rasterio.crs.CRS.from_wkt(wkt)

    x = ds["x"].values
    y = ds["y"].values
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    height = ds.dims["y"]
    width  = ds.dims["x"]
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    return crs, transform


# -----------------------------
# Crop helper (EPSG:4326 bbox -> dataset CRS, writes temp_crop.nc)
# -----------------------------
def crop_to_temp(nc_path: str, out_dir: str, bbox: dict | None, rho_var="rho_w_g21"):
    """
    Returns path to dataset to use:
      - if bbox is None: nc_path
      - else: writes and returns temp_crop.nc in out_dir
    """
    if bbox is None:
        return nc_path

    ds = xr.open_dataset(nc_path, engine="netcdf4")

    crs, _ = get_crs_and_transform(ds, data_var=rho_var)
    transformer = Transformer.from_crs("EPSG:4326", crs.to_string(), always_xy=True)

    x_min, y_min = transformer.transform(bbox["lon"][0], bbox["lat"][0])
    x_max, y_max = transformer.transform(bbox["lon"][1], bbox["lat"][1])

    # y may be descending
    y_desc = bool(ds["y"][0] > ds["y"][-1])
    if y_desc:
        y0, y1 = y_max, y_min
    else:
        y0, y1 = y_min, y_max

    ds_crop = ds.sel(x=slice(x_min, x_max), y=slice(y0, y1))

    os.makedirs(out_dir, exist_ok=True)
    temp_crop = os.path.join(out_dir, "temp_crop.nc")
    ds_crop.to_netcdf(temp_crop)

    ds_crop.close()
    ds.close()
    return temp_crop


# -----------------------------
# Segmentation scale from pixel size + smallest feature size
# -----------------------------
def target_pixels_per_segment(pixel_size_m=1.5, feature_size_m=10.0):
    # smallest features ~10m square area => about (10/1.5)^2 pixels
    return int(round((feature_size_m / pixel_size_m) ** 2))


# -----------------------------
# Segment on PCA of normalized spectra using SLIC
# -----------------------------
def segment_rho_slic(ds: xr.Dataset,
                     rho_var="rho_w_g21",
                     n_pca=6,
                     compactness=0.3,
                     target_pps=50,
                     seed=1,
                     mask=None):
    rho = ds[rho_var]  # (wl,y,x)
    wl_dim = rho.dims[0]
    wl = rho[wl_dim].values

    R = rho.values.astype(np.float32)  # (wl,y,x)
    nwl, ny, nx = R.shape

    if mask is None:
        valid = np.all(np.isfinite(R), axis=0)
    else:
        valid = mask & np.all(np.isfinite(R), axis=0)

    X = np.moveaxis(R, 0, -1).reshape(-1, nwl)   # (n_pix, wl)
    v = valid.reshape(-1)
    n_valid = int(v.sum())
    if n_valid == 0:
        raise ValueError("No valid pixels for segmentation")

    # n_segments from target pixels/segment
    n_segments = max(200, int(round(n_valid / max(1, target_pps))))
    n_segments = int(np.clip(n_segments, 200, 200000))

    Xv = X[v]

    # magnitude feature (log brightness)
    mag = np.log(np.sum(np.clip(Xv, 0, None), axis=1) + 1e-12).astype(np.float32)

    # normalize to emphasize spectral shape
    denom = np.sum(Xv, axis=1, keepdims=True)
    Xv_norm = Xv / (denom + 1e-12)

    # PCA fit on subsample
    rng = np.random.default_rng(seed)
    fit_n = min(300_000, n_valid)
    fit_idx = rng.choice(n_valid, size=fit_n, replace=False)
    Xfit = Xv_norm[fit_idx]

    ipca = IncrementalPCA(n_components=n_pca, batch_size=100_000)
    ipca.fit(Xfit)

    PCs_v = ipca.transform(Xv_norm).astype(np.float32)

    # standardize PCs
    mu = PCs_v.mean(axis=0, keepdims=True)
    sd = PCs_v.std(axis=0, keepdims=True) + 1e-12
    PCs_v = (PCs_v - mu) / sd

    # add magnitude as an extra channel (standardized)
    mag = (mag - mag.mean()) / (mag.std() + 1e-12)
    feat_v = np.concatenate([PCs_v, mag[:, None]], axis=1)  # (n_valid, n_pca+1)

    # image features (y,x,channels)
    feat = np.zeros((ny * nx, feat_v.shape[1]), dtype=np.float32)
    feat[v] = feat_v
    feat_img = feat.reshape(ny, nx, feat_v.shape[1])

    print("PC std (valid):", PCs_v.std(axis=0))
    print("mag std:", mag.std())

    # labels = slic(
    #     feat_img,
    #     n_segments=n_segments,
    #     compactness=float(compactness),   # try 0.1 first
    #     sigma=1.0,                        # add
    #     start_label=1,
    #     channel_axis=2,
    #     enforce_connectivity=True,
    #     slic_zero=True
    # ).astype(np.int32)

    from skimage.segmentation import felzenszwalb

    labels = felzenszwalb(
        feat_img[..., :5],  # e.g., first 4 PCs + mag
        scale=500,  # larger => fewer/larger regions
        sigma=1.0,
        min_size=50  # ~ your smallest feature in pixels
    ).astype(np.int32)
    labels[~valid] = 0

    labels[~valid] = 0
    return labels, wl



# -----------------------------
# Per-segment median + MAD spectra + some summary stats
# -----------------------------
def segment_stats(ds: xr.Dataset,
                  labels: np.ndarray,
                  rho_var="rho_w_g21",
                  sun_var="sun_zenith",
                  view_var="view_zenith",
                  depth_var="bathymetry_nonna10"):
    R = ds[rho_var].values.astype(np.float32)  # (wl,y,x)
    nwl, ny, nx = R.shape
    X = np.moveaxis(R, 0, -1).reshape(-1, nwl)
    lab = labels.reshape(-1)

    sun = ds[sun_var].values.reshape(-1).astype(np.float32) if sun_var in ds else None
    view = ds[view_var].values.reshape(-1).astype(np.float32) if view_var in ds else None
    dep = ds[depth_var].values.reshape(-1).astype(np.float32) if depth_var in ds else None

    seg_ids = np.unique(lab)
    seg_ids = seg_ids[seg_ids != 0]

    med = np.full((len(seg_ids), nwl), np.nan, dtype=np.float32)
    mad = np.full((len(seg_ids), nwl), np.nan, dtype=np.float32)
    npx = np.zeros(len(seg_ids), dtype=np.int32)

    sun_m = np.full(len(seg_ids), np.nan, dtype=np.float32)
    view_m = np.full(len(seg_ids), np.nan, dtype=np.float32)
    dep_m = np.full(len(seg_ids), np.nan, dtype=np.float32)

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

        if sun is not None:  sun_m[i] = np.nanmedian(sun[idx])
        if view is not None: view_m[i] = np.nanmedian(view[idx])
        if dep is not None:  dep_m[i] = np.nanmedian(dep[idx])

    return seg_ids, npx, med, mad, sun_m, view_m, dep_m


# -----------------------------
# Write seg_id into NetCDF in-place
# -----------------------------
def write_seg_id_inplace(nc_path: str, labels: np.ndarray, var_name="seg_id", fill_value=0):
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
            v = nc.createVariable(var_name, "i4", ("y", "x"),
                                  zlib=True, complevel=1, fill_value=fill_value)
            v.long_name = "Segmentation label id (0 = invalid/masked)"

        v[:, :] = labels.astype(np.int32, copy=False)


# -----------------------------
# Write polygons + spectra into GeoPackage
# -----------------------------
def write_segments_gpkg(labels: np.ndarray,
                        ds: xr.Dataset,
                        wl: np.ndarray,
                        seg_ids: np.ndarray,
                        npx: np.ndarray,
                        rho_med: np.ndarray,
                        rho_mad: np.ndarray,
                        out_gpkg: str,
                        layer="segments",
                        store_wavelength_layer=True):
    crs, transform = get_crs_and_transform(ds, data_var="rho_w_g21")

    mask = labels != 0
    geoms, vals = [], []
    for geom, val in shapes(labels.astype(np.int32), mask=mask, transform=transform):
        geoms.append(shp_shape(geom))
        vals.append(int(val))

    gdf = gpd.GeoDataFrame({"seg_id": vals, "geometry": geoms}, crs=crs)

    sid_to_i = {int(s): i for i, s in enumerate(seg_ids.astype(int))}

    n_pix_col, med_json, mad_json = [], [], []
    for sid in gdf["seg_id"].astype(int).values:
        i = sid_to_i.get(sid, None)
        if i is None:
            n_pix_col.append(None); med_json.append(None); mad_json.append(None)
        else:
            n_pix_col.append(int(npx[i]))
            med_json.append(json.dumps(rho_med[i].astype(float).tolist()))
            mad_json.append(json.dumps(rho_mad[i].astype(float).tolist()))

    gdf["n_pix"] = n_pix_col
    gdf["rho_med"] = med_json
    gdf["rho_mad"] = mad_json

    os.makedirs(os.path.dirname(out_gpkg), exist_ok=True)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    gdf.to_file(out_gpkg, layer=layer, driver="GPKG")

    if store_wavelength_layer:
        # store wavelength vector once in a non-spatial layer
        meta = gpd.GeoDataFrame(
            {"key": ["wavelength_nm"], "value": [json.dumps(np.asarray(wl).astype(float).tolist())]},
            geometry=[None],
            crs=crs
        )
        meta.to_file(out_gpkg, layer="meta", driver="GPKG")

    return gdf

def write_segments_gpkg_bandcols(labels: np.ndarray,
                                ds: xr.Dataset,
                                wl: np.ndarray,
                                seg_ids: np.ndarray,
                                npx: np.ndarray,
                                rho_med: np.ndarray,
                                rho_mad: np.ndarray,
                                out_gpkg: str,
                                layer="segments",
                                store_mad=True):
    crs, transform = get_crs_and_transform(ds, data_var="rho_w_g21")

    mask = labels != 0
    geoms, vals = [], []
    for geom, val in shapes(labels.astype(np.int32), mask=mask, transform=transform):
        geoms.append(shp_shape(geom))
        vals.append(int(val))

    gdf = gpd.GeoDataFrame({"seg_id": vals, "geometry": geoms}, crs=crs)

    sid_to_i = {int(s): i for i, s in enumerate(seg_ids.astype(int))}

    # base attrs
    gdf["n_pix"] = [int(npx[sid_to_i[sid]]) if sid in sid_to_i else None for sid in gdf["seg_id"].astype(int)]

    # band columns
    wl_int = [int(round(w)) for w in wl]
    for j, w in enumerate(wl_int):
        gdf[f"m_{w}"] = [float(rho_med[sid_to_i[sid], j]) if sid in sid_to_i else np.nan
                         for sid in gdf["seg_id"].astype(int)]
        if store_mad:
            gdf[f"d_{w}"] = [float(rho_mad[sid_to_i[sid], j]) if sid in sid_to_i else np.nan
                             for sid in gdf["seg_id"].astype(int)]

    os.makedirs(os.path.dirname(out_gpkg), exist_ok=True)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
    return gdf

# -----------------------------
# Main
# -----------------------------
def segment_and_export(
    nc_path: str,
    out_dir: str,
    bbox: dict | None,
    rho_var="rho_w_g21",
    pixel_size_m=1.5,
    feature_size_m=100.0,   # smallest feature ~100m square
    n_pca=6,
    slic_compactness=0.1,
):
    # 1) Create temp_crop.nc if bbox provided (matches your workflow)
    temp_nc = crop_to_temp(nc_path, out_dir=out_dir, bbox=bbox, rho_var=rho_var)

    # 2) Open cropped dataset for segmentation
    ds = xr.open_dataset(temp_nc, engine="netcdf4")

    # Basic valid mask (plug your water mask here if you have one)
    R = ds[rho_var]
    valid = np.isfinite(R).all(dim=R.dims[0]).values
    valid &= (np.nanmean(R.values, axis=0) > 0)

    # 3) Segment
    # target_pps = target_pixels_per_segment(pixel_size_m=pixel_size_m, feature_size_m=feature_size_m)
    # labels, wl = segment_rho_slic(
    #     ds,
    #     rho_var=rho_var,
    #     n_pca=n_pca,
    #     compactness=slic_compactness,
    #     target_pps=target_pps,
    #     seed=1,
    #     mask=valid
    # )

    rgb, lab, valid, wl = build_rgb_lab(ds, rho_var="rho_w_g21", rgb_wl=(665, 560, 490))
    labels = segment_on_lab(lab, valid, scale=400, sigma=1.0, min_size=50)

    # 4) Write seg_id into the SAME cropped NetCDF (in-place)
    ds.close()  # close xarray handle before netCDF4 append
    write_seg_id_inplace(temp_nc, labels, var_name="seg_id", fill_value=0)

    # 5) Re-open to compute per-segment spectra and export GPKG
    ds = xr.open_dataset(temp_nc, engine="netcdf4")
    seg_ids, npx, rho_med, rho_mad, *_ = segment_stats(ds, labels, rho_var=rho_var)

    out_gpkg = os.path.join(out_dir, "segments.gpkg")
    # write_segments_gpkg(
    #     labels=labels,
    #     ds=ds,
    #     wl=wl,
    #     seg_ids=seg_ids,
    #     npx=npx,
    #     rho_med=rho_med,
    #     rho_mad=rho_mad,
    #     out_gpkg=out_gpkg,
    #     layer="segments",
    #     store_wavelength_layer=True
    # )

    # write_segments_gpkg_bandcols(
    #     labels=labels,
    #     ds=ds,
    #     wl=wl,
    #     seg_ids=seg_ids,
    #     npx=npx,
    #     rho_med=rho_med,
    #     rho_mad=rho_mad,
    #     out_gpkg=out_gpkg,
    #     layer="segments",
    # )

    # write_spectra_long_table_gpkg(
    #     out_gpkg=out_gpkg,
    #     seg_ids=seg_ids,
    #     wl=wl,
    #     rho_med=rho_med,
    #     rho_mad=rho_mad,
    #     npx=npx,
    #     layer="spectra"
    # )

    write_spectra_layer_same_gpkg(out_gpkg, seg_ids, wl, rho_med, rho_mad, layer="spectra")

    ds.close()
    return temp_nc, out_gpkg


from skimage.color import rgb2lab
from skimage.exposure import rescale_intensity

def build_rgb_lab(ds, rho_var="rho_w_g21", rgb_wl=(665, 560, 490), clip=(1, 99)):
    rho = ds[rho_var]                      # (wl,y,x)
    wl = rho[rho.dims[0]].values.astype(float)

    def pick_band(target):
        return int(np.argmin(np.abs(wl - target)))

    iR, iG, iB = map(pick_band, rgb_wl)

    R = rho.isel({rho.dims[0]: iR}).values.astype(np.float32)
    G = rho.isel({rho.dims[0]: iG}).values.astype(np.float32)
    B = rho.isel({rho.dims[0]: iB}).values.astype(np.float32)

    # valid mask: finite and non-negative
    valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
    valid &= (R >= 0) & (G >= 0) & (B >= 0)

    # robust stretch each channel for segmentation stability
    def stretch(ch):
        lo, hi = np.nanpercentile(ch[valid], clip)
        ch2 = np.clip(ch, lo, hi)
        ch2 = (ch2 - lo) / (hi - lo + 1e-12)
        return ch2

    Rn, Gn, Bn = stretch(R), stretch(G), stretch(B)
    rgb = np.stack([Rn, Gn, Bn], axis=-1)     # (y,x,3) in [0,1]

    lab = rgb2lab(rgb)                        # (y,x,3)
    return rgb, lab, valid, wl

from skimage.segmentation import felzenszwalb

def segment_on_lab(lab, valid, scale=400, sigma=1.0, min_size=50):
    # Felzenszwalb expects finite everywhere; fill invalid with 0
    lab2 = lab.copy()
    lab2[~valid] = 0.0

    labels = felzenszwalb(
        lab2,
        scale=scale,     # higher => larger segments
        sigma=sigma,     # smoothing
        min_size=min_size
    ).astype(np.int32)

    labels[~valid] = 0
    # relabel to be compact and start at 1
    u = np.unique(labels)
    u = u[u != 0]
    remap = {int(old): i+1 for i, old in enumerate(u)}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out

# import pandas as pd
# import geopandas as gpd
# import numpy as np
# import os
#
# def write_spectra_long_table_gpkg(
#     out_gpkg: str,
#     seg_ids: np.ndarray,
#     wl: np.ndarray,
#     rho_med: np.ndarray,
#     rho_mad: np.ndarray | None = None,
#     npx: np.ndarray | None = None,
#     layer: str = "spectra"
# ):
#     # Build long table
#     rows = []
#     wl = np.asarray(wl).astype(float)
#
#     for i, sid in enumerate(seg_ids.astype(int)):
#         for j, w in enumerate(wl):
#             r = {
#                 "seg_id": int(sid),
#                 "wl_nm": float(w),
#                 "rho_med": float(rho_med[i, j]),
#             }
#             if rho_mad is not None:
#                 r["rho_mad"] = float(rho_mad[i, j])
#             if npx is not None:
#                 r["n_pix"] = int(npx[i])
#             rows.append(r)
#
#     df = pd.DataFrame(rows)
#
#     # Write as a non-spatial table: GeoDataFrame with null geometry
#     gdf = gpd.GeoDataFrame(df, geometry=[None] * len(df), crs=None)
#
#     # Append to existing gpkg (segments already written), or create if missing
#     os.makedirs(os.path.dirname(out_gpkg), exist_ok=True)
#     gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
#
#     return out_gpkg

import pandas as pd
import geopandas as gpd
import numpy as np

def write_spectra_layer_same_gpkg(out_gpkg, seg_ids, wl, rho_med, rho_mad=None, layer="spectra"):
    wl = np.asarray(wl).astype(float)

    rows = []
    for i, sid in enumerate(seg_ids.astype(int)):
        for j, w in enumerate(wl):
            r = {"seg_id": int(sid), "wl_nm": float(w), "rho_med": float(rho_med[i, j])}
            if rho_mad is not None:
                r["rho_mad"] = float(rho_mad[i, j])
            rows.append(r)

    df = pd.DataFrame(rows)
    # non-spatial table inside gpkg
    gdf = gpd.GeoDataFrame(df, geometry=[None]*len(df), crs=None)
    gdf.to_file(out_gpkg, layer=layer, driver="GPKG")

if __name__ == "__main__":
    # ---- user inputs ----
    # nc_path = "/D/Data/WISE/ACI-13A/el_jetski/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc"
    nc_path = "/D/Data/WISE/ACI-12A/jetski_el/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc"
    out_dir = os.path.join(os.path.dirname(nc_path), "segmentation_outputs")

    # Use your bbox or None
    # bbox = {"lon": (-64.36871, -64.3615), "lat": (49.80857, 49.81336)}
    bbox = {"lon": (-64.37024, -64.35578), "lat": (49.77804, 49.78492)}
    # bbox = None

    temp_nc, gpkg = segment_and_export(
        nc_path=nc_path,
        out_dir=out_dir,
        bbox=bbox,
        rho_var="rho_w_g21",
        pixel_size_m=1.5,
        feature_size_m=10.0,
        n_pca=6,
        slic_compactness=0.01
    )

    print("Wrote cropped NetCDF:", temp_nc)
    print("Wrote segments GeoPackage:", gpkg)
