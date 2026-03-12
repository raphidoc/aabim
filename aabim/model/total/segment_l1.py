#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional, Tuple, Dict

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


# -----------------------------------------------------------------------------
# CRS + Affine transform from NetCDF grid_mapping
# -----------------------------------------------------------------------------
def get_crs_and_transform(ds: xr.Dataset, data_var: str) -> Tuple[rasterio.crs.CRS, rasterio.Affine]:
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

    # netCDF4/xarray dims may be MappingProxyType; use sizes
    height = int(ds.sizes["y"])
    width = int(ds.sizes["x"])
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    return crs, transform


# -----------------------------------------------------------------------------
# Optional crop (EPSG:4326 bbox -> dataset CRS). Writes temp_crop.nc in out_dir.
# -----------------------------------------------------------------------------
def crop_to_temp(
    nc_path: str,
    out_dir: str,
    bbox: Optional[dict],
    rho_var: str,
    out_name: str = "temp_crop.nc",
) -> str:
    if bbox is None:
        return nc_path

    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        crs, _ = get_crs_and_transform(ds, data_var=rho_var)
        transformer = Transformer.from_crs("EPSG:4326", crs.to_string(), always_xy=True)

        x_min, y_min = transformer.transform(bbox["lon"][0], bbox["lat"][0])
        x_max, y_max = transformer.transform(bbox["lon"][1], bbox["lat"][1])

        # y may be descending in projected grids
        y_desc = bool(ds["y"][0] > ds["y"][-1])
        y0, y1 = (y_max, y_min) if y_desc else (y_min, y_max)

        ds_crop = ds.sel(x=slice(x_min, x_max), y=slice(y0, y1))

        os.makedirs(out_dir, exist_ok=True)
        temp_crop = os.path.join(out_dir, out_name)
        ds_crop.to_netcdf(temp_crop)

    return temp_crop


# -----------------------------------------------------------------------------
# Build RGB + Lab for segmentation (from reflectance)
# -----------------------------------------------------------------------------
def build_rgb_lab(
    ds: xr.Dataset,
    rho_var: str,
    rgb_wl: Tuple[float, float, float] = (665, 560, 490),
    clip: Tuple[float, float] = (1, 99),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = ds[rho_var]
    wl = rho[rho.dims[0]].values.astype(float)

    def pick_band(target: float) -> int:
        return int(np.argmin(np.abs(wl - target)))

    iR, iG, iB = map(pick_band, rgb_wl)

    R = rho.isel({rho.dims[0]: iR}).values.astype(np.float32)
    G = rho.isel({rho.dims[0]: iG}).values.astype(np.float32)
    B = rho.isel({rho.dims[0]: iB}).values.astype(np.float32)

    valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
    valid &= (R >= 0) & (G >= 0) & (B >= 0)

    def stretch(ch: np.ndarray) -> np.ndarray:
        if not np.any(valid):
            return np.zeros_like(ch, dtype=np.float32)
        lo, hi = np.nanpercentile(ch[valid], clip)
        ch2 = np.clip(ch, lo, hi)
        return ((ch2 - lo) / (hi - lo + 1e-12)).astype(np.float32)

    rgb = np.stack([stretch(R), stretch(G), stretch(B)], axis=-1)  # (y,x,3)
    lab = rgb2lab(rgb).astype(np.float32)
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

    # remap labels to 1..N (keep 0)
    u = np.unique(labels)
    u = u[u != 0]
    remap = {int(old): i + 1 for i, old in enumerate(u)}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out


# -----------------------------------------------------------------------------
# Per-segment median + MAD spectra + ancillary medians
# -----------------------------------------------------------------------------
def segment_stats(
    ds: xr.Dataset,
    labels: np.ndarray,
    rho_var: str,
    aux_vars: Dict[str, str],
):
    R = ds[rho_var].values.astype(np.float32)  # (wl,y,x)
    if R.ndim != 3:
        raise ValueError(f"{rho_var} must be 3D (wl,y,x). Got shape {R.shape}")

    nwl, ny, nx = R.shape
    X = np.moveaxis(R, 0, -1).reshape(-1, nwl)  # (n_pix, wl)
    lab = labels.reshape(-1)

    seg_ids = np.unique(lab)
    seg_ids = seg_ids[seg_ids != 0]

    med = np.full((len(seg_ids), nwl), np.nan, dtype=np.float32)
    mad = np.full((len(seg_ids), nwl), np.nan, dtype=np.float32)
    npx = np.zeros(len(seg_ids), dtype=np.int32)

    # per segment value for each aux key
    aux_med = {k: np.full(len(seg_ids), np.nan, dtype=np.float32) for k in aux_vars.keys()}

    # pre-load aux arrays: scalar float OR flattened (ny*nx,)
    aux_cache: dict[str, object] = {}
    for out_name, var in aux_vars.items():
        if var not in ds:
            aux_cache[out_name] = None
            continue
        da = ds[var]

        if da.ndim == 0:
            aux_cache[out_name] = float(da.values)
            continue

        if set(da.dims) == {"y", "x"} and da.ndim == 2:
            aux_cache[out_name] = da.values.astype(np.float32).reshape(-1)
            continue

        if da.ndim == 1 and da.dims[0] in ("y", "x"):
            if da.dims[0] == "y":
                arr2 = np.repeat(da.values.astype(np.float32)[:, None], nx, axis=1)
            else:
                arr2 = np.repeat(da.values.astype(np.float32)[None, :], ny, axis=0)
            aux_cache[out_name] = arr2.reshape(-1)
            continue

        # last resort: try broadcast to (y,x)
        try:
            template = (ds["y"] * 0 + ds["x"] * 0)  # (y,x)
            da2 = da.broadcast_like(template)
            aux_cache[out_name] = da2.values.astype(np.float32).reshape(-1)
        except Exception as e:
            raise ValueError(f"Unsupported dims for aux var '{var}': {da.dims}") from e

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

        for k in aux_vars.keys():
            src = aux_cache.get(k, None)
            if src is None:
                continue
            if isinstance(src, float):
                aux_med[k][i] = src
            else:
                aux_med[k][i] = np.nanmedian(src[idx])

    wl = ds[rho_var][ds[rho_var].dims[0]].values.astype(float)
    return seg_ids, npx, wl, med, mad, aux_med


# -----------------------------------------------------------------------------
# Write seg_id into NetCDF in-place
# -----------------------------------------------------------------------------
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
# Write polygons layer + attach aux fields directly
# PATCH: dissolve by seg_id so seg_id is unique in the layer
# -----------------------------------------------------------------------------
def write_segments_layer_with_aux(
    out_gpkg: str,
    layer: str,
    labels: np.ndarray,
    ds: xr.Dataset,
    seg_ids: np.ndarray,
    npx: np.ndarray,
    rho_var: str,
    aux_med: Dict[str, np.ndarray],
) -> gpd.GeoDataFrame:
    crs, transform = get_crs_and_transform(ds, data_var=rho_var)

    mask = labels != 0
    records = []
    for geom, val in shapes(labels.astype(np.int32), mask=mask, transform=transform):
        sid = int(val)
        if sid == 0:
            continue
        records.append({"seg_id": sid, "geometry": shp_shape(geom)})

    gdf = gpd.GeoDataFrame(records, crs=crs)

    if not gdf.empty:
        # dissolve islands -> one feature per seg_id
        gdf = gdf.dissolve(by="seg_id", as_index=False)

    sid_to_i = {int(s): i for i, s in enumerate(seg_ids.astype(int))}

    gdf["n_pix"] = [
        int(npx[sid_to_i[sid]]) if sid in sid_to_i else np.nan
        for sid in gdf["seg_id"].astype(int)
    ]

    for k, arr in aux_med.items():
        col = f"{k}_med"
        gdf[col] = [
            float(arr[sid_to_i[sid]]) if sid in sid_to_i and np.isfinite(arr[sid_to_i[sid]]) else np.nan
            for sid in gdf["seg_id"].astype(int)
        ]

    # safety
    if not gdf["seg_id"].is_unique:
        raise RuntimeError("segments layer seg_id is still not unique after dissolve")

    gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
    return gdf


def write_segments_layer(
    out_gpkg: str,
    layer: str,
    labels: np.ndarray,
    ds: xr.Dataset,
    seg_ids: np.ndarray,
    npx: np.ndarray,
    rho_var: str,
) -> gpd.GeoDataFrame:
    return write_segments_layer_with_aux(
        out_gpkg=out_gpkg,
        layer=layer,
        labels=labels,
        ds=ds,
        seg_ids=seg_ids,
        npx=npx,
        rho_var=rho_var,
        aux_med={},
    )


# -----------------------------------------------------------------------------
# Write spectra table layer (no geometry) into same GeoPackage (append)
# -----------------------------------------------------------------------------
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

    # write as non-spatial table: geopandas requires a geometry column; write None geometry
    gdf = gpd.GeoDataFrame(df, geometry=[None] * len(df), crs=None)
    gdf.to_file(out_gpkg, layer=layer, driver="GPKG", mode="a")


def write_aux_layer(
    out_gpkg: str,
    layer: str,
    seg_ids: np.ndarray,
    aux_med: Dict[str, np.ndarray],
) -> None:
    rows = []
    for i, sid in enumerate(seg_ids.astype(int)):
        row = {"seg_id": int(sid)}
        for k, arr in aux_med.items():
            row[f"{k}_med"] = float(arr[i]) if np.isfinite(arr[i]) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry=[None] * len(df), crs=None)
    gdf.to_file(out_gpkg, layer=layer, driver="GPKG", mode="a")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def segment_and_export(
    nc_path: str,
    out_dir: str,
    bbox: Optional[dict],
    rho_var: str = "rho_at_sensor",
) -> Tuple[str, str]:
    temp_nc = crop_to_temp(nc_path, out_dir=out_dir, bbox=bbox, rho_var=rho_var)

    with xr.open_dataset(temp_nc, engine="netcdf4") as ds:
        _, lab, valid, _ = build_rgb_lab(ds, rho_var=rho_var, rgb_wl=(665, 560, 490))
        labels = segment_on_lab(lab, valid, scale=400, sigma=1.0, min_size=50)

    write_seg_id_inplace(temp_nc, labels, var_name="seg_id", fill_value=0)

    with xr.open_dataset(temp_nc, engine="netcdf4") as ds:
        aux_vars = {
            "sun_zenith": "sun_zenith",
            "view_zenith": "view_zenith",
            "relative_azimuth": "relative_azimuth",
            "pressure": "surface_air_pressure",
            "water_vapor": "atmosphere_mass_content_of_water_vapor",
            "ozone": "equivalent_thickness_at_stp_of_atmosphere_ozone_content",
            "altitude": "z",
            "h_w": "bathymetry_nonna10",
        }

        seg_ids, npx, wl, rho_med, rho_mad, aux_med = segment_stats(
            ds, labels, rho_var=rho_var, aux_vars=aux_vars
        )

        out_gpkg = os.path.join(out_dir, "segments.gpkg")
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(out_gpkg):
            os.remove(out_gpkg)

        write_segments_layer_with_aux(
            out_gpkg=out_gpkg,
            layer="segments",
            labels=labels,
            ds=ds,
            seg_ids=seg_ids,
            npx=npx,
            rho_var=rho_var,
            aux_med=aux_med,
        )
        write_spectra_layer(out_gpkg, "spectra", seg_ids, wl, rho_med, rho_mad)

        # optional: keep a separate aux table too (comment out if not needed)
        # write_aux_layer(out_gpkg, "aux", seg_ids, aux_med)

    return temp_nc, out_gpkg


if __name__ == "__main__":
    nc_path = "/D/Data/WISE/ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l1rg.nc"
    out_dir = os.path.join(os.path.dirname(nc_path), "segmentation_outputs")

    bbox = {"lon": (-64.37024, -64.35578), "lat": (49.77804, 49.78492)}
    # bbox = None

    temp_nc, gpkg = segment_and_export(
        nc_path=nc_path,
        out_dir=out_dir,
        bbox=bbox,
        rho_var="rho_at_sensor_calibrated",
    )

    print("Wrote cropped NetCDF:", temp_nc)
    print("Wrote GeoPackage:", gpkg)
