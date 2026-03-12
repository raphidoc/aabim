#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, Dict, Set, List

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

from skimage.segmentation import slic
from skimage.morphology import remove_small_objects

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


# ----------------------------- CRS / transform -----------------------------
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

    height = int(ds.sizes["y"])
    width = int(ds.sizes["x"])
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    return crs, transform


# ----------------------------- Crop / wl filter -----------------------------
def crop_to_temp(
    nc_path: str,
    out_dir: str,
    bbox: Optional[dict],
    rho_var: str = "rho_w",
    out_name: str = "temp_crop.nc",
    wavelength_filter: tuple = None,
) -> str:
    if bbox is None:
        return nc_path

    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        if wavelength_filter is not None:
            lo, hi = wavelength_filter
            if lo > hi:
                lo, hi = hi, lo

            rho = ds[rho_var]
            wl_dim = rho.dims[0]
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


def _remap_consecutive(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32, copy=False)
    u = np.unique(labels)
    u = u[u != 0]
    remap = {int(old): i + 1 for i, old in enumerate(u)}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out


# ----------------------------- Spectral PCA features -----------------------------
def spectral_pca_features(
    ds: xr.Dataset,
    rho_var: str,
    valid: np.ndarray,
    n_pca: int = 6,
    use_log1p: bool = True,
) -> np.ndarray:
    """
    Returns (ny,nx,n_pca_eff) feature cube for segmentation.
    """
    rho = ds[rho_var]  # (wl,y,x)
    R = rho.values.astype(np.float32)  # (nwl,ny,nx)
    nwl, ny, nx = R.shape

    X = np.moveaxis(R, 0, -1).reshape(-1, nwl)  # (N,nwl)
    m = valid.reshape(-1)

    idx = np.where(m)[0]
    Xv = X[idx]
    ok = np.all(np.isfinite(Xv), axis=1) & np.all(Xv >= 0, axis=1)
    idx = idx[ok]
    Xv = Xv[ok]

    if Xv.shape[0] == 0:
        return np.zeros((ny, nx, 1), dtype=np.float32)

    if use_log1p:
        Xv = np.log1p(Xv)

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(10, 90))
    Xs = scaler.fit_transform(Xv)

    n_pca_eff = int(min(n_pca, Xs.shape[1], max(1, Xs.shape[0] - 1)))
    pca = PCA(n_components=n_pca_eff, random_state=0)
    Z = pca.fit_transform(Xs).astype(np.float32)  # (n_valid, n_pca_eff)

    feats = np.zeros((ny * nx, n_pca_eff), dtype=np.float32)
    feats[idx] = Z
    return feats.reshape(ny, nx, n_pca_eff)


# ----------------------------- Adjacency + merging -----------------------------
def adjacency_4(labels: np.ndarray) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}

    def add(a: int, b: int):
        if a == 0 or b == 0 or a == b:
            return
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # right neighbors
    a = labels[:, :-1]
    b = labels[:, 1:]
    diff = (a != b) & (a != 0) & (b != 0)
    for x, y in zip(a[diff].ravel(), b[diff].ravel()):
        add(int(x), int(y))

    # down neighbors
    a = labels[:-1, :]
    b = labels[1:, :]
    diff = (a != b) & (a != 0) & (b != 0)
    for x, y in zip(a[diff].ravel(), b[diff].ravel()):
        add(int(x), int(y))

    return adj


class UnionFind:
    def __init__(self, items: np.ndarray):
        items = [int(x) for x in items]
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return a
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return a


def region_stats_from_pixels(
    labels: np.ndarray,
    valid: np.ndarray,
    feats: np.ndarray,  # (ny,nx,c) features (PCs)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-label:
      ids (K,), counts (K,), mean feature vector (K,c)
    """
    lab = labels.reshape(-1)
    m = valid.reshape(-1)
    F = feats.reshape(-1, feats.shape[-1])

    idx = np.where((lab != 0) & m)[0]
    labv = lab[idx].astype(np.int32)
    Fv = F[idx].astype(np.float32)

    ids = np.unique(labv)
    K = len(ids)
    c = Fv.shape[1]
    cnt = np.zeros(K, dtype=np.int32)
    mu = np.zeros((K, c), dtype=np.float32)

    for i, sid in enumerate(ids):
        sel = (labv == sid)
        Xi = Fv[sel]
        if Xi.shape[0] == 0:
            continue
        cnt[i] = Xi.shape[0]
        mu[i] = Xi.mean(axis=0)

    return ids, cnt, mu


def merge_by_mean_feature_distance(
    labels: np.ndarray,
    valid: np.ndarray,
    feats: np.ndarray,
    dist_thresh: float = 0.55,
    max_passes: int = 6,
) -> np.ndarray:
    """
    Merge adjacent regions if ||mu_i - mu_j|| <= dist_thresh in PCA feature space.
    This is the “grow coherent regions” step you’re missing.
    """
    labels = labels.astype(np.int32, copy=True)
    labels[~valid] = 0

    for _ in range(max_passes):
        ids, cnt, mu = region_stats_from_pixels(labels, valid, feats)
        if len(ids) == 0:
            return labels

        # normalize feature means so dist_thresh is stable-ish
        mu_n = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)

        sid_to_i = {int(s): i for i, s in enumerate(ids.astype(int))}
        adj = adjacency_4(labels)
        uf = UnionFind(ids)

        merged_any = False
        # one pass over adjacency
        for a, neighs in adj.items():
            ra = uf.find(a)
            ia = sid_to_i.get(ra, None)
            if ia is None:
                continue
            for b in neighs:
                rb = uf.find(b)
                if ra == rb:
                    continue
                ib = sid_to_i.get(rb, None)
                if ib is None:
                    continue
                d = float(np.linalg.norm(mu_n[ia] - mu_n[ib]))
                if d <= dist_thresh:
                    uf.union(ra, rb)
                    merged_any = True

        if not merged_any:
            break

        # apply unions to raster
        flat = labels.reshape(-1)
        nz = flat != 0
        flat[nz] = np.array([uf.find(v) for v in flat[nz]], dtype=np.int32)
        labels = labels.reshape(labels.shape)

        labels = _remap_consecutive(labels)

    return labels


# ----------------------------- Segmentation: SLIC on PCs + merge -----------------------------
def segment_on_slic_pcs_then_merge(
    ds: xr.Dataset,
    rho_var: str,
    valid: np.ndarray,
    n_pca: int = 6,
    n_segments: int = 1600,
    compactness: float = 8.0,
    sigma: float = 0.5,
    min_size: int = 80,
    merge_dist: float = 0.55,
    merge_passes: int = 6,
) -> np.ndarray:
    feats = spectral_pca_features(ds, rho_var=rho_var, valid=valid, n_pca=n_pca, use_log1p=True)

    labels0 = slic(
        feats,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=1,
        channel_axis=-1,
        mask=valid,
    ).astype(np.int32)
    labels0[~valid] = 0

    # if min_size and min_size > 1:
    #     labels0 = remove_small_objects(labels0, min_size=int(min_size), connectivity=1).astype(np.int32)
    #     labels0[~valid] = 0
    #     labels0 = _remap_consecutive(labels0)

    labels = merge_by_mean_feature_distance(
        labels0,
        valid=valid,
        feats=feats,
        dist_thresh=float(merge_dist),
        max_passes=int(merge_passes),
    )
    labels[~valid] = 0
    return _remap_consecutive(labels)


# ----------------------------- Stats / outputs (unchanged) -----------------------------
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
    gdf["theta_sun"] = [_get(theta_sun, sid) for sid in gdf["seg_id"].astype(int)]
    gdf["theta_view"] = [_get(theta_view, sid) for sid in gdf["seg_id"].astype(int)]
    gdf["h_w_mu"] = [_get(h_w_mu, sid) for sid in gdf["seg_id"].astype(int)]
    gdf["h_w_sd"] = [_get(h_w_sd, sid) for sid in gdf["seg_id"].astype(int)]

    gdf["seg_id"] = gdf["seg_id"].astype(int)
    gdf["n_pix"] = pd.to_numeric(gdf["n_pix"], errors="coerce")

    agg = {
        "n_pix": "sum",
        "theta_sun": "first",
        "theta_view": "first",
        "h_w_mu": "first",
        "h_w_sd": "first",
    }

    gdf = gdf.dissolve(by="seg_id", as_index=False, aggfunc=agg)
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


def auto_n_segments(valid: np.ndarray, target_area_px: int = 25, clamp: Tuple[int, int] = (200, 8000)) -> int:
    """
    target_area_px: desired superpixel size in pixels (smaller -> more segments)
    clamp: (min_segments, max_segments) safety bounds
    """
    n_valid = int(np.sum(valid))
    if n_valid <= 0:
        return clamp[0]
    n = int(np.ceil(n_valid / max(1, int(target_area_px))))
    return int(np.clip(n, clamp[0], clamp[1]))

def segment_and_export(
    nc_path: str,
    out_dir: str,
    bbox: Optional[dict],
    rho_var: str = "rho_w",
    wavelength_filter: Optional[Tuple[float, float]] = (370, 730),
    rgb_wl: Tuple[float, float, float] = (665, 560, 490),  # kept for signature compatibility
) -> Tuple[str, str]:
    temp_nc = crop_to_temp(nc_path, out_dir=out_dir, bbox=bbox, rho_var=rho_var, wavelength_filter=wavelength_filter)

    with xr.open_dataset(temp_nc, engine="netcdf4") as ds:
        depth = ds["bathymetry_nonna10"].values.astype(np.float32)
        theta_view = ds["view_zenith"].values.astype(np.float32)

        # base valid: require depth/view + finite spectra (we’ll check full spectrum inside PCA build too)
        rho = ds[rho_var]
        R0 = rho.isel({rho.dims[0]: 0}).values.astype(np.float32)
        valid = np.isfinite(R0)
        valid &= np.isfinite(depth) & (depth <= 20) & np.isfinite(theta_view)

        # --- SLIC on spectral PCs + merging
        n_segments = auto_n_segments(valid, target_area_px=25, clamp=(300, 12000))

        labels = segment_on_slic_pcs_then_merge(
            ds,
            rho_var=rho_var,
            valid=valid,
            n_pca=6,
            n_segments=n_segments,
            compactness=8.0,
            sigma=0.5,
            min_size=60,
            merge_dist=0.35,
            merge_passes=3,
        )
        print("auto n_segments:", n_segments)

        print("valid px:", int(np.sum(valid)))
        print("labels nonzero px:", int(np.sum(labels != 0)))
        u, c = np.unique(labels, return_counts=True)
        print("n labels (incl 0):", len(u), "unique:", u[:10], "...")
        print("largest counts:", np.sort(c)[-10:])
        print("max label:", int(labels.max()))

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
            out_gpkg,
            "segments",
            labels,
            ds,
            seg_ids,
            npx,
            theta_sun,
            theta_view,
            h_w_mu,
            h_w_sd,
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