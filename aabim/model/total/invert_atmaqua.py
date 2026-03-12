#!/usr/bin/env python3
from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr
import netCDF4
from pyproj import Transformer
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from reverie import ReveCube
from reverie.correction.aquatic.stan_data import make_stan_data_base_from_csv


# =============================================================================
# Logging / silence
# =============================================================================
def quiet_logging() -> None:
    logging.disable(logging.ERROR)
    os.environ["CMDSTANPY_SILENT"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"


def silence_everything() -> None:
    logging.disable(logging.ERROR)
    for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
    os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"


# =============================================================================
# Worker globals
# =============================================================================
_STAN_MODEL = None
_BASE = None
_RB_MU = None
_RB_SD = None
_ATM_LUT_PATH = None


# =============================================================================
# Config
# =============================================================================
@dataclass(frozen=True)
class StanCfg:
    method: str = "optimize"  # "optimize" or "sample"
    seed: int = 1234
    chains: int = 2
    iter_warmup: int = 300
    iter_sampling: int = 300


@dataclass(frozen=True)
class PathsCfg:
    stan_model_path: str
    user_header: Optional[str]
    a_w_csv: str
    a0_a1_phyto_csv: str
    r_b_gamache_csv: str
    atm_lut_nc: str



@dataclass(frozen=True)
class VarsCfg:
    rho_var: str = "rho_at_sensor"
    sun_var: str = "sun_zenith"
    view_var: str = "view_zenith"
    raa_var: str = "relative_azimuth"
    pressure_var: str = "pressure"
    altitude_var: str = "altitude"
    depth_var: str = "bathymetry_nonna10"
    seg_id_var: str = "seg_id"


# =============================================================================
# Worker init: compile Stan, load base constants + rb priors
# =============================================================================
def _collapse_rb_prior_from_lib(base: Dict[str, object], sd_floor: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse class-conditional (mu,sd) to a single (mu,sd) prior for r_b(λ).
    Uses equal weights over base["bottom_class_ids"].
    """
    mu_lib = np.asarray(base["r_b_mu_lib"], dtype=np.float64)  # (n_wl, n_class)
    sd_lib = np.asarray(base["r_b_sd_lib"], dtype=np.float64)  # (n_wl, n_class)
    ids = np.asarray(base["bottom_class_ids"], dtype=int)      # 1-based ids of the 3 chosen classes
    cols = ids - 1                                             # to 0-based column indices

    mu = mu_lib[:, cols]                                       # (n_wl, K)
    sd = sd_lib[:, cols]                                       # (n_wl, K)

    # mixture collapsed moments
    mu_mix = np.mean(mu, axis=1)                               # (n_wl,)
    ex2 = np.mean(sd**2 + mu**2, axis=1)
    var_mix = np.maximum(ex2 - mu_mix**2, sd_floor**2)
    sd_mix = np.sqrt(var_mix)

    # keep physically safe (Stan later logit() needs (0,1))
    mu_mix = np.clip(mu_mix, 1e-6, 1.0 - 1e-6)
    sd_mix = np.maximum(sd_mix, sd_floor)

    return mu_mix, sd_mix


def _init_worker(
    paths: PathsCfg,
    wavelength: np.ndarray,
    water_type: int,
    shallow: int,
    bottom_class_names: Tuple[str, str, str],
    # keep your old fixed params if your base builder still requires them
    a_nap_star: float,
    bb_p_star: float,
    a_g_s: float,
    a_nap_s: float,
    bb_p_gamma: float,
):
    global _STAN_MODEL, _BASE, _RB_MU, _RB_SD, _ATM_LUT_PATH
    silence_everything()

    from cmdstanpy import CmdStanModel
    _STAN_MODEL = CmdStanModel(stan_file=paths.stan_model_path, user_header=paths.user_header)

    _BASE = make_stan_data_base_from_csv(
        wavelength=wavelength,
        water_type=water_type,
        shallow=shallow,
        bottom_class_names=bottom_class_names,
        a_nap_star=a_nap_star,
        bb_p_star=bb_p_star,
        a_g_s=a_g_s,
        a_nap_s=a_nap_s,
        bb_p_gamma=bb_p_gamma,
        a_w_csv=paths.a_w_csv,
        a0_a1_phyto_csv=paths.a0_a1_phyto_csv,
        r_b_gamache_csv=paths.r_b_gamache_csv,
    )

    _RB_MU, _RB_SD = _collapse_rb_prior_from_lib(_BASE, sd_floor=1e-6)
    _ATM_LUT_PATH = paths.atm_lut_nc



# =============================================================================
# Atmospheric LUT slicing (nearest on fixed dims, keep AOD axis)
# -----------------------------------------------------------------------------
# Expected LUT layout (xarray):
#   dims: aod550, wavelength, sol_zen, view_zen, relative_azimuth, pressure, altitude
#   vars: rho_path_ra, t_ra, s_ra
#
# If your dim/var names differ, edit here only.
# =============================================================================
def slice_atm_lut_for_segment(
    sun: float,
    view: float,
    raa: float,
    pressure: float,
    altitude: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _ATM_LUT_PATH

    with xr.open_dataset(_ATM_LUT_PATH) as lut:
        # adjust these names if needed:
        sel = lut.sel(
            sol_zen=sun,
            view_zen=view,
            relative_azimuth=raa,
            target_pressure=pressure,
            sensor_alt=altitude,
            method="nearest",
        )

        aod_grid = sel["aod550"].values.astype(np.float64)

        # assume vars are shaped (aod550, wavelength)
        rho_path_ra = sel["rho_path_ra"].values.astype(np.float64)
        t_ra = sel["t_ra"].values.astype(np.float64)
        s_ra = sel["s_ra"].values.astype(np.float64)

    return aod_grid, rho_path_ra, t_ra, s_ra


# =============================================================================
# Invert a single segment
# =============================================================================
def invert_segment(
    seg_id: int,
    rho_obs: np.ndarray,       # (Nwl,)
    rho_mad: np.ndarray,       # (Nwl,)
    sun: float,
    view: float,
    raa: float,
    pressure: float,
    altitude: float,
    hw_med: float,
    stan_cfg: StanCfg,
    out_vars: Tuple[str, ...],
) -> Tuple[int, Optional[Dict[str, float]]]:
    global _STAN_MODEL, _BASE, _RB_MU, _RB_SD

    if not np.all(np.isfinite(rho_obs)):
        return seg_id, None

    # robust -> gaussian equiv
    rho_sigma = 1.4826 * np.asarray(rho_mad, dtype=np.float64)
    rho_sigma = np.clip(np.nan_to_num(rho_sigma, nan=np.nanmedian(rho_sigma)), 1e-6, 0.2)

    if not (np.isfinite(sun) and np.isfinite(view) and np.isfinite(raa) and np.isfinite(pressure) and np.isfinite(altitude)):
        return seg_id, None

    # slice atmospheric LUT (nearest on fixed dims), keep AOD axis for Stan
    aod_grid, rho_path_ra, t_ra, s_ra = slice_atm_lut_for_segment(
        sun=float(sun),
        view=float(view),
        raa=float(raa),
        pressure=float(pressure),
        altitude=float(altitude),
    )

    stan_data = dict(_BASE)

    # canonical keys for the new atmaqua model
    stan_data.update({
        "Nwl": int(len(rho_obs)),
        "wl_nm": np.asarray(stan_data.get("wavelength", stan_data.get("wl", None)), dtype=np.float64),

        "rho_obs": np.asarray(rho_obs, dtype=np.float64),
        "rho_sigma": np.asarray(rho_sigma, dtype=np.float64),

        "theta_sun_deg": float(sun),
        "theta_view_deg": float(view),

        "Naod": int(len(aod_grid)),
        "aod_grid": np.asarray(aod_grid, dtype=np.float64),
        "rho_path_ra": np.asarray(rho_path_ra, dtype=np.float64),
        "t_ra": np.asarray(t_ra, dtype=np.float64),
        "s_ra": np.asarray(s_ra, dtype=np.float64),

        "rb_mu": np.clip(_RB_MU, 1e-6, 1 - 1e-6),
        "rb_sd": np.clip(_RB_SD, 1e-6, 1.0),

        "hw_prior_mu": float(max(hw_med, 0.0)),
        "hw_prior_sd": float(2.0),
    })

    try:
        if stan_cfg.method == "optimize":
            fit = _STAN_MODEL.optimize(
                data=stan_data,
                seed=stan_cfg.seed,
            )
            est = fit.optimized_params_dict
            out = {k: float(est[k]) for k in out_vars if k in est}
        else:
            fit = _STAN_MODEL.sample(
                data=stan_data,
                seed=stan_cfg.seed,
                chains=stan_cfg.chains,
                parallel_chains=1,
                iter_warmup=stan_cfg.iter_warmup,
                iter_sampling=stan_cfg.iter_sampling,
                show_progress=False,
                show_console=False,
            )
            draws = fit.draws_pd()
            out = {k: float(draws[k].mean()) for k in out_vars if k in draws.columns}

        return seg_id, out

    except Exception:
        return seg_id, None


# =============================================================================
# Segment table IO (from seg_id raster + per-seg med/mad)
# -----------------------------------------------------------------------------
# Assumes seg_id is already in the NetCDF and you want to invert per segment
# by aggregating rho_at_sensor and aux vars.
# =============================================================================
def compute_segment_stats_from_nc(
    nc_path: str,
    vars_cfg: VarsCfg,
    wavelength_filter: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, float]], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Returns:
      wl (Nwl,), seg_ids (Ns,),
      aux_by_seg[seg_id] -> dict of aux medians,
      rho_med_by_seg[seg_id] -> (Nwl,),
      rho_mad_by_seg[seg_id] -> (Nwl,)
    """
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        rho = ds[vars_cfg.rho_var]  # (wavelength, y, x)
        wl = rho[rho.dims[0]].values.astype(np.float64)

        if wavelength_filter is not None:
            lo, hi = float(wavelength_filter[0]), float(wavelength_filter[1])
            keep = (wl >= lo) & (wl <= hi)
            rho = rho.isel({rho.dims[0]: np.where(keep)[0]})
            wl = wl[keep]

        seg = ds[vars_cfg.seg_id_var].values.astype(np.int32)  # (y,x)

        # flatten
        R = rho.values.astype(np.float64)         # (Nwl, y, x)
        X = np.moveaxis(R, 0, -1).reshape(-1, R.shape[0])  # (Npix, Nwl)
        lab = seg.reshape(-1)

        # aux flatten
        sun = ds[vars_cfg.sun_var].values.reshape(-1).astype(np.float64)
        view = ds[vars_cfg.view_var].values.reshape(-1).astype(np.float64)
        raa = ds[vars_cfg.raa_var].values.reshape(-1).astype(np.float64)
        pres = ds[vars_cfg.pressure_var].values.reshape(-1).astype(np.float64)
        alt = ds[vars_cfg.altitude_var].values.reshape(-1).astype(np.float64)
        hw = ds[vars_cfg.depth_var].values.reshape(-1).astype(np.float64)

        seg_ids = np.unique(lab)
        seg_ids = seg_ids[seg_ids != 0]

        aux_by_seg: Dict[int, Dict[str, float]] = {}
        rho_med_by_seg: Dict[int, np.ndarray] = {}
        rho_mad_by_seg: Dict[int, np.ndarray] = {}

        for sid in seg_ids.astype(int):
            idx = np.where(lab == sid)[0]
            Xi = X[idx]
            ok = np.all(np.isfinite(Xi), axis=1)
            idx = idx[ok]
            Xi = Xi[ok]
            if Xi.shape[0] == 0:
                continue

            m = np.median(Xi, axis=0)
            mad = np.median(np.abs(Xi - m[None, :]), axis=0)

            rho_med_by_seg[sid] = m.astype(np.float64)
            rho_mad_by_seg[sid] = mad.astype(np.float64)

            aux_by_seg[sid] = {
                "sun": float(np.nanmedian(sun[idx])),
                "view": float(np.nanmedian(view[idx])),
                "raa": float(np.nanmedian(raa[idx])),
                "pressure": float(np.nanmedian(pres[idx])),
                "altitude": float(np.nanmedian(alt[idx])),
                "h_w": float(np.nanmedian(hw[idx])),
            }

    return wl, seg_ids.astype(int), aux_by_seg, rho_med_by_seg, rho_mad_by_seg


# =============================================================================
# Main driver: invert per segment + write rasters back to new NetCDF
# =============================================================================
def run_atmaqua_inversion(
    l2_path: str,
    out_path: str,
    paths: PathsCfg,
    vars_cfg: VarsCfg = VarsCfg(),
    wavelength_filter: Optional[Tuple[float, float]] = (400.0, 700.0),
    bbox: Optional[dict] = None,
    n_workers: Optional[int] = None,
    stan_cfg: StanCfg = StanCfg(method="optimize"),
    out_vars: Tuple[str, ...] = ("aod550", "chl", "a_g_440", "spm", "bb_p_gamma", "a_g_slope", "h_w"),
) -> str:
    quiet_logging()

    # Open with ReveCube to preserve your tooling / CRS handling
    l2w = ReveCube.from_reve_nc(l2_path)

    # optional crop (in dataset CRS)
    temp_crop = None
    work_path = l2_path
    if bbox is not None:
        transformer = Transformer.from_crs("EPSG:4326", l2w.CRS, always_xy=True)
        x_min, y_min = transformer.transform(bbox["lon"][0], bbox["lat"][0])
        x_max, y_max = transformer.transform(bbox["lon"][1], bbox["lat"][1])

        l2w.in_ds = l2w.in_ds.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

        tmp_dir = os.path.join(os.path.dirname(l2_path), "atmaqua_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        temp_crop = os.path.join(tmp_dir, "temp_crop.nc")
        l2w.in_ds.to_netcdf(temp_crop)
        l2w.update_attributes()
        work_path = temp_crop

    # Make sure seg_id exists
    if vars_cfg.seg_id_var not in l2w.in_ds:
        raise ValueError(f"Missing '{vars_cfg.seg_id_var}' in dataset. Run segmentation first.")

    # Segment stats
    wl, seg_ids, aux_by_seg, rho_med_by_seg, rho_mad_by_seg = compute_segment_stats_from_nc(
        work_path, vars_cfg=vars_cfg, wavelength_filter=wavelength_filter
    )

    # Create output dataset: copy coordinates + add per-pixel rasters using seg_id mapping
    with xr.open_dataset(work_path, engine="netcdf4") as ds_in:
        ds_out = xr.Dataset(coords={"y": ds_in["y"].values, "x": ds_in["x"].values})
        seg_map = ds_in[vars_cfg.seg_id_var].values.astype(np.int32)  # (y,x)

        # allocate rasters
        out_rasters = {v: np.full(seg_map.shape, np.nan, dtype=np.float32) for v in out_vars}

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    bottom_class_names = ("coraline_crustose_algae", "saccharina_latissima", "sediment")

    # worker init
    ctx = mp.get_context("spawn")
    initargs = (
        paths,
        wl.astype(float),
        2,  # water_type
        1,  # shallow
        bottom_class_names,
        0.0051, 0.0047, 0.017, 0.006, 0.65,
    )

    # dispatch segment jobs
    futures = []
    results: Dict[int, Dict[str, float]] = {}

    t0 = time.time()
    logging.info(f"Inverting {len(seg_ids)} segments with {n_workers} workers; method={stan_cfg.method}")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker, initargs=initargs) as ex:
        for sid in seg_ids:
            aux = aux_by_seg.get(int(sid))
            if aux is None:
                continue
            fut = ex.submit(
                invert_segment,
                int(sid),
                rho_med_by_seg[int(sid)],
                rho_mad_by_seg[int(sid)],
                aux["sun"],
                aux["view"],
                aux["raa"],
                aux["pressure"],
                aux["altitude"],
                aux["h_w"],
                stan_cfg,
                out_vars,
            )
            futures.append(fut)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Stan segments"):
            sid, est = fut.result()
            if est is not None:
                results[int(sid)] = est

    logging.info(f"Done. Successful segments: {len(results)}/{len(seg_ids)} in {time.time()-t0:.1f}s")

    # map segment estimates back to rasters
    for sid, est in results.items():
        mask = (seg_map == sid)
        for v in out_vars:
            if v in est:
                out_rasters[v][mask] = np.float32(est[v])

    # write output netcdf
    for v in out_vars:
        ds_out[v] = xr.DataArray(out_rasters[v], dims=("y", "x"), coords={"y": ds_out["y"], "x": ds_out["x"]})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds_out.to_netcdf(out_path)

    # cleanup
    ds_out.close()
    if temp_crop is not None:
        try:
            os.remove(temp_crop)
        except Exception:
            pass

    return out_path


# =============================================================================
# CLI-ish entrypoint
# =============================================================================
if __name__ == "__main__":
    quiet_logging()

    image_dir = "/D/Data/WISE/"
    # image = "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l1r.nc"
    image = "ACI-12A/segmentation_outputs/temp_crop.nc"
    l2_path = os.path.join(image_dir, image)

    bbox = {"lon": (-64.36871, -64.3615), "lat": (49.80857, 49.81336)}

    paths = PathsCfg(
        stan_model_path="/home/raphael/R/SABER/inst/stan/model_atmaqua_rb.stan",
        user_header="/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp",
        a_w_csv="/home/raphael/R/SABER/inst/extdata/a_w.csv",
        a0_a1_phyto_csv="/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv",
        r_b_gamache_csv="/home/raphael/R/SABER/inst/extdata/r_b_gamache.csv",
        atm_lut_nc="/D/Data/WISE/LUT/atm_lut_sliced.nc",
    )

    vars_cfg = VarsCfg(
        rho_var="rho_at_sensor",
        sun_var="sun_zenith",
        view_var="view_zenith",
        raa_var="relative_azimuth",
        pressure_var="pressure",
        altitude_var="altitude",
        depth_var="bathymetry_nonna10",
        seg_id_var="seg_id",
    )

    out_path = re.sub(r"l1r\.nc$", "l2_atmaqua_inverted.nc", l2_path)

    run_atmaqua_inversion(
        l2_path=l2_path,
        out_path=out_path,
        paths=paths,
        vars_cfg=vars_cfg,
        wavelength_filter=(400, 700),
        bbox=bbox,
        n_workers=None,
        stan_cfg=StanCfg(method="optimize"),
        out_vars=("aod550", "chl", "a_g_440", "spm", "bb_p_gamma", "a_g_slope", "h_w"),
    )

    print("Wrote:", out_path)
