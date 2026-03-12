#!/usr/bin/env python3
from __future__ import annotations

import os
import logging
import sqlite3
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from reverie.correction.aquatic.stan_data import make_stan_data_base_from_csv
from reverie.correction.atmospheric import lut

# =============================================================================
# Silence
# =============================================================================
def silence_everything() -> None:
    logging.disable(logging.ERROR)
    for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
    os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"
    os.environ["CMDSTANPY_SILENT"] = "true"


# =============================================================================
# USER: rename these to match your LUT file(s)
# =============================================================================
# AER LUT dims
AER_SOL = "sol_zen"
AER_VIEW = "view_zen"
AER_RAA = "relative_azimuth"
AER_AOD = "aot550"  # or "aod550"
AER_PRESS = "target_pressure"
AER_ALT = "sensor_altitude"

# AER LUT vars (at sensor)
AER_RHO_PATH = "atmospheric_reflectance_at_sensor"
AER_T = "total_scattering_trans_total"
AER_S = "spherical_albedo_total"
AER_G = "sky_glint_total"

# GAS LUT dims
GAS_SOL = "sol_zen"
GAS_VIEW = "view_zen"
GAS_RAA = "relative_azimuth"
GAS_WV = "water"
GAS_O3 = "ozone"
GAS_PRESS = "target_pressure"
GAS_ALT = "sensor_altitude"

# GAS LUT var
GAS_TGAS = "global_gas_trans_total"


# =============================================================================
# USER: Stan data keys (change to match your atmaqua.stan)
# =============================================================================
K_NWL = "n_wl"
K_WL = "wavelength"

K_RHO_OBS = "rho_obs"          # at-sensor reflectance (segment median)
K_SIGMA_RHO = "rho_sigma"      # segment uncertainty (MAD->sigma)

K_THETA_SUN = "theta_sun_deg"
K_THETA_VIEW = "theta_view_deg"

K_Naod = "Naod"
K_AOD_GRID = "aod_grid"
K_RHO_PATH_AER = "rho_path_ra"
K_T_AER = "t_ra"
K_S_AER = "s_ra"
K_G_AER = "sky_glint_ra"

K_TGAS = "t_gas"

K_RB_MU = "rb_mu"
K_RB_SD = "rb_sd"


# =============================================================================
# CmdStan globals
# =============================================================================
_STAN_MODEL = None
_BASE: Dict[str, Any] | None = None

_AER_LUT_PATH: str | None = None
_GAS_LUT_PATH: str | None = None

_RB_MU: np.ndarray | None = None
_RB_SD: np.ndarray | None = None


# =============================================================================
# Compile once in parent
# =============================================================================
def compile_model(stan_model_path: str, user_header: str | None = None) -> str:
    from cmdstanpy import CmdStanModel
    m = CmdStanModel(stan_file=stan_model_path, user_header=user_header)
    return m.exe_file


# =============================================================================
# GeoPackage helpers
# =============================================================================
def sql_safe(name: str) -> str:
    return (
        name.replace("[", "_")
        .replace("]", "")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "_")
        .replace(")", "_")
    )


def read_table(gpkg_path: str, table: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    conn = sqlite3.connect(gpkg_path)
    try:
        if columns is None:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        else:
            cols = ", ".join(f'"{c}"' for c in columns)
            df = pd.read_sql_query(f'SELECT {cols} FROM "{table}"', conn)
    finally:
        conn.close()
    return df


def write_table_sqlite(gpkg_path: str, table: str, df: pd.DataFrame, overwrite: bool = True) -> None:
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    try:
        if overwrite:
            cur.execute(f'DROP TABLE IF EXISTS "{table}"')

        cols_def = []
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.integer):
                t = "INTEGER"
            elif np.issubdtype(df[c].dtype, np.floating):
                t = "REAL"
            else:
                t = "TEXT"
            cols_def.append(f'"{c}" {t}')

        cur.execute(f'CREATE TABLE "{table}" ({", ".join(cols_def)})')

        placeholders = ",".join(["?"] * len(df.columns))
        cols = ", ".join(f'"{c}"' for c in df.columns)
        ins = f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'
        cur.executemany(ins, df.itertuples(index=False, name=None))

        conn.commit()
    finally:
        conn.close()

def write_rb_spectrum_table(
    out_gpkg: str,
    table: str,
    results: List[Dict[str, Any]],
) -> None:
    rows = []
    for r in results:
        if not r.get("ok"):
            continue
        rb = r.get("r_b_hat", None)
        wl = r.get("wl_nm", None)
        if rb is None or wl is None:
            continue
        if len(rb) != len(wl):
            continue
        sid = int(r["seg_id"])
        for w, v in zip(wl, rb):
            rows.append({"seg_id": sid, "wl_nm": float(w), "r_b_hat": float(v)})

    if not rows:
        return

    df = pd.DataFrame(rows)
    write_table_sqlite(out_gpkg, table, df, overwrite=True)


# =============================================================================
# rb prior: collapse mu/sd library into a single prior per wavelength
# =============================================================================
def collapse_rb_prior_from_lib(base: Dict[str, Any], sd_floor: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mu_lib = np.asarray(base["r_b_mu_lib"], dtype=np.float64)  # (n_wl, n_class)
    sd_lib = np.asarray(base["r_b_sd_lib"], dtype=np.float64)  # (n_wl, n_class)
    ids = np.asarray(base["bottom_class_ids"], dtype=int)      # 1-based length=3
    cols = ids - 1

    mu = mu_lib[:, cols]  # (n_wl, 3)
    sd = sd_lib[:, cols]

    mu_mix = np.mean(mu, axis=1)
    ex2 = np.mean(sd**2 + mu**2, axis=1)
    var_mix = np.maximum(ex2 - mu_mix**2, sd_floor**2)
    sd_mix = np.sqrt(var_mix)

    mu_mix = np.clip(mu_mix, 1e-6, 1.0 - 1e-6)
    sd_mix = np.maximum(sd_mix, sd_floor)
    return mu_mix, sd_mix

def _extract_vector_from_optimize(est: Dict[str, Any], base: str, n: int) -> Optional[np.ndarray]:
    """
    CmdStan optimize returns keys like 'rb_logit.1', 'rb_logit.2', ...
    Return vector length n if all present, else None.
    """
    vals = []
    for i in range(1, n + 1):
        k = f"{base}[{i}]"
        if k not in est:
            return None
        vals.append(float(est[k]))
    return np.asarray(vals, dtype=np.float64)


def _extract_vector_from_draws(draws: pd.DataFrame, base: str, n: int) -> Optional[np.ndarray]:
    """
    CmdStan sample draws_df has columns like 'rb_logit[1]' ...
    Return posterior mean vector length n if all present, else None.
    """
    cols = [f"{base}[{i}]" for i in range(1, n + 1)]
    if not all(c in draws.columns for c in cols):
        return None
    return draws[cols].mean(axis=0).to_numpy(dtype=np.float64)

# =============================================================================
# LUT slicing
# =============================================================================
def slice_aer_lut_per_segment(
    sun: float, view: float, raa: float, press: float, alt: float, wavelength
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      aod_grid: (Naod,)
      rho_path: (Naod, Nwl)
      t_ra:     (Naod, Nwl)
      s_ra:     (Naod, Nwl)
      sky_g:    (Naod, Nwl)
    """
    assert _AER_LUT_PATH is not None
    with xr.open_dataset(_AER_LUT_PATH) as ds:
        ds = lut.slice_lut_to_wavelengths(ds, wavelength)

        sel = ds.sel(
            {
                AER_SOL: sun,
                AER_VIEW: view,
                AER_RAA: raa,
                AER_PRESS: press,
                AER_ALT: alt,
            },
            method="nearest",
        )

        aod_grid = np.asarray(sel[AER_AOD].values, dtype=np.float64)

        rho = np.asarray(sel[AER_RHO_PATH].values, dtype=np.float64)
        t = np.asarray(sel[AER_T].values, dtype=np.float64)
        s = np.asarray(sel[AER_S].values, dtype=np.float64)
        g = np.asarray(sel[AER_G].values, dtype=np.float64)

    # Ensure (Naod, Nwl)
    if rho.shape[0] != aod_grid.size and rho.shape[1] == aod_grid.size:
        rho = rho.T
        t = t.T
        s = s.T
        g = g.T

    return aod_grid, rho, t, s, g


def slice_gas_lut_per_segment(
    sun: float, view: float, raa: float,
    water: float, ozone: float, press: float, alt: float, wavelength
) -> np.ndarray:
    """
    Returns:
      t_gas: (Nwl,)
    """
    assert _GAS_LUT_PATH is not None
    with xr.open_dataset(_GAS_LUT_PATH) as ds:
        ds = lut.slice_lut_to_wavelengths(ds, wavelength)
        sel = ds.sel(
            {
                GAS_SOL: sun,
                GAS_VIEW: view,
                GAS_RAA: raa,
                GAS_WV: water,
                GAS_O3: ozone,
                GAS_PRESS: press,
                GAS_ALT: alt,
            },
            method="nearest",
        )
        tgas = np.asarray(sel[GAS_TGAS].values, dtype=np.float64)

    return tgas


# =============================================================================
# Worker init
# =============================================================================
def _init_worker(
    exe_file: str,
    wavelength: np.ndarray,
    water_type: int,
    shallow: int,
    bottom_class_names: Tuple[str, str, str],
    a_nap_star: float,
    bb_p_star: float,
    a_g_s: float,
    a_nap_s: float,
    bb_p_gamma: float,
    a_w_csv: str,
    a0_a1_phyto_csv: str,
    r_b_gamache_csv: str,
    aer_lut_path: str,
    gas_lut_path: str,
) -> None:
    global _STAN_MODEL, _BASE, _AER_LUT_PATH, _GAS_LUT_PATH, _RB_MU, _RB_SD
    silence_everything()
    from cmdstanpy import CmdStanModel

    _STAN_MODEL = CmdStanModel(exe_file=exe_file)
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
        a_w_csv=a_w_csv,
        a0_a1_phyto_csv=a0_a1_phyto_csv,
        r_b_gamache_csv=r_b_gamache_csv,
    )

    _RB_MU, _RB_SD = collapse_rb_prior_from_lib(_BASE, sd_floor=1e-6)
    _AER_LUT_PATH = aer_lut_path
    _GAS_LUT_PATH = gas_lut_path


# =============================================================================
# One segment inversion (ATMAQUA)
# =============================================================================
def invert_one_segment(job: Dict[str, Any], stan_cfg: Dict[str, Any], out_vars: Tuple[str, ...]) -> Dict[str, Any]:
    global _STAN_MODEL, _BASE, _RB_MU, _RB_SD
    assert _STAN_MODEL is not None and _BASE is not None
    assert _RB_MU is not None and _RB_SD is not None

    seg_id = int(job["seg_id"])
    wl = np.asarray(job["wl_nm"], dtype=np.float64)
    rho_obs = np.asarray(job["rho_med"], dtype=np.float64)
    rho_mad = np.asarray(job["rho_mad"], dtype=np.float64)

    if not np.all(np.isfinite(rho_obs)):
        return {"seg_id": seg_id, "ok": False, "err": "non-finite rho_obs"}

    rho_sigma = 1.4826 * rho_mad
    rho_sigma = np.clip(np.nan_to_num(rho_sigma, nan=np.nanmedian(rho_sigma)), 1e-6, 0.2)

    theta_sun = float(job["theta_sun"])
    theta_view = float(job["theta_view"])
    raa = float(job["raa"])
    press = float(job["pressure"])
    alt = float(job["altitude"])

    water = float(job["water_vapor"])
    ozone = float(job["ozone"])

    try:
        aod_grid, rho_path, t_ra, s_ra, sky_g = slice_aer_lut_per_segment(
            sun=theta_sun, view=theta_view, raa=raa, press=press, alt=alt, wavelength =wl
        )
        t_gas = slice_gas_lut_per_segment(
            sun=theta_sun, view=theta_view, raa=raa,
            water=water, ozone=ozone, press=press, alt=alt, wavelength = wl
        )
    except Exception as e:
        return {"seg_id": seg_id, "ok": False, "err": f"lut_slice: {repr(e)}"}

    nwl = wl.size
    if t_gas.shape[0] != nwl:
        return {"seg_id": seg_id, "ok": False, "err": f"t_gas shape {t_gas.shape} != ({nwl},)"}
    if rho_path.shape[1] != nwl:
        return {"seg_id": seg_id, "ok": False, "err": f"aer LUT nwl mismatch: {rho_path.shape} vs nwl={nwl}"}

    stan_data = dict(_BASE)

    base_wl = np.asarray(stan_data.get("wavelength", []), dtype=np.float64)
    if base_wl.size != nwl or np.max(np.abs(base_wl - wl)) > 1e-6:
        return {"seg_id": seg_id, "ok": False, "err": "wl mismatch vs base (export spectra on common wl grid)"}

    Naod = int(len(aod_grid))

    stan_data.update({
        K_NWL: int(nwl),
        K_WL: wl.tolist(),
        K_RHO_OBS: rho_obs.tolist(),
        K_SIGMA_RHO: rho_sigma.tolist(),

        K_THETA_SUN: theta_sun,
        K_THETA_VIEW: theta_view,

        K_Naod: Naod,
        K_AOD_GRID: aod_grid.tolist(),
        K_RHO_PATH_AER: rho_path.tolist(),
        K_T_AER: t_ra.tolist(),
        K_S_AER: s_ra.tolist(),
        K_G_AER: sky_g.tolist(),

        K_TGAS: t_gas.tolist(),

        K_RB_MU: _RB_MU.tolist(),
        K_RB_SD: _RB_SD.tolist(),
    })

    # only if your Stan uses it
    stan_data["h_w_obs"] = float(job["h_w"])

    pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")

    try:
        r_b_hat = None

        if stan_cfg["method"] == "optimize":
            fit = _STAN_MODEL.optimize(
                data=stan_data,
                seed=int(stan_cfg["seed"]),
                output_dir=pid_dir,
                show_console=False,
            )
            est = fit.optimized_params_dict

            scalars = {v: float(est[v]) for v in out_vars if v in est}

            rb_logit = _extract_vector_from_optimize(est, "rb_logit", nwl)
            if rb_logit is not None:
                r_b_hat = 1.0 / (1.0 + np.exp(-rb_logit))

        else:
            fit = _STAN_MODEL.sample(
                data=stan_data,
                seed=int(stan_cfg["seed"]),
                chains=int(stan_cfg["chains"]),
                parallel_chains=1,
                iter_warmup=int(stan_cfg["iter_warmup"]),
                iter_sampling=int(stan_cfg["iter_sampling"]),
                show_progress=False,
                show_console=False,
                output_dir=pid_dir,
            )
            draws = fit.draws_pd()
            scalars = {v: float(draws[v].mean()) for v in out_vars if v in draws.columns}

            rb_logit = _extract_vector_from_draws(draws, "rb_logit", nwl)
            if rb_logit is not None:
                r_b_hat = 1.0 / (1.0 + np.exp(-rb_logit))

        return {
            "seg_id": seg_id,
            "ok": True,
            "scalars": scalars,
            "wl_nm": wl,  # keep for table writing
            "r_b_hat": r_b_hat,  # vector (Nwl,) or None
        }

    except Exception as e:
        return {"seg_id": seg_id, "ok": False, "err": repr(e)}


# =============================================================================
# Build jobs from gpkg tables
# IMPORTANT: aux fields are now stored on the segments layer
# =============================================================================
def build_jobs_from_gpkg(
    gpkg_path: str,
    spectra_table: str,
    segments_gdf: gpd.GeoDataFrame,
) -> List[Dict[str, Any]]:
    # spectra are still in a separate (attribute) table
    spec = read_table(gpkg_path, spectra_table, columns=["seg_id", "wl_nm", "rho_med", "rho_mad"])

    # pull aux from segments layer attributes (already in the gpkg)
    need_cols = [
        "seg_id",
        "sun_zenith_med", "view_zenith_med",
        "relative_azimuth_med",
        "pressure_med", "altitude_med",
        "h_w_mu",
        "water_vapor_med", "ozone_med",
    ]
    missing = [c for c in need_cols if c not in segments_gdf.columns]
    if missing:
        raise RuntimeError(f"segments layer missing required columns: {missing}")

    aux = pd.DataFrame(segments_gdf[need_cols].copy())
    aux = aux.rename(columns={
        "sun_zenith_med": "theta_sun",
        "view_zenith_med": "theta_view",
        "relative_azimuth_med": "raa",
        "pressure_med": "pressure",
        "water_vapor_med": "water_vapor",
        "ozone_med": "ozone",
        "altitude_med": "altitude",
        "h_w_mu": "h_w",
    })

    # --- DEBUG: check uniqueness ---
    print("spec rows per seg (min/med/max):",
          int(spec.groupby("seg_id").size().min()),
          int(spec.groupby("seg_id").size().median()),
          int(spec.groupby("seg_id").size().max()))

    aux_counts = aux.groupby("seg_id").size()
    print("aux rows per seg (min/med/max):",
          int(aux_counts.min()), int(aux_counts.median()), int(aux_counts.max()))

    bad_aux = aux_counts[aux_counts > 1]
    print("segments with duplicate aux rows:", bad_aux.shape[0])
    if not bad_aux.empty:
        print(bad_aux.head(20))

    merged = spec.merge(aux, on="seg_id", how="inner")
    if merged.empty:
        raise RuntimeError("No rows after joining spectra table to segments attributes (seg_id mismatch?).")

    jobs: List[Dict[str, Any]] = []
    for seg_id, gg in merged.groupby("seg_id", sort=True):
        gg = gg.sort_values("wl_nm")
        row0 = gg.iloc[0]

        wl = gg["wl_nm"].to_numpy(float)
        rho_med = gg["rho_med"].to_numpy(float)
        rho_mad = gg["rho_mad"].to_numpy(float)

        job = dict(
            seg_id=int(seg_id),
            wl_nm=wl,
            rho_med=rho_med,
            rho_mad=rho_mad,
            theta_sun=float(row0["theta_sun"]),
            theta_view=float(row0["theta_view"]),
            raa=float(row0["raa"]),
            pressure=float(row0["pressure"]),
            altitude=float(row0["altitude"]),
            h_w=float(row0["h_w"]),
            water_vapor=float(row0["water_vapor"]),
            ozone=float(row0["ozone"]),
        )

        if not np.all(np.isfinite([job["theta_sun"], job["theta_view"], job["raa"], job["pressure"], job["altitude"]])):
            continue
        if not np.all(np.isfinite(rho_med)):
            continue

        jobs.append(job)

    if not jobs:
        raise RuntimeError("No valid segments to invert.")

    w0 = jobs[0]["wl_nm"]
    for j in jobs[1:]:
        if j["wl_nm"].shape != w0.shape or np.max(np.abs(j["wl_nm"] - w0)) > 1e-6:
            raise ValueError("Wavelength grid differs between segments (export common wl grid).")

    return jobs


# =============================================================================
# Main driver
# =============================================================================
def run_gpkg_atmaqua_inversion(
    in_gpkg: str,
    out_gpkg: str,
    *,
    segments_layer: str = "segments",
    spectra_table: str = "spectra",
    stan_model_path: str,
    aer_lut_path: str,
    gas_lut_path: str,
    out_vars: Tuple[str, ...],
    method: str = "optimize",
    n_workers: Optional[int] = None,
) -> int:
    gdf = gpd.read_file(in_gpkg, layer=segments_layer)
    if gdf.empty:
        raise RuntimeError(f"Empty layer '{segments_layer}'")
    if "seg_id" not in gdf.columns:
        raise RuntimeError("segments layer must have seg_id")

    jobs = build_jobs_from_gpkg(in_gpkg, spectra_table=spectra_table, segments_gdf=gdf)

    out_vars_list = list(out_vars)
    if "lp__" not in out_vars_list:
        out_vars_list.append("lp__")
    out_vars = tuple(out_vars_list)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    bottom_class_names = ("coraline_crustose_algae", "saccharina_latissima", "sediment")

    user_header = "/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp"
    a_w_csv = "/home/raphael/R/SABER/inst/extdata/a_w.csv"
    a0_a1_phyto_csv = "/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv"
    r_b_gamache_csv = "/home/raphael/R/SABER/inst/extdata/r_b_gamache.csv"

    exe_file = compile_model(stan_model_path, user_header=user_header)

    wavelength = np.asarray(jobs[0]["wl_nm"], dtype=float)
    initargs = (
        exe_file,
        wavelength,
        2,  # water_type
        1,  # shallow
        bottom_class_names,
        0.0051, 0.0047, 0.017, 0.006, 0.65,
        a_w_csv, a0_a1_phyto_csv, r_b_gamache_csv,
        aer_lut_path,
        gas_lut_path,
    )

    stan_cfg = dict(method=method, seed=1234, chains=2, iter_warmup=300, iter_sampling=300)

    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=initargs,
    ) as ex:
        futs = [ex.submit(invert_one_segment, job, stan_cfg, out_vars) for job in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            if r.get("ok"):
                results.append(r)
            else:
                failed.append(r)

    if not results:
        err = failed[0].get("err") if failed else "unknown"
        raise RuntimeError(f"All inversions failed. Example error: {err}")

    scal_df = pd.DataFrame(
        [{"seg_id": r["seg_id"], **{sql_safe(k): r["scalars"].get(k, np.nan) for k in out_vars}} for r in results]
    )
    gdf["seg_id"] = gdf["seg_id"].astype(int)
    scal_df["seg_id"] = scal_df["seg_id"].astype(int)
    gdf_out = gdf.merge(scal_df, on="seg_id", how="left")

    out_dir = os.path.dirname(out_gpkg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    gdf_out.to_file(out_gpkg, layer=segments_layer, driver="GPKG")

    if failed:
        write_table_sqlite(out_gpkg, "failed_segments", pd.DataFrame(failed), overwrite=True)

    # write r_b spectrum table
    write_rb_spectrum_table(out_gpkg, "spectra", results)


    return len(results)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    n_ok = run_gpkg_atmaqua_inversion(
        in_gpkg="/D/Data/WISE/ACI-12A/segmentation_outputs/segments.gpkg",
        out_gpkg="/D/Data/WISE/ACI-12A/segmentation_outputs/segments_atmaqua_inverted.gpkg",
        stan_model_path="/home/raphael/R/SABER/inst/stan/model_atmaqua_rb.stan",
        aer_lut_path="/home/raphael/PycharmProjects/reverie/reverie/data/lut/aer_lut_WISE.nc",
        gas_lut_path="/home/raphael/PycharmProjects/reverie/reverie/data/lut/gas_lut_WISE.nc",
        out_vars=("aod550", "chl", "a_g_440", "spm", "bb_p_gamma", "a_g_slope", "h_w", "r_b_a"),
        method="optimize",
        n_workers=None,
    )
    print("segments inverted:", n_ok)

# QGIS action code to map polygon selection -> spectrum table filtering
# from qgis.core import QgsProject, QgsVectorLayer
#
# seg_id = [% "seg_id" %]
#
# proj = QgsProject.instance()
#
# tbl = proj.mapLayersByName("segments_atmaqua_inverted — spectra")[0]  # <-- exact layer name
# tbl.removeSelection()
#
# # build expression depending on seg_id type
# if isinstance(seg_id, (int, float)):
#     expr = f"\"seg_id\" = {int(seg_id)}"
# else:
#     expr = f"\"seg_id\" = '{seg_id}'"
#
# tbl.selectByExpression(expr, QgsVectorLayer.SetSelection)