from __future__ import annotations

import os
import logging
import sqlite3
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional, Sequence, Tuple
import functools

import numpy as np
import pandas as pd
import geopandas as gpd


# =============================================================================
# Logging / silence
# =============================================================================
_STAN_MODEL = None
_BASE: dict[str, Any] | None = None


def silence_everything() -> None:
    logging.disable(logging.ERROR)
    for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
    os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"
    os.environ["CMDSTANPY_SILENT"] = "true"


# =============================================================================
# Small numerics helpers
# =============================================================================
def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xout = np.asarray(xout, float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    yout = np.interp(xout, x, y)
    yout = np.where(xout < x[0], y[0], yout)
    yout = np.where(xout > x[-1], y[-1], yout)
    return yout


def load_optics_luts_from_csv(
    wavelength: Sequence[float],
    a_w_csv: str,
    a0_a1_phyto_csv: str,
) -> Dict[str, np.ndarray]:
    wl = np.asarray(wavelength, float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wavelength must be 1D length>=2")

    a_w_df = pd.read_csv(a_w_csv)
    a01_df = pd.read_csv(a0_a1_phyto_csv)

    for c in ("wavelength", "a_w"):
        if c not in a_w_df.columns:
            raise ValueError(f"{a_w_csv} missing column '{c}'")
    for c in ("wavelength", "a0", "a1"):
        if c not in a01_df.columns:
            raise ValueError(f"{a0_a1_phyto_csv} missing column '{c}'")

    a_w = interp1_flat(a_w_df["wavelength"].to_numpy(float), a_w_df["a_w"].to_numpy(float), wl)
    a0 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a0"].to_numpy(float), wl)
    a1 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a1"].to_numpy(float), wl)

    bb_w = 0.00111 * (wl / 500.0) ** (-4.32)
    return {"a_w": a_w, "a0": a0, "a1": a1, "bb_w": bb_w}


def compile_model(stan_model_path: str, user_header: str | None = None) -> str:
    from cmdstanpy import CmdStanModel
    m = CmdStanModel(stan_file=stan_model_path, user_header=user_header)
    return m.exe_file


# =============================================================================
# Worker init: build _BASE for ETA model
# =============================================================================
def _init_worker(
    exe_file: str,
    wavelength: np.ndarray,
    *,
    water_type: int,
    shallow: int,
    a_w_csv: str,
    a0_a1_phyto_csv: str,
    rb_obs_path: str,
    # ETA prior config
    q: int,
    eps: float,
    cov_jitter: float,
    sigma2_floor: float,
    # prior hyperparameters (must match Stan data names)
    priors: dict,
) -> None:
    global _STAN_MODEL, _BASE
    silence_everything()

    from cmdstanpy import CmdStanModel
    # IMPORT the ETA builder you generated earlier (change module path as needed)
    from reverie.correction.aquatic.stan_data_gmm import make_stan_data_eta_model  # <-- adjust to your file/module name

    _STAN_MODEL = CmdStanModel(exe_file=exe_file)

    wl = np.asarray(wavelength, float)
    optics = load_optics_luts_from_csv(wl, a_w_csv=a_w_csv, a0_a1_phyto_csv=a0_a1_phyto_csv)

    dummy_rrs = np.zeros_like(wl, dtype=float)
    dummy_sig = np.full_like(wl, 1e-6, dtype=float)

    base = make_stan_data_eta_model(
        wavelength=wl,
        rrs_obs=dummy_rrs,
        sigma_rrs=dummy_sig,
        a_w=optics["a_w"],
        a0=optics["a0"],
        a1=optics["a1"],
        bb_w=optics["bb_w"],
        rb_obs_path=rb_obs_path,
        q=q,
        eps=eps,
        cov_jitter=cov_jitter,
        sigma2_floor=sigma2_floor,
        water_type=water_type,
        theta_sun=30.0,   # overwritten per segment
        theta_view=0.0,   # overwritten per segment
        shallow=shallow,
        # hypers
        a_nap_star_mu=float(priors["a_nap_star_mu"]),
        a_nap_star_sd=float(priors["a_nap_star_sd"]),
        bb_p_star_mu=float(priors["bb_p_star_mu"]),
        bb_p_star_sd=float(priors["bb_p_star_sd"]),
        a_g_s_mu=float(priors["a_g_s_mu"]),
        a_g_s_sd=float(priors["a_g_s_sd"]),
        a_nap_s_mu=float(priors["a_nap_s_mu"]),
        a_nap_s_sd=float(priors["a_nap_s_sd"]),
        bb_p_gamma_mu=float(priors["bb_p_gamma_mu"]),
        bb_p_gamma_sd=float(priors["bb_p_gamma_sd"]),
        h_w_mu=float(priors["h_w_mu"]),
        h_w_sd=float(priors["h_w_sd"]),
    )

    # Ensure eta-fields exist (defensive)
    required = ("eta_U", "eta_mu_k", "eta_L_k", "eta_sigma2_k", "eta_pi")
    miss = [k for k in required if k not in base]
    if miss:
        raise RuntimeError(f"ETA base missing keys: {miss}")

    _BASE = base


# =============================================================================
# Segment inversion
# =============================================================================
def make_init(stan_data):
    K = int(stan_data["K"])
    # pick the most-weighted component
    # k_star = int(np.argmax(stan_data["eta_pi"]))  # 0-based
    # eta_mu = np.array(stan_data["eta_mu_k"][k_star])  # length n_wl
    eta_mu = np.array(np.mean(stan_data["eta_mu_k"], 0))#[k_star])  # length n_wl

    init = dict(
        chl=0.9,
        a_g_440=0.25,
        spm=2.0,
        a_nap_star=float(stan_data["a_nap_star_mu"]),
        bb_p_star=float(stan_data["bb_p_star_mu"]),
        a_g_s=float(stan_data["a_g_s_mu"]),
        a_nap_s=float(stan_data["a_nap_s_mu"]),
        bb_p_gamma=float(stan_data["bb_p_gamma_mu"]),
        h_w=float(np.clip(stan_data["h_w_mu"], 0.0, 30.0)),
        eta_b=eta_mu.tolist(),
        sigma_model=5e-5,
    )
    return init

def invert_one_segment(job: dict, stan_cfg: dict, out_vars: tuple[str, ...]) -> dict:
    global _STAN_MODEL, _BASE
    assert _STAN_MODEL is not None
    assert _BASE is not None

    seg_id = int(job["seg_id"])
    wl = job["wl_nm"].astype(float)

    rho_est = job["rho_med"].astype(float)
    rho_unc = job["rho_mad"].astype(float)

    rrs_0p = rho_est / np.pi
    rrs_0m = rrs_0p / (0.52 + 1.7 * rrs_0p)
    if not np.all(np.isfinite(rrs_0m)):
        return {"seg_id": seg_id, "ok": False, "err": "non-finite rrs_0m"}

    rrs_0p_unc = rho_unc / np.pi
    rrs_0m_unc = rrs_0p_unc / (0.52 + 1.7 * rrs_0p_unc)
    rrs_0m_unc = np.maximum(rrs_0m_unc, 1e-8)

    stan_data = dict(_BASE)
    stan_data["rrs_obs"] = rrs_0m.tolist()
    stan_data["sigma_rrs"] = rrs_0m_unc.tolist()
    stan_data["theta_sun"] = float(job["theta_sun"])
    stan_data["theta_view"] = float(job["theta_view"])
    stan_data["h_w_mu"] = float(job["h_w_mu"])
    stan_data["h_w_sd"] = float(job["h_w_sd"])

    # sanity: eta fields must be present
    if "eta_U" not in stan_data:
        return {"seg_id": seg_id, "ok": False, "err": "stan_data missing eta_U (wrong base/builder?)"}

    pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")

    try:
        if stan_cfg["method"] == "optimize":
            fit = _STAN_MODEL.optimize(
                data=stan_data,
                seed=int(stan_cfg["seed"]),
                output_dir=pid_dir,
                show_console=False,
                algorithm="lbfgs",
                # algorithm="newton",
                inits=make_init(stan_data),
                iter=50000,
                require_converged=True,
                # tol_rel_obj=1e-10,
                # tol_grad=1e-8,
                # tol_param=1e-10,
            )
            est = fit.optimized_params_dict
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
            # Mean of the posterior
            draws = fit.draws_pd()
            est = {v: float(draws[v].mean()) for v in draws.columns}

            # MAP
            # lp = draws["lp__"].to_numpy()
            # map_row = draws.iloc[int(lp.argmax())]
            # est = {k: float(map_row[k]) for k in draws.columns if
            #            k not in ["lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__", "divergent__",
            #                      "energy__"]}

        scalars = {v: float(est[v]) for v in out_vars if v in est}

        nwl = len(wl)
        r_b_hat = np.full(nwl, np.nan, dtype=float)
        rrs_hat = np.full(nwl, np.nan, dtype=float)
        eta_b_hat = np.full(nwl, np.nan, dtype=float)

        for i in range(nwl):
            kb = f"r_b_hat[{i + 1}]"
            kr = f"rrs_hat[{i + 1}]"
            ke = f"eta_b[{i + 1}]"
            if kb in est:
                r_b_hat[i] = float(est[kb])
            if kr in est:
                rrs_hat[i] = float(est[kr])
            if ke in est:
                eta_b_hat[i] = float(est[ke])

        # rrs(0-) -> rrs(0+) -> rho
        rho_hat = np.full(nwl, np.nan, dtype=float)
        ok = np.isfinite(rrs_hat) & (1 - 1.7 * rrs_hat > 1e-12)
        rrs_0p_hat = np.full(nwl, np.nan, dtype=float)
        rrs_0p_hat[ok] = (0.52 * rrs_hat[ok]) / (1 - 1.7 * rrs_hat[ok])
        rho_hat[ok] = rrs_0p_hat[ok] * np.pi

        # physical reflectance already via inv_logit in Stan; keep it bounded
        r_b_hat = np.clip(np.nan_to_num(r_b_hat, nan=0.0), 0.0, 1.0)

        return {
            "seg_id": seg_id,
            "ok": True,
            "scalars": scalars,
            "wl_nm": wl.astype(float),
            "r_b_med": r_b_hat.astype(float),
            "rho_hat": rho_hat.astype(float),
            "eta_b_med": eta_b_hat.astype(float),  # optional debug output
        }

    except Exception as e:
        return {"seg_id": seg_id, "ok": False, "err": repr(e)}


# =============================================================================
# SQLite helpers
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


def write_table_sqlite(gpkg_path: str, table: str, df: pd.DataFrame, overwrite: bool = True) -> None:
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
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
    conn.close()


def sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {r[1] for r in rows}

def pick_existing_column(conn: sqlite3.Connection, table: str, candidates: Sequence[str]) -> str:
    cols = sqlite_table_columns(conn, table)
    for c in candidates:
        if c in cols:
            return c
    raise RuntimeError(
        f'None of these columns exist in "{table}": {list(candidates)}. Found: {sorted(cols)}'
    )

# =============================================================================
# Main driver (ONLY changes: initkwargs uses eps/cov_jitter; aux must provide h_w_mu/h_w_sd)
# =============================================================================
def run_gpkg_inversion_to_new_file(
    in_gpkg: str,
    out_gpkg: str,
    segments_layer: str = "segments",
    spectra_table: str = "spectra",
    seg_id_field: str = "seg_id",
    wl_field: str = "wl_nm",
    rho_med_field: str = "rho_med",
    rho_mad_field: str = "rho_mad",
    sun_field: str = "theta_sun",
    view_field: str = "theta_view",
    hw_mu_field: str = "h_w_mu",
    hw_sd_field: str = "h_w_sd",
    stan_model_path: str = "",
    out_vars: tuple[str, ...] = ("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
    method: str = "optimize",
    n_workers: int | None = None,
    h_w_mu_override: float | None = None,
    h_w_sd_override: float | None = None,
) -> int:
    # segments layer (now contains the scalar fields)
    gdf = gpd.read_file(in_gpkg, layer=segments_layer)
    if seg_id_field not in gdf.columns:
        raise RuntimeError(f"'{seg_id_field}' not found in segments layer")

    for c in (sun_field, view_field, hw_mu_field, hw_sd_field):
        if c not in gdf.columns:
            raise RuntimeError(f"'{c}' not found in segments layer. Available: {list(gdf.columns)}")

    # read spectra table (SQLite) from the same gpkg
    conn = sqlite3.connect(in_gpkg)
    spec = pd.read_sql_query(
        f"""
        SELECT
            {seg_id_field}   AS seg_id,
            {wl_field}       AS wl_nm,
            {rho_med_field}  AS rho_med,
            {rho_mad_field}  AS rho_mad
        FROM "{spectra_table}"
        """,
        conn,
    )
    conn.close()

    # build a scalars dataframe from the segments layer (no aux table anymore)
    seg_scal = pd.DataFrame(
        {
            "seg_id": pd.to_numeric(gdf[seg_id_field], errors="coerce").astype("Int64"),
            "theta_sun": pd.to_numeric(gdf[sun_field], errors="coerce"),
            "theta_view": pd.to_numeric(gdf[view_field], errors="coerce"),
            "h_w_mu": pd.to_numeric(gdf[hw_mu_field], errors="coerce"),
            "h_w_sd": pd.to_numeric(gdf[hw_sd_field], errors="coerce"),
        }
    ).dropna(subset=["seg_id"])
    seg_scal["seg_id"] = seg_scal["seg_id"].astype(int)

    # merge spectra + scalars
    spec["seg_id"] = pd.to_numeric(spec["seg_id"], errors="coerce")
    spec = spec.dropna(subset=["seg_id"])
    spec["seg_id"] = spec["seg_id"].astype(int)

    merged = spec.merge(seg_scal, on="seg_id", how="inner")

    jobs: list[dict] = []
    for seg_id, gg in merged.groupby("seg_id"):
        gg = gg.sort_values("wl_nm")

        theta_sun = float(gg["theta_sun"].iloc[0])
        theta_view = float(gg["theta_view"].iloc[0])

        if not np.isfinite(theta_sun) or not np.isfinite(theta_view):
            continue

        if h_w_mu_override is None:
            h_w_mu = float(gg["h_w_mu"].iloc[0])
            if not np.isfinite(h_w_mu):
                continue
        else:
            h_w_mu = float(h_w_mu_override)

        if h_w_sd_override is None:
            h_w_sd = float(gg["h_w_sd"].iloc[0])
            if not np.isfinite(h_w_sd):
                continue
        else:
            h_w_sd = float(h_w_sd_override)

        jobs.append(
            {
                "seg_id": int(seg_id),
                "wl_nm": gg["wl_nm"].to_numpy(float),
                "rho_med": gg["rho_med"].to_numpy(float),
                "rho_mad": gg["rho_mad"].to_numpy(float),
                "theta_sun": theta_sun,
                "theta_view": theta_view,
                "h_w_mu": h_w_mu,
                "h_w_sd": h_w_sd,
            }
        )

    if not jobs:
        raise RuntimeError("No valid segments to invert (check NaNs in segments scalars or join).")

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    wavelength = np.asarray(jobs[0]["wl_nm"], dtype=float)

    user_header = "/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp"
    exe_file = compile_model(stan_model_path, user_header=user_header)

    a_w_csv = "/home/raphael/R/SABER/inst/extdata/a_w.csv"
    a0_a1_phyto_csv = "/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv"
    rb_obs_path = "/home/raphael/R/SABER/inst/extdata/r_b_gamache_obs.csv"

    priors = dict(
        a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
        bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
        a_g_s_mu=0.017,       a_g_s_sd=0.0012,
        a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
        bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
        h_w_mu=3.0,           h_w_sd=2.0,
    )

    initkwargs = dict(
        water_type=2,
        shallow=1,
        a_w_csv=a_w_csv,
        a0_a1_phyto_csv=a0_a1_phyto_csv,
        rb_obs_path=rb_obs_path,
        q=8,
        eps=1e-6,
        cov_jitter=1e-6,
        sigma2_floor=1e-6,
        priors=priors,
    )

    ctx = mp.get_context("spawn")
    stan_cfg = dict(method=method, seed=1234, chains=2, iter_warmup=300, iter_sampling=300)

    init_fn = functools.partial(_init_worker, **initkwargs)

    results: list[dict] = []
    failed: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_fn,
        initargs=(exe_file, wavelength),
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

    gdf[seg_id_field] = pd.to_numeric(gdf[seg_id_field], errors="coerce")
    gdf = gdf.dropna(subset=[seg_id_field])
    gdf[seg_id_field] = gdf[seg_id_field].astype(int)
    scal_df["seg_id"] = scal_df["seg_id"].astype(int)

    gdf_out = gdf.merge(scal_df, left_on=seg_id_field, right_on="seg_id", how="left", suffixes=("", "_inv"))
    if "seg_id_inv" in gdf_out.columns:
        gdf_out = gdf_out.drop(columns=["seg_id_inv"])

    rb_rows = []
    for r in results:
        seg_id = int(r["seg_id"])
        wl_nm = r["wl_nm"]
        rb_med = r["r_b_med"]
        rho_hat = r["rho_hat"]
        eta_med = r.get("eta_b_med", None)

        for j, (wl_i, rb_i, rho_i) in enumerate(zip(wl_nm, rb_med, rho_hat)):
            row = {"seg_id": seg_id, "wl_nm": float(wl_i), "r_b_hat": float(rb_i), "rho_hat": float(rho_i)}
            if eta_med is not None and j < len(eta_med) and np.isfinite(eta_med[j]):
                row["eta_b_hat"] = float(eta_med[j])
            rb_rows.append(row)

    rb_df = pd.DataFrame(rb_rows)

    out_dir = os.path.dirname(out_gpkg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    gdf_out.to_file(out_gpkg, layer=segments_layer, driver="GPKG")
    write_table_sqlite(out_gpkg, "spectra", rb_df, overwrite=True)

    return (len(results), failed)

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    n_ok, failed = run_gpkg_inversion_to_new_file(
        in_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments.gpkg",
        out_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments_inverted.gpkg",
        segments_layer="segments",
        spectra_table="spectra",
        stan_model_path="/home/raphael/R/SABER/inst/stan/model_gmm.stan",
        out_vars=("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
        method="optimize",
        # method="sample",
        n_workers=None,
        # h_w_mu_override=5,
        # h_w_sd_override=5,
        h_w_mu_override=None,
        h_w_sd_override=1,
    )
    print("segments inverted:", n_ok)
    print("segments failed:", len(failed))
    # print(failed[0].get("err"))

# from qgis.core import QgsProject, QgsVectorLayer
#
# seg_id = [% "seg_id" %]
#
# proj = QgsProject.instance()
#
# tbl = proj.mapLayersByName("segments_inverted — spectra")[0]  # <-- exact layer name
# tbl.removeSelection()
#
# # build expression depending on seg_id type
# if isinstance(seg_id, (int, float)):
#     expr = f"\"seg_id\" = {int(seg_id)}"
# else:
#     expr = f"\"seg_id\" = '{seg_id}'"
#
# tbl.selectByExpression(expr, QgsVectorLayer.SetSelection)
