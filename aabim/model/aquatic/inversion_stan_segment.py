#!/usr/bin/env python3
from __future__ import annotations

import os
import logging
import sqlite3
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd

from reverie.correction.aquatic.stan_data import make_stan_data_base_from_csv
import reverie.correction.aquatic.rtm as rtm


# ------------------ Stan worker globals ------------------
_STAN_MODEL = None
_BASE: dict[str, Any] | None = None


def silence_everything() -> None:
    logging.disable(logging.ERROR)
    for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False
    os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"


# ---- compile ONCE in parent (avoids multi-proc compile race) ----
def compile_model(stan_model_path: str, user_header: str | None = None) -> str:
    from cmdstanpy import CmdStanModel
    m = CmdStanModel(stan_file=stan_model_path, user_header=user_header)
    return m.exe_file  # path to compiled executable


def _init_worker(
    exe_file: str,  # compiled executable, not .stan
    wavelength: np.ndarray,
    water_type: int,
    shallow: int,
    bottom_class_names,
    a_nap_star: float,
    bb_p_star: float,
    a_g_s: float,
    a_nap_s: float,
    bb_p_gamma: float,
    a_w_csv: str,
    a0_a1_phyto_csv: str,
    r_b_gamache_csv: str,
) -> None:
    global _STAN_MODEL, _BASE
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


def invert_one_segment(job: dict, stan_cfg: dict, out_vars: tuple[str, ...], sigma_rrs: float = 0.0003) -> dict:
    global _STAN_MODEL, _BASE
    assert _STAN_MODEL is not None, "Stan model not initialized in worker"
    assert _BASE is not None, "Base stan data not initialized in worker"

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
    rrs_0m_unc = np.maximum(rrs_0m_unc, 10e-6)

    stan_data = dict(_BASE)
    stan_data["rrs_obs"] = rrs_0m.astype(float)
    stan_data["sigma_rrs"] = rrs_0m_unc.astype(float)#np.full(rrs_0m.size, sigma_rrs, dtype=float)
    stan_data["theta_sun"] = float(job["theta_sun"])
    stan_data["theta_view"] = float(job["theta_view"])
    stan_data["h_w"] = float(job["h_w"])

    # Always write CmdStan outputs to a local temp dir (avoid /D mounts/perms/noexec issues)
    pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")

    try:
        if stan_cfg["method"] == "optimize":
            fit = _STAN_MODEL.optimize(
                data=stan_data,
                seed=stan_cfg["seed"],
                output_dir=pid_dir,
                show_console=False,
            )
            est = fit.optimized_params_dict
        else:
            fit = _STAN_MODEL.sample(
                data=stan_data,
                seed=stan_cfg["seed"],
                chains=int(stan_cfg["chains"]),
                parallel_chains=1,
                iter_warmup=int(stan_cfg["iter_warmup"]),
                iter_sampling=int(stan_cfg["iter_sampling"]),
                show_progress=False,
                show_console=False,
                output_dir=pid_dir,
            )
            draws = fit.draws_pd()
            est = {v: float(draws[v].mean()) for v in out_vars if v in draws.columns}

        scalars = {v: float(est[v]) for v in out_vars if v in est}

        # Required for the rb solve
        chl = float(est["chl"])
        a_g_440 = float(est["a_g_440"])
        spm = float(est["spm"])

        # fixed params (you can later pass these in if you want them configurable)
        a_nap_star = 0.0051
        bb_p_star = 0.0047
        a_g_s = 0.017
        a_nap_s = 0.006
        bb_p_gamma = 0.65

        a, bb = rtm.iop_from_oac_spm_np(
            wl=wl,
            a_w=np.asarray(_BASE["a_w"], dtype=float),
            a0=np.asarray(_BASE["a0"], dtype=float),
            a1=np.asarray(_BASE["a1"], dtype=float),
            bb_w=np.asarray(_BASE["bb_w"], dtype=float),
            chl=chl,
            a_g_440=a_g_440,
            spm=spm,
            a_g_s=a_g_s,
            a_nap_star=a_nap_star,
            a_nap_s=a_nap_s,
            bb_p_star=bb_p_star,
            bb_p_gamma=bb_p_gamma,
        )

        rb = rtm.solve_rb_am03_np(
            wl=wl,
            a=a,
            bb=bb,
            water_type=2,
            theta_sun_deg=float(job["theta_sun"]),
            theta_view_deg=float(job["theta_view"]),
            h_w=float(job["h_w"]),
            rrs_obs=rrs_0m.astype(float),
        )

        rb = np.clip(np.nan_to_num(rb, nan=0.0), 0.0, 1.0)

        # --- rebuild rrs_hat vector from Stan outputs ---
        nwl = len(wl)
        rrs_hat = np.full(nwl, np.nan, dtype=float)

        for i in range(nwl):
            key = f"rrs_hat[{i + 1}]"  # Stan uses 1-based indexing
            if key in est:
                rrs_hat[i] = float(est[key])

        # rrs_0p = rho / np.pi
        # rrs_0m = rrs_0p / (0.52 + 1.7 * rrs_0p)

        rrs_0p = (0.52 * rrs_hat) / (1 - 1.7 * rrs_hat)
        rho_hat = rrs_0p * np.pi

        return {
            "seg_id": seg_id,
            "ok": True,
            "scalars": scalars,
            "wl_nm": wl.astype(float),
            "r_b_med": rb.astype(float),
            "rho_hat": rho_hat,
        }

    except Exception as e:
        return {"seg_id": seg_id, "ok": False, "err": repr(e)}


# ---------- helpers ----------
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


# ------------------ main driver ------------------
def run_gpkg_inversion_to_new_file(
    in_gpkg: str,
    out_gpkg: str,
    segments_layer: str = "segments",
    spectra_table: str = "spectra",
    aux_table: str = "aux",
    seg_id_field: str = "seg_id",
    wl_field: str = "wl_nm",
    rho_med_field: str = "rho_med",
    rho_mad_field: str = "rho_mad",
    sun_field: str = "sun_zenith_med",
    view_field: str = "view_zenith_med",
    hw_field: str = "h_w_mu",
    stan_model_path: str = "",
    out_vars: tuple[str, ...] = ("chl", "a_g_440", "spm", "r_b_a", "sigma_model"),
    method: str = "optimize",
    n_workers: int | None = None,
) -> int:
    # --- read inputs ---
    gdf = gpd.read_file(in_gpkg, layer=segments_layer)

    conn = sqlite3.connect(in_gpkg)
    spec = pd.read_sql_query(
        f"""
        SELECT
            {seg_id_field} AS seg_id,
            {wl_field}     AS wl_nm,
            {rho_med_field} AS rho_med,
            {rho_mad_field} AS rho_mad
        FROM "{spectra_table}"
        """,
        conn,
    )
    aux = pd.read_sql_query(
        f"""
        SELECT
            {seg_id_field} AS seg_id,
            {sun_field}    AS theta_sun,
            {view_field}   AS theta_view,
            {hw_field}     AS h_w
        FROM "{aux_table}"
        """,
        conn,
    )
    conn.close()

    merged = spec.merge(aux, on="seg_id", how="inner")

    jobs: list[dict] = []
    for seg_id, gg in merged.groupby("seg_id"):
        gg = gg.sort_values("wl_nm")
        if not np.isfinite(gg["theta_sun"].iloc[0]) or not np.isfinite(gg["h_w"].iloc[0]):
            continue
        jobs.append(
            {
                "seg_id": int(seg_id),
                "wl_nm": gg["wl_nm"].to_numpy(float),
                "rho_med": gg["rho_med"].to_numpy(float),
                "rho_mad": gg["rho_mad"].to_numpy(float),
                "theta_sun": float(gg["theta_sun"].iloc[0]),
                "theta_view": float(gg["theta_view"].iloc[0]),
                "h_w": float(gg["h_w"].iloc[0]),
            }
        )

    if not jobs:
        raise RuntimeError("No valid segments to invert (check aux join / NaNs).")

    # add lp__ once
    out_vars_list = list(out_vars)
    if "lp__" not in out_vars_list:
        out_vars_list.append("lp__")
    out_vars = tuple(out_vars_list)

    # --- multiprocessing init ---
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    wavelength = np.asarray(jobs[0]["wl_nm"], dtype=float)
    bottom_class_names = ("coraline_crustose_algae", "saccharina_latissima", "sediment")

    # your headers/csv paths
    user_header = "/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp"
    a_w_csv = "/home/raphael/R/SABER/inst/extdata/a_w.csv"
    a0_a1_phyto_csv = "/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv"
    r_b_gamache_csv = "/home/raphael/R/SABER/inst/extdata/r_b_gamache.csv"

    # compile once
    exe_file = compile_model(stan_model_path, user_header=user_header)

    initargs = (
        exe_file,
        wavelength,
        2,
        1,
        bottom_class_names,
        0.0051,
        0.0047,
        0.017,
        0.006,
        0.65,
        a_w_csv,
        a0_a1_phyto_csv,
        r_b_gamache_csv,
    )

    ctx = mp.get_context("spawn")
    stan_cfg = dict(method=method, seed=1234, chains=2, iter_warmup=300, iter_sampling=300)

    results: list[dict] = []
    failed: list[dict] = []

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
        # show one error to help debugging
        err = failed[0].get("err") if failed else "unknown"
        raise RuntimeError(f"All inversions failed. Example error: {err}")

    # --- build outputs ---
    # 1) polygon scalars as columns on segments layer
    scal_df = pd.DataFrame(
        [
            {"seg_id": r["seg_id"], **{sql_safe(k): r["scalars"].get(k, np.nan) for k in out_vars}}
            for r in results
        ]
    )

    # ensure seg_id type matches join
    gdf[seg_id_field] = gdf[seg_id_field].astype(int)
    scal_df["seg_id"] = scal_df["seg_id"].astype(int)

    gdf_out = gdf.merge(scal_df, left_on=seg_id_field, right_on="seg_id", how="left", suffixes=("", "_inv"))
    if "seg_id_inv" in gdf_out.columns:
        gdf_out = gdf_out.drop(columns=["seg_id_inv"])
    #
    # # 2) spectral table (non-spatial)
    # rb_rows = []
    # for r in results:
    #     for wl, rb in zip(r["wl_nm"], r["r_b_med"]):
    #         rb_rows.append({"seg_id": int(r["seg_id"]), "wl_nm": float(wl), "r_b_med": float(rb)})
    # rb_df = pd.DataFrame(rb_rows)

    rb_rows = []
    for r in results:
        seg_id = int(r["seg_id"])
        wl_nm = r["wl_nm"]
        rb_med = r["r_b_med"]
        rho_hat = r["rho_hat"]  # <-- add this in your results

        if not (len(wl_nm) == len(rb_med) == len(rho_hat)):
            raise ValueError(f"Length mismatch for seg_id={seg_id}: "
                             f"wl={len(wl_nm)} rb={len(rb_med)} rho={len(rho_hat)}")

        for wl, rb, rho in zip(wl_nm, rb_med, rho_hat):
            rb_rows.append({
                "seg_id": seg_id,
                "wl_nm": float(wl),
                "r_b_med": float(rb),
                "rho_hat": float(rho) if rho is not None else None,
            })

    rb_df = pd.DataFrame(rb_rows)

    # --- write new gpkg ---
    out_dir = os.path.dirname(out_gpkg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    # write polygons (spatial) with GDAL via geopandas
    gdf_out.to_file(out_gpkg, layer=segments_layer, driver="GPKG")

    # write rb_hat as non-spatial sqlite table into the same file
    write_table_sqlite(out_gpkg, "spectra", rb_df, overwrite=True)



    return len(results)

# from qgis.core import (
#     QgsVectorLayer,
#     QgsAction,
# )
#
# def install_action_into_gpkg(gpkg_path: str,
#                              target_layer_name: str,
#                              action_name: str = "Select seg_id in segments — spectra",
#                              style_name: str = "default_with_actions"):
#     """
#     Writes a Python action into the GPKG by saving it as part of the layer's default style
#     in the GeoPackage layer_styles table.
#     """
#
#     # Load target layer directly from GPKG
#     uri = f"{gpkg_path}|layername={target_layer_name}"
#     lyr = QgsVectorLayer(uri, target_layer_name, "ogr")
#     if not lyr.isValid():
#         raise RuntimeError(f"Could not load layer '{target_layer_name}' from: {gpkg_path}")
#
#     # Your action code (expression placeholder gets evaluated by QGIS before running)
#     action_code = r"""
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
# """.strip()
#
#     # Remove any existing action with same name to avoid duplicates
#     mgr = lyr.actions()
#     for a in list(mgr.actions()):
#         if a.name() == action_name:
#             mgr.removeAction(a.id())
#
#     # Add the Python action
#     act = QgsAction(QgsAction.GenericPython, action_name, action_code, action_name)
#     mgr.addAction(act)
#
#     # Save style (including actions) INTO the GeoPackage as default
#     ok, err = lyr.saveStyleToDatabase(style_name, "Stored by script (includes actions)", True, "")
#     if not ok:
#         raise RuntimeError(f"saveStyleToDatabase failed: {err}")
#
#     print(f"OK: Action saved into {gpkg_path} for layer '{target_layer_name}' as default style '{style_name}'.")


# ------------------ run ------------------
if __name__ == "__main__":
    n_ok = run_gpkg_inversion_to_new_file(
        in_gpkg="/D/Data/WISE/ACI-12A/jetski_el/segmentation_outputs/segments.gpkg",
        out_gpkg="/D/Data/WISE/ACI-12A/jetski_el/segmentation_outputs/segments_inverted.gpkg",
        segments_layer="segments",
        spectra_table="spectra",
        aux_table="aux",
        stan_model_path="/home/raphael/R/SABER/inst/stan/model_sc_spm.stan",
        out_vars=("chl", "a_g_440", "spm", "r_b_mix[1]", "r_b_mix[2]", "r_b_mix[3]", "r_b_a", "sigma_model"),
        method="optimize",
        n_workers=None,
    )
    print("segments inverted:", n_ok)

    # install_action_into_gpkg(
    #     gpkg_path=r"/D/Data/WISE/ACI-12A/jetski_el/segmentation_outputs/segment_inverted.gpkg",
    #     target_layer_name="segments_inverted — aux"  # or the layer that should *own* the action
    # )
