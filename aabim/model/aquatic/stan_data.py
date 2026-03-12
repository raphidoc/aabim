from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from functools import lru_cache


def interp1(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """Linear interp with flat extrapolation (R approx(rule=2))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xout = np.asarray(xout, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    yout = np.interp(xout, x, y)
    yout = np.where(xout < x[0], y[0], yout)
    yout = np.where(xout > x[-1], y[-1], yout)
    return yout


@dataclass(frozen=True)
class LUTs:
    a_w: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    bb_w: np.ndarray
    r_b_mu_lib: np.ndarray
    r_b_sd_lib: np.ndarray
    classes: List[str]


@lru_cache(maxsize=8)
def _read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_library_csvs(
    a_w_csv: str,
    a0_a1_phyto_csv: str,
    r_b_gamache_csv: str,
) -> Dict[str, pd.DataFrame]:
    """
    Reads the three CSVs. Cached per process via lru_cache.
    """
    a_w_df = _read_csv_cached(a_w_csv)
    a0_a1_df = _read_csv_cached(a0_a1_phyto_csv)
    rb_df = _read_csv_cached(r_b_gamache_csv)

    # Minimal column normalization (fail fast if not found)
    for col in ("wavelength", "a_w"):
        if col not in a_w_df.columns:
            raise ValueError(f"{a_w_csv} missing column '{col}'")

    for col in ("wavelength", "a0", "a1"):
        if col not in a0_a1_df.columns:
            raise ValueError(f"{a0_a1_phyto_csv} missing column '{col}'")

    # Accept a few common spellings for rb columns
    rb_df = rb_df.rename(columns={
        "wl": "wavelength",
        "r_b_mean": "r_b_mean",
        "r_b_sd": "r_b_sd",
    })

    required_rb = {"class", "wavelength", "r_b_mean", "r_b_sd"}
    missing = required_rb.difference(rb_df.columns)
    if missing:
        raise ValueError(f"{r_b_gamache_csv} missing columns: {', '.join(sorted(missing))}")

    return {"a_w": a_w_df, "a0_a1": a0_a1_df, "rb": rb_df}


def make_luts_from_csv(
    wavelength: Sequence[float],
    a_w_csv: str,
    a0_a1_phyto_csv: str,
    r_b_gamache_csv: str,
    sd_floor: float = 1e-6,
) -> LUTs:
    wl = np.asarray(wavelength, dtype=float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wavelength must be a 1D vector of length >= 2")

    dfs = load_library_csvs(a_w_csv, a0_a1_phyto_csv, r_b_gamache_csv)
    a_w_df = dfs["a_w"]
    a0_a1_df = dfs["a0_a1"]
    rb = dfs["rb"].copy()
    rb["class"] = rb["class"].astype(str)

    a_w_int = interp1(a_w_df["wavelength"].to_numpy(), a_w_df["a_w"].to_numpy(), wl)
    a0_int = interp1(a0_a1_df["wavelength"].to_numpy(), a0_a1_df["a0"].to_numpy(), wl)
    a1_int = interp1(a0_a1_df["wavelength"].to_numpy(), a0_a1_df["a1"].to_numpy(), wl)

    bb_w = 0.00111 * (wl / 500.0) ** (-4.32)

    classes = sorted(rb["class"].unique().tolist())
    mu_wide = rb.pivot_table(index="wavelength", columns="class", values="r_b_mean").sort_index()
    sd_wide = rb.pivot_table(index="wavelength", columns="class", values="r_b_sd").sort_index()
    rb_wl_master = mu_wide.index.to_numpy(dtype=float)

    mu_cols, sd_cols = [], []
    for cl in classes:
        mu_cols.append(interp1(rb_wl_master, mu_wide[cl].to_numpy(dtype=float), wl))
        sd_cols.append(interp1(rb_wl_master, sd_wide[cl].to_numpy(dtype=float), wl))

    r_b_mu_lib = np.column_stack(mu_cols).astype(float)
    r_b_sd_lib = np.maximum(np.column_stack(sd_cols).astype(float), float(sd_floor))

    return LUTs(
        a_w=a_w_int,
        a0=a0_int,
        a1=a1_int,
        bb_w=bb_w,
        r_b_mu_lib=r_b_mu_lib,
        r_b_sd_lib=r_b_sd_lib,
        classes=classes,
    )


def make_stan_data_base_from_csv(
    wavelength: Sequence[float],
    water_type: int,
    shallow: int,
    bottom_class_names: Sequence[str],
    *,
    a_nap_star: float,
    bb_p_star: float,
    a_g_s: float,
    a_nap_s: float,
    bb_p_gamma: float,
    a_w_csv: str,
    a0_a1_phyto_csv: str,
    r_b_gamache_csv: str,
    sd_floor: float = 1e-6,
) -> Dict[str, Any]:
    luts = make_luts_from_csv(
        wavelength=wavelength,
        a_w_csv=a_w_csv,
        a0_a1_phyto_csv=a0_a1_phyto_csv,
        r_b_gamache_csv=r_b_gamache_csv,
        sd_floor=sd_floor,
    )

    bottom_class_names = list(bottom_class_names)
    if len(bottom_class_names) != 3:
        raise ValueError("Stan model expects exactly 3 bottom classes.")

    name_to_id = {name: i for i, name in enumerate(luts.classes, start=1)}  # 1-based
    missing = [n for n in bottom_class_names if n not in name_to_id]
    if missing:
        raise ValueError("Bottom class not found in library: " + ", ".join(missing))
    bottom_class_ids = [int(name_to_id[n]) for n in bottom_class_names]

    wl = np.asarray(wavelength, dtype=float)
    return {
        "n_wl": int(wl.size),
        "wavelength": wl.tolist(),
        "a_w": luts.a_w.tolist(),
        "a0": luts.a0.tolist(),
        "a1": luts.a1.tolist(),
        "bb_w": luts.bb_w.tolist(),
        "n_class": int(len(luts.classes)),
        "r_b_mu_lib": luts.r_b_mu_lib.tolist(),  # matrix[n_wl, n_class]
        "r_b_sd_lib": luts.r_b_sd_lib.tolist(),  # matrix[n_wl, n_class]
        "bottom_class_ids": bottom_class_ids,     # int[3]
        "water_type": int(water_type),
        "shallow": int(shallow),
        "a_nap_star": float(a_nap_star),
        "bb_p_star": float(bb_p_star),
        "a_g_s": float(a_g_s),
        "a_nap_s": float(a_nap_s),
        "bb_p_gamma": float(bb_p_gamma),
    }