# #!/usr/bin/env python3
# from __future__ import annotations
#
# import os
# import sys
# import math
# import logging
# import tempfile
# import multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import Any, Dict, Optional, Sequence, Tuple
#
# import numpy as np
# import xarray as xr
# import netCDF4
# from pyproj import CRS, Transformer
#
# # -----------------------------------------------------------------------------
# # CmdStan globals (per worker)
# # -----------------------------------------------------------------------------
# _STAN_MODEL = None
# _BASE: dict[str, Any] | None = None
#
#
# # =============================================================================
# # Logging / silence
# # =============================================================================
# def silence_everything() -> None:
#     logging.disable(logging.ERROR)
#     for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
#         lg = logging.getLogger(name)
#         lg.setLevel(logging.CRITICAL)
#         lg.propagate = False
#     os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"
#     os.environ["CMDSTANPY_SILENT"] = "true"
#
#
# # =============================================================================
# # Small numerics helpers
# # =============================================================================
# def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)
#     xout = np.asarray(xout, float)
#
#     order = np.argsort(x)
#     x = x[order]
#     y = y[order]
#
#     yout = np.interp(xout, x, y)
#     yout = np.where(xout < x[0], y[0], yout)
#     yout = np.where(xout > x[-1], y[-1], yout)
#     return yout
#
#
# def load_optics_luts_from_csv(
#     wavelength: Sequence[float],
#     a_w_csv: str,
#     a0_a1_phyto_csv: str,
# ) -> Dict[str, np.ndarray]:
#     import pandas as pd
#
#     wl = np.asarray(wavelength, float)
#     if wl.ndim != 1 or wl.size < 2:
#         raise ValueError("wavelength must be 1D length>=2")
#
#     a_w_df = pd.read_csv(a_w_csv)
#     a01_df = pd.read_csv(a0_a1_phyto_csv)
#
#     for c in ("wavelength", "a_w"):
#         if c not in a_w_df.columns:
#             raise ValueError(f"{a_w_csv} missing column '{c}'")
#     for c in ("wavelength", "a0", "a1"):
#         if c not in a01_df.columns:
#             raise ValueError(f"{a0_a1_phyto_csv} missing column '{c}'")
#
#     a_w = interp1_flat(a_w_df["wavelength"].to_numpy(float), a_w_df["a_w"].to_numpy(float), wl)
#     a0 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a0"].to_numpy(float), wl)
#     a1 = interp1_flat(a01_df["wavelength"].to_numpy(float), a01_df["a1"].to_numpy(float), wl)
#
#     bb_w = 0.00111 * (wl / 500.0) ** (-4.32)
#     return {"a_w": a_w, "a0": a0, "a1": a1, "bb_w": bb_w}
#
#
# def compile_model(stan_model_path: str, user_header: str | None = None) -> str:
#     from cmdstanpy import CmdStanModel
#     m = CmdStanModel(stan_file=stan_model_path, user_header=user_header)
#     return m.exe_file
#
#
# # =============================================================================
# # CRS / bbox crop
# # =============================================================================
# def _get_crs_wkt_from_grid_mapping(ds: xr.Dataset, data_var: str) -> Optional[str]:
#     gm_name = ds[data_var].attrs.get("grid_mapping")
#     if not gm_name or gm_name not in ds.variables:
#         return None
#     gm = ds[gm_name]
#     return (
#         gm.attrs.get("spatial_ref")
#         or gm.attrs.get("crs_wkt")
#         or gm.attrs.get("WKT")
#         or gm.attrs.get("proj4text")
#     )
#
#
# def crop_dataset_bbox(ds: xr.Dataset, rho_var: str, bbox: Optional[dict]) -> xr.Dataset:
#     """
#     bbox formats supported:
#       - projected coords: {"x": (xmin, xmax), "y": (ymin, ymax)}
#       - lon/lat in EPSG:4326: {"lon": (lon0, lon1), "lat": (lat0, lat1)}
#
#     Uses ds["x"], ds["y"]. Handles descending y.
#     """
#     if bbox is None:
#         return ds
#
#     if "x" in bbox and "y" in bbox:
#         x0, x1 = bbox["x"]
#         y0, y1 = bbox["y"]
#     elif "lon" in bbox and "lat" in bbox:
#         wkt = _get_crs_wkt_from_grid_mapping(ds, rho_var)
#         if not wkt:
#             raise ValueError("bbox lon/lat provided but CRS WKT not found in grid_mapping.")
#         crs = CRS.from_wkt(wkt)
#         tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
#         x0, y0 = tr.transform(bbox["lon"][0], bbox["lat"][0])
#         x1, y1 = tr.transform(bbox["lon"][1], bbox["lat"][1])
#     else:
#         raise ValueError('bbox must have keys ("x","y") or ("lon","lat").')
#
#     xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
#     ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
#
#     # descending y axis support
#     y = ds["y"].values
#     y_desc = bool(y[0] > y[-1])
#     y_slice = slice(ymax, ymin) if y_desc else slice(ymin, ymax)
#
#     return ds.sel(x=slice(xmin, xmax), y=y_slice)
#
#
# # =============================================================================
# # Uncertainty estimation (pixel-by-pixel)
# # =============================================================================
# def _nan_gaussian_2d(img: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
#     """NaN-safe Gaussian smoothing via normalized convolution."""
#     from skimage.filters import gaussian
#
#     M = np.isfinite(img).astype(np.float32)
#     X = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
#
#     num = gaussian(X, sigma=sigma, truncate=truncate, preserve_range=True, channel_axis=None)
#     den = gaussian(M, sigma=sigma, truncate=truncate, preserve_range=True, channel_axis=None)
#     out = num / (den + 1e-12)
#     out[den < 1e-6] = np.nan
#     return out.astype(np.float32)
#
#
# def estimate_sigma_rho_local(
#     R: np.ndarray,                 # (nwl, ny, nx)
#     sigma_xy: float = 1.2,
#     truncate: float = 4.0,
#     rel_floor: float = 0.02,
#     abs_floor: float = 2e-5,
# ) -> np.ndarray:
#     """
#     Per-band local spatial variability + floors:
#       mu = G(R)
#       var = G((R-mu)^2)
#       sigma = sqrt(var) + rel_floor*|mu|
#     """
#     R = np.asarray(R, float)
#     nwl, ny, nx = R.shape
#     sig = np.full((nwl, ny, nx), np.nan, dtype=np.float32)
#
#     for k in range(nwl):
#         band = R[k]
#         mu = _nan_gaussian_2d(band, sigma=sigma_xy, truncate=truncate)
#         d2 = (band - mu) ** 2
#         var = _nan_gaussian_2d(d2, sigma=sigma_xy, truncate=truncate)
#         s = np.sqrt(np.maximum(var, 0.0))
#         s = s + rel_floor * np.abs(mu)
#         s = np.maximum(s, abs_floor)
#         sig[k] = s.astype(np.float32)
#
#     return sig
#
#
# # =============================================================================
# # Worker init: build Stan base for ETA model
# # =============================================================================
# def _init_worker(exe_file: str, wavelength: np.ndarray, cfg: dict) -> None:
#     global _STAN_MODEL, _BASE
#     silence_everything()
#
#     from cmdstanpy import CmdStanModel
#
#     eta_builder_import = cfg["eta_builder_import"]
#     mod_name, fn_name = eta_builder_import.rsplit(":", 1)
#     mod = __import__(mod_name, fromlist=[fn_name])
#     make_stan_data_eta_model = getattr(mod, fn_name)
#
#     _STAN_MODEL = CmdStanModel(exe_file=exe_file)
#
#     wl = np.asarray(wavelength, float)
#     optics = load_optics_luts_from_csv(wl, a_w_csv=cfg["a_w_csv"], a0_a1_phyto_csv=cfg["a0_a1_phyto_csv"])
#
#     dummy_rrs = np.zeros_like(wl, dtype=float)
#     dummy_sig = np.full_like(wl, 1e-6, dtype=float)
#
#     priors = cfg["priors"]
#
#     base = make_stan_data_eta_model(
#         wavelength=wl,
#         rrs_obs=dummy_rrs,
#         sigma_rrs=dummy_sig,
#         a_w=optics["a_w"],
#         a0=optics["a0"],
#         a1=optics["a1"],
#         bb_w=optics["bb_w"],
#         rb_obs_path=cfg["rb_obs_path"],
#         q=cfg["q"],
#         eps=cfg["eps"],
#         cov_jitter=cfg["cov_jitter"],
#         sigma2_floor=cfg["sigma2_floor"],
#         water_type=cfg["water_type"],
#         theta_sun=30.0,
#         theta_view=0.0,
#         shallow=cfg["shallow"],
#         a_nap_star_mu=float(priors["a_nap_star_mu"]),
#         a_nap_star_sd=float(priors["a_nap_star_sd"]),
#         bb_p_star_mu=float(priors["bb_p_star_mu"]),
#         bb_p_star_sd=float(priors["bb_p_star_sd"]),
#         a_g_s_mu=float(priors["a_g_s_mu"]),
#         a_g_s_sd=float(priors["a_g_s_sd"]),
#         a_nap_s_mu=float(priors["a_nap_s_mu"]),
#         a_nap_s_sd=float(priors["a_nap_s_sd"]),
#         bb_p_gamma_mu=float(priors["bb_p_gamma_mu"]),
#         bb_p_gamma_sd=float(priors["bb_p_gamma_sd"]),
#         h_w_mu=float(priors["h_w_mu"]),
#         h_w_sd=float(priors["h_w_sd"]),
#     )
#
#     required = ("eta_U", "eta_mu_k", "eta_L_k", "eta_sigma2_k", "eta_pi")
#     miss = [k for k in required if k not in base]
#     if miss:
#         raise RuntimeError(f"ETA base missing keys: {miss}")
#
#     _BASE = base
#
#
# # =============================================================================
# # Inversion: init + one pixel
# # =============================================================================
# def make_init(stan_data: dict) -> dict:
#     eta_mu = np.array(np.mean(stan_data["eta_mu_k"], axis=0), dtype=float)
#     return dict(
#         chl=0.9,
#         a_g_440=0.25,
#         spm=2.0,
#         a_nap_star=float(stan_data["a_nap_star_mu"]),
#         bb_p_star=float(stan_data["bb_p_star_mu"]),
#         a_g_s=float(stan_data["a_g_s_mu"]),
#         a_nap_s=float(stan_data["a_nap_s_mu"]),
#         bb_p_gamma=float(stan_data["bb_p_gamma_mu"]),
#         h_w=float(np.clip(stan_data["h_w_mu"], 0.0, 30.0)),
#         eta_b=eta_mu.tolist(),
#         sigma_model=5e-5,
#     )
#
#
# def _rho_to_rrs0m_and_sigma(rho: np.ndarray, sigma_rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     rho = np.asarray(rho, float)
#     sigma_rho = np.asarray(sigma_rho, float)
#
#     rrs0p = rho / math.pi
#     denom = 0.52 + 1.7 * rrs0p
#     rrs0m = rrs0p / denom
#
#     sigma_rrs0p = sigma_rho / math.pi
#     d = 0.52 / (denom ** 2)
#     sigma_rrs0m = np.abs(d) * sigma_rrs0p
#     sigma_rrs0m = np.maximum(sigma_rrs0m, 1e-8)
#
#     return rrs0m, sigma_rrs0m
#
#
# def invert_one_pixel(job: dict, out_vars: tuple[str, ...], save_spectra: bool) -> dict:
#     global _STAN_MODEL, _BASE
#     assert _STAN_MODEL is not None
#     assert _BASE is not None
#
#     iy = int(job["iy"])
#     ix = int(job["ix"])
#
#     wl = job["wl_nm"].astype(float)
#     rho = job["rho"].astype(float)
#     sig_rho = job["sigma_rho"].astype(float)
#
#     if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(sig_rho)):
#         return {"ok": False, "iy": iy, "ix": ix, "err": "non-finite input"}
#
#     rrs_0m, sigma_rrs = _rho_to_rrs0m_and_sigma(rho, sig_rho)
#     if not np.all(np.isfinite(rrs_0m)) or not np.all(np.isfinite(sigma_rrs)):
#         return {"ok": False, "iy": iy, "ix": ix, "err": "non-finite rrs/sigma"}
#
#     stan_data = dict(_BASE)
#     stan_data["rrs_obs"] = rrs_0m.tolist()
#     stan_data["sigma_rrs"] = sigma_rrs.tolist()
#     stan_data["theta_sun"] = float(job["theta_sun"])
#     stan_data["theta_view"] = float(job["theta_view"])
#     stan_data["h_w_mu"] = float(job["h_w_mu"])
#     stan_data["h_w_sd"] = float(job["h_w_sd"])
#
#     pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")
#
#     try:
#         fit = _STAN_MODEL.optimize(
#             data=stan_data,
#             seed=int(job["seed"]),
#             output_dir=pid_dir,
#             show_console=False,
#             algorithm="lbfgs",
#             inits=make_init(stan_data),
#             iter=int(job.get("iter", 20000)),
#             require_converged=True,
#         )
#         est = fit.optimized_params_dict
#         scalars = {v: float(est[v]) for v in out_vars if v in est}
#         out = {"ok": True, "iy": iy, "ix": ix, "scalars": scalars}
#
#         if save_spectra:
#             nwl = len(wl)
#             r_b_hat = np.full(nwl, np.nan, dtype=float)
#             rrs_hat = np.full(nwl, np.nan, dtype=float)
#
#             for i in range(nwl):
#                 kb = f"r_b_hat[{i + 1}]"
#                 kr = f"rrs_hat[{i + 1}]"
#                 if kb in est:
#                     r_b_hat[i] = float(est[kb])
#                 if kr in est:
#                     rrs_hat[i] = float(est[kr])
#
#             rho_hat = np.full(nwl, np.nan, dtype=float)
#             ok = np.isfinite(rrs_hat) & (1 - 1.7 * rrs_hat > 1e-12)
#             rrs_0p_hat = np.full(nwl, np.nan, dtype=float)
#             rrs_0p_hat[ok] = (0.52 * rrs_hat[ok]) / (1 - 1.7 * rrs_hat[ok])
#             rho_hat[ok] = rrs_0p_hat[ok] * math.pi
#
#             r_b_hat = np.clip(np.nan_to_num(r_b_hat, nan=0.0), 0.0, 1.0)
#
#             out["r_b_hat"] = r_b_hat.astype(np.float32)
#             out["rho_hat"] = rho_hat.astype(np.float32)
#
#         return out
#
#     except Exception as e:
#         return {"ok": False, "iy": iy, "ix": ix, "err": repr(e)}
#
#
# # =============================================================================
# # NetCDF output helpers
# # =============================================================================
# def _create_out_nc(
#     out_path: str,
#     ds: xr.Dataset,
#     rho_var: str,
#     wl_dim: str,
#     y_dim: str,
#     x_dim: str,
#     save_spectra: bool,
#     out_vars: tuple[str, ...],
# ) -> None:
#     if os.path.exists(out_path):
#         os.remove(out_path)
#
#     with netCDF4.Dataset(out_path, "w") as nc:
#         nc.createDimension(wl_dim, ds.sizes[wl_dim])
#         nc.createDimension(y_dim, ds.sizes[y_dim])
#         nc.createDimension(x_dim, ds.sizes[x_dim])
#
#         # coords
#         for d in (wl_dim, y_dim, x_dim):
#             v = nc.createVariable(d, "f4", (d,), zlib=True, complevel=1)
#             v[:] = ds[d].values.astype(np.float32)
#             for k, val in ds[d].attrs.items():
#                 try:
#                     setattr(v, k, val)
#                 except Exception:
#                     pass
#
#         # copy grid mapping var if any (CF)
#         gm_name = ds[rho_var].attrs.get("grid_mapping")
#         if gm_name and gm_name in ds.variables:
#             gm_in = ds[gm_name]
#             gm = nc.createVariable(gm_name, "i4", ())
#             for k, val in gm_in.attrs.items():
#                 try:
#                     setattr(gm, k, val)
#                 except Exception:
#                     pass
#
#         # scalar outputs
#         for name in out_vars:
#             v = nc.createVariable(name, "f4", (y_dim, x_dim), zlib=True, complevel=1, fill_value=np.nan)
#             v.long_name = name
#
#         v_ok = nc.createVariable("inv_ok", "i1", (y_dim, x_dim), zlib=True, complevel=1, fill_value=0)
#         v_ok.long_name = "inversion success flag (1=ok, 0=failed)"
#
#         if save_spectra:
#             v_rb = nc.createVariable("r_b_hat", "f4", (wl_dim, y_dim, x_dim), zlib=True, complevel=1, fill_value=np.nan)
#             v_rb.long_name = "retrieved bottom reflectance (0..1)"
#             v_rh = nc.createVariable("rho_hat", "f4", (wl_dim, y_dim, x_dim), zlib=True, complevel=1, fill_value=np.nan)
#             v_rh.long_name = "model-predicted rho"
#
#         for k, val in ds.attrs.items():
#             try:
#                 setattr(nc, k, val)
#             except Exception:
#                 pass
#
#         prev = getattr(nc, "history", "")
#         nc.history = (prev + "\n" if prev else "") + "pixel inversion written by invert_pixelwise.py"
#
#
# def _write_tile_to_out_nc(
#     out_path: str,
#     y0: int,
#     y1: int,
#     x0: int,
#     x1: int,
#     scal_maps: dict[str, np.ndarray],   # (ty,tx)
#     ok_map: np.ndarray,                # (ty,tx)
#     rb_cube: Optional[np.ndarray],      # (nwl,ty,tx)
#     rho_hat_cube: Optional[np.ndarray], # (nwl,ty,tx)
# ) -> None:
#     with netCDF4.Dataset(out_path, "a") as nc:
#         for k, arr in scal_maps.items():
#             nc.variables[k][y0:y1, x0:x1] = arr.astype(np.float32)
#         nc.variables["inv_ok"][y0:y1, x0:x1] = ok_map.astype(np.int8)
#
#         if rb_cube is not None:
#             nc.variables["r_b_hat"][:, y0:y1, x0:x1] = rb_cube.astype(np.float32)
#         if rho_hat_cube is not None:
#             nc.variables["rho_hat"][:, y0:y1, x0:x1] = rho_hat_cube.astype(np.float32)
#
#
# # =============================================================================
# # Main pixelwise driver
# # =============================================================================
# def invert_netcdf_pixelwise(
#     in_nc: str,
#     out_nc: str,
#     *,
#     bbox: Optional[dict] = None,                 # {"lon":(lon0,lon1),"lat":(lat0,lat1)} or {"x":(...),"y":(...)}
#     rho_var: str = "rho_w_g21",                  # (wl,y,x)
#     wl_range: tuple[float, float] = (370, 730),
#     tile_yx: tuple[int, int] = (48, 48),
#     n_workers: Optional[int] = None,
#     save_spectra: bool = True,
#     # uncertainty estimation
#     unc_sigma_xy: float = 1.2,
#     unc_rel_floor: float = 0.02,
#     unc_abs_floor: float = 2e-5,
#     # per-pixel auxiliaries
#     sun_var: str = "sun_zenith",
#     view_var: str = "view_zenith",
#     depth_var: str = "bathymetry_nonna10",
#     max_depth: float = 10.0,
#     h_w_sd_override: Optional[float] = 1.0,
#     # Stan stuff
#     stan_model_path: str = "",
#     user_header: Optional[str] = None,
#     eta_builder_import: str = "reverie.correction.aquatic.stan_data_gmm:make_stan_data_eta_model",
#     a_w_csv: str = "",
#     a0_a1_phyto_csv: str = "",
#     rb_obs_path: str = "",
#     priors: Optional[dict] = None,
#     water_type: int = 2,
#     shallow: int = 1,
#     q: int = 8,
#     eps: float = 1e-6,
#     cov_jitter: float = 1e-6,
#     sigma2_floor: float = 1e-6,
#     out_vars: tuple[str, ...] = ("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
#     iter_opt: int = 20000,
# ) -> None:
#     if priors is None:
#         priors = dict(
#             a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
#             bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
#             a_g_s_mu=0.017,       a_g_s_sd=0.0012,
#             a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
#             bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
#             h_w_mu=3.0,           h_w_sd=2.0,
#         )
#
#     if n_workers is None:
#         n_workers = max(1, (os.cpu_count() or 2) - 1)
#
#     lo, hi = wl_range
#     if lo > hi:
#         lo, hi = hi, lo
#
#     ctx = mp.get_context("spawn")
#     ty, tx = tile_yx
#     seed0 = 1234
#
#     with xr.open_dataset(in_nc, engine="netcdf4") as ds0:
#         if rho_var not in ds0:
#             raise KeyError(f"{rho_var} not found in {in_nc}")
#
#         rho0 = ds0[rho_var]
#         if rho0.ndim != 3:
#             raise ValueError(f"{rho_var} must be (wl,y,x). got dims={rho0.dims}")
#
#         wl_dim, y_dim, x_dim = rho0.dims
#
#         ds = ds0.sel({wl_dim: slice(lo, hi)})
#         ds = crop_dataset_bbox(ds, rho_var=rho_var, bbox=bbox)
#
#         rho_da = ds[rho_var]
#         wl = rho_da[wl_dim].values.astype(float)
#         ny = rho_da.sizes[y_dim]
#         nx = rho_da.sizes[x_dim]
#         nwl = rho_da.sizes[wl_dim]
#
#         sun = ds[sun_var].values.astype(np.float32) if sun_var in ds else None
#         view = ds[view_var].values.astype(np.float32) if view_var in ds else None
#         depth = ds[depth_var].values.astype(np.float32) if depth_var in ds else None
#
#         _create_out_nc(out_nc, ds, rho_var, wl_dim, y_dim, x_dim, save_spectra, out_vars)
#
#         exe_file = compile_model(stan_model_path, user_header=user_header)
#
#         cfg = dict(
#             water_type=water_type,
#             shallow=shallow,
#             a_w_csv=a_w_csv,
#             a0_a1_phyto_csv=a0_a1_phyto_csv,
#             rb_obs_path=rb_obs_path,
#             q=q,
#             eps=eps,
#             cov_jitter=cov_jitter,
#             sigma2_floor=sigma2_floor,
#             priors=priors,
#             eta_builder_import=eta_builder_import,
#         )
#
#         with ProcessPoolExecutor(
#             max_workers=n_workers,
#             mp_context=ctx,
#             initializer=_init_worker,
#             initargs=(exe_file, wl, cfg),
#         ) as ex:
#
#             for y0 in range(0, ny, ty):
#                 y1 = min(ny, y0 + ty)
#                 for x0 in range(0, nx, tx):
#                     x1 = min(nx, x0 + tx)
#
#                     R = rho_da.isel({y_dim: slice(y0, y1), x_dim: slice(x0, x1)}).values.astype(np.float32)
#
#                     valid = np.all(np.isfinite(R), axis=0)
#                     valid &= np.all(R >= 0, axis=0)
#
#                     if depth is not None:
#                         d = depth[y0:y1, x0:x1]
#                         valid &= np.isfinite(d) & (d <= max_depth)
#                     if view is not None:
#                         vv = view[y0:y1, x0:x1]
#                         valid &= np.isfinite(vv)
#                     if sun is not None:
#                         ss = sun[y0:y1, x0:x1]
#                         valid &= np.isfinite(ss)
#
#                     if not np.any(valid):
#                         ok_map = np.zeros((y1 - y0, x1 - x0), dtype=np.int8)
#                         scal_maps = {k: np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32) for k in out_vars}
#                         _write_tile_to_out_nc(out_nc, y0, y1, x0, x1, scal_maps, ok_map, None, None)
#                         continue
#
#                     Sig = estimate_sigma_rho_local(
#                         R,
#                         sigma_xy=unc_sigma_xy,
#                         rel_floor=unc_rel_floor,
#                         abs_floor=unc_abs_floor,
#                     )
#
#                     scal_maps = {k: np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32) for k in out_vars}
#                     ok_map = np.zeros((y1 - y0, x1 - x0), dtype=np.int8)
#
#                     rb_cube = np.full((nwl, y1 - y0, x1 - x0), np.nan, dtype=np.float32) if save_spectra else None
#                     rho_hat_cube = np.full((nwl, y1 - y0, x1 - x0), np.nan, dtype=np.float32) if save_spectra else None
#
#                     futs = []
#                     for iy in range(y0, y1):
#                         for ix in range(x0, x1):
#                             ly = iy - y0
#                             lx = ix - x0
#                             if not valid[ly, lx]:
#                                 continue
#
#                             theta_sun = float(sun[iy, ix]) if sun is not None else 30.0
#                             theta_view = float(view[iy, ix]) if view is not None else 0.0
#                             h_w_mu = float(depth[iy, ix]) if depth is not None else float(priors["h_w_mu"])
#                             h_w_sd = float(h_w_sd_override) if h_w_sd_override is not None else float(priors["h_w_sd"])
#
#                             job = dict(
#                                 iy=iy,
#                                 ix=ix,
#                                 wl_nm=wl,
#                                 rho=R[:, ly, lx],
#                                 sigma_rho=Sig[:, ly, lx],
#                                 theta_sun=theta_sun,
#                                 theta_view=theta_view,
#                                 h_w_mu=h_w_mu,
#                                 h_w_sd=h_w_sd,
#                                 seed=seed0 + (iy * 1000003 + ix),
#                                 iter=iter_opt,
#                             )
#                             futs.append(ex.submit(invert_one_pixel, job, out_vars, save_spectra))
#
#                     for fut in as_completed(futs):
#                         r = fut.result()
#                         iy = int(r["iy"])
#                         ix = int(r["ix"])
#                         ly = iy - y0
#                         lx = ix - x0
#
#                         if not r.get("ok", False):
#                             continue
#
#                         ok_map[ly, lx] = 1
#                         scal = r["scalars"]
#                         for k in out_vars:
#                             if k in scal and np.isfinite(scal[k]):
#                                 scal_maps[k][ly, lx] = float(scal[k])
#
#                         if save_spectra:
#                             rb_cube[:, ly, lx] = r["r_b_hat"]
#                             rho_hat_cube[:, ly, lx] = r["rho_hat"]
#
#                     _write_tile_to_out_nc(out_nc, y0, y1, x0, x1, scal_maps, ok_map, rb_cube, rho_hat_cube)
#                     print(f"[tile] y={y0}:{y1} x={x0}:{x1} ok={int(ok_map.sum())}/{ok_map.size}", file=sys.stderr)
#
#
# # =============================================================================
# # Run
# # =============================================================================
# if __name__ == "__main__":
#     in_nc = "/D/Data/WISE/ACI-12A/el_sma/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc"
#     out_nc = "/D/Data/WISE/ACI-12A/el_sma/220705_ACI-12A-WI-1x1x1_v01-l2rg_inverted_pixelwise_bbox.nc"
#
#     # reduced for optimization testing
#     # bbox = {"lon": (-64.37024, -64.35578), "lat": (49.77804, 49.78492)}
#     # Ultra reduce for sampling testing
#     bbox = {"lon": (-64.37031, -64.36776), "lat": (49.778227, 49.780921)}
#     # For publication
#     # bbox = {"lon": (-64.3771, -64.35086), "lat": (49.77186, 49.79783)}
#
#     invert_netcdf_pixelwise(
#         in_nc=in_nc,
#         out_nc=out_nc,
#         bbox=bbox,
#         rho_var="rho_w_g21",
#         wl_range=(370, 730),
#         tile_yx=(48, 48),
#         n_workers=None,
#         save_spectra=True,
#
#         unc_sigma_xy=1.2,
#         unc_rel_floor=0.02,
#         unc_abs_floor=2e-5,
#
#         sun_var="sun_zenith",
#         view_var="view_zenith",
#         depth_var="bathymetry_nonna10",
#         max_depth=10,
#         h_w_sd_override=1.0,
#
#         stan_model_path="/home/raphael/R/SABER/inst/stan/model_gmm.stan",
#         user_header="/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp",
#         eta_builder_import="reverie.correction.aquatic.stan_data_gmm:make_stan_data_eta_model",
#         a_w_csv="/home/raphael/R/SABER/inst/extdata/a_w.csv",
#         a0_a1_phyto_csv="/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv",
#         rb_obs_path="/home/raphael/R/SABER/inst/extdata/r_b_gamache_obs.csv",
#
#         priors=dict(
#             a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
#             bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
#             a_g_s_mu=0.017,       a_g_s_sd=0.0012,
#             a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
#             bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
#             h_w_mu=3.0,           h_w_sd=2.0,
#         ),
#
#         water_type=2,
#         shallow=1,
#         q=8,
#         eps=1e-6,
#         cov_jitter=1e-6,
#         sigma2_floor=1e-6,
#
#         out_vars=("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
#         iter_opt=20000,
#     )
#
#     print("Wrote:", out_nc)

#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import math
import logging
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import netCDF4
from pyproj import CRS, Transformer

# -----------------------------------------------------------------------------
# CmdStan globals (per worker)
# -----------------------------------------------------------------------------
_STAN_MODEL = None
_BASE: dict[str, Any] | None = None


# =============================================================================
# Logging / silence
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
    import pandas as pd

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
# CRS / bbox crop
# =============================================================================
def _get_crs_wkt_from_grid_mapping(ds: xr.Dataset, data_var: str) -> Optional[str]:
    gm_name = ds[data_var].attrs.get("grid_mapping")
    if not gm_name or gm_name not in ds.variables:
        return None
    gm = ds[gm_name]
    return (
        gm.attrs.get("spatial_ref")
        or gm.attrs.get("crs_wkt")
        or gm.attrs.get("WKT")
        or gm.attrs.get("proj4text")
    )


def crop_dataset_bbox(ds: xr.Dataset, rho_var: str, bbox: Optional[dict]) -> xr.Dataset:
    """
    bbox formats supported:
      - projected coords: {"x": (xmin, xmax), "y": (ymin, ymax)}
      - lon/lat in EPSG:4326: {"lon": (lon0, lon1), "lat": (lat0, lat1)}
    Uses ds["x"], ds["y"]. Handles descending y.
    """
    if bbox is None:
        return ds

    if "x" in bbox and "y" in bbox:
        x0, x1 = bbox["x"]
        y0, y1 = bbox["y"]
    elif "lon" in bbox and "lat" in bbox:
        wkt = _get_crs_wkt_from_grid_mapping(ds, rho_var)
        if not wkt:
            raise ValueError("bbox lon/lat provided but CRS WKT not found in grid_mapping.")
        crs = CRS.from_wkt(wkt)
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x0, y0 = tr.transform(bbox["lon"][0], bbox["lat"][0])
        x1, y1 = tr.transform(bbox["lon"][1], bbox["lat"][1])
    else:
        raise ValueError('bbox must have keys ("x","y") or ("lon","lat").')

    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)

    y = ds["y"].values
    y_desc = bool(y[0] > y[-1])
    y_slice = slice(ymax, ymin) if y_desc else slice(ymin, ymax)

    return ds.sel(x=slice(xmin, xmax), y=y_slice)


# =============================================================================
# Uncertainty estimation (pixel-by-pixel)
# =============================================================================
def _nan_gaussian_2d(img: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """NaN-safe Gaussian smoothing via normalized convolution."""
    from skimage.filters import gaussian

    M = np.isfinite(img).astype(np.float32)
    X = np.where(np.isfinite(img), img, 0.0).astype(np.float32)

    num = gaussian(X, sigma=sigma, truncate=truncate, preserve_range=True, channel_axis=None)
    den = gaussian(M, sigma=sigma, truncate=truncate, preserve_range=True, channel_axis=None)
    out = num / (den + 1e-12)
    out[den < 1e-6] = np.nan
    return out.astype(np.float32)


def estimate_sigma_rho_local(
    R: np.ndarray,                 # (nwl, ny, nx)
    sigma_xy: float = 1.2,
    truncate: float = 4.0,
    rel_floor: float = 0.02,
    abs_floor: float = 2e-5,
) -> np.ndarray:
    """
    Per-band local spatial variability + floors:
      mu = G(R)
      var = G((R-mu)^2)
      sigma = sqrt(var) + rel_floor*|mu|
    """
    R = np.asarray(R, float)
    nwl, ny, nx = R.shape
    sig = np.full((nwl, ny, nx), np.nan, dtype=np.float32)

    for k in range(nwl):
        band = R[k]
        mu = _nan_gaussian_2d(band, sigma=sigma_xy, truncate=truncate)
        d2 = (band - mu) ** 2
        var = _nan_gaussian_2d(d2, sigma=sigma_xy, truncate=truncate)
        s = np.sqrt(np.maximum(var, 0.0))
        s = s + rel_floor * np.abs(mu)
        s = np.maximum(s, abs_floor)
        sig[k] = s.astype(np.float32)

    return sig


# =============================================================================
# Worker init: build Stan base for ETA model
# =============================================================================
def _init_worker(exe_file: str, wavelength: np.ndarray, cfg: dict) -> None:
    global _STAN_MODEL, _BASE
    silence_everything()

    from cmdstanpy import CmdStanModel

    eta_builder_import = cfg["eta_builder_import"]
    mod_name, fn_name = eta_builder_import.rsplit(":", 1)
    mod = __import__(mod_name, fromlist=[fn_name])
    make_stan_data_eta_model = getattr(mod, fn_name)

    _STAN_MODEL = CmdStanModel(exe_file=exe_file)

    wl = np.asarray(wavelength, float)
    optics = load_optics_luts_from_csv(wl, a_w_csv=cfg["a_w_csv"], a0_a1_phyto_csv=cfg["a0_a1_phyto_csv"])

    dummy_rrs = np.zeros_like(wl, dtype=float)
    dummy_sig = np.full_like(wl, 1e-6, dtype=float)

    priors = cfg["priors"]

    base = make_stan_data_eta_model(
        wavelength=wl,
        rrs_obs=dummy_rrs,
        sigma_rrs=dummy_sig,
        a_w=optics["a_w"],
        a0=optics["a0"],
        a1=optics["a1"],
        bb_w=optics["bb_w"],
        rb_obs_path=cfg["rb_obs_path"],
        q=cfg["q"],
        eps=cfg["eps"],
        cov_jitter=cfg["cov_jitter"],
        sigma2_floor=cfg["sigma2_floor"],
        water_type=cfg["water_type"],
        theta_sun=30.0,
        theta_view=0.0,
        shallow=cfg["shallow"],
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

    required = ("eta_U", "eta_mu_k", "eta_L_k", "eta_sigma2_k", "eta_pi")
    miss = [k for k in required if k not in base]
    if miss:
        raise RuntimeError(f"ETA base missing keys: {miss}")

    _BASE = base


# =============================================================================
# Inversion: init + one pixel
# =============================================================================
def make_init(stan_data: dict) -> dict:
    eta_mu = np.array(np.mean(stan_data["eta_mu_k"], axis=0), dtype=float)
    return dict(
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


def _rho_to_rrs0m_and_sigma(rho: np.ndarray, sigma_rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, float)
    sigma_rho = np.asarray(sigma_rho, float)

    rrs0p = rho / math.pi
    denom = 0.52 + 1.7 * rrs0p
    rrs0m = rrs0p / denom

    sigma_rrs0p = sigma_rho / math.pi
    d = 0.52 / (denom ** 2)
    sigma_rrs0m = np.abs(d) * sigma_rrs0p
    sigma_rrs0m = np.maximum(sigma_rrs0m, 1e-8)

    return rrs0m, sigma_rrs0m


def invert_one_pixel(job: dict, out_vars: tuple[str, ...], save_spectra: bool) -> dict:
    global _STAN_MODEL, _BASE
    assert _STAN_MODEL is not None
    assert _BASE is not None

    iy = int(job["iy"])
    ix = int(job["ix"])

    wl = job["wl_nm"].astype(float)
    rho = job["rho"].astype(float)
    sig_rho = job["sigma_rho"].astype(float)

    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(sig_rho)):
        return {"ok": False, "iy": iy, "ix": ix, "err": "non-finite input"}

    rrs_0m, sigma_rrs = _rho_to_rrs0m_and_sigma(rho, sig_rho)
    if not np.all(np.isfinite(rrs_0m)) or not np.all(np.isfinite(sigma_rrs)):
        return {"ok": False, "iy": iy, "ix": ix, "err": "non-finite rrs/sigma"}

    stan_data = dict(_BASE)
    stan_data["rrs_obs"] = rrs_0m.tolist()
    stan_data["sigma_rrs"] = sigma_rrs.tolist()
    stan_data["theta_sun"] = float(job["theta_sun"])
    stan_data["theta_view"] = float(job["theta_view"])
    stan_data["h_w_mu"] = float(job["h_w_mu"])
    stan_data["h_w_sd"] = float(job["h_w_sd"])

    pid_dir = tempfile.mkdtemp(prefix="cmdstan_", dir="/tmp")

    try:
        fit = _STAN_MODEL.optimize(
            data=stan_data,
            seed=int(job["seed"]),
            output_dir=pid_dir,
            show_console=False,
            algorithm="lbfgs",
            inits=make_init(stan_data),
            iter=int(job.get("iter", 20000)),
            require_converged=True,
        )
        est = fit.optimized_params_dict
        scalars = {v: float(est[v]) for v in out_vars if v in est}
        out = {"ok": True, "iy": iy, "ix": ix, "scalars": scalars}

        if save_spectra:
            nwl = len(wl)
            r_b_hat = np.full(nwl, np.nan, dtype=float)
            rrs_hat = np.full(nwl, np.nan, dtype=float)

            for i in range(nwl):
                kb = f"r_b_hat[{i + 1}]"
                kr = f"rrs_hat[{i + 1}]"
                if kb in est:
                    r_b_hat[i] = float(est[kb])
                if kr in est:
                    rrs_hat[i] = float(est[kr])

            rho_hat = np.full(nwl, np.nan, dtype=float)
            ok = np.isfinite(rrs_hat) & (1 - 1.7 * rrs_hat > 1e-12)
            rrs_0p_hat = np.full(nwl, np.nan, dtype=float)
            rrs_0p_hat[ok] = (0.52 * rrs_hat[ok]) / (1 - 1.7 * rrs_hat[ok])
            rho_hat[ok] = rrs_0p_hat[ok] * math.pi

            r_b_hat = np.clip(np.nan_to_num(r_b_hat, nan=0.0), 0.0, 1.0)

            out["r_b_hat"] = r_b_hat.astype(np.float32)
            out["rho_hat"] = rho_hat.astype(np.float32)

        return out

    except Exception as e:
        return {"ok": False, "iy": iy, "ix": ix, "err": repr(e)}


# =============================================================================
# CF/GDAL helpers
# =============================================================================
def _infer_geotransform_from_xy(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float, float, float, float, float], bool]:
    """
    Given 1D pixel-center coords x, y, return GDAL GeoTransform:
      (x_ul, dx, 0, y_ul, 0, dy)
    where (x_ul,y_ul) is upper-left corner of upper-left pixel.
    Returns (gt, y_desc) where y_desc=True if y is descending (north-up typical).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least 2 x and 2 y values to infer geotransform")

    dx = float(np.median(np.diff(x)))
    dy_raw = float(np.median(np.diff(y)))
    y_desc = dy_raw < 0

    # Use magnitude for pixel size in y
    dy = -abs(dy_raw) if y_desc else abs(dy_raw)

    x_ul = float(x[0] - dx / 2.0)
    # upper-left corner depends on y direction
    if y_desc:
        y_ul = float(y[0] + abs(dy) / 2.0)
    else:
        y_ul = float(y[0] - abs(dy) / 2.0)

    gt = (x_ul, dx, 0.0, y_ul, 0.0, dy)
    return gt, y_desc


def _copy_cf_grid_mapping(nc: netCDF4.Dataset, ds: xr.Dataset, rho_var: str) -> str:
    """
    Copy the input CF grid mapping variable, and ensure it contains
    GDAL-friendly fields (spatial_ref/crs_wkt + GeoTransform).
    Returns grid mapping variable name.
    """
    gm_name = ds[rho_var].attrs.get("grid_mapping")
    if not gm_name or gm_name not in ds.variables:
        # No grid mapping in input -> best effort: create one from proj WKT if present in attrs
        raise ValueError(f"No CF grid_mapping found on {rho_var}; cannot ensure GDAL compliance.")

    gm_in = ds[gm_name]
    if gm_name in nc.variables:
        gm = nc.variables[gm_name]
    else:
        gm = nc.createVariable(gm_name, "i4", ())

    # copy attrs
    for k, val in gm_in.attrs.items():
        try:
            setattr(gm, k, val)
        except Exception:
            pass

    # Ensure WKT fields exist (GDAL reads these)
    wkt = (
        gm_in.attrs.get("spatial_ref")
        or gm_in.attrs.get("crs_wkt")
        or gm_in.attrs.get("WKT")
        or gm_in.attrs.get("proj4text")
    )
    if wkt:
        try:
            setattr(gm, "spatial_ref", wkt)
        except Exception:
            pass
        try:
            setattr(gm, "crs_wkt", wkt)
        except Exception:
            pass

    return gm_name


# =============================================================================
# NetCDF output helpers (CF + GDAL compliant)
# =============================================================================
def _create_out_nc(
    out_path: str,
    ds: xr.Dataset,
    rho_var: str,
    wl_dim: str,
    y_dim: str,
    x_dim: str,
    save_spectra: bool,
    out_vars: tuple[str, ...],
) -> None:
    if os.path.exists(out_path):
        os.remove(out_path)

    x = ds["x"].values.astype(float)
    y = ds["y"].values.astype(float)
    gt, y_desc = _infer_geotransform_from_xy(x, y)

    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as nc:
        # Dimensions
        nc.createDimension(wl_dim, ds.sizes[wl_dim])
        nc.createDimension(y_dim, ds.sizes[y_dim])
        nc.createDimension(x_dim, ds.sizes[x_dim])

        # Global CF conventions + GDAL geotransform
        nc.Conventions = "CF-1.8"
        # GDAL accepts GeoTransform as string of 6 numbers
        nc.GeoTransform = " ".join(f"{v:.15g}" for v in gt)

        # Coordinate variables
        vwl = nc.createVariable(wl_dim, "f4", (wl_dim,), zlib=True, complevel=1)
        vwl[:] = ds[wl_dim].values.astype(np.float32)
        # keep any wl attrs
        for k, val in ds[wl_dim].attrs.items():
            try:
                setattr(vwl, k, val)
            except Exception:
                pass
        # make it more CF-ish if absent
        if not hasattr(vwl, "standard_name"):
            vwl.standard_name = "radiation_wavelength"
        if not hasattr(vwl, "units"):
            vwl.units = "nm"

        vx = nc.createVariable(x_dim, "f8", (x_dim,), zlib=True, complevel=1)
        vx[:] = x.astype(np.float64)
        vx.standard_name = "projection_x_coordinate"
        vx.long_name = "x coordinate of projection"
        vx.units = ds["x"].attrs.get("units", "m")
        vx.axis = "X"

        vy = nc.createVariable(y_dim, "f8", (y_dim,), zlib=True, complevel=1)
        vy[:] = y.astype(np.float64)
        vy.standard_name = "projection_y_coordinate"
        vy.long_name = "y coordinate of projection"
        vy.units = ds["y"].attrs.get("units", "m")
        vy.axis = "Y"

        # Copy / ensure CF grid mapping variable + add GeoTransform to it too (many tools like this)
        gm_name = _copy_cf_grid_mapping(nc, ds, rho_var)
        try:
            setattr(nc.variables[gm_name], "GeoTransform", " ".join(f"{v:.15g}" for v in gt))
        except Exception:
            pass

        # Helper to create CF-ish output vars
        def _mk_scalar(name: str, units: str = ""):
            v = nc.createVariable(name, "f4", (y_dim, x_dim), zlib=True, complevel=1, fill_value=np.nan)
            v.long_name = name
            if units:
                v.units = units
            v.grid_mapping = gm_name
            v.coordinates = f"{y_dim} {x_dim}"
            return v

        # Scalar outputs
        for name in out_vars:
            _mk_scalar(name)

        v_ok = nc.createVariable("inv_ok", "i1", (y_dim, x_dim), zlib=True, complevel=1, fill_value=0)
        v_ok.long_name = "inversion success flag (1=ok, 0=failed)"
        v_ok.grid_mapping = gm_name
        v_ok.coordinates = f"{y_dim} {x_dim}"

        # Optional spectral outputs
        if save_spectra:
            v_rb = nc.createVariable("r_b_hat", "f4", (wl_dim, y_dim, x_dim), zlib=True, complevel=1, fill_value=np.nan)
            v_rb.long_name = "retrieved bottom reflectance"
            v_rb.units = "1"
            v_rb.valid_min = 0.0
            v_rb.valid_max = 1.0
            v_rb.grid_mapping = gm_name
            v_rb.coordinates = f"{wl_dim} {y_dim} {x_dim}"

            v_rh = nc.createVariable("rho_hat", "f4", (wl_dim, y_dim, x_dim), zlib=True, complevel=1, fill_value=np.nan)
            v_rh.long_name = "model-predicted water-leaving reflectance (rho)"
            v_rh.units = "1"
            v_rh.grid_mapping = gm_name
            v_rh.coordinates = f"{wl_dim} {y_dim} {x_dim}"

        # Copy global attrs (best effort)
        for k, val in ds.attrs.items():
            if k in ("Conventions", "GeoTransform"):
                continue
            try:
                setattr(nc, k, val)
            except Exception:
                pass

        prev = getattr(nc, "history", "")
        nc.history = (prev + "\n" if prev else "") + "pixel inversion written by invert_pixelwise.py (CF-1.8 + GDAL GeoTransform)"


def _write_tile_to_out_nc(
    out_path: str,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    scal_maps: dict[str, np.ndarray],   # (ty,tx)
    ok_map: np.ndarray,                # (ty,tx)
    rb_cube: Optional[np.ndarray],      # (nwl,ty,tx)
    rho_hat_cube: Optional[np.ndarray], # (nwl,ty,tx)
) -> None:
    with netCDF4.Dataset(out_path, "a") as nc:
        for k, arr in scal_maps.items():
            nc.variables[k][y0:y1, x0:x1] = arr.astype(np.float32)
        nc.variables["inv_ok"][y0:y1, x0:x1] = ok_map.astype(np.int8)

        if rb_cube is not None:
            nc.variables["r_b_hat"][:, y0:y1, x0:x1] = rb_cube.astype(np.float32)
        if rho_hat_cube is not None:
            nc.variables["rho_hat"][:, y0:y1, x0:x1] = rho_hat_cube.astype(np.float32)


# =============================================================================
# Main pixelwise driver
# =============================================================================
def invert_netcdf_pixelwise(
    in_nc: str,
    out_nc: str,
    *,
    bbox: Optional[dict] = None,
    rho_var: str = "rho_w_g21",                  # (wl,y,x)
    wl_range: tuple[float, float] = (370, 730),
    tile_yx: tuple[int, int] = (48, 48),
    n_workers: Optional[int] = None,
    save_spectra: bool = True,
    # uncertainty estimation
    unc_sigma_xy: float = 1.2,
    unc_rel_floor: float = 0.02,
    unc_abs_floor: float = 2e-5,
    # per-pixel auxiliaries
    sun_var: str = "sun_zenith",
    view_var: str = "view_zenith",
    depth_var: str = "bathymetry_nonna10",
    max_depth: float = 10.0,
    h_w_sd_override: Optional[float] = 1.0,
    # Stan stuff
    stan_model_path: str = "",
    user_header: Optional[str] = None,
    eta_builder_import: str = "reverie.correction.aquatic.stan_data_gmm:make_stan_data_eta_model",
    a_w_csv: str = "",
    a0_a1_phyto_csv: str = "",
    rb_obs_path: str = "",
    priors: Optional[dict] = None,
    water_type: int = 2,
    shallow: int = 1,
    q: int = 8,
    eps: float = 1e-6,
    cov_jitter: float = 1e-6,
    sigma2_floor: float = 1e-6,
    out_vars: tuple[str, ...] = ("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
    iter_opt: int = 20000,
) -> None:
    if priors is None:
        priors = dict(
            a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
            bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
            a_g_s_mu=0.017,       a_g_s_sd=0.0012,
            a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
            bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
            h_w_mu=3.0,           h_w_sd=2.0,
        )

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    lo, hi = wl_range
    if lo > hi:
        lo, hi = hi, lo

    ctx = mp.get_context("spawn")
    ty, tx = tile_yx
    seed0 = 1234

    with xr.open_dataset(in_nc, engine="netcdf4") as ds0:
        if rho_var not in ds0:
            raise KeyError(f"{rho_var} not found in {in_nc}")

        rho0 = ds0[rho_var]
        if rho0.ndim != 3:
            raise ValueError(f"{rho_var} must be (wl,y,x). got dims={rho0.dims}")

        wl_dim, y_dim, x_dim = rho0.dims

        ds = ds0.sel({wl_dim: slice(lo, hi)})
        ds = crop_dataset_bbox(ds, rho_var=rho_var, bbox=bbox)

        rho_da = ds[rho_var]
        wl = rho_da[wl_dim].values.astype(float)
        ny = rho_da.sizes[y_dim]
        nx = rho_da.sizes[x_dim]
        nwl = rho_da.sizes[wl_dim]

        sun = ds[sun_var].values.astype(np.float32) if sun_var in ds else None
        view = ds[view_var].values.astype(np.float32) if view_var in ds else None
        depth = ds[depth_var].values.astype(np.float32) if depth_var in ds else None

        # CF/GDAL-compliant output creation
        _create_out_nc(out_nc, ds, rho_var, wl_dim, y_dim, x_dim, save_spectra, out_vars)

        exe_file = compile_model(stan_model_path, user_header=user_header)

        cfg = dict(
            water_type=water_type,
            shallow=shallow,
            a_w_csv=a_w_csv,
            a0_a1_phyto_csv=a0_a1_phyto_csv,
            rb_obs_path=rb_obs_path,
            q=q,
            eps=eps,
            cov_jitter=cov_jitter,
            sigma2_floor=sigma2_floor,
            priors=priors,
            eta_builder_import=eta_builder_import,
        )

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(exe_file, wl, cfg),
        ) as ex:

            for y0 in range(0, ny, ty):
                y1 = min(ny, y0 + ty)
                for x0 in range(0, nx, tx):
                    x1 = min(nx, x0 + tx)

                    R = rho_da.isel({y_dim: slice(y0, y1), x_dim: slice(x0, x1)}).values.astype(np.float32)

                    valid = np.all(np.isfinite(R), axis=0)
                    valid &= np.all(R >= 0, axis=0)

                    if depth is not None:
                        d = depth[y0:y1, x0:x1]
                        valid &= np.isfinite(d) & (d <= max_depth)
                    if view is not None:
                        vv = view[y0:y1, x0:x1]
                        valid &= np.isfinite(vv)
                    if sun is not None:
                        ss = sun[y0:y1, x0:x1]
                        valid &= np.isfinite(ss)

                    if not np.any(valid):
                        ok_map = np.zeros((y1 - y0, x1 - x0), dtype=np.int8)
                        scal_maps = {k: np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32) for k in out_vars}
                        _write_tile_to_out_nc(out_nc, y0, y1, x0, x1, scal_maps, ok_map, None, None)
                        continue

                    Sig = estimate_sigma_rho_local(
                        R,
                        sigma_xy=unc_sigma_xy,
                        rel_floor=unc_rel_floor,
                        abs_floor=unc_abs_floor,
                    )

                    scal_maps = {k: np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32) for k in out_vars}
                    ok_map = np.zeros((y1 - y0, x1 - x0), dtype=np.int8)

                    rb_cube = np.full((nwl, y1 - y0, x1 - x0), np.nan, dtype=np.float32) if save_spectra else None
                    rho_hat_cube = np.full((nwl, y1 - y0, x1 - x0), np.nan, dtype=np.float32) if save_spectra else None

                    futs = []
                    for iy in range(y0, y1):
                        for ix in range(x0, x1):
                            ly = iy - y0
                            lx = ix - x0
                            if not valid[ly, lx]:
                                continue

                            theta_sun = float(sun[iy, ix]) if sun is not None else 30.0
                            theta_view = float(view[iy, ix]) if view is not None else 0.0
                            h_w_mu = float(depth[iy, ix]) if depth is not None else float(priors["h_w_mu"])
                            h_w_sd = float(h_w_sd_override) if h_w_sd_override is not None else float(priors["h_w_sd"])

                            job = dict(
                                iy=iy,
                                ix=ix,
                                wl_nm=wl,
                                rho=R[:, ly, lx],
                                sigma_rho=Sig[:, ly, lx],
                                theta_sun=theta_sun,
                                theta_view=theta_view,
                                h_w_mu=h_w_mu,
                                h_w_sd=h_w_sd,
                                seed=seed0 + (iy * 1000003 + ix),
                                iter=iter_opt,
                            )
                            futs.append(ex.submit(invert_one_pixel, job, out_vars, save_spectra))

                    for fut in as_completed(futs):
                        r = fut.result()
                        iy = int(r["iy"])
                        ix = int(r["ix"])
                        ly = iy - y0
                        lx = ix - x0

                        if not r.get("ok", False):
                            continue

                        ok_map[ly, lx] = 1
                        scal = r["scalars"]
                        for k in out_vars:
                            if k in scal and np.isfinite(scal[k]):
                                scal_maps[k][ly, lx] = float(scal[k])

                        if save_spectra:
                            rb_cube[:, ly, lx] = r["r_b_hat"]
                            rho_hat_cube[:, ly, lx] = r["rho_hat"]

                    _write_tile_to_out_nc(out_nc, y0, y1, x0, x1, scal_maps, ok_map, rb_cube, rho_hat_cube)
                    print(f"[tile] y={y0}:{y1} x={x0}:{x1} ok={int(ok_map.sum())}/{ok_map.size}", file=sys.stderr)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    in_nc = "/D/Data/WISE/ACI-12A/el_sma/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc"
    out_nc = "/D/Data/WISE/ACI-12A/el_sma/220705_ACI-12A-WI-1x1x1_v01-l2rg_inverted_pixelwise_bbox_cf_gdal.nc"

    bbox = {"lon": (-64.37031, -64.36776), "lat": (49.778227, 49.780921)}

    invert_netcdf_pixelwise(
        in_nc=in_nc,
        out_nc=out_nc,
        bbox=bbox,
        rho_var="rho_w_g21",
        wl_range=(370, 730),
        tile_yx=(48, 48),
        n_workers=None,
        save_spectra=True,

        unc_sigma_xy=1.2,
        unc_rel_floor=0.02,
        unc_abs_floor=2e-5,

        sun_var="sun_zenith",
        view_var="view_zenith",
        depth_var="bathymetry_nonna10",
        max_depth=10,
        h_w_sd_override=1.0,

        stan_model_path="/home/raphael/R/SABER/inst/stan/model_gmm.stan",
        user_header="/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp",
        eta_builder_import="reverie.correction.aquatic.stan_data_gmm:make_stan_data_eta_model",
        a_w_csv="/home/raphael/R/SABER/inst/extdata/a_w.csv",
        a0_a1_phyto_csv="/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv",
        rb_obs_path="/home/raphael/R/SABER/inst/extdata/r_b_gamache_obs.csv",

        priors=dict(
            a_nap_star_mu=0.0051, a_nap_star_sd=0.0012,
            bb_p_star_mu=0.0047,  bb_p_star_sd=0.0012,
            a_g_s_mu=0.017,       a_g_s_sd=0.0012,
            a_nap_s_mu=0.006,     a_nap_s_sd=0.0012,
            bb_p_gamma_mu=0.65,   bb_p_gamma_sd=0.12,
            h_w_mu=3.0,           h_w_sd=2.0,
        ),

        water_type=2,
        shallow=1,
        q=8,
        eps=1e-6,
        cov_jitter=1e-6,
        sigma2_floor=1e-6,

        out_vars=("chl", "a_g_440", "spm", "h_w", "sigma_model", "lp__"),
        iter_opt=20000,
    )

    print("Wrote:", out_nc)