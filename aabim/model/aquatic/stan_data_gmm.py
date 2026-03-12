# from __future__ import annotations
#
# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, Optional, Sequence, Tuple, List
#
# import numpy as np
# import pandas as pd
#
#
# def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
#     """Linear interpolation with flat extrapolation (like R approx(rule=2))."""
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
# def pca_basis(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     PCA via SVD on centered data Xc [n_spec, n_wl] (already centered).
#     Returns:
#       Vt: right singular vectors transpose [n_wl, n_wl] (rows are PCs in feature space)
#       s: singular values
#     """
#     # Xc = U S Vt
#     U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
#     # V = Vt.T, columns are PC loadings (n_wl, n_comp)
#     return Vt.T, s
#
#
# @dataclass
# class PCAPrior:
#     classes: List[str]
#     K: int
#     q: int
#     rb_U: np.ndarray                 # (n_wl, q)
#     rb_mu_k: List[np.ndarray]        # list length K, each (n_wl,)
#     rb_tau_k: List[np.ndarray]       # list length K, each (q,)
#     rb_sigma2_k: np.ndarray          # (K,)
#     rb_pi: np.ndarray                # (K,)
#
#
# def build_lowrank_pca_prior_from_replicates(
#     wavelength: Sequence[float],                 # Stan grid [n_wl]
#     rb_obs_path: str,
#     q: int = 8,
#     tau_floor: float = 1e-6,
#     sigma2_floor: float = 1e-8,
#     class_col: str = "class",
#     wl_col: str = "wavelength",
#     rb_col: str = "r_b",
#     rep_id_col_candidates: Tuple[str, ...] = ("time_target",),
# ) -> PCAPrior:
#     wl_target = np.asarray(wavelength, float)
#     n_wl = wl_target.size
#     if n_wl < 2:
#         raise ValueError("wavelength must have length >= 2")
#
#     rb_obs = pd.read_csv(rb_obs_path)
#     for c in (class_col, wl_col, rb_col):
#         if c not in rb_obs.columns:
#             raise ValueError(f"{rb_obs_path} missing column '{c}'")
#
#     rb_obs[class_col] = rb_obs[class_col].astype(str)
#     rb_obs[wl_col] = pd.to_numeric(rb_obs[wl_col], errors="coerce")
#     rb_obs[rb_col] = pd.to_numeric(rb_obs[rb_col], errors="coerce")
#
#     # replicate id: use time_target if present; else fabricate per class
#     rep_id_col = next((c for c in rep_id_col_candidates if c in rb_obs.columns), None)
#     if rep_id_col is None:
#         rb_obs = rb_obs.sort_values([class_col, wl_col], kind="mergesort")
#         # replicate increments when wavelength decreases (new spectrum)
#         def _add_rep_id(df: pd.DataFrame) -> pd.DataFrame:
#             w = df[wl_col].to_numpy(float)
#             rep = np.ones(len(df), dtype=int)
#             rep[1:] = (np.diff(w) < 0).astype(int)
#             df = df.copy()
#             df["rep_id"] = np.cumsum(rep)
#             return df
#
#         rb_obs = rb_obs.groupby(class_col, group_keys=False).apply(_add_rep_id)
#         rep_id_col = "rep_id"
#
#     # pivot: one row per (class, rep_id), columns = wavelengths
#     spectra_df = (
#         rb_obs[[class_col, rep_id_col, wl_col, rb_col]]
#         .pivot_table(index=[class_col, rep_id_col], columns=wl_col, values=rb_col, aggfunc="first")
#         .reset_index()
#     )
#
#     # native wavelength grid (from pivot columns)
#     wl_native = np.array([c for c in spectra_df.columns if isinstance(c, (int, float, np.integer, np.floating))], float)
#     if wl_native.size < 2:
#         raise ValueError("Not enough wavelength columns found in replicate table after pivot.")
#
#     wl_native = np.sort(wl_native)
#     # matrix of native spectra
#     X_native = spectra_df[wl_native].to_numpy(float)  # (n_spec, n_wl_native)
#
#     # interpolate each replicate spectrum onto Stan grid
#     X = np.full((X_native.shape[0], n_wl), np.nan, float)
#     for i in range(X_native.shape[0]):
#         y = X_native[i, :]
#         ok = np.isfinite(y)
#         if ok.sum() < 2:
#             continue
#         X[i, :] = interp1_flat(wl_native[ok], y[ok], wl_target)
#
#     # fill any non-finite with column medians
#     if not np.isfinite(X).all():
#         col_med = np.nanmedian(X, axis=0)
#         bad = ~np.isfinite(X)
#         # broadcast replace
#         X[bad] = np.take(col_med, np.where(bad)[1])
#
#     cls_vec = spectra_df[class_col].astype(str).to_numpy()
#     classes = sorted(pd.unique(cls_vec).tolist())
#     K = len(classes)
#
#     # global PCA basis shared across classes:
#     mu_global = X.mean(axis=0)
#     Xc = X - mu_global[None, :]
#     V, s = pca_basis(Xc)  # V: (n_wl, n_comp)
#     q_eff = int(min(q, V.shape[1], n_wl))
#     rb_U = V[:, :q_eff].copy()  # (n_wl, q)
#
#     rb_mu_k: List[np.ndarray] = []
#     rb_tau_k: List[np.ndarray] = []
#     rb_sigma2_k = np.zeros(K, float)
#     counts = np.zeros(K, int)
#
#     for i, cl in enumerate(classes):
#         idx = np.where(cls_vec == cl)[0]
#         counts[i] = idx.size
#         Xk = X[idx, :]  # (n_k, n_wl)
#
#         mu_k = Xk.mean(axis=0)
#         rb_mu_k.append(mu_k.astype(float, copy=True))
#
#         Rk = Xk - mu_k[None, :]          # (n_k, n_wl)
#         Ak = Rk @ rb_U                   # (n_k, q)
#
#         tau = Ak.std(axis=0, ddof=1) if Ak.shape[0] > 1 else np.zeros(q_eff, float)
#         tau[~np.isfinite(tau)] = 0.0
#         tau = np.maximum(tau, float(tau_floor))
#         rb_tau_k.append(tau.astype(float, copy=True))
#
#         Rhat = Ak @ rb_U.T               # (n_k, n_wl)
#         Ek = Rk - Rhat
#         sigma2 = float(np.mean(Ek**2))
#         rb_sigma2_k[i] = max(sigma2, float(sigma2_floor))
#
#     rb_pi = counts.astype(float) / float(counts.sum()) if counts.sum() > 0 else np.full(K, 1.0 / K)
#
#     return PCAPrior(
#         classes=classes,
#         K=K,
#         q=q_eff,
#         rb_U=rb_U,
#         rb_mu_k=rb_mu_k,
#         rb_tau_k=rb_tau_k,
#         rb_sigma2_k=rb_sigma2_k,
#         rb_pi=rb_pi,
#     )
#
#
# def make_stan_data_with_lowrank_pca(
#     *,
#     wavelength: Sequence[float],
#     rrs_obs: Sequence[float],
#     sigma_rrs: Sequence[float],
#     a_w: Sequence[float],
#     a0: Sequence[float],
#     a1: Sequence[float],
#     bb_w: Sequence[float],
#     rb_obs_path: str,
#     q: int = 8,
#     tau_floor: float = 1e-6,
#     sigma2_floor: float = 1e-8,
#     water_type: int = 2,
#     theta_sun: float = 30.0,
#     theta_view: float = 0.0,
#     shallow: int = 1,
#     # prior hypers (match Stan data names)
#     a_nap_star_mu: float = 0.0051,
#     a_nap_star_sd: float = 0.0012,
#     bb_p_star_mu: float = 0.0047,
#     bb_p_star_sd: float = 0.0012,
#     a_g_s_mu: float = 0.017,
#     a_g_s_sd: float = 0.0012,
#     a_nap_s_mu: float = 0.006,
#     a_nap_s_sd: float = 0.0012,
#     bb_p_gamma_mu: float = 0.65,
#     bb_p_gamma_sd: float = 0.12,
#     h_w_mu: float = 3.0,
#     h_w_sd: float = 2.0,
#     # output
#     out_json: Optional[str] = None,
# ) -> Dict[str, Any]:
#     wl = np.asarray(wavelength, float)
#     n_wl = int(wl.size)
#
#     rrs_obs = np.asarray(rrs_obs, float)
#     sigma_rrs = np.asarray(sigma_rrs, float)
#     if rrs_obs.size != n_wl or sigma_rrs.size != n_wl:
#         raise ValueError("rrs_obs and sigma_rrs must have same length as wavelength")
#     sigma_rrs = np.maximum(sigma_rrs, 1e-8)
#
#     def _vec(x, name):
#         x = np.asarray(x, float)
#         if x.size != n_wl:
#             raise ValueError(f"{name} must have length {n_wl}")
#         return x
#
#     a_w = _vec(a_w, "a_w")
#     a0 = _vec(a0, "a0")
#     a1 = _vec(a1, "a1")
#     bb_w = _vec(bb_w, "bb_w")
#
#     pca = build_lowrank_pca_prior_from_replicates(
#         wavelength=wl,
#         rb_obs_path=rb_obs_path,
#         q=q,
#         tau_floor=tau_floor,
#         sigma2_floor=sigma2_floor,
#     )
#
#     # CmdStan JSON representation:
#     # - matrices as list-of-rows
#     rb_U_rows = pca.rb_U.tolist()  # n_wl rows, each length q
#     rb_mu_k_list = [v.tolist() for v in pca.rb_mu_k]      # list length K, each n_wl
#     rb_tau_k_list = [v.tolist() for v in pca.rb_tau_k]    # list length K, each q
#
#     stan_data: Dict[str, Any] = {
#         "n_wl": n_wl,
#         "wavelength": wl.tolist(),
#         "rrs_obs": rrs_obs.tolist(),
#         "sigma_rrs": sigma_rrs.tolist(),
#
#         "a_w": a_w.tolist(),
#         "a0": a0.tolist(),
#         "a1": a1.tolist(),
#         "bb_w": bb_w.tolist(),
#
#         "K": int(pca.K),
#         "q": int(pca.q),
#         "rb_U": rb_U_rows,
#         "rb_mu_k": rb_mu_k_list,
#         "rb_tau_k": rb_tau_k_list,
#         "rb_sigma2_k": pca.rb_sigma2_k.tolist(),
#         "rb_pi": pca.rb_pi.tolist(),
#
#         "water_type": int(water_type),
#         "theta_sun": float(theta_sun),
#         "theta_view": float(theta_view),
#         "shallow": int(shallow),
#
#         "a_nap_star_mu": float(a_nap_star_mu),
#         "a_nap_star_sd": float(a_nap_star_sd),
#         "bb_p_star_mu": float(bb_p_star_mu),
#         "bb_p_star_sd": float(bb_p_star_sd),
#
#         "a_g_s_mu": float(a_g_s_mu),
#         "a_g_s_sd": float(a_g_s_sd),
#         "a_nap_s_mu": float(a_nap_s_mu),
#         "a_nap_s_sd": float(a_nap_s_sd),
#         "bb_p_gamma_mu": float(bb_p_gamma_mu),
#         "bb_p_gamma_sd": float(bb_p_gamma_sd),
#
#         "h_w_mu": float(h_w_mu),
#         "h_w_sd": float(h_w_sd),
#     }
#
#     if out_json is not None:
#         Path(out_json).write_text(json.dumps(stan_data, indent=2))
#
#     return stan_data

# from __future__ import annotations
#
# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Dict, Optional, Sequence, Tuple, List
#
# import numpy as np
# import pandas as pd
#
#
# def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
#     """Linear interpolation with flat extrapolation (like R approx(rule=2))."""
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
# def pca_basis(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     PCA via SVD on centered data Xc [n_spec, n_wl] (already centered).
#     Returns V (loadings) and singular values.
#     """
#     U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
#     return Vt.T, s  # V: (n_wl, n_comp)
#
#
# def _cov_ddof1(A: np.ndarray) -> np.ndarray:
#     """Sample covariance like R cov() (ddof=1). Returns (q,q)."""
#     n = A.shape[0]
#     q = A.shape[1]
#     if n <= 1:
#         return np.zeros((q, q), float)
#     Ac = A - A.mean(axis=0, keepdims=True)
#     return (Ac.T @ Ac) / float(n - 1)
#
#
# @dataclass
# class PCAPrior:
#     classes: List[str]
#     K: int
#     q: int
#     rb_U: np.ndarray                  # (n_wl, q)
#     rb_mu_k: List[np.ndarray]         # list length K, each (n_wl,)
#     rb_L_k: List[np.ndarray]          # list length K, each (q, q) lower Cholesky
#     rb_sigma2_k: np.ndarray           # (K,)
#     rb_pi: np.ndarray                 # (K,)
#
#
# def build_lowrank_pca_prior_from_replicates(
#     wavelength: Sequence[float],                 # Stan grid [n_wl]
#     rb_obs_path: str,
#     q: int = 8,
#     tau_floor: float = 1e-6,                     # used as jitter scale (tau_floor^2 on diag), like R
#     sigma2_floor: float = 1e-8,
#     class_col: str = "class",
#     wl_col: str = "wavelength",
#     rb_col: str = "r_b",
#     rep_id_col_candidates: Tuple[str, ...] = ("time_target",),
# ) -> PCAPrior:
#     wl_target = np.asarray(wavelength, float)
#     n_wl = wl_target.size
#     if n_wl < 2:
#         raise ValueError("wavelength must have length >= 2")
#
#     rb_obs = pd.read_csv(rb_obs_path)
#     for c in (class_col, wl_col, rb_col):
#         if c not in rb_obs.columns:
#             raise ValueError(f"{rb_obs_path} missing column '{c}'")
#
#     rb_obs[class_col] = rb_obs[class_col].astype(str)
#     rb_obs[wl_col] = pd.to_numeric(rb_obs[wl_col], errors="coerce")
#     rb_obs[rb_col] = pd.to_numeric(rb_obs[rb_col], errors="coerce")
#
#     # replicate id: prefer time_target if present; otherwise fabricate within class
#     rep_id_col = next((c for c in rep_id_col_candidates if c in rb_obs.columns), None)
#     if rep_id_col is None:
#         rb_obs = rb_obs.sort_values([class_col, wl_col], kind="mergesort")
#
#         def _add_rep_id(df: pd.DataFrame) -> pd.DataFrame:
#             w = df[wl_col].to_numpy(float)
#             rep = np.ones(len(df), dtype=int)
#             rep[1:] = (np.diff(w) < 0).astype(int)
#             out = df.copy()
#             out["rep_id"] = np.cumsum(rep)
#             return out
#
#         rb_obs = rb_obs.groupby(class_col, group_keys=False).apply(_add_rep_id)
#         rep_id_col = "rep_id"
#
#     # pivot: one row per (class, rep_id), columns = wavelengths
#     spectra_df = (
#         rb_obs[[class_col, rep_id_col, wl_col, rb_col]]
#         .pivot_table(index=[class_col, rep_id_col], columns=wl_col, values=rb_col, aggfunc="first")
#         .reset_index()
#     )
#
#     # wavelength columns (numeric)
#     wl_native = np.array(
#         [c for c in spectra_df.columns if isinstance(c, (int, float, np.integer, np.floating))],
#         float,
#     )
#     if wl_native.size < 2:
#         raise ValueError("Not enough wavelength columns found after pivot.")
#     wl_native = np.sort(wl_native)
#
#     X_native = spectra_df[wl_native].to_numpy(float)  # (n_spec, n_wl_native)
#
#     # interpolate each replicate spectrum onto Stan grid
#     X = np.full((X_native.shape[0], n_wl), np.nan, float)
#     for i in range(X_native.shape[0]):
#         y = X_native[i, :]
#         ok = np.isfinite(y)
#         if ok.sum() < 2:
#             continue
#         X[i, :] = interp1_flat(wl_native[ok], y[ok], wl_target)
#
#     # fill any non-finite with column medians
#     if not np.isfinite(X).all():
#         col_med = np.nanmedian(X, axis=0)
#         bad = ~np.isfinite(X)
#         X[bad] = np.take(col_med, np.where(bad)[1])
#
#     cls_vec = spectra_df[class_col].astype(str).to_numpy()
#     classes = sorted(pd.unique(cls_vec).tolist())
#     K = len(classes)
#
#     # global PCA basis (shared across classes), like R: prcomp(Xc, center=FALSE) after global centering
#     mu_global = X.mean(axis=0)
#     Xc = X - mu_global[None, :]
#     V, _s = pca_basis(Xc)
#     q_eff = int(min(q, V.shape[1], n_wl))
#     rb_U = V[:, :q_eff].copy()  # (n_wl, q)
#
#     rb_mu_k: List[np.ndarray] = []
#     rb_L_k: List[np.ndarray] = []
#     rb_sigma2_k = np.zeros(K, float)
#     counts = np.zeros(K, int)
#
#     jitter = float(tau_floor) ** 2
#
#     for i, cl in enumerate(classes):
#         idx = np.where(cls_vec == cl)[0]
#         counts[i] = idx.size
#         Xk = X[idx, :]  # (n_k, n_wl)
#
#         mu_k = Xk.mean(axis=0)
#         rb_mu_k.append(mu_k.astype(float, copy=True))
#
#         # residuals around class mean
#         Rk = Xk - mu_k[None, :]          # (n_k, n_wl)
#         Ak = Rk @ rb_U                   # (n_k, q)
#
#         # Sk = cov(Ak) + diag(tau_floor^2), then lower Cholesky (matches R: cov + diag(tau_floor^2); L = t(chol))
#         Sk = _cov_ddof1(Ak)
#         Sk = Sk + np.eye(q_eff) * jitter
#
#         # robust chol: in case of near-singularity, add more jitter progressively
#         added = 0.0
#         for _ in range(6):
#             try:
#                 Lk = np.linalg.cholesky(Sk + np.eye(q_eff) * added)
#                 break
#             except np.linalg.LinAlgError:
#                 added = (10.0 ** (_ + 1)) * jitter
#         else:
#             # last resort: diagonal
#             Lk = np.linalg.cholesky(np.eye(q_eff) * max(jitter, 1e-12))
#
#         rb_L_k.append(Lk.astype(float, copy=True))  # lower
#
#         # residual variance in orthogonal complement
#         Rhat = Ak @ rb_U.T               # (n_k, n_wl)
#         Ek = Rk - Rhat
#         sigma2 = float(np.mean(Ek ** 2))
#         rb_sigma2_k[i] = max(sigma2, float(sigma2_floor))
#
#     rb_pi = counts.astype(float) / float(counts.sum()) if counts.sum() > 0 else np.full(K, 1.0 / K)
#
#     return PCAPrior(
#         classes=classes,
#         K=K,
#         q=q_eff,
#         rb_U=rb_U,
#         rb_mu_k=rb_mu_k,
#         rb_L_k=rb_L_k,
#         rb_sigma2_k=rb_sigma2_k,
#         rb_pi=rb_pi,
#     )
#
#
# def make_stan_data_with_lowrank_pca(
#     *,
#     wavelength: Sequence[float],
#     rrs_obs: Sequence[float],
#     sigma_rrs: Sequence[float],
#     a_w: Sequence[float],
#     a0: Sequence[float],
#     a1: Sequence[float],
#     bb_w: Sequence[float],
#     rb_obs_path: str,
#     q: int = 8,
#     tau_floor: float = 1e-6,
#     sigma2_floor: float = 1e-8,
#     water_type: int = 2,
#     theta_sun: float = 30.0,
#     theta_view: float = 0.0,
#     shallow: int = 1,
#     # prior hypers (match Stan data names)
#     a_nap_star_mu: float = 0.0051,
#     a_nap_star_sd: float = 0.0012,
#     bb_p_star_mu: float = 0.0047,
#     bb_p_star_sd: float = 0.0012,
#     a_g_s_mu: float = 0.017,
#     a_g_s_sd: float = 0.0012,
#     a_nap_s_mu: float = 0.006,
#     a_nap_s_sd: float = 0.0012,
#     bb_p_gamma_mu: float = 0.65,
#     bb_p_gamma_sd: float = 0.12,
#     h_w_mu: float = 3.0,
#     h_w_sd: float = 2.0,
#     out_json: Optional[str] = None,
# ) -> Dict[str, Any]:
#     wl = np.asarray(wavelength, float)
#     n_wl = int(wl.size)
#
#     rrs_obs = np.asarray(rrs_obs, float)
#     sigma_rrs = np.asarray(sigma_rrs, float)
#     if rrs_obs.size != n_wl or sigma_rrs.size != n_wl:
#         raise ValueError("rrs_obs and sigma_rrs must have same length as wavelength")
#     sigma_rrs = np.maximum(sigma_rrs, 1e-8)
#
#     def _vec(x, name):
#         x = np.asarray(x, float)
#         if x.size != n_wl:
#             raise ValueError(f"{name} must have length {n_wl}")
#         return x
#
#     a_w = _vec(a_w, "a_w")
#     a0 = _vec(a0, "a0")
#     a1 = _vec(a1, "a1")
#     bb_w = _vec(bb_w, "bb_w")
#
#     pca = build_lowrank_pca_prior_from_replicates(
#         wavelength=wl,
#         rb_obs_path=rb_obs_path,
#         q=q,
#         tau_floor=tau_floor,
#         sigma2_floor=sigma2_floor,
#     )
#
#     stan_data: Dict[str, Any] = {
#         "n_wl": n_wl,
#         "wavelength": wl.tolist(),
#         "rrs_obs": rrs_obs.tolist(),
#         "sigma_rrs": sigma_rrs.tolist(),
#         "a_w": a_w.tolist(),
#         "a0": a0.tolist(),
#         "a1": a1.tolist(),
#         "bb_w": bb_w.tolist(),
#         "K": int(pca.K),
#         "q": int(pca.q),
#         "rb_U": pca.rb_U.tolist(),                              # [n_wl][q]
#         "rb_mu_k": [v.tolist() for v in pca.rb_mu_k],            # K x [n_wl]
#         "rb_L_k": [L.tolist() for L in pca.rb_L_k],              # K x [q][q]  (lower)
#         "rb_sigma2_k": pca.rb_sigma2_k.tolist(),                 # [K]
#         "rb_pi": pca.rb_pi.tolist(),                             # [K]
#         "water_type": int(water_type),
#         "theta_sun": float(theta_sun),
#         "theta_view": float(theta_view),
#         "shallow": int(shallow),
#         "a_nap_star_mu": float(a_nap_star_mu),
#         "a_nap_star_sd": float(a_nap_star_sd),
#         "bb_p_star_mu": float(bb_p_star_mu),
#         "bb_p_star_sd": float(bb_p_star_sd),
#         "a_g_s_mu": float(a_g_s_mu),
#         "a_g_s_sd": float(a_g_s_sd),
#         "a_nap_s_mu": float(a_nap_s_mu),
#         "a_nap_s_sd": float(a_nap_s_sd),
#         "bb_p_gamma_mu": float(bb_p_gamma_mu),
#         "bb_p_gamma_sd": float(bb_p_gamma_sd),
#         "h_w_mu": float(h_w_mu),
#         "h_w_sd": float(h_w_sd),
#     }
#
#     if out_json is not None:
#         Path(out_json).write_text(json.dumps(stan_data, indent=2))
#
#     return stan_data

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd


def interp1_flat(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """Linear interpolation with flat extrapolation (like R approx(rule=2))."""
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


def pca_basis(Xc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA via SVD on centered data Xc [n_spec, n_wl] (already centered).
    Returns V (loadings) and singular values.
    """
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt.T, s  # V: (n_wl, n_comp)


def _cov_ddof1(A: np.ndarray) -> np.ndarray:
    """Sample covariance like R cov() (ddof=1). Returns (q,q)."""
    n = A.shape[0]
    q = A.shape[1]
    if n <= 1:
        return np.zeros((q, q), float)
    Ac = A - A.mean(axis=0, keepdims=True)
    return (Ac.T @ Ac) / float(n - 1)


def clamp01(x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, float)
    return np.minimum(np.maximum(x, eps), 1.0 - eps)


@dataclass
class EtaPCAPrior:
    classes: List[str]
    K: int
    q: int
    eta_U: np.ndarray                  # (n_wl, q)
    eta_mu_k: List[np.ndarray]         # list length K, each (n_wl,)
    eta_L_k: List[np.ndarray]          # list length K, each (q, q) lower Cholesky
    eta_sigma2_k: np.ndarray           # (K,)
    eta_pi: np.ndarray                 # (K,)


def build_eta_lowrank_pca_prior_from_replicates(
    wavelength: Sequence[float],                 # Stan grid [n_wl]
    rb_obs_path: str,
    q: int = 8,
    eps: float = 1e-6,                           # clamp for logit
    cov_jitter: float = 1e-6,                    # jitter added to PC-score covariance diag
    sigma2_floor: float = 1e-8,
    class_col: str = "class",
    wl_col: str = "wavelength",
    rb_col: str = "r_b",
    rep_id_col_candidates: Tuple[str, ...] = ("time_target",),
) -> EtaPCAPrior:
    wl_target = np.asarray(wavelength, float)
    n_wl = wl_target.size
    if n_wl < 2:
        raise ValueError("wavelength must have length >= 2")

    rb_obs = pd.read_csv(rb_obs_path)
    for c in (class_col, wl_col, rb_col):
        if c not in rb_obs.columns:
            raise ValueError(f"{rb_obs_path} missing column '{c}'")

    rb_obs[class_col] = rb_obs[class_col].astype(str)
    rb_obs[wl_col] = pd.to_numeric(rb_obs[wl_col], errors="coerce")
    rb_obs[rb_col] = pd.to_numeric(rb_obs[rb_col], errors="coerce")

    # replicate id: prefer time_target if present; otherwise fabricate within class
    rep_id_col = next((c for c in rep_id_col_candidates if c in rb_obs.columns), None)
    if rep_id_col is None:
        rb_obs = rb_obs.sort_values([class_col, wl_col], kind="mergesort")

        def _add_rep_id(df: pd.DataFrame) -> pd.DataFrame:
            w = df[wl_col].to_numpy(float)
            rep = np.ones(len(df), dtype=int)
            rep[1:] = (np.diff(w) < 0).astype(int)
            out = df.copy()
            out["rep_id"] = np.cumsum(rep)
            return out

        rb_obs = rb_obs.groupby(class_col, group_keys=False).apply(_add_rep_id)
        rep_id_col = "rep_id"

    # pivot: one row per (class, rep_id), columns = wavelengths
    spectra_df = (
        rb_obs[[class_col, rep_id_col, wl_col, rb_col]]
        .pivot_table(index=[class_col, rep_id_col], columns=wl_col, values=rb_col, aggfunc="first")
        .reset_index()
    )

    # wavelength columns (numeric)
    wl_native = np.array(
        [c for c in spectra_df.columns if isinstance(c, (int, float, np.integer, np.floating))],
        float,
    )
    if wl_native.size < 2:
        raise ValueError("Not enough wavelength columns found after pivot.")
    wl_native = np.sort(wl_native)

    X_native = spectra_df[wl_native].to_numpy(float)  # (n_spec, n_wl_native)

    # interpolate each replicate spectrum onto Stan grid
    X = np.full((X_native.shape[0], n_wl), np.nan, float)
    for i in range(X_native.shape[0]):
        y = X_native[i, :]
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        X[i, :] = interp1_flat(wl_native[ok], y[ok], wl_target)

    # fill any non-finite with column medians
    if not np.isfinite(X).all():
        col_med = np.nanmedian(X, axis=0)
        bad = ~np.isfinite(X)
        X[bad] = np.take(col_med, np.where(bad)[1])

    cls_vec = spectra_df[class_col].astype(str).to_numpy()
    classes = sorted(pd.unique(cls_vec).tolist())
    K = len(classes)
    if K == 0:
        raise ValueError("No classes found after processing library (check rb_obs_path and columns).")

    # reflectance -> eta (logit space)
    eta_X = np.log(clamp01(X, eps) / (1.0 - clamp01(X, eps)))  # qlogis

    # global PCA basis (shared across classes) in eta-space
    mu_global = eta_X.mean(axis=0)
    Xc = eta_X - mu_global[None, :]
    V, _s = pca_basis(Xc)
    q_eff = int(min(q, V.shape[1], n_wl))
    eta_U = V[:, :q_eff].copy()  # (n_wl, q)

    eta_mu_k: List[np.ndarray] = []
    eta_L_k: List[np.ndarray] = []
    eta_sigma2_k = np.zeros(K, float)
    counts = np.zeros(K, int)

    jitter = float(cov_jitter)

    for i, cl in enumerate(classes):
        idx = np.where(cls_vec == cl)[0]
        counts[i] = idx.size
        Xk = eta_X[idx, :]  # (n_k, n_wl) in eta-space

        mu_k = Xk.mean(axis=0)
        eta_mu_k.append(mu_k.astype(float, copy=True))

        # residuals around class mean
        Rk = Xk - mu_k[None, :]          # (n_k, n_wl)
        Ak = Rk @ eta_U                  # (n_k, q)

        # PC-score covariance + jitter, then lower Cholesky
        Sk = _cov_ddof1(Ak)
        Sk = Sk + np.eye(q_eff) * jitter

        added = 0.0
        for t in range(8):
            try:
                Lk = np.linalg.cholesky(Sk + np.eye(q_eff) * added)
                break
            except np.linalg.LinAlgError:
                added = (10.0 ** (t + 1)) * jitter
        else:
            Lk = np.linalg.cholesky(np.eye(q_eff) * max(jitter, 1e-12))

        eta_L_k.append(Lk.astype(float, copy=True))

        # residual variance in orthogonal complement
        Rhat = Ak @ eta_U.T              # (n_k, n_wl)
        Ek = Rk - Rhat
        sigma2 = float(np.mean(Ek ** 2))
        eta_sigma2_k[i] = max(sigma2, float(sigma2_floor))

    # eta_pi = counts.astype(float) / float(counts.sum()) if counts.sum() > 0 else np.full(K, 1.0 / K)
    eta_pi = np.full(K, 1.0 / K)
    return EtaPCAPrior(
        classes=classes,
        K=K,
        q=q_eff,
        eta_U=eta_U,
        eta_mu_k=eta_mu_k,
        eta_L_k=eta_L_k,
        eta_sigma2_k=eta_sigma2_k,
        eta_pi=eta_pi,
    )


def make_stan_data_eta_model(
    *,
    wavelength: Sequence[float],
    rrs_obs: Sequence[float],
    sigma_rrs: Sequence[float],
    a_w: Sequence[float],
    a0: Sequence[float],
    a1: Sequence[float],
    bb_w: Sequence[float],
    rb_obs_path: str,
    q: int = 8,
    eps: float = 1e-6,
    cov_jitter: float = 1e-6,
    sigma2_floor: float = 1e-8,
    water_type: int = 2,
    theta_sun: float = 30.0,
    theta_view: float = 0.0,
    shallow: int = 1,
    # prior hypers (match Stan data names)
    a_nap_star_mu: float = 0.0051,
    a_nap_star_sd: float = 0.0012,
    bb_p_star_mu: float = 0.0047,
    bb_p_star_sd: float = 0.0012,
    a_g_s_mu: float = 0.017,
    a_g_s_sd: float = 0.0012,
    a_nap_s_mu: float = 0.006,
    a_nap_s_sd: float = 0.0012,
    bb_p_gamma_mu: float = 0.65,
    bb_p_gamma_sd: float = 0.12,
    h_w_mu: float = 3.0,
    h_w_sd: float = 2.0,
    out_json: Optional[str] = None,
) -> Dict[str, Any]:
    wl = np.asarray(wavelength, float)
    n_wl = int(wl.size)

    rrs_obs = np.asarray(rrs_obs, float)
    sigma_rrs = np.asarray(sigma_rrs, float)
    if rrs_obs.size != n_wl or sigma_rrs.size != n_wl:
        raise ValueError("rrs_obs and sigma_rrs must have same length as wavelength")
    sigma_rrs = np.maximum(sigma_rrs, 1e-8)

    def _vec(x, name):
        x = np.asarray(x, float)
        if x.size != n_wl:
            raise ValueError(f"{name} must have length {n_wl}")
        return x

    a_w = _vec(a_w, "a_w")
    a0 = _vec(a0, "a0")
    a1 = _vec(a1, "a1")
    bb_w = _vec(bb_w, "bb_w")

    prior = build_eta_lowrank_pca_prior_from_replicates(
        wavelength=wl,
        rb_obs_path=rb_obs_path,
        q=q,
        eps=eps,
        cov_jitter=cov_jitter,
        sigma2_floor=sigma2_floor,
    )

    stan_data: Dict[str, Any] = {
        "n_wl": n_wl,
        "wavelength": wl.tolist(),
        "rrs_obs": rrs_obs.tolist(),
        "sigma_rrs": sigma_rrs.tolist(),
        "a_w": a_w.tolist(),
        "a0": a0.tolist(),
        "a1": a1.tolist(),
        "bb_w": bb_w.tolist(),
        "K": int(prior.K),
        "q": int(prior.q),
        "eta_U": prior.eta_U.tolist(),                               # [n_wl][q]
        "eta_mu_k": [v.tolist() for v in prior.eta_mu_k],            # K x [n_wl]
        "eta_L_k": [L.tolist() for L in prior.eta_L_k],              # K x [q][q] (lower)
        "eta_sigma2_k": prior.eta_sigma2_k.tolist(),                 # [K]
        "eta_pi": prior.eta_pi.tolist(),                             # [K]  (Stan: simplex[K])
        "water_type": int(water_type),
        "theta_sun": float(theta_sun),
        "theta_view": float(theta_view),
        "shallow": int(shallow),
        "a_nap_star_mu": float(a_nap_star_mu),
        "a_nap_star_sd": float(a_nap_star_sd),
        "bb_p_star_mu": float(bb_p_star_mu),
        "bb_p_star_sd": float(bb_p_star_sd),
        "a_g_s_mu": float(a_g_s_mu),
        "a_g_s_sd": float(a_g_s_sd),
        "a_nap_s_mu": float(a_nap_s_mu),
        "a_nap_s_sd": float(a_nap_s_sd),
        "bb_p_gamma_mu": float(bb_p_gamma_mu),
        "bb_p_gamma_sd": float(bb_p_gamma_sd),
        "h_w_mu": float(h_w_mu),
        "h_w_sd": float(h_w_sd),
    }

    if out_json is not None:
        Path(out_json).write_text(json.dumps(stan_data, indent=2))

    return stan_data