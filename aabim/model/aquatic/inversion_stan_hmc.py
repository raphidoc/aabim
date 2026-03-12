import logging
import math
import os
import shutil
import time

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import re
import numpy as np
import xarray as xr
import glob
from functools import partial
import netCDF4
from pyproj import Transformer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from reverie import ReveCube
from reverie.correction.surface.rayleigh import fresnel_reflectance
from reverie.correction.aquatic.stan_data import make_stan_data_base_from_csv
import reverie.correction.aquatic.rtm as rtm

# ------------------ Stan worker init ------------------
_STAN_MODEL = None
_BASE = None

def _init_worker(stan_model_path: str,
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
                 user_header: str | None = None):
    global _STAN_MODEL, _BASE
    silence_everything()
    from cmdstanpy import CmdStanModel

    _STAN_MODEL = CmdStanModel(
        stan_file=stan_model_path,
        user_header=user_header
    )

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

# ------------------ Per-window inversion ------------------
def process_window_stan(
        window,
        l2_path,
        output_dir,
        stan_cfg=None,
        out_vars=None,
        rho_var="rho_w_g21",
        mask_var=None
):
    """
    Reads only a window from l2_path, runs Stan inversion, writes one file for this window.

    window: rasterio.windows.Window-like (row_off, col_off, height, width)
    bbox: optional crop in lon/lat already applied outside, OR keep None here
    mask_var: optional name of mask in dataset; if provided, skip non-water pixels
    """
    assert stan_cfg is not None and out_vars is not None

    # Open lazily (no load)
    ds = xr.open_dataset(l2_path, engine="netcdf4")

    y0 = int(window.row_off)
    x0 = int(window.col_off)
    y1 = y0 + int(window.height)
    x1 = x0 + int(window.width)

    # Slice lazily; then load only this window into RAM
    sub = ds.isel(y=slice(y0, y1), x=slice(x0, x1))
    rho = sub[rho_var].load()  # dims: (wavelength, y, x)
    sun_zenith = sub["sun_zenith"].values
    view_zenith = sub["view_zenith"].values
    h_w = sub["bathymetry_nonna10"].values

    if mask_var is not None:
        mask = sub[mask_var].load().values.astype(bool)  # (y,x)
    else:
        mask = None

    # wl = rho["wavelength"].values
    rho_np = rho.values  # (wl, y, x)
    nwl, ny, nx = rho_np.shape

    # Prepare output arrays for this window only (bounded memory: window_size * n_params)
    out = {v: np.full((ny, nx), np.nan, dtype=np.float32) for v in out_vars}

    rb_hat = np.full((nwl, ny, nx), np.nan, dtype=np.float32)

    pid_dir = os.path.join(output_dir, f"cmdstan_{os.getpid()}")
    os.makedirs(pid_dir, exist_ok=True)

    # Invert per pixel (you can micro-chunk here if you want)
    global _STAN_MODEL, _BASE
    for iy in range(ny):
        for ix in range(nx):
            if mask is not None and not mask[iy, ix]:
                continue
            rrs_0p = rho_np[:, iy, ix] / np.pi
            rrs_0m = rrs_0p / (0.52 + 1.7 * rrs_0p)
            sun_zenith_pix = sun_zenith[iy, ix]
            view_zenith_pix = view_zenith[iy, ix]
            if not np.all(np.isfinite(rrs_0m)):
                continue

            stan_data = dict(_BASE)
            stan_data["rrs_obs"] = rrs_0m
            stan_data["sigma_rrs"] = np.full(len(rrs_0m), 0.0003)
            stan_data["theta_sun"] = sun_zenith_pix
            stan_data["theta_view"] = view_zenith_pix
            stan_data["h_w"] = h_w[iy, ix]

            try:
                if stan_cfg["method"] == "optimize":
                    fit = _STAN_MODEL.optimize(
                        data=stan_data,
                        seed=stan_cfg["seed"],
                        output_dir=pid_dir,
                    )
                    est = fit.optimized_params_dict
                else:
                    fit = _STAN_MODEL.sample(
                        data=stan_data,
                        seed=stan_cfg["seed"],
                        chains=stan_cfg["chains"],
                        parallel_chains=1,
                        iter_warmup=stan_cfg["iter_warmup"],
                        iter_sampling=stan_cfg["iter_sampling"],
                        show_progress=False,
                        show_console=False,
                    )
                    draws = fit.draws_pd()
                    est = {v: float(draws[v].mean()) for v in out_vars}

                for v in out_vars:
                    out[v][iy, ix] = float(est[v])

                # --- after you have est ---
                chl = float(est["chl"])
                a_g_440 = float(est["a_g_440"])
                spm = float(est["spm"])

                # fixed params like your R logic for "sc"
                a_nap_star = 0.0051
                bb_p_star = 0.0047
                a_g_s = 0.017
                a_nap_s = 0.006
                bb_p_gamma = 0.65

                # base LUTs from _BASE (must exist there)
                wl = stan_data["wavelength"] if "wavelength" in stan_data else stan_data[
                    "wl"] if "wl" in stan_data else None
                if wl is None:
                    wl = np.asarray(_BASE["wavelength"], dtype=float)
                else:
                    wl = np.asarray(wl, dtype=float)

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
                    theta_sun_deg=float(sun_zenith_pix),
                    theta_view_deg=float(view_zenith_pix),  # or 0.0 if you want exactly like your R snippet
                    h_w=float(h_w[iy, ix]),
                    rrs_obs=rrs_0m.astype(float),
                )

                rb = np.clip(np.nan_to_num(rb, nan=0.0), 0.0, 1.0)

                rb_hat[:, iy, ix] = rb.astype(np.float32, copy=False)

            except Exception:
                continue

    # Write window file (no shared write contention)
    y_coords = sub["y"].values
    x_coords = sub["x"].values

    out_path = os.path.join(
        output_dir,
        f"stan_window_{y0}-{y1}_{x0}-{x1}.nc"
    )

    wl_coords = rho["wavelength"].values

    ds_out = xr.Dataset(
        {v: xr.DataArray(out[v], dims=["y", "x"], coords={"y": y_coords, "x": x_coords})
         for v in out_vars}
    )

    ds_out["r_b_hat"] = xr.DataArray(
        rb_hat,
        dims=["wavelength", "y", "x"],
        coords={"wavelength": wl_coords, "y": y_coords, "x": x_coords},
    )

    ds_out.attrs.update({"row_off": y0, "row_end": y1, "col_off": x0, "col_end": x1})

    ds_out.to_netcdf(out_path)

    ds_out.close()
    ds.close()
    return out_path

def quiet_logging():
    # kill Python logging (yours + libs using logging)
    logging.disable(logging.ERROR)

    # common native/lib chatty env vars
    os.environ["CMDSTANPY_SILENT"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # harmless if TF not used

def silence_everything():
    # silence all python logging (root + libs)
    logging.disable(logging.ERROR)

    # also silence cmdstanpy’s named loggers (belt + suspenders)
    for name in ("cmdstanpy", "cmdstanpy.model", "cmdstanpy.utils", "cmdstanpy.stanfit"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False

    # avoid some cmdstanpy verbosity flags (harmless if unused)
    os.environ["CMDSTANPY_LOG_LEVEL"] = "CRITICAL"

def run_l2_inversion(
        l2w,
        stan_model_path,
        n_workers=None,
        chunk_size=256,
        method="optimize",
        out_vars=("chl","a_gnap_440", "a_gnap_s", "bb_p_550", "bb_p_gamma", "h_w","r_b_mix", "r_b_a", "sigma_model"),
        wavelength_filter = None,
        bbox = None
):
    quiet_logging()

    out_vars = tuple(list(out_vars) + ["lp__"])

    scale_factor = 1e-5  # NetCDF divide by scale when writing data and multiply when reading

    log_dir = os.path.dirname(l2w.in_path)
    log_file = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(l2w.in_path))[0]}_l2bottom.log")

    # Get the root logger and remove all handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set up file and stream handlers
    file_handler = logging.FileHandler(log_file, mode='w')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)



    start_time = time.time()
    logging.info("Starting inversion processing")

    output_dir = os.path.join(os.path.dirname(l2w.in_path), "temp_windows_" + l2w.image_name)  # Must exist
    os.makedirs(output_dir, exist_ok=True)

    if wavelength_filter is not None:
        l2w.mask_wavelength(wavelength_filter)

    wavelength = l2w.in_ds.wavelength.values

    temp_crop = None
    if bbox is not None:
        transformer = Transformer.from_crs("EPSG:4326", l2w.CRS, always_xy=True)
        x_min, y_min = transformer.transform(bbox["lon"][0], bbox["lat"][0])
        x_max, y_max = transformer.transform(bbox["lon"][1], bbox["lat"][1])

        # projected y is descending from north to south, so slice from max to min
        #l1c.y[0] > l1c.y[-1]

        l2w.in_ds = l2w.in_ds.sel(
            x=slice(x_min, x_max),
            y=slice(y_max, y_min)
        )

        temp_crop = os.path.join(output_dir, "temp_crop.nc")
        l2w.in_ds.to_netcdf(temp_crop)

        # Necessary after crop to maintain consistency between class attributes and self.in_ds
        l2w.update_attributes()

    out_name = re.sub(r'l2rg', 'l2_inverted', l2w.in_path)
    l2w.create_reve_nc(out_name)

    # logging.info("Residual glint (ref_ratio) g21  rho_f({}) = {}".format(wavelength[nir_i], ref_ratio))

    # Create output variables (one per retrieved parameter)
    for v in out_vars:
        l2w.create_var_nc(
            name=v,
            type="i4",
            dims=("y", "x"),
            comp="zlib",
            complevel=1,
            scale=scale_factor,
        )

    l2w.create_var_nc(
        name="r_b_hat",
        type="i4",
        dims=("wavelength", "y", "x"),
        comp="zlib",
        complevel=1,
        scale=scale_factor,
    )

    if l2w.no_data is None:
        l2w.no_data = math.trunc(netCDF4.default_fillvals["i4"] * scale_factor)


    window_size = 100
    windows = l2w.create_windows(window_size)
    non_empty_windows = []
    for window in tqdm(windows, desc="Filtering windows"):
        y_slice = slice(window.row_off, window.row_off + window.height)
        x_slice = slice(window.col_off, window.col_off + window.width)

        rho_t = l2w.in_ds.isel(
            x=x_slice,
            y=y_slice
        )["rho_w_g21"]

        if not np.isnan(rho_t).all():
            non_empty_windows.append(window)
        else:
            for v in out_vars:
                l2w.out_ds.variables[v][y_slice, x_slice] = l2w.no_data * scale_factor

    logging.info(
        f"{len(non_empty_windows)} non-empty windows to process"
    )

    # ---------- multiprocessing ----------
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    stan_cfg = dict(
        method=method,
        seed=1234,
        chains=2,
        iter_warmup=300,
        iter_sampling=300,
        out_vars=list(out_vars),
    )

    ctx = mp.get_context("spawn")  # safer with CmdStanPy across platforms
    logging.info(f"Launching pool: workers={n_workers}, chunk_size={chunk_size}, method={method}")

    # build these once in the parent
    wavelength = np.asarray(l2w.wavelength, dtype=float)  # small
    bottom_class_names = ("coraline_crustose_algae", "saccharina_latissima", "sediment")
    user_header = "/home/raphael/R/SABER/inst/stan/fct_rtm_stan.hpp"

    a_w_csv = "/home/raphael/R/SABER/inst/extdata/a_w.csv"
    a0_a1_phyto_csv = "/home/raphael/R/SABER/inst/extdata/a0_a1_phyto.csv"
    r_b_gamache_csv = "/home/raphael/R/SABER/inst/extdata/r_b_gamache.csv"

    initargs = (
        stan_model_path,
        wavelength,  # numpy array is pickleable; list(wavelength) also OK
        2,  # water_type
        1,  # shallow
        bottom_class_names,
        0.0051,  # a_nap_star
        0.0047,  # bb_p_star
        0.017,  # a_g_s
        0.006,  # a_nap_s
        0.65,  # bb_p_gamma
        a_w_csv,
        a0_a1_phyto_csv,
        r_b_gamache_csv,
        user_header,
    )

    if temp_crop is not None:
        ds_path = temp_crop
    else:
        ds_path = l2w.in_ds

    # dispatch window jobs
    win_files = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=initargs,
    ) as ex:
        futs = []
        for w in non_empty_windows:
            futs.append(ex.submit(
                process_window_stan,
                w,
                ds_path,
                output_dir,
                stan_cfg=stan_cfg,
                out_vars=out_vars,
                rho_var="rho_w_g21",
                mask_var=None,
            ))

        for fut in as_completed(futs):
            win_files.append(fut.result())

    logging.debug("Merging results")
    # all_files = sorted(glob.glob(f"{output_dir}/rho_surface_window_*.nc"))

    vhandles = {v: l2w.out_ds.variables[v] for v in out_vars}

    rb_handle = l2w.out_ds.variables["r_b_hat"]

    for file in tqdm(win_files, desc="Writing results"):
        ds = xr.open_dataset(file)

        y0 = int(ds.attrs["row_off"])
        y1 = int(ds.attrs["row_end"])
        x0 = int(ds.attrs["col_off"])
        x1 = int(ds.attrs["col_end"])

        for v in out_vars:
            block = ds[v].values
            np.nan_to_num(block, copy=False, nan=l2w.no_data * scale_factor)
            vhandles[v][y0:y1, x0:x1] = block

        rb_block = ds["r_b_hat"].values  # (wavelength, y, x)
        np.nan_to_num(rb_block, copy=False, nan=l2w.no_data * scale_factor)
        rb_handle[:, y0:y1, x0:x1] = rb_block

        ds.close()

    l2w.out_ds.close()
    #
    # shutil.rmtree(output_dir)

    return 0

if __name__ == "__main__":




    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-l2rg.nc",
        # "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-l2r.nc",
        # "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-l2r.nc",
        "ACI-13A/el_jetski/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc",
        # "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-l2rg.nc",
    ]

    # bbox = None
    # Reduced bbox for dev
    # bbox = {"lon": (-64.37016, -64.3554), "lat": (49.80531, 49.81682)}
    bbox = {"lon": (-64.36871, -64.3615), "lat": (49.80857, 49.81336)}

    # stan_model_path = "/home/raphael/R/SABER/inst/stan/model_uc_spm.stan"
    # stan_model_path = "/home/raphael/R/SABER/inst/stan/model_hc_spm.stan"
    # stan_model_path = "/home/raphael/R/SABER/inst/stan/model_mc_spm.stan"
    stan_model_path = "/home/raphael/R/SABER/inst/stan/model_sc_spm.stan"
    # stan_model_path = "/home/raphael/R/SABER/inst/stan/model_msc_spm.stan"

    for image in images:

        l2w = ReveCube.from_reve_nc(os.path.join(image_dir, image))
        run_l2_inversion(
            l2w,
            stan_model_path,
            n_workers=None,
            chunk_size=256,
            method="optimize",
            # out_vars=("chl","a_g_440", "a_g_s", "spm", "a_nap_star", "a_nap_s", "bb_p_star", "bb_p_gamma", "h_w", "r_b_mix[1]","r_b_mix[2]", "r_b_mix[3]", "r_b_a", "sigma_model"),
            # out_vars=("chl","a_g_440", "a_g_s", "spm", "a_nap_star", "a_nap_s", "bb_p_star", "bb_p_gamma", "r_b_mix[1]","r_b_mix[2]", "r_b_mix[3]", "r_b_a", "sigma_model"),
            # out_vars=("chl","a_g_440", "a_g_s", "spm", "a_nap_s", "bb_p_gamma", "r_b_mix[1]","r_b_mix[2]", "r_b_mix[3]", "r_b_a", "sigma_model"),
            out_vars=("chl","a_g_440", "spm", "r_b_mix[1]","r_b_mix[2]", "r_b_mix[3]", "r_b_a", "sigma_model"),
            # out_vars=("chl","a_g_440", "spm", "h_w", "r_b_mix[1]","r_b_mix[2]", "r_b_mix[3]", "r_b_a", "sigma_model"),
            wavelength_filter = (400, 700),
            bbox = bbox
)
