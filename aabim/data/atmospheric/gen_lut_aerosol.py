"""Generate the aerosol radiative-transfer LUT using 6S.

Calls 6S (sixs_json conda-forge package) once per grid point, collects JSON
output, computes sky-glint reflectance via the Fresnel-phase-function formula,
and writes a CF-compliant NetCDF.

Grid dimensions
---------------
sol_zen          : solar zenith angle [deg]
view_zen         : sensor zenith angle [deg]
relative_azimuth : relative azimuth (sun→sensor) [deg] — full 0–180 range
aot550           : aerosol optical depth at 550 nm
target_pressure  : surface pressure at target [mbar]
sensor_altitude  : sensor altitude above geoid [m]   (6S input: −alt/1000 km)
wavelength       : [nm]  (passed to 6S in µm)

Usage
-----
python gen_lut_aerosol.py --output lut_aerosol.nc --n-workers 400

References
----------
Vermote et al. (1997) Second Simulation of the Satellite Signal in the Solar
    Spectrum (6S). IEEE TGRS 35(3).
Quan & Fry (1995) Empirical equation for the index of refraction of seawater.
    Appl. Opt. 34(18).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Grid definition — edit here to change LUT resolution / bounds
# ---------------------------------------------------------------------------

SOL_ZEN   = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]              # deg
VIEW_ZEN  = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]               # deg
REL_AZI   = list(np.arange(0, 181, 10, dtype=float))     # 0–180 deg, 19 pts
AOT550    = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]   # unitless
PRESSURE  = [750.0, 1013.0]                              # mbar
ALT_M     = [2000.0, 3000.0, 4000.0]                             # m (6S: −alt/1000 km)
WL_NM     = list(np.round(np.arange(340, 1000, 10, dtype=float), 1))  # nm

# ---------------------------------------------------------------------------
# Physics helpers (no external dependencies)
# ---------------------------------------------------------------------------


def _water_refractive_index(
    wl_nm: float, salinity: float = 30.0, temperature: float = 12.0
) -> float:
    """Quan & Fry (1995) seawater refractive index. λ must be in nm."""
    return (
        1.31405
        + (1.779e-4 - 1.05e-6 * temperature + 1.6e-8 * temperature**2) * salinity
        - 2.02e-6 * temperature**2
        + (15.868 + 0.01155 * salinity - 0.00423 * temperature) / wl_nm
        - 4382.0 / wl_nm**2
        + 1.1455e6 / wl_nm**3
    )


def _fresnel_reflectance(theta_rad: float, n: float) -> float:
    """Unpolarized Fresnel reflectance at air–water interface."""
    sin_t = np.sin(theta_rad) / n
    if abs(sin_t) >= 1.0:
        return 1.0
    cos_i = np.cos(theta_rad)
    cos_t = np.sqrt(max(0.0, 1.0 - sin_t**2))
    rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t)) ** 2
    rp = ((n * cos_i - cos_t) / (n * cos_i + cos_t)) ** 2
    return 0.5 * (rs + rp)


# ---------------------------------------------------------------------------
# 6S worker
# ---------------------------------------------------------------------------


def _run_sixs(task: tuple) -> tuple[int, dict]:
    """Run one 6S call; return (flat_index, result_dict).

    sensor_altitude_m is converted to −alt/1000 km for the XPP 6S parameter.
    """
    idx, sixs_bin, sol_zen, view_zen, raa, aot, pressure, alt_m, wl_nm = task

    wl_um = wl_nm / 1000.0
    alt_km_neg = -(alt_m / 1000.0)  # 6S convention: negative km = above target

    sixs_input = (
        "\n"
        "0\n"  # IGEOM user-defined
        f"{sol_zen} 0.0 {view_zen} {raa} 1 1\n"
        "0\n"  # IDATM no gas absorption (pure aerosol LUT)
        "2\n"  # IAER maritime
        "0\n"  # visibility: driven by AOD below
        f"{aot}\n"
        f"{pressure}\n"
        f"{alt_km_neg}\n"
        "-1.0 -1.0\n"  # UH2O UO3 below sensor
        "-1.0\n"       # taer below sensor
        "-1\n"         # IWAVE monochromatic
        f"{wl_um}\n"
        "0\n"  # INHOMO
        "0\n"  # IDIREC
        "0\n"  # IGROUN
        "0\n"  # surface reflectance
        "-1\n" # IRAPP
    )

    proc = subprocess.run(
        [sixs_bin], input=sixs_input, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"6S failed (idx={idx}, sol={sol_zen}, view={view_zen}, "
            f"raa={raa}, aot={aot}, wl={wl_nm}): {proc.stderr[:300]}"
        )

    out = json.loads(proc.stdout)

    tau_ra   = float(out["optical_depth_total_total"])
    phase_ra = float(out["phase_function_I_total"])
    theta_s  = float(out["solar_zenith_[degree]"]) * np.pi / 180.0
    theta_v  = float(out["view_zenith_[degree]"])  * np.pi / 180.0
    wl_out   = float(out["monochromatic_wavelength_[um]"]) * 1e3

    n_w  = _water_refractive_index(wl_out)
    fr_s = _fresnel_reflectance(theta_s, n_w)
    fr_v = _fresnel_reflectance(theta_v, n_w)
    denom = 4.0 * max(np.cos(theta_s), 1e-6) * max(np.cos(theta_v), 1e-6)
    sky_glint = tau_ra * (fr_s + fr_v) * phase_ra / denom

    return idx, {
        "rho_path":  float(out["atmospheric_reflectance_at_sensor"]),
        "t_total":   float(out["total_scattering_trans_total"]),
        "s_total":   float(out["spherical_albedo_total"]),
        "sky_glint": sky_glint,
    }


# ---------------------------------------------------------------------------
# LUT assembly
# ---------------------------------------------------------------------------


def generate_aerosol_lut(
    output_path: str | Path,
    n_workers: int = -1,
    complevel: int = 4,
    sixs_bin: str = "sixs_json",
    # Grid overrides — None means use the module-level defaults above
    sol_zen: list | None = None,
    view_zen: list | None = None,
    rel_azi: list | None = None,
    aot550: list | None = None,
    pressure: list | None = None,
    alt_m: list | None = None,
    wl_nm: list | None = None,
) -> Path:
    """Generate the aerosol LUT and write it to *output_path*.

    Parameters
    ----------
    output_path : destination NetCDF file
    n_workers   : parallel threads (-1 → cpu_count)
    complevel   : zlib compression level (0–9)
    sixs_bin    : sixs_json binary name or path (default: "sixs_json" from PATH)
    sol_zen … wl_nm : grid axis overrides; None → use module-level defaults.
        Useful for tests or custom resolutions without editing the source.
    """
    import os
    import shutil
    import sys

    if n_workers < 1:
        n_workers = os.cpu_count() or 4

    # Resolve binary: PATH first, then same conda env as the running Python.
    sixs_bin = shutil.which(sixs_bin) or str(Path(sys.executable).parent / "sixs_json")
    if not Path(sixs_bin).exists():
        raise FileNotFoundError(
            f"sixs_json binary not found. Install with: conda install -c conda-forge sixs_json"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dims = [
        sol_zen  if sol_zen  is not None else SOL_ZEN,
        view_zen if view_zen is not None else VIEW_ZEN,
        rel_azi  if rel_azi  is not None else REL_AZI,
        aot550   if aot550   is not None else AOT550,
        pressure if pressure is not None else PRESSURE,
        alt_m    if alt_m    is not None else ALT_M,
        wl_nm    if wl_nm    is not None else WL_NM,
    ]
    dim_names = ["sol_zen", "view_zen", "relative_azimuth",
                 "aot550", "target_pressure", "sensor_altitude", "wavelength"]
    shape = [len(d) for d in dims]
    n_total = 1
    for s in shape:
        n_total *= s

    # Flat result buffers
    buf = {k: np.full(n_total, np.nan, dtype=np.float32)
           for k in ("rho_path", "t_total", "s_total", "sky_glint")}

    tasks = [
        (i, sixs_bin, *combo)
        for i, combo in enumerate(itertools.product(*dims))
    ]

    print(f"Aerosol LUT: {n_total:,} calls, {n_workers} threads")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_sixs, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=n_total, desc="AerLUT"):
            idx, res = fut.result()
            buf["rho_path"][idx]  = res["rho_path"]
            buf["t_total"][idx]   = res["t_total"]
            buf["s_total"][idx]   = res["s_total"]
            buf["sky_glint"][idx] = res["sky_glint"]

    # Reshape to (sol_zen, view_zen, raa, aot, pressure, alt, wavelength)
    def reshape(arr):
        return arr.reshape(shape)

    coords = {n: np.array(d, dtype=np.float32)
              for n, d in zip(dim_names, dims)}

    ds = xr.Dataset(
        {
            "atmospheric_reflectance_at_sensor": (dim_names, reshape(buf["rho_path"])),
            "total_scattering_trans_total":      (dim_names, reshape(buf["t_total"])),
            "spherical_albedo_total":            (dim_names, reshape(buf["s_total"])),
            "sky_glint_total":                   (dim_names, reshape(buf["sky_glint"])),
        },
        coords=coords,
    )

    # CF attributes
    ds["sol_zen"].attrs.update(
        standard_name="solar_zenith_angle", units="degree")
    ds["view_zen"].attrs.update(
        standard_name="sensor_zenith_angle", units="degree")
    ds["relative_azimuth"].attrs.update(
        standard_name="angle_of_rotation_from_solar_azimuth_to_platform_azimuth",
        long_name="relative_azimuth_angle", units="degree")
    ds["aot550"].attrs.update(
        standard_name="aerosol_optical_thickness_at_550_nm", units="1")
    ds["target_pressure"].attrs.update(units="mbar")
    ds["sensor_altitude"].attrs.update(
        standard_name="altitude",
        long_name="sensor altitude above geoid, positive upward",
        units="m")
    ds["wavelength"].attrs.update(
        standard_name="radiation_wavelength", units="nm")

    enc = {v: {"zlib": True, "complevel": complevel}
           for v in ds.data_vars}
    ds.to_netcdf(output_path, encoding=enc)
    print(f"Written → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",    default="lut_aerosol.nc",
                   help="Output NetCDF path (default: lut_aerosol.nc)")
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Parallel threads (-1 = cpu_count)")
    p.add_argument("--complevel", type=int, default=4,
                   help="zlib compression level 0-9 (default: 4)")
    p.add_argument("--sixs",      default="sixs_json",
                   help="sixs_json binary name or path (default: sixs_json)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_aerosol_lut(
        output_path=args.output,
        n_workers=args.n_workers,
        complevel=args.complevel,
        sixs_bin=args.sixs,
    )
