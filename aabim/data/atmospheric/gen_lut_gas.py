"""Generate the gas transmittance LUT using 6S.

Calls 6S (sixs_json conda-forge package) once per grid point with user-defined
water vapour and ozone (IDATM=8, IAER=0), collects the JSON gas-transmittance
output, and writes a CF-compliant NetCDF.

Grid dimensions
---------------
sol_zen          : solar zenith angle [deg]
view_zen         : sensor zenith angle [deg]
relative_azimuth : relative azimuth (sun→sensor) [deg] — full 0–180 range
water            : atmospheric water vapour column [g cm⁻²]
ozone            : ozone column [atm-cm]
target_pressure  : surface pressure at target [mbar]
sensor_altitude  : sensor altitude above geoid [m]   (6S input: −alt/1000 km)
wavelength       : [nm]  (passed to 6S in µm)

Usage
-----
python gen_lut_gas.py --output lut_gas.nc --n-workers 400
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

SOL_ZEN   = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]           # deg
VIEW_ZEN  = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]            # deg
REL_AZI   = list(np.arange(0, 181, 10, dtype=float))  # 0–180 deg, 19 pts
WATER     = [0.0, 1.0, 2.0, 3.0, 4.0]               # g cm⁻²
OZONE     = [0.0, 0.25, 0.35, 0.50]                  # atm-cm
PRESSURE  = [750.0, 1013.0]                           # mbar
ALT_M     = [2000.0, 3000.0, 4000.0]                          # m (6S: −alt/1000 km)
WL_NM     = list(np.round(np.arange(340, 1000, 10, dtype=float), 1))  # nm

# ---------------------------------------------------------------------------
# 6S worker
# ---------------------------------------------------------------------------


def _run_sixs(task: tuple) -> tuple[int, dict]:
    """Run one 6S call; return (flat_index, result_dict).

    Uses IDATM=8 (user H2O + O3) and IAER=0 (no aerosol) to isolate gas
    transmittance.
    """
    idx, sixs_bin, sol_zen, view_zen, raa, water, ozone, pressure, alt_m, wl_nm = task

    wl_um = wl_nm / 1000.0
    alt_km_neg = -(alt_m / 1000.0)

    sixs_input = (
        "\n"
        "0\n"    # IGEOM user-defined
        f"{sol_zen} 0.0 {view_zen} {raa} 1 1\n"
        "8\n"    # IDATM user-defined H2O + O3
        f"{water}\n"
        f"{ozone}\n"
        "0\n"    # IAER no aerosol
        "-1\n"   # visibility (unused when IAER=0)
        f"{pressure}\n"
        f"{alt_km_neg}\n"
        "-1.0 -1.0\n"
        "-1.0\n"
        "-1\n"   # IWAVE monochromatic
        f"{wl_um}\n"
        "0\n"    # INHOMO
        "0\n"    # IDIREC
        "0\n"    # IGROUN
        "0\n"    # surface reflectance
        "-1\n"   # IRAPP
    )

    proc = subprocess.run(
        [sixs_bin], input=sixs_input, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"6S failed (idx={idx}, sol={sol_zen}, view={view_zen}, "
            f"raa={raa}, H2O={water}, O3={ozone}, wl={wl_nm}): {proc.stderr[:300]}"
        )

    out = json.loads(proc.stdout)

    t_gas = (
        float(out["ozone_gas_trans_total"])
        * float(out["water_gas_trans_total"])
    )

    return idx, {"t_gas": t_gas}


# ---------------------------------------------------------------------------
# LUT assembly
# ---------------------------------------------------------------------------


def generate_gas_lut(
    output_path: str | Path,
    n_workers: int = -1,
    complevel: int = 4,
    sixs_bin: str = "sixs_json",
    # Grid overrides — None means use the module-level defaults above
    sol_zen: list | None = None,
    view_zen: list | None = None,
    rel_azi: list | None = None,
    water: list | None = None,
    ozone: list | None = None,
    pressure: list | None = None,
    alt_m: list | None = None,
    wl_nm: list | None = None,
) -> Path:
    """Generate the gas transmittance LUT and write it to *output_path*.

    Parameters
    ----------
    output_path : destination NetCDF file
    n_workers   : parallel threads (-1 → cpu_count)
    complevel   : zlib compression level (0–9)
    sixs_bin    : sixs_json binary name or path (default: "sixs_json" from PATH)
    sol_zen … wl_nm : grid axis overrides; None → use module-level defaults.
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
        water    if water    is not None else WATER,
        ozone    if ozone    is not None else OZONE,
        pressure if pressure is not None else PRESSURE,
        alt_m    if alt_m    is not None else ALT_M,
        wl_nm    if wl_nm    is not None else WL_NM,
    ]
    dim_names = ["sol_zen", "view_zen", "relative_azimuth",
                 "water", "ozone", "target_pressure", "sensor_altitude", "wavelength"]
    shape = [len(d) for d in dims]
    n_total = 1
    for s in shape:
        n_total *= s

    buf_tgas = np.full(n_total, np.nan, dtype=np.float32)

    tasks = [
        (i, sixs_bin, *combo)
        for i, combo in enumerate(itertools.product(*dims))
    ]

    print(f"Gas LUT: {n_total:,} calls, {n_workers} threads")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_run_sixs, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=n_total, desc="GasLUT"):
            idx, res = fut.result()
            buf_tgas[idx] = res["t_gas"]

    coords = {n: np.array(d, dtype=np.float32)
              for n, d in zip(dim_names, dims)}

    ds = xr.Dataset(
        {
            "global_gas_trans_total": (
                dim_names, buf_tgas.reshape(shape)
            ),
        },
        coords=coords,
    )

    ds["sol_zen"].attrs.update(
        standard_name="solar_zenith_angle", units="degree")
    ds["view_zen"].attrs.update(
        standard_name="sensor_zenith_angle", units="degree")
    ds["relative_azimuth"].attrs.update(
        standard_name="angle_of_rotation_from_solar_azimuth_to_platform_azimuth",
        long_name="relative_azimuth_angle", units="degree")
    ds["water"].attrs.update(
        standard_name="atmosphere_mass_content_of_water_vapor", units="g cm-2")
    ds["ozone"].attrs.update(
        standard_name="equivalent_thickness_at_stp_of_atmosphere_ozone_content",
        units="atm-cm")
    ds["target_pressure"].attrs.update(units="mbar")
    ds["sensor_altitude"].attrs.update(
        standard_name="altitude",
        long_name="sensor altitude above geoid, positive upward",
        units="m")
    ds["wavelength"].attrs.update(
        standard_name="radiation_wavelength", units="nm")

    enc = {"global_gas_trans_total": {"zlib": True, "complevel": complevel}}
    ds.to_netcdf(output_path, encoding=enc)
    print(f"Written → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",    default="lut_gas.nc",
                   help="Output NetCDF path (default: lut_gas.nc)")
    p.add_argument("--n-workers", type=int, default=-1,
                   help="Parallel threads (-1 = cpu_count)")
    p.add_argument("--complevel", type=int, default=4,
                   help="zlib compression level 0-9 (default: 4)")
    p.add_argument("--sixs",      default="sixs_json",
                   help="sixs_json binary name or path (default: sixs_json)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_gas_lut(
        output_path=args.output,
        n_workers=args.n_workers,
        complevel=args.complevel,
        sixs_bin=args.sixs,
    )
