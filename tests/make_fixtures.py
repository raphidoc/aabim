"""One-time script to create small test fixtures from the raw ACI13 .pix data.

Run locally (requires access to /D/Data/WISE/):

    python tests/make_fixtures.py

The output NetCDF is committed to tests/data/ and used by all tests via
conftest.py.

Pipeline
--------
1. Read the raw WISE .dpix directory with ``Pix`` (computes full geometry).
2. Crop to the test bounding box (in memory, before any disk write).
3. Write directly to tests/data/ACI13_bbox_l1c.nc  (no ancillary data).

Ancillary data (MERRA-2) is added separately via ``add_ancillary()`` during
processing and is not baked into this fixture.
"""

# import os
# import sys
# # Ensure CONDA_PREFIX is set so CuPy (imported transitively via zarr) can
# # locate the CUDA libraries.  This is a no-op when the env is activated
# # normally; it only matters when the interpreter is invoked directly (e.g.
# # from a VSCode "Run File" button) without conda activate.
# if "CONDA_PREFIX" not in os.environ:
#     os.environ["CONDA_PREFIX"] = sys.prefix

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(name)s — %(message)s")

from pathlib import Path

from aabim.converter.wise.read_pix import Pix

# Path to the raw WISE L1A .dpix directory
PIX_DIR = "/D/Data/WISE/ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.dpix"
BBOX    = {"lon": (-64.36871, -64.3615), "lat": (49.80857, 49.81336)}
OUT_DIR = Path(__file__).parent / "data"

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    pix = Pix(PIX_DIR, bbox=BBOX)

    out = OUT_DIR / "ACI13_bbox_l1c.nc"
    pix.to_aabim_nc(str(out))

    print(
        f"Fixture saved → {out}\n"
        f"  {pix.n_rows} rows × {pix.n_cols} cols × {len(pix.wavelength)} bands\n"
        f"  No ancillary data — run add_ancillary() before processing."
    )
