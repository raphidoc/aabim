"""
get_gmao.py — Download and interpolate MERRA-2 ancillary data for an image.

Downloads two consecutive hourly MERRA-2 files (MET + AER) from the NASA
Ocean Data server, caches them locally, then interpolates each field to the
image centre lat/lon and acquisition time.

Returns an xr.Dataset with scalar variables keyed by MERRA-2 field name:
  U10M, V10M, PS, TQV, TO3  (from MET)
  TOTEXTTAU                  (from AER)
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import xarray as xr

from aabim.ancillary.obpg.interp_gmao import interp_gmao
from aabim.ancillary.obpg import OBPGSession

log = logging.getLogger(__name__)

_SERVER = "oceandata.sci.gsfc.nasa.gov"


def get_gmao(image, anc_dir: str | Path | None = None) -> xr.Dataset:
    """Download and interpolate MERRA-2 data for *image*.

    Parameters
    ----------
    image : Image
        aabim Image object.  Must have ``acq_time_z``, ``center_lon``,
        ``center_lat`` populated (call ``image.expand_coordinate()`` first
        if needed).
    anc_dir : path, optional
        Local directory for caching downloaded files.
        Defaults to a sibling ``anc/`` directory next to the image file.

    Returns
    -------
    xr.Dataset
        Scalar dataset with MERRA-2 fields at the image location and time.
    """
    if image.center_lon is None or image.center_lat is None:
        image.expand_coordinate()

    anc_dir = Path(anc_dir) if anc_dir else Path(image.in_path).parent / "anc"
    anc_dir.mkdir(parents=True, exist_ok=True)

    dt         = image.acq_time_z
    center_lon = image.center_lon
    center_lat = image.center_lat

    log.info(
        "Fetching MERRA-2 for %s  %.3f°E  %.3f°N",
        dt.strftime("%Y-%m-%dT%H:%M:%SZ"), center_lon, center_lat,
    )

    met_files = _local_files(dt, "MET", anc_dir)
    aer_files = _local_files(dt, "AER", anc_dir)

    met_anc = interp_gmao(met_files, center_lon, center_lat, dt)
    aer_anc = interp_gmao(aer_files, center_lon, center_lat, dt)

    return xr.merge([met_anc, aer_anc])


# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #

def _bracket_hours(dt: datetime.datetime) -> tuple[str, str]:
    """Return the two HH strings bracketing *dt* for MERRA-2 file selection."""
    hh0 = str(dt.hour).zfill(2)
    if dt.hour < 23:
        hh1 = str(dt.hour + 1).zfill(2)
        yyyymmdd1 = dt.strftime("%Y%m%d")
    else:
        hh1 = "00"
        yyyymmdd1 = (dt + datetime.timedelta(days=1)).strftime("%Y%m%d")
    return (dt.strftime("%Y%m%d"), hh0), (yyyymmdd1, hh1)


def _local_files(dt: datetime.datetime, kind: str, anc_dir: Path) -> list[str]:
    """Ensure both bracketing hourly MERRA-2 *kind* files are cached locally."""
    paths = []
    for yyyymmdd, hh in _bracket_hours(dt):
        fname = f"GMAO_MERRA2.{yyyymmdd}T{hh}0000.{kind}.nc"
        fpath = anc_dir / fname
        if not fpath.exists():
            log.info("Downloading %s", fname)
            status = OBPGSession.httpdl(
                _SERVER,
                f"/ob/getfile/{fname}",
                localpath=str(anc_dir),
                outputfilename=fname,
                uncompress=False,
                verbose=2,
            )
            if status in (400, 401, 403, 404, 416):
                raise IOError(
                    f"Failed to download {fname} from {_SERVER} (HTTP {status})"
                )
        else:
            log.debug("Cache hit: %s", fpath)
        paths.append(str(fpath))
    return paths
