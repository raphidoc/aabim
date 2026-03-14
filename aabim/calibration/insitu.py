"""
insitu.py — In-situ water reflectance data for system vicarious calibration.

Expected CSV format (long)
--------------------------
uuid        : unique identifier for each in-situ measurement
date_time   : ISO-8601 timestamp (UTC)
lat         : decimal latitude  (°N)
lon         : decimal longitude (°E)
wavelength  : wavelength (nm)
rho         : remote-sensing water reflectance ρ_w (dimensionless)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_REQUIRED_COLS = {"uuid", "date_time", "lat", "lon", "wavelength", "rho"}


class InSitu:
    """
    Long-format in-situ water reflectance data.

    Parameters
    ----------
    df : pd.DataFrame
        Validated long-format DataFrame with columns defined in
        ``_REQUIRED_COLS``.

    Examples
    --------
    >>> in_situ = InSitu.from_csv("rho_w_measurements.csv")
    >>> wl, rho = in_situ.rho_for_uuid("abc-123")
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_csv(cls, path: str | Path) -> "InSitu":
        """Load and validate in-situ data from a CSV file.

        Parameters
        ----------
        path : str or Path
            Path to a long-format CSV file.

        Returns
        -------
        InSitu
        """
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()

        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"In-situ CSV is missing required columns: {missing}. "
                f"Found: {set(df.columns)}"
            )

        df["date_time"] = pd.to_datetime(df["date_time"], utc=True)

        log.info(
            "Loaded %d rows (%d UUIDs) from %s",
            len(df),
            df["uuid"].nunique(),
            path,
        )
        return cls(df)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def df(self) -> pd.DataFrame:
        """Underlying long-format DataFrame."""
        return self._df

    @property
    def uuids(self) -> list:
        """Unique measurement identifiers."""
        return self._df["uuid"].unique().tolist()

    @property
    def wavelengths(self) -> np.ndarray:
        """Sorted array of wavelengths present in the dataset (nm)."""
        return np.sort(self._df["wavelength"].unique())

    # ------------------------------------------------------------------ #
    # Per-measurement accessors                                            #
    # ------------------------------------------------------------------ #

    def rho_for_uuid(self, uuid) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(wavelength, rho)`` arrays for *uuid*, sorted by wavelength.

        Parameters
        ----------
        uuid :
            Measurement identifier.

        Returns
        -------
        wavelength : np.ndarray
        rho        : np.ndarray
        """
        sub = self._df[self._df["uuid"] == uuid].sort_values("wavelength")
        return sub["wavelength"].values.astype(float), sub["rho"].values.astype(float)

    def location_for_uuid(self, uuid) -> dict:
        """Return ``{"lat", "lon", "date_time"}`` for *uuid* (first row).

        Parameters
        ----------
        uuid :
            Measurement identifier.

        Returns
        -------
        dict
        """
        row = self._df[self._df["uuid"] == uuid].iloc[0]
        return {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "date_time": row["date_time"],
        }

    def __repr__(self) -> str:
        return (
            f"InSitu(n_uuids={len(self.uuids)}, "
            f"wavelength=[{self.wavelengths.min():.0f}, {self.wavelengths.max():.0f}] nm)"
        )
