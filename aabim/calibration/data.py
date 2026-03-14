"""
data.py — Matchup database: measured vs predicted TOA reflectance.

Design
------
CalibrationData.compute() co-locates image pixels with in-situ ρ_w measurements,
runs the atmospheric forward model using AerLUT + GasLUT, and returns a
per-pixel matchup DataFrame.

To aggregate across multiple flight lines, use CalibrationData.concat().

Forward model (from vicarious_calibration.py)
---------------------------------------------
    ρ_s     = ρ_w + ρ_sky_glint          (surface reflectance, water + glint)
    ρ_t_hat = t_g × (ρ_path + t_ra × ρ_s / (1 − s_ra × ρ_s))

The AOD used in the LUT query is read from the image ancillary data
(``image.in_ds[AOD_VAR]``), which is expected to come from MERRA-2 reanalysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from rasterio.windows import Window
from tqdm import tqdm

from aabim.calibration.insitu import InSitu
from aabim.data.atmospheric.atmospheric import AerLUT, GasLUT

log = logging.getLogger(__name__)

# Variable name for AOD in the image ancillary dataset (MERRA-2 convention)
AOD_VAR = "aerosol_optical_depth_at_550_nm"


class CalibrationData:
    """
    Per-pixel matchup database of observed (ρ_t) and predicted (ρ_t_hat) TOA
    reflectance, used to fit a CalibrationModel.

    Use ``CalibrationData.compute()`` to build from a single image + InSitu,
    then ``CalibrationData.concat()`` to aggregate across multiple flight lines.

    Stored columns
    --------------
    uuid, image_name, land, x, y, wavelength,
    rho_t, rho_t_hat, rho_w, sky_glint, rho_path, t_ra, s_ra, t_g

    Examples
    --------
    >>> cd1 = CalibrationData.compute(image1, in_situ)
    >>> cd2 = CalibrationData.compute(image2, in_situ)
    >>> cd  = CalibrationData.concat([cd1, cd2])
    >>> cd.summary()
    >>> cd.save("matchups.nc")
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def compute(
        cls,
        image,
        in_situ: InSitu,
        window_size: int = 7,
        land: bool = False,
        max_time_diff: float = 12.0,
    ) -> "CalibrationData":
        """Build the matchup database for one image.

        Parameters
        ----------
        image : Image
            L1 image with ancillary atmospheric data (AOD, pressure, etc.).
        in_situ : InSitu
            In-situ water reflectance data.
        window_size : int
            Spatial averaging window in pixels (must be odd).
        land : bool
            If True, sky glint is not added to ρ_w (land target calibration).
        max_time_diff : float
            Maximum time difference (hours) allowed between in-situ and image
            acquisition for a match to be accepted.

        Returns
        -------
        CalibrationData
        """
        log.info("Building CalibrationData for image '%s'", image.image_name)

        window_dict = cls._find_windows(image, in_situ, max_time_diff, window_size)
        if not window_dict:
            raise ValueError(
                "No matchups found after spatial/temporal filtering. "
                "Check max_time_diff and that in-situ coordinates overlap the image."
            )
        log.info("Found %d matchup windows", len(window_dict))

        aer_lut = AerLUT.from_image(image)
        gas_lut = GasLUT.from_image(image)

        if AOD_VAR not in image.in_ds:
            raise KeyError(
                f"AOD variable '{AOD_VAR}' not found in image dataset. "
                f"Available variables: {list(image.in_ds.data_vars)}"
            )
        aod550 = float(image.in_ds[AOD_VAR].values)
        wavelength = np.asarray(image.wavelength, dtype=np.float64)

        frames = []

        for uuid, window in tqdm(window_dict.items(), desc="computing ρ_t_hat"):
            y_slice = slice(int(window.row_off), int(window.row_off + window.height))
            x_slice = slice(int(window.col_off), int(window.col_off + window.width))
            image_sub = image.in_ds.isel(x=x_slice, y=y_slice)

            # --- LUT queries ------------------------------------------------
            ra = aer_lut.get_ra(image, window, aod550)
            gas = gas_lut.get_gas(image, window)

            rho_path = ra["rho_path"].values  # (n_wl, ny, nx)
            t_ra = ra["trans_ra"].values
            s_ra = ra["spherical_albedo_ra"].values
            sky_glint = ra["sky_glint_ra"].values
            t_g = gas["t_gas"].values

            # --- In-situ ρ_w interpolated to image wavelengths --------------
            wl_insitu, rho_insitu = in_situ.rho_for_uuid(uuid)
            rho_w_1d = np.interp(wavelength, wl_insitu, rho_insitu)

            n_wl = len(wavelength)
            ny = image_sub.sizes["y"]
            nx = image_sub.sizes["x"]

            # Broadcast ρ_w to (n_wl, ny, nx) without Python loops
            rho_w = np.broadcast_to(
                rho_w_1d[:, np.newaxis, np.newaxis], (n_wl, ny, nx)
            ).copy()

            # --- Forward model ----------------------------------------------
            rho_s = rho_w + sky_glint if not land else rho_w
            rho_t_hat = t_g * (rho_path + t_ra * (rho_s / (1.0 - s_ra * rho_s)))

            rho_t = image_sub["rho_at_sensor"].values  # (n_wl, ny, nx)

            # --- Build coordinate grids (vectorised) ------------------------
            x_coords = image_sub.x.values  # (nx,)
            y_coords = image_sub.y.values  # (ny,)

            # Meshgrid over space, then tile over wavelength
            y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
            # Shapes: y_grid / x_grid → (ny, nx)
            wl_grid = np.tile(wavelength[:, np.newaxis, np.newaxis], (1, ny, nx))

            def _flat(arr):
                return arr.ravel()

            frame = pd.DataFrame(
                {
                    "uuid": uuid,
                    "image_name": image.image_name,
                    "land": land,
                    "x": np.tile(x_grid[np.newaxis], (n_wl, 1, 1)).ravel(),
                    "y": np.tile(y_grid[np.newaxis], (n_wl, 1, 1)).ravel(),
                    "wavelength": _flat(wl_grid),
                    "rho_t": _flat(rho_t),
                    "rho_t_hat": _flat(rho_t_hat),
                    "rho_w": _flat(rho_w),
                    "sky_glint": _flat(sky_glint),
                    "rho_path": _flat(rho_path),
                    "t_ra": _flat(t_ra),
                    "s_ra": _flat(s_ra),
                    "t_g": _flat(t_g),
                }
            )
            frames.append(frame)

        df = pd.concat(frames, ignore_index=True)
        log.info(
            "CalibrationData: %d rows, %d matchups, %d images",
            len(df),
            df["uuid"].nunique(),
            df["image_name"].nunique(),
        )
        return cls(df)

    @classmethod
    def concat(cls, datasets: list["CalibrationData"]) -> "CalibrationData":
        """Combine CalibrationData from multiple images into one database.

        Parameters
        ----------
        datasets : list[CalibrationData]

        Returns
        -------
        CalibrationData
        """
        df = pd.concat([d._df for d in datasets], ignore_index=True)
        log.info(
            "CalibrationData.concat: %d rows, %d matchups, %d images",
            len(df),
            df["uuid"].nunique(),
            df["image_name"].nunique(),
        )
        return cls(df)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationData":
        """Load a CalibrationData previously saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Path to a NetCDF file written by :meth:`save`.

        Returns
        -------
        CalibrationData
        """
        ds = xr.open_dataset(path)
        df = ds.to_dataframe().reset_index()
        log.info("CalibrationData loaded from %s (%d rows)", path, len(df))
        return cls(df)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save to NetCDF.

        Parameters
        ----------
        path : str or Path
            Output path (``*.nc``).
        """
        ds = self._to_dataset()
        ds.to_netcdf(path)
        log.info("CalibrationData saved → %s", path)

    # ------------------------------------------------------------------ #
    # Properties / inspection                                              #
    # ------------------------------------------------------------------ #

    @property
    def df(self) -> pd.DataFrame:
        """Underlying long-format DataFrame."""
        return self._df

    @property
    def wavelengths(self) -> np.ndarray:
        """Sorted wavelength array (nm)."""
        return np.sort(self._df["wavelength"].unique())

    def summary(self) -> pd.DataFrame:
        """Per-wavelength statistics: bias, RMSE, and observation count.

        Returns
        -------
        pd.DataFrame
            Index = wavelength (nm).  Columns: ``bias``, ``rmse``, ``n``.
        """
        grp = self._df.groupby("wavelength")
        bias = grp.apply(lambda g: (g["rho_t_hat"] - g["rho_t"]).mean())
        rmse = grp.apply(lambda g: np.sqrt(((g["rho_t_hat"] - g["rho_t"]) ** 2).mean()))
        n = grp["rho_t"].count()
        return pd.DataFrame({"bias": bias, "rmse": rmse, "n": n})

    def plot(self, model=None, save_path=None) -> None:
        """Scatter plot of ρ_t_hat vs ρ_t per wavelength.

        Parameters
        ----------
        model : CalibrationModel, optional
            If provided, overlays the fitted calibration line on each panel.
        save_path : str or Path, optional
            If provided, saves the figure to this path instead of displaying it.
        """
        import matplotlib.pyplot as plt

        wls = self.wavelengths
        ncols = min(8, len(wls))
        nrows = (len(wls) + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5), squeeze=False
        )
        axes_flat = axes.ravel()

        use_gain = model is not None and "gain" in model.coeffs

        for i, wl in enumerate(wls):
            sub = self._df[self._df["wavelength"] == wl].dropna(
                subset=["rho_t", "rho_t_hat"]
            )
            ax = axes_flat[i]
            if sub.empty:
                ax.set_visible(False)
                continue
            ax.scatter(sub["rho_t"], sub["rho_t_hat"], s=4, alpha=0.5)
            lim = max(sub["rho_t"].max(), sub["rho_t_hat"].max()) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1")

            if model is not None:
                c = model.coeffs.sel(wavelength=wl, method="nearest")
                x_fit = np.linspace(0, lim, 100)
                if use_gain:
                    gain = float(c["gain"])
                    if not np.isnan(gain):
                        ax.plot(x_fit, gain * x_fit, "r-", lw=1.2, label="calibration")
                else:
                    a, b = float(c["a"]), float(c["b"])
                    if not (np.isnan(a) or np.isnan(b)):
                        ax.plot(x_fit, a * x_fit + b, "r-", lw=1.2, label="calibration")

            ax.set_title(f"{wl:.0f} nm", fontsize=8)
            ax.set_xlabel("ρ_t measured", fontsize=7)
            ax.set_ylabel("ρ_t predicted", fontsize=7)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("ρ_t_hat vs ρ_t  (predicted vs measured)")
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Calibration plot saved → %s", save_path)
        else:
            plt.show()

    def __repr__(self) -> str:
        return (
            f"CalibrationData("
            f"n_rows={len(self._df)}, "
            f"n_matchups={self._df['uuid'].nunique()}, "
            f"n_images={self._df['image_name'].nunique()})"
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _to_dataset(self) -> xr.Dataset:
        """Convert to xr.Dataset (dim = observation) for NetCDF storage."""
        df = self._df.reset_index(drop=True)
        df.index.name = "observation"
        ds = xr.Dataset.from_dataframe(df)
        ds.attrs["description"] = "aabim CalibrationData matchup database"
        return ds

    @staticmethod
    def _find_windows(
        image,
        in_situ: InSitu,
        max_time_diff: float,
        window_size: int,
    ) -> dict:
        """Return ``{uuid: Window}`` for in-situ points co-located with *image*.

        Applies temporal filtering (``max_time_diff``) and nearest-pixel
        lookup in the projected image coordinate system.

        Parameters
        ----------
        image : Image
        in_situ : InSitu
        max_time_diff : float   hours
        window_size : int       must be odd

        Returns
        -------
        dict[uuid, rasterio.windows.Window]
        """
        df = in_situ.df[["uuid", "date_time", "lat", "lon"]].drop_duplicates()

        # --- Temporal filter ------------------------------------------------
        df = df[
            abs(df["date_time"] - image.acq_time_z)
            < pd.Timedelta(max_time_diff, unit="h")
        ]
        if df.empty:
            log.warning(
                "No in-situ points within %.1f h of image acquisition time %s",
                max_time_diff,
                image.acq_time_z,
            )
            return {}

        # --- Project lat/lon to image CRS -----------------------------------
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", image.CRS.to_epsg(), always_xy=True
        )
        df = df.copy()
        df["x_proj"], df["y_proj"] = transformer.transform(
            df["lon"].values, df["lat"].values
        )

        x_coords = image.in_ds.x.values
        y_coords = image.in_ds.y.values
        half = window_size // 2

        window_dict = {}
        for _, row in df.iterrows():
            xi = int(np.abs(x_coords - row["x_proj"]).argmin())
            yi = int(np.abs(y_coords - row["y_proj"]).argmin())

            col_start = max(0, xi - half)
            row_start = max(0, yi - half)
            col_stop = min(len(x_coords), xi + half + 1)
            row_stop = min(len(y_coords), yi + half + 1)

            win = Window(
                col_off=col_start,
                row_off=row_start,
                width=col_stop - col_start,
                height=row_stop - row_start,
            )
            window_dict[row["uuid"]] = win
            log.debug("UUID %s → Window %s", row["uuid"], win)

        log.info(
            "%d/%d in-situ points matched (temporal+spatial filter)",
            len(window_dict),
            in_situ.df["uuid"].nunique(),
        )
        return window_dict
