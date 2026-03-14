"""
model.py — Calibration model hierarchy for System Vicarious Calibration.

Design
------
* CalibrationModel  — abstract base (fit / apply / save / load / summary / plot)
* RatioModel        — gain[λ] = median(ρ_t_hat / ρ_t)
* OLSModel          — per-wavelength OLS:  ρ_t_hat = a·ρ_t + b  (statsmodels)
* SMAModel          — per-wavelength SMA:  ρ_t_hat = a·ρ_t + b  (pylr2 Type-II)

Application (apply)
-------------------
    RatioModel : ρ_t_cal = gain[λ]  · ρ_t
    OLSModel   : ρ_t_cal = a[λ]    · ρ_t + b[λ]
    SMAModel   : ρ_t_cal = a[λ]    · ρ_t + b[λ]

The apply() method operates on any array with wavelength on axis 0,
making it compatible with both scalar spectra and full tiled images.

Persistence
-----------
All models are saved to / loaded from CSV.  The file format is::

    # model_name: ratio
    # metadata: {"n_matchups": 10, ...}
    wavelength,gain,gain_std
    361.51,1.023,0.012
    …

Comment lines (``#``) carry the model type and provenance metadata;
the data block is a plain CSV with ``wavelength`` as the first column
and one column per coefficient.

Coefficients produced by external software (R, Excel, …) can be loaded
without the comment header by calling the concrete class directly::

    model = RatioModel.load("gains_from_r.csv")   # columns: wavelength, gain

Extending
---------
Subclass CalibrationModel, set a unique ``model_name`` class attribute,
and implement ``fit()`` and ``apply()``.  The subclass is automatically
registered in ``_MODEL_REGISTRY`` and can be loaded with
``CalibrationModel.load(path)``.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# Populated automatically by __init_subclass__
_MODEL_REGISTRY: dict[str, type["CalibrationModel"]] = {}


class CalibrationModel(ABC):
    """Abstract base for SVC calibration models.

    Parameters
    ----------
    wavelength : np.ndarray, shape (n_wl,)
        Wavelengths corresponding to the fitted coefficients (nm).
    coeffs : xr.Dataset
        Model coefficients indexed by ``wavelength``.
        Variables depend on the subclass (see concrete classes below).
    metadata : dict
        Fit provenance: n_matchups, n_images, statistics, etc.
        Stored as a JSON attribute in the saved NetCDF.
    """

    model_name: ClassVar[str]

    def __init__(
        self,
        wavelength: np.ndarray,
        coeffs: xr.Dataset,
        metadata: dict,
    ) -> None:
        self.wavelength = wavelength
        self.coeffs     = coeffs
        self.metadata   = metadata

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "model_name"):
            _MODEL_REGISTRY[cls.model_name] = cls

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    @abstractmethod
    def fit(cls, cal_data) -> "CalibrationModel":
        """Fit the model to a :class:`~aabim.calibration.data.CalibrationData`.

        Parameters
        ----------
        cal_data : CalibrationData

        Returns
        -------
        CalibrationModel
        """
        ...

    @abstractmethod
    def apply(self, rho_t: np.ndarray) -> np.ndarray:
        """Apply calibration to a reflectance array.

        Parameters
        ----------
        rho_t : np.ndarray
            Array with **wavelength on axis 0**, e.g. shape
            ``(n_wl,)``, ``(n_wl, ny, nx)``.

        Returns
        -------
        np.ndarray  same shape as *rho_t*
        """
        ...

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save coefficients and metadata to a CSV file.

        The file starts with comment lines carrying the model name and
        provenance metadata, followed by a standard CSV table with
        ``wavelength`` as the first column.

        Parameters
        ----------
        path : str
            Output path (``*.csv``).
        """
        import pandas as pd

        df = self.coeffs.to_dataframe().reset_index()
        with open(path, "w") as fh:
            fh.write(f"# model_name: {self.model_name}\n")
            fh.write(f"# metadata: {json.dumps(self.metadata)}\n")
            df.to_csv(fh, index=False)
        log.info("%s model saved → %s", self.model_name, path)

    @classmethod
    def load(cls, path: str) -> "CalibrationModel":
        """Load a saved model from CSV, dispatching to the correct subclass.

        Can be called on the base class (``CalibrationModel.load(path)``)
        when the file contains a ``# model_name:`` comment header, or
        directly on a concrete subclass to skip the header requirement::

            RatioModel.load("gains_from_r.csv")   # no comment header needed

        Parameters
        ----------
        path : str
            Path to a CSV file.  All columns except ``wavelength`` are
            treated as coefficient variables.

        Returns
        -------
        CalibrationModel subclass instance
        """
        import pandas as pd

        model_name: str | None = None
        metadata: dict = {}

        with open(path) as fh:
            for line in fh:
                if line.startswith("# model_name:"):
                    model_name = line.split(":", 1)[1].strip()
                elif line.startswith("# metadata:"):
                    metadata = json.loads(line.split(":", 1)[1].strip())
                elif not line.startswith("#"):
                    break

        df = pd.read_csv(path, comment="#")
        wavelength = df["wavelength"].values

        if cls is CalibrationModel:
            if model_name is None:
                raise ValueError(
                    f"No '# model_name:' header found in '{path}'. "
                    "Call a concrete class (e.g. RatioModel.load()) instead, "
                    "or add '# model_name: ratio' as the first line."
                )
            target_cls = _MODEL_REGISTRY.get(model_name)
            if target_cls is None:
                raise ValueError(
                    f"Unknown model '{model_name}'. "
                    f"Registered models: {list(_MODEL_REGISTRY)}"
                )
        else:
            target_cls = cls

        coeff_cols = [c for c in df.columns if c != "wavelength"]
        coeffs = xr.Dataset(
            {col: ("wavelength", df[col].values) for col in coeff_cols},
            coords={"wavelength": wavelength},
        )
        return target_cls(wavelength, coeffs, metadata)

    # ------------------------------------------------------------------ #
    # Inspection                                                           #
    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        """Print a formatted summary of model name, provenance, and coefficients."""
        print(f"Model      : {self.model_name}")
        print(f"Wavelengths: {len(self.wavelength)} bands "
              f"[{self.wavelength.min():.0f}–{self.wavelength.max():.0f} nm]")
        for k, v in self.metadata.items():
            print(f"  {k}: {v}")
        print(self.coeffs)

    def plot(self) -> None:
        """Plot calibration coefficients as a function of wavelength."""
        import matplotlib.pyplot as plt

        vars_  = list(self.coeffs.data_vars)
        fig, axes = plt.subplots(
            1, len(vars_), figsize=(4 * len(vars_), 3), squeeze=False
        )
        for ax, var in zip(axes[0], vars_):
            ax.plot(self.wavelength, self.coeffs[var].values)
            ax.axhline(0 if var != "gain" else 1, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel(var)
            ax.set_title(f"{self.model_name} — {var}")
        plt.tight_layout()
        plt.show()

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for embedding in image NetCDF attrs.

        Returns
        -------
        dict with keys ``model_name``, ``metadata``, ``coeffs``.
        """
        return {
            "model_name": self.model_name,
            "metadata":   self.metadata,
            "coeffs":     {
                v: self.coeffs[v].values.tolist() for v in self.coeffs.data_vars
            },
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_wl={len(self.wavelength)}, "
            f"metadata={self.metadata})"
        )

    # ------------------------------------------------------------------ #
    # Shared broadcasting helper                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _broadcast(coeffs_1d: np.ndarray, rho_t: np.ndarray) -> np.ndarray:
        """Reshape a 1-D wavelength coefficient array for broadcasting with *rho_t*.

        Parameters
        ----------
        coeffs_1d : np.ndarray, shape (n_wl,)
        rho_t     : np.ndarray, shape (n_wl, ...)

        Returns
        -------
        np.ndarray  shape (n_wl, 1, 1, …)
        """
        slices = (slice(None),) + (np.newaxis,) * (rho_t.ndim - 1)
        return coeffs_1d[slices]


# --------------------------------------------------------------------------- #
# Concrete models                                                              #
# --------------------------------------------------------------------------- #


class RatioModel(CalibrationModel):
    """Per-wavelength ratio calibration.

    Gain is the per-wavelength median of ρ_t_hat / ρ_t over all matchup pixels.

    Coefficients (``self.coeffs``)
    --------------------------------
    gain     : median ratio per wavelength
    gain_std : standard deviation of ratios

    Application
    -----------
    ρ_t_cal = gain[λ] · ρ_t
    """

    model_name = "ratio"

    @classmethod
    def fit(cls, cal_data) -> "RatioModel":
        """Fit a ratio model.

        Parameters
        ----------
        cal_data : CalibrationData

        Returns
        -------
        RatioModel
        """
        df          = cal_data.df
        wavelengths = np.sort(df["wavelength"].unique())
        n           = len(wavelengths)

        gain     = np.full(n, np.nan)
        gain_std = np.full(n, np.nan)
        n_obs    = np.zeros(n, dtype=int)

        for i, wl in enumerate(wavelengths):
            sub = df[df["wavelength"] == wl].dropna(subset=["rho_t", "rho_t_hat"])
            sub = sub[sub["rho_t"] != 0]
            if len(sub) == 0:
                continue
            ratio       = sub["rho_t_hat"].values / sub["rho_t"].values
            gain[i]     = float(np.nanmedian(ratio))
            gain_std[i] = float(np.nanstd(ratio))
            n_obs[i]    = len(sub)

        coeffs = xr.Dataset(
            {
                "gain":     ("wavelength", gain),
                "gain_std": ("wavelength", gain_std),
            },
            coords={"wavelength": wavelengths},
        )
        metadata = {
            "n_matchups":    int(df["uuid"].nunique()),
            "n_images":      int(df["image_name"].nunique()),
            "total_pixels":  int(n_obs.sum()),
            "gain_range":    [float(np.nanmin(gain)), float(np.nanmax(gain))],
        }
        log.info(
            "RatioModel fit: %d matchups, gain ∈ [%.4f, %.4f]",
            metadata["n_matchups"],
            *metadata["gain_range"],
        )
        return cls(wavelengths, coeffs, metadata)

    def apply(self, rho_t: np.ndarray) -> np.ndarray:
        gain = self._broadcast(self.coeffs["gain"].values, rho_t)
        return rho_t * gain


class OLSModel(CalibrationModel):
    """Per-wavelength Ordinary Least Squares regression.

    Fits ρ_t_hat = a·ρ_t + b per wavelength using statsmodels OLS.

    Coefficients (``self.coeffs``)
    --------------------------------
    a  : slope
    b  : intercept
    r2 : coefficient of determination

    Application
    -----------
    ρ_t_cal = a[λ] · ρ_t + b[λ]
    """

    model_name = "ols"

    @classmethod
    def fit(cls, cal_data) -> "OLSModel":
        """Fit an OLS model.

        Parameters
        ----------
        cal_data : CalibrationData

        Returns
        -------
        OLSModel
        """
        import statsmodels.api as sm

        df          = cal_data.df
        wavelengths = np.sort(df["wavelength"].unique())
        n           = len(wavelengths)

        a    = np.full(n, np.nan)
        b    = np.full(n, np.nan)
        r2   = np.full(n, np.nan)
        n_obs = np.zeros(n, dtype=int)

        for i, wl in enumerate(wavelengths):
            sub = df[df["wavelength"] == wl].dropna(subset=["rho_t", "rho_t_hat"])
            if len(sub) < 2:
                continue
            X       = sm.add_constant(sub["rho_t"].values)
            res     = sm.OLS(sub["rho_t_hat"].values, X).fit()
            b[i]    = float(res.params[0])
            a[i]    = float(res.params[1])
            r2[i]   = float(res.rsquared)
            n_obs[i]= len(sub)

        coeffs = xr.Dataset(
            {"a": ("wavelength", a), "b": ("wavelength", b), "r2": ("wavelength", r2)},
            coords={"wavelength": wavelengths},
        )
        metadata = {
            "n_matchups": int(df["uuid"].nunique()),
            "n_images":   int(df["image_name"].nunique()),
            "mean_r2":    float(np.nanmean(r2)),
        }
        log.info(
            "OLSModel fit: %d matchups, mean R²=%.4f",
            metadata["n_matchups"],
            metadata["mean_r2"],
        )
        return cls(wavelengths, coeffs, metadata)

    def apply(self, rho_t: np.ndarray) -> np.ndarray:
        a = self._broadcast(self.coeffs["a"].values, rho_t)
        b = self._broadcast(self.coeffs["b"].values, rho_t)
        return a * rho_t + b


class SMAModel(CalibrationModel):
    """Per-wavelength Standard Major Axis (Type-II) regression.

    Fits ρ_t_hat = a·ρ_t + b per wavelength using pylr2 (reduced major axis).
    Unlike OLS, SMA accounts for measurement error in both variables.

    Coefficients (``self.coeffs``)
    --------------------------------
    a  : slope
    b  : intercept
    r2 : squared Pearson correlation coefficient

    Application
    -----------
    ρ_t_cal = a[λ] · ρ_t + b[λ]
    """

    model_name = "sma"

    @classmethod
    def fit(cls, cal_data) -> "SMAModel":
        """Fit a SMA (Type-II) model.

        Requires ``pylr2`` (``pip install pylr2``).

        Parameters
        ----------
        cal_data : CalibrationData

        Returns
        -------
        SMAModel
        """
        try:
            from pylr2 import regress2
        except ImportError as exc:
            raise ImportError(
                "SMAModel requires pylr2.  Install it with: pip install pylr2"
            ) from exc

        df          = cal_data.df
        wavelengths = np.sort(df["wavelength"].unique())
        n           = len(wavelengths)

        a    = np.full(n, np.nan)
        b    = np.full(n, np.nan)
        r2   = np.full(n, np.nan)
        n_obs = np.zeros(n, dtype=int)

        for i, wl in enumerate(wavelengths):
            sub = df[df["wavelength"] == wl].dropna(subset=["rho_t", "rho_t_hat"])
            if len(sub) < 2:
                continue
            res   = regress2(
                sub["rho_t"].values,
                sub["rho_t_hat"].values,
                _method_type_2="reduced major axis",
            )
            a[i]    = float(res["slope"])
            b[i]    = float(res["intercept"])
            r2[i]   = float(res["r"]) ** 2
            n_obs[i]= len(sub)

        coeffs = xr.Dataset(
            {"a": ("wavelength", a), "b": ("wavelength", b), "r2": ("wavelength", r2)},
            coords={"wavelength": wavelengths},
        )
        metadata = {
            "n_matchups": int(df["uuid"].nunique()),
            "n_images":   int(df["image_name"].nunique()),
            "mean_r2":    float(np.nanmean(r2)),
        }
        log.info(
            "SMAModel fit: %d matchups, mean R²=%.4f",
            metadata["n_matchups"],
            metadata["mean_r2"],
        )
        return cls(wavelengths, coeffs, metadata)

    def apply(self, rho_t: np.ndarray) -> np.ndarray:
        a = self._broadcast(self.coeffs["a"].values, rho_t)
        b = self._broadcast(self.coeffs["b"].values, rho_t)
        return a * rho_t + b
