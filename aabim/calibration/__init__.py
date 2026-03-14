"""
aabim.calibration — System Vicarious Calibration pipeline.

Classes
-------
InSitu            Load in-situ water reflectance data from CSV.
CalibrationData   Build the ρ_t / ρ_t_hat matchup database from image + InSitu.
CalibrationModel  Abstract base class for calibration models.
RatioModel        Per-wavelength ratio gain: ρ_t_cal = gain · ρ_t.
OLSModel          Per-wavelength OLS regression: ρ_t_cal = a·ρ_t + b.
SMAModel          Per-wavelength SMA (Type-II) regression: ρ_t_cal = a·ρ_t + b.
"""

from aabim.calibration.insitu import InSitu
from aabim.calibration.data import CalibrationData
from aabim.calibration.model import CalibrationModel, RatioModel, OLSModel, SMAModel

__all__ = [
    "InSitu",
    "CalibrationData",
    "CalibrationModel",
    "RatioModel",
    "OLSModel",
    "SMAModel",
]
