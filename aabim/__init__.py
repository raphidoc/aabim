"""aabim — Atmospheric Aquatic Bayesian Inversion Model.

Public API
----------
Image             Hyperspectral image (load with Image.from_aabim_nc()).
AerLUT            Aerosol scattering / transmittance look-up table.
GasLUT            Gas transmittance look-up table.
InSitu            In-situ water reflectance data loader.
CalibrationData   ρ_t / ρ_t_hat matchup database.
CalibrationModel  Abstract base class for SVC calibration models.
RatioModel        Per-wavelength ratio gain.
OLSModel          Per-wavelength OLS regression.
SMAModel          Per-wavelength SMA (Type-II) regression.
"""

from aabim.image import Image
from aabim.data import AerLUT, GasLUT
from aabim.calibration import (
    InSitu,
    CalibrationData,
    CalibrationModel,
    RatioModel,
    OLSModel,
    SMAModel,
)

__all__ = [
    "Image",
    "AerLUT",
    "GasLUT",
    "InSitu",
    "CalibrationData",
    "CalibrationModel",
    "RatioModel",
    "OLSModel",
    "SMAModel",
]
