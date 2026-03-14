"""
calibrate.py — CalibrateMixin: apply a CalibrationModel to an Image.

The calibration is expressed as a per-wavelength operation and is applied
lazily through xarray so that only the requested chunks are materialised.
This keeps memory usage constant regardless of image size.

Traceability
------------
The calibrated image dataset carries three global attributes:
    calibration_model          : model name string
    calibration_applied_at     : ISO-8601 UTC timestamp
    calibration_params_json    : JSON-serialised coefficients + metadata
"""
from __future__ import annotations

import copy
import datetime
import json
import logging

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# Variable in the image dataset that holds TOA reflectance
_RHO_VAR = "rho_at_sensor"


class CalibrateMixin:
    """Mixin that adds :meth:`calibrate` to the Image class."""

    def calibrate(self, cal_model, output: str | None = None):
        """Apply *cal_model* to the image's TOA reflectance variable.

        The per-wavelength calibration coefficients are interpolated to the
        image wavelengths so that the model can be applied even when it was
        fitted on a slightly different wavelength grid.

        The operation is fully lazy: only the data needed to write each chunk
        is materialised, so large images can be processed with constant memory.

        Parameters
        ----------
        cal_model : CalibrationModel
            A fitted calibration model (RatioModel, OLSModel, or SMAModel).
        output : str, optional
            Path for the output NetCDF.  When provided the calibrated dataset
            is written to disk.  The method always returns the calibrated Image
            regardless.

        Returns
        -------
        Image  (same concrete class as ``self``)
            New Image instance whose ``in_ds`` contains the calibrated
            ``rho_at_sensor`` variable and updated processing_level.
        """
        if _RHO_VAR not in self.in_ds:
            raise KeyError(
                f"Variable '{_RHO_VAR}' not found in image dataset. "
                f"Available: {list(self.in_ds.data_vars)}"
            )

        log.info(
            "Applying '%s' calibration to image '%s'",
            cal_model.model_name,
            self.image_name,
        )

        # --- Interpolate coefficients to image wavelengths -----------------
        # The model may have been fitted on a different (possibly coarser)
        # wavelength grid; linear interpolation keeps things consistent.
        image_wl = xr.DataArray(
            np.asarray(self.wavelength, dtype=np.float64),
            dims=["wavelength"],
        )
        coeffs = cal_model.coeffs.interp(wavelength=image_wl, method="linear")

        # --- Apply calibration lazily via xarray arithmetic ----------------
        ds_cal = self.in_ds.copy()

        if cal_model.model_name == "ratio":
            gain = coeffs["gain"]
            ds_cal[_RHO_VAR] = self.in_ds[_RHO_VAR] * gain
        else:
            # OLS and SMA share the same functional form: a·ρ_t + b
            a = coeffs["a"]
            b = coeffs["b"]
            ds_cal[_RHO_VAR] = a * self.in_ds[_RHO_VAR] + b

        # --- Traceability attributes ----------------------------------------
        ds_cal.attrs["calibration_model"]       = cal_model.model_name
        ds_cal.attrs["calibration_applied_at"]  = (
            datetime.datetime.utcnow().isoformat() + "Z"
        )
        ds_cal.attrs["calibration_params_json"] = json.dumps(cal_model.to_dict())
        ds_cal.attrs["processing_level"]        = "L1C"

        # --- Write to disk if requested -------------------------------------
        if output is not None:
            ds_cal.to_netcdf(output)
            log.info("Calibrated image written → %s", output)

        # --- Return a new Image with the calibrated dataset -----------------
        cal_image = copy.copy(self)
        cal_image.in_ds = ds_cal
        cal_image.level = "L1C"
        return cal_image
