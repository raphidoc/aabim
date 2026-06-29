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

        # --- Clip image to the model's spectral range ----------------------
        # Bands outside the model range are dropped entirely so that no
        # uncalibrated reflectance propagates through the pipeline.
        cal_image = copy.copy(self)
        model_wl = cal_model.coeffs.wavelength
        cal_image.mask_wavelength([float(model_wl.min()), float(model_wl.max())])
        log.info(
            "Wavelength range after calibration clip: %.2f – %.2f nm  (%d bands)",
            float(model_wl.min()), float(model_wl.max()), len(cal_image.wavelength),
        )

        # --- Interpolate coefficients to image wavelengths -----------------
        # Round to 2 dp before interpolation so float32-stored wavelengths
        # (e.g. 374.78 → 374.77999878) align with 2-dp CSV model coordinates.
        image_wl = xr.DataArray(
            np.round(np.asarray(cal_image.wavelength, dtype=np.float64), 2),
            dims=["wavelength"],
        )
        coeffs = cal_model.coeffs.interp(wavelength=image_wl, method="linear")

        # --- Apply calibration lazily via xarray arithmetic ----------------
        ds_cal = cal_image.in_ds.copy()

        if cal_model.model_name == "ratio":
            ds_cal[_RHO_VAR] = cal_image.in_ds[_RHO_VAR] * coeffs["gain"]
        else:
            ds_cal[_RHO_VAR] = coeffs["a"] * cal_image.in_ds[_RHO_VAR] + coeffs["b"]

        # --- Traceability attributes ----------------------------------------
        ds_cal.attrs["calibration_model"]       = cal_model.model_name
        ds_cal.attrs["calibration_applied_at"]  = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        ds_cal.attrs["calibration_params_json"] = json.dumps(cal_model.to_dict())
        ds_cal.attrs["processing_level"]        = "L1C"

        # --- Write to disk if requested -------------------------------------
        if output is not None:
            ds_cal.to_netcdf(output)
            log.info("Calibrated image written → %s", output)

        cal_image.in_ds = ds_cal
        cal_image.level = "L1C"
        return cal_image
