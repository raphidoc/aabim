from dataclasses import dataclass, field
import numpy as np

@dataclass
class Sensor:
    name:        str
    wavelengths: np.ndarray
    fwhm:        np.ndarray | None = None   # per-band FWHM (nm), optional
    metadata:    dict = field(default_factory=dict)

    def to_gaussian_srf(
        self, resolution: float = 0.001
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a Gaussian spectral response function from per-band FWHM.

        The SRF is evaluated on a fine regular wavelength grid that spans
        4 σ beyond the outermost band centres.  Each band's SRF is an
        unnormalized Gaussian (peak = 1); normalization is handled at use
        time when computing the weighted average.

        Parameters
        ----------
        resolution : float
            Wavelength step of the fine grid in nm.  Default 0.1 nm.

        Returns
        -------
        srf_wl : np.ndarray, shape (n_fine,)
            Fine wavelength axis (nm).
        srf : np.ndarray, shape (n_fine, n_bands), float32
            Per-band Gaussian SRF values (peak-normalized to 1).

        Raises
        ------
        ValueError
            If ``fwhm`` is not set on this sensor.
        """
        if self.fwhm is None:
            raise ValueError(
                "Sensor.fwhm is not set; cannot compute Gaussian SRF."
            )

        sigma = self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        half_width = 4.0 * sigma.max()

        wl_min = self.wavelengths.min() - half_width
        wl_max = self.wavelengths.max() + half_width
        srf_wl = np.arange(wl_min, wl_max + resolution / 2.0, resolution)

        n_fine  = len(srf_wl)
        n_bands = len(self.wavelengths)
        srf = np.zeros((n_fine, n_bands), dtype=np.float32)

        for i, (wl_c, sig) in enumerate(zip(self.wavelengths, sigma)):
            srf[:, i] = np.exp(-0.5 * ((srf_wl - wl_c) / sig) ** 2)

        return srf_wl, srf
