class SolarIrradiance:
    """Extraterrestrial solar irradiance."""

    @classmethod
    def load(cls, wavelengths, doy): ...

    def get_f0(doy: int, wavelength):

        f0 = xr.open_dataset(
            "/home/raphael/PycharmProjects/reverie/reverie/data/solar_irradiance/hybrid_reference_spectrum_c2022-11-30_with_unc.nc"
        )

        _waves = f0["Vacuum Wavelength"].values
        _f0 = f0["SSI"].values
        # unit = f0["SSI"].units

        # calculate extraterrestrial solar irradiance on a specific day considering the sun-earth distance
        # (https://en.wikipedia.org/wiki/Sunlight#:~:text=If%20the%20extraterrestrial%20solar%20radiation,hitting%20the%20ground%20is%20around)
        # IESNA, 1997. Recommended practice for the calculation of daylight availability. Illuminating
        # Engineering Society of North America, New York.
        distance_factor = 1 + 0.034 * np.cos(2 * np.pi * (doy - 2) / 365.0)
        _f0_cal = _f0 * distance_factor

        wrange = (_waves[0], _waves[-1] + 0.0001)

        resolution = 0.001

        rsr, wave = fwhm_2_rsr(
            np.full(len(wavelength), 5.05),
            wavelength,
            wrange=wrange,
            resolution=resolution,
        )
        f0_wise = np.zeros_like(wavelength, dtype=float)
        for i, item in enumerate(rsr):
            f0_wise[i] = np.sum(_f0_cal * item) / np.sum(item)

        # Convert F0 from W m-2 nm-1 to uW cm-2 nm-1
        f0 = f0_wise * 1e2

        return f0
