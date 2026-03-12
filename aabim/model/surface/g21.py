def g21(rho_w, wavelength):
    """Gao & Li 2021 glint correction"""
    nir_i   = np.abs(wavelength - 800).argmin()
    n_w     = get_water_refractive_index(30, 12, wavelength)
    rho_f   = fresnel_reflectance(0, n_w)
    ref_ratio = rho_w[nir_i] / rho_f[nir_i]
    return rho_w - rho_f[:, None, None] * ref_ratio[None, :, :]