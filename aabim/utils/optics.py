import numpy as np

def get_water_refractive_index(salinity, temperature, wavelength):
    # Quan, X. and Fry, E.S. (1995) ‘Empirical equation for the index
    # of refraction of seawater.’, Appl. Opt., 34, pp. 3477–3480.

    n0 = 1.31405
    n1 = 1.779 * 1e-4
    n2 = -1.05 * 1e-6
    n3 = 1.6 * 1e-8
    n4 = -2.02 * 1e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455 * 1e6

    nw = n0 + (n1 + n2 * temperature + n3 * np.power(temperature, 2)) * salinity + n4 * np.power(temperature, 2) + \
         (n5 + n6 * salinity + n7 * temperature) / wavelength + n8 / np.power(wavelength, 2) + n9 / np.power(wavelength, 3)

    return nw