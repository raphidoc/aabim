import numpy as np
import xarray as xr

from reverie.correction.surface import z17


def get_glint_z17(image_sub, wavelength, aod555):

    x = image_sub.x.values
    y = image_sub.y.values

    if aod555 is not None:
        aod555_xi = aod555
    else:
        aod555_xi = float(image_sub.variables["aerosol_optical_thickness_at_555_nm"].mean().values)

    rho_z17 = z17.get_sky_sun_rho(
        aot_550=aod555_xi,
        sun_zen=float(image_sub.variables["sun_zenith"].mean().values),
        view_ang=[float(image_sub.variables["view_zenith"].mean().values),
                  float(image_sub.variables["relative_azimuth"].mean().values)],
        water_salinity=30,
        water_temperature=17,
        wavelength=wavelength,
        wind_speed=float(image_sub.variables["wind_speed"].mean().values)
    )

    sky = rho_z17["sky"]
    sky_3d = sky[:, np.newaxis, np.newaxis]
    sky_3d = np.repeat(sky_3d, len(y), axis=1)
    sky_3d = np.repeat(sky_3d, len(x), axis=2)

    sun = rho_z17["sun"]
    sun_3d = sun[:, np.newaxis, np.newaxis]
    sun_3d = np.repeat(sun_3d, len(y), axis=1)
    sun_3d = np.repeat(sun_3d, len(x), axis=2)

    rho = rho_z17["rho"]
    rho_3d = rho[:, np.newaxis, np.newaxis]
    rho_3d = np.repeat(rho_3d, len(y), axis=1)
    rho_3d = np.repeat(rho_3d, len(x), axis=2)

    rho_sky = xr.Dataset(
        data_vars=dict(
            rho_surface_sky=(
                ["wavelength", "y", "x"],
                sky_3d,
            ),
            rho_surface_sun=(
                ["wavelength", "y", "x"],
                sun_3d,
            ),
            rho_surface_total=(
                ["wavelength", "y", "x"],
                rho_3d,
            ),
        ),
        coords=dict(
            wavelength=wavelength,
            y=("y", y),
            x=("x", x),
        ),
    )

    return rho_sky
