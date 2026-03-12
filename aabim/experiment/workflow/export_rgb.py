import os
import numpy as np

from reverie import ReveCube
from reverie.converter import wise


if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"

    images = [
        # "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-L1CG.nc",
        # "MC-50A/190818_MC-50A-WI-2x1x1_v02-L1CG.nc",
        # "MC-37A/190818_MC-37A-WI-1x1x1_v02-L1CG.nc",
        # "MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc",
    ]

    for image in images:
        reve_cube = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        # Define the wavelengths for R, G, B
        wavelengths_rgb = [652, 551, 449]  # in nm

        # Find the index of the closest wavelength for R, G, B
        indices_rgb = [
            np.abs(reve_cube.wavelength - wl).argmin() for wl in wavelengths_rgb
        ]

        # Create a subset of the dataset at those wavelengths
        subset = reve_cube.in_ds.isel(wavelength=indices_rgb)

        subset = subset[["radiance_at_sensor", "grid_mapping"]]

        subset.to_netcdf(os.path.join("/D/Data/WISE/", f"{reve_cube.image_name}_rgb.nc"))
