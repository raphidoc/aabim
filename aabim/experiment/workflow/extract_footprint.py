import os

from reverie import ReveCube
from reverie.converter import wise


if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"

    images = [
        "ACI-10A/220705_ACI-10A-WI-1x1x1_v01-L1CG.nc",
        "ACI-11A/220705_ACI-11A-WI-1x1x1_v01-L1CG.nc",
        "ACI-12A/220705_ACI-12A-WI-1x1x1_v01-L1CG.nc",
        "ACI-13A/220705_ACI-13A-WI-1x1x1_v01-L1CG.nc",
        "ACI-14A/220705_ACI-14A-WI-1x1x1_v01-L1CG.nc"
    ]

    for image in images:
        reve_cube = ReveCube.from_reve_nc(os.path.join(image_dir, image))

        footprint_gdf = reve_cube.get_footprint()

        image_name = os.path.basename(reve_cube.in_path).split(".")[0]

        output_name = f"{image_name}_footprint.geojson"

        # Write the GeoDataFrame to a GeoJSON file
        footprint_gdf.to_file(os.path.join("/D/Data/WISE/", output_name), driver="GeoJSON")


