import os

from reverie import ReveCube
from reverie.converter import wise


if __name__ == "__main__":
    image_dir = "/D/Data/WISE/"
    matchup_dir = "/D/Documents/phd/thesis/3_chapter/data/wise/pixex"

    # matchups = [
    #     "ACI-11A/el_ols/220705_ACI-11A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-12A/el_ols/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-13A/el_ols/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-14A/el_ols/220705_ACI-14A-WI-1x1x1_v01-l2rg.nc",
    # ]
    # matchups = [
    #     "ACI-11A/el_sma/220705_ACI-11A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-12A/el_sma/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-13A/el_sma/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-14A/el_sma/220705_ACI-14A-WI-1x1x1_v01-l2rg.nc",
    # ]
    # matchups = [
    #     "ACI-11A/ratio/220705_ACI-11A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-12A/ratio/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-13A/ratio/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-14A/ratio/220705_ACI-14A-WI-1x1x1_v01-l2rg.nc",
    # ]
    # matchups = [
    #     "ACI-11A/ratio_sas/220705_ACI-11A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-12A/ratio_sas/220705_ACI-12A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-13A/ratio_sas/220705_ACI-13A-WI-1x1x1_v01-l2rg.nc",
    #     "ACI-14A/ratio_sas/220705_ACI-14A-WI-1x1x1_v01-l2rg.nc",
    # ]
    matchups = [
        "ACI-11A/dsf/220705_ACI-11A-WI-1x1x1_v01-l2r.nc",
        "ACI-12A/dsf/220705_ACI-12A-WI-1x1x1_v01-l2r.nc",
        "ACI-13A/dsf/220705_ACI-13A-WI-1x1x1_v01-l2r.nc",
        "ACI-14A/dsf/220705_ACI-14A-WI-1x1x1_v01-l2r.nc",
    ]



    for matchup in matchups:
        reve_cube = ReveCube.from_reve_nc(os.path.join(image_dir, matchup))

        test = reve_cube.extract_pixel(
            os.path.join(matchup_dir, "jetski_coord.csv"),
            max_time_diff=5,
            window_size=7,
        )

        image_name = os.path.basename(reve_cube.in_path).split(".")[0]

        # output_name = f"{image_name}_pixex_el_ols.csv"
        # output_name = f"{image_name}_pixex_el_sma.csv"
        # output_name = f"{image_name}_pixex_ratio.csv"
        # output_name = f"{image_name}_pixex_ratio_sas.csv"
        output_name = f"{image_name}_pixex_dsf.csv"

        test[0].to_csv(
            path_or_buf=os.path.join(matchup_dir, output_name)
        )
