from reverie import ReveCube
from reverie.converter import wise

if __name__ == "__main__":
    # image = wise.read_pix.Pix(
    #     pix_dir="/D/Data/TEST/TEST_WISE/MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.dpix",
    # )
    #
    # image.to_reve_nc("/D/Data/TEST/TEST_WISE/MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.nc")

    reve = ReveCube.from_zarr(
        "/D/Data/TEST/TEST_WISE/MC-10A/190820_MC-10A-WI-1x1x1_v02-L1G.zarr"
    )

    test = reve.extract_pixel_line(
        line_index=9130,
        line_window=3,
    )

    test.to_csv(
        path_or_buf="/D/Documents/PhD/Thesis/Chapter2/Data/WISE/Lt_pixex_line_MC10A.csv"
    )
