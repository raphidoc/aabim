import os
from reverie.converter import wise
import xarray as xr

if __name__ == "__main__":
    # Create a zarr dataset from a netcdf with the xarray method to inspect it's construction
    # test = xr.open_dataset(
    #     "/D/Data/TEST/TEST_WISE/MC-51A/190818_MC-51A-WI-2x1x1_v02-L1CG.nc"
    # )
    # test.to_zarr(
    #     "/D/Data/TEST/TEST_WISE/MC-51A/xarray_190818_MC-51A-WI-2x1x1_v02-L1C.zarr"
    # )

    # test = xr.open_zarr(
    #     "/D/Data/TEST/TEST_WISE/MC-51A/xarray_190818_MC-51A-WI-2x1x1_v02-L1C.zarr"
    # )

    l1 = wise.read_pix.Pix(
        pix_dir="/D/Data/TEST/TEST_WISE/MC-51A/190818_MC-51A-WI-2x1x1_v02-L1CG.dpix"
    )

    l1.to_zarr("/D/Data/TEST/TEST_WISE/MC-51A/190818_MC-51A-WI-2x1x1_v02-L1C.zarr")

    test = xr.open_zarr(
        "/D/Data/TEST/TEST_WISE/MC-51A/190818_MC-51A-WI-2x1x1_v02-L1C.zarr"
    )

    # image = xr.open_zarr(
    #     "/D/Data/TEST/TEST_WISE/MC-12A/190820_MC-12A-WI-1x1x1_v02-L1C.zarr"
    # )
