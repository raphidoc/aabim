from osgeo import gdal
import netCDF4 as nc
from reverie.converter.acolite.read_netcdf import AcoliteNetCDF

file1 = "/D/Data/TEST/merge/S2A_MSI_2023_08_24_15_29_07_T20UMA_L2W.nc"
file2 = "/D/Data/TEST/merge/S2A_MSI_2023_08_24_15_29_12_T20ULA_L2W.nc"

acolite_nc = AcoliteNetCDF(file1)
acolite_nc.to_reve_nc()

# TODO: read
#  https://github.com/remicres/otb-mosaic
#  https://doi.org/10.1109/JSTARS.2015.2449233


# ds1 = gdal.Open()
# ds2 = gdal.Open()
# dstDS = gdal.Warp("/D/Data/TEST/merged.nc", ds1)
