# Standard library imports
import os
import time
from datetime import datetime

# Third party imports
from osgeo import gdal
from tqdm import tqdm
import numpy as np
import re
import netCDF4
import h5py

import pyproj

# REVERIE import
from reverie.image import ReveCube

gdal.UseExceptions()


class PrismaHe5(ReveCube):
    def __init__(self, he5_file):
        if os.path.isfile(he5_file):
            src_ds = netCDF4.Dataset(he5_file, "r")

            src_ds = h5py.File(he5_file,'r')

        else:
            raise Exception(f"File {he5_file} does not exist")

        h5_gatts = {}
        for a in src_ds.attrs.keys(): h5_gatts[a] = src_ds.attrs[a]

        waves_vnir = h5_gatts['List_Cw_Vnir']
        bands_vnir = ['{:.0f}'.format(w) for w in waves_vnir]
        fwhm_vnir = h5_gatts['List_Fwhm_Vnir']
        n_vnir = len(waves_vnir)

        waves_swir = h5_gatts['List_Cw_Swir']
        bands_swir = ['{:.0f}'.format(w) for w in waves_swir]
        fwhm_swir = h5_gatts['List_Fwhm_Swir']
        n_swir = len(waves_swir)

        waves = [w for w in waves_vnir] + [w for w in waves_swir]
        fwhm = [f for f in fwhm_vnir] + [f for f in fwhm_swir]
        waves_names = ['{:.0f}'.format(w) for w in waves]
        instrument = ['vnir']*n_vnir + ['swir']*n_swir
        band_index = [i for i in range(n_vnir)] + [i for i in range(n_swir)]

        band_names_vnir = ['vnir_{}'.format(b) for b in range(0, n_vnir)]
        band_names_swir = ['swir_{}'.format(b) for b in range(0, n_swir)]

        # rsr_vnir = {'vnir_{}'.format(b): ac.shared.gauss_response(waves_vnir[b], fwhm_vnir[b], step=0.1) for b in range(0, n_vnir)}
        # rsr_swir = {'swir_{}'.format(b): ac.shared.gauss_response(waves_swir[b], fwhm_swir[b], step=0.1) for b in range(0, n_swir)}
        #
        # band_names = band_names_vnir + band_names_swir
        # band_rsr = {}
        # for b in rsr_vnir: band_rsr[b] = {'wave': rsr_vnir[b][0]/1000, 'response': rsr_vnir[b][1]}
        # for b in rsr_swir: band_rsr[b] = {'wave': rsr_swir[b][0]/1000, 'response': rsr_swir[b][1]}
        #
        # ## use same rsr as acolite_l2r
        # #rsr = ac.shared.rsr_hyper(gatts['band_waves'], gatts['band_widths'], step=0.1)
        # # rsrd = ac.shared.rsr_dict(rsrd={sensor:{'rsr':band_rsr}})
        # # waves = [rsrd[sensor]['wave_nm'][b] for b in band_names]
        # # waves_names = [rsrd[sensor]['wave_name'][b] for b in band_names]
        #
        # idx = np.argsort(waves)
        # f0d = ac.shared.rsr_convolute_dict(f0['wave']/1000, f0['data'], band_rsr)
        #
        # bands = {}
        # for i in idx:
        #     cwave = waves[i]
        #     if cwave == 0: continue
        #     swave = '{:.0f}'.format(cwave)
        #     bands[swave]= {'wave':cwave, 'wavelength':cwave, 'wave_mu':cwave/1000.,
        #                    'wave_name':waves_names[i],
        #                    'width': fwhm[i],
        #                    'i':i, 'index':band_index[i],
        #                    'rsr': band_rsr[band_names[i]],
        #                    'f0': f0d[band_names[i]],
        #                    'instrument':instrument[i],}
        #
        # # print(rsrd[sensor]['wave_name'])
        # # print(bands)
        # # stop
        #
        # gatts = {}
        #
        # isotime = h5_gatts['Product_StartTime']
        # time = dateutil.parser.parse(isotime)
        #
        # doy = int(time.strftime('%j'))
        # d = ac.shared.distance_se(doy)
        #
        # ## lon and lat keys
        # lat_key = 'Latitude_SWIR'
        # lon_key = 'Longitude_SWIR'
        # if 'PRS_L1G_STD_OFFL_' in os.path.basename(file):
        #     lat_key = 'Latitude'
        #     lon_key = 'Longitude'
        #
        # ## mask for L1G format
        # mask_value = 65535
        # dem = None
        #
        # ## reading settings
        # src = 'HCO' ## coregistered radiance cube
        # read_cube = True
        #
        # ## get geometry from l2 file if present
        # l2file = os.path.dirname(file) + os.path.sep + os.path.basename(file).replace('PRS_L1_STD_OFFL_', 'PRS_L2C_STD_')
        # if not os.path.exists(l2file):
        #     print('PRISMA processing only supported when L2 geometry is present.')
        #     print('Please put {} in the same directory as {}'.format(os.path.basename(l2file), os.path.basename(file)))
        #     continue
        #
        # ## read geolocation
        # with h5py.File(file, mode='r') as f:
        #     lat = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Geolocation Fields'][lat_key][:]
        #     lon = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Geolocation Fields'][lon_key][:]
        #     lat[lat>=mask_value] = np.nan
        #     lon[lon>=mask_value] = np.nan
        # sub = None
        # if limit is not None:
        #     sub = ac.shared.geolocation_sub(lat, lon, limit)
        #     if sub is None:
        #         print('Limit outside of scene {}'.format(file))
        #         continue
        #     ## crop to sub
        #     lat = lat[sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #     lon = lon[sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        # ## end read geolocation
        #
        # ## read geometry
        # vza, vaa, sza, saa, raa = None, None, None, None, None
        # with h5py.File(l2file, mode='r') as f:
        #     ## L1G format
        #     if 'PRS_L1G_STD_OFFL_' in os.path.basename(file):
        #         #lat_key = 'Latitude'
        #         #lon_key = 'Longitude'
        #         if sub is None:
        #             vza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Zenith_Angle'][:]
        #             vaa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Azimuth_Angle'][:]
        #             sza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Zenith_Angle'][:]
        #             saa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Azimuth_Angle'][:]
        #         else:
        #             vza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Zenith_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #             vaa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Azimuth_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #             sza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Zenith_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #             saa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Azimuth_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #
        #         ## apply mask
        #         vza[vza>=mask_value] = np.nan
        #         sza[sza>=mask_value] = np.nan
        #         saa[saa>=mask_value] = np.nan
        #         vaa[vaa>=mask_value] = np.nan
        #
        #         ## compute relative azimuth
        #         raa = np.abs(saa - vaa)
        #         raa[raa>180] = 360 - raa[raa>180]
        #
        #         ## get DEM data
        #         if sub is None:
        #             dem = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Terrain Fields']['DEM'][:]
        #         else:
        #             dem = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Terrain Fields']['DEM'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #         dem[dem>=mask_value] = np.nan
        #     else:
        #         if sub is None:
        #             vza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Observing_Angle'][:]
        #             raa = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Rel_Azimuth_Angle'][:]
        #             sza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Solar_Zenith_Angle'][:]
        #         else:
        #             vza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Observing_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #             raa = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Rel_Azimuth_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #             sza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Solar_Zenith_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        #
        # gatts['vza'] = np.nanmean(np.abs(vza))
        # gatts['raa'] = np.nanmean(np.abs(raa))
        # gatts['sza'] = np.nanmean(np.abs(sza))
        #
        # with h5py.File(file, mode='r') as f:
        #     ## read bands in spectral order
        #     if read_cube:
        #         if sub is None:
        #             vnir_data = h5_gatts['Offset_Vnir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][:]/h5_gatts['ScaleFactor_Vnir']
        #             swir_data = h5_gatts['Offset_Swir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][:]/h5_gatts['ScaleFactor_Swir']
        #         else:
        #             vnir_data = h5_gatts['Offset_Vnir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Vnir']
        #             swir_data = h5_gatts['Offset_Swir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Swir']
        #
        #         vnir_data[vnir_data>=mask_value] = np.nan
        #         swir_data[swir_data>=mask_value] = np.nan
        #
        #     ## read LOS vectors
        #     x_ = f['KDP_AUX']['LOS_Vnir'][:,0]
        #     y_ = f['KDP_AUX']['LOS_Vnir'][:,1]
        #     z_ = f['KDP_AUX']['LOS_Vnir'][:,2]
        #
        # ## get vza/vaa
        # #dtor = np.pi/180
        # #vza = np.arctan2(y_,x_)/dtor
        # #vaa = np.arctan2(z_,np.sqrt(x_**2+y_**2))/dtor
        # #vza_ave = np.nanmean(np.abs(vza))
        # #vaa_ave = np.nanmean(np.abs(vaa))
        # sza_ave = h5_gatts['Sun_zenith_angle']
        # saa_ave = h5_gatts['Sun_azimuth_angle']
        #
        # if setu['prisma_rhot_per_pixel_sza']:
        #     cossza = np.cos(np.radians(sza))
        # else:
        #     cossza = np.cos(np.radians(sza_ave))
        #
        # vza_ave = 0
        # vaa_ave = 0
        #
        # if 'sza' not in gatts: gatts['sza'] = sza_ave
        # if 'vza' not in gatts: gatts['vza'] = vza_ave
        # if 'saa' not in gatts: gatts['saa'] = saa_ave
        # if 'vaa' not in gatts: gatts['vaa'] = vaa_ave
        #
        # if 'raa' not in gatts:
        #     raa_ave = abs(gatts['saa'] - gatts['vaa'])
        #     while raa_ave >= 180: raa_ave = abs(raa_ave-360)
        #     gatts['raa'] = raa_ave
        #
        # mu0 = np.cos(gatts['sza']*(np.pi/180))
        # muv = np.cos(gatts['vza']*(np.pi/180))
        #
        # if output is None:
        #     odir = os.path.dirname(file)
        # else:
        #     odir = output
        #
        # gatts['sensor'] = sensor
        # gatts['isodate'] = time.isoformat()
        #
        # obase  = '{}_{}_L1R'.format(gatts['sensor'],  time.strftime('%Y_%m_%d_%H_%M_%S'))
        # if not os.path.exists(odir): os.makedirs(odir)
        # ofile = '{}/{}.nc'.format(odir, obase)
        #
        # gatts['obase'] = obase
        #
        # gatts['band_waves'] = [bands[w]['wave'] for w in bands]
        # gatts['band_widths'] = [bands[w]['width'] for w in bands]
        #
        # new = True
        # if (setu['output_geolocation']) & (new):
        #     if verbosity > 1: print('Writing geolocation lon/lat')
        #     ac.output.nc_write(ofile, 'lon', np.flip(np.rot90(lon)), new=new, attributes=gatts)
        #     if verbosity > 1: print('Wrote lon ({})'.format(lon.shape))
        #     new = False
        #     if not (store_l2c & store_l2c_separate_file): lon = None
        #
        #     ac.output.nc_write(ofile, 'lat', np.flip(np.rot90(lat)), new=new, attributes=gatts)
        #     if verbosity > 1: print('Wrote lat ({})'.format(lat.shape))
        #     if not (store_l2c & store_l2c_separate_file): lat = None
        #
        # ## write geometry
        # if os.path.exists(l2file):
        #     if (setu['output_geometry']):
        #         if verbosity > 1: print('Writing geometry')
        #         ac.output.nc_write(ofile, 'vza', np.flip(np.rot90(vza)), attributes=gatts, new=new)
        #         if verbosity > 1: print('Wrote vza ({})'.format(vza.shape))
        #         vza = None
        #         new = False
        #         if vaa is not None:
        #             ac.output.nc_write(ofile, 'vaa', np.flip(np.rot90(vaa)), attributes=gatts, new=new)
        #             if verbosity > 1: print('Wrote vaa ({})'.format(vaa.shape))
        #             vaa = None
        #
        #         ac.output.nc_write(ofile, 'sza', np.flip(np.rot90(sza)), attributes=gatts, new=new)
        #         if verbosity > 1: print('Wrote sza ({})'.format(sza.shape))
        #
        #         if saa is not None:
        #             ac.output.nc_write(ofile, 'saa', np.flip(np.rot90(saa)), attributes=gatts, new=new)
        #             if verbosity > 1: print('Wrote saa ({})'.format(saa.shape))
        #             saa = None
        #
        #         ac.output.nc_write(ofile, 'raa', np.flip(np.rot90(raa)), attributes=gatts, new=new)
        #         if verbosity > 1: print('Wrote raa ({})'.format(raa.shape))
        #         raa = None
        #
        #     if dem is not None:
        #         ac.output.nc_write(ofile, 'dem', np.flip(np.rot90(dem)))
        #
        # ## store l2c data
        # if store_l2c & read_cube:
        #     if store_l2c_separate_file:
        #         obase_l2c  = '{}_{}_converted_L2C'.format('PRISMA',  time.strftime('%Y_%m_%d_%H_%M_%S'))
        #         ofile_l2c = '{}/{}.nc'.format(odir, obase_l2c)
        #         ac.output.nc_write(ofile_l2c, 'lat', np.flip(np.rot90(lat)), new=True, attributes=gatts)
        #         lat = None
        #         ac.output.nc_write(ofile_l2c, 'lon', np.flip(np.rot90(lon)))
        #         lon = None
        #     else:
        #         ofile_l2c = '{}'.format(ofile)
        #
        #     ## get l2c details for reflectance conversion
        #     h5_l2c_gatts = ac.prisma.attributes(l2file)
        #     scale_max = h5_l2c_gatts['L2ScaleVnirMax']
        #     scale_min = h5_l2c_gatts['L2ScaleVnirMin']
        #
        #     ##  read in data cube
        #     with h5py.File(l2file, mode='r') as f:
        #         if sub is None:
        #             vnir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['VNIR_Cube'][:]
        #             swir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['SWIR_Cube'][:]
        #         else:
        #             vnir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['VNIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]
        #             swir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['SWIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]
        #
        # ## write TOA data
        # for bi, b in enumerate(bands):
        #     wi = bands[b]['index']
        #     i = bands[b]['i']
        #     print('Reading rhot_{}'.format(bands[b]['wave_name']))
        #
        #     if bands[b]['instrument'] == 'vnir':
        #         if read_cube:
        #             cdata_radiance = vnir_data[:,wi,:]
        #             cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)
        #             if store_l2c:
        #                 cdata_l2c = scale_min + (vnir_l2c_data[:, wi, :] * (scale_max - scale_min)) / 65535
        #         else:
        #             if sub is None:
        #                 cdata_radiance = h5_gatts['Offset_Vnir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][:,i,:]/h5_gatts['ScaleFactor_Vnir']
        #             else:
        #                 cdata_radiance = h5_gatts['Offset_Vnir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][sub[1]:sub[1]+sub[3], i, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Vnir']
        #             cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)
        #
        #     if bands[b]['instrument'] == 'swir':
        #         if read_cube:
        #             cdata_radiance = swir_data[:,wi,:]
        #             cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)
        #             if store_l2c:
        #                 cdata_l2c = scale_min + (swir_l2c_data[:, wi, :] * (scale_max - scale_min)) / 65535
        #         else:
        #             if sub is None:
        #                 cdata_radiance = h5_gatts['Offset_Swir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][:,i,:]/h5_gatts['ScaleFactor_Swir']
        #             else:
        #                 cdata_radiance = h5_gatts['Offset_Swir'] + \
        #                         f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][sub[1]:sub[1]+sub[3], i, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Swir']
        #             cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)
        #
        #     ds_att = {k:bands[b][k] for k in bands[b] if k not in ['rsr']}
        #
        #     if output_lt:
        #         ## write toa radiance
        #         ac.output.nc_write(ofile, 'Lt_{}'.format(bands[b]['wave_name']),
        #                             np.flip(np.rot90(cdata_radiance)),dataset_attributes = ds_att)
        #         cdata_radiance = None
        #
        #     ## write toa reflectance
        #     ac.output.nc_write(ofile, 'rhot_{}'.format(bands[b]['wave_name']),
        #                             np.flip(np.rot90(cdata)), dataset_attributes = ds_att)
        #     cdata = None
        #     print('Wrote rhot_{}'.format(bands[b]['wave_name']))
        #
        #     ## store L2C data
        #     if store_l2c & read_cube:
        #         ac.output.nc_write(ofile_l2c, 'rhos_l2c_{}'.format(bands[b]['wave_name']),
        #                             np.flip(np.rot90(cdata_l2c)),dataset_attributes = ds_att)
        #         ofile_l2c_new = False
        #         cdata_l2c = None
        #         print('Wrote rhos_l2c_{}'.format(bands[b]['wave_name']))
        #
        # ## output PAN
        # if setu['prisma_output_pan']:
        #     psrc = src.replace('H', 'P')
        #     with h5py.File(file, mode='r') as f:
        #         if sub is None:
        #             pan = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Data Fields']['Cube'][:]
        #             plat = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Latitude'][:]
        #             plon = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Longitude'][:]
        #         else:
        #             psub = [s*6 for s in sub]
        #             pan = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Data Fields']['Cube'][psub[1]:psub[1]+psub[3], psub[0]:psub[0]+psub[2]]
        #             plat = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Latitude'][psub[1]:psub[1]+psub[3], psub[0]:psub[0]+psub[2]]
        #             plon = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Longitude'][psub[1]:psub[1]+psub[3], psub[0]:psub[0]+psub[2]]
        #
        #     ## convert to radiance
        #     pan = h5_gatts['Offset_Pan'] + pan / h5_gatts['ScaleFactor_Pan']
        #
        #     ## output netcdf
        #     ofile_pan = '{}/{}_pan.nc'.format(odir, obase)
        #     ac.output.nc_write(ofile_pan, 'lon', np.flip(np.rot90(plon)),new = True) #dataset_attributes = ds_att,
        #     plon = None
        #     ac.output.nc_write(ofile_pan, 'lat', np.flip(np.rot90(plat))) #dataset_attributes = ds_att,
        #     plat = None
        #     ac.output.nc_write(ofile_pan, 'pan', np.flip(np.rot90(pan))) #dataset_attributes = ds_att,
        #     pan = None
        #  ## end PAN
        #
        # ofiles.append(ofile)

        # ACOLITE store spectral variable in a wide format (individual varaible by wavelength)
        # and wavelength in variables name and attributes
        # First list all variable that are wavelength dependant
        pattern = re.compile(r".*_\d{3}")

        def find_matching_keys(pattern, dictionary):
            return list(filter(lambda key: re.search(pattern, key), dictionary.keys()))

        spectral_var = find_matching_keys(pattern, src_ds.variables)

        # Create a list of wavelength from the variable attributes
        wavelength = []
        for var in spectral_var:
            wavelength.append(src_ds.variables[var].wavelength)

        # Convert list to numpy array
        wavelength = np.round(np.array(wavelength), 2)

        # Create a dictionary mapping variable name to wavelength
        self.spectral_var = dict(zip(spectral_var, wavelength))

        # Geographic attributes
        # As ACOLITE only process satellite imagery we give a symbolic altitude of 800km
        z = 800  # As we have only one altitude, could be a scalar

        grid_mapping = src_ds.variables["transverse_mercator"]
        crs = pyproj.CRS.from_wkt(grid_mapping.crs_wkt)
        affine = None
        n_rows = src_ds.dimensions["y"].size
        n_cols = src_ds.dimensions["x"].size

        x_var = src_ds.variables["x"]
        y_var = src_ds.variables["y"]
        lon_var = src_ds.variables["lon"]
        lat_var = src_ds.variables["lat"]
        x = x_var[:].data
        y = y_var[:].data
        # ACOLITE lon and lat have both y, x dimension as in CF-1.11 section 5.2
        # don't know why this was done
        lon = lon_var[:, 0].data
        lat = lat_var[0, :].data

        # Time attributes
        time_var = src_ds.isodate

        acq_time_z = datetime.fromisoformat(time_var)

        super().__init__(
            he5_file,
            src_ds,
            wavelength,
            acq_time_z,
            z,
            y,
            x,
            lon,
            lat,
            n_rows,
            n_cols,
            affine,
            crs,
        )

    def to_reve_nc(self, out_file=None):
        """
        Convert ACOLITE NetCDF to REVE NetCDF
        """
        if out_file is None:
            out_file = f"{os.path.splitext(self.in_ds.filepath())[0]}-reve.nc"

        # Create REVE NetCDF
        self.create_reve_nc(out_file)

        # Create radiometric variable
        # ACOLITE doesn't use a scale factor
        self.create_var_nc(
            name="rho_w",
            type="f4",
            dims=(
                "W",
                "Y",
                "X",
            ),
            scale=1,
        )

        for var_name, wavelength in tqdm(
            self.spectral_var.items(), desc="Writing band: "
        ):
            data = self.in_ds.variables[var_name][:, :].data
            # Cannot directly index with wavelength value as to find the index of the wavelength
            # we need to round the value as the conversion to list appear to modify it
            wave_ix = np.where(self.out_ds.variables["W"][:].data == wavelength)[0][0]
            # Could also just assume the order is correct and use the index of the variable
            self.out_ds.variables["rho_w"][wave_ix, :, :] = data

        # # Create geometric variables
        # geom = {
        #     "SolAzm": self.sun_azimuth,
        #     "SolZen": self.sun_zenith,
        #     "ViewAzm": self.view_azimuth,
        #     "ViewZen": self.view_zenith,
        #     "RelativeAzimuth": self.relative_azimuth,
        #     "SampleIndex": self.sample_index,
        # }
        #
        # for name in tqdm(geom, desc="Writing geometry"):
        #     self.create_var_nc(
        #         name=name,
        #         dims=(
        #             "Y",
        #             "X",
        #         ),
        #     )
        #
        #     geom[name][np.isnan(geom[name])] = self.no_data * self.scale
        #
        #     self.out_ds.variables[name][:, :] = geom[name]

        self.out_ds.close()
        return
