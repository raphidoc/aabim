"""
Define application-specific aliases for CF standard names (sometimes very long).
Some of these are not in the CF standard name table, and should be added according to the guideline at:
https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html

The xml table could be parsed for automatic retrieval of standard names, but this is not implemented yet.
"""
# import requests
# import untangle
# import xmltodict
# from defusedxml.dom.minidom import parse, parseString, Node
# import defusedxml.ElementTree as ET
import warnings

def get_cf_std_name(url=None, alias: str = "radiance_at_sensor"):
    std_name = None
    std_unit = None

    # Template
    # if alias in ['']:
    #     std_name = ''
    #     std_unit = ''

    # Radiometric quantities
    if alias in ["radiance_at_sensor"]:
        std_name = "upwelling_radiance_per_unit_wavelength_in_air"
        std_unit = "uW cm-2 nm-1 sr-1"

    # No standard
    if alias in ["rho_at_sensor"]:
        std_name = "at_sensor_reflectance"
        std_unit = "1"

    # No standard
    if alias in ["rho_w"]:
        std_name = "water_leaving_reflectance"
        std_unit = "1"

    # Standard ?
    if alias in ["rho_surface"]:
        std_name = "surface_reflectance"
        std_unit = "1"

    if alias in ["rho_remote_sensing"]:
        std_name = "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air"
        std_unit = "sr-1"

    if alias in ["F0"]:
        std_name = "solar_irradiance_per_unit_wavelength"
        std_unit = "W cm-2 nm-1"

    # Indices
    if alias in ['ndwi']:
        std_name = 'normalized_difference_water_index'
        std_unit = 'unitless'

    if alias in ['ndvi']:
        std_name = 'normalized_difference_vegetation_index'
        std_unit = 'unitless'

    if alias in ['s_vis_nir']:
        std_name = 's_vis_nir'
        std_unit = 'unitless'

    # Mask
    if alias in ['mask_water']:
        std_name = 'mask_water'
        std_unit = 'boolean'

    # Geometric quantities:
    # solar_azimuth_angle [degree]
    # solar_zenith_angle [degree]

    # sensor_azimuth_angle [degree]
    # sensor_zenith_angle [degree]

    # Relative azimuth angle = angle_of_rotation_from_solar_azimuth_to_platform_azimuth [degree]
    # platform_azimuth_angle / sensor_azimuth_angle / relative_platform_azimuth_angle / relative_sensor_azimuth_angle ?

    if alias in ["sun_zenith"]:
        std_name = "solar_zenith_angle"
        std_unit = "degree"

    if alias in ["sun_azimuth"]:
        std_name = "solar_azimuth_angle"
        std_unit = "degree"

    if alias in ["view_zenith"]:
        std_name = "sensor_zenith_angle"
        std_unit = "degree"

    if alias in ["view_azimuth"]:
        std_name = "sensor_azimuth_angle"
        std_unit = "degree"

    if alias in ["relative_azimuth"]:
        # Should be: angle_of_rotation_from_solar_azimuth_to_sensor_azimuth
        std_name = "angle_of_rotation_from_solar_azimuth_to_platform_azimuth"
        std_unit = "degree"

    # No standard
    if alias in ["sample_index"]:
        std_name = "sensor_array_index_of_across_track_samples) "
        std_unit = ""

    # No standard
    if alias in ["line_index"]:
        std_name = "index_of_along_track_acquisition_line) "
        std_unit = ""

    if std_name is None:
        warnings.warn(f"No match for {alias}")
        std_name = "NA"
        std_unit = "NA"

    # TODO: automate retrieval of standard from the XML table at:
    #  http://cfconventions.org/Data/cf-standard-names/current/src/cf-standard-name-table.xml
    # response = requests.get(url)
    # tree = ET.ElementTree(ET.fromstring(response.content))
    # # tests = tree.findall('entry')
    # qry=f"entry[@id='{std_name}']"
    # tests = tree.find(qry)
    # dic = {}
    # for x in tests.iter():
    #     print(x.text)

    return std_name, std_unit
