import rasterio
import collections
import pyproj
import numpy as np
import rasterio.transform as transform
import pandas as pd
from timezonefinder import TimezoneFinder
from datetime import datetime
import pendulum
import math
from charset_normalizer import from_path
import re
import xarray as xr
import psutil

BADVALUE = -999


def get_available_ram_gb():
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    return available_gb


def estimate_window_size(l1, wavelength, ram_gb, n_variables=6, dtype=np.float32):
    # Get dataset dims
    # n_wl = len(l1.in_ds.wavelength)
    n_wl = wavelength.size
    # n_y = len(l1.in_ds.y)
    # n_x = len(l1.in_ds.x)
    # Data type size in bytes
    dtype_size = np.dtype(dtype).itemsize
    ram_bytes = ram_gb * (1024 ** 3)
    max_pixels = ram_bytes // dtype_size
    max_yx = max_pixels // (n_wl * n_variables)
    window_size = int(np.sqrt(max_yx))
    return window_size

def get_f0(doy : int, wavelength):

    f0 = xr.open_dataset(
        "/home/raphael/PycharmProjects/reverie/reverie/data/solar_irradiance/hybrid_reference_spectrum_c2022-11-30_with_unc.nc"
    )

    _waves = f0["Vacuum Wavelength"].values
    _f0 = f0["SSI"].values
    # unit = f0["SSI"].units

    # calculate extraterrestrial solar irradiance on a specific day considering the sun-earth distance
    # (https://en.wikipedia.org/wiki/Sunlight#:~:text=If%20the%20extraterrestrial%20solar%20radiation,hitting%20the%20ground%20is%20around)
    # IESNA, 1997. Recommended practice for the calculation of daylight availability. Illuminating
    # Engineering Society of North America, New York.
    distance_factor = 1 + 0.034 * np.cos(2 * np.pi * (doy - 2) / 365.0)
    _f0_cal = _f0 * distance_factor

    wrange = (_waves[0], _waves[-1] + 0.0001)

    resolution = 0.001

    rsr, wave = fwhm_2_rsr(
        np.full(len(wavelength), 5.05),
        wavelength,
        wrange=wrange,
        resolution=resolution,
    )
    f0_wise = np.zeros_like(wavelength, dtype=float)
    for i, item in enumerate(rsr):
        f0_wise[i] = np.sum(_f0_cal * item) / np.sum(item)

    # Convert F0 from W m-2 nm-1 to uW cm-2 nm-1
    f0 = f0_wise * 1e2

    return f0

def read_envi_hdr(hdr_f: str) -> dict:
    result = from_path(hdr_f).best()._string

    # TODO: find a regex that can parse all the header (including the nested 'description') as key-value pairs?
    # Working version that failed to parse multiline list (bbl, ...) but correctly divide the nested 'description'
    # pattern = re.compile(r'([^\n]+?(?= *[=:] *)) *[=:] *{?([^\r\n;}]*)', re.DOTALL)

    # Working version that parse multiline list but fail to unnest 'description'
    pattern = re.compile(r"([^\r\n]*)\s[=]\s({[\s\S]*?}|[^\r\n]*)")
    matches = pattern.findall(result)

    # Create a dictionary from the matches
    result_dict = {
        key.strip().lower(): re.sub(r"[{}]", "", value).strip()
        for key, value in matches
    }

    # Additional parsing for the nested 'description'
    pattern = re.compile(r"([^\n]+?(?= *[=:] *)) *[=:] *{?([^\r\n;}]*)", re.DOTALL)
    matches = pattern.findall(result_dict["description"])

    result_dict_desc = {
        key.strip().lower(): re.sub(r"[{}]", "", value).strip()
        for key, value in matches
    }

    # Clean up the list, cannot do it before because of the 'description' parsing
    result_dict = {
        key: re.sub(r"[\r\n]", "", value) for key, value in result_dict.items()
    }

    # Merge the dict
    result_dict.pop("description")
    result_dict.update(result_dict_desc)

    return result_dict


def fwhm_2_rsr(fwhm, centers, resolution, wrange=(200, 2000), model="gaussian"):
    """
    convert FWMH and center wavelenth to RSR
    :param fwhm: 1D ndarray,  full width at half maximum
    :param centers: 1D ndarray, center wavelength
    :param resolution: the spectral resolution for intergration
    :param wrange: the range of the wavelength for intergration
    :param model: model for simulation of RSR,the default is Gaussian
    :return: list of rsr ordered by bands,wavelenghs
    """
    # resolution, wrange = 0.01, (_waves[0],_waves[-1]+0.0001)

    wave = np.arange(wrange[0], wrange[1], resolution)
    if model == "gaussian":
        ##let's use Gaussian funcrion f=a*exp[-(x-b)^2/2c^2] to simulate RSR (relative spectral response)
        ##since Maximum of RSR is 1, a is set to 1.
        ## references:
        ### https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        ### http://blog.sina.com.cn/s/blog_4a1c6f7f0100061m.html
        cs = [fwmh / 2.0 / np.sqrt(2 * np.log(2)) for fwmh in fwhm]
        rsr = [np.exp(-np.power(wave - b, 2) / (2 * c**2)) for c, b in zip(cs, centers)]
    else:
        return None
    return rsr, wave


def parse_mapinfo(mapinfo):
    """
    compose affine, crs and proj4 string from map info in ENVI Header

    Parameters
    ----------
    mapinfo: str
      Structure: [UTM, 1, 1, 552591.000, 5450146.500, 1.500, 1.500, 19, North,  WGS84]

    Returns
    -------
    affine: rasterio.transform.Affine
    crs: pyproj.crs.CRS
    proj4string: str
    """

    (
        proj_name,
        x,
        y,
        reference_x,
        reference_y,
        x_size,
        y_size,
        proj_zone,
        north_or_south,
        datum,
    ) = mapinfo.split(",")

    x, y = float(x), float(y)
    x_size, y_size = float(x_size), float(y_size)

    proj_zone = int(proj_zone)

    if x == 1.0 and y == 1.0:
        ## means the reference point is (left,upper), this is the most common case
        left, upper = float(reference_x), float(reference_y)
        affine = rasterio.transform.from_origin(left, upper, x_size, y_size)

    else:
        ## if x==1.5 and y==1.5, the reference point is at the center
        return None
    unit = "m"
    proj4string = (
        "+proj={0} +zone={1} +ellps={2} +datum={2} +units={3} +no_defs".format(
            str.lower(proj_name), proj_zone, str.upper(datum), unit
        )
    )

    # TODO: when CRS.from_proj4 is exported to_wtk it lack PROJCRS and BASEGEOGCRS
    #   When created from_epsg, to_wtk is complete
    #   see: https://proj4.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems
    crs = pyproj.crs.CRS.from_proj4(proj4string)
    crs = pyproj.crs.CRS.from_epsg(crs.to_epsg())

    return affine, crs, proj4string


def transform_xy(transform, rows, cols, offset="center"):
    """
    rewrite rasterio.transform.xy, transform rows and cols of image to y, x of the coordinate reference system
    :param transform:  instance of rasterio.transform.Affine
    :param rows: List or int
    :param cols: List or int
    :param offset:
    :return:  x, and y
    """
    single_col = False
    single_row = False

    # collections.Iterable has been moved to collections.abc.Iterable since python 3.8
    # https://stackoverflow.com/questions/53978542/how-to-use-collections-abc-from-both-python-3-8-and-python-2-7/53978543#53978543
    if not isinstance(cols, collections.abc.Iterable):
        cols = [cols]
        single_col = True
    if not isinstance(rows, collections.abc.Iterable):
        rows = [rows]
        single_row = True

    if offset == "center":
        coff, roff = (0.5, 0.5)
    elif offset == "ul":
        coff, roff = (0, 0)
    elif offset == "ur":
        coff, roff = (1, 0)
    elif offset == "ll":
        coff, roff = (0, 1)
    elif offset == "lr":
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")

    xs = []
    ys = []
    for col, row in zip(cols, rows):
        x, y = transform * transform.translation(coff, roff) * (col, row)
        xs.append(x)
        ys.append(y)

    if len(cols) < len(rows):
        for col, row in zip(cols[-1:] * (len(rows) - len(cols)), rows[len(cols) :]):
            x, y = transform * transform.translation(coff, roff) * (col, row)
            ys.append(y)

    if len(rows) < len(cols):
        for col, row in zip(cols[len(rows) :], rows[-1:] * (len(cols) - len(rows))):
            x, y = transform * transform.translation(coff, roff) * (col, row)
            xs.append(x)

    if single_row:
        ys = ys[0]
    if single_col:
        xs = xs[0]

    return xs, ys

def transform_rowcol(affine, ys, xs, precision):
    """
    transform x and y, to row and col.
    :param affine: instance of rasterio.transform.Affine
    :param ys:
    :param xs:
    :param precision:
    :return:  row and cols
    """
    rows, cols = transform.rowcol(affine, xs=xs, ys=ys, precision=precision)
    return np.asarray(rows), np.asarray(cols)


def fill_na_2d(arr: np.ndarray):
    # mask = arr>0
    # xx, yy = np.meshgrid(x, y)
    # f = intp.interp2d(x, y, arr, kind='cubic')
    df = pd.DataFrame(data=arr)
    df_ = (
        df.fillna(method="ffill", axis=1)
        .fillna(method="ffill", axis=0)
        .fillna(method="bfill", axis=1)
        .fillna(method="ffill", axis=0)
    )
    return df_.values

# -------------------------------SOLAR GEOMETRY-------------------------------------------------#
def cal_solar_hour_angle(longitude, longitude_timezone, time):
    """
    calcuate the Solar Hour Angle, Observing the sun from earth, the solar hour angle is an expression of time, expressed in angular measurement,
    usually degrees, from solar noon. At solar noon the hour angle is 0.000 degree, with the time before solar noon expressed as negative degrees,
    and the local time after solar noon expressed as positive degrees.
    :param longitude: local longitude
    :param longitude_timezone: longitude for the local time zone
    :param time: local time
    :return: solar hour angle (degree)
    """
    # calculete the solar time
    # temp = time + dt.timedelta(minutes=(longitude_timezone-longitude)*4)
    if not isinstance(time, datetime):
        raise TypeError(message="time should be instance of datetime.datetime")

    # seconds =  dt.timedelta(minutes=(longitude - longitude_timezone) * 4).seconds
    seconds = (longitude - longitude_timezone) * 4 * 60.0
    # temp = time + dt.timedelta(minutes=(longitude - longitude_timezone) * 4)
    # realSolarTime = temp.hour+temp.minute/60.0+temp.second/3600.0

    realSolarTime = (
        time.hour + time.minute / 60.0 + time.second / 3600.0 + seconds / 3600.0
    )
    # print(realSolarTime)
    hourAngle = (realSolarTime - 12) * 15
    return hourAngle


def cal_declination(year, month, day):
    """
    calcuate declination of the sun, The declination of the Sun,
    is the angle between the rays of the Sun and the plane of the Earth's equator.
    The Earth's axial tilt (called the obliquity of the ecliptic by astronomers)
    is the angle between the Earth's axis and a line perpendicular to the Earth's orbit.
    :param year: year
    :param month: month
    :param day: day
    :return:  sun declination (degree)
    """
    delta = datetime(year, month, day) - datetime(year, 1, 1)

    days = delta.days + 1

    b = 2 * np.pi * (days - 1) / 365.0

    declination = (
        0.006918
        - 0.399912 * np.cos(b)
        + 0.070257 * np.sin(b)
        - 0.006758 * np.cos(2 * b)
        + 0.000907 * np.sin(2 * b)
        - 0.002697 * np.cos(3 * b)
        + 0.00148 * np.sin((3 * b))
    )

    return math.degrees(declination)


def cal_solar_zenith_azimuth(latitude, hour_angle, declination):
    """
    calculate solar elevation and azimuth
    :param latitude: latitude
    :param hour_angle: solar hour angle  (degree)
    :param declination: solar declination (degree)
    :return: Solar zenith angle, and azimuth, 0 degree refers to the real north direction, negative as western, and positive as eastern
    """
    if not isinstance(latitude, collections.abc.Iterable):
        latitude = [latitude]
    if not isinstance(hour_angle, collections.abc.Iterable):
        hour_angle = [hour_angle]

    latitude, hour_angle = np.asarray(latitude), np.asarray(hour_angle)
    if latitude.shape != hour_angle.shape:
        return BADVALUE, BADVALUE

    sinHs = np.sin(np.deg2rad(latitude)) * np.sin(np.deg2rad(declination)) + np.cos(
        np.deg2rad(latitude)
    ) * np.cos(np.deg2rad(declination)) * np.cos(np.deg2rad(hour_angle))

    cosHs = np.cos(np.arcsin(sinHs))

    cosAs = (sinHs * np.sin(np.deg2rad(latitude)) - np.sin(np.deg2rad(declination))) / (
        cosHs * np.cos(np.deg2rad(latitude))
    )
    # print(cosAs)

    # np.rad2deg(np.arccos(cosAs))   azimuth, 0 degree refers to the real southern direction, negative as eastern, and positive as westerm

    SolZen = 90 - np.rad2deg(np.arcsin(sinHs))
    SolAzm = np.rad2deg(np.arccos(cosAs)) + 180

    return SolZen, SolAzm


# -----------------------------------TIME ZONE-----------------------------------------------#
def findLocalTimeZone(longitude, latitude):
    timezone_local_name = TimezoneFinder().timezone_at(lng=longitude, lat=latitude)

    if timezone_local_name not in pendulum.timezones():
        print(
            "{} can not be recongnized as pendulum.timezones".format(
                timezone_local_name
            )
        )
        return None
    tz = pendulum.timezone(timezone_local_name)
    return tz
