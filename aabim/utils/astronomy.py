import numpy as np
import datetime


def date_time_to_julian_day(date_time: datetime):
    """Convert a datetime to a Julian Day
    Parameters
    ----------
    date_time: datetime
        The date time to convert to Julian Day

    Returns
    -------
    float
        The Julian Day

    Notes
    -----
    The Julian Day is the continuous count of days since the beginning of the Julian Period used primarily by astronomers.
    The algorithm used here is eq. 7.1 from [1]_ and is valid for any date in the Gregorian calendar.

    References
    ----------
    .. [1] Meeus, J. (1998) Astronomical algorithms. 2nd ed. Richmond, Va: Willmann-Bell.
    """

    # TODO: not sure about the meaning of INT() in Meeus (1998) is it int or trunc ?
    year = date_time.year
    month = date_time.month
    day = date_time.day

    decimal_hour = date_time.hour / 24
    decimal_minute = date_time.minute / 1440
    decimal_second = date_time.second / 86400

    day = day + decimal_hour + decimal_minute + decimal_second

    if month <= 2:
        year = year - 1
        month = month + 12

    a = int(year / 100)
    b = 2 - a + int(a / 4)

    julian_day = (
        int(365.25 * (year + 4716)) + int(30.61 * (month + 1)) + day + b - 1524.5
    )

    return julian_day


def sun_geom_meeus(date_time, obs_latitude):
    """Compute the solar azimuth and altitude using the Meeus (1998) algorithm !!! NOT WORKING !!!
    Parameters
    ----------
    date_time: datetime
        The date time to convert to Julian Day
    obs_latitude: float
        The latitude of the observation in decimal degrees, positive north

    Returns
    -------
    sun_azimuth: float
        The solar azimuth angle in decimal degrees measured from the south, positive westward (0 - 180)
    sun_altitude: float
        The solar altitude angle in decimal degrees measured from the horizon, positive above the horizon (0 - 90)

    Notes
    -----
    The algorithm used here is from [1]_.

    References
    ----------
    .. [1] Meeus, J. (1998) Astronomical algorithms. 2nd ed. Richmond, Va: Willmann-Bell.
    """
    julian_day = date_time_to_julian_day(date_time)

    # eq. 24.1
    julian_centuries = (julian_day - 2451545.0) / 36525

    # Mean longitude of the sun eq. 24.2: $L_o$
    mean_lon_sun = (
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * julian_centuries**2
    )

    # In nooa calc, it is mean_lon_sun % 360
    # mean_lon_sun = mean_lon_sun % 360

    # Mean longitude of the moon: $L^'$
    mean_lon_moon = 218.3165 + 481267.8813 * julian_centuries

    # Mean anomaly of the sun eq. 24.3: $M$ [degree]
    mean_anomaly_sun = (
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries**2
        - 0.00000048 * julian_centuries**3
    )

    # Eccentricity of the Earth's orbit eq. 24.4: $e$
    eccentricity_earth = (
        0.016708617
        - 0.000042037 * julian_centuries
        - 0.0000001236 * julian_centuries**2
    )

    # Sun equation of center chapter 24: $C$
    center_sun = (
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * julian_centuries**2)
        * np.sin(np.radians(mean_anomaly_sun))
        + (0.019993 - 0.000101 * julian_centuries)
        * np.sin(2 * np.radians(mean_anomaly_sun))
        + 0.000290 * np.sin(3 * np.radians(mean_anomaly_sun))
    )

    # Sun true longitude chapter 24: $\theta$ [degree]
    true_longitude_sun = mean_lon_sun + center_sun

    # Sun true anomaly: $v$ [degree]
    true_anomaly_sun = mean_anomaly_sun + center_sun

    # Sun radius vector (sun-earth distance) eq. 24.5: $R$ [astronomical units]
    # radius_vector_sun = (1.000001018 * (1 - eccentricity_earth**2)) / (
    #     1 + eccentricity_earth * np.cos(np.radians(true_anomaly_sun))
    # )

    # Longitude of the ascending node of the moon mean orbit: $\omega$ [degree]
    # Simplified polynomial for the reduced precision nutation obliquity
    # See page 132 of Meeus (1998)
    # Also called as a correction factor for nutation and aberration on page 152
    lon_asc_node_moon_ecliptic = (
        125.04452 - 1934.136 * julian_centuries
    )  # + 0.0020708 * julian_centuries**2 + julian_centuries**3/450000

    # Sun apparent longitude: $\lambda$
    apparent_longitude_sun = (
        true_longitude_sun
        - 0.00569
        - 0.00478 * np.sin(np.radians(lon_asc_node_moon_ecliptic))
    )

    nutation_longitude = (
        -17.20 * np.sin(np.radians(lon_asc_node_moon_ecliptic))
        - 1.32 * np.sin(2 * np.radians(mean_lon_sun))
        - 0.23 * np.sin(2 * np.radians(mean_lon_moon))
        + 0.21 * np.sin(2 * np.radians(lon_asc_node_moon_ecliptic))
    )

    # Nutation in obliquity: $\delta\epsilon$
    # Reduced precision at 0".1
    nutation_obliquity = (
        9.20 * np.cos(np.radians(lon_asc_node_moon_ecliptic))
        + 0.57 * np.cos(2 * np.radians(mean_lon_sun))
        + 0.10 * np.cos(2 * np.radians(mean_lon_moon))
        - 0.09 * np.cos(2 * np.radians(lon_asc_node_moon_ecliptic))
    )

    # Mean obliquity of the ecliptic eq. 21.2: $\epsilon_o$
    # Here, $\epsilon_o$ == $\epsilon$
    # The input value is given as 23° 26' 21".448
    # TODO: different values for the two formulas that should be the same.
    # mean_obliquity_ecliptic = (
    #     (23 + (26 / 60) + (21.448 / 3600))
    #     - 46.8150 * julian_centuries
    #     - 0.00059 * julian_centuries**2
    #     + 0.001813 * julian_centuries**3
    # )

    # Fromm noaa
    mean_obliquity_ecliptic = (
        23
        + (
            26
            + (
                (
                    21.448
                    - julian_centuries
                    * (
                        46.815
                        + julian_centuries * (0.00059 - julian_centuries * 0.001813)
                    )
                )
            )
            / 60
        )
        / 60
    )

    # True obliquity of the ecliptic chapter 21: $\epsilon$
    # Apparently, from NOAA spreadsheet, we use only the corrected mean obliquity of the ecliptic
    # true_obliquity_ecliptic = mean_obliquity_ecliptic + nutation_obliquity

    # Correction for the apparent position of the sun chapter 24: $\epsilon$
    # Here, $\epsilon_o$ == $\epsilon$
    mean_obliquity_ecliptic_cor = mean_obliquity_ecliptic
    +0.00256 * np.cos(np.radians(lon_asc_node_moon_ecliptic))

    # Right ascension of the sun eq. 24.6: $a$
    # + 0.00256 * math.cos(omega) is the correction factor when computing apparent position
    right_ascension_sun = np.degrees(
        np.arctan2(
            np.cos(np.radians(mean_obliquity_ecliptic_cor))
            * np.sin(np.radians(apparent_longitude_sun)),
            np.cos(np.radians(apparent_longitude_sun)),
        )
    )

    # Declination of the sun eq. 24.7: $\delta$
    declination_sun = np.degrees(
        np.arcsin(
            np.sin(np.radians(mean_obliquity_ecliptic_cor))
            * np.sin(np.radians(apparent_longitude_sun))
        )
    )

    # TODO: debug from here

    # Greenwich mean sideral time eq. 11.4: $\theta_o$
    greenwich_mean_sideral_time = (
        280.46061837
        + 360.9864736629 * (julian_day - 2451545.0)
        + 0.000387933 * julian_centuries**2
        - julian_centuries**3 / 38710000
    )

    # nutation in right ascension
    nutation_right_ascension = nutation_longitude * np.cos(
        np.radians(mean_obliquity_ecliptic_cor)
    )

    # Greenwich mean sideral time corrected for nutation
    greenwich_mean_sideral_time = greenwich_mean_sideral_time + nutation_right_ascension

    # Local hour angle: $H$
    local_hour_angle = greenwich_mean_sideral_time - obs_latitude - right_ascension_sun

    # Sun azimuth: $A$
    # Positive westward from south
    sun_azimuth = np.arctan2(
        np.sin(local_hour_angle),
        np.cos(local_hour_angle) * np.sin(obs_latitude)
        - np.tan(declination_sun) * np.cos(obs_latitude),
    )

    # Sun altitude: $h$
    # Positive above the horizon
    sun_altitude = np.sin(obs_latitude) * np.sin(declination_sun) + np.cos(
        obs_latitude
    ) * np.cos(declination_sun) * np.cos(local_hour_angle)

    sun_azimuth = np.degrees(sun_azimuth)
    sun_altitude = np.degrees(sun_altitude)

    return sun_azimuth, sun_altitude


def sun_geom_noaa(date_time: datetime.datetime, utc_offset, obs_latitude, obs_longitude):
    """ Calculate the solar zenith and azimuth angles with the NOAA algorithm
    Parameters
    ----------
    date_time: datetime
        The date time to calculate the solar zenith and azimuth angles
    utc_offset: int
        The UTC offset of the location in hours
    obs_latitude: float
        The latitude of the location in decimal degrees
    obs_longitude: float
        The longitude of the location in decimal degrees

    Returns
    -------
    sun_zenith: float
        The solar zenith angle in decimal degrees measured from the vertical (0 - 180)
    sun_azimuth: float
        The solar azimuth angle in decimal degrees measured clockwise from the north (0 - 360)

    Notes
    -----
    The algorithm used here is the one from the NOAA spreadsheet available at
    https://www.esrl.noaa.gov/gmd/grad/solcalc/calcdetails.html
    The main difference with the Meeus (1998) algorithm is the use of the equation of time
    """

    julian_day = date_time_to_julian_day(date_time)

    julian_centuries = (julian_day - 2451545.0) / 36525

    mean_lon_sun = (
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * julian_centuries**2
    ) % 360

    # Mean anomaly of the sun: $M$ [degree]
    mean_anomaly_sun = (
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries**2
        - 0.00000048 * julian_centuries**3
    )

    # Eccentricity of the Earth's orbit: $e$
    eccentricity_earth = (
        0.016708617
        - 0.000042037 * julian_centuries
        - 0.0000001236 * julian_centuries**2
    )

    # Sun equation of center: $C$
    center_sun = (
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * julian_centuries**2)
        * np.sin(np.radians(mean_anomaly_sun))
        + (0.019993 - 0.000101 * julian_centuries)
        * np.sin(2 * np.radians(mean_anomaly_sun))
        + 0.000290 * np.sin(3 * np.radians(mean_anomaly_sun))
    )

    # Sun true longitude: $\theta$ [degree]
    true_longitude_sun = mean_lon_sun + center_sun

    # Sun true anomaly: $v$ [degree]
    true_anomaly_sun = mean_anomaly_sun + center_sun

    # Longitude of the ascending node of the moon mean orbit: $\omega$ [degree]
    # Simplified polynomial for the reduced precision nutation obliquity
    # See page 132 of Meeus (1998)
    # Also called as a correction factor for nutation and aberration on page 152
    lon_asc_node_moon_ecliptic = (
        125.04452 - 1934.136261 * julian_centuries
    )  # + 0.0020708 * julian_centuries**2 + julian_centuries**3/450000

    # Sun apparent longitude: $\lambda$
    apparent_longitude_sun = (
        true_longitude_sun
        - 0.00569
        - 0.00478 * np.sin(np.radians(lon_asc_node_moon_ecliptic))
    )

    # Fromm noaa
    mean_obliquity_ecliptic = (
        23
        + (
            26
            + (
                (
                    21.448
                    - julian_centuries
                    * (
                        46.815
                        + julian_centuries * (0.00059 - julian_centuries * 0.001813)
                    )
                )
            )
            / 60
        )
        / 60
    )

    # Mean obliquity of the ecliptic [degree]
    mean_obliquity_ecliptic_cor = mean_obliquity_ecliptic
    +0.00256 * np.cos(np.radians(lon_asc_node_moon_ecliptic))

    # Right ascension of the sun: $a$ [degree]
    right_ascension_sun = np.degrees(
        np.arctan2(
            np.cos(np.radians(mean_obliquity_ecliptic_cor))
            * np.sin(np.radians(apparent_longitude_sun)),
            np.cos(np.radians(apparent_longitude_sun)),
        )
    )

    # Declination of the sun
    declination_sun = np.degrees(
        np.arcsin(
            np.sin(np.radians(mean_obliquity_ecliptic_cor))
            * np.sin(np.radians(apparent_longitude_sun))
        )
    )

    var_y = np.tan(np.radians(mean_obliquity_ecliptic_cor / 2)) * np.tan(
        np.radians(mean_obliquity_ecliptic_cor / 2)
    )

    # Equation of time from NOAA, don't know where it comes from
    eq_of_time = 4 * np.degrees(
        var_y * np.sin(2 * np.radians(mean_lon_sun))
        - 2 * eccentricity_earth * np.sin(np.radians(mean_anomaly_sun))
        + 4
        * eccentricity_earth
        * var_y
        * np.sin(np.radians(mean_anomaly_sun))
        * np.cos(2 * np.radians(mean_lon_sun))
        - 0.5 * var_y * var_y * np.sin(4 * np.radians(mean_lon_sun))
        - 1.25
        * eccentricity_earth
        * eccentricity_earth
        * np.sin(2 * np.radians(mean_anomaly_sun))
    )

    decimal_day = (
        date_time.hour * 3600 + date_time.minute * 60 + date_time.second
    ) / 86400

    # True solar time from NOAA [minute]
    # In spreadsheet E is time in decimal day
    # B4 is longitude positive towards east
    # B5 is time zone, positive towards east
    true_solar_time = (
        decimal_day * 1440 + eq_of_time + 4 * obs_longitude - 60 * utc_offset
    ) % 1440

    # Hour angle from NOAA [degree]
    if np.all(true_solar_time / 4 < 0):
        local_hour_angle = true_solar_time / 4 + 180
    elif np.all(true_solar_time / 4 >= 0):
        local_hour_angle = true_solar_time / 4 - 180
    else:
        raise ValueError("The grid span on both side of 0 degree")

    # Sun zenith angle from NOAA [deg]
    # B3 is latitude positive to north
    sun_zenith = np.degrees(
        np.arccos(
            np.sin(np.radians(obs_latitude)) * np.sin(np.radians(declination_sun))
            + np.cos(np.radians(obs_latitude))
            * np.cos(np.radians(declination_sun))
            * np.cos(np.radians(local_hour_angle))
        )
    )

    # Sun azimuth from NOAA from [degree]
    if np.all(local_hour_angle > 0):
        sun_azimuth = (
            np.degrees(
                np.arccos(
                    (
                        (
                            np.sin(np.radians(obs_latitude))
                            * np.cos(np.radians(sun_zenith))
                        )
                        - np.sin(np.radians(declination_sun))
                    )
                    / (
                        np.cos(np.radians(obs_latitude))
                        * np.sin(np.radians(sun_zenith))
                    )
                )
            )
            + 180
        ) % 360
    elif np.all(local_hour_angle <= 0):
        sun_azimuth = (
            540
            - np.degrees(
                np.arccos(
                    (
                        (
                            np.sin(np.radians(obs_latitude))
                            * np.cos(np.radians(sun_zenith))
                        )
                        - np.sin(np.radians(declination_sun))
                    )
                    / (
                        np.cos(np.radians(obs_latitude))
                        * np.sin(np.radians(sun_zenith))
                    )
                )
            )
        ) % 360
    else:
        raise ValueError("The grid span on both side of 0 degree")

    return sun_zenith, sun_azimuth


if __name__ == "__main__":
    obs_latitude = 49.077464
    obs_longitude = -68.323974
    time_zone = -4
    date_time = datetime.datetime(2019, 8, 18, 17, 35, 29)

    sun_geom_noaa(date_time, time_zone, obs_latitude, obs_longitude)

    # sun_geom_meeus(date_time, obs_latitude)
