import datetime
import xarray as xr


def interp_gmao(files, lon, lat, dt):
    """
    Interpolates GMAO GEOS data from hourly files to image lon, lat and time
    Parameters
    ----------
    files
    lon
    lat
    dt

    Returns
    -------

    """

    def prepro_gmao(ds):
        """
        add time coordinate to gmao dataset
        Parameters
        ----------
        ds

        Returns
        -------

        """
        ds = ds.assign_coords(
            {
                "T": datetime.datetime.strptime(
                    ds.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=datetime.timezone.utc).timestamp()
            }
        )
        ds = ds.expand_dims("T")
        return ds

    net_ds = xr.open_mfdataset(
        files, preprocess=prepro_gmao, parallel=False, engine="netcdf4"
    )

    # Strange bug with type error numpy.float64 since saved lat lon as f8 in nc L1 conversion
    gmao_interp = net_ds.interp(lat=float(lat), lon=float(lon), T=dt.timestamp())
    gmao_interp = gmao_interp.compute()

    # interpolate for the whole image
    # interpolated = da.interp_like(image)

    return gmao_interp
