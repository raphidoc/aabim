import math
from collections import defaultdict

import numpy as np
import fiona
from rasterio.features import rasterize
from rasterio.transform import from_origin
from netCDF4 import Dataset
from pyproj import CRS


def _ceildiv(a, b):
    return int(math.ceil(a / b))


def _build_grid(bounds, pixel_size):
    minx, miny, maxx, maxy = bounds
    res = float(pixel_size)

    width = _ceildiv(maxx - minx, res)
    height = _ceildiv(maxy - miny, res)

    transform = from_origin(minx, maxy, res, res)  # north-up
    x = minx + res * (0.5 + np.arange(width))
    y = maxy - res * (0.5 + np.arange(height))  # descending
    return width, height, transform, x, y


def _affine_to_gdal_geotransform(transform):
    # rasterio Affine: (a, b, c, d, e, f) where:
    # x = a*col + b*row + c
    # y = d*col + e*row + f
    # GDAL GeoTransform: [c, a, b, f, d, e]
    return (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)


def _nc_dtype_and_fill(arr: np.ndarray):
    """
    netCDF4 can't store boolean attrs and can't use NaN fill for integer variables.
    - floats -> f4 with NaN fill
    - ints/bools -> i4 with -9999 fill
    - others -> f4 with NaN fill
    """
    if np.issubdtype(arr.dtype, np.floating):
        return "f4", np.nan, arr.astype(np.float32, copy=False)

    if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
        fill = np.int32(-9999)
        return "i4", fill, arr.astype(np.int32, copy=False)

    return "f4", np.nan, arr.astype(np.float32)


def write_netcdf_cf_gdal(
    output_nc: str,
    x: np.ndarray,
    y: np.ndarray,
    wl: np.ndarray,
    transform,
    crs_wkt: str,
    seg_vars_2d: dict,   # name -> (y,x)
    spec_vars_3d: dict,  # name -> (y,x,wl) in memory; we will write as (wl,y,x)
    wl_units: str = "nm",
    compress: bool = True,
):
    H, W = len(y), len(x)
    B = len(wl)

    if not crs_wkt:
        raise ValueError("crs_wkt is empty; cannot write CF/GDAL georeferencing.")

    crs = CRS.from_wkt(crs_wkt)
    cf = crs.to_cf()

    gt = _affine_to_gdal_geotransform(transform)
    geotransform_str = " ".join(str(v) for v in gt)

    comp = dict(zlib=True, complevel=4, shuffle=True) if compress else {}

    with Dataset(output_nc, "w", format="NETCDF4") as nc:
        # Dimensions
        nc.createDimension("y", H)
        nc.createDimension("x", W)
        nc.createDimension("wl", B)

        # Coordinates
        vx = nc.createVariable("x", "f8", ("x",))
        vy = nc.createVariable("y", "f8", ("y",))
        vwl = nc.createVariable("wl", "f4", ("wl",))

        vx[:] = x
        vy[:] = y
        vwl[:] = wl

        # CF coordinate metadata
        vx.standard_name = "projection_x_coordinate"
        vy.standard_name = "projection_y_coordinate"
        vx.long_name = "x coordinate of projection"
        vy.long_name = "y coordinate of projection"
        vx.axis = "X"
        vy.axis = "Y"

        vwl.standard_name = "radiation_wavelength"
        vwl.units = wl_units
        vwl.axis = "Z"  # not strictly required, but helps some readers

        # Units for x/y from CRS (usually meters for UTM)
        try:
            units = crs.axis_info[0].unit_name
            ux = "m" if units in ("metre", "meter") else units
            vx.units = ux
            vy.units = ux
        except Exception:
            vx.units = "m"
            vy.units = "m"

        # grid_mapping variable (scalar)
        gm = nc.createVariable("grid_mapping", "i4", ())
        for k, v in cf.items():
            if isinstance(v, (np.bool_, bool)):
                v = int(v)
            gm.setncattr(k, v)

        if "grid_mapping_name" in cf:
            gm.grid_mapping_name = cf["grid_mapping_name"]

        # GDAL extras
        wkt_out = crs.to_wkt()
        gm.spatial_ref = wkt_out
        gm.crs_wkt = wkt_out
        gm.GeoTransform = geotransform_str

        gm.affine_transform = (
            float(transform.a), float(transform.b), float(transform.c),
            float(transform.d), float(transform.e), float(transform.f),
        )

        def _tag_2d(v):
            v.grid_mapping = "grid_mapping"
            v.coordinates = "x y"

        def _tag_3d(v):
            v.grid_mapping = "grid_mapping"
            # include wl coord explicitly
            v.coordinates = "wl x y"

        # 2D segment vars (y,x)
        for name, arr in seg_vars_2d.items():
            nc_dtype, fill_value, out = _nc_dtype_and_fill(arr)
            v = nc.createVariable(name, nc_dtype, ("y", "x"), fill_value=fill_value, **comp)
            v[:, :] = out
            _tag_2d(v)

        # 3D spectra vars: WRITE AS (wl, y, x) for GDAL/QGIS sanity
        for name, arr in spec_vars_3d.items():
            # arr comes in as (y,x,wl); transpose to (wl,y,x)
            arr_wyx = np.transpose(arr, (2, 0, 1))
            nc_dtype, fill_value, out = _nc_dtype_and_fill(arr_wyx)
            v = nc.createVariable(name, nc_dtype, ("wl", "y", "x"), fill_value=fill_value, **comp)
            v[:, :, :] = out
            v.wl_units = wl_units
            _tag_3d(v)

        # Global attrs (netCDF-safe)
        nc.setncattr("Conventions", "CF-1.8")
        nc.setncattr("crs_wkt", wkt_out)
        nc.setncattr("GeoTransform", geotransform_str)


def main(
    input_gpkg: str,
    output_nc: str,
    segments_layer: str = "segments",
    spectra_layer: str = "spectra",
    seg_id_field: str = "seg_id",
    wl_field: str = "wl_nm",
    pixel_size: float = 10.0,
    all_touched: bool = False,
    compress: bool = True,
    wl_units: str = "nm",
):
    # --- Read segments ---
    with fiona.open(input_gpkg, layer=segments_layer) as seg_src:
        seg_schema = seg_src.schema
        seg_crs_wkt = seg_src.crs_wkt
        seg_bounds = seg_src.bounds

        seg_fields = list(seg_schema["properties"].keys())
        if seg_id_field not in seg_fields:
            raise ValueError(f"{segments_layer} layer must contain '{seg_id_field}'")

        width, height, transform, x, y = _build_grid(seg_bounds, pixel_size)

        shapes_seg_id = []
        seg_attrs_by_id = {}

        for feat in seg_src:
            geom = feat.get("geometry", None)
            if geom is None:
                continue
            props = dict(feat.get("properties", {}))
            sid = props.get(seg_id_field, None)
            if sid is None:
                continue
            sid = int(sid)
            shapes_seg_id.append((geom, sid))
            seg_attrs_by_id[sid] = props

    seg_id_raster = rasterize(
        shapes=shapes_seg_id,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=all_touched,
    )

    seg_ids_present = np.unique(seg_id_raster)
    seg_ids_present = seg_ids_present[seg_ids_present != 0]

    # --- Read spectra table ---
    with fiona.open(input_gpkg, layer=spectra_layer) as sp_src:
        sp_fields = list(sp_src.schema["properties"].keys())
        if seg_id_field not in sp_fields or wl_field not in sp_fields:
            raise ValueError(f"{spectra_layer} must contain '{seg_id_field}' and '{wl_field}'")

        spectra_value_fields = [f for f in sp_fields if f not in (seg_id_field, wl_field)]

        spectra_by_seg = defaultdict(dict)  # spectra_by_seg[seg_id][wl] -> dict(field->value)
        wls = set()

        for feat in sp_src:
            p = dict(feat.get("properties", {}))
            sid = p.get(seg_id_field, None)
            wlv = p.get(wl_field, None)
            if sid is None or wlv is None:
                continue
            sid = int(sid)
            wlv = float(wlv)
            wls.add(wlv)
            spectra_by_seg[sid][wlv] = {k: p.get(k, None) for k in spectra_value_fields}

    wl = np.array(sorted(wls), dtype=np.float32)

    # IMPORTANT: avoid float-key mismatch by indexing wavelengths by rounded key
    # (optional but robust if wl values are like 549.9999997)
    def wl_key(v):
        return round(float(v), 6)

    spectra_by_seg_rounded = defaultdict(dict)
    for sid, d in spectra_by_seg.items():
        for wv, rec in d.items():
            spectra_by_seg_rounded[sid][wl_key(wv)] = rec

    wl_keys = np.array([wl_key(v) for v in wl], dtype=np.float64)

    # --- 2D outputs ---
    seg_vars_2d = {seg_id_field: seg_id_raster.astype(np.int32)}

    for field in seg_fields:
        if field == seg_id_field:
            continue
        arr = np.full((height, width), np.nan, dtype=np.float32)
        for sid in seg_ids_present:
            sid_int = int(sid)
            val = seg_attrs_by_id.get(sid_int, {}).get(field, None)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            arr[seg_id_raster == sid_int] = fval
        seg_vars_2d[field] = arr

    # --- 3D outputs in memory as (y,x,wl) ---
    spec_vars_3d = {
        sf: np.full((height, width, wl.size), np.nan, dtype=np.float32)
        for sf in spectra_value_fields
    }

    for sid in seg_ids_present:
        sid_int = int(sid)
        mask = (seg_id_raster == sid_int)
        if not mask.any():
            continue

        spec_map = spectra_by_seg_rounded.get(sid_int, {})
        if not spec_map:
            continue

        for j, wk in enumerate(wl_keys):
            rec = spec_map.get(float(wk), None)
            if rec is None:
                continue
            for sf in spectra_value_fields:
                v = rec.get(sf, None)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                spec_vars_3d[sf][mask, j] = fv

    # --- Write NetCDF (spectra written as wl,y,x) ---
    write_netcdf_cf_gdal(
        output_nc=output_nc,
        x=x.astype(np.float64),
        y=y.astype(np.float64),
        wl=wl.astype(np.float32),
        transform=transform,
        crs_wkt=seg_crs_wkt,
        seg_vars_2d=seg_vars_2d,
        spec_vars_3d=spec_vars_3d,
        wl_units=wl_units,
        compress=compress,
    )

    return {
        "output_nc": output_nc,
        "shape_yx": (height, width),
        "n_wl": int(wl.size),
        "n_seg_vars": int(len(seg_vars_2d)),
        "n_spec_vars": int(len(spec_vars_3d)),
    }


if __name__ == "__main__":
    main(
        input_gpkg="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments_inverted.gpkg",
        output_nc="/D/Data/WISE/ACI-12A/el_sma/segmentation_outputs/segments_inverted.nc",
        pixel_size=5.0,
        all_touched=False,
        compress=True,
    )