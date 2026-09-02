"""Writers for mod22 (TRITON gtiff -> map).

Streams the consolidated space-time cube to a compressed netCDF one timestep at a
time and emits the pixel-maximum GeoTIFF once the stream has been reduced. Both
carry the grid geometry and CRS taken from the source TRITON ``.vrt`` (no
reprojection); missing/masked cells use the shared NODATA fill value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import netCDF4
import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()

# Per-variable CF metadata for the netCDF data variable.
VAR_META = {
    "H": {"units": "m", "long_name": "water depth"},
    "MH": {"units": "m", "long_name": "maximum water depth (running envelope)"},
    "QX": {"units": "m2 s-1", "long_name": "unit discharge, x-direction"},
    "QY": {"units": "m2 s-1", "long_name": "unit discharge, y-direction"},
    "V": {"units": "m s-1", "long_name": "depth-averaged velocity magnitude"},
}
_TIME_UNITS = "hours since 1970-01-01 00:00:00"


def open_cube(
    path: Path, var: str, grid: Dict, times: pd.DatetimeIndex, nodata: float
) -> Tuple[netCDF4.Dataset, netCDF4.Variable]:
    """Create the netCDF cube and return the open dataset + data variable handle.

    Time slices are written incrementally by the caller via ``data[t] = slice``.
    """
    ny, nx = grid["ny"], grid["nx"]
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", len(times))
    ds.createDimension("y", ny)
    ds.createDimension("x", nx)

    tv = ds.createVariable("time", "f8", ("time",))
    tv.units = _TIME_UNITS
    tv.calendar = "standard"
    tv.standard_name = "time"
    tv[:] = netCDF4.date2num(times.to_pydatetime(), _TIME_UNITS, "standard")

    yv = ds.createVariable("y", "f8", ("y",))
    yv.standard_name = "projection_y_coordinate"
    yv.units = "m"
    yv[:] = grid["y"]
    xv = ds.createVariable("x", "f8", ("x",))
    xv.standard_name = "projection_x_coordinate"
    xv.units = "m"
    xv[:] = grid["x"]

    crs = ds.createVariable("crs", "i4")
    if grid.get("wkt"):
        crs.crs_wkt = grid["wkt"]
        crs.spatial_ref = grid["wkt"]
    if grid.get("epsg"):
        crs.epsg_code = grid["epsg"]

    chunk = (1, min(ny, 512), min(nx, 512))
    data = ds.createVariable(
        var,
        "f4",
        ("time", "y", "x"),
        zlib=True,
        complevel=4,
        chunksizes=chunk,
        fill_value=np.float32(nodata),
    )
    meta = VAR_META.get(var, {})
    data.units = meta.get("units", "")
    data.long_name = meta.get("long_name", var)
    data.grid_mapping = "crs"

    ds.Conventions = "CF-1.8"
    ds.source = "TRITON 2D hydraulic model gtiff outputs (mod22)"
    return ds, data


def write_max_tiff(path: Path, arr: np.ndarray, grid: Dict, nodata: float) -> None:
    """Write the pixel-maximum 2-D *arr* as a single-band GeoTIFF (LZW, tiled)."""
    ny, nx = grid["ny"], grid["nx"]
    out = np.where(np.isfinite(arr), arr, np.float32(nodata)).astype(np.float32)
    # clear the target and any stale sidecars so GDAL's implicit delete can't fail
    for sidecar in (path, Path(f"{path}.ovr"), Path(f"{path}.aux.xml")):
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(path), nx, ny, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"]
    )
    ds.SetGeoTransform(grid["gt"])
    if grid.get("wkt"):
        ds.SetProjection(grid["wkt"])
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    band.WriteArray(out)
    band.FlushCache()
    ds = None
