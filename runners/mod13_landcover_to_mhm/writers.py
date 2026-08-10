"""Writers for the mHM ArcGIS ASCII land-cover file and an optional QA NetCDF."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def write_asc(path: Path, header: dict, grid: np.ndarray, nodata: int) -> None:
    """
    Write an ArcGIS ASCII grid mHM can read.

    Six-line header + row-major integer grid, north-up (first row = northernmost).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if grid.shape != (header["nrows"], header["ncols"]):
        raise AssertionError(
            f"Grid shape {grid.shape} does not match header "
            f"({header['nrows']}, {header['ncols']})."
        )

    header_lines = [
        f"ncols        {header['ncols']}",
        f"nrows        {header['nrows']}",
        f"xllcorner    {float(header['xllcorner'])}",
        f"yllcorner    {float(header['yllcorner'])}",
        f"cellsize     {int(header['cellsize'])}",
        f"NODATA_value {nodata}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        np.savetxt(f, grid.astype(np.int32), fmt="%d")

    log.info("Wrote %s", path)


def write_nc_copy(
    path: Path,
    grid: np.ndarray,
    header: dict,
    crs_wkt: str,
    nodata: int,
    year: int,
) -> None:
    """Optional NetCDF copy for GIS/QA inspection. Not read by mHM."""
    import netCDF4 as nc

    path.parent.mkdir(parents=True, exist_ok=True)
    cs = header["cellsize"]
    xll, yll = header["xllcorner"], header["yllcorner"]
    ncols, nrows = header["ncols"], header["nrows"]

    xs = xll + cs / 2 + np.arange(ncols) * cs
    ys = yll + cs / 2 + np.arange(nrows) * cs
    ys = ys[::-1]   # north-up

    with nc.Dataset(path, "w", format="NETCDF4") as fh:
        fh.createDimension("x", ncols)
        fh.createDimension("y", nrows)

        xv = fh.createVariable("x", "f8", ("x",)); xv.axis = "X"; xv.units = "m"; xv[:] = xs
        yv = fh.createVariable("y", "f8", ("y",)); yv.axis = "Y"; yv.units = "m"; yv[:] = ys

        lc = fh.createVariable(
            "land_cover", "i4", ("y", "x"),
            zlib=True, complevel=4, fill_value=nodata,
        )
        lc.long_name    = "mHM land-cover class"
        lc.units        = "1"
        lc.flag_values  = np.array([1, 2, 3], dtype="i4")
        lc.flag_meanings = "forest impervious pervious"
        lc.missing_value = nodata
        lc[:] = grid.astype(np.int32)

        crs_var = fh.createVariable("crs", "i4")
        crs_var.spatial_ref = crs_wkt
        lc.grid_mapping = "crs"

        fh.year        = year
        fh.source      = "Space Intelligence GHL (via Earthmover Arraylake)"
        fh.description = "mHM land cover (1=Forest, 2=Impervious, 3=Pervious)"

    log.info("Wrote %s", path)