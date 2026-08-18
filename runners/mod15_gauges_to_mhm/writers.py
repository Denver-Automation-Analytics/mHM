"""Grid writers for idgauges output."""

from __future__ import annotations
import logging
from pathlib import Path

import netCDF4 as nc4
import numpy as np

log = logging.getLogger(__name__)

_CHUNK_SIZE = 32

def write_asc(path: Path, header: dict, grid: np.ndarray, nodata: int) -> None:
    """Six-line header + row-major integer grid, north-up (row 0 = northernmost)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if grid.shape != (header["nrows"], header["ncols"]):
        raise AssertionError(
            f"Grid shape {grid.shape} does not match header "
            f"({header['nrows']}, {header['ncols']})."
        )

    header_lines = [
        f"ncols        {header['ncols']}",
        f"nrows        {header['nrows']}",
        f"xllcorner    {header['xllcorner']}",
        f"yllcorner    {header['yllcorner']}",
        f"cellsize     {int(header['cellsize'])}",
        f"NODATA_value {nodata}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        np.savetxt(f, grid.astype(np.int32), fmt="%d")

    log.info("Wrote %s", path)


def write_nc(path: Path, header: dict, grid: np.ndarray, nodata: int,
             block_size: int = 256) -> None:
    """Write idgauges as NetCDF formatted for mHM ingestion (EPSG:4326)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    ncols = header["ncols"]
    nrows = header["nrows"]
    cs    = header["cellsize"]
    xll   = header["xllcorner"]
    yll   = header["yllcorner"]

    # Center-of-cell coordinates; y is decreasing (north-up, row 0 = northernmost)
    x_coords = xll + (np.arange(ncols) + 0.5) * cs
    y_coords = yll + (nrows - 0.5 - np.arange(nrows)) * cs

    with nc4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.6"

        ds.createDimension("x", ncols)
        ds.createDimension("y", nrows)

        xv = ds.createVariable("x", "f8", ("x",))
        xv.axis = "X"
        xv.standard_name = "longitude"
        xv.units = "degree_east"
        xv[:] = x_coords

        yv = ds.createVariable("y", "f8", ("y",))
        yv.axis = "Y"
        yv.standard_name = "latitude"
        yv.units = "degree_north"
        yv[:] = y_coords

        # mHM requires (x, y) dimension order; grid is (nrows, ncols) so transpose
        dv = ds.createVariable("idgauges", "i4", ("x", "y"),
                               fill_value=np.int32(nodata),
                               zlib=True, complevel=4,
                               chunksizes=(_CHUNK_SIZE, _CHUNK_SIZE))
        dv.coordinates = "x y"
        for row_start in range(0, nrows, block_size):
            row_count = min(block_size, nrows - row_start)
            block = grid[row_start:row_start + row_count, :]   # (row_count, ncols)
            dv[:, row_start:row_start + row_count] = block.T   # (ncols, row_count) = (x, y)

    log.info("Wrote %s", path)