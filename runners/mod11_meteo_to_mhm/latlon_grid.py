"""
Create mHM-compatible latlon.nc with multi-level (L0/L1/L11) grid definitions.

Migrated from mHM's pre-proc/create_latlon.py:
  * CLI (optparse) removed
  * ufz.writenetcdf replaced with direct netCDF4 calls
  * pyproj.Proj(init=...) replaced with pyproj.Transformer.from_crs(...)
  * Headers accepted as either a dict or a path to header.txt
  * L1 required; L0 and L11 optional (L0 will be added when the morphology
    module is implemented)
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import netCDF4 as nc
from pyproj import Transformer

log = logging.getLogger(__name__)

HeaderLike = Union[dict, str, Path]


# --------------------------------------------------------------------------- #
# Header handling
# --------------------------------------------------------------------------- #
def _header_from_nc(path: Path) -> dict:
    """Derive an mHM-style header dict from the 1-D x/y coords of a NetCDF file."""
    with nc.Dataset(path) as ds:
        x = ds["x"][:]
        y = ds["y"][:]
        fill = getattr(ds.variables.get(list(ds.variables)[-1], object()), "_FillValue", -9999.0)
    cs = float(x[1] - x[0])          # positive west→east step
    cs_y = float(y[0] - y[1])        # positive north→south step (y is north-down)
    return {
        "ncols":        int(x.shape[0]),
        "nrows":        int(y.shape[0]),
        "xllcorner":    float(x[0])   - cs   / 2,
        "yllcorner":    float(y[-1])  - cs_y / 2,
        "cellsize":     cs,
        "NODATA_value": float(fill),
    }


def _load_header(header: HeaderLike) -> dict:
    """Accept an in-memory header dict or a path to an mHM header.txt/.nc file."""
    if isinstance(header, dict):
        parsed = header
    else:
        path = Path(header)
        if not path.is_file():
            raise FileNotFoundError(f"Header file not found: {path}")
        if path.suffix == ".nc":
            return _header_from_nc(path)
        parsed = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0][0].isdigit() or parts[0][0] == "-":
                break  # hit the data rows of an .asc file
            parsed[parts[0]] = parts[1]

    # normalize types
    return {
        "ncols":        int(parsed["ncols"]),
        "nrows":        int(parsed["nrows"]),
        "xllcorner":    float(parsed["xllcorner"]),
        "yllcorner":    float(parsed["yllcorner"]),
        "cellsize":     float(parsed["cellsize"]),
        "NODATA_value": float(parsed["NODATA_value"]),
    }


# --------------------------------------------------------------------------- #
# Coordinate transformation
# --------------------------------------------------------------------------- #
def xx_to_latlon(xx: np.ndarray,
                 yy: np.ndarray,
                 coord_sys: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project (x, y) grids in `coord_sys` to (lon, lat) in WGS84.
    An empty `coord_sys` string means: no transformation (data already lon/lat).
    """
    if not coord_sys:
        return xx, yy
    transformer = Transformer.from_crs(coord_sys, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return lon, lat


# --------------------------------------------------------------------------- #
# Header -> lat/lon grids
# --------------------------------------------------------------------------- #
def header_to_latlon(header: HeaderLike,
                     coord_sys: str,
                     do_corners: bool = False):
    """Build cell-center (and optionally corner) lat/lon arrays from a header."""
    h = _load_header(header)
    ncols, nrows = h["ncols"], h["nrows"]
    xll, yll, cs = h["xllcorner"], h["yllcorner"], h["cellsize"]
    missing      = h["NODATA_value"]

    # cell-center grid (yy runs north-down so row 0 = northernmost)
    xx = np.linspace(xll + cs / 2,                  xll + cs / 2 + (ncols - 1) * cs, ncols)
    yy = np.linspace(yll + cs / 2 + (nrows - 1) * cs, yll + cs / 2,                   nrows)
    xx, yy = np.meshgrid(xx, yy)

    lons, lats = xx_to_latlon(xx, yy, coord_sys)

    if not do_corners:
        return lons, lats, xx, yy, missing

    # cell-corner grids (preserved from the original for optional CF bounds use)
    def _corner(x_off: float, y_off: float):
        cxx = np.linspace(xll + x_off,                 xll + x_off + (ncols - 1) * cs, ncols)
        cyy = np.linspace(yll + y_off + (nrows - 1) * cs, yll + y_off,                  nrows)
        cxx, cyy = np.meshgrid(cxx, cyy)
        return xx_to_latlon(cxx, cyy, coord_sys)

    ul_lons, ul_lats = _corner(0.0, cs)     # upper-left
    ur_lons, ur_lats = _corner(cs,  cs)     # upper-right
    lr_lons, lr_lats = _corner(cs,  0.0)    # lower-right
    ll_lons, ll_lats = _corner(0.0, 0.0)    # lower-left

    return (lons, lats, xx, yy, missing,
            ll_lons, lr_lons, ur_lons, ul_lons,
            ll_lats, lr_lats, ur_lats, ul_lats)


# --------------------------------------------------------------------------- #
# NetCDF writing
# --------------------------------------------------------------------------- #
_CHUNK_ROWS = 512   # rows per batch; keeps peak memory ~400 MB at 3-km resolution


def _write_level(fh: nc.Dataset,
                 lons: np.ndarray,
                 lats: np.ndarray,
                 xx: np.ndarray,
                 yy: np.ndarray,
                 missing: float,
                 suffix: Tuple[str, str]) -> None:
    """Write xc*, yc*, lon*, lat* variables for one mHM level."""
    sfx, long_sfx = suffix

    x_dim = f"xc{sfx}"
    y_dim = f"yc{sfx}"

    if x_dim not in fh.dimensions:
        fh.createDimension(x_dim, xx.shape[1])
    if y_dim not in fh.dimensions:
        fh.createDimension(y_dim, yy.shape[0])

    xc = fh.createVariable(x_dim, "f8", (x_dim,))
    xc.axis = "X"
    xc[:] = xx[0, :]

    yc = fh.createVariable(y_dim, "f8", (y_dim,))
    yc.axis = "Y"
    yc[:] = yy[:, 0]

    lon_var = fh.createVariable(f"lon{sfx}", "f8", (y_dim, x_dim),
                                zlib=True, complevel=4)
    lon_var.units         = "degrees_east"
    lon_var.long_name     = f"longitude{long_sfx}"
    lon_var.missing_value = float(missing)
    lon_var[:] = lons

    lat_var = fh.createVariable(f"lat{sfx}", "f8", (y_dim, x_dim),
                                zlib=True, complevel=4)
    lat_var.units         = "degrees_north"
    lat_var.long_name     = f"latitude{long_sfx}"
    lat_var.missing_value = float(missing)
    lat_var[:] = lats


def _write_level_chunked(
    fh: nc.Dataset,
    header: HeaderLike,
    coord_sys: str,
    suffix: Tuple[str, str],
) -> None:
    """Like _write_level but streams row-batches so peak memory stays bounded."""
    sfx, long_sfx = suffix
    h = _load_header(header)
    nrows, ncols = h["nrows"], h["ncols"]
    xll, yll, cs = h["xllcorner"], h["yllcorner"], h["cellsize"]
    missing = float(h["NODATA_value"])

    x_centers = np.linspace(xll + cs / 2, xll + cs / 2 + (ncols - 1) * cs, ncols)
    y_centers = np.linspace(yll + cs / 2 + (nrows - 1) * cs, yll + cs / 2,  nrows)

    x_dim, y_dim = f"xc{sfx}", f"yc{sfx}"
    if x_dim not in fh.dimensions:
        fh.createDimension(x_dim, ncols)
    if y_dim not in fh.dimensions:
        fh.createDimension(y_dim, nrows)

    xc = fh.createVariable(x_dim, "f8", (x_dim,))
    xc.axis = "X"
    xc[:] = x_centers

    yc = fh.createVariable(y_dim, "f8", (y_dim,))
    yc.axis = "Y"
    yc[:] = y_centers

    lon_var = fh.createVariable(
        f"lon{sfx}", "f8", (y_dim, x_dim),
        zlib=True, complevel=4, chunksizes=(_CHUNK_ROWS, ncols),
    )
    lon_var.units         = "degrees_east"
    lon_var.long_name     = f"longitude{long_sfx}"
    lon_var.missing_value = missing

    lat_var = fh.createVariable(
        f"lat{sfx}", "f8", (y_dim, x_dim),
        zlib=True, complevel=4, chunksizes=(_CHUNK_ROWS, ncols),
    )
    lat_var.units         = "degrees_north"
    lat_var.long_name     = f"latitude{long_sfx}"
    lat_var.missing_value = missing

    transformer = (
        Transformer.from_crs(coord_sys, "EPSG:4326", always_xy=True)
        if coord_sys else None
    )

    for row0 in range(0, nrows, _CHUNK_ROWS):
        row1 = min(row0 + _CHUNK_ROWS, nrows)
        # broadcast x across the batch without copying; copy y to allow transform
        xx_batch = np.broadcast_to(x_centers, (row1 - row0, ncols))
        yy_batch = np.repeat(y_centers[row0:row1, None], ncols, axis=1)
        if transformer is not None:
            lons, lats = transformer.transform(xx_batch, yy_batch)
        else:
            lons = np.array(xx_batch)   # materialise the broadcast view
            lats = yy_batch
        lon_var[row0:row1, :] = lons
        lat_var[row0:row1, :] = lats
        log.debug("lat/lon%s: rows %d–%d / %d", sfx, row0, row1, nrows)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def create_latlon(
    out_file: Union[str, Path],
    coord_sys: str,
    header_l1: HeaderLike,
    header_l0: Optional[HeaderLike]  = None,
    header_l11: Optional[HeaderLike] = None,
) -> None:
    """
    Create the latlon.nc file that mHM requires.

    Parameters
    ----------
    out_file   : path
        Output NetCDF path (mHM expects the filename `latlon.nc`).
    coord_sys  : str
        CRS for the projected header coordinates. Accepts any string pyproj can
        parse: 'EPSG:xxxx', a PROJ pipeline, or a full WKT. Pass '' if the
        headers are already in geographic lon/lat.
    header_l1  : dict or path (REQUIRED)
        Level-1 (hydrologic) header. Written as xc, yc, lon, lat.
    header_l0  : dict or path (optional)
        Level-0 (morphological) header. Written as xc_l0, yc_l0, lon_l0, lat_l0.
        Add later when the DEM/morphology module is implemented.
    header_l11 : dict or path (optional)
        Level-11 (routing) header. Written as xc_l11, yc_l11, lon_l11, lat_l11.
    """
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(out_path, "w", format="NETCDF4") as fh:
        fh.description = "lat lon file"
        fh.projection  = coord_sys if coord_sys else "geographic"
        fh.history     = "Created " + time.ctime(time.time())

        if header_l0 is not None:
            _write_level_chunked(fh, header_l0, coord_sys, ("_l0", " at level 0"))

        # L1 is always required
        lons, lats, xx, yy, miss = header_to_latlon(header_l1, coord_sys)
        _write_level(fh, lons, lats, xx, yy, miss, ("", " at level 1"))

        if header_l11 is not None:
            lons, lats, xx, yy, miss = header_to_latlon(header_l11, coord_sys)
            _write_level(fh, lons, lats, xx, yy, miss, ("_l11", " at level 11"))

    log.info("Wrote %s", out_path)
