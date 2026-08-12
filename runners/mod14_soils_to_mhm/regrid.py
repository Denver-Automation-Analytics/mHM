"""Reproject a SoilGrids GeoTIFF from Homolosine to the target LCC soil grid."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

log = logging.getLogger(__name__)

# SoilGrids INT16 nodata sentinel (used when metadata nodata is absent)
_SG_NODATA = -32768

# SoilGrids native CRS; the WCS-written tifs carry no PROJ-parseable CRS, so
# assign this when src.crs is None to keep the tif geotransform georeferenced.
_IGH_PROJ4 = "+proj=igh +datum=WGS84 +no_defs"


def reproject_to_header(
    src_path: Path,
    header: dict,
    target_crs: str,
    nodata_out: int = -9999,
) -> np.ndarray:
    """
    Reproject a single SoilGrids GeoTIFF onto the grid described by `header`.

    Uses nearest-neighbour resampling to preserve raw INT16 integer values.
    Returns an (nrows, ncols) int32 array; nodata pixels are nodata_out.

    Resampling is done manually via pyproj because GDAL's warp cannot invert
    the interrupted Goode homolosine (+proj=igh) source projection.
    """
    ncols, nrows = header["ncols"], header["nrows"]
    cs = header["cellsize"]
    xll, yll = header["xllcorner"], header["yllcorner"]
    top = yll + nrows * cs

    # Destination cell-centre coordinates in the target CRS.
    xs = xll + (np.arange(ncols) + 0.5) * cs
    ys = top - (np.arange(nrows) + 0.5) * cs
    dst_x, dst_y = np.meshgrid(xs, ys)

    dst = np.full((nrows, ncols), nodata_out, dtype=np.int32)

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else _SG_NODATA
        src_crs = src.crs.to_wkt() if src.crs is not None else _IGH_PROJ4
        arr = src.read(1)
        inv = ~src.transform  # world -> pixel

        tf = Transformer.from_crs(target_crs, src_crs, always_xy=True)
        sx, sy = tf.transform(dst_x.ravel(), dst_y.ravel())
        sx = sx.reshape(nrows, ncols)
        sy = sy.reshape(nrows, ncols)

        col = inv.a * sx + inv.b * sy + inv.c
        row = inv.d * sx + inv.e * sy + inv.f
        ci = np.floor(col).astype(np.int64)
        ri = np.floor(row).astype(np.int64)

        h, w = arr.shape
        ok = (
            (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
            & np.isfinite(sx) & np.isfinite(sy)
        )
        vals = arr[ri[ok], ci[ok]]
        good = vals != src_nodata
        rr, cc = np.nonzero(ok)
        dst[rr[good], cc[good]] = vals[good].astype(np.int32)

    valid = int(np.sum(dst != nodata_out))
    log.debug("%s -> %d valid pixels / %d total", src_path.name, valid, dst.size)
    return dst
