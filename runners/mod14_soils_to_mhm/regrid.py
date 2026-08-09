"""Reproject a SoilGrids GeoTIFF from Homolosine to the target LCC soil grid."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject as _rio_reproject

log = logging.getLogger(__name__)

# SoilGrids INT16 nodata sentinel (used when metadata nodata is absent)
_SG_NODATA = -32768


def _affine_from_header(header: dict) -> Affine:
    """Build a north-up affine transform from an mHM-style header dict."""
    cs = header["cellsize"]
    return Affine(
        cs, 0.0, header["xllcorner"],
        0.0, -cs, header["yllcorner"] + header["nrows"] * cs,
    )


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
    """
    dst_transform = _affine_from_header(header)
    dst_crs = CRS.from_user_input(target_crs)
    dst_shape = (header["nrows"], header["ncols"])
    dst = np.full(dst_shape, nodata_out, dtype=np.int32)

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else _SG_NODATA
        _rio_reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_crs=src.crs,
            src_transform=src.transform,
            src_nodata=src_nodata,
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            resampling=Resampling.nearest,
            dst_nodata=nodata_out,
        )

    valid = np.sum(dst != nodata_out)
    log.debug("%s -> %d valid pixels / %d total", src_path.name, valid, dst.size)
    return dst
