"""Watershed reading, reprojection, buffering, and 3-km grid snapping."""

from __future__ import annotations
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np

log = logging.getLogger(__name__)


def load_and_prepare_watershed(path: str, target_crs) -> gpd.GeoDataFrame:
    """Read the boundary, reproject to HRRR LCC, and buffer."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Watershed file {path} has no CRS defined.")
    gdf = gdf.to_crs(target_crs)
    return gdf


def snap_bbox_to_grid(bounds, cell_size_m: int):
    """
    Snap a (minx, miny, maxx, maxy) bbox outward to align with a cell_size_m grid
    anchored at the origin (0, 0) of the projected CRS. This guarantees
    xllcorner/yllcorner are integer multiples of cell_size_m so that all mHM
    grids (meteo, latlon, and later morphology) share consistent extents.
    """
    minx, miny, maxx, maxy = bounds
    xll = math.floor(minx / cell_size_m) * cell_size_m
    yll = math.floor(miny / cell_size_m) * cell_size_m
    xur = math.ceil(maxx / cell_size_m)  * cell_size_m
    yur = math.ceil(maxy / cell_size_m)  * cell_size_m
    return (xll, yll, xur, yur)