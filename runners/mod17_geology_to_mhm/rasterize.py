"""Reproject karst vector layers and burn them onto the L0 grid."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin


def _grid_transform(grid_def: dict):
    return from_origin(
        grid_def["xtop"],
        grid_def["ytop"],
        grid_def["cellsize"],
        grid_def["cellsize"],
    )


def rasterize_mask(
    gdf: gpd.GeoDataFrame, grid_def: dict, target_crs: str, all_touched: bool = True
) -> np.ndarray:
    """Reproject a class layer to the grid CRS and return its L0 coverage mask."""
    shape = (grid_def["nrows"], grid_def["ncols"])
    if gdf is None or gdf.empty:
        return np.zeros(shape, dtype=bool)

    projected = gdf.to_crs(target_crs)
    geoms = [g for g in projected.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool)

    burned = rasterize(
        ((g, 1) for g in geoms),
        out_shape=shape,
        transform=_grid_transform(grid_def),
        fill=0,
        default_value=1,
        all_touched=all_touched,
        dtype="uint8",
    )
    return burned.astype(bool)
