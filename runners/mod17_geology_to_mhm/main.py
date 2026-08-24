"""
USGS karst geology -> mHM geology inputs.

Acquires USGS "Karst in the United States" vector classes (karst_access.py),
rasterises them onto the L0 grid defined by mod10's dem.nc, and writes the two
files mHM reads from mhm_input/morph/:

  geology_class.asc            gridded integer ClassUnit ids (L0, EPSG:5070)
  geology_classdefinition.txt  LUT with a Karstic flag per unit

Unit 1 is the non-karst background over the DEM-valid domain; each karst class
present becomes its own karstic unit (first-wins on overlap, in KARST_SERVICES
order). mod19's _bootstrap_geology() skips when both files exist, so no
namelist changes are required.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import OUTPUT_CRS, WORKING_DIR, NODATA  # noqa: E402

from karst_access import acquire_karst, KARST_SERVICES  # noqa: E402
from lut import build_geology  # noqa: E402
from rasterize import rasterize_mask  # noqa: E402
from utils import load_dem_grid, setup_logging  # noqa: E402


def _read_gpkg_layers(path: Path) -> dict[str, gpd.GeoDataFrame]:
    import fiona  # noqa: PLC0415

    return {lyr: gpd.read_file(path, layer=lyr) for lyr in fiona.listlayers(str(path))}


def _acquire_or_load(boundary: gpd.GeoDataFrame, cache_path: Path,
                     log: logging.Logger) -> dict[str, gpd.GeoDataFrame]:
    if cache_path.exists():
        log.info("Using cached karst vectors: %s", cache_path)
        return _read_gpkg_layers(cache_path)
    log.info("Acquiring karst vectors -> %s", cache_path)
    return acquire_karst(boundary_gdf=boundary, output_path=str(cache_path), clip=True)


def _group_by_class(frames: dict[str, gpd.GeoDataFrame]) -> dict[str, list[gpd.GeoDataFrame]]:
    by_class: dict[str, list[gpd.GeoDataFrame]] = {}
    for key, gdf in frames.items():
        if gdf is None or gdf.empty:
            continue
        if "karst_class" in gdf.columns:
            for label, sub in gdf.groupby("karst_class"):
                by_class.setdefault(str(label), []).append(sub)
        else:
            by_class.setdefault(key.split("__", 1)[0], []).append(gdf)
    return by_class


def main() -> None:
    setup_logging()
    log = logging.getLogger("geology_to_mhm")

    morph_dir = Path(WORKING_DIR) / "mhm_input" / "morph"
    grid_def = load_dem_grid(morph_dir / "dem.nc", nodata=NODATA)
    log.info("L0 grid: %d x %d at %d m", grid_def["ncols"], grid_def["nrows"],
             int(grid_def["cellsize"]))

    cache_path = morph_dir / "raw" / "karst.gpkg"
    WATERSHED_FILE = os.path.join(WORKING_DIR, "mhm_input/domain/watershed.geojson")
    if not os.path.exists(WATERSHED_FILE):
        raise FileNotFoundError(
            f"Watershed file {WATERSHED_FILE} not found. Run mod10_dem_to_mhm first."
        )
    boundary = gpd.read_file(WATERSHED_FILE)
    frames = _acquire_or_load(boundary, cache_path, log)
    by_class = _group_by_class(frames)

    shape = (grid_def["nrows"], grid_def["ncols"])
    class_masks: list[tuple[str, np.ndarray]] = []
    for label, _ in KARST_SERVICES:  # KARST_SERVICES order = overlap priority
        gdfs = by_class.get(label)
        if not gdfs:
            continue
        mask = np.zeros(shape, dtype=bool)
        for gdf in gdfs:
            mask |= rasterize_mask(gdf, grid_def, OUTPUT_CRS, all_touched=True)
        if not mask.any():
            continue
        log.info("  %-22s %d L0 cell(s)", label, int(mask.sum()))
        class_masks.append((label, mask))

    id_of = build_geology(class_masks, grid_def, morph_dir, nodata=NODATA)
    if not id_of:
        log.info("No karst intersects the domain; wrote non-karst-only geology.")


if __name__ == "__main__":
    main()
