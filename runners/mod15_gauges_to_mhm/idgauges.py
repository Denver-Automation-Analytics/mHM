"""Build the L0 idgauges.asc raster and write the id_map.csv lookup."""

from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from pyproj import Transformer

log = logging.getLogger(__name__)


def build_idgauges_grid(
    survivors: List[Dict],
    l0_header: dict,
    target_crs_wkt: str,
    nodata: int,
) -> np.ndarray:
    """
    Return an int32 L0 grid with each surviving gauge's local id burned into
    the L0 cell that contains it. Collisions resolve to the gauge with the
    longer record (survivors already ordered by discovery, so we sort here).
    """
    ncols = l0_header["ncols"]
    nrows = l0_header["nrows"]
    cs = l0_header["cellsize"]
    xll = l0_header["xllcorner"]
    yll = l0_header["yllcorner"]

    grid = np.full((nrows, ncols), nodata, dtype=np.int32)
    transformer = Transformer.from_crs("EPSG:4326", target_crs_wkt, always_xy=True)

    # Sort by record length (longest first) so collisions favor the longer record.
    ordered = sorted(
        survivors,
        key=lambda g: (g["series"].index.max() - g["series"].index.min()).days,
        reverse=True,
    )

    used_cells: dict[tuple[int, int], int] = {}
    for g in ordered:
        x, y = transformer.transform(g["lon"], g["lat"])
        col = int((x - xll) // cs)
        row_from_bottom = int((y - yll) // cs)
        row = nrows - 1 - row_from_bottom  # ASCII rows are north-down

        if not (0 <= col < ncols and 0 <= row < nrows):
            log.warning(
                "Gauge %s (%s) projected outside L0 grid; skipping burn.",
                g["site_no"],
                g["name"],
            )
            continue

        if (row, col) in used_cells:
            keeper = used_cells[(row, col)]
            log.warning(
                "Cell (%d, %d) already holds gauge id %d; %s (%s) demoted "
                "(shorter record, no burn).",
                row,
                col,
                keeper,
                g["site_no"],
                g["name"],
            )
            continue

        grid[row, col] = g["local_id"]
        used_cells[(row, col)] = g["local_id"]

    burned = int((grid != nodata).sum())
    log.info("Burned %d gauge(s) into idgauges grid.", burned)
    return grid


def write_id_map(
    path: Path, survivors: List[Dict], start: str, end: str, cadence: str
) -> None:
    """Write id_map.csv linking local id -> USGS site number + metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "local_id",
                "site_no",
                "name",
                "lat",
                "lon",
                "cadence",
                "start",
                "end",
            ]
        )
        for g in sorted(survivors, key=lambda x: x["local_id"]):
            writer.writerow(
                [
                    g["local_id"],
                    g["site_no"],
                    g["name"],
                    f"{g['lat']:.6f}",
                    f"{g['lon']:.6f}",
                    cadence,
                    start,
                    end,
                ]
            )
    log.info("Wrote %s", path)
