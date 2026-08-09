"""ArcGIS ASCII grid writer for idgauges.asc."""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


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