"""
latlon.nc assembly for mHM.

Combines headers from all grid-defining modules into the single latlon.nc
file that mHM requires.  Run this after mod10, mod11, and mod14.

Grid levels:
    L0   = 10 m    morph/dem.nc          (mod10 DEM)
    L1   = 250 m   morph/soil_class.asc  (mod14 soils)
    L11  = 250 m   same as L1            (routing == hydrological)

L2 (3 km meteo) is NOT written to latlon.nc; mHM reads meteo files directly.
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "mod11_meteo_to_mhm"))
from latlon_grid import create_latlon, derive_l_header  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import L1_CELL_SIZE_M, WORKING_DIR, OUTPUT_CRS  # noqa: E402

# ---- USER INPUTS --------------------------------------------------
L0_HEADER   = os.path.join(WORKING_DIR, "input", "morph", "dem.nc")
OUTPUT_DIR  = os.path.join(WORKING_DIR, "input", "latlon")
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("latlon_to_mhm")


def _require(path: str, module: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found — run {module} first."
        )
    return p


if __name__ == "__main__":
    l0  = _require(L0_HEADER,  "mod10_dem_to_mhm")
    l1  = derive_l_header(l0, L1_CELL_SIZE_M)
    l11 = l1

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / "latlon.nc"

    create_latlon(
        out_file   = out_file,
        coord_sys  = OUTPUT_CRS,
        header_l0  = l0,
        header_l1  = l1,
        header_l11 = l11,
    )
    log.info("Written: %s", out_file)
