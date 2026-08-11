"""
SoilGrids -> mHM soil data preparation.

Fetches bulk density (bdod), clay, and sand for six GlobalSoilMap depth
intervals from the ISRIC SoilGrids WCS, clips to the watershed on the
native Homolosine CRS, reprojects to the mHM Lambert Conformal Conic grid
at 250 m resolution (12× refinement of the 3 km meteo grid), and writes:

  18 ArcGIS ASCII grids   bd01-06.txt  cl01-06.txt  sn01-06.txt
  soil_classdefinition.txt             (mHM soil LUT, iFlag_soilDB = 0)
  soil_class.asc                       (gridded soil-type index)

Copy soil_classdefinition.txt to the domain's dirCommonFiles directory and
soil_class.asc to dir_Morpho before running mHM.

mHM configuration expected:
    iFlag_soilDB        = 0    (classical mHM format)
    iFlag_cordinate_sys = 0    (projected LCC metres)
"""

from __future__ import annotations
import logging
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import L0_CELL_SIZE_M, OUTPUT_CRS, DOMAIN_FILE, WORKING_DIR, NODATA

from soilgrids_access import download_soilgrids
from regrid          import reproject_to_header
from unit_convert    import convert
from writers         import write_all_layers, write_layers_nc
from lut             import build_lut
from utils           import (
    setup_logging,
    load_header,
)

# ---- USER INPUTS ---------------------------------------------------
L0_HEADER_PATH    = os.path.join(WORKING_DIR, "input", "gauge", "header.txt")
METEO_HEADER_PATH = os.path.join(WORKING_DIR, "input", "meteo", "pre", "header.txt")
OUTPUT_DIR        = os.path.join(WORKING_DIR, "input", "morph")
STATS             = "Q0.5"  # SoilGrids statistic: Q0.5 | Q0.05 | Q0.95 | mean
SOIL_CELL_SIZE_M  = L0_CELL_SIZE_M     # native SoilGrids resolution; 3000 / 250 = 12
TARGET_CRS_WKT    = OUTPUT_CRS
OUTPUT_FORMAT     = "asc"   # "asc" (18 .txt files) or "nc" (single soil_layers.nc)

# --------------------------------------------------------------------


def main() -> None:
    setup_logging()
    log = logging.getLogger("soils_to_mhm")

    out_root  = Path(OUTPUT_DIR)
    cache_dir = out_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Read the canonical L0 header (gauge/luse domain) as the soil grid definition.
    soil_header  = load_header(L0_HEADER_PATH)
    meteo_header = load_header(METEO_HEADER_PATH)
    factor = int(meteo_header["cellsize"]) // SOIL_CELL_SIZE_M
    log.info(
        "Soil grid: %d x %d cells at %d m  "
        "(meteo %d x %d at %d m, ×%d refinement)",
        soil_header["ncols"], soil_header["nrows"], SOIL_CELL_SIZE_M,
        meteo_header["ncols"], meteo_header["nrows"],
        int(meteo_header["cellsize"]), factor,
    )

    # 2. Resolve LCC CRS (read from latlon.nc produced by precip_to_mhm).
    target_crs = TARGET_CRS_WKT

    # 3. Download 18 SoilGrids GeoTIFFs (cached after first run).
    tif_paths = download_soilgrids(
        DOMAIN_FILE, cache_dir, stat=STATS,
    )

    # 4. Reproject each GeoTIFF to the soil grid and apply unit conversion.
    layers: dict[str, dict[int, object]] = {"bd": {}, "cl": {}, "sn": {}}
    for prop, prop_paths in tif_paths.items():
        for layer_num, tif_path in sorted(prop_paths.items()):
            raw = reproject_to_header(tif_path, soil_header, target_crs, nodata_out=NODATA)
            layers[prop][layer_num] = convert(raw, prop, nodata=NODATA)
            log.info("Converted %s layer %02d", prop, layer_num)

    # 5. Write 18 intermediate soil-layer grids.
    if OUTPUT_FORMAT == "nc":
        write_layers_nc(out_root, soil_header, layers, nodata=NODATA)
    else:
        write_all_layers(out_root, soil_header, layers, nodata=NODATA)

    # 6. Generate soil_classdefinition.txt + soil_class.asc (Python LUT builder).
    build_lut(layers, soil_header, out_root, nodata=NODATA)

    log.info("Done. Outputs in %s", out_root)


if __name__ == "__main__":
    main()
