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

from soilgrids_access import download_soilgrids
from regrid          import reproject_to_header
from unit_convert    import convert
from writers         import write_all_layers
from lut             import build_lut
from utils           import (
    setup_logging,
    load_header,
    build_soil_header_from_meteo,
    read_lcc_crs_from_latlon,
)

# ---- USER INPUTS ---------------------------------------------------
WATERSHED_PATH    = "/workspace/test_domain_3/input/domain/niver.geojson"
METEO_HEADER_PATH = "/workspace/test_domain_3/input/latlon/header.txt"
OUTPUT_DIR        = "/workspace/test_domain_3/input/morph"
BUFFER_KM         = 6       # buffer around watershed for WCS request
STATS             = "Q0.5"  # SoilGrids statistic: Q0.5 | Q0.05 | Q0.95 | mean
NODATA            = -9999
SOIL_CELL_SIZE_M  = 250     # native SoilGrids resolution; 3000 / 250 = 12

# Set to a CRS WKT string to skip the latlon.nc lookup (useful if meteo
# hasn't been run yet or for a different projection)
TARGET_CRS_WKT = None
# --------------------------------------------------------------------


def main() -> None:
    setup_logging()
    log = logging.getLogger("soils_to_mhm")

    out_root  = Path(OUTPUT_DIR)
    cache_dir = out_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Build soil grid header anchored to the meteo grid at 250 m.
    meteo_header = load_header(METEO_HEADER_PATH)
    soil_header  = build_soil_header_from_meteo(meteo_header, SOIL_CELL_SIZE_M)
    factor = int(meteo_header["cellsize"]) // SOIL_CELL_SIZE_M
    log.info(
        "Soil grid: %d x %d cells at %d m  "
        "(meteo %d x %d at %d m, ×%d refinement)",
        soil_header["ncols"], soil_header["nrows"], SOIL_CELL_SIZE_M,
        meteo_header["ncols"], meteo_header["nrows"],
        int(meteo_header["cellsize"]), factor,
    )

    # 2. Resolve LCC CRS (read from latlon.nc produced by precip_to_mhm).
    target_crs = TARGET_CRS_WKT or read_lcc_crs_from_latlon(
        Path(METEO_HEADER_PATH).parent.parent / "latlon" / "latlon.nc"
    )

    # 3. Download 18 SoilGrids GeoTIFFs (cached after first run).
    tif_paths = download_soilgrids(
        WATERSHED_PATH, cache_dir, buffer_km=BUFFER_KM, stat=STATS,
    )

    # 4. Reproject each GeoTIFF to the soil grid and apply unit conversion.
    layers: dict[str, dict[int, object]] = {"bd": {}, "cl": {}, "sn": {}}
    for prop, prop_paths in tif_paths.items():
        for layer_num, tif_path in sorted(prop_paths.items()):
            raw = reproject_to_header(tif_path, soil_header, target_crs, nodata_out=NODATA)
            layers[prop][layer_num] = convert(raw, prop, nodata=NODATA)
            log.info("Converted %s layer %02d", prop, layer_num)

    # 5. Write 18 intermediate ASCII grids.
    write_all_layers(out_root, soil_header, layers, nodata=NODATA)

    # 6. Generate soil_classdefinition.txt + soil_class.asc (Python LUT builder).
    build_lut(layers, soil_header, out_root, nodata=NODATA)

    log.info("Done. Outputs in %s", out_root)


if __name__ == "__main__":
    main()
