"""
GHL Land Cover -> mHM preparation.

Fetches the Space Intelligence Global Harmonised Layers (GHL) 30 m land cover
scene from Earthmover (Arraylake), clips to a watershed, reprojects to the mHM
Lambert Conformal Conic L0 grid using majority resampling, reclassifies from
GHL's 10 classes to mHM's 3 classes (Forest / Impervious / Pervious), and
writes ArcGIS ASCII grids that mHM reads directly.

mHM configuration expected:
    iFlag_cordinate_sys = 0                (projected LCC meters)
    L0 cellsize        = L0_CELL_SIZE_M    (runners/config.py)
    Land cover classes : 1=Forest, 2=Impervious, 3=Pervious
"""

from __future__ import annotations
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import L0_CELL_SIZE_M, OUTPUT_CRS, WORKING_DIR, NODATA
from pathlib import Path
from dotenv import load_dotenv
from pyproj import CRS as ProjCRS

from ghl_access   import open_ghl, select_scene, detect_class_var
from reclassify   import to_mhm_classes, log_class_stats
from regrid       import clip_and_reproject_to_grid
from writers      import write_asc, write_nc_copy
from utils        import (
    load_header_from_nc,
    write_header_txt,
)

load_dotenv()

# Fail fast if the datalake key is not set; dataretrieval reads it implicitly.
if not os.environ.get("GHL_REPO"):
    raise EnvironmentError(
        "GHL_REPO env var is required. Register at Earthmover Marketplace for "
        "an API key and add it to your .env file."
    )

# ---- OUTPUTS ---------------------------------------------------
OUTPUT_DIR = os.path.join(WORKING_DIR, "input", "luse")

# --- GHL subscription -------------------------------------------
# Repo name is read from the GHL_REPO env var so it isn't hard-coded in source.
GHL_REPO              = os.environ["GHL_REPO"]
GHL_BRANCH_OR_TAG     = os.environ.get("GHL_BRANCH_OR_TAG", "main")
GHL_VARIABLE          = os.environ.get("GHL_VARIABLE")   # None -> auto-detect
SCENE_YEARS           = [2024]                            # one .asc per year

# --- Grid targeting --------------------------------------------------
# L0_CELL_SIZE_M and OUTPUT_CRS are read from runners/config.py at runtime.

# --- Reclassification knobs ----------------------------------------
PLANTATION_MHM_CLASS  = 1             # 1=Forest (default) or 3=Pervious

# GHL -> mHM class map. Water/snow/ice -> Pervious per mHM devs' guidance.
CLASS_MAP_GHL = {
    1:  1,   # Tree cover                    -> Forest
    2:  1,   # Mangrove                      -> Forest
    3:  PLANTATION_MHM_CLASS,  # Plantation
    4:  3,   # Grassland / shrubland         -> Pervious
    5:  3,   # Cropland                      -> Pervious
    6:  2,   # Built-up                      -> Impervious
    7:  3,   # Bare / sparse vegetation      -> Pervious
    8:  3,   # Snow / ice                    -> Pervious
    9:  3,   # Water                         -> Pervious
    10: 3,   # Flooded non-forest vegetation -> Pervious
}
NODATA_SRC            = 255           # GHL native NoData
NODATA_MHM            = NODATA         # mHM convention

# --- QA -------------------------------------------------------------
WRITE_NETCDF_COPY     = True
LOG_LEVEL             = logging.INFO


# --------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("land_cover_to_mhm")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Load the canonical L0 grid produced by the DEM runner (mod10).
    #    All L0 inputs must share this exact grid; deriving it independently
    #    from the watershed bbox produces a misaligned (inset) grid.
    WATERSHED_FILE = os.path.join(WORKING_DIR, "input", "domain", "watershed.geojson")
    if not os.path.exists(WATERSHED_FILE):
        raise FileNotFoundError(
            f"Watershed file not found: {WATERSHED_FILE}. "
            "Run runners/mod10_dem_to_mhm first."
        )
    L0_MORPH_NC_PATH = os.path.join(WORKING_DIR, "input", "morph", "dem.nc")
    if not os.path.exists(L0_MORPH_NC_PATH):
        raise FileNotFoundError(
            f"Canonical L0 morph grid not found: {L0_MORPH_NC_PATH}. "
            "Run runners/mod10_dem_to_mhm first."
        )
    l0_header = load_header_from_nc(L0_MORPH_NC_PATH)
    log.info("L0 cellsize=%s m, ncols=%d, nrows=%d",
             l0_header["cellsize"], l0_header["ncols"], l0_header["nrows"])

    # Persist the L0 header so downstream modules can reuse it.
    write_header_txt(l0_header, out_root / "header.txt")

    # 2. Resolve target CRS (WKT) from shared config.
    target_crs = ProjCRS.from_user_input(OUTPUT_CRS).to_wkt()

    # 3. Open GHL Zarr store via Arraylake.
    ds = open_ghl(GHL_REPO, GHL_BRANCH_OR_TAG)
    var_name = GHL_VARIABLE or detect_class_var(ds)
    log.info("Using GHL variable: %s", var_name)

    # 4. Process one .asc per requested scene year.
    for year in SCENE_YEARS:
        log.info("---- Processing GHL year %s ----", year)
        scene = select_scene(ds, var_name, year)

        # Clip + reproject to L0 with majority (mode) resampling.
        lc_l0 = clip_and_reproject_to_grid(
            scene,
            watershed_path=WATERSHED_FILE,
            target_crs_wkt=target_crs,
            header=l0_header,
            src_nodata=NODATA_SRC,
            l0_cellsize_m=L0_CELL_SIZE_M,
        )

        # Reclassify GHL -> mHM 3-class scheme.
        grid = to_mhm_classes(
            lc_l0.values,
            class_map=CLASS_MAP_GHL,
            nodata_src=NODATA_SRC,
            nodata_mhm=NODATA_MHM,
        )
        log_class_stats(grid, nodata=NODATA_MHM, year=year)

        # Write outputs.
        asc_path = out_root / f"lc_{year}.asc"
        write_asc(asc_path, l0_header, grid, nodata=NODATA_MHM)

        if WRITE_NETCDF_COPY:
            write_nc_copy(
                out_root / f"lc_{year}.nc",
                grid, l0_header, target_crs, nodata=NODATA_MHM, year=year,
            )

    log.info("Done. Outputs in %s", out_root)



if __name__ == "__main__":
    main()