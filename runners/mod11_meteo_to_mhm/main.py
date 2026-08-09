"""
HRRR → mHM meteorology preparation.

Pulls NOAA HRRR 48-hour forecast from dynamical.org's Icechunk Zarr, clips to a
watershed on the native 3-km Lambert Conformal Conic grid, and writes mHM-ready
`pre.nc`, `tavg.nc`, `latlon.nc` files plus companion `header.txt` files.

mHM configuration expected:
    iFlag_cordinate_sys = 0   (projected LCC meters)
    variable names       : pre (mm), tavg (degC)
    NODATA_value         : -9999
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

try:
    import rioxarray  # noqa: F401  # registers xarray .rio accessor
except ImportError as exc:
    raise ImportError(
        "The precip_to_mhm workflow requires 'rioxarray'. "
        "Install it in your environment (e.g. `pip install rioxarray`)."
    ) from exc

from hrrr_access   import open_hrrr, resolve_init_time, select_window
from io_watershed  import load_and_prepare_watershed, snap_bbox_to_grid
from mhm_format    import format_for_mhm
from writers       import write_meteo, write_header_txt
from utils         import setup_logging, assert_grid_consistency

load_dotenv()
# Fail fast if the datalake key is not set; dataretrieval reads it implicitly.
if not os.environ.get("HRRR_FORECAST_REPO"):
    raise EnvironmentError(
        "HRRR_FORECAST_REPO env var is required. Register at Earthmover Marketplace"
    )

# ---- USER INPUTS ---------------------------------------------------
WATERSHED_PATH   = "/workspace/test_domain_3/input/domain/niver.geojson"     # .shp / .geojson / .gpkg
INIT_TIME        = "latest"                      # "latest" or "YYYY-MM-DDTHH" (UTC)
METEO_OUTPUT_DIR       = "/workspace/test_domain_3/input/meteo"  # output directory for mHM-ready files
LATLON_OUTPUT_DIR      = "/workspace/test_domain_3/input/latlon" # output directory for latlon.nc
BUFFER_KM        = 6                             # extra buffer around watershed
FORECAST         = False                            # use forecast (True) or analysis (False) HRRR subscription

# ---- FIXED (dictated by HRRR native grid & mHM contract) -----------
CELL_SIZE_M      = 3000
NODATA           = -9999.0

# --- HRRR subscription -----------------------------------------------
# Repo name is read from the HRRR_REPO env var so it isn't hard-coded in source.
if FORECAST:
    HRRR_REPO             = os.environ["HRRR_FORECAST_REPO"]
    HRRR_BRANCH_OR_TAG    = os.environ.get("HRRR_FORECAST_BRANCH_OR_TAG", "main")
else:
    HRRR_REPO             = os.environ["HRRR_ANALYSIS_REPO"]
    HRRR_BRANCH_OR_TAG    = os.environ.get("HRRR_ANALYSIS_BRANCH_OR_TAG", "main")

# --------------------------------------------------------------------


def main() -> None:
    setup_logging()
    log = logging.getLogger("hrrr_to_mhm")

    meteo_out_root = Path(METEO_OUTPUT_DIR)
    meteo_out_root.mkdir(parents=True, exist_ok=True)
    latlon_out_root = Path(LATLON_OUTPUT_DIR)
    latlon_out_root.mkdir(parents=True, exist_ok=True)

    # 1. Open source dataset (needed early to expose the HRRR LCC CRS)
    ds = open_hrrr(HRRR_REPO, HRRR_BRANCH_OR_TAG)
    hrrr_crs = ds.rio.crs
    log.info("HRRR CRS: %s", hrrr_crs)

    # 2. Watershed → LCC → buffer → snap to 3-km grid
    ws_lcc = load_and_prepare_watershed(WATERSHED_PATH, hrrr_crs, BUFFER_KM)
    bbox_lcc = snap_bbox_to_grid(ws_lcc.total_bounds, CELL_SIZE_M)
    log.info("Snapped LCC bbox (m): %s", bbox_lcc)

    # 3. Select time window
    if FORECAST:
        init_time = resolve_init_time(ds, INIT_TIME)
        log.info("Using init_time = %s UTC", init_time)
        ds_window = select_window(ds, init_time, forecast_hours=48)
    else:
        if "time" in ds.coords and ds["time"].size > 0:
            t0 = str(ds["time"].values[0])
            t1 = str(ds["time"].values[-1])
            log.info("Using full historical period: %s .. %s", t0, t1)
        ds_window = ds
        # Not used by format_for_mhm when a time axis already exists.
        init_time = resolve_init_time(ds, "latest")

    # 4. Clip on native LCC grid (no reprojection, no regridding)
    ds_clip = ds_window.rio.clip_box(*bbox_lcc, crs=hrrr_crs).compute()
    log.info("Clipped grid shape (y, x): (%d, %d)",
             ds_clip.sizes["y"], ds_clip.sizes["x"])

    # 5. Format to mHM contract (rename, units check, time-as-int, DOUBLE, fill)
    ds_mhm, ref_time = format_for_mhm(ds_clip, init_time, NODATA)

    # 6. Derive header dict from the clipped grid
    header = build_header(ds_clip, bbox_lcc, CELL_SIZE_M, NODATA)
    assert_grid_consistency(ds_clip, header, CELL_SIZE_M)

    # 7. Write outputs
    write_meteo(ds_mhm[["pre"]],       meteo_out_root / "pre"       / "pre.nc",       ref_time, NODATA)
    write_meteo(ds_mhm[["tavg"]],      meteo_out_root / "tavg"      / "tavg.nc",      ref_time, NODATA)
    write_meteo(ds_mhm[["windspeed"]], meteo_out_root / "windspeed" / "windspeed.nc", ref_time, NODATA)
    write_meteo(ds_mhm[["rhavg"]],     meteo_out_root / "rhavg"     / "rhavg.nc",     ref_time, NODATA)
    write_meteo(ds_mhm[["ssrd"]],      meteo_out_root / "ssrd"      / "ssrd.nc",      ref_time, NODATA)
    write_meteo(ds_mhm[["strd"]],      meteo_out_root / "strd"      / "strd.nc",      ref_time, NODATA)
    write_header_txt(header, meteo_out_root / "pre"       / "header.txt")
    write_header_txt(header, meteo_out_root / "tavg"      / "header.txt")
    write_header_txt(header, meteo_out_root / "windspeed" / "header.txt")
    write_header_txt(header, meteo_out_root / "rhavg"     / "header.txt")
    write_header_txt(header, meteo_out_root / "ssrd"      / "header.txt")
    write_header_txt(header, meteo_out_root / "strd"      / "header.txt")

    write_header_txt(header, latlon_out_root / "header.txt")

    log.info("Done. Outputs in %s", meteo_out_root)


def build_header(ds_clip, bbox_lcc, cell_size, nodata):
    xll, yll, _xur, _yur = bbox_lcc
    return {
        "ncols":        int(ds_clip.sizes["x"]),
        "nrows":        int(ds_clip.sizes["y"]),
        "xllcorner":    float(xll),
        "yllcorner":    float(yll),
        "cellsize":     int(cell_size),
        "NODATA_value": int(nodata),
    }


if __name__ == "__main__":
    main()