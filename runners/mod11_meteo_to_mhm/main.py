"""
HRRR → mHM meteorology preparation.

Pulls NOAA HRRR 48-hour forecast from dynamical.org's Icechunk Zarr, clips to a
watershed on the native 3-km Lambert Conformal Conic grid, and writes mHM-ready
NetCDF files for precipitation, temperature, wind speed, relative humidity, and
shortwave/longwave radiation. Also writes a header.txt file for each variable and
a latlon.nc file for the clipped grid.
"""

from __future__ import annotations
import os
import sys
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

from affine import Affine
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_CRS, L2_CELL_SIZE_M
from latlon_grid import mhm_l2_from_l0

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

    # 2. Watershed → LCC → snap to native HRRR grid
    ws_lcc = load_and_prepare_watershed(WATERSHED_PATH, hrrr_crs)
    bbox_lcc = snap_bbox_to_grid(ws_lcc.total_bounds, L2_CELL_SIZE_M)
    log.info("Snapped LCC bbox (m): %s", bbox_lcc)

    # 3. Select time window
    if FORECAST:
        init_time = resolve_init_time(ds, INIT_TIME)
        log.info("Using init_time = %s UTC", init_time)
        ds_window = select_window(ds, init_time, forecast_hours=48)
    else:
        ds_window = ds.sel(time=slice(START_DATE, END_DATE))
        if ds_window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No timesteps found in [{START_DATE}, {END_DATE}]. "
                "Check START_DATE / END_DATE against the available analysis period."
            )
        t0 = str(ds_window["time"].values[0])
        t1 = str(ds_window["time"].values[-1])
        log.info("Analysis window: %s .. %s", t0, t1)
        # Not used by format_for_mhm when a time axis already exists.
        init_time = resolve_init_time(ds, "latest")

    # 4. Clip on native LCC grid
    ds_clip = ds_window.rio.clip_box(*bbox_lcc, crs=hrrr_crs).compute()
    log.info("Clipped grid shape (y, x): (%d, %d)",
             ds_clip.sizes["y"], ds_clip.sizes["x"])

    # 4b. Reproject to the exact L2 grid that mHM derives from L0 at runtime
    if ds_clip.rio.crs is None:
        ds_clip = ds_clip.rio.set_crs(hrrr_crs)
    l2 = mhm_l2_from_l0(L0_DEM, L2_CELL_SIZE_M)
    _cs = l2["cellsize"]
    _target_transform = Affine(_cs, 0.0, l2["xllcorner"],
                               0.0, -_cs, l2["yllcorner"] + l2["nrows"] * _cs)
    ds_reproj = ds_clip.rio.reproject(
        OUTPUT_CRS,
        shape=(l2["nrows"], l2["ncols"]),
        transform=_target_transform,
        resampling=Resampling.bilinear,
    )
    # precipitation uses sum resampling to preserve mass
    pre_sum = ds_clip[["precipitation_surface"]].rio.reproject_match(
        ds_reproj, resampling=Resampling.sum
    )
    ds_reproj["precipitation_surface"] = pre_sum["precipitation_surface"]
    log.info("Reprojected to %s: (y, x) = (%d, %d)",
             OUTPUT_CRS, ds_reproj.sizes["y"], ds_reproj.sizes["x"])

    # 5. Format to mHM contract (rename, units check, time-as-int, DOUBLE, fill)
    ds_mhm, ref_time = format_for_mhm(ds_reproj, init_time, NODATA)

    # 6. Header from mhm_l2_from_l0 — same grid mHM derives internally
    header = {**l2, "NODATA_value": int(NODATA)}
    assert_grid_consistency(ds_reproj, header, L2_CELL_SIZE_M)

    # 7. Write outputs
    write_meteo(ds_mhm[["pre"]],       meteo_out_root / "pre"       / "pre.nc",       ref_time, NODATA, crs=OUTPUT_CRS)
    write_meteo(ds_mhm[["tavg"]],      meteo_out_root / "tavg"      / "tavg.nc",      ref_time, NODATA, crs=OUTPUT_CRS)
    write_meteo(ds_mhm[["windspeed"]], meteo_out_root / "windspeed" / "windspeed.nc", ref_time, NODATA, crs=OUTPUT_CRS)
    write_meteo(ds_mhm[["rhavg"]],     meteo_out_root / "rhavg"     / "rhavg.nc",     ref_time, NODATA, crs=OUTPUT_CRS)
    write_meteo(ds_mhm[["ssrd"]],      meteo_out_root / "ssrd"      / "ssrd.nc",      ref_time, NODATA, crs=OUTPUT_CRS)
    write_meteo(ds_mhm[["strd"]],      meteo_out_root / "strd"      / "strd.nc",      ref_time, NODATA, crs=OUTPUT_CRS)
    write_header_txt(header, meteo_out_root / "pre"       / "header.txt")
    write_header_txt(header, meteo_out_root / "tavg"      / "header.txt")
    write_header_txt(header, meteo_out_root / "windspeed" / "header.txt")
    write_header_txt(header, meteo_out_root / "rhavg"     / "header.txt")
    write_header_txt(header, meteo_out_root / "ssrd"      / "header.txt")
    write_header_txt(header, meteo_out_root / "strd"      / "header.txt")

    write_header_txt(header, latlon_out_root / "header.txt")

    log.info("Done. Outputs in %s", meteo_out_root)




if __name__ == "__main__":
    # ---- USER INPUTS ---------------------------------------------------
    WATERSHED_PATH   = "/workspace/test_domain_3/input/domain/huc4_1211.geojson"     # .shp / .geojson / .gpkg
    L0_DEM           = "/workspace/test_domain_3/input/morph/dem.nc"
    INIT_TIME        = "latest"                      # "latest" or "YYYY-MM-DDTHH" (UTC)
    METEO_OUTPUT_DIR       = "/workspace/test_domain_3/input/meteo"  # output directory for mHM-ready files
    LATLON_OUTPUT_DIR      = "/workspace/test_domain_3/input/latlon" # output directory for latlon.nc
    FORECAST         = False                            # use forecast (True) or analysis (False) HRRR subscription
    START_DATE       = "2024-09-01"                     # analysis only; "YYYY-MM-DD" or None for full period
    END_DATE         = "2024-09-30"                     # analysis only; "YYYY-MM-DD" or None for full period

    # ---- FIXED -----------------------------------------------------------
    NODATA           = -9999.0

    # --- HRRR subscription -----------------------------------------------
    # Repo name is read from the HRRR_REPO env var.
    if FORECAST:
        HRRR_REPO             = os.environ["HRRR_FORECAST_REPO"]
        HRRR_BRANCH_OR_TAG    = os.environ.get("HRRR_FORECAST_BRANCH_OR_TAG", "main")
    else:
        HRRR_REPO             = os.environ["HRRR_ANALYSIS_REPO"]
        HRRR_BRANCH_OR_TAG    = os.environ.get("HRRR_ANALYSIS_BRANCH_OR_TAG", "main")
    main()