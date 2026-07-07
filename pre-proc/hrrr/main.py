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
import logging
from pathlib import Path

from hrrr_access   import open_hrrr, resolve_init_time, select_window
from io_watershed  import load_and_prepare_watershed, snap_bbox_to_grid
from mhm_format    import format_for_mhm
from writers       import write_meteo, write_header_txt
from latlon_grid   import create_latlon
from utils         import setup_logging, assert_grid_consistency

# ---- USER INPUTS ---------------------------------------------------
WATERSHED_PATH   = r"path/to/watershed.shp"     # .shp / .geojson / .gpkg
FORECAST_LENGTH  = 48                            # hours, 1–48
INIT_TIME        = "latest"                      # "latest" or "YYYY-MM-DDTHH" (UTC)
OUTPUT_DIR       = r"path/to/mhm_meteo"
BUFFER_KM        = 6                             # extra buffer around watershed

# ---- FIXED (dictated by HRRR native grid & mHM contract) -----------
CELL_SIZE_M      = 3000
NODATA           = -9999.0

# --------------------------------------------------------------------


def main() -> None:
    setup_logging()
    log = logging.getLogger("hrrr_to_mhm")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Open source dataset (needed early to expose the HRRR LCC CRS)
    ds = open_hrrr()
    hrrr_crs = ds.rio.crs
    log.info("HRRR CRS: %s", hrrr_crs)

    # 2. Watershed → LCC → buffer → snap to 3-km grid
    ws_lcc = load_and_prepare_watershed(WATERSHED_PATH, hrrr_crs, BUFFER_KM)
    bbox_lcc = snap_bbox_to_grid(ws_lcc.total_bounds, CELL_SIZE_M)
    log.info("Snapped LCC bbox (m): %s", bbox_lcc)

    # 3. Resolve init_time and slice the forecast window
    init_time = resolve_init_time(ds, INIT_TIME)
    log.info("Using init_time = %s UTC", init_time)
    ds_window = select_window(ds, init_time, FORECAST_LENGTH)

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
    write_meteo(ds_mhm[["pre"]],  out_root / "pre"  / "pre.nc",  ref_time, NODATA)
    write_meteo(ds_mhm[["tavg"]], out_root / "tavg" / "tavg.nc", ref_time, NODATA)
    write_header_txt(header, out_root / "pre"  / "header.txt")
    write_header_txt(header, out_root / "tavg" / "header.txt")

    # latlon.nc via migrated mHM create_latlon logic.
    # L0 will be supplied when the DEM/morphology module is implemented.
    create_latlon(
        out_file   = out_root / "latlon" / "latlon.nc",
        coord_sys  = hrrr_crs.to_wkt(),   # HRRR native LCC
        header_l1  = header,               # 3 km meteo == hydrology grid
        header_l11 = header,               # 3 km routing grid (same for now)
        header_l0  = None,                 # deferred
    )
    write_header_txt(header, out_root / "latlon" / "header.txt")

    log.info("Done. Outputs in %s", out_root)


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