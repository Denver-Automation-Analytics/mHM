"""
HRRR → mHM meteorology preparation.

Pulls NOAA HRRR 48-hour forecast from dynamical.org's Icechunk Zarr, clips to a
watershed on the native 3-km Lambert Conformal Conic grid, and writes mHM-ready
NetCDF files for precipitation, temperature, wind speed, relative humidity, and
shortwave/longwave radiation. Also writes a header.txt file for each variable and
a latlon.nc file for the clipped grid.

Output cadence follows config.TIMESTEP: "hourly" writes the native HRRR steps,
"daily" aggregates each calendar day (fluxes summed, states averaged).
"""

from __future__ import annotations
import os
import sys
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

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
from config import OUTPUT_CRS, L2_CELL_SIZE_M, START_DATE, END_DATE, DOMAIN_FILE, TIMESTEP, WORKING_DIR, NODATA
from latlon_grid import mhm_l2_from_l0

from hrrr_access   import open_hrrr, resolve_init_time, select_window
from io_watershed  import load_and_prepare_watershed, snap_bbox_to_grid
from mhm_format    import format_for_mhm, aggregate_to_daily
from writers       import write_meteo, write_header_txt, finalize_variable
from utils         import setup_logging, assert_grid_consistency

load_dotenv()
# Fail fast if the datalake key is not set; dataretrieval reads it implicitly.
if not os.environ.get("HRRR_FORECAST_REPO"):
    raise EnvironmentError(
        "HRRR_FORECAST_REPO env var is required. Register at Earthmover Marketplace"
    )

# mHM variable name → (output subdir, output filename)
OUT_VARS = {
    "pre":       ("pre",       "pre.nc"),
    "tavg":      ("tavg",      "tavg.nc"),
    "windspeed": ("windspeed", "windspeed.nc"),
    "rhavg":     ("rhavg",     "rhavg.nc"),
    "ssrd":      ("ssrd",      "ssrd.nc"),
    "strd":      ("strd",      "strd.nc"),
}

# --------------------------------------------------------------------


def _iter_monthly_windows(ds, start_date: str, end_date: str):
    """Yield (batch_id, ds_slice) for each calendar month in [start, end]."""
    win_start = pd.Timestamp(start_date)
    win_end   = pd.Timestamp(end_date)

    month_starts = pd.date_range(win_start, win_end, freq="MS")
    if len(month_starts) == 0 or month_starts[0] > win_start:
        month_starts = month_starts.insert(0, win_start)

    for i, s in enumerate(month_starts):
        if i == len(month_starts) - 1:
            e = win_end
        else:
            e = month_starts[i + 1] - pd.Timedelta(seconds=1)
        ds_slice = ds.sel(time=slice(s, e))
        if ds_slice.sizes.get("time", 0) == 0:
            continue
        yield i, ds_slice


def process_window(ds_window, batch_id, *, hrrr_crs, bbox_lcc, l2,
                   target_transform, header, init_time, nodata,
                   ref_time, temp_dirs, temp_files, log):
    """Clip → reproject → format one time window, writing per-variable temp files.

    Returns the effective reference time (shared across batches for consistent
    time encoding).
    """
    # Clip on native LCC grid; compute materializes only this batch.
    ds_clip = ds_window.rio.clip_box(*bbox_lcc, crs=hrrr_crs).compute()
    if ds_clip.rio.crs is None:
        ds_clip = ds_clip.rio.set_crs(hrrr_crs)

    # Reproject to the exact L2 grid mHM derives from L0 at runtime.
    ds_reproj = ds_clip.rio.reproject(
        OUTPUT_CRS,
        shape=(l2["nrows"], l2["ncols"]),
        transform=target_transform,
        resampling=Resampling.bilinear,
    )
    # precipitation uses sum resampling to preserve mass
    pre_sum = ds_clip[["precipitation_surface"]].rio.reproject_match(
        ds_reproj, resampling=Resampling.sum
    )
    ds_reproj["precipitation_surface"] = pre_sum["precipitation_surface"]

    if batch_id == 0:
        assert_grid_consistency(ds_reproj, header, L2_CELL_SIZE_M)

    ds_mhm, eff_ref = format_for_mhm(ds_reproj, init_time, nodata, ref_time=ref_time)
    if TIMESTEP == "daily":
        ds_mhm = aggregate_to_daily(ds_mhm, nodata)

    for var, (sub, _fname) in OUT_VARS.items():
        tmp = temp_dirs[var] / f"{var}_{batch_id:04d}.nc"
        write_meteo(ds_mhm[[var]], tmp, eff_ref, nodata, crs=OUTPUT_CRS)
        temp_files[var].append(tmp)

    log.info("Batch %04d: %d %s timesteps → temp files",
             batch_id, ds_mhm.sizes["time"], TIMESTEP)
    return eff_ref


def main() -> None:
    setup_logging()
    log = logging.getLogger("hrrr_to_mhm")

    if TIMESTEP not in ("hourly", "daily"):
        raise ValueError(f"Unexpected TIMESTEP {TIMESTEP!r}. Must be 'hourly' or 'daily'.")
    log.info("Meteo output cadence: %s", TIMESTEP)

    meteo_out_root = Path(METEO_OUTPUT_DIR)
    meteo_out_root.mkdir(parents=True, exist_ok=True)
    latlon_out_root = Path(LATLON_OUTPUT_DIR)
    latlon_out_root.mkdir(parents=True, exist_ok=True)

    # 1. Open source dataset (needed early to expose the HRRR LCC CRS)
    ds = open_hrrr(HRRR_REPO, HRRR_BRANCH_OR_TAG)
    hrrr_crs = ds.rio.crs
    log.info("HRRR CRS: %s", hrrr_crs)

    # 2. Watershed → LCC → snap to native HRRR grid
    ws_lcc = load_and_prepare_watershed(DOMAIN_FILE, hrrr_crs)
    bbox_lcc = snap_bbox_to_grid(ws_lcc.total_bounds, L2_CELL_SIZE_M)
    log.info("Snapped LCC bbox (m): %s", bbox_lcc)

    # 3. Static target grid mHM derives from L0 — computed once, shared by all batches.
    l2 = mhm_l2_from_l0(L0_DEM, L2_CELL_SIZE_M)
    _cs = l2["cellsize"]
    target_transform = Affine(_cs, 0.0, l2["xllcorner"],
                              0.0, -_cs, l2["yllcorner"] + l2["nrows"] * _cs)
    header = {**l2, "NODATA_value": int(NODATA)}

    # 4. Build the batch list (monthly for analysis; single window for forecast).
    if FORECAST:
        init_time = resolve_init_time(ds, INIT_TIME)
        log.info("Using init_time = %s UTC", init_time)
        windows = [(0, select_window(ds, init_time, forecast_hours=48))]
    else:
        ds_full = ds.sel(time=slice(START_DATE, END_DATE))
        if ds_full.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No timesteps found in [{START_DATE}, {END_DATE}]. "
                "Check START_DATE / END_DATE against the available analysis period."
            )
        log.info("Analysis window: %s .. %s",
                 str(ds_full["time"].values[0]), str(ds_full["time"].values[-1]))
        # Not used by format_for_mhm when a time axis already exists.
        init_time = resolve_init_time(ds, "latest")
        windows = _iter_monthly_windows(ds_full, START_DATE, END_DATE)

    # 5. Prepare (and clear stale) temp directories per variable.
    temp_dirs = {}
    for var, (sub, _fname) in OUT_VARS.items():
        tdir = meteo_out_root / sub / "_tmp"
        if tdir.exists():
            shutil.rmtree(tdir)
        tdir.mkdir(parents=True, exist_ok=True)
        temp_dirs[var] = tdir
    temp_files = {var: [] for var in OUT_VARS}

    # 6. Stream each batch through clip → reproject → format → temp write.
    ref_time = None
    for batch_id, ds_window in windows:
        ref_time = process_window(
            ds_window, batch_id,
            hrrr_crs=hrrr_crs, bbox_lcc=bbox_lcc, l2=l2,
            target_transform=target_transform, header=header,
            init_time=init_time, nodata=NODATA, ref_time=ref_time,
            temp_dirs=temp_dirs, temp_files=temp_files, log=log,
        )

    if ref_time is None:
        raise RuntimeError("No batches were processed — nothing to write.")

    # 7. Concatenate temp files per variable into the final meteo NetCDF, then
    #    remove the temp directories.
    try:
        for var, (sub, fname) in OUT_VARS.items():
            finalize_variable(
                temp_files[var], meteo_out_root / sub / fname,
                ref_time, NODATA, crs=OUTPUT_CRS,
            )
    finally:
        for tdir in temp_dirs.values():
            shutil.rmtree(tdir, ignore_errors=True)

    # 8. Write headers (per variable + latlon).
    for var, (sub, _fname) in OUT_VARS.items():
        write_header_txt(header, meteo_out_root / sub / "header.txt")
    write_header_txt(header, latlon_out_root / "header.txt")

    log.info("Done. Outputs in %s", meteo_out_root)


if __name__ == "__main__":
    # ---- USER INPUTS ---------------------------------------------------
    L0_DEM           = os.path.join(WORKING_DIR, "input/morph/dem.nc")
    INIT_TIME        = "latest"                      # "latest" or "YYYY-MM-DDTHH" (UTC)
    METEO_OUTPUT_DIR = os.path.join(WORKING_DIR, "input/meteo")  # output directory for mHM-ready files
    LATLON_OUTPUT_DIR = os.path.join(WORKING_DIR, "input/latlon") # output directory for latlon.nc
    FORECAST         = False                            # use forecast (True) or analysis (False) HRRR subscription

    # --- HRRR subscription -----------------------------------------------
    # Repo name is read from the HRRR_REPO env var.
    if FORECAST:
        HRRR_REPO             = os.environ["HRRR_FORECAST_REPO"]
        HRRR_BRANCH_OR_TAG    = os.environ.get("HRRR_FORECAST_BRANCH_OR_TAG", "main")
    else:
        HRRR_REPO             = os.environ["HRRR_ANALYSIS_REPO"]
        HRRR_BRANCH_OR_TAG    = os.environ.get("HRRR_ANALYSIS_BRANCH_OR_TAG", "main")
    main()