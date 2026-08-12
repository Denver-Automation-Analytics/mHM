"""
mHM calibration assembler.

Validates all outputs from mod10–mod17, introspects grid metadata, writes a
calibration-ready mhm.nml (optimize=.TRUE.) plus companion namelists into the
domain directory, then launches the mHM binary and streams its output.

Run order: mod10 → mod11 → mod12 → mod13 → mod14 → mod15 → mod16 → mod17 → mod18.

mHM invocation: the binary is run with cwd=WORKING_DIR so that it finds all
four nml files without path arguments.

Geology: no dedicated runner exists; _bootstrap_geology() creates a single-class
placeholder when geology files are absent.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import date
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import L0_CELL_SIZE_M, L1_CELL_SIZE_M, L2_CELL_SIZE_M, OUTPUT_CRS, N_OMP_THREADS, START_DATE, END_DATE, TIMESTEP, WARMUP_DAYS, WORKING_DIR, ROUTING_METHOD, OPTI_OBJECTIVE, N_ITERATIONS
from pathlib import Path

from readers import derive_eval_period, read_gauge_info, read_gauge_obs_window, read_lcover_scenes, read_meteo_dates, read_soil_info
from nml_writer import write_mhm_nml

# ---------------------------------------------------------------------------
# USER INPUTS — edit these paths and settings to reconfigure
# ---------------------------------------------------------------------------
MHM_BINARY       = "/workspace/build/mhm"

OPTI_METHOD      = 1     # 1=DDS, 2=Simulated Annealing, 3=SCE
# Objective function -> mHM opti_function from config.OPTI_OBJECTIVE.
OPTI_FUNCTION = {
    "nse":       1,   # 1 - NSE(Q)
    "lnnse":     2,   # 1 - lnNSE(Q); emphasises low flows
    "nse_lnnse": 3,   # 1 - 0.5*(NSE + lnNSE)
    "kge":       9,   # 1 - KGE(Q)
    "multi_kge": 14,  # power-6 combination of per-gauge KGE
}.get(OPTI_OBJECTIVE)
if OPTI_FUNCTION is None:
    raise ValueError(
        f"Unexpected OPTI_OBJECTIVE {OPTI_OBJECTIVE!r}. Must be "
        "'nse', 'lnnse', 'nse_lnnse', 'kge', or 'multi_kge'.")
WARMING_DAYS     = WARMUP_DAYS  # spin-up days before eval period (from config.py)
# Model timestep [h] derived from config.TIMESTEP (single source of truth).
MODEL_TIMESTEP_H = {"hourly": 1, "daily": 24}.get(TIMESTEP)
if MODEL_TIMESTEP_H is None:
    raise ValueError(f"Unexpected TIMESTEP {TIMESTEP!r}. Must be 'hourly' or 'daily'.")
# mRM routing scheme -> mHM processCase(8) from config.ROUTING_METHOD.
ROUTING_CASE = {"muskingum": 1, "adaptive": 2, "adaptive_varying": 3}.get(ROUTING_METHOD)
if ROUTING_CASE is None:
    raise ValueError(
        f"Unexpected ROUTING_METHOD {ROUTING_METHOD!r}. "
        "Must be 'muskingum', 'adaptive', or 'adaptive_varying'.")
SNAP_RADIUS_CELLS = 2    # gauge stream-snap search radius in L0 cells (±)
# Minimum flow accumulation (in L0 cells) for a snapped gauge to count as on-channel.
MIN_CHANNEL_FACC_CELLS = 50

REPO_PARAM_NML   = "/workspace/mhm_parameter.nml"
REPO_OUTPUT_NML  = "/workspace/mhm_outputs.nml"
REPO_MRM_OUT_NML = "/workspace/mrm_outputs.nml"
# ---------------------------------------------------------------------------

if L2_CELL_SIZE_M % L1_CELL_SIZE_M != 0:
    raise ValueError(
        f"L2_CELL_SIZE_M={L2_CELL_SIZE_M} is not a whole-number multiple "
        f"of L1_CELL_SIZE_M={L1_CELL_SIZE_M} (runners/config.py)."
    )

if L1_CELL_SIZE_M % L0_CELL_SIZE_M != 0:
    raise ValueError(
        f"L1_CELL_SIZE_M={L1_CELL_SIZE_M} is not a whole-number multiple "
        f"of L0_CELL_SIZE_M={L0_CELL_SIZE_M} (runners/config.py)."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("calibrate_to_mhm")


# ---------------------------------------------------------------------------
# Geology placeholder bootstrap
# ---------------------------------------------------------------------------

# Single formation: the bootstrap map only ever assigns class 1, so extra
# GeoParam entries would be dead parameters that DDS needlessly optimizes.
_GEO_CLASSDEF = """\
nGeo_Formations  1
GeoParam(i)   ClassUnit     Karstic      Description
         1                1           0      GeoUnit-1
!<-END
"""


def _bootstrap_geology(morph_dir: Path) -> None:
    """Create single-class geology files from the DEM mask if absent."""
    classdef = morph_dir / "geology_classdefinition.txt"
    classmap  = morph_dir / "geology_class.asc"
    if classdef.exists() and classmap.exists():
        return

    import netCDF4 as nc4
    import numpy as np

    log.info("Bootstrapping geology files from DEM mask (single class)...")

    dem_nc = morph_dir / "dem.nc"
    with nc4.Dataset(dem_nc) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        dem = ds.variables["dem"][:]        # masked array, (y, x)
        fill = ds.variables["dem"]._FillValue

    nrows, ncols = dem.shape
    xres = float(x[1] - x[0])
    xll  = float(x[0]) - xres / 2
    yll  = float(y[-1]) - xres / 2

    if not classdef.exists():
        classdef.write_text(_GEO_CLASSDEF)
        log.info("Written: %s", classdef)

    if not classmap.exists():
        # class 1 where DEM is valid, nodata elsewhere
        grid = np.where(np.asarray(dem) != fill, 1, -9999).astype(np.int32)
        with open(classmap, "w") as fh:
            fh.write(f"ncols         {ncols}\n")
            fh.write(f"nrows         {nrows}\n")
            fh.write(f"xllcorner     {xll:.1f}\n")
            fh.write(f"yllcorner     {yll:.1f}\n")
            fh.write(f"cellsize      {int(xres)}\n")
            fh.write("NODATA_value  -9999\n")
            for row in grid:
                fh.write(" ".join(str(v) for v in row) + "\n")
        log.info("Written: %s", classmap)


def _sync_geoparameter(param_nml: Path, morph_dir: Path) -> None:
    """Trim the &geoparameter block to match nGeo_Formations (mHM requires equality)."""
    classdef = morph_dir / "geology_classdefinition.txt"
    if not (classdef.exists() and param_nml.exists()):
        return
    n_geo = int(classdef.read_text().split(None, 2)[1])

    lines = param_nml.read_text().splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.strip().startswith("&geoparameter"))
    except StopIteration:
        return
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "/")
    templates = [lines[i] for i in range(start + 1, end) if "GeoParam(" in lines[i]]
    if not templates or len(templates) == n_geo:
        return

    rebuilt = []
    for idx in range(1, n_geo + 1):
        src = templates[idx - 1] if idx <= len(templates) else templates[-1]
        rebuilt.append(re.sub(r"GeoParam\(\d+,:\)", f"GeoParam({idx},:)", src, count=1))
    new_lines = lines[:start + 1] + rebuilt + lines[end:]
    param_nml.write_text("\n".join(new_lines) + "\n")
    log.info("Synced &geoparameter to %d geology unit(s): %s", n_geo, param_nml)


def _ascii_header(path: Path) -> dict:
    """Read the 6-line header of an ESRI ASCII grid, return lowercase-key dict."""
    header = {}
    with open(path) as fh:
        for _ in range(6):
            k, v = fh.readline().split(None, 1)
            header[k.lower()] = v.strip()
    return header


def _resample_ascii_to_l0(src_asc: Path, dem_nc: Path) -> None:
    """Resample a categorical ESRI ASCII grid to the L0 grid in-place.

    Reads the target grid shape from *dem_nc* (mod10 output); CRS is the
    shared OUTPUT_CRS constant from config.py.
    Uses GDAL Warp with mode resampling (appropriate for class integers).
    Skips the operation when the source header already matches L0.
    """
    from osgeo import gdal

    import netCDF4 as nc4
    import numpy as np

    with nc4.Dataset(dem_nc) as ds:
        x    = ds.variables["x"][:]
        y    = ds.variables["y"][:]
    xres  = float(x[1] - x[0])
    xmin  = float(x[0])  - xres / 2
    xmax  = float(x[-1]) + xres / 2
    ymin  = float(y[-1]) - xres / 2
    ymax  = float(y[0])  + xres / 2
    ncols = len(x)
    nrows = len(y)

    # --- check if resampling is needed --------------------------------------
    hdr = _ascii_header(src_asc)
    if (int(hdr["ncols"]) == ncols and int(hdr["nrows"]) == nrows
            and abs(float(hdr["cellsize"]) - xres) < 1e-6):
        return   # already on L0 grid

    log.info("Resampling %s to L0 grid (mode, %d m) …", src_asc.name, int(xres))

    src_cellsize = float(hdr["cellsize"])
    src_xll      = float(hdr["xllcorner"])
    src_yll      = float(hdr["yllcorner"])
    src_ncols    = int(hdr["ncols"])
    src_nrows    = int(hdr["nrows"])
    nodata_val   = int(float(hdr.get("nodata_value", "-9999")))

    # read data rows (skip 6-line header)
    with open(src_asc) as fh:
        for _ in range(6):
            fh.readline()
        data = np.loadtxt(fh, dtype=np.int32)

    # --- write source into an in-memory GeoTIFF with the known CRS ----------
    mem = gdal.GetDriverByName("MEM")
    src_ds = mem.Create("", src_ncols, src_nrows, 1, gdal.GDT_Int32)
    ull_y  = src_yll + src_nrows * src_cellsize      # upper-left y
    src_ds.SetGeoTransform((src_xll, src_cellsize, 0.0, ull_y, 0.0, -src_cellsize))
    src_ds.SetProjection(OUTPUT_CRS)
    band = src_ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(nodata_val)

    # --- warp to L0 grid ----------------------------------------------------
    dst_ds = gdal.Warp(
        "",
        src_ds,
        format="MEM",
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xres,
        yRes=xres,
        dstSRS=OUTPUT_CRS,
        resampleAlg="mode",
        dstNodata=nodata_val,
    )

    out = dst_ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    src_ds = dst_ds = None   # close GDAL datasets

    # --- write result back as ESRI ASCII grid in-place ----------------------
    with open(src_asc, "w") as fh:
        fh.write(f"ncols         {ncols}\n")
        fh.write(f"nrows         {nrows}\n")
        fh.write(f"xllcorner     {xmin:.1f}\n")
        fh.write(f"yllcorner     {ymin:.1f}\n")
        fh.write(f"cellsize      {int(xres)}\n")
        fh.write(f"NODATA_value  {nodata_val}\n")
        for row in out:
            fh.write(" ".join(str(v) for v in row) + "\n")
    log.info("Resampled %s → %d×%d @ %d m", src_asc.name, ncols, nrows, int(xres))


def _build_idgauges_asc(morph_dir: Path, gauge_dir: Path, dem_nc: Path) -> None:
    """Burn gauge local_ids into an L0 ASCII raster in *morph_dir*.

    Reads gauge lat/lon from id_map.csv, projects to LCC, snaps each gauge to
    the highest-facc valid cell within SNAP_RADIUS_CELLS of the nearest cell
    (so gauges land on the channel), and writes morph/idgauges.asc.
    Always rebuilds (a stale file silently corrupts calibration), and validates
    that every gauge snapped onto a channel before writing.
    """
    import csv
    import netCDF4 as nc4_mod
    import numpy as np
    from pyproj import Transformer

    dst = morph_dir / "idgauges.asc"

    # Load L0 grid
    with nc4_mod.Dataset(dem_nc) as ds:
        x_l0 = np.array(ds.variables["x"][:])
        y_l0 = np.array(ds.variables["y"][:])
        dem_vals = np.array(ds.variables["dem"][:])           # (nrows, ncols)
        dem_fill = float(ds.variables["dem"]._FillValue)
    ncols, nrows = len(x_l0), len(y_l0)

    # Flow accumulation for stream-snapping (same L0 grid/mask as the DEM)
    with nc4_mod.Dataset(morph_dir / "facc.nc") as ds:
        facc_vals = np.array(ds.variables["facc"][:])
        facc_fill = int(ds.variables["facc"]._FillValue)

    # Always rebuild: a stale idgauges.asc silently corrupts the calibration.
    log.info("Building morph/idgauges.asc at L0 grid (forced rebuild) …")

    # Transform gauges from WGS84 (lon, lat) → LCC (x, y)
    tf = Transformer.from_crs("EPSG:4326", OUTPUT_CRS, always_xy=True)

    # Valid-domain mask (True where DEM has data)
    valid = (dem_vals != dem_fill)
    # facc restricted to valid cells; -1 marks nodata so argmax ignores it
    facc_stream = np.where(valid & (facc_vals != facc_fill), facc_vals, -1)

    grid = np.full((nrows, ncols), -9999, dtype=np.int32)

    # Per-gauge snap record: gid -> (row, col, facc_before, facc_after) or None.
    placements: dict[int, tuple[int, int, int, int] | None] = {}

    id_map_path = gauge_dir / "id_map.csv"
    with open(id_map_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = int(row["local_id"])
            lon = float(row["lon"])
            lat = float(row["lat"])

            gx, gy = tf.transform(lon, lat)

            # Snap to nearest L0 cell
            ci = int(np.argmin(np.abs(x_l0 - gx)))   # column index
            ri = int(np.argmin(np.abs(y_l0 - gy)))   # row index (y descends)

            # Stay in bounds
            ci = max(0, min(ci, ncols - 1))
            ri = max(0, min(ri, nrows - 1))

            # Stream-snap onto the highest-facc valid cell in a small window
            r0, r1 = max(0, ri - SNAP_RADIUS_CELLS), min(nrows, ri + SNAP_RADIUS_CELLS + 1)
            c0, c1 = max(0, ci - SNAP_RADIUS_CELLS), min(ncols, ci + SNAP_RADIUS_CELLS + 1)
            win = facc_stream[r0:r1, c0:c1]
            if win.max() >= 0:
                lr, lc = np.unravel_index(int(np.argmax(win)), win.shape)
                sri, sci = r0 + lr, c0 + lc
                if (sri, sci) != (ri, ci):
                    log.info("Gauge %d snapped (%d,%d)→(%d,%d)  facc %d→%d",
                             gid, ri, ci, sri, sci,
                             int(facc_stream[ri, ci]), int(facc_stream[sri, sci]))
                grid[sri, sci] = gid
                placements[gid] = (sri, sci, int(facc_stream[ri, ci]),
                                   int(facc_stream[sri, sci]))
            elif valid[ri, ci]:
                grid[ri, ci] = gid
                placements[gid] = (ri, ci, int(facc_stream[ri, ci]),
                                   int(facc_stream[ri, ci]))
            else:
                placements[gid] = None

    xres = float(x_l0[1] - x_l0[0])
    xll  = float(x_l0[0])  - xres / 2
    yll  = float(y_l0[-1]) - xres / 2

    # Validate gauge snapping before writing so a bad file never lands on disk.
    cell_km2 = (xres * xres) / 1e6
    problems: list[str] = []
    log.info("Gauge snapping report (cell area %.4f km²):", cell_km2)
    for gid in sorted(placements):
        info = placements[gid]
        if info is None:
            log.warning("  gauge %d: NOT PLACED (projected outside valid domain)", gid)
            problems.append(f"gauge {gid}: not placed (outside valid domain)")
            continue
        sri, sci, f0, f1 = info
        area = f1 * cell_km2
        flag = "OK" if f1 >= MIN_CHANNEL_FACC_CELLS else "OFF-CHANNEL"
        log.info("  gauge %d: (%d,%d) facc %d→%d  area %.1f km²  [%s]",
                 gid, sri, sci, f0, f1, area, flag)
        if f1 < MIN_CHANNEL_FACC_CELLS:
            problems.append(
                f"gauge {gid}: snapped facc {f1} cells (< {MIN_CHANNEL_FACC_CELLS}); "
                f"area {area:.1f} km² looks off-channel")

    placed_cells = [(v[0], v[1]) for v in placements.values() if v is not None]
    if len(set(placed_cells)) != len(placed_cells):
        problems.append("two gauges share the same cell (collision)")
    if problems:
        raise ValueError(
            "idgauges.asc gauge-snapping validation failed:\n  "
            + "\n  ".join(problems))
    log.info("All %d gauges snapped onto channels (facc ≥ %d cells).",
             len(placements), MIN_CHANNEL_FACC_CELLS)

    with open(dst, "w") as fh:
        fh.write(f"ncols         {ncols}\n")
        fh.write(f"nrows         {nrows}\n")
        fh.write(f"xllcorner     {xll:.1f}\n")
        fh.write(f"yllcorner     {yll:.1f}\n")
        fh.write(f"cellsize      {int(xres)}\n")
        fh.write("NODATA_value  -9999\n")
        for row_data in grid:
            fh.write(" ".join(str(v) for v in row_data) + "\n")

    n_placed = int((grid != -9999).sum())
    log.info("Written: %s  (%d gauges placed)", dst, n_placed)


def _ensure_latlon_nc(latlon_dir: Path, dem_nc: Path, resolution_hydrology: int) -> None:
    """Regenerate latlon.nc when the L0 shape no longer matches dem.nc."""
    import netCDF4 as nc4_mod

    # Expected L0 shape
    with nc4_mod.Dataset(dem_nc) as ds:
        x_l0 = ds.variables["x"][:]
        y_l0 = ds.variables["y"][:]
    ncols_l0, nrows_l0 = len(x_l0), len(y_l0)

    # Skip if latlon.nc already has the right L0 shape
    out_file = latlon_dir / "latlon.nc"
    if out_file.exists():
        with nc4_mod.Dataset(out_file) as ds:
            if "lat_l0" in ds.variables:
                sh = ds.variables["lat_l0"].shape
                if sh[0] == nrows_l0 and sh[1] == ncols_l0:
                    return

    xres_l0 = float(x_l0[1] - x_l0[0])
    log.info(
        "Regenerating latlon.nc: L0=%dm (%d×%d), L1=%dm …",
        int(xres_l0), ncols_l0, nrows_l0, resolution_hydrology,
    )

    xll = float(x_l0[0])  - xres_l0 / 2
    yll = float(y_l0[-1]) - xres_l0 / 2

    l0_header = {
        "ncols": ncols_l0,
        "nrows": nrows_l0,
        "xllcorner": xll,
        "yllcorner": yll,
        "cellsize": xres_l0,
        "NODATA_value": -9999.0,
    }

    factor = resolution_hydrology // int(xres_l0)
    l1_header = {
        "ncols": ncols_l0 // factor,
        "nrows": nrows_l0 // factor,
        "xllcorner": xll,
        "yllcorner": yll,
        "cellsize": float(resolution_hydrology),
        "NODATA_value": -9999.0,
    }

    sys.path.insert(0, str(Path(__file__).parent.parent / "mod11_meteo_to_mhm"))
    from latlon_grid import create_latlon  # noqa: E402

    latlon_dir.mkdir(parents=True, exist_ok=True)
    create_latlon(
        out_file   = out_file,
        coord_sys  = OUTPUT_CRS,
        header_l0  = l0_header,
        header_l1  = l1_header,
        header_l11 = l1_header,
    )
    log.info("Written: %s", out_file)


# ---------------------------------------------------------------------------
# Phase 1: input validation
# ---------------------------------------------------------------------------

def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        log.error("Missing: %s  (run %s first)", path, produced_by)
        sys.exit(1)
    return path


def validate_inputs(domain: Path) -> None:
    inp = domain / "input"

    # mod10 — terrain morphology (NC format)
    for name in ("dem.nc", "slope.nc", "aspect.nc", "fdir.nc", "facc.nc"):
        _require(inp / "morph" / name, "mod10_dem_to_mhm")

    # mod13 — land cover
    if not list((inp / "luse").glob("lc_*.asc")):
        log.error("Missing: lc_*.asc in %s/input/luse/  (run mod13_landcover_to_mhm first)", domain)
        sys.exit(1)

    # mod14 — soils
    for name in ("soil_class.asc", "soil_classdefinition.txt"):
        _require(inp / "morph" / name, "mod14_soils_to_mhm")

    # mod11 — meteorology (NC format)
    _require(inp / "meteo" / "pre"  / "pre.nc",   "mod11_meteo_to_mhm")
    _require(inp / "meteo" / "tavg" / "tavg.nc",  "mod11_meteo_to_mhm")

    # mod12 — PET (NC format)
    _require(inp / "meteo" / "pet" / "pet.nc",    "mod12_pet_to_mhm")

    # mod15 — gauges (NC format for raster, CSV for metadata)
    _require(inp / "gauge" / "id_map.csv",         "mod15_gauges_to_mhm")
    _require(inp / "gauge" / "idgauges.nc",         "mod15_gauges_to_mhm")

    # mod16 — LAI gridded NetCDF
    _require(inp / "lai" / "lai.nc",               "mod16_lai_to_mhm")

    # mod17 — latlon grid
    _require(inp / "latlon" / "latlon.nc",          "mod17_latlon_to_mhm")

    # mhm binary
    _require(Path(MHM_BINARY), "cmake build")

    log.info("All prerequisite files present.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    domain = Path(WORKING_DIR)
    inp    = domain / "input"

    # Phase 1 — validate
    validate_inputs(domain)

    # Phase 2 — extract metadata
    first_meteo, last_meteo = read_meteo_dates(inp / "meteo" / "pre" / "pre.nc")
    log.info("Meteo period: %s – %s", first_meteo, last_meteo)

    gauges = read_gauge_info(inp / "gauge" / "id_map.csv")
    if not gauges:
        log.error("id_map.csv contains no gauges — calibration requires at least one.")
        sys.exit(1)
    log.info("Gauges: %s", [g["filename"] for g in gauges])

    # Clamp the eval window to the dates covered by every gauge file so the
    # simulation period never exceeds the observations mHM reads for calibration.
    gauge_start, gauge_end = read_gauge_obs_window(inp / "gauge", gauges)
    log.info("Gauge obs window (all gauges): %s – %s", gauge_start, gauge_end)

    try:
        eval_start, eval_end = derive_eval_period(
            first_meteo, last_meteo, WARMING_DAYS,
            obs_start=max(date.fromisoformat(START_DATE), gauge_start),
            obs_end=min(date.fromisoformat(END_DATE), gauge_end),
        )
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)
    log.info("Eval period:  %s – %s  (warming_days=%d)", eval_start, eval_end, WARMING_DAYS)

    lcover_scenes = read_lcover_scenes(inp / "luse")
    log.info("Land cover scenes: %s", [f for _, f in lcover_scenes])


    resolution = L1_CELL_SIZE_M
    log.info("L1 resolution: %d m (from config.py; L0 terrain is %d m)", resolution, L0_CELL_SIZE_M)

    n_soil_horizons, soil_depths = read_soil_info(inp / "morph" / "soil_classdefinition.txt")
    log.info("Soil horizons: %d  depths: %s mm", n_soil_horizons, soil_depths)

    # Phase 2b — bootstrap geology if absent; resample ASCII inputs to L0
    _bootstrap_geology(inp / "morph")
    dem_nc = inp / "morph" / "dem.nc"
    _resample_ascii_to_l0(inp / "morph" / "soil_class.asc", dem_nc)
    _build_idgauges_asc(inp / "morph", inp / "gauge", dem_nc)
    _ensure_latlon_nc(inp / "latlon", dem_nc, resolution)

    # Phase 3 — generate mhm.nml
    nml_path = domain / "mhm.nml"
    write_mhm_nml(
        nml_path,
        domain,
        resolution_hydrology = resolution,
        timestep             = MODEL_TIMESTEP_H,
        opti_method          = OPTI_METHOD,
        opti_function        = OPTI_FUNCTION,
        routing_case         = ROUTING_CASE,
        n_iterations         = N_ITERATIONS,
        warming_days         = WARMING_DAYS,
        eval_start           = eval_start,
        eval_end             = eval_end,
        lcover_scenes        = lcover_scenes,
        gauges               = gauges,
        n_soil_horizons      = n_soil_horizons,
        soil_depths          = soil_depths,
    )
    log.info("Written: %s", nml_path)

    # Phase 4 — create output directories and copy companion namelists
    (domain / "output").mkdir(parents=True, exist_ok=True)
    (domain / "restart").mkdir(parents=True, exist_ok=True)
    (inp / "optional_data").mkdir(parents=True, exist_ok=True)

    for src, name in (
        (REPO_PARAM_NML,   "mhm_parameter.nml"),
        (REPO_OUTPUT_NML,  "mhm_outputs.nml"),
        (REPO_MRM_OUT_NML, "mrm_outputs.nml"),
    ):
        dst = domain / name
        shutil.copy2(src, dst)
        log.info("Copied → %s", dst)

    _sync_geoparameter(domain / "mhm_parameter.nml", inp / "morph")

    # Phase 5 — run mHM calibration
    log.info("Launching mHM calibration: %s  (cwd=%s)", MHM_BINARY, domain)
    log.info("opti_method=%d  opti_function=%d  n_iterations=%d", OPTI_METHOD, OPTI_FUNCTION, N_ITERATIONS)
    log.info("OMP_NUM_THREADS=%d", N_OMP_THREADS)

    proc = subprocess.Popen(
        [MHM_BINARY],
        cwd=str(domain),
        env={**os.environ, "OMP_NUM_THREADS": str(N_OMP_THREADS)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        log.info("[mhm] %s", line.rstrip())
    proc.wait()

    if proc.returncode != 0:
        log.error("mHM exited with code %d", proc.returncode)
        sys.exit(proc.returncode)

    log.info("mHM calibration finished. Output in %s/output/", domain)


if __name__ == "__main__":
    main()
