"""
mHM calibration assembler.

Validates all outputs from mod10–mod18, introspects grid metadata, writes a
calibration-ready mhm.nml (optimize=.TRUE.) plus companion namelists into the
domain directory, then launches the mHM binary and streams its output.

Run order: mod10 → mod11 → mod12 → mod13 → mod14 → mod15 → mod16 → mod17 → mod18 → mod19.

mHM invocation: the binary is run with cwd=WORKING_DIR so that it finds all
four nml files without path arguments.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    L0_CELL_SIZE_M,
    L1_CELL_SIZE_M,
    L2_CELL_SIZE_M,
    OUTPUT_CRS,
    N_OMP_THREADS,
    START_DATE,
    END_DATE,
    EVAL_START_DATE,
    TIMESTEP,
    WARMUP_DAYS,
    WORKING_DIR,
    ROUTING_METHOD,
    OPTI_OBJECTIVE,
    N_ITERATIONS,
    SEED,
    RESUME,
    GAUGE_DRAINAGE_AREAS_SQMI,
)
from pathlib import Path

from readers import (
    derive_eval_period,
    read_gauge_info,
    read_gauge_obs_window,
    read_lcover_scenes,
    read_meteo_dates,
    read_soil_info,
)
from nml_writer import write_mhm_nml

# ---------------------------------------------------------------------------
# USER INPUTS — edit these paths and settings to reconfigure
# ---------------------------------------------------------------------------
MHM_BINARY = "/workspace/build/mhm"

OPTI_METHOD = 1  # 1=DDS, 2=Simulated Annealing, 3=SCE
# Objective function -> mHM opti_function from config.OPTI_OBJECTIVE.
OPTI_FUNCTION = {
    "nse": 1,  # 1 - NSE(Q)
    "lnnse": 2,  # 1 - lnNSE(Q); emphasises low flows
    "nse_lnnse": 3,  # 1 - 0.5*(NSE + lnNSE)
    "kge": 9,  # 1 - KGE(Q)
    "multi_kge": 14,  # power-6 combination of per-gauge KGE
    "wnse": 31,  # 1 - weighted NSE(Q); weights errors by observed flow (Hundecha & Bardossy 2004)
    "kge_q_et": 29,  # combines KGE(Q) with catchment-average actual-ET (needs et.nc)
    "multi_objective_lnnse_highflow_lnnse_lowflow": 18,  # power-6 combination of per-gauge lnnse_highflow and lnnse_lowflow
}.get(OPTI_OBJECTIVE)
if OPTI_FUNCTION is None:
    raise ValueError(
        f"Unexpected OPTI_OBJECTIVE {OPTI_OBJECTIVE!r}. Must be "
        "'nse', 'lnnse', 'nse_lnnse', 'kge', 'multi_kge', 'wnse', or 'kge_q_et'."
    )
# Model timestep [h] derived from config.TIMESTEP (single source of truth).
# Always 1 h: mHM's daily model timestep (24) mis-indexes daily meteo (iMeteoTS bug in
# mo_meteo_handler.f90 advances forcing only every 24 model-days), so run the model
# hourly and let it disaggregate the daily forcing (as the reference test_domain does).
MODEL_TIMESTEP_H = {"hourly": 1, "daily": 1}.get(TIMESTEP)
if MODEL_TIMESTEP_H is None:
    raise ValueError(f"Unexpected TIMESTEP {TIMESTEP!r}. Must be 'hourly' or 'daily'.")
# Gridded-output write frequency -> mhm_outputs.nml timeStep_model_outputs,
# matched to the input/model resolution from config.TIMESTEP.
#   hourly -> 1  (after each 1-hour model step); daily -> -1 (daily preset)
OUTPUT_TIMESTEP = {"hourly": 1, "daily": -1}.get(TIMESTEP)
if OUTPUT_TIMESTEP is None:
    raise ValueError(f"Unexpected TIMESTEP {TIMESTEP!r}. Must be 'hourly' or 'daily'.")
# mRM routing scheme -> mHM processCase(8) from config.ROUTING_METHOD.
ROUTING_CASE = {"muskingum": 1, "adaptive": 2, "adaptive_varying": 3}.get(
    ROUTING_METHOD
)
if ROUTING_CASE is None:
    raise ValueError(
        f"Unexpected ROUTING_METHOD {ROUTING_METHOD!r}. "
        "Must be 'muskingum', 'adaptive', or 'adaptive_varying'."
    )
SNAP_RADIUS_CELLS = 8  # gauge stream-snap search radius in L0 cells (±)
# Fraction of the search-window's peak flow accumulation that defines the "major
# channel". Used only for gauges WITHOUT a known drainage area: the gauge snaps to
# the nearest such cell, ignoring small tributaries that bend closer while avoiding
# the downstream drift a plain max-facc pick would cause past a confluence.
SNAP_CHANNEL_FRACTION = 0.5
# Minimum flow accumulation (in L0 cells) for a snapped gauge to count as on-channel.
MIN_CHANNEL_FACC_CELLS = 50
SQMI_TO_KM2 = 2.589988110336  # exact statute square mile -> km²
# Max relative mismatch between a gauge's known drainage area and its snapped cell's
# upstream area before the snap is rejected (guards against wrong coords/area/units).
AREA_MATCH_TOL = 0.5

REPO_PARAM_NML = "/workspace/mhm_parameter.nml"
REPO_OUTPUT_NML = "/workspace/mhm_outputs.nml"
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


def _sync_geoparameter(param_nml: Path, morph_dir: Path) -> None:
    """Trim the &geoparameter block to match nGeo_Formations (mHM requires equality)."""
    classdef = morph_dir / "geology_classdefinition.txt"
    if not (classdef.exists() and param_nml.exists()):
        return
    n_geo = int(classdef.read_text().split(None, 2)[1])

    lines = param_nml.read_text().splitlines()
    try:
        start = next(
            i for i, l in enumerate(lines) if l.strip().startswith("&geoparameter")
        )
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
    new_lines = lines[: start + 1] + rebuilt + lines[end:]
    param_nml.write_text("\n".join(new_lines) + "\n")
    log.info("Synced &geoparameter to %d geology unit(s): %s", n_geo, param_nml)


# name = lower, upper, VALUE, flag, flag  -> groups: prefix, name, lower, upper, value, rest
# Numbers may be Fortran scientific notation (e.g. 7.880E-01), so consume the exponent.
_NUM = r"-?[0-9.]+(?:[eEdD][+-]?[0-9]+)?"
_PARAM_VALUE_RE = re.compile(
    rf"^(\s*([^=]+?)\s*=\s*({_NUM})\s*,\s*({_NUM})\s*,\s*)({_NUM})(.*)$"
)


def _pf(s: str) -> float:
    """Parse a Fortran real that may use D/d exponents."""
    return float(s.replace("D", "E").replace("d", "e"))


def _apply_resume(param_nml: Path, final_nml: Path) -> None:
    """Seed DDS from the previous calibration.

    Copies each parameter's final value from *final_nml* into the value (3rd)
    column of *param_nml*, clamped to this file's (possibly changed) bounds, so
    the optimiser restarts from the last result while honouring current ranges.
    """
    if not final_nml.exists():
        log.warning(
            "RESUME=True but %s not found; using template start values.", final_nml
        )
        return
    finals: dict[str, float] = {}
    for line in final_nml.read_bytes().decode("latin-1").splitlines():
        m = _PARAM_VALUE_RE.match(line)
        if m:
            finals[m.group(2).strip()] = _pf(m.group(5))

    out: list[str] = []
    n = 0
    for line in param_nml.read_bytes().decode("latin-1").splitlines():
        m = _PARAM_VALUE_RE.match(line)
        name = m.group(2).strip() if m else None
        if m and name in finals:
            lo, hi = _pf(m.group(3)), _pf(m.group(4))
            val = min(
                max(finals[name], lo), hi
            )  # clamp resumed value to current bounds
            out.append(f"{m.group(1)}{val:.10g}{m.group(6)}")
            n += 1
        else:
            out.append(line)
    param_nml.write_bytes(("\n".join(out) + "\n").encode("latin-1"))
    log.info("RESUME: seeded %d parameter start value(s) from %s.", n, final_nml.name)


def _strip_finalparam_garbage(final_nml: Path) -> None:
    """Drop the corrupted trailing block mHM's at-exit crash appends to FinalParam.nml.

    A heap-corruption SIGABRT/SIGSEGV in mHM's cleanup scribbles binary bytes into a
    throwaway namelist block after all real parameters are written; strip from the first
    non-printable line onward so the file is clean ASCII.
    """
    if not final_nml.exists():
        return
    lines = final_nml.read_bytes().decode("latin-1").splitlines()

    def _garbage(s: str) -> bool:
        return any(ord(c) < 9 or (13 < ord(c) < 32) or ord(c) > 126 for c in s)

    cut = next((i for i, l in enumerate(lines) if _garbage(l)), len(lines))
    if cut == len(lines):
        return
    clean = lines[:cut]
    while clean and clean[-1].strip() == "":
        clean.pop()
    final_nml.write_text("\n".join(clean) + "\n")
    log.info(
        "Stripped %d corrupted trailing line(s) from %s.",
        len(lines) - cut,
        final_nml.name,
    )


def _write_mhm_outputs_nml(src: str, dst: Path, output_timestep: int) -> None:
    """Copy mhm_outputs.nml to *dst*, forcing timeStep_model_outputs.

    The write frequency is overridden to match the model/input resolution
    (config.TIMESTEP) rather than the value carried by the repo template.
    """
    text = Path(src).read_text()
    new_text, n = re.subn(
        r"^(\s*timeStep_model_outputs\s*=\s*)-?\d+",
        rf"\g<1>{output_timestep}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        raise ValueError(f"timeStep_model_outputs not found in {src}")
    dst.write_text(new_text)


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
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
    xres = float(x[1] - x[0])
    xmin = float(x[0]) - xres / 2
    xmax = float(x[-1]) + xres / 2
    ymin = float(y[-1]) - xres / 2
    ymax = float(y[0]) + xres / 2
    ncols = len(x)
    nrows = len(y)

    # --- check if resampling is needed --------------------------------------
    hdr = _ascii_header(src_asc)
    if (
        int(hdr["ncols"]) == ncols
        and int(hdr["nrows"]) == nrows
        and abs(float(hdr["cellsize"]) - xres) < 1e-6
    ):
        return  # already on L0 grid

    log.info("Resampling %s to L0 grid (mode, %d m) …", src_asc.name, int(xres))

    src_cellsize = float(hdr["cellsize"])
    src_xll = float(hdr["xllcorner"])
    src_yll = float(hdr["yllcorner"])
    src_ncols = int(hdr["ncols"])
    src_nrows = int(hdr["nrows"])
    nodata_val = int(float(hdr.get("nodata_value", "-9999")))

    # read data rows (skip 6-line header)
    with open(src_asc) as fh:
        for _ in range(6):
            fh.readline()
        data = np.loadtxt(fh, dtype=np.int32)

    # --- write source into an in-memory GeoTIFF with the known CRS ----------
    mem = gdal.GetDriverByName("MEM")
    src_ds = mem.Create("", src_ncols, src_nrows, 1, gdal.GDT_Int32)
    ull_y = src_yll + src_nrows * src_cellsize  # upper-left y
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
    src_ds = dst_ds = None  # close GDAL datasets

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


def _build_idgauges_asc(
    morph_dir: Path,
    gauge_dir: Path,
    dem_nc: Path,
    drainage_areas_sqmi: dict[str, float],
) -> None:
    """Burn gauge local_ids into an L0 ASCII raster in *morph_dir*.

    Reads gauge lat/lon from id_map.csv, projects to LCC, and snaps each gauge to
    a flow-accumulation cell within SNAP_RADIUS_CELLS. When the gauge's USGS site_no
    is in *drainage_areas_sqmi*, it snaps to the cell whose upstream area best matches
    that known area (resolving confluences); otherwise it snaps to the nearest major
    channel. Writes morph/idgauges.asc, always rebuilding (a stale file silently
    corrupts calibration), and validates every gauge before writing.
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
        dem_vals = np.array(ds.variables["dem"][:])  # (nrows, ncols)
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
    valid = dem_vals != dem_fill
    # facc restricted to valid cells; -1 marks nodata so argmax ignores it
    facc_stream = np.where(valid & (facc_vals != facc_fill), facc_vals, -1)

    grid = np.full((nrows, ncols), -9999, dtype=np.int32)

    xres = float(x_l0[1] - x_l0[0])
    cell_km2 = (xres * xres) / 1e6

    # Per-gauge snap record: gid -> (row, col, facc_before, facc_after, target) or None.
    placements: dict[int, tuple[int, int, int, int, float | None] | None] = {}

    id_map_path = gauge_dir / "id_map.csv"
    with open(id_map_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = int(row["local_id"])
            site_no = row["site_no"].strip()
            lon = float(row["lon"])
            lat = float(row["lat"])

            # Known drainage area (if provided) as an expected facc cell count.
            area_sqmi = drainage_areas_sqmi.get(site_no)
            target_cells = (
                area_sqmi * SQMI_TO_KM2 / cell_km2 if area_sqmi is not None else None
            )

            gx, gy = tf.transform(lon, lat)

            # Snap to nearest L0 cell
            ci = int(np.argmin(np.abs(x_l0 - gx)))  # column index
            ri = int(np.argmin(np.abs(y_l0 - gy)))  # row index (y descends)

            # Stay in bounds
            ci = max(0, min(ci, ncols - 1))
            ri = max(0, min(ri, nrows - 1))

            # Stream-snap onto the highest-facc valid cell in a small window
            r0, r1 = (
                max(0, ri - SNAP_RADIUS_CELLS),
                min(nrows, ri + SNAP_RADIUS_CELLS + 1),
            )
            c0, c1 = (
                max(0, ci - SNAP_RADIUS_CELLS),
                min(ncols, ci + SNAP_RADIUS_CELLS + 1),
            )
            win = facc_stream[r0:r1, c0:c1]
            win_max = int(win.max())
            if win_max >= 0:
                rows_w, cols_w = np.nonzero(win >= 0)  # valid cells in the window
                faccs = win[rows_w, cols_w].astype(np.int64)
                d2 = (rows_w + r0 - ri) ** 2 + (cols_w + c0 - ci) ** 2
                if target_cells is not None:
                    # Known area: snap to the cell whose upstream area best matches it
                    # (tie-break on distance). This lands on the correct side of a
                    # confluence, which a facc-only heuristic cannot distinguish.
                    order = np.lexsort((d2, np.abs(faccs - target_cells)))
                else:
                    # No area given: nearest cell on the major channel (>= a fraction
                    # of the window peak). Avoids downstream drift from a plain max-facc
                    # pick and small-tributary snaps from nearest-on-any-channel.
                    channel_thresh = max(
                        MIN_CHANNEL_FACC_CELLS, SNAP_CHANNEL_FRACTION * win_max
                    )
                    on_channel = faccs >= channel_thresh
                    if not on_channel.any():  # no real channel in reach; keep the peak
                        on_channel = faccs >= win_max
                    d2_masked = np.where(on_channel, d2, d2.max() + 1)
                    order = np.lexsort((-faccs, d2_masked))
                best = int(order[0])
                sri, sci = int(rows_w[best] + r0), int(cols_w[best] + c0)
                if (sri, sci) != (ri, ci):
                    log.info(
                        "Gauge %d snapped (%d,%d)→(%d,%d)  facc %d→%d",
                        gid,
                        ri,
                        ci,
                        sri,
                        sci,
                        int(facc_stream[ri, ci]),
                        int(facc_stream[sri, sci]),
                    )
                grid[sri, sci] = gid
                placements[gid] = (
                    sri,
                    sci,
                    int(facc_stream[ri, ci]),
                    int(facc_stream[sri, sci]),
                    target_cells,
                )
            elif valid[ri, ci]:
                grid[ri, ci] = gid
                placements[gid] = (
                    ri,
                    ci,
                    int(facc_stream[ri, ci]),
                    int(facc_stream[ri, ci]),
                    target_cells,
                )
            else:
                placements[gid] = None

    xll = float(x_l0[0]) - xres / 2
    yll = float(y_l0[-1]) - xres / 2

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
        sri, sci, f0, f1, target = info
        area = f1 * cell_km2
        flag = "OK" if f1 >= MIN_CHANNEL_FACC_CELLS else "OFF-CHANNEL"
        if target is not None:
            exp_area = target * cell_km2
            area_err = abs(f1 - target) / target
            log.info(
                "  gauge %d: (%d,%d) facc %d→%d  area %.1f km² (expected %.1f km², "
                "%.1f%% off)  [%s]",
                gid,
                sri,
                sci,
                f0,
                f1,
                area,
                exp_area,
                area_err * 100.0,
                flag,
            )
            if area_err > AREA_MATCH_TOL:
                problems.append(
                    f"gauge {gid}: snapped area {area:.1f} km² is {area_err * 100:.0f}% "
                    f"off the expected {exp_area:.1f} km² (> {AREA_MATCH_TOL * 100:.0f}%); "
                    "check gauge coordinates, drainage area, or search radius"
                )
        else:
            log.info(
                "  gauge %d: (%d,%d) facc %d→%d  area %.1f km²  [%s]",
                gid,
                sri,
                sci,
                f0,
                f1,
                area,
                flag,
            )
        if f1 < MIN_CHANNEL_FACC_CELLS:
            problems.append(
                f"gauge {gid}: snapped facc {f1} cells (< {MIN_CHANNEL_FACC_CELLS}); "
                f"area {area:.1f} km² looks off-channel"
            )

    placed_cells = [(v[0], v[1]) for v in placements.values() if v is not None]
    if len(set(placed_cells)) != len(placed_cells):
        problems.append("two gauges share the same cell (collision)")
    if problems:
        raise ValueError(
            "idgauges.asc gauge-snapping validation failed:\n  " + "\n  ".join(problems)
        )
    log.info(
        "All %d gauges snapped onto channels (facc ≥ %d cells).",
        len(placements),
        MIN_CHANNEL_FACC_CELLS,
    )

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


def _ensure_latlon_nc(
    latlon_dir: Path, dem_nc: Path, resolution_hydrology: int
) -> None:
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
        int(xres_l0),
        ncols_l0,
        nrows_l0,
        resolution_hydrology,
    )

    xll = float(x_l0[0]) - xres_l0 / 2
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
        out_file=out_file,
        coord_sys=OUTPUT_CRS,
        header_l0=l0_header,
        header_l1=l1_header,
        header_l11=l1_header,
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
    inp = domain / "mhm_input"

    # mod10 — terrain morphology (NC format)
    for name in ("dem.nc", "slope.nc", "aspect.nc", "fdir.nc", "facc.nc"):
        _require(inp / "morph" / name, "mod10_dem_to_mhm")

    # mod13 — land cover
    if not list((inp / "luse").glob("lc_*.asc")):
        log.error(
            "Missing: lc_*.asc in %s/mhm_input/luse/  (run mod13_landcover_to_mhm first)",
            domain,
        )
        sys.exit(1)

    # mod14 — soils
    for name in ("soil_class.asc", "soil_classdefinition.txt"):
        _require(inp / "morph" / name, "mod14_soils_to_mhm")

    # mod11 — meteorology (NC format)
    _require(inp / "meteo" / "pre" / "pre.nc", "mod11_meteo_to_mhm")
    _require(inp / "meteo" / "tavg" / "tavg.nc", "mod11_meteo_to_mhm")

    # mod12 — PET (NC format)
    _require(inp / "meteo" / "pet" / "pet.nc", "mod12_pet_to_mhm")

    # mod15 — gauges (NC format for raster, CSV for metadata)
    _require(inp / "gauge" / "id_map.csv", "mod15_gauges_to_mhm")
    _require(inp / "gauge" / "idgauges.nc", "mod15_gauges_to_mhm")

    # mod16 — LAI gridded NetCDF
    _require(inp / "lai" / "lai.nc", "mod16_lai_to_mhm")

    # mod17 — latlon grid
    _require(inp / "latlon" / "latlon.nc", "mod17_latlon_to_mhm")

    # mhm binary
    _require(Path(MHM_BINARY), "cmake build")

    log.info("All prerequisite files present.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    domain = Path(WORKING_DIR)
    inp = domain / "mhm_input"

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
        eval_start, eval_end, warming_days = derive_eval_period(
            first_meteo,
            last_meteo,
            eval_start_date=date.fromisoformat(EVAL_START_DATE),
            warmup_days=WARMUP_DAYS,
            obs_start=max(date.fromisoformat(START_DATE), gauge_start),
            obs_end=min(date.fromisoformat(END_DATE), gauge_end),
        )
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)
    log.info(
        "Eval period:  %s – %s  (warming_days=%d, spin-up %s – %s)",
        eval_start,
        eval_end,
        warming_days,
        eval_start - timedelta(days=warming_days),
        eval_start - timedelta(days=1),
    )

    lcover_scenes = read_lcover_scenes(inp / "luse")
    log.info("Land cover scenes: %s", [f for _, f in lcover_scenes])

    resolution = L1_CELL_SIZE_M
    log.info(
        "L1 resolution: %d m (from config.py; L0 terrain is %d m)",
        resolution,
        L0_CELL_SIZE_M,
    )

    n_soil_horizons, soil_depths = read_soil_info(
        inp / "morph" / "soil_classdefinition.txt"
    )
    log.info("Soil horizons: %d  depths: %s mm", n_soil_horizons, soil_depths)

    # Phase 2b — resample ASCII inputs to L0
    dem_nc = inp / "morph" / "dem.nc"
    _resample_ascii_to_l0(inp / "morph" / "soil_class.asc", dem_nc)
    _build_idgauges_asc(
        inp / "morph", inp / "gauge", dem_nc, GAUGE_DRAINAGE_AREAS_SQMI
    )
    _ensure_latlon_nc(inp / "latlon", dem_nc, resolution)

    # Phase 3 — generate mhm.nml
    nml_path = domain / "mhm.nml"
    write_mhm_nml(
        nml_path,
        domain,
        resolution_hydrology=resolution,
        timestep=MODEL_TIMESTEP_H,
        opti_method=OPTI_METHOD,
        opti_function=OPTI_FUNCTION,
        routing_case=ROUTING_CASE,
        n_iterations=N_ITERATIONS,
        seed=SEED,
        warming_days=warming_days,
        eval_start=eval_start,
        eval_end=eval_end,
        lcover_scenes=lcover_scenes,
        gauges=gauges,
        n_soil_horizons=n_soil_horizons,
        soil_depths=soil_depths,
    )
    log.info("Written: %s", nml_path)

    # Phase 4 — create output directories and copy companion namelists
    (domain / "mhm_output").mkdir(parents=True, exist_ok=True)
    (domain / "restart").mkdir(parents=True, exist_ok=True)
    (inp / "optional_data").mkdir(parents=True, exist_ok=True)

    for src, name in (
        (REPO_PARAM_NML, "mhm_parameter.nml"),
        (REPO_MRM_OUT_NML, "mrm_outputs.nml"),
    ):
        dst = domain / name
        shutil.copy2(src, dst)
        log.info("Copied → %s", dst)

    # mhm_outputs.nml: gridded-output write frequency wired to config.TIMESTEP.
    out_nml = domain / "mhm_outputs.nml"
    _write_mhm_outputs_nml(REPO_OUTPUT_NML, out_nml, OUTPUT_TIMESTEP)
    log.info(
        "Written → %s (timeStep_model_outputs=%d, %s)",
        out_nml,
        OUTPUT_TIMESTEP,
        TIMESTEP,
    )

    _sync_geoparameter(domain / "mhm_parameter.nml", inp / "morph")

    # RESUME: reseed DDS start values from the previous run's FinalParam.nml.
    if RESUME:
        _apply_resume(domain / "mhm_parameter.nml", domain / "FinalParam.nml")

    # Phase 5 — run mHM calibration
    log.info("Launching mHM calibration: %s  (cwd=%s)", MHM_BINARY, domain)
    log.info(
        "opti_method=%d  opti_function=%d  n_iterations=%d",
        OPTI_METHOD,
        OPTI_FUNCTION,
        N_ITERATIONS,
    )
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

    final_nml = domain / "FinalParam.nml"
    # mHM can crash in its at-exit cleanup (SIGABRT/SIGSEGV, negative code) AFTER writing
    # FinalParam.nml; treat that as success if the calibrated file exists. A positive exit
    # code is a real mHM error.
    if proc.returncode < 0 and final_nml.exists():
        log.warning(
            "mHM terminated by signal %d after writing FinalParam.nml; "
            "treating as benign at-exit crash.",
            -proc.returncode,
        )
    elif proc.returncode != 0:
        log.error("mHM exited with code %d", proc.returncode)
        sys.exit(proc.returncode)

    _strip_finalparam_garbage(final_nml)
    log.info("mHM calibration finished. Output in %s/mhm_output/", domain)


if __name__ == "__main__":
    main()
