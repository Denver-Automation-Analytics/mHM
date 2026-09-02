"""TRITON gtiff outputs -> consolidated maps (mod22).

Consolidates the per-timestep TRITON GeoTIFF series into compressed netCDF cubes
and a pixel-maximum GeoTIFF per variable:

  H.nc  / H_max.tif   water depth
  MH.nc / MH_max.tif  running-maximum water depth (TRITON envelope)
  V.nc  / V_max.tif   depth-averaged velocity magnitude, V = sqrt(QX^2+QY^2)/H

H and MH are copied straight from the gtiff series; V is derived from the QX/QY
unit-discharge series and H, and is only produced when QX and QY are present. The
space-time cube is streamed one timestep at a time, so arbitrarily long/large
TRITON runs are handled without loading the whole cube into memory. Outputs keep
the native CRS and grid of the TRITON ``.vrt`` (no reprojection).

H.gif/MH.gif/V.gif animations are also rendered over a DEM hillshade, subsampled
to a bounded number of frames spanning the full run. TRITON's performance.txt/
performance/*.txt timing logs are turned into a load-balance figure and a
per-step time series (next to the H wet-cell/volume series when available) ->
perf_load_balance.png, perf_timeseries.png. Both steps skip gracefully (with a
warning) when their required inputs (DEM, clip, performance logs) are missing.

Run order: a TRITON run producing gtiff outputs -> mod22.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    END_DATE,
    L1_CELL_SIZE_M,
    NODATA,
    TRITON_COMPARE_OUT_DIR,
    TRITON_COMPARE_POINTS,
    TRITON_COMPARE_TZ,
    TRITON_GIF_CMAP,
    TRITON_GIF_FPS,
    TRITON_GIF_MAX_FRAMES,
    TRITON_GIF_VARS,
    TRITON_MAP_CFG,
    TRITON_MAP_CLIP,
    TRITON_MAP_DEM_TIF,
    TRITON_MAP_GTIFF_DIR,
    TRITON_MAP_HMIN,
    TRITON_MAP_MIN_DEPTH,
    TRITON_MAP_OUT_DIR,
    TRITON_MAP_SERIES_DIR,
    TRITON_PERF_DIR,
    TRITON_PERF_ROFF,
    TRITON_PERF_SUMMARY,
    TRITON_PERF_WET_VAR,
    TRITON_START_DATE,
    TRITON_START_FILE,
)

import compare
import gifs
import perf
import perf_plots
import readers
import series
import writers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("triton_to_map")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Consolidate TRITON gtiff outputs into netCDF + max maps."
    )
    p.add_argument(
        "--gtiff-dir",
        default=TRITON_MAP_GTIFF_DIR,
        help="directory of TRITON per-timestep GeoTIFFs",
    )
    p.add_argument(
        "--out-dir",
        default=TRITON_MAP_OUT_DIR,
        help="output directory for netCDF + max GeoTIFFs",
    )
    p.add_argument(
        "--cfg", default=TRITON_MAP_CFG, help="TRITON .cfg parsed for print_interval"
    )
    p.add_argument(
        "--start-date",
        default=TRITON_START_DATE,
        help="anchor date 'YYYY-MM-DD' for the time axis",
    )
    p.add_argument(
        "--start-file",
        default=TRITON_START_FILE,
        help="sidecar with the mod21-resolved sim start datetime; overrides --start-date when present",
    )
    p.add_argument(
        "--hmin",
        type=float,
        default=TRITON_MAP_HMIN,
        help="depth floor [m] below which velocity is masked",
    )
    p.add_argument(
        "--min-depth",
        type=float,
        default=TRITON_MAP_MIN_DEPTH,
        help="depth floor [m]; H/MH cells shallower than this are masked to NODATA (0 disables)",
    )
    p.add_argument(
        "--remax",
        action="store_true",
        help="rebuild only the max GeoTIFF from the cached netCDF (skips re-reading the gtiffs)",
    )
    p.add_argument(
        "--clip",
        default=TRITON_MAP_CLIP,
        help="watershed polygon to clip maps to (cells outside -> NODATA); empty string disables",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="regenerate outputs even if they already exist",
    )
    p.add_argument(
        "--gif-fps", type=int, default=TRITON_GIF_FPS, help="GIF playback frame rate"
    )
    p.add_argument(
        "--gif-max-frames",
        type=int,
        default=TRITON_GIF_MAX_FRAMES,
        help="evenly-strided timestep cap per GIF",
    )
    p.add_argument(
        "--dem",
        default=TRITON_MAP_DEM_TIF,
        help="DEM GeoTIFF used as the GIF hillshade background",
    )
    p.add_argument(
        "--perf-summary",
        default=TRITON_PERF_SUMMARY,
        help="TRITON performance.txt final per-rank summary",
    )
    p.add_argument(
        "--perf-dir",
        default=TRITON_PERF_DIR,
        help="directory of TRITON performanceN.txt per-step timing files",
    )
    p.add_argument(
        "--roff",
        default=TRITON_PERF_ROFF,
        help="TRITON .roff gridded runoff input (domain applied-runoff overlay)",
    )
    p.add_argument(
        "--series-dir",
        default=TRITON_MAP_SERIES_DIR,
        help="directory of TRITON stage time-series files (<name>_at_Xsec.txt)",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="overlay TRITON depth at TRITON_COMPARE_POINTS lat/lon against the observed USGS gauge stage",
    )
    p.add_argument(
        "--compare-out",
        default=TRITON_COMPARE_OUT_DIR,
        help="output directory for the compare_<gauge>.png hydrograph overlays",
    )
    return p.parse_args()


def _consolidate(
    var: str,
    producer: Callable[[int], np.ndarray],
    n_steps: int,
    grid: Dict,
    times,
    out_dir: Path,
    force: bool,
    remax: bool,
    mask: np.ndarray = None,
) -> None:
    """Stream *n_steps* slices from *producer* into ``<var>.nc`` and ``<var>_max.tif``.

    ``producer(t)`` returns a (ny, nx) float32 array with NaN at invalid cells;
    invalid cells are stored as NODATA in the netCDF and ignored in the max. When
    *mask* is given, cells outside it are also set to NODATA (watershed clip). When
    a cached ``<var>.nc`` already exists, the max GeoTIFF is derived from it rather
    than re-reading the gtiff series (unless *force* rebuilds the netCDF).
    """
    nc_path = out_dir / f"{var}.nc"
    tif_path = out_dir / f"{var}_max.tif"
    nc_cached = nc_path.exists() and nc_path.stat().st_size > 0

    if not force and nc_cached and (remax or not tif_path.exists()):
        running_max = readers.max_from_netcdf(nc_path, var, NODATA)
        if mask is not None:
            running_max = np.where(mask, running_max, np.float32(np.nan))
        writers.write_max_tiff(tif_path, running_max, grid, NODATA)
        log.info("Wrote %s from cached %s", tif_path.name, nc_path.name)
        return
    if not force and nc_cached and tif_path.exists():
        log.info("Skip (exists): %s, %s", nc_path.name, tif_path.name)
        return

    ny, nx = grid["ny"], grid["nx"]
    running_max = np.full((ny, nx), np.nan, dtype=np.float32)
    ds, data = writers.open_cube(nc_path, var, grid, times, NODATA)
    try:
        for t in range(n_steps):
            arr = producer(t)
            if mask is not None:
                arr = np.where(mask, arr, np.float32(np.nan))
            data[t, :, :] = np.where(np.isfinite(arr), arr, np.float32(NODATA))
            np.fmax(running_max, arr, out=running_max)
    finally:
        ds.close()
    writers.write_max_tiff(tif_path, running_max, grid, NODATA)
    log.info("Wrote %s (%d steps) + %s", nc_path.name, n_steps, tif_path.name)


def main() -> int:
    args = parse_args()
    gtiff = Path(args.gtiff_dir)
    out = Path(args.out_dir)

    if args.compare:
        h_nc = out / "H.nc"
        if not h_nc.exists():
            log.error("%s not found; run the default mod22 consolidation first.", h_nc)
            return 1
        dem = Path(args.dem)
        if not dem.exists():
            log.error("DEM not found: %s (needed for bed elevation).", dem)
            return 1
        start_date = args.start_date
        start_file = Path(args.start_file) if args.start_file else None
        if start_file and start_file.exists():
            start_date = start_file.read_text().strip()
        written = compare.run(
            TRITON_COMPARE_POINTS,
            h_nc,
            dem,
            TRITON_COMPARE_TZ,
            start_date,
            END_DATE,
            NODATA,
            Path(args.compare_out),
        )
        log.info("Compare: wrote %d plot(s) to %s", len(written), args.compare_out)
        return 0

    if not gtiff.is_dir():
        log.error("gtiff directory not found: %s", gtiff)
        return 1

    out.mkdir(parents=True, exist_ok=True)

    avail = readers.available_vars(gtiff)
    if not avail:
        log.error("No TRITON gtiff series (<VAR>_<NN>.vrt) found in %s", gtiff)
        return 1
    log.info(
        "Variables present: %s", ", ".join(f"{v}({len(p)})" for v, p in avail.items())
    )

    start_file = Path(args.start_file) if args.start_file else None
    if start_file and start_file.exists():
        args.start_date = start_file.read_text().strip()
        log.info(
            "Anchoring time axis to mod21-resolved sim start %s (from %s)",
            args.start_date,
            start_file.name,
        )

    interval = readers.parse_print_interval(Path(args.cfg))
    log.info(
        "Output cadence: print_interval=%ds, time anchored at %s",
        interval,
        args.start_date,
    )

    grid = readers.read_grid(next(iter(avail.values()))[0])
    log.info(
        "Grid: %d x %d cells, CRS %s (native, no reprojection)",
        grid["nx"],
        grid["ny"],
        grid["epsg"] or "unknown",
    )

    mask = None
    if args.clip:
        clip = Path(args.clip)
        if not clip.exists():
            log.error("Clip boundary not found: %s", clip)
            return 1
        mask = readers.build_watershed_mask(clip, grid)
        log.info(
            "Clipping maps to %s (%d/%d cells inside)",
            clip.name,
            int(mask.sum()),
            mask.size,
        )

    # H and MH: copied from the gtiff series, masking cells below the depth floor.
    min_depth = float(args.min_depth)
    if min_depth > 0 and not args.remax:
        log.info("Depth-map floor: masking H/MH cells < %.3f m to NODATA", min_depth)
    for var in ("H", "MH"):
        paths = avail.get(var)
        if not paths:
            log.warning("Variable %s absent; skipping.", var)
            continue
        times = readers.build_time_axis(len(paths), args.start_date, interval)

        def producer(t: int, _paths=paths) -> np.ndarray:
            arr = readers.read_slice(_paths[t], grid["nx"], grid["ny"])
            if min_depth > 0:
                arr = np.where(arr >= min_depth, arr, np.float32(np.nan))
            return arr

        _consolidate(
            var, producer, len(paths), grid, times, out, args.force, args.remax, mask
        )

    # V: derived from QX/QY unit discharge and H (only when QX and QY exist).
    if {"QX", "QY"}.issubset(avail) and "H" in avail:
        qx, qy, h = avail["QX"], avail["QY"], avail["H"]
        n = min(len(qx), len(qy), len(h))
        if not (len(qx) == len(qy) == len(h)):
            log.warning(
                "QX/QY/H step counts differ (%d/%d/%d); using first %d.",
                len(qx),
                len(qy),
                len(h),
                n,
            )
        times = readers.build_time_axis(n, args.start_date, interval)
        hmin = float(args.hmin)

        def producer_v(t: int) -> np.ndarray:
            vx = readers.read_slice(qx[t], grid["nx"], grid["ny"])
            vy = readers.read_slice(qy[t], grid["nx"], grid["ny"])
            depth = readers.read_slice(h[t], grid["nx"], grid["ny"])
            wet = depth >= hmin
            v = np.full_like(depth, np.nan)
            np.divide(np.hypot(vx, vy), depth, out=v, where=wet)
            return v

        _consolidate("V", producer_v, n, grid, times, out, args.force, args.remax, mask)
    else:
        log.warning("QX/QY (and H) not all present; skipping velocity (V).")

    dem = Path(args.dem)
    if not dem.exists():
        log.warning("GIF DEM not found: %s; skipping GIF animations.", dem)
    else:
        hillshade = readers.read_hillshade(dem, grid)
        boundary = (
            readers.read_boundary_line(Path(args.clip), grid) if args.clip else None
        )
        for var in TRITON_GIF_VARS:
            nc_path = out / f"{var}.nc"
            gif_path = out / f"{var}.gif"
            if not nc_path.exists():
                log.warning("%s not found; skipping GIF for %s.", nc_path.name, var)
                continue
            if gif_path.exists() and not args.force:
                log.info("Skip (exists): %s", gif_path.name)
                continue
            n_used = gifs.make_gif(
                nc_path,
                var,
                grid,
                hillshade,
                boundary,
                TRITON_GIF_CMAP[var],
                args.gif_fps,
                NODATA,
                gif_path,
                args.gif_max_frames,
            )
            log.info("Wrote %s (%d frames)", gif_path.name, n_used)

    summary_path = Path(args.perf_summary)
    perf_dir = Path(args.perf_dir)
    if not summary_path.exists() or not perf_dir.is_dir():
        log.warning(
            "TRITON performance logs not found (%s / %s); skipping performance diagnostics.",
            summary_path,
            perf_dir,
        )
    else:
        balance_png = out / "perf_load_balance.png"
        perf_plots.plot_load_balance(perf.read_summary(summary_path), balance_png)
        log.info("Wrote %s", balance_png.name)

        deltas = perf.step_deltas(perf.read_series(perf_dir))
        wet_nc = out / f"{TRITON_PERF_WET_VAR}.nc"
        wet = (
            perf.wet_stats(wet_nc, TRITON_PERF_WET_VAR, NODATA)
            if wet_nc.exists()
            else None
        )
        if wet is None:
            log.warning(
                "%s not found; per-step time series will omit the wet-cell overlay.",
                wet_nc.name,
            )
        roff_path = Path(args.roff)
        roff = perf.read_roff(roff_path, L1_CELL_SIZE_M) if roff_path.exists() else None
        if roff is None:
            log.warning(
                "%s not found; per-step time series will omit the applied-runoff overlay.",
                roff_path.name,
            )
        timeseries_png = out / "perf_timeseries.png"
        perf_plots.plot_timeseries(
            deltas,
            wet,
            timeseries_png,
            roff,
            start_date=args.start_date,
            interval_s=interval,
        )
        log.info("Wrote %s", timeseries_png.name)

    series_dir = Path(args.series_dir)
    series_files = series.discover_series(series_dir) if series_dir.is_dir() else []
    if not series_files:
        log.warning(
            "No TRITON stage time-series files found in %s; skipping stage hydrographs.",
            series_dir,
        )
    else:
        for sf in series_files:
            stage_png = out / f"stage_{sf.stem}.png"
            n_pts = series.plot_stage_hydrographs(sf, stage_png, args.start_date)
            log.info("Wrote %s (%d monitoring point(s))", stage_png.name, n_pts)

    log.info("Done. Maps written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
