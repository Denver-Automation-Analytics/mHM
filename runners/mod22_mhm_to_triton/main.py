"""mHM/mRM -> TRITON input generation (mod22).

Converts the prepared mHM inputs and outputs of a domain into the ASCII inputs
that drive the TRITON 2D hydraulic model:

  <name>.dem    reprojected/clipped/resampled mod10 hydro-corrected DEM
  <name>.rmap   runoff-zone id per TRITON cell (one zone per mHM L1 cell)
  <name>.roff   gridded runoff time series [mm/hr] per zone (from mHM Q)
  <name>.extbc  outlet open-boundary segment (default: normal-slope / Manning)
  <name>.obs    observation points at the mHM gauges
  <name>.cfg    TRITON configuration referencing the above

Run order: mod10 (DEM) + mod15 (gauges) + an mHM run producing gridded Q -> mod22.
The Manning field (.mann) is intentionally not generated yet.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (DOMAIN_FILE, OUTPUT_CRS, TIMESTEP, WORKING_DIR,
                    TRITON_CONST_MANN, TRITON_DEM_CELLSIZE_M, TRITON_DOMAIN_NAME,
                    TRITON_EXTBC_SEG_LEN_M, TRITON_EXTBC_TYPE, TRITON_EXTBC_VALUE,
                    TRITON_OUT_DIR, TRITON_PRINT_INTERVAL_S, TRITON_PROJECTION)

import grid as gridmod
import readers
import writers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mhm_to_triton")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate TRITON inputs from mHM/mRM data.")
    p.add_argument("--work-dir", default=WORKING_DIR, help="mHM domain directory")
    p.add_argument("--out-dir", default=TRITON_OUT_DIR, help="output directory for TRITON inputs")
    p.add_argument("--name", default=TRITON_DOMAIN_NAME, help="basename for the TRITON files")
    p.add_argument("--cellsize", type=float, default=TRITON_DEM_CELLSIZE_M, help="TRITON grid resolution [m]")
    p.add_argument("--timestep", default=TIMESTEP, choices=["daily", "hourly"], help="mHM output cadence")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    work = Path(args.work_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name

    flux_nc = work / "output" / "mHM_Fluxes_States.nc"
    src_dem = work / "input" / "dem" / "dem_corrected.tif"
    facc_nc = work / "input" / "morph" / "facc.nc"
    id_map = work / "input" / "gauge" / "id_map.csv"
    for pth in (flux_nc, src_dem, facc_nc, id_map):
        if not pth.exists():
            log.error("Required input missing: %s", pth)
            return 1

    paths = {k: out / f"{name}.{k}" for k in ("dem", "rmap", "roff", "extbc", "obs", "cfg")}
    names = {"domain": name, **{k: f"{name}.{k}" for k in ("dem", "rmap", "roff", "extbc", "obs")}}

    step_h = readers.step_hours(args.timestep)
    log.info("Reading mHM runoff cube: %s", flux_nc)
    runoff = readers.read_runoff(flux_nc)
    log.info("Runoff units=%r, %d steps, L1 %dx%d @ %.0f m",
             runoff["units"], runoff["time"].size,
             runoff["easting"].size, runoff["northing"].size, runoff["cellsize"])

    log.info("Reprojecting/clipping DEM to %s @ %.0f m", OUTPUT_CRS, args.cellsize)
    warped_tif = out / f"{name}_dem_{OUTPUT_CRS.split(':')[-1]}.tif"
    dem_grid = gridmod.warp_dem(src_dem, Path(DOMAIN_FILE), OUTPUT_CRS, args.cellsize, warped_tif)
    log.info("TRITON grid: %d x %d = %.1fM cells @ %.0f m",
             dem_grid["ncols"], dem_grid["nrows"],
             dem_grid["ncols"] * dem_grid["nrows"] / 1e6, dem_grid["cellsize"])

    zone_ids, valid, n_zones = gridmod.build_zones(runoff)
    ix1, iy1 = gridmod.dem_axis_to_l1(dem_grid, runoff)
    log.info("Runoff zones (num_runoffs): %d", n_zones)

    log.info("Writing DEM + runoff map ...")
    writers.write_dem_and_rmap(dem_grid, zone_ids, ix1, iy1, paths["dem"], paths["rmap"])
    log.info("Writing runoff time series ...")
    n_rows = writers.write_roff(runoff, valid, step_h, paths["roff"])

    outlet = readers.find_outlet(facc_nc)
    log.info("Outlet (max facc) at x=%.1f y=%.1f", *outlet)
    num_extbc = writers.write_extbc(dem_grid, outlet, TRITON_EXTBC_TYPE,
                                    TRITON_EXTBC_VALUE, TRITON_EXTBC_SEG_LEN_M, paths["extbc"])

    gauges = readers.read_gauges(id_map)
    n_obs = writers.write_obs(gauges, paths["obs"])
    log.info("Wrote %d observation point(s)", n_obs)

    sim_duration_s = int(n_rows * step_h * 3600)
    writers.write_cfg(paths["cfg"], names, TRITON_PROJECTION, n_zones, n_rows,
                      num_extbc, sim_duration_s, TRITON_PRINT_INTERVAL_S, TRITON_CONST_MANN)

    # cross-file consistency checks
    assert n_zones == int(zone_ids.max()), "zone id range does not match num_runoffs"
    log.info("Done. TRITON inputs written to %s", out)
    log.info("  num_runoffs=%d runoff_row_size=%d num_extbc=%d obs=%d sim_duration=%ds",
             n_zones, n_rows, num_extbc, n_obs, sim_duration_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
