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
from config import (NODATA, OUTPUT_CRS, TIMESTEP, WORKING_DIR,
                    TRITON_BF_CHANNEL_KM2, TRITON_BF_SLOPE_MIN, TRITON_BF_WIDTH_A,
                    TRITON_BF_WIDTH_B, TRITON_CONST_MANN, TRITON_DEM_CELLSIZE_M,
                    TRITON_DOMAIN_NAME, TRITON_END_DATE, TRITON_EXTBC_SEG_LEN_M,
                    TRITON_EXTBC_TYPE, TRITON_EXTBC_VALUE, TRITON_INITH,
                    TRITON_MANN_SOURCE, TRITON_OUT_DIR, TRITON_PRINT_INTERVAL_S,
                    TRITON_PROJECTION, TRITON_START_DATE)

from osgeo import gdal

import grid as gridmod
import readers
import writers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mhm_to_triton")

WATERSHED_FILE = os.path.join(WORKING_DIR, "input", "domain", "watershed.geojson")
if not os.path.exists(WATERSHED_FILE):
    raise FileNotFoundError(f"Watershed file not found: {WATERSHED_FILE}. Run mod10 first to generate it.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate TRITON inputs from mHM/mRM data.")
    p.add_argument("--work-dir", default=WORKING_DIR, help="mHM domain directory")
    p.add_argument("--out-dir", default=TRITON_OUT_DIR, help="output directory for TRITON inputs")
    p.add_argument("--name", default=TRITON_DOMAIN_NAME, help="basename for the TRITON files")
    p.add_argument("--cellsize", type=float, default=TRITON_DEM_CELLSIZE_M, help="TRITON grid resolution [m]")
    p.add_argument("--timestep", default=TIMESTEP, choices=["daily", "hourly"], help="mHM output cadence")
    p.add_argument("--start-date", default=TRITON_START_DATE, help="event-window start 'YYYY-MM-DD' (subset the runoff)")
    p.add_argument("--end-date", default=TRITON_END_DATE, help="event-window end 'YYYY-MM-DD' (inclusive)")
    p.add_argument("--force", action="store_true", help="regenerate outputs even if they already exist")
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

    keys = ("dem", "rmap", "roff", "extbc", "obs", "mann", "inith", "initqx", "inityq", "cfg")
    paths = {k: out / f"{name}.{k}" for k in keys}
    names = {"domain": name, **{k: f"{name}.{k}" for k in keys if k != "cfg"}}
    epsg = OUTPUT_CRS.split(":")[-1]

    def skip(key: str) -> bool:
        p = paths[key]
        if p.exists() and p.stat().st_size > 0 and not args.force:
            log.info("Skip (exists): %s", p.name)
            return True
        return False

    step_h = readers.step_hours(args.timestep)
    log.info("Reading mHM runoff cube: %s", flux_nc)
    runoff = readers.read_runoff(flux_nc, start=args.start_date, end=args.end_date)
    log.info("Runoff units=%r, %d steps (%s..%s), L1 %dx%d @ %.0f m",
             runoff["units"], runoff["time"].size,
             str(runoff["time"][0])[:10], str(runoff["time"][-1])[:10],
             runoff["easting"].size, runoff["northing"].size, runoff["cellsize"])

    warped_tif = out / f"{name}_dem_{epsg}.tif"
    if warped_tif.exists() and not args.force:
        log.info("Reusing warped DEM: %s", warped_tif.name)
        dem_grid = gridmod.read_grid(warped_tif)
    else:
        log.info("Reprojecting/clipping DEM to %s @ %.0f m", OUTPUT_CRS, args.cellsize)
        dem_grid = gridmod.warp_dem(src_dem, Path(WATERSHED_FILE), OUTPUT_CRS, args.cellsize, warped_tif)
    log.info("TRITON grid: %d x %d = %.1fM cells @ %.0f m",
             dem_grid["ncols"], dem_grid["nrows"],
             dem_grid["ncols"] * dem_grid["nrows"] / 1e6, dem_grid["cellsize"])

    zone_ids, valid, n_zones = gridmod.build_zones(runoff)
    ix1, iy1 = gridmod.dem_axis_to_l1(dem_grid, runoff)
    n_rows = runoff["time"].size
    log.info("Runoff zones (num_runoffs): %d", n_zones)

    if not (skip("dem") and skip("rmap")):
        log.info("Writing DEM + runoff map ...")
        writers.write_dem_and_rmap(dem_grid, zone_ids, ix1, iy1, paths["dem"], paths["rmap"])

    if not skip("roff"):
        log.info("Writing runoff time series ...")
        n_rows = writers.write_roff(runoff, valid, step_h, paths["roff"])

    outlet = readers.find_outlet(facc_nc)
    log.info("Outlet (max facc) at x=%.1f y=%.1f", *outlet)
    num_extbc = 1
    if not skip("extbc"):
        num_extbc = writers.write_extbc(dem_grid, outlet, TRITON_EXTBC_TYPE,
                                        TRITON_EXTBC_VALUE, TRITON_EXTBC_SEG_LEN_M, paths["extbc"])

    gauges = readers.read_gauges(id_map)
    n_obs = len(gauges)
    if not skip("obs"):
        n_obs = writers.write_obs(gauges, paths["obs"])
    log.info("Observation point(s): %d", n_obs)

    lc_cache: dict = {}

    def ensure_lc() -> dict:
        """Warp the land-cover raster onto the DEM grid once; cache lut + tif."""
        if not lc_cache:
            lut, nod = readers.read_manning_lookup(Path(TRITON_MANN_SOURCE))
            lc_tif = out / f"{name}_lc_{epsg}.tif"
            if not lc_tif.exists() or args.force:
                gridmod.warp_to_dem_grid(Path(TRITON_MANN_SOURCE), Path(WATERSHED_FILE),
                                         OUTPUT_CRS, dem_grid, lc_tif, resample="near",
                                         dst_nodata=nod if nod is not None else -128)
            lc_cache.update(lut=lut, nod=nod, tif=lc_tif)
        return lc_cache

    if not skip("mann"):
        log.info("Generating Manning field from %s", TRITON_MANN_SOURCE)
        lc = ensure_lc()
        writers.write_mann(lc["tif"], dem_grid, lc["lut"], TRITON_CONST_MANN, lc["nod"], paths["mann"])

    init_names = None
    if TRITON_INITH:
        ic_keys = ("inith", "initqx", "inityq")
        ic_files = [paths[k] for k in ic_keys] + [Path(f"{paths[k]}.tif") for k in ic_keys]
        ready = all(p.exists() and p.stat().st_size > 0 for p in ic_files)
        facc_hi = work / "input" / "dem" / "facc.tif"   # full-resolution routing (mod10)
        fdir_hi = work / "input" / "dem" / "fdir.tif"
        if ready and not args.force:
            for k in ic_keys:
                log.info("Skip (exists): %s", paths[k].name)
        else:
            if not (facc_hi.exists() and fdir_hi.exists()):
                log.error("Initial conditions require full-resolution routing rasters "
                          "%s and %s. Run mod10 to generate them.", facc_hi, fdir_hi)
                return 1
            log.info("Seeding channels from pre-event baseflow (Method A) ...")
            lc = ensure_lc()
            # slope at the TRITON resolution, derived from the reprojected DEM
            slope_g = out / f"{name}_slope_{epsg}.tif"
            if not slope_g.exists() or args.force:
                gdal.DEMProcessing(str(slope_g), str(dem_grid["tif"]), "slope", slopeFormat="degree")
            log.info("Using full-resolution fdir/facc from input/dem/ (reprojected to the TRITON grid)")
            cell_area = gridmod.pixel_area_m2(facc_hi)
            facc_g = gridmod.warp_to_dem_grid(facc_hi, Path(WATERSHED_FILE), OUTPUT_CRS,
                        dem_grid, out / f"{name}_facc_{epsg}.tif", resample="near", dst_nodata=NODATA)
            fdir_g = gridmod.warp_to_dem_grid(fdir_hi, Path(WATERSHED_FILE), OUTPUT_CRS,
                        dem_grid, out / f"{name}_fdir_{epsg}.tif", resample="near", dst_nodata=0)
            bf = readers.read_baseflow_preevent(flux_nc, start_date=args.start_date)
            log.info("Pre-event baseflow step: %s | source cell area %.1f m2", str(bf["time"])[:16], cell_area)
            ic = writers.write_initial_conditions(
                dem_grid, facc_g, Path(slope_g), fdir_g, lc["tif"], lc["lut"], TRITON_CONST_MANN,
                bf["rate"], ix1, iy1, cell_area, TRITON_BF_CHANNEL_KM2,
                TRITON_BF_WIDTH_A, TRITON_BF_WIDTH_B, TRITON_BF_SLOPE_MIN,
                paths["inith"], paths["initqx"], paths["inityq"])
            log.info("Initial-condition channel cells: %d", ic["n_chan"])
            log.info("  depth h [m]      : min=%.3f mean=%.3f median=%.3f p90=%.3f max=%.3f",
                     *(ic["depth_m"][s] for s in ("min", "mean", "median", "p90", "max")))
            log.info("  unit q |q| [m2/s]: min=%.4f mean=%.4f median=%.4f p90=%.4f max=%.4f",
                     *(ic["unit_q_m2s"][s] for s in ("min", "mean", "median", "p90", "max")))
            log.info("  velocity [m/s]   : min=%.3f mean=%.3f median=%.3f p90=%.3f max=%.3f",
                     *(ic["velocity_ms"][s] for s in ("min", "mean", "median", "p90", "max")))
            log.info("  GeoTIFFs: %s", ", ".join(t.name for t in ic["tifs"].values()))
        if all(paths[k].exists() for k in ic_keys):
            init_names = {"h": names["inith"], "qx": names["initqx"], "qy": names["inityq"]}

    sim_duration_s = int(n_rows * step_h * 3600)
    if not skip("cfg"):
        writers.write_cfg(paths["cfg"], names, TRITON_PROJECTION, n_zones, n_rows,
                          num_extbc, sim_duration_s, TRITON_PRINT_INTERVAL_S, TRITON_CONST_MANN,
                          init_names=init_names)

    # cross-file consistency checks
    assert n_zones == int(zone_ids.max()), "zone id range does not match num_runoffs"
    log.info("Done. TRITON inputs written to %s", out)
    log.info("  num_runoffs=%d runoff_row_size=%d num_extbc=%d obs=%d sim_duration=%ds",
             n_zones, n_rows, num_extbc, n_obs, sim_duration_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
