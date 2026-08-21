"""mHM/mRM -> TRITON input generation (mod22).

Converts the prepared mHM inputs and outputs of a domain into the ASCII inputs
that drive the TRITON 2D hydraulic model:

  <name>.dem    reprojected/clipped/resampled mod10 hydro-corrected DEM
  <name>.rmap   runoff-zone id per TRITON cell (one zone per mHM L1 cell)
  <name>.roff   gridded runoff time series [mm/hr] per zone (from mHM Q)
  <name>.extbc  single open outlet on the downstream grid edge (default: normal-slope / Manning)
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
from config import (NODATA, OUTPUT_CRS, TIMESTEP, TRITON_HYDROGRAPH_INTERVAL_S, WORKING_DIR,
                    IO_MANNING_N,
                    TRITON_BF_CHANNEL_KM2, TRITON_BF_SLOPE_MIN, TRITON_BF_WIDTH_A,
                    TRITON_BF_WIDTH_B, TRITON_CHANNEL_MANN, TRITON_CONST_MANN,
                    TRITON_DECOMP_FACTOR,
                    TRITON_DECOMP_TYPE,
                    TRITON_COURANT,
                    TRITON_DEM_CELLSIZE_M, TRITON_DOMAIN_NAME, TRITON_END_DATE,
                    TRITON_EXTBC_TYPE, TRITON_EXTBC_VALUE,
                    TRITON_INIT_FILL, TRITON_INIT_FILL_MAX_H,
                    TRITON_IO_LULC_PATH, TRITON_IO_LULC_YEAR,
                    TRITON_MAP_HMIN,
                    TRITON_OUT_DIR,
                    TRITON_MAPPING_INTERVAL_S, TRITON_PROJECTION, TRITON_START_DATE,
                    TRITON_WARM_START,
                    TRITON_WATERBODIES, TRITON_WATERBODY_LAYER_ID,
                    TRITON_WATERBODY_EXCLUDE_FTYPES, TRITON_WATERBODY_LAKEPOND_STAGE,
                    TRITON_WATERBODY_PATH, TRITON_WATERBODY_SERVICE_URL)

from osgeo import gdal

import esri_lulc
import grid as gridmod
import nhd_waterbodies
import readers
import writers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mhm_to_triton")

WATERSHED_FILE = os.path.join(WORKING_DIR, "mhm_input", "domain", "watershed.geojson")
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

    flux_nc = work / "mhm_output" / "mHM_Fluxes_States.nc"
    src_dem = work / "mhm_input" / "dem" / "dem_corrected.tif"
    facc_nc = work / "mhm_input" / "morph" / "facc.nc"
    id_map = work / "mhm_input" / "gauge" / "id_map.csv"
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

    num_extbc = 1
    if not skip("extbc"):
        outlet = readers.find_outlet(facc_nc)
        num_extbc, outlet_edge = writers.write_extbc(dem_grid, TRITON_EXTBC_TYPE,
                                                     TRITON_EXTBC_VALUE, paths["extbc"], outlet)
        log.info("Downstream-only outlet boundary on the %s edge (num_extbc=%d)", outlet_edge, num_extbc)

    gauges = readers.read_gauges(id_map)
    n_obs = len(gauges)
    if not skip("obs"):
        n_obs = writers.write_obs(gauges, paths["obs"])
    log.info("Observation point(s): %d", n_obs)

    facc_hi = work / "mhm_input" / "dem" / "facc.tif"   # full-resolution routing (mod10)
    fdir_hi = work / "mhm_input" / "dem" / "fdir.tif"
    lc_cache: dict = {}
    facc_g_cache: dict = {}
    wb_cache: dict = {}

    def ensure_lc() -> dict:
        """Warp the land-cover raster onto the DEM grid once; cache lut + tif."""
        if not lc_cache:
            src = Path(TRITON_IO_LULC_PATH)
            if not src.exists() or args.force:
                log.info("Acquiring ESRI/IO 10 m land cover -> %s", src)
                esri_lulc.acquire(Path(WATERSHED_FILE), OUTPUT_CRS, src,
                                  year=TRITON_IO_LULC_YEAR, resolution=TRITON_DEM_CELLSIZE_M)
            lut, nod = readers.read_manning_lookup(src, IO_MANNING_N)
            lc_tif = out / f"{name}_lc_{epsg}.tif"
            if not lc_tif.exists() or args.force:
                gridmod.warp_to_dem_grid(src, Path(WATERSHED_FILE),
                                         OUTPUT_CRS, dem_grid, lc_tif, resample="near",
                                         dst_nodata=nod if nod is not None else -128)
            lc_cache.update(lut=lut, nod=nod, tif=lc_tif)
        return lc_cache

    def ensure_facc_g() -> dict:
        """Warp full-res facc onto the DEM grid once; cache path + source pixel area."""
        if not facc_g_cache:
            fg = out / f"{name}_facc_{epsg}.tif"
            if not fg.exists() or args.force:
                gridmod.warp_to_dem_grid(facc_hi, Path(WATERSHED_FILE), OUTPUT_CRS,
                                         dem_grid, fg, resample="near", dst_nodata=NODATA)
            facc_g_cache.update(path=fg, cell_area=gridmod.pixel_area_m2(facc_hi))
        return facc_g_cache

    def ensure_waterbodies() -> dict:
        """Acquire waterbodies and rasterize them to the DEM grid (open-water .mann mask)."""
        if not wb_cache:
            vec = Path(TRITON_WATERBODY_PATH)
            if not vec.exists() or args.force:
                log.info("Acquiring NHDPlus HR waterbodies -> %s", vec)
                nhd_waterbodies.acquire(Path(WATERSHED_FILE), OUTPUT_CRS, vec,
                                        TRITON_WATERBODY_SERVICE_URL, TRITON_WATERBODY_LAYER_ID,
                                        exclude_ftypes=TRITON_WATERBODY_EXCLUDE_FTYPES,
                                        lakepond_stage=TRITON_WATERBODY_LAKEPOND_STAGE)
            mask_tif = out / f"{name}_wb_{epsg}.tif"
            if not mask_tif.exists() or args.force:
                gridmod.rasterize_waterbodies_to_dem_grid(vec, dem_grid, mask_tif)
            wb_cache.update(mask=mask_tif)
        return wb_cache

    # Channel roughness is burned into .mann using the same facc channel network.
    burn_channels = TRITON_CHANNEL_MANN is not None
    use_wb = TRITON_WATERBODIES

    mann_tif = Path(f'{paths["mann"]}.tif')
    mann_ready = paths["mann"].exists() and paths["mann"].stat().st_size > 0 and mann_tif.exists()
    if mann_ready and not args.force:
        log.info("Skip (exists): %s", paths["mann"].name)
    else:
        log.info("Generating Manning field from %s", TRITON_IO_LULC_PATH)
        lc = ensure_lc()
        burn_kwargs: dict = {}
        if burn_channels:
            if not facc_hi.exists():
                log.error("Channel roughness burn requires %s. Run mod10 to generate it.", facc_hi)
                return 1
            fg = ensure_facc_g()
            burn_kwargs = dict(channel_facc_tif=fg["path"], channel_km2=TRITON_BF_CHANNEL_KM2,
                               cell_area_m2=fg["cell_area"], channel_n=TRITON_CHANNEL_MANN)
        if use_wb:
            burn_kwargs.update(waterbody_mask_tif=ensure_waterbodies()["mask"],
                               water_n=TRITON_CHANNEL_MANN)
        n_burn = writers.write_mann(lc["tif"], dem_grid, lc["lut"], TRITON_CONST_MANN,
                                    lc["nod"], paths["mann"], **burn_kwargs)
        log.info("Manning GeoTIFF: %s", mann_tif.name)
        if burn_channels:
            log.info("Burned channel roughness n=%.3f into %d cell(s) of %s",
                     TRITON_CHANNEL_MANN, n_burn, paths["mann"].name)

    init_names = None
    if TRITON_WARM_START:
        ic_keys = ("inith", "initqx", "inityq")
        ic_files = [paths[k] for k in ic_keys] + [Path(f"{paths[k]}.tif") for k in ic_keys]
        ready = all(p.exists() and p.stat().st_size > 0 for p in ic_files)
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
            log.info("Using full-resolution fdir/facc from mhm_input/dem/ (reprojected to the TRITON grid)")
            fg = ensure_facc_g()
            cell_area = fg["cell_area"]
            facc_g = fg["path"]
            fdir_g = gridmod.warp_to_dem_grid(fdir_hi, Path(WATERSHED_FILE), OUTPUT_CRS,
                        dem_grid, out / f"{name}_fdir_{epsg}.tif", resample="near", dst_nodata=0)
            bf = readers.read_baseflow_preevent(flux_nc, start_date=args.start_date)
            log.info("Pre-event baseflow step: %s | source cell area %.1f m2", str(bf["time"])[:16], cell_area)
            ic = writers.write_initial_conditions(
                dem_grid, facc_g, Path(slope_g), fdir_g, lc["tif"], lc["lut"], TRITON_CONST_MANN,
                bf["rate"], ix1, iy1, cell_area, TRITON_BF_CHANNEL_KM2,
                TRITON_BF_WIDTH_A, TRITON_BF_WIDTH_B, TRITON_BF_SLOPE_MIN,
                paths["inith"], paths["initqx"], paths["inityq"],
                do_fill=TRITON_INIT_FILL, fill_max_h=TRITON_INIT_FILL_MAX_H,
                min_h=TRITON_MAP_HMIN)
            log.info("Initial-condition channel seed cells: %d", ic["n_chan"])
            if TRITON_INIT_FILL:
                log.info("Filled channel-storage cells: %d (cap %.1f m)", ic["n_fill"], TRITON_INIT_FILL_MAX_H)
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
        writers.write_cfg(cfg_path=paths["cfg"],
                          names=names,
                          projection=TRITON_PROJECTION,
                          num_runoffs=n_zones,
                          runoff_rows=n_rows,
                          num_extbc=num_extbc,
                          sim_duration_s=sim_duration_s,
                          mapping_interval_s=TRITON_MAPPING_INTERVAL_S,
                          obs_interval_s=TRITON_HYDROGRAPH_INTERVAL_S,
                          const_mann=TRITON_CONST_MANN,
                          decomp_factor=TRITON_DECOMP_FACTOR,
                          decomp_type=TRITON_DECOMP_TYPE,
                          courant=TRITON_COURANT,
                          init_names=init_names,
                          )

    # cross-file consistency checks
    assert n_zones == int(zone_ids.max()), "zone id range does not match num_runoffs"
    log.info("Done. TRITON inputs written to %s", out)
    log.info("  num_runoffs=%d runoff_row_size=%d num_extbc=%d obs=%d sim_duration=%ds",
             n_zones, n_rows, num_extbc, n_obs, sim_duration_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
