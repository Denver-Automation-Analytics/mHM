"""
USGS NWIS -> mHM streamflow gauge preparation.

Discovers USGS streamflow gauges strictly inside a user-supplied watershed
polygon, downloads daily or hourly discharge, converts ft^3/s -> m^3/s,
applies a quality-control policy, and writes:

  * one <local_id>.txt file per gauge in the mHM gauge-file format,
  * idgauges.asc  -- L0 raster with the local id burned into each gauge cell,
  * id_map.csv    -- mapping local_id <-> USGS site number + metadata.

If no gauges are found inside the watershed, the script prints a message and
exits cleanly with status 0.

mHM configuration expected:
    iFlag_cordinate_sys = 0     (projected LCC meters)
    L0 cellsize        = 300 m  (shared with land-cover module)
    Discharge units    : m^3/s
    NODATA_value       : -9999
"""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import OUTPUT_CRS

import geopandas as gpd

from hyriver      import get_usgs_stations, get_nwis, aggregate_to_hourly, interpolate_gaps
from mhm_format   import to_m3s, filter_by_qualifiers, write_gauge_file
from idgauges     import build_idgauges_grid, write_id_map
from writers      import write_nc
from utils        import load_header_from_nc, write_header_txt, read_projection_wkt

# Load env from repo root explicitly so runs from any CWD behave the same.
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)

# ---- USER INPUTS ---------------------------------------------------
WATERSHED_PATH        = "/workspace/test_domain_3/input/domain/huc4_1211.geojson"
OUTPUT_DIR            = "/workspace/test_domain_3/input/gauge"

# --- L0 grid (derived from morph/dem.nc produced by mod10) ---------
L0_MORPH_NC_PATH      = "/workspace/test_domain_3/input/morph/dem.nc"
TARGET_CRS_WKT        = OUTPUT_CRS
LATLON_NC_PATH        = "/workspace/test_domain_3/input/latlon/latlon.nc"

# --- Time & cadence ------------------------------------------------
CADENCE               = "hourly"    # "daily" or "hourly"
START_DATE            = "2024-09-01"
END_DATE              = "2024-09-30"

# --- Retrieval knobs -----------------------------------------------
SITE_TYPE_CODE        = "ST"       # Stream sites
PARAMETER_CODE        = "00060"    # Discharge, m^3/s
MIN_RECORD_YEARS      = 1          # skip gauges with valid record shorter than this
QUALIFIER_POLICY      = "keep_approved_only"   # or "keep_approved_provisional" or "keep_all"
CHUNK_YEARS           = 1          # break continuous requests into 1-yr chunks
MAX_WORKERS           = 4          # parallel per-gauge network calls

# --- Constants -----------------------------------------------------
NODATA                = -9999
MAX_GAP_HOURS         = None       # None -> interpolate all gaps; int to cap gap size
LOG_LEVEL             = logging.INFO


# --------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Avoid leaking api_key query values in verbose request URL logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("hydro")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. L0 header + target CRS.
    l0_header  = load_header_from_nc(L0_MORPH_NC_PATH)
    target_crs = TARGET_CRS_WKT
    log.info("L0 grid: %d x %d cells @ %s m", l0_header["ncols"],
             l0_header["nrows"], l0_header["cellsize"])

    # 2. Discover gauges strictly inside the watershed polygon.
    model_perimeter = gpd.read_file(WATERSHED_PATH).to_crs("EPSG:4326")
    gauges_in = get_usgs_stations(
        model_perimeter=model_perimeter,
        variable_type="flow",
        dates=(START_DATE, END_DATE),
    )

    if gauges_in.empty:
        log.warning(
            "No USGS streamflow gauges found strictly inside %s. "
            "Nothing to write.", WATERSHED_PATH,
        )
        print(f"[hydro] No gauges found inside {WATERSHED_PATH}. Exiting.")
        sys.exit(0)

    log.info("%d gauge(s) strictly inside watershed.", len(gauges_in))

    # 4. Fetch iv, aggregate to hourly, QC-filter, convert units, interpolate gaps.
    survivors = []          # list of (site_no, name, lat, lon, series_df)
    for _, row in gauges_in.iterrows():
        site_no = row["site_no"]
        name    = row.get("station_nm", row.get("name", str(site_no)))
        raw = get_nwis(
            site       = site_no,
            parameter  = "Flow",
            frequency  = "iv",
            start_date = START_DATE,
            end_date   = END_DATE,
        )
        if raw is None:
            log.warning("Skipping %s (%s): fetch returned None.", site_no, name)
            continue

        hourly = aggregate_to_hourly(raw)
        if hourly.empty:
            log.warning("Skipping %s (%s): no data after hourly aggregation.", site_no, name)
            continue

        clean = filter_by_qualifiers(hourly, policy=QUALIFIER_POLICY, nodata=NODATA)
        clean = to_m3s(clean, nodata=NODATA)
        clean = interpolate_gaps(clean, nodata=NODATA, max_gap_hours=MAX_GAP_HOURS)

        valid = clean.loc[clean["value"] != NODATA]
        if valid.empty:
            log.warning("Skipping %s (%s): no valid values after QC.", site_no, name)
            continue

        # span_yrs = (valid.index.max() - valid.index.min()).days / 365.25
        # if span_yrs < MIN_RECORD_YEARS:
        #     log.warning("Skipping %s (%s): record span %.2f yr < %d yr threshold.",
        #                 site_no, name, span_yrs, MIN_RECORD_YEARS)
        #     continue

        survivors.append({
            "site_no": site_no,
            "name":    name,
            "lat":     float(row["dec_lat_va"]),
            "lon":     float(row["dec_long_va"]),
            "series":  clean,
        })

    if not survivors:
        log.warning("All discovered gauges failed retrieval or QC. Nothing to write.")
        print("[hydro] No gauges survived QC. Exiting.")
        sys.exit(0)

    # 5. Assign sequential local IDs and write per-gauge files.
    for local_id, g in enumerate(survivors, start=1):
        g["local_id"] = local_id
        write_gauge_file(
            path         = out_root / f"{local_id}.txt",
            local_id     = local_id,
            site_no      = g["site_no"],
            name         = g["name"],
            series       = g["series"],
            cadence      = CADENCE,
            nodata       = NODATA,
        )

    # 6. Build and write idgauges.asc + id_map.csv + header.txt.
    grid = build_idgauges_grid(
        survivors      = survivors,
        l0_header      = l0_header,
        target_crs_wkt = target_crs,
        nodata         = NODATA,
    )
    write_nc(out_root / "idgauges.nc", l0_header, grid, nodata=NODATA)
    write_header_txt(l0_header, out_root / "header.txt")
    write_id_map(out_root / "id_map.csv", survivors,
                 start=START_DATE, end=END_DATE, cadence=CADENCE)

    log.info("Done. %d gauge(s) written to %s", len(survivors), out_root)


if __name__ == "__main__":
    main()