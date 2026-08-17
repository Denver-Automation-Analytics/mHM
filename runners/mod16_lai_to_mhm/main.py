import os
import sys
import numpy as np
import geopandas as gpd
import logging
from pathlib import Path

import rioxarray  # noqa: F401  registers the .rio accessor
import netCDF4 as nc
from affine import Affine
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import L0_CELL_SIZE_M, OUTPUT_CRS, START_DATE, END_DATE, WORKING_DIR
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

from acquire_modis_lai import acquire_lai_map
from writers import write_lai_nc

log = logging.getLogger("mod16_lai_to_mhm")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _l0_geobox(l0_nc_path: str):
    """Return (affine_transform, (nrows, ncols)) of the canonical L0 grid."""
    with nc.Dataset(l0_nc_path) as ds:
        x = ds["x"][:].data
        y = ds["y"][:].data
    cs = float(x[1] - x[0])                 # +ve W->E step
    x_left = float(x[0]) - cs / 2.0
    y_top = float(y[0]) + cs / 2.0          # y is N->S (descending)
    transform = Affine(cs, 0.0, x_left, 0.0, -cs, y_top)
    return transform, (len(y), len(x))


def _snap_to_l0_grid(snapshot, l0_nc_path: str):
    """Regrid the LAI snapshot onto the exact L0 grid so it byte-aligns with mHM."""
    if snapshot.rio.crs is None:
        snapshot = snapshot.rio.write_crs(OUTPUT_CRS)
    transform, (nrows, ncols) = _l0_geobox(l0_nc_path)
    snapped = snapshot.rio.reproject(
        dst_crs=snapshot.rio.crs,
        transform=transform,
        shape=(nrows, ncols),
        resampling=Resampling.nearest,   # L0-aligned & same resolution: lossless
    )
    # reproject stamps a _FillValue attr; drop it so the writer owns encoding.
    for name in snapped.variables:
        snapped[name].attrs.pop("_FillValue", None)
        snapped[name].encoding.pop("_FillValue", None)
    log.info("Snapped LAI to L0 grid: %d x %d (ncols x nrows).", ncols, nrows)
    return snapped


def main(start_date: str,
         end_date: str,
         out_nc: str,
         chunks: int | None = None) -> int:

    WATERSHED_FILE = os.path.join(WORKING_DIR, "input/domain/watershed.geojson") # derived from mod10_dem_to_mhm
    if not os.path.exists(WATERSHED_FILE):
        raise FileNotFoundError(
            f"Watershed file {WATERSHED_FILE} not found. Run mod10_dem_to_mhm first."
        )
    
    gdf = gpd.read_file(WATERSHED_FILE)
    log.info("Loaded boundary '%s' (%d features, CRS=%s).",
             WATERSHED_FILE, len(gdf), gdf.crs)

    chunk_dict = {"x": chunks, "y": chunks} if chunks else None
    snapshot = acquire_lai_map(
        boundary=gdf,
        start_date=start_date,
        end_date=end_date,
        reducer="median",
        max_scf_qc=1,
        scale_m=L0_CELL_SIZE_M,
        output_crs=OUTPUT_CRS,
        chunks=chunk_dict,
        monthly=True,
    )

    try:
        mean_lai = float(np.nanmean(snapshot["Lai"].values))
        valid_frac = float(np.isfinite(snapshot["Lai"].values).mean())
        log.info("Area-mean LAI (m2/m2): %.3f  |  valid-pixel fraction: %.1f%%",
                 mean_lai, 100 * valid_frac)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not compute area-mean summary: %s", exc)

    # Align to the canonical L0 grid produced by mod10 so mHM's generated
    # x/y match the LAI NetCDF dimensions exactly.
    L0_MORPH_NC_PATH = os.path.join(WORKING_DIR, "input", "morph", "dem.nc")
    if not os.path.exists(L0_MORPH_NC_PATH):
        raise FileNotFoundError(
            f"Canonical L0 morph grid not found: {L0_MORPH_NC_PATH}. "
            "Run runners/mod10_dem_to_mhm first."
        )
    snapshot = _snap_to_l0_grid(snapshot, L0_MORPH_NC_PATH)

    write_lai_nc(snapshot, out_nc, ref_month=start_date)
    return 0


if __name__ == "__main__":

    main(START_DATE,
         END_DATE,
         out_nc=os.path.join(WORKING_DIR, "input", "lai", "lai.nc")
         )
