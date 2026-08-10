

import sys
import numpy as np
import geopandas as gpd
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import L0_CELL_SIZE_M, OUTPUT_CRS

from acquire_modis_lai import acquire_lai_map
from writers import write_lai_nc

log = logging.getLogger("mod16_lai_to_mhm")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def main(boundary: str,
         start_date: str,
         end_date: str,
         out_nc: str,
         chunks: int | None = None) -> int:
    gdf = gpd.read_file(boundary)
    log.info("Loaded boundary '%s' (%d features, CRS=%s).",
             boundary, len(gdf), gdf.crs)

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
    )

    try:
        mean_lai = float(np.nanmean(snapshot["Lai"].values))
        valid_frac = float(np.isfinite(snapshot["Lai"].values).mean())
        log.info("Area-mean LAI (m2/m2): %.3f  |  valid-pixel fraction: %.1f%%",
                 mean_lai, 100 * valid_frac)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not compute area-mean summary: %s", exc)

    write_lai_nc(snapshot, out_nc, ref_month=start_date)
    return 0


if __name__ == "__main__":

    DOMAIN_FILE = "/workspace/test_domain_3/input/domain/huc4_1211.geojson"
    START_DATE = "2024-09-01"
    END_DATE = "2024-09-30"

    sys.exit(main(DOMAIN_FILE,
                  START_DATE,
                  END_DATE,
                  out_nc="/workspace/test_domain_3/input/lai/lai.nc"))