"""Shared configuration for all runner modules."""
WORKING_DIR = "/workspace/test_domain_3"  # working directory for all modules
DOMAIN_FILE = "/workspace/test_domain_3/input/domain/huc4_1211.geojson"  # watershed boundary for all modules
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution (mod10, mod13, mod14, mod15, mod16)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution (i.e., mHM output resolution)
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
START_DATE      = "2014-10-01"  # start date for all simulations
END_DATE        = "2026-07-31"  # end date for all simulations
TIMESTEP        = "daily"  # "hourly" or "daily"; used by mod11, mod12, mod15, mod18
N_OMP_THREADS  = 20     # OpenMP threads for mHM; requires binary built with -DCMAKE_WITH_OpenMP=ON
PET_METHOD     = "penman_monteith"  # one of: "hargreaves_samani", "oudin", "priestley_taylor", "penman_monteith"
WANTED_GAUGE_IDS = ["08210000", "08200000", "08201500", "08204005", "08200720"]
NODATA = -9999