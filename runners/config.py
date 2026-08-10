"""Shared grid configuration for all runner modules."""
DOMAIN_FILE = "/workspace/test_domain_3/input/domain/huc4_1211.geojson"  # watershed boundary for all modules
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution in metres (mod10, mod13, mod14, mod15, mod16)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution in metres
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution in metres (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
START_DATE      = "2026-07-01"  # start date for all simulations
END_DATE        = "2026-07-31"  # end date for all simulations
N_OMP_THREADS  = 10     # OpenMP threads for mHM; requires binary built with -DCMAKE_WITH_OpenMP=ON
