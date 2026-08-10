"""Shared grid configuration for all runner modules. Edit here to change resolution or CRS."""
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution in metres (mod10, mod13, mod14)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution in metres
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution in metres (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
N_OMP_THREADS  = 4     # OpenMP threads for mHM; requires binary built with -DCMAKE_WITH_OpenMP=ON
