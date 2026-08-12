"""Shared configuration for all runner modules."""
WORKING_DIR = "/workspace/test_domain_3"  # working directory for all modules
DOMAIN_FILE = "/workspace/test_domain_3/input/domain/NuecesRvNrThreeRivers.geojson"  # watershed boundary for all modules
DOMAIN_BUFFER_M = 1000     # buffer [m] grown around the domain polygon when clipping/gridding inputs
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution (mod10, mod13, mod14, mod15, mod16)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution (i.e., mHM output resolution)
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
START_DATE      = "2016-10-01"  # start date for all simulations
END_DATE        = "2026-07-31"  # end date for all simulations
WARMUP_DAYS     = 0  # spin-up days consumed from the start of the forcing before the eval period
TIMESTEP        = "daily"  # "hourly" or "daily"; used by mod11, mod12, mod15, mod19
ROUTING_METHOD  = "muskingum"  # mRM routing (mod19): "muskingum" | "adaptive" | "adaptive_varying"
OPTI_OBJECTIVE  = "kge"  # mod18 calibration objective: "nse" | "lnnse" | "nse_lnnse" | "kge" | "multi_kge"
N_ITERATIONS   = 500     # mod19 DDS optimizer trials; more = better calibration, longer runtime
N_OMP_THREADS  = 10     # OpenMP threads for mHM; requires binary built with -DCMAKE_WITH_OpenMP=ON
PET_METHOD     = "penman_monteith"  # one of: "hargreaves_samani", "oudin", "priestley_taylor", "penman_monteith"
WANTED_GAUGE_IDS = ["08208000","08206600","08206700"] # "08210000",
NODATA = -9999
