"""Shared configuration for all runner modules."""
WORKING_DIR = "/workspace/test_domain_3"  # working directory for all modules
DOMAIN_FILE = "/workspace/test_domain_3/input/domain/NuecesRvNrThreeRivers.geojson"  # watershed boundary for all modules
DOMAIN_BUFFER_M = 1000     # buffer [m] grown around the domain polygon when clipping/gridding inputs
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution (mod10, mod13, mod14, mod15, mod16)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution (i.e., mHM output resolution)
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
START_DATE      = "2017-10-01"  # forcing/simulation start; earliest date mHM may draw spin-up from
END_DATE        = "2020-10-01"  # forcing/simulation and evaluation end. NOTE: keep the sim span <=2 yrs from START_DATE; longer spans trigger an mHM large-domain meteo-zeroing bug (3 yr=48 zero-PET days, 10 yr=312 + overflow)
EVAL_START_DATE = "2018-10-01"  # calibration scoring starts here; keep FIXED so warm-up length changes don't move the scored window
WARMUP_DAYS     = 0  # spin-up days drawn from forcing in [EVAL_START_DATE - WARMUP_DAYS, EVAL_START_DATE); capped by START_DATE. Does NOT shift the eval window.
TIMESTEP        = "daily"  # "hourly" or "daily"; used by mod11, mod12, mod15, mod19
ROUTING_METHOD  = "muskingum"  # mRM routing (mod19): "muskingum" | "adaptive" | "adaptive_varying"
OPTI_OBJECTIVE  = "multi_kge"  # mod19 calibration objective: "nse" | "lnnse" | "nse_lnnse" | "kge" | "multi_kge" | "kge_q_et"
N_ITERATIONS   = 500     # mod19 DDS optimizer trials; more = better calibration, longer runtime
SEED           = 32      # mod19 DDS random seed; -9 = clock-based (nondeterministic). Set a positive int for reproducible A/B runs.
N_OMP_THREADS  = 20     # OpenMP threads for mHM; requires binary built with -DCMAKE_WITH_OpenMP=ON
PET_METHOD     = "penman_monteith"  # one of: "hargreaves_samani", "oudin", "priestley_taylor", "penman_monteith"
WANTED_GAUGE_IDS = ["08208000","08206600","08206700","08194500"]
NODATA = -9999
