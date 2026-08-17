"""Shared configuration for all runner modules."""
WORKING_DIR = "/workspace/data/frio_river"  # working directory for all modules
DOMAIN_FILE = "/workspace/data/frio_river/input/domain/data.geojson"  # watershed boundary for all modules
DOMAIN_BUFFER_M = 1000     # buffer [m] grown around the domain polygon when clipping/gridding inputs
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution (mod10, mod13, mod14, mod15, mod16)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution (i.e., mHM output resolution)
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
START_DATE      = "2025-07-09"  # forcing/simulation start; earliest date mHM may draw spin-up from
END_DATE        = "2026-07-31"  # forcing/simulation and evaluation end.
EVAL_START_DATE = "2026-07-10"  # calibration scoring starts here; keep FIXED so warm-up length changes don't move the scored window
WARMUP_DAYS     = 366  # spin-up days drawn from forcing in [EVAL_START_DATE - WARMUP_DAYS, EVAL_START_DATE); capped by START_DATE. Does NOT shift the eval window.
TIMESTEP        = "hourly"  # "hourly" or "daily"; used by mod11, mod12, mod15, mod19
ROUTING_METHOD  = "muskingum"  # mRM routing (mod19): "muskingum" | "adaptive" | "adaptive_varying"
OPTI_OBJECTIVE  = "nse"  # mod19 calibration objective: "nse" | "lnnse" | "nse_lnnse" | "kge" | "multi_kge" | "wnse" | "kge_q_et"
N_ITERATIONS   = 1000     # mod19 DDS optimizer trials; more = better calibration, longer runtime
SEED           = 32      # mod19 DDS random seed; -9 = clock-based (nondeterministic). Set a positive int for reproducible A/B runs.
N_OMP_THREADS  = 25     # OpenMP threads for mHM; requires binary built with -DCMAKE_WITH_OpenMP=ON
PET_METHOD     = "penman_monteith"  # one of: "hargreaves_samani", "oudin", "priestley_taylor", "penman_monteith"
WANTED_GAUGE_IDS = ["08206600",]
NODATA = -9999
RESUME = False  # mod19: if True, reseed DDS start values from the previous run's FinalParam.nml

# --- mod22 (mHM/mRM -> TRITON hydraulic model inputs) ---
TRITON_OUT_DIR         = "/workspace/data/frio_river/triton"  # directory that receives the generated TRITON input files (matches the input/<domain>/ paths in the .cfg)
TRITON_DOMAIN_NAME     = "frio_river"     # basename for the TRITON files (<name>.dem, <name>.roff, ...)
TRITON_DEM_CELLSIZE_M  = 10           # TRITON grid resolution [m]; mod10 dem_corrected.tif is reprojected/resampled to this
TRITON_PROJECTION      = OUTPUT_CRS   # projected CRS written to the TRITON .cfg (must match the runoff grid)
TRITON_EXTBC_TYPE      = 2            # default outlet boundary: 2 = normal slope (velocity from bed slope + Manning roughness)
TRITON_EXTBC_VALUE     = 0.001        # value for the boundary condition; for type 2 this is the bed slope [-]
TRITON_EXTBC_SEG_LEN_M = 2000         # length [m] of the auto-derived outlet boundary segment
TRITON_PRINT_INTERVAL_S = 3600        # TRITON spatial output interval [s]
TRITON_CONST_MANN      = 0.035        # fallback constant Manning roughness used where land cover is unavailable
TRITON_MANN_SOURCE     = "/workspace/data/01-source/lulc_nvalue.tif"  # nationwide land-cover raster; its RAT MANNINGS_N column gives per-class roughness
TRITON_START_DATE      = "2026-07-06"         # event-window start 'YYYY-MM-DD' for the TRITON runoff subset; None = full mHM record
TRITON_END_DATE        = "2026-08-01"         # event-window end 'YYYY-MM-DD' (inclusive); None = full mHM record
