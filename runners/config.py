"""Shared configuration for all runner modules."""
from datetime import date, timedelta

WORKING_DIR = "/workspace/data/eastfork_whitewater"  # working directory for all modules
DOMAIN_FILE = "/workspace/data/eastfork_whitewater/input/domain/data.geojson"  # watershed boundary for all modules
DOMAIN_BUFFER_M = 1000     # buffer [m] grown around the domain polygon when clipping/gridding inputs
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution (mod10, mod13, mod14, mod15, mod16)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution (i.e., mHM output resolution)
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
END_DATE        = "2026-08-14"  # forcing/simulation and evaluation end.
EVAL_START_DATE = "2026-07-28"  # calibration scoring starts here; keep FIXED so warm-up length changes don't move the scored window
WARMUP_DAYS     = 350  # spin-up days drawn from forcing in [EVAL_START_DATE - WARMUP_DAYS, EVAL_START_DATE). Does NOT shift the eval window.
# forcing/simulation start (earliest date mHM may draw spin-up from), derived as exactly WARMUP_DAYS before EVAL_START_DATE.
START_DATE      = (date.fromisoformat(EVAL_START_DATE) - timedelta(days=WARMUP_DAYS)).isoformat()
TIMESTEP        = "hourly"  # "hourly" or "daily"; used by mod11, mod12, mod15, mod19
ROUTING_METHOD  = "muskingum"  # mRM routing (mod19): "muskingum" | "adaptive" | "adaptive_varying"
OPTI_OBJECTIVE  = "nse_lnnse"  # mod19 calibration objective: "nse" | "lnnse" | "nse_lnnse" | "kge" | "multi_kge" | "wnse" | "kge_q_et"
N_ITERATIONS   = 1000     # mod19 DDS optimizer trials; more = better calibration, longer runtime
SEED           = 32      # mod19 DDS random seed; -9 = clock-based (nondeterministic). Set a positive int for reproducible A/B runs.
N_OMP_THREADS  = 25     # OpenMP threads for mHM; requires binary built with -DCMAKE_WITH_OpenMP=ON
PET_METHOD     = "penman_monteith"  # one of: "hargreaves_samani", "oudin", "priestley_taylor", "penman_monteith"
WANTED_GAUGE_IDS = ["03275600",]
NODATA = -9999
RESUME = False  # mod19: if True, reseed DDS start values from the previous run's FinalParam.nml

# --- mod22 (mHM/mRM -> TRITON hydraulic model inputs) ---
TRITON_OUT_DIR         = "/workspace/data/eastfork_whitewater/triton"  # directory that receives the generated TRITON input files (matches the input/<domain>/ paths in the .cfg)
TRITON_DOMAIN_NAME     = "eastfork_whitewater"     # basename for the TRITON files (<name>.dem, <name>.roff, ...)
TRITON_DEM_CELLSIZE_M  = 10           # TRITON grid resolution [m]; mod10 dem_corrected.tif is reprojected/resampled to this
TRITON_PROJECTION      = OUTPUT_CRS   # projected CRS written to the TRITON .cfg (must match the runoff grid)
TRITON_EXTBC_TYPE      = 2            # default outlet boundary: 2 = normal slope (velocity from bed slope + Manning roughness)
TRITON_EXTBC_VALUE     = 0.001        # value for the boundary condition; for type 2 this is the bed slope [-]
TRITON_EXTBC_SEG_LEN_M = 2000         # length [m] of the auto-derived outlet boundary segment
TRITON_PRINT_INTERVAL_S = 900         # TRITON spatial output interval [s]
TRITON_CONST_MANN      = 0.060        # fallback constant Manning roughness used where land cover is unavailable
TRITON_CHANNEL_MANN    = 0.038         # roughness burned into .mann for channel cells (facc >= TRITON_BF_CHANNEL_KM2); None disables
TRITON_IO_LULC_PATH    = f"{TRITON_OUT_DIR}/io_lulc.tif"  # cached ESRI/IO 10 m land-cover raster (auto-acquired by mod22 when missing)
TRITON_IO_LULC_YEAR    = None         # IO annual mosaic year (e.g. "2023"); None = latest available
# Per-ESRI/IO-class Manning n for the TRITON .mann field (io-lulc-annual-v02).
# NoData (0) and Clouds (10) are omitted and fall back to TRITON_CONST_MANN.
IO_MANNING_N = {
    1:  0.038,  # Water
    2:  0.150,  # Trees
    4:  0.070,  # Flooded Vegetation
    5:  0.040,  # Crops
    7:  0.022,  # Built Area
    8:  0.030,  # Bare Ground
    9:  0.030,  # Snow/Ice
    11: 0.040,  # Rangeland
}
TRITON_START_DATE      = "2026-08-05"         # event-window start 'YYYY-MM-DD' for the TRITON runoff subset; None = full mHM record
TRITON_END_DATE        = "2026-08-15"         # event-window end 'YYYY-MM-DD' (inclusive); None = full mHM record
TRITON_INITH           = True         # warm-start channels: seed initial depth/discharge (h,qx,qy) from mHM pre-event baseflow
TRITON_BF_CHANNEL_KM2  = 0.5          # drainage-area threshold [km2] above which a cell is treated as channel for the baseflow seed
TRITON_BF_WIDTH_A      = 3.0          # channel width w = a * A^b [m] with drainage area A in km2 (downstream hydraulic geometry)
TRITON_BF_WIDTH_B      = 0.5          # width exponent b
TRITON_BF_SLOPE_MIN    = 1e-4         # floor on bed slope [m/m] in the Manning normal-depth calculation
