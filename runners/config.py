"""Shared configuration for all runner modules."""
from datetime import date, timedelta

WORKING_DIR = "/workspace/data/frio_river"  # working directory for all modules
DOMAIN_FILE = "/workspace/data/frio_river/mhm_input/domain/data.geojson"  # watershed boundary for all modules
DOMAIN_BUFFER_M = 1000     # buffer [m] grown around the domain polygon when clipping/gridding inputs
L0_CELL_SIZE_M = 250       # Landscape detail grid resolution (mod10, mod13, mod14, mod15, mod16)
L1_CELL_SIZE_M = 1000      # Hydrologic simulation grid resolution (i.e., mHM output resolution)
L2_CELL_SIZE_M = 3000      # Meteorological grid resolution (mod11, mod12)
OUTPUT_CRS     = "EPSG:5070"  # common projected CRS for all spatial outputs
END_DATE        = "2026-07-31"  # forcing/simulation and evaluation end.
EVAL_START_DATE = "2026-07-10"  # calibration scoring starts here; keep FIXED so warm-up length changes don't move the scored window
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
WANTED_GAUGE_IDS = ["08206600",]
NODATA = -9999
RESUME = False  # mod19: if True, reseed DDS start values from the previous run's FinalParam.nml

# --- TRITON model configuration ---
TRITON_OUT_DIR         = "/workspace/data/frio_river/triton_input"  # directory that receives the generated TRITON input files (matches the input/<domain>/ paths in the .cfg)
TRITON_DOMAIN_NAME     = "frio_river"     # basename for the TRITON files (<name>.dem, <name>.roff, ...)
TRITON_DEM_CELLSIZE_M  = 10           # TRITON grid resolution [m]; mod10 dem_corrected.tif is reprojected/resampled to this
TRITON_PROJECTION      = OUTPUT_CRS   # projected CRS written to the TRITON .cfg (must match the runoff grid)
TRITON_MAPPING_INTERVAL_S = 1800         # TRITON spatial output interval [s]
TRITON_HYDROGRAPH_INTERVAL_S = 900     # TRITON hydrograph output interval [s]
TRITON_COURANT         = 0.5          # CFL stability factor (0.5 = 50% of the max stable timestep)
TRITON_DECOMP_TYPE     = "dynamic"     # domain decomposition: "static" (single partition) or "dynamic" (multi-partition, MPI+CUDA)
TRITON_DECOMP_FACTOR   = 5           # dynamic domain-decomposition re-partition interval; 1 (every step) trips a CUDA illegal-address bug, >=10 is stable
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
TRITON_START_DATE      = "2026-07-10"         # event-window start 'YYYY-MM-DD' for the TRITON runoff subset; with TRITON_AUTO_START it is the earliest allowed start (search lower bound). None = full mHM record
TRITON_END_DATE        = "2026-07-25"         # event-window end 'YYYY-MM-DD' (inclusive); None = full mHM record
TRITON_AUTO_START      = True         # trim the sim start to one mHM step before runoff onset (skip pre-event dry/baseflow steps -> shorter TRITON run); False = start at TRITON_START_DATE
TRITON_ONSET_MM_HR     = 0          # domain-mean runoff intensity [mm/hr] that marks event onset for TRITON_AUTO_START
TRITON_START_FILE      = f"{TRITON_OUT_DIR}/{TRITON_DOMAIN_NAME}.startdate"  # sidecar mod21 writes with the resolved sim start datetime; mod22 reads it to anchor output time axes
TRITON_WARM_START      = False         # produce warm-start init files (inith/initqx/inityq) seeding channel depth/discharge from mHM pre-event baseflow; False = cold start (no init files, cfg omits them)
TRITON_BF_CHANNEL_KM2  = 0.5          # drainage-area threshold [km2] above which a cell is treated as channel for the baseflow seed
TRITON_BF_WIDTH_A      = 3.0          # channel width w = a * A^b [m] with drainage area A in km2 (downstream hydraulic geometry)
TRITON_BF_WIDTH_B      = 0.5          # width exponent b
TRITON_BF_SLOPE_MIN    = 1e-4         # floor on bed slope [m/m] in the Manning normal-depth calculation
TRITON_INIT_FILL       = True         # expand the baseflow channel seed outward to fill DEM channel storage (False = seed-only)
TRITON_INIT_FILL_MAX_H = 0.1          # cap [m] on the level-pool fill depth grown from channel seeds (anti-runaway on flat terrain)

# --- TRITON known-waterbody acquisition + init integration (mod21) ---
TRITON_WATERBODIES         = True      # acquire NHDPlus HR waterbodies and seed them wet in the init fields + .mann
TRITON_WATERBODY_SERVICE_URL = "https://services5.arcgis.com/7weheFjxuNkGGiZi/arcgis/rest/services/National_Hydrography_Dataset_Plus_High_Resolution/FeatureServer"  # ESRI FeatureServer root
TRITON_WATERBODY_LAYER_ID  = 1         # layer id of "Waterbodies and Areas" (polygons) on the FeatureServer
TRITON_WATERBODY_PATH      = f"{TRITON_OUT_DIR}/waterbodies.gpkg"  # cached waterbody polygons (auto-acquired by mod21 when missing)
TRITON_WATERBODY_MAX_H     = 10.0      # cap [m] on the fill-to-rim initial depth inside a waterbody
TRITON_WATERBODY_BASE_H    = 0.3       # guaranteed warm-start depth [m] for every mapped waterbody cell (whole polygon reads wet)
TRITON_WATERBODY_EXCLUDE_FTYPES = ("SwampMarsh", "Wetland", "Inundation Area")  # NHD Feature_Type classes to drop before rasterizing
TRITON_WATERBODY_LAKEPOND_STAGE = "Stage = Normal Pool"      # keep LakePonds only if their FCode description contains this text; None disables

# --- TRITON output -> map configuration (mod22) ---
TRITON_MAP_GTIFF_DIR   = f"{WORKING_DIR}/triton_output/gtiff"  # directory holding the per-timestep TRITON GeoTIFFs (<VAR>_<NN>_<MM>.tif + <VAR>_<NN>.vrt)
TRITON_MAP_OUT_DIR     = f"{WORKING_DIR}/triton_output/maps"   # directory that receives the consolidated netCDFs and pixel-max GeoTIFFs
TRITON_MAP_CFG         = f"{WORKING_DIR}/triton_output/cfg/config_1.cfg"  # TRITON .cfg parsed for print_interval (output cadence in seconds)
TRITON_MAP_HMIN        = 0.01         # water-depth floor [m] below which velocity is undefined (masked to NODATA)
TRITON_MAP_MIN_DEPTH   = 0.01        # depth-map floor [m]; H/MH cells shallower than this are masked to NODATA (0 disables)
TRITON_MAP_CLIP        = f"{WORKING_DIR}/mhm_input/domain/watershed.geojson"  # polygon boundary the maps are clipped to (cells outside -> NODATA); None/"" disables
TRITON_MAP_DEM_TIF     = f"{TRITON_OUT_DIR}/{TRITON_DOMAIN_NAME}_dem_{OUTPUT_CRS.split(':')[-1]}.tif"  # mod21's warped DEM, reused as the GIF hillshade background
TRITON_MAP_SERIES_DIR  = f"{WORKING_DIR}/triton_output/series"  # directory of TRITON stage time-series files (<name>_at_Xsec.txt) at the observation points

# --- TRITON H/MH/V GIF animations (mod22 --gif) ---
TRITON_GIF_VARS        = ("H", "MH", "V")   # variables animated as GIFs
TRITON_GIF_FPS         = 8            # playback frame rate
TRITON_GIF_CMAP        = {"H": "Blues", "MH": "PuBu", "V": "viridis"}  # colormap per variable
TRITON_GIF_MAX_FRAMES  = None          # evenly-strided timestep cap per GIF (bounds render time/file size on long runs)
TRITON_GIF_MAX_DIM     = None         # optional cap [px] on the larger grid dimension for GIF rendering; None disables spatial downsampling

# --- TRITON performance diagnostics (mod22 --perf) ---
TRITON_PERF_SUMMARY    = f"{WORKING_DIR}/triton_output/performance.txt"       # final per-rank timing summary
TRITON_PERF_DIR        = f"{WORKING_DIR}/triton_output/performance"          # per-print-step cumulative timing files (performanceN.txt)
TRITON_PERF_WET_VAR    = "H"          # variable used for the wet-cell/volume overlay (must have a cached <var>.nc)
TRITON_PERF_ROFF       = f"{TRITON_OUT_DIR}/{TRITON_DOMAIN_NAME}.roff"       # gridded runoff time series driving TRITON (mod21 output)

# --- TRITON stage-vs-gauge comparison (mod22 --compare) ---
TRITON_COMPARE_OUT_DIR = f"{WORKING_DIR}/triton_output/compare"  # directory that receives the compare_<gauge>.png hydrograph overlays
TRITON_COMPARE_TZ      = "America/Chicago"  # IANA tz the TRITON start date (mHM forcing) is expressed in; naive H.nc times are localized to it before converting to UTC for alignment with UTC gauge data
# Comparison points sampled from H.nc at an explicit lat/lon (WGS84) rather than the gauge's own (imprecise) coordinate.
# Observed USGS gauge height (00065, ft) is converted to water depth at the point via:
#   depth[m] = (gauge_altitude_ft + gauge_stage_ft) * 0.3048 - bed_elevation[m]
# where bed_elevation is sampled from TRITON_MAP_DEM_TIF; altitude_accuracy_ft sets the shaded uncertainty band.
TRITON_COMPARE_POINTS = (
    {
        "name": "Frio River at Tilden",              # label used in the output filename/plot title
        "lat": 28.4674927922804,                 # sampling latitude  [deg, WGS84]
        "lon": -98.5475173731209,                # sampling longitude [deg, WGS84]
        "gauge_id": "08206600",        # USGS site whose stage (00065) is compared
        "gauge_altitude_ft": 212.66,   # altitude of the gauge zero datum [ft]; set to the site's true datum (near the DEM bed at the point)
        "altitude_accuracy_ft": 0.23,   # datum/DEM vertical uncertainty [ft] -> +/- band on the converted depth
    },
)