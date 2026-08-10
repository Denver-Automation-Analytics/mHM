"""
DEM → mHM morphology preparation.

Given a model perimeter, this script retrieves DEM data from the 3DEP Seamless server,
performs terrain conditioning (breach and fill), calculates flow direction and accumulation,
computes slope and aspect, and writes the results to NetCDF files suitable for mHM input.
"""
import math
import os
import gc
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import L0_CELL_SIZE_M, OUTPUT_CRS, DOMAIN_FILE
import geopandas as gpd
import netCDF4 as nc4
import overflow
import seamless_3dep as s3dep
import numpy as np
from osgeo import gdal
from pyproj import CRS as ProjCRS
from clip import clip_mosaic
from mosaic import mosaic_tiles

NODATA = -9999

def get_dem_tiles(model_perimeter: gpd.GeoDataFrame,
            res: int = 10,
            save_dir: str = "/workspace/test_domain_3/input/dem"):
    """
    Get tiled DEM data within the model perimeter

    Parameters
    ----------
    model_perimeter : gpd.GeoDataFrame
        The perimeter of the model
    res : int, optional
        The resolution of the DEM data to retrieve (default is 10 m)
    save_dir : str, optional
        The directory to save the DEM data (default is "/workspace/test_domain_3/input/dem")

    Returns
    -------
    list
        List of paths to the downloaded DEM tiles
    """
    # Get the bounding box in WGS84 (decimal degrees) as required by seamless_3dep
    bounds = model_perimeter.to_crs("EPSG:4326").total_bounds  # [minx, miny, maxx, maxy]
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])  # (west, south, east, north)
    import gc
    # Download DEM tiles (hash-named intermediates)
    try:
        tiff_files = s3dep.get_dem(bbox=bbox, res=res, save_dir=save_dir)
        num_tiles = len(tiff_files)
        print(f"Downloaded {num_tiles} DEM tiles.")
        return tiff_files
    except Exception as e:
        print(f"Error downloading DEM tiles: {e}")
        return []

def write_asc(tif_path: str, asc_path: str, coord_scale: float = 1.0, value_scale: float = 1.0):
    """
    Write a GeoTIFF to an ASCII grid file (ESRI ASCII raster format) with optional scaling.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF file.
    asc_path : str
        Path to the output ASCII grid file.
    coord_scale : float, optional
        Scale factor for the coordinates (default is 1.0).
    value_scale : float, optional
        Scale factor for the raster values (default is 1.0).
    """
    ds = gdal.Open(tif_path)
    gt = ds.GetGeoTransform()
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    xllcorner = gt[0] * coord_scale
    yllcorner = (gt[3] + nrows * gt[5]) * coord_scale  # gt[5] < 0 for north-up rasters
    cellsize = gt[1] * coord_scale
    band = ds.GetRasterBand(1)
    src_nodata = band.GetNoDataValue()
    # stream one scanline at a time to avoid loading the full raster into RAM
    with open(asc_path, "w") as f:
        f.write(f"ncols\t{ncols}\n")
        f.write(f"nrows\t{nrows}\n")
        f.write(f"xllcorner\t{xllcorner}\n")
        f.write(f"yllcorner\t{yllcorner}\n")
        f.write(f"cellsize\t{cellsize}\n")
        f.write(f"NODATA_value\t{NODATA}\n")
        for i in range(nrows):
            row = band.ReadAsArray(0, i, ncols, 1)[0].astype(np.float64)
            if src_nodata is not None:
                row[row == src_nodata] = NODATA
            row[row != NODATA] *= value_scale
            f.write(" ".join(str(int(v) if v == int(v) else v) for v in row) + "\n")
    band = None
    ds = None


def write_nc(tif_path: str, nc_path: str, dtype: str, var_name: str,
            value_scale: float = 1.0, block_size: int = 256):
    """Write a GeoTIFF to a NetCDF file formatted for mHM ingestion.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF file.
    nc_path : str
        Path to the output NetCDF file.
    dtype : str
        NumPy dtype string for the output variable ('float64' or 'int32').
    var_name : str
        Name of the data variable in the NetCDF file.
    value_scale : float, optional
        Scale factor applied to non-nodata values (default is 1.0).
    block_size : int, optional
        Number of rows to read and write per iteration (default is 256).
    """
    ds = gdal.Open(tif_path)
    gt = ds.GetGeoTransform()       # (xmin, xres, 0, ymax, 0, -yres)
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    band = ds.GetRasterBand(1)
    src_nodata = band.GetNoDataValue()

    # Center-of-cell coordinate arrays
    x_coords = gt[0] + (np.arange(ncols) + 0.5) * gt[1]
    y_coords = gt[3] + (np.arange(nrows) + 0.5) * gt[5]  # gt[5] < 0

    fill_val = np.dtype(dtype).type(NODATA)

    with nc4.Dataset(nc_path, "w", format="NETCDF4") as ds_nc:
        ds_nc.Conventions = "CF-1.6"

        ds_nc.createDimension("x", ncols)
        ds_nc.createDimension("y", nrows)

        # projected coordinates in metres (LCC)
        xv = ds_nc.createVariable("x", "f8", ("x",))
        xv.axis = "X"
        xv.standard_name = "projection_x_coordinate"
        xv.units = "m"
        xv[:] = x_coords

        yv = ds_nc.createVariable("y", "f8", ("y",))
        yv.axis = "Y"
        yv.standard_name = "projection_y_coordinate"
        yv.units = "m"
        yv[:] = y_coords

        # mHM requires dimension order (y, x); GDAL ReadAsArray returns (row, col) = (y, x)
        _dtype_map = {"float64": "f8", "float32": "f4", "int32": "i4", "int16": "i2"}
        nc_dtype = _dtype_map.get(dtype, "f4")
        dv = ds_nc.createVariable(var_name, nc_dtype, ("y", "x"),
                                  fill_value=fill_val, chunksizes=(block_size, block_size),
                                  zlib=True, complevel=4)
        dv.coordinates = "y x"

        for row_start in range(0, nrows, block_size):
            row_count = min(block_size, nrows - row_start)
            block = band.ReadAsArray(0, row_start, ncols, row_count).astype(np.dtype(dtype))
            if src_nodata is not None:
                block[block == src_nodata] = fill_val
            if value_scale != 1.0:
                mask = block != fill_val
                block[mask] *= value_scale
            dv[row_start:row_start + row_count, :] = block

    band = None
    ds = None


def _build_l0_grid_from_domain(domain: gpd.GeoDataFrame, cellsize_m: float, crs: str) -> tuple:
    """Return (xmin, ymin, xmax, ymax, cellsize_m, crs_wkt) snapped to cellsize_m multiples."""
    projected = domain.to_crs(crs)
    raw_xmin, raw_ymin, raw_xmax, raw_ymax = projected.total_bounds
    xmin = math.floor(raw_xmin / cellsize_m) * cellsize_m
    ymin = math.floor(raw_ymin / cellsize_m) * cellsize_m
    xmax = math.ceil(raw_xmax / cellsize_m) * cellsize_m
    ymax = math.ceil(raw_ymax / cellsize_m) * cellsize_m
    crs_wkt = ProjCRS.from_user_input(crs).to_wkt()
    return xmin, ymin, xmax, ymax, cellsize_m, crs_wkt


def _warp_to_l0(src_tif: str, dst_tif: str, resample_alg: str,
                xmin: float, ymin: float, xmax: float, ymax: float,
                cellsize: float, crs_wkt: str) -> None:
    """Reproject and resample src_tif to the L0 grid, writing dst_tif."""
    gdal.Warp(
        dst_tif,
        src_tif,
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=cellsize,
        yRes=cellsize,
        dstSRS=crs_wkt,
        resampleAlg=resample_alg,
        format="GTiff",
        multithread=True,
    )


# overflow D8 codes (0=E,1=NE,2=N,3=NW,4=W,5=SW,6=S,7=SE) → ArcGIS powers-of-2 (mHM)
_OVERFLOW_TO_ARCGIS = {0: 1, 1: 128, 2: 64, 3: 32, 4: 16, 5: 8, 6: 4, 7: 2}


def _remap_fdir_to_arcgis(fdir_nc: str) -> None:
    """Remap fdir.nc from overflow's 0-7 D8 codes to ArcGIS powers-of-2 in place.

    Undefined cells (overflow code 8) become 0, which mHM treats as an outlet.
    """
    with nc4.Dataset(fdir_nc, 'r+') as ds:
        fdir = np.array(ds['fdir'][:])
        fv = int(ds['fdir']._FillValue)
        out = fdir.copy()
        for src, dst in _OVERFLOW_TO_ARCGIS.items():
            out[fdir == src] = dst
        out[fdir == 8] = 0
        out[fdir == fv] = fv
        ds['fdir'][:] = out


if __name__ == "__main__":

    # ---- USER INPUTS --------------------------------------------------
    DEM_CELL_SIZE_M   = 10        # native resolution of the source DEM (m)
    RADIUS_CELLS       = 50        # radius for breaching (cells)
    CHUNK_SIZE         = 256       # chunk size for tiled processing (cells)
    DEM_DIR  = "/workspace/test_domain_3/input/dem"
    TILES_DIR = "/workspace/test_domain_3/input/dem/tiles"
    MORPH_DIR = "/workspace/test_domain_3/input/morph"
    # -------------------------------------------------------------------

    # 1. Get DEM data within the model perimeter
    print("Getting DEM data within the model perimeter...")
    mosaic_file = os.path.join(DEM_DIR, "mosaic.tif")
    mosaic_clip_file = os.path.join(DEM_DIR, "mosaic_clipped.tif")
    model_perimeter = gpd.read_file(DOMAIN_FILE)
    if not os.path.exists(mosaic_clip_file):
        tiles = get_dem_tiles(model_perimeter, res=DEM_CELL_SIZE_M, save_dir=TILES_DIR)
        if len(tiles) == 0:
            print("Failed: No DEM tiles were downloaded.")
            sys.exit(1)
        else:
            print("Mosaicing DEM tiles")
            mosaic_tiles(tiles, mosaic_file)

        print("Clipping mosaic to model perimeter")
        clip_mosaic(mosaic_file, model_perimeter, mosaic_clip_file)
    gc.collect()

    # 2. Terrain Conditioning (Breach + Fill)
    print("Conditioning DEM (breach + fill)...")
    if not os.path.exists(f"{DEM_DIR}/dem_breached.tif"):
        overflow.breach_paths_least_cost(mosaic_clip_file,
                                         f"{DEM_DIR}/dem_breached.tif",
                                         search_radius=RADIUS_CELLS,
                                         chunk_size=CHUNK_SIZE)
    if not os.path.exists(f"{DEM_DIR}/dem_corrected.tif"):
        overflow.fill_depressions_tiled(f"{DEM_DIR}/dem_breached.tif",
                                        f"{DEM_DIR}/dem_corrected.tif",
                                        chunk_size=CHUNK_SIZE,
                                        working_dir=DEM_DIR)

    # 3. Terrain Attributes
    print("Calculating slope and aspect...")
    if not os.path.exists(f"{DEM_DIR}/slope.tif"):
        gdal.DEMProcessing(f"{DEM_DIR}/slope.tif",
                           f"{DEM_DIR}/dem_corrected.tif",
                           "slope",
                           slopeFormat="degree")
    if not os.path.exists(f"{DEM_DIR}/aspect.tif"):
        gdal.DEMProcessing(f"{DEM_DIR}/aspect.tif",
                           f"{DEM_DIR}/dem_corrected.tif",
                           "aspect")

    # 4. Resample terrain derivatives to L0 grid
    print("Resampling terrain derivatives to L0 grid...")
    xmin, ymin, xmax, ymax, cellsize, crs_wkt = _build_l0_grid_from_domain(
        model_perimeter, L0_CELL_SIZE_M, OUTPUT_CRS
    )
    print(f"  L0 grid: cellsize={cellsize:.0f} m  extent=({xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f})")

    l0_dir = os.path.join(DEM_DIR, "l0")
    os.makedirs(l0_dir, exist_ok=True)

    # algorithm per variable: average for continuous derivatives
    _resample_jobs = [
        (f"{DEM_DIR}/dem_corrected.tif", f"{l0_dir}/dem_l0.tif",    "average"),
        (f"{DEM_DIR}/slope.tif",          f"{l0_dir}/slope_l0.tif",  "average"),
        (f"{DEM_DIR}/aspect.tif",         f"{l0_dir}/aspect_l0.tif", "average"),
    ]
    for src, dst, alg in _resample_jobs:
        if not os.path.exists(dst):
            print(f"  {os.path.basename(src)} → {os.path.basename(dst)} ({alg})")
            _warp_to_l0(src, dst, alg, xmin, ymin, xmax, ymax, cellsize, crs_wkt)

    # 4b. Derive flow direction and accumulation on the L0 DEM.
    # Flow direction is categorical and cannot be resampled; recompute it on the
    # L0 DEM so the coarse network stays hydrologically connected. fix_flats
    # resolves directions across filled/flat areas (else the network fragments).
    print("Deriving flow direction and accumulation on the L0 grid...")
    dem_l0        = f"{l0_dir}/dem_l0.tif"
    dem_l0_filled = f"{l0_dir}/dem_l0_corrected.tif"
    fdir_l0_raw   = f"{l0_dir}/fdir_l0_raw.tif"
    fdir_l0       = f"{l0_dir}/fdir_l0.tif"
    facc_l0       = f"{l0_dir}/facc_l0.tif"
    if not os.path.exists(dem_l0_filled):
        overflow.fill_depressions_tiled(dem_l0, dem_l0_filled,
                                        chunk_size=CHUNK_SIZE, working_dir=l0_dir)
    if not os.path.exists(fdir_l0_raw):
        overflow.flow_direction(dem_l0_filled, fdir_l0_raw, chunk_size=CHUNK_SIZE)
    if not os.path.exists(fdir_l0):
        overflow.fix_flats_tiled(dem_l0_filled, fdir_l0_raw, fdir_l0,
                                 chunk_size=CHUNK_SIZE, working_dir=l0_dir)
    if not os.path.exists(facc_l0):
        overflow.flow_accumulation_tiled(fdir_l0, facc_l0, chunk_size=CHUNK_SIZE)

    # 5. Write mHM-ready NetCDF files from L0-resampled TIFs
    print("Writing mHM-ready NetCDF files...")
    os.makedirs(MORPH_DIR, exist_ok=True)
    write_nc(f"{l0_dir}/dem_l0.tif",    f"{MORPH_DIR}/dem.nc",    dtype="float32", var_name="dem",    block_size=CHUNK_SIZE)
    write_nc(f"{l0_dir}/slope_l0.tif",  f"{MORPH_DIR}/slope.nc",  dtype="float32", var_name="slope",  block_size=CHUNK_SIZE)
    write_nc(f"{l0_dir}/aspect_l0.tif", f"{MORPH_DIR}/aspect.nc", dtype="float32", var_name="aspect", block_size=CHUNK_SIZE)
    write_nc(f"{l0_dir}/fdir_l0.tif",   f"{MORPH_DIR}/fdir.nc",   dtype="int32",   var_name="fdir",   block_size=CHUNK_SIZE)
    write_nc(f"{l0_dir}/facc_l0.tif",   f"{MORPH_DIR}/facc.nc",   dtype="int32",   var_name="facc",   block_size=CHUNK_SIZE)
    _remap_fdir_to_arcgis(f"{MORPH_DIR}/fdir.nc")
