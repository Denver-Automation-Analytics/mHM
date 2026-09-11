"""
DEM → mHM morphology preparation.

Given a model perimeter, this script retrieves DEM data from the 3DEP Seamless server,
performs terrain conditioning (breach and fill), calculates flow direction and accumulation,
computes slope and aspect, and writes the results to NetCDF files suitable for mHM input.
"""

import math
import os
import gc
import glob
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    L0_CELL_SIZE_M,
    OUTPUT_CRS,
    DOMAIN_FILE,
    DOMAIN_BUFFER_M,
    WORKING_DIR,
    NODATA,
)
import geopandas as gpd
import netCDF4 as nc4
import overflow
import seamless_3dep as s3dep
import numpy as np
from osgeo import gdal
from pyproj import CRS as ProjCRS
import rasterio
from rasterio import features
from shapely.geometry import shape, box
from shapely.ops import unary_union

from clip import clip_mosaic
from mosaic import mosaic_tiles


def load_domain(buffer_m=None, crs=None):
    """Return the domain polygon as a GeoDataFrame, grown by ``buffer_m`` meters.

    Buffering is performed in OUTPUT_CRS (projected metres).
    """

    if buffer_m is None:
        buffer_m = DOMAIN_BUFFER_M
    gdf = gpd.read_file(DOMAIN_FILE)
    src_crs = gdf.crs
    projected = gdf.to_crs(OUTPUT_CRS)
    if buffer_m:
        projected["geometry"] = projected.geometry.buffer(buffer_m)
    return projected.to_crs(crs if crs is not None else src_crs)


def acquisition_perimeter(margin_m: float = None) -> gpd.GeoDataFrame:
    """Rectangle over the domain's OUTPUT_CRS bounding box (+margin), in OUTPUT_CRS.

    DEM tiles/clip use this so that after reprojection to OUTPUT_CRS the raster fully
    covers the projected (axis-aligned) grid: a source-CRS rectangle reprojects to a
    rotated quad and would leave nodata in the projected bbox corners, so acquisition
    must be sized in the target CRS, not the geographic source CRS.
    """
    if margin_m is None:
        margin_m = 2 * DOMAIN_BUFFER_M
    dom = load_domain().to_crs(OUTPUT_CRS)
    minx, miny, maxx, maxy = dom.total_bounds
    rect = box(minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m)
    return gpd.GeoDataFrame({"id": [1]}, geometry=[rect], crs=OUTPUT_CRS)


def get_dem_tiles(
    model_perimeter: gpd.GeoDataFrame,
    res: int = 10,
    save_dir: str = os.path.join(WORKING_DIR, "mhm_input/dem"),
):
    """
    Get tiled DEM data within the model perimeter

    Parameters
    ----------
    model_perimeter : gpd.GeoDataFrame
        The perimeter of the model
    res : int, optional
        The resolution of the DEM data to retrieve (default is 10 m)
    save_dir : str, optional
        The directory to save the DEM data (default is os.path.join(WORKING_DIR, "mhm_input/dem"))

    Returns
    -------
    list
        List of paths to the downloaded DEM tiles
    """
    # Get the bounding box in WGS84 (decimal degrees) as required by seamless_3dep
    bounds = model_perimeter.to_crs(
        "EPSG:4326"
    ).total_bounds  # [minx, miny, maxx, maxy]
    bbox = (bounds[0], bounds[1], bounds[2], bounds[3])  # (west, south, east, north)
    # Snapshot tiles already on disk so we can report cache hits vs new downloads
    cached_before = set(glob.glob(os.path.join(save_dir, "*.tiff")))
    try:
        # get_dem is resumable: it only fetches hash-named tiles missing from save_dir
        tiff_files = s3dep.get_dem(bbox=bbox, res=res, save_dir=save_dir)
        num_downloaded = sum(1 for f in tiff_files if os.fspath(f) not in cached_before)
        num_cached = len(tiff_files) - num_downloaded
        if num_downloaded == 0:
            print(
                f"All {len(tiff_files)} DEM tiles already cached — skipping download."
            )
        else:
            print(
                f"Downloaded {num_downloaded} DEM tiles ({num_cached} reused from cache)."
            )
        return tiff_files
    except Exception as e:
        print(f"Error downloading DEM tiles: {e}")
        return []


def write_asc(
    tif_path: str, asc_path: str, coord_scale: float = 1.0, value_scale: float = 1.0
):
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


def write_nc(
    tif_path: str,
    nc_path: str,
    dtype: str,
    var_name: str,
    value_scale: float = 1.0,
    block_size: int = 256,
):
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
    gt = ds.GetGeoTransform()  # (xmin, xres, 0, ymax, 0, -yres)
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
        dv = ds_nc.createVariable(
            var_name,
            nc_dtype,
            ("y", "x"),
            fill_value=fill_val,
            chunksizes=(block_size, block_size),
            zlib=True,
            complevel=4,
        )
        dv.coordinates = "y x"

        for row_start in range(0, nrows, block_size):
            row_count = min(block_size, nrows - row_start)
            block = band.ReadAsArray(0, row_start, ncols, row_count).astype(
                np.dtype(dtype)
            )
            if src_nodata is not None:
                block[block == src_nodata] = fill_val
            if value_scale != 1.0:
                mask = block != fill_val
                block[mask] *= value_scale
            dv[row_start : row_start + row_count, :] = block

    band = None
    ds = None


def _build_l0_grid_from_domain(
    domain: gpd.GeoDataFrame, cellsize_m: float, crs: str
) -> tuple:
    """Return (xmin, ymin, xmax, ymax, cellsize_m, crs_wkt) snapped to cellsize_m multiples."""
    projected = domain.to_crs(crs)
    raw_xmin, raw_ymin, raw_xmax, raw_ymax = projected.total_bounds
    xmin = math.floor(raw_xmin / cellsize_m) * cellsize_m
    ymin = math.floor(raw_ymin / cellsize_m) * cellsize_m
    xmax = math.ceil(raw_xmax / cellsize_m) * cellsize_m
    ymax = math.ceil(raw_ymax / cellsize_m) * cellsize_m
    crs_wkt = ProjCRS.from_user_input(crs).to_wkt()
    return xmin, ymin, xmax, ymax, cellsize_m, crs_wkt


def _warp_to_l0(
    src_tif: str,
    dst_tif: str,
    resample_alg: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    cellsize: float,
    crs_wkt: str,
) -> None:
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

_ARCGIS_D8_OFFSETS = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}


def _remap_fdir_to_arcgis(fdir_nc: str) -> None:
    """Remap fdir.nc from overflow's 0-7 D8 codes to ArcGIS powers-of-2 in place.

    Undefined cells (overflow code 8) become 0, which mHM treats as an outlet.
    """
    with nc4.Dataset(fdir_nc, "r+") as ds:
        fdir = np.array(ds["fdir"][:])
        fv = int(ds["fdir"]._FillValue)
        out = fdir.copy()
        for src, dst in _OVERFLOW_TO_ARCGIS.items():
            out[fdir == src] = dst
        out[fdir == 8] = 0
        out[fdir == fv] = fv
        ds["fdir"][:] = out


def _write_fdir_arcgis(src_tif: str, dst_tif: str) -> None:
    """Copy an overflow 0-7 D8 flow-direction raster to ArcGIS powers-of-2 codes.

    Written block-wise so full-resolution grids never load fully into memory;
    overflow's undefined code 8 becomes 0 (outlet).
    """
    with rasterio.open(src_tif) as s:
        profile = s.profile
        with rasterio.open(dst_tif, "w", **profile) as d:
            for _, win in s.block_windows(1):
                a = s.read(1, window=win)
                out = a.copy()
                for src, dst in _OVERFLOW_TO_ARCGIS.items():
                    out[a == src] = dst
                out[a == 8] = 0
                d.write(out, 1, window=win)


def _rasterize_domain_mask(domain: gpd.GeoDataFrame, shape, transform) -> np.ndarray:
    """Rasterize the unbuffered model domain on the canonical L0 grid."""
    geoms = [geom for geom in domain.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError(f"Domain file contains no usable geometry: {DOMAIN_FILE}")
    if any(not geom.is_valid for geom in geoms):
        raise ValueError(f"Domain file contains invalid geometry: {DOMAIN_FILE}")
    return features.rasterize(
        ((geom, 1) for geom in geoms),
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _set_boundary_outlets(fdir: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Set ArcGIS D8 cells whose downstream target leaves mask to outlet code 0."""
    if fdir.shape != mask.shape:
        raise ValueError(f"Flow-direction shape {fdir.shape} != mask shape {mask.shape}")

    valid_codes = np.array([0, *_ARCGIS_D8_OFFSETS], dtype=fdir.dtype)
    invalid = mask & ~np.isin(fdir, valid_codes)
    if invalid.any():
        values = np.unique(fdir[invalid]).tolist()
        raise ValueError(f"Invalid ArcGIS flow-direction codes inside HUC8: {values}")

    out = fdir.copy()
    nrows, ncols = mask.shape
    for code, (row_offset, col_offset) in _ARCGIS_D8_OFFSETS.items():
        rows, cols = np.where(mask & (fdir == code))
        target_rows = rows + row_offset
        target_cols = cols + col_offset
        in_grid = (
            (target_rows >= 0)
            & (target_rows < nrows)
            & (target_cols >= 0)
            & (target_cols < ncols)
        )
        crosses_boundary = ~in_grid
        crosses_boundary[in_grid] = ~mask[
            target_rows[in_grid], target_cols[in_grid]
        ]
        out[rows[crosses_boundary], cols[crosses_boundary]] = 0
    return out, int(np.count_nonzero(mask & (out == 0)))


def _mask_to_original_domain(l0_dir: str, morph_dir: str, domain_out: str) -> None:
    """Mask all L0 morphology grids to the original HUC8 and retain all outlets."""
    facc_l0 = os.path.join(l0_dir, "facc_l0.tif")
    with rasterio.open(facc_l0) as reference:
        transform = reference.transform
        crs = reference.crs
        raster_shape = reference.shape

    domain = load_domain(buffer_m=0, crs=crs)
    mask = _rasterize_domain_mask(domain, raster_shape, transform)
    if not mask.any():
        raise ValueError("Original HUC8 does not intersect the L0 grid")

    outlet_count = 0
    for var in ("dem", "slope", "aspect", "fdir", "facc"):
        path = os.path.join(morph_dir, f"{var}.nc")
        with nc4.Dataset(path, "r+") as ds:
            data = ds[var][:]
            if data.shape != mask.shape:
                raise ValueError(f"{var}.nc shape {data.shape} != L0 shape {mask.shape}")
            missing = np.ma.getmaskarray(data)
            values = np.asarray(data.filled(ds[var]._FillValue))
            if np.issubdtype(values.dtype, np.floating):
                missing |= ~np.isfinite(values)
            missing_inside = mask & missing
            if var == "aspect" and missing_inside.any():
                values[missing_inside] = 0.0
                missing[missing_inside] = False
                print(
                    f"  Set {int(missing_inside.sum())} undefined flat-cell "
                    "aspect values to 0 degrees"
                )
                missing_inside = mask & missing
            if missing_inside.any():
                raise ValueError(
                    f"{var}.nc has {int(missing_inside.sum())} missing HUC8 cells"
                )
            if var == "fdir":
                values, outlet_count = _set_boundary_outlets(values, mask)
            ds[var][:] = np.where(mask, values, ds[var]._FillValue)

    geoms = [
        shape(g)
        for g, v in features.shapes(
            mask.astype(np.uint8), mask=mask, transform=transform
        )
        if v == 1
    ]
    gdf = gpd.GeoDataFrame(
        {"id": [1], "source_method": ["original_huc8_l0_mask"]},
        geometry=[unary_union(geoms)],
        crs=crs,
    )
    os.makedirs(os.path.dirname(domain_out), exist_ok=True)
    gdf.to_file(domain_out, driver="GeoJSON")
    area_km2 = int(mask.sum()) * abs(transform.a * transform.e) / 1e6
    print(
        f"  HUC8: {int(mask.sum())} cells ({area_km2:.0f} km2), "
        f"{outlet_count} outlet(s)"
    )
    print(f"  Wrote grid-aligned HUC8 boundary: {domain_out}")


def main():
    """Run the DEM → mHM morphology preparation workflow."""

    # 1. Get DEM data within the model perimeter
    print("Getting DEM data within the model perimeter...")
    mosaic_file = os.path.join(DEM_DIR, "mosaic.tif")
    mosaic_clip_file = os.path.join(DEM_DIR, "mosaic_clipped.tif")
    model_perimeter = load_domain()  # domain polygon grown by DOMAIN_BUFFER_M
    # DEM acquisition/clip is sized in OUTPUT_CRS so the reprojected raster fills the
    # projected grid corners (a geographic rectangle rotates and leaves nodata otherwise)
    acquire_perimeter = acquisition_perimeter()

    if not os.path.exists(mosaic_clip_file):
        if os.path.exists(mosaic_file):
            print(f"Reusing cached mosaic: {mosaic_file}")
        else:
            tiles = get_dem_tiles(
                acquire_perimeter, res=DEM_CELL_SIZE_M, save_dir=TILES_DIR
            )
            if len(tiles) == 0:
                print("Failed: No DEM tiles were downloaded.")
                sys.exit(1)
            print("Mosaicing DEM tiles")
            mosaic_tiles(tiles, mosaic_file)

        print(
            "Clipping mosaic to OUTPUT_CRS bounding box (rectilinear, gap-free after reprojection)"
        )
        clip_mosaic(mosaic_file, acquire_perimeter, mosaic_clip_file, to_bbox=True)
    gc.collect()

    # 2. Terrain Conditioning (Breach + Fill)
    print("Conditioning DEM (breach + fill)...")
    if not os.path.exists(f"{DEM_DIR}/dem_breached.tif"):
        overflow.breach_paths_least_cost(
            mosaic_clip_file,
            f"{DEM_DIR}/dem_breached.tif",
            search_radius=RADIUS_CELLS,
            chunk_size=CHUNK_SIZE,
        )
    if not os.path.exists(f"{DEM_DIR}/dem_corrected.tif"):
        overflow.fill_depressions_tiled(
            f"{DEM_DIR}/dem_breached.tif",
            f"{DEM_DIR}/dem_corrected.tif",
            chunk_size=CHUNK_SIZE,
            working_dir=DEM_DIR,
        )

    # 3. Terrain Attributes
    print("Calculating slope and aspect...")
    if not os.path.exists(f"{DEM_DIR}/slope.tif"):
        gdal.DEMProcessing(
            f"{DEM_DIR}/slope.tif",
            f"{DEM_DIR}/dem_corrected.tif",
            "slope",
            slopeFormat="degree",
        )
    if not os.path.exists(f"{DEM_DIR}/aspect.tif"):
        gdal.DEMProcessing(
            f"{DEM_DIR}/aspect.tif", f"{DEM_DIR}/dem_corrected.tif", "aspect"
        )

    # 3b. Full-resolution flow direction & accumulation for downstream hydraulic
    # coupling (mod22). dem_corrected is already breach+filled, so route directly.
    # facc is accumulated on overflow's 0-7 codes; fdir.tif is the ArcGIS-coded copy.
    print("Deriving full-resolution flow direction and accumulation...")
    fdir_raw = f"{DEM_DIR}/fdir_raw.tif"
    fdir_d8 = f"{DEM_DIR}/fdir_d8.tif"
    if not os.path.exists(fdir_raw):
        overflow.flow_direction(
            f"{DEM_DIR}/dem_corrected.tif", fdir_raw, chunk_size=CHUNK_SIZE
        )
    if not os.path.exists(fdir_d8):
        overflow.fix_flats_tiled(
            f"{DEM_DIR}/dem_corrected.tif",
            fdir_raw,
            fdir_d8,
            chunk_size=CHUNK_SIZE,
            working_dir=DEM_DIR,
        )
    if not os.path.exists(f"{DEM_DIR}/facc.tif"):
        overflow.flow_accumulation_tiled(
            fdir_d8, f"{DEM_DIR}/facc.tif", chunk_size=CHUNK_SIZE
        )
    if not os.path.exists(f"{DEM_DIR}/fdir.tif"):
        _write_fdir_arcgis(fdir_d8, f"{DEM_DIR}/fdir.tif")

    # 4. Resample terrain derivatives to L0 grid
    print("Resampling terrain derivatives to L0 grid...")
    xmin, ymin, xmax, ymax, cellsize, crs_wkt = _build_l0_grid_from_domain(
        model_perimeter, L0_CELL_SIZE_M, OUTPUT_CRS
    )
    print(
        f"  L0 grid: cellsize={cellsize:.0f} m  extent=({xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f})"
    )

    l0_dir = os.path.join(DEM_DIR, "l0")
    os.makedirs(l0_dir, exist_ok=True)

    # algorithm per variable: average for continuous derivatives
    _resample_jobs = [
        (f"{DEM_DIR}/dem_corrected.tif", f"{l0_dir}/dem_l0.tif", "average"),
        (f"{DEM_DIR}/slope.tif", f"{l0_dir}/slope_l0.tif", "average"),
        (f"{DEM_DIR}/aspect.tif", f"{l0_dir}/aspect_l0.tif", "average"),
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
    dem_l0 = f"{l0_dir}/dem_l0.tif"
    dem_l0_filled = f"{l0_dir}/dem_l0_corrected.tif"
    fdir_l0_raw = f"{l0_dir}/fdir_l0_raw.tif"
    fdir_l0 = f"{l0_dir}/fdir_l0.tif"
    facc_l0 = f"{l0_dir}/facc_l0.tif"
    if not os.path.exists(dem_l0_filled):
        overflow.fill_depressions_tiled(
            dem_l0, dem_l0_filled, chunk_size=CHUNK_SIZE, working_dir=l0_dir
        )
    if not os.path.exists(fdir_l0_raw):
        overflow.flow_direction(dem_l0_filled, fdir_l0_raw, chunk_size=CHUNK_SIZE)
    if not os.path.exists(fdir_l0):
        overflow.fix_flats_tiled(
            dem_l0_filled,
            fdir_l0_raw,
            fdir_l0,
            chunk_size=CHUNK_SIZE,
            working_dir=l0_dir,
        )
    if not os.path.exists(facc_l0):
        overflow.flow_accumulation_tiled(fdir_l0, facc_l0, chunk_size=CHUNK_SIZE)

    # 5. Write mHM-ready NetCDF files from L0-resampled TIFs
    print("Writing mHM-ready NetCDF files...")
    os.makedirs(MORPH_DIR, exist_ok=True)
    write_nc(
        f"{l0_dir}/dem_l0.tif",
        f"{MORPH_DIR}/dem.nc",
        dtype="float32",
        var_name="dem",
        block_size=WRITE_CHUNK_SIZE,
    )
    write_nc(
        f"{l0_dir}/slope_l0.tif",
        f"{MORPH_DIR}/slope.nc",
        dtype="float32",
        var_name="slope",
        block_size=WRITE_CHUNK_SIZE,
    )
    write_nc(
        f"{l0_dir}/aspect_l0.tif",
        f"{MORPH_DIR}/aspect.nc",
        dtype="float32",
        var_name="aspect",
        block_size=WRITE_CHUNK_SIZE,
    )
    write_nc(
        f"{l0_dir}/fdir_l0.tif",
        f"{MORPH_DIR}/fdir.nc",
        dtype="int32",
        var_name="fdir",
        block_size=WRITE_CHUNK_SIZE,
    )
    write_nc(
        f"{l0_dir}/facc_l0.tif",
        f"{MORPH_DIR}/facc.nc",
        dtype="int32",
        var_name="facc",
        block_size=WRITE_CHUNK_SIZE,
    )
    _remap_fdir_to_arcgis(f"{MORPH_DIR}/fdir.nc")

    # 6. Mask morphology to the original HUC8. Boundary-crossing flow paths become
    #    outlets; mHM supports multiple outlets within one routed domain.
    print("Masking morphology to the original HUC8 boundary...")
    _mask_to_original_domain(
        l0_dir,
        MORPH_DIR,
        os.path.join(WORKING_DIR, "mhm_input/domain/watershed.geojson"),
    )


if __name__ == "__main__":
    # ---- USER INPUTS --------------------------------------------------
    DEM_CELL_SIZE_M = 10  # native resolution of the source DEM (m)
    RADIUS_CELLS = 50  # radius for breaching (cells)
    CHUNK_SIZE = 256  # chunk size for tiled processing (cells)
    WRITE_CHUNK_SIZE = 32  # chunk size for writing NetCDF (cells)
    DEM_DIR = os.path.join(WORKING_DIR, "mhm_input/dem")  # directory for DEM processing
    TILES_DIR = os.path.join(WORKING_DIR, "mhm_input/dem/tiles")
    MORPH_DIR = os.path.join(WORKING_DIR, "mhm_input/morph")
    # -------------------------------------------------------------------

    main()
