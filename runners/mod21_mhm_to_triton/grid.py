"""Grid handling for mod22: reproject/clip the fine DEM and map it onto L1 zones.

The TRITON simulation grid is defined by reprojecting mod10's hydro-corrected
DEM into the projected TRITON CRS and resampling to the configured cellsize.
Every TRITON cell is then tagged with the id of the mHM L1 runoff zone whose
footprint contains the cell centre, producing the .rmap / .roff pairing.
"""
from __future__ import annotations

import os
import sys
import math
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
from osgeo import gdal, osr
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NODATA

gdal.UseExceptions()


def _dissolved_polygon(domain_file: Path, dst_crs: str) -> gpd.GeoDataFrame:
    """Return the domain as a single (multi)polygon in *dst_crs* for cutlining."""
    dom = gpd.read_file(domain_file).to_crs(dst_crs)
    geom = dom.union_all() if hasattr(dom, "union_all") else dom.unary_union
    polys = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, (Polygon, MultiPolygon))]
    if not polys:
        raise ValueError(f"{domain_file} contains no polygon geometry to use as a cutline.")
    return gpd.GeoDataFrame(geometry=[MultiPolygon(
        [p for poly in polys for p in (poly.geoms if isinstance(poly, MultiPolygon) else [poly])]
    )], crs=dst_crs)


def warp_dem(src_tif: Path, domain_file: Path, dst_crs: str, cellsize: float,
             out_tif: Path) -> Dict:
    """Reproject and resample the DEM to *cellsize* metres over the domain bbox.

    Fills the whole domain bounding box (no cutline) so the TRITON DEM is
    rectilinear/gap-free — the source dem_corrected already covers this bbox with
    real terrain. Returns the output grid geometry (origin of the upper-left
    corner, cellsize, ncols, nrows). Pixels are aligned to the cellsize grid so
    the output nests cleanly regardless of the source extent.
    """
    dom = _dissolved_polygon(domain_file, dst_crs)
    minx, miny, maxx, maxy = (float(v) for v in dom.total_bounds)
    gdal.Warp(
        str(out_tif), str(src_tif),
        dstSRS=dst_crs, xRes=cellsize, yRes=cellsize,
        outputBounds=(minx, miny, maxx, maxy),
        dstNodata=NODATA, resampleAlg="average",
        targetAlignedPixels=True, multithread=True,
        outputType=gdal.GDT_Float32,
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )

    ds = gdal.Open(str(out_tif))
    gt = ds.GetGeoTransform()
    grid = {
        "tif": Path(out_tif),
        "x0": gt[0],              # left edge (upper-left x)
        "y0": gt[3],              # top edge (upper-left y)
        "cellsize": gt[1],        # gt[5] == -cellsize for north-up
        "ncols": ds.RasterXSize,
        "nrows": ds.RasterYSize,
        "nodata": NODATA,
    }
    ds = None
    return grid


def read_grid(tif: Path) -> Dict:
    """Return the grid geometry of an existing (already warped) DEM GeoTIFF."""
    ds = gdal.Open(str(tif))
    gt = ds.GetGeoTransform()
    grid = {
        "tif": Path(tif), "x0": gt[0], "y0": gt[3], "cellsize": gt[1],
        "ncols": ds.RasterXSize, "nrows": ds.RasterYSize, "nodata": NODATA,
    }
    ds = None
    return grid


def warp_to_dem_grid(src_tif: Path, domain_file: Path, dst_crs: str, dem_grid: Dict,
                     out_tif: Path, resample: str = "near",
                     dst_nodata: float = NODATA) -> Path:
    """Warp *src_tif* onto the exact TRITON DEM grid (same extent/cellsize/dims).

    Pixels outside the watershed are masked to *dst_nodata*. ``near`` resampling
    preserves categorical land-cover codes.
    """
    dom = _dissolved_polygon(domain_file, dst_crs)
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        cutline = tmp.name
    dom.to_file(cutline, driver="GPKG")
    cs = dem_grid["cellsize"]
    xmin, ymax = dem_grid["x0"], dem_grid["y0"]
    xmax, ymin = xmin + dem_grid["ncols"] * cs, ymax - dem_grid["nrows"] * cs
    try:
        gdal.Warp(
            str(out_tif), str(src_tif),
            dstSRS=dst_crs, xRes=cs, yRes=cs, outputBounds=[xmin, ymin, xmax, ymax],
            cutlineDSName=cutline, cropToCutline=False,
            dstNodata=dst_nodata, resampleAlg=resample, multithread=True,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
        )
    finally:
        os.unlink(cutline)
    return Path(out_tif)


def nc_to_tif(nc_path: Path, var: str, out_tif: Path, crs: str) -> Path:
    """Write a single-variable mHM morphology NetCDF (x/y in *crs*) to a GeoTIFF."""
    ds = xr.open_dataset(nc_path)
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    arr = np.asarray(ds[var].values, dtype=float)
    ds.close()
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    ny, nx = arr.shape
    arr = np.where(np.isfinite(arr), arr, NODATA)
    drv = gdal.GetDriverByName("GTiff")
    o = drv.Create(str(out_tif), nx, ny, 1, gdal.GDT_Float64)
    o.SetGeoTransform((x[0] - dx / 2.0, dx, 0.0, y[0] - dy / 2.0, 0.0, dy))
    srs = osr.SpatialReference(); srs.SetFromUserInput(crs)
    o.SetProjection(srs.ExportToWkt())
    band = o.GetRasterBand(1)
    band.SetNoDataValue(float(NODATA))
    band.WriteArray(arr)
    o.FlushCache(); o = None
    return Path(out_tif)


def prepare_field_on_dem_grid(nc_path: Path, var: str, domain_file: Path, dst_crs: str,
                              dem_grid: Dict, out_tif: Path, resample: str = "near") -> Path:
    """Rasterize an mHM morphology field and warp it onto the exact DEM grid."""
    tmp = Path(f"{out_tif}.src.tif")
    nc_to_tif(nc_path, var, tmp, dst_crs)
    try:
        warp_to_dem_grid(tmp, domain_file, dst_crs, dem_grid, out_tif,
                         resample=resample, dst_nodata=NODATA)
    finally:
        if tmp.exists():
            os.unlink(tmp)
    return Path(out_tif)


def pixel_area_m2(tif: Path) -> float:
    """Area [m2] of one source pixel; handles projected and geographic rasters.

    Flow-accumulation counts upstream cells, so drainage area = count x this.
    """
    ds = gdal.Open(str(tif))
    gt = ds.GetGeoTransform()
    ny = ds.RasterYSize
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    ds = None
    if srs.IsProjected():
        return abs(gt[1] * gt[5])
    cy = gt[3] + (ny / 2.0) * gt[5]  # mean latitude of the raster
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(cy))
    return abs(gt[1]) * m_per_deg_lon * abs(gt[5]) * m_per_deg_lat


def rasterize_waterbodies_to_dem_grid(vector_path: Path, dem_grid: Dict, out_tif: Path,
                                      all_touched: bool = True) -> Path:
    """Burn waterbody polygons onto the exact DEM grid -> uint8 mask (1 = water)."""
    cs = dem_grid["cellsize"]
    ncols, nrows = dem_grid["ncols"], dem_grid["nrows"]
    xmin, ymax = dem_grid["x0"], dem_grid["y0"]
    xmax, ymin = xmin + ncols * cs, ymax - nrows * cs
    tif = gdal.GetDriverByName("GTiff").Create(
        str(out_tif), ncols, nrows, 1, gdal.GDT_Byte,
        ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    tif.SetGeoTransform((xmin, cs, 0.0, ymax, 0.0, -cs))
    tif.SetProjection(gdal.Open(str(dem_grid["tif"])).GetProjection())
    tif.GetRasterBand(1).Fill(0)
    src = gdal.OpenEx(str(vector_path), gdal.OF_VECTOR)
    if src is not None and src.GetLayer(0).GetFeatureCount() > 0:
        opts = ["ALL_TOUCHED=TRUE"] if all_touched else []
        gdal.RasterizeLayer(tif, [1], src.GetLayer(0), burn_values=[1], options=opts)
    src = None
    tif.FlushCache()
    tif = None
    return Path(out_tif)


def build_waterbody_depth(dem_grid: Dict, wb_mask_tif: Path, out_tif: Path,
                          max_h: float, base_h: float) -> Dict:
    """Fill each waterbody to its rim/DEM level and write an initial-depth GeoTIFF.

    Connected waterbody cells form one pool; its water-surface level is the lowest
    DEM elevation on the pool's one-cell rim (the spill point), or the pool's own
    maximum where no valid rim exists. Depth = level - DEM, clamped to [0, *max_h*].
    Every mapped waterbody cell is then floored to *base_h* so the full polygon
    reads wet (mapped water is not a DEM depression everywhere), while the DEM
    still adds variable depth along the deeper thalweg. Returns the GeoTIFF path
    plus pool/cell counts and the maximum seeded depth.
    """
    ds = gdal.Open(str(dem_grid["tif"]))
    dem = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    prj = ds.GetProjection()
    ds = None
    dm = gdal.Open(str(wb_mask_tif))
    mask = dm.GetRasterBand(1).ReadAsArray()
    dm = None

    valid = np.isfinite(dem) & (dem != NODATA)
    pools = (mask == 1) & valid
    depth = np.zeros(dem.shape, dtype=np.float64)
    n_pools = n_cells = 0
    if pools.any():
        structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
        labels, _ = ndimage.label(pools, structure=structure)
        ny, nx = dem.shape
        for lbl, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            r0, r1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, ny)
            c0, c1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, nx)
            sub_dem = dem[r0:r1, c0:c1]
            sub_valid = valid[r0:r1, c0:c1]
            pool = labels[r0:r1, c0:c1] == lbl
            rim = ndimage.binary_dilation(pool, structure=structure) & ~pool & sub_valid
            level = float(sub_dem[rim].min()) if rim.any() else float(sub_dem[pool].max())
            d = np.clip(level - sub_dem, 0.0, max_h)
            d[pool] = np.maximum(d[pool], base_h)  # guarantee every mapped water cell is wet
            d[~pool] = 0.0
            depth[r0:r1, c0:c1] = np.maximum(depth[r0:r1, c0:c1], d)
            n_pools += 1
            n_cells += int(pool.sum())

    cs = dem_grid["cellsize"]
    tif = gdal.GetDriverByName("GTiff").Create(
        str(out_tif), dem_grid["ncols"], dem_grid["nrows"], 1, gdal.GDT_Float32,
        ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    tif.SetGeoTransform((dem_grid["x0"], cs, 0.0, dem_grid["y0"], 0.0, -cs))
    tif.SetProjection(prj)
    tif.GetRasterBand(1).SetNoDataValue(0.0)
    tif.GetRasterBand(1).WriteArray(depth.astype(np.float32))
    tif.FlushCache()
    tif = None
    return {"tif": Path(out_tif), "n_pools": n_pools, "n_cells": n_cells,
            "max_depth_m": float(depth.max()) if n_cells else 0.0}



def build_zones(runoff: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """Assign sequential runoff-zone ids (1..N) to valid L1 cells.

    A cell is valid where mHM wrote finite runoff (i.e. inside the domain). Ids
    are assigned in row-major order; the same order indexes the .roff columns.
    Returns (zone_ids grid [ny1, nx1] int32, valid mask [ny1, nx1] bool, N).
    """
    q0 = runoff["da"].isel(time=0).values
    valid = np.isfinite(q0)
    n = int(valid.sum())
    zone_ids = np.zeros(valid.shape, dtype=np.int32)
    zone_ids[valid] = np.arange(1, n + 1, dtype=np.int32)
    return zone_ids, valid, n


def dem_axis_to_l1(grid: Dict, runoff: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Map DEM column/row indices to L1 column/row indices (-1 when outside L1).

    Uses DEM cell centres and the L1 grid edges. Because both axes are separable
    the mapping is computed per-axis and broadcast when building each rmap row.
    """
    cs = grid["cellsize"]
    cs1 = runoff["cellsize"]
    xs = grid["x0"] + (np.arange(grid["ncols"]) + 0.5) * cs
    ys = grid["y0"] - (np.arange(grid["nrows"]) + 0.5) * cs
    ix1 = np.floor((xs - runoff["left"]) / cs1).astype(np.int64)
    iy1 = np.floor((runoff["top"] - ys) / cs1).astype(np.int64)
    nx1, ny1 = runoff["easting"].size, runoff["northing"].size
    ix1[(ix1 < 0) | (ix1 >= nx1)] = -1
    iy1[(iy1 < 0) | (iy1 >= ny1)] = -1
    return ix1, iy1
