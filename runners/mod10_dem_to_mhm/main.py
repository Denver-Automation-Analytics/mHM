import os
import sys
from pathlib import Path

import numpy as np
import overflow
from osgeo import gdal, osr

sys.path.insert(0, str(Path(__file__).parent.parent / "mod11_meteo_to_mhm"))
from io_watershed import snap_bbox_to_grid  # noqa: E402

gdal.UseExceptions()

NODATA = -9999
FEET_TO_METERS = 0.3048

# ---- USER INPUTS --------------------------------------------------
TARGET_CRS        = "EPSG:5070"
DEM_CELL_SIZE_M   = 10           # native resolution of the source DEM (m)
# -------------------------------------------------------------------


def write_asc(tif_path, asc_path, coord_scale=1.0, value_scale=1.0):
    ds = gdal.Open(tif_path)
    gt = ds.GetGeoTransform()
    ncols = ds.RasterXSize
    nrows = ds.RasterYSize
    xllcorner = gt[0] * coord_scale
    yllcorner = (gt[3] + nrows * gt[5]) * coord_scale  # gt[5] < 0 for north-up rasters
    cellsize = gt[1] * coord_scale
    band = ds.GetRasterBand(1)
    src_nodata = band.GetNoDataValue()
    data = band.ReadAsArray().astype(float)
    if src_nodata is not None:
        data[data == src_nodata] = NODATA
    data[data != NODATA] *= value_scale
    ds = None
    with open(asc_path, "w") as f:
        f.write(f"ncols\t{ncols}\n")
        f.write(f"nrows\t{nrows}\n")
        f.write(f"xllcorner\t{xllcorner}\n")
        f.write(f"yllcorner\t{yllcorner}\n")
        f.write(f"cellsize\t{cellsize}\n")
        f.write(f"NODATA_value\t{NODATA}\n")
        for row in data:
            f.write(" ".join(str(int(v) if v == int(v) else v) for v in row) + "\n")


if __name__ == "__main__":
    # 1. Setup paths
    dem_file = "/workspace/test_domain_3/input/dem/dem_raw_10m.tif"
    dem_dir = "/workspace/test_domain_3/input/dem"
    morph_dir = "/workspace/test_domain_3/input/morph"

    # 2. Reproject raw DEM to TARGET_CRS with bounds snapped to the EPSG:5070 origin
    if not os.path.exists(f"{dem_dir}/dem_reprojected.tif"):
        src_ds = gdal.Open(dem_file)
        src_srs = osr.SpatialReference(wkt=src_ds.GetProjection())
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(int(TARGET_CRS.split(":")[1]))
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(src_srs, dst_srs)
        gt = src_ds.GetGeoTransform()
        nx, ny = src_ds.RasterXSize, src_ds.RasterYSize
        corners = [
            ct.TransformPoint(gt[0],              gt[3]),
            ct.TransformPoint(gt[0] + nx * gt[1], gt[3]),
            ct.TransformPoint(gt[0],              gt[3] + ny * gt[5]),
            ct.TransformPoint(gt[0] + nx * gt[1], gt[3] + ny * gt[5]),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        src_ds = None
        snapped = snap_bbox_to_grid((min(xs), min(ys), max(xs), max(ys)), DEM_CELL_SIZE_M)
        gdal.Warp(
            f"{dem_dir}/dem_reprojected.tif", dem_file,
            dstSRS=TARGET_CRS, outputBounds=snapped,
            xRes=DEM_CELL_SIZE_M, yRes=DEM_CELL_SIZE_M,
        )

    # 3. Terrain Conditioning (Breach + Fill)
    # Note: Python API uses cell count
    radius_cells = 50
    chunk_size = 256

    if not os.path.exists(f"{dem_dir}/dem_breached.tif"):
        overflow.breach_paths_least_cost(f"{dem_dir}/dem_reprojected.tif", f"{dem_dir}/dem_breached.tif", search_radius=radius_cells, chunk_size=chunk_size)
    if not os.path.exists(f"{dem_dir}/dem_corrected.tif"):
        overflow.fill_depressions_tiled(f"{dem_dir}/dem_breached.tif", f"{dem_dir}/dem_corrected.tif", chunk_size=chunk_size, working_dir=dem_dir)

    # 4. Flow Routing
    if not os.path.exists(f"{dem_dir}/fdr.tif"):
        overflow.flow_direction(f"{dem_dir}/dem_corrected.tif", f"{dem_dir}/fdr.tif", chunk_size=chunk_size)
    if not os.path.exists(f"{dem_dir}/accum.tif"):
        overflow.flow_accumulation_tiled(f"{dem_dir}/fdr.tif", f"{dem_dir}/accum.tif", chunk_size=chunk_size)

    # 5. Terrain Attributes
    if not os.path.exists(f"{dem_dir}/slope.tif"):
        gdal.DEMProcessing(f"{dem_dir}/slope.tif", f"{dem_dir}/dem_corrected.tif", "slope", slopeFormat="degree")
    if not os.path.exists(f"{dem_dir}/aspect.tif"):
        gdal.DEMProcessing(f"{dem_dir}/aspect.tif", f"{dem_dir}/dem_corrected.tif", "aspect")

    # 6. Write mHM-ready ASC files
    os.makedirs(morph_dir, exist_ok=True)
    write_asc(f"{dem_dir}/dem_corrected.tif", f"{morph_dir}/dem.asc",   value_scale=FEET_TO_METERS)
    write_asc(f"{dem_dir}/slope.tif",         f"{morph_dir}/slope.asc")
    write_asc(f"{dem_dir}/aspect.tif",         f"{morph_dir}/aspect.asc")
    write_asc(f"{dem_dir}/fdr.tif",            f"{morph_dir}/fdir.asc")
    write_asc(f"{dem_dir}/accum.tif",          f"{morph_dir}/facc.asc")

    # 7. Write elevation TIF in meters for downstream PET use
    if not os.path.exists(f"{dem_dir}/dem_m.tif"):
        src_ds = gdal.Open(f"{dem_dir}/dem_corrected.tif")
        band = src_ds.GetRasterBand(1)
        data = band.ReadAsArray().astype("float32")
        src_nodata = band.GetNoDataValue()
        if src_nodata is not None:
            data[data == src_nodata] = NODATA
        data[data != NODATA] *= FEET_TO_METERS
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            f"{dem_dir}/dem_m.tif",
            src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Float32,
        )
        out_ds.SetGeoTransform(src_ds.GetGeoTransform())
        out_ds.SetProjection(src_ds.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(data)
        out_ds.GetRasterBand(1).SetNoDataValue(NODATA)
        out_ds = src_ds = None


