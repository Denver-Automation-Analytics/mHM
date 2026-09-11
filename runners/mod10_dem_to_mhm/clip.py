import os
import tempfile
import geopandas as gpd
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

NODATA = -9999
CHUNK_SIZE = 256

COG_CREATION_OPTIONS = [
    "COMPRESS=DEFLATE",
    "PREDICTOR=2",
    "BIGTIFF=IF_SAFER",
    "TILED=YES",
]


def _fill_nodata_with_zero(ds: gdal.Dataset, chunk_size: int) -> None:
    """Replace each band's NoData pixels with valid zero values block-wise."""
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        band_nodata = band.GetNoDataValue()
        if band_nodata is None:
            continue

        nodata_is_nan = np.isnan(band_nodata)
        for row_start in range(0, ds.RasterYSize, chunk_size):
            row_count = min(chunk_size, ds.RasterYSize - row_start)
            for col_start in range(0, ds.RasterXSize, chunk_size):
                col_count = min(chunk_size, ds.RasterXSize - col_start)
                block = band.ReadAsArray(
                    col_start, row_start, col_count, row_count
                )
                nodata_mask = (
                    np.isnan(block) if nodata_is_nan else block == band_nodata
                )
                if np.any(nodata_mask):
                    block[nodata_mask] = 0
                    band.WriteArray(block, col_start, row_start)

        band.FlushCache()


def clip_mosaic(
    mosaic_file: str,
    perimeter: gpd.GeoDataFrame,
    output_file: str | None = None,
    nodata: float = NODATA,
    chunk_size: int = CHUNK_SIZE,
    to_bbox: bool = False,
) -> None:
    """Clip a GeoTIFF in-place to the model perimeter polygon.

    Uses GDAL Warp with a cutline so the full raster never loads into Python
    memory — GDAL reads and writes in tiles controlled by chunk_size. Bounding-box
    clips replace NoData gaps with valid zero elevation; polygon clips preserve
    NoData outside the cutline.
    The clip is written to a temp file then atomically swapped over mosaic_file.

    Parameters
    ----------
    mosaic_file : str
        Path to the GeoTIFF to clip (overwritten in place).
    perimeter : gpd.GeoDataFrame
        Model perimeter polygon(s); any CRS accepted.
    output_file : str, optional
        If provided, the clipped raster is written to this path instead of
        overwriting mosaic_file.
    nodata : float, optional
        Nodata value to carry through to the output (default -9999).
    chunk_size : int, optional
        Controls GDAL's internal warp tile size in cells (default 256).
    to_bbox : bool, optional
        If True, clip to the perimeter's bounding box (rectilinear, full coverage,
        with NoData gaps set to zero) instead of the polygon cutline. Used so the
        2D-hydraulic coupling (mod21/TRITON) gets a gap-free DEM. Default False
        (polygon clip).
    """
    # Read source CRS so the cutline geometry is always co-registered
    src_ds = gdal.Open(mosaic_file, gdal.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open {mosaic_file}")
    crs_wkt = src_ds.GetProjection()
    src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
    src_ds = None

    perimeter_reprojected = perimeter.to_crs(crs_wkt)

    # float32 pixel × chunk grid × 8× headroom keeps the warper in a similar
    # memory envelope to the overflow tiled operations elsewhere in the pipeline
    warp_memory_bytes = chunk_size * chunk_size * 4 * 8

    tmp_output = mosaic_file + ".tmp.tif"
    tmp_geojson = None
    try:
        source_nodata_options = (
            {"srcNodata": src_nodata} if src_nodata is not None else {}
        )
        if to_bbox:
            minx, miny, maxx, maxy = perimeter_reprojected.total_bounds
            warp_options = gdal.WarpOptions(
                outputBounds=(minx, miny, maxx, maxy),
                dstNodata=nodata,
                warpMemoryLimit=warp_memory_bytes,
                creationOptions=COG_CREATION_OPTIONS,
                **source_nodata_options,
            )
        else:
            tmp_geojson = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
            tmp_geojson.close()
            perimeter_reprojected.to_file(tmp_geojson.name, driver="GeoJSON")
            warp_options = gdal.WarpOptions(
                cutlineDSName=tmp_geojson.name,
                cropToCutline=True,
                dstNodata=nodata,
                warpMemoryLimit=warp_memory_bytes,
                warpOptions=["CUTLINE_ALL_TOUCHED=TRUE"],
                creationOptions=COG_CREATION_OPTIONS,
                **source_nodata_options,
            )
        ds = gdal.Warp(tmp_output, mosaic_file, options=warp_options)
        if ds is None:
            raise RuntimeError("gdal.Warp returned None — clip failed.")
        if to_bbox:
            _fill_nodata_with_zero(ds, chunk_size)
        ds.FlushCache()
        ds = None
        if output_file is not None:
            os.replace(tmp_output, output_file)
        else:
            os.replace(tmp_output, mosaic_file)
    except Exception:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        raise
    finally:
        if tmp_geojson is not None and os.path.exists(tmp_geojson.name):
            os.remove(tmp_geojson.name)


if __name__ == "__main__":
    # Example usage
    import geopandas as gpd

    mosaic_file = "/workspace/test_domain_3/mhm_input/dem/mosaic.tif"
    perimeter_file = "/workspace/test_domain_3/mhm_input/domain/huc4_1211.geojson"
    output_file = "/workspace/test_domain_3/mhm_input/dem/mosaic_clipped.tif"

    perimeter = gpd.read_file(perimeter_file)
    clip_mosaic(mosaic_file, perimeter, output_file=output_file)
