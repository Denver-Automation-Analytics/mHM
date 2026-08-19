import os
import tempfile
import geopandas as gpd
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


def clip_mosaic(
    mosaic_file: str,
    perimeter: gpd.GeoDataFrame,
    output_file: str | None = None,
    nodata: float = NODATA,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Clip a GeoTIFF in-place to the model perimeter polygon.

    Uses GDAL Warp with a cutline so the full raster never loads into Python
    memory — GDAL reads and writes in tiles controlled by chunk_size.
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
    """
    # Read source CRS so the cutline geometry is always co-registered
    src_ds = gdal.Open(mosaic_file, gdal.GA_ReadOnly)
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open {mosaic_file}")
    crs_wkt = src_ds.GetProjection()
    src_ds = None

    perimeter_reprojected = perimeter.to_crs(crs_wkt)

    tmp_geojson = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
    tmp_geojson.close()
    perimeter_reprojected.to_file(tmp_geojson.name, driver="GeoJSON")

    # float32 pixel × chunk grid × 8× headroom keeps the warper in a similar
    # memory envelope to the overflow tiled operations elsewhere in the pipeline
    warp_memory_bytes = chunk_size * chunk_size * 4 * 8

    tmp_output = mosaic_file + ".tmp.tif"
    try:
        warp_options = gdal.WarpOptions(
            cutlineDSName=tmp_geojson.name,
            cropToCutline=True,
            dstNodata=nodata,
            srcNodata=nodata,
            warpMemoryLimit=warp_memory_bytes,
            warpOptions=["CUTLINE_ALL_TOUCHED=TRUE"],
            creationOptions=COG_CREATION_OPTIONS,
        )
        ds = gdal.Warp(tmp_output, mosaic_file, options=warp_options)
        if ds is None:
            raise RuntimeError("gdal.Warp returned None — clip failed.")
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
        if os.path.exists(tmp_geojson.name):
            os.remove(tmp_geojson.name)

if __name__ == "__main__":
    # Example usage
    import geopandas as gpd

    mosaic_file = "/workspace/test_domain_3/mhm_input/dem/mosaic.tif"
    perimeter_file = "/workspace/test_domain_3/mhm_input/domain/huc4_1211.geojson"
    output_file = "/workspace/test_domain_3/mhm_input/dem/mosaic_clipped.tif"

    perimeter = gpd.read_file(perimeter_file)
    clip_mosaic(mosaic_file, perimeter, output_file=output_file)