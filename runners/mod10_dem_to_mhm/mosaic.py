"""
mosaic_rasters.py

For each subfolder in the Fathom Data directory, mosaics all constituent
GeoTIFF tiles into a single Cloud Optimized GeoTIFF (COG) and saves it flat
to the Mosaic output directory, named after the subfolder.

Behaviour:
    - Skips subfolders with no .tif files
    - Skips output files that already exist
    - Preserves source CRS, nodata value, and data type in the output COG
    - COG creation options: DEFLATE compression, tiled, overviews built by driver

Processing applied to every pixel before mosaicking:
    1. Pixels with value < 0 (excluding nodata) are replaced with the nodata value.
"""

import glob
import logging
import os
import tempfile
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# ----------------------------- Configuration ----------------------------------

COG_CREATION_OPTIONS = [
    "COMPRESS=DEFLATE",
    "PREDICTOR=2",
    "BIGTIFF=IF_SAFER",
    "RESAMPLING=NEAREST",
    "OVERVIEW_RESAMPLING=NEAREST",
]

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("mosaic_rasters")

# ----------------------------- Helpers ---------------------------------------

_TIFF_MAGIC = {b"II\x2a\x00", b"MM\x00\x2a", b"II\x2b\x00", b"MM\x00\x2b"}


def _is_tiff(path: str) -> bool:
    """Return True if *path* starts with a TIFF magic number."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) in _TIFF_MAGIC
    except OSError:
        return False


def _process_tile(
    src_path: str,
    dst_path: str,
    nodata_value: float | None,
) -> None:
    """Write a processed copy of *src_path* (GTiff) to *dst_path*.

    Processing steps applied per band:
        Pixels with value < 0 (and not equal to nodata) are replaced with nodata.
    """
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        dst_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        src_ds.RasterCount,
        gdal.GDT_Float32,
    )
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())

    nd = np.float32(nodata_value) if nodata_value is not None else None

    for band_idx in range(1, src_ds.RasterCount + 1):
        data = src_ds.GetRasterBand(band_idx).ReadAsArray().astype(np.float32)

        nodata_mask = (
            (data == nd) if nd is not None else np.zeros(data.shape, dtype=bool)
        )
        negative_mask = (data < 0) & ~nodata_mask

        # Replace out-of-range negatives with nodata (or 0 if no nodata defined)
        data[negative_mask] = nd if nd is not None else np.float32(0.0)

        out_band = out_ds.GetRasterBand(band_idx)
        out_band.WriteArray(data)
        if nd is not None:
            out_band.SetNoDataValue(float(nd))
        out_band.FlushCache()
        out_band = None

    src_ds = None
    out_ds.FlushCache()
    out_ds = None


# ----------------------------- Core function ----------------------------------


def mosaic_tiles(tiles, output_path: str) -> None:
    """Mosaic all .tif files in *tiles* into a single COG at *output_path*.

    *tiles* may be a directory to search recursively or an iterable of tile paths.
    """

    if isinstance(tiles, (str, os.PathLike)):
        candidates = glob.glob(os.path.join(os.fspath(tiles), "**", "*"), recursive=True)
    else:
        candidates = [os.fspath(t) for t in tiles]

    tif_files = sorted(
        f
        for f in candidates
        if not f.lower().endswith(".aux.xml") and _is_tiff(f)
    )

    if not tif_files:
        log.warning("No .tif files found in %s — skipping.", tiles)
        return

    log.info("  Tiles found : %d", len(tif_files))

    # --- Read metadata from the first tile -----------------------------------
    src_ds = gdal.Open(tif_files[0], gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Could not open {tif_files[0]} with GDAL.")

    nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()
    src_ds = None  # close

    # --- Process each tile -------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        processed_files = []
        for tif in tif_files:
            dst = os.path.join(tmp_dir, os.path.basename(tif))
            _process_tile(tif, dst, nodata_value)
            processed_files.append(dst)

        # --- Build VRT from processed tiles ----------------------------------
        vrt_path = os.path.join(tmp_dir, "mosaic.vrt")
        vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
        vrt_ds = gdal.BuildVRT(vrt_path, processed_files, options=vrt_options)
        if vrt_ds is None:
            raise RuntimeError(f"gdal.BuildVRT failed for {subfolder_path}.")
        vrt_ds.FlushCache()
        vrt_ds = None  # close before Translate reads it from disk

        # --- Translate VRT → COG ---------------------------------------------
        translate_options = gdal.TranslateOptions(
            format="COG",
            creationOptions=COG_CREATION_OPTIONS,
            noData=nodata_value,
        )
        out_ds = gdal.Translate(output_path, vrt_path, options=translate_options)
        if out_ds is None:
            raise RuntimeError(f"gdal.Translate failed for {output_path}.")
        out_ds.FlushCache()
        out_ds = None  # close / finalise file

    log.info("  Output      : %s", output_path)

# ----------------------------- Main ------------------------------------------


if __name__ == "__main__":
    DATA_DIR = "/workspace/test_domain_3/mhm_input/dem"
    OUTPUT_PATH = "/workspace/test_domain_3/mhm_input/dem/mosaic.tif"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    mosaic_tiles(DATA_DIR, OUTPUT_PATH)