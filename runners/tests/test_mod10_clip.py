import geopandas as gpd
import numpy as np
from osgeo import gdal, osr
from shapely.geometry import Polygon, box

from runners.mod10_dem_to_mhm.clip import NODATA, clip_mosaic


def _write_dem(path, data, nodata):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, data.shape[0], 0, -1))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(3857)
    ds.SetProjection(spatial_ref.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    ds = None


def _perimeter(geometry):
    return gpd.GeoDataFrame({"id": [1]}, geometry=[geometry], crs="EPSG:3857")


def _read_band(path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    data_type = band.DataType
    ds = None
    return data, nodata, data_type


def test_bbox_replaces_source_nodata_with_valid_zero(tmp_path):
    source = tmp_path / "source.tif"
    output = tmp_path / "clipped.tif"
    source_nodata = -32768.0
    data = np.array(
        [
            [source_nodata, 2, 3, 4],
            [5, -2.5, 7, 8],
            [9, 10, source_nodata, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.float32,
    )
    _write_dem(source, data, source_nodata)

    clip_mosaic(
        str(source),
        _perimeter(box(0, 0, 4, 4)),
        output_file=str(output),
        to_bbox=True,
        chunk_size=2,
    )

    result, output_nodata, data_type = _read_band(output)
    assert result[0, 0] == 0
    assert result[2, 2] == 0
    assert result[1, 1] == -2.5
    assert output_nodata == NODATA
    assert output_nodata != 0
    assert data_type == gdal.GDT_Float32


def test_bbox_replaces_nan_nodata_with_zero(tmp_path):
    source = tmp_path / "source_nan.tif"
    output = tmp_path / "clipped_nan.tif"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    data[1, 2] = np.nan
    _write_dem(source, data, np.nan)

    clip_mosaic(
        str(source),
        _perimeter(box(0, 0, 4, 4)),
        output_file=str(output),
        to_bbox=True,
        chunk_size=2,
    )

    result, output_nodata, _ = _read_band(output)
    assert result[1, 2] == 0
    assert not np.isnan(result).any()
    assert output_nodata == NODATA


def test_polygon_clip_preserves_cutline_nodata(tmp_path):
    source = tmp_path / "source_polygon.tif"
    output = tmp_path / "clipped_polygon.tif"
    data = np.arange(36, dtype=np.float32).reshape(6, 6) + 1
    _write_dem(source, data, None)
    triangle = Polygon([(0, 0), (6, 0), (0, 6)])

    clip_mosaic(
        str(source),
        _perimeter(triangle),
        output_file=str(output),
        to_bbox=False,
        chunk_size=2,
    )

    result, output_nodata, _ = _read_band(output)
    assert output_nodata == NODATA
    assert np.any(result == NODATA)
    assert np.any(result > 0)