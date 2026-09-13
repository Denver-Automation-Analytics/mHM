import netCDF4 as nc4
import numpy as np
from osgeo import gdal, osr

from runners.mod10_dem_to_mhm.main import _write_l2_dem


def _write_l0_dem(path):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), 4, 4, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 250.0, 0.0, 1000.0, 0.0, -250.0))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(5070)
    ds.SetProjection(spatial_ref.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(np.arange(16, dtype=np.float32).reshape(4, 4))
    ds = None
    return spatial_ref.ExportToWkt()


def test_write_l2_dem_uses_exact_grid_and_average_values(tmp_path):
    dem_l0 = tmp_path / "dem_l0.tif"
    dem_l2 = tmp_path / "l2" / "dem_l2.tif"
    dem_nc = tmp_path / "meteo" / "dem" / "dem.nc"
    crs_wkt = _write_l0_dem(dem_l0)
    l0_header = {
        "ncols": 4,
        "nrows": 4,
        "xllcorner": 0.0,
        "yllcorner": 0.0,
        "cellsize": 250.0,
        "NODATA_value": -9999.0,
    }

    l2 = _write_l2_dem(
        str(dem_l0),
        l0_header,
        crs_wkt,
        str(dem_l2),
        str(dem_nc),
        l2_cellsize=500.0,
        block_size=32,
    )

    assert l2 == {
        "ncols": 2,
        "nrows": 2,
        "xllcorner": 0.0,
        "yllcorner": 0.0,
        "cellsize": 500.0,
        "NODATA_value": -9999.0,
    }
    with nc4.Dataset(dem_nc) as ds:
        assert ds["dem"].dimensions == ("y", "x")
        np.testing.assert_allclose(ds["x"][:], [250.0, 750.0])
        np.testing.assert_allclose(ds["y"][:], [750.0, 250.0])
        np.testing.assert_allclose(ds["dem"][:], [[2.5, 4.5], [10.5, 12.5]])
        assert ds["dem"].dtype == np.dtype("float32")
        assert ds["dem"]._FillValue == np.float32(-9999.0)
        assert ds["dem"].units == "m"
