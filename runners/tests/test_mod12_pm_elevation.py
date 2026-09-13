import numpy as np
import xarray as xr

from runners.mod12_pet_to_mhm.main import _compute_pm_inputs, _load_l2_elevation


def _template():
    return xr.DataArray(
        np.zeros((2, 2, 3), dtype=np.float32),
        dims=("time", "y", "x"),
        coords={
            "time": np.array([0, 1]),
            "y": np.array([750.0, 250.0]),
            "x": np.array([250.0, 750.0, 1250.0]),
        },
        name="tavg",
    )


def _write_dem(path, values, x=None):
    template = _template()
    xr.Dataset(
        {
            "dem": (
                ("y", "x"),
                np.asarray(values, dtype=np.float32),
            )
        },
        coords={
            "y": template.y.values,
            "x": template.x.values if x is None else np.asarray(x),
        },
    ).to_netcdf(path)


def test_load_l2_elevation_returns_broadcastable_aligned_array(tmp_path):
    path = tmp_path / "dem.nc"
    values = np.arange(6, dtype=np.float32).reshape(2, 3) + 100.0
    _write_dem(path, values)

    elevation = _load_l2_elevation(str(path), _template())

    assert elevation.shape == (1, 2, 3)
    np.testing.assert_array_equal(elevation[0], values)


def test_load_l2_elevation_rejects_shifted_coordinates(tmp_path):
    path = tmp_path / "dem_shifted.nc"
    _write_dem(path, np.ones((2, 3)), x=[251.0, 751.0, 1251.0])

    try:
        _load_l2_elevation(str(path), _template())
    except ValueError as exc:
        assert "'x' coordinates do not match" in str(exc)
    else:
        raise AssertionError("Shifted DEM coordinates were accepted")


def test_load_l2_elevation_rejects_missing_cells(tmp_path):
    path = tmp_path / "dem_missing.nc"
    values = np.ones((2, 3), dtype=np.float32)
    values[0, 1] = np.nan
    _write_dem(path, values)

    try:
        _load_l2_elevation(str(path), _template())
    except ValueError as exc:
        assert "1 missing or non-finite cells" in str(exc)
    else:
        raise AssertionError("DEM with a missing cell was accepted")


def test_compute_pm_inputs_varies_gamma_with_elevation():
    shape = (1, 1, 2)
    tavg = np.full(shape, 20.0)
    rhavg = np.full(shape, 50.0)
    ssrd = np.full(shape, 1.0e6)
    strd = np.full(shape, 0.5e6)
    windspeed = np.full(shape, 3.0)
    elevation = np.array([[[0.0, 2000.0]]])

    result = _compute_pm_inputs(
        tavg, rhavg, ssrd, strd, windspeed, 3600.0, elevation
    )

    expected_pressure = 101.325 * (1.0 - 2.2577e-5 * elevation) ** 5.2568
    np.testing.assert_allclose(result["gamma"], 0.000665 * expected_pressure)
    assert result["gamma"][0, 0, 0] > result["gamma"][0, 0, 1]
