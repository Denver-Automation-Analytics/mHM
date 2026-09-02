"""Transform the clipped HRRR subset into an mHM-compliant xarray Dataset."""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def format_for_mhm(
    ds_clip: xr.Dataset,
    init_time: pd.Timestamp,
    nodata: float,
    ref_time: pd.Timestamp | None = None,
) -> tuple[xr.Dataset, pd.Timestamp]:
    """
    Rename variables to mHM's hard-coded names, ensure DOUBLE, build a single
    time dimension (init_time + lead_time), and set fill values / attrs.

    ``ref_time`` pins the time-encoding origin so batched calls share one origin;
    when None it defaults to this window's first timestamp.
    Returns (ds_mhm, reference_time).
    """
    # Rename to mHM's required variable names
    ds = ds_clip.rename(
        {
            "temperature_2m": "tavg",
            "precipitation_surface": "pre",
            "relative_humidity_2m": "rhavg",
            "downward_long_wave_radiation_flux_surface": "strd",
            "downward_short_wave_radiation_flux_surface": "ssrd",
        }
    )
    ds["windspeed"] = np.sqrt(ds["wind_u_10m"] ** 2 + ds["wind_v_10m"] ** 2)
    ds = ds.drop_vars(["wind_u_10m", "wind_v_10m"])

    # Build a proper 1-D time coordinate.
    # Forecast-style: init_time + lead_time. Analysis-style: existing time coord.
    if "lead_time" in ds.coords or "lead_time" in ds.dims:
        lead_vals = ds["lead_time"].values
        if np.issubdtype(np.asarray(lead_vals).dtype, np.timedelta64):
            lead_td = pd.to_timedelta(lead_vals)
        else:
            lead_td = pd.to_timedelta(lead_vals, unit="s")
        valid_time = pd.to_datetime(init_time) + lead_td
        ds = ds.assign_coords(time=("lead_time", valid_time))
        ds = ds.swap_dims({"lead_time": "time"}).drop_vars("lead_time", errors="ignore")
    elif "time" in ds.coords:
        if "time" in ds.dims:
            valid_time = pd.to_datetime(ds["time"].values)
        else:
            valid_time = pd.DatetimeIndex([pd.to_datetime(ds["time"].values)])
            ds = ds.expand_dims(time=valid_time)
    else:
        valid_time = pd.DatetimeIndex([pd.to_datetime(init_time)])
        ds = ds.expand_dims(time=valid_time)

    # Frequency-aware rate → per-timestep conversion.
    if len(valid_time) >= 2:
        dt_s = float((valid_time[1] - valid_time[0]).total_seconds())
    else:
        log.warning(
            "Single-timestep dataset — falling back to dt_s = 3600 s for rate conversion"
        )
        dt_s = 3600.0
    # HRRR precipitation is a rate (kg m-2 s-1 = mm s-1) → mm per timestep.
    ds["pre"] = ds["pre"] * dt_s
    for rad_var in ("ssrd", "strd"):
        ds[rad_var] = ds[rad_var] * dt_s
    # Cast data variables to DOUBLE and fill NaN with nodata
    for v in ds.data_vars:
        ds[v] = ds[v].astype("float64").fillna(nodata)

    # CF-style attributes
    ds["pre"].attrs.update(
        {
            "units": "mm",
            "long_name": "precipitation",
            "standard_name": "precipitation_amount",
            "missing_value": nodata,
        }
    )
    ds["tavg"].attrs.update(
        {
            "units": "degC",
            "long_name": "average air temperature",
            "standard_name": "air_temperature",
            "missing_value": nodata,
        }
    )
    ds["windspeed"].attrs.update(
        {
            "units": "m s-1",
            "long_name": "10 m wind speed magnitude",
            "standard_name": "wind_speed",
            "missing_value": nodata,
        }
    )
    ds["rhavg"].attrs.update(
        {
            "units": "%",
            "long_name": "relative humidity 2 m",
            "standard_name": "relative_humidity",
            "missing_value": nodata,
        }
    )
    ds["ssrd"].attrs.update(
        {
            "units": "J m-2",
            "long_name": "surface downwelling shortwave radiation",
            "standard_name": "surface_downwelling_shortwave_flux_in_air",
            "cell_methods": "time: sum",
            "missing_value": nodata,
        }
    )
    ds["strd"].attrs.update(
        {
            "units": "J m-2",
            "long_name": "surface downwelling longwave radiation",
            "standard_name": "surface_downwelling_longwave_flux_in_air",
            "cell_methods": "time: sum",
            "missing_value": nodata,
        }
    )

    ref_time = (
        pd.Timestamp(valid_time[0]) if ref_time is None else pd.Timestamp(ref_time)
    )
    return ds, ref_time


# Per-variable reducers for hourly → daily aggregation.
_DAILY_SUM_VARS = ("pre", "ssrd", "strd")  # fluxes/accumulations: conserve totals
_DAILY_MEAN_VARS = ("tavg", "rhavg", "windspeed")  # state variables: daily average


def aggregate_to_daily(ds_mhm: xr.Dataset, nodata: float) -> xr.Dataset:
    """Aggregate an hourly mHM-formatted dataset to daily (calendar-day, UTC).

    Sums fluxes (pre, ssrd, strd) and averages states (tavg, rhavg, windspeed).
    Radiation is already J m-2 per hourly step, so summing yields J m-2 per day.
    """
    # Drop the nodata sentinel before reducing so it does not corrupt sums/means.
    ds_valid = ds_mhm.where(ds_mhm != nodata)

    sum_vars = [v for v in ds_mhm.data_vars if v in _DAILY_SUM_VARS]
    mean_vars = [v for v in ds_mhm.data_vars if v in _DAILY_MEAN_VARS]
    other = set(ds_mhm.data_vars) - set(sum_vars) - set(mean_vars)
    if other:
        raise KeyError(f"No daily reducer defined for variable(s) {sorted(other)}.")

    # min_count=1 so a fully-missing day sums to NaN (not 0), letting the
    # downstream temporal gap-fill interpolate it instead of leaving a spurious 0.
    daily_sum = ds_valid[sum_vars].resample(time="1D").sum(skipna=True, min_count=1)
    daily_mean = ds_valid[mean_vars].resample(time="1D").mean(skipna=True)
    ds_daily = xr.merge([daily_sum, daily_mean])

    for v in ds_daily.data_vars:
        ds_daily[v].attrs = dict(ds_mhm[v].attrs)
        ds_daily[v] = ds_daily[v].astype("float64").fillna(nodata)
    return ds_daily
