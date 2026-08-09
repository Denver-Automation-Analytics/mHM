"""Transform the clipped HRRR subset into an mHM-compliant xarray Dataset."""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def format_for_mhm(ds_clip: xr.Dataset,
                   init_time: pd.Timestamp,
                   nodata: float) -> tuple[xr.Dataset, pd.Timestamp]:
    """
    Rename variables to mHM's hard-coded names, ensure DOUBLE, build a single
    time dimension (init_time + lead_time), and set fill values / attrs.
    Returns (ds_mhm, reference_time).
    """
    # Rename to mHM's required variable names
    ds = ds_clip.rename({
        "temperature_2m":                             "tavg",
        "precipitation_surface":                      "pre",
        "relative_humidity_2m":                       "rhavg",
        "downward_long_wave_radiation_flux_surface":  "strd",
        "downward_short_wave_radiation_flux_surface": "ssrd",
    })
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

    # Frequency-aware radiation conversion: W m-2 → J m-2 per timestep
    if len(valid_time) >= 2:
        dt_s = float((valid_time[1] - valid_time[0]).total_seconds())
    else:
        log.warning("Single-timestep dataset — falling back to dt_s = 3600 s for radiation conversion")
        dt_s = 3600.0
    for rad_var in ("ssrd", "strd"):
        ds[rad_var] = ds[rad_var] * dt_s
    # Cast data variables to DOUBLE and fill NaN with nodata
    for v in ds.data_vars:
        ds[v] = ds[v].astype("float64").fillna(nodata)

    # CF-style attributes
    ds["pre"].attrs.update({
        "units":         "mm",
        "long_name":     "precipitation",
        "standard_name": "precipitation_amount",
        "missing_value": nodata,
    })
    ds["tavg"].attrs.update({
        "units":         "degC",
        "long_name":     "average air temperature",
        "standard_name": "air_temperature",
        "missing_value": nodata,
    })
    ds["windspeed"].attrs.update({
        "units":         "m s-1",
        "long_name":     "10 m wind speed magnitude",
        "standard_name": "wind_speed",
        "missing_value": nodata,
    })
    ds["rhavg"].attrs.update({
        "units":         "%",
        "long_name":     "relative humidity 2 m",
        "standard_name": "relative_humidity",
        "missing_value": nodata,
    })
    ds["ssrd"].attrs.update({
        "units":         "J m-2",
        "long_name":     "surface downwelling shortwave radiation",
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "cell_methods":  "time: sum",
        "missing_value": nodata,
    })
    ds["strd"].attrs.update({
        "units":         "J m-2",
        "long_name":     "surface downwelling longwave radiation",
        "standard_name": "surface_downwelling_longwave_flux_in_air",
        "cell_methods":  "time: sum",
        "missing_value": nodata,
    })

    ref_time = pd.Timestamp(valid_time[0])
    return ds, ref_time