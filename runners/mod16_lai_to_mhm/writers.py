"""Write mHM-compatible lai.nc from a MODIS LAI snapshot dataset."""

from __future__ import annotations

import json
import logging
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# mHM rejects lai <= 1e-10 (mo_read_nc.f90 lower bound); use a safe margin
_LAI_MIN = 1e-6
_LAI_MAX = 30.0


def _sanitize_attr(value):
    if isinstance(value, dict):
        return json.dumps(value, default=str, sort_keys=True)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (str, bytes, numbers.Number, np.number, np.ndarray, list, tuple)):
        return value
    return str(value)


def _sanitize_ds(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy(deep=False)
    out.attrs = {k: _sanitize_attr(v) for k, v in out.attrs.items()}
    for name in out.variables:
        out[name].attrs = {k: _sanitize_attr(v) for k, v in out[name].attrs.items()}
    return out


def write_lai_nc(
    snapshot_ds: xr.Dataset,
    out_path: Path,
    ref_month: str | pd.Timestamp,
    nodata: float = -9999.0,
) -> None:
    """
    Write a 12-month flat-climatology lai.nc for mHM timeStep_LAI_input = -2.

    ``snapshot_ds`` must contain a 2-D ``Lai`` variable (y, x) in m2/m2,
    as returned by ``acquire_lai_map``. The same snapshot is tiled for every
    month of the reference year so mHM's monthly reader sees ≥ 2 time steps
    and the file covers any simulation period within that year.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref_year = pd.Timestamp(ref_month).year
    monthly_times = pd.date_range(f"{ref_year}-01-01", periods=12, freq="MS")

    lai_base = snapshot_ds["Lai"].clip(_LAI_MIN, _LAI_MAX)
    # Fill cloud/QC gaps with domain median so mHM finds no nodata within its mask.
    # Load into memory first: dask's nanmedian can't reduce over all axes at once.
    domain_median = float(np.nanmedian(lai_base.values))
    if np.isnan(domain_median):
        domain_median = 1.0  # fallback when the whole snapshot is missing
    lai_base = lai_base.fillna(domain_median)

    # Repeat the snapshot for each month to build a flat annual climatology.
    lai = (
        xr.concat([lai_base] * 12, dim=pd.DatetimeIndex(monthly_times, name="time"))
        .rename("lai")                       # Fortran reader looks for variable 'lai'
        .to_dataset()
    )

    lai.attrs["Conventions"] = "CF-1.6"
    lai["lai"].attrs.update(
        long_name="Leaf Area Index",
        units="m2 m-2",
        missing_value=nodata,
    )

    encoding = {
        "lai": {
            "dtype":      "f4",
            "_FillValue": nodata,
            "zlib":       True,
            "complevel":  4,
        },
        "time": {
            "dtype":    "i4",
            "units":    "hours since 1900-01-01 00:00:00",
            "calendar": "standard",
        },
    }

    _sanitize_ds(lai).to_netcdf(out_path, encoding=encoding, format="NETCDF4")
    log.info("Wrote %s (12-month climatology, year %d)", out_path, ref_year)
