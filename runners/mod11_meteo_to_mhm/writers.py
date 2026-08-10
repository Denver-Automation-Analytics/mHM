"""Output writers for mHM meteo NetCDFs and header.txt files."""

from __future__ import annotations
import json
import logging
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  # registers .rio accessor

log = logging.getLogger(__name__)


def _sanitize_attr_value(value):
    """Convert attrs to NetCDF-safe scalar/list-like values."""
    if isinstance(value, dict):
        return json.dumps(value, default=str, sort_keys=True)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (str, bytes, numbers.Number, np.number, np.ndarray, list, tuple)):
        return value
    return str(value)


def _sanitize_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Return a copy with attrs normalized for NetCDF serialization."""
    out = ds.copy(deep=False)
    out.attrs = {k: _sanitize_attr_value(v) for k, v in out.attrs.items()}

    for name in out.variables:
        out[name].attrs = {k: _sanitize_attr_value(v) for k, v in out[name].attrs.items()}

    return out


def write_meteo(ds: xr.Dataset,
                out_path: Path,
                ref_time: pd.Timestamp,
                nodata: float,
                crs: str | None = None) -> None:
    """Write a single-variable meteo NetCDF (pre.nc or tavg.nc)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if crs is not None:
        ds = ds.rio.write_crs(crs)

    var = list(ds.data_vars)[0]
    encoding = {
        var: {
            "dtype":     "f8",
            "_FillValue": nodata,
            "zlib":       True,
            "complevel":  4,
        },
        "time": {
            "dtype":    "i4",
            "units":    f"hours since {ref_time:%Y-%m-%d %H:%M:%S}",
            "calendar": "standard",
        },
    }
    ds_safe = _sanitize_dataset_attrs(ds)
    ds_safe.to_netcdf(out_path, encoding=encoding, format="NETCDF4")
    log.info("Wrote %s", out_path)


def write_header_txt(header: dict, out_path: Path) -> None:
    """Write an mHM header.txt with the six required fields."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"ncols        {header['ncols']}",
        f"nrows        {header['nrows']}",
        f"xllcorner    {header['xllcorner']}",
        f"yllcorner    {header['yllcorner']}",
        f"cellsize     {header['cellsize']}",
        f"NODATA_value {header['NODATA_value']}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    log.info("Wrote %s", out_path)