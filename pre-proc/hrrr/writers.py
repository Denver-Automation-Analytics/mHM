"""Output writers for mHM meteo NetCDFs and header.txt files."""

from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def write_meteo(ds: xr.Dataset,
                out_path: Path,
                ref_time: pd.Timestamp,
                nodata: float) -> None:
    """Write a single-variable meteo NetCDF (pre.nc or tavg.nc)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    ds.to_netcdf(out_path, encoding=encoding, format="NETCDF4")
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