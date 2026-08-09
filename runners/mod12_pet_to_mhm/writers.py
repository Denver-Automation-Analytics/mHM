"""Write PET NetCDF and companion header.txt following the mHM meteo convention."""

from __future__ import annotations
import json
import logging
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


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


def write_pet(
    pet_data: np.ndarray,
    template_ds: xr.Dataset,
    tavg_var: str,
    out_path: Path,
    ref_time: pd.Timestamp,
    nodata: float = -9999.0,
    stat_freq: str = "hourly",
) -> None:
    """Write *pet_data* as a CF-compliant NetCDF using *template_ds* for coordinates."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    template_var = template_ds[tavg_var]
    dims = list(template_var.dims)          # e.g. ["time", "y", "x"]
    ntime = pet_data.shape[0]
    coords = {
        d: (template_ds[d].values[:ntime] if d == "time" else template_ds[d])
        for d in dims
        if d in template_ds.coords
    }

    units = "mm" if stat_freq == "daily" else "mm h-1"
    pet_attrs: dict = {
        "units": units,
        "long_name": "potential evapotranspiration",
        "standard_name": "water_potential_evaporation_amount",
        "missing_value": nodata,
    }
    if "grid_mapping" in template_var.attrs:
        pet_attrs["grid_mapping"] = template_var.attrs["grid_mapping"]

    pet_da = xr.DataArray(pet_data, dims=dims, coords=coords, name="pet", attrs=pet_attrs)
    pet_ds = pet_da.to_dataset()

    # carry scalar CRS variables (spatial_ref / grid_mapping) from data_vars and coords
    crs_vars = []
    for collection in (template_ds.data_vars, template_ds.coords):
        for var in collection:
            if var not in pet_ds and var not in dims and template_ds[var].ndim == 0:
                pet_ds[var] = template_ds[var]
                crs_vars.append(var)
    # link pet to the CRS variable via grid_mapping if not already set
    if crs_vars and "grid_mapping" not in pet_attrs:
        pet_ds["pet"].attrs["grid_mapping"] = crs_vars[0]

    encoding = {
        "pet": {
            "dtype":      "f8",
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
    _sanitize_ds(pet_ds).to_netcdf(out_path, encoding=encoding, format="NETCDF4")
    log.info("Wrote %s", out_path)


def write_header_txt(header: dict, out_path: Path) -> None:
    """Write an mHM header.txt with the six required fields."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"ncols        {header['ncols']}",
        f"nrows        {header['nrows']}",
        f"xllcorner    {header['xllcorner']}",
        f"yllcorner    {header['yllcorner']}",
        f"cellsize     {int(header['cellsize'])}",
        f"NODATA_value {int(header['NODATA_value'])}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    log.info("Wrote %s", out_path)
