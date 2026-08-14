"""Write PET NetCDF and companion header.txt following the mHM meteo convention."""

from __future__ import annotations
import json
import logging
import numbers
from pathlib import Path

import netCDF4 as nc4
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
            "units":    f"{'days' if stat_freq == 'daily' else 'hours'} since {ref_time:%Y-%m-%d %H:%M:%S}",
            "calendar": "standard",
        },
    }
    _sanitize_ds(pet_ds).to_netcdf(out_path, encoding=encoding, format="NETCDF4")
    log.info("Wrote %s", out_path)


def create_pet_nc(
    out_path: Path,
    template_ds: xr.Dataset,
    tavg_var: str,
    ref_time: pd.Timestamp,
    nodata: float = -9999.0,
    stat_freq: str = "hourly",
    nc_chunk_t: int = 24,
) -> None:
    """Create an empty, unlimited-time pet.nc; fill with write_pet_chunk()."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    template_var = template_ds[tavg_var]
    dims = list(template_var.dims)           # e.g. ["time", "y", "x"]
    spatial_dims = [d for d in dims if d != "time"]

    shape = template_var.shape
    spatial_shape = tuple(template_ds.sizes[d] for d in spatial_dims)

    units = "mm" if stat_freq == "daily" else "mm h-1"

    # resolve grid_mapping variable name from template
    gm_name = template_var.attrs.get("grid_mapping")
    if gm_name is None:
        for col in (template_ds.data_vars, template_ds.coords):
            for v in col:
                if v not in dims and template_ds[v].ndim == 0:
                    gm_name = v
                    break
            if gm_name:
                break

    with nc4.Dataset(out_path, "w", format="NETCDF4") as f:
        f.Conventions = "CF-1.6"

        f.createDimension("time", None)  # unlimited
        for dim in spatial_dims:
            f.createDimension(dim, template_ds.sizes[dim])

        # time variable
        tv = f.createVariable("time", "i4", ("time",))
        tv.units    = f"{'days' if stat_freq == 'daily' else 'hours'} since {ref_time:%Y-%m-%d %H:%M:%S}"
        tv.calendar = "standard"
        tv.axis     = "T"

        # spatial coordinate variables
        for dim in spatial_dims:
            if dim in template_ds.coords:
                coord = template_ds[dim]
                cv = f.createVariable(dim, "f8", (dim,))
                for attr in ("units", "long_name", "standard_name", "axis"):
                    if attr in coord.attrs:
                        setattr(cv, attr, coord.attrs[attr])
                cv[:] = coord.values

        # scalar CRS variable
        if gm_name and gm_name in {**dict(template_ds.data_vars), **dict(template_ds.coords)}:
            crs_src = template_ds[gm_name]
            crs_v = f.createVariable(gm_name, "i4")
            for k, v in crs_src.attrs.items():
                try:
                    setattr(crs_v, k, _sanitize_attr(v))
                except Exception:
                    pass

        # pet variable
        chunk_t = min(nc_chunk_t, 1)  # at least 1; actual size set here
        pv = f.createVariable(
            "pet", "f4", ("time", *spatial_dims),
            fill_value=np.float32(nodata),
            zlib=True, complevel=4,
            chunksizes=(nc_chunk_t, *spatial_shape),
        )
        pv.units         = units
        pv.long_name     = "potential evapotranspiration"
        pv.standard_name = "water_potential_evaporation_amount"
        pv.missing_value = np.float32(nodata)
        if gm_name:
            pv.grid_mapping = gm_name

    log.info("Created %s (unlimited time, nc_chunk_t=%d)", out_path, nc_chunk_t)


def write_pet_chunk(
    out_path: Path,
    chunk_data: np.ndarray,
    time_offsets: np.ndarray,
    time_start_idx: int,
) -> None:
    """Append one chunk of PET data to a file previously created by create_pet_nc()."""
    t_end = time_start_idx + len(chunk_data)
    with nc4.Dataset(out_path, "a") as f:
        f["pet"][time_start_idx:t_end, :, :]  = chunk_data
        f["time"][time_start_idx:t_end]        = time_offsets


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
