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

# Precipitation is the only variable zero-filled for whole-missing timesteps
# (no rain assumed). States and radiation fluxes are continuous physical
# quantities that are never zero over a full step, so they are interpolated in
# time instead — zero-filling radiation injects a spurious energy deficit that
# drives Penman-Monteith net radiation negative and collapses PET.
_STATE_VARS = ("tavg", "rhavg", "windspeed")
_ACCUM_VARS = ("pre", "ssrd", "strd")
_ZEROFILL_VARS = ("pre",)


def _gap_fill_series(
    da: xr.DataArray, var: str, nodata: float, crs: str | None
) -> xr.DataArray:
    """Fill nodata gaps so no sentinel survives inside the domain mask.

    Spatially fills edge/partial gaps per timestep from nearest valid neighbours,
    then temporally fills whole-missing timesteps: precipitation is zero-filled
    (no rain assumed), while states and radiation fluxes (tavg, rhavg, windspeed,
    ssrd, strd) are interpolated in time to avoid inventing an energy deficit.
    """
    da = da.where(da != nodata).load()

    n_missing = int(np.isnan(da).sum())
    empty_steps = int(np.isnan(da).all(dim=("y", "x")).sum())

    if n_missing:
        filled = da
        if crs is not None:
            filled = filled.rio.write_crs(crs)
        # rioxarray keys "missing" off rio.nodata; point it at NaN so the
        # gaps we just unmasked are the cells that get interpolated.
        filled = filled.rio.write_nodata(np.nan).rio.interpolate_na(method="nearest")

        n_after_spatial = int(np.isnan(filled).sum())
        if var in _ZEROFILL_VARS:
            filled = filled.fillna(0.0)
        else:
            filled = (
                filled.interpolate_na(dim="time", method="linear")
                .ffill("time")
                .bfill("time")
            )

        residual = int(np.isnan(filled).sum())
        log.info(
            "gap-fill %s: filled %d cells (spatial) + %d cells across %d "
            "empty timesteps (temporal); residual nodata: %d",
            var,
            n_missing - n_after_spatial,
            n_after_spatial,
            empty_steps,
            residual,
        )
        da = filled

    return da.drop_vars("spatial_ref", errors="ignore")


def _sanitize_attr_value(value):
    """Convert attrs to NetCDF-safe scalar/list-like values."""
    if isinstance(value, dict):
        return json.dumps(value, default=str, sort_keys=True)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(
        value, (str, bytes, numbers.Number, np.number, np.ndarray, list, tuple)
    ):
        return value
    return str(value)


def _sanitize_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Return a copy with attrs normalized for NetCDF serialization."""
    out = ds.copy(deep=False)
    out.attrs = {k: _sanitize_attr_value(v) for k, v in out.attrs.items()}

    for name in out.variables:
        out[name].attrs = {
            k: _sanitize_attr_value(v) for k, v in out[name].attrs.items()
        }

    return out


def write_meteo(
    ds: xr.Dataset,
    out_path: Path,
    ref_time: pd.Timestamp,
    nodata: float,
    crs: str | None = None,
) -> None:
    """Write a single-variable meteo NetCDF (pre.nc or tavg.nc)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if crs is not None:
        ds = ds.rio.write_crs(crs)

    var = list(ds.data_vars)[0]
    # mHM mis-reads daily data encoded as "hours since" (dt=24); use the unit
    # giving an integer step of 1 (days for daily/monthly, hours for hourly).
    _t = pd.DatetimeIndex(ds["time"].values)
    _tu = "days" if (len(_t) < 2 or (_t[1] - _t[0]) >= pd.Timedelta("1D")) else "hours"
    encoding = {
        var: {
            "dtype": "f8",
            "_FillValue": nodata,
            "zlib": True,
            "complevel": 4,
        },
        "time": {
            "dtype": "i4",
            "units": f"{_tu} since {ref_time:%Y-%m-%d %H:%M:%S}",
            "calendar": "standard",
        },
    }
    ds_safe = _sanitize_dataset_attrs(ds)
    # _FillValue is owned by encoding; drop any stray attr (e.g. left by
    # mask_and_scale=False reads) so xarray does not raise on the conflict.
    for name in ds_safe.variables:
        ds_safe[name].attrs.pop("_FillValue", None)
    ds_safe.to_netcdf(out_path, encoding=encoding, format="NETCDF4")
    log.info("Wrote %s", out_path)


def finalize_variable(
    temp_files: list[Path],
    out_path: Path,
    ref_time: pd.Timestamp,
    nodata: float,
    crs: str | None = None,
) -> None:
    """Concatenate per-batch temp NetCDFs along time and write the final meteo file.

    ``mask_and_scale=False`` keeps the nodata sentinel as data so ``write_meteo``
    is the only place a ``_FillValue`` is applied (no double masking).
    """
    ordered = sorted(temp_files)
    if not ordered:
        raise ValueError(f"No temp files to concatenate for {out_path}")

    ds = xr.open_mfdataset(
        ordered,
        combine="nested",
        concat_dim="time",
        mask_and_scale=False,
        decode_times=True,
    )
    try:
        var = list(ds.data_vars)[0]
        ds[var] = _gap_fill_series(ds[var], var, nodata, crs)
        write_meteo(ds, out_path, ref_time, nodata, crs=crs)
    finally:
        ds.close()


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
