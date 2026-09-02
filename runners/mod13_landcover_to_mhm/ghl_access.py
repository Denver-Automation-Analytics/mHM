"""Access utilities for the Space Intelligence GHL land cover via Arraylake."""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import xarray as xr
from arraylake import Client
import zarr

log = logging.getLogger(__name__)


def open_ghl(repo_name: str, ref: str = "main") -> xr.Dataset:
    """
    Open the GHL Icechunk repo (subscription mirror) via Arraylake.

    Access pattern (matches the recommendation returned when subscribing):

        client  = Client()
        repo    = client.get_repo(repo_name)
        session = repo.readonly_session(branch=ref)
        root    = zarr.open_group(session.store, zarr_format=3, mode="r")
        ds      = xr.open_zarr(session.store, zarr_format=3, consolidated=False)

    Authentication is handled by the Arraylake client's built-in auth cache
    (`arraylake auth login`). No token is passed here.

    Parameters
    ----------
    repo_name : str
        Fully-qualified Arraylake repo, e.g.
        "mbi-daa/global-harmonised-layers-2024-subscription".
        Typically read from the GHL_REPO env var in main.py.
    ref : str, optional
        Branch, tag, or snapshot id. Default "main". Pin a snapshot id for
        reproducible production runs.
    """

    client = Client()
    repo = client.get_repo(repo_name)
    session = repo.readonly_session(branch=ref)
    store = session.store

    # Sanity: open with zarr first so we surface any v3 metadata issues clearly
    # and can log the group hierarchy before xarray reads it.
    root = zarr.open_group(store, zarr_format=3, mode="r")
    log.info(
        "Opened %s @ %s; top-level arrays: %s", repo_name, ref, list(root.array_keys())
    )
    log.info("Top-level groups: %s", list(root.group_keys()))

    # xarray consumes the same Icechunk-backed store
    ds = xr.open_zarr(store, zarr_format=3, consolidated=False)
    log.info("Dataset dims: %s | vars: %s", dict(ds.sizes), list(ds.data_vars))
    return ds


def detect_class_var(ds: xr.Dataset) -> str:
    """
    Pick the GHL class variable automatically.

    Heuristic:
      * integer dtype (uint8 / int8 / int16)
      * has (y, x) or (lat, lon) among its dims
      * prefers variables named 'land_cover' / 'lc' / 'class' / 'map'
    """
    spatial_dims = {"y", "x", "lat", "lon", "latitude", "longitude"}
    candidates = [
        v
        for v in ds.data_vars
        if np.issubdtype(ds[v].dtype, np.integer) and set(ds[v].dims) & spatial_dims
    ]
    if not candidates:
        raise RuntimeError(
            f"No integer-typed spatial variable found in {list(ds.data_vars)}. "
            "Set GHL_VARIABLE explicitly in main.py."
        )

    preferred = ("land_cover", "landcover", "lc", "class", "classification", "map")
    lc = [c for c in candidates if any(p in c.lower() for p in preferred)]
    chosen = lc[0] if lc else candidates[0]
    log.info("Auto-detected GHL class variable: %s", chosen)
    return chosen


def select_scene(ds: xr.Dataset, var_name: str, year: Optional[int]) -> xr.DataArray:
    """
    Return a single-scene DataArray for `year`.

    Handles three cases:
      * Static dataset (no time dim)          -> returned as-is.
      * time indexed by datetime64            -> .sel(time=str(year))
      * time indexed by integer years         -> .sel(time=year)
    """
    da = ds[var_name]

    if "time" not in da.dims:
        if year is not None:
            log.warning("GHL variable has no 'time' dim; ignoring year=%s", year)
        return da

    time_vals = da["time"].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        return da.sel(time=str(year)).squeeze("time", drop=True)
    if np.issubdtype(time_vals.dtype, np.integer):
        return da.sel(time=year).squeeze("time", drop=True)

    return da.sel(time=str(year)).squeeze("time", drop=True)
