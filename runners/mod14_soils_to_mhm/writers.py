"""Write per-layer ArcGIS ASCII grids and optional NetCDF QA copies."""

from __future__ import annotations
import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np

log = logging.getLogger(__name__)


def write_ascii(path: Path, header: dict, grid: np.ndarray, nodata: int = -9999) -> None:
    """Write a single ArcGIS ASCII grid readable by mHM and the Fortran prep code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if grid.shape != (header["nrows"], header["ncols"]):
        raise AssertionError(
            f"Grid shape {grid.shape} does not match header "
            f"({header['nrows']}, {header['ncols']})."
        )
    header_lines = [
        f"ncols        {header['ncols']}",
        f"nrows        {header['nrows']}",
        f"xllcorner    {header['xllcorner']}",
        f"yllcorner    {header['yllcorner']}",
        f"cellsize     {int(header['cellsize'])}",
        f"NODATA_value {nodata}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(header_lines) + "\n")
        np.savetxt(fh, grid.astype(np.int32), fmt="%d")
    log.info("Wrote %s", path)


def write_all_layers(
    out_dir: Path,
    header: dict,
    layers: dict[str, dict[int, np.ndarray]],
    nodata: int = -9999,
) -> None:
    """
    Write 18 ASCII grids: bd01-06.txt, cl01-06.txt, sn01-06.txt.
    These are the direct inputs to the LUT generator (lut.py).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for prop, prop_layers in layers.items():
        for layer_num in sorted(prop_layers):
            fname = f"{prop}{layer_num:02d}.txt"
            write_ascii(out_dir / fname, header, prop_layers[layer_num], nodata)


_PROP_UNITS = {"bd": "mg cm-3", "cl": "%", "sn": "%"}
_PROP_LONG  = {"bd": "bulk density", "cl": "clay fraction", "sn": "sand fraction"}


def write_layers_nc(
    out_dir: Path,
    header: dict,
    layers: dict[str, dict[int, np.ndarray]],
    nodata: int = -9999,
) -> None:
    """Write all 18 soil layers to a single NetCDF (bd, cl, sn × 6 layers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_nums = sorted(next(iter(layers.values())))
    out_path = out_dir / "soil_layers.nc"

    with nc.Dataset(out_path, "w") as ds:
        ds.xllcorner    = float(header["xllcorner"])
        ds.yllcorner    = float(header["yllcorner"])
        ds.cellsize     = int(header["cellsize"])
        ds.NODATA_value = int(nodata)

        ds.createDimension("layer", len(layer_nums))
        ds.createDimension("y",     header["nrows"])
        ds.createDimension("x",     header["ncols"])

        lv = ds.createVariable("layer", "i4", ("layer",))
        lv[:] = layer_nums

        for prop, prop_layers in layers.items():
            stack = np.stack([prop_layers[k] for k in layer_nums], axis=0).astype(np.int32)
            v = ds.createVariable(
                prop, "i4", ("layer", "y", "x"),
                fill_value=nodata, zlib=True,
            )
            v.units         = _PROP_UNITS[prop]
            v.long_name     = _PROP_LONG[prop]
            v.missing_value = nodata
            v[:] = stack

    log.info("Wrote %s", out_path)
