"""Writers for mod22: emit the TRITON ASCII input files from mHM/mRM data.

Large grids are streamed row-by-row so the full-resolution DEM and runoff map are
never held in memory at once.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NODATA

# Time block used when streaming the runoff cube out to the .roff file.
_ROFF_TIME_BLOCK = 365


def write_dem_and_rmap(grid: Dict, zone_ids: np.ndarray, ix1: np.ndarray,
                       iy1: np.ndarray, dem_path: Path, rmap_path: Path) -> None:
    """Write the ESRI-ASCII DEM and the matching integer runoff map in one pass.

    Both files share the DEM grid. A TRITON cell takes the zone id of the L1
    cell containing its centre, or 0 where the DEM is nodata or the cell falls
    outside the L1 grid.
    """
    ds = gdal.Open(str(grid["tif"]))
    band = ds.GetRasterBand(1)
    ncols, nrows, cs = grid["ncols"], grid["nrows"], grid["cellsize"]
    xll = grid["x0"]
    yll = grid["y0"] - nrows * cs
    header = (f"ncols\t{ncols}\nnrows\t{nrows}\n"
              f"xllcorner\t{xll}\nyllcorner\t{yll}\n"
              f"cellsize\t{cs}\nNODATA_value\t{NODATA}\n")

    ny1, nx1 = zone_ids.shape
    with open(dem_path, "w") as fd, open(rmap_path, "w") as fr:
        fd.write(header)
        fr.write(header)
        for j in range(nrows):
            row = band.ReadAsArray(0, j, ncols, 1)[0].astype(np.float64)
            valid = row != NODATA
            # zone id per cell: valid DEM cell whose (iy1[j], ix1) lands inside L1
            rmap_row = np.zeros(ncols, dtype=np.int32)
            jy = iy1[j]
            if jy >= 0:
                col_ok = valid & (ix1 >= 0)
                if col_ok.any():
                    rmap_row[col_ok] = zone_ids[jy, ix1[col_ok]]
            dem_txt = np.where(valid, np.char.mod("%.2f", row), str(NODATA))
            fd.write(" ".join(dem_txt.tolist()))
            fd.write("\n")
            fr.write(" ".join(np.char.mod("%d", rmap_row).tolist()))
            fr.write("\n")
    ds = None


def write_roff(runoff: Dict, valid: np.ndarray, step_h: int, roff_path: Path) -> int:
    """Write the gridded runoff time series [mm/hr]; returns the number of rows.

    Column order matches the zone ids written to .rmap (row-major over valid L1
    cells). mHM runoff (mm per output step) is converted to a mm/hr rate.
    """
    da = runoff["da"]
    nt = da.sizes["time"]
    with open(roff_path, "w") as f:
        f.write("% Time(hr) Discharge(mm/hr)\n")
        for start in range(0, nt, _ROFF_TIME_BLOCK):
            stop = min(start + _ROFF_TIME_BLOCK, nt)
            block = da.isel(time=slice(start, stop)).values  # (b, ny1, nx1)
            block = block[:, valid] / float(step_h)          # (b, N) mm/hr
            block = np.nan_to_num(block, nan=0.0)
            for k in range(block.shape[0]):
                t = (start + k) * step_h
                vals = ",".join(np.char.mod("%.6g", block[k]).tolist())
                f.write(f"{t},{vals}\n")
    return nt


def _snap_segment(grid: Dict, outlet: Tuple[float, float],
                  seg_len: float) -> Tuple[float, float, float, float]:
    """Snap the outlet to the nearest grid edge and return a segment on it."""
    cs = grid["cellsize"]
    left, top = grid["x0"], grid["y0"]
    right = left + grid["ncols"] * cs
    bottom = top - grid["nrows"] * cs
    ox, oy = outlet
    d = {"left": abs(ox - left), "right": abs(ox - right),
         "top": abs(oy - top), "bottom": abs(oy - bottom)}
    edge = min(d, key=d.get)
    half = seg_len / 2.0
    if edge in ("left", "right"):
        x = left if edge == "left" else right
        yc = min(max(oy, bottom), top)
        return x, min(max(yc - half, bottom), top), x, min(max(yc + half, bottom), top)
    y = top if edge == "top" else bottom
    xc = min(max(ox, left), right)
    return min(max(xc - half, left), right), y, min(max(xc + half, left), right), y


def write_extbc(grid: Dict, outlet: Tuple[float, float], bc_type: int,
                bc_value: float, seg_len: float, extbc_path: Path) -> int:
    """Write a single outlet boundary segment; returns num_extbc (1)."""
    x1, y1, x2, y2 = _snap_segment(grid, outlet, seg_len)
    with open(extbc_path, "w") as f:
        f.write("% BC Type, X1, Y1, X2, Y2, BC\n")
        f.write(f"{bc_type},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{bc_value}\n")
    return 1


def write_obs(gauges: List[Dict], obs_path: Path) -> int:
    """Write projected gauge coordinates as TRITON observation points."""
    with open(obs_path, "w") as f:
        f.write("%X-Location,Y-Location\n")
        for g in gauges:
            f.write(f"{g['x']:.3f},{g['y']:.3f}\n")
    return len(gauges)


def write_cfg(cfg_path: Path, names: Dict[str, str], projection: str,
              num_runoffs: int, runoff_rows: int, num_extbc: int,
              sim_duration_s: int, print_interval_s: int,
              const_mann: float) -> None:
    """Write the TRITON configuration tying the generated inputs together."""
    rel = lambda key: f"input/{names['domain']}/{names[key]}"
    lines = [
        "#---------------------------------------------------------------------------",
        "# TRITON config file (generated by mod22_mhm_to_triton)",
        "#---------------------------------------------------------------------------",
        f'dem_filename="{rel("dem")}"',
        f'output_folder="output/{names["domain"]}/"',
        "input_format=ASC",
        "output_format=ASC",
        'outfile_pattern="%s/%s/%s_%02d_%02d"',
        f'projection="{projection}"',
        "output_option=SEQ",
        "",
        "# Manning roughness (constant until the .mann file is generated)",
        f"const_mann={const_mann}",
        "",
        "# Hydrograph point sources are not used in this coupling",
        "num_sources=0",
        "",
        "# Gridded runoff from mHM",
        f"num_runoffs={num_runoffs}",
        f"runoff_row_size={runoff_rows}",
        f'runoff_filename="{rel("roff")}"',
        f'runoff_map="{rel("rmap")}"',
        "",
        "# External (outlet) boundary",
        f"num_extbc={num_extbc}",
        f'extbc_dir="input/{names["domain"]}/"',
        f'extbc_file="{rel("extbc")}"',
        "",
        "# Observation points (mHM gauges)",
        "time_series_flag=1",
        f'observation_loc_file="{rel("obs")}"',
        "",
        "print_option=h",
        "max_value_print_option=h",
        "sim_start_time=0",
        f"sim_duration={sim_duration_s}",
        "checkpoint_id=0",
        "time_increment_fixed=0",
        "time_step=0.01",
        f"print_interval={print_interval_s}",
        "courant=0.5",
        "hextra=0.001",
        "",
    ]
    cfg_path.write_text("\n".join(lines))
