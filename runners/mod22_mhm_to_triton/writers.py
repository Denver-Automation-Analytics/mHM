"""Writers for mod22: emit the TRITON ASCII input files from mHM/mRM data.

Large grids are streamed row-by-row so the full-resolution DEM and runoff map are
never held in memory at once.
"""
from __future__ import annotations

import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import NODATA

# Time block used when streaming the runoff cube out to the .roff file.
_ROFF_TIME_BLOCK = 365

# ESRI D8 codes -> (east, north) unit vectors: +qx = east (increasing col), +qy = north (decreasing row).
_D8 = {1: (1, 0), 2: (1, -1), 4: (0, -1), 8: (-1, -1),
       16: (-1, 0), 32: (-1, 1), 64: (0, 1), 128: (1, 1)}


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
        fd.write(header)  # only the DEM carries the ESRI header; .rmap is a bare grid
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


def write_mann(lc_tif: Path, dem_grid: Dict, lut: Dict[int, float],
               const_mann: float, nodata_lc, mann_path: Path) -> None:
    """Write the per-cell Manning grid [-] aligned to the DEM (headerless).

    Land-cover codes on the DEM grid are mapped to their RAT roughness; nodata
    or unmapped cells fall back to *const_mann*.
    """
    ds = gdal.Open(str(lc_tif))
    band = ds.GetRasterBand(1)
    ncols, nrows = dem_grid["ncols"], dem_grid["nrows"]
    maxcode = max(c for c in lut if c >= 0)
    table = np.full(maxcode + 1, const_mann, dtype=np.float64)
    for code, nval in lut.items():
        if 0 <= code <= maxcode:
            table[code] = nval
    nod = None if nodata_lc is None else int(nodata_lc)
    with open(mann_path, "w") as f:
        for j in range(nrows):
            row = band.ReadAsArray(0, j, ncols, 1)[0].astype(np.int64)
            out = np.full(ncols, const_mann, dtype=np.float64)
            m = (row >= 0) & (row <= maxcode)
            if nod is not None:
                m &= row != nod
            out[m] = table[row[m]]
            f.write(" ".join(np.char.mod("%.4f", out).tolist()))
            f.write("\n")
    ds = None


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


def _series_stats(a: np.ndarray) -> Dict[str, float]:
    """min/mean/median/p90/max over a 1-D array (empty -> zeros)."""
    if a.size == 0:
        return {"n": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()),
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def write_initial_conditions(dem_grid: Dict, facc_tif: Path, slope_tif: Path,
                             fdir_tif: Path, lc_tif: Path, mann_lut: Dict[int, float],
                             const_mann: float, qb_rate: np.ndarray,
                             ix1: np.ndarray, iy1: np.ndarray, l0_cell_area_m2: float,
                             channel_km2: float, width_a: float, width_b: float,
                             slope_min: float, h_path: Path, qx_path: Path,
                             qy_path: Path) -> Dict:
    """Seed initial depth/discharge in channel cells from mHM baseflow (Method A).

    For every channel cell (drainage area >= *channel_km2*) the accumulated
    baseflow discharge Q_b = pre-event baseflow rate x drainage area is converted
    to Manning normal depth h = (n Q_b / (w sqrt(S)))^(3/5), with width
    w = a A^b. The unit discharge Q_b/w is split along the D8 flow direction into
    qx/qy so the channel starts already flowing.

    Writes the headerless ASCII grids (aligned to the DEM) plus GeoTIFF mirrors
    for inspection, and returns the channel-cell count, the GeoTIFF paths, and
    depth/unit-discharge/velocity statistics over the channel cells.
    """
    dsf = gdal.Open(str(facc_tif)); bf = dsf.GetRasterBand(1)
    dss = gdal.Open(str(slope_tif)); bs = dss.GetRasterBand(1)
    dsd = gdal.Open(str(fdir_tif)); bd = dsd.GetRasterBand(1)
    dsl = gdal.Open(str(lc_tif)); bl = dsl.GetRasterBand(1)
    ncols, nrows = dem_grid["ncols"], dem_grid["nrows"]

    maxcode = max(c for c in mann_lut if c >= 0)
    ntab = np.full(maxcode + 1, const_mann, dtype=np.float64)
    for code, nval in mann_lut.items():
        if 0 <= code <= maxcode:
            ntab[code] = nval
    d8x = np.zeros(129, dtype=np.float64)
    d8y = np.zeros(129, dtype=np.float64)
    for code, (ux, uy) in _D8.items():
        norm = math.hypot(ux, uy)
        d8x[code], d8y[code] = ux / norm, uy / norm

    # GeoTIFF mirrors aligned to the DEM grid, for manual inspection (0 = dry/off-channel)
    prj = gdal.Open(str(dem_grid["tif"])).GetProjection()
    cs = dem_grid["cellsize"]
    gt = (dem_grid["x0"], cs, 0.0, dem_grid["y0"], 0.0, -cs)
    drv = gdal.GetDriverByName("GTiff")
    co = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
    tifs = {"h": Path(f"{h_path}.tif"), "qx": Path(f"{qx_path}.tif"), "qy": Path(f"{qy_path}.tif")}

    def _new_tif(path: Path):
        d = drv.Create(str(path), ncols, nrows, 1, gdal.GDT_Float32, co)
        d.SetGeoTransform(gt); d.SetProjection(prj)
        d.GetRasterBand(1).SetNoDataValue(0.0)
        return d

    th, tqx, tqy = _new_tif(tifs["h"]), _new_tif(tifs["qx"]), _new_tif(tifs["qy"])
    bth, btqx, btqy = th.GetRasterBand(1), tqx.GetRasterBand(1), tqy.GetRasterBand(1)

    h_vals, q_vals, v_vals = [], [], []
    n_chan = 0
    with open(h_path, "w") as fh, open(qx_path, "w") as fqx, open(qy_path, "w") as fqy:
        for j in range(nrows):
            facc = bf.ReadAsArray(0, j, ncols, 1)[0].astype(np.float64)
            slope = bs.ReadAsArray(0, j, ncols, 1)[0].astype(np.float64)
            fdir = bd.ReadAsArray(0, j, ncols, 1)[0].astype(np.float64)
            lc = bl.ReadAsArray(0, j, ncols, 1)[0].astype(np.int64)
            h = np.zeros(ncols, dtype=np.float64)
            qx = np.zeros(ncols, dtype=np.float64)
            qy = np.zeros(ncols, dtype=np.float64)

            a_m2 = facc * l0_cell_area_m2
            a_km2 = a_m2 / 1e6
            qbrow = np.zeros(ncols, dtype=np.float64)
            jy = iy1[j]
            if jy >= 0:
                ok = ix1 >= 0
                qbrow[ok] = qb_rate[jy, ix1[ok]]

            chan = (facc > NODATA + 1) & (slope > NODATA + 1) & (a_km2 >= channel_km2) & (qbrow > 0)
            if chan.any():
                qb = qbrow[chan] * a_m2[chan]                      # m3 s-1
                w = np.maximum(width_a * np.power(a_km2[chan], width_b), 1e-3)
                s = np.maximum(np.tan(np.radians(slope[chan])), slope_min)
                nch = ntab[np.clip(lc[chan], 0, maxcode)]
                hc = np.power(nch * qb / (w * np.sqrt(s)), 0.6)
                q_unit = qb / w                                    # m2 s-1
                code = np.clip(fdir[chan].astype(np.int64), 0, 128)
                h[chan] = hc
                qx[chan] = q_unit * d8x[code]
                qy[chan] = q_unit * d8y[code]
                n_chan += int(chan.sum())
                h_vals.append(hc)
                q_vals.append(q_unit)
                v_vals.append(q_unit / np.maximum(hc, 1e-9))       # m s-1

            fh.write(" ".join(np.char.mod("%.4f", h).tolist())); fh.write("\n")
            fqx.write(" ".join(np.char.mod("%.6g", qx).tolist())); fqx.write("\n")
            fqy.write(" ".join(np.char.mod("%.6g", qy).tolist())); fqy.write("\n")
            bth.WriteArray(h.reshape(1, -1).astype(np.float32), 0, j)
            btqx.WriteArray(qx.reshape(1, -1).astype(np.float32), 0, j)
            btqy.WriteArray(qy.reshape(1, -1).astype(np.float32), 0, j)
    for d in (th, tqx, tqy):
        d.FlushCache()
    th = tqx = tqy = None
    dsf = dss = dsd = dsl = None

    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)
    return {
        "n_chan": n_chan,
        "tifs": tifs,
        "depth_m": _series_stats(cat(h_vals)),
        "unit_q_m2s": _series_stats(cat(q_vals)),
        "velocity_ms": _series_stats(cat(v_vals)),
    }


def write_cfg(cfg_path: Path, names: Dict[str, str], projection: str,
              num_runoffs: int, runoff_rows: int, num_extbc: int,
              sim_duration_s: int, print_interval_s: int,
              const_mann: float, init_names: Optional[Dict[str, str]] = None) -> None:
    """Write the TRITON configuration tying the generated inputs together."""
    rel = lambda key: f"input/{names['domain']}/{names[key]}"
    ic = init_names or {}
    h_file = f"input/{names['domain']}/{ic['h']}" if ic.get("h") else ""
    qx_file = f"input/{names['domain']}/{ic['qx']}" if ic.get("qx") else ""
    qy_file = f"input/{names['domain']}/{ic['qy']}" if ic.get("qy") else ""
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
        "output_option=PAR",
        "",
        "# Manning roughness field from land cover (const_mann is the fallback)",
        f'n_infile="{rel("mann")}"',
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
        "# Initial conditions (mHM pre-event baseflow warm start)",
        f'h_infile="{h_file}"',
        f'qx_infile="{qx_file}"',
        f'qy_infile="{qy_file}"',
        "",
        "it_count=0",
        "gpu_direct_flag=1",
        "domain_decomposition=dynamic",
        "factor_interval_domain_decomposition=10",
        "open_boundaries=1",
        "print_option=h",
        "max_value_print_option=h",
        "sim_start_time=0",
        f"sim_duration={sim_duration_s}",
        "checkpoint_id=0",
        "time_increment_fixed=0",
        f"print_interval={print_interval_s}",
        "courant=0.5",
        "hextra=0.001",
        "",
    ]
    cfg_path.write_text("\n".join(lines))
