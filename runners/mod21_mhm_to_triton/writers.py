"""Writers for mod22: emit the TRITON ASCII input files from mHM/mRM data.

Large grids are streamed row-by-row so the full-resolution DEM and runoff map are
never held in memory at once.
"""
from __future__ import annotations

import os
import sys
import math
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal
from scipy import ndimage

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
    # space-separated header: TRITON's ASCII parser splits on spaces, not tabs
    header = (f"ncols {ncols}\nnrows {nrows}\n"
              f"xllcorner {xll}\nyllcorner {yll}\n"
              f"cellsize {cs}\nNODATA_value {NODATA}\n")

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
               const_mann: float, nodata_lc, mann_path: Path,
               channel_facc_tif: Path = None, channel_km2: float = None,
               cell_area_m2: float = None, channel_n: float = None,
               waterbody_mask_tif: Path = None, water_n: float = None) -> int:
    """Write the per-cell Manning grid [-] aligned to the DEM (headerless).

    Land-cover codes on the DEM grid are mapped to their RAT roughness; nodata
    or unmapped cells fall back to *const_mann*. When *channel_facc_tif* and
    *channel_n* are given, cells whose drainage area (facc x *cell_area_m2*)
    reaches *channel_km2* are overwritten with *channel_n* (channel roughness).
    When *waterbody_mask_tif* and *water_n* are given, known-waterbody cells are
    finally overwritten with *water_n* (open-water roughness). A GeoTIFF mirror
    (<mann_path>.tif) is written for inspection. Returns the number of channel
    cells burned in.
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

    burn = channel_facc_tif is not None and channel_n is not None
    bfacc = None
    if burn:
        dsf = gdal.Open(str(channel_facc_tif))
        bfacc = dsf.GetRasterBand(1)

    burn_wb = waterbody_mask_tif is not None and water_n is not None
    bwb = None
    if burn_wb:
        dsw = gdal.Open(str(waterbody_mask_tif))
        bwb = dsw.GetRasterBand(1)

    # GeoTIFF mirror aligned to the DEM grid, for manual inspection
    cs = dem_grid["cellsize"]
    tif = gdal.GetDriverByName("GTiff").Create(
        f"{mann_path}.tif", ncols, nrows, 1, gdal.GDT_Float32,
        ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    tif.SetGeoTransform((dem_grid["x0"], cs, 0.0, dem_grid["y0"], 0.0, -cs))
    tif.SetProjection(gdal.Open(str(dem_grid["tif"])).GetProjection())
    tband = tif.GetRasterBand(1)

    n_chan = 0
    with open(mann_path, "w") as f:
        for j in range(nrows):
            row = band.ReadAsArray(0, j, ncols, 1)[0].astype(np.int64)
            out = np.full(ncols, const_mann, dtype=np.float64)
            m = (row >= 0) & (row <= maxcode)
            if nod is not None:
                m &= row != nod
            out[m] = table[row[m]]
            if burn:
                facc = bfacc.ReadAsArray(0, j, ncols, 1)[0].astype(np.float64)
                chan = (facc > NODATA + 1) & (facc * cell_area_m2 / 1e6 >= channel_km2)
                out[chan] = channel_n
                n_chan += int(chan.sum())
            if burn_wb:
                wb = bwb.ReadAsArray(0, j, ncols, 1)[0]
                out[wb == 1] = water_n
            f.write(" ".join(np.char.mod("%.4f", out).tolist()))
            f.write("\n")
            tband.WriteArray(out.reshape(1, -1).astype(np.float32), 0, j)
    tif.FlushCache()
    tif = None
    ds = None
    if burn:
        dsf = None
    if burn_wb:
        dsw = None
    return n_chan


def write_extbc(grid: Dict, bc_type: int, bc_value: float, extbc_path: Path,
                outlet: Tuple[float, float]) -> Tuple[int, str]:
    """Write a single open outlet boundary on the downstream grid edge.

    Returns ``(num_extbc, edge_name)`` with ``num_extbc == 1``. TRITON only
    accepts boundary segments that lie on the rectangular grid edge and are
    axis-aligned (see extbc.h check_extreme_extbc). To route water out at the
    downstream outlet only, we emit one full-length segment on whichever edge
    (W, E, N, S) the *outlet* (max flow-accumulation cell) is closest to; the
    other three edges stay closed walls so water leaves solely at the downstream
    boundary. Endpoints use edge cell-centre coordinates so TRITON's
    calc_src_col/row map them to columns 0/ncols-1 and rows 0/nrows-1.
    """
    cs = grid["cellsize"]
    x0, y0 = grid["x0"], grid["y0"]
    ncols, nrows = grid["ncols"], grid["nrows"]
    xw = x0 + 0.5 * cs                    # column 0 centre
    xe = x0 + (ncols - 0.5) * cs          # column ncols-1 centre
    yn = y0 - 0.5 * cs                    # row 0 (top) centre
    ys = y0 - (nrows - 0.5) * cs          # row nrows-1 (bottom) centre
    ox, oy = outlet
    col = (ox - x0) / cs
    row = (y0 - oy) / cs
    dist = {"west": col, "east": (ncols - 1) - col,
            "north": row, "south": (nrows - 1) - row}
    edge = min(dist, key=dist.get)
    segs = {
        "west":  (xw, yn, xw, ys),
        "east":  (xe, yn, xe, ys),
        "north": (xw, yn, xe, yn),
        "south": (xw, ys, xe, ys),
    }
    x1, y1, x2, y2 = segs[edge]
    with open(extbc_path, "w") as f:
        f.write("% BC Type, X1, Y1, X2, Y2, BC\n")
        f.write(f"{bc_type},{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f},{bc_value}\n")
    return 1, edge


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
                             qy_path: Path, do_fill: bool = True,
                             fill_max_h: float = 5.0, min_h: float = 0.0) -> Dict:
    """Seed initial depth/discharge from mHM baseflow, then fill the DEM channel storage.

    For every channel cell (drainage area >= *channel_km2*) the accumulated
    baseflow discharge Q_b = pre-event baseflow rate x drainage area is converted
    to Manning normal depth h = (n Q_b / (w sqrt(S)))^(3/5), with width w = a A^b;
    the unit discharge Q_b/w is split along the D8 flow direction into qx/qy. These
    seeded cells define a water-surface elevation WSE = DEM + h.

    When *do_fill* is set, a level-pool priority flood grows each seed outward into
    8-neighbours whose DEM lies below the reaching WSE (taking the max WSE across
    seeds), stopping at DEM rims or once the fill depth exceeds *fill_max_h*. The
    resulting depth field (inith) is the authoritative product; its wet mask then
    drives qx/qy: every wet cell inherits the full velocity vector of its nearest
    channel seed (Voronoi fill), so the qx/qy coverage matches h exactly. Cells
    thinner than *min_h* are dropped from the mask so none carries flow over a
    near-zero depth.

    Writes the headerless ASCII grids (aligned to the DEM) plus GeoTIFF mirrors for
    inspection, and returns the seed-cell count, the filled-cell count, the GeoTIFF
    paths, and depth/unit-discharge/velocity statistics over the wet cells.
    """
    ncols, nrows = dem_grid["ncols"], dem_grid["nrows"]
    cs = dem_grid["cellsize"]

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

    # Full-array reads: the level-pool flood needs the whole grid in memory at once.
    dsdem = gdal.Open(str(dem_grid["tif"]))
    dem = dsdem.GetRasterBand(1).ReadAsArray().astype(np.float64)
    prj = dsdem.GetProjection()
    dsdem = None
    dsf = gdal.Open(str(facc_tif)); facc = dsf.GetRasterBand(1).ReadAsArray().astype(np.float64); dsf = None
    dss = gdal.Open(str(slope_tif)); slope = dss.GetRasterBand(1).ReadAsArray().astype(np.float64); dss = None
    dsd = gdal.Open(str(fdir_tif)); fdir = dsd.GetRasterBand(1).ReadAsArray(); dsd = None
    dsl = gdal.Open(str(lc_tif)); lc = dsl.GetRasterBand(1).ReadAsArray().astype(np.int64); dsl = None

    valid_dem = np.isfinite(dem) & (dem != NODATA)

    # Map the L1 baseflow rate onto the DEM grid via the per-axis L1 index arrays.
    valid_row = iy1 >= 0
    valid_col = ix1 >= 0
    rr = np.where(valid_row, iy1, 0)
    cc = np.where(valid_col, ix1, 0)
    qbrow = qb_rate[np.ix_(rr, cc)].astype(np.float64)
    qbrow[~(valid_row[:, None] & valid_col[None, :])] = 0.0

    a_m2 = facc * l0_cell_area_m2
    a_km2 = a_m2 / 1e6
    seed = ((facc > NODATA + 1) & (slope > NODATA + 1) &
            (a_km2 >= channel_km2) & (qbrow > 0) & valid_dem)

    h = np.zeros((nrows, ncols), dtype=np.float64)
    qx = np.zeros((nrows, ncols), dtype=np.float64)  # seed velocities (later filled over the h mask)
    qy = np.zeros((nrows, ncols), dtype=np.float64)

    if seed.any():
        qb = qbrow[seed] * a_m2[seed]                          # m3 s-1
        w = np.maximum(width_a * np.power(a_km2[seed], width_b), 1e-3)
        s = np.maximum(np.tan(np.radians(slope[seed])), slope_min)
        nch = ntab[np.clip(lc[seed], 0, maxcode)]
        hc = np.power(nch * qb / (w * np.sqrt(s)), 0.6)
        q_unit = qb / w                                        # m2 s-1
        code = np.clip(fdir[seed].astype(np.int64), 0, 128)
        h[seed] = hc
        qx[seed] = q_unit * d8x[code]
        qy[seed] = q_unit * d8y[code]

    # inith is the authoritative product: its wet mask drives the qx/qy extent below.
    wet = seed.copy()
    if do_fill and seed.any():
        # Level-pool priority flood: process the highest reaching WSE first so each
        # cell settles at the maximum seed water level able to reach it below the cap.
        wse = np.full((nrows, ncols), -np.inf, dtype=np.float64)
        heap: list = []
        sr, sc = np.nonzero(seed)
        for r, c in zip(sr.tolist(), sc.tolist()):
            W = dem[r, c] + h[r, c]
            wse[r, c] = W
            heapq.heappush(heap, (-W, r, c))

        neigh = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        while heap:
            negW, r, c = heapq.heappop(heap)
            W = -negW
            if wse[r, c] > W:                                  # stale (lower) entry
                continue
            for dr, dc in neigh:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= nrows or nc < 0 or nc >= ncols:
                    continue
                if not valid_dem[nr, nc]:
                    continue
                dn = dem[nr, nc]
                if dn >= W or (W - dn) > fill_max_h:            # rim reached / depth cap
                    continue
                if wse[nr, nc] >= W:                           # already at >= this level
                    continue
                wse[nr, nc] = W
                heapq.heappush(heap, (-W, nr, nc))

        wet = np.isfinite(wse)
        h = np.where(wet, np.clip(wse - dem, 0.0, fill_max_h), 0.0)

    # Drop paper-thin cells (depth < min_h) so no wet cell carries a finite discharge
    # over a near-zero depth (which would blow up |q|/h at TRITON's first step).
    if min_h > 0.0:
        wet = wet & (h >= min_h)
        h = np.where(wet, h, 0.0)

    # Fill qx/qy over the exact inith mask: every wet cell inherits the full velocity
    # vector of its nearest channel seed (Voronoi fill), so qx/qy cover matches h.
    if seed.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~seed, return_indices=True)
        qxf = np.zeros((nrows, ncols), dtype=np.float64)
        qyf = np.zeros((nrows, ncols), dtype=np.float64)
        qxf[wet] = qx[iy[wet], ix[wet]]
        qyf[wet] = qy[iy[wet], ix[wet]]
        qx, qy = qxf, qyf

    n_chan = int((seed & wet).sum())
    n_fill = int(wet.sum()) - n_chan

    # Hard guarantee that h/qx/qy share one wet/dry footprint: h is nonzero exactly on
    # the wet cells and qx/qy are zero wherever h is dry (their own axis-aligned zeros
    # inside the wet area are physical and expected).
    dry = ~wet
    assert np.array_equal(h != 0.0, wet), "init h nonzero footprint disagrees with the wet mask"
    assert not qx[dry].any() and not qy[dry].any(), "init qx/qy are nonzero on h-dry cells"

    # GeoTIFF mirrors aligned to the DEM grid, for manual inspection (0 = dry).
    gt = (dem_grid["x0"], cs, 0.0, dem_grid["y0"], 0.0, -cs)
    drv = gdal.GetDriverByName("GTiff")
    co = ["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"]
    tifs = {"h": Path(f"{h_path}.tif"), "qx": Path(f"{qx_path}.tif"), "qy": Path(f"{qy_path}.tif")}

    def _write_tif(path: Path, arr: np.ndarray) -> None:
        d = drv.Create(str(path), ncols, nrows, 1, gdal.GDT_Float32, co)
        d.SetGeoTransform(gt); d.SetProjection(prj)
        d.GetRasterBand(1).SetNoDataValue(0.0)
        d.GetRasterBand(1).WriteArray(arr.astype(np.float32))
        d.FlushCache()

    with open(h_path, "w") as fh, open(qx_path, "w") as fqx, open(qy_path, "w") as fqy:
        for j in range(nrows):
            fh.write(" ".join(np.char.mod("%.4f", h[j]).tolist())); fh.write("\n")
            fqx.write(" ".join(np.char.mod("%.6g", qx[j]).tolist())); fqx.write("\n")
            fqy.write(" ".join(np.char.mod("%.6g", qy[j]).tolist())); fqy.write("\n")
    _write_tif(tifs["h"], h)
    _write_tif(tifs["qx"], qx)
    _write_tif(tifs["qy"], qy)

    qmag = np.hypot(qx[wet], qy[wet])
    h_wet = h[wet]
    return {
        "n_chan": n_chan,
        "n_fill": n_fill,
        "tifs": tifs,
        "depth_m": _series_stats(h_wet),
        "unit_q_m2s": _series_stats(qmag),
        "velocity_ms": _series_stats(qmag / np.maximum(h_wet, 1e-9)),
    }


def write_cfg(cfg_path: Path,
              names: Dict[str, str],
              projection: str,
              num_runoffs: int,
              runoff_rows: int,
              num_extbc: int,
              sim_duration_s: int,
              mapping_interval_s: int,
              obs_interval_s: int,
              const_mann: float,
              decomp_factor: int = 10,
              decomp_type: str = "static",
              courant: float = 0.5,
              init_names: Optional[Dict[str, str]] = None) -> None:
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
        "output_format=GTIFF",
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
        f'print_observation={obs_interval_s}',
        "",
        "# Initial conditions (mHM pre-event baseflow warm start)",
        f'h_infile="{h_file}"',
        f'qx_infile="{qx_file}"',
        f'qy_infile="{qy_file}"',
        "",
        "it_count=0",
        "gpu_direct_flag=0",
        f"domain_decomposition={decomp_type}",
        f"factor_interval_domain_decomposition={decomp_factor}",
        "open_boundaries=1",
        "print_option=h",
        "max_value_print_option=h",
        "sim_start_time=0",
        f"sim_duration={sim_duration_s}",
        "checkpoint_id=0",
        "time_increment_fixed=0",
        f"print_interval={mapping_interval_s}",
        f"courant={courant}",
        "hextra=0.001",
        "",
    ]
    cfg_path.write_text("\n".join(lines))
