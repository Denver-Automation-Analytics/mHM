"""
mHM calibration assembler.

Validates all outputs from mod10–mod16, introspects grid metadata, writes a
calibration-ready mhm.nml (optimize=.TRUE.) plus companion namelists into the
domain directory, then launches the mHM binary and streams its output.

Run order: mod10 → mod11 → mod12 → mod13 → mod14 → mod15 → mod16 → mod17.

mHM invocation: the binary is run with cwd=DOMAIN_DIR so that it finds all
four nml files without path arguments.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import L0_CELL_SIZE_M
from pathlib import Path

from readers import derive_eval_period, read_gauge_info, read_lcover_scenes, read_meteo_dates, read_soil_info
from nml_writer import write_mhm_nml

# ---------------------------------------------------------------------------
# USER INPUTS — edit these paths and settings to reconfigure
# ---------------------------------------------------------------------------
DOMAIN_DIR       = "/workspace/test_domain_3"
MHM_BINARY       = "/workspace/build/mhm"

OPTI_METHOD      = 1     # 1=DDS, 2=Simulated Annealing, 3=SCE
OPTI_FUNCTION    = 9     # 9=1-KGE(Q); see mhm.nml comments for full list
N_ITERATIONS     = 1000
WARMING_DAYS     = 0     # spin-up days before eval period; increase for multi-year meteo
TIMESTEP         = 1     # model timestep [h]: 1=hourly, 24=daily
# L1 simulation resolution in metres.  Must be a whole-number multiple of
# L0_CELL_SIZE_M (runners/config.py) and strictly coarser: 600, 900, 1200, ...
RESOLUTION_HYDROLOGY = 600

REPO_PARAM_NML   = "/workspace/mhm_parameter.nml"
REPO_OUTPUT_NML  = "/workspace/mhm_outputs.nml"
REPO_MRM_OUT_NML = "/workspace/mrm_outputs.nml"
# ---------------------------------------------------------------------------

if RESOLUTION_HYDROLOGY % L0_CELL_SIZE_M != 0:
    raise ValueError(
        f"RESOLUTION_HYDROLOGY={RESOLUTION_HYDROLOGY} is not a whole-number multiple "
        f"of L0_CELL_SIZE_M={L0_CELL_SIZE_M} (runners/config.py)."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("calibrate_to_mhm")


# ---------------------------------------------------------------------------
# Morph NC axis-order repair
# ---------------------------------------------------------------------------

_MORPH_NC = ("dem.nc", "slope.nc", "aspect.nc", "fdir.nc", "facc.nc")


def _fix_morph_nc_axis_order(morph_dir: Path) -> None:
    """Transpose any (x, y) morph NC file to (y, x) as mHM requires."""
    import shutil
    import xarray as xr

    for name in _MORPH_NC:
        path = morph_dir / name
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            first_var = next(iter(ds.data_vars))
            if ds[first_var].dims[0] != "x":
                continue  # already (y, x)
        log.info("Fixing axis order (x,y)→(y,x): %s", path)
        tmp = path.with_suffix(".tmp.nc")
        with xr.open_dataset(path, chunks={"x": 512, "y": 512}) as ds:
            ds.transpose("y", "x").to_netcdf(tmp)
        shutil.move(str(tmp), str(path))
        log.info("Fixed: %s", path)


# ---------------------------------------------------------------------------
# Geology placeholder bootstrap
# ---------------------------------------------------------------------------

_GEO_CLASSDEF = """\
nGeo_Formations  10
GeoParam(i)   ClassUnit     Karstic      Description
         1                1           0      GeoUnit-1
         2            2           0      GeoUnit-2
         3            3           0      GeoUnit-3
         4            4           0      GeoUnit-4
         5            5           0      GeoUnit-5
         6            6           0      GeoUnit-6
         7            7           0      GeoUnit-7
         8            8           0      GeoUnit-8
         9            9           0      GeoUnit-9
        10               10           0      GeoUnit-10
!<-END
"""


def _bootstrap_geology(morph_dir: Path) -> None:
    """Create single-class geology files from the DEM mask if absent."""
    classdef = morph_dir / "geology_classdefinition.txt"
    classmap  = morph_dir / "geology_class.asc"
    if classdef.exists() and classmap.exists():
        return

    import netCDF4 as nc4
    import numpy as np

    log.info("Bootstrapping geology files from DEM mask (single class)...")

    dem_nc = morph_dir / "dem.nc"
    with nc4.Dataset(dem_nc) as ds:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        dem = ds.variables["dem"][:]        # masked array, (y, x)
        fill = ds.variables["dem"]._FillValue

    nrows, ncols = dem.shape
    xres = float(x[1] - x[0])
    xll  = float(x[0]) - xres / 2
    yll  = float(y[-1]) - xres / 2

    if not classdef.exists():
        classdef.write_text(_GEO_CLASSDEF)
        log.info("Written: %s", classdef)

    if not classmap.exists():
        # class 1 where DEM is valid, nodata elsewhere
        grid = np.where(np.asarray(dem) != fill, 1, -9999).astype(np.int32)
        with open(classmap, "w") as fh:
            fh.write(f"ncols         {ncols}\n")
            fh.write(f"nrows         {nrows}\n")
            fh.write(f"xllcorner     {xll:.1f}\n")
            fh.write(f"yllcorner     {yll:.1f}\n")
            fh.write(f"cellsize      {int(xres)}\n")
            fh.write("NODATA_value  -9999\n")
            for row in grid:
                fh.write(" ".join(str(v) for v in row) + "\n")
        log.info("Written: %s", classmap)


# LAI monthly values for 3 land-cover classes (South Texas climate):
#   1 = Forest (riparian/mixed woodland)
#   2 = Impervious (sealed surfaces — minimal LAI)
#   3 = Pervious (grassland/shrubland)
_LAI_CLASSDEF = (
    "NoLAIclasses           3\n"
    "ID   LAND-USE        Jan.   Feb.   Mar.   Apr.   May    Jun.   Jul.   Aug.   Sep.   Oct.   Nov.   Dec.\n"
    " 1    Forest          2.0    2.0    3.0    4.0    5.0    5.5    5.5    5.5    5.0    4.0    2.5    2.0\n"
    " 2    Impervious      0.01   0.01   0.01   0.01   0.01   0.01   0.01   0.01   0.01   0.01   0.01   0.01\n"
    " 3    Pervious        0.5    0.5    1.0    1.5    2.0    2.5    2.0    2.0    2.0    1.5    0.8    0.5\n"
)


def _bootstrap_lai(morph_dir: Path, luse_dir: Path) -> None:
    """Create LAI class files from the land-cover map if absent."""
    classdef = morph_dir / "LAI_classdefinition.txt"
    classmap  = morph_dir / "LAI_class.asc"
    if classdef.exists() and classmap.exists():
        return

    log.info("Bootstrapping LAI files from land-cover map (3 classes)...")

    if not classdef.exists():
        classdef.write_text(_LAI_CLASSDEF)
        log.info("Written: %s", classdef)

    if not classmap.exists():
        # Reuse lc_2024.asc directly — its class IDs (1,2,3) match our LAI classes
        import shutil
        lc_asc = next(luse_dir.glob("lc_*.asc"), None)
        if lc_asc is None:
            log.error("No lc_*.asc found in %s — cannot create LAI_class.asc", luse_dir)
            sys.exit(1)
        shutil.copy2(lc_asc, classmap)
        log.info("Written: %s  (copied from %s)", classmap, lc_asc.name)


def _ascii_header(path: Path) -> dict:
    """Read the 6-line header of an ESRI ASCII grid, return lowercase-key dict."""
    header = {}
    with open(path) as fh:
        for _ in range(6):
            k, v = fh.readline().split(None, 1)
            header[k.lower()] = v.strip()
    return header


def _resample_ascii_to_l0(src_asc: Path, l0_nc: Path) -> None:
    """Resample a categorical ESRI ASCII grid to the L0 300 m grid in-place.

    Reads the target grid from *l0_nc* (land-cover NC with embedded CRS).
    Uses GDAL Warp with mode resampling (appropriate for class integers).
    Skips the operation when the source header already matches L0.
    """
    from osgeo import gdal

    # --- read L0 target grid params from land-cover NC ----------------------
    import netCDF4 as nc4
    import numpy as np

    with nc4.Dataset(l0_nc) as ds:
        x    = ds.variables["x"][:]
        y    = ds.variables["y"][:]
        crs_wkt = str(ds.variables["crs"].spatial_ref)
    xres  = float(x[1] - x[0])
    xmin  = float(x[0])  - xres / 2
    xmax  = float(x[-1]) + xres / 2
    ymin  = float(y[-1]) - xres / 2
    ymax  = float(y[0])  + xres / 2
    ncols = len(x)
    nrows = len(y)

    # --- check if resampling is needed --------------------------------------
    hdr = _ascii_header(src_asc)
    if (int(hdr["ncols"]) == ncols and int(hdr["nrows"]) == nrows
            and abs(float(hdr["cellsize"]) - xres) < 1e-6):
        return   # already on L0 grid

    log.info("Resampling %s to L0 grid (mode, %d m) …", src_asc.name, int(xres))

    src_cellsize = float(hdr["cellsize"])
    src_xll      = float(hdr["xllcorner"])
    src_yll      = float(hdr["yllcorner"])
    src_ncols    = int(hdr["ncols"])
    src_nrows    = int(hdr["nrows"])
    nodata_val   = int(float(hdr.get("nodata_value", "-9999")))

    # read data rows (skip 6-line header)
    with open(src_asc) as fh:
        for _ in range(6):
            fh.readline()
        data = np.loadtxt(fh, dtype=np.int32)

    # --- write source into an in-memory GeoTIFF with the known CRS ----------
    mem = gdal.GetDriverByName("MEM")
    src_ds = mem.Create("", src_ncols, src_nrows, 1, gdal.GDT_Int32)
    ull_y  = src_yll + src_nrows * src_cellsize      # upper-left y
    src_ds.SetGeoTransform((src_xll, src_cellsize, 0.0, ull_y, 0.0, -src_cellsize))
    src_ds.SetProjection(crs_wkt)
    band = src_ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(nodata_val)

    # --- warp to L0 grid ----------------------------------------------------
    dst_ds = gdal.Warp(
        "",
        src_ds,
        format="MEM",
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=xres,
        yRes=xres,
        dstSRS=crs_wkt,
        resampleAlg="mode",
        dstNodata=nodata_val,
    )

    out = dst_ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    src_ds = dst_ds = None   # close GDAL datasets

    # --- write result back as ESRI ASCII grid in-place ----------------------
    with open(src_asc, "w") as fh:
        fh.write(f"ncols         {ncols}\n")
        fh.write(f"nrows         {nrows}\n")
        fh.write(f"xllcorner     {xmin:.1f}\n")
        fh.write(f"yllcorner     {ymin:.1f}\n")
        fh.write(f"cellsize      {int(xres)}\n")
        fh.write(f"NODATA_value  {nodata_val}\n")
        for row in out:
            fh.write(" ".join(str(v) for v in row) + "\n")
    log.info("Resampled %s → %d×%d @ %d m", src_asc.name, ncols, nrows, int(xres))


def _build_idgauges_asc(morph_dir: Path, gauge_dir: Path, dem_nc: Path, l0_nc: Path) -> None:
    """Burn gauge local_ids into an L0 ASCII raster in *morph_dir*.

    Reads gauge lat/lon from id_map.csv, projects to LCC, snaps each gauge to
    the nearest valid L0 cell, and writes morph/idgauges.asc.
    Skips the operation if the file already exists with the correct grid.
    """
    import csv
    import netCDF4 as nc4_mod
    import numpy as np
    from pyproj import Transformer

    dst = morph_dir / "idgauges.asc"

    # Load L0 grid
    with nc4_mod.Dataset(dem_nc) as ds:
        x_l0 = np.array(ds.variables["x"][:])
        y_l0 = np.array(ds.variables["y"][:])
        dem_vals = np.array(ds.variables["dem"][:])           # (nrows, ncols)
        dem_fill = float(ds.variables["dem"]._FillValue)
    ncols, nrows = len(x_l0), len(y_l0)

    # Check if file already matches
    if dst.exists():
        hdr = _ascii_header(dst)
        if int(hdr["ncols"]) == ncols and int(hdr["nrows"]) == nrows:
            return

    log.info("Building morph/idgauges.asc at L0 300 m grid …")

    # Read LCC CRS WKT
    with nc4_mod.Dataset(l0_nc) as ds:
        crs_wkt = str(ds.variables["crs"].spatial_ref)

    # Transform gauges from WGS84 (lon, lat) → LCC (x, y)
    tf = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)

    # Valid-domain mask (True where DEM has data)
    valid = (dem_vals != dem_fill)

    grid = np.full((nrows, ncols), -9999, dtype=np.int32)

    id_map_path = gauge_dir / "id_map.csv"
    with open(id_map_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = int(row["local_id"])
            lon = float(row["lon"])
            lat = float(row["lat"])

            gx, gy = tf.transform(lon, lat)

            # Snap to nearest L0 cell
            ci = int(np.argmin(np.abs(x_l0 - gx)))   # column index
            ri = int(np.argmin(np.abs(y_l0 - gy)))   # row index (y descends)

            # Stay in bounds and within valid domain
            ci = max(0, min(ci, ncols - 1))
            ri = max(0, min(ri, nrows - 1))
            if valid[ri, ci]:
                grid[ri, ci] = gid

    xres = float(x_l0[1] - x_l0[0])
    xll  = float(x_l0[0])  - xres / 2
    yll  = float(y_l0[-1]) - xres / 2

    with open(dst, "w") as fh:
        fh.write(f"ncols         {ncols}\n")
        fh.write(f"nrows         {nrows}\n")
        fh.write(f"xllcorner     {xll:.1f}\n")
        fh.write(f"yllcorner     {yll:.1f}\n")
        fh.write(f"cellsize      {int(xres)}\n")
        fh.write("NODATA_value  -9999\n")
        for row_data in grid:
            fh.write(" ".join(str(v) for v in row_data) + "\n")

    n_placed = int((grid != -9999).sum())
    log.info("Written: %s  (%d gauges placed)", dst, n_placed)


# pysheds sequential D8 → ArcGIS D8 powers-of-2 (what mHM expects)
_PYSHEDS_TO_ARCGIS = {0: -9999, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128}


def _remap_fdir_to_arcgis(morph_dir: Path) -> None:
    """Remap fdir.nc from pysheds sequential (0–8) to ArcGIS D8 encoding.

    pysheds: 1=E 2=SE 3=S 4=SW 5=W 6=NW 7=N 8=NE  (0=flat/nodata)
    ArcGIS:  1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE  (mHM convention)
    Skips the operation if the file already uses ArcGIS encoding.
    """
    import netCDF4 as nc4_mod
    import numpy as np

    fdir_nc = morph_dir / "fdir.nc"
    with nc4_mod.Dataset(fdir_nc) as ds:
        fdir = np.array(ds.variables["fdir"][:])
        fill = int(ds.variables["fdir"]._FillValue)

    valid = fdir[fdir != fill]
    unique = set(int(v) for v in np.unique(valid))
    arcgis_vals = {1, 2, 4, 8, 16, 32, 64, 128}
    if unique.issubset(arcgis_vals):
        return   # already ArcGIS encoded

    log.info("Remapping fdir.nc: pysheds 0–8 → ArcGIS D8 powers-of-2 …")
    out = np.full_like(fdir, fill)
    for src_val, dst_val in _PYSHEDS_TO_ARCGIS.items():
        mask = fdir == src_val
        out[mask] = dst_val

    # Fill any remaining nodata within the DEM valid mask using the nearest
    # valid fdir cell (handles flat/boundary cells that got pysheds value 0).
    with nc4_mod.Dataset(morph_dir / "dem.nc") as ds:
        dem_fill = float(ds.variables["dem"]._FillValue)
        dem_arr  = np.array(ds.variables["dem"][:])

    broken = (out == fill) & (dem_arr != dem_fill)
    n_broken = int(broken.sum())
    if n_broken > 0:
        log.info("Filling %d flat/nodata fdir cells via nearest-valid-neighbour …", n_broken)
        from scipy.ndimage import distance_transform_edt
        valid_mask = out != fill
        _, (nr, nc_) = distance_transform_edt(
            ~valid_mask, return_distances=True, return_indices=True
        )
        out[broken] = out[nr[broken], nc_[broken]]

    with nc4_mod.Dataset(fdir_nc, "r+") as ds:
        ds.variables["fdir"][:] = out
    log.info("fdir.nc remapped.")


def _fix_facc_boundary_nodata(morph_dir: Path) -> None:
    """Fill facc nodata cells that lie within the DEM valid mask with 1.

    At domain boundaries, GDAL sum-resampling of the 10m accumulation raster
    can leave nodata in 300m cells that the DEM (average-resampled) considers
    valid.  mHM requires facc to be defined everywhere the DEM is defined.
    A value of 1 (self-drainage) is the minimum physically meaningful value.
    """
    import netCDF4 as nc4_mod
    import numpy as np

    dem_nc  = morph_dir / "dem.nc"
    facc_nc = morph_dir / "facc.nc"

    with nc4_mod.Dataset(dem_nc) as ds:
        dem  = np.array(ds.variables["dem"][:])
        dfil = float(ds.variables["dem"]._FillValue)

    with nc4_mod.Dataset(facc_nc) as ds:
        facc_var = ds.variables["facc"]
        facc     = np.array(facc_var[:])
        ffil     = int(facc_var._FillValue)

    # Cells with valid DEM but missing facc
    broken = (dem != dfil) & (facc == ffil)
    n_broken = int(broken.sum())
    if n_broken == 0:
        return

    log.info("Filling %d facc boundary nodata cells with 1 …", n_broken)
    facc[broken] = 1

    with nc4_mod.Dataset(facc_nc, "r+") as ds:
        ds.variables["facc"][:] = facc
    log.info("facc.nc updated.")


def _ensure_latlon_nc(latlon_dir: Path, dem_nc: Path, l0_nc: Path, resolution_hydrology: int) -> None:
    """Regenerate latlon.nc when the L0 shape no longer matches dem.nc.

    Uses the LCC CRS embedded in *l0_nc* to transform projected metre
    coordinates to WGS84 lat/lon before writing.
    """
    import netCDF4 as nc4_mod

    # Expected L0 shape
    with nc4_mod.Dataset(dem_nc) as ds:
        x_l0 = ds.variables["x"][:]
        y_l0 = ds.variables["y"][:]
    ncols_l0, nrows_l0 = len(x_l0), len(y_l0)

    # Skip if latlon.nc already has the right L0 shape
    out_file = latlon_dir / "latlon.nc"
    if out_file.exists():
        with nc4_mod.Dataset(out_file) as ds:
            if "lat_l0" in ds.variables:
                sh = ds.variables["lat_l0"].shape
                if sh[0] == nrows_l0 and sh[1] == ncols_l0:
                    return

    xres_l0 = float(x_l0[1] - x_l0[0])
    log.info(
        "Regenerating latlon.nc: L0=%dm (%d×%d), L1=%dm …",
        int(xres_l0), ncols_l0, nrows_l0, resolution_hydrology,
    )

    # Read LCC CRS WKT from the land-cover NC (all grids share this projection)
    with nc4_mod.Dataset(l0_nc) as ds:
        crs_wkt = str(ds.variables["crs"].spatial_ref)

    xll = float(x_l0[0])  - xres_l0 / 2
    yll = float(y_l0[-1]) - xres_l0 / 2

    l0_header = {
        "ncols": ncols_l0,
        "nrows": nrows_l0,
        "xllcorner": xll,
        "yllcorner": yll,
        "cellsize": xres_l0,
        "NODATA_value": -9999.0,
    }

    factor = resolution_hydrology // int(xres_l0)
    l1_header = {
        "ncols": ncols_l0 // factor,
        "nrows": nrows_l0 // factor,
        "xllcorner": xll,
        "yllcorner": yll,
        "cellsize": float(resolution_hydrology),
        "NODATA_value": -9999.0,
    }

    sys.path.insert(0, str(Path(__file__).parent.parent / "mod11_meteo_to_mhm"))
    from latlon_grid import create_latlon  # noqa: E402

    latlon_dir.mkdir(parents=True, exist_ok=True)
    create_latlon(
        out_file   = out_file,
        coord_sys  = crs_wkt,
        header_l0  = l0_header,
        header_l1  = l1_header,
        header_l11 = l1_header,
    )
    log.info("Written: %s", out_file)


# ---------------------------------------------------------------------------
# Phase 1: input validation
# ---------------------------------------------------------------------------

def _require(path: Path, produced_by: str) -> Path:
    if not path.exists():
        log.error("Missing: %s  (run %s first)", path, produced_by)
        sys.exit(1)
    return path


def validate_inputs(domain: Path) -> None:
    inp = domain / "input"

    # mod10 — terrain morphology (NC format)
    for name in ("dem.nc", "slope.nc", "aspect.nc", "fdir.nc", "facc.nc"):
        _require(inp / "morph" / name, "mod10_dem_to_mhm")

    # mod13 — land cover
    if not list((inp / "luse").glob("lc_*.asc")):
        log.error("Missing: lc_*.asc in %s/input/luse/  (run mod13_landcover_to_mhm first)", domain)
        sys.exit(1)

    # mod14 — soils
    for name in ("soil_class.asc", "soil_classdefinition.txt"):
        _require(inp / "morph" / name, "mod14_soils_to_mhm")

    # mod11 — meteorology (NC format)
    _require(inp / "meteo" / "pre"  / "pre.nc",   "mod11_meteo_to_mhm")
    _require(inp / "meteo" / "tavg" / "tavg.nc",  "mod11_meteo_to_mhm")

    # mod12 — PET (NC format)
    _require(inp / "meteo" / "pet" / "pet.nc",    "mod12_pet_to_mhm")

    # mod15 — gauges (NC format for raster, CSV for metadata)
    _require(inp / "gauge" / "id_map.csv",         "mod15_gauges_to_mhm")
    _require(inp / "gauge" / "idgauges.nc",         "mod15_gauges_to_mhm")

    # mod16 — latlon grid
    _require(inp / "latlon" / "latlon.nc",          "mod16_latlon_to_mhm")

    # mhm binary
    _require(Path(MHM_BINARY), "cmake build")

    log.info("All prerequisite files present.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    domain = Path(DOMAIN_DIR)
    inp    = domain / "input"

    # Phase 1 — validate
    validate_inputs(domain)

    # Phase 2 — extract metadata
    first_meteo, last_meteo = read_meteo_dates(inp / "meteo" / "pre" / "pre.nc")
    log.info("Meteo period: %s – %s", first_meteo, last_meteo)

    try:
        eval_start, eval_end = derive_eval_period(first_meteo, last_meteo, WARMING_DAYS)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)
    log.info("Eval period:  %s – %s  (warming_days=%d)", eval_start, eval_end, WARMING_DAYS)

    lcover_scenes = read_lcover_scenes(inp / "luse")
    log.info("Land cover scenes: %s", [f for _, f in lcover_scenes])

    gauges = read_gauge_info(inp / "gauge" / "id_map.csv")
    if not gauges:
        log.error("id_map.csv contains no gauges — calibration requires at least one.")
        sys.exit(1)
    log.info("Gauges: %s", [g["filename"] for g in gauges])

    resolution = RESOLUTION_HYDROLOGY
    log.info("L1 resolution: %d m (user-configured; L0 terrain is %d m)", resolution, L0_CELL_SIZE_M)

    n_soil_horizons, soil_depths = read_soil_info(inp / "morph" / "soil_classdefinition.txt")
    log.info("Soil horizons: %d  depths: %s mm", n_soil_horizons, soil_depths)

    # Phase 2b — bootstrap geology and LAI files if absent; resample ASCII inputs to L0
    _bootstrap_geology(inp / "morph")
    _bootstrap_lai(inp / "morph", inp / "luse")
    l0_nc = inp / "luse" / "lc_2024.nc"
    _resample_ascii_to_l0(inp / "morph" / "soil_class.asc", l0_nc)
    _build_idgauges_asc(inp / "morph", inp / "gauge", inp / "morph" / "dem.nc", l0_nc)
    _remap_fdir_to_arcgis(inp / "morph")
    _fix_facc_boundary_nodata(inp / "morph")
    _ensure_latlon_nc(inp / "latlon", inp / "morph" / "dem.nc", l0_nc, RESOLUTION_HYDROLOGY)

    # Phase 4 — create output directories and copy companion namelists
    (domain / "output").mkdir(parents=True, exist_ok=True)
    (domain / "restart").mkdir(parents=True, exist_ok=True)
    (inp / "optional_data").mkdir(parents=True, exist_ok=True)

    for src, name in (
        (REPO_PARAM_NML,   "mhm_parameter.nml"),
        (REPO_OUTPUT_NML,  "mhm_outputs.nml"),
        (REPO_MRM_OUT_NML, "mrm_outputs.nml"),
    ):
        dst = domain / name
        shutil.copy2(src, dst)
        log.info("Copied → %s", dst)

    # Phase 3 — generate mhm.nml
    nml_path = domain / "mhm.nml"
    write_mhm_nml(
        nml_path,
        domain,
        resolution_hydrology = resolution,
        timestep             = TIMESTEP,
        opti_method          = OPTI_METHOD,
        opti_function        = OPTI_FUNCTION,
        n_iterations         = N_ITERATIONS,
        warming_days         = WARMING_DAYS,
        eval_start           = eval_start,
        eval_end             = eval_end,
        lcover_scenes        = lcover_scenes,
        gauges               = gauges,
        n_soil_horizons      = n_soil_horizons,
        soil_depths          = soil_depths,
    )
    log.info("Written: %s", nml_path)

    # Phase 4b — repair morph NC axis order if files were written by old mod10
    _fix_morph_nc_axis_order(inp / "morph")

    # Phase 5 — run mHM calibration
    log.info("Launching mHM calibration: %s  (cwd=%s)", MHM_BINARY, domain)
    log.info("opti_method=%d  opti_function=%d  n_iterations=%d", OPTI_METHOD, OPTI_FUNCTION, N_ITERATIONS)

    proc = subprocess.Popen(
        [MHM_BINARY],
        cwd=str(domain),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        log.info("[mhm] %s", line.rstrip())
    proc.wait()

    if proc.returncode != 0:
        log.error("mHM exited with code %d", proc.returncode)
        sys.exit(proc.returncode)

    log.info("mHM calibration finished. Output in %s/output/", domain)


if __name__ == "__main__":
    main()
