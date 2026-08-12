"""
Generate soil_classdefinition.txt and soil_class.asc from the 6-layer grids.

Replicates the logic of pre-proc/process_soilgrid/_main_prepare_lut.f90 in
Python, removing the need to compile or run any Fortran code.

Layer depths (mm) match the six GlobalSoilMap standard intervals exactly:
  Layer  1:    0 –   50 mm  (0 –   5 cm)
  Layer  2:   50 –  150 mm  (5 –  15 cm)
  Layer  3:  150 –  300 mm  (15 –  30 cm)
  Layer  4:  300 –  600 mm  (30 –  60 cm)
  Layer  5:  600 – 1000 mm  (60 – 100 cm)
  Layer  6: 1000 – 2000 mm  (100 – 200 cm)
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_LAYER_DEPTHS: list[tuple[int, int]] = [
    (0,    50),
    (50,   150),
    (150,  300),
    (300,  600),
    (600,  1000),
    (1000, 2000),
]

# Default fallback soil type appended as the last entry (fills masked cells)
_DEFAULT = {"cl": 33, "sn": 33, "bd_gcm3": 1.5}

# Quantisation bin sizes for soil-type dedup (aggressive: ~20k types).
# clay/sand in %, bulk density in mg/cm³ (200 = 0.2 g/cm³).
_CLSN_BIN_PCT = 20
_BD_BIN_MGCM3 = 200


def _quantise_profiles(keys: np.ndarray) -> np.ndarray:
    """
    Bin clay/sand (%) and bulk density (mg/cm³) so near-identical soil columns
    map to one type. Clay and sand are clamped to keep clay + sand ≤ 100 %
    (silt ≥ 0) per horizon after rounding.
    """
    q = keys.astype(np.int32).copy()
    cl_idx = np.arange(0, 18, 3)
    sn_idx = np.arange(1, 18, 3)
    bd_idx = np.arange(2, 18, 3)

    cl_q = np.clip(np.round(q[:, cl_idx] / _CLSN_BIN_PCT) * _CLSN_BIN_PCT, 0, 100)
    sn_q = np.round(q[:, sn_idx] / _CLSN_BIN_PCT) * _CLSN_BIN_PCT
    sn_q = np.clip(sn_q, 0, 100 - cl_q)
    bd_q = np.round(q[:, bd_idx] / _BD_BIN_MGCM3) * _BD_BIN_MGCM3

    q[:, cl_idx] = cl_q
    q[:, sn_idx] = sn_q
    q[:, bd_idx] = bd_q
    return q


def build_lut(
    layers: dict[str, dict[int, np.ndarray]],
    header: dict,
    out_dir: Path,
    nodata: int = -9999,
) -> None:
    """
    Build soil_classdefinition.txt and soil_class.asc.

    layers["bd"][k]  int32 mg/cm³   (divide by 1000 → g/cm³)
    layers["cl"][k]  int32 integer % clay
    layers["sn"][k]  int32 integer % sand
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Zero-value replacement (mirrors Fortran: zeros are unphysical fill values)
    for k in range(1, 7):
        layers["bd"][k] = np.where(layers["bd"][k] == 0, 1500, layers["bd"][k])
        layers["cl"][k] = np.where(layers["cl"][k] == 0, _DEFAULT["cl"], layers["cl"][k])
        layers["sn"][k] = np.where(layers["sn"][k] == 0, _DEFAULT["sn"], layers["sn"][k])

    nrows, ncols = header["nrows"], header["ncols"]
    # Valid cells require every property in every horizon (no nodata leaks into
    # a soil type; cells missing any layer fall back to the default type).
    valid = np.ones((nrows, ncols), dtype=bool)
    for prop in ("cl", "sn", "bd"):
        for k in range(1, 7):
            valid &= layers[prop][k] >= 0

    # Stack the 18 per-cell values (cl, sn, bd x 6 horizons) into a profile key,
    # then quantise so near-identical columns collapse to one soil type. Without
    # binning, continuous SoilGrids texture yields ~one type per cell.
    profile = np.stack(
        [layers[prop][k] for k in range(1, 7) for prop in ("cl", "sn", "bd")],
        axis=-1,
    ).astype(np.int32)
    keys = _quantise_profiles(profile[valid])

    soil_id = np.full((nrows, ncols), nodata, dtype=np.int32)
    lut_rows: list[tuple] = []

    if keys.size:
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        soil_id[valid] = np.ravel(inverse).astype(np.int32) + 1
        for uid, prof in enumerate(uniq, start=1):
            for layer_idx, (ud, ld) in enumerate(_LAYER_DEPTHS, start=1):
                base = (layer_idx - 1) * 3
                cl = int(prof[base])
                sn = int(prof[base + 1])
                bd = int(prof[base + 2]) / 1000.0
                lut_rows.append((uid, layer_idx, ud, ld, cl, sn, bd))
        soil_count = int(uniq.shape[0])
    else:
        soil_count = 0

    # Append the default fallback type; masked cells in soil_id point to it
    default_id = soil_count + 1
    for layer_idx, (ud, ld) in enumerate(_LAYER_DEPTHS, start=1):
        lut_rows.append((
            default_id, layer_idx, ud, ld,
            _DEFAULT["cl"], _DEFAULT["sn"], _DEFAULT["bd_gcm3"],
        ))
    soil_id = np.where(soil_id == nodata, default_id, soil_id)

    log.info(
        "Soil types: %d unique + 1 default = %d total",
        soil_count, default_id,
    )

    _write_classdefinition(out_dir / "soil_classdefinition.txt", lut_rows, default_id)
    _write_soil_class_asc(out_dir / "soil_class.asc", header, soil_id, nodata)


def _write_classdefinition(path: Path, rows: list[tuple], n_types: int) -> None:
    with open(path, "w") as fh:
        fh.write(f"nSoil_Types  {n_types}\n")
        fh.write("MU_GLOBAL\tHORIZON\tUD[mm]\tLD[mm]\tCLAY[%]\tSAND[%]\tBD[gcm-3]\n")
        for sid, hor, ud, ld, cl, sn, bd in rows:
            fh.write(f"{sid}\t{hor}\t{ud}\t{ld}\t{cl:.1f}\t{sn:.1f}\t{bd:.3f}\n")
    log.info("Wrote %s  (%d data rows)", path, len(rows))


def _write_soil_class_asc(
    path: Path, header: dict, soil_id: np.ndarray, nodata: int
) -> None:
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
        np.savetxt(fh, soil_id, fmt="%d")
    log.info("Wrote %s", path)
