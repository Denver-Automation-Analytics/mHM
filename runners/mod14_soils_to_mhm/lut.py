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
    # Valid cells: sand layer-1 has a real value (nodata = -9999 < 0)
    valid = layers["sn"][1] >= 0

    soil_id = np.full((nrows, ncols), nodata, dtype=np.int32)
    lut_rows: list[tuple] = []
    soil_count = 0

    # Column-major iteration matches the Fortran loop order
    for col in range(ncols):
        for row in range(nrows):
            if not valid[row, col]:
                continue
            soil_count += 1
            soil_id[row, col] = soil_count
            for layer_idx, (ud, ld) in enumerate(_LAYER_DEPTHS, start=1):
                cl = int(layers["cl"][layer_idx][row, col])
                sn = int(layers["sn"][layer_idx][row, col])
                bd = int(layers["bd"][layer_idx][row, col]) / 1000.0
                lut_rows.append((soil_count, layer_idx, ud, ld, cl, sn, bd))

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
