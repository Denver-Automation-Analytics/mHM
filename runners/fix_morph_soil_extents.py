#!/usr/bin/env python
"""Crop/resample morph soil ASCII files to the canonical L0 domain.

Source: test_domain_3/input/morph/{bd,cl,sn}0[1-6].txt  (1440 x 1944)
Target: canonical domain matching gauge/luse              (1374 x 1921)
"""

from __future__ import annotations
import os
import re
import tempfile
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Canonical L0 domain
# ---------------------------------------------------------------------------
XMIN, YMIN, XMAX, YMAX = -441000.0, 312750.0, -97500.0, 793000.0
NCOLS, NROWS = 1374, 1921
CELLSIZE = 250
NODATA = -9999

MORPH_DIR = Path(__file__).parent.parent / "test_domain_3/input/morph"

FILES = (
    [f"bd{i:02d}.txt" for i in range(1, 7)]
    + [f"cl{i:02d}.txt" for i in range(1, 7)]
    + [f"sn{i:02d}.txt" for i in range(1, 7)]
)

WARP_OPTS = gdal.WarpOptions(
    outputBounds=(XMIN, YMIN, XMAX, YMAX),
    width=NCOLS,
    height=NROWS,
    resampleAlg="near",
    srcNodata=NODATA,
    dstNodata=NODATA,
)


def _fix_header(path: Path) -> None:
    """Rewrite the 6-line header to the canonical mHM format."""
    text = path.read_text()
    lines = text.splitlines()

    # Replace the first 6 lines with canonical values; keep data rows intact.
    canonical = [
        f"ncols        {NCOLS}",
        f"nrows        {NROWS}",
        f"xllcorner    {float(XMIN)}",
        f"yllcorner    {float(YMIN)}",
        f"cellsize     {CELLSIZE}",
        f"NODATA_value {NODATA}",
    ]

    header_count = 0
    new_lines = []
    for line in lines:
        if header_count < 6 and re.match(r"^[a-zA-Z]", line.strip()):
            new_lines.append(canonical[header_count])
            header_count += 1
        else:
            new_lines.append(line)

    path.write_text("\n".join(new_lines) + "\n")


def process(fname: str) -> None:
    src = MORPH_DIR / fname
    if not src.is_file():
        raise FileNotFoundError(src)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
        tmp_tif_path = tmp_tif.name
    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp_asc:
        tmp_asc_path = tmp_asc.name

    try:
        gdal.Warp(tmp_tif_path, str(src), options=WARP_OPTS)
        gdal.Translate(tmp_asc_path, tmp_tif_path, format="AAIGrid",
                       noData=NODATA)

        # GDAL AAIGrid may write lowercase/float keys — normalise to mHM format.
        _fix_header(Path(tmp_asc_path))

        # Overwrite the original in-place.
        src.write_text(Path(tmp_asc_path).read_text())
        print(f"  updated {fname}")
    finally:
        for p in (tmp_tif_path, tmp_asc_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        # gdal_translate also creates a .prj sidecar
        for p in (tmp_asc_path.replace(".asc", ".prj"),):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    print(f"Target domain: {NCOLS}x{NROWS}, xll={XMIN}, yll={YMIN}")
    for fname in FILES:
        process(fname)
    print("Done. Verify with: head -6 test_domain_3/input/morph/bd01.txt")
