#!/usr/bin/env python3
"""
run_mhm.py — Runner script for mHM basin simulations using pre-calibrated parameters.

Reads a JSON config file that specifies the basin ID, meteorology inputs, static inputs,
simulation period, and output paths. Pre-calibrated parameters are loaded from the
basin's calib_001/ subfolder.

Usage:
    python run_mhm.py run_mhm_config.json

Outputs include mHM_Fluxes_States.nc containing spatially distributed fluxes and states,
including Q (total surface runoff per grid cell, L1_total_runoff, [mm/timestep]).

Meteo inputs must follow the test_domain format:
    <dir_precipitation>/pre.nc
    <dir_temperature>/tavg.nc
    <dir_reference_et>/pet.nc
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Required config keys
# ---------------------------------------------------------------------------

_REQUIRED_METEO = ["dir_precipitation", "dir_temperature", "dir_reference_et"]
_REQUIRED_STATIC = ["dir_morpho", "dir_lcover", "dir_common_files", "file_latlon", "dir_gauges", "gauges"]
_REQUIRED_SIM = ["start_date", "end_date"]


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def validate_config(cfg: dict) -> None:
    """Validate required keys and verify all referenced paths exist."""
    errors = []

    # Top-level keys
    for key in ("basin_id", "basins_root", "mhm_binary", "meteo", "static_inputs",
                "simulation", "output_dir"):
        if key not in cfg:
            errors.append(f"Missing top-level key: '{key}'")
    if errors:
        _abort(errors)

    # Meteo keys
    for k in _REQUIRED_METEO:
        if k not in cfg["meteo"]:
            errors.append(f"Missing meteo key: '{k}'")

    # Static input keys
    for k in _REQUIRED_STATIC:
        if k not in cfg["static_inputs"]:
            errors.append(f"Missing static_inputs key: '{k}'")

    # Simulation keys
    for k in _REQUIRED_SIM:
        if k not in cfg["simulation"]:
            errors.append(f"Missing simulation key: '{k}'")

    if errors:
        _abort(errors)

    # Parse and validate dates
    try:
        start = datetime.strptime(cfg["simulation"]["start_date"], "%Y-%m-%d")
        end = datetime.strptime(cfg["simulation"]["end_date"], "%Y-%m-%d")
        if end <= start:
            errors.append("simulation.end_date must be after simulation.start_date")
    except ValueError as exc:
        errors.append(f"Invalid date format (expected YYYY-MM-DD): {exc}")

    # Timestep value
    ts = cfg["simulation"].get("timestep", 1)
    if ts not in (1, 24):
        errors.append(f"simulation.timestep must be 1 or 24, got {ts}")

    # Basin calib_001/exe directory and required NML files
    basin_exe = Path(cfg["basins_root"]) / cfg["basin_id"] / "calib_001" / "exe"
    if not basin_exe.is_dir():
        errors.append(f"Basin calib_001/exe not found: {basin_exe}")
    else:
        for fname in ("mhm.nml", "mhm_parameter.nml"):
            if not (basin_exe / fname).exists():
                errors.append(f"Missing {fname} in {basin_exe}")

    # mHM binary
    binary = Path(cfg["mhm_binary"])
    if not binary.exists():
        errors.append(f"mHM binary not found: {binary}")
    elif not os.access(binary, os.X_OK):
        errors.append(f"mHM binary is not executable: {binary}")

    # Meteo directories
    for key in _REQUIRED_METEO:
        p = Path(cfg["meteo"][key])
        if not p.is_dir():
            errors.append(f"Meteo directory not found: {p}  ({key})")

    # Static input directories and files
    si = cfg["static_inputs"]
    for key in ("dir_morpho", "dir_lcover", "dir_common_files", "dir_gauges"):
        p = Path(si[key])
        if not p.is_dir():
            errors.append(f"Static input directory not found: {p}  ({key})")
    if not Path(si["file_latlon"]).exists():
        errors.append(f"LatLon file not found: {si['file_latlon']}")

    # Gauge list
    gauges = si["gauges"]
    if not isinstance(gauges, list) or len(gauges) == 0:
        errors.append("static_inputs.gauges must be a non-empty list of {id, filename} objects")
    else:
        for i, g in enumerate(gauges):
            if not isinstance(g, dict) or "id" not in g or "filename" not in g:
                errors.append(f"Gauge entry [{i}] must have 'id' and 'filename' keys")
            else:
                gfile = Path(si["dir_gauges"]) / g["filename"]
                if not gfile.exists():
                    errors.append(f"Gauge file not found: {gfile}")

    # Workspace NML files (next to this script)
    script_dir = Path(__file__).parent
    for fname in ("mhm_outputs.nml", "mrm_outputs.nml"):
        if not (script_dir / fname).exists():
            errors.append(f"Workspace NML not found: {script_dir / fname}")

    if errors:
        _abort(errors)


def _abort(errors: list) -> None:
    print("Configuration errors:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# NML patching
# ---------------------------------------------------------------------------

def _quote_dir(path) -> str:
    """Return a path string with a trailing slash, wrapped in double quotes."""
    return '"' + str(path).rstrip("/") + '/"'


def _quote_file(path) -> str:
    """Return a file path string wrapped in double quotes."""
    return f'"{path}"'


def _sub(pattern: str, replacement: str, text: str, warn_label: str = "") -> str:
    """Apply a regex substitution; warn if the pattern is not found."""
    new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
    if n == 0:
        label = warn_label or pattern
        print(f"  [warn] NML pattern not matched (skipped): {label}", file=sys.stderr)
    return new_text


def patch_mhm_nml(text: str, cfg: dict, restart_dir: Path) -> str:
    """
    Apply all path, date, and timestep substitutions to the mhm.nml text.
    Returns the patched NML as a string.
    """
    si = cfg["static_inputs"]
    mt = cfg["meteo"]
    sim = cfg["simulation"]
    out = cfg["output_dir"]

    ts = sim.get("timestep", 1)
    tsmi = sim.get("time_step_model_inputs", 0)
    wd = sim.get("warming_days", 0)

    start = datetime.strptime(sim["start_date"], "%Y-%m-%d")
    end = datetime.strptime(sim["end_date"], "%Y-%m-%d")

    substitutions = [
        # timestep appears in both &mainconfig and &mainconfig_mhm_mrm — replace all
        (r'((?:^|\s)timestep\s*=\s*)\d+',
         rf'\g<1>{ts}',
         "timestep"),

        # &directories_general
        (r'(dirConfigOut\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(out)}',
         "dirConfigOut"),
        (r'(dirCommonFiles\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(si["dir_common_files"])}',
         "dirCommonFiles"),
        (r'(dir_Morpho\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(si["dir_morpho"])}',
         "dir_Morpho(1)"),
        (r'(dir_LCover\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(si["dir_lcover"])}',
         "dir_LCover(1)"),
        (r'(dir_RestartIn\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(restart_dir)}',
         "dir_RestartIn(1)"),
        (r'(dir_RestartOut\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(restart_dir)}',
         "dir_RestartOut(1)"),
        (r'(dir_Out\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(out)}',
         "dir_Out(1)"),
        (r'(file_LatLon\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_file(si["file_latlon"])}',
         "file_LatLon(1)"),

        # &directories_mRM
        (r'(dir_Gauges\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(si["dir_gauges"])}',
         "dir_Gauges(1)"),

        # &directories_mHM
        (r'(dir_Precipitation\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(mt["dir_precipitation"])}',
         "dir_Precipitation(1)"),
        (r'(dir_Temperature\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(mt["dir_temperature"])}',
         "dir_Temperature(1)"),
        (r'(dir_ReferenceET\(1\)\s*=\s*)"[^"]*"',
         rf'\1{_quote_dir(mt["dir_reference_et"])}',
         "dir_ReferenceET(1)"),

        # &time_periods
        (r'(warming_Days\(1\)\s*=\s*)\d+',
         rf'\g<1>{wd}',
         "warming_Days(1)"),
        (r'(eval_Per\(1\)%yStart\s*=\s*)\d+',
         rf'\g<1>{start.year}',
         "eval_Per(1)%yStart"),
        (r'(eval_Per\(1\)%mStart\s*=\s*)\d+',
         rf'\g<1>{start.month:02d}',
         "eval_Per(1)%mStart"),
        (r'(eval_Per\(1\)%dStart\s*=\s*)\d+',
         rf'\g<1>{start.day:02d}',
         "eval_Per(1)%dStart"),
        (r'(eval_Per\(1\)%yEnd\s*=\s*)\d+',
         rf'\g<1>{end.year}',
         "eval_Per(1)%yEnd"),
        (r'(eval_Per\(1\)%mEnd\s*=\s*)\d+',
         rf'\g<1>{end.month:02d}',
         "eval_Per(1)%mEnd"),
        (r'(eval_Per\(1\)%dEnd\s*=\s*)\d+',
         rf'\g<1>{end.day:02d}',
         "eval_Per(1)%dEnd"),
        (r'(time_step_model_inputs\(1\)\s*=\s*)-?\d+',
         rf'\g<1>{tsmi}',
         "time_step_model_inputs(1)"),
    ]

    for pattern, replacement, label in substitutions:
        text = _sub(pattern, replacement, text, warn_label=label)

    # Regenerate the &evaluation_gauges block
    text = _patch_evaluation_gauges(text, si["gauges"])

    return text


def _patch_evaluation_gauges(text: str, gauges: list) -> str:
    """
    Replace the entire &evaluation_gauges namelist block with entries
    derived from the gauges list: [{id: ..., filename: ...}, ...].
    All gauges belong to basin (subbasin index 1).
    """
    n = len(gauges)
    lines = [
        "&evaluation_gauges",
        f"nGaugesTotal        = {n}",
        f"NoGauges_basin(1)   = {n}",
    ]
    for j, g in enumerate(gauges, start=1):
        lines.append(f'Gauge_id(1,{j})       = {g["id"]}')
        lines.append(f'gauge_filename(1,{j}) = "{g["filename"]}"')
    lines.append("/")
    new_block = "\n".join(lines)

    # Match from &evaluation_gauges to the next standalone /
    pattern = r'&evaluation_gauges\b.*?^/'
    result, n_subs = re.subn(pattern, new_block, text, flags=re.DOTALL | re.MULTILINE)
    if n_subs == 0:
        raise ValueError("Could not locate &evaluation_gauges block in mhm.nml")
    return result


def ensure_total_runoff_enabled(text: str) -> str:
    """
    Ensure outputFlxState(11) = .TRUE. in mhm_outputs.nml text.
    L1_total_runoff is the spatially distributed total surface runoff [mm/timestep].
    """
    # Replace .FALSE. → .TRUE. if already present
    patched, n = re.subn(
        r'(outputFlxState\(11\)\s*=\s*)\.FALSE\.',
        r'\1.TRUE.',
        text,
        flags=re.IGNORECASE,
    )
    if n > 0:
        return patched
    # Already .TRUE. or not present — check for it
    if re.search(r'outputFlxState\(11\)', patched, re.IGNORECASE):
        return patched  # already enabled
    # Insert before closing / of the NLoutputResults namelist
    patched = re.sub(
        r'(/\s*\Z)',
        '  outputFlxState(11)=.TRUE.  ! L1_total_runoff [mm/T]\n/',
        patched,
        count=1,
        flags=re.MULTILINE,
    )
    return patched


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)
    validate_config(cfg)

    basin_id = cfg["basin_id"]
    basin_exe = Path(cfg["basins_root"]) / basin_id / "calib_001" / "exe"
    script_dir = Path(__file__).parent
    output_dir = Path(cfg["output_dir"]).resolve()

    # Set up working directory
    work_dir_cfg = cfg.get("work_dir")
    if work_dir_cfg:
        work_dir = Path(work_dir_cfg).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        keep_work = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"mhm_{basin_id}_"))
        keep_work = cfg.get("keep_work_dir", False)

    restart_dir = work_dir / "restart"

    print(f"Basin:            {basin_id}")
    print(f"Parameters:       {basin_exe / 'mhm_parameter.nml'}")
    print(f"Working directory:{work_dir}")
    print(f"Output directory: {output_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        restart_dir.mkdir(parents=True, exist_ok=True)

        # mhm_parameter.nml — copy pre-calibrated parameters verbatim
        shutil.copy2(basin_exe / "mhm_parameter.nml", work_dir / "mhm_parameter.nml")

        # mhm.nml — copy from calib_001, patch all paths and time period
        nml_text = (basin_exe / "mhm.nml").read_text()
        nml_text = patch_mhm_nml(nml_text, cfg, restart_dir)
        (work_dir / "mhm.nml").write_text(nml_text)

        # mhm_outputs.nml — copy from workspace root, ensure L1_total_runoff is on
        outputs_text = (script_dir / "mhm_outputs.nml").read_text()
        outputs_text = ensure_total_runoff_enabled(outputs_text)
        (work_dir / "mhm_outputs.nml").write_text(outputs_text)

        # mrm_outputs.nml — copy verbatim
        shutil.copy2(script_dir / "mrm_outputs.nml", work_dir / "mrm_outputs.nml")

        # Run mHM binary
        binary = str(Path(cfg["mhm_binary"]).resolve())
        sim = cfg["simulation"]
        print(
            f"\nRunning: {binary}\n"
            f"  Period:   {sim['start_date']} → {sim['end_date']}\n"
            f"  Timestep: {sim.get('timestep', 1)}h\n"
            f"  Gauges:   {len(cfg['static_inputs']['gauges'])}"
        )

        result = subprocess.run(
            [binary],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print(result.stdout)

        last_lines = result.stdout.strip().splitlines()[-10:]
        finished = any("mHM: Finished!" in line for line in last_lines)

        if result.returncode != 0 or not finished:
            print("ERROR: mHM did not complete successfully.", file=sys.stderr)
            if not finished:
                print('  "mHM: Finished!" not found in output.', file=sys.stderr)
            sys.exit(result.returncode or 1)

        print(f"\nSuccess. Output written to: {output_dir}")
        nc_files = sorted(output_dir.glob("*.nc"))
        if nc_files:
            print("NetCDF output files:")
            for f in nc_files:
                print(f"  {f.name}")
            print(
                "\nNote: Total surface runoff (L1_total_runoff) is available in\n"
                "      mHM_Fluxes_States.nc as variable 'Q' [mm/month or mm/day]."
            )

    finally:
        if keep_work or work_dir_cfg:
            print(f"\nWork directory retained: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.json>", file=sys.stderr)
        sys.exit(1)
    run('/workspace/run_mhm_config.json')
