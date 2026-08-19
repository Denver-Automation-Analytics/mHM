"""mHM results diagnostics (mod20).

Runs mHM once in forward/evaluation mode using the calibrated parameter set
(FinalParam.nml produced by mod19), then inspects the gridded fluxes/states
against the meteo inputs to judge whether the runoff coefficient, baseflow
fraction, actual ET and groundwater recharge (L1_percol) look plausible.

Run order: mod10 → … → mod19 (calibration) → mod20 (diagnostics).

The forward run reads the fixed-name namelists from WORKING_DIR (cwd), so this
module edits mhm.nml / mhm_parameter.nml in place after backing up the
pre-diagnostics originals (*.precal). Verdicts are warn-only; the process
always exits 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import N_OMP_THREADS, WORKING_DIR, START_DATE, END_DATE, EVAL_START_DATE

from metrics import compute_metrics, metrics_to_dict
import plots
import stats
from readers import (read_fluxes, read_inputs, resolve_window,
                     read_discharge, read_discharge_from_qrouted, read_terrain,
                     read_parameters)

MHM_BINARY = "/workspace/build/mhm"
FLUX_FILE = "mHM_Fluxes_States.nc"
MRM_FLUX_FILE = "mRM_Fluxes_States.nc"

# outputFlxState cases the diagnostics read (mo_write_fluxes_states.f90):
#   9=PET, 10=aET, 11=Q(total runoff), 15=QB(baseflow), 16=recharge, 20=preEffect
DIAG_OUTPUT_CASES = {9, 10, 11, 15, 16, 20}
# Storage states needed to evaluate the ΔS water-balance-closure term.
STORAGE_OUTPUT_CASES = {1, 2, 3, 6, 7, 8}
# True also writes storage states so closure can be computed (much larger file);
# False keeps only the six flux fields for the smallest possible output.
INCLUDE_STORAGE_STATES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("diagnostics_to_mhm")


def _backup_once(path: Path) -> None:
    """Copy *path* to *path*.precal the first time, preserving the original."""
    bak = path.with_suffix(path.suffix + ".precal")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        log.info("Backed up %s → %s", path.name, bak.name)


def inject_calibrated_params(work: Path) -> None:
    """Swap the calibrated FinalParam.nml in as the run's mhm_parameter.nml."""
    final = work / "FinalParam.nml"
    param = work / "mhm_parameter.nml"
    if not final.exists():
        raise FileNotFoundError(
            f"Calibrated parameter file not found: {final}\n"
            "mod20 requires the calibrated parameters. Run mod19 calibration to "
            "completion first (it writes FinalParam.nml to WORKING_DIR).")
    _backup_once(param)
    shutil.copy2(final, param)
    log.info("Injected calibrated parameters: %s → %s", final.name, param.name)


def _set_eval_period(text: str, start: str, end: str) -> str:
    """Rewrite eval_Per(1) in mhm.nml to span *start*..*end* ('YYYY-MM-DD')."""
    ys, ms, ds = start.split("-")
    ye, me, de = end.split("-")
    subs = [
        (r"(eval_Per\(1\)%yStart\s*=\s*)\d+", int(ys)),
        (r"(eval_Per\(1\)%mStart\s*=\s*)\d+", f"{int(ms):02d}"),
        (r"(eval_Per\(1\)%dStart\s*=\s*)\d+", f"{int(ds):02d}"),
        (r"(eval_Per\(1\)%yEnd\s*=\s*)\d+", int(ye)),
        (r"(eval_Per\(1\)%mEnd\s*=\s*)\d+", f"{int(me):02d}"),
        (r"(eval_Per\(1\)%dEnd\s*=\s*)\d+", f"{int(de):02d}"),
    ]
    for pat, val in subs:
        text, n = re.subn(pat, rf"\g<1>{val}", text, count=1)
        if n == 0:
            raise ValueError(f"eval_Per key not found in mhm.nml: {pat}")
    return text


def write_forward_nml(work: Path) -> None:
    """Set optimize=.FALSE. and evaluate over the calibration window.

    eval_Per spans EVAL_START_DATE..END_DATE (not START_DATE): with a warm-up,
    START_DATE is the spin-up start and warming_Days already draws that lead-in
    forcing before EVAL_START_DATE. Using START_DATE here would push the sim
    period a warm-up length before the forcing record, so mHM rejects it
    ("time period of input data not matching modelling period")."""
    nml = work / "mhm.nml"
    if not nml.exists():
        raise FileNotFoundError(f"Missing {nml}. Run mod19 to assemble it first.")
    _backup_once(nml)
    text = nml.read_text()
    new_text, n = re.subn(
        r"^(\s*optimize\s*=\s*)\.TRUE\.",
        r"\1.FALSE.",
        text,
        count=1,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if n == 0 and not re.search(r"^\s*optimize\s*=\s*\.FALSE\.", text,
                                re.MULTILINE | re.IGNORECASE):
        raise ValueError(f"Could not find an 'optimize' switch in {nml}.")
    new_text = _set_eval_period(new_text, EVAL_START_DATE, END_DATE)
    new_text = _extend_lcover_start(new_text, START_DATE)
    nml.write_text(new_text)
    log.info("Forward-mode namelist ready (optimize=.FALSE., eval_Per=%s..%s): %s",
             EVAL_START_DATE, END_DATE, nml)


def _extend_lcover_start(text: str, start: str) -> str:
    """Extend the first land-cover scene's start year back to the sim start.

    Only one NLCD scene is usually available; mHM requires a land-cover period
    covering the whole simulation, so the single scene is applied to all years.
    """
    ys = int(start.split("-")[0])
    m = re.search(r"LCoverYearStart\(1\)\s*=\s*(\d+)", text)
    if m and int(m.group(1)) > ys:
        text = re.sub(r"(LCoverYearStart\(1\)\s*=\s*)\d+", rf"\g<1>{ys}", text, count=1)
        log.info("Extended LCoverYearStart(1) %s → %s to cover the run "
                 "(single land-cover scene applied to all years).", m.group(1), ys)
    return text


def write_minimal_outputs_nml(work: Path) -> None:
    """Enable only the outputFlxState fields the diagnostics read, to shrink the file."""
    nml = work / "mhm_outputs.nml"
    if not nml.exists():
        raise FileNotFoundError(f"Missing {nml}. Run mod19 to assemble it first.")
    _backup_once(nml)
    wanted = DIAG_OUTPUT_CASES | (STORAGE_OUTPUT_CASES if INCLUDE_STORAGE_STATES else set())

    def _toggle(m: "re.Match[str]") -> str:
        flag = ".TRUE." if int(m.group(2)) in wanted else ".FALSE."
        return f"{m.group('head')}{flag}"

    text = nml.read_text()
    new_text, n = re.subn(
        r"(?P<head>outputFlxState\((\d+)\)\s*=\s*)\.(?:TRUE|FALSE)\.",
        _toggle, text, flags=re.IGNORECASE)
    if n == 0:
        raise ValueError(f"No outputFlxState switches found in {nml}.")
    nml.write_text(new_text)
    log.info("Minimal mhm_outputs.nml: %d field-case(s) enabled (%s)",
             len(wanted), ", ".join(map(str, sorted(wanted))))


def _set_mrm_output_timestep(work: Path, ts: int = 1) -> None:
    """Set the mRM gridded-output cadence so Qrouted is written at the model step.

    Hourly Qrouted lets mod20 recover the gauge hydrograph from mRM_Fluxes_States.nc
    when mHM's at-exit crash truncates discharge.nc.
    """
    nml = work / "mrm_outputs.nml"
    if not nml.exists():
        raise FileNotFoundError(f"Missing {nml}. Run mod19 to assemble it first.")
    _backup_once(nml)
    text = nml.read_text()
    new_text, n = re.subn(r"^(\s*timeStep_model_outputs_mrm\s*=\s*)-?\d+",
                          rf"\g<1>{ts}", text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"timeStep_model_outputs_mrm not found in {nml}.")
    nml.write_text(new_text)
    log.info("Set timeStep_model_outputs_mrm=%d (hourly Qrouted fallback source).", ts)


def run_mhm(work: Path) -> None:
    """Launch the mHM binary in *work* and stream its output to the log."""
    if not Path(MHM_BINARY).exists():
        raise FileNotFoundError(f"mHM binary not found: {MHM_BINARY} (build it first).")
    log.info("Launching mHM forward run: %s  (cwd=%s)", MHM_BINARY, work)
    proc = subprocess.Popen(
        [MHM_BINARY],
        cwd=str(work),
        env={**os.environ, "OMP_NUM_THREADS": str(N_OMP_THREADS)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        log.info("[mhm] %s", line.rstrip())
    proc.wait()
    # mHM can crash in its at-exit cleanup (SIGSEGV/SIGABRT, negative code) AFTER
    # writing all outputs; the caller validates the flux file, so only a real STOP
    # error (positive exit code) should abort here.
    if proc.returncode < 0:
        log.warning("mHM terminated by signal %d after finishing; continuing "
                    "(outputs are validated next).", -proc.returncode)
    elif proc.returncode != 0:
        raise RuntimeError(f"mHM exited with code {proc.returncode}.")


def _print_summary(metrics, window) -> None:
    log.info("=" * 78)
    log.info("mHM RESULTS DIAGNOSTICS  (water-year window %s .. %s)", window[0], window[1])
    log.info("%-28s %12s %-14s %-6s", "metric", "value", "units", "verdict")
    log.info("-" * 78)
    for m in metrics:
        val = "n/a" if m.value is None else f"{m.value:12.3f}"
        log.info("%-28s %12s %-14s %-6s", m.key, val, m.units, m.verdict)
        if m.note:
            log.info("    %s", m.note)
    warns = [m.key for m in metrics if m.verdict == "WARN"]
    log.info("-" * 78)
    if warns:
        log.warning("%d metric(s) flagged for review: %s", len(warns), ", ".join(warns))
    else:
        log.info("All metrics within plausible ranges.")
    log.info("=" * 78)


def _print_hydro_summary(dstats) -> None:
    log.info("=" * 78)
    log.info("HYDROGRAPH SKILL (simulated vs. observed discharge)")
    log.info("%-10s %8s %8s %8s %9s %10s", "gauge", "KGE", "NSE", "logNSE", "PBIAS%", "n_obs")
    log.info("-" * 78)
    for s in dstats:
        if s.get("n_obs", 0) == 0:
            log.info("%-10s %8s %8s %8s %9s %10d", s["site_no"], "-", "-", "-", "-", 0)
            continue
        def _f(v):
            return "n/a" if v is None else f"{v:8.3f}"
        log.info("%-10s %8s %8s %8s %9s %10d", s["site_no"], _f(s["kge"]), _f(s["nse"]),
                 _f(s["lognse"]),
                 "n/a" if s["pbias_pct"] is None else f"{s['pbias_pct']:8.1f}",
                 s["n_obs"])
    log.info("=" * 78)


def _print_param_summary(pstat) -> None:
    log.info("=" * 78)
    log.info("CALIBRATED PARAMETERS: %d free, %d rail-pinned, %d fixed",
             pstat["n_free"], pstat["n_railed"], pstat["n_fixed"])
    if pstat["railed"]:
        log.warning("Rail-pinned (value at a bound): %s", ", ".join(pstat["railed"]))
    else:
        log.info("No free parameter is rail-pinned — healthy calibration spread.")
    log.info("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect mHM results (water balance).")
    ap.add_argument("--skip-run", action="store_true",
                    help="Reuse the existing flux file instead of running mHM.")
    ap.add_argument("--spinup-years", type=int, default=1,
                    help="Water years to drop as spin-up before analysis (default 1).")
    args = ap.parse_args()

    work = Path(WORKING_DIR)
    out_dir = work / "output"
    flux_nc = out_dir / FLUX_FILE

    if args.skip_run:
    # if True:
        if not flux_nc.exists():
            raise FileNotFoundError(
                f"--skip-run set but {flux_nc} is missing; run without it first.")
        log.info("Skipping forward run; reusing %s", flux_nc)
    else:
        inject_calibrated_params(work)
        write_forward_nml(work)
        write_minimal_outputs_nml(work)
        _set_mrm_output_timestep(work, 1)   # hourly Qrouted for the discharge fallback
        # Remove stale gridded outputs so mHM writes fresh files (avoids file locks).
        for stale in (flux_nc, out_dir / "mRM_Fluxes_States.nc"):
            if stale.exists():
                stale.unlink()
        run_mhm(work)
        if not flux_nc.exists():
            raise FileNotFoundError(
                f"Forward run finished but {flux_nc} was not written. "
                "Check outputFlxState switches in mhm_outputs.nml.")

    log.info("Reading fluxes: %s", flux_nc)
    window = resolve_window(flux_nc, args.spinup_years)
    log.info("Diagnostics window (full water years): %s .. %s", window[0], window[1])
    flux = read_fluxes(flux_nc, window)
    inp = read_inputs(
        work / "mhm_input/meteo/pre/pre.nc",
        work / "mhm_input/meteo/pet/pet.nc",
        window,
    )

    metrics = compute_metrics(flux, inp)
    _print_summary(metrics, window)

    # Hydrograph (sim vs obs), terrain and precipitation diagnostics. Each is
    # optional: a missing file is warned about but never aborts the run.
    disch = dstats = terr = tstats = None
    try:
        # mRM writes gauge discharge daily to discharge.nc and, for sub-daily
        # (hourly) runs, additionally at the model step to subdaily_discharge.nc;
        # prefer the latter so the hydrograph and skill metrics stay hourly.
        disch_nc = out_dir / "subdaily_discharge.nc"
        if not disch_nc.exists():
            disch_nc = out_dir / "discharge.nc"
        try:
            disch = read_discharge(disch_nc, work / "mhm_input/gauge", window)
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            # mHM's at-exit crash can truncate discharge.nc; recover the hydrograph
            # from the routed-flow grid (written before the crash).
            log.warning("discharge file unusable (%s); recovering hydrograph from %s.",
                        exc, MRM_FLUX_FILE)
            disch = read_discharge_from_qrouted(out_dir / MRM_FLUX_FILE,
                                                work / "mhm_input/gauge", window)
        dstats = stats.discharge_stats(disch)
        _print_hydro_summary(dstats)
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        log.warning("Hydrograph diagnostics skipped: %s", exc)
    try:
        terr = read_terrain(work / "mhm_input/morph")
        tstats = stats.terrain_stats(terr)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        log.warning("Terrain diagnostics skipped: %s", exc)
    pstats = stats.precip_stats(inp, window)

    params = param_stats = None
    try:
        pfile = work / "FinalParam.nml"
        if not pfile.exists():
            pfile = work / "mhm_parameter.nml"
        params = read_parameters(pfile)
        param_stats = stats.parameter_stats(params)
        _print_param_summary(param_stats)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("Parameter diagnostics skipped: %s", exc)

    report = {
        "eval_start": window[0],
        "eval_end": window[1],
        "totals_mm": {**flux["totals"],
                      "pre_input": inp["totals"]["pre"],
                      "pet_input": inp["totals"]["pet"],
                      "delta_storage": flux["delta_storage"]},
        "metrics": metrics_to_dict(metrics),
        "hydrograph": dstats,
        "terrain": tstats,
        "precipitation": pstats,
        "parameters": param_stats,
    }
    report_path = out_dir / "diagnostics_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote report: %s", report_path)

    plot_dir = out_dir / "diagnostics"
    paths = plots.write_all(flux, inp, metrics, plot_dir,
                            disch=disch, dstats=dstats,
                            terr=terr, tstats=tstats, pstats=pstats,
                            params=params)
    log.info("Wrote %d diagnostic plots to %s", len(paths), plot_dir)


if __name__ == "__main__":
    main()
