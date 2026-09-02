"""Water-balance diagnostics and plausibility verdicts for mod20.

Each metric answers one of the review questions and carries a warn-only verdict:
values outside the plausible range are flagged but never abort the run.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class Metric:
    key: str
    question: str
    value: Optional[float]
    units: str
    low: Optional[float]
    high: Optional[float]
    verdict: str  # "OK", "WARN", or "N/A"
    note: str = ""


def _safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den not in (0.0, None) else None


def _verdict(
    value: Optional[float], low: Optional[float], high: Optional[float]
) -> str:
    if value is None:
        return "N/A"
    if low is not None and value < low:
        return "WARN"
    if high is not None and value > high:
        return "WARN"
    return "OK"


def compute_metrics(flux: Dict, inp: Dict) -> List[Metric]:
    """Build the diagnostic metric list from flux totals and input totals [mm]."""
    ft = flux["totals"]
    P_gross = inp["totals"]["pre"]
    PET_in = inp["totals"]["pet"]
    aET, PET_f = ft["aET"], ft["PET"]
    Q, QB, rech, preEff = ft["Q"], ft["QB"], ft["recharge"], ft["preEffect"]
    dS = flux["delta_storage"]

    rc_gross = _safe_div(Q, P_gross)
    rc_eff = _safe_div(Q, preEff)
    bfi = _safe_div(QB, Q)
    aet_pet = _safe_div(aET, PET_f)
    et_index = _safe_div(aET, P_gross)
    aridity = _safe_div(PET_in, P_gross)
    rech_ratio = _safe_div(rech, P_gross)
    rech_bf = _safe_div(rech, QB)
    if dS is None:
        residual = None
        closure_rel = None
        closure_note = (
            "storage states not written; "
            "set INCLUDE_STORAGE_STATES=True in main.py for ΔS closure"
        )
    else:
        residual = P_gross - aET - Q - dS
        closure_rel = _safe_div(residual, P_gross)
        closure_note = f"residual={residual:.0f} mm, ΔS={dS:.0f} mm"

    metrics: List[Metric] = [
        Metric(
            "runoff_coefficient",
            "Does the runoff coefficient look reasonable?",
            rc_gross,
            "Q/P (gross)",
            0.01,
            0.9,
            _verdict(rc_gross, 0.01, 0.9),
            f"ΣQ={Q:.0f} mm, ΣP_gross={P_gross:.0f} mm; gross P is an L2 (3km) "
            "clip and can read ~20% low vs the L1 grid — prefer Q/preEffect",
        ),
        Metric(
            "runoff_coefficient_effective",
            "Runoff coefficient vs. mHM effective precip (cross-check).",
            rc_eff,
            "Q/preEffect",
            0.01,
            1.0,
            _verdict(rc_eff, 0.01, 1.0),
            f"ΣpreEffect={preEff:.0f} mm",
        ),
        Metric(
            "baseflow_fraction",
            "Does the baseflow fraction look realistic?",
            bfi,
            "QB/Q",
            0.05,
            0.95,
            _verdict(bfi, 0.05, 0.95),
            f"ΣQB={QB:.0f} mm",
        ),
        Metric(
            "aet_pet_ratio",
            "Does ET look realistic (demand-limited)?",
            aet_pet,
            "aET/PET",
            None,
            1.0,
            _verdict(aet_pet, None, 1.0),
            f"ΣaET={aET:.0f} mm, ΣPET={PET_f:.0f} mm; aET must not exceed PET",
        ),
        Metric(
            "evaporative_index",
            "Does ET look realistic (Budyko water limit)?",
            et_index,
            "aET/P",
            None,
            1.0,
            _verdict(et_index, None, 1.0),
            f"aridity index PET/P={aridity:.2f}" if aridity is not None else "",
        ),
        Metric(
            "recharge_ratio",
            "Does groundwater recharge (L1_percol) make sense?",
            rech_ratio,
            "recharge/P",
            None,
            0.6,
            _verdict(rech_ratio, None, 0.6),
            f"Σrecharge={rech:.0f} mm",
        ),
        Metric(
            "recharge_vs_baseflow",
            "Long-term recharge should approximately balance baseflow.",
            rech_bf,
            "recharge/QB",
            0.5,
            2.0,
            _verdict(rech_bf, 0.5, 2.0),
        ),
        Metric(
            "water_balance_closure",
            "Overall water balance closure P - aET - Q - ΔS.",
            closure_rel,
            "residual/P",
            -0.1,
            0.1,
            _verdict(closure_rel, -0.1, 0.1),
            closure_note,
        ),
    ]
    return metrics


def metrics_to_dict(metrics: List[Metric]) -> List[dict]:
    return [asdict(m) for m in metrics]
