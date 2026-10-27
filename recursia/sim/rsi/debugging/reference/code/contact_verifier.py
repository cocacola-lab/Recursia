"""Independent verifier for contact / grasp / hold / release postconditions (round contact_primitive_20260917_112613).

Pure functions over public data only (the primitive's recorded trace and the public gripper opening fraction). Separate code
path from contact_primitive.py with its own thresholds; it cannot see which body was touched (runner truth decides offline).
Gripper constants: opening fraction = finger qpos / 0.04; an empty Panda gripper closes to ~0.015; max opening 1.0 (8 cm).
"""
from __future__ import annotations

import numpy as np

V_STATIONARY = 0.003      # m over the hold
V_SHORTFALL = 0.007       # m along the axis
V_ALIGN = 0.5
V_RATIO = 0.5             # actual/commanded advance over the pre-hold waypoints
GRASP_MIN_OPENING = 0.06  # fraction: >= ~5 mm object width between the pads (empty closed ~0.015)
GRASP_MIN_FLOOR = 0.03    # N15 amendment (user 2026-10-09: "allow changing ... thin-walled object threshold"): a caller may pass a measured
                          # min_opening for a thin wall/handle; it is clamped at 2x the empty-closed reading (~0.015), ~2.4 mm.
GRASP_MIN_TRAVEL = 0.02   # the fingers must have moved
HOLD_TOL = 0.03
RELEASE_MIN = 0.85
STATUSES = ("verified", "refuted", "uncertain", "failed")


def verify_contact(result) -> dict:
    """Re-derive contact evidence from the recorded trace/hold of a contact_move result."""
    if result is None or not isinstance(result, dict):
        return {"status": "uncertain", "reason": "no result", "checks": {}}
    hold = result.get("hold"); trace = result.get("trace") or []
    if hold is None:
        return {"status": "refuted" if result.get("status") == "contact_established" else "uncertain", "reason": "no confirmation hold recorded", "checks": {}}
    checks = {}
    checks["stationary"] = bool(hold.get("moved_m", 1.0) <= V_STATIONARY)
    checks["shortfall"] = bool(hold.get("shortfall_along", 0.0) >= V_SHORTFALL)
    checks["aligned"] = bool(hold.get("align", 0.0) >= V_ALIGN)
    wps = [t for t in trace if "wp" in t and not t.get("hold")]
    if len(wps) >= 2:
        pre = wps[-2:]; c_adv = pre[-1]["c"] - (wps[-3]["c"] if len(wps) >= 3 else 0.0); a_adv = pre[-1]["a"] - (wps[-3]["a"] if len(wps) >= 3 else 0.0)
        ratio = a_adv / c_adv if c_adv > 1e-9 else 1.0
        checks["low_progress"] = bool(ratio <= V_RATIO); checks["pre_hold_ratio"] = round(float(ratio), 3)
    else:
        checks["low_progress"] = False
    ok = all(checks[k] for k in ("stationary", "shortfall", "aligned", "low_progress"))
    if ok:
        return {"status": "verified", "reason": "hold stationary, axial shortfall above sag, low pre-hold progress", "checks": checks}
    if result.get("status") in ("contact_established",):
        return {"status": "refuted", "reason": "primitive claimed contact but the independent checks disagree: " + ", ".join(k for k in ("stationary", "shortfall", "aligned", "low_progress") if not checks[k]), "checks": checks}
    return {"status": "uncertain", "reason": "evidence insufficient: " + ", ".join(k for k in ("stationary", "shortfall", "aligned", "low_progress") if not checks[k]), "checks": checks}


def _min_opening(v):
    if v is None:
        return GRASP_MIN_OPENING
    return max(float(v), GRASP_MIN_FLOOR)


def _frac(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def verify_grasp(opening_before, opening_after, min_opening=None) -> dict:
    b = _frac(opening_before); a = _frac(opening_after); MIN = _min_opening(min_opening)
    if a is None or b is None:
        return {"status": "uncertain", "reason": "gripper opening not observable", "before": b, "after": a}
    if a < MIN:
        return {"status": "failed", "detail": "nothing_between_fingers", "reason": f"opening after close {a:.3f} < {MIN} (empty gripper)", "before": b, "after": a}
    if b - a < GRASP_MIN_TRAVEL:
        return {"status": "uncertain", "reason": f"fingers did not move ({b:.3f} -> {a:.3f})", "before": b, "after": a}
    return {"status": "verified", "reason": f"object between the pads: opening {a:.3f} after closing from {b:.3f}", "before": b, "after": a, "object_width_est_m": round(a * 0.08, 4)}


def verify_hold(opening_ref, opening_now, min_opening=None) -> dict:
    r = _frac(opening_ref); n = _frac(opening_now); MIN = _min_opening(min_opening)
    if r is None or n is None:
        return {"status": "uncertain", "reason": "gripper opening not observable", "ref": r, "now": n}
    if n < MIN:
        return {"status": "failed", "detail": "object_lost", "reason": f"opening {n:.3f} collapsed to the empty-closed value", "ref": r, "now": n}
    if abs(n - r) > HOLD_TOL:
        return {"status": "failed", "detail": "object_lost", "reason": f"opening changed by {abs(n-r):.3f} > {HOLD_TOL}", "ref": r, "now": n}
    return {"status": "verified", "reason": "opening unchanged; object still between the pads", "ref": r, "now": n}


def verify_release(opening_after_open) -> dict:
    a = _frac(opening_after_open)
    if a is None:
        return {"status": "uncertain", "reason": "gripper opening not observable", "after": a}
    if a >= RELEASE_MIN:
        return {"status": "verified", "reason": f"fingers open ({a:.3f})", "after": a}
    return {"status": "failed", "detail": "gripper_not_open", "reason": f"opening {a:.3f} < {RELEASE_MIN} after the release command", "after": a}
