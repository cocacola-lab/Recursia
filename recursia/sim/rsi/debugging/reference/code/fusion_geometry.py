"""Module F — general fusion of grasp geometry and placement target generation (round zeroshot_general_fusion_20260917_CRQQPG).

Consumes module B (grasp_geometry, frozen) and PUBLIC quantities only: point sets from SAM3 + depth, the gripper model, the
caller's public IK (passed in as a probe callable), verifier verdicts (module A, frozen). No object, task, scene or seed word
enters any decision; every constant is a gripper/controller quantity or a documented shared perception bound.

Mechanisms (protocol §1 MF):
  plan_grasp              module B candidates; body-pinch height = max(palm clearance, fingertips at the estimated true base)
  rest_reference          resting height of the destination plane (on_support: support top; beside: binding base level if
                          consistent with the support's table estimate, else that estimate; never a raw minimum)
  select_descent_target   touchdown target (undershoot inside the unchanged MC window) probed for reachability BEFORE descending
                          at the sweep bottom and release height; fallback = touchdown without undershoot; else infeasible
  validate_beside_target  footprint half-extent along the support direction must clear the rim (bounded outward shift)
  correction              bounded corrective action after a readiness verdict (re-descend after a no-contact stall, recenter
                          on a measured XY offset, descend by a measured gap); a stall is never accepted as contact
"""
from __future__ import annotations

import numpy as np

import grasp_geometry as GG
from grasp_geometry import GRIPPER, PERCEPTION

PLACE_TOUCHDOWN_UNDERSHOOT = 0.015   # m, identical to MG (half the visible-surface base-bias interval)
PLACE_RETRIES_TOTAL = 2              # bounded corrective actions per placement
RECENTER_MAX = 0.04                  # m, largest measured XY offset a recentering may act on (beyond it the measurement is not trusted)
RECENTER_MIN = 0.005                 # m, below this an XY offset is within the perception sigma: no action
REGION_CLEAR = 0.005                 # m, required free space between a footprint edge and a support rim (beside placement)
REGION_SHIFT_MAX = 0.03              # m, bounded outward correction of a caller's nominal beside target
REST_CONSISTENCY = 0.03              # m, a binding base level counts as resting on the support's table only within this band
PALM_CLEAR = 0.005                   # m, palm clearance above an object top


# ----------------------------------------------------------------------------- grasp
def plan_grasp(pts, cam_xy=None, yaws=(0, 90)):
    """Module B plan with the body-pinch height rule replaced: the fingertips are aimed at the estimated true base
    (base_p2 - half the bias band) so the unchanged contact-seeking descent meets the support inside its window and the pads
    engage the whole body height; the palm clearance bounds it from above for tall objects."""
    plan = GG.plan_grasp(pts, cam_xy, yaws)
    plan["rule"] = "module_B"
    if plan.get("mode") != "body_pinch" or plan.get("profile") is None:
        return plan
    prof = plan["profile"]
    z_palm = float(prof["top"] - (GRIPPER["finger_len"] - PALM_CLEAR))
    z_base = float(prof["base_est"] + GRIPPER["tcp_to_fingertip"])
    z_new = max(z_palm, z_base)
    z_ft = z_new - GRIPPER["tcp_to_fingertip"]
    engaged = float(min(prof["top"], z_ft + GRIPPER["pad_engage"]) - max(prof["base_est"], z_ft))   # pad/body overlap height
    shortfall_to_perceived_base = float(prof["base_p2"] - z_ft)                                       # >0: fingertips aimed below the perceived base
    tgt = np.asarray(plan["target"], float).copy(); tgt[2] = z_new
    plan.update({"target": tgt, "rule": "F_body_pinch_base", "z_module_B": float(plan["target"][2]) if plan.get("target") is not None else None,
                 "hang_est": float(z_new - prof["base_est"]), "hang_interval": [float(z_new - prof["base_hi"]), float(z_new - prof["base_lo"])],
                 "pad_engaged_m": engaged, "expected": "contact_established" if shortfall_to_perceived_base > PERCEPTION["z_noise"] else "reached",
                 "predicted_contact_tcp_z": float(prof["base_p2"] + GRIPPER["tcp_to_fingertip"]) if shortfall_to_perceived_base > PERCEPTION["z_noise"] else None,
                 "contact_range": [float(prof["base_lo"] + GRIPPER["tcp_to_fingertip"]), float(prof["base_hi"] + GRIPPER["tcp_to_fingertip"])]})
    return plan


# ----------------------------------------------------------------------------- placement reference and target
def rest_reference(relation, support, binding_base_z=None):
    """Resting height of the destination plane for the object base. Returns (z, source)."""
    if relation == "on_support":
        return float(support["rest_z"]), "support_rest_z"
    tz = float(support["table_z"])
    if binding_base_z is not None and np.isfinite(binding_base_z) and abs(float(binding_base_z) - tz) <= REST_CONSISTENCY:
        return float(binding_base_z), "binding_base_level"       # same estimator as the held-object base -> bias-consistent gap
    return tz, "support_table_z"


def touchdown_targets(ref_z, drop_used, undershoot=PLACE_TOUCHDOWN_UNDERSHOOT):
    """Ordered TCP release heights: touchdown with undershoot (preferred), then touchdown without undershoot."""
    return [float(ref_z + drop_used - undershoot), float(ref_z + drop_used)]


def select_descent_target(ik_probe, xy, quat, ref_z, drop_used, overshoot, undershoot=PLACE_TOUCHDOWN_UNDERSHOOT):
    """Reachability of the whole contact-seeking sweep is checked BEFORE the descent: the public IK must solve at the sweep
    bottom (z_release - overshoot) and at z_release. Returns dict(z_release|None, feasible, undershoot_used, probes)."""
    xy = np.asarray(xy, float).reshape(2); probes = []
    for z in touchdown_targets(ref_z, drop_used, undershoot):
        ok_all = True
        for zp in (z - float(overshoot), z):
            ok = bool(ik_probe([float(xy[0]), float(xy[1]), float(zp)], quat)); probes.append({"z": round(float(zp), 4), "ok": ok})
            if not ok:
                ok_all = False; break
        if ok_all:
            return {"z_release": float(z), "feasible": True, "undershoot_used": float(ref_z + drop_used - z), "probes": probes, "expected_touchdown_tcp_z": float(ref_z + drop_used)}
    return {"z_release": None, "feasible": False, "undershoot_used": None, "probes": probes, "expected_touchdown_tcp_z": float(ref_z + drop_used), "reason": "descent sweep unreachable by the public IK at every touchdown target"}


def half_extent_along(pts, direction):
    """Half of the 3-97 percentile extent of a point set's XY projected on a unit direction."""
    pts = np.asarray(pts, float); d = np.asarray(direction, float)[:2]; n = float(np.linalg.norm(d))
    if len(pts) < 8 or n < 1e-9:
        return None
    proj = pts[:, :2] @ (d / n)
    return float(0.5 * (np.percentile(proj, 97) - np.percentile(proj, 3)))


def validate_beside_target(nominal_xy, support_center_xy, support_r, half_extent, sigma=PERCEPTION["xy_sigma"], clear=REGION_CLEAR, max_shift=REGION_SHIFT_MAX):
    """Feasible-region validation of a caller's nominal beside-support target: the footprint edge facing the support must clear
    the rim by clear + sigma. If it overlaps, the target is pushed outward along the support->target direction by the overlap
    (bounded). Returns (xy, info)."""
    nominal = np.asarray(nominal_xy, float)[:2]; c = np.asarray(support_center_xy, float)[:2]; rel = nominal - c; dist = float(np.linalg.norm(rel))
    if dist < 1e-6 or half_extent is None:
        return nominal, {"shift_m": 0.0, "reason": "no direction or extent", "valid": half_extent is not None}
    need = float(support_r) + float(half_extent) + clear + sigma; overlap = need - dist
    if overlap <= 0:
        return nominal, {"shift_m": 0.0, "clearance_m": -overlap, "valid": True}
    shift = min(overlap, max_shift)
    return nominal + rel / dist * shift, {"shift_m": float(shift), "overlap_m": float(overlap), "valid": shift >= overlap - 1e-9, "bounded": shift < overlap}


# ----------------------------------------------------------------------------- bounded corrections
def correction(verdict, descent_result, retries_left, offset_xy=None):
    """Bounded corrective action after a pre-release readiness verdict, or None. `offset_xy` = measured object base XY minus
    the desired base XY (caller-computed from the verifier's measurement and the contract). Kinds:
      redescend             the descent stalled without confirmed contact (a stall is never accepted as contact)
      recenter              containment/region not ok with a measured offset in (RECENTER_MIN, RECENTER_MAX]
      descend_for_touchdown descent reached its height without physical touchdown evidence
      descend_by_gap        measured support gap above the pass band"""
    if retries_left <= 0 or not isinstance(verdict, dict):
        return None
    rd = descent_result or {}; st = rd.get("status"); ch = verdict.get("checks", {}) or {}
    if verdict.get("status") == "ready" and st == "contact_established":
        return None
    if st == "stalled":
        return {"kind": "redescend", "why": "stalled without contact confirmation"}
    cont = ch.get("containment") or ch.get("region")
    if cont is not None and cont.get("status") != "ok" and offset_xy is not None:
        off = np.asarray(offset_xy, float)[:2]; n = float(np.linalg.norm(off))
        if RECENTER_MIN < n <= RECENTER_MAX:
            return {"kind": "recenter", "shift_xy": (-off).tolist(), "offset_cm": round(n * 100, 2), "why": cont.get("reason")}
        if n > RECENTER_MAX:
            return None   # measurement not trusted for an action; verdict stands
    gap = (ch.get("support_gap") or {}).get("value")
    if rd.get("success") and st != "contact_established":
        return {"kind": "descend_for_touchdown", "step": float(max(gap if (gap is not None and gap > 0) else 0.0, PLACE_TOUCHDOWN_UNDERSHOOT))}
    if rd.get("success") and gap is not None and (ch.get("support_gap") or {}).get("status") != "ok" and gap > 0:
        return {"kind": "descend_by_gap", "step": float(gap)}
    return None
