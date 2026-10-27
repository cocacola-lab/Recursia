"""Shared placement contract + verifier (module A, round placement_geometry_20260917_122900).

Explicit contract: target relation, support geometry (resting height, rim radius, table height), object identity (prompts,
expected footprint radius/height), region for `beside_support`. Verifier statuses: exactly one of success | failed |
uncertain, with per-check evidence (support gap, XY containment/region with margin, orientation, identity, release
stability). Pure functions over public point sets; segmentation / depth access is injected (seg_fn, pts_fn) so the same
code runs in unit tests (synthetic geometry) and in a rollout (public API). The verifier never reads controller claims.

Thresholds are shared perception bounds (identical to the parent's shadow verifier V1 bands; nothing widened):
  gap_pass 0.020 / gap_uncertain 0.045   object base above the resting surface (visible-surface base bias is 0-2 cm)
  xy_margin 0.010                          footprint margin inside the support rim
  tilt_pass 15 deg / tilt_uncertain 30 deg
  settle_move 0.010 / settle_fail 0.030    base displacement between two post-release frames
"""
from __future__ import annotations

import numpy as np

import grasp_geometry as GG

THR = {"gap_pass": 0.020, "gap_uncertain": 0.045, "gap_below": -0.030, "xy_margin": 0.010, "tilt_pass_deg": 15.0, "tilt_uncertain_deg": 30.0, "min_pts": 40, "min_score": 0.10,
       "identity_gap": 0.10, "identity_radius": 0.08, "cluster_merge": 0.04, "settle_move": 0.010, "settle_fail": 0.030, "beside_min_gap": 0.02, "beside_max": 0.35, "beside_perp_max": 0.15,
       "ws": {"x": (0.1, 0.9), "y": (-0.6, 0.6), "z": (-0.05, 0.40)}}
STATUSES = ("success", "failed", "uncertain")            # post-release placement claim ("placed")
PRE_STATUSES = ("ready", "not_ready", "uncertain")       # pre-release readiness claim ("ready_to_release"); never asserts a placement
RELATIONS = ("on_support", "beside_support")
# Amendment (continuation 2026-09-17 15:00, audit item 3): a pre-release frame shows a HELD object, so its geometry can never
# establish a placement; the pre-release verdict is a readiness claim with its own vocabulary. A post-release `success`
# additionally requires (i) gripper-clearance evidence from public robot state (the end-effector is out of reach of the
# object, so an in-hand hover cannot masquerade as a placement) and (ii) the two-frame stability check. No band was widened.
THR.update({"clear_xy_extra": 0.05, "clear_z": 0.12})   # gripper model: finger length 0.054 + hang/base-bias margin; xy: footprint + finger reach


# ----------------------------------------------------------------------------- contract
def support_from_points(pts):
    """Support geometry from the support's visible points: rim circle from the farthest point per angular bin (occluded bins
    dropped), resting height max(p50, p90 - 1.5 cm), table height p2."""
    pts = np.asarray(pts, float); xy = pts[:, :2]; cen = xy.mean(0)
    ang = np.arctan2(xy[:, 1] - cen[1], xy[:, 0] - cen[0]); rad = np.hypot(*(xy - cen).T); nb = 24
    bins = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb; ext = []; reach = np.full(nb, np.nan)
    for b in range(nb):
        m = bins == b
        if m.any():
            i = np.argmax(np.where(m, rad, -1)); ext.append(xy[i]); reach[b] = rad[i]
    ext = np.array(ext) if ext else np.zeros((0, 2)); med = float(np.nanmedian(reach)) if len(ext) else 0.1
    keep = ext[np.hypot(*(ext - cen).T) >= 0.8 * med] if len(ext) else ext
    fit = GG.fit_circle(keep) if len(keep) >= 6 else None
    z = pts[:, 2]; p50, p90 = float(np.percentile(z, 50)), float(np.percentile(z, 90))
    out = {"centroid_xy": cen, "n_pts": int(len(pts)), "n_bins_kept": int(len(keep)), "rest_z": max(p50, p90 - 0.015), "table_z": float(np.percentile(z, 2)), "top_z": p90}
    if fit is not None and 0.05 <= fit[2] <= 0.16 and fit[3] < 0.012:
        out.update({"center_xy": np.array(fit[:2]), "r": float(fit[2]), "rms": float(fit[3]), "fit_ok": True})
    else:
        out.update({"center_xy": cen, "r": med, "rms": None, "fit_ok": False})
    return out


def make_contract(relation, support, object_prompts, object_r, object_h, region_dir=None, region_min=None, region_max=None, others_xy=(), label=""):
    """Validated placement contract. `support` = support_from_points(...) (or a dict with center_xy, r, rest_z, table_z)."""
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}, got {relation!r}")
    for k in ("center_xy", "r", "rest_z", "table_z"):
        if k not in support:
            raise ValueError(f"support geometry lacks {k}")
    if not (np.isfinite(float(object_r)) and 0.005 <= float(object_r) <= 0.15):
        raise ValueError(f"object_r out of range: {object_r}")
    c = {"relation": relation, "label": label, "support": {"center_xy": np.asarray(support["center_xy"], float), "r": float(support["r"]), "rest_z": float(support["rest_z"]), "table_z": float(support["table_z"])},
         "object": {"prompts": list(object_prompts), "r": float(object_r), "h": float(object_h) if object_h is not None else None}, "others_xy": [np.asarray(o, float)[:2] for o in others_xy]}
    if relation == "beside_support":
        d = np.asarray(region_dir, float)[:2]; n = float(np.linalg.norm(d))
        if n < 1e-6:
            raise ValueError("region_dir must be a non-zero 2-vector")
        c["region"] = {"dir": d / n, "min": float(region_min if region_min is not None else support["r"] + THR["beside_min_gap"]), "max": float(region_max if region_max is not None else THR["beside_max"])}
    return c


# ----------------------------------------------------------------------------- measurement
def measure_object(pts, cam_xy=None):
    """Public geometry of one object candidate: base xy (lowest band), base z, top z, footprint radius, axis tilt."""
    pts = np.asarray(pts, float)
    if len(pts) < THR["min_pts"]:
        return None
    bxy, how = GG.base_xy(pts, cam_xy); prof = GG.object_profile(pts, cam_xy)
    tilt, tinfo = GG.axis_tilt_deg(pts, cam_xy)
    if bxy is None or prof is None:
        return None
    return {"base_xy": np.asarray(bxy, float), "base_z": prof["base_p2"], "top_z": prof["top"], "r_base": float(how.get("r", prof["r_mid"])), "r_mid": prof["r_mid"], "axis_xy": prof["axis_xy"], "tilt_deg": tilt, "tilt_info": tinfo, "base_how": how["method"], "n_pts": int(len(pts))}


def _check(value, ok, unc, failed_reason, unc_reason):
    if ok:
        return {"value": value, "status": "ok"}
    if unc:
        return {"value": value, "status": "uncertain", "reason": unc_reason}
    return {"value": value, "status": "failed", "reason": failed_reason}


def _claim(phase):
    return "ready_to_release" if phase == "pre_release" else "placed"


def verify(contract, meas, phase="pre_release", prev_meas=None, identity="ok", robot=None):
    """Verdict for one measured object against the contract. `identity` in {ok, ambiguous, none}. prev_meas (post-release
    only): the measurement from the previous post-release frame, for the release-stability check. `robot` (post-release
    only): public robot state {"eef": xyz in the base frame} for the gripper-clearance check.
    pre_release  -> status in PRE_STATUSES (readiness claim; never `success`).
    post_release -> status in STATUSES; `success` requires every check ok including gripper_clear and stability."""
    if phase not in ("pre_release", "post_release"):
        raise ValueError(f"phase must be pre_release or post_release, got {phase!r}")
    sup = contract["support"]; obj = contract["object"]; checks = {}; evidential = []
    if identity == "none" or meas is None:
        return {"status": "uncertain", "phase": phase, "claim": _claim(phase), "reason": "object not found / too few points", "checks": {"identity": {"status": "uncertain", "reason": identity}}, "evidential_only": True, "measured_violation": False}
    checks["identity"] = {"status": "ok"} if identity == "ok" else {"status": "uncertain", "reason": "identity ambiguous"}
    if identity != "ok":
        evidential.append("identity")
    # support gap
    ref_z = sup["rest_z"] if contract["relation"] == "on_support" else sup["table_z"]
    gap = float(meas["base_z"] - ref_z)
    checks["support_gap"] = _check(gap, THR["gap_below"] <= gap <= THR["gap_pass"], gap <= THR["gap_uncertain"] and gap > THR["gap_pass"] or gap < THR["gap_below"],
                                   f"object base {gap*100:.1f} cm above the resting surface", f"base gap {gap*100:.1f} cm in the uncertain band" if gap > 0 else f"base {abs(gap)*100:.1f} cm below the support estimate (geometry inconsistent)")
    # containment / region
    rel = np.asarray(meas["base_xy"], float) - sup["center_xy"]
    if contract["relation"] == "on_support":
        d = float(np.hypot(*rel)); r_obj = float(obj["r"])
        checks["containment"] = _check({"xy_dist": d, "r_obj": r_obj, "r_support": sup["r"]}, d + r_obj + THR["xy_margin"] <= sup["r"], d < sup["r"],
                                       f"object centre {d*100:.1f} cm from the support centre, beyond the rim ({sup['r']*100:.1f} cm)", f"footprint reaches over the rim by {(d + r_obj + THR['xy_margin'] - sup['r'])*100:.1f} cm")
    else:
        rg = contract["region"]; proj = float(rel @ rg["dir"]); perp = float(abs(rel @ np.array([-rg["dir"][1], rg["dir"][0]])))
        ok = rg["min"] <= proj <= rg["max"] and perp <= THR["beside_perp_max"]
        failed = proj <= 0.5 * sup["r"] or proj > rg["max"] + 0.05 or perp > THR["beside_perp_max"] + 0.07
        checks["region"] = _check({"proj": proj, "perp": perp, "min": rg["min"], "max": rg["max"]}, ok, not failed, "outside the region", "near the region boundary")
    # orientation
    tilt = meas.get("tilt_deg")
    if tilt is None:
        checks["orientation"] = {"value": None, "status": "uncertain", "reason": "axis tilt not measurable"}; evidential.append("orientation")
    else:
        checks["orientation"] = _check(tilt, tilt <= THR["tilt_pass_deg"], tilt <= THR["tilt_uncertain_deg"], f"object tilted {tilt:.0f} deg", f"tilt {tilt:.0f} deg in the uncertain band")
    # release stability (two post-release frames)
    if phase == "post_release":
        if prev_meas is None:
            checks["stability"] = {"value": None, "status": "uncertain", "reason": "single frame"}; evidential.append("stability")
        else:
            mv = float(np.hypot(*(np.asarray(meas["base_xy"]) - np.asarray(prev_meas["base_xy"]))))
            checks["stability"] = _check(mv, mv <= THR["settle_move"], mv <= THR["settle_fail"], f"object moved {mv*100:.1f} cm between frames", f"object moved {mv*100:.1f} cm (settling)")
        # gripper clearance: a placement can only be asserted when the end-effector is out of reach of the object
        eef = None if not isinstance(robot, dict) else robot.get("eef")
        if eef is None:
            checks["gripper_clear"] = {"value": None, "status": "uncertain", "reason": "robot state not available: in-hand hover cannot be excluded"}; evidential.append("gripper_clear")
        else:
            eef = np.asarray(eef, float).reshape(3); dxy = float(np.hypot(*(eef[:2] - np.asarray(meas["base_xy"], float)))); dz = float(eef[2] - meas["top_z"])
            clear = dxy > float(obj["r"]) + THR["clear_xy_extra"] or dz > THR["clear_z"]
            checks["gripper_clear"] = _check({"eef_dxy": dxy, "eef_dz_above_top": dz}, clear, True, "", f"end-effector within reach of the object (dxy {dxy*100:.1f} cm, dz {dz*100:.1f} cm): held/in-hand hover cannot be excluded")
    sts = [c["status"] for c in checks.values()]
    if "failed" in sts:
        status = "failed"
    elif all(s == "ok" for s in sts):
        status = "success"
    else:
        status = "uncertain"
    if phase == "pre_release":
        status = {"success": "ready", "failed": "not_ready"}.get(status, status)
    measured_unc = [k for k, c in checks.items() if c["status"] == "uncertain" and k not in evidential]
    reason = "; ".join(f"{k}: {c.get('reason')}" for k, c in checks.items() if c["status"] != "ok") or "all checks passed"
    return {"status": status, "phase": phase, "claim": _claim(phase), "reason": reason, "checks": checks, "evidential_only": status == "uncertain" and not measured_unc, "measured_violation": status in ("failed", "not_ready") or bool(measured_unc)}


# ----------------------------------------------------------------------------- frame-level driver (segmentation injected)
def frame_candidates(frame, prompts, seg_fn, pts_fn, exclude_xy=(), max_masks=5):
    """Object candidates from the prompts: SAM3 masks -> public points -> measurement; clustered by base xy."""
    cands = []; ws = THR["ws"]; calls = 0
    for prompt in prompts:
        masks = seg_fn(frame["rgb"], prompt); calls += 1
        for m in (masks or [])[:max_masks]:
            if m["score"] < THR["min_score"]:
                continue
            pts = pts_fn(m["mask"].astype(np.uint8), frame["depth"], frame["K"], frame["E"])
            meas = measure_object(pts, frame.get("cam_xy")) if pts is not None else None
            if meas is None:
                continue
            c = meas["base_xy"]
            if not (ws["x"][0] < c[0] < ws["x"][1] and ws["y"][0] < c[1] < ws["y"][1] and ws["z"][0] < meas["base_z"] < ws["z"][1]):
                continue
            if any(np.hypot(*(c - np.asarray(e, float)[:2])) < 0.05 for e in exclude_xy):
                continue
            meas.update({"score": float(m["score"]), "prompt": prompt}); cands.append(meas)
    clusters = []
    for g in sorted(cands, key=lambda d: -d["score"]):
        for cl in clusters:
            if np.hypot(*(g["base_xy"] - cl["base_xy"])) < THR["cluster_merge"]:
                cl["members"] += 1; break
        else:
            clusters.append({**g, "members": 1})
    return clusters, calls


def select_identity(clusters, expected_xy, radius=None):
    """Nearest candidate to the expected position; ambiguous when a second candidate of similar score is also within radius."""
    radius = THR["identity_radius"] if radius is None else radius
    if not clusters:
        return None, "none"
    exp = np.asarray(expected_xy, float)[:2]
    near = sorted([c for c in clusters if np.hypot(*(c["base_xy"] - exp)) <= radius], key=lambda c: np.hypot(*(c["base_xy"] - exp)))
    if not near:
        return None, "none"
    if len(near) > 1 and near[1]["score"] >= near[0]["score"] - THR["identity_gap"] and np.hypot(*(near[1]["base_xy"] - near[0]["base_xy"])) > THR["cluster_merge"]:
        return near[0], "ambiguous"
    return near[0], "ok"


def verify_frame(contract, frame, seg_fn, pts_fn, expected_xy, phase="pre_release", prev_meas=None, exclude_xy=()):
    """Full verdict from one public frame: candidates -> identity -> verify. expected_xy = where the object should be
    (pre-release: TCP xy + hang offset; post-release: the placement target)."""
    clusters, calls = frame_candidates(frame, contract["object"]["prompts"], seg_fn, pts_fn, exclude_xy=list(contract.get("others_xy", [])) + list(exclude_xy))
    meas, ident = select_identity(clusters, expected_xy)
    v = verify(contract, meas, phase=phase, prev_meas=prev_meas, identity=ident, robot=frame.get("robot"))
    v["evidence"] = {"n_candidates": len(clusters), "identity": ident, "sam3_calls": calls, "expected_xy": np.asarray(expected_xy, float)[:2].tolist(),
                     "meas": None if meas is None else {k: (v_.tolist() if hasattr(v_, "tolist") else v_) for k, v_ in meas.items() if k not in ("tilt_info",)}}
    v["meas"] = meas
    return v
