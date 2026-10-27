"""General grasp/placement geometry calibration (module B, round placement_geometry_20260917_122900).

Inputs are PUBLIC point sets (SAM3 mask + depth -> base-frame points) and the gripper model. No object, task, scene or seed
information enters any decision; every constant is a gripper quantity or a documented shared perception bound.

Gripper model (Franka Hand, URDF):
  max_opening 0.080   distance between the pad inner faces when fully open
  finger_len  0.054   finger length below the hand body (the palm meets an object top when the fingertips are this far below it)
  pad_engage  0.018   height of the pad friction surface above the fingertip
  finger_thick 0.010  finger body thickness along the closing axis (outside the pad face)
  finger_width 0.020  finger width across the closing axis
  tcp_to_fingertip 0.005  the public TCP (IK point) is 5 mm above the fingertips
  finger_clear 0.008  minimum free space between a pad face and the object at the grasp height (per side)
Perception bounds (shared, measured offline on the parent rounds' public frames vs runner truth; not per object):
  base_bias (0.0, 0.02)  the 2nd-percentile z of the visible points lies 0-2 cm ABOVE the true base
  xy_sigma 0.010         1-sigma XY error of a centre estimate from visible-surface points
  z_noise 0.004          depth noise on a top/base percentile
"""
from __future__ import annotations

import math

import numpy as np

GRIPPER = {"max_opening": 0.080, "finger_len": 0.054, "pad_engage": 0.018, "finger_thick": 0.010, "finger_width": 0.020, "tcp_to_fingertip": 0.005, "finger_clear": 0.008}
PERCEPTION = {"base_bias": (0.0, 0.02), "xy_sigma": 0.010, "z_noise": 0.004, "min_pts": 40, "centroid_shift": 0.012}
CLOSING_DIRS = {0: np.array([0.0, 1.0]), 90: np.array([1.0, 0.0])}   # yaw of the caller's top-down quaternion -> pad closing axis (base frame)
MODES = ("body_pinch", "rim_pinch", "infeasible")
EXPECTED = ("reached", "contact_established", "premature_contact")


# ----------------------------------------------------------------------------- primitives
def fit_circle(xy):
    """Algebraic (Kasa) circle fit; returns (cx, cy, r, rms) or None."""
    xy = np.asarray(xy, float)
    if len(xy) < 8:
        return None
    A = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]; b = (xy ** 2).sum(1)
    try:
        s, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:  # noqa: BLE001
        return None
    r2 = s[2] + s[0] ** 2 + s[1] ** 2
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2)); res = np.hypot(xy[:, 0] - s[0], xy[:, 1] - s[1]) - r
    return float(s[0]), float(s[1]), r, float(np.sqrt((res ** 2).mean()))


def slice_geometry(pts, z_lo, z_hi, cam_xy=None, r_range=(0.012, 0.09), view="side", allow_circle=True):
    """Horizontal cross-section estimate of a z-band: circle fit (one outlier-rejection pass) when the band is round enough;
    otherwise, for a band seen from above (view="top": the whole top face is visible) the midpoint of the 3-97 percentile
    extents, and for a partial side view the centroid shifted away from the camera by the visible-surface prior."""
    pts = np.asarray(pts, float); band = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
    if len(band) < 12:
        return None
    xy = band[:, :2]; cen = xy.mean(0)
    ext = np.percentile(xy, 97, axis=0) - np.percentile(xy, 3, axis=0)
    fit = fit_circle(xy) if allow_circle else None
    if fit is not None:
        cx, cy, r, rms = fit; res = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r); inl = xy[res < 0.01]
        if len(inl) >= 10:
            f2 = fit_circle(inl)
            if f2 is not None:
                cx, cy, r, rms = f2
        ring_frac = float(np.mean(np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) < 0.006))   # a ring/arc has its points AT the radius; a filled face does not
        if r_range[0] <= r <= r_range[1] and rms < 0.006 and ring_frac >= 0.6 and np.hypot(cx - cen[0], cy - cen[1]) < 0.06:
            return {"center": np.array([cx, cy]), "r": float(r), "rms": float(rms), "method": "circle", "n": int(len(band)), "extent": ext, "ring_frac": ring_frac}
    if view == "top":
        mid = 0.5 * (np.percentile(xy, 97, axis=0) + np.percentile(xy, 3, axis=0))
        return {"center": mid, "r": float(max(ext) / 2.0), "rms": None, "method": "extent_mid", "n": int(len(band)), "extent": ext}
    shift = np.zeros(2)
    if cam_xy is not None:
        d = cen - np.asarray(cam_xy, float)[:2]; n = float(np.linalg.norm(d))
        if n > 1e-6:
            shift = d / n * PERCEPTION["centroid_shift"]
    return {"center": cen + shift, "r": float(max(ext) / 2.0), "rms": None, "method": "centroid_shift", "n": int(len(band)), "extent": ext}


def object_profile(pts, cam_xy=None):
    """Base/top/height, per-band cross-sections, hollowness and protrusions of one object's visible points."""
    pts = np.asarray(pts, float)
    if len(pts) < PERCEPTION["min_pts"]:
        return None
    z = pts[:, 2]; base_p2 = float(np.percentile(z, 2)); top = float(np.percentile(z, 98)); H = max(top - base_p2, 1e-3)
    topb = slice_geometry(pts, top - 0.02, top + 0.01, cam_xy, view="top")
    round_ = topb is not None and topb["method"] == "circle"    # roundness is decided on the fully visible top band; two side faces of a box can fake a circle
    bottom = slice_geometry(pts, base_p2, base_p2 + max(0.015, 0.25 * H), cam_xy, allow_circle=round_)
    mid = slice_geometry(pts, base_p2 + 0.25 * H, base_p2 + 0.75 * H, cam_xy, allow_circle=round_)
    ref = topb or mid or bottom          # the top band sees the whole section from above; side bands are partial views
    axis_xy = ref["center"]; r_mid = (mid or ref)["r"]; r_top = (topb or ref)["r"]
    # hollow: a solid body shows the camera only its near half below the top; an open-top body also shows its FAR inner wall
    # (and possibly its inner bottom). Test: fraction of below-top points on the far half (angle from the camera direction
    # > 100 deg) or clearly inside the top ring.
    tc = (topb or ref)["center"]; rr = np.hypot(pts[:, 0] - tc[0], pts[:, 1] - tc[1]); below = (z < top - 0.015) & (z > base_p2 + 0.3 * H)
    interior = int(np.sum((rr < 0.7 * r_top) & (z < top - 0.015) & (z > base_p2 + 0.005)))
    far_frac = 0.0; n_far = 0
    if cam_xy is not None and below.sum() >= 30:
        d_cam = np.asarray(cam_xy, float)[:2] - tc; a_cam = math.atan2(d_cam[1], d_cam[0]); a_pts = np.arctan2(pts[below, 1] - tc[1], pts[below, 0] - tc[0])
        far = np.abs(np.angle(np.exp(1j * (a_pts - a_cam)))) > math.radians(100); n_far = int(far.sum()); far_frac = float(far.mean())
    hollow = bool(r_top >= 0.02 and (topb or ref)["method"] == "circle" and (interior >= 20 or (n_far >= 30 and far_frac >= 0.15)))
    # protrusions (e.g. a handle): mid-band points beyond the fitted radius, binned by direction
    band = pts[(z > base_p2 + 0.2 * H) & (z < top - 0.005)]; out = {}
    d = band[:, :2] - axis_xy; rad = np.hypot(d[:, 0], d[:, 1]); ang = np.arctan2(d[:, 1], d[:, 0])
    for name, a0 in (("+x", 0.0), ("+y", math.pi / 2), ("-x", math.pi), ("-y", -math.pi / 2)):
        sel = np.abs(np.angle(np.exp(1j * (ang - a0)))) < math.pi / 6
        out[name] = float(np.mean(rad[sel] > r_mid + 0.012)) if sel.sum() >= 5 else 0.0
    return {"n": int(len(pts)), "base_p2": base_p2, "base_est": base_p2 - 0.5 * sum(PERCEPTION["base_bias"]), "base_lo": base_p2 - PERCEPTION["base_bias"][1], "base_hi": base_p2 - PERCEPTION["base_bias"][0],
            "top": top, "height": H, "axis_xy": axis_xy, "r_mid": float(r_mid), "r_top": float(r_top), "bottom": bottom, "mid": mid, "top_band": topb, "hollow": hollow, "hollow_evidence": {"interior_pts": interior, "far_pts": n_far, "far_frac": round(far_frac, 3)}, "protrusion": out}


def width_across(profile, pts, d):
    """Object width across the closing axis d at the grasp band (mid band): 2r for a round section, else the projected extent."""
    mid = profile["mid"]; topb = profile["top_band"]
    if topb is not None and topb["method"] == "circle":
        return 2.0 * max(profile["r_top"], mid["r"] if (mid is not None and mid["method"] == "circle") else 0.0)
    pts = np.asarray(pts, float); z = pts[:, 2]; b = profile["base_p2"]; H = profile["height"]; d = np.asarray(d, float)
    widths = []
    for band in (pts[(z > b + 0.25 * H) & (z < b + 0.75 * H)], pts[z >= profile["top"] - 0.02]):
        if len(band) >= 12:
            proj = band[:, :2] @ d; widths.append(float(np.percentile(proj, 97) - np.percentile(proj, 3)))
    return max(widths) if widths else float(np.ptp(pts[:, :2] @ d))


def points_in_footprints(pts_xy, tcp_xy, d, opening, g=GRIPPER):
    """Boolean mask of points lying under either finger body when the pads are `opening` apart, closing along d."""
    d = np.asarray(d, float) / np.linalg.norm(d); n = np.array([-d[1], d[0]]); rel = np.asarray(pts_xy, float) - np.asarray(tcp_xy, float)
    along = rel @ d; across = rel @ n
    inner = opening / 2.0; outer = inner + g["finger_thick"]
    return (np.abs(across) <= g["finger_width"] / 2.0) & (np.abs(along) >= inner - 0.002) & (np.abs(along) <= outer)


def predict_contact(pts, tcp_xy, d, z_target, opening=None, g=GRIPPER, p=PERCEPTION):
    """First-contact prediction for a straight descent of the open fingers to TCP height z_target: the highest object point
    under a finger footprint that lies above the fingertip target. Evaluated at the nominal XY and at 8 perturbations of
    xy_sigma (uncertainty). Returns predicted TCP height at contact (None = no contact before arrival) and its range."""
    opening = g["max_opening"] if opening is None else opening
    pts = np.asarray(pts, float); z_ft = z_target - g["tcp_to_fingertip"] + 0.002
    shifts = [np.zeros(2)] + [p["xy_sigma"] * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 9)[:-1]]
    zs = []
    for s in shifts:
        m = points_in_footprints(pts[:, :2], np.asarray(tcp_xy, float) + s, d, opening, g) & (pts[:, 2] > z_ft)
        zs.append(float(pts[m, 2].max()) + g["tcp_to_fingertip"] if m.any() else None)
    hit = [v for v in zs if v is not None]
    return {"predicted_contact_tcp_z": zs[0], "range": [min(hit), max(hit)] if hit else None, "p_contact": len(hit) / len(zs), "n_perturbations": len(zs), "sigma_z": p["z_noise"]}


def expected_status(pred, z_target, contact_tol_before=0.03):
    """MC status implied by the prediction for a descent whose target is z_target with the caller's unchanged tolerance."""
    zc = pred["predicted_contact_tcp_z"]
    if zc is None and pred["p_contact"] <= 0.5:
        return "reached"
    zc = zc if zc is not None else pred["range"][1]
    return "contact_established" if zc <= z_target + contact_tol_before else "premature_contact"


# ----------------------------------------------------------------------------- grasp planning
def plan_grasp(pts, cam_xy=None, yaws=(0, 90), g=GRIPPER, p=PERCEPTION, contact_tol_before=0.03):
    """Grasp target from geometry only. Returns a dict with mode, target (TCP xyz), yaw_order (caller quaternions, best
    first), predicted contact, hang estimate with interval, and the per-candidate table. `infeasible` when no candidate
    straddles free material without a predicted premature contact."""
    prof = object_profile(pts, cam_xy)
    if prof is None:
        return {"mode": "infeasible", "reason": "too few points", "candidates": [], "profile": None}
    pts = np.asarray(pts, float); cands = []
    for yaw in yaws:
        d = CLOSING_DIRS[yaw]
        w = width_across(prof, pts, d)
        # --- body pinch: pads straddle the whole section
        if w <= g["max_opening"] - 2 * g["finger_clear"]:
            centre = prof["base_est"] + 0.5 * prof["height"]
            z_g = max(prof["top"] - (g["finger_len"] - 0.005), prof["base_hi"] + 0.010 + g["tcp_to_fingertip"], min(prof["top"] - g["pad_engage"] - 0.005, centre))
            tcp_xy = prof["axis_xy"]
            pred = predict_contact(pts, tcp_xy, d, z_g, g=g, p=p); st = expected_status(pred, z_g, contact_tol_before)
            cands.append({"mode": "body_pinch", "yaw": yaw, "closing_dir": d.tolist(), "width_across": w, "tcp_xy": np.asarray(tcp_xy, float), "z": float(z_g), "pred": pred, "expected": st,
                          "score": (0 if st == "reached" else 2) + 0.1 * (yaw != 0) + 0.5 * (pred["p_contact"])})
        # --- rim pinch: hollow object, one pad inside the cavity and one outside the wall
        if prof["hollow"] and prof["r_top"] >= 0.02:
            r_wall = prof["r_top"] - 0.003
            for sgn, name in ((1.0, None), (-1.0, None)):
                n_hat = sgn * d; key = ("+" if sgn > 0 else "-") + ("y" if yaw == 0 else "x")
                tcp_xy = prof["axis_xy"] + n_hat * r_wall
                z_g = prof["top"] - g["pad_engage"] - 0.002
                if z_g - g["tcp_to_fingertip"] < prof["base_hi"] + 0.010:     # cavity too shallow for the inner finger
                    continue
                pred = predict_contact(pts, tcp_xy, d, z_g, g=g, p=p); st = expected_status(pred, z_g, contact_tol_before)
                prot = prof["protrusion"].get(key, 0.0)
                cands.append({"mode": "rim_pinch", "yaw": yaw, "closing_dir": d.tolist(), "side": key, "width_across": w, "wall_r": r_wall, "tcp_xy": np.asarray(tcp_xy, float), "z": float(z_g), "pred": pred, "expected": st,
                              "protrusion": prot, "score": (0 if st == "reached" else 2) + 1.0 * prot + 0.1 * (yaw != 0) + 0.5 * pred["p_contact"]})
    feas = [c for c in cands if c["expected"] != "premature_contact"]
    if not feas:
        return {"mode": "infeasible", "reason": "every candidate predicts contact before the grasp height (object wider than the finger span and not hollow, or cavity too shallow)", "candidates": cands, "profile": prof}
    feas.sort(key=lambda c: c["score"]); best = feas[0]
    yaw_order = list(dict.fromkeys([best["yaw"]] + [c["yaw"] for c in feas[1:]]))
    hang = best["z"] - prof["base_est"]
    return {"mode": best["mode"], "target": np.array([best["tcp_xy"][0], best["tcp_xy"][1], best["z"]]), "yaw_order": yaw_order, "side": best.get("side"), "expected": best["expected"],
            "predicted_contact_tcp_z": best["pred"]["predicted_contact_tcp_z"], "contact_range": best["pred"]["range"], "p_contact": best["pred"]["p_contact"],
            "hang_est": float(hang), "hang_interval": [float(best["z"] - prof["base_hi"]), float(best["z"] - prof["base_lo"])], "xy_sigma": p["xy_sigma"], "z_sigma": p["z_noise"],
            "profile_summary": {"base_p2": prof["base_p2"], "top": prof["top"], "height": prof["height"], "r_mid": prof["r_mid"], "r_top": prof["r_top"], "hollow": prof["hollow"], "axis_xy": prof["axis_xy"].tolist(), "protrusion": prof["protrusion"]},
            "candidates": cands, "profile": prof}


# ----------------------------------------------------------------------------- held-object geometry (placement side)
def base_xy(pts, cam_xy=None):
    """XY of the object's BASE — what lands on the support. Primary estimator: the fully visible top section's centre
    projected down the object axis (top-plane normal) by the estimated height, so a tilted hanging object's base is
    displaced from its mid-height axis by H*tan(tilt). Fallback: the lowest band's section (handles/flare can bias it)."""
    pts = np.asarray(pts, float); prof = object_profile(pts, cam_xy)
    if prof is None:
        return None, {"method": "none"}
    tilt, info = axis_tilt_deg(pts, cam_xy); topb = prof["top_band"]
    if topb is not None and tilt is not None and info.get("normal") is not None:
        n = np.asarray(info["normal"], float); H = prof["top"] - prof["base_est"]
        if n[2] > 0.5:
            bxy = np.asarray(topb["center"], float) - H * n[:2] / n[2]
            return bxy, {"method": "top_projected", "tilt_deg": round(tilt, 1), "r": float(prof["r_top"]), "n": int(topb["n"]), "H_used": round(float(H), 4)}
    b = prof["base_p2"]; s = slice_geometry(pts, b, b + 0.015, cam_xy, allow_circle=False) or slice_geometry(pts, b, b + 0.03, cam_xy, allow_circle=False)
    if s is None:
        return prof["axis_xy"], {"method": "axis_fallback", "r": float(prof["r_mid"])}
    return s["center"], {"method": "bottom_band_" + s["method"], "r": s["r"], "n": s["n"]}


def axis_tilt_deg(pts, cam_xy=None, band=0.015, iters=4):
    """Tilt of the object axis from vertical = tilt of the normal of its top plane (rim ring or top face), which is fully
    visible from above and planar for the objects a parallel gripper can pick. The top band is re-selected along the fitted
    normal (iterated), so a strongly tilted rim is recovered. None when the band is too small or not planar."""
    pts = np.asarray(pts, float)
    if len(pts) < PERCEPTION["min_pts"]:
        return None, {}
    n = np.array([0.0, 0.0, 1.0]); info = {}
    for _ in range(iters):
        h = pts @ n; top = np.percentile(h, 98); sel = pts[h >= top - band]
        if len(sel) < 30:
            return None, {"n_top": int(len(sel))}
        c = sel.mean(0); u, s, vt = np.linalg.svd(sel - c, full_matrices=False); n_new = vt[-1]
        if n_new[2] < 0:
            n_new = -n_new
        rms = float(np.sqrt(np.mean(((sel - c) @ n_new) ** 2)))
        info = {"n_top": int(len(sel)), "plane_rms": rms, "normal": n_new.round(4).tolist()}
        if np.dot(n_new, n) > 0.99999:
            n = n_new; break
        n = n_new
    if info["plane_rms"] > 0.006:
        return None, {**info, "reason": "top band not planar"}
    return float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))), info
