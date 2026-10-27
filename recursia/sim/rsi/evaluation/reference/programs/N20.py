import numpy as np
import json
from scipy.spatial.transform import Rotation

# ===== shared descend-place-retreat primitive (round 20260914_558080) =====
# Public-API only: get_observation, solve_ik, move_to_joints, open_gripper, goto_home_joint_position,
# segment_sam3_text_prompt, mask_to_world_points, get_oriented_bounding_box_from_3d_points.
# No simulator state, no scene-specific coordinates, no ground truth. `diag_mark` (if present in the
# namespace) is a one-way annotation hook that returns None; it is absent in plain replay.
import json as _json
import time as _time
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

PRIM = {
    "version": "P1",              # P1 | P2 | P3 (set by the program builder)
    "tcp_offset_along_z": 0.107,  # robot_cartesian_pos -> IK/TCP point along the public quaternion z axis (probe_frames.json)
    "retreat_seg": 0.04, "retreat_safe_z": 0.25, "retreat_max_segments": 8,
    "descend_coarse": 0.03, "descend_fine": 0.012, "descend_fine_zone": 0.03,
    "descend_max_segments": 14, "descend_stall_frac": 0.5, "descend_time_cap_s": 60.0,
    "contact_z_short": 0.012,     # a segment ending >1.2 cm above its command (free-space sag is ~0.5 cm) => contact inferred, stop
    "reach_tol": 0.010,
    "release_clearance": 0.020,   # calibrated on dev states (recorder, offline): SAM3 base-rim bias ~1 cm + in-gripper slip during descent ~1 cm; a <=2 cm drop is harmless
    "sag_default": 0.005, "sag_max": 0.010,  # steady-state controller sag (segment ends above its command); estimated online from reached segments
    "drop_bounds": (0.03, 0.16),  # sanity range for the measured hang depth (TCP to object base)
    "held_offset_max": 0.08, "held_agree_tol": 0.015, "held_min_pts": 40,
    "centroid_shift_fallback": 0.012,  # SAM3 visible-surface centroid bias toward camera (offline-measured prior, cm-level)
    "shadow_verifier": True,
}
_PRIM_STATS = {"motion_calls": 0, "sam3_calls": 0, "segments": 0, "stalls": 0, "unreached": 0}


def _mark(**kw):
    kw.setdefault("t", round(_time.time(), 3))
    s = _json.dumps(kw, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("PRIM " + s, flush=True)
    f = globals().get("diag_mark")
    if callable(f):
        f(s)


def _R(q_wxyz):
    w, x, y, z = [float(v) for v in q_wxyz]
    return _Rot.from_quat([x, y, z, w]).as_matrix()


def tcp_from_obs(obs):
    """Public TCP (the IK target point) from robot_cartesian_pos: p + R(q) @ [0,0,+0.107]."""
    c = np.asarray(obs["robot_cartesian_pos"], dtype=float)
    q = c[3:7].copy()
    p = c[:3] + _R(q) @ np.array([0.0, 0.0, PRIM["tcp_offset_along_z"]])
    return p, q


def read_tcp():
    obs = get_observation()
    p, q = tcp_from_obs(obs)
    return p, q, obs


def _ik(pos, quat):
    try:
        return solve_ik(np.asarray(pos, float).tolist(), np.asarray(quat, float).tolist())
    except Exception as e:  # noqa: BLE001
        print(f"PRIM ik_failed pos={np.round(pos, 4).tolist()} err={e!r}", flush=True)
        return None


def move_tcp_segment(target_p, q, label):
    """One IK segment to target_p with orientation q; verify arrival from the public TCP. Returns a dict."""
    target_p = np.asarray(target_p, float)
    _mark(ev="segment_start", label=label, cmd=target_p)
    t0 = _time.time()
    # MR: straight-line verified segment; the interface reports reached/stalled/ik_failed/... from the public joints
    mres = move_to_pose(target_p, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=label)
    _PRIM_STATS["motion_calls"] += 1; _PRIM_STATS["segments"] += 1
    if mres["status"] in _PRE_MOTION:
        _mark(ev="segment_ik_failed", label=label, cmd=target_p, status=mres["status"])
        return {"label": label, "cmd": target_p, "ik_failed": True, "reached": False, "status": mres["status"]}
    p, q_act, _ = read_tcp()
    err = p - target_p
    r = {"label": label, "cmd": target_p, "act": p, "err_xy_cm": float(np.hypot(err[0], err[1]) * 100), "err_z_cm": float(err[2] * 100),
         "reached": bool(mres["reached"]), "wall_s": round(_time.time() - t0, 2), "ik_failed": False, "status": mres["status"], "steps": mres["steps_used"]}
    if not r["reached"]:
        _PRIM_STATS["unreached"] += 1
    _mark(ev="segment_end", label=label, cmd=target_p, act=p, err_xy_cm=round(r["err_xy_cm"], 2), err_z_cm=round(r["err_z_cm"], 2), reached=r["reached"])
    return r


# ----------------------------------------------------------------------------- retreat (P1)
def safe_retreat(sg="sg", q_hold=None):
    """Vertical segmented ascent from the ACTUAL public TCP pose (orientation held), then goto_home."""
    p0, q0, _ = read_tcp()
    q = np.asarray(q_hold, float) if q_hold is not None else q0
    _mark(ev="retreat_start", sg=sg, tcp=p0, q=q)
    z_safe = max(PRIM["retreat_safe_z"], p0[2] + 0.10)
    z = p0[2]; segs = []; n = 0
    stopped = None
    while z < z_safe - 1e-3 and n < PRIM["retreat_max_segments"]:
        z = min(z + PRIM["retreat_seg"], z_safe)
        segs.append(move_tcp_segment([p0[0], p0[1], z], q, f"{sg}:retreat{n}"))
        n += 1
        if not segs[-1].get("reached"):   # G1 I1: the next segment depends on this arrival -> the dependent chain ends here
            stopped = {"segment": n - 1, "status": segs[-1].get("status")}
            print("RETREAT_SEGMENT_NONARRIVAL " + _json.dumps({"sg": sg, **stopped}), flush=True)
            break
    p1, _, _ = read_tcp()
    drift = float(np.hypot(*(p1[:2] - p0[:2])) * 100)
    _mark(ev="retreat_ascent_done", sg=sg, tcp=p1, xy_drift_cm=round(drift, 2), n_segments=n, stopped=stopped)
    hres = goto_home_joint_position()   # G1 I1: independent recovery re-planned from the actual joints; its actual termination is reported, never assumed
    _PRIM_STATS["motion_calls"] += 1
    hstat = hres.get("status") if isinstance(hres, dict) else "unreported"
    _mark(ev="retreat_home_done", sg=sg, home_status=hstat)
    return {"segments": segs, "xy_drift_cm": drift, "n_segments": n, "stopped": stopped, "home": hstat}


# ----------------------------------------------------------------------------- geometry helpers (public)
def support_height(pts):
    """Resting-surface estimate from the support's visible points: median z, clamped to >= p90 - 1.5 cm."""
    z = np.asarray(pts)[:, 2]
    p50, p90 = float(np.percentile(z, 50)), float(np.percentile(z, 90))
    return max(p50, p90 - 0.015), {"p50": p50, "p90": p90}


def object_base_z(pts):
    return float(np.percentile(np.asarray(pts)[:, 2], 2))


def fit_circle_xy(xy):
    """Algebraic (Kasa) circle fit. Returns (cx, cy, r, rms) or None."""
    xy = np.asarray(xy, float)
    if len(xy) < 8:
        return None
    A = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:  # noqa: BLE001
        return None
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2)); res = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r
    return float(cx), float(cy), r, float(np.sqrt((res ** 2).mean()))


def object_axis_center_xy(pts, cam_xy, r_range=(0.02, 0.07)):
    """Axis (body) center of a roughly cylindrical object from its visible surface points.
    Circle fit on the middle z-band with one outlier-rejection pass; fallback = centroid shifted away from the camera."""
    pts = np.asarray(pts, float)
    zmin, zmax = np.percentile(pts[:, 2], 2), np.percentile(pts[:, 2], 98)
    h = max(zmax - zmin, 1e-3)
    band = pts[(pts[:, 2] > zmin + 0.25 * h) & (pts[:, 2] < zmin + 0.75 * h)]
    cen = pts[:, :2].mean(0)
    fit = fit_circle_xy(band[:, :2]) if len(band) >= 12 else None
    if fit is not None:
        cx, cy, r, rms = fit
        res = np.abs(np.hypot(band[:, 0] - cx, band[:, 1] - cy) - r)
        inl = band[res < 0.01]
        if len(inl) >= 10:
            fit2 = fit_circle_xy(inl[:, :2])
            if fit2 is not None:
                cx, cy, r, rms = fit2
        if r_range[0] <= r <= r_range[1] and rms < 0.006 and np.hypot(cx - cen[0], cy - cen[1]) < 0.05:
            return np.array([cx, cy]), {"method": "circle_fit", "r": r, "rms": rms, "n_band": int(len(band))}
    d = cen - np.asarray(cam_xy, float)[:2]
    d = d / (np.linalg.norm(d) + 1e-9)
    return cen + d * PRIM["centroid_shift_fallback"], {"method": "centroid_shift", "shift": PRIM["centroid_shift_fallback"]}


# ===== C6I4 place_primitive: the hang is measured on the points a held body can occupy (round 20260924, chunk 6, iteration 4) =====
# A body held by the pads hangs between PRIM["drop_bounds"] below the TCP - the module's own sanity range, used by prim_place to
# accept or discard the lift-pose measurement. A mask whose base percentile lies deeper than the upper bound therefore carries
# points of another body (the surface under the open hand, seen through the mask's border pixels). Those points also destroy the
# height band of the axis fit (it is built from the mask's own p2/p98), so the whole measurement is lost and the caller falls back
# to the grasp-frame hang, which is the hang the object had before it settled into the pinch. Here the impossible points are
# dropped instead, using the same bound that declares them impossible.
def ph_held_body_points(pts, tcp_p, min_pts):
    """The subset of a held-object mask that a body hanging from the pads can occupy.

    Returns the mask unchanged (the same array) whenever its base is inside the module's maximum hang below the TCP, so every
    measurement the module could already use is untouched. Otherwise the points deeper than that bound are dropped, and the
    trimmed cloud is adopted only if it keeps the majority of the mask and at least `min_pts` points and its own hang lies inside
    PRIM["drop_bounds"]; any other case returns the mask unchanged (fail-closed: the caller then falls back exactly as before)."""
    pts = np.asarray(pts, float)
    lo, hi = PRIM["drop_bounds"]
    z_bound = float(tcp_p[2]) - float(hi)
    if object_base_z(pts) >= z_bound:
        return pts
    keep = pts[pts[:, 2] >= z_bound]
    if len(keep) < max(int(min_pts), 0.5 * len(pts)):
        return pts
    return keep if float(lo) <= float(tcp_p[2]) - object_base_z(keep) <= float(hi) else pts
# ===== end C6I4 place_primitive =====


def measure_held_object(prompts, obs, min_pts=None):
    """Public measurement of the object held in the gripper (elevated). Returns dict or None."""
    min_pts = min_pts or PRIM["held_min_pts"]
    cam = obs["agentview"]; rgb = cam["images"]["rgb"]; depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    K, E = cam["intrinsics"], cam["pose_mat"]
    tcp_p, tcp_q = tcp_from_obs(obs)
    cam_xy = np.asarray(E)[:2, 3]
    best = None
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt); _PRIM_STATS["sam3_calls"] += 1
        for m in (masks or [])[:5]:
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < min_pts:
                continue
            zc = float(np.median(pts[:, 2]))
            cxy = pts[:, :2].mean(0)
            # must be elevated (near the TCP height band) and laterally near the TCP
            if not (tcp_p[2] - 0.20 < zc < tcp_p[2] + 0.05) or np.hypot(*(cxy - tcp_p[:2])) > 0.12:
                continue
            if best is None or m["score"] > best["score"]:
                best = {"score": float(m["score"]), "prompt": prompt, "pts": pts}
        if best is not None:
            break
    if best is None:
        return None
    pts = ph_held_body_points(best["pts"], tcp_p, min_pts)   # C6I4: measure on the points a body held by the pads can occupy
    axis_xy, how = object_axis_center_xy(pts, cam_xy)
    if len(pts) < len(best["pts"]):
        how = dict(how, dropped=int(len(best["pts"]) - len(pts)))   # C6I4: logged in the held_measure mark
    out = {"prompt": best["prompt"], "score": best["score"], "n_pts": int(len(pts)), "pts": pts,   # C3I3: the points themselves (t_footprint_misfit)
           "axis_xy": axis_xy, "centroid_xy": pts[:, :2].mean(0), "z_min": object_base_z(pts), "z_max": float(np.percentile(pts[:, 2], 98)),
           "tcp_p": tcp_p, "tcp_q": tcp_q, "offset_xy_base": axis_xy - tcp_p[:2], "drop_below_tcp": float(tcp_p[2] - object_base_z(pts)), "how": how}
    return out


def compensate_xy(place_xy, measurements, q_place):
    """P3: held-object XY compensation with a confidence check. measurements: list of measure_held_object dicts.
    The offset is measured in the base frame at the lift orientation; it is rotated into the gripper frame and
    re-expressed at the placement orientation. Returns (place_xy_tcp, info)."""
    ms = [m for m in measurements if m is not None and np.linalg.norm(m["offset_xy_base"]) < PRIM["held_offset_max"]]
    info = {"n_valid": len(ms), "applied": False}
    if len(ms) < 2:
        info["reason"] = "fewer than two valid held-object measurements"
        return np.asarray(place_xy, float), info
    offs = np.array([m["offset_xy_base"] for m in ms])
    spread = float(np.max(np.linalg.norm(offs - offs.mean(0), axis=1)))
    info["spread_cm"] = round(spread * 100, 2)
    if spread > PRIM["held_agree_tol"]:
        info["reason"] = f"measurements disagree ({spread*100:.1f} cm > {PRIM['held_agree_tol']*100:.1f} cm)"
        return np.asarray(place_xy, float), info
    off_b = np.array([*offs.mean(0), 0.0])
    q_lift = ms[-1]["tcp_q"]
    off_g = _R(q_lift).T @ off_b                    # gripper-frame offset
    off_b_place = _R(np.asarray(q_place, float)) @ off_g  # re-expressed at placement orientation
    info.update({"applied": True, "offset_base_lift_cm": (off_b[:2] * 100).round(2), "offset_base_place_cm": (off_b_place[:2] * 100).round(2),
                 "rot_change_deg": round(float(np.degrees(np.arccos(np.clip(abs(np.dot(_Rot.from_matrix(_R(q_lift)).as_quat(), _Rot.from_matrix(_R(q_place)).as_quat())), 0, 1)) * 2)), 1)})
    return np.asarray(place_xy, float) - off_b_place[:2], info


# ----------------------------------------------------------------------------- constrained descent (P2/P3)
def constrained_descend(xy, q, z_release, sg="sg"):
    """Short IK segments straight down at fixed XY (re-commanded each segment). Stops when the TCP reaches
    z_release, when progress stalls (contact inferred from public TCP only), or at the segment/time caps."""
    xy = np.asarray(xy, float)
    p, _, _ = read_tcp()
    _mark(ev="descend_start", sg=sg, tcp=p, xy_cmd=xy, z_release=z_release)
    t0 = _time.time(); segs = []; stalled = False; n = 0; reached = False
    z = p[2]; sag = PRIM["sag_default"]; sags = []
    while n < PRIM["descend_max_segments"] and _time.time() - t0 < PRIM["descend_time_cap_s"]:
        remaining = z - z_release
        if remaining <= PRIM["reach_tol"]:
            reached = True; break
        step = PRIM["descend_fine"] if remaining < PRIM["descend_fine_zone"] else PRIM["descend_coarse"]
        z_goal = max(z_release, z - step)
        z_cmd = z_goal - sag  # compensate the steady-state sag so the segment ends near z_goal
        r = move_tcp_segment([xy[0], xy[1], z_cmd], q, f"{sg}:descend{n}")
        segs.append(r); n += 1
        if r.get("ik_failed"):
            break
        z_new = float(r["act"][2]); drop_cmd = z - z_goal; drop_act = z - z_new
        r["drop_cmd_cm"] = round(drop_cmd * 100, 2); r["drop_act_cm"] = round(drop_act * 100, 2); r["sag_used_cm"] = round(sag * 100, 2)
        z_short = z_new - z_cmd  # raw shortfall vs the actual command (free-space sag ~0.5 cm)
        if (drop_act < PRIM["descend_stall_frac"] * drop_cmd or z_short > PRIM["contact_z_short"]) and z_new > z_release + PRIM["reach_tol"]:
            stalled = True; _PRIM_STATS["stalls"] += 1
            _mark(ev="descend_stall", sg=sg, seg=n - 1, drop_cmd_cm=r["drop_cmd_cm"], drop_act_cm=r["drop_act_cm"], z_short_cm=round(z_short * 100, 2), tcp=r["act"])
            break
        if r["reached"] or z_short < PRIM["contact_z_short"]:
            sags.append(max(0.0, z_short)); sag = float(np.clip(np.median(sags), 0.0, PRIM["sag_max"]))
        z = z_new
        if abs(z - z_release) <= PRIM["reach_tol"]:
            reached = True; break
    p1, _, _ = read_tcp()
    out = {"segments": segs, "n_segments": n, "stalled": stalled, "reached_release_height": reached or abs(p1[2] - z_release) <= PRIM["reach_tol"],
           "final_tcp": p1, "xy_err_cm": float(np.hypot(*(p1[:2] - xy)) * 100), "z_short_cm": float((p1[2] - z_release) * 100), "wall_s": round(_time.time() - t0, 2)}
    _mark(ev="descend_end", sg=sg, n_segments=n, stalled=stalled, xy_err_cm=round(out["xy_err_cm"], 2), z_short_cm=round(out["z_short_cm"], 2), tcp=p1)
    return out


# ----------------------------------------------------------------------------- integrated placement

# ===== M1 T: transport orientation fallback (vertical-axis alternatives, planned before moving) =====
def _yaw_about_vertical(q, deg):
    R = Rotation.from_euler("z", deg, degrees=True).as_matrix() @ Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    x = Rotation.from_matrix(R).as_quat()
    return np.array([x[3], x[0], x[1], x[2]])
def t_select_transport(pose, q, sg="sg"):
    q = np.asarray(q, float)
    cands = [("as_is", q), ("flip", _flip_tool_z(q)), ("yaw+90", _yaw_about_vertical(q, 90)), ("flip_yaw+90", _flip_tool_z(_yaw_about_vertical(q, 90))),
             ("yaw-90", _yaw_about_vertical(q, -90)), ("flip_yaw-90", _flip_tool_z(_yaw_about_vertical(q, -90)))]
    plans = [move_to_pose(np.asarray(pose, float), qq, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=f"{sg}:transport_plan_{name}") for name, qq in cands]
    feas = [(int(p["steps_planned"]), k) for k, p in enumerate(plans) if p.get("detail") == "plan_only" and p.get("steps_planned") is not None]
    chosen = 0 if any(k == 0 for _, k in feas) else (min(feas)[1] if feas else 0)
    print("TRANSPORT_PLAN " + json.dumps({"sg": sg, "candidates": [c[0] for c in cands], "planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": cands[chosen][0]}), flush=True)
    return cands[chosen][1], cands[chosen][0]
def t_select_transport_p3(place_xy, meas, q, lift_z, sg="sg", opening=None, slab=None):   # C3I3: opening = the container / bay opening record (None: the previous rule)   # C12I1: slab = the held slab's rectangle (t_held_slab; None: the previous rule)
    """Amendment 2: each orientation candidate is planned at ITS OWN compensated TCP (the held offset, measured once at the
    lift pose, re-expressed at that orientation by compensate_xy), because the IK margin at the compensated pose can be
    millimetres while the nominal pose is feasible. Returns (q, tcp_xy, compensation_info, chosen_name)."""
    q = np.asarray(q, float)
    cands = [("as_is", q), ("flip", _flip_tool_z(q)), ("yaw+90", _yaw_about_vertical(q, 90)), ("flip_yaw+90", _flip_tool_z(_yaw_about_vertical(q, 90))),
             ("yaw-90", _yaw_about_vertical(q, -90)), ("flip_yaw-90", _flip_tool_z(_yaw_about_vertical(q, -90)))]
    comp = [bf_compensate_xy(np.asarray(place_xy, float), meas, qq, slab, opening) for _, qq in cands]   # C13I1: a held slab into a measured bay -> the slab offset (t_slab_compensate_xy); else gh_compensate_xy as before   # C12I4: a protrusion-pinched body's centre is placed on the target (gh_compensate_xy); every other grasp: compensate_xy, unchanged
    plans = [move_to_pose(np.array([c[0][0], c[0][1], lift_z]), qq, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=f"{sg}:transport_plan_{name}") for (name, qq), c in zip(cands, comp)]
    feas = [(int(p["steps_planned"]), k) for k, p in enumerate(plans) if p.get("detail") == "plan_only" and p.get("steps_planned") is not None]
    misfit = t_footprint_misfit(meas, q, [qq for _, qq in cands], opening)   # C3I3: None without an opening / a held measurement
    passing = [] if misfit is None else [k for _, k in feas if misfit[k] <= CP_FIT_TOL]   # C3I3: the feasible candidates whose held footprint passes the opening
    fit = passing if passing else [k for _, k in feas]   # none passes (or no opening / measurement): the previous rule among all feasible candidates; all pass: the same set
    chosen = 0 if 0 in fit else (min((s, k) for s, k in feas if k in fit)[1] if fit else 0)   # as before among those: as_is, else the fewest planned steps
    print("TRANSPORT_PLAN " + json.dumps({"sg": sg, "candidates": [c[0] for c in cands], "planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": cands[chosen][0], "at": "compensated_tcp", "misfit_cm": None if misfit is None else [round(100.0 * m, 2) for m in misfit], "passing": [cands[k][0] for k in passing]}), flush=True)
    _al = t_align_slab(slab, opening, place_xy, meas, q, lift_z, cands, comp, plans, chosen, sg=sg) if (slab is not None and opening is not None) else None   # C12I1: a squared slab that does not fit at the previous choice is turned onto the opening's long axis
    if _al is not None:
        return _al
    return cands[chosen][1], comp[chosen][0], comp[chosen][1], cands[chosen][0]
# ===== end M1 T =====


# ===== N1 P: differential pre-release slip check + bounded re-centering (Q1 amendment: executed-correction residual) =====
P_SLIP_BOUND = 0.008     # m, unchanged (chosen from 134 recorded placements, dev data)
P_SLIP_RETRIES = 2       # bounded corrections; release regardless afterwards
P_SLIP_UP = 0.03         # m, lift before the lateral shift
P_SHIFT_TOL = 0.003      # m, tolerance of the correction move (Q1; the parent used 0.015, which accepted an unexecuted ~1 cm shift as reached)
P_SHIFT_MIN_FRAC = 0.5   # Q1: a correction that executes less than this fraction of its command means the motion is blocked -> stop correcting
def _yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
def p_slip_check(summary, tcp_xy, q_place, z_release, held_prompts, meas_lift, sg, opening=None):   # C11I4: opening = the placement's opening record (None otherwise)
    xy_now = np.asarray(tcp_xy, float).copy(); hist = []
    ref = [m for m in (meas_lift or []) if m is not None]
    if not held_prompts or not ref:
        print("PLACE_SLIP " + _json.dumps({"sg": sg, "status": "no_reference"}), flush=True)
        return {"tcp_xy": xy_now, "history": hist, "status": "no_reference"}
    off_ref = np.mean([np.asarray(m["offset_xy_base"], float) for m in ref], axis=0); q_ref = np.asarray(ref[0]["tcp_q"], float)
    p0, _, _ = read_tcp(); tcp0 = np.asarray(p0[:2], float)   # Q1: executed corrections are the TCP displacement from here
    _occ = p_slip_occluded(opening, ref, float(p0[2]))   # C11I4: the held body crosses the opening's rim plane -> no comparable measurement
    if _occ is not None:
        rec = {"k": 0, "status": "unmeasured_in_opening", "action": "release", **_occ}; hist.append(rec)
        print("PLACE_SLIP " + _json.dumps({"sg": sg, **rec}), flush=True)
        return {"tcp_xy": xy_now, "history": hist, "status": "release"}
    for k in range(P_SLIP_RETRIES + 1):
        m = ps_measure_held(held_prompts, get_observation(), ref, sg=sg, k=k)   # C15I1: only a mask consistent with the lift-pose held body is the held body
        if m is None:
            hist.append({"k": k, "status": "unmeasured", "action": "release"}); print("PLACE_SLIP " + _json.dumps({"sg": sg, "k": k, "status": "unmeasured", "action": "release"}), flush=True); break
        dyaw = _yaw_of(m["tcp_q"]) - _yaw_of(q_ref); c, s = np.cos(dyaw), np.sin(dyaw); Rz = np.array([[c, -s], [s, c]])
        slip = np.asarray(m["offset_xy_base"], float) - Rz @ off_ref          # in-hand slip since the lift; a TCP shift does not change it
        p_now, _, _ = read_tcp(); p_now = np.asarray(p_now, float); applied = p_now[:2] - tcp0   # Q1: correction executed so far
        resid = slip + applied; mag = float(np.hypot(*resid))
        rec = {"k": k, "slip_cm": [round(100 * float(v), 2) for v in slip], "applied_cm": [round(100 * float(v), 2) for v in applied], "residual_cm": round(100 * mag, 2),
               "mag_cm": round(100 * mag, 2), "dyaw_deg": round(float(np.degrees(dyaw)), 1), "how": m["how"]["method"], "bound_cm": 100 * P_SLIP_BOUND}
        if mag <= P_SLIP_BOUND or k == P_SLIP_RETRIES:
            rec["action"] = "release" if mag <= P_SLIP_BOUND else "release_after_retries"; hist.append(rec); print("PLACE_SLIP " + _json.dumps({"sg": sg, **rec}), flush=True); break
        z_up = float(p_now[2] + P_SLIP_UP)
        r1 = move_to_pose(np.array([p_now[0], p_now[1], z_up]), q_place, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:slip_up"); _PRIM_STATS["motion_calls"] += 1
        xy_new = p_now[:2] - resid                                            # command relative to the actual TCP
        r2 = move_to_pose(np.array([xy_new[0], xy_new[1], z_up]), q_place, mode="free", tol_pos=P_SHIFT_TOL, step_budget=BUDGET_FREE, label=f"{sg}:slip_shift"); _PRIM_STATS["motion_calls"] += 1
        p2, _, _ = read_tcp(); p2 = np.asarray(p2, float); executed = p2[:2] - p_now[:2]; frac = float(np.hypot(*executed) / max(mag, 1e-9))
        rec.update(action="recenter", up=r1.get("status"), shift=r2.get("status"), xy_from=[float(v) for v in p_now[:2]], xy_to=[float(v) for v in xy_new],
                   executed_cm=[round(100 * float(v), 2) for v in executed], executed_frac=round(frac, 2)); hist.append(rec)
        print("PLACE_SLIP " + _json.dumps({"sg": sg, **rec}), flush=True)
        blocked = r1.get("status") != "reached" or frac < P_SHIFT_MIN_FRAC
        xy_now = xy_new if not blocked else p2[:2].copy()
        summary["descend"] = mc_place_descend(xy_now, q_place, float(z_release), z_up, sg)   # back onto the support before any release
        ok = summary["descend"].get("mc_result") is not None and (summary["descend"]["mc_result"] or {}).get("success")
        if blocked or not ok:
            rec2 = {"k": k, "action": "release_after_shift_blocked" if blocked else "descend_failed", "descend_ok": bool(ok)}; hist.append(rec2); print("PLACE_SLIP " + _json.dumps({"sg": sg, **rec2}), flush=True); break
    return {"tcp_xy": xy_now, "history": hist, "status": hist[-1].get("action", hist[-1].get("status")) if hist else "none"}
# ===== C15I1 place_primitive: the pre-release slip measurement is taken on the held body only (N15 group A, slot chunk15_iter1) =====
# The N1 P slip check (p_slip_check) re-measures the held body at the release height with measure_held_object and reads the change
# of its axis offset since the lift pose as an in-hand slip. measure_held_object keeps every mask whose median z lies in TCP z
# -0.20..+0.05 and whose centroid lies within 12 cm of the TCP, and takes the best score. At the lift pose only the held body is
# there; at the release height the support under it, or the reference body beside it, is there too. Recorded (development rows,
# proposals/chunk15_iter1/DIAGNOSIS.md): the best-scoring mask was then a static body - a container under a held block (base 6.5-7.9 cm
# below the TCP against 1.1-1.7 cm at the lift), a flat disc beside a held hollow body (top 3.0 cm below the TCP against 0.5 cm above),
# a patch of the support surface under a held vessel (top 14.7 cm below the TCP against 0.1 cm above). Its offset was read as a slip
# of 5.1-12.4 cm while the true in-hand offset changed by <= 0.7 cm, and the recentre walked the held body off a target it had
# already reached (into a container wall: stalled re-descent, refused release; or 7-12 cm away and back).
# A body held between the pads keeps its height relative to the TCP. Rule: at the release height a mask counts as the held body
# only if its top (p98 z - TCP z) agrees with the lift-pose top within PRIM["held_agree_tol"] (the module's own agreement tolerance
# between two held-body measurements) and its base (p2 z - TCP z) is not deeper than the lift-pose base by more than
# PRIM["release_clearance"] (at the release height the held body's base is that far above its resting surface, whose pixels at the
# mask border may join the mask; a shallower base is allowed: the lower part of the held body can be hidden by the support or a
# wall). Measured margins (DIAGNOSIS.md section 6): held-body masks top within 0.9 cm and base within 1.7 cm; static bodies top off by
# 3.5-14.8 cm or base 4.7-6.9 cm deeper. Among those the
# previous rule applies (best score, first prompt with a candidate). None consistent: the measurement is unavailable and the check
# takes its existing branch for an unavailable measurement (release). Without a usable lift reference: measure_held_object as before.
def ps_extent(pts, tcp_z):
    """(base, top) of a cloud relative to the TCP height: p2 z - tcp_z (object_base_z), p98 z - tcp_z (measure_held_object's z_max)."""
    pts = np.asarray(pts, float)
    return float(object_base_z(pts)) - float(tcp_z), float(np.percentile(pts[:, 2], 98)) - float(tcp_z)


def ps_reference(ref):
    """Median (base, top) relative to the TCP of the lift-pose held-body measurements, or None without a usable one."""
    rows = [(float(m["z_min"]) - float(m["tcp_p"][2]), float(m["z_max"]) - float(m["tcp_p"][2])) for m in (ref or [])
            if m is not None and m.get("z_min") is not None and m.get("z_max") is not None and m.get("tcp_p") is not None]
    if not rows:
        return None
    return float(np.median([r[0] for r in rows])), float(np.median([r[1] for r in rows]))


def ps_held_body(ext, ref_ext):
    """True when a candidate's (base, top) relative to the TCP is that of the held body measured at the lift pose."""
    return bool(abs(float(ext[1]) - float(ref_ext[1])) <= PRIM["held_agree_tol"] and float(ext[0]) >= float(ref_ext[0]) - PRIM["release_clearance"])


def ps_measure_held(prompts, obs, ref, min_pts=None, sg="sg", k=0):
    """measure_held_object restricted to the masks consistent with the lift-pose held body (ps_held_body). Same return format;
    None when no mask is consistent. Without a usable reference: measure_held_object(prompts, obs, min_pts) unchanged."""
    ref_ext = ps_reference(ref)
    if ref_ext is None:
        return measure_held_object(prompts, obs, min_pts)
    min_pts = min_pts or PRIM["held_min_pts"]
    cam = obs["agentview"]; rgb = cam["images"]["rgb"]; depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    K, E = cam["intrinsics"], cam["pose_mat"]
    tcp_p, tcp_q = tcp_from_obs(obs)
    cam_xy = np.asarray(E)[:2, 3]
    best = None; rejected = []
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt); _PRIM_STATS["sam3_calls"] += 1
        for m in (masks or [])[:5]:
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth_img, K, E)
            if pts is None or len(pts) < min_pts:
                continue
            zc = float(np.median(pts[:, 2]))
            cxy = pts[:, :2].mean(0)
            if not (tcp_p[2] - 0.20 < zc < tcp_p[2] + 0.05) or np.hypot(*(cxy - tcp_p[:2])) > 0.12:   # measure_held_object's own band / radius
                continue
            held = ph_held_body_points(pts, tcp_p, min_pts)   # the cloud measure_held_object would measure (C6I4)
            ext = ps_extent(held, tcp_p[2])
            if not ps_held_body(ext, ref_ext):
                rejected.append({"prompt": prompt, "score": round(float(m["score"]), 3), "n_pts": int(len(held)), "base_cm": round(100.0 * ext[0], 2), "top_cm": round(100.0 * ext[1], 2)})
                continue
            if best is None or m["score"] > best["score"]:
                best = {"score": float(m["score"]), "prompt": prompt, "pts": pts, "held": held}
        if best is not None:
            break
    if rejected:
        print("PLACE_SLIP_IDENTITY " + _json.dumps({"sg": sg, "k": k, "ref_base_cm": round(100.0 * ref_ext[0], 2), "ref_top_cm": round(100.0 * ref_ext[1], 2),
                                                     "top_tol_cm": round(100.0 * PRIM["held_agree_tol"], 2), "base_tol_cm": round(100.0 * PRIM["release_clearance"], 2), "rejected": rejected[:6], "n_rejected": len(rejected),
                                                     "accepted": None if best is None else {"prompt": best["prompt"], "score": round(best["score"], 3), "n_pts": int(len(best["held"]))}}), flush=True)
    if best is None:
        return None
    pts = best["held"]
    axis_xy, how = object_axis_center_xy(pts, cam_xy)
    if len(pts) < len(best["pts"]):
        how = dict(how, dropped=int(len(best["pts"]) - len(pts)))
    return {"prompt": best["prompt"], "score": best["score"], "n_pts": int(len(pts)), "pts": pts,
            "axis_xy": axis_xy, "centroid_xy": pts[:, :2].mean(0), "z_min": object_base_z(pts), "z_max": float(np.percentile(pts[:, 2], 98)),
            "tcp_p": tcp_p, "tcp_q": tcp_q, "offset_xy_base": axis_xy - tcp_p[:2], "drop_below_tcp": float(tcp_p[2] - object_base_z(pts)), "how": how}
# ===== end C15I1 place_primitive =====
# ===== end N1 P =====


# ===== Q1 C: transport clearance from the public depth (height map of the static part of the workspace) =====
C_CELL = 0.02                         # m, height-map cell
C_WS = ((0.1, 0.9), (-0.6, 0.6), (-0.05, 0.40))   # base-frame workspace box for the map (x, y, z); same box the verifier uses
C_CLEAR = 0.07                        # m, margin over the corridor maximum (hang bias <= 2 cm, sag ~0.5 cm, map noise; R1: + linear-path bow, predicted p90 1.1 / max 1.7 cm)
C_W_EXTRA = 0.07                      # m, corridor half-width beyond the held object's radius (offset error, swing; R1: + linear-path lateral wander <= 2.4 cm)
C_EXCL_EXTRA = 0.04                   # m, disk around the held object's axis always excluded (plus its elevated blob)
C_ARM_STRIP = 0.12                    # m, half-width of the base->TCP strip in each view (robot arm)
C_VIEW_MIN_DIST = 0.10                # m, a second view must have been taken with the TCP at least this far away
C_Z_MAX = 0.40                        # m, cap on the transport height
C_ELEV = 0.03                         # m, elevation above the support that counts as "object" for the blob
C_R_DEFAULT = 0.06                    # m, held-object radius when no circle fit is available
C_VIEWS_KEEP = 4
_C_VIEWS = []                         # (tcp_xy, obs) of recent observations (stashed at the observe step and at the lift-pose measurement)
def _c_stash_view(obs):
    try:
        p, _ = tcp_from_obs(obs); _C_VIEWS.append((np.asarray(p[:2], float).copy(), obs)); del _C_VIEWS[:-C_VIEWS_KEEP]
    except Exception:  # noqa: BLE001
        pass
def _c_grid():
    nx = int(round((C_WS[0][1] - C_WS[0][0]) / C_CELL)); ny = int(round((C_WS[1][1] - C_WS[1][0]) / C_CELL))
    cx = C_WS[0][0] + (np.arange(nx) + 0.5) * C_CELL; cy = C_WS[1][0] + (np.arange(ny) + 0.5) * C_CELL
    return np.meshgrid(cx, cy, indexing="ij")
def _c_height_map(obs):
    cam = obs["agentview"]; depth = cam["images"]["depth"]; depth = depth[:, :, 0] if len(depth.shape) == 3 else depth
    pts = mask_to_world_points(np.ones(depth.shape, np.uint8), depth, cam["intrinsics"], cam["pose_mat"])
    CX, CY = _c_grid(); nx, ny = CX.shape; H = np.full(nx * ny, -np.inf)
    if pts is not None and len(pts):
        pts = np.asarray(pts, float)
        m = (pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] > C_WS[2][0]) & (pts[:, 2] < C_WS[2][1])
        p = pts[m]; ix = np.clip(((p[:, 0] - C_WS[0][0]) / C_CELL).astype(int), 0, nx - 1); iy = np.clip(((p[:, 1] - C_WS[1][0]) / C_CELL).astype(int), 0, ny - 1)
        np.maximum.at(H, ix * ny + iy, p[:, 2])
    H = H.reshape(nx, ny); H[np.isinf(H)] = np.nan; return H
def _c_seg_dist(px, py, a, b):
    d = np.asarray(b, float) - np.asarray(a, float); L2 = float(d @ d)
    t = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2, 0.0, 1.0) if L2 > 1e-12 else 0.0
    return np.hypot(px - (a[0] + t * d[0]), py - (a[1] + t * d[1]))
def _c_blob(H, support_z, axis_xy, r_obj):
    """Connected elevated cells around the held object's axis (its body, handle and the gripper above it) in one view."""
    CX, CY = _c_grid(); nx, ny = H.shape; elev = np.nan_to_num(H, nan=-np.inf) > float(support_z) + C_ELEV
    near = np.hypot(CX - axis_xy[0], CY - axis_xy[1]) <= r_obj + C_EXCL_EXTRA
    starts = list(zip(*np.where(elev & near))); blob = np.zeros_like(elev); stack = list(starts)
    for s in starts:
        blob[s] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and elev[a, b] and not blob[a, b]:
                blob[a, b] = True; stack.append((a, b))
    return blob | near
def c_transport_height(lift_z, support_z, obj_xy, target_xy, r_obj, hang, sg="sg"):
    """z for the transport: max(lift_z, corridor max height + hang + C_CLEAR), capped. Public depth only."""
    info = {"sg": sg, "lift_z": float(lift_z), "n_views": 0, "hmax": None, "z": float(lift_z), "reason": "no_view"}
    if not _C_VIEWS or hang is None:
        print("TRANSPORT_CLEAR " + _json.dumps(info), flush=True); return float(lift_z), info
    cur_xy, cur_obs = _C_VIEWS[-1]; views = [(cur_xy, cur_obs)]
    for xy, obs in reversed(_C_VIEWS[:-1]):
        if float(np.hypot(*(np.asarray(xy) - cur_xy))) >= C_VIEW_MIN_DIST:
            views.append((xy, obs)); break
    Hs = [_c_height_map(o) for _, o in views]; CX, CY = _c_grid()
    st = np.stack(Hs); has = ~np.isnan(st)
    with np.errstate(all="ignore"):
        C = np.nanmin(st, 0)
    strips = np.stack([_c_seg_dist(CX, CY, np.zeros(2), np.asarray(xy, float)) < C_ARM_STRIP for xy, _ in views])
    only_arm = np.all(~has | strips, 0); C[only_arm | ~has.any(0)] = np.nan      # a cell is kept only where some view sees it outside its own arm strip
    obj_xy = np.asarray(obj_xy, float); target_xy = np.asarray(target_xy, float); r = float(r_obj) if r_obj else C_R_DEFAULT
    excl = _c_blob(Hs[0], support_z, obj_xy, r)
    corridor = (_c_seg_dist(CX, CY, obj_xy, target_xy) < r + C_W_EXTRA) & ~excl
    vals = C[corridor]; ok = vals[~np.isnan(vals)]
    hmax = float(ok.max()) if len(ok) else None
    z = float(lift_z) if hmax is None else max(float(lift_z), hmax + float(hang) + C_CLEAR); z_c = min(z, C_Z_MAX)
    info.update(n_views=len(views), hmax=None if hmax is None else round(hmax, 4), hang=round(float(hang), 4), r_obj=round(r, 4), corridor_cells=int(corridor.sum()), missing_cells=int(np.isnan(vals).sum()),
                excluded_cells=int((excl & (_c_seg_dist(CX, CY, obj_xy, target_xy) < r + C_W_EXTRA)).sum()), z=round(z_c, 4), raised_cm=round(100 * (z_c - float(lift_z)), 2), capped=bool(z > C_Z_MAX), reason="ok" if hmax is not None else "empty_corridor")
    print("TRANSPORT_CLEAR " + _json.dumps(info), flush=True)
    return z_c, info
# ===== end Q1 C =====

# ===== C13I5 place_primitive: the bay's own measured rim bounds the transport height (N13 group J, slot C13I5) =====
# Recorded (development rows, DIAGNOSIS.md section 4): the corridor clearance (c_transport_height) keeps a map cell only where
# some view sees it outside that view's own base->TCP arm strip (C_ARM_STRIP). A bay lying under the arm's line in the observe
# and lift views (target 0.0-3.3 cm from the segment, strip half-width 12 cm) is never kept (0 of 64 bay cells, 10/10 such
# placements): the corridor maximum fell to the table (hmax -0.030 / -0.016 m), the transport ran at the fixed 0.25 m lift
# with the held body's measured base 1.6 cm below the bay's rim, the body caught on the bay's wall (tilt 15.5-15.9 deg) and
# the guarded descent refused (premature_contact). The bay's rim is measured on the first public frame (the opening record) and is a
# static surface the held body must pass over to enter the bay; the corridor rule is applied to it:
#     z_tr >= rim_z + max(hang, CA_HAND tip) + C_CLEAR, capped at C_Z_MAX
# rim_z / hang / the lowest-point rule: as place_entry_height (the same opening record and drop_used); C_CLEAR / C_Z_MAX: the
# corridor rule's own clearance and cap. Only bay openings (centre_src == BF_SRC); otherwise the parent's height unchanged.
def rf_rim_floor(opening, hang):
    """Transport TCP height at which the held body's lowest point clears the bay opening's rim by C_CLEAR (uncapped), or None
    when the rule does not apply (no bay opening record, no rim, or no hang)."""
    if not opening or opening.get("centre_src") != BF_SRC or opening.get("rim_z") is None or hang is None:
        return None
    return float(opening["rim_z"]) + max(float(hang), float(CA_HAND["tip"])) + C_CLEAR


def rf_transport_height(z_tr, opening, hang, sg="sg"):
    """The parent's corridor transport height z_tr, raised to the bay rim floor (capped at C_Z_MAX) when it is lower."""
    z_floor = rf_rim_floor(opening, hang)
    if z_floor is None:
        return float(z_tr)
    z = max(float(z_tr), min(z_floor, C_Z_MAX))
    print("TRANSPORT_RIM_FLOOR " + _json.dumps({"sg": sg, "rim_z": round(float(opening["rim_z"]), 4), "hang": round(float(hang), 4),
                                                 "z_floor": round(z_floor, 4), "z_corridor": round(float(z_tr), 4), "z": round(z, 4),
                                                 "raised_cm": round(100.0 * (z - float(z_tr)), 2), "capped": bool(z_floor > C_Z_MAX)}), flush=True)
    return z
# ===== end C13I5 place_primitive =====


def prim_place(place_xy, q_place, lift_z, p0_descent_zs, support_z, drop_est, held_prompts=None, sg="sg1", settle_obs=5, alt_quat=None, opening=None, support_rim=None):   # C11I5: support_rim = the rim of a round hollow on-support (place_support_rim; None otherwise)   # C2I4: opening = the container's opening record of a containment placement (None otherwise)
    """Transport (as P0: one IK move above the target) -> descent -> release -> retreat.
    P1: P0's own descent moves (list of TCP z targets `p0_descent_zs`, each one IK move), then safe_retreat.
    P2: constrained descent to support_z + drop_est + clearance, then safe_retreat.
    P3: P2 with held-object XY compensation (two public measurements at the lift pose).
    Returns a summary dict (also printed as PRIM json)."""
    ver = PRIM["version"]
    place_xy = np.asarray(place_xy, float); q_place = np.asarray(q_place, float)
    summary = {"version": ver, "sg": sg, "place_xy_nominal": place_xy.copy()}
    p0_descent_zs = [float(z) for z in np.atleast_1d(p0_descent_zs)]
    p0_release_z = p0_descent_zs[-1]
    _mark(ev="place_start", sg=sg, version=ver, place_xy=place_xy, support_z=support_z, drop_est=drop_est, p0_descent_zs=p0_descent_zs)
    tcp_xy = place_xy.copy(); _tsel = None; _meas_lift = []; z_tr = float(lift_z); _cinfo = None
    drop_used = float(drop_est); drop_source = "grasp_frame_estimate"
    if ver in ("P2", "P3") and held_prompts:
        # Hang depth (TCP to object base) measured publicly at the lift pose: P2 one measurement, P3 two (also used for XY).
        meas = []
        for k in range(2 if ver == "P3" else 1):
            obs = get_observation(); _c_stash_view(obs)   # Q1 C: lift-pose view
            m = measure_held_object(held_prompts, obs)
            meas.append(m)
            if m is not None:
                _mark(ev="held_measure", sg=sg, k=k, offset_xy_base_cm=(m["offset_xy_base"] * 100).round(2), drop_below_tcp_cm=round(m["drop_below_tcp"] * 100, 2),
                      score=round(m["score"], 3), n_pts=m["n_pts"], how=m["how"])
            else:
                _mark(ev="held_measure", sg=sg, k=k, result="none")
        _meas_lift = [m for m in meas if m is not None]   # N1 P: lift-pose reference for the pre-release slip check
        ds = [m["drop_below_tcp"] for m in meas if m is not None and PRIM["drop_bounds"][0] <= m["drop_below_tcp"] <= PRIM["drop_bounds"][1]]
        if ds:
            drop_used = float(np.median(ds)); drop_source = f"lifted_measurement(n={len(ds)})"
            summary["drop_measured"] = drop_used
        _mark(ev="drop_estimate", sg=sg, drop_grasp_frame_cm=round(drop_est * 100, 2), drop_used_cm=round(drop_used * 100, 2), source=drop_source)
        if ver == "P3":
            _mv = [m for m in meas if m is not None]   # Q1 C: transport height from the height map under the held object's path
            _p_now, _, _ = read_tcp(); _p_now = np.asarray(_p_now, float)
            _obj_xy = _p_now[:2] + (np.mean([np.asarray(m["offset_xy_base"], float) for m in _mv], axis=0) if _mv else np.zeros(2))
            _r_obj = next((m["how"].get("r") for m in _mv if m["how"].get("method") == "circle_fit"), None)
            z_tr, _cinfo = c_transport_height(lift_z, support_z, _obj_xy, place_xy, _r_obj, drop_used if drop_source != "grasp_frame_estimate" else None, sg=sg)
            z_tr = rf_transport_height(z_tr, opening, drop_used, sg=sg)   # C13I5: a bay opening's measured rim bounds the transport height
            q_place, tcp_xy, cinfo, _tsel = t_select_transport_p3(place_xy, meas, q_place, z_tr, sg=sg, opening=opening, slab=(t_held_slab(q_place) if opening is not None else None))   # C12I1/C12I5: the held slab's rectangle (containment only)   # M1 T (Amendment 2): orientation chosen at each candidate's compensated TCP
            summary["compensation"] = cinfo
            _mark(ev="compensation", sg=sg, **{k: v for k, v in cinfo.items()})
    summary["place_xy_tcp"] = tcp_xy
    # transport above the target (P0-identical form: one IK move at lift height)
    _mark(ev="transport_start", sg=sg, cmd=[tcp_xy[0], tcp_xy[1], z_tr], z_tr=z_tr)
    # MR: free-space transport with verified arrival; the caller keeps its orientation (alt_quat is only used when the
    # requested orientation is reported unreachable, i.e. an explicit caller-provided alternative)
    if _tsel is None:   # M1 T: no compensation happened (P1/P2 or no measurement) -> select at the transport pose itself
        q_place, _tsel = t_select_transport([tcp_xy[0], tcp_xy[1], z_tr], q_place, sg=sg)
    if z_tr > float(lift_z) + 0.005:   # Q1 C: vertical raise at the current xy before the horizontal transport
        _pr, _qr, _ = read_tcp(); _pr = np.asarray(_pr, float)
        rres = move_to_pose(np.array([_pr[0], _pr[1], z_tr]), np.asarray(_qr, float), mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:transport_raise"); _PRIM_STATS["motion_calls"] += 1
        _mark(ev="transport_raise", sg=sg, z_tr=z_tr, status=rres["status"])
    tres = move_to_pose([tcp_xy[0], tcp_xy[1], z_tr], q_place, mode="linear", tol_pos=0.015, step_budget=BUDGET_TRANSPORT, label=f"{sg}:transport")   # R1 L: straight-line transport; S1 B: own budget
    _tmode = "linear"
    if tres["status"] == "unreachable":   # R1 L: linear path refused (IK / discontinuous) -> the parent's direct joint move
        tres = move_to_pose([tcp_xy[0], tcp_xy[1], z_tr], q_place, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:transport_free"); _tmode = "free_fallback"
    print("TRANSPORT_MODE " + _json.dumps({"sg": sg, "mode": _tmode, "status": tres["status"], "detail": tres.get("detail"), "steps": tres.get("steps_used"), "planned": tres.get("steps_planned"), "n_wp": tres.get("n_waypoints")}), flush=True)
    if tres["status"] == "unreachable" and alt_quat is not None:
        q_place = np.asarray(alt_quat, float)
        tres = move_to_pose([tcp_xy[0], tcp_xy[1], z_tr], q_place, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:transport_alt")
    _PRIM_STATS["motion_calls"] += 1
    p, _, _ = read_tcp()
    _mark(ev="transport_end", sg=sg, tcp=p, xy_err_cm=round(float(np.hypot(*(p[:2] - tcp_xy)) * 100), 2), status=tres["status"], steps=tres["steps_used"])
    if not tres["reached"]:
        # precondition for the descent not met: no descent, no release; object retained; safe retreat
        _mark(ev="transport_unreached", sg=sg, status=tres["status"], reason=tres.get("reason"))
        print("PLACE_RESULT " + _json.dumps({"sg": sg, "status": "failed", "reason": f"transport_{tres['status']}", "pos_err_cm": round(tres["pos_err"] * 100, 2)}), flush=True)
        summary["retreat"] = safe_retreat(sg=sg, q_hold=q_place)
        summary["status"] = "failed"; summary["reason"] = f"transport_{tres['status']}"
        _mark(ev="place_done", sg=sg, stats=dict(_PRIM_STATS))
        return summary
    _z_entry, _einfo = place_entry_height(opening, drop_used, lift_z, z_tr) if ver in ("P2", "P3") else (float(lift_z), {"applies": False, "z_entry": round(float(lift_z), 4)})   # C11I1: the guarded descent begins above the opening's rim plane
    summary["z_entry"] = _z_entry
    print("PLACE_ENTRY " + _json.dumps({"sg": sg, **_einfo}), flush=True)
    if z_tr > float(_z_entry) + 0.005:   # Q1 C: vertical lower to the descent start height above the target (C11I1: the entry height)
        _lin = _einfo.get("bound") == "clearance"   # C11I1: a raised entry is approached on the straight vertical line (the transport's linear mode), so the guarded descent starts over its target
        lres = move_to_pose([tcp_xy[0], tcp_xy[1], _z_entry], q_place, mode="linear" if _lin else "free", tol_pos=0.015, step_budget=BUDGET_LINEAR if _lin else BUDGET_FREE, label=f"{sg}:transport_lower"); _PRIM_STATS["motion_calls"] += 1
        if _lin and lres["status"] == "unreachable":   # C11I1: linear path refused (IK / discontinuous) -> the free-mode lowering, as the transport's R1 L fallback
            lres = move_to_pose([tcp_xy[0], tcp_xy[1], _z_entry], q_place, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:transport_lower_free"); _PRIM_STATS["motion_calls"] += 1
        _mark(ev="transport_lower", sg=sg, status=lres["status"])
        if not lres["reached"]:
            print("PLACE_RESULT " + _json.dumps({"sg": sg, "status": "failed", "reason": f"transport_lower_{lres['status']}", "pos_err_cm": round(lres["pos_err"] * 100, 2)}), flush=True)
            summary["retreat"] = safe_retreat(sg=sg, q_hold=q_place)
            summary["status"] = "failed"; summary["reason"] = f"transport_lower_{lres['status']}"
            _mark(ev="place_done", sg=sg, stats=dict(_PRIM_STATS))
            return summary
    # descent
    if ver == "P1":
        _mark(ev="descend_start", sg=sg, tcp=p, xy_cmd=tcp_xy, z_release=p0_release_z, mode="p0_moves", zs=p0_descent_zs)
        for k, zc in enumerate(p0_descent_zs):
            j = _ik([tcp_xy[0], tcp_xy[1], zc], q_place)
            if j is not None:
                move_to_joints(j); _PRIM_STATS["motion_calls"] += 1
        p1, _, _ = read_tcp()
        summary["descend"] = {"n_segments": len(p0_descent_zs), "final_tcp": p1, "xy_err_cm": float(np.hypot(*(p1[:2] - tcp_xy)) * 100), "z_short_cm": float((p1[2] - p0_release_z) * 100)}
        _mark(ev="descend_end", sg=sg, n_segments=len(p0_descent_zs), tcp=p1, xy_err_cm=round(summary["descend"]["xy_err_cm"], 2), z_short_cm=round(summary["descend"]["z_short_cm"], 2))
    else:
        _hz, _hinfo = place_hand_limit(opening, tcp_xy, q_place, gripper_opening(), float(support_z) + float(drop_used) + PRIM["release_clearance"]) if opening is not None else (None, {"applies": False})   # C2I4: evaluated at the final orientation / compensated TCP, from the rest-based height
        z_release, _hl = place_release_height(support_z, drop_used, _hz)   # C2I4: = support_z + drop_used + release_clearance unless the hand limit is higher
        z_release, _rrec = place_rim_release(z_release, support_rim, drop_used)   # C11I5: over a round hollow support the held body is released with its base above the rim plane
        summary["rim_release"] = _rrec
        if _rrec.get("applies"):
            print("PLACE_RIM_RELEASE " + _json.dumps({"sg": sg, **_rrec}), flush=True)
        summary["z_release"] = z_release; summary["drop_used"] = drop_used; summary["drop_source"] = drop_source; summary["hand_limited"] = _hl
        print("PLACE_HAND_LIMIT " + _json.dumps({"sg": sg, "hand_limited": _hl, "z_release": round(float(z_release), 4), "z_rest_based": round(float(support_z) + float(drop_used) + PRIM["release_clearance"], 4), **_hinfo}), flush=True)
        _mark(ev="hand_limit", sg=sg, hand_limited=_hl, z_release=z_release, z_tcp_min=_hz)
        summary["descend"] = mc_place_descend(tcp_xy, q_place, z_release, _z_entry, sg)   # MC: contact-aware placement descent (C11I1: from the entry height)
        if summary["descend"].get("mc_result") is not None and (summary["descend"]["mc_result"] or {}).get("success"):
            summary["slip"] = p_slip_check(summary, tcp_xy, q_place, z_release, held_prompts, _meas_lift, sg, opening=opening)   # N1 P (C11I4: with the opening record)
            tcp_xy = np.asarray(summary["slip"]["tcp_xy"], float)
        summary["mg_verify"] = mf_pre_release(summary, tcp_xy, q_place, place_xy, sg)   # F: verifier in shadow (logged, never gates)
    # release (MC: only when the placement descent succeeded — release height reached, or axial contact confirmed within
    # PLACE_CONTACT_TOL above it; any other status blocks the release and retreats with the object retained)
    _rd = summary["descend"].get("mc_result") if isinstance(summary.get("descend"), dict) else None
    _ok_rel, _why_rel = gate_allow(_rd, "release")
    if not _ok_rel:
        _mark(ev="release_blocked", sg=sg, reason=_why_rel)
        print("PLACE_RESULT " + _json.dumps({"sg": sg, "status": "failed", "reason": "release_blocked: " + _why_rel}), flush=True)
        summary["retreat"] = safe_retreat(sg=sg, q_hold=q_place); summary["status"] = "failed"; summary["reason"] = "release_blocked"
        _mark(ev="place_done", sg=sg, stats=dict(_PRIM_STATS))
        return summary
    _mark(ev="release_start", sg=sg)
    open_gripper(); gate_record("release"); _PRIM_STATS["motion_calls"] += 1
    print("RELEASE_VERIFY " + _json.dumps({"sg": sg, **verify_release(gripper_opening())}), flush=True)
    _mark(ev="release_end", sg=sg)
    for _ in range(settle_obs):
        get_observation()
    # retreat (P1+)
    summary["retreat"] = safe_retreat(sg=sg, q_hold=q_place)
    _post = mf_post_release(place_xy, sg)   # F: post-release verdict in shadow (logged only; MC status kept)
    summary["status"] = "released"; summary["reason"] = "arrived_and_released"
    print("PLACE_RESULT " + _json.dumps({"sg": sg, "status": "released", "reason": "arrived_and_released", "shadow_post": _post["status"]}), flush=True)
    _mark(ev="place_done", sg=sg, stats=dict(_PRIM_STATS))
    return summary
# ===== end shared primitive =====

# ===== MR caller-side consumption of the general motion interface R (round rel_20260916_212500) =====
# Constants are gripper/controller quantities, not object or task quantities (see code/build_programs_r.py).
APPROACH_CORRIDOR = 0.04     # m, lateral corridor of a straight-line grasp approach = half the finger span (0.08 m)
SEGMENT_CORRIDOR = 0.015     # m, lateral corridor of placement/retreat segments
BUDGET_FREE = 240            # control steps for a free-space move (legacy cap 120)
BUDGET_LINEAR = 240          # control steps for a straight-line approach/segment
BUDGET_TRANSPORT = 900       # S1 B: control steps for the linear-mode transport (2 cm waypoints, each converged; R1 plan estimates up to 227)
_PRE_MOTION = ("unreachable", "invalid_input", "uncertain")   # R statuses that mean "no configuration / no motion"
def _motion_ok_to_close(r):
    """Dependent action only when the motion result is `reached`. No stalled/near-target exception; `stalled` is
    robot-motion evidence, not a grasp verdict (contract.md)."""
    if r is None:
        return False, "no_motion_result"
    if r.get("status") == "reached" and r.get("reached") is True:
        return True, "reached"
    return False, f"{r.get('status')}: pos_err {r.get('pos_err', 9)*100:.1f} cm, lateral {r.get('lateral_dev_max', 0)*100:.1f} cm"
# ===== end MR header =====

# ===== MC caller-side contact-aware consumption (round contact_primitive_20260917_112613) =====
# Constants are gripper/controller quantities or a shared perception-uncertainty bound; no object/task/scene/seed constant.
GRIPPER_CONTACT_TOL = 0.03   # m, half the Franka finger length (URDF panda_finger ~0.054 m): a contact within this distance before the grasp point still has the pads around the target
PLACE_CONTACT_TOL = 0.04     # m, uncertainty bound of the estimated release height (support surface + measured hang depth), identical for every task
CONTACT_OVERSHOOT = 0.01     # m, the contact search may command up to this beyond the target
PLACE_CORRIDOR = 0.02        # m, lateral corridor of the placement descent
BUDGET_CONTACT = 240         # control steps per contact call
def mc_grasp_descend(grasp_pos, approach_pos, q, label):
    """Contact-seeking straight descent from the verified approach pose to the grasp point (open gripper expected)."""
    r = contact_move(np.asarray(grasp_pos, float), np.asarray(q, float), approach_axis=[0.0, 0.0, -1.0], contact_tol_before=GRIPPER_CONTACT_TOL, overshoot_after=CONTACT_OVERSHOOT,
                     start_pose=np.asarray(approach_pos, float), start_tol=0.02, gripper_expect="open", lateral_corridor=APPROACH_CORRIDOR, step_budget=BUDGET_CONTACT, label=label)
    _PRIM_STATS["motion_calls"] += 1
    print("CONTACT_MOVE " + json.dumps({"label": label, "status": r["status"], "detail": r.get("detail"), "success": r["success"], "steps": r["steps_used"], "budget": r["budget"], "wall_s": r.get("wall_s"),
                                        "along_cm": round((r.get("along_final") or 0) * 100, 2), "travel_cm": round((r.get("travel") or 0) * 100, 2), "window_cm": [round(x * 100, 2) for x in (r.get("window") or [])],
                                        "pos_err_cm": round(r.get("pos_err", 9) * 100, 2), "evidence": r.get("evidence"), "verifier": verify_contact(r).get("status")}), flush=True)
    return r
def mc_close_and_verify(rg, sg):
    """Gated close + independent grasp verification from the public gripper opening. Returns the verifier dict."""
    g0 = gripper_opening()
    close_gripper(); gate_record("close"); _PRIM_STATS["motion_calls"] += 1
    g1 = gripper_opening(); _lmo = lt_min_opening_for(rg.get("target") if isinstance(rg, dict) else None)   # N15 C15I2: LT2 measured thin wall (None: default bound)
    gv = verify_grasp(g0, g1) if _lmo is None else {**verify_grasp(g0, g1, min_opening=_lmo), "lt_min_opening": _lmo}
    print("GRASP_VERIFY " + json.dumps({"sg": sg, **gv}), flush=True)
    g1_width_shadow(gv)   # G1 I3: shadow diagnostic only
    if _FCFG.get("width_gate") and (_MF.get("last_plan") or {}).get("mode") == "body_pinch" and gv.get("status") == "verified" and _WG.get("pts") is not None and _WG.get("quat") is not None and rg.get("final_tcp") is not None:   # Amendment 1: body-pinch plans only (a rim pinch closes on the wall)
        wv = verify_width(_WG["pts"], float(rg["final_tcp"][2]), _WG["quat"], gv.get("after"))   # W: closed width vs body width at the pinch band
        print("GRASP_WIDTH_GATE " + json.dumps({"sg": sg, **wv}), flush=True)
        if wv.get("status") == "failed":
            gv = {**gv, "status": "failed", "detail": "width_inconsistent", "reason": wv.get("reason")}
    return gv
def mc_place_descend(xy, q, z_release, lift_z, sg):
    """Contact-seeking placement descent (holding expected): success = reached the release height or confirmed axial contact
    within PLACE_CONTACT_TOL above it."""
    xy = np.asarray(xy, float)
    _mark(ev="descend_start", sg=sg, xy_cmd=xy, z_release=z_release, mode="contact_move")
    r = contact_move(np.array([xy[0], xy[1], float(z_release)]), np.asarray(q, float), approach_axis=[0.0, 0.0, -1.0], contact_tol_before=PLACE_CONTACT_TOL, overshoot_after=CONTACT_OVERSHOOT,
                     start_pose=np.array([xy[0], xy[1], float(lift_z)]), start_tol=0.02, gripper_expect="holding", lateral_corridor=PLACE_CORRIDOR, step_budget=BUDGET_CONTACT, label=f"{sg}:place_descent")
    _PRIM_STATS["motion_calls"] += 1; _PRIM_STATS["segments"] += 1
    if not r["success"]:
        _PRIM_STATS["unreached"] += 1
    print("CONTACT_MOVE " + json.dumps({"label": f"{sg}:place_descent", "status": r["status"], "detail": r.get("detail"), "success": r["success"], "steps": r["steps_used"], "budget": r["budget"], "wall_s": r.get("wall_s"),
                                        "along_cm": round((r.get("along_final") or 0) * 100, 2), "travel_cm": round((r.get("travel") or 0) * 100, 2), "window_cm": [round(x * 100, 2) for x in (r.get("window") or [])],
                                        "pos_err_cm": round(r.get("pos_err", 9) * 100, 2), "evidence": r.get("evidence"), "verifier": verify_contact(r).get("status")}), flush=True)
    p1 = np.asarray(r.get("final_tcp", [np.nan] * 3), float)
    _mark(ev="descend_end", sg=sg, status=r["status"], detail=r.get("detail"), tcp=p1, z_short_cm=round(float(p1[2] - z_release) * 100, 2) if np.all(np.isfinite(p1)) else None)
    return {"mc_result": r, "segments": [], "n_segments": r.get("n_waypoints_executed"), "status": r["status"], "reached_release_height": bool(r["success"]), "final_tcp": p1,
            "z_short_cm": float((p1[2] - z_release) * 100) if np.all(np.isfinite(p1)) else None, "wall_s": r.get("wall_s")}
# ===== end MC header =====

# ===== E1 shared failure-propagation guards (round zeroshot_minimal_fusion_20260918_t7saYg) =====
# Gripper/interface quantities only; no object, task, scene or seed constant. Pure checks + logging; every motion they trigger is
# an EXISTING recovery path of the frozen program (backoff / safe_retreat / goto_home), entered through the same shared interface.
_E1_LAST = {"grasp_after": None, "hold": None, "stop": False}
def e1_lift_allowed(gv):
    """G1: the lift is a dependent action; it needs the gate's permission on the grasp verdict (no motion here)."""
    ok, why = gate_allow(gv, "lift")
    print("E1_GUARD " + json.dumps({"action": "lift", "allow": bool(ok), "why": why}), flush=True)
    return bool(ok), why
def e1_arrived(r, action):
    """G2: a dependent action needs the previous motion to be `reached` (interface verdict, caller tolerance). No motion here."""
    ok = isinstance(r, dict) and r.get("status") == "reached" and r.get("reached") is True
    print("E1_GUARD " + json.dumps({"action": action, "allow": bool(ok), "prev_status": (r or {}).get("status") if isinstance(r, dict) else None, "prev_detail": (r or {}).get("detail") if isinstance(r, dict) else None,
                                    "pos_err_cm": round(float(r.get("pos_err", 9)) * 100, 2) if isinstance(r, dict) else None}), flush=True)
    return bool(ok)
def e1_hold_evidence():
    """G3: public hold evidence only: the last hold verification was `verified` and the opening is still consistent with it."""
    ref = _E1_LAST.get("grasp_after"); last = _E1_LAST.get("hold")
    _lmo = _E1_LAST.get("min_opening")   # N15 C15I2: LT2 min_opening of the held grasp (None: default bound)
    hv = (verify_hold(ref, gripper_opening()) if _lmo is None else verify_hold(ref, gripper_opening(), min_opening=_lmo)) if ref is not None else {"status": "uncertain", "reason": "no reference opening"}
    ok = last == "verified" and hv.get("status") == "verified"
    print("E1_HOLD_EVIDENCE " + json.dumps({"last_hold": last, "now": hv.get("status"), "reason": hv.get("reason"), "ok": bool(ok)}), flush=True)
    return bool(ok)
# ===== end E1 guards =====

# ===== G1 shared interface patches (round zeroshot_general_grasp_contract_20260918_OxvXZB) =====
# Robot/gripper constants only (public IK box of libero_reduced.py = the contact primitive's precondition; TCP-to-fingertip 5 mm).
# No object, task, scene or seed constant. Pure checks + logging; no new motion path.
_G1_WS_LO = (-0.1, -0.5, 0.005); _G1_WS_HI = (0.75, 0.5, 0.9)
_G1_TCP_TO_FINGERTIP = 0.005; _G1_WIDTH_TOL = 0.02
def g1_descent_feasible(target, z_start):
    """I2: workspace feasibility of a straight vertical contact descent (target and overshoot end inside the public IK box), before any motion."""
    t = np.asarray(target, float).reshape(3); end = t.copy(); end[2] -= CONTACT_OVERSHOOT
    for nm, p in (("target", t), ("overshoot_end", end)):
        if np.any(p < lt_box_lo(t) - 1e-6) or np.any(p > np.asarray(_G1_WS_HI) + 1e-6):   # N15 C15I2: lt_box_lo = the parent box, with the measured floor for the LT1 target only; 1e-6: floating tolerance so a candidate exactly at the box face is not filtered when the primitive would accept it
            r = {"ok": False, "why": f"{nm} {np.round(p, 3).tolist()} outside the public IK box"}; print("CANDIDATE_FILTER " + json.dumps(r), flush=True); return r
    travel = float(z_start - t[2])
    if not (0.005 <= travel <= 0.5):
        r = {"ok": False, "why": f"travel {travel:.3f} m outside [0.005, 0.5]"}; print("CANDIDATE_FILTER " + json.dumps(r), flush=True); return r
    r = {"ok": True, "why": "workspace_ok"}; print("CANDIDATE_FILTER " + json.dumps(r), flush=True); return r
def g1_actual_target_z(rg, plan):
    """I3: the descent target actually commanded (from the primitive's result), not the stale first plan."""
    t = rg.get("target") if isinstance(rg, dict) else None
    if t is not None:
        return float(np.asarray(t, float).reshape(3)[2])
    return None if plan.get("target") is None else float(plan["target"][2])
def g1_engagement(rg):
    """I3: pad engagement estimate from public quantities: perceived top (binding frame) minus the actual fingertip height."""
    prof = _MF.get("prof") or {}; ft = rg.get("final_tcp") if isinstance(rg, dict) else None; top = prof.get("top")
    if ft is None or top is None:
        return None
    z_ft = float(np.asarray(ft, float)[2]) - _G1_TCP_TO_FINGERTIP
    return {"final_fingertip_z": round(z_ft, 4), "perceived_top": round(float(top), 4), "pad_below_top_m": round(float(top) - z_ft, 4)}
def g1_width_shadow(gv):
    """I3 (shadow): closed opening vs the perceived width across the first planned closing axis (body pinch only). Logged, never gates."""
    plan = _MF.get("last_plan") or {}
    if plan.get("mode") != "body_pinch" or not isinstance(gv, dict) or gv.get("after") is None:
        return None
    yaw0 = (plan.get("yaw_order") or [None])[0]
    cands = [c for c in (plan.get("candidates") or []) if c.get("mode") == "body_pinch" and c.get("yaw") == yaw0]
    wp = float(cands[0]["width_across"]) if cands else None; wm = float(gv["after"]) * 0.08
    rec = {"shadow": True, "mode": plan.get("mode"), "yaw_of_reference": yaw0, "width_perceived_m": None if wp is None else round(wp, 4), "width_measured_m": round(wm, 4),
           "consistent": None if wp is None else bool(abs(wm - wp) <= _G1_WIDTH_TOL)}
    print("GRASP_WIDTH_SHADOW " + json.dumps(rec), flush=True); return rec
# ===== end G1 patches =====

# ===== F caller-side header (round zeroshot_general_fusion_20260917_CRQQPG) =====
# Constants are gripper/controller quantities or shared perception bounds; no object/task/scene/seed constant. Injected by the
# runner: gg_* (module B), pv_* (module A), pg_* (placement gate), fg_* (module F). MC (contact_move / gate_allow / verifiers)
# and R are unchanged. _FCFG holds the experimental factors of this arm (set by the builder).
_WG = {'pts': None, 'quat': None}
_CUR = {'prompts': None}
_FCFG = {'grasp_geometry': 'B+G2depth', 'arm_h': 'H2', 'arm_i': 'I2', 'arm_l': 'L1', 'width_gate': True, 'arm_m': 'M1', 'arm_n': 'N1', 'arm_q': 'Q1', 'arm_r': 'R1', 'arm_s': 'S1', 'arm_k': 'K2', 'arm_90': 'N20', 'placement_target': 'MC', 'enforce': False, 'corrections': False}
PLACE_TOUCHDOWN_UNDERSHOOT = 0.015   # m (identical to MG)
PLACE_RETRIES_TOTAL = 2              # bounded corrective actions per placement (module F)
_MF = {"contract": None, "last_plan": None, "rest_ref": None, "rest_src": None, "target_xy": None, "prof": None}
def _mfj(o):
    return json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
def mf_frame(obs):
    cam = obs["agentview"]; depth = cam["images"]["depth"]; _p_eef, _ = tcp_from_obs(obs)
    return {"rgb": cam["images"]["rgb"], "depth": depth[:, :, 0] if len(depth.shape) == 3 else depth, "K": cam["intrinsics"], "E": cam["pose_mat"], "cam_xy": np.asarray(cam["pose_mat"])[:2, 3], "robot": {"eef": np.asarray(_p_eef, float)}}
def mf_make_contract(relation, support_pts, obj_pts, prompts, E, region_dir=None, others_xy=(), label="", binding_base_z=None, target_xy=None, support_geom=None):
    sup = pv_support_from_points(np.asarray(support_pts, float)); prof = gg_object_profile(np.asarray(obj_pts, float), np.asarray(E)[:2, 3]); _MF["prof"] = prof
    r_obj = float(max(prof["r_top"], prof["r_mid"])) if prof else 0.04
    if r_obj > MF_OBJECT_R_MAX:   # C4I4 (verifier plumbing): the shared verifier accepts object radii up to MF_OBJECT_R_MAX and raises beyond it; the shadow contract gets the clamped radius instead of ending the episode
        print("MF_CONTRACT_CLAMP " + _mfj({"label": label, "object_r": r_obj, "object_r_contract": MF_OBJECT_R_MAX}), flush=True)
    v_rel = relation
    if relation in ("in_support", "in_part_support", "in_sliding_support"):   # C1I2 (C3I3: the bay likewise; C6I3: the sliding part's interior likewise): the shared verifier has no containment form; "inside the rim, on the interior floor" is its on-support form on the container's interior geometry
        if support_geom is not None:
            sup = {**sup, "center_xy": np.asarray(support_geom["center_xy"], float), "r": float(support_geom["r"]), "rest_z": float(support_geom["rest_z"]), "table_z": float(support_geom["table_z"])}
        v_rel = "on_support"
    c = pv_make_contract(v_rel, sup, list(prompts), min(r_obj, MF_OBJECT_R_MAX), prof["height"] if prof else None, region_dir=(None if relation in ("in_part_support", "in_sliding_support") else region_dir), others_xy=others_xy, label=label)
    c["placement_relation"] = relation
    _MF["contract"] = c
    _MF["rest_ref"], _MF["rest_src"] = fg_rest_reference(v_rel, sup, binding_base_z)
    _MF["target_xy"] = np.asarray(sup["center_xy"], float) if target_xy is None else np.asarray(target_xy, float)
    print("MF_CONTRACT " + _mfj({"label": label, "relation": relation, "verifier_relation": v_rel, "support": {"center_xy": sup["center_xy"], "r": sup["r"], "rest_z": sup["rest_z"], "table_z": sup["table_z"], "fit_ok": sup["fit_ok"]}, "object_r": r_obj, "object_h": prof["height"] if prof else None,
                                 "region": c.get("region"), "rest_ref": _MF["rest_ref"], "rest_src": _MF["rest_src"], "binding_base_z": binding_base_z, "target_xy": _MF["target_xy"]}), flush=True)
    return c
# ===== C3I4 grasp_geometry: closing axis from the top band's rectangle (round 20260924, chunk 3, iteration 4) =====
# Module B evaluates two pad closing axes only, the base axes (yaw 0 closes along base y, yaw 90 along base x). A body standing on
# its edge (a slab: one thin horizontal axis) at any other yaw shows both of them an oblique section: wider than the finger span
# (mode infeasible; the fallback descents land on the body: premature_contact) or, within ~20 deg of the thin axis, narrow enough
# to be planned but pinched across two opposite edges (recorded: closed 3.5-3.6 cm on a 2.9 cm slab, hold failed 3 of 4; closed
# 2.9 cm when the axis matched the thin axis, hold verified 2 of 2). Rule:
#   thin axis = the minor axis of the minimum-area rectangle of the object's top band (the points within GA_TOP_BAND of the
#   CONTAINER_PCT_HI top: the section seen whole from above; the C3I3 object_footprint_halves statistic, _ca_min_rect, 1 deg grid);
#   psi = the yaw whose pad closing axis (tool y of make_topdown_quat(psi) = (sin psi, -cos psi)) is that axis, in (-90, 90];
#   the rect-frame plan = module B on the points (and camera xy) rotated about the vertical through the rectangle centre by -psi,
#   single candidate yaw 0, rotated back: module B's own profile / body-pinch height (G2 depth, H palm clearance) / contact
#   prediction, evaluated along the body's own axis (module B is rotation-equivariant apart from its fixed axis set);
#   it replaces module B's plan only (a) when module B is infeasible and the rect frame yields a body pinch, or (b) when module B's
#   body pinch closes across a section wider than the rect frame's by more than GA_WIDTH_MARGIN (2 x PERCEPTION xy_sigma: two
#   widths of one section, each a 3-97 percentile extent with a 1 cm sigma per edge, differ by less). Every other plan (rim
#   pinches, body pinches along the thin axis, round sections whose rectangle is square) is module B's, the same object.
GA_TOP_BAND = 0.02               # m, = CP_TOP_BAND: the profile's top band (gg_object_profile: top - 0.02 .. top), as object_footprint_halves (a literal: CP_TOP_BAND is defined later in the file)
GA_MIN_PTS = 40                  # = CONTAINER_MIN_PTS: points the band needs (as object_footprint_halves; a literal for the same reason)
GA_WIDTH_MARGIN = 0.02           # m, 2 x grasp_geometry PERCEPTION xy_sigma (0.010)


def _ga_rot(xy, c, deg):
    """xy rotated by deg about the vertical through c."""
    t = np.radians(float(deg)); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (np.asarray(xy, float) - np.asarray(c, float)) @ R.T + np.asarray(c, float)


def ga_rect_axis(obj_pts):
    """The top band's minimum-area rectangle and the yaw psi whose pads close along its minor axis; None without enough band points."""
    P = np.asarray(obj_pts, float).reshape(-1, 3)
    if len(P) < GA_MIN_PTS:
        return None
    top = float(np.percentile(P[:, 2], CONTAINER_PCT_HI)); band = P[P[:, 2] >= top - GA_TOP_BAND]
    if len(band) < GA_MIN_PTS:
        return None
    c, a, b, hu, hv = _ca_min_rect(band[:, :2])
    psi = float(np.degrees(np.arctan2(b[0], -b[1])))
    psi = float(round(((psi + 90.0) % 180.0) - 90.0))
    if psi <= -90.0:
        psi += 180.0
    return {"centre_xy": [float(c[0]), float(c[1])], "minor_axis": [float(b[0]), float(b[1])], "half_major": float(hu), "half_minor": float(hv), "psi": psi, "n_band": int(len(band))}


def _ga_pinch_width(plan):
    """module B's width across the closing axis of the chosen body-pinch candidate (None for other modes)."""
    if plan.get("mode") != "body_pinch":
        return None
    y0 = (plan.get("yaw_order") or [None])[0]
    cs = [c for c in (plan.get("candidates") or []) if c.get("mode") == "body_pinch" and c.get("yaw") == y0]
    return float(cs[0]["width_across"]) if cs else None


def ga_plan_grasp(obj_pts, cam_xy):
    """Module B's plan, or its rect-frame plan along the body's thin axis (rule above). Logs GRASP_AXIS."""
    P = np.asarray(obj_pts, float); plan = gg_plan_grasp(P, cam_xy)
    rec = {"old_mode": plan.get("mode"), "old_width": _ga_pinch_width(plan), "decision": "module_B"}
    try:
        ax = ga_rect_axis(P)
        if ax is None or plan.get("mode") not in ("infeasible", "body_pinch"):
            rec["why"] = "no_band" if ax is None else "mode_kept"; print("GRASP_AXIS " + _mfj(rec), flush=True); return plan
        psi = ax["psi"]; c = np.asarray(ax["centre_xy"], float); rec.update(rect=ax)
        Pr = P.copy(); Pr[:, :2] = _ga_rot(P[:, :2], c, -psi)
        cam_r = None if cam_xy is None else _ga_rot(np.asarray(cam_xy, float)[:2], c, -psi)
        pr = gg_plan_grasp(Pr, cam_r, yaws=(0,))
        w_new = _ga_pinch_width(pr); rec.update(rect_mode=pr.get("mode"), rect_expected=pr.get("expected"), rect_width=w_new)
        if pr.get("mode") != "body_pinch" or pr.get("target") is None or pr.get("expected") == "premature_contact":
            rec["why"] = "rect_frame_not_a_body_pinch"; print("GRASP_AXIS " + _mfj(rec), flush=True); return plan
        if plan.get("mode") == "body_pinch":
            w_old = rec["old_width"]
            if w_old is None or w_new is None or not (w_old - w_new > GA_WIDTH_MARGIN):
                rec["why"] = "module_B_pinch_within_margin"; print("GRASP_AXIS " + _mfj(rec), flush=True); return plan
        t = np.asarray(pr["target"], float); txy = _ga_rot(t[:2], c, psi)
        cands = []
        for cd in (pr.get("candidates") or []):
            cd = dict(cd); cd["yaw"] = psi
            if cd.get("tcp_xy") is not None:
                cd["tcp_xy"] = _ga_rot(np.asarray(cd["tcp_xy"], float)[:2], c, psi)
            if cd.get("closing_dir") is not None:
                cd["closing_dir"] = _ga_rot(np.asarray(cd["closing_dir"], float)[:2], np.zeros(2), psi).tolist()
            cands.append(cd)
        new = dict(pr)
        ps = dict(pr.get("profile_summary") or {})
        if ps.get("axis_xy") is not None:
            ps["axis_xy"] = _ga_rot(np.asarray(ps["axis_xy"], float), c, psi).tolist()
        new.update(target=np.array([float(txy[0]), float(txy[1]), float(t[2])]), yaw_order=[psi], candidates=cands, profile_summary=ps, profile=plan.get("profile"),
                   rule=str(pr.get("rule")) + "+rect_axis", rect_axis=ax, plan_module_B={"mode": plan.get("mode"), "yaw_order": plan.get("yaw_order"), "target": plan.get("target"), "width": rec["old_width"]})
        rec.update(decision="rect_axis", why=("module_B_infeasible" if plan.get("mode") == "infeasible" else "module_B_oblique_pinch"), target=new["target"], yaw=psi)
        print("GRASP_AXIS " + _mfj(rec), flush=True)
        return new
    except Exception as e:  # noqa: BLE001  (a rect-frame failure never changes module B's plan)
        rec.update(decision="module_B", why=f"rect_frame_error: {type(e).__name__}: {e}"); print("GRASP_AXIS " + _mfj(rec), flush=True); return plan
# ===== end C3I4 grasp_geometry =====


# ===== C4I4 grasp_geometry: the grasp of a wide hollow body is planned on its own body (round 20260924, chunk 4, iteration 4) =====
# Module B (+ C3I4) plans on every point of the bound mask and pinches either a whole section (<= the finger span) or, for a body it
# recognises as hollow and round (top-band circle of radius <= 0.09 m, ring fraction >= 0.6), its rim wall. A wide hollow body with a handle
# fails both: its lip (radius ~0.10 m) is beyond the circle range and the handle, higher than the lip, fills the top band, so module B
# is infeasible on the body itself; and when the mask also covers a neighbour's part that the image shows touching the handle (in 3D
# 7-8 cm away), module B plans on that part (the top band of the union) and the fingers meet the neighbour (recorded: finger vs the
# neighbour's body at the first descent). Rule (gw_plan_grasp, wrapping ga_plan_grasp, which is always called first and unchanged):
#   main body = the largest 3D-connected component of the bound points (GW_CELL occupancy, 26-connectivity: the ca_parts clustering);
#   the wrapper acts only when ga_plan_grasp is infeasible or its target lies farther than GW_DETACH from every main-body point (a
#   pinch of a body the clustering separates from the object); otherwise its plan is returned unchanged. A mask whose two largest
#   components are within LOW_SIZE_RATIO of each other (two comparable bodies: the binding's look-alike size band) keeps its plan.
#   Acting: (a) for a plan on a detached part: module B on the main body alone, taken when feasible and on the main body; else
#   (b) the main body as a wide hollow round body: outline = farthest point per GW_BINS angular bin about the centre (first pass about
#   the centroid without the bins beyond GW_REACH_FRAC x the median reach: a handle / spout), trimmed circle fit (GW_RING_TOL, three
#   passes), accepted with module B's circle / hollow criteria (hollow: interior points or the far inner wall, object_profile's test
#   with the rim as the top); candidates at the direction d of every outline bin whose rim point lies on the fitted circle (the wall
#   was seen there at the pinch radius: not across the handle, an occlusion or a flat side): TCP on the wall at c + d (r -
#   GW_RIM_INSET), pads closing along d, height = max(module B's rim height, module B's cavity bound, the H palm bound of the points
#   under the hand body, the public IK floor + the contact overshoot); valid when module B's finger-contact model predicts no
#   premature contact, the pads overlap the wall by GW_ENGAGE_MIN at the height where the descent is predicted to stop, and the open
#   hand swept down the approach column is CA_CLEAR_MARGIN clear of every other raised point of the observe view (the hand body also
#   of the object's own points); ranked by that clearance, first with an IK solution for the grasp and approach poses. No valid
#   candidate -> the original plan. DIAGNOSIS.md sections 3-4.
MF_OBJECT_R_MAX = 0.15           # m, placement_verifier.make_contract's accepted object radius (its ValueError bound); the shadow contract's radius is clamped to it
GW_CELL = 0.02                   # m, = CA_CLUSTER_CELL (3D clustering cell of ca_parts; a literal: CA_CLUSTER_CELL is defined later in the file)
GW_DETACH = 0.042                # m, (max_opening - 2 finger_clear) / 2 + xy_sigma: the farthest a pinched section's surface lies from its pinch target
GW_BINS = 36                     # angular bins of the outline (10 deg: one pad width on a 11.5 cm rim); the candidate directions are the bin centres
GW_REACH_FRAC = 1.2              # first pass only: outline bins beyond 1.2 x the median reach about the centroid are a protrusion
GW_RING_TOL = 0.01               # m, module B's circle inlier band (slice_geometry)
GW_RING_RMS = 0.006              # m, module B's circle rms bound
GW_RING_FRAC = 0.6               # module B's ring fraction, over the occupied outline bins
GW_R_MIN = 0.032                 # m, (max_opening - 2 finger_clear) / 2: a narrower round section is module B's body pinch
GW_RIM_INSET = 0.003             # m, module B's rim-pinch TCP radius r_top - 0.003
GW_ENGAGE_MIN = 0.009            # m, pad_engage / 2: pads overlapping the wall at least this much
GW_APPROACH_UP = 0.12            # m, attempt_grasp_verify's approach pose above the grasp point
GW_G = {"max_opening": 0.080, "pad_engage": 0.018, "finger_thick": 0.010, "finger_width": 0.020, "tcp_to_fingertip": 0.005, "xy_sigma": 0.010,
        "contact_tol_before": 0.03, "rim_below": 0.002, "cavity_clear": 0.010, "inner_frac": 0.7, "inner_below": 0.015, "inner_min": 20}   # grasp_geometry GRIPPER / PERCEPTION / plan_grasp constants, verbatim
GW_H = {"hand_clear_above_tcp": 0.0335, "palm_clear": 0.005, "overshoot": 0.010}   # grasp_geometry_h palm rule, verbatim


def gw_components(P):
    """Component label of every point (GW_CELL occupancy grid, 26-connectivity), the largest label and the component sizes."""
    g = np.floor((P - P.min(0)) / GW_CELL).astype(int); occ = np.zeros(tuple(int(v) + 1 for v in g.max(0)), bool); occ[g[:, 0], g[:, 1], g[:, 2]] = True
    lab, n = _ndi.label(occ, structure=np.ones((3, 3, 3), bool)); L = lab[g[:, 0], g[:, 1], g[:, 2]]
    cnt = np.bincount(L, minlength=n + 1); k = int(np.argmax(cnt[1:])) + 1
    return L, k, sorted((int(c) for c in cnt[1:]), reverse=True)


def gw_xy_gap(xy, M):
    """xy distance from a point to the nearest point of M."""
    return float(np.min(np.hypot(M[:, 0] - float(xy[0]), M[:, 1] - float(xy[1]))))


def _gw_outline(xy, c):
    """Farthest point per GW_BINS angular bin about c: (indices into xy, reach, bin numbers)."""
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]); rad = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1])
    b = ((ang + np.pi) / (2 * np.pi) * GW_BINS).astype(int) % GW_BINS; idx = []; bins = []
    for k in range(GW_BINS):
        sel = np.flatnonzero(b == k)
        if len(sel):
            idx.append(int(sel[np.argmax(rad[sel])])); bins.append(k)
    idx = np.asarray(idx, int); return idx, rad[idx], bins


def gw_ring(M, cam_xy=None):
    """Rim of a round hollow body from its outline (rule above). Hollow = module B's test (grasp_geometry.object_profile) with the
    rim as the top: >= 20 points inside 0.7 r and >= 1.5 cm below the rim, or the far inner wall visible (>= 30 points of the band
    between 30 % of the height and 1.5 cm below the rim lie > 100 deg from the camera direction, >= 15 % of that band). Returns (ring | None, info)."""
    xy = M[:, :2]; c = xy.mean(0); r = None
    for _ in range(4):
        idx, reach, _ = _gw_outline(xy, c)
        keep = reach <= GW_REACH_FRAC * float(np.median(reach)) if r is None else np.abs(np.hypot(xy[idx, 0] - c[0], xy[idx, 1] - c[1]) - r) <= GW_RING_TOL
        if int(keep.sum()) < 8:
            return None, {"why": "too_few_outline_points"}
        f = fit_circle_xy(xy[idx][keep])
        if f is None:
            return None, {"why": "no_circle"}
        c = np.array([f[0], f[1]]); r = float(f[2])
    idx, _, bins = _gw_outline(xy, c); res = np.abs(np.hypot(xy[idx, 0] - c[0], xy[idx, 1] - c[1]) - r); keep = res <= GW_RING_TOL
    rms = float(np.sqrt(np.mean(res[keep] ** 2))) if keep.any() else None
    info = {"centre_xy": [round(float(c[0]), 4), round(float(c[1]), 4)], "r": round(r, 4), "rms": None if rms is None else round(rms, 4), "n_bins": int(len(idx)), "n_inliers": int(keep.sum())}
    if rms is None or rms > GW_RING_RMS or keep.sum() < GW_RING_FRAC * len(idx):
        return None, {**info, "why": "not_round"}
    if r < GW_R_MIN:
        return None, {**info, "why": "narrower_than_the_finger_span"}
    z_rim = float(np.median(M[idx[keep], 2])); base = float(np.percentile(M[:, 2], 2))
    rr = np.hypot(M[:, 0] - c[0], M[:, 1] - c[1])
    n_in = int(np.sum((rr < GW_G["inner_frac"] * r) & (M[:, 2] < z_rim - GW_G["inner_below"]) & (M[:, 2] > base + 0.005)))
    below = (M[:, 2] < z_rim - GW_G["inner_below"]) & (M[:, 2] > base + 0.3 * max(z_rim - base, 1e-3)); n_far = 0; far_frac = 0.0
    if cam_xy is not None and int(below.sum()) >= 30:
        a_cam = np.arctan2(float(cam_xy[1]) - c[1], float(cam_xy[0]) - c[0]); a_pts = np.arctan2(M[below, 1] - c[1], M[below, 0] - c[0])
        far = np.abs(np.angle(np.exp(1j * (a_pts - a_cam)))) > np.radians(100.0); n_far = int(far.sum()); far_frac = float(far.mean())
    info.update(z_rim=round(z_rim, 4), base=round(base, 4), interior_pts=n_in, far_pts=n_far, far_frac=round(far_frac, 3))
    if n_in < GW_G["inner_min"] and not (n_far >= 30 and far_frac >= 0.15):
        return None, {**info, "why": "not_hollow"}
    return {"c": c, "r": r, "z_rim": z_rim, "base": base, "bins": [k for k, ok in zip(bins, keep) if ok]}, info   # the bins where the rim was seen at the fitted radius: the only pinch directions


def gw_contact(P, tcp_xy, d, z_target):
    """Module B's first-contact prediction of the open fingers (grasp_geometry.predict_contact + expected_status, verbatim)."""
    g = GW_G; P = np.asarray(P, float); d = np.asarray(d, float) / float(np.linalg.norm(d)); n = np.array([-d[1], d[0]])
    z_ft = float(z_target) - g["tcp_to_fingertip"] + 0.002; inner = g["max_opening"] / 2.0; outer = inner + g["finger_thick"]; zs = []
    for s in [np.zeros(2)] + [g["xy_sigma"] * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 9)[:-1]]:
        rel = P[:, :2] - (np.asarray(tcp_xy, float) + s); al = rel @ d; ac = rel @ n
        m = (np.abs(ac) <= g["finger_width"] / 2.0) & (np.abs(al) >= inner - 0.002) & (np.abs(al) <= outer) & (P[:, 2] > z_ft)
        zs.append(float(P[m, 2].max()) + g["tcp_to_fingertip"] if m.any() else None)
    hit = [v for v in zs if v is not None]; p = len(hit) / len(zs); zc = zs[0]
    if zc is None and p <= 0.5:
        st = "reached"
    else:
        st = "contact_established" if (zc if zc is not None else max(hit)) <= float(z_target) + g["contact_tol_before"] else "premature_contact"
    return {"predicted_contact_tcp_z": zc, "range": [min(hit), max(hit)] if hit else None, "p_contact": p, "expected": st}


def gw_hand_boxes():
    """The open hand in tool coordinates (x = pad closing axis, z toward the tips): two fingers at the full opening (each as thick
    as the closed pair's half width, CA_HAND) and the hand body. Returns (finger boxes, body boxes)."""
    H = CA_HAND; w = GW_G["max_opening"] / 2.0; t = H["finger_half_wide"]; fy = (-H["finger_half_thin"], H["finger_half_thin"]); fz = (H["finger_base"], H["tip"])
    return [((w, w + t), fy, fz), ((-w - t, -w), fy, fz)], [((-H["body_half_wide"], H["body_half_wide"]), (-H["body_half_thin"], H["body_half_thin"]), (H["body_top"], H["body_bottom"]))]


def gw_hand_R(d):
    """Rotation (columns = tool axes) of a top-down hand whose pads close along the horizontal d."""
    x = np.array([float(d[0]), float(d[1]), 0.0]); x /= float(np.linalg.norm(x)); z = np.array([0.0, 0.0, -1.0]); return np.stack([x, np.cross(z, x), z], 1)


def gw_box_gap(P, tcp, R, boxes):
    """Smallest distance from the points to the boxes of a hand at (tcp, R) (negative inside); 9.0 without points."""
    if P is None or len(P) == 0:
        return 9.0
    L = (np.asarray(P, float) - np.asarray(tcp, float)) @ R; best = 9.0
    for (x0, x1), (y0, y1), (z0, z1) in boxes:
        c = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0]); s = np.array([(x1 - x0) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0])
        q = np.abs(L - c) - s; dd = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0); best = min(best, float(dd.min()))
    return best


def gw_obstacles(M):
    """Raised points of the observe view (the last stashed view: run_subgoal's observe frame) that are not the main body, and the
    floor height; (None, None) without a view or a floor."""
    if not _C_VIEWS:
        return None, None
    tcp_xy, obs = _C_VIEWS[-1]; cam = obs["agentview"]; dep = cam["images"]["depth"]; dep = dep[:, :, 0] if len(dep.shape) == 3 else dep
    floor = _CG.get("floor_z")
    if floor is None:
        floor = support_floor_z(dep, cam["intrinsics"], cam["pose_mat"])
    if floor is None:
        return None, None
    S = ca_frame_raised(dep, cam["intrinsics"], cam["pose_mat"], floor, tcp_xy)
    return S[~ca_near(S, M, CA_SAME_POINT)], float(floor)


# ===== C11I5 grasp_geometry: the out-of-box rim pinch is re-planned around the rim (C11I3's rule, narrowed; N11, slot C11I5) =====
# g1_descent_feasible refuses, before any motion, a grasp whose TCP target or contact-overshoot end (target - CONTACT_OVERSHOOT in
# z) lies outside the public IK box [_G1_WS_LO, _G1_WS_HI] (1e-6 floating tolerance). gb_box_ok is that test without logging.
# gc_column_replan re-plans module B's rim pinch with gw_rim_plan when module B's target fails it (as for an occupied finger
# column), and only in that case asks gw_rim_plan to admit box-feasible rim candidates only (box=True): the ring planner's IK
# probe is solve_ik, which clamps into the box and cannot see it. With box=False (every existing caller) gw_rim_plan is unchanged.
# No new constant. DIAGNOSIS.md section 3.2.
def gb_box_ok(target):
    """True when the TCP target and its contact-overshoot end lie inside the public IK box (g1_descent_feasible's box test)."""
    if target is None:
        return False
    t = np.asarray(target, float).reshape(-1)[:3]; end = t.copy(); end[2] -= CONTACT_OVERSHOOT
    lo = np.asarray(_G1_WS_LO, float) - 1e-6; hi = np.asarray(_G1_WS_HI, float) + 1e-6
    return bool(np.all(np.isfinite(t)) and all(np.all(p >= lo) and np.all(p <= hi) for p in (t, end)))
# ===== end C11I5 grasp_geometry helper =====


def gw_rim_plan(M, obst, ik_probe, cam_xy=None, box=False):   # C11I5: box=True (the out-of-box re-plan only) admits box-feasible rim candidates only
    """Rim pinch of a wide hollow round main body (rule above). Returns (plan | None, info)."""
    ring, info = gw_ring(M, cam_xy)
    if ring is None:
        return None, info
    c = ring["c"]; r = ring["r"]; z_rim = ring["z_rim"]; g = GW_G; fb, bb = gw_hand_boxes(); rows = []
    z_min = float(_G1_WS_LO[2]) + CONTACT_OVERSHOOT + 1e-4   # the contact descent's overshoot end must stay inside the public IK box (g1_descent_feasible)
    for k in ring["bins"]:   # a pinch only where the outline shows the wall at the fitted radius (+- GW_RING_TOL): not across a protrusion, an occlusion or a flat side
        th = -np.pi + (k + 0.5) * 2 * np.pi / GW_BINS; d = np.array([np.cos(th), np.sin(th)]); t = c + d * (r - GW_RIM_INSET); R = gw_hand_R(d)
        foot = (M - np.array([t[0], t[1], 0.0])) @ R   # own points under the hand body (its xy footprint)
        under = M[(np.abs(foot[:, 0]) <= CA_HAND["body_half_wide"]) & (np.abs(foot[:, 1]) <= CA_HAND["body_half_thin"])]
        z_palm = (float(under[:, 2].max()) - GW_H["hand_clear_above_tcp"] + GW_H["palm_clear"] + GW_H["overshoot"]) if len(under) else -9.0
        z = max(z_rim - g["pad_engage"] - g["rim_below"], ring["base"] + g["cavity_clear"] + g["tcp_to_fingertip"], z_palm, z_min)
        pred = gw_contact(M, t, d, z)
        zc = pred["predicted_contact_tcp_z"] if pred["predicted_contact_tcp_z"] is not None else (pred["range"][1] if pred["p_contact"] > 0.5 else None)
        engaged = z_rim - (max(z, zc if zc is not None else z) - g["tcp_to_fingertip"])   # pad overlap with the wall where the descent is predicted to stop (a finger landing inside the rim stops it early)
        zs = list(np.arange(z + GW_APPROACH_UP, z, -CA_DESCENT_STEP)) + [z]
        gap_obst = min(gw_box_gap(obst, [t[0], t[1], zz], R, fb + bb) for zz in zs)
        gap_own = min(gw_box_gap(M, [t[0], t[1], zz], R, bb) for zz in zs)
        psi = int(round(float(np.degrees(th)) + 90.0)); psi = ((psi + 90) % 180) - 90
        psi = 90 if psi == -90 else psi
        ok = pred["expected"] != "premature_contact" and engaged >= GW_ENGAGE_MIN and gap_obst >= CA_CLEAR_MARGIN and gap_own >= CA_CLEAR_MARGIN
        _inb = gb_box_ok([round(float(t[0]), 4), round(float(t[1]), 4), round(float(z), 4)]) if box else None   # C11I5: the executor's box test on the target it would receive
        ok = ok and (_inb is not False)
        rows.append({"deg": round(float(np.degrees(th)), 1), "psi": psi, "tcp": [round(float(t[0]), 4), round(float(t[1]), 4), round(float(z), 4)], "expected": pred["expected"],
                     "p_contact": round(pred["p_contact"], 2), "engaged": round(float(engaged), 4), "gap_obst": round(gap_obst, 4), "gap_own": round(gap_own, 4), "ok": bool(ok), "_pred": pred, "_d": d})
        if box:
            rows[-1]["in_box"] = bool(_inb)   # C11I5
    ranked = sorted([x for x in rows if x["ok"]], key=lambda x: -x["gap_obst"]); chosen = None
    for x in ranked:
        q = make_topdown_quat(x["psi"]); tt = np.asarray(x["tcp"], float)
        x["ik"] = bool(ik_probe(tt, q) and ik_probe(tt + np.array([0.0, 0.0, GW_APPROACH_UP]), q))
        if x["ik"]:
            chosen = x; break
    info.update(n_candidates=len(rows), n_valid=len(ranked), candidates=[{k: v for k, v in x.items() if not k.startswith("_")} for x in rows])
    if chosen is None:
        info["why"] = "no_valid_candidate" if not ranked else "no_ik"; return None, info
    pred = chosen["_pred"]; z = float(chosen["tcp"][2]); base_est = ring["base"] - 0.01   # module B's base_est: base_p2 - mean base bias (0.0, 0.02)
    plan = {"mode": "rim_pinch", "target": np.asarray(chosen["tcp"], float), "yaw_order": [chosen["psi"]], "side": f"ring{chosen['deg']:+.0f}", "expected": pred["expected"],
            "predicted_contact_tcp_z": pred["predicted_contact_tcp_z"], "contact_range": pred["range"], "p_contact": pred["p_contact"], "hang_est": float(z - base_est),
            "hang_interval": [float(z - ring["base"]), float(z - (ring["base"] - 0.02))], "xy_sigma": g["xy_sigma"], "z_sigma": 0.004, "rule": "C4I4_wide_rim",
            "candidates": [{"mode": "rim_pinch", "yaw": x["psi"], "closing_dir": x["_d"].tolist(), "tcp_xy": np.asarray(x["tcp"][:2], float), "z": x["tcp"][2], "expected": x["expected"]} for x in ranked],
            "profile_summary": {"base_p2": ring["base"], "top": z_rim, "r_top": r, "hollow": True, "axis_xy": [float(c[0]), float(c[1])], "rim": {k: v for k, v in info.items() if k != "candidates"}}}
    info.update(chosen={k: v for k, v in chosen.items() if not k.startswith("_")})
    return plan, info


# ===== C6I2 grasp_geometry: a rim pinch whose finger column is occupied is re-planned around the rim (round 20260924, chunk 6, iteration 2) =====
# Module B pinches a round hollow body's wall at one fixed side of the rim (side +y: pads closing along base y) and checks the fingers
# against the object's own points only. When another body stands beside the rim there, the open hand meets it on the way down (recorded:
# the outer finger / the hand body stopped 2.8-5.8 cm above the target by the open part of the neighbouring fixture, the pads closing
# above the rim), while every other side of the same rim is free. Rule, for a module-B rim pinch only (ga_plan_grasp's output, unchanged):
#   finger column gap = the smallest distance from the observe view's raised points that are not the object's main body (gw_obstacles,
#   C4I4) to the two open fingers (gw_hand_boxes' finger boxes at the full opening: the part of the hand that goes down beside the wall
#   to below the rim) at the plan's target xy, pads closing along the plan's first yaw (tool y of make_topdown_quat(psi) =
#   (sin psi, -cos psi)), at every CA_DESCENT_STEP from GW_APPROACH_UP above the target down to the target (gw_rim_plan's column steps);
#   the hand-body box (an envelope of the housing, from 3.1 cm above the TCP) ranks replacements but does not veto module B (recorded
#   passing rim pinches overlap it by up to 1.04 cm without any robot contact);
#   gap >= GC_BLOCKED (0: no visible point inside the nominal fingers) -> module B's plan, unchanged;
#   gap < GC_BLOCKED -> gw_rim_plan on the main body (C4I4: rim directions where the wall was seen at the fitted radius, module B's
#   contact prediction and pad engagement, CA_CLEAR_MARGIN column clearance of the whole open hand from the other bodies and of the hand
#   body over its own points, ranked by clearance, first with IK for the grasp and approach poses); its plan when it has one, else module B's.
# Two comparable bodies in the mask (LOW_SIZE_RATIO, the C4I4 fail-closed rule), no view, or any error -> module B's plan.
GC_BLOCKED = 0.0                 # m, box-distance sign: a visible surface point inside a swept open finger is a predicted collision of the nominal plan


def gc_column_gap(plan, obst):
    """Smallest distance from the points obst to the two open fingers swept down the plan's approach column (gw_rim_plan's steps)."""
    t = np.asarray(plan["target"], float).reshape(-1); psi = float((plan.get("yaw_order") or [0])[0])
    d = np.array([np.sin(np.radians(psi)), -np.cos(np.radians(psi))]); R = gw_hand_R(d); fb, _ = gw_hand_boxes()
    zs = list(np.arange(float(t[2]) + GW_APPROACH_UP, float(t[2]), -CA_DESCENT_STEP)) + [float(t[2])]
    return float(min(gw_box_gap(obst, [float(t[0]), float(t[1]), zz], R, fb) for zz in zs))


def gc_column_replan(plan, M, sizes, cam_xy):
    """Module B's rim pinch, or the C4I4 ring plan on the main body M when module B's finger column is occupied (rule above). Logs GRASP_COLUMN."""
    rec = {"old_side": plan.get("side"), "old_target": plan.get("target"), "old_yaw": (plan.get("yaw_order") or [None])[0], "decision": "module_B"}
    try:
        if plan.get("target") is None or not plan.get("yaw_order"):
            rec["why"] = "no_target"; print("GRASP_COLUMN " + _mfj(rec), flush=True); return plan
        if len(sizes) > 1 and sizes[0] < LOW_SIZE_RATIO * sizes[1]:
            rec["why"] = "two_comparable_bodies"; print("GRASP_COLUMN " + _mfj(rec), flush=True); return plan
        obst, floor = gw_obstacles(M)
        if obst is None:
            rec["why"] = "no_view"; print("GRASP_COLUMN " + _mfj(rec), flush=True); return plan
        g = gc_column_gap(plan, obst); rec.update(gap=round(g, 4), n_obst=int(len(obst)), floor_z=floor)
        _inbox = gb_box_ok(plan["target"]); rec["in_box"] = _inbox   # C11I5: module B's target against the executor's box test
        _outside = g >= GC_BLOCKED and not _inbox   # C11I5: a clear column but a target the executor refuses before any motion
        if g >= GC_BLOCKED and _inbox:
            rec["why"] = "column_clear"; print("GRASP_COLUMN " + _mfj(rec), flush=True); return plan
        _why = "outside_ik_box" if _outside else "column_occupied"
        rp, info = gw_rim_plan(M, obst, mf_ik_probe, cam_xy, box=_outside); rec["rim"] = {k: v for k, v in info.items() if k != "candidates"}
        if rp is None:
            rec["why"] = _why + "_no_ring_plan: " + str(info.get("why")); print("GRASP_COLUMN " + _mfj(rec), flush=True); return plan
        rp = dict(rp); rp["rule"] = "C11I5_in_box_rim" if _outside else "C6I2_clear_column"; rp["plan_module_B"] = {"mode": plan.get("mode"), "side": plan.get("side"), "target": plan.get("target"), "yaw_order": plan.get("yaw_order"), "column_gap": round(g, 4)}
        rec.update(decision="ring_replan", why=_why, target=rp["target"], yaw=rp["yaw_order"][0], side=rp.get("side")); print("GRASP_COLUMN " + _mfj(rec), flush=True)
        return rp
    except Exception as e:  # noqa: BLE001  (a failure of the check never changes module B's plan)
        rec.update(decision="module_B", why=f"error: {type(e).__name__}: {e}"); print("GRASP_COLUMN " + _mfj(rec), flush=True); return plan
# ===== end C6I2 grasp_geometry =====


# ===== C12I2 grasp_geometry: a wide hollow body with a protrusion is pinched across the protrusion next to its rim (N12, slot C12I2) =====
# C4I4 pinches a wide hollow round main body (module B infeasible) at its rim wall, pads closing along the radius d. Recorded on the
# development rows (30 lifts): the hold was reported lost in 27; in all 27 the body was still between the pads after the lift
# (diagnosis-only truth) but had pivoted in the pinch by 7-36 deg about a horizontal axis close to the tangent (perpendicular to d)
# while the pads closed further on the rotating lip (gap 1.4-2.1 cm -> 0.8-1.3 cm); the pads hold a thin curved lip whose vertical
# overlap is bounded by the public IK floor. One of the 3 verified rim holds was lost in transport. When the same body was pinched
# across its protrusion (a bar the rim fit excludes), the pads closed on 2.6-2.9 cm of bar and the hold stayed verified in 4/4
# lifts (the gravity torque axis is then the closing axis, resisted by the pad patch, not by a lip line contact).
# Rule (gh_handle_plan, tried by gw_plan_grasp before gw_rim_plan on the same main body; every other path unchanged):
#   ring = gw_ring(main body) (C4I4: round, hollow; else no change);
#   protrusion = main-body points farther than r + GW_RING_TOL from the ring centre and above the lowest pad height of a top-down
#   pinch (the public IK floor + the contact overshoot, or module B's cavity bound, minus the TCP-to-fingertip offset; floor pixels
#   of the mask lie below it); its direction u = the mean direction of the points of the contiguous run of occupied GW_BINS angular
#   bins around the fullest bin; none (fewer than GA_MIN_PTS points in the run, or a run wider than the bins a bar of width
#   2 GW_R_MIN subtends at r + GW_RING_TOL plus one quantisation bin per side) -> no change;
#   sections across u at reach s = r + finger_width / 2 + xy_sigma (the pads' 2 cm along u clear the rim lip by the perception sigma),
#   then outward in steps of finger_width / 2 while every section from the rim on is a bar (the first section that is not one ends
#   the protrusion: a blob detached from the rim is never pinched); a section = run points within
#   finger_width / 2 of s along u and within max_opening / 2 across; valid when it has >= inner_min points, its width across u
#   <= 2 GW_R_MIN (module B's body-pinch span: max opening minus both finger clearances) and its height spread <= the finger length
#   below the hand body (GW_H hand_clear_above_tcp + tcp_to_fingertip: a taller section cannot be straddled), TCP = its centre, pads closing across u, height = max(section top -
#   pad_engage - rim_below, the H palm bound of the own points under the hand body, the public IK floor + the contact overshoot),
#   module B's contact prediction not premature, pad overlap >= GW_ENGAGE_MIN, the open hand swept down the column CA_CLEAR_MARGIN
#   clear of every other raised point of the observe view and the hand body of the own points, target in the public IK box, and
#   IK for the grasp and approach poses (gw_rim_plan's tests, verbatim); the first valid section (the nearest to the body: the
#   smallest lever arm and held-body offset) is the plan; none -> gw_rim_plan exactly as before.
# Every constant is an existing gripper / perception constant (GW_G, GW_H, GW_RING_TOL, GW_R_MIN, GW_ENGAGE_MIN, GW_BINS, GA_MIN_PTS,
# CA_CLEAR_MARGIN, CA_DESCENT_STEP, GW_APPROACH_UP). DIAGNOSIS.md sections 3-5.
def gh_protrusion(M, ring):
    """Points of the protrusion of a ring body and its unit direction u from the ring centre: (points | None, u | None, info)."""
    c = np.asarray(ring["c"], float); r = float(ring["r"]); g = GW_G
    z_low = max(float(_G1_WS_LO[2]) + CONTACT_OVERSHOOT, float(ring["base"]) + g["cavity_clear"] + g["tcp_to_fingertip"]) - g["tcp_to_fingertip"]
    rr = np.hypot(M[:, 0] - c[0], M[:, 1] - c[1]); Q = M[(rr > r + GW_RING_TOL) & (M[:, 2] > z_low)]
    info = {"n_out": int(len(Q)), "z_low": round(float(z_low), 4)}
    if len(Q) < GA_MIN_PTS:
        return None, None, {**info, "why": "no_protrusion"}
    ang = np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0]); b = ((ang + np.pi) / (2 * np.pi) * GW_BINS).astype(int) % GW_BINS
    cnt = np.bincount(b, minlength=GW_BINS); k0 = int(np.argmax(cnt)); run = {k0}
    for step in (1, -1):
        k = (k0 + step) % GW_BINS
        while cnt[k] > 0 and k not in run:
            run.add(k); k = (k + step) % GW_BINS
    S = Q[np.isin(b, sorted(run))]
    n_max = int(np.ceil(np.degrees(2.0 * np.arcsin(min(1.0, GW_R_MIN / (r + GW_RING_TOL)))) / (360.0 / GW_BINS))) + 2   # bins a bar of width <= 2 GW_R_MIN subtends at r + GW_RING_TOL, + one bin of quantisation per side
    info.update(n_run=int(len(S)), run_bins=len(run), run_bins_max=n_max)
    if len(S) < GA_MIN_PTS or len(run) > n_max:   # wider than a pinchable bar: a flat side, a second body or a mask spill, not a protrusion
        return None, None, {**info, "why": "no_compact_protrusion"}
    v = S[:, :2] - c; v = v / np.linalg.norm(v, axis=1, keepdims=True); u = v.mean(0); u = u / float(np.linalg.norm(u))
    info["u"] = [round(float(u[0]), 3), round(float(u[1]), 3)]
    return S, u, info


def gh_handle_plan(M, obst, ik_probe, cam_xy=None):
    """Pinch across the protrusion of a wide hollow round main body (rule above). Returns (plan | None, info)."""
    ring, rinfo = gw_ring(M, cam_xy)
    if ring is None:
        return None, {"why": "no_ring: " + str(rinfo.get("why"))}
    S, u, info = gh_protrusion(M, ring)
    if S is None:
        return None, info
    c = np.asarray(ring["c"], float); r = float(ring["r"]); g = GW_G; n = np.array([-u[1], u[0]]); fb, bb = gw_hand_boxes()
    al = (S[:, :2] - c) @ u; ac = (S[:, :2] - c) @ n; half = g["finger_width"] / 2.0
    z_min = float(_G1_WS_LO[2]) + CONTACT_OVERSHOOT + 1e-4
    s = r + half + g["xy_sigma"]; rows = []; chosen = None; R = gw_hand_R(n)
    while s + half <= float(al.max()) + 1e-9:
        sec = (np.abs(al - s) <= half) & (np.abs(ac) <= g["max_opening"] / 2.0); row = {"s": round(float(s), 4), "n": int(sec.sum())}; rows.append(row)
        s += half
        if row["n"] < g["inner_min"]:
            row["why"] = "few_points"; break   # the protrusion ends here: only sections joined to the rim by bar sections are pinched
        a_lo, a_hi = np.percentile(ac[sec], [2, 98]); z_lo, z_hi = np.percentile(S[sec, 2], [2, 98])
        row.update(width=round(float(a_hi - a_lo), 4), z_spread=round(float(z_hi - z_lo), 4), top=round(float(z_hi), 4))
        if a_hi - a_lo > 2 * GW_R_MIN or z_hi - z_lo > GW_H["hand_clear_above_tcp"] + g["tcp_to_fingertip"]:
            row["why"] = "not_a_bar"; break
        t = c + u * row["s"] + n * float((a_lo + a_hi) / 2.0)
        foot = (M - np.array([t[0], t[1], 0.0])) @ R
        under = M[(np.abs(foot[:, 0]) <= CA_HAND["body_half_wide"]) & (np.abs(foot[:, 1]) <= CA_HAND["body_half_thin"])]
        z_palm = (float(under[:, 2].max()) - GW_H["hand_clear_above_tcp"] + GW_H["palm_clear"] + GW_H["overshoot"]) if len(under) else -9.0
        z = max(float(z_hi) - g["pad_engage"] - g["rim_below"], z_palm, z_min)
        pred = gw_contact(M, t, n, z)
        zc = pred["predicted_contact_tcp_z"] if pred["predicted_contact_tcp_z"] is not None else (pred["range"][1] if pred["p_contact"] > 0.5 else None)
        engaged = float(z_hi) - (max(z, zc if zc is not None else z) - g["tcp_to_fingertip"])
        zs = list(np.arange(z + GW_APPROACH_UP, z, -CA_DESCENT_STEP)) + [z]
        gap_obst = min(gw_box_gap(obst, [t[0], t[1], zz], R, fb + bb) for zz in zs)
        gap_own = min(gw_box_gap(M, [t[0], t[1], zz], R, bb) for zz in zs)
        th = float(np.arctan2(n[1], n[0])); psi = int(round(float(np.degrees(th)) + 90.0)); psi = ((psi + 90) % 180) - 90; psi = 90 if psi == -90 else psi
        tcp = [round(float(t[0]), 4), round(float(t[1]), 4), round(float(z), 4)]
        row.update(tcp=tcp, psi=psi, expected=pred["expected"], engaged=round(float(engaged), 4), gap_obst=round(gap_obst, 4), gap_own=round(gap_own, 4))
        ok = pred["expected"] != "premature_contact" and engaged >= GW_ENGAGE_MIN and gap_obst >= CA_CLEAR_MARGIN and gap_own >= CA_CLEAR_MARGIN and gb_box_ok(tcp)
        if ok:
            q = make_topdown_quat(psi); tt = np.asarray(tcp, float)
            ok = bool(ik_probe(tt, q) and ik_probe(tt + np.array([0.0, 0.0, GW_APPROACH_UP]), q)); row["ik"] = ok
        row["ok"] = bool(ok)
        if ok:
            chosen = (row, pred, t, z); break
    info.update(r=round(r, 4), sections=rows)
    _tilt = None
    if chosen is None:   # C13I4: every bar section refused -> the sections refused only by the hand clearance, tilted (stage 1 vertical column, stage 2 axial approach; gt_tilt_section)
        try:
            _tilt, _tinfo = gt_tilt_section(rows, M, obst, ik_probe, n)
        except Exception as _e:  # noqa: BLE001
            _tilt, _tinfo = None, {"why": f"error: {type(_e).__name__}: {_e}"}
        info["tilt"] = _tinfo
        if _tilt is None:
            info["why"] = "no_valid_section"; return None, info
        _row = dict(_tilt[0]); _t = np.asarray(_row["tcp"][:2], float); _z = float(_row["tcp"][2]) + float(_tilt[2]["raise"])
        _row["tcp"] = [round(float(_t[0]), 4), round(float(_t[1]), 4), round(_z, 4)]; _z = _row["tcp"][2]
        chosen = (_row, gw_contact(M, _t, n, _z), _t, _z)
    row, pred, t, z = chosen; base_est = ring["base"] - 0.01   # gw_rim_plan's base estimate (module B's base_p2 - mean base bias)
    plan = {"mode": "handle_pinch", "target": np.asarray(row["tcp"], float), "yaw_order": [row["psi"]], "side": f"protrusion@{row['s']:.3f}", "expected": pred["expected"],
            "predicted_contact_tcp_z": pred["predicted_contact_tcp_z"], "contact_range": pred["range"], "p_contact": pred["p_contact"], "hang_est": float(z - base_est),
            "hang_interval": [float(z - ring["base"]), float(z - (ring["base"] - 0.02))], "xy_sigma": g["xy_sigma"], "z_sigma": 0.004, "rule": "C12I2_protrusion",
            "candidates": [{"mode": "handle_pinch", "yaw": row["psi"], "closing_dir": n.tolist(), "tcp_xy": np.asarray(row["tcp"][:2], float), "z": row["tcp"][2], "expected": pred["expected"]}],
            "profile_summary": {"base_p2": ring["base"], "top": ring["z_rim"], "r_top": r, "hollow": True, "axis_xy": [float(c[0]), float(c[1])], "protrusion_u": info.get("u"), "section": {k: v for k, v in row.items() if k not in ("tcp",)}}}
    if _tilt is not None:   # C13I4: the tilted pinch (ga_reachable_body_quat applies tilt_deg at the plan's target; an axial plan also sets the executor's approach)
        if _tilt[2]["axial"]:
            plan.update(rule="C13I4_protrusion_axial", tilt_deg=float(_tilt[1]), approach="axial", approach_axis=list(_tilt[2]["axis"]), raise_m=float(_tilt[2]["raise"]),
                        side=plan["side"] + f"/tilt{_tilt[1]:+.0f}/axial")
        else:
            plan.update(rule="C13I2_protrusion_tilt", tilt_deg=float(_tilt[1]), side=plan["side"] + f"/tilt{_tilt[1]:+.0f}")
    info["chosen"] = row
    return plan, info
# ===== end C12I2 grasp_geometry =====


# ===== C12I4 grasp_geometry / place_primitive: the held body's centre, not the pinch point, is placed on the target (N12, slot C12I4) =====
# A protrusion pinch (C12I2 block above) holds a wide body at its protrusion, one section reach s (~ r + finger_width/2 + xy_sigma)
# from the body's ring centre. Recorded on the development rows (20 protrusion-pinched rows, 3 tasks): the hold after the lift was
# verified and the VB transport gate allowed in 20/20, but compensate_xy discards both lift measurements (their offsets, 10.4-11.7 cm,
# exceed PRIM["held_offset_max"], a bound for centre-near grasps), so the TCP - the pinch point - was commanded to the target and the
# body centre landed 11.5-15.3 cm from it (diagnosis-only truth: relation false in 20/20). The visible-centroid offset itself is
# 2.6-3.8 cm wrong (the protrusion and the underside pull it toward the pads); the ring centre is not: the observe-view ring fit that
# planned the pinch gives the in-hand offset within 0.2-0.6 cm, and the same ring fit (gw_ring) on the held mask after the lift within
# 0.2-0.4 cm where it fits (15/20 rows).
# Rule (gh_held_offset; gh_compensate_xy replaces compensate_xy in t_select_transport_p3, every other path calls compensate_xy unchanged):
#   applies only when the plan of the held grasp (_MF["last_plan"]) is a protrusion pinch and the lift-pose TCP lies within
#   GW_G["xy_sigma"] of the plan's target (the executed grasp is that plan; a later strategy grasps elsewhere);
#   offset (base frame, lift orientation) = (i) the mean of ring centre - TCP over the lift measurements whose held points fit a ring
#   (gw_ring) of the plan's radius within GW_RING_TOL, when those agree within PRIM["held_agree_tol"]; else (ii) the plan's
#   ring centre - plan target, when every lift measurement's offset lies within the plan's ring radius of it (the visible held body
#   is on the planned body's footprint); else none -> compensate_xy unchanged;
#   the offset may not exceed the pinch's own reach (the section distance s + PRIM["held_agree_tol"]); it is rotated into the
#   gripper frame at the lift orientation and re-expressed at each candidate orientation exactly as compensate_xy does, and the TCP
#   is place_xy - offset: the held body's centre goes to the target.
# No new constant (GW_G["xy_sigma"], GW_RING_TOL, PRIM["held_agree_tol"], the plan's measured r_top and section s). No guard,
# margin, bound or budget changes: held_offset_max keeps its meaning on every centroid-based path. DIAGNOSIS.md sections 2 and 5.
def gh_held_offset(measurements):
    """In-hand offset of the body held by the current protrusion pinch: (offset_xy_base | None, info)."""
    plan = _MF.get("last_plan") or {}
    if plan.get("mode") != "handle_pinch":
        return None, {"applies": False}
    info = {"applies": True, "rule": "C12I4_held_body_centre"}
    try:
        prof = plan.get("profile_summary") or {}; sec = prof.get("section") or {}
        t = np.asarray(plan["target"], float).reshape(-1); c = np.asarray(prof["axis_xy"], float).reshape(-1)[:2]; r = float(prof["r_top"]); s = float(sec["s"])
        ms = [m for m in (measurements or []) if m is not None]
        info["n_meas"] = len(ms)
        if not ms:
            info["why"] = "no_held_measurement"; return None, info
        tcp = np.asarray(ms[-1]["tcp_p"], float)
        info["tcp_to_plan_cm"] = round(100.0 * float(np.hypot(*(tcp[:2] - t[:2]))), 2)
        if float(np.hypot(*(tcp[:2] - t[:2]))) > GW_G["xy_sigma"]:
            info["why"] = "held_grasp_not_the_plan"; return None, info
        off_plan = c - t[:2]; rings = []
        for m in ms:
            ring, _ = gw_ring(np.asarray(m["pts"], float))
            if ring is not None and abs(float(ring["r"]) - r) <= GW_RING_TOL:
                rings.append(np.asarray(ring["c"], float)[:2] - np.asarray(m["tcp_p"], float)[:2])
        info["n_ring"] = len(rings)
        off = None
        if rings:
            R_ = np.array(rings); spread = float(np.max(np.linalg.norm(R_ - R_.mean(0), axis=1)))
            if spread <= PRIM["held_agree_tol"]:
                off = R_.mean(0); info["how"] = "held_ring"
        if off is None:
            d = [float(np.hypot(*(np.asarray(m["offset_xy_base"], float)[:2] - off_plan))) for m in ms]
            info["meas_to_plan_cm"] = [round(100.0 * v, 2) for v in d]
            if max(d) <= r:
                off = off_plan; info["how"] = "planned_ring"
            else:
                info["why"] = "held_body_off_the_planned_body"; return None, info
        info["offset_cm"] = [round(100.0 * float(v), 2) for v in off]; info["plan_offset_cm"] = [round(100.0 * float(v), 2) for v in off_plan]
        if float(np.hypot(*off)) > s + PRIM["held_agree_tol"]:
            info["why"] = "offset_beyond_the_pinch_reach"; return None, info
        return off, info
    except Exception as e:  # noqa: BLE001  (any error: the parent's compensation)
        info["why"] = f"error: {type(e).__name__}: {e}"; return None, info


def gh_compensate_xy(place_xy, measurements, q_place):
    """compensate_xy, except that a protrusion-pinched body's centre (gh_held_offset) is placed on the target. Returns (place_xy_tcp, info)."""
    off, hinfo = gh_held_offset(measurements)
    if off is None:
        xy, info = compensate_xy(place_xy, measurements, q_place)
        return xy, (dict(info, held_centre=hinfo) if hinfo.get("applies") else info)
    ms = [m for m in measurements if m is not None]
    off_b = np.array([float(off[0]), float(off[1]), 0.0])
    off_g = _R(ms[-1]["tcp_q"]).T @ off_b                  # gripper-frame offset (lift orientation)
    off_p = _R(np.asarray(q_place, float)) @ off_g          # re-expressed at the placement orientation
    info = {"n_valid": len(ms), "applied": True, "held_centre": hinfo, "offset_base_lift_cm": (off_b[:2] * 100).round(2), "offset_base_place_cm": (off_p[:2] * 100).round(2)}
    return np.asarray(place_xy, float) - off_p[:2], info
# ===== end C12I4 =====


# ===== C13I4 grasp_geometry: a protrusion pinch refused by the hand clearance is tried with the hand pitched about the closing axis (C13I2 rule) and then on an axial approach (N13, slot C13I4) =====
# gh_handle_plan (C12I2) pinches a wide hollow body across its protrusion with a vertical hand. Recorded on the development rows: in 5 of
# 10 layouts every bar section was refused by gap_obst < CA_CLEAR_MARGIN alone (contact, engagement, own clearance and IK box passed);
# the conflicting points lay only inside the HAND-BODY box, 2-4 cm beyond the pinch along the protrusion, 6-11 cm to one side (a tall
# neighbouring body). The C4I4 rim pinch then ran and the body pivoted in the pads after the lift (the wall's flare) -> hold failed.
# Stage 1 (C13I2, verbatim): the hand pitched about the pad closing axis by GT_TILTS_DEG (ga_reachable_body_quat's bounded jaw-axis tilt)
#   on the parent's vertical approach column. Recorded (C13I2 development rows): fired in 4 rows, held and placed in 4/4.
# Stage 2 (C13I4, only when stage 1 finds nothing): the same pinch, the hand approaching and retreating along its own tilted tool axis.
#   Recorded geometry: in the layouts stage 1 refuses, the binding obstacle gap of the tilted hand is NOT at the grasp pose but 3-6 cm up
#   the vertical column (the hand body, raised straight up, meets the neighbour's widest part); moving back along the tilted tool axis
#   carries the hand body over the held body, away from the neighbour, so the axial column's binding level is the grasp pose itself.
#   Candidates: tilt b = +-1 .. +-GT_AXIAL_MAX_DEG deg in GT_AXIAL_STEP_DEG steps, TCP raise dz = 0, GT_RAISE_STEP, ... while the tilted
#   engagement (engaged(dz)) cos b - (finger_width / 2) sin b >= GW_ENGAGE_MIN; evaluated in order of that engagement, largest first (the
#   pinch closest to the parent's full-depth vertical pinch); the first candidate that passes is the plan:
#     axial column TCP - t a (a = tool axis toward the tips; t = GW_APPROACH_UP .. 0 in CA_DESCENT_STEP): fingers + hand body >= CA_CLEAR_MARGIN
#     from the other raised points, hand body and fingers >= CA_CLEAR_MARGIN from the own points; the retreat-to-lift segment (approach
#     pose -> the lift pose above the target, rr_lift_z) sampled at CA_DESCENT_STEP: fingers + hand body >= CA_CLEAR_MARGIN from the other
#     raised points; contact prediction at the raised TCP not premature; target in the IK box; IK at the approach pose, the grasp pose and
#     the contact-overshoot end along a.
#   The plan stays mode handle_pinch (C12I4's held-centre placement applies unchanged) with tilt_deg, approach "axial", approach_axis a
#   and raise_m dz; the executor (attempt_grasp_verify) approaches, descends (contact_move along a), backs off and - holding - first
#   retreats along a to the approach pose, then lifts exactly as before. Every other plan and call is the parent's.
# Constants: GT_TILTS_DEG = ga_reachable_body_quat's angles; GT_AXIAL_MAX_DEG = 30 = CA_PUSH_TILTS_DEG's smaller tool tilt (the program's
# commanded hand tilts span 0..45 deg); GT_AXIAL_STEP_DEG = 1 = the yaw quantum of make_topdown_quat's integer psi; GT_RAISE_STEP =
# GW_G["rim_below"] (the parent's own offset of the pinch TCP under the bar's top band). CA_CLEAR_MARGIN, GW_ENGAGE_MIN, GW_G finger_width,
# CA_DESCENT_STEP, GW_APPROACH_UP, CONTACT_OVERSHOOT unchanged. DIAGNOSIS.md sections 2 and 5.
GT_TILTS_DEG = (-15.0, 15.0)     # deg, ga_reachable_body_quat's bounded jaw-axis tilt (both signs), verbatim (stage 1)
GT_AXIAL_MAX_DEG = 30            # deg, CA_PUSH_TILTS_DEG's smaller tilt (stage 2 upper bound)
GT_AXIAL_STEP_DEG = 1            # deg, make_topdown_quat's integer-degree yaw quantum (stage 2 tilt step)
GT_RAISE_STEP = GW_G["rim_below"]   # m, the parent's pinch-TCP offset under the bar's top band (stage 2 raise step)


def gt_tilt_quat(psi, deg):
    """wxyz commanded quaternion of a top-down hand at yaw psi pitched by deg about the pad closing axis (tool y)."""
    q = make_topdown_quat(psi); xyzw = (Rotation.from_quat(np.asarray(q, float)[[1, 2, 3, 0]]) * Rotation.from_euler("y", float(deg), degrees=True)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def gt_only_clearance(row):
    """True when the vertical loop refused this bar section ONLY for the hand clearance gap_obst."""
    return row.get("tcp") is not None and row.get("expected") != "premature_contact" and float(row["engaged"]) >= GW_ENGAGE_MIN and float(row["gap_own"]) >= CA_CLEAR_MARGIN and gb_box_ok(row["tcp"]) and float(row["gap_obst"]) < CA_CLEAR_MARGIN


def gt_axial_candidates(engaged):
    """Stage-2 (engagement, tilt_deg, raise) candidates, largest tilted engagement first (rule above)."""
    half = GW_G["finger_width"] / 2.0; out = []
    for deg in range(GT_AXIAL_STEP_DEG, GT_AXIAL_MAX_DEG + 1, GT_AXIAL_STEP_DEG):
        b = np.radians(float(deg)); k = 0
        while True:
            dz = k * GT_RAISE_STEP; eng = (float(engaged) - dz) * float(np.cos(b)) - half * float(np.sin(b))
            if eng < GW_ENGAGE_MIN:
                break
            out.extend([(eng, float(deg), dz), (eng, -float(deg), dz)]); k += 1
    return sorted(out, key=lambda x: (-x[0], x[2], abs(x[1]), -x[1]))


def gt_axial_section(row, M, obst, ik_probe, n, tried):
    """Stage 2 on one bar section: (deg, dz, q, a, rec) | None."""
    fb, bb = gw_hand_boxes(); half = GW_G["finger_width"] / 2.0; t0 = np.asarray(row["tcp"], float); z_hi = float(row["top"])
    for _, deg, dz in gt_axial_candidates(row["engaged"]):
        tcp = t0 + np.array([0.0, 0.0, dz]); q = gt_tilt_quat(row["psi"], deg); Rt = _R(q); Rm = Rt @ OP_TOOL_FROM_MODEL.T; a = Rt[:, 2] / float(np.linalg.norm(Rt[:, 2]))
        if gw_box_gap(obst, tcp, Rm, fb + bb) < CA_CLEAR_MARGIN:   # the grasp pose itself (cheapest test first)
            continue
        col = [tcp - t * a for t in list(np.arange(GW_APPROACH_UP, 0.0, -CA_DESCENT_STEP))] + [tcp]
        g_obst = min(gw_box_gap(obst, p, Rm, fb + bb) for p in col)
        if g_obst < CA_CLEAR_MARGIN:
            continue
        g_own = min(gw_box_gap(M, p, Rm, bb) for p in col); g_fown = min(gw_box_gap(M, p, Rm, fb) for p in col)
        ap = col[0]; top = np.array([tcp[0], tcp[1], float(rr_lift_z(tcp)[0])]); L = float(np.linalg.norm(top - ap)); ns_ = max(1, int(np.ceil(L / CA_DESCENT_STEP)))
        g_seg = min(gw_box_gap(obst, ap + (top - ap) * k / ns_, Rm, fb + bb) for k in range(ns_ + 1))
        pred = gw_contact(M, tcp[:2], n, float(tcp[2]))
        zc = pred["predicted_contact_tcp_z"] if pred["predicted_contact_tcp_z"] is not None else (pred["range"][1] if pred["p_contact"] > 0.5 else None)
        eng = (z_hi - (max(float(tcp[2]), zc if zc is not None else float(tcp[2])) - GW_G["tcp_to_fingertip"])) * float(np.cos(np.radians(abs(deg)))) - half * float(np.sin(np.radians(abs(deg))))
        rec = {"s": row["s"], "deg": deg, "raise": round(dz, 4), "gap_obst": round(g_obst, 4), "gap_own": round(g_own, 4), "gap_fingers_own": round(g_fown, 4), "gap_retreat": round(g_seg, 4),
               "engaged": round(eng, 4), "expected": pred["expected"], "axial": True}
        rec["ok"] = bool(g_own >= CA_CLEAR_MARGIN and g_fown >= CA_CLEAR_MARGIN and g_seg >= CA_CLEAR_MARGIN and eng >= GW_ENGAGE_MIN and pred["expected"] != "premature_contact" and gb_box_ok(tcp))
        if not rec["ok"]:
            tried.append(rec); continue
        rec["ik"] = bool(ik_probe(ap, q) and ik_probe(tcp, q) and ik_probe(tcp + CONTACT_OVERSHOOT * a, q)); tried.append(rec)
        if rec["ik"]:
            return deg, dz, q, a, rec
    return None


def gt_tilt_section(rows, M, obst, ik_probe, n=None):
    """A bar section of gh_handle_plan refused only by the hand clearance, pinched with the hand pitched about the closing axis: stage 1
    (vertical column, GT_TILTS_DEG), then stage 2 (axial approach). Returns ((row, tilt_deg, extra) | None, info)."""
    fb, bb = gw_hand_boxes(); half = GW_G["finger_width"] / 2.0; tried = []
    for row in rows:
        if not gt_only_clearance(row):
            continue
        tcp = np.asarray(row["tcp"], float); zs = list(np.arange(tcp[2] + GW_APPROACH_UP, tcp[2], -CA_DESCENT_STEP)) + [float(tcp[2])]; cands = []
        for deg in GT_TILTS_DEG:
            b = np.radians(abs(float(deg))); eng = float(row["engaged"]) * float(np.cos(b)) - half * float(np.sin(b))
            q = gt_tilt_quat(row["psi"], deg); Rm = _R(q) @ OP_TOOL_FROM_MODEL.T
            g_obst = min(gw_box_gap(obst, [tcp[0], tcp[1], zz], Rm, fb + bb) for zz in zs)
            g_own = min(gw_box_gap(M, [tcp[0], tcp[1], zz], Rm, bb) for zz in zs)
            g_fown = min(gw_box_gap(M, [tcp[0], tcp[1], zz], Rm, fb) for zz in zs)
            rec = {"s": row["s"], "deg": float(deg), "gap_obst": round(g_obst, 4), "gap_own": round(g_own, 4), "gap_fingers_own": round(g_fown, 4), "engaged": round(eng, 4)}
            rec["ok"] = bool(g_obst >= CA_CLEAR_MARGIN and g_own >= CA_CLEAR_MARGIN and g_fown >= CA_CLEAR_MARGIN and eng >= GW_ENGAGE_MIN); tried.append(rec)
            if rec["ok"]:
                cands.append((min(g_obst, g_own, g_fown), float(deg), q, rec))
        for _, deg, q, rec in sorted(cands, key=lambda x: -x[0]):
            rec["ik"] = bool(ik_probe(tcp + np.array([0.0, 0.0, GW_APPROACH_UP]), q) and ik_probe(tcp, q) and ik_probe(tcp - np.array([0.0, 0.0, CONTACT_OVERSHOOT]), q))
            if rec["ik"]:
                return (row, deg, {"axial": False, "raise": 0.0}), {"tried": tried, "chosen": rec, "stage": 1}
    if n is None:
        return None, {"tried": tried, "why": "no_tilted_section"}
    for row in rows:   # stage 2 (C13I4): the axial approach, only after stage 1 found nothing
        if not gt_only_clearance(row) or row.get("top") is None:
            continue
        res = gt_axial_section(row, M, obst, ik_probe, n, tried)
        if res is not None:
            deg, dz, q, a, rec = res
            return (row, deg, {"axial": True, "raise": float(dz), "axis": [float(v) for v in a]}), {"tried": tried[-40:], "n_tried": len(tried), "chosen": rec, "stage": 2}
    return None, {"tried": tried[-40:], "n_tried": len(tried), "why": "no_tilted_section"}


def gt_tilted_quat(position, quat, plan):
    """ga_reachable_body_quat's hook for a tilted protrusion pinch: the plan's tilt at the plan's own target and top-down quaternion."""
    try:
        t = np.asarray(plan["target"], float).reshape(-1)[:3]; psi = (plan.get("yaw_order") or [None])[0]
        if psi is None or not np.allclose(np.asarray(position, float).reshape(-1)[:3], t, rtol=0.0, atol=1e-6):
            return quat
        if not np.allclose(np.asarray(quat, float), make_topdown_quat(psi), rtol=0.0, atol=1e-6):
            return quat
        q = gt_tilt_quat(psi, float(plan["tilt_deg"]))
        print("GRASP_TILT " + json.dumps({"degrees": float(plan["tilt_deg"]), "reason": "handle_pinch_hand_clearance", "approach": plan.get("approach", "vertical"), "raise_m": plan.get("raise_m", 0.0), "grasp_xyz_preserved": True, "closing_axis_preserved": True}), flush=True)
        return q
    except Exception as e:  # noqa: BLE001  (any error: the vertical quaternion)
        print("GRASP_TILT " + json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True); return quat
# ===== end C13I4 grasp_geometry =====


def gw_plan_grasp(obj_pts, cam_xy):
    """ga_plan_grasp (module B + C3I4), re-planned on the object's own body when it is infeasible or grasps a detached part (rule above)."""
    P = np.asarray(obj_pts, float); plan = ga_plan_grasp(P, cam_xy)
    try:
        if len(P) < GA_MIN_PTS:
            return plan
        L, k, sizes = gw_components(P); M = P[L == k]
        gap = None if plan.get("target") is None else gw_xy_gap(np.asarray(plan["target"], float)[:2], M)
        detached = gap is not None and gap > GW_DETACH
        if plan.get("mode") != "infeasible" and not detached:
            return gc_column_replan(plan, M, sizes, cam_xy) if plan.get("mode") == "rim_pinch" else plan   # C6I2: a module-B rim pinch whose open-finger column holds another body's visible points is re-planned around the rim (gc_column_replan)
        rec = {"trigger": "infeasible" if plan.get("mode") == "infeasible" else "detached_target", "old_mode": plan.get("mode"), "old_target": plan.get("target"), "old_gap_to_main": None if gap is None else round(gap, 4),
               "n_pts": int(len(P)), "n_main": int(len(M)), "components": sizes[:5], "decision": "module_B"}
        if len(sizes) > 1 and sizes[0] < LOW_SIZE_RATIO * sizes[1]:   # two bodies of comparable size in one mask: the binding's look-alike size band (C1I3), which body is the object is not decided here (fail closed)
            rec["why"] = "two_comparable_bodies"; print("GRASP_WIDE " + _mfj(rec), flush=True); return plan
        if detached and len(M) < len(P) and len(M) >= GA_MIN_PTS:   # only a plan on a detached part is re-planned by module B on the main body (an infeasible plan goes to the rim test only: its fallback strategies are kept for every body that is not a wide hollow ring)
            pm = ga_plan_grasp(M, cam_xy)
            if pm.get("mode") != "infeasible" and pm.get("target") is not None and gw_xy_gap(np.asarray(pm["target"], float)[:2], M) <= GW_DETACH:
                rec.update(decision="main_body_module_B", mode=pm.get("mode"), target=pm.get("target")); print("GRASP_WIDE " + _mfj(rec), flush=True); return pm
        obst, floor = gw_obstacles(M)
        if obst is None:
            rec["why"] = "no_view"; print("GRASP_WIDE " + _mfj(rec), flush=True); return plan
        try:   # C12I2: a protrusion of the ring body is pinched next to the rim (gh_handle_plan); any error or no valid section -> the rim plan below, unchanged
            hp, hinfo = gh_handle_plan(M, obst, mf_ik_probe, cam_xy)
        except Exception as _e:  # noqa: BLE001
            hp, hinfo = None, {"why": f"error: {type(_e).__name__}: {_e}"}
        rec["protrusion"] = {k: v for k, v in hinfo.items() if k != "sections"}
        if hp is not None:
            rec.update(decision="handle_pinch", target=hp["target"], yaw=hp["yaw_order"][0], floor_z=floor, n_obst=int(len(obst))); print("GRASP_WIDE " + _mfj(rec), flush=True)
            return hp
        rp, info = gw_rim_plan(M, obst, mf_ik_probe, cam_xy); rec["rim"] = info; rec["floor_z"] = floor; rec["n_obst"] = int(len(obst))
        if rp is None:
            rec["why"] = info.get("why"); print("GRASP_WIDE " + _mfj(rec), flush=True); return plan
        rec.update(decision="rim_pinch", target=rp["target"], yaw=rp["yaw_order"][0]); print("GRASP_WIDE " + _mfj(rec), flush=True)
        return rp
    except Exception as e:  # noqa: BLE001  (a failure of the wrapper never changes ga_plan_grasp's plan)
        print("GRASP_WIDE " + _mfj({"decision": "module_B", "why": f"error: {type(e).__name__}: {e}"}), flush=True); return plan
# ===== end C4I4 grasp_geometry =====



# ===== N15 C15I2 grasp_geometry / motion_preconditions / hold_contract: low-body and thin-wall grasp (N15 group D, slot C15I2) =====
# Amended interface (N15 protocol section 3): the contact primitive accepts TCP z >= -0.035, and verify_grasp / verify_hold accept a
# caller min_opening clamped at 0.03. The policy derives both from measurement; nothing changes unless a measurement says the
# parent's grasp cannot work. DIAGNOSIS.md sections 3-5.
# LT1, low body (lt_low_body): a body-pinch plan whose TCP target or contact-overshoot end the parent's box mirror refuses (gb_box_ok
#   False; the parent then never moves for it) is re-planned. Support plane z_t = lt_table_z: the dominant height of the observe
#   view's points in the ring where the open fingers land around the body (outline radius + xy_sigma .. + max_opening / 2 +
#   finger_thick), no higher than the body's perceived base + LT_Z_NOISE. Floor (TCP target) = z_t + tcp_to_fingertip - LT_Z_NOISE
#   (the fingertips are never aimed lower than the measured plane's own noise band) and >= LT_IK_LO_Z + CONTACT_OVERSHOOT (the
#   overshoot end stays in the harness box). Height = max(floor, palm bounds (G2 and H, verbatim), min(G2 depth z = base_hi +
#   LT_DEPTH_CLEAR + tcp_to_fingertip, engagement z = top - GW_ENGAGE_MIN - LT_ARRIVE_TOL + tcp_to_fingertip)): the G2 height,
#   lowered only so far that the pads still overlap the body by GW_ENGAGE_MIN when the descent stops anywhere within its 1 cm reach
#   tolerance; a support met first is a contact inside the descent's window (contact_established, as in G2). If even the floor
#   leaves less than GW_ENGAGE_MIN, module B's finger-contact model (gw_contact) predicts a premature contact on the body's own
#   points at the new height, or no plane is measured, the parent's plan is kept. Only this target is tested against the
#   measured floor (lt_box_lo, used by g1_descent_feasible); every other candidate keeps the parent's box.
# LT2, thin wall (lt_wall_min_opening): for a rim-pinch plan, on the plan's main body M (gw_components) and its ring (gw_ring, C4I4):
#   pad band = [plan TCP z - tcp_to_fingertip, rim z]; inner wall = the band's points on the far side (> 100 deg from the camera
#   direction: gw_ring's far-wall test, the only side where the inner wall is visible); span w = ring radius - p10 radius of those
#   points (the outer surface at the rim minus the innermost inner surface the pads reach: the gap vertical pads close to).
#   Measurement resolution px = one pixel of depth footprint at the rim (camera range / fx). min_opening = (w - px) / max_opening,
#   clamped to [LT_MIN_OPEN_FLOOR, LT_MIN_OPEN_DEFAULT]; applied only when below the default (a measured wall thinner than the
#   verifier's 4.8 mm empty bound), only for the attempt whose target is that plan's target, for its grasp verification and its
#   hold checks (lift and E1 hold evidence). Every other attempt, and every other plan, calls the verifier exactly as before.
# Constants (all existing gripper / perception / harness quantities):
LT_Z_NOISE = 0.004            # m, grasp_geometry PERCEPTION z_noise: depth noise on a percentile height (the measured plane's band)
LT_ARRIVE_TOL = 0.01          # m, contact_move's tol_pos default used by mc_grasp_descend: a descent counts as reached within 1 cm of its target
LT_DEPTH_CLEAR = 0.002        # m, grasp_geometry_g2 DEPTH_CLEAR (fingertips above the perceived base upper bound)
LT_PALM_CLEAR = 0.005         # m, grasp_geometry_g2 / grasp_geometry_h PALM_CLEAR
LT_FINGER_LEN = 0.054         # m, grasp_geometry GRIPPER finger_len (palm meets a top this far above the fingertips)
LT_IK_LO_Z = -0.035           # m, amended harness IK box floor (motion_interface_r IK_WORKSPACE_LO z, N15 protocol section 3)
LT_MIN_OPEN_DEFAULT = 0.06    # contact_verifier GRASP_MIN_OPENING (unchanged default)
LT_MIN_OPEN_FLOOR = 0.03      # contact_verifier GRASP_MIN_FLOOR (the harness clamp)
LT_FAR_DEG = 100.0            # deg, gw_ring / object_profile far-wall angle
LT_FAR_MIN = 30               # points, gw_ring's far-wall count


def _lt_view():
    """(depth, K, E) of the observe view (the last stashed view: run_subgoal's observe frame, as gw_obstacles), or None."""
    if not _C_VIEWS:
        return None
    _, obs = _C_VIEWS[-1]; cam = obs["agentview"]; dep = cam["images"]["depth"]; dep = dep[:, :, 0] if len(dep.shape) == 3 else dep
    return np.asarray(dep), np.asarray(cam["intrinsics"], float), np.asarray(cam["pose_mat"], float)


def lt_table_z(P):
    """Support plane under the body P from the observe view's public depth (rule above): (z | None, info)."""
    v = _lt_view()
    if v is None:
        return None, {"why": "no_view"}
    dep, K, E = v; P = np.asarray(P, float); g = GW_G
    A = np.asarray(mask_to_world_points(np.ones(dep.shape, np.uint8), dep, K, E), float).reshape(-1, 3)
    c = np.median(P[:, :2], 0); r_o = float(np.percentile(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]), 95)); base = float(np.percentile(P[:, 2], 2))
    rr = np.hypot(A[:, 0] - c[0], A[:, 1] - c[1])
    sel = (rr >= r_o + g["xy_sigma"]) & (rr <= r_o + g["max_opening"] / 2.0 + g["finger_thick"]) & (A[:, 2] <= base + LT_Z_NOISE) & np.all(np.isfinite(A), 1)
    z = A[sel, 2]; info = {"n": int(len(z)), "r_outline": round(r_o, 4), "base_p2": round(base, 4)}
    if len(z) < SUPPORT_FLOOR_MIN_PTS:
        return None, {**info, "why": "few_points"}
    w = LT_Z_NOISE / 2.0; nb = max(1, int(np.ceil((float(z.max()) - float(z.min())) / w)))
    h, e = np.histogram(z, bins=nb, range=(float(z.min()), float(z.min()) + nb * w)); k = int(np.argmax(h)); zm = 0.5 * float(e[k] + e[k + 1])
    near = z[np.abs(z - zm) <= LT_Z_NOISE]; zt = float(np.median(near))
    return zt, {**info, "z": round(zt, 4), "n_mode": int(len(near))}


# ===== C16I5 grasp_geometry LT3: late-arrival engagement of an in-box body pinch (round N16, chunk 16, iteration 5) =====
# The contact descent counts as reached anywhere within LT_ARRIVE_TOL (1 cm) above its target and the gripper closes there.
# Recorded (development rows, body pinches on boxes 1.5-1.7 cm tall): the descents stopped 0.06-0.97 cm above the target in
# free space (no gripper contact); G2's depth target (perceived base + 0.7 cm) then leaves the pads at most 0.5 cm under the top
# when the stop is late, and the close is empty (opening 0.015) or the body is lost at the lift; every close with >= 0.65 cm
# overlap held. LT1 (C15I2) already places a low body's target so that the pads overlap by GW_ENGAGE_MIN at the latest
# arrival, but only for targets outside the parent's box. LT3 applies the same LT1 height rule (same floor, palm bounds,
# engagement height, module B premature-contact refusal, same record) to an in-box body pinch whose own target leaves less
# than GW_ENGAGE_MIN at the latest arrival: top - (z + LT_ARRIVE_TOL - tcp_to_fingertip) < GW_ENGAGE_MIN. Only a LOWER
# target replaces the parent's; every other plan (taller bodies, rim pinches, other modes) is untouched. No new constant.
def lt3_late_short(plan):
    """True when an in-box body pinch leaves less than GW_ENGAGE_MIN pad overlap at its latest accepted arrival (LT3)."""
    prof = plan.get("profile") or {}
    if prof.get("top") is None:
        return False
    z = float(np.asarray(plan["target"], float).reshape(-1)[2])
    return float(prof["top"]) - (z + LT_ARRIVE_TOL - GW_G["tcp_to_fingertip"]) < GW_ENGAGE_MIN - 1e-9
# ===== end C16I5 LT3 helper =====


def lt_low_body(plan, P):
    """LT1 (rule above): the re-planned body pinch of a body too low for the parent's box, or the parent's plan. (plan, record | None)."""
    if plan.get("mode") != "body_pinch" or plan.get("target") is None or plan.get("profile") is None:
        return plan, None
    _inbox = gb_box_ok(plan["target"])   # C16I5 LT3: an in-box target is re-planned only when it is short of GW_ENGAGE_MIN at the latest arrival
    if _inbox and not lt3_late_short(plan):
        return plan, None
    prof = plan["profile"]; ft = GW_G["tcp_to_fingertip"]; rec = {"rule": "LT3_late_arrival" if _inbox else "LT1_low_body", "parent_rule": plan.get("rule"), "parent_target": plan.get("target")}
    zt, tinfo = lt_table_z(P); rec["table"] = tinfo
    if zt is None:
        rec["decision"] = "parent"; rec["why"] = "no_support_plane"; return plan, rec
    top = float(prof["top"]); z_floor = max(zt + ft - LT_Z_NOISE, LT_IK_LO_Z + CONTACT_OVERSHOOT + 1e-4)
    z_palm = max(top - (LT_FINGER_LEN - LT_PALM_CLEAR), top - GW_H["hand_clear_above_tcp"] + GW_H["palm_clear"] + GW_H["overshoot"])
    z_g2 = float(prof["base_hi"]) + LT_DEPTH_CLEAR + ft; z_eng = top - GW_ENGAGE_MIN - LT_ARRIVE_TOL + ft
    z = max(z_floor, z_palm, min(z_g2, z_eng)); eng_late = top - (z + LT_ARRIVE_TOL - ft)
    rec.update(table_z=round(zt, 4), floor_tcp=round(z_floor, 4), z_g2=round(z_g2, 4), z_engage=round(z_eng, 4), z_palm=round(z_palm, 4), z=round(z, 4), engaged_at_latest_arrival=round(eng_late, 4))
    if eng_late < GW_ENGAGE_MIN - 1e-9:
        rec["decision"] = "parent"; rec["why"] = "body_lower_than_the_engagement_at_the_measured_floor"; return plan, rec
    if _inbox and z >= float(np.asarray(plan["target"], float).reshape(-1)[2]) - 1e-6:   # C16I5 LT3: only a lower target replaces the parent's
        rec["decision"] = "parent"; rec["why"] = "no_lower_height"; return plan, rec
    psi = float((plan.get("yaw_order") or [0])[0]); d = np.array([np.sin(np.radians(psi)), -np.cos(np.radians(psi))])   # pad closing axis of make_topdown_quat(psi)
    pred = gw_contact(np.asarray(P, float), np.asarray(plan["target"], float).reshape(-1)[:2], d, z); rec["expected"] = pred["expected"]
    if pred["expected"] == "premature_contact":   # module B's finger-contact model on the body's own points at the new height
        rec["decision"] = "parent"; rec["why"] = "premature_contact_predicted"; return plan, rec
    t = np.asarray(plan["target"], float).copy(); t[2] = z; z_ft = z - ft
    new = dict(plan)
    new.update({"target": t, "rule": rec["rule"], "lt_floor_end": float(z_floor - CONTACT_OVERSHOOT), "lt": rec, "hang_est": float(z - prof["base_est"]),
                "hang_interval": [float(z - prof["base_hi"]), float(z - prof["base_lo"])], "pad_engaged_m": float(min(top, z_ft + GW_G["pad_engage"]) - max(float(prof["base_est"]), z_ft)),
                "expected": pred["expected"], "predicted_contact_tcp_z": pred["predicted_contact_tcp_z"], "contact_range": [float(prof["base_lo"] + ft), float(prof["base_hi"] + ft)],
                "plan_parent": {"rule": plan.get("rule"), "target": plan.get("target")}})
    rec["decision"] = "measured_floor"
    return new, rec


def lt_wall_min_opening(plan, P, E):
    """LT2 (rule above): (min_opening | None, record | None) for a rim-pinch plan."""
    if plan.get("mode") != "rim_pinch" or plan.get("target") is None:
        return None, None
    rec = {"rule": "LT2_thin_wall"}; v = _lt_view()
    if v is None:
        rec["why"] = "no_view"; return None, rec
    K = v[1]; P = np.asarray(P, float)
    L, k, _ = gw_components(P); M = P[L == k]; cam = np.asarray(E, float)[:3, 3]
    ring, info = gw_ring(M, cam[:2])
    if ring is None:
        rec["why"] = "no_ring: " + str(info.get("why")); return None, rec
    c = np.asarray(ring["c"], float); r = float(ring["r"]); z_rim = float(ring["z_rim"]); z_lo = float(np.asarray(plan["target"], float).reshape(-1)[2]) - GW_G["tcp_to_fingertip"]
    rr = np.hypot(M[:, 0] - c[0], M[:, 1] - c[1]); a_cam = np.arctan2(cam[1] - c[1], cam[0] - c[0])
    far = np.abs(np.angle(np.exp(1j * (np.arctan2(M[:, 1] - c[1], M[:, 0] - c[0]) - a_cam)))) > np.radians(LT_FAR_DEG)
    band = far & (M[:, 2] >= z_lo) & (M[:, 2] <= z_rim)
    rec.update(r=round(r, 4), z_rim=round(z_rim, 4), z_pad_low=round(z_lo, 4), n_band=int(band.sum()))
    if int(band.sum()) < LT_FAR_MIN:
        rec["why"] = "inner_wall_not_seen"; return None, rec
    w = r - float(np.percentile(rr[band], 10)); px = float(np.linalg.norm(cam - np.array([c[0], c[1], z_rim]))) / float(K[0, 0])
    m = min(LT_MIN_OPEN_DEFAULT, max(LT_MIN_OPEN_FLOOR, (w - px) / GW_G["max_opening"]))
    rec.update(span_m=round(w, 4), px_m=round(px, 4), min_opening=round(m, 4))
    if m >= LT_MIN_OPEN_DEFAULT:
        rec["why"] = "wall_not_thin"; return None, rec
    rec["why"] = "thin_wall"
    return m, rec


def lt_plan(plan, P, E):
    """LT1 / LT2 on the program's grasp plan (mf_plan_grasp). Returns the parent's plan object unchanged unless a rule applies."""
    try:
        new, rec = lt_low_body(plan, P)
        m, wrec = lt_wall_min_opening(new, P, E)
        if m is not None:
            new = dict(new); new["lt_min_opening"] = float(m)
        if rec is not None or wrec is not None:
            print("LT_PLAN " + _mfj({"low_body": rec, "thin_wall": wrec, "target": new.get("target"), "min_opening": new.get("lt_min_opening")}), flush=True)
        return new
    except Exception as e:  # noqa: BLE001  (a failure of the measurement never changes the parent's plan)
        print("LT_PLAN " + _mfj({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return plan


def _lt_same(a, b):
    return a is not None and b is not None and bool(np.allclose(np.asarray(a, float).reshape(-1)[:3], np.asarray(b, float).reshape(-1)[:3], rtol=0.0, atol=1e-6))


def lt_box_lo(target):
    """The box floor g1_descent_feasible applies to this target: the measured floor (LT1) for the LT1 plan's own target, else the parent's."""
    lo = np.asarray(_G1_WS_LO, float).copy(); plan = _MF.get("last_plan") or {}
    if plan.get("lt_floor_end") is not None and _lt_same(target, plan.get("target")):
        lo[2] = max(float(plan["lt_floor_end"]), LT_IK_LO_Z)
    return lo


def lt_min_opening_for(target):
    """LT2's measured min_opening for the attempt at the current plan's own target, else None (the verifier's default)."""
    plan = _MF.get("last_plan") or {}
    m = plan.get("lt_min_opening")
    return float(m) if (m is not None and _lt_same(target, plan.get("target"))) else None


def lt_verify_hold(gv, opening_now):
    """verify_hold against the grasp's reference with the grasp's own min_opening (LT2), else exactly the parent's call."""
    m = gv.get("lt_min_opening") if isinstance(gv, dict) else None
    return verify_hold(gv["after"], opening_now) if m is None else verify_hold(gv["after"], opening_now, min_opening=m)
# ===== end N15 C15I2 =====


def mf_plan_grasp(obj_pts, E, label):
    plan = (fg_plan_grasp if _FCFG["grasp_geometry"] == "F" else gw_plan_grasp)(np.asarray(obj_pts, float), np.asarray(E)[:2, 3]); _MF["last_plan"] = plan; plan = lt_plan(plan, obj_pts, E); _MF["last_plan"] = plan   # N15 C15I2: LT1 measured floor / LT2 measured thin wall (lt_plan); C3I4: module B, along the body's thin axis when module B has no pinch or an oblique one; C4I4: re-planned on the object's own body (main 3D component; a wide hollow body at its rim wall) when that plan is infeasible or grasps a detached part
    print("MF_GRASP_PLAN " + _mfj({"label": label, "rule": plan.get("rule"), "mode": plan["mode"], "reason": plan.get("reason"), "target": plan.get("target"), "yaw_order": plan.get("yaw_order"), "side": plan.get("side"), "expected": plan.get("expected"),
                                  "predicted_contact_tcp_z": plan.get("predicted_contact_tcp_z"), "contact_range": plan.get("contact_range"), "hang_est": plan.get("hang_est"), "hang_interval": plan.get("hang_interval"), "z_module_B": plan.get("z_module_B"),
                                  "pad_engaged_m": plan.get("pad_engaged_m"), "profile": plan.get("profile_summary")}), flush=True)
    return plan
def mf_record_grasp(rg):
    plan = _MF.get("last_plan") or {}; h = rg.get("hold") or {}
    zc = (h.get("p") or [None, None, None])[2] if (rg.get("status") == "contact_established" or rg.get("detail") == "premature_contact") else None
    print("MF_GRASP_ACTUAL " + _mfj({"label": rg.get("label"), "status": rg.get("status"), "detail": rg.get("detail"), "mode": plan.get("mode"), "rule": plan.get("rule"), "target_z": g1_actual_target_z(rg, plan), "planned_z_first_plan": None if plan.get("target") is None else float(plan["target"][2]), "engagement": g1_engagement(rg), "expected": plan.get("expected"),
                                    "predicted_contact_tcp_z": plan.get("predicted_contact_tcp_z"), "actual_contact_tcp_z": zc, "final_tcp_z": (rg.get("final_tcp") if rg.get("final_tcp") is not None else [None] * 3)[2], "steps": rg.get("steps_used")}), flush=True)
def mf_ik_probe(pos, quat):
    try:
        j = solve_ik([float(v) for v in pos], [float(v) for v in quat])
        return j is not None and len(np.asarray(j, float).reshape(-1)) >= 7 and bool(np.all(np.isfinite(np.asarray(j, float))))
    except Exception:  # noqa: BLE001
        return False
def mf_select_target(tcp_xy, q_place, support_z, drop_used, sg):
    ref = _MF["rest_ref"] if _MF.get("rest_ref") is not None else float(support_z)
    sel = fg_select_descent_target(mf_ik_probe, tcp_xy, q_place, float(ref), float(drop_used), CONTACT_OVERSHOOT, PLACE_TOUCHDOWN_UNDERSHOOT)
    sel = {**sel, "rest_ref": float(ref), "rest_src": _MF.get("rest_src"), "program_support_z": float(support_z), "drop_used": float(drop_used)}
    print("MF_TARGET " + _mfj({"sg": sg, **sel}), flush=True); _mark(ev="mf_target", sg=sg, feasible=sel["feasible"], z_release=sel["z_release"], rest_src=sel["rest_src"])
    return sel
def mf_validate_beside_target(nominal_xy, support_pts, obj_pts, E):
    sup = pv_support_from_points(np.asarray(support_pts, float)); c = np.asarray(sup["center_xy"], float); rel = np.asarray(nominal_xy, float)[:2] - c
    he = fg_half_extent_along(np.asarray(obj_pts, float), rel)
    xy, info = fg_validate_beside_target(nominal_xy, c, sup["r"], he)
    print("MF_BESIDE_TARGET " + _mfj({"nominal": nominal_xy, "validated": xy, "half_extent_along": he, "support_r": sup["r"], **info}), flush=True)
    return float(xy[0]), float(xy[1])
def _mf_verify(phase, place_xy, prev_meas=None):
    con = _MF.get("contract")
    if con is None:
        return {"status": "uncertain", "reason": "no placement contract", "evidential_only": True, "checks": {}, "claim": "ready_to_release" if phase == "pre_release" else "placed"}, None
    obs = get_observation(); fr = mf_frame(obs); p_now, _ = tcp_from_obs(obs)
    v = pv_verify_frame(con, fr, segment_sam3_text_prompt, mask_to_world_points, np.asarray(place_xy, float), phase=phase, prev_meas=prev_meas)
    m = v.get("meas")
    if isinstance(m, dict) and m.get("base_xy") is not None and _MF.get("target_xy") is not None:
        v["offset_xy"] = (np.asarray(m["base_xy"], float)[:2] - np.asarray(_MF["target_xy"], float)[:2]).tolist()
    v["tcp_now"] = np.asarray(p_now, float).tolist()
    return v, m
def _mf_rec(v, rd, k):
    ch = v.get("checks", {}) or {}
    return {"k": k, "descent": (rd or {}).get("status"), "status": v.get("status"), "claim": v.get("claim"), "evidential_only": v.get("evidential_only"), "reason": v.get("reason"), "offset_xy": v.get("offset_xy"), "tcp_z": (v.get("tcp_now") or [None, None, None])[2],
            "checks": {kk: {"status": c.get("status"), "value": c.get("value")} for kk, c in ch.items()}, "evidence": {kk: vv for kk, vv in (v.get("evidence") or {}).items() if kk != "meas"}}
def mf_pre_release(summary, tcp_xy, q_place, place_xy, sg):
    """Readiness verdict(s) from fresh frames (object still held). MF: bounded corrections from module F. Shadow arms: one
    verdict, logged only. Returns verdict + retries_left + history (+ the possibly recentred TCP xy)."""
    hist = []; retries = PLACE_RETRIES_TOTAL if _FCFG["corrections"] else 0; verdict = None; xy_now = np.asarray(tcp_xy, float).copy(); k = 0
    while True:
        rd = (summary.get("descend") or {}).get("mc_result") or {}
        verdict, meas = _mf_verify("pre_release", place_xy)
        rec = _mf_rec(verdict, rd, k); hist.append(rec)
        print(("MF_PRE_RELEASE " if _FCFG["enforce"] else "MF_SHADOW_PRE ") + _mfj({"sg": sg, **rec}), flush=True); _mark(ev="mf_pre_release", sg=sg, k=k, status=verdict.get("status"), claim=verdict.get("claim"), shadow=not _FCFG["enforce"])
        act = fg_correction(verdict, rd, retries, verdict.get("offset_xy")) if _FCFG["corrections"] else None
        if act is None:
            if _FCFG["corrections"] and not (verdict.get("status") == "ready" and rd.get("status") == "contact_established"):
                retries = 0   # no applicable bounded correction: retries are exhausted by definition
            break
        retries -= 1; k += 1; p_now = np.asarray(verdict.get("tcp_now"), float); z_rel = summary.get("z_release")
        print("MF_CORRECTIVE " + _mfj({"sg": sg, "k": k, "retries_left": retries, **act, "z_now": float(p_now[2]), "z_release": z_rel}), flush=True); _mark(ev="mf_corrective", sg=sg, kind=act["kind"], k=k)
        if act["kind"] == "recenter":
            gap = ((verdict.get("checks") or {}).get("support_gap") or {}).get("value"); z_up = float(p_now[2] + max(float(gap) if gap is not None and gap > 0 else 0.0, 0.0) + 0.02)
            r1 = move_to_pose(np.array([p_now[0], p_now[1], z_up]), q_place, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:recenter_up"); _PRIM_STATS["motion_calls"] += 1
            xy_new = xy_now + np.asarray(act["shift_xy"], float)
            r2 = move_to_pose(np.array([xy_new[0], xy_new[1], z_up]), q_place, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:recenter_shift"); _PRIM_STATS["motion_calls"] += 1
            print("MF_RECENTER_MOVES " + _mfj({"sg": sg, "up": r1.get("status"), "shift": r2.get("status"), "xy_from": xy_now, "xy_to": xy_new, "z_up": z_up}), flush=True)
            if r1.get("status") != "reached" or r2.get("status") != "reached" or z_rel is None:
                retries = 0; break
            xy_now = xy_new; summary["descend"] = mc_place_descend(xy_now, q_place, float(z_rel), z_up, sg)
        elif act["kind"] == "redescend":
            if z_rel is None:
                retries = 0; break
            summary["descend"] = mc_place_descend(xy_now, q_place, float(z_rel), float(p_now[2]), sg)
        else:   # descend_for_touchdown | descend_by_gap
            z_new = float(p_now[2] - min(float(act["step"]), 0.06)); summary["descend"] = mc_place_descend(xy_now, q_place, z_new, float(p_now[2]), sg)
    summary["place_xy_tcp_final"] = xy_now
    return {"verdict": verdict, "retries_left": retries, "history": hist, "tcp_xy": xy_now}
def mf_post_release(place_xy, sg):
    v1, m1 = _mf_verify("post_release", place_xy); v2, _m2 = _mf_verify("post_release", place_xy, prev_meas=m1)
    out = {**_mf_rec(v2, None, 1), "first_frame_status": v1.get("status")}
    if _FCFG["enforce"]:
        print("MF_POST_RELEASE " + _mfj({"sg": sg, **out}), flush=True); _mark(ev="mg_post_release", sg=sg, status=out["status"], claim=out["claim"])   # recorder mark: evidence-time alignment with the official predicate
    else:
        print("MF_SHADOW_POST " + _mfj({"sg": sg, **out}), flush=True); _mark(ev="mf_shadow_post", sg=sg, status=out["status"], claim=out["claim"])
    return out
# ===== end F header =====

# ===== visual verifier V1 (shadow mode; public frames only) =====
# Differences from V0 (verifier_standalone.verify_mug_on_plate):
#  - height reference = support *resting surface* (plate p50/p90 geometry) memorised from the pre-placement
#    frame, not the visible-surface centroid of whatever the plate mask is at verification time;
#  - object XY = body-axis centre (circle fit on the visible cylinder band), not the visible-surface centroid;
#  - plate XY = rim circle fit with occluded angular bins dropped; pre-placement memory is used unless the
#    current rim evidence says the plate moved;
#  - identity: candidates from all prompts, clustered; memorised positions of already-placed objects are
#    excluded; ambiguous identity -> uncertain;
#  - explicit `uncertain` whenever evidence is insufficient (no default success/failure).
# Image-only verification cannot equal the official contact predicate; it approximates On(obj, plate).
import json as _vj
import numpy as np

V1 = {
    "xy_pass": 0.030, "xy_uncertain": 0.045, "base_dz_pass": 0.020, "base_dz_uncertain": 0.045,
    "min_pts": 40, "min_score": 0.10, "cluster_merge": 0.04, "identity_gap": 0.10, "exclude_radius": 0.05,
    "plate_moved_tol": 0.025, "plate_r_range": (0.06, 0.16), "rim_bins": 24, "rim_keep_frac": 0.8,
    "ws": {"x": (0.1, 0.9), "y": (-0.6, 0.6), "z": (-0.05, 0.40)},
}
_V1_STATS = {"sam3_calls": 0}


def v1_frame(obs):
    cam = obs["agentview"]; depth = cam["images"]["depth"]
    return {"rgb": cam["images"]["rgb"], "depth": depth[:, :, 0] if len(depth.shape) == 3 else depth,
            "K": cam["intrinsics"], "E": cam["pose_mat"], "cam_xy": np.asarray(cam["pose_mat"])[:2, 3]}


def _v1_circle(xy):
    xy = np.asarray(xy, float)
    if len(xy) < 6:
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


def v1_plate_geometry(pts):
    """Rim circle fit from the plate's visible points: angular bins around the centroid, keep the farthest
    point per bin, drop bins whose reach is < keep_frac of the median (occluded/partial), fit a circle."""
    pts = np.asarray(pts, float); xy = pts[:, :2]; cen = xy.mean(0)
    ang = np.arctan2(xy[:, 1] - cen[1], xy[:, 0] - cen[0]); rad = np.hypot(*(xy - cen).T)
    nb = V1["rim_bins"]; bins = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    ext = []; reach = np.full(nb, np.nan)
    for b in range(nb):
        m = bins == b
        if m.any():
            i = np.argmax(np.where(m, rad, -1)); ext.append(xy[i]); reach[b] = rad[i]
    ext = np.array(ext); med = np.nanmedian(reach)
    keep = ext[np.hypot(*(ext - cen).T) >= V1["rim_keep_frac"] * med] if len(ext) else ext
    fit = _v1_circle(keep) if len(keep) >= 6 else None
    z = pts[:, 2]; p50, p90 = float(np.percentile(z, 50)), float(np.percentile(z, 90))
    out = {"centroid_xy": cen, "n_pts": int(len(pts)), "n_bins_kept": int(len(keep)), "top_z": p90, "rest_z": max(p50, p90 - 0.015), "table_z": float(np.percentile(z, 2))}
    if fit is not None and V1["plate_r_range"][0] <= fit[2] <= V1["plate_r_range"][1] and fit[3] < 0.012:
        out.update({"center_xy": np.array(fit[:2]), "r": fit[2], "rms": fit[3], "fit_ok": True})
    else:
        out.update({"center_xy": cen, "r": float(np.nanmedian(reach)) if len(ext) else 0.1, "rms": None, "fit_ok": False})
    return out


def v1_object_geometry(pts, cam_xy):
    pts = np.asarray(pts, float)
    axis_xy, how = object_axis_center_xy(pts, cam_xy)  # from place_primitive (shared helper)
    return {"axis_xy": axis_xy, "centroid_xy": pts[:, :2].mean(0), "base_z": float(np.percentile(pts[:, 2], 2)),
            "top_z": float(np.percentile(pts[:, 2], 98)), "n_pts": int(len(pts)), "how": how}


def v1_build_memory(frame, plate_pts, obj_pts, others_xy=()):
    """Pre-placement memory built from the SAME frame/masks the policy used for binding (no extra SAM3)."""
    pg = v1_plate_geometry(plate_pts); og = v1_object_geometry(obj_pts, frame["cam_xy"])
    return {"plate": pg, "obj_initial": og, "others_xy": [np.asarray(o, float) for o in others_xy]}


def v1_candidates(frame, prompts, exclude_xy=(), max_masks=5):
    cands = []
    ws = V1["ws"]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(frame["rgb"], prompt); _V1_STATS["sam3_calls"] += 1
        for m in (masks or [])[:max_masks]:
            if m["score"] < V1["min_score"]:
                continue
            pts = mask_to_world_points(m["mask"].astype(np.uint8), frame["depth"], frame["K"], frame["E"])
            if pts is None or len(pts) < V1["min_pts"]:
                continue
            g = v1_object_geometry(pts, frame["cam_xy"]); c = g["axis_xy"]
            if not (ws["x"][0] < c[0] < ws["x"][1] and ws["y"][0] < c[1] < ws["y"][1] and ws["z"][0] < g["base_z"] < ws["z"][1]):
                continue
            if any(np.hypot(*(c - np.asarray(e)[:2])) < V1["exclude_radius"] for e in exclude_xy):
                continue
            g.update({"score": float(m["score"]), "prompt": prompt}); cands.append(g)
    # cluster by axis xy
    clusters = []
    for g in sorted(cands, key=lambda d: -d["score"]):
        for cl in clusters:
            if np.hypot(*(g["axis_xy"] - cl["axis_xy"])) < V1["cluster_merge"]:
                cl["members"] += 1; break
        else:
            clusters.append({**g, "members": 1})
    return clusters


def v1_current_plate(frame, memory, select_prompt="plate"):
    """Re-localise the memorised plate in the current frame (nearest candidate); decide whether it moved."""
    masks = segment_sam3_text_prompt(frame["rgb"], select_prompt); _V1_STATS["sam3_calls"] += 1
    best = None
    for m in (masks or [])[:15]:
        pts = mask_to_world_points(m["mask"].astype(np.uint8), frame["depth"], frame["K"], frame["E"])
        if pts is None or len(pts) < 30:
            continue
        pg = v1_plate_geometry(pts)
        if not (-0.03 < pg["rest_z"] < 0.08):
            continue
        d = float(np.hypot(*(pg["center_xy"] - memory["plate"]["center_xy"])))
        if d < 0.10 and (best is None or d < best[0]):
            best = (d, pg)
    if best is None:
        return {"found": False}
    d, pg = best
    partial = pg["n_pts"] < 0.5 * memory["plate"]["n_pts"] or pg["n_bins_kept"] < 12
    moved = pg["fit_ok"] and not partial and d > V1["plate_moved_tol"]
    return {"found": True, "geom": pg, "dist_to_memory_cm": round(d * 100, 2), "partial": bool(partial), "moved": bool(moved)}


def v1_verify_on(frame, prompts, memory, placed_others_xy=()):
    """3-state verdict for On(object, plate) from one public frame + pre-placement memory."""
    ev = {"sam3_before": _V1_STATS["sam3_calls"]}
    exclude = list(memory.get("others_xy", [])) + [np.asarray(p, float) for p in placed_others_xy]
    cl = v1_candidates(frame, prompts, exclude_xy=exclude)
    ev["n_candidate_clusters"] = len(cl)
    if not cl:
        return {"status": "uncertain", "reason": "object not found", "evidence": ev}
    # plate reference: memory unless the current frame shows it moved
    cur = v1_current_plate(frame, memory)
    ref = memory["plate"]; ev["plate_ref"] = "memory"
    if cur.get("found") and cur.get("moved"):
        ref = cur["geom"]; ev["plate_ref"] = "current(moved %.1f cm)" % cur["dist_to_memory_cm"]
    ev["plate_current"] = {k: v for k, v in cur.items() if k != "geom"}
    ev["plate_center_xy"] = ref["center_xy"]; ev["plate_rest_z"] = ref["rest_z"]
    # identity
    best = cl[0]
    if len(cl) > 1 and cl[1]["score"] >= best["score"] - V1["identity_gap"] and np.hypot(*(cl[1]["axis_xy"] - best["axis_xy"])) > 0.05:
        d0 = np.hypot(*(best["axis_xy"] - ref["center_xy"])); d1 = np.hypot(*(cl[1]["axis_xy"] - ref["center_xy"]))
        ev["ambiguous_pair_xy_cm"] = [round(float(d0) * 100, 1), round(float(d1) * 100, 1)]
        if not (d0 > V1["xy_uncertain"] and d1 > V1["xy_uncertain"]):
            return {"status": "uncertain", "reason": "identity ambiguous", "evidence": ev}
    xy = float(np.hypot(*(best["axis_xy"] - ref["center_xy"]))); dz = float(best["base_z"] - ref["rest_z"])
    ev.update({"obj_axis_xy": best["axis_xy"], "obj_base_z": best["base_z"], "xy_dist": xy, "base_dz": dz, "obj_score": best["score"],
               "obj_prompt": best["prompt"], "obj_how": best["how"], "n_pts": best["n_pts"]})
    if xy <= V1["xy_pass"] and abs(dz) <= V1["base_dz_pass"]:
        st, why = "verified", "axis within plate tolerance and base at resting surface"
    elif xy > V1["xy_uncertain"]:
        st, why = "failed", f"xy {xy*100:.1f} cm > {V1['xy_uncertain']*100:.1f}"
    elif dz > V1["base_dz_uncertain"]:
        st, why = "failed", f"object base {dz*100:.1f} cm above support (lifted/stacked)"
    elif dz < -V1["base_dz_uncertain"]:
        st, why = "uncertain", "object base below support estimate (geometry inconsistent)"
    else:
        st, why = "uncertain", f"xy {xy*100:.1f} cm / base dz {dz*100:.1f} cm in the uncertain band"
    ev["sam3_calls"] = _V1_STATS["sam3_calls"] - ev.pop("sam3_before")
    return {"status": st, "reason": why, "evidence": ev}


def v1_verify_right_of(frame, prompts, memory, min_gap=0.02, max_dy=0.35):
    """Secondary check for 'object to the right of the plate' (image right = +Y): object rests on the table,
    y beyond the plate rim, x within the plate's x span. Weak approximation of a site-region predicate."""
    ev = {}
    cl = v1_candidates(frame, prompts, exclude_xy=memory.get("others_xy", []))
    if not cl:
        return {"status": "uncertain", "reason": "object not found", "evidence": ev}
    best = cl[0]; pl = memory["plate"]
    dy = float(best["centroid_xy"][1] - pl["center_xy"][1]); dx = float(best["centroid_xy"][0] - pl["center_xy"][0]); dz = float(best["base_z"] - pl["table_z"])
    ev.update({"dy": dy, "dx": dx, "base_dz_table": dz, "plate_r": pl["r"], "obj_score": best["score"]})
    if dz > 0.05:
        return {"status": "failed", "reason": "object lifted", "evidence": ev}
    if dy > pl["r"] + min_gap and abs(dx) < 0.15 and dy < max_dy and abs(dz) < 0.03:
        return {"status": "verified", "reason": "on table, right of plate rim", "evidence": ev}
    if dy < 0.5 * pl["r"] or dy > max_dy + 0.05 or abs(dx) > 0.22:
        return {"status": "failed", "reason": "not in the right-of-plate region", "evidence": ev}
    return {"status": "uncertain", "reason": "near region boundary", "evidence": ev}


def v1_report(tag, res):
    print("V1_SHADOW " + _vj.dumps({"tag": tag, **res}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), flush=True)
# ===== end verifier V1 =====

PRIM["version"] = "P3"

def make_topdown_quat(yaw_deg=0):
    R = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ \
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def localize_object(rgb, depth_img, K, E, prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        best = max(masks, key=lambda d: d["score"])
        mask = best["mask"].astype(np.uint8)
        pts = mask_to_world_points(mask, depth_img, K, E)
        if pts is None or len(pts) < 10:
            continue
        obb = get_oriented_bounding_box_from_3d_points(pts)
        return obb["center"], pts, mask
    return None, None, None


# ===== C1I3 binding: consistent low-score acceptance (round 20260924, chunk 1, iteration 3) =====
# The shared bind_object (frozen) answers not_found whenever the top instance's support (mean best score over the responding
# prompts) is below its absolute minimum, whatever the agreement between the prompts and whatever the gap to the next instance.
# A unique body that every responding prompt places on the SAME instance, with nothing of comparable size anywhere near it in
# support, is found with low absolute confidence, not absent. The program accepts such an instance only when ALL of the
# following hold on bind_object's public result (per-prompt best scores and pixel counts of the reported instances):
#   (1) agreement:    at least LOW_MIN_HITS prompts respond on the top instance and they are a majority of the prompt list;
#   (2) consistency:  the top instance is the best-scoring instance under EVERY responding prompt (a prompt that answers on
#                     another instance without it is a disagreement, whatever its score);
#   (3) uniqueness:   every other instance of comparable size (pixel count within a factor LOW_SIZE_RATIO of the top's: the
#                     size band of a look-alike; sub-masks of the same body and background-sized regions lie outside it) has
#                     support <= (1 - LOW_REL_MARGIN) x the top support; a full reported list whose last entry is above that
#                     level means unseen instances might be too -> rejected;
#   (4) plausibility: the top instance covers at most LOW_MAX_FRAC of the image and its support is at least LOW_MIN_SUPPORT.
# Constants (DIAGNOSIS.md section 3, from the recorded calls): LOW_MIN_HITS = the smallest majority of the adapters' 2-3 prompts
# (a single low-score prompt is the legacy wrong-object mode); LOW_SIZE_RATIO = the recorded look-alike size band 0.60-1.35 x
# with a factor >= 2.2 to spare, excluding recorded sub-masks (<= 0.25 x) and container regions (>= 21 x); LOW_REL_MARGIN = the
# smallest relative margin of a correct initial binding among same-kind distractors (0.401; ambiguous cases <= 0.045);
# LOW_MAX_FRAC = between the largest correctly bound instance (2.10 % of the image) and the smallest union / container region
# (3.25 %); LOW_MIN_SUPPORT = 2 x the shared module's per-mask noise floor (0.05). Applies only to the not_found branch.
LOW_MIN_HITS = 2
LOW_SIZE_RATIO = 3.0
LOW_REL_MARGIN = 0.40
LOW_MAX_FRAC = 0.025
LOW_MIN_SUPPORT = 0.10


def bind_accept_consistent(r, n_prompts, image_px):
    """Consistent low-score acceptance of bind_object's top instance (see the block comment). Returns (ok, info); info names
    the first failing criterion in "why" (or "consistent_low" on acceptance) together with the measured quantities."""
    ins = list(r.get("instances") or [])
    if not ins or r.get("pts") is None or r.get("mask") is None:
        return False, {"why": "no_instance"}
    I = ins[0]; sc = [float(s) for s in I["scores"]]; s0 = float(I["support"]); px0 = int(I["n_px"])
    n_prompts = len(sc) if len(sc) else int(n_prompts)
    hit = [p for p in range(len(sc)) if sc[p] > 0.0]
    n_resp = r.get("n_responding_prompts")
    if n_resp is None:   # the shared definition support = mean over responding prompts gives their number back exactly
        n_resp = int(round(sum(sc) / s0)) if s0 > 0 else 0
    others = ins[1:]
    comp = [J for J in others if px0 / LOW_SIZE_RATIO <= int(J["n_px"]) <= px0 * LOW_SIZE_RATIO]
    s1 = max((float(J["support"]) for J in comp), default=0.0)
    rel = (s0 - s1) / s0 if s0 > 0 else 0.0
    info = {"n_prompts": n_prompts, "n_hit": len(hit), "n_resp": int(n_resp), "support": round(s0, 4), "px_frac": round(px0 / float(image_px), 5),
            "n_sized_competitors": len(comp), "rel_margin_sized": round(rel, 4)}
    if len(hit) < LOW_MIN_HITS or 2 * len(hit) <= n_prompts:
        return False, dict(info, why="agreement")
    if len(hit) != int(n_resp) or any(float(J["scores"][p]) > 0.0 for J in others for p in range(len(sc)) if sc[p] <= 0.0):
        return False, dict(info, why="prompt_elsewhere")
    for p in hit:
        if any(float(J["scores"][p]) >= sc[p] for J in others):
            return False, dict(info, why="not_top_under_prompt", prompt=p)
    if rel < LOW_REL_MARGIN:
        return False, dict(info, why="comparable_instance")
    if len(ins) >= 5 and float(ins[-1]["support"]) > (1.0 - LOW_REL_MARGIN) * s0:
        return False, dict(info, why="list_truncated")
    if px0 > LOW_MAX_FRAC * float(image_px):
        return False, dict(info, why="background_sized")
    if s0 < LOW_MIN_SUPPORT:
        return False, dict(info, why="support_floor")
    return True, dict(info, why="consistent_low")
# ===== end C1I3 binding =====


# ===== C3I1 binding: side-selected object instance (round 20260924, chunk 3, iteration 1) =====
# A side word that qualifies the object phrase ("<side> <kind>", "<kind> at the <side>", "<kind> on the <side>") names one
# instance of the kind among several. The shared bind_object rates the look-alike instances alike (recorded: supports 0.5703 /
# 0.5688, rel_margin 0.003) and answers ambiguous, or "bound" on whichever scored a little higher, which is not the instance
# the word names. Rule: (1) instances of the kind = the public segmenter's masks under every prompt of the phrase (score >=
# SIDE_MASK_MIN_SCORE, the SIDE_MAX_MASKS best, >= SIDE_MIN_PTS points), highest score first, joined to an existing instance
# when the overlap relative to the smaller mask reaches SUPPORT_SAME_MASK (sub-masks of one body merge into it; the same rule
# the container path uses); support = mean over the responding prompts of the best score under that prompt (the shared
# module's definition); centre = median of the representative (highest-scoring) mask's points. (2) candidates = the instances
# the binding cannot tell apart from the top one: support >= (1 - LOW_REL_MARGIN) x top and pixel count within a factor
# LOW_SIZE_RATIO of the top's (the C1I3 look-alike band). (3) The axis is the observer's: the words describe the picture the
# camera takes. lateral (codes min / max) = the horizontal projection of the camera's image-x axis (public E, column 0: the
# image's +x side in either camera convention); depth (codes xmin / xmax) = its in-plane perpendicular, signed away from the
# camera position toward the candidates (xmax = the farthest from the camera); vertical (zmin / zmax) = base z; mid = the
# middle candidate along whichever of lateral / depth spreads them more (an odd count of at least 3). In every recorded frame
# of this round the lateral direction is +y of the base frame exactly (the reading localize_support_by_side uses for support
# instances) and the depth direction is -x; the recorded truth of the depth code fixes this reading (DIAGNOSIS.md section 2).
# (4) The extreme candidate along the axis (the middle one for mid) is the instance when its projection is separated from the
# runner-up (both neighbours for mid) by at least SIDE_MIN_GAP; a lone candidate is its own extreme (the support rule's
# precedent); otherwise the binding is ambiguous (fail closed, no motion). The selection runs only when the shared binding
# reports the kind present (bound or ambiguous); its not_found branch is unchanged (consistent-low acceptance).
# Constants: SIDE_MIN_GAP = 2 sigma of the difference of two centre estimates along one axis (each the stack's xy_sigma 0.010 m:
# sqrt(2) x 0.010 x 2 = 0.028 -> 0.03); SIDE_MAX_MASKS / SIDE_MASK_MIN_SCORE = the shared binding module's own per-prompt caps
# (efficiency only); SIDE_MIN_PTS = the legacy minimum point count of a mask (localize_object; the shared module's MIN_POINTS).
SIDE_MIN_GAP = 0.03
SIDE_MAX_MASKS = 20
SIDE_MASK_MIN_SCORE = 0.05
SIDE_MIN_PTS = 10
SIDE_LATERAL = ("min", "max")
SIDE_DEPTH = ("xmin", "xmax")
SIDE_VERTICAL = ("zmin", "zmax")
SIDE_MIDDLE = "mid"


def bind_instances(rgb, depth_img, K, E, prompts):
    """Instances of the kind named by the prompts (rule (1)): [{mask, pts, scores, support, n_px, centre}] sorted by support, descending."""
    prompts = list(prompts); flat = []; responding = []
    for pi, prompt in enumerate(prompts):
        masks = segment_sam3_text_prompt(rgb, prompt) or []; _PRIM_STATS["sam3_calls"] += 1
        keep = sorted((m for m in masks if float(m["score"]) >= SIDE_MASK_MIN_SCORE), key=lambda d: -float(d["score"]))[:SIDE_MAX_MASKS]
        got = False
        for m in keep:
            mask = np.asarray(m["mask"]).astype(np.uint8); pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < SIDE_MIN_PTS:
                continue
            flat.append((float(m["score"]), pi, mask, np.asarray(pts, float))); got = True
        if got:
            responding.append(pi)
    flat.sort(key=lambda t: -t[0])
    inst = []
    for score, pi, mask, pts in flat:
        hit = next((I for I in inst if mask_overlap_frac(mask, I["mask"]) >= SUPPORT_SAME_MASK), None)
        if hit is None:
            sc = [0.0] * len(prompts); sc[pi] = score
            inst.append({"mask": mask, "pts": pts, "scores": sc, "n_px": int(np.count_nonzero(mask)), "centre": np.median(pts, axis=0)})
        else:
            hit["scores"][pi] = max(hit["scores"][pi], score)
    for I in inst:
        I["support"] = float(np.mean([I["scores"][pi] for pi in responding])) if responding else 0.0
    inst.sort(key=lambda I: -I["support"])
    return inst


def bind_side_dirs(E, ref_xy):
    """Observer axes in the base plane (rule (3)): (lateral, depth, info); (None, None, info) when the image-x axis is vertical."""
    E = np.asarray(E, float); cam = E[:3, 3]; lat = E[:2, 0].copy(); n = float(np.linalg.norm(lat))
    if n < 1e-6:
        return None, None, {"why": "image_axis_vertical"}
    lat = lat / n; dep = np.array([-lat[1], lat[0]]); v = np.asarray(ref_xy, float)[:2] - cam[:2]
    if float(v @ dep) < 0.0:
        dep = -dep
    return lat, dep, {"cam_xy": [round(float(cam[0]), 4), round(float(cam[1]), 4)], "lateral": [round(float(lat[0]), 4), round(float(lat[1]), 4)],
                      "depth": [round(float(dep[0]), 4), round(float(dep[1]), 4)], "away_m": round(float(abs(v @ dep)), 4)}


def bind_select_by_side(rgb, depth_img, K, E, prompts, code):
    """The instance of the kind that the side code selects (rules (1)-(4)). Returns (instance | None, info); info["why"] names the
    outcome: selected | lone_instance | no_instance | unknown_code | image_axis_vertical | observer_above | no_middle | gap_below_min."""
    code = str(code).strip().lower(); inst = bind_instances(rgb, depth_img, K, E, prompts)
    info = {"code": code, "n_instances": len(inst), "supports": [round(float(I["support"]), 4) for I in inst[:5]]}
    if not inst:
        info["why"] = "no_instance"; return None, info
    s0 = float(inst[0]["support"]); px0 = int(inst[0]["n_px"])
    cands = [I for I in inst if float(I["support"]) >= (1.0 - LOW_REL_MARGIN) * s0 and px0 / LOW_SIZE_RATIO <= int(I["n_px"]) <= px0 * LOW_SIZE_RATIO]
    info.update(n_candidates=len(cands), centres=[[round(float(v), 4) for v in I["centre"]] for I in cands])
    if code in SIDE_VERTICAL:
        axis = "vertical"; proj = [float(I["centre"][2]) for I in cands]
    else:
        lat, dep, dinfo = bind_side_dirs(E, np.mean([I["centre"][:2] for I in cands], axis=0)); info["dirs"] = dinfo
        if lat is None:
            info["why"] = dinfo["why"]; return None, info
        if code in SIDE_LATERAL:
            axis, d = "lateral", lat
        elif code in SIDE_DEPTH or code == SIDE_MIDDLE:
            if code in SIDE_DEPTH and float(dinfo["away_m"]) < SIDE_MIN_GAP:   # the observer stands over the candidates: no near / far side
                info["why"] = "observer_above"; return None, info
            axis, d = "depth", dep
            if code == SIDE_MIDDLE:
                pl = [float(I["centre"][:2] @ lat) for I in cands]; pd = [float(I["centre"][:2] @ dep) for I in cands]
                if (max(pl) - min(pl)) >= (max(pd) - min(pd)):
                    axis, d = "lateral", lat
        else:
            info["why"] = "unknown_code"; return None, info
        proj = [float(I["centre"][:2] @ d) for I in cands]
    order = [int(i) for i in np.argsort(proj)]; info.update(axis=axis, projections=[round(p, 4) for p in proj])
    if code == SIDE_MIDDLE:
        if len(cands) < 3 or len(cands) % 2 == 0:
            info["why"] = "no_middle"; return None, info
        k = order[len(cands) // 2]; gap = min(proj[k] - proj[order[len(cands) // 2 - 1]], proj[order[len(cands) // 2 + 1]] - proj[k])
    elif code.endswith("min"):
        k = order[0]; gap = None if len(cands) == 1 else proj[order[1]] - proj[k]
    else:
        k = order[-1]; gap = None if len(cands) == 1 else proj[k] - proj[order[-2]]
    info.update(chosen=int(k), chosen_centre=[round(float(v), 4) for v in cands[k]["centre"]], gap_m=None if gap is None else round(float(gap), 4))
    if gap is not None and gap < SIDE_MIN_GAP:
        info["why"] = "gap_below_min"; return None, info
    info["why"] = "selected" if len(cands) > 1 else "lone_instance"
    return cands[k], info
# ===== end C3I1 binding =====


# ===== C4I3 binding: per-prompt majority among look-alike candidates (round 20260924, chunk 4, iteration 3) =====
# The shared bind_object ranks the instances of a kind by their support = the MEAN of their best scores over the responding
# prompts. The prompts of one object are several descriptions of it (its own phrase, a longer name, an appearance description);
# a description that also fits a look-alike well (recorded: an appearance prompt scores a look-alike box 0.52-0.58 against the
# named object's 0.19-0.30) moves the mean by more than the named object leads under the other two prompts (0.02-0.13), so the
# mean binds the look-alike or ties. Rule: every responding prompt casts one vote, for the candidate it scores highest (when that
# lead is decisive, see below); candidates = the instances the shared binding cannot tell apart from its top one (the C3I1
# look-alike band: support >= (1 - LOW_REL_MARGIN) x top and pixel count within a factor LOW_SIZE_RATIO of the top's; the top is always a candidate). An instance
# holding a strict majority of the votes of the responding prompts is the binding: it confirms the shared top, resolves an
# ambiguous result, or replaces a top that the majority of the descriptions ranks below it. Without a strict majority the shared
# result stands. The vote reads the shared result's own instance summaries (per-prompt best scores, pixel counts; its top 5 by
# support); the geometry of a winner other than the top is its best mask under its best-scoring prompt (the segmenter's mask
# with that score), deprojected as every mask. A prompt votes only when it separates its winner from the next candidate by
# BIND_VOTE_REL_MARGIN relative to the winner's score: the shared binding's own decisiveness criterion (object_binding
# MIN_REL_MARGIN = 0.05, applied there to the supports), applied here to each prompt's scores. Two look-alikes that every prompt
# rates alike (recorded: 0.4-1.7 % apart under the generic prompts) therefore get no votes and stay ambiguous (fail closed).
# The vote decides only when the decisive prompts DISAGREE (votes for at least two candidates): that is the case the mean gets
# wrong (the most emphatic dissenter decides). When they agree, or none is decisive, the shared result and its margin stand.
BIND_VOTE_REL_MARGIN = 0.05   # = object_binding.MIN_REL_MARGIN (the shared module's relative margin of a decisive top)


def bind_prompt_vote(r):
    """Per-prompt majority among the look-alike candidates of the shared result r (rule above). Returns (index | None, info):
    the index into r["instances"] of the strict-majority winner, None when there is none."""
    ins = list(r.get("instances") or []); info = {"n_instances": len(ins)}
    if not ins:
        info["why"] = "no_instance"; return None, info
    s0 = float(ins[0]["support"]); px0 = int(ins[0]["n_px"]); n_p = len(ins[0]["scores"])
    cand = [i for i, J in enumerate(ins) if float(J["support"]) >= (1.0 - LOW_REL_MARGIN) * s0 and px0 / LOW_SIZE_RATIO <= int(J["n_px"]) <= px0 * LOW_SIZE_RATIO]
    resp = [p for p in range(n_p) if any(float(J["scores"][p]) > 0.0 for J in ins)]
    n_resp = int(r.get("n_responding_prompts") or len(resp))
    votes = [0] * len(ins); ballot = []
    for p in resp:
        sc = sorted(((float(ins[i]["scores"][p]), i) for i in cand), reverse=True)
        if not sc or sc[0][0] <= 0.0 or (len(sc) > 1 and (sc[0][0] - sc[1][0]) < BIND_VOTE_REL_MARGIN * sc[0][0]):
            ballot.append(None); continue   # no decisive winner under this prompt: no vote
        votes[sc[0][1]] += 1; ballot.append(sc[0][1])
    k = int(np.argmax(votes)) if votes else None
    info.update(candidates=cand, n_resp=n_resp, ballot=ballot, votes=[votes[i] for i in cand])
    if len({v for v in ballot if v is not None}) < 2:   # the decisive prompts do not disagree: the shared mean and its margin decide, as before
        info["why"] = "no_disagreement"; return None, info
    if k is None or 2 * votes[k] <= n_resp:
        info["why"] = "no_majority"; return None, info
    info["why"] = "majority"; info["winner"] = k
    return k, info


def bind_instance_geometry(rgb, depth_img, K, E, prompts, inst):
    """Mask and points of a summarised instance: the segmenter's mask with the instance's best score under its best prompt."""
    sc = [float(s) for s in inst["scores"]]; p = int(np.argmax(sc))
    masks = segment_sam3_text_prompt(rgb, list(prompts)[p]) or []; _PRIM_STATS["sam3_calls"] += 1
    best = min(masks, key=lambda m: abs(float(m["score"]) - sc[p]), default=None)
    if best is None or abs(float(best["score"]) - sc[p]) > 5e-4:
        return None, None
    mask = np.asarray(best["mask"]).astype(np.uint8); pts = mask_to_world_points(mask, depth_img, K, E)
    if pts is None or len(pts) < SIDE_MIN_PTS:
        return None, None
    return mask, np.asarray(pts, float)
# ===== end C4I3 binding =====


# ===== H1 B: task-agnostic multi-prompt target binding (runner-injected bind_object; fail-closed) =====
_BIND = {"last": None, "n": 0}
def bind_specific_prompt(r, prompts):
    """Resolve a close support tie only when the first description has a clear lead.

    The other descriptions must agree that the two leading instances look alike;
    they must not contain conflicting identity evidence. Fail closed otherwise.
    """
    ins = list(r.get("instances") or [])
    info = {"why": "insufficient_evidence"}
    if r.get("status") != "ambiguous" or len(ins) < 2:
        return False, info
    if not prompts or len(str(list(prompts)[0]).split()) < 2:
        return False, {"why": "phrase_not_specific"}
    top, rival = ins[:2]
    a = [float(x) for x in top["scores"]]
    b = [float(x) for x in rival["scores"]]
    if len(a) < 3 or len(a) != len(b) or a[0] < 0.10:
        return False, info
    if not (float(rival["support"]) >= 0.60 * float(top["support"])):
        return False, info
    if not (int(top["n_px"]) / 3.0 <= int(rival["n_px"]) <= 3.0 * int(top["n_px"])):
        return False, info
    lead = (a[0] - b[0]) / a[0]
    generic_gaps = [abs(x - y) / max(x, y) if max(x, y) > 0 else 0.0
                    for x, y in zip(a[1:], b[1:])]
    comparable = [v for v in ins[1:]
                  if float(v["support"]) >= 0.60 * float(top["support"])
                  and int(top["n_px"]) / 3.0 <= int(v["n_px"]) <= 3.0 * int(top["n_px"])]
    other_leads = [(a[0] - float(v["scores"][0])) / a[0] for v in comparable]
    info = {"lead": round(lead, 4), "generic_gaps": [round(x, 4) for x in generic_gaps],
            "other_leads": [round(x, 4) for x in other_leads]}
    ok = (lead >= 0.07 and all(x <= 0.05 for x in generic_gaps)
          and all(x >= 0.07 for x in other_leads))
    info["why"] = "specific_prompt" if ok else "insufficient_evidence"
    return ok, info


def bind_recheck_reference(pick, info):
    """Snapshot the selected footprint and competing instance centres."""
    if pick is None or not isinstance(info, dict):
        return None
    if info.get("code") not in SIDE_LATERAL + SIDE_DEPTH + (SIDE_MIDDLE,):
        return None
    if info.get("why") != "selected":
        return None
    try:
        pts = np.asarray(pick["pts"], float)
        centres = np.asarray(info["centres"], float)
        chosen = int(info["chosen"])
        if (pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < SIDE_MIN_PTS
                or centres.ndim != 2 or centres.shape[1] != 3 or len(centres) < 2
                or not 0 <= chosen < len(centres)
                or not np.isfinite(pts).all() or not np.isfinite(centres).all()):
            return None
        bounds = np.percentile(pts[:, :2], [CONTAINER_PCT_LO, CONTAINER_PCT_HI], axis=0)
        rivals = np.delete(centres, chosen, axis=0)
        return {"bounds": bounds.copy(), "rivals": rivals.copy()}
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def bind_post_lift(rgb, depth_img, K, E, prompts, source_reference=None,
                   hold_status=None, lift_status=None):
    """A stationary competing instance cannot refute a selected instance's hold."""
    result = localize_target(rgb, depth_img, K, E, prompts)
    centre, pts, mask = result
    if (centre is None or float(centre[2]) > 0.10 or hold_status != "verified"
            or lift_status != "reached" or not isinstance(source_reference, dict)):
        return result
    try:
        pts = np.asarray(pts, float)
        bounds = np.asarray(source_reference["bounds"], float)
        rivals = np.asarray(source_reference["rivals"], float)
        if (pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < SIDE_MIN_PTS
                or bounds.shape != (2, 2) or rivals.ndim != 2 or rivals.shape[1] != 3
                or len(rivals) == 0 or not np.isfinite(pts).all()
                or not np.isfinite(bounds).all() or not np.isfinite(rivals).all()
                or np.any(bounds[0] > bounds[1])):
            return result
        now = np.percentile(pts[:, :2], [CONTAINER_PCT_LO, CONTAINER_PCT_HI], axis=0)
        gap = float(np.linalg.norm(np.maximum(np.maximum(bounds[0] - now[1], now[0] - bounds[1]), 0.0)))
        rival_gap = float(np.min(np.linalg.norm(rivals - np.median(pts, axis=0), axis=1)))
        if gap <= SIDE_MIN_GAP or rival_gap > SIDE_MIN_GAP:
            return result
    except (KeyError, TypeError, ValueError, IndexError):
        return result
    _BIND["last"] = "ambiguous"
    print("BINDING_RECHECK " + json.dumps({"status": "ambiguous", "why": "stationary_competitor",
          "source_gap_m": round(gap, 6), "rival_gap_m": round(rival_gap, 6)}), flush=True)
    return None, None, None


def localize_target(rgb, depth_img, K, E, prompts, object_side=None):   # C3I1: object_side = the side code of a side-qualified object phrase (None = as before)
    r = bind_object(rgb, depth_img, K, E, list(prompts))
    status = r["status"]; rule = "shared"; low_info = None; pick = None; side_info = None
    if status == "not_found":   # C1I3: consistent low-score acceptance on the shared module's own result (fail-closed otherwise)
        ok_low, low_info = bind_accept_consistent(r, len(list(prompts)), int(np.asarray(rgb).shape[0]) * int(np.asarray(rgb).shape[1]))
        if ok_low:
            status = "bound"; rule = "consistent_low"
    vote = None; vote_info = None; vote_geom = None
    if object_side is None and r["status"] in ("bound", "ambiguous"):   # C4I3: per-prompt majority among the look-alike candidates (bind_prompt_vote)
        vote, vote_info = bind_prompt_vote(r)
        if vote is not None and vote > 0:
            vote_geom = bind_instance_geometry(rgb, depth_img, K, E, list(prompts), r["instances"][vote])
            if vote_geom[0] is None:
                vote = None; vote_info["why"] = "winner_geometry_unavailable"
        if vote is not None:
            status = "bound"; rule = "prompt_majority" if (vote > 0 or r["status"] != "bound") else rule
    if object_side is None and status == "ambiguous":
        specific_ok, specific_info = bind_specific_prompt(r, prompts)
        print("BINDING_SPECIFIC " + json.dumps({"k": _BIND["n"] + 1, "accepted": specific_ok, **specific_info}), flush=True)
        if specific_ok:
            status = "bound"; rule = "specific_prompt"
    if object_side is not None and r["status"] in ("bound", "ambiguous"):   # C3I1: the kind is present; the side word selects the instance (bind_select_by_side)
        pick, side_info = bind_select_by_side(rgb, depth_img, K, E, list(prompts), object_side)
        status = "bound" if pick is not None else "ambiguous"; rule = "side_select"
    _BIND["reference"] = bind_recheck_reference(pick, side_info)
    _BIND["last"] = status; _BIND["n"] += 1
    _BIND["contested"] = vote_info is not None and len({v for v in vote_info.get("ballot", []) if v is not None}) >= 2   # C16I1: the decisive prompts disagreed
    print("BINDING " + json.dumps({"k": _BIND["n"], "status": status, "support": round(float(r["support"]), 3),
                                   "rel_margin": None if r["rel_margin"] is None else round(float(r["rel_margin"]), 3),
                                   "instances": r["instances"][:3], "sam3_calls": r["sam3_calls"], "rule": rule}), flush=True)
    if low_info is not None:
        print("BINDING_LOW " + json.dumps({"k": _BIND["n"], "accepted": status == "bound", **low_info}), flush=True)
    if side_info is not None:   # C3I1
        print("BINDING_SIDE " + json.dumps({"k": _BIND["n"], "accepted": pick is not None, **side_info}), flush=True)
    if vote_info is not None:   # C4I3
        print("BINDING_VOTE " + json.dumps({"k": _BIND["n"], "winner": vote, "shared_status": r["status"], **vote_info}), flush=True)
    if status != "bound":
        if _VB.get("phase") in ("task_start", "pregrasp") and vb_enabled():   # VB1 sub-patch 1
            _vb = vb_point_localize(rgb, depth_img, K, E, str(list(prompts)[0]), _VB["phase"])
            if _vb is not None:
                return _vb[0], _vb[1], _vb[2]
        return None, None, None
    if pick is not None:   # C3I1: the selected instance's own geometry
        pts = pick["pts"]; obb = get_oriented_bounding_box_from_3d_points(pts)
        return obb["center"], pts, np.asarray(pick["mask"]).astype(np.uint8)
    if vote is not None and vote > 0:   # C4I3: the majority winner's own geometry
        pts = vote_geom[1]; obb = get_oriented_bounding_box_from_3d_points(pts)
        return obb["center"], pts, vote_geom[0]
    pts = r["pts"]; obb = get_oriented_bounding_box_from_3d_points(pts)
    return obb["center"], pts, np.asarray(r["mask"]).astype(np.uint8)
# ===== end H1 B =====


# ===== VB1 binding: VLM-assisted localization + target identity memory (user-specified patch, 2026-10-02) =====
# Sub-patch 1 ("where is the target"): when the shared text binding ends ambiguous / not_found, the runner-injected VLM
# (query_vlm; public RGB, instruction text and phase only) names one pixel of the instruction's target or abstains. That pixel
# prompts SAM3; the mask must contain the point, cover at most VB_MASK_MAX_FRAC of the image and give at least VB_MIN_DEPTH_PTS
# deprojected points; depth + calibration give the 3D points and centre. No reranking over the SAM3 text-candidate pool.
# Sub-patch 2 ("is it still the instance the instruction named"): before any motion every object subgoal's target is resolved,
# its mask reviewed by the VLM, and its image, mask, 3D centre and an internal handle are frozen. Before a grasp the candidate is
# the instance associated with the frozen reference (SAM3 point prompt at the frozen mask's interior pixel, accepted when its
# centre stays within VB_ASSOC_M of the frozen centre; otherwise VLM re-localization with the frozen image as exemplar), never a
# re-reading of the spatial words on the current arrangement, and the VLM must confirm it against the frozen reference. After the
# lift the held object must be confirmed against both the frozen reference and the confirmed pre-grasp image; transport also
# needs the verified hold and motion with the gripper (the held body measured near the TCP, its base raised by >= VB_RISE_M).
# identity_mismatch and identity_unknown are never a confirmation. The handle is a memory key of this module, not a simulator id.
VB_MASK_MAX_FRAC = 0.08     # a point-prompt mask larger than 8 % of the image is a support / background region
VB_MIN_DEPTH_PTS = 50       # minimum deprojected points of a point-prompt mask
VB_ASSOC_M = 0.03           # = SIDE_MIN_GAP: association radius between a frozen centre and a current centre
VB_RISE_M = 0.05            # motion with the gripper: the held body's base rose at least this much above its resting base
VB_ACCEPT_CONF = ("medium", "high")
VB_VETO_TEXT_BINDING = False   # rev3: the VLM steps in only where text binding fails; it never vetoes a confident text binding
VB_REF_IOU = 0.5            # rev2: majority overlap with the frozen mask = the same body (the stack's SUPPORT_SAME_MASK majority criterion)
VB_TINT = np.array([255, 0, 255], np.uint8)   # magenta highlight
_VB = {"refs": {}, "active": None, "phase": None, "confirmed": None, "pregrasp_img": None, "base_z": None}
VB_POINT_SCHEMA = {"type": "object", "properties": {"abstain": {"type": "boolean"}, "x": {"type": "integer"}, "y": {"type": "integer"},
                   "confidence": {"type": "string", "enum": ["low", "medium", "high"]}, "reason": {"type": "string"}},
                   "required": ["abstain", "x", "y", "confidence", "reason"], "additionalProperties": False}
VB_VERDICT_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "no", "unsure"]}, "reason": {"type": "string"}},
                     "required": ["verdict", "reason"], "additionalProperties": False}


def vb_enabled():
    return "query_vlm" in globals()


def vb_log(kind, **kw):
    print("VB_" + kind + " " + json.dumps(kw, default=str), flush=True)


def vb_lang():
    try:
        return str(env.handle.task_language)
    except Exception:  # noqa: BLE001 - no instruction available offline
        return ""


def vb_highlight(rgb, mask, point=None):
    """Public RGB with the mask tinted magenta and outlined; optional point marked with a magenta cross."""
    im = np.asarray(rgb, np.uint8).copy(); m = np.asarray(mask).astype(bool)
    if m.shape == im.shape[:2] and m.any():
        im[m] = (0.55 * im[m] + 0.45 * VB_TINT).astype(np.uint8)
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        im[edge] = VB_TINT
    if point is not None:
        x, y = int(point[0]), int(point[1]); H, W = im.shape[:2]
        im[max(0, y - 1):y + 2, max(0, x - 8):x + 9] = VB_TINT; im[max(0, y - 8):y + 9, max(0, x - 1):x + 2] = VB_TINT
    return im


def vb_interior_pixel(mask):
    """(x, y) of the mask pixel nearest its pixel centroid (inside the mask even when it is not convex)."""
    ys, xs = np.nonzero(np.asarray(mask))
    if len(xs) == 0:
        return None
    cx, cy = xs.mean(), ys.mean(); i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return int(xs[i]), int(ys[i])


def vb_ask(images, prompt, schema, tag):
    if not vb_enabled():
        return None
    try:
        out = query_vlm(list(images), prompt, schema, tag=tag)
    except Exception as e:  # noqa: BLE001 - a failed query is an abstention
        vb_log("QUERY_ERROR", tag=tag, error=repr(e)[:200]); return None
    return out if isinstance(out, dict) else None


def vb_point_masks(rgb, depth_img, K, E, x, y):
    """SAM3 point prompt at (x, y). The segmenter answers with nested masks (a part such as a label or an inner surface / the
    whole object / the object merged with a neighbour or its support). Returns the valid ones (containing the point, not
    background-sized, enough depth) largest first, as [(centre, pts, mask), ...], and info. rev2: size order is only a listing
    order; which nested mask is the target is decided by vb_pick_by_reference / vb_pick_by_size / vb_choose_nested."""
    H, W = np.asarray(rgb).shape[:2]
    if not (0 <= x < W and 0 <= y < H):
        return [], {"why": "point_outside_image"}
    masks = segment_sam3_point_prompt(rgb, (float(x), float(y))) or []; _PRIM_STATS["sam3_calls"] += 1
    valid = []; info = {"n_masks": len(masks), "candidates": []}
    for d in masks:
        m = np.asarray(d["mask"]).astype(np.uint8)
        if m.shape != (H, W) or not m[y, x]:
            continue
        frac = float(np.count_nonzero(m)) / float(H * W)
        if frac > VB_MASK_MAX_FRAC:
            info["candidates"].append({"frac": round(frac, 5), "why": "large"}); continue
        pts = mask_to_world_points(m, depth_img, K, E)
        if pts is None or len(pts) < VB_MIN_DEPTH_PTS:
            info["candidates"].append({"frac": round(frac, 5), "why": "depth", "n_pts": 0 if pts is None else int(len(pts))}); continue
        valid.append((frac, float(d.get("score", 0.0)), m, np.asarray(pts, float)))
        info["candidates"].append({"frac": round(frac, 5), "why": "valid", "score": round(float(d.get("score", 0.0)), 4), "n_pts": int(len(pts))})
    valid.sort(key=lambda t: -t[0])
    info["why"] = "ok" if valid else "no_valid_mask"
    return [(get_oriented_bounding_box_from_3d_points(pts)["center"], pts, m) for _, _, m, pts in valid], info


def vb_mask_iou(a, b):
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    u = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b)) / u if u else 0.0


def vb_pick_by_reference(geoms, ref_mask):
    """rev2: the nested candidate that best overlaps the frozen mask (fixed camera, unmoved body). Returns (geom | None, iou)."""
    if not geoms:
        return None, 0.0
    ious = [vb_mask_iou(g[2], ref_mask) for g in geoms]; i = int(np.argmax(ious))
    return geoms[i], ious[i]


def vb_pick_by_size(geoms, ref_mask):
    """rev2: the nested candidate whose pixel area is closest (log ratio) to the frozen mask's - the same body seen again
    after it may have moved (a part is much smaller, a merged region much larger)."""
    a0 = max(1, int(np.count_nonzero(ref_mask)))
    return min(geoms, key=lambda g: abs(float(np.log(max(1, int(np.count_nonzero(g[2]))) / a0)))) if geoms else None


def vb_choose_nested(rgb, geoms, phrase, phase, point=None):
    """rev2: one VLM call chooses among the nested candidates (this is the mask review): image 1 = the full view with the
    point marked; images 2.. = the same zoomed region, each with one candidate highlighted. Returns the chosen geom or None."""
    if not geoms:
        return None
    H, W = np.asarray(rgb).shape[:2]
    ys, xs = np.nonzero(np.any([np.asarray(g[2]).astype(bool) for g in geoms], axis=0))
    pad = max(24, int(0.5 * max(int(xs.max() - xs.min()), int(ys.max() - ys.min()))))
    x0, y0, x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)
    images = [vb_highlight(rgb, np.zeros((H, W), np.uint8), point)] + [vb_highlight(rgb, g[2])[y0:y1, x0:x1] for g in geoms]
    labels = [str(i + 2) for i in range(len(geoms))]
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": labels + ["none"]}, "reason": {"type": "string"}},
              "required": ["choice", "reason"], "additionalProperties": False}
    text = (f"Task instruction: \"{vb_lang()}\". The robot's current step is to pick up the object the instruction calls \"{phrase}\". "
            f"Phase: {phase}. Image 1 is the full camera view; the magenta cross marks a point on the object. Images {labels[0]}"
            f"{'' if len(labels) == 1 else '-' + labels[-1]} show the same zoomed region, each with one candidate region highlighted in "
            "magenta. Which image highlights exactly that whole object instance - the complete object, not only a part of it (for "
            "example a label, an inner surface or a lid), not merged with a neighbouring object or its support, and not a different "
            "instance? Answer with that image number, or none if no image does.")
    a = vb_ask(images, text, schema, tag=f"choose:{phase}")
    ch = None if a is None else str(a.get("choice"))
    pick = geoms[labels.index(ch)] if ch in labels else None
    vb_log("CHOOSE", phase=phase, phrase=phrase, n_candidates=len(geoms), choice=ch, answer=a,
           areas=[int(np.count_nonzero(g[2])) for g in geoms])
    return pick


def vb_point_localize(rgb, depth_img, K, E, phrase, phase, exemplar=None):
    """Sub-patch 1: VLM pixel -> SAM3 point mask -> 3D points. exemplar = (rgb, mask) of the frozen reference (same instance)."""
    H, W = np.asarray(rgb).shape[:2]
    text = (f"Image 1 is the robot's current camera view, {W}x{H} pixels (x counts columns from 0, y counts rows from 0 at the top). "
            f"Task instruction: \"{vb_lang()}\". In the current step the robot must pick up the object the instruction calls \"{phrase}\". "
            f"Phase: {phase}. ")
    images = [rgb]
    if exemplar is not None:
        images.append(vb_highlight(exemplar[0], exemplar[1]))
        text += ("Image 2 is the same camera at the start of the task with that exact object instance highlighted in magenta; the robot "
                 "may have moved objects since. Find that same physical instance in image 1 (look-alike objects are different instances). ")
    else:
        text += ("Resolve which single instance the instruction refers to, reading its spatial words for the arrangement shown. ")
    text += ("Return the pixel at the centre of the visible body of that instance. Set abstain=true if you are not sure which instance "
             "it is or it is not visible.")
    a = vb_ask(images, text, VB_POINT_SCHEMA, tag=f"point:{phase}")
    if a is None or a.get("abstain") or a.get("confidence") not in VB_ACCEPT_CONF:
        vb_log("POINT", phase=phase, phrase=phrase, accepted=False, answer=a); return None
    geoms, info = vb_point_masks(rgb, depth_img, K, E, int(a["x"]), int(a["y"]))
    vb_log("POINT", phase=phase, phrase=phrase, accepted=bool(geoms), answer=a, mask=info)
    if not geoms:
        return None
    if exemplar is not None:   # rev2: the frozen body's own size selects among the nested masks
        pick = vb_pick_by_size(geoms, exemplar[1])
    else:                      # rev2: the VLM chooses the whole-instance mask among the nested candidates
        pick = vb_choose_nested(rgb, geoms, phrase, phase, point=(int(a["x"]), int(a["y"])))
    if pick is None:
        return None
    return tuple(pick) + (int(a["x"]), int(a["y"]), geoms)


def vb_review(rgb, mask, phrase, phase):
    """VLM review of one mask: 'yes' only when the highlighted region is the whole instance the instruction names."""
    text = (f"Task instruction: \"{vb_lang()}\". The robot's current step is to pick up the object the instruction calls \"{phrase}\". "
            f"Phase: {phase}. Is the region highlighted in magenta exactly that object instance - the whole object, not part of it, not "
            "a different look-alike instance and not a container or support? Answer yes, no, or unsure.")
    a = vb_ask([vb_highlight(rgb, mask)], text, VB_VERDICT_SCHEMA, tag=f"review:{phase}")
    v = "unsure" if a is None else str(a.get("verdict"))
    vb_log("REVIEW", phase=phase, phrase=phrase, verdict=v, answer=a)
    return v


def vb_same_instance(views, phrase, phase, held=False):
    """VLM cross-frame identity: views = [(rgb, mask|None), ...]; the last view is the current one. Returns
    identity_verified | identity_mismatch | identity_unknown."""
    imgs = [vb_highlight(r, m) if m is not None else np.asarray(r, np.uint8) for r, m in views]
    n = len(imgs)
    text = (f"Task instruction: \"{vb_lang()}\". The object the instruction calls \"{phrase}\" was identified at the start of the task; "
            "image 1 shows that instance highlighted in magenta. ")
    if n == 3:
        text += "Image 2 shows the instance the robot confirmed just before grasping, highlighted in magenta. "
    if held:
        text += (f"Image {n} is the current view: the robot has lifted an object between its fingers. Is the lifted object the same "
                 "physical instance as the highlighted one(s)? Use where objects were and which place is now empty. ")
    else:
        text += (f"Image {n} is the current view with a candidate highlighted in magenta. Is the candidate the same physical instance as "
                 "in image 1? Objects may have been moved by the robot; look-alike objects are different instances; use the "
                 "surrounding unmoved objects as context. ")
    text += "Answer yes, no, or unsure."
    a = vb_ask(imgs, text, VB_VERDICT_SCHEMA, tag=f"identity:{phase}")
    v = "unsure" if a is None else str(a.get("verdict"))
    out = {"yes": "identity_verified", "no": "identity_mismatch"}.get(v, "identity_unknown")
    vb_log("IDENTITY", phase=phase, phrase=phrase, result=out, answer=a)
    return out


def vb_observe():
    obs = get_observation(); cam = obs["agentview"]; depth = cam["images"]["depth"]
    return obs, cam["images"]["rgb"], (depth[:, :, 0] if len(depth.shape) == 3 else depth), cam["intrinsics"], cam["pose_mat"]


def vb_resolve_references(subgoals):
    """Sub-patch 2, task start: resolve and freeze every object subgoal's target before any object moves."""
    if not vb_enabled():
        return
    _VB["phase"] = "task_start"; obs, rgb, depth_img, K, E = vb_observe()
    for k, sg in enumerate(subgoals, 1):
        if sg.get("relation") in LA_PART_RELATIONS or sg.get("object_prompts") is None:
            continue
        phrase = str(sg.get("object") or list(sg["object_prompts"])[0])
        _VB["phase"] = "task_start_text"   # the explicit VLM path below replaces sub-patch 1's in-binding fallback here
        c, pts, mask = localize_target(rgb, depth_img, K, E, sg["object_prompts"], object_side=sg.get("object_side"))
        _VB["phase"] = "task_start"; how = "text_binding"
        if c is not None and VB_VETO_TEXT_BINDING and vb_review(rgb, mask, phrase, "task_start") != "yes":
            c = None
        if sg.get("object_side") is None and (c is None or _BIND.get("contested")):   # C16I1: gallery choice among the text candidates
            g = vg_gallery_localize(rgb, depth_img, K, E, sg["object_prompts"], phrase, "task_start")
            if g is not None and c is None:
                c, pts, mask = g; how = "vlm_gallery"
            elif g is not None and not vg_same_body(g, (c, pts, mask)):
                c, pts, mask = g; how = "gallery_contested"
        if c is None:
            r = vb_point_localize(rgb, depth_img, K, E, phrase, "task_start"); how = "vlm_point"
            if r is not None:   # rev2: vb_choose_nested inside vb_point_localize already reviewed the chosen nested mask
                c, pts, mask = r[0], r[1], r[2]
        if c is None:
            _VB["refs"][k] = {"handle": f"ref{k}", "status": "unresolved", "phrase": phrase}
        else:
            _VB["refs"][k] = {"handle": f"ref{k}", "status": "frozen", "phrase": phrase, "rgb": np.asarray(rgb, np.uint8).copy(),
                              "mask": np.asarray(mask).astype(np.uint8), "centre": np.asarray(c, float), "pixel": vb_interior_pixel(mask),
                              "base_z": float(object_base_z(pts))}
        R = _VB["refs"][k]
        vb_log("REFERENCE", k=k, handle=R["handle"], status=R["status"], phrase=phrase, how=how if c is not None else None,
               centre=None if c is None else [round(float(v), 4) for v in c])
    _VB["phase"] = None


def vb_reference_active():
    R = _VB["refs"].get(_VB["active"]) if _VB["active"] is not None else None
    return R is not None


# ===== C17I1 binding VBC: the frozen body's re-found mask keeps only the body (round N17, chunk 17, iteration 1) =====
# Before a grasp the frozen reference is re-found by a SAM3 point prompt at its interior pixel; the nested mask that best
# overlaps the frozen mask is taken with all its pixels. Recorded (development, task-start reference bound by the gallery): the
# chosen mask (IoU 0.966 with the frozen mask) held 1629 px on the bound body plus three detached specks (3 + 2 + 13 px) on a
# taller neighbour 13 cm away (z up to 0.082 m). Their points enter the object profile, and the footprint radius
# (max(r_top, r_mid)) exceeded the container's inner radius 0.0647, so the container was refused (smaller_than_object) and
# the episode stopped with no motion; the same body's radius is 0.035-0.036 in every other recorded episode. Rule: when the
# accepted mask has more than one connected pixel region, keep the region that holds the prompt pixel (the frozen body's
# interior; the largest region when the pixel is not covered) plus every other region whose 3D points come within VB_ASSOC_M
# of it (a body split by an occluder or by the depth edge); a region farther than the association radius (= SIDE_MIN_GAP,
# the gap that separates two bodies) is another body. A kept set below VB_MIN_DEPTH_PTS points keeps the parent's mask.
# A single-region mask (every other recorded pre-grasp mask but 3 of 79, whose extra regions are 1-12 px) is untouched.
def vb_body_component(geom, pixel, depth_img, K, E):
    """VBC: (centre, pts, mask) restricted to the prompt pixel's connected region plus regions within VB_ASSOC_M in 3D.
    Returns (geom, info); info is None when the mask is one region (geom unchanged)."""
    from scipy import ndimage as _vbc_ndi
    from scipy.spatial import cKDTree as _vbc_kd
    c, pts, m = geom
    mb = np.asarray(m) > 0
    lab, n = _vbc_ndi.label(mb)
    if n <= 1:
        return geom, None
    sizes = _vbc_ndi.sum(mb, lab, index=np.arange(1, n + 1))
    x, y = (int(pixel[0]), int(pixel[1])) if pixel is not None else (-1, -1)
    k0 = int(lab[y, x]) if (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1] and lab[y, x] > 0) else int(np.argmax(sizes)) + 1
    keep = lab == k0
    p0 = mask_to_world_points(keep.astype(np.uint8), depth_img, K, E)
    if p0 is None or len(p0) < VB_MIN_DEPTH_PTS:
        return geom, {"regions": int(n), "kept": None, "why": "main_region_without_depth"}
    tree = _vbc_kd(np.asarray(p0, float)); dropped = []
    for k in range(1, n + 1):
        if k == k0:
            continue
        reg = lab == k; pk = mask_to_world_points(reg.astype(np.uint8), depth_img, K, E)
        if pk is not None and len(pk) and float(np.min(tree.query(np.asarray(pk, float))[0])) <= VB_ASSOC_M:
            keep |= reg
        else:
            dropped.append(int(sizes[k - 1]))
    info = {"regions": int(n), "dropped_px": dropped, "kept_px": int(keep.sum()), "px": int(mb.sum())}
    if not dropped:
        return geom, info
    pk = mask_to_world_points(keep.astype(np.uint8), depth_img, K, E)
    if pk is None or len(pk) < VB_MIN_DEPTH_PTS:
        info["why"] = "kept_without_depth"; return geom, info
    pk = np.asarray(pk, float)
    return (get_oriented_bounding_box_from_3d_points(pk)["center"], pk, keep.astype(np.uint8)), info
# ===== end C17I1 VBC =====


def vb_localize_for_grasp(rgb, depth_img, K, E, prompts, object_side):
    """Sub-patch 2, before grasping: the candidate associated with the frozen reference, confirmed by the VLM; else no motion."""
    R = _VB["refs"].get(_VB["active"]) if _VB["active"] is not None else None
    if R is None:   # no reference was taken for this subgoal (no VLM / not an object subgoal): the parent binding, plus sub-patch 1
        _VB["phase"] = "pregrasp"; out = localize_target(rgb, depth_img, K, E, prompts, object_side=object_side); _VB["phase"] = None
        return out
    if R["status"] != "frozen":
        vb_log("PREGRASP", handle=R["handle"], result="identity_unknown", why="reference_unresolved"); return None, None, None
    cand = None; how = None
    if R["pixel"] is not None:
        geoms, _ = vb_point_masks(rgb, depth_img, K, E, R["pixel"][0], R["pixel"][1])
        geom, iou = vb_pick_by_reference(geoms, R["mask"])   # rev2: the nested mask that is the frozen body, not the largest
        if geom is not None and (iou >= VB_REF_IOU or float(np.linalg.norm(np.asarray(geom[0], float) - R["centre"])) <= VB_ASSOC_M):
            cand, how = geom, "frozen_pixel"
            cand, _vbc = vb_body_component(cand, R["pixel"], depth_img, K, E)   # C17I1 VBC: detached regions of other bodies are dropped
            if _vbc is not None:
                vb_log("BODY_REGIONS", handle=R["handle"], **_vbc)
        vb_log("ASSOC", handle=R["handle"], n_candidates=len(geoms), best_iou=round(float(iou), 3), accepted=cand is not None)
    if cand is None:
        r = vb_point_localize(rgb, depth_img, K, E, R["phrase"], "pregrasp", exemplar=(R["rgb"], R["mask"]))
        if r is not None:
            cand, how = r[:3], "vlm_exemplar"
    if cand is None:
        vb_log("PREGRASP", handle=R["handle"], result="identity_unknown", why="not_relocalized"); return None, None, None
    c, pts, mask = cand
    conf = _VB.get("confirmed")
    if conf is not None and conf[0] == R["handle"] and float(np.linalg.norm(np.asarray(c, float) - conf[1])) <= VB_ASSOC_M:
        result = "identity_verified"   # the same confirmed body, unmoved since its confirmation in this subgoal
    elif how == "frozen_pixel" and not VB_VETO_TEXT_BINDING:
        result = "identity_verified"   # rev3: the frozen body re-found in place (majority overlap with the frozen mask) - no VLM veto
    else:
        result = vb_same_instance([(R["rgb"], R["mask"]), (rgb, mask)], R["phrase"], "pregrasp")
    vb_log("PREGRASP", handle=R["handle"], result=result, how=how, centre=[round(float(v), 4) for v in c])
    if result != "identity_verified":
        _VB["confirmed"] = None; return None, None, None
    _VB["confirmed"] = (R["handle"], np.asarray(c, float)); _VB["pregrasp_img"] = (np.asarray(rgb, np.uint8).copy(), np.asarray(mask).astype(np.uint8))
    _VB["base_z"] = float(object_base_z(pts))
    return c, pts, mask


# ===== C17I1 binding VBH: post-grasp identity question (round N17, chunk 17, iteration 1) =====
# After the lift the parent asks the VLM whether the lifted object is the frozen instance, with the reference and pre-grasp
# views TINTED magenta. Recorded (development rows, 79 post-grasp questions): the lifted body was the frozen body in 79 of
# 79 (runner truth), the VLM answered no in 14 (18 %), and each no refused the transport of the correct body (one episode lost
# both attempts this way). The recorded reasons: the tinted body was read as a different object (a magenta vessel instead
# of the tinted can), and the lifted body, raised above the table, appears higher in the picture and hides the body behind
# it, which was read as "that other object is gone, so the robot holds it". Rule (wording and images only; the verdict
# mapping, the hold / motion checks and the transport gate are unchanged): the reference and pre-grasp views carry only a
# magenta outline (vg_outline, as the gallery), one zoomed outlined crop of the reference body (vg_crop) shows its own print
# and colours, and the text says what the outline is, that a lifted object appears higher and can hide what stands behind
# it, and to judge by the lifted object's own appearance and by which resting place is now empty. At most 4 images (the
# interface limit): reference, pre-grasp (when taken), crop, current view.
def vb_held_identity(views, phrase):
    """VBH: post-grasp identity. views = [(ref_rgb, ref_mask), (pregrasp_rgb, pregrasp_mask)?, (current_rgb, None)].
    Returns identity_verified | identity_mismatch | identity_unknown (the parent's mapping of yes / no / unsure)."""
    ref_rgb, ref_mask = views[0]
    imgs = [vg_outline(r, m) if m is not None else np.asarray(r, np.uint8) for r, m in views[:-1]]
    has_pre = len(imgs) == 2
    imgs.append(vg_crop(ref_rgb, ref_mask)); imgs.append(np.asarray(views[-1][0], np.uint8))
    n = len(imgs); kc = n - 1
    text = (f"Task instruction: \"{vb_lang()}\". The object the instruction calls \"{phrase}\" was identified at the start of the task: "
            "image 1 is the camera view at that time with that instance outlined by a thin magenta line (the line is only drawn on the "
            "picture; it is not part of any object). ")
    if has_pre:
        text += "Image 2 is the view just before grasping, with the instance the robot confirmed outlined the same way. "
    text += (f"Image {kc} is a zoomed view of the outlined instance from image 1, so you can see its own print, colours and shape. "
             f"Image {n} is the current view: the robot has lifted an object between its fingers. The lifted object is held well above "
             "the table, so in the picture it appears higher (further up) than the spot where it stood, and it can hide objects that "
             "stand behind it; an object hidden behind the robot or behind the lifted object has not necessarily moved. Is the lifted "
             "object the same physical instance as the outlined one? Judge by (a) whether the lifted object's own appearance (print, "
             "colours, shape, size) matches the zoomed instance, and (b) whether the outlined instance's resting spot is now empty. "
             "Look-alike objects are different instances. Answer yes, no, or unsure.")
    a = vb_ask(imgs, text, VB_VERDICT_SCHEMA, tag="identity:post_grasp")
    v = "unsure" if a is None else str(a.get("verdict"))
    out = {"yes": "identity_verified", "no": "identity_mismatch"}.get(v, "identity_unknown")
    vb_log("IDENTITY", phase="post_grasp", phrase=phrase, result=out, answer=a, rule="VBH")
    return out
# ===== end C17I1 VBH =====


def vb_post_grasp(obs_c, hv, rl):
    """Sub-patch 2, after grasping and before transport: identity verified AND hold retained AND motion with the gripper."""
    R = _VB["refs"].get(_VB["active"]); rgb_c = obs_c["agentview"]["images"]["rgb"]
    views = [(R["rgb"], R["mask"])] + ([_VB["pregrasp_img"]] if _VB.get("pregrasp_img") is not None else []) + [(rgb_c, None)]
    identity = vb_held_identity(views, R["phrase"])   # C17I1 VBH: outlined views + zoomed reference + the lift's view effects
    retained = (hv or {}).get("status") == "verified" and (rl or {}).get("status") == "reached"
    meas = measure_held_object(list(_CUR["prompts"]), obs_c)
    rise = None if (meas is None or _VB.get("base_z") is None) else float(meas["z_min"]) - float(_VB["base_z"])
    moved = rise is not None and rise >= VB_RISE_M
    ok = identity == "identity_verified" and retained and moved
    vb_log("TRANSPORT_GATE", handle=R["handle"], identity=identity, retained=retained, moved_with_gripper=moved,
           rise_m=None if rise is None else round(rise, 4), allow=ok)
    if not ok:
        _VB["confirmed"] = None
    return ok
# ===== end VB1 binding =====


# ===== C16I1 binding: VLM choice among the text candidates (gallery) with the descriptor's phrases (round N16, chunk 16, iteration 1) =====
# Where the text binding fails at task start (ambiguous / not_found) the parent asks the VLM for one free pixel over the full
# view. In the recorded development answers the VLM then reasons from a prior about how a product "usually" looks and lands
# on a look-alike (the named box in 6 of 30 answers; a can pixel on the other can). Rule (task start, object phrases without a
# side word only): the instances the text prompts already segmented (bind_instances: one entry per body, in support order)
# are shown as a gallery - image 1 the full view (spatial context), images 2.. one magnified crop per candidate with only its
# outline drawn (the body's own colours and print stay visible) - together with every phrase of the task descriptor (its own
# name, the longer name, the appearance description). The VLM chooses one image by what is printed / visible on the body, or
# answers none.
#   (a) text binding failed: a chosen candidate at medium / high confidence is the reference (its own text mask and points);
#       none, low confidence, no candidate or no answer -> the parent's free pixel, unchanged.
#   (b) text binding succeeded but its decisive prompts disagreed (the C4I3 ballot holds votes for two or more candidates -
#       the mixed-evidence case where the name and the appearance prompt point at different bodies): the gallery is asked;
#       only a confident choice of a DIFFERENT body (mask IoU < VB_REF_IOU and centre farther than VB_ASSOC_M from the text
#       body) replaces the text body. Agreement, none, low confidence or no answer keep the text binding.
# A text binding whose decisive prompts agree is never touched, so is every side-qualified phrase (C3I1 path).
# Constants: VG_MAX_CANDIDATES = 3 = the VLM interface's 4-image limit per request (5 or more are rejected) minus the full
# view; the named body ranks 1st-3rd by support in 39/39 development task-start frames. Candidates pass the parent's own
# point-mask validity bounds (VB_MASK_MAX_FRAC, VB_MIN_DEPTH_PTS). Crop: tight padding max(VG_PAD_MIN_PX, VG_PAD_FRAC x the
# longer side) and an integer magnification towards VG_ZOOM_PX (at most VG_MAX_ZOOM): a 40-80 px body is shown at about
# 5-6x. With the parent-style padding (half the longer side) and 240 px the can labels were reported illegible (DIAGNOSIS.md
# section 4). Same-body test = the stack's VB_REF_IOU / VB_ASSOC_M association criteria.
VG_MAX_CANDIDATES = 3
VG_ZOOM_PX = 480
VG_MAX_ZOOM = 6
VG_PAD_FRAC = 0.15
VG_PAD_MIN_PX = 8


def vg_outline(rgb, mask):
    """Public RGB with only the mask's outline drawn (2 px, magenta)."""
    im = np.asarray(rgb, np.uint8).copy(); m = np.asarray(mask).astype(bool)
    if m.shape == im.shape[:2] and m.any():
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
        im[edge] = VB_TINT
    return im


def vg_crop(rgb, mask):
    """Magnified crop around one candidate with its outline (integer magnification, nearest pixel)."""
    H, W = np.asarray(rgb).shape[:2]; ys, xs = np.nonzero(np.asarray(mask))
    pad = max(VG_PAD_MIN_PX, int(VG_PAD_FRAC * max(int(xs.max() - xs.min()), int(ys.max() - ys.min()))))
    x0, y0, x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(W, int(xs.max()) + pad + 1), min(H, int(ys.max()) + pad + 1)
    c = vg_outline(rgb, mask)[y0:y1, x0:x1]
    f = int(min(VG_MAX_ZOOM, max(1, round(VG_ZOOM_PX / float(max(c.shape[:2]))))))
    return np.repeat(np.repeat(c, f, axis=0), f, axis=1)


def vg_candidates(rgb, depth_img, K, E, prompts):
    """The text candidates: bind_instances entries (support order) whose mask passes the point-mask validity bounds."""
    H, W = np.asarray(rgb).shape[:2]; out = []
    for I in bind_instances(rgb, depth_img, K, E, list(prompts)):
        if int(I["n_px"]) > VB_MASK_MAX_FRAC * H * W or len(I["pts"]) < VB_MIN_DEPTH_PTS:
            continue
        out.append(I)
        if len(out) >= VG_MAX_CANDIDATES:
            break
    return out


def vg_same_body(a, b):
    """Two (centre, pts, mask) geometries are the same body: majority mask overlap or centres within the association radius."""
    return vb_mask_iou(a[2], b[2]) >= VB_REF_IOU or float(np.linalg.norm(np.asarray(a[0], float) - np.asarray(b[0], float))) <= VB_ASSOC_M


def vg_gallery_localize(rgb, depth_img, K, E, prompts, phrase, phase):
    """One VLM call chooses the named object among the text candidates. Returns (centre, pts, mask) or None."""
    if not vb_enabled():
        return None
    cands = vg_candidates(rgb, depth_img, K, E, prompts)
    if not cands:
        vb_log("GALLERY", phase=phase, phrase=phrase, n_candidates=0, accepted=False); return None
    labels = [str(i + 2) for i in range(len(cands))]
    others = [str(p) for p in list(prompts) if str(p) != str(phrase)]
    hint = "" if not others else " (the task also describes it as: " + "; ".join("\"" + p + "\"" for p in others) + ")"
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": labels + ["none"]},
              "confidence": {"type": "string", "enum": ["low", "medium", "high"]}, "reason": {"type": "string"}},
              "required": ["choice", "confidence", "reason"], "additionalProperties": False}
    text = (f"Task instruction: \"{vb_lang()}\". The robot must pick up the object the instruction calls \"{phrase}\"{hint}. "
            f"Image 1 is the full camera view. Images {labels[0]}{'' if len(labels) == 1 else '-' + labels[-1]} each show one candidate "
            "object from that view, zoomed in, with its outline drawn in magenta. Identify the object by what you can actually see on "
            "it: read the words and pictures printed on its label or packaging, and compare its colours and shape with the "
            "descriptions above. Do not rely on assumptions about how such a product usually looks in a simulator or dataset; "
            "several candidates may be similar packages, and only one is the named object. Which image's outlined object is the "
            f"\"{phrase}\"? Answer with that image number, or none if no candidate is it or you cannot tell, and give your confidence.")
    images = [np.asarray(rgb, np.uint8)] + [vg_crop(rgb, I["mask"]) for I in cands]
    a = vb_ask(images, text, schema, tag=f"gallery:{phase}")
    ch = None if a is None else str(a.get("choice")); conf = None if a is None else a.get("confidence")
    ok = ch in labels and conf in VB_ACCEPT_CONF
    vb_log("GALLERY", phase=phase, phrase=phrase, n_candidates=len(cands), choice=ch, confidence=conf, accepted=ok, answer=a,
           supports=[round(float(I["support"]), 4) for I in cands], areas=[int(I["n_px"]) for I in cands])
    if not ok:
        return None
    I = cands[labels.index(ch)]; pts = np.asarray(I["pts"], float)
    return get_oriented_bounding_box_from_3d_points(pts)["center"], pts, np.asarray(I["mask"]).astype(np.uint8)
# ===== end C16I1 binding =====


# ===== H2 A: approach orientation (parallel-jaw 180 deg symmetry) + planned approach budget =====
H_APPROACH_CAP = 600
def _flip_tool_z(q):
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() @ np.diag([-1.0, -1.0, 1.0])
    x = Rotation.from_matrix(R).as_quat()
    return np.array([x[3], x[0], x[1], x[2]])
def h_select_approach(gpos, gquat):
    ap = np.array([gpos[0], gpos[1], gpos[2] + 0.12])
    cands = [np.asarray(gquat, float), _flip_tool_z(np.asarray(gquat, float))]
    plans = [move_to_pose(ap, q, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=f"sg1:approach_plan{k}") for k, q in enumerate(cands)]
    ok = [(int(p["steps_planned"]), k) for k, p in enumerate(plans) if p.get("detail") == "plan_only" and p.get("steps_planned") is not None]
    if not ok:
        print("APPROACH_PLAN " + json.dumps({"planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": None, "budget": BUDGET_FREE}), flush=True)
        return cands[0], BUDGET_FREE
    steps, k = min(ok)
    budget = int(min(H_APPROACH_CAP, max(BUDGET_FREE, int(np.ceil(1.25 * steps)))))
    print("APPROACH_PLAN " + json.dumps({"planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": k, "budget": budget}), flush=True)
    return cands[k], budget
# ===== end H2 A =====

def ga_tilt_ik_ok(position, quat):
    """Only an expected, verified IK refusal is a negative probe."""
    try:
        joints = np.asarray(solve_ik(position, quat), float)
    except RuntimeError as error:
        if str(error).startswith("IK ik_failed:") or str(error).startswith("IK out_of_workspace:"):
            return False
        raise
    return joints.shape == (7,) and bool(np.all(np.isfinite(joints)))


def ga_tilt_clearance(position, quat, target_points, obstacles, floor_z):
    """Nominal open fingers and housing clear the measured vertical approach column."""
    rt = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
    rm = rt @ OP_TOOL_FROM_MODEL.T
    fingers, housing = gw_hand_boxes()
    if float(position[2]) - CONTACT_OVERSHOOT + op_box_low_z(rm, fingers + housing) < float(floor_z) + CA_FLOOR_CLEAR:
        return False
    heights = list(np.arange(float(position[2]) + 0.12, float(position[2]), -CA_DESCENT_STEP)) + [float(position[2])]
    for height in heights:
        tcp = [float(position[0]), float(position[1]), height]
        if gw_box_gap(obstacles, tcp, rm, fingers + housing) < CA_CLEAR_MARGIN:
            return False
        # Target contact after the nominal goal remains governed by contact_move.
        if gw_box_gap(target_points, tcp, rm, fingers + housing) < CA_CLEAR_MARGIN:
            return False
    return True


def ga_reachable_body_quat(position, quat):
    """Bounded jaw-axis tilt only after both original vertical endpoints fail IK."""
    plan = _MF.get("last_plan") or {}
    target = plan.get("target")
    if plan.get("mode") == "handle_pinch" and plan.get("tilt_deg") is not None and target is not None:   # C13I4: a tilted protrusion pinch (gt_tilt_section)
        return gt_tilted_quat(position, quat, plan)
    if plan.get("mode") != "body_pinch" or target is None or _WG.get("pts") is None:
        return quat
    if not np.allclose(np.asarray(position, float), np.asarray(target, float), rtol=0.0, atol=1e-6):
        return quat
    rot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
    if float(rot.as_matrix()[2, 2]) > -0.999:
        return quat
    if ga_tilt_ik_ok(position, quat) or ga_tilt_ik_ok(position, _flip_tool_z(quat)):
        return quat
    profile = plan.get("profile_summary") or {}
    radius = max(float(profile.get("r_top", 0.0)), float(profile.get("r_mid", 0.0)))
    engagement = float(plan.get("pad_engaged_m") or 0.0)
    angle = np.radians(15.0)
    if radius <= 0.0 or engagement * np.cos(angle) - radius * np.sin(angle) < GW_ENGAGE_MIN:
        return quat
    obstacles, floor_z = gw_obstacles(_WG["pts"])
    if obstacles is None or floor_z is None:
        return quat
    for degrees in (-15.0, 15.0):
        xyzw = (rot * Rotation.from_euler("y", degrees, degrees=True)).as_quat()
        candidate = xyzw[[3, 0, 1, 2]]
        if not ga_tilt_clearance(position, candidate, _WG["pts"], obstacles, floor_z):
            continue
        approach = np.asarray(position, float) + np.array([0.0, 0.0, 0.12])
        overshoot = np.asarray(position, float) - np.array([0.0, 0.0, CONTACT_OVERSHOOT])
        if not all(ga_tilt_ik_ok(p, candidate) for p in (approach, position, overshoot)):
            continue
        print("GRASP_TILT " + json.dumps({"degrees": degrees, "reason": "vertical_endpoint_unreachable", "grasp_xyz_preserved": True, "closing_axis_preserved": True}), flush=True)
        return candidate
    return quat


# ===== C13I3 manipulation_primitive: pick from a raised body (N13 group L, slot C13I3) =====
# Recorded (development rows, DIAGNOSIS.md sections 1-3): a body resting on a raised support (measured top 0.196 m) is planned a
# sound grasp (approach IK solved); the parent's free (joint-interpolated) approach from home then stalls in 20/20 strategy-0
# approaches with the TCP 3-7 cm below both endpoint heights at the raised body's near edge (approach_plan1's ik_failed is the
# plan-only probe of the 180 deg-flipped jaw, refused at the wrist joint limit; plan 0 was chosen every time). With a measured
# clearance route (raise / level traverse / lower; N12 slot C12I3) the same approaches reached 20/20, the grasp closed on the
# body in 18/20 and the hold was verified after the lift in 16/20 - and then the transport gate refused all 16 because the
# fixed 0.25 m lift had raised the body only 0.023-0.028 m < VB_RISE_M = 0.05.
# (1) rr_approach: the parent's free move first, unchanged. A `stalled` result (it moved, it did not arrive) is the evidence that
#     the free path is obstructed; only then is the parent's own non-arrival recovery (goto_home) taken before, not after, a
#     second approach on the measured route: h_max = highest seen static surface under the hand's straight horizontal path
#     (corridor half-width CA_HAND body_half_wide = the hand's largest horizontal half-extent about the TCP; the subgoal's
#     observe view in the transport map _c_height_map; robot strip C_ARM_STRIP and the target's own disk r + C_EXCL_EXTRA
#     excluded, as in c_transport_height); z_path = max(z_approach, h_max + C_CLEAR) <= C_Z_MAX; raise at the current xy
#     (orientation held), traverse at z_path in the grasp orientation, lower to the approach pose; all linear with the
#     program's BUDGET_LINEAR / BUDGET_TRANSPORT and the parent's tolerance 0.015. No seen corridor, z_path above C_Z_MAX,
#     or any non-stalled parent result: the parent's result is returned. A stage refused or stalled is the approach result,
#     so the parent's approach_nonarrival recovery is unchanged.
# (2) rr_lift_z: the transport gate needs the held body's base raised by VB_RISE_M; the parent lifts the TCP to the fixed
#     RR_PARENT_LIFT_Z. When RR_PARENT_LIFT_Z - z_grasp < VB_RISE_M that lift cannot satisfy the gate (the measured rise is
#     at most the commanded one: recorded 0.4-0.6 cm below it), so the lift goes to z_grasp + RR_APPROACH_DZ, the approach
#     pose height, reached on the way in with the same orientation. Otherwise the parent's lift, unchanged.
RR_PARENT_LIFT_Z = 0.25     # m, the parent's fixed lift height of attempt_grasp_verify (unchanged; prim_place keeps it as lift_z)
RR_APPROACH_DZ = 0.12       # m, the parent's approach offset above the grasp point (attempt_grasp_verify, h_select_approach)


def rr_target_disk(target_pts):
    """(centre_xy, radius) of the target's footprint from its visible points (median xy; p98 xy distance); None without points."""
    if target_pts is None:
        return None
    P = np.asarray(target_pts, float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) == 0:
        return None
    c = np.median(P[:, :2], axis=0)
    return c, float(np.percentile(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]), 98))


def rr_corridor_height(start_xy, goal_xy, target_pts):
    """Highest seen static surface under the hand's straight horizontal path start_xy -> goal_xy in the last stashed view
    (robot strip and the target's own disk excluded). Returns (h_max | None, info)."""
    info = {"n_views": 0, "h_max": None, "cells": 0, "seen": 0, "excluded_target": 0, "excluded_strip": 0}
    if not _C_VIEWS:
        return None, info
    view_xy, obs = _C_VIEWS[-1]
    H = _c_height_map(obs); CX, CY = _c_grid()
    corridor = _c_seg_dist(CX, CY, np.asarray(start_xy, float)[:2], np.asarray(goal_xy, float)[:2]) < float(CA_HAND["body_half_wide"])
    strip = _c_seg_dist(CX, CY, np.zeros(2), np.asarray(view_xy, float)[:2]) < C_ARM_STRIP
    disk = rr_target_disk(target_pts); excl = np.zeros(corridor.shape, bool)
    if disk is not None:
        excl = np.hypot(CX - disk[0][0], CY - disk[0][1]) <= disk[1] + C_EXCL_EXTRA
    cells = corridor & ~strip & ~excl
    vals = H[cells]; ok = vals[~np.isnan(vals)]
    h = float(ok.max()) if len(ok) else None
    info.update(n_views=1, h_max=None if h is None else round(h, 4), cells=int(corridor.sum()), seen=int(len(ok)),
                excluded_target=int((corridor & excl).sum()), excluded_strip=int((corridor & strip).sum()),
                target_r=None if disk is None else round(disk[1], 4))
    return h, info


def rr_route(ap, h_max):
    """Pure decision after a stalled free approach to pose ap: ('keep', why, z | None) or ('routed', why, z_path)."""
    if h_max is None:
        return "keep", "no_seen_corridor", None
    z_path = max(float(ap[2]), float(h_max) + C_CLEAR)
    if z_path > C_Z_MAX:
        return "keep", "clearance_above_cap", z_path
    return "routed", "stalled_free_approach", z_path


def rr_approach(ap, q, budget):
    """The grasp approach to pose (ap, q): the parent's free move; after a `stalled` free move, goto_home and the measured
    raise / traverse / lower route (rule above). Returns the move_to_pose result of the last executed move."""
    ap = np.asarray(ap, float); q = np.asarray(q, float)
    r = move_to_pose(ap, q, mode="free", tol_pos=0.015, step_budget=budget, label="sg1:approach")
    if r.get("status") != "stalled":
        return r
    goto_home_joint_position()   # the parent's own recovery from an approach non-arrival, taken before the route
    p0, q0, _ = read_tcp(); p0 = np.asarray(p0, float); q0 = np.asarray(q0, float)
    h, info = rr_corridor_height(p0[:2], ap[:2], _WG.get("pts"))
    decision, why, z_path = rr_route(ap, h)
    rec = {"free": f"{r.get('status')}/{r.get('detail')}", "decision": decision, "why": why, "z_path": None if z_path is None else round(float(z_path), 4),
           "start": [round(float(v), 4) for v in p0], "approach": [round(float(v), 4) for v in ap], "clear": C_CLEAR, **info}
    if decision != "routed":
        rec["stages"] = []
        print("APPROACH_REROUTE " + json.dumps(rec), flush=True)
        return r
    plan = []
    if z_path > float(p0[2]) + 0.005:
        plan.append(("raise", np.array([p0[0], p0[1], z_path]), q0, BUDGET_LINEAR))
    plan.append(("traverse", np.array([ap[0], ap[1], z_path]), q, BUDGET_TRANSPORT))
    if z_path > float(ap[2]) + 0.005:
        plan.append(("lower", ap.copy(), q, BUDGET_LINEAR))
    stages = []
    for name, target, qq, bud in plan:
        r = move_to_pose(target, qq, mode="linear", tol_pos=0.015, step_budget=bud, label=f"sg1:approach_{name}")
        _PRIM_STATS["motion_calls"] += 1
        stages.append(f"{name}:{r.get('status')}")
        if r.get("status") != "reached":   # refused or stalled: the approach result (the caller's approach_nonarrival recovery)
            break
    rec["stages"] = stages
    print("APPROACH_REROUTE " + json.dumps(rec), flush=True)
    return r


def rr_lift_z(gpos):
    """(lift TCP height, rule): the parent's RR_PARENT_LIFT_Z unless it raises the body less than VB_RISE_M above the grasp
    point; then the approach pose height z_grasp + RR_APPROACH_DZ."""
    gz = float(np.asarray(gpos, float)[2])
    if RR_PARENT_LIFT_Z - gz >= VB_RISE_M:
        return RR_PARENT_LIFT_Z, "parent"
    return gz + RR_APPROACH_DZ, "approach_height"
# ===== end C13I3 manipulation_primitive =====


# ===== C13I4 manipulation_primitive: the axial approach / retreat of a stage-2 tilted protrusion pinch (N13, slot C13I4) =====
# Used only when the current plan (_MF["last_plan"]) is a handle_pinch with approach "axial" (gt_tilt_section stage 2), the grasp position
# is that plan's target and the commanded tool axis is the plan's (a 180-deg jaw flip keeps it). Then the approach pose is GW_APPROACH_UP
# back along the tool axis, the contact descent runs along the axis (contact_move with approach_axis = a, as the program's push and pull
# primitives already command), backoffs go to that approach pose, and the hand, holding, first retreats linearly to it before the
# parent's lift. Every other grasp: gt_axial_axis returns None and attempt_grasp_verify runs the parent's code unchanged.
def gt_axial_axis(gpos, gquat):
    """The unit tool axis (toward the tips) of an axial-approach protrusion pinch at its own target, else None."""
    try:
        plan = _MF.get("last_plan") or {}
        if plan.get("mode") != "handle_pinch" or plan.get("approach") != "axial" or plan.get("approach_axis") is None or plan.get("target") is None:
            return None
        if not np.allclose(np.asarray(gpos, float).reshape(-1)[:3], np.asarray(plan["target"], float).reshape(-1)[:3], rtol=0.0, atol=1e-6):
            return None
        a = np.asarray(plan["approach_axis"], float); a = a / float(np.linalg.norm(a)); z = _R(np.asarray(gquat, float))[:, 2]
        if not np.allclose(z, a, rtol=0.0, atol=1e-6):
            return None
        return a
    except Exception:  # noqa: BLE001  (any error: the parent's vertical path)
        return None


def gt_approach_point(gpos, a):
    """The approach pose: the parent's vertical point, or GW_APPROACH_UP back along the tool axis a."""
    gpos = np.asarray(gpos, float)
    if a is None:
        return np.array([gpos[0], gpos[1], gpos[2] + 0.12])
    return gpos - GW_APPROACH_UP * np.asarray(a, float)


def gt_select_approach(gpos, gquat, a):
    """h_select_approach (H2 A) with the plan-only probes at the axial approach pose: (quat, budget)."""
    ap = gt_approach_point(gpos, a)
    cands = [np.asarray(gquat, float), _flip_tool_z(np.asarray(gquat, float))]
    plans = [move_to_pose(ap, q, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=f"sg1:approach_plan{k}") for k, q in enumerate(cands)]
    ok = [(int(p["steps_planned"]), k) for k, p in enumerate(plans) if p.get("detail") == "plan_only" and p.get("steps_planned") is not None]
    if not ok:
        print("APPROACH_PLAN " + json.dumps({"planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": None, "budget": BUDGET_FREE, "axial": True}), flush=True)
        return cands[0], BUDGET_FREE
    steps, k = min(ok)
    budget = int(min(H_APPROACH_CAP, max(BUDGET_FREE, int(np.ceil(1.25 * steps)))))
    print("APPROACH_PLAN " + json.dumps({"planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": k, "budget": budget, "axial": True}), flush=True)
    return cands[k], budget


def gt_axial_descend(grasp_pos, approach_pos, q, a, label):
    """mc_grasp_descend along the tool axis a instead of straight down (same tolerances, corridor and budget)."""
    r = contact_move(np.asarray(grasp_pos, float), np.asarray(q, float), approach_axis=[float(v) for v in a], contact_tol_before=GRIPPER_CONTACT_TOL, overshoot_after=CONTACT_OVERSHOOT,
                     start_pose=np.asarray(approach_pos, float), start_tol=0.02, gripper_expect="open", lateral_corridor=APPROACH_CORRIDOR, step_budget=BUDGET_CONTACT, label=label)
    _PRIM_STATS["motion_calls"] += 1
    print("CONTACT_MOVE " + json.dumps({"label": label, "axial": True, "status": r["status"], "detail": r.get("detail"), "success": r["success"], "steps": r["steps_used"], "budget": r["budget"], "wall_s": r.get("wall_s"),
                                        "along_cm": round((r.get("along_final") or 0) * 100, 2), "travel_cm": round((r.get("travel") or 0) * 100, 2), "window_cm": [round(x * 100, 2) for x in (r.get("window") or [])],
                                        "pos_err_cm": round(r.get("pos_err", 9) * 100, 2), "evidence": r.get("evidence"), "verifier": verify_contact(r).get("status")}), flush=True)
    return r


def gt_axial_retreat(ap, q):
    """Holding: the linear retreat back along the approach axis to the approach pose (the segment the planner checked), before the lift."""
    r = move_to_pose(np.asarray(ap, float), np.asarray(q, float), mode="linear", tol_pos=0.015, step_budget=BUDGET_LINEAR, label="sg1:lift_axial")
    _PRIM_STATS["motion_calls"] += 1
    print("LIFT_AXIAL " + json.dumps({"status": r.get("status"), "detail": r.get("detail"), "pos_err_cm": round(float(r.get("pos_err", 9)) * 100, 2)}), flush=True)
    return r
# ===== end C13I4 manipulation_primitive =====


def attempt_grasp_verify(depth, K, mask, E, pud_c, use_topdown=False, yaw=0, grasp_z=None, grasp_xy=None):
    """Execute one grasp and verify. Returns (success, gpos, gquat). (identical to P0)"""
    _WG["pts"] = mask_to_world_points(mask, depth, K, E)   # W: the target's visible points for the width gate
    if use_topdown:
        gquat = make_topdown_quat(yaw)
        gz = grasp_z if grasp_z is not None else pud_c[2]
        gxy = grasp_xy if grasp_xy is not None else pud_c[:2]   # Amendment 2: the plan-derived strategy uses the plan's xy (a rim pinch targets the wall, not the axis)
        gpos = np.array([gxy[0], gxy[1], gz])
    else:
        grasp_poses, grasp_scores = plan_grasp(depth, K, mask)
        if grasp_poses is None or len(grasp_poses) == 0:
            return False, None, None
        best_g, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
        if best_g is None:
            best_g = E @ grasp_poses[grasp_scores.argmax()]
        gpos, gquat = decompose_transform(best_g)

    if use_topdown:
        gquat = ga_reachable_body_quat(gpos, gquat)
    _feas = g1_descent_feasible(gpos, gpos[2] + 0.12)
    if not _feas["ok"]:   # G1 I2: infeasible candidate filtered BEFORE any motion (no open, no approach); the strategy loop continues
        print(f"GRASP_ABORT reason=candidate_infeasible: {_feas['why']}", flush=True)
        return False, gpos, gquat
    open_gripper()
    _ax = gt_axial_axis(gpos, gquat)   # C13I4: the tool axis of an axial-approach protrusion pinch (gt_tilt_section stage 2), else None
    if _ax is None:
        gquat, _abud = h_select_approach(gpos, gquat)
    else:
        gquat, _abud = gt_select_approach(gpos, gquat, _ax)
    _ap = gt_approach_point(gpos, _ax)   # C13I4: the parent's vertical approach point, or GW_APPROACH_UP back along the tilted tool axis
    _WG["quat"] = np.asarray(gquat, float)   # W: actual descent orientation (closing axis)
    ra = rr_approach(_ap, gquat, _abud)   # C13I3: the parent free move; only after it stalls, goto_home + raise / traverse / lower over the measured corridor
    if ra["status"] != "reached":
        print("GRASP_MOTION " + json.dumps({"approach": ra["status"], "descent": None, "detail": ra.get("detail"), "close": False, "why": "approach_nonarrival"}), flush=True)
        print(f"GRASP_ABORT reason=approach_nonarrival: {ra['status']}/{ra.get('detail')}", flush=True)
        goto_home_joint_position()
        return False, gpos, gquat
    rg = mc_grasp_descend(gpos, _ap, gquat, "sg1:grasp_descent") if _ax is None else gt_axial_descend(gpos, _ap, gquat, _ax, "sg1:grasp_descent")   # MC: contact-aware grasp descent
    mf_record_grasp(rg)   # F: predicted vs actual contact height
    ok_close, why = gate_allow(rg, "close")
    print("GRASP_MOTION " + json.dumps({"approach": ra["status"], "descent": rg["status"], "detail": rg.get("detail"), "pos_err_cm": round(rg.get("pos_err", 9) * 100, 2), "lateral_cm": round(rg.get("lateral_max", 0) * 100, 2), "close": ok_close, "why": why}), flush=True)
    if not ok_close:
        print(f"GRASP_ABORT reason={why}", flush=True)
        move_to_pose(_ap, gquat, mode="free", tol_pos=0.02, step_budget=BUDGET_FREE, label="sg1:backoff")
        goto_home_joint_position()
        return False, gpos, gquat
    gv = mc_close_and_verify(rg, "sg1")
    _E1_LAST.update({"grasp_after": None, "hold": None})
    if gv["status"] != "verified":
        print(f"GRASP_ABORT reason=grasp_not_verified: {gv.get('reason')}", flush=True)
        open_gripper(); gate_record("open", role="recovery_open")
        move_to_pose(_ap, gquat, mode="free", tol_pos=0.02, step_budget=BUDGET_FREE, label="sg1:backoff")
        goto_home_joint_position()
        return False, gpos, gquat

    _okl, _whyl = e1_lift_allowed(gv)
    if not _okl:   # E1 G1: lift refused -> existing not-verified recovery path (open, backoff, home); no lift
        print(f"GRASP_ABORT reason=lift_refused: {_whyl}", flush=True)
        open_gripper(); gate_record("open", role="recovery_open")
        move_to_pose(_ap, gquat, mode="free", tol_pos=0.02, step_budget=BUDGET_FREE, label="sg1:backoff")
        goto_home_joint_position()
        return False, gpos, gquat
    _lz, _lzr = rr_lift_z(gpos)   # C13I3: 0.25 as before unless it raises the body less than VB_RISE_M; then the approach height
    if _lzr != "parent":
        print("LIFT_HEIGHT " + json.dumps({"rule": _lzr, "z_grasp": round(float(gpos[2]), 4), "lift_z": round(float(_lz), 4), "parent_lift_z": RR_PARENT_LIFT_Z, "rise_needed": VB_RISE_M}), flush=True)
    if _ax is not None:   # C13I4: holding, back out along the approach axis to the approach pose first (the planned, checked retreat)
        gt_axial_retreat(_ap, gquat)
    rl = move_to_pose(np.array([gpos[0], gpos[1], _lz]), gquat, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label="sg1:lift"); gate_record("lift")
    hv = lt_verify_hold(gv, gripper_opening())   # N15 C15I2: LT2 min_opening of this grasp (else the parent call)
    _E1_LAST.update({"grasp_after": gv["after"], "hold": hv["status"], "min_opening": gv.get("lt_min_opening")})
    print("LIFT_MOTION " + json.dumps({"status": rl["status"], "pos_err_cm": round(rl["pos_err"] * 100, 2), "hold": hv["status"]}), flush=True)
    if hv["status"] != "verified":
        print("GRASP_ABORT reason=object_lost_after_lift", flush=True)
        open_gripper(); gate_record("open", role="recovery_open"); goto_home_joint_position()
        return False, gpos, gquat
    if not e1_arrived(rl, "transport"):   # E1 G2/G4: transport depends on the lift arrival; object retained, existing retreat path, strategy loop ends
        print("PLACE_RESULT " + json.dumps({"sg": "sg1", "status": "failed", "reason": f"lift_{rl['status']}"}), flush=True)
        safe_retreat(sg="sg1", q_hold=gquat); _E1_LAST["stop"] = True
        return False, gpos, gquat

    obs_c = get_observation()
    cam_c = obs_c["agentview"]
    rgb_c = cam_c["images"]["rgb"]
    d_c = cam_c["images"]["depth"]
    di_c = d_c[:, :, 0] if len(d_c.shape) == 3 else d_c

    if vb_reference_active():   # VB1: transport needs identity_verified, the verified hold and motion with the gripper
        if vb_post_grasp(obs_c, hv, rl):
            return True, gpos, gquat
        open_gripper()
        goto_home_joint_position()
        return False, gpos, gquat
    chk, _, _ = bind_post_lift(rgb_c, di_c, cam_c["intrinsics"], cam_c["pose_mat"],
        list(_CUR["prompts"]), source_reference=_BIND.get("reference"),
        hold_status=hv.get("status"), lift_status=rl.get("status"))
    print("RECHECK " + json.dumps({"bind": _BIND["last"], "z": None if chk is None else round(float(chk[2]), 4), "held": bool(chk is None or chk[2] > 0.10)}), flush=True)

    if chk is not None and chk[2] > 0.10:
        return True, gpos, gquat
    elif chk is None:
        return True, gpos, gquat  # not found = assume grasped
    else:
        open_gripper()
        goto_home_joint_position()
        return False, gpos, gquat

# ===== K2 place_primitive: relation-aware placement target (round rsi_subset, iteration 2) =====
# The adapter provides the relation as DATA ("on": the object goes onto the support; "beside": next to it, on the side its side
# word selects). The side axis is the base-frame axis along which the side words order support instances
# (localize_support_by_side sorts on index 1; "max" = +, "min" = -). A beside target is derived from measured geometry only:
#     target = fitted support centre + sign(side) * (r_support + r_object + BESIDE_CLEARANCE) along the side axis
# with r_support = rim radius of the fitted support (pv_support_from_points) and r_object = the object's profile radius
# (gg_object_profile, the same max(r_top, r_mid) the contract uses), so the whole footprint lies outside the rim on the requested
# side; the nominal target then passes the shared feasible-region validation (mf_validate_beside_target: bounded outward shift
# only). BESIDE_CLEARANCE = 0.010 m = the stack's documented 1-sigma XY error of a centre estimate from visible-surface points
# (grasp_geometry PERCEPTION xy_sigma). In the iteration-1 recordings the along-side error of a released object relative to its
# target is <= 0.16 cm (n = 10) and the fitted support centre is within 0.09 cm of the true centre along that axis (n = 26);
# the profile radius of a box-shaped body exceeds its along-side half-width by ~1.5 cm (proposals/iter_2/DIAGNOSIS.md).
SIDE_AXIS = 1
BESIDE_CLEARANCE = 0.010
def relation_geometry(td, select, E=None, ref_xy=None):   # C10I1: E / ref_xy = observer pose and a reference point (depth side codes of beside)
    """(contract relation, region direction | None) from the adapter data. Only the datum "beside" with a side selection of
    "min" / "max" yields the region form; any other value, or an absent datum, is the on-support relation (parent behaviour)."""
    rel = str((td or {}).get("relation") or "").strip().lower(); sel = str(select).strip().lower()
    if rel == "in":
        return "in_support", None   # C1I2: containment (a side code with "in" is refused by the dispatch before this point)
    if rel == "in_part" and sel in CP_SIDES:
        return "in_part_support", sel   # C3I3: containment into the bay the side word names (the code is the region datum; other codes are refused by the dispatch)
    if rel == "in_sliding" and sel in CS_SIDES:
        return "in_sliding_support", sel   # C6I3: containment into the open sliding part the side word names (the code is the region datum; other codes are refused by the dispatch)
    if rel == "beside" and sel in BESIDE_DEPTH_CODES:   # C10I1: region along the observer's depth axis (None when undefined: run_subgoal stops)
        return "beside_support", beside_depth_dir(E, ref_xy, sel)
    if rel != "beside" or sel not in ("min", "max"):
        return "on_support", None
    d = np.zeros(2); d[SIDE_AXIS] = 1.0 if sel == "max" else -1.0
    return "beside_support", d
def place_target(relation, region_dir, support_c, support_pts, obj_pts, E, others_xy=()):
    """Placement target xy for the relation. on_support: the support centre exactly as before. beside_support: outside the rim
    on the requested side (rule above). in_support (C1I2): the interior centre of the container (container_geometry on the bound
    support's points), shifted along its long axis only to clear an object already placed inside (others_xy). Returns (x, y, info)."""
    if relation == "in_part_support":   # C3I3: the bay the side code (region_dir) names on the bound support's points
        halves = object_footprint_halves(obj_pts)   # identical to the gate's object footprint
        g = bay_geometry(np.asarray(support_pts, float), _CG.get("floor_z"), region_dir, E, obj_half=halves)
        if not g.get("ok"):   # cannot happen after the gate accepted this candidate (same function, same points, same code); logged fallback
            info = {"relation": relation, "rule": "support_centre_fallback", "why": g.get("why"), "target": [float(support_c[0]), float(support_c[1])]}
            print("PLACE_TARGET " + _mfj(info), flush=True)
            return float(support_c[0]), float(support_c[1]), info
        bay = g["bay"]; xy, _bfi = bf_face_centre(support_pts, _CG.get("floor_z"), g, bay)   # C13I1: the mid-point of the measured wall faces (the cell centre is up to half a cell off)
        print("BAY_FACES " + _mfj({"code": g["code"], "target": [round(float(v), 4) for v in xy], **_bfi}), flush=True)
        info = {"relation": relation, "rule": "bay_centre", "code": g["code"], "target": [float(xy[0]), float(xy[1])], "outline_centre_xy": g["centre_xy"], "outline_half_extent": g["half_extent"],
                "n_bays": g["n_bays"], "axis": g["axis"], "scores": g["scores"], "gap": g["gap"], "bay_half_extent": bay["half_extent"], "bay_n_cells": bay["n_cells"], "rim_z": bay["rim_z"], "top": g["top"],
                "rest_z": bay["rest_z"], "rest_src": bay["rest_src"], "depth": g["depth"], "z_low": g["z_low"], "z_wall": g["z_wall"], "free_max": g.get("free_max"), "wall_min": g.get("wall_min"), "object_halves": halves, "floor_z": _CG.get("floor_z"), "dirs": g.get("dirs"),
                "support_geom": {"center_xy": [float(xy[0]), float(xy[1])], "r": float(min(bay["half_extent"])), "rest_z": float(bay["rest_z"]), "table_z": float(_CG["floor_z"])}}
        info["opening"] = {"rim_z": float(bay["rim_z"]), "centre_xy": [float(xy[0]), float(xy[1])], "inner_half_extent": [float(bay["half_extent"][0]), float(bay["half_extent"][1])], "centre_src": BF_SRC}   # what the hand must pass to go below the bay's rim (place_hand_limit)
        print("PLACE_TARGET " + _mfj(info), flush=True)
        return float(xy[0]), float(xy[1]), info
    if relation == "in_sliding_support":   # C6I3: the open sliding part of the bound assembly (region_dir = the side code)
        r_obj = object_footprint_radius(obj_pts, E)   # identical to the gate's / the contract's object radius
        g = sliding_geometry(np.asarray(support_pts, float), _CG.get("floor_z"), region_dir, r_object=r_obj)
        if not g.get("ok"):   # cannot happen after the gate accepted this assembly (same function, same points, same code); logged fallback
            info = {"relation": relation, "rule": "support_centre_fallback", "why": g.get("why"), "target": [float(support_c[0]), float(support_c[1])]}
            print("PLACE_TARGET " + _mfj(info), flush=True)
            return float(support_c[0]), float(support_c[1]), info
        xy = g["centre_xy"]
        info = {"relation": relation, "rule": "sliding_interior_centre", "code": g["code"], "target": [float(xy[0]), float(xy[1])], "half_extent": g["half_extent"],
                "region_rule": g["rule"], "rim_z": g["rim_z"], "top": g.get("top"), "rest_z": g["rest_z"], "rest_src": g["rest_src"], "depth": g["depth"],
                "n_open_parts": g["n_open_parts"], "levels": g.get("levels"), "chosen": g.get("chosen"), "gap": g.get("gap"), "plane_frac": g.get("plane_frac"),
                "n_walled": g.get("n_walled"), "r_inside": g["r_inside"], "r_object": r_obj, "floor_z": _CG.get("floor_z"), "interior": g.get("interior"),
                "support_geom": {"center_xy": [float(xy[0]), float(xy[1])], "r": float(g["r_inside"]), "rest_z": float(g["rest_z"]), "table_z": float(_CG["floor_z"])}}
        info["opening"] = {"rim_z": float(g["rim_z"]), "centre_xy": [float(xy[0]), float(xy[1])], "inner_half_extent": [float(g["half_extent"][0]), float(g["half_extent"][1])]}   # C2I4: what the hand must pass to go below the rim
        print("PLACE_TARGET " + _mfj(info), flush=True)
        return float(xy[0]), float(xy[1]), info
    if relation == "in_support":
        r_obj = object_footprint_radius(obj_pts, E)   # C1I4: identical to the gate's / the contract's object radius
        g = container_geometry(np.asarray(support_pts, float), _CG.get("floor_z"), r_object=r_obj)
        if not g.get("ok"):   # cannot happen after the gate accepted this candidate (same function, same points); logged fallback
            info = {"relation": relation, "rule": "support_centre_fallback", "why": g.get("why"), "target": [float(support_c[0]), float(support_c[1])]}
            print("PLACE_TARGET " + _mfj(info), flush=True)
            return float(support_c[0]), float(support_c[1]), info
        xy, av = container_target(g, r_obj, others_xy)
        info = {"relation": relation, "rule": "interior_centre", "target": [float(xy[0]), float(xy[1])], "outline_centre_xy": g["centre_xy"], "half_extent": g["half_extent"],
                "rim_z": g["rim_z"], "top": g["top"], "rest_z": g["rest_z"], "rest_src": g["rest_src"], "depth": g["depth"], "plane_frac": g["plane_frac"], "n_walled": g["n_walled"],
                "r_inside": g["r_inside"], "r_object": r_obj, "avoid": av, "floor_z": _CG.get("floor_z"), "floor_hidden": g.get("floor_hidden"), "deep_frac": g.get("deep_frac"), "interior_low_z": g.get("interior_low_z"),
                "support_geom": {"center_xy": [float(g["centre_xy"][0]), float(g["centre_xy"][1])], "r": float(g["r_inside"]), "rest_z": float(g["rest_z"]), "table_z": float(_CG["floor_z"])}}
        info["opening"] = {"rim_z": float(g["rim_z"]), "centre_xy": [float(g["centre_xy"][0]), float(g["centre_xy"][1])], "inner_half_extent": [float((1.0 - CONTAINER_RING_FRAC) * g["half_extent"][0]), float((1.0 - CONTAINER_RING_FRAC) * g["half_extent"][1])]}   # C2I4: what the hand must pass to go below the rim (place_hand_limit)
        print("PLACE_TARGET " + _mfj(info), flush=True)
        return float(xy[0]), float(xy[1]), info
    if relation != "beside_support" or region_dir is None:
        info = {"relation": relation, "rule": "support_centre", "target": [float(support_c[0]), float(support_c[1])]}
        print("PLACE_TARGET " + _mfj(info), flush=True)
        return float(support_c[0]), float(support_c[1]), info
    sup = pv_support_from_points(np.asarray(support_pts, float)); prof = gg_object_profile(np.asarray(obj_pts, float), np.asarray(E)[:2, 3])
    r_obj = float(max(prof["r_top"], prof["r_mid"])) if prof else 0.04   # identical to the contract's object radius (mf_make_contract)
    d = np.asarray(region_dir, float)[:2]; d = d / (float(np.linalg.norm(d)) + 1e-12)
    offset = float(sup["r"]) + r_obj + BESIDE_CLEARANCE
    c = np.asarray(sup["center_xy"], float)[:2]; nominal = c + d * offset
    ext = None
    if not sup.get("fit_ok"):   # C4I1: no rim circle (the reference is a body, not a round support): beyond its far edge along d (beside_extent_target)
        reach = object_footprint_reach(obj_pts, prof)   # C4I2: the object's whole footprint about its axis, not its round body only
        ext = beside_extent_target(support_pts, d, r_obj if reach is None else max(r_obj, reach))
        if ext is not None:
            c = np.asarray(ext["centre_xy"], float); offset = float(ext["offset"]); nominal = np.asarray(ext["nominal"], float)
    x, y = mf_validate_beside_target(nominal, support_pts, obj_pts, E)   # shared feasible-region rule: bounded outward shift only
    info = {"relation": relation, "rule": "outside_rim_on_side", "dir": d, "support_center_xy": c, "r_support": float(sup["r"]), "fit_ok": sup.get("fit_ok"),
            "r_object": r_obj, "clearance": BESIDE_CLEARANCE, "offset": offset, "nominal": nominal, "target": [float(x), float(y)], "shift_m": float(np.hypot(float(x) - nominal[0], float(y) - nominal[1]))}
    if ext is not None:   # C4I1: the extent rule's record (a round reference's record is unchanged)
        info.update(rule="outside_extent_on_side", extent={"half_along": round(float(ext["half_along"]), 4), "far_edge": round(float(ext["far_edge"]), 4), "n": ext["n"], "clear": BESIDE_EXTENT_CLEAR,
                                                           "r_reach": None if reach is None else round(float(reach), 4), "reserved": round(float(r_obj if reach is None else max(r_obj, reach)), 4)})   # C4I2
    print("PLACE_TARGET " + _mfj(info), flush=True)
    return float(x), float(y), info
# ===== end K2 place_primitive =====


def run_subgoal(pud_prompts, pl_select, others_xy=(), relation=None, support_prompts=None, object_side=None, assembly_prompts=None, support_track=None):   # C17I2: support_track = {xy} of the body an earlier subgoal placed (None = as before)   # C6I3: assembly_prompts = the reference vocabulary of an assembly relation (la_assembly_prompts)   # C1I1: relation datum + reference prompts from the parsed subgoal (defaults = the adapter fields, as before); C3I1: object_side = the side code of a side-qualified object phrase
    if relation in LP_RELATIONS:   # C15I3: an in_layer subgoal runs the horizontal-insertion primitive (no other relation reaches it)
        return lp_run_subgoal(pud_prompts, pl_select, support_prompts, object_side)
    _TD = task_descriptors()   # T1: task adapter - object/support descriptors are DATA from the manifest (runner-injected); no execution logic
    _E1_LAST.update({"stop": False, "grasp_after": None, "hold": None})   # L: per-subgoal guard state
    _CUR["prompts"] = list(pud_prompts)   # Amendment 3: prompts of the current target for the post-lift re-check
    # --- Observe (identical to P0) ---
    obs = get_observation(); _c_stash_view(obs)   # Q1 C: view from the observe pose
    cam = obs["agentview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
    K = cam["intrinsics"]
    E = cam["pose_mat"]

    pud_c, pud_pts, pud_mask = vb_localize_for_grasp(rgb, depth_img, K, E, pud_prompts, object_side)   # VB1: frozen-reference identity (falls back to localize_target without a reference)
    if pud_c is None:
        print("BINDING_STOP " + json.dumps({"status": _BIND["last"], "reason": "target not bound unambiguously; no motion"}), flush=True)
        return

    _rel, _rdir = relation_geometry(_TD if relation is None else {"relation": relation}, pl_select, E=E, ref_xy=np.asarray(pud_c, float)[:2])   # K2: contract relation + region direction from the adapter DATA (relation datum, side selection); C1I1: the parsed relation when given; C1I2: computed before the support binding (the gate is relation-dependent)
    if _rel == "beside_support" and _rdir is None:   # C10I1: a beside side code without a defined region direction -> no motion
        print("BINDING_STOP " + json.dumps({"status": "region_direction_undefined", "select": pl_select, "relation": _rel}), flush=True)
        return
    _top = _rel == "on_support" and pl_select == TOP_FACE_CODE   # C6I1: the support is a top face (bind_top_face)
    _r_obj = object_footprint_radius(pud_pts, E) if (_rel in ("in_support", "in_sliding_support") or _top) else None   # C6I3: the sliding part must hold the footprint too   # C6I1: the top face must hold the footprint too   # C1I4: the container must hold the held object's footprint (in_support only)
    _part = {"code": _rdir, "E": E, "obj_half": object_footprint_halves(pud_pts)} if _rel == "in_part_support" else None   # C3I3: the bay datum (side code, observer pose, object footprint) for the gate
    _ref_prompts = list(_TD["support_prompts"] if support_prompts is None else support_prompts)
    if support_track is not None:   # C17I2: the support is the body the carry subgoal placed (sc_tracked_support); every other subgoal runs the code below unchanged
        pl_c, pl_pts, _ = sc_tracked_support(rgb, depth_img, K, E, _ref_prompts, support_track.get("xy"), pud_mask)
    elif _rel == "in_sliding_support":   # C6I3: the reference of a sliding-containment relation is an assembly, not a resting-support instance
        pl_c, pl_pts, _ = localize_sliding_assembly(rgb, depth_img, K, E, list(assembly_prompts or _ref_prompts), _rdir, r_object=_r_obj)
    else:
        pl_c, pl_pts, _ = localize_support_by_side(rgb, depth_img, K, E, _ref_prompts, (None if _rel == "in_part_support" else pl_select), exclude_mask=pud_mask, relation=_rel, r_object=_r_obj, part=_part)   # K1: the bound target is never its own support; C1I1: prompts of the parsed reference when given; C1I2: container gate for in_support
    if pl_c is None and _top and support_track is None and _r_obj is not None:   # C19I5: a top face that cannot hold the protrusion's reach may still hold the resting base of a handled ring body (tf_fit_radius); every other case stops as before
        _r_fit = tf_fit_radius(pud_pts, E, _r_obj)
        if _r_fit < float(_r_obj):
            pl_c, pl_pts, _ = localize_support_by_side(rgb, depth_img, K, E, _ref_prompts, pl_select, exclude_mask=pud_mask, relation=_rel, r_object=_r_fit, part=_part)
    if pl_c is None:
        print("BINDING_STOP " + json.dumps({"status": "support_not_found", "select": pl_select, "relation": _rel}), flush=True)
        return
    pl_obb = get_oriented_bounding_box_from_3d_points(pl_pts)
    pl_r = max(pl_obb["extent"][0], pl_obb["extent"][1]) / 2.0
    table_z = pl_pts[:, 2].min()

    pud_base_z0 = object_base_z(pud_pts)
    print(f"PRIM_GEOM table_z={table_z:.4f} pud_base_z0={pud_base_z0:.4f}", flush=True)
    target_x, target_y, _tinfo = place_target(_rel, _rdir, pl_c, pl_pts, pud_pts, E, others_xy=tuple(others_xy))   # K2: on_support = support centre (unchanged); beside_support = outside the rim on the requested side; C1I2: in_support = interior centre of the container
    _srim = place_support_rim(pl_pts, E, (target_x, target_y)) if (_rel == "on_support" and not _top) else None   # C11I5: the rim of a round hollow on-support (None otherwise)
    support_z = float(_tinfo.get("rest_z", table_z))   # C1I2: rest reference of the relation: in_support = the measured interior floor; on / beside = table_z exactly as before
    print(f"Object: {pud_c}, Support: {pl_c}, Target: [{target_x:.3f}, {target_y:.3f}], support_z: {support_z:.4f}")
    _MF["contract"] = mf_make_contract(_rel, pl_pts, pud_pts, pud_prompts, E, region_dir=_rdir, others_xy=tuple(others_xy), label="sg1", binding_base_z=pud_base_z0, target_xy=(target_x, target_y), support_geom=_tinfo.get("support_geom"))   # K2: shadow contract in the relation's own form (region form for beside; never gates); C1I2: interior geometry for in_support
    v1_mem = v1_build_memory(v1_frame(obs), pl_pts, pud_pts) if PRIM["shadow_verifier"] else None

    # --- Grasp strategies (identical to P0) ---
    strategies = [
        {"use_topdown": True, "yaw": 0, "grasp_z": 0.015},
        {"use_topdown": True, "yaw": 0, "grasp_z": 0.005},
        {"use_topdown": False},  # GraspNet
        {"use_topdown": True, "yaw": 90, "grasp_z": 0.015},
        {"use_topdown": True, "yaw": 0, "grasp_z": 0.010},
        {"use_topdown": False},  # GraspNet retry
    ]

    _gplan = mf_plan_grasp(pud_pts, E, "sg1")   # F: geometry-derived grasp height for the first top-down strategy (xy = per-attempt mask centre as before)
    if _gplan["mode"] != "infeasible":
        strategies[0] = {"use_topdown": True, "yaw": int(_gplan["yaw_order"][0]), "grasp_z": float(_gplan["target"][2]), "grasp_xy": [float(_gplan["target"][0]), float(_gplan["target"][1])]}
    grasp_ok = False
    gpos = gquat = None
    for i, strat in enumerate(strategies):
        obs = get_observation()
        cam = obs["agentview"]
        rgb = cam["images"]["rgb"]
        depth = cam["images"]["depth"]
        depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
        K = cam["intrinsics"]
        E = cam["pose_mat"]

        pud_c2, _, pud_mask2 = vb_localize_for_grasp(rgb, depth_img, K, E, pud_prompts, object_side)   # VB1: the same frozen instance, confirmed
        if pud_c2 is None:
            if gpos is not None and e1_hold_evidence():   # E1 G3: 'not visible' is not grasp evidence; public hold evidence required
                print("Object not localized; verified hold retained -> continue with the held object", flush=True)
                grasp_ok = True
            else:
                print("GRASP_ABORT reason=object_not_localized_no_hold_evidence", flush=True)
            break
        if pud_c2[0] < 0 or pud_c2[0] > 1.0:
            continue

        ok, gpos, gquat = attempt_grasp_verify(depth, K, pud_mask2, E, pud_c2, **strat)
        td = "TD" if strat.get("use_topdown") else "GN"
        z = strat.get("grasp_z", "?")
        yaw = strat.get("yaw", "?")
        print(f"Attempt {i+1} ({td} y={yaw} z={z}): {'OK' if ok else 'FAIL'}")
        if ok:
            grasp_ok = True
            break
        if _E1_LAST.get("stop"):   # E1 G4: a lift non-arrival with a possibly held object ends the strategy loop (no blind open)
            print("GRASP_ABORT reason=strategy_loop_stopped_after_lift_nonarrival", flush=True)
            break

    if not grasp_ok or gpos is None:
        print("All grasp attempts failed!")
        return

    # --- Transport / descend / release / retreat: shared primitive ---
    # P0 descended in four moves to table_z + [0.15, 0.10, 0.06, 0.04] then released, settled 10 obs, goto_home.
    drop_est = float(gpos[2] - pud_base_z0)
    _lift = 0.25
    if _top:   # C6I1: the descent over a top face starts at the transport-clearance height at the target (top_face_lift), reached vertically at the lift xy
        _lift = top_face_lift(0.25, support_z, drop_est)
        print("TOP_FACE_LIFT " + json.dumps({"rest_z": round(float(support_z), 4), "hang": round(drop_est, 4), "lift_z": None if _lift is None else round(_lift, 4), "cap": C_Z_MAX}), flush=True)
        if _lift is None:
            print("PLACE_RESULT " + json.dumps({"sg": "sg1", "status": "failed", "reason": "top_face_above_transport_cap"}), flush=True)
            safe_retreat(sg="sg1", q_hold=gquat); return
        _p, _q, _ = read_tcp()
        if float(_lift) > float(_p[2]) + 0.005:
            _rr = move_to_pose(np.array([float(_p[0]), float(_p[1]), float(_lift)]), np.asarray(_q, float), mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label="sg1:top_face_raise")
            print("TOP_FACE_RAISE " + json.dumps({"status": _rr["status"], "reached": bool(_rr.get("reached"))}), flush=True)
            if not _rr.get("reached"):
                print("PLACE_RESULT " + json.dumps({"sg": "sg1", "status": "failed", "reason": "top_face_raise_" + str(_rr["status"])}), flush=True)
                safe_retreat(sg="sg1", q_hold=gquat); return
    prim_place(np.array([target_x, target_y]), gquat, _lift, [table_z + rz for rz in (0.15, 0.10, 0.06, 0.04)],
               float(support_z), drop_est, held_prompts=pud_prompts, sg="sg1", settle_obs=10, opening=_tinfo.get("opening"), support_rim=_srim)   # C1I2: support_z = the relation's rest reference (== table_z for on / beside); C2I4: the container's opening (in_support only) bounds the descent by the hand's reach
    for _ in range(3):
        get_observation()

    if PRIM["shadow_verifier"]:
        res = v1_verify_on(v1_frame(get_observation()), pud_prompts, v1_mem) if _rel == "on_support" else {"status": "not_applicable", "reason": "V1 approximates the on-support relation only; the region verdict is the shared contract verifier's (MF_SHADOW_POST)", "evidence": {"relation": _rel}}   # K2: shadow only (never gates)
        v1_report("sg1_final", res)
    print("Done!")
    return (float(target_x), float(target_y))


# ===== K1 binding: support-instance geometry gate (round rsi_subset, iteration 1) =====
# A support instance is recognised by public geometry that holds for any support, never by a name:
#   (1) it is not the object that was just bound: a mask that covers, or is covered by, the bound object's mask by a majority
#       (relative to the smaller mask) is that object or a part of it -- the same majority criterion as the binding module's
#       IOU_SAME = 0.5;
#   (2) it rests ON the workspace floor and stands above it: the median height of its visible points over the floor is at
#       least SUPPORT_RAISED_MIN. The floor is the dominant surface of the depth image inside the public workspace box C_WS,
#       estimated as a low percentile of the in-box heights; the floor surface itself (which the segmenter returns as a
#       low-score mask) has zero rise and is rejected;
#   (3) it is low: the parent's candidate bound admits OBB centres up to 0.06 m, i.e. bodies standing on the floor up to
#       0.12 m tall; the same bound is applied to the visible top (98th percentile height over the floor), which, unlike the
#       OBB centre, is not pulled down by floor pixels inside a union mask that spans several tall bodies and the floor
#       between them (such unions are also returned by the segmenter at low score).
# Constants derived from the recorded public depth of the iteration-0 observe frames (proposals/iter_1/DIAGNOSIS.md): the
# in-box floor heights span 0.0006 m between p2 and p80 (0.0057 -> 0.0063 m) in all 23 frames; the thinnest support rim
# measured stands 0.021 m over the floor. SUPPORT_RAISED_MIN = 0.01 m is half that rise and ~15x the floor spread.
SUPPORT_SAME_MASK = 0.5       # majority overlap relative to the smaller mask -> the same instance (or a part of it)
SUPPORT_RAISED_MIN = 0.01     # m, minimum median height of a support's visible points above the floor
# ===== C3I2 binding: the rise bound of the K1 support gate (round 20260924, chunk 3, iteration 2) =====
# The 0.010 m bound above was derived on frames of one table family whose rendered floor lies 11.1 mm BELOW the surface bodies
# rest on (truth table minus the p10 floor estimate; constant per family: +11.1 / +3.8 / -0.9 mm in the three families recorded
# in this round). Every rise measured there carries that offset: 0.021 m = 0.010 m physical + 0.011 m offset, so the bound
# equals the thinnest recorded support's physical median rise (10.0 mm over its own table in both families where it was
# recorded); in the family without the offset (-0.9 mm) that support measures 0.0091 m and is rejected (s14, 30 records).
# What the rise test separates is a floor-surface mask (median rise 0.0001-0.0003 m in 195 recorded rejections and every
# replay frame; the floor's own spread p50 - p10 is 0.2-0.4 mm in every family; a floor mask bleeding onto tall bodies keeps its
# median on the floor while its p98 rises, which is why the median and not the top is the statistic) from a body standing on
# the floor. The resolving power of that height difference is the stack's depth noise z_noise = 0.004 m (grasp_geometry
# PERCEPTION; the height resolution the program already uses: CONTAINER_PLANE_BIN 5 mm bins), which also covers the largest
# measured floor estimate above the resting surface (0.9 mm). SUPPORT_RAISED_MIN keeps its value for its other uses
# (ca_assembly / ca_frame_raised raised threshold, CONTAINER_RIM_MIN term); only the on / beside support gate's rise test
# reads SUPPORT_RISE_MIN. Recorded margins: floor masks <= 0.0003, thinnest support 0.0091 (DIAGNOSIS.md sections 2-4).
SUPPORT_RISE_MIN = 0.004      # m, C3I2: minimum median height of a support's visible points above the floor = the stack's depth noise (z_noise)
# ===== end C3I2 binding =====
SUPPORT_TOP_MAX = 0.12        # m, maximum p98 height of a support's visible points above the floor (= 2 x the parent's 0.06 m OBB-centre bound)
SUPPORT_FLOOR_PCT = 10        # percentile of the in-box depth heights taken as the floor height
SUPPORT_FLOOR_MIN_PTS = 200   # minimum in-box points for a floor estimate; without one the rise test is skipped, not failed
def support_floor_z(depth_img, K, E):
    """Floor height from the public depth image alone (low percentile of all in-box heights). None when unavailable."""
    d = np.asarray(depth_img); d = d[:, :, 0] if d.ndim == 3 else d
    pts = mask_to_world_points(np.ones(d.shape, np.uint8), d, K, E)
    if pts is None or len(pts) < SUPPORT_FLOOR_MIN_PTS:
        return None
    pts = np.asarray(pts, float)
    m = ((pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] > C_WS[2][0]) & (pts[:, 2] < C_WS[2][1]))
    if int(m.sum()) < SUPPORT_FLOOR_MIN_PTS:
        return None
    return float(np.percentile(pts[m, 2], SUPPORT_FLOOR_PCT))
def mask_overlap_frac(a, b):
    """Overlap of two masks relative to the smaller one (1.0 = one contains the other)."""
    a = np.asarray(a) > 0; b = np.asarray(b) > 0
    den = min(int(a.sum()), int(b.sum()))
    return 0.0 if den == 0 else float(np.logical_and(a, b).sum()) / float(den)
def support_candidate_ok(pts, mask, exclude_mask, floor_z, relation=None, r_object=None, part=None):   # C3I3: part = {code, E, obj_half} of a bay relation
    """Geometry gate for one support candidate (public data only). Returns (ok, reason, features). C1I2: for the containment
    relation ("in_support") the candidate must be an open-top container (container_geometry) and the K1 flat-support bounds do
    not apply; for every other relation (default None) the K1 gate is exactly as before. C3I2: the rise test's bound is
    SUPPORT_RISE_MIN (the depth noise; the C3I2 note above), the top bound and the order of the tests are unchanged."""
    feat = {"n_pts": int(len(pts))}
    if exclude_mask is not None:
        feat["overlap_bound"] = round(mask_overlap_frac(mask, exclude_mask), 3)
        if feat["overlap_bound"] >= SUPPORT_SAME_MASK:
            return False, "is_bound_object", feat
    if relation == "in_part_support":   # C3I3: bay gate (the same function place_target measures the bay with)
        g = bay_geometry(pts, floor_z, (part or {}).get("code"), (part or {}).get("E"), obj_half=(part or {}).get("obj_half"))
        feat.update({k: g.get(k) for k in ("top", "z_low", "z_wall", "free_max", "wall_min", "n_bays", "axis", "gap", "rest_src", "depth") if g.get(k) is not None})
        return (True, "ok", feat) if g.get("ok") else (False, "not_a_bay:" + str(g.get("why")), feat)
    if relation == "in_support":   # C1I2: container gate
        g = container_geometry(pts, floor_z, r_object=r_object)   # C1I4: hidden-floor branch + footprint size rule
        feat.update({k: g.get(k) for k in ("top", "depth", "rest_src", "plane_frac", "deep_frac", "n_walled", "r_inside") if g.get(k) is not None})
        return (True, "ok", feat) if g.get("ok") else (False, "not_a_container:" + str(g.get("why")), feat)
    if floor_z is not None:
        z = np.asarray(pts, float)[:, 2]
        feat["rise"] = round(float(np.median(z)) - float(floor_z), 4); feat["top"] = round(float(np.percentile(z, 98)) - float(floor_z), 4)
        if feat["rise"] < SUPPORT_RISE_MIN:   # C3I2: the depth-noise bound (SUPPORT_RAISED_MIN keeps its other uses; DIAGNOSIS.md section 3)
            return False, "not_raised_over_floor", feat
        if feat["top"] > SUPPORT_TOP_MAX:
            return False, "too_tall_for_a_support", feat
    return True, "ok", feat
# ===== end K1 binding =====


# ===== C1I2 place_primitive: containment placement (round 20260924, chunk 1, iteration 2) =====
# An open-top container is recognised by public geometry that holds for any container, never by a name: its visible points are
# (a) a rim standing over the workspace floor, (b) an inner region (inside the outline) whose points lie well below the rim,
# with (c) one dominant horizontal level = the interior floor, and (d) wall material around the inner region. A solid body has
# no inner points below its top; a flat support's lip is shallower than the verifier's resting band; a union mask spanning
# several bodies and the floor between them is not walled around. The placement target is the interior centre = midpoint of
# the robust xy extents of the outline (the interior-plane centroid is biased by the strip of floor the near walls hide);
# the rest reference is the interior floor. Constants (DIAGNOSIS.md section 4): CONTAINER_MIN_PTS = the stack's minimum point
# count of a measurement (PERCEPTION min_pts); CONTAINER_DEPTH_MIN = the verifier's gap_pass band (walls shallower than the
# resting band are the lip of a flat support); CONTAINER_RIM_MIN = SUPPORT_RAISED_MIN + CONTAINER_DEPTH_MIN; the ring is the
# outer fifth of the reach (the verifier's rim-bin keep rule, 0.8 x reach) over the verifier's 24 rim bins grouped into 8
# sectors, of which 6 must be walled (two may be hidden by the arm or by a body standing in front); the interior-floor mode is
# taken in 5 mm bins (> PERCEPTION z_noise) and must hold a majority of the inner points (the SUPPORT_SAME_MASK majority rule;
# without it the candidate is not accepted: the reference is measured, never guessed); percentiles 2 / 98 are the stack's
# base_p2 / top_p98 convention. Sub-part masks of one container (the segmenter returns its end walls as separate low-score
# masks) are merged into the instance whose mask covers them by the same majority rule (SUPPORT_SAME_MASK).
CONTAINER_MIN_PTS = 40
CONTAINER_DEPTH_MIN = 0.020
CONTAINER_RIM_MIN = SUPPORT_RAISED_MIN + CONTAINER_DEPTH_MIN
CONTAINER_RING_FRAC = 0.2
CONTAINER_BINS = 24
CONTAINER_SECTORS = 8
CONTAINER_WALLED_MIN = 6
CONTAINER_SECTOR_MIN_PTS = 5
CONTAINER_PLANE_BIN = 0.005
CONTAINER_PLANE_FRAC = 0.5
CONTAINER_PCT_LO = 2               # outline extents: the stack's base_p2 percentile
CONTAINER_PCT_HI = 98              # rim height / outline reach: the stack's top_p98 percentile (the K1 "top" statistic)
CONTAINER_PCT_WALL = 90            # wall statistic per sector: the stack's p90 support-top percentile (robust to the bled outline points at the floor)
CONTAINER_AVOID_MARGIN = 0.010     # m, = the verifier's xy_margin: clearance between two footprints inside one container
_CG = {"floor_z": None}            # floor estimate of the last support localisation (reused by place_target for the container geometry)


# ===== C1I4 place_primitive: hidden interior floor + footprint size rule (round 20260924, chunk 1, iteration 4) =====
# A deep container seen from a low elevation shows no floor: the near wall hides it and the inner region holds only the far
# interior faces (recorded: 14-18 % of the points, no level above 24 %). What is still visible of an open-top body is (a) a rim
# over the floor, (b) an inner region that lies BELOW the rim by at least the minimum depth for a majority of its points, and
# (c) a wall band that reaches the rim in most sectors. A solid body has no inner points below its top; a tapering body's
# outline is below its cap, so its ring does not reach the rim (recorded: 1/8 sectors); a flat support's lip is shallower than
# the depth band. With the floor hidden, the rest reference is the workspace floor (the lowest position the interior floor can
# have: the container stands on it); the contact-seeking placement descent then sets the object down inside its window
# (PLACE_CONTACT_TOL above the release height) or releases it from at most the release clearance. The lowest inner level with
# CONTAINER_MIN_PTS points is logged only. A candidate whose inner region cannot hold the held object's footprint (r_inside,
# the containment contract's rim radius, below the object's profile radius, the contract's object radius) is rejected: a
# tiny mask whose boundary bleeds onto the floor behind a small body shows a "plane" under a "rim" but no room for anything.
CONTAINER_DEEP_FRAC = CONTAINER_PLANE_FRAC   # majority of the inner points at least CONTAINER_DEPTH_MIN below the rim = a deep inside (the same majority rule as the plane)


def object_footprint_radius(obj_pts, E):
    """The held object's footprint radius = max(r_top, r_mid) of its profile (the contract's object radius; 0.04 without a profile)."""
    prof = gg_object_profile(np.asarray(obj_pts, float), np.asarray(E)[:2, 3])
    return float(max(prof["r_top"], prof["r_mid"])) if prof else 0.04
# ===== end C1I4 place_primitive =====


# ===== C2I4 place_primitive: hand-limited containment descent (round 20260924, chunk 2, iteration 4) =====
# The placement descent of a containment relation seeks the object's rest height inside the container (rest reference + hang +
# release clearance; C1I2 / C1I4). The hand that carries the object is not a point: below the rim only those parts of it can go
# whose horizontal footprint passes through the opening. The hand body (CA_HAND: 20.8 x 6.3 cm, its bottom face 3.1 cm ABOVE
# the TCP) never passes an opening narrower than itself, and the finger pair (the pads the gripper opening apart, each finger
# body finger_half_wide beyond its pad, 2.1 cm thin, the tips 0.9 cm below the TCP) passes only when it fits inside the inner
# region at the commanded TCP. Recorded (10 + 5 descents into a 15 cm deep container): the descent stops with the body's bottom
# face on the rim (TCP = true rim - 3.1 cm in every state), 8.5 cm above the rest-based release height -> premature_contact,
# release refused (or released by the window's chance when the hang was long enough). Rule: the release height is the larger
# of the rest-based height and the hand limit = the lowest TCP height at which, for every hand box, the part below the rim plane
# (rim + PLACE_RIM_CLEAR) fits inside the inner region; the object is released there and falls the remaining distance inside the walls
# (its footprint is inside the inner region by the C1I4 size gate and the descent's xy accuracy). A box that fits imposes no
# limit (a wide container; a tilted hand whose bottom face is above the plane at the rest-based height). The on / beside
# relations have no opening: the expression is unchanged.
PLACE_RIM_CLEAR = 0.01     # m, margin of the hand's overhanging faces above the rim (= CA_CLEAR_MARGIN, the stack's depth / xy sigma; the rim is the p98 height of the container's points, which errs upward by 0.8 cm in the recorded frames)
PLACE_GRIP_SPAN = 0.08     # m, the finger span at full opening (the public gripper opening is a fraction of it; APPROACH_CORRIDOR is half of it)
PLACE_BISECT_ITERS = 14    # bisection steps of the limit height (a 0.5 m bracket -> 0.03 mm, below the interface's position tolerance)
_PLACE_BOX_EDGES = ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7))   # edges of a box whose corner index is 4 ix + 2 iy + iz


def _place_part_fits(P, z_tcp, z_plane, off, hin):
    """Whether the part of a hand box (its 8 corners P relative to the TCP, rotated) that lies below the plane z_plane when the TCP
    is at z_tcp has a footprint inside the inner region (TCP offset off from its centre, half extents hin). The part is a convex
    polytope whose vertices are the corners below the plane and the crossings of the box edges with it; nothing below the plane
    fits trivially."""
    z = z_tcp + P[:, 2]; below = z < z_plane
    if not below.any():
        return True
    V = [P[i, :2] for i in range(8) if below[i]]
    for i, j in _PLACE_BOX_EDGES:
        if below[i] != below[j]:
            t = (z_plane - z[i]) / (z[j] - z[i]); V.append(P[i, :2] + t * (P[j, :2] - P[i, :2]))
    foot = np.max(np.abs(np.array(V)), axis=0)
    return bool(np.all(off + foot <= hin + 1e-9))


def place_hand_limit(opening, tcp_xy, q, opening_frac, z_floor):
    """Lowest TCP height, at or above the rest-based release height z_floor, at which every hand box (CA_HAND: the finger pair
    widened by the gripper opening, the body), rotated by the placement orientation q, keeps whatever part of it lies below the
    rim plane (rim_z + PLACE_RIM_CLEAR) inside the container's inner region about the commanded TCP xy. opening = {rim_z,
    centre_xy, inner_half_extent}. Returns (z_tcp_min | None, info): None when every box already fits at z_floor (no limit); for a
    box that does not, the limit is bisected between z_floor and the height at which the box is entirely above the plane (the
    part below the plane grows as the TCP goes down, so the fit is monotone). An unobservable gripper opening counts as the full
    span (the widest finger pair)."""
    if not opening:
        return None, {"applies": False}
    R = _R(q); c = np.asarray(opening["centre_xy"], float)[:2]; hin = np.asarray(opening["inner_half_extent"], float)[:2]; rim = float(opening["rim_z"]); z_plane = rim + PLACE_RIM_CLEAR
    off = np.abs(np.asarray(tcp_xy, float)[:2] - c); z_floor = float(z_floor)
    w = PLACE_GRIP_SPAN * (1.0 if opening_frac is None else float(np.clip(float(opening_frac), 0.0, 1.0)))
    boxes = []
    for name, ((x0, x1), (y0, y1), (z0, z1)) in zip(("fingers", "body"), _ca_hand_boxes()):
        if name == "fingers":
            x0, x1 = x0 - 0.5 * w, x1 + 0.5 * w   # the pads stand w apart; each finger body reaches finger_half_wide beyond its pad
        P = np.array([R @ np.array([xv, yv, zv]) for xv in (x0, x1) for yv in (y0, y1) for zv in (z0, z1)])
        foot = np.max(np.abs(P[:, :2]), axis=0); low = float(P[:, 2].min()); z_clear = z_plane - low   # the whole box above the plane
        fits_floor = _place_part_fits(P, z_floor, z_plane, off, hin)
        if fits_floor or z_clear <= z_floor:
            z_min = None
        else:
            lo, hi = z_floor, z_clear
            for _ in range(PLACE_BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                if _place_part_fits(P, mid, z_plane, off, hin):
                    hi = mid
                else:
                    lo = mid
            z_min = float(hi)
        boxes.append({"box": name, "footprint_half": [round(float(foot[0]), 4), round(float(foot[1]), 4)], "low_rel_tcp": round(low, 4), "fits": bool(fits_floor), "z_tcp_min": None if z_min is None else round(z_min, 4)})
    zs = [float(b["z_tcp_min"]) for b in boxes if b["z_tcp_min"] is not None]
    z = None if not zs else float(max(zs))
    return z, {"applies": True, "rim_z": rim, "inner_half_extent": [float(hin[0]), float(hin[1])], "tcp_offset": [round(float(off[0]), 4), round(float(off[1]), 4)], "opening_m": round(w, 4), "z_floor": round(z_floor, 4), "boxes": boxes, "z_tcp_min": z}


def place_release_height(support_z, drop_used, z_tcp_min):
    """Release height of the placement descent: the rest-based height (rest reference + hang + release clearance, as before) or,
    when the hand cannot go that deep into the opening, the hand limit. Returns (z_release, hand_limited)."""
    z = float(support_z) + float(drop_used) + PRIM["release_clearance"]
    if z_tcp_min is not None and float(z_tcp_min) > z:
        return float(z_tcp_min), True
    return z, False
# ===== end C2I4 place_primitive =====


# ===== C11I1 place_primitive: the guarded placement descent begins above the opening (N11, slot C11I1) =====
# A containment placement (an opening record {rim_z, centre_xy, inner_half_extent}: C1I2 / C3I3 / C6I3) lowered the held body
# with the free-mode transport_lower (joint-space, tol_pos 1.5 cm, no contact guard) to the fixed lift height and started the
# guarded contact descent there. When rim + hang lies above that height the held body passed the rim plane inside the unguarded
# move. Recorded (proposals/chunk11_iter1/DIAGNOSIS.md, 18 development placements on two tasks): the executed xy error of that
# move is 0.7-1.4 cm, the held body met the walls inside it (tilt 0.2 -> 8.5-10.6 deg, centre displaced 3-4 cm) and the guarded
# primitive, starting from a body already caught, refused (lateral_corridor / premature_contact). Rule: the guarded descent
# starts at the lowest TCP height at which the lowest point below the TCP (the measured hang; the fingertips, CA_HAND tip, when
# the hang is shorter) clears the rim plane the hand limit uses (rim_z + PLACE_RIM_CLEAR); never below the previous start
# (lift_z), never above the transport height or the transport cap (C_Z_MAX). The free lowering then ends above the rim and the
# whole passage through the opening is made by the guarded descent with its unchanged corridor, contact window and budget.
# A raised entry is reached on the straight vertical line (the transport's linear mode and its free fallback): the free-mode
# lowering ended 0.7-1.4 cm off its xy (recorded), the linear transport 0.3-0.4 cm, and the guarded descent runs on the line
# from where the lowering ends, so its start error is carried into the opening.
# Without an opening (on / beside), or when that height is not above lift_z, the start is lift_z exactly as before.
def place_entry_height(opening, hang, lift_z, z_tr):
    """TCP height at which the guarded placement descent begins, and its record. opening = {rim_z, ...} or None; hang = the
    TCP-to-base distance of the held body (drop_used); lift_z = the previous start; z_tr = the transport height."""
    lift_z = float(lift_z)
    if not opening or hang is None:
        return lift_z, {"applies": False, "z_entry": round(lift_z, 4)}
    low = max(float(hang), float(CA_HAND["tip"]))
    z_need = float(opening["rim_z"]) + PLACE_RIM_CLEAR + low
    z_top = min(float(z_tr), C_Z_MAX)
    z = max(lift_z, min(z_need, z_top))
    bound = "lift" if z <= lift_z + 1e-9 else ("clearance" if z_need <= z_top else "transport")
    return float(z), {"applies": True, "rim_z": round(float(opening["rim_z"]), 4), "hang": round(float(hang), 4), "z_need": round(z_need, 4), "z_tr": round(float(z_tr), 4),
                      "lift_z": round(lift_z, 4), "z_entry": round(float(z), 4), "raised_cm": round(100.0 * (float(z) - lift_z), 2), "bound": bound}
# ===== end C11I1 place_primitive =====


# ===== C11I4 place_primitive: the pre-release slip measurement is unavailable while the held body crosses an opening's rim plane =====
# The N1 P slip check compares the held body's axis offset (measure_held_object) at the release height with the same offset at
# the lift pose, where the body hangs free. When the placement is into an opening (an opening record {rim_z, centre_xy,
# inner_half_extent}) and the held body reaches below the opening's rim plane (rim_z + PLACE_RIM_CLEAR, the plane of the hand limit
# and of the C11I1 entry), the part below the plane shares its image region and its column with the opening's walls and interior:
# its mask either merges with the interior (recorded: 91-97 % of the points below the plane, points down to the interior floor,
# a fitted radius equal to the opening's own half extent) or is cut by the rim wall (recorded: visible base 1.4-2.6 cm higher than
# at the lift pose). The part above the plane is not the part the lift reference measured. The differential is then not an in-hand
# slip: 15 of 47 recorded in-opening measurements exceeded P_SLIP_BOUND while the true in-hand change stayed below it, and the
# lateral recentre they commanded runs inside the walls (blocked shift, failed re-descent, refused release) - see
# proposals/chunk11_iter4/DIAGNOSIS.md. Rule: in that case the measurement is unavailable and the check takes its existing branch
# for an unavailable measurement (release; the release itself stays gated by the guarded descent's own result). Inputs are the
# opening record, the hang measured at the lift pose (the drop_below_tcp values inside PRIM drop_bounds, the quantity drop_used is
# made of) and the current TCP height. Without an opening, without a usable hang, or with the body wholly above the plane: None,
# and the check is the previous one.
def p_slip_occluded(opening, ref, tcp_z):
    """None when the slip measurement is available; otherwise the record of why not (the held body crosses the rim plane)."""
    if not opening or not ref:
        return None
    lo, hi = PRIM["drop_bounds"]
    ds = [float(m["drop_below_tcp"]) for m in ref if m is not None and m.get("drop_below_tcp") is not None and lo <= float(m["drop_below_tcp"]) <= hi]
    if not ds:
        return None
    hang = float(np.median(ds)); plane = float(opening["rim_z"]) + PLACE_RIM_CLEAR; low = float(tcp_z) - hang
    if low >= plane:
        return None
    return {"rim_plane": round(plane, 4), "tcp_z": round(float(tcp_z), 4), "hang": round(hang, 4), "body_low_z": round(low, 4), "below_plane_cm": round(100.0 * (plane - low), 2)}
# ===== end C11I4 place_primitive =====


# ===== C11I5 place_primitive: the release height over a raised hollow support clears its rim plane (N11, slot C11I5) =====
# The on relation's release height is rest-based: support_z (= table_z, the lowest point of the support's mask) + the measured
# hang + PRIM release_clearance, so the held body's base is commanded to the clearance above the table. When the support is a
# round hollow body whose rim stands above the table, that target lies inside the support's opening. Recorded (20 development
# placements of one hollow body into another, DIAGNOSIS.md section 1.3): the rim fit below measures the support's rim within
# 0.03 cm of truth; the rest-based target lay 2.95-3.0 cm below it; the rim pinch let the held body slip 0.8-2.4 cm below its
# lift-pose hang measurement (in transport and at the descent onset, after the last public measurement), so it met the inner
# wall 1.5-2.6 cm above the target, slid up between the pads instead of stopping the arm (no axial contact is ever confirmed) and
# pushed the hand 1.2-1.7 cm sideways: stalled / no_contact_no_arrival, release refused although the body rested in the support.
# Rule: for an on placement (not a top face) whose bound support's main body (gw_components, the C4I4 clustering; two comparable
# bodies -> no rule, LOW_SIZE_RATIO) is accepted by the program's rim fit gw_ring (module B's round / hollow criteria; a body
# whose rim rises less than the release clearance above its base cannot pass the hollow test: it needs points GW_G inner_below
# below the rim and 0.5 cm above the base) and whose rim circle contains the placement target, the release height is
# max(rest-based height, z_rim + PLACE_RIM_CLEAR + drop_used): the lowest TCP height at which the held body's measured lowest
# point clears the rim plane that the C2I4 hand limit, the C11I1 entry and the C11I4 slip occlusion use. A body above that plane
# and over the opening cannot touch the support, so the guarded descent arrives in free space; the slip only lowers the body,
# by less than the room between the plane and the first wall contact. The body is released there and drops into (or onto) the
# opening. Any other support, relation or failure: the parent's height, unchanged. No new constant.
def place_support_rim(pts, E, target_xy):
    """The rim {z_rim, r, centre_xy, base} of a round hollow support whose rim circle contains target_xy, else None. Logs PLACE_SUPPORT_RIM."""
    rec = {"applies": False}; out = None
    try:
        P = np.asarray(pts, float)
        if P.ndim != 2 or len(P) < GA_MIN_PTS:
            rec["why"] = "too_few_points"; return None
        L, k, sizes = gw_components(P)
        if len(sizes) > 1 and sizes[0] < LOW_SIZE_RATIO * sizes[1]:
            rec["why"] = "two_comparable_bodies"; return None
        M = P[L == k]
        ring, info = gw_ring(M, np.asarray(E)[:2, 3])
        rec.update({kk: info.get(kk) for kk in ("centre_xy", "r", "rms", "n_bins", "n_inliers", "z_rim", "base", "interior_pts", "far_frac") if kk in info})
        if ring is None:
            rec["why"] = str(info.get("why")); return None
        c = np.asarray(ring["c"], float)[:2]; dist = float(np.hypot(*(np.asarray(target_xy, float)[:2] - c))); rec["target_from_centre"] = round(dist, 4)
        if dist > float(ring["r"]):
            rec["why"] = "target_outside_rim"; return None
        out = {"z_rim": float(ring["z_rim"]), "r": float(ring["r"]), "centre_xy": [float(c[0]), float(c[1])], "base": float(ring["base"])}
        rec.update(applies=True, why="round_hollow_support")
        return out
    except Exception as e:  # noqa: BLE001  (a failure of the measurement never changes the parent's release height)
        rec["why"] = f"error: {type(e).__name__}: {e}"; return None
    finally:
        print("PLACE_SUPPORT_RIM " + _mfj(rec), flush=True)


def place_rim_release(z_release, rim, hang):
    """(release height, record): max(z_release, rim z_rim + PLACE_RIM_CLEAR + hang) when a support rim is given, else z_release."""
    z0 = float(z_release)
    if rim is None or hang is None:
        return z0, {"applies": False}
    z_need = float(rim["z_rim"]) + PLACE_RIM_CLEAR + float(hang)
    rec = {"applies": True, "z_rim": round(float(rim["z_rim"]), 4), "hang": round(float(hang), 4), "z_rest_based": round(z0, 4), "z_need": round(z_need, 4), "raised": bool(z_need > z0 + 1e-9)}
    z = max(z0, z_need); rec["z_release"] = round(z, 4); rec["raised_cm"] = round(100.0 * (z - z0), 2)
    return z, rec
# ===== end C11I5 place_primitive =====


def container_geometry(pts, floor_z, r_object=None):
    """Open-top container test + interior geometry of one candidate from its visible points and the workspace floor height.
    Returns a dict with ok / why and, when measurable: rim_z, top (rim over the floor), centre_xy, half_extent (outline),
    n_inner, n_ring, plane_z, plane_frac, rest_z (the measured interior floor), rest_src, depth, walled (per sector), n_walled,
    r_inside. A candidate without a measured interior floor (no majority level in the inner region) is not a container here:
    the rest reference is never guessed from the workspace floor (a solid body with a cap or a shoulder would pass otherwise)."""
    P = np.asarray(pts, float).reshape(-1, 3); g = {"ok": False, "n_pts": int(len(P))}
    if floor_z is None:
        g["why"] = "no_floor_estimate"; return g
    if len(P) < 2 * CONTAINER_MIN_PTS:
        g["why"] = "too_few_points"; return g
    floor_z = float(floor_z); z = P[:, 2]; rim_z = float(np.percentile(z, CONTAINER_PCT_HI)); g.update(rim_z=rim_z, top=round(rim_z - floor_z, 4))
    if rim_z - floor_z < CONTAINER_RIM_MIN:
        g["why"] = "rim_not_raised_over_floor"; return g
    lo, hi = np.percentile(P[:, :2], [CONTAINER_PCT_LO, CONTAINER_PCT_HI], axis=0); c = 0.5 * (lo + hi); he = 0.5 * (hi - lo)
    d = P[:, :2] - c; rad = np.hypot(d[:, 0], d[:, 1]); b = (((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2.0 * np.pi)) * CONTAINER_BINS).astype(int) % CONTAINER_BINS
    reach = np.full(CONTAINER_BINS, np.nan)
    for k in range(CONTAINER_BINS):
        m = b == k
        if int(m.sum()) >= CONTAINER_SECTOR_MIN_PTS:
            reach[k] = float(np.percentile(rad[m], CONTAINER_PCT_HI))
    lim = reach[b]; covered = np.isfinite(lim); inner = covered & (rad < (1.0 - CONTAINER_RING_FRAC) * lim); ring = covered & ~inner
    g.update(centre_xy=[float(c[0]), float(c[1])], half_extent=[float(he[0]), float(he[1])], n_inner=int(inner.sum()), n_ring=int(ring.sum()))
    if int(inner.sum()) < CONTAINER_MIN_PTS:
        g["why"] = "no_interior"; return g
    zi = z[inner]; edges = np.arange(float(zi.min()) - CONTAINER_PLANE_BIN, float(zi.max()) + 2.0 * CONTAINER_PLANE_BIN, CONTAINER_PLANE_BIN)
    h, _ = np.histogram(zi, bins=edges); k = int(np.argmax(h)); sel = (zi >= edges[k] - CONTAINER_PLANE_BIN) & (zi < edges[k + 1] + CONTAINER_PLANE_BIN)
    plane_frac = float(sel.sum()) / float(len(zi)); plane_z = float(np.median(zi[sel])) if sel.any() else float(np.median(zi))
    g.update(plane_z=plane_z, plane_frac=round(plane_frac, 3))
    if plane_frac >= CONTAINER_PLANE_FRAC:   # (a) visible interior floor: the majority level (C1I2, unchanged)
        rest_z = max(plane_z, floor_z)   # a bottom cannot lie below the floor it stands on
        g.update(rest_z=float(rest_z), rest_src="interior_plane", depth=round(rim_z - rest_z, 4), floor_hidden=False)
    else:   # (b) C1I4: no majority level -> the floor is hidden by the walls, or there is no interior at all
        deep_frac = float(np.mean(zi < rim_z - CONTAINER_DEPTH_MIN))
        cum = np.cumsum(h); kl = int(np.argmax(cum >= CONTAINER_MIN_PTS)); low_z = float(np.median(zi[zi < edges[kl + 1]])) if cum[-1] >= CONTAINER_MIN_PTS else float(np.median(zi))
        g.update(deep_frac=round(deep_frac, 3), interior_low_z=low_z)
        if deep_frac < CONTAINER_DEEP_FRAC:   # what is visible inside the outline is not below the rim: a solid body's top, a lip
            g["why"] = "inner_not_below_rim"; return g
        rest_z = floor_z   # unknown floor: the workspace floor is the lowest position it can have; the contact descent finds it
        g.update(rest_z=float(rest_z), rest_src="workspace_floor", depth=round(rim_z - rest_z, 4), floor_hidden=True)
    if rim_z - rest_z < CONTAINER_DEPTH_MIN:
        g["why"] = "too_shallow"; return g
    sec = (b * CONTAINER_SECTORS) // CONTAINER_BINS; walled = []
    for s in range(CONTAINER_SECTORS):
        m = ring & (sec == s)
        if int(m.sum()) < CONTAINER_SECTOR_MIN_PTS:
            walled.append(None)
        elif g["floor_hidden"]:   # C1I4: the wall band must reach the rim (a tapering body's outline lies below its cap)
            walled.append(bool(float(np.percentile(z[m], CONTAINER_PCT_WALL)) >= rim_z - CONTAINER_DEPTH_MIN))
        else:
            walled.append(bool(float(np.percentile(z[m], CONTAINER_PCT_WALL)) - rest_z >= CONTAINER_DEPTH_MIN))
    g.update(walled=walled, n_walled=int(sum(1 for w in walled if w)), n_covered=int(sum(1 for w in walled if w is not None)))
    if g["n_walled"] < CONTAINER_WALLED_MIN:
        g["why"] = "not_walled_around"; return g
    g.update(r_inside=float((1.0 - CONTAINER_RING_FRAC) * min(float(he[0]), float(he[1]))))
    if r_object is not None and g["r_inside"] < float(r_object):   # C1I4: the inner region must hold the held object's footprint
        g["why"] = "smaller_than_object"; g["r_object"] = float(r_object); return g
    g.update(ok=True, why="ok")
    return g


def container_target(geom, r_obj, others_xy=()):
    """Interior centre of an accepted container; when an object already placed inside (others_xy) lies closer than
    r_obj + C_R_DEFAULT + CONTAINER_AVOID_MARGIN, the target moves along the container's long axis away from it by that distance,
    bounded so the footprint stays inside the inner region. Returns (xy, info)."""
    c = np.asarray(geom["centre_xy"], float)[:2]; he = np.asarray(geom["half_extent"], float)[:2]; ax = int(np.argmax(he)); u = np.zeros(2); u[ax] = 1.0
    room = max(0.0, (1.0 - CONTAINER_RING_FRAC) * float(he[ax]) - float(r_obj)); need = float(r_obj) + C_R_DEFAULT + CONTAINER_AVOID_MARGIN
    near = [np.asarray(o, float)[:2] for o in others_xy if float(np.hypot(*(np.asarray(o, float)[:2] - c))) < need]
    info = {"n_near": len(near), "need_m": round(need, 4), "room_m": round(room, 4), "axis": ax, "shift_m": 0.0}
    if not near:
        return c, info
    along = sum(float((o - c) @ u) for o in near); sign = -1.0 if along > 0 else 1.0; shift = sign * min(need, room)
    info.update(shift_m=round(float(abs(shift)), 4), bounded=bool(need > room))
    return c + shift * u, info
# ===== end C1I2 place_primitive =====


# ===== C3I3 place_primitive: containment into a bay of a partitioned support (round 20260924, chunk 3, iteration 3) =====
# "in the <side> bay of the <support>" (the data relation in_part; "bay" here = the language's own word for one division of a
# partitioned support) puts the object INTO one bay of a support whose top view is divided by walls. The C1I2 / C1I4 open-top gate
# sees one rim and one interior; a partitioned support has several interiors behind partitions of different heights (recorded: rim
# 17.5 cm, partitions 11-17.5 cm, one outer wall 8 cm over the floor) and fails its "walled around to the rim" test in the low
# wall's sectors (5 of 8 sectors, 20/20 frames). Rule:
#   (1) height map of the support's visible points: max height over the floor per CP_CELL cell (the close primitive's top-face
#       occupancy cell) over the outline padded by one cell. A cell is a WALL when it stands at least CONTAINER_DEPTH_MIN (the
#       verifier's resting band: what retains an object) above the bay floor, whose visible level z_low (the lowest
#       CONTAINER_PLANE_BIN level holding CONTAINER_MIN_PTS points inside the outline, the C1I4 low-level statistic) errs low by at
#       most the stack's base bias: z_wall = z_low + CP_FLOOR_BIAS + CONTAINER_DEPTH_MIN (recorded: z_low 1.6 cm, floor cells at
#       most 3.0 cm, the lowest wall 8.0 cm, z_wall 5.6 cm);
#   (2) bays = 4-connected components of the non-wall cells (empty cells included: an occluded interior) that do not reach the
#       padded border (a region open to the outside is not a bay) and hold at least CP_MIN_CELLS cells; each bay has an inner
#       extent (the span of its cells), a rim (its highest adjacent wall cell) and a floor (the lowest level of its own points when
#       that level lies a resting band below the wall level, else the workspace floor as in C1I4);
#   (3) the side word names the bay on the observer's axes (C3I1 bind_side_dirs: lateral for min / max, depth for xmin / xmax with
#       xmax the farthest from the camera): for a max code the bay whose extent BEGINS farthest along the axis, for a min code the
#       one whose extent ENDS first (a bay spanning the whole support along the axis is neither its front nor its back); the
#       runner-up must trail by SIDE_MIN_GAP (2 sigma of two centre estimates), a lone bay is its own extreme (the support instance
#       rule's precedent), otherwise gap_below_min: no motion;
#   (4) the object must fit: the minimum-area rectangle of its top band (object_footprint_halves: the section seen whole from
#       above, _ca_min_rect) against the bay's inner half extents in either orientation (the transport turns the hand by a quarter
#       turn when needed): half_minor <= min(inner half), half_major <= max(inner half); else smaller_than_object.
# The placement runs the containment machinery on the bay: its centre is the target (rule bay_centre), its floor the rest
# reference, its extent and rim the opening of the hand-limited release (C2I4, unchanged), and the transport orientation (M1 T
# candidates, t_select_transport_p3) prefers the feasible candidates whose held footprint PASSES the opening: footprint misfit =
# sum over the two axes of max(0, held half extent - inner half extent), the held half extents being the 2-98 percentile half
# widths of the held object's points below the pads (measure_held_object at the lift pose) rotated by the candidate's yaw change
# about the vertical, passing = misfit within CP_FIT_TOL; among the passing candidates the previous rule decides (as_is, else the
# fewest planned steps), and when none or all pass (a wide container; a measurement that bled onto the surroundings: recorded
# 17.7 cm half extent in one state of a passing task) the previous rule decides among all feasible candidates: the choice is
# then identical. Without an opening or a measurement the rule is the previous one.
CP_CELL = 0.01                 # m, height-map cell = CA_CELL, the close primitive's top-face occupancy cell (that block is defined later in the file)
CP_FLOOR_BIAS = 0.02           # m, the stack's base-bias bound (grasp_geometry PERCEPTION base_bias: a visible low level lies 0-2 cm above the true base)
CP_MIN_CELLS = 4               # cells, the smallest bay considered (2 x 2 cm: below any footprint the stack places)
CP_TOP_BAND = 0.02             # m, the object's top band (the profile's top band: the section seen whole from above)
CP_PAD_ENGAGE = 0.018          # m, the pad friction surface's height above the fingertips (grasp_geometry GRIPPER pad_engage): the held object's points below TCP - tip - pad_engage lie below the pads
CP_SIDES = SIDE_LATERAL + SIDE_DEPTH   # side codes of a bay (the horizontal observer axes)
CP_FIT_TOL = 0.01              # m, a held footprint passes the opening when its misfit is within the stack's xy sigma (grasp_geometry PERCEPTION xy_sigma; = PLACE_RIM_CLEAR): the held extents are centre-estimate statistics of visible-surface points


def cp_low_level(zs):
    """Lowest CONTAINER_PLANE_BIN level holding CONTAINER_MIN_PTS points (the C1I4 interior_low_z statistic); None with fewer points."""
    zs = np.asarray(zs, float)
    if len(zs) < CONTAINER_MIN_PTS:
        return None
    edges = np.arange(float(zs.min()) - CONTAINER_PLANE_BIN, float(zs.max()) + 2.0 * CONTAINER_PLANE_BIN, CONTAINER_PLANE_BIN)
    h, _ = np.histogram(zs, bins=edges); cum = np.cumsum(h); kl = int(np.argmax(cum >= CONTAINER_MIN_PTS))
    return float(np.median(zs[zs < edges[kl + 1]]))


def cp_components(free):
    """4-connected components of the True cells of a grid: a list of (n, 2) index arrays."""
    nx, ny = free.shape; lab = -np.ones((nx, ny), int); comps = []
    for i in range(nx):
        for j in range(ny):
            if free[i, j] and lab[i, j] < 0:
                k = len(comps); stack = [(i, j)]; lab[i, j] = k; cells = []
                while stack:
                    a, b = stack.pop(); cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, d = a + da, b + db
                        if 0 <= c < nx and 0 <= d < ny and free[c, d] and lab[c, d] < 0:
                            lab[c, d] = k; stack.append((c, d))
                comps.append(np.array(cells, int))
    return comps


def object_footprint_halves(obj_pts):
    """(half_major, half_minor) of the minimum-area rectangle of the object's top band (the section seen whole from above); None with
    fewer than CONTAINER_MIN_PTS band points."""
    P = np.asarray(obj_pts, float).reshape(-1, 3)
    if len(P) < CONTAINER_MIN_PTS:
        return None
    top = float(np.percentile(P[:, 2], CONTAINER_PCT_HI)); band = P[P[:, 2] >= top - CP_TOP_BAND]
    if len(band) < CONTAINER_MIN_PTS:
        return None
    _c, _a, _b, hu, hv = _ca_min_rect(band[:, :2])
    return float(hu), float(hv)


def bay_geometry(pts, floor_z, code, E, obj_half=None):
    """Partitioned-support test + the geometry of the bay a side code names, from the support's visible points, the floor height,
    the observer's pose E and (optionally) the object's footprint halves. Returns a dict with ok / why and, when measurable: rim_z,
    top, centre_xy / half_extent (outline), z_low, z_wall, n_bays, bays (each: n_cells, centre_xy, half_extent, extent, rim_z,
    rest_z, rest_src, n_pts), dirs, axis, scores, chosen, gap, and for an accepted bay: bay, rest_z, rest_src, depth."""
    P = np.asarray(pts, float).reshape(-1, 3); g = {"ok": False, "n_pts": int(len(P)), "code": None if code is None else str(code)}
    if floor_z is None:
        g["why"] = "no_floor_estimate"; return g
    if len(P) < 2 * CONTAINER_MIN_PTS:
        g["why"] = "too_few_points"; return g
    code = str(code or "").strip().lower()
    if code not in CP_SIDES:
        g["why"] = "unknown_code"; return g
    floor_z = float(floor_z); z = P[:, 2] - floor_z; rim = float(np.percentile(z, CONTAINER_PCT_HI)); g.update(rim_z=rim + floor_z, top=round(rim, 4))
    if rim < CONTAINER_RIM_MIN:
        g["why"] = "rim_not_raised_over_floor"; return g
    lo, hi = np.percentile(P[:, :2], [CONTAINER_PCT_LO, CONTAINER_PCT_HI], axis=0); c = 0.5 * (lo + hi); he = 0.5 * (hi - lo)
    g.update(centre_xy=[float(c[0]), float(c[1])], half_extent=[float(he[0]), float(he[1])])
    inside = (P[:, 0] >= lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1]); z_low = cp_low_level(z[inside])
    if z_low is None:
        g["why"] = "no_interior"; return g
    z_wall = z_low + CP_FLOOR_BIAS + CONTAINER_DEPTH_MIN; g.update(z_low=round(z_low, 4), z_wall=round(z_wall, 4))
    if z_wall > rim - CONTAINER_DEPTH_MIN:
        g["why"] = "too_shallow"; return g
    org = np.asarray(lo, float) - CP_CELL; nx = int(np.ceil((hi[0] - lo[0]) / CP_CELL)) + 3; ny = int(np.ceil((hi[1] - lo[1]) / CP_CELL)) + 3
    ix = np.floor((P[:, 0] - org[0]) / CP_CELL).astype(int); iy = np.floor((P[:, 1] - org[1]) / CP_CELL).astype(int); ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    H = np.full((nx, ny), -np.inf); np.maximum.at(H, (ix[ok], iy[ok]), z[ok]); wall = np.isfinite(H) & (H >= z_wall)
    free = np.isfinite(H) & ~wall; g.update(free_max=(round(float(H[free].max()), 4) if free.any() else None), wall_min=(round(float(H[wall].min()), 4) if wall.any() else None))   # the wall level's margins (logged)
    key = ix * ny + iy; bays = []
    for comp in cp_components(~wall):
        if len(comp) < CP_MIN_CELLS or (comp[:, 0] == 0).any() or (comp[:, 1] == 0).any() or (comp[:, 0] == nx - 1).any() or (comp[:, 1] == ny - 1).any():
            continue   # too small, or open to the outside (reaches the padded border)
        e0 = org + comp.min(0) * CP_CELL; e1 = org + (comp.max(0) + 1) * CP_CELL
        adj = set()
        for a, b in comp:
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if wall[a + da, b + db]:
                    adj.add((a + da, b + db))
        rim_b = float(max(H[a, b] for a, b in adj)); sel = ok & np.isin(key, comp[:, 0] * ny + comp[:, 1]); lv = cp_low_level(z[sel])
        rest, src = (lv, "bay_floor") if (lv is not None and lv < z_wall - CONTAINER_DEPTH_MIN) else (0.0, "workspace_floor")
        bays.append({"n_cells": int(len(comp)), "centre_xy": [float(0.5 * (e0[0] + e1[0])), float(0.5 * (e0[1] + e1[1]))], "half_extent": [float(0.5 * (e1[0] - e0[0])), float(0.5 * (e1[1] - e0[1]))],
                     "extent": [[float(e0[0]), float(e1[0])], [float(e0[1]), float(e1[1])]], "rim_z": rim_b + floor_z, "rest_z": float(rest) + floor_z, "rest_src": src, "n_pts": int(sel.sum())})
    g.update(n_bays=len(bays), bays=bays)
    if not bays:
        g["why"] = "no_bay"; return g
    lat, dep, dinfo = bind_side_dirs(E, c); g["dirs"] = dinfo
    if lat is None:
        g["why"] = dinfo["why"]; return g
    if code in SIDE_DEPTH and float(dinfo["away_m"]) < SIDE_MIN_GAP:   # the observer stands over the support: no near / far side
        g["why"] = "observer_above"; return g
    d = lat if code in SIDE_LATERAL else dep; g["axis"] = "lateral" if code in SIDE_LATERAL else "depth"
    if code.endswith("max"):   # the bay whose extent begins farthest along d (the lowest projection of its extent)
        score = [sum(min(b["extent"][i][0] * d[i], b["extent"][i][1] * d[i]) for i in range(2)) for b in bays]; order = [int(i) for i in np.argsort(score)[::-1]]
    else:                      # the bay whose extent ends first along d (the highest projection of its extent)
        score = [sum(max(b["extent"][i][0] * d[i], b["extent"][i][1] * d[i]) for i in range(2)) for b in bays]; order = [int(i) for i in np.argsort(score)]
    k = order[0]; gap = None if len(bays) == 1 else float(abs(score[k] - score[order[1]]))
    g.update(scores=[round(float(s), 4) for s in score], chosen=k, gap=None if gap is None else round(gap, 4))
    if gap is not None and gap < SIDE_MIN_GAP:
        g["why"] = "gap_below_min"; return g
    bay = bays[k]; hin = sorted(bay["half_extent"])
    if obj_half is not None and (float(obj_half[1]) > hin[0] or float(obj_half[0]) > hin[1]):   # the object's minor half against the bay's minor, its major against the major
        g.update(why="smaller_than_object", obj_half=[float(obj_half[0]), float(obj_half[1])]); return g
    g.update(bay=bay, rest_z=bay["rest_z"], rest_src=bay["rest_src"], depth=round(bay["rim_z"] - bay["rest_z"], 4), ok=True, why="ok")
    return g


def t_footprint_misfit(meas, q_lift, cand_quats, opening):
    """Per transport candidate: the excess (m, summed over the two axes) of the held object's horizontal half extents beyond the
    opening's inner half extents once the hand turns from the lift orientation q_lift to the candidate's. The extents are the 2-98
    percentile half widths of the held object's points below the pads (measure_held_object at the lift pose), rotated about the
    vertical by the candidate's yaw change. None without an opening or a measurement (the previous rule then applies)."""
    ms = [m for m in (meas or []) if m is not None and m.get("pts") is not None and m.get("tcp_p") is not None]
    if not opening or not ms:
        return None
    hin = np.asarray(opening["inner_half_extent"], float)[:2]; P = np.asarray(ms[-1]["pts"], float); tcp = np.asarray(ms[-1]["tcp_p"], float)
    body = P[P[:, 2] < tcp[2] - CA_HAND["tip"] - CP_PAD_ENGAGE]
    if len(body) < CONTAINER_MIN_PTS:
        body = P
    xy = body[:, :2]; y0 = _yaw_of(np.asarray(q_lift, float)); out = []
    for qq in cand_quats:
        t = _yaw_of(np.asarray(qq, float)) - y0; cs, sn = np.cos(t), np.sin(t); r = xy @ np.array([[cs, sn], [-sn, cs]])
        h = 0.5 * (np.percentile(r, CONTAINER_PCT_HI, axis=0) - np.percentile(r, CONTAINER_PCT_LO, axis=0)); out.append(float(np.sum(np.maximum(0.0, h - hin))))
    return out
# ===== C12I1 place_primitive: continuous yaw alignment of a body-pinched slab with the opening (N12, slot C12I1; carried by C12I5) =====
# A slab (a body standing on its edge, its top band an elongated rectangle) grasped by a body pinch across its thin axis is squared
# between the pads: its horizontal long axis lies in the pad plane, which for a top-down hand orientation q is the angle _yaw_of(q)
# (make_topdown_quat(psi) closes along (sin psi, -cos psi), so the pad plane is at psi; a tool-z flip adds 180 deg, the same axis).
# Recorded (proposals/chunk12_iter1/DIAGNOSIS.md, 43 development placements into the bays of a partitioned support): the long axis
# after the close is the grasp yaw within 0.2 deg in 37 rect-axis grasps, a module-B pinch 10 deg oblique squared the body by the
# descent, and the axis follows every later hand turn within 0.5 deg. The previous M1 T / C3I3 rule chose among discrete candidates
# (as_is, flip, +-90 deg) with a held-cloud misfit that read the body's extent wrong in both directions; the held long axis reached
# the 5-6 cm wide bays 19.7-89.9 deg off their long axis in 33 placements. Rule (inputs: the executed grasp, the opening record):
#   held slab = the grasp plan was a body pinch, the executed orientation has the plan's pad-plane yaw (mod 180 deg, within the
#   1 deg grid psi is planned on), and the top-band rectangle of the executed attempt's points is elongated:
#   half_major - half_minor > GA_WIDTH_MARGIN (two widths of one body are distinct beyond that margin);
#   misfit of a candidate q = sum over the base axes of max(0, rectangle extent - opening inner half extent), the rectangle turned to
#   the pad plane _yaw_of(q) (the bay extents are measured on base-axis cells);
#   if the previous choice's misfit is 0 nothing changes; otherwise the two candidates that turn the pad plane onto the opening's
#   long axis (the larger inner half extent; x on a tie) - the turn about the vertical in (-90, 90] deg and its tool-z flip - are
#   planned like the others (plan_only at their own compensated TCP) and the feasible candidate with the smallest misfit is taken,
#   ties go to an aligned candidate (the fewest planned steps), else to the previous order (as_is, else the fewest planned steps);
#   taken only when its misfit is below the previous choice's.
# Without an opening, without a held slab, or with a previous choice whose rectangle lies inside the opening: the previous rule.
SL_YAW_TOL = 1.0   # deg, the grid of ga_rect_axis (psi is planned in whole degrees): the executed pad plane is the planned one


def t_slab_rect(pts):
    """(centre_xy, half_major, half_minor) of the minimum-area rectangle of the top band of `pts` (the object_footprint_halves
    statistic, with its centre); None with fewer than CONTAINER_MIN_PTS band points."""
    P = np.asarray(pts, float).reshape(-1, 3)
    if len(P) < CONTAINER_MIN_PTS:
        return None
    top = float(np.percentile(P[:, 2], CONTAINER_PCT_HI)); band = P[P[:, 2] >= top - CP_TOP_BAND]
    if len(band) < CONTAINER_MIN_PTS:
        return None
    c, _a, _b, hu, hv = _ca_min_rect(band[:, :2])
    return np.asarray(c, float)[:2], float(hu), float(hv)


def t_held_slab(q_lift):
    """The held slab's rectangle {half_major, half_minor, centre_xy, ...} from the executed grasp, or None (not a squared slab)."""
    plan = _MF.get("last_plan") or {}
    if plan.get("mode") != "body_pinch" or not plan.get("yaw_order") or _WG.get("pts") is None or _WG.get("quat") is None:
        return None
    y_plan = float(plan["yaw_order"][0]); y_exec = float(np.degrees(_yaw_of(np.asarray(_WG["quat"], float))))
    y_lift = float(np.degrees(_yaw_of(np.asarray(q_lift, float))))
    d_exec = (y_exec - y_plan + 90.0) % 180.0 - 90.0; d_lift = (y_lift - y_exec + 90.0) % 180.0 - 90.0
    if abs(d_exec) > SL_YAW_TOL or abs(d_lift) > SL_YAW_TOL:
        return None
    r = t_slab_rect(_WG["pts"])
    if r is None or float(r[1]) - float(r[2]) <= GA_WIDTH_MARGIN:
        return None
    return {"half_major": float(r[1]), "half_minor": float(r[2]), "centre_xy": [float(r[0][0]), float(r[0][1])], "yaw_plan": round(y_plan, 2), "yaw_exec": round(y_exec, 2)}


def t_slab_misfit(slab, q, hin):
    """Excess (m, summed over the base axes) of the slab's rectangle, long axis in the pad plane of q, beyond the inner half extents."""
    t = _yaw_of(np.asarray(q, float)); a = float(slab["half_major"]); b = float(slab["half_minor"]); c, s = abs(np.cos(t)), abs(np.sin(t))
    ext = np.array([a * c + b * s, a * s + b * c]); hin = np.asarray(hin, float)[:2]
    return float(np.sum(np.maximum(0.0, ext - hin)))


def t_align_slab(slab, opening, place_xy, meas, q, lift_z, cands, comp, plans, chosen, sg="sg"):
    """The C12I1 choice: None (keep the previous choice) or (q, tcp_xy, compensation_info, name)."""
    if not slab or not opening:
        return None
    hin = np.asarray(opening["inner_half_extent"], float)[:2]
    names = [c[0] for c in cands]; qs = [np.asarray(c[1], float) for c in cands]; comps = list(comp); pl = list(plans)
    mis = [t_slab_misfit(slab, qq, hin) for qq in qs]
    rec = {"sg": sg, "half_major": round(slab["half_major"], 4), "half_minor": round(slab["half_minor"], 4), "inner_half_extent": [round(float(v), 4) for v in hin],
           "previous": names[chosen], "previous_misfit_cm": round(100.0 * mis[chosen], 2)}
    if mis[chosen] <= 1e-9:
        rec.update(applies=False, why="previous_fits"); print("TRANSPORT_ALIGN " + _json.dumps(rec), flush=True); return None
    phi = 0.0 if hin[0] >= hin[1] else 0.5 * np.pi
    dlt = float(np.degrees((phi - _yaw_of(np.asarray(q, float)) + 0.5 * np.pi) % np.pi - 0.5 * np.pi))
    qa = _yaw_about_vertical(np.asarray(q, float), dlt)
    for name, qq in (("align", qa), ("flip_align", _flip_tool_z(qa))):
        cc = t_slab_compensate_xy(np.asarray(place_xy, float), meas, qq, slab)   # C12I5: the held slab's centre (compensate_xy when its offset is not available)
        pp = move_to_pose(np.array([cc[0][0], cc[0][1], lift_z]), qq, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=f"{sg}:transport_plan_{name}")
        names.append(name); qs.append(np.asarray(qq, float)); comps.append(cc); pl.append(pp); mis.append(t_slab_misfit(slab, qq, hin))
    feas = [(int(p["steps_planned"]), k) for k, p in enumerate(pl) if p.get("detail") == "plan_only" and p.get("steps_planned") is not None]
    rec.update(turn_deg=round(dlt, 2), misfit_cm=[round(100.0 * m, 2) for m in mis], planned=[p.get("steps_planned") for p in pl], candidates=names)
    if not feas:
        rec.update(applies=False, why="no_feasible_candidate"); print("TRANSPORT_ALIGN " + _json.dumps(rec), flush=True); return None
    m_best = min(mis[k] for _, k in feas); tie = [k for _, k in feas if mis[k] <= m_best + 1e-9]
    al = [k for k in tie if k >= len(cands)]   # an aligned candidate among the best: it alone leaves the full margin across the opening's narrow axis
    best = min((s, k) for s, k in feas if k in al)[1] if al else (0 if 0 in tie else min((s, k) for s, k in feas if k in tie)[1])
    if not mis[best] < mis[chosen] - 1e-9:
        rec.update(applies=False, why="no_improvement", best=names[best]); print("TRANSPORT_ALIGN " + _json.dumps(rec), flush=True); return None
    rec.update(applies=True, chosen=names[best], chosen_misfit_cm=round(100.0 * mis[best], 2), slab_axis_deg=round(float(np.degrees(_yaw_of(qs[best]))), 2))
    print("TRANSPORT_ALIGN " + _json.dumps(rec), flush=True)
    return qs[best], comps[best][0], comps[best][1], names[best]
# ===== end C12I1 place_primitive =====


# ===== C12I5 place_primitive: the held slab's centre, not its visible surface's centroid, is placed on the bay centre (N12, slot C12I5) =====
# compensate_xy places the TCP at target - offset, the offset being the lift-pose measurement (measure_held_object: visible-surface
# centroid shifted 1.2 cm away from the camera, PRIM centroid_shift_fallback) re-expressed in the gripper frame and turned with the
# hand to the placement orientation. Recorded (DIAGNOSIS.md sections 2-3; 44 development placements of a body-pinched slab into
# the bays of a partitioned support; truth for diagnosis only): the measured offset is 0.5-1.8 cm, pointing toward the camera,
# while the slab's true centre lies 0.0-0.6 cm from the TCP at the lift and 0.0-0.2 cm at the descent start - the offset is the
# camera-facing view bias of a 3 cm thick body, not a property of the grasp, so turning it with the hand (46-90 deg for the
# aligned slab) moves the TCP 0.8-1.8 cm along the bay's long axis. The slab, aligned (C12I1) and 0.6-0.8 cm shorter than the
# bay at each end, then landed with its end on the bay's end wall in 39/39 aligned placements (truth base height = that wall's top),
# the one placement whose error fell the other way succeeded. The grasp knows where the slab is held:
#   across the pads: zero (the two fingers close symmetrically about the TCP; the slab is squared between them, C12I1);
#   along the pad plane: (body axis - executed grasp point) . u_pad, the body axis being the grasp plan's own body centre
#   (profile_summary axis_xy, the point the pinch was planned about; 0 for a pinch at the body's centre) and u_pad the unit vector
#   of the executed pad plane _yaw_of(_WG quat). Recorded (truth, diagnosis only): the plan's axis lies 0.09 cm (median, 45 grasps)
#   from the slab's truth centre along the pad plane, the top-band rectangle centre 0.22 cm, and the truth in-hand offset after the
#   close is -0.06 cm along / +0.03 cm across (median; max 0.25 cm);
# Rule (t_slab_offset / t_slab_compensate_xy; only the two ALIGNED candidates of the C12I1 rule use it - the turned slab whose long
# axis is laid along the bay's long axis, where the turned bias lands; every candidate the parent already had, and every placement
# the C12I1 rule leaves unchanged, keeps gh_compensate_xy: not established there, see DIAGNOSIS.md section 5):
#   the offset exists only when the grasp point lies on the slab (|along| <= half_major) and every valid lift measurement's own
#   offset lies within half_major + PRIM["held_agree_tol"] of the slab offset carried to that measurement's orientation (the visible
#   held body is on the carried slab's footprint; else, or with no valid measurement: compensate_xy / gh_compensate_xy unchanged);
#   gripper-frame offset = R(q_grasp)^T [along * u_pad, 0], re-expressed at the aligned orientation exactly as compensate_xy does;
#   TCP = place_xy - that offset: the slab's centre goes to the bay centre.
# No new constant (half_major / half_minor measured; PRIM["held_agree_tol"], PRIM["held_offset_max"] existing). No guard, margin, bound
# or budget changes; every other placement path keeps gh_compensate_xy / compensate_xy.
def t_slab_offset(slab):
    """Gripper-frame (3,) in-hand offset of the held slab's centre from the TCP, from the executed grasp plan, and an info dict; (None, info)."""
    info = {"applies": slab is not None}
    if not slab:
        return None, info
    plan = _MF.get("last_plan") or {}
    t = plan.get("target"); q = _WG.get("quat"); c = (plan.get("profile_summary") or {}).get("axis_xy")
    if t is None or q is None or c is None:
        info["why"] = "no_grasp_geometry"; return None, info
    q = np.asarray(q, float); yg = _yaw_of(q); u = np.array([np.cos(yg), np.sin(yg)])
    along = float((np.asarray(c, float)[:2] - np.asarray(t, float).reshape(-1)[:2]) @ u)
    info["along_cm"] = round(100.0 * along, 2)
    if abs(along) > float(slab["half_major"]):
        info["why"] = "grasp_point_off_the_slab"; return None, info
    off_b = np.array([along * u[0], along * u[1], 0.0])
    return _R(q).T @ off_b, info


def t_slab_compensate_xy(place_xy, measurements, q_place, slab):
    """gh_compensate_xy, except that a held slab's centre (t_slab_offset) is placed on the target. Returns (place_xy_tcp, info)."""
    off_g, sinfo = t_slab_offset(slab) if slab is not None else (None, {"applies": False})
    if off_g is None:
        xy, info = gh_compensate_xy(place_xy, measurements, q_place)
        return xy, (dict(info, held_slab=sinfo) if sinfo.get("applies") else info)
    ms = [m for m in (measurements or []) if m is not None and m.get("offset_xy_base") is not None and m.get("tcp_q") is not None and np.linalg.norm(np.asarray(m["offset_xy_base"], float)) < PRIM["held_offset_max"]]
    if not ms:
        xy, info = gh_compensate_xy(place_xy, measurements, q_place)
        return xy, dict(info, held_slab=dict(sinfo, why="no_valid_held_measurement"))
    reach = float(slab["half_major"]) + PRIM["held_agree_tol"]
    d = [float(np.hypot(*(np.asarray(m["offset_xy_base"], float)[:2] - (_R(np.asarray(m["tcp_q"], float)) @ off_g)[:2]))) for m in ms]
    sinfo["meas_to_slab_cm"] = [round(100.0 * v, 2) for v in d]
    if max(d) > reach:
        xy, info = gh_compensate_xy(place_xy, measurements, q_place)
        return xy, dict(info, held_slab=dict(sinfo, why="held_body_off_the_slab"))
    off_l = _R(np.asarray(ms[-1]["tcp_q"], float)) @ off_g; off_p = _R(np.asarray(q_place, float)) @ off_g
    info = {"n_valid": len(ms), "applied": True, "how": "held_slab_grasp", "held_slab": sinfo,
            "offset_base_lift_cm": (off_l[:2] * 100).round(2), "offset_base_place_cm": (off_p[:2] * 100).round(2),
            "measured_offset_lift_cm": (np.mean([np.asarray(m["offset_xy_base"], float)[:2] for m in ms], axis=0) * 100).round(2)}
    return np.asarray(place_xy, float) - off_p[:2], info
# ===== end C12I5 place_primitive =====


# ===== C13I1 place_primitive: the held slab's centre is placed on the bay's measured centre (N13, slot C13I1) =====
# bay_geometry measures a bay as the span of whole wall-free CP_CELL cells: the cell holding a wall's inner face is a wall cell, so
# the span loses 0..1 cell at each end and its centre is off the faces' mid-point by up to CP_CELL / 2 per axis. Recorded
# (DIAGNOSIS.md sections 1-3; development placements into the bays of a partitioned support): the front / back bays' target is
# 0.44-0.45 cm off their measured faces' mid-point along the long axis, toward the tall divider, in every row; an aligned slab 0.84 cm
# shorter than the bay at each end then rests its end on the divider's top (premature contact) unless its in-hand residual
# falls the other way. Rule (bf_face_centre; inputs: the support's visible points, the floor, the accepted bay record):
#   wall points = the support's points at least z_wall over the floor (the level that made the wall cells walls);
#   along each base axis i, within the bay's span on the other axis, the low face is found in the cell band [e0 - CP_CELL, e0) just
#   outside the span's low end e0 (the high face in [e1, e1 + CP_CELL)): in each CP_CELL slice of the span the wall point nearest
#   the bay, and the face = the median over the slices (a corner fillet or a stray point moves one slice; recorded: the corner
#   slice of one wall reads 0.8 cm inside the face, the other slices within 1 mm); a side without a visible wall point keeps its
#   cell edge; centre_i = mid-point of the two faces.
#   The faces lie outside the cell span by construction (0..CP_CELL), so the shift is below CP_CELL / 2 per axis and the opening,
#   which keeps the cell half extents, stays inside the measured faces: no extent, margin or gate changes.
# And (bf_compensate_xy): for that bay placement every transport candidate of a held slab (t_held_slab) is compensated by the
# grasp-derived slab offset (t_slab_compensate_xy, C12I5; it falls back to gh_compensate_xy by its own checks), so the slab's
# centre, not the camera-facing part of its visible surface, goes to that centre - C12I5 applied it to the two aligned
# candidates only, because the cell-centre bias (0.4 cm toward the near wall in the square bay) then made an unturned slab
# marginal; with (1) that bias is gone. Without a bay (opening without centre_src bay_faces) or without a held slab: the parent.
BF_SRC = "bay_faces"


def bf_face(band, i, j, f0, nearest, edge):
    """(face coordinate along axis i, n slices): the median over the CP_CELL slices of the bay's span along axis j of the wall point
    nearest the bay in each slice (a corner's or a single point's artefact moves one slice, not the face); the cell edge when the
    band holds no wall point."""
    if len(band) == 0:
        return float(edge), 0
    k = np.floor((band[:, j] - f0) / CP_CELL).astype(int)
    v = [float(nearest(band[k == s, i])) for s in np.unique(k)]
    return float(np.median(v)), len(v)


def bf_face_centre(pts, floor_z, g, bay):
    """((x, y) mid-point of the bay's measured wall faces, info). g = the accepted bay_geometry record, bay = its chosen bay."""
    P = np.asarray(pts, float).reshape(-1, 3); c = [float(v) for v in bay["centre_xy"]]
    info = {"cell_centre_xy": [round(v, 4) for v in c], "faces": [], "shift_cm": [0.0, 0.0]}
    if floor_z is None or g.get("z_wall") is None or len(P) == 0:
        info["why"] = "no_wall_level"; return c, info
    Wp = P[(P[:, 2] - float(floor_z)) >= float(g["z_wall"])]
    for i in (0, 1):
        j = 1 - i; e0, e1 = (float(v) for v in bay["extent"][i]); f0, f1 = (float(v) for v in bay["extent"][j])
        on = Wp[(Wp[:, j] >= f0) & (Wp[:, j] <= f1)]
        lo = on[(on[:, i] >= e0 - CP_CELL) & (on[:, i] < e0)]; hi = on[(on[:, i] >= e1) & (on[:, i] < e1 + CP_CELL)]
        a = bf_face(lo, i, j, f0, max, e0); b = bf_face(hi, i, j, f0, min, e1)
        c[i] = 0.5 * (a[0] + b[0]); info["faces"].append([round(a[0], 4), round(b[0], 4), a[1], b[1]])
        info["shift_cm"][i] = round(100.0 * (c[i] - float(bay["centre_xy"][i])), 2)
    return c, info


def bf_compensate_xy(place_xy, meas, q, slab, opening):
    """The compensation of one transport candidate: a held slab placed into a bay whose centre is its measured faces' mid-point ->
    t_slab_compensate_xy; anything else -> gh_compensate_xy (the parent)."""
    if slab is not None and opening is not None and opening.get("centre_src") == BF_SRC:
        return t_slab_compensate_xy(place_xy, meas, q, slab)
    return gh_compensate_xy(place_xy, meas, q)
# ===== end C13I1 place_primitive =====


# ===== end C3I3 place_primitive =====


# ===== C6I3 place_primitive: containment into the open sliding part of an assembly (round 20260924, chunk 6, iteration 3) =====
# "put the <object> in the <side> <part> of the <reference>" (the data relation in_sliding; its anaphora form copies the part and
# its side from the previous clause) puts the object INTO a part that slides out of a body. The stack has the whole containment
# machinery (C1I2 gate / interior centre / rest reference, C1I4 hidden floor, C2I4 hand-limited release, C3I3 the region of a
# divided support) and, from the close / open primitives, the decomposition of an assembly into a body (its largest planar upper
# level, ca_body) and the components standing outside that body's outline (ca_parts, each with the body face it protrudes from:
# outward normal n_out, face offset face_off). What it lacks is the dispatch and one piece of geometry. Rule:
#   (1) the container is a component of the assembly that is itself an open container (container_geometry, unchanged: a rim over
#       the floor, an inner region below it, a measured interior floor or the hidden-floor branch, walled around). Recorded (30
#       frames of three states sets): exactly one such component per frame while a part stands out; the thin bars of the closed
#       parts fail it (too_shallow) and a closed assembly has none (10/10 frames), so no motion is planned into one;
#   (2) a side code names one of several by its own rim height (the levels of the body): a max code the highest, a min code the
#       lowest, the runner-up at least SIDE_MIN_GAP away, a lone candidate its own extreme - the rule bind_select_by_side already
#       applies to the vertical codes (CS_SIDES = SIDE_VERTICAL); any other code is refused;
#   (3) the interior region, in the part's own frame (s = outward distance from the body's face plane along n_out, t across it):
#         far bound s = 0, the body's own face plane (body centre . n_out + face_off): behind it the body stands over the part and
#         nothing can be lowered into it (measured within 0.3 mm of that face in 20 frames, conservative by 0.6-1.0 cm in 10 more);
#         near bound s = s_near, the outer edge of the part's VISIBLE interior floor: a CP_CELL height map of the part's points in
#         that frame, a cell being a wall when it stands CP_FLOOR_BIAS + CONTAINER_DEPTH_MIN above the part's lowest level (the
#         C3I3 wall level, unchanged), and s_near = the outer edge of the largest 4-connected component of the observed non-wall
#         cells. A height map, not a percentile: the part's front panel and its bar stand at the floor's own height in front of
#         it and move any percentile of the point cloud outward (measured: up to 2.2 cm);
#         across that axis: the unchanged C1I2 inner extent ((1 - CONTAINER_RING_FRAC) x the outline half extent), which is already
#         accurate there (both lateral walls are visible; measured error 0.2 cm) - the outline is only wrong along the slide axis,
#         where it reads the occlusion boundary (measured: the C1I2 centre lies 1.9-4.6 cm toward the observer, and a wide held
#         object then meets the near inner wall: truth sweep, 7 of 9 reconstructible transports, gap -1.5..-0.2 cm);
#       the region is written in base axes, which is exact only while n_out is a base axis, so it is used only while the
#       misalignment costs less than the stack's xy sigma over the body (|off axis| x max(body half) <= CA_CLEAR_MARGIN; measured
#       0.000 in all 30 frames); otherwise the plain C1I2 region is kept and logged.
#   (4) everything downstream is the existing containment expression: the target is that region's centre, the rest reference the
#       part's measured interior floor, the opening of the hand-limited descent (C2I4) that region with the part's own rim, the
#       transport-orientation rule (C3I3 t_footprint_misfit) the same opening, and the object must fit it (the C1I4 size rule).
# Constants: the stack's own (CP_CELL, CP_FLOOR_BIAS, CP_MIN_CELLS, CONTAINER_*, SIDE_VERTICAL, SIDE_MIN_GAP, CA_CLEAR_MARGIN);
# CS_MIN_SPAN = two cells of the height map = CONTAINER_DEPTH_MIN, the verifier's resting band: below it there is no region to set
# an object down in. DIAGNOSIS.md section 5.
CS_SIDES = SIDE_VERTICAL       # side codes that name one sliding part among the levels of a body (the codes the vertical instance rule implements)
CS_MIN_SPAN = 2 * CP_CELL      # m, smallest interior span along the slide axis that is still a region (two height-map cells; = CONTAINER_DEPTH_MIN, the resting band)


def slide_open_parts(pts, floor_z):
    """(body, [(component, container_geometry), ...]) of an assembly: the components standing outside the body's outline
    (ca_parts) that are themselves open containers (container_geometry). (None, []) when the assembly has no body."""
    P = np.asarray(pts, float).reshape(-1, 3); body = ca_body(P)
    if body is None:
        return None, []
    out = []
    for q in ca_parts(P, body):
        g = container_geometry(q["pts"], floor_z)
        if g.get("ok"):
            out.append((q, g))
    return body, out


def slide_interior(part, body, geom, floor_z):
    """Interior region of one open sliding part (rule (3)): (centre_xy, half_extent, info) in base axes, or (None, None, info)."""
    n = np.asarray(part["n_out"], float)[:2]; w = np.array([-n[1], n[0]]); c = np.asarray(body["centre"], float)[:2]
    ax = int(np.argmax(np.abs(n))); info = {"axis": ax, "off_axis": round(float(abs(n[1 - ax])), 4)}
    if abs(float(n[1 - ax])) * float(max(body["half"])) > CA_CLEAR_MARGIN:
        info["why"] = "face_not_axis_aligned"; return None, None, info
    P = np.asarray(part["pts"], float).reshape(-1, 3)
    s = (P[:, :2] - c) @ n - float(part["face_off"]); t = (P[:, :2] - c) @ w; z = P[:, 2] - float(floor_z)
    z_low = cp_low_level(z)
    if z_low is None:
        info["why"] = "no_low_level"; return None, None, info
    z_wall = z_low + CP_FLOOR_BIAS + CONTAINER_DEPTH_MIN; info.update(z_low=round(float(z_low), 4), z_wall=round(float(z_wall), 4))
    s_hi = float(np.percentile(s, CONTAINER_PCT_HI)); t_lo, t_hi = [float(v) for v in np.percentile(t, [CONTAINER_PCT_LO, CONTAINER_PCT_HI])]
    info["s_out"] = round(s_hi, 4)
    if s_hi <= CS_MIN_SPAN:
        info["why"] = "not_protruding"; return None, None, info
    ns = int(np.ceil(s_hi / CP_CELL)) + 1; nt = int(np.ceil((t_hi - t_lo) / CP_CELL)) + 3; org_t = t_lo - CP_CELL
    i = np.floor(s / CP_CELL).astype(int); j = np.floor((t - org_t) / CP_CELL).astype(int)
    m = (i >= 0) & (i < ns) & (j >= 0) & (j < nt)
    H = np.full((ns, nt), -np.inf); np.maximum.at(H, (i[m], j[m]), z[m])
    comps = [cc for cc in cp_components(np.isfinite(H) & (H < z_wall)) if len(cc) >= CP_MIN_CELLS]
    if not comps:
        info["why"] = "no_interior_floor"; return None, None, info
    comp = max(comps, key=len); s_near = min(float((int(comp[:, 0].max()) + 1) * CP_CELL), s_hi)
    info.update(n_cells=int(len(comp)), n_regions=len(comps), s_near=round(s_near, 4))
    if s_near <= CS_MIN_SPAN:
        info["why"] = "no_room_before_the_wall"; return None, None, info
    plane = (float(c @ n) + float(part["face_off"])) / float(n[ax]); sgn = 1.0 if float(n[ax]) > 0 else -1.0
    ctr = np.zeros(2); he = np.zeros(2)
    ctr[ax] = plane + sgn * 0.5 * s_near; he[ax] = 0.5 * s_near
    ctr[1 - ax] = float(geom["centre_xy"][1 - ax]); he[1 - ax] = (1.0 - CONTAINER_RING_FRAC) * float(geom["half_extent"][1 - ax])
    info.update(why="ok", face_plane=round(float(plane), 4))
    return ctr, he, info


def sliding_geometry(pts, floor_z, code, r_object=None):
    """Open sliding part of an assembly named by a side code, and its interior region (rules (1)-(3)). Returns a dict with ok /
    why and, when measurable: body_centre / body_half, n_open_parts, parts, levels, chosen, gap, centre_xy, half_extent, rim_z,
    rest_z, rest_src, depth, r_inside, interior (the region record)."""
    P = np.asarray(pts, float).reshape(-1, 3); g = {"ok": False, "n_pts": int(len(P)), "code": None if code is None else str(code)}
    if floor_z is None:
        g["why"] = "no_floor_estimate"; return g
    code = str(code or "").strip().lower()
    if code not in CS_SIDES:
        g["why"] = "unknown_code"; return g
    if len(P) < 2 * CONTAINER_MIN_PTS:
        g["why"] = "too_few_points"; return g
    body, cands = slide_open_parts(P, floor_z)
    if body is None:
        g["why"] = "no_body"; return g
    g.update(body_centre=[round(float(v), 4) for v in body["centre"]], body_half=[round(float(v), 4) for v in body["half"]], body_z_top=round(float(body["z_top"]), 4),
             n_open_parts=len(cands), parts=[{"n": int(q["n"]), "centroid": [round(float(v), 4) for v in q["centroid"]], "rim_z": round(float(cg["rim_z"]), 4),
                                              "rest_z": round(float(cg["rest_z"]), 4), "rest_src": cg["rest_src"]} for q, cg in cands])
    if not cands:
        g["why"] = "no_open_part"; return g
    lv = [float(cg["rim_z"]) for _q, cg in cands]; order = [int(k) for k in np.argsort(lv)]
    k = order[-1] if code.endswith("max") else order[0]
    gap = None if len(cands) == 1 else float(abs(lv[k] - lv[order[-2] if code.endswith("max") else order[1]]))
    g.update(levels=[round(v, 4) for v in lv], chosen=int(k), gap=None if gap is None else round(gap, 4))
    if gap is not None and gap < SIDE_MIN_GAP:
        g["why"] = "gap_below_min"; return g
    part, cg = cands[k]
    ctr, he, rinfo = slide_interior(part, body, cg, floor_z); g["interior"] = rinfo
    if ctr is None:   # the axis-aligned region is not measurable: the plain C1I2 interior of the same component
        ctr = np.asarray(cg["centre_xy"], float); he = (1.0 - CONTAINER_RING_FRAC) * np.asarray(cg["half_extent"], float)
        g["rule"] = "outline_interior"
    else:
        g["rule"] = "face_plane_interior"
    g.update(centre_xy=[float(ctr[0]), float(ctr[1])], half_extent=[float(he[0]), float(he[1])], rim_z=float(cg["rim_z"]), rest_z=float(cg["rest_z"]),
             rest_src=cg["rest_src"], depth=round(float(cg["rim_z"]) - float(cg["rest_z"]), 4), top=cg.get("top"), plane_frac=cg.get("plane_frac"),
             n_walled=cg.get("n_walled"), r_inside=float(min(float(he[0]), float(he[1]))))
    if float(g["depth"]) < CONTAINER_DEPTH_MIN:
        g["why"] = "too_shallow"; return g
    if r_object is not None and g["r_inside"] < float(r_object):   # C1I4 size rule on the region the object is set down in
        g["why"] = "smaller_than_object"; g["r_object"] = float(r_object); return g
    g.update(ok=True, why="ok")
    return g


def la_assembly_prompts(sg, td, previous=()):
    """Vocabulary of the assembly a relation refers to: the subgoal's part phrase, the reference phrase its clause spells out and
    its support phrase (the prompt list run_open_articulated / run_close_articulated build for the same assembly), and, when the
    clause refers to a previous one (an anaphora subgoal: its reference IS that clause's), the previous subgoals' as well.
    Language order, duplicates removed; None when the subgoals name no reference at all. Measured: the part vocabulary alone
    finds the assembly in 11 of 40 recorded frames (the part is a small piece of one face), the union in 40 of 40, identically to
    the reference vocabulary alone wherever the clause spells it out."""
    out = []
    for s in (list(previous) + [sg] if sg.get("refers") else [sg]):
        ph = la_phrase_in(td, "supports", s.get("clause") or "")
        for lst in (s.get("part_prompts"), ((td or {}).get("supports") or {}).get(ph) if ph else None, s.get("support_prompts")):
            for q in (lst or []):
                if q not in out:
                    out.append(q)
    return out or None


def localize_sliding_assembly(rgb, depth_img, K, E, prompts, code, r_object=None):
    """Reference of a sliding-containment relation: the assembly of the reference prompts (ca_assembly, the open / close
    primitives' own rule) whose open sliding part the side code names. Returns (region centre, assembly points, None) as
    localize_support_by_side does, or (None, None, None); logs SUPPORT_GATE and, on acceptance, BIND_SUPPORT."""
    floor_z = support_floor_z(depth_img, K, E); _CG["floor_z"] = floor_z
    pts, ainfo = (None, {"n_candidates": 0, "why": "no_floor_estimate"}) if floor_z is None else ca_assembly(rgb, depth_img, K, E, list(prompts), floor_z)
    g = {"ok": False, "why": "assembly_not_found"} if pts is None else sliding_geometry(pts, floor_z, code, r_object=r_object)
    rec = {"prompt": list(prompts), "relation": "in_sliding_support", "select": None if code is None else str(code), "floor_z": None if floor_z is None else round(float(floor_z), 4),
           "assembly": {kk: ainfo.get(kk) for kk in ("n_candidates", "rejected", "anchor", "n_merged", "n_pts", "why") if kk in ainfo}}
    rec.update({kk: g.get(kk) for kk in ("ok", "why", "rule", "n_open_parts", "parts", "levels", "chosen", "gap", "centre_xy", "half_extent", "rim_z", "rest_z",
                                         "rest_src", "depth", "r_inside", "r_object", "interior", "body_centre", "body_half") if kk in g})
    print("SUPPORT_GATE " + json.dumps(rec, default=str), flush=True)
    if not g.get("ok"):
        return None, None, None
    c = np.array([float(g["centre_xy"][0]), float(g["centre_xy"][1]), float(g["rest_z"])], float)
    print("BIND_SUPPORT " + json.dumps({"prompt": list(prompts), "n_instances": int(g["n_open_parts"]), "select": str(code), "center": [round(float(v), 4) for v in c],
                                        "half_extent": [round(float(v), 4) for v in g["half_extent"]], "rule": "open_sliding_part"}), flush=True)
    return c, np.asarray(pts, float), None
# ===== end C6I3 place_primitive =====


# ===== C4I1 place_primitive: the reference of a beside relation is a body (round 20260924, chunk 4, iteration 1) =====
# "Put X to the <side> of Y" names Y as a landmark, not as a surface X rests on. The K1 gate (support_candidate_ok, the 0.06 m
# OBB-centre candidate bound of localize_support_by_side) recognises a resting support: low, raised over the floor. A landmark
# that stands taller (recorded: a partitioned body, visible top 14.1 cm over the floor, OBB centre 0.060-0.062 m) never enters
# the candidate set, and the tall look-alike the reference prompts also match is (rightly) rejected as too tall; the parent
# answers support_not_found. Rule (binding branch, relation == "beside_support" only, reached only where the parent returns no
# support): the reference is bound as a BODY by the program's object-binding rule: the shared bind_object over the reference
# prompts (instance support = mean best score over the responding prompts; ambiguous -> fail closed) with the C1I3 consistent-low
# acceptance of its not_found branch (bind_accept_consistent, unchanged); the body must not be the bound object (overlap >=
# SUPPORT_SAME_MASK relative to the smaller mask, the K1 criterion) and must stand over the floor (median rise >= SUPPORT_RISE_MIN,
# the K1 rise test). Returns what localize_support_by_side returns: (OBB centre, points, mask) or (None, None, None).
# Place target (beside branch of place_target): the K2 rule offsets the target from a rim CIRCLE (pv_support_from_points); when
# its circle fit fails (the reference is no round support: the verifier then reports the median reach of the outline, 12 cm for
# a 43 cm long body) the offset is taken from the reference's own extent along the side direction d: target = the middle of its
# extent across d + d x (its far edge along d + r_object + BESIDE_EXTENT_CLEAR), extents from the 3-97 percentile band
# (BESIDE_EXTENT_PCT: fusion_geometry.half_extent_along, the statistic the stack already uses for the object's half extent in
# the same beside validation); the band spans 94 % of a uniformly covered extent, so the far edge is the band's middle + its half
# width / 0.94 (the band alone ends 1.3 cm inside a 43 cm body's true end). A round reference (fit ok) keeps the K2 rule byte for byte.
BESIDE_EXTENT_PCT = (3, 97)   # percentiles of the extent band = fusion_geometry.half_extent_along (3-97)
BESIDE_EXTENT_CLEAR = 0.015   # m, footprint edge to the reference's far edge = fusion_geometry REGION_CLEAR (0.005) + PERCEPTION xy_sigma (0.010): the clearance the stack's beside validation demands (validate_beside_target's need), applied to the extent it is measured on


def bind_beside_reference(rgb, depth_img, K, E, prompts, exclude_mask, floor_z):
    """The body a beside relation names as its landmark (rule above). Logs BIND_REFERENCE (+ BIND_SUPPORT on acceptance)."""
    r = bind_object(rgb, depth_img, K, E, list(prompts))
    status = r.get("status"); rule = "shared"; low_info = None
    if status == "not_found":
        ok_low, low_info = bind_accept_consistent(r, len(list(prompts)), int(np.asarray(rgb).shape[0]) * int(np.asarray(rgb).shape[1]))
        if ok_low:
            status = "bound"; rule = "consistent_low"
    rec = {"status": status, "rule": rule, "support": None if r.get("support") is None else round(float(r["support"]), 3), "rel_margin": None if r.get("rel_margin") is None else round(float(r["rel_margin"]), 3),
           "instances": (r.get("instances") or [])[:3], "low": low_info}
    pts = r.get("pts"); mask = r.get("mask")
    if status != "bound" or pts is None or mask is None or len(pts) < 10:
        rec["why"] = "not_bound"; print("BIND_REFERENCE " + json.dumps(rec, default=str), flush=True); return None, None, None
    pts = np.asarray(pts, float); mask = np.asarray(mask).astype(np.uint8)
    if exclude_mask is not None:
        rec["overlap_bound"] = round(mask_overlap_frac(mask, exclude_mask), 3)
        if rec["overlap_bound"] >= SUPPORT_SAME_MASK:
            rec["why"] = "is_bound_object"; print("BIND_REFERENCE " + json.dumps(rec, default=str), flush=True); return None, None, None
    if floor_z is not None:
        rec["rise"] = round(float(np.median(pts[:, 2])) - float(floor_z), 4)
        if rec["rise"] < SUPPORT_RISE_MIN:
            rec["why"] = "not_raised_over_floor"; print("BIND_REFERENCE " + json.dumps(rec, default=str), flush=True); return None, None, None
    c = get_oriented_bounding_box_from_3d_points(pts)["center"]
    rec["why"] = "bound_as_body"; rec["center"] = [round(float(v), 4) for v in c]
    print("BIND_REFERENCE " + json.dumps(rec, default=str), flush=True)
    print("BIND_SUPPORT " + json.dumps({"prompt": list(prompts), "n_instances": 1, "select": None, "center": [round(float(v), 4) for v in c], "rule": "body_binding"}), flush=True)
    return c, pts, mask


# ===== C18I3 binding: a tall beside landmark is not replaced by a low look-alike (N18 group P, slot chunk18_iter3) =====
# "Put X to the <side> of Y" names Y as a landmark (C4I1). The K1 resting-support gate of localize_support_by_side sees only low
# bodies (OBB centre below 0.06 m, visible top <= SUPPORT_TOP_MAX); a landmark that stands taller never enters its candidate set
# and is dropped without a record. C4I1 binds such a landmark as a body, but only when the gate keeps NO candidate. When a low
# body that the reference prompts barely match passes the gate, the parent binds it as the lone instance and places beside the
# wrong body (recorded: 7/7 development rows over 3 initial states; the kept body scores at most 0.04 under every reference
# prompt, the excluded landmark 0.016 / 0.56 / 0.07; the released object lands beside the look-alike: official failure in 7/7).
# Rule (beside_support only, reached only where the parent would return its single_instance pick): the landmark is bound by the
# C4I1 body rule (the shared bind_object over the reference prompts, the C1I3 consistent-low acceptance of its not_found branch,
# not the bound object, raised over the floor). The body replaces the parent's pick only when
#   (a) it is a different instance (mask overlap with the pick < SUPPORT_SAME_MASK, the K1 same-instance criterion), and
#   (b) it lies inside the parent's candidate xy bounds but OUTSIDE the K1 resting-support bounds (OBB centre z >= the beside
#       bound 0.06 m, or visible top over the floor > SUPPORT_TOP_MAX): exactly the body the K1 gate cannot see. A body the gate
#       did see and rejected or ranked lower keeps the parent's pick.
# The shared rule's support (mean best score over the responding prompts) already ranks the landmark above the look-alike when it
# is bound (MIN_SUPPORT, MIN_REL_MARGIN). Otherwise (not bound, same instance, low body, outside xy) the parent's pick is returned
# unchanged. Placement (place_target's C4I1 extent rule, validation, contract, prim_place) is the parent's. No new constant.
def beside_tall_body(rgb, depth_img, K, E, prompts, exclude_mask, floor_z, pick):
    """The landmark body that replaces a lone K1 pick of a beside relation (rule above), or None (keep the pick). Logs BESIDE_BODY."""
    r = bind_object(rgb, depth_img, K, E, list(prompts))
    status = r.get("status"); rule = "shared"; low_info = None
    if status == "not_found":
        ok_low, low_info = bind_accept_consistent(r, len(list(prompts)), int(np.asarray(rgb).shape[0]) * int(np.asarray(rgb).shape[1]))
        if ok_low:
            status = "bound"; rule = "consistent_low"
    rec = {"status": status, "rule": rule, "support": None if r.get("support") is None else round(float(r["support"]), 3), "rel_margin": None if r.get("rel_margin") is None else round(float(r["rel_margin"]), 3),
           "instances": (r.get("instances") or [])[:3], "low": low_info, "pick_center": [round(float(v), 4) for v in np.asarray(pick[0], float)]}
    pts = r.get("pts"); mask = r.get("mask")
    if status != "bound" or pts is None or mask is None or len(pts) < 10:
        rec["why"] = "not_bound"; print("BESIDE_BODY " + json.dumps(rec, default=str), flush=True); return None
    pts = np.asarray(pts, float); mask = np.asarray(mask).astype(np.uint8)
    rec["overlap_pick"] = round(mask_overlap_frac(mask, pick[2]), 3)
    if rec["overlap_pick"] >= SUPPORT_SAME_MASK:
        rec["why"] = "same_instance"; print("BESIDE_BODY " + json.dumps(rec, default=str), flush=True); return None
    if exclude_mask is not None:
        rec["overlap_bound"] = round(mask_overlap_frac(mask, exclude_mask), 3)
        if rec["overlap_bound"] >= SUPPORT_SAME_MASK:
            rec["why"] = "is_bound_object"; print("BESIDE_BODY " + json.dumps(rec, default=str), flush=True); return None
    c = get_oriented_bounding_box_from_3d_points(pts)["center"]; rec["center"] = [round(float(v), 4) for v in c]
    if not (0.2 < c[0] < 0.8 and -0.5 < c[1] < 0.5):
        rec["why"] = "outside_candidate_xy"; print("BESIDE_BODY " + json.dumps(rec, default=str), flush=True); return None
    if floor_z is not None:
        rec["rise"] = round(float(np.median(pts[:, 2])) - float(floor_z), 4); rec["top"] = round(float(np.percentile(pts[:, 2], 98)) - float(floor_z), 4)
        if rec["rise"] < SUPPORT_RISE_MIN:
            rec["why"] = "not_raised_over_floor"; print("BESIDE_BODY " + json.dumps(rec, default=str), flush=True); return None
    tall = float(c[2]) >= 0.06 or (floor_z is not None and float(rec["top"]) > SUPPORT_TOP_MAX)
    if not tall:
        rec["why"] = "inside_resting_bounds"; print("BESIDE_BODY " + json.dumps(rec, default=str), flush=True); return None
    rec["why"] = "tall_body_over_lookalike"; print("BESIDE_BODY " + json.dumps(rec, default=str), flush=True)
    print("BIND_SUPPORT " + json.dumps({"prompt": list(prompts), "n_instances": 1, "select": None, "center": rec["center"], "rule": "tall_body_over_lookalike"}), flush=True)
    return c, pts, mask
# ===== end C18I3 binding =====


def beside_extent_target(support_pts, d, r_obj):
    """Beside target from the reference's extent along the unit side direction d (rule above): dict(centre_xy, half_along, far_edge,
    offset, nominal, n) or None with fewer than CONTAINER_MIN_PTS points."""
    P = np.asarray(support_pts, float).reshape(-1, 3)[:, :2]
    if len(P) < CONTAINER_MIN_PTS:
        return None
    d = np.asarray(d, float)[:2]; d = d / (float(np.linalg.norm(d)) + 1e-12); w = np.array([-d[1], d[0]])
    lo_d, hi_d = np.percentile(P @ d, BESIDE_EXTENT_PCT); lo_w, hi_w = np.percentile(P @ w, BESIDE_EXTENT_PCT)
    c = d * (0.5 * (lo_d + hi_d)) + w * (0.5 * (lo_w + hi_w)); cover = 0.01 * float(BESIDE_EXTENT_PCT[1] - BESIDE_EXTENT_PCT[0])
    half = 0.5 * float(hi_d - lo_d) / cover   # the band spans `cover` of a uniformly covered extent: the full half extent
    offset = half + float(r_obj) + BESIDE_EXTENT_CLEAR
    return {"centre_xy": c, "half_along": half, "band_half": 0.5 * float(hi_d - lo_d), "far_edge": float(0.5 * (lo_d + hi_d) + half), "offset": float(offset), "nominal": c + d * offset, "n": int(len(P))}
# ===== C4I2 place_primitive: the clearance beside a tall reference covers the object's whole footprint (round 20260924, chunk 4, iteration 2) =====
# The extent rule above reserves r_object (max(r_top, r_mid) of the object's profile: the radius of its round body) between the
# reference's far edge + BESIDE_EXTENT_CLEAR and the object's axis. A held object is not always a disc about its axis: a part
# that stands out of the round body (the profile's "protrusion" bins) still descends with it, and beside a reference that stands
# taller than the object's release height it meets the reference's wall (recorded: 10/10 descents stopped by the held object
# against the reference; its observed points reach 8.0 cm from the axis, the reserved radius 4.8 cm). Rule: the reach of the
# object's footprint = the BESIDE_REACH_PCT percentile of the horizontal distances of its observed points from the profile's
# axis (the axis the placement positions); the extent rule reserves max(r_object, reach). The reach is a radius, so it holds
# for any yaw of the object at the release (a protrusion facing away, a transport orientation other than as_is). The percentile
# is the stack's outline-reach statistic (CONTAINER_PCT_HI), which leaves out mask-edge points that fall off the body.
BESIDE_REACH_PCT = CONTAINER_PCT_HI   # percentile of the point distances from the axis taken as the footprint's reach (the stack's p98 outline reach)


def object_footprint_reach(obj_pts, prof):
    """Reach of the object's footprint about its profile axis (rule above): the BESIDE_REACH_PCT percentile of the horizontal
    distances of its observed points from prof["axis_xy"]; None without a profile or with fewer than CONTAINER_MIN_PTS points."""
    if not prof or prof.get("axis_xy") is None:
        return None
    P = np.asarray(obj_pts, float).reshape(-1, 3)[:, :2]
    if len(P) < CONTAINER_MIN_PTS:
        return None
    a = np.asarray(prof["axis_xy"], float)[:2]
    return float(np.percentile(np.hypot(P[:, 0] - a[0], P[:, 1] - a[1]), BESIDE_REACH_PCT))
# ===== end C4I2 place_primitive =====


# ===== end C4I1 place_primitive (the place_target branch and the localize_support_by_side fallback are marked C4I1) =====


# ===== C6I1 binding: the top face of a tall support (round 20260924, chunk 6, iteration 1) =====
# "on top of the <reference>" names the reference's top face (the template's literal side word, la_literal_sides: side code zmax).
# The K1 gate above admits supports resting on the floor only (OBB centre < 0.06 m, visible top <= SUPPORT_TOP_MAX): a tall body's
# top face (recorded: 0.227 m over the floor) is out of both bounds, and a low part in front of it (an open sliding part) is admitted
# instead. With side zmax on the on-support relation every candidate is measured by its top face and the highest face is the support:
#   (1) candidates: the parent's xy bounds; the OBB-centre z bound is the workspace box (C_WS), as for the container relations;
#       the kind band (LOW_REL_MARGIN below the best in-bounds score) and the bound-object overlap test (SUPPORT_SAME_MASK) unchanged;
#   (2) the top face = the candidate's points within CONTAINER_PLANE_BIN (5 mm bins > the depth noise) of its p98 level (the K1
#       "top" statistic); at least CONTAINER_MIN_PTS points; its centre and half extents are the p2 / p98 extents along the axes of
#       its minimum-area rectangle (robust to the edge points of the side faces inside the band); its level is the median height;
#   (3) the body stands over the floor: median rise >= SUPPORT_RISE_MIN (the K1 rise test; a floor mask has no raised face);
#   (4) the face holds the held object's footprint: object_footprint_radius (the contract's object radius) <= the smaller half extent;
#   (5) among the admitted candidates of the first prompt that has any, the highest face (zmax is the extreme along z, as min / max are
#       the extremes along y); the placement target is its centre, the rest reference (table_z / support_z) its level (the returned
#       points are the face's).
# The descent over a top face starts at the stack's transport-clearance height at the target (Q1 C rule: face + hang + C_CLEAR, with
# the grasp-frame hang), never below the parent's lift height, and is refused above C_Z_MAX (c_transport_height's cap).
TOP_FACE_CODE = "zmax"                  # the side code of the adapter's "top" (side_words); the code the on relation implements for a top face
TOP_FACE_BAND = CONTAINER_PLANE_BIN     # m, the face band under the p98 level (5 mm bins, > PERCEPTION z_noise)


def support_top_face(pts, floor_z, r_object=None):
    """Top face of one support candidate (public points only) -> (face | None, reason, features). face = {centre_xy, half_extent,
    rest_z, pts}."""
    P = np.asarray(pts, float).reshape(-1, 3); z = P[:, 2]
    top = float(np.percentile(z, CONTAINER_PCT_HI)); face = P[z >= top - TOP_FACE_BAND]
    feat = {"n_pts": int(len(P)), "top": round(top, 4), "n_face": int(len(face))}
    if floor_z is not None:
        feat["rise"] = round(float(np.median(z)) - float(floor_z), 4); feat["top_over_floor"] = round(top - float(floor_z), 4)
        if feat["rise"] < SUPPORT_RISE_MIN:
            return None, "not_raised_over_floor", feat
    if len(face) < CONTAINER_MIN_PTS:
        return None, "no_top_face", feat
    _c, a, b, _hu, _hv = _ca_min_rect(face[:, :2])
    u = face[:, :2] @ a; v = face[:, :2] @ b
    ulo, uhi = (float(x) for x in np.percentile(u, [CONTAINER_PCT_LO, CONTAINER_PCT_HI])); vlo, vhi = (float(x) for x in np.percentile(v, [CONTAINER_PCT_LO, CONTAINER_PCT_HI]))
    centre = a * (ulo + uhi) / 2.0 + b * (vlo + vhi) / 2.0; half = ((uhi - ulo) / 2.0, (vhi - vlo) / 2.0)
    rest_z = float(np.median(face[:, 2]))
    feat.update({"centre_xy": [round(float(centre[0]), 4), round(float(centre[1]), 4)], "half_extent": [round(half[0], 4), round(half[1], 4)], "rest_z": round(rest_z, 4),
                 "r_object": None if r_object is None else round(float(r_object), 4)})
    if r_object is not None and min(half) < float(r_object):
        return None, "top_face_too_small", feat
    return {"centre_xy": np.asarray(centre, float), "half_extent": half, "rest_z": rest_z, "pts": face}, "ok", feat


def bind_top_face(rgb, depth_img, K, E, prompts, exclude_mask, floor_z, r_object=None):
    """Support of an on relation with side zmax: the highest top face among the admitted candidates of the first prompt that has
    any (rule above). Returns (centre [x, y, rest_z], face points, mask) or (None, None, None)."""
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        cands = []; rejected = []; s_top = None
        for m in sorted(masks, key=lambda d: -float(d["score"]))[:15]:
            mask = m["mask"].astype(np.uint8); pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 10:
                continue
            c = get_oriented_bounding_box_from_3d_points(pts)["center"]
            if not (0.2 < c[0] < 0.8 and -0.5 < c[1] < 0.5 and -0.02 < c[2] < C_WS[2][1]):
                continue
            if s_top is None:
                s_top = float(m["score"])
            if float(m["score"]) < (1.0 - LOW_REL_MARGIN) * s_top:
                rejected.append({"why": "below_kind_band", "score": round(float(m["score"]), 3)}); continue
            if exclude_mask is not None and mask_overlap_frac(mask, exclude_mask) >= SUPPORT_SAME_MASK:
                rejected.append({"why": "is_bound_object", "score": round(float(m["score"]), 3)}); continue
            f, why, feat = support_top_face(pts, floor_z, r_object=r_object)
            if f is None:
                rejected.append({"why": why, "score": round(float(m["score"]), 3), **{k: feat.get(k) for k in ("top_over_floor", "half_extent", "r_object", "n_face") if feat.get(k) is not None}}); continue
            cands.append((f, mask, float(m["score"]), feat))
        print("SUPPORT_GATE " + json.dumps({"prompt": prompt, "relation": "on_support", "rule": "top_face", "floor_z": None if floor_z is None else round(floor_z, 4), "n_kept": len(cands), "score_top": s_top,
                                            "faces": [{"score": round(s, 3), "rest_z": ft["rest_z"], "centre_xy": ft["centre_xy"], "half_extent": ft["half_extent"]} for _f, _m, s, ft in cands[:6]], "rejected": rejected[:6]}), flush=True)
        if cands:
            f, mask, s, ft = max(cands, key=lambda t: t[0]["rest_z"])
            centre = np.array([float(f["centre_xy"][0]), float(f["centre_xy"][1]), float(f["rest_z"])])
            print("BIND_SUPPORT " + json.dumps({"prompt": prompt, "n_instances": len(cands), "select": TOP_FACE_CODE, "center": [round(float(v), 4) for v in centre], "half_extent": ft["half_extent"], "rule": "top_face_highest"}), flush=True)
            return centre, np.asarray(f["pts"], float), mask
    return None, None, None


# ===== C19I5 binding: the resting base of a handled ring body is what a top face must hold (N19 group P2, slot chunk19_iter5) =====
# C6I1 admits a top face only when its smaller half extent holds object_footprint_radius = max(r_top, r_mid) of the held body's
# profile. For a round hollow body with a compact protrusion (C4I4 gw_ring + C12I2 gh_protrusion) that radius is the protrusion's
# reach (recorded: 0.166 m against a 0.092 m face, development rows 7/7 stop before motion). What rests on a horizontal face is the
# body's lowest level, its base; the flared rim and a protrusion at rim height lie above that level and touch nothing past the
# face edge. C12I4 places the ring centre, not the pinch point, on the target (the face centre). Rule (tf_rest_radius):
#   base = the ring body's points (within r + GW_RING_TOL of the ring centre) within TOP_FACE_BAND of its lowest level (the ring's
#   p2 base): the mirror of C6I1's face band under the support's p98 top; >= CONTAINER_MIN_PTS points, and fewer than GA_MIN_PTS
#   points outside the ring at that level that stand over the floor by SUPPORT_RISE_MIN (the K1 / C3I2 rise test; nothing but the
#   base rests there - points of the mask boundary that bleed onto the floor beside the protrusion have zero rise: recorded 41-68
#   per live pre-grasp mask, which made the C19I4 version fail closed; without a floor estimate every outside point counts); r_rest = the CONTAINER_PCT_HI radius of the base about
#   the ring centre (recorded 0.0755-0.0757 m, flat floor of a flared vessel whose rim is 0.101 m).
# Reached only on a top-face relation, and only after the parent's support binding returned nothing with the parent's radius (the
# point of the parent's support_not_found stop): when r_rest is smaller, the same localize_support_by_side call is repeated with
# r_object = r_rest; every other gate is the parent's. Anything else (no ring, no protrusion, too few base points, a protrusion at
# the base level, r_rest not smaller, any error) keeps the parent's stop. No new constant; no guard, threshold or budget change.
def tf_rest_radius(obj_pts, E):
    """Resting-base radius of a round hollow body with a compact protrusion (rule above): (r_rest | None, info)."""
    P = np.asarray(obj_pts, float).reshape(-1, 3)
    ring, rinfo = gw_ring(P, np.asarray(E)[:2, 3])
    if ring is None:
        return None, {"why": "no_ring", "ring": rinfo.get("why")}
    S, _u, pinfo = gh_protrusion(P, ring)
    info = {"ring_r": round(float(ring["r"]), 4), "base": round(float(ring["base"]), 4), "z_rim": round(float(ring["z_rim"]), 4)}
    if S is None:
        return None, {**info, "why": "no_protrusion", "protrusion": pinfo.get("why")}
    c = np.asarray(ring["c"], float)[:2]; rr = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    band = P[:, 2] <= float(ring["base"]) + TOP_FACE_BAND
    inner = band & (rr <= float(ring["r"]) + GW_RING_TOL); outer = band & (rr > float(ring["r"]) + GW_RING_TOL)
    floor_z = _CG.get("floor_z"); n_floor = 0
    if floor_z is not None:   # C19I5: floor pixels (zero rise) are not a part of the body resting at the base level
        on_floor = outer & (P[:, 2] - float(floor_z) < SUPPORT_RISE_MIN); n_floor = int(on_floor.sum()); outer = outer & ~on_floor
    info.update(n_base=int(inner.sum()), n_base_outside=int(outer.sum()), n_outside_floor=n_floor, floor_z=None if floor_z is None else round(float(floor_z), 4), protrusion_u=pinfo.get("u"))
    if int(inner.sum()) < CONTAINER_MIN_PTS:
        return None, {**info, "why": "no_base_level"}
    if int(outer.sum()) >= GA_MIN_PTS:
        return None, {**info, "why": "protrusion_at_base_level"}
    r_rest = float(np.percentile(rr[inner], CONTAINER_PCT_HI))
    info["r_rest"] = round(r_rest, 4)
    return r_rest, info


def tf_fit_radius(obj_pts, E, r_object):
    """The radius a top face must hold for the retry: r_rest when the held body is a handled ring body whose base is smaller than
    r_object, else r_object (no retry). Logs TOP_FACE_FIT."""
    try:
        r_rest, info = tf_rest_radius(obj_pts, E)
    except Exception as e:  # noqa: BLE001  (any error: the parent's stop)
        r_rest, info = None, {"why": f"error: {type(e).__name__}: {e}"}
    use = r_rest is not None and float(r_rest) < float(r_object)
    print("TOP_FACE_FIT " + json.dumps({"r_object": round(float(r_object), 4), "use_rest_base": bool(use), **info}, default=str), flush=True)
    return float(r_rest) if use else float(r_object)
# ===== end C19I5 binding =====


def top_face_lift(lift_z, rest_z, hang):
    """Descent-start height over a top face: the transport-clearance rule at the target (rest level + hang + C_CLEAR), never below the
    parent's lift height; None above the transport cap C_Z_MAX (the transport would not reach it)."""
    z = max(float(lift_z), float(rest_z) + float(hang) + C_CLEAR)
    return None if z > C_Z_MAX else z
# ===== end C6I1 binding =====


# ===== C10I1 binding: the reference side of a position-named support / a depth side of beside (N10, slot C10I1) =====
# (1) on + an instance code (mid, or a depth code xmin / xmax): the side word names WHICH instance of the reference kind is the
# support. The instance is chosen by the unchanged C3I1 rule (bind_select_by_side: instances of the kind under every reference
# prompt, the look-alike candidate band, the observer's axes from the public E, the SIDE_MIN_GAP separation test; mid = the
# middle of an odd set of at least 3 along whichever of lateral / depth spreads the candidates more). The set is the whole
# arrangement, the bound object's instance included: "the middle one" is defined over all instances, and removing the object
# first (the min / max path) leaves an even set without a middle. When object and reference are of one kind the object was
# chosen by the same rule on the same set, so the two choices are consistent. The chosen instance must then pass the parent's
# K1 support gates unchanged: the candidate bounds of localize_support_by_side, and support_candidate_ok (not the bound object:
# SUPPORT_SAME_MASK majority overlap; raised over the floor by SUPPORT_RISE_MIN; top <= SUPPORT_TOP_MAX). Any failure returns no
# support (support_not_found: no motion). No new constant.
# (2) beside + a depth code: the side word names a REGION of the reference along the observer's depth axis (bind_side_dirs: the
# horizontal perpendicular of the image-x axis, signed away from the camera; xmin = toward the camera, xmax = away), the
# reading C3I1 uses for instances. The reference point only fixes the sign (the bound object's centre). No axis (image-x axis
# vertical) or an observer standing over the reference (away < SIDE_MIN_GAP) -> no direction -> the subgoal stops before motion.
# The placement itself (place_target's outside-rim / outside-extent offset, validation, contract, prim_place) is the parent's.
# (3) beside + a depth code with several reference instances passing the K1 gate: ordered along the same depth axis (the
# parent orders min / max beside references along the lateral axis the same way).
SUPPORT_INSTANCE_CODES = SIDE_DEPTH + (SIDE_MIDDLE,)   # on: side codes that select the support instance by bind_select_by_side
BESIDE_DEPTH_CODES = SIDE_DEPTH                         # beside: side codes whose region direction is the observer's depth axis


def beside_depth_dir(E, ref_xy, code):
    """Unit region direction (base xy) of a beside relation with a depth side code (rule (2)); None when it is undefined."""
    if E is None or ref_xy is None or str(code) not in BESIDE_DEPTH_CODES:
        return None
    lat, dep, info = bind_side_dirs(E, np.asarray(ref_xy, float)[:2])
    if dep is None or float(info.get("away_m", 0.0)) < SIDE_MIN_GAP:
        print("BESIDE_DEPTH_DIR " + json.dumps({"code": str(code), "ok": False, **info}), flush=True)
        return None
    d = -np.asarray(dep, float) if str(code) == "xmin" else np.asarray(dep, float)
    print("BESIDE_DEPTH_DIR " + json.dumps({"code": str(code), "ok": True, "dir": [round(float(v), 4) for v in d], **info}), flush=True)
    return d


def bind_support_instance(rgb, depth_img, K, E, prompts, select, exclude_mask, floor_z, r_object=None):
    """Support of an on relation whose side code names the instance (rule (1)). Returns (OBB centre, points, mask) or
    (None, None, None). Logs SUPPORT_SIDE (+ BIND_SUPPORT on acceptance)."""
    inst, info = bind_select_by_side(rgb, depth_img, K, E, list(prompts), select)
    rec = {"select": str(select), "rule": "side_select", "side": {k: info.get(k) for k in ("why", "n_instances", "n_candidates", "axis", "projections", "chosen", "gap_m", "centres")}}
    if inst is None:
        rec["why"] = "side_" + str(info.get("why")); print("SUPPORT_SIDE " + json.dumps(rec, default=str), flush=True)
        return None, None, None
    pts = np.asarray(inst["pts"], float); mask = np.asarray(inst["mask"]).astype(np.uint8)
    c = get_oriented_bounding_box_from_3d_points(pts)["center"]; rec["center"] = [round(float(v), 4) for v in c]
    if not (0.2 < c[0] < 0.8 and -0.5 < c[1] < 0.5 and -0.02 < c[2] < 0.06):   # the candidate bounds of localize_support_by_side (on relation)
        rec["why"] = "outside_candidate_bounds"; print("SUPPORT_SIDE " + json.dumps(rec, default=str), flush=True)
        return None, None, None
    ok, why, feat = support_candidate_ok(pts, mask, exclude_mask, floor_z, relation="on_support", r_object=r_object)
    rec.update(gate=why, **{k: v for k, v in feat.items()})
    if not ok:
        rec["why"] = why; print("SUPPORT_SIDE " + json.dumps(rec, default=str), flush=True)
        return None, None, None
    rec["why"] = "selected"; print("SUPPORT_SIDE " + json.dumps(rec, default=str), flush=True)
    print("BIND_SUPPORT " + json.dumps({"prompt": list(prompts), "n_instances": info.get("n_candidates"), "select": str(select), "center": rec["center"], "rule": "side_select"}), flush=True)
    return c, pts, mask
# ===== end C10I1 binding =====


def localize_support_by_side(rgb, depth_img, K, E, prompts, select, exclude_mask=None, relation=None, r_object=None, part=None):   # C3I3: part = the bay datum (in_part_support)   # C1I2: relation-dependent gate (in_support = container)
    """Support instance by side. Candidates = top masks of a support prompt inside the same candidate bounds as the parent
    two-support program that pass the K1 geometry gate (not the bound object; raised over the floor; low). Distinct instances are
    > 5 cm apart. The first prompt that yields >= 2 instances selects 'min' / 'max' along y (rule unchanged); when no prompt
    yields two, the single instance of the first prompt that yields one is the support (a lone instance is its own extreme)."""
    floor_z = support_floor_z(depth_img, K, E); _CG["floor_z"] = floor_z   # C1I2: reused by place_target for the container geometry
    if relation == "on_support" and select == TOP_FACE_CODE:   # C6I1: "on top of": the support is the highest top face (bind_top_face); every other relation / side runs the code below unchanged
        return bind_top_face(rgb, depth_img, K, E, prompts, exclude_mask, floor_z, r_object=r_object)
    if relation == "on_support" and select in SUPPORT_INSTANCE_CODES:   # C10I1: the side word names the support instance (bind_support_instance)
        return bind_support_instance(rgb, depth_img, K, E, prompts, select, exclude_mask, floor_z, r_object=r_object)
    z_hi = C_WS[2][1] if relation in ("in_support", "in_part_support") else 0.06   # C1I2: the parent's tall-body bound on the OBB centre is replaced by the container gate for in_support; unchanged otherwise
    single = None
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        cands = []; rejected = []; s_top = None
        for m in sorted(masks, key=lambda d: -float(d["score"]))[:15]:
            mask = m["mask"].astype(np.uint8); pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 10:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts); c = obb["center"]
            if 0.2 < c[0] < 0.8 and -0.5 < c[1] < 0.5 and -0.02 < c[2] < z_hi:
                if s_top is None:
                    s_top = float(m["score"])   # C3I1: the best-scoring mask of the kind inside the candidate bounds, gate or not (masks come in score order)
                if float(m["score"]) < (1.0 - LOW_REL_MARGIN) * s_top:   # C3I1: below the kind band of that mask: not an instance of the kind (the C1I3 look-alike band, as in bind_select_by_side)
                    rejected.append({"why": "below_kind_band", "score": round(float(m["score"]), 3), "center": [round(float(v), 4) for v in c], "score_top": round(s_top, 3)}); continue
                ok, why, feat = support_candidate_ok(pts, mask, exclude_mask, floor_z, relation=relation, r_object=r_object, part=part)   # K1: geometry gate; C1I2: container gate for in_support; C1I4: footprint size rule
                if ok:
                    cands.append((c, pts, mask))
                else:
                    rejected.append({"why": why, "score": round(float(m["score"]), 3), "center": [round(float(v), 4) for v in c], **feat})
        uniq = []
        for c in cands:
            if relation in ("in_support", "in_part_support") and any(mask_overlap_frac(c[2], u[2]) >= SUPPORT_SAME_MASK for u in uniq):
                continue   # C1I2: a part of an already kept container (its mask is covered by the kept mask): the same instance
            if all(float(np.hypot(*(np.asarray(c[0][:2]) - np.asarray(u[0][:2])))) > 0.05 for u in uniq):
                uniq.append(c)
        print("SUPPORT_GATE " + json.dumps({"prompt": prompt, "relation": relation, "floor_z": None if floor_z is None else round(floor_z, 4), "n_kept": len(cands), "n_instances": len(uniq), "score_top": s_top, "rejected": rejected[:6]}), flush=True)   # C3I1: score_top = the kind band's reference
        if len(uniq) >= 2:
            if select in BESIDE_DEPTH_CODES:   # C10I1: depth side code -> order along the observer's depth axis (xmin = nearest the camera)
                _lat, _dep, _dinfo = bind_side_dirs(E, np.mean([np.asarray(u[0][:2], float) for u in uniq], axis=0))
                if _dep is None:
                    print("BIND_SUPPORT " + json.dumps({"prompt": prompt, "n_instances": len(uniq), "select": select, "rule": "depth_axis_undefined", **_dinfo}), flush=True)
                    return None, None, None
                uniq.sort(key=lambda x: float(np.asarray(x[0][:2], float) @ _dep)); pick = uniq[0] if select == "xmin" else uniq[-1]
            else:
                uniq.sort(key=lambda x: float(x[0][1])); pick = uniq[0] if select == "min" else uniq[-1]
            print("BIND_SUPPORT " + json.dumps({"prompt": prompt, "n_instances": len(uniq), "select": select, "center": [round(float(v), 4) for v in pick[0]]}), flush=True)
            return pick
        if len(uniq) == 1 and single is None:
            single = (prompt, uniq[0])
    if single is not None:
        prompt, pick = single
        if relation == "beside_support":   # C18I3: a tall landmark the K1 gate cannot see is not replaced by this low pick (beside_tall_body)
            _body = beside_tall_body(rgb, depth_img, K, E, prompts, exclude_mask, floor_z, pick)
            if _body is not None:
                return _body
        print("BIND_SUPPORT " + json.dumps({"prompt": prompt, "n_instances": 1, "select": select, "center": [round(float(v), 4) for v in pick[0]], "rule": "single_instance"}), flush=True)
        return pick
    if relation == "beside_support":   # C4I1: no resting-support instance of a beside reference -> the reference is bound as a body (bind_beside_reference)
        return bind_beside_reference(rgb, depth_img, K, E, prompts, exclude_mask, floor_z)
    return None, None, None


# ===== C2I1 manipulation_primitive: close_articulated (round 20260924, chunk 2, iteration 1) =====
# Closing an articulated part by a straight push. Public data only: the segmenter's masks of the adapter's part / support prompts,
# the depth frame, the interface's motion contracts. The part is the component of the assembly that protrudes beyond the body's
# top-face outline; the body is the largest planar top level of the assembly. Two closing forms, decided by the part's horizontal
# footprint: a box protruding from a body face (a sliding part: its front is pushed along the face's inward normal until the joint stop)
# and a thin panel (a hinged door: pushed on its outer face along a direction rotated 20 deg from the panel normal toward the hinge
# end, so that one straight push keeps a sliding contact and sweeps the panel through ~73 deg). The hand pushes with the closed fingertips, either horizontal
# (tall parts: the fingers point along the push, the hand's 6.3 cm thin side is lateral) or tilted 45 deg (low parts: the tips lead
# the hand body by 1.3 cm, the body corner 4.4 cm above the tips). Constants: hand geometry (robot), the stack's percentiles and
# the xy sigma; DIAGNOSIS.md section 4.
from scipy import ndimage as _ndi
from scipy.spatial import cKDTree as _KDTree
CA_TOP_BAND = 0.02            # m, the body's top face = assembly points within this band below the assembly's top (p98): the verifier's gap band
CA_CELL = 0.01                # m, grid cell of the top-face occupancy
CA_OPEN_CELLS = 5             # cells: binary opening removes top-band structures thinner than 5 cm (a panel's top edge, 2.4 cm recorded); a body top is >= 19 cm
CA_MIN_TOP_CELLS = 100        # 100 cm2 of top face: the smallest body top (a 10 x 10 cm box)
CA_OUTLINE_MARGIN = 0.015     # m, points this far outside the body outline belong to a part (the body's own faces lie within the stack's 1 cm xy sigma of the outline)
CA_PART_MIN_PTS = 200         # points of a part component (a quarter of the smallest recorded handle bar, 650-800 points)
CA_CLUSTER_CELL = 0.02        # m, 3D cell of the part clustering (= C_CELL)
CA_PANEL_THIN = 0.08          # m, horizontal minor extent below which a component is a panel (recorded: panel top edge + handle 5-7 cm; sliding box 16-22 cm)
CA_FRONT_BAND = 0.015         # m, the front surface = points within this band of the part's extreme protrusion (1.5 x xy sigma)
CA_PUSH_GAP = 0.02            # m, pre-push standoff of the fingertip from the measured front surface (2 x xy sigma)
CA_PUSH_EXTRA = 0.02          # m, commanded travel beyond the closing face (a stop up to 2 cm past flush is reached; the stop ends the push)
CA_PUSH_TILTS_DEG = (45.0, 30.0)   # deg, tilt of the tool axis from vertical toward the push, tried in this order: the fingertips lead the hand body's corner by 1.3 cm (45) / 0.2 cm (30), the corner 4.4 / 4.5 cm above the tips; the finger's rear reaches 5.3 / 4.5 cm behind the tips
CA_PANEL_TILTS_DEG = (0.0, 15.0, 30.0, 45.0)   # deg, tilt candidates of a PANEL push (a box keeps CA_PUSH_TILTS_DEG): the hand body's bottom corner is 4.0-4.5 cm above the tips at every tilt of 0..45 (CA_HAND: 2.11 sin t + 4.0 cos t cm) while the tips are CA_PANEL_TIP_BELOW under the panel's top, so the body overhangs the panel at all of them; the tilt loads the wrist pitch joint toward its limit (recorded solutions 2.3 -> 3.5 rad from 0 to 45, limit 3.60), so the candidates span the range downward from the box list at the 15 deg quantum (0.26 rad of a wrist joint: adjacent candidates are kinematic neighbours under the interface's 0.45 rad continuity limit)

CA_PROBE_MAX = 4              # candidates whose approach may be executed and probed in one cycle: every panel tilt once at its better flip; a refused probe costs one free move (recorded 100-300 steps), so four stay under one cycle's budget of 1440 and the step guard still admits a second cycle
CA_PANEL_SWEEP_DEG = 20.0     # deg, panel push direction rotated from the panel normal toward the hinge end (72-73 deg swept per push with the push point at half the panel; the palm's far end (10.4 cm x sin 20 = 3.6 cm toward the closing face) stays outside that face at the closed position)
CA_PANEL_R0_FRAC = 0.5        # push point at half the panel length from the hinge end
CA_PROGRESS_MIN = 0.02        # m, the smallest protrusion change read as motion of the part (2 x xy sigma)
CA_CYCLES_MAX = 4             # push cycles per subgoal
CA_STEPS_GUARD = 2200         # control steps used by this primitive after which no new cycle starts (a cycle is <= 1440 steps of budgets)
CA_SAME_POINT = 0.005         # m, a frame point within this distance of an assembly point is that point (the same pixel)
CA_HANDLE_MAX = 0.05          # m, a body wholly within this distance in front of the part's front, inside its lateral extent and not continuing outside that zone, is a handle of the part
CA_TIP_BELOW = 0.005          # m, the fingertip is set this far below the lowest level of the front surface (a box) so the inclined finger face meets the surface's top edge
CA_PANEL_TIP_BELOW = 0.03     # m, the fingertip on a panel is this far below the panel's top: the hand body (corner 4.4 cm above the tips) overhangs the panel
CA_CLEAR_MARGIN = 0.01        # m, no visible surface point may lie within this margin of the closed hand's two boxes (depth / xy sigma)
CA_DESCENT_STEP = 0.03        # m, the vertical descent column is checked at this spacing
CA_FLOOR_CLEAR = 0.02         # m, lowest hand point over the floor
CA_HIGH_CLEAR = 0.05          # m, lowest hand point over the assembly's top at the high pose
CA_BACKOFF = 0.03             # m, retreat along -push after the push
BUDGET_PUSH = 480             # control steps of a push (contact primitive: ~10 steps per cm recorded; 45 cm)
PUSH_CORRIDOR = 0.03          # m, lateral corridor of the push (the hand slides along a rotating panel)
# Panda hand geometry relative to the TCP (grip site, 0.097 along the tool axis from the hand base), tool frame: x = finger-opening
# axis, y = thin axis, z = tool axis toward the tips (robosuite panda gripper collision geometry, fingers closed).
CA_HAND = {"tip": 0.009, "finger_half_wide": 0.0264, "finger_half_thin": 0.0105, "finger_base": -0.0445,
           "body_half_wide": 0.104, "body_half_thin": 0.0316, "body_bottom": -0.031, "body_top": -0.123}


def _ca_min_rect(xy):
    """Minimum-area bounding rectangle of 2D points: (centre, axis_major, axis_minor, half_major, half_minor)."""
    xy = np.asarray(xy, float); best = None
    for deg in range(0, 90):
        t = np.radians(deg); a = np.array([np.cos(t), np.sin(t)]); b = np.array([-np.sin(t), np.cos(t)])
        u = xy @ a; v = xy @ b; area = (u.max() - u.min()) * (v.max() - v.min())
        if best is None or area < best[0]:
            best = (area, a, b, u, v)
    _, a, b, u, v = best
    c = a * (u.max() + u.min()) / 2 + b * (v.max() + v.min()) / 2; hu = (u.max() - u.min()) / 2; hv = (v.max() - v.min()) / 2
    if hv > hu:
        a, b, hu, hv = b, a, hv, hu
    return c, a, b, float(hu), float(hv)


def ca_assembly(rgb, depth_img, K, E, prompts, floor_z):
    """Union of the masks (all prompts, top 15 each) whose in-box raised points are a majority of the mask; the largest is the anchor,
    masks overlapping it by SUPPORT_SAME_MASK are merged. Returns (pts, info) or (None, info)."""
    cands = []; rej = 0
    for prompt in prompts:
        for m in sorted(segment_sam3_text_prompt(rgb, prompt) or [], key=lambda d: -float(d["score"]))[:15]:
            mask = np.asarray(m["mask"]).astype(np.uint8); pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < CA_PART_MIN_PTS:
                rej += 1; continue
            pts = np.asarray(pts, float)
            inbox = (pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] < C_WS[2][1])
            raised = inbox & (pts[:, 2] > float(floor_z) + SUPPORT_RAISED_MIN)
            if int(inbox.sum()) < 0.5 * len(pts) or int(raised.sum()) < CA_PART_MIN_PTS:
                rej += 1; continue
            cands.append({"prompt": prompt, "score": float(m["score"]), "mask": mask, "n": int(raised.sum())})
    if not cands:
        return None, {"n_candidates": 0, "rejected": rej}
    anchor = max(cands, key=lambda c: c["n"]); merged = [c for c in cands if c is anchor or mask_overlap_frac(c["mask"], anchor["mask"]) >= SUPPORT_SAME_MASK]
    union = np.zeros_like(anchor["mask"])
    for c in merged:
        union |= c["mask"]
    pts = np.asarray(mask_to_world_points(union, depth_img, K, E), float)
    keep = (pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] < C_WS[2][1]) & (pts[:, 2] > float(floor_z) + SUPPORT_RAISED_MIN)
    info = {"n_candidates": len(cands), "rejected": rej, "anchor": {"prompt": anchor["prompt"], "score": round(anchor["score"], 3), "n": anchor["n"]}, "n_merged": len(merged), "n_pts": int(keep.sum()),
            "merged": [{"prompt": c["prompt"], "score": round(c["score"], 3), "n": c["n"]} for c in merged]}
    return pts[keep], info


def ca_body(pts):
    """Body = the largest planar top level of the assembly: the top-band points on a 1 cm grid, opened with CA_OPEN_CELLS (thin edges of
    a panel vanish), the largest connected cell region, its minimum-area rectangle. None when no top face survives."""
    z = pts[:, 2]; z_top = float(np.percentile(z, CONTAINER_PCT_HI)); top = pts[z >= z_top - CA_TOP_BAND]
    if len(top) < CA_MIN_TOP_CELLS:
        return None
    x0, y0 = top[:, 0].min(), top[:, 1].min(); ix = ((top[:, 0] - x0) / CA_CELL).astype(int); iy = ((top[:, 1] - y0) / CA_CELL).astype(int)
    occ = np.zeros((ix.max() + 1 + 2 * CA_OPEN_CELLS, iy.max() + 1 + 2 * CA_OPEN_CELLS), bool); occ[ix + CA_OPEN_CELLS, iy + CA_OPEN_CELLS] = True
    opened = _ndi.binary_opening(occ, structure=np.ones((CA_OPEN_CELLS, CA_OPEN_CELLS), bool))
    lab, n = _ndi.label(opened)
    if n == 0:
        return None
    sizes = _ndi.sum(opened, lab, index=np.arange(1, n + 1)); k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < CA_MIN_TOP_CELLS:
        return None
    ci, cj = np.where(lab == k); cells = np.stack([x0 + (ci - CA_OPEN_CELLS + 0.5) * CA_CELL, y0 + (cj - CA_OPEN_CELLS + 0.5) * CA_CELL], 1)
    c, a, b, hu, hv = _ca_min_rect(cells)
    return {"centre": c, "axes": (a, b), "half": (hu + CA_CELL / 2, hv + CA_CELL / 2), "z_top": z_top, "n_top": int(len(top)), "n_cells": int(sizes[k - 1]), "n_regions": int(n)}


def ca_parts(pts, body):
    """Components of the assembly outside the body outline (3D grid clustering). Each: points, centroid, the body face it protrudes from
    (outward normal n_out, offset), its protrusion (p98 beyond the face), and its horizontal minimum rectangle."""
    c = body["centre"]; a, b = body["axes"]; hu, hv = body["half"]
    rel = pts[:, :2] - c; u = rel @ a; v = rel @ b
    out = (np.abs(u) > hu + CA_OUTLINE_MARGIN) | (np.abs(v) > hv + CA_OUTLINE_MARGIN)
    P = pts[out]
    if len(P) < CA_PART_MIN_PTS:
        return []
    g = np.floor((P - P.min(0)) / CA_CLUSTER_CELL).astype(int); occ = np.zeros(g.max(0) + 1, bool); occ[g[:, 0], g[:, 1], g[:, 2]] = True
    lab, n = _ndi.label(occ, structure=np.ones((3, 3, 3), bool)); L = lab[g[:, 0], g[:, 1], g[:, 2]]
    comps = []
    for k in range(1, n + 1):
        Q = P[L == k]
        if len(Q) < CA_PART_MIN_PTS:
            continue
        m = Q.mean(0); mu = float((m[:2] - c) @ a); mv = float((m[:2] - c) @ b)
        faces = [(mu - hu, a, hu), (-mu - hu, -a, hu), (mv - hv, b, hv), (-mv - hv, -b, hv)]
        excess, n_out, off = max(faces, key=lambda f: f[0])
        s = (Q[:, :2] - c) @ n_out - off   # protrusion coordinate of every point beyond the face
        cc, pa, pb, phu, phv = _ca_min_rect(Q[:, :2])
        comps.append({"pts": Q, "n": int(len(Q)), "centroid": m, "n_out": n_out, "face_off": float(off), "excess": float(excess), "protrusion": float(np.percentile(s, CONTAINER_PCT_HI)),
                      "rect": {"centre": cc, "major": pa, "minor": pb, "half_major": phu, "half_minor": phv}, "z_lo": float(np.percentile(Q[:, 2], CONTAINER_PCT_LO)), "z_hi": float(np.percentile(Q[:, 2], CONTAINER_PCT_HI))})
    return comps


def ca_select_part(comps, side):
    """The part named by the side code (data: [axis]min|max; no axis = SIDE_AXIS) among the components; no side = the most protruding."""
    if not comps:
        return None
    if side is None:
        return max(comps, key=lambda q: q["protrusion"])
    sd = str(side).lower(); ext = sd[-3:]; ax = {"x": 0, "y": 1, "z": 2}.get(sd[:-3], SIDE_AXIS) if sd[:-3] in ("", "x", "y", "z") else None
    if ext not in ("min", "max") or ax is None:
        return None
    key = lambda q: float(q["centroid"][ax])   # noqa: E731
    return min(comps, key=key) if ext == "min" else max(comps, key=key)


def _ca_hand_boxes():
    """The closed hand as two axis-aligned boxes in tool coordinates: ((lo, hi) per axis) for the finger pair and the hand body."""
    H = CA_HAND
    return [((-H["finger_half_wide"], H["finger_half_wide"]), (-H["finger_half_thin"], H["finger_half_thin"]), (H["finger_base"], H["tip"])),
            ((-H["body_half_wide"], H["body_half_wide"]), (-H["body_half_thin"], H["body_half_thin"]), (H["body_top"], H["body_bottom"]))]


def ca_hand_frame(u, tilt_deg, flip=False):
    """Rotation (3x3, columns = tool axes) and wxyz quaternion of the pushing hand: tool z tilted tilt_deg from vertical toward the push
    (the fingertips point down and forward); tool x (finger opening, the wide axis) horizontal and perpendicular to the push."""
    u = np.asarray(u, float); u = np.array([u[0], u[1], 0.0]); u /= np.linalg.norm(u)
    w = np.array([-u[1], u[0], 0.0]) * (-1.0 if flip else 1.0)
    t = np.radians(float(tilt_deg)); zt = np.sin(t) * u - np.cos(t) * np.array([0.0, 0.0, 1.0])
    xt = w; yt = np.cross(zt, xt); R = np.stack([xt, yt, zt], 1)
    q = Rotation.from_matrix(R).as_quat(); return R, np.array([q[3], q[0], q[1], q[2]])


def ca_tip_offset(R, u):
    """World offset from the TCP to the leading fingertip point: the finger face that faces the push (tool +/-y), tip end (tool z)."""
    yt = R[:, 1]; sgn = 1.0 if float(yt[0] * u[0] + yt[1] * u[1]) >= 0 else -1.0
    return R @ np.array([0.0, sgn * CA_HAND["finger_half_thin"], CA_HAND["tip"]])


def ca_hand_low_z(R):
    """Height of the lowest corner of the hand's boxes relative to the TCP for the rotation R."""
    lo = 0.0
    for (x0, x1), (y0, y1), (z0, z1) in _ca_hand_boxes():
        for xv in (x0, x1):
            for yv in (y0, y1):
                for zv in (z0, z1):
                    lo = min(lo, float((R @ np.array([xv, yv, zv]))[2]))
    return lo


def ca_clearance(obst, tcp, R):
    """No obstacle point inside either hand box grown by CA_CLEAR_MARGIN, with the hand at (tcp, R). Returns (ok, n_conflicts, nearest)."""
    if obst is None or len(obst) == 0:
        return True, 0, None
    P = (np.asarray(obst, float) - np.asarray(tcp, float)) @ R; n = 0; nearest = None
    for (x0, x1), (y0, y1), (z0, z1) in _ca_hand_boxes():
        m = CA_CLEAR_MARGIN
        inside = (P[:, 0] >= x0 - m) & (P[:, 0] <= x1 + m) & (P[:, 1] >= y0 - m) & (P[:, 1] <= y1 + m) & (P[:, 2] >= z0 - m) & (P[:, 2] <= z1 + m)
        k = int(inside.sum()); n += k
        if k and nearest is None:
            nearest = np.round(np.asarray(obst, float)[inside][0], 3).tolist()
    return n == 0, n, nearest


def ca_column_clear(obst, tcp_low, z_high, R):
    """The vertical descent column from z_high to the pre-push pose: the hand at every CA_DESCENT_STEP is clear."""
    zs = np.arange(float(z_high), float(tcp_low[2]) - 1e-9, -CA_DESCENT_STEP).tolist() + [float(tcp_low[2])]
    worst = (True, 0, None)
    for z in zs:
        ok, n, near = ca_clearance(obst, [tcp_low[0], tcp_low[1], z], R)
        if not ok:
            return False, n, near, round(float(z), 3)
    return True, 0, None, None


def ca_frame_raised(depth_img, K, E, floor_z, tcp_xy):
    """All in-box points of the frame raised over the floor, without the robot's own strip (base -> TCP, C_ARM_STRIP)."""
    d = np.asarray(depth_img); d = d[:, :, 0] if d.ndim == 3 else d
    pts = mask_to_world_points(np.ones(d.shape, np.uint8), d, K, E)
    if pts is None or len(pts) == 0:
        return np.empty((0, 3))
    pts = np.asarray(pts, float)
    m = (pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] > float(floor_z) + SUPPORT_RAISED_MIN) & (pts[:, 2] < C_WS[2][1])
    pts = pts[m]
    if tcp_xy is not None:
        pts = pts[_c_seg_dist(pts[:, 0], pts[:, 1], np.zeros(2), np.asarray(tcp_xy, float)[:2]) >= C_ARM_STRIP]
    return pts


def ca_near(pts, ref, r):
    """Boolean mask of the points of pts within r (3D) of some point of ref (empty ref: all False)."""
    if ref is None or len(ref) == 0 or len(pts) == 0:
        return np.zeros(len(pts), bool)
    d, _ = _KDTree(np.asarray(ref, float)).query(np.asarray(pts, float), distance_upper_bound=float(r)); return np.isfinite(d)


def ca_part_handles(part, body, frame_pts, assembly_pts):
    """Frame points in the part's front zone (within CA_HANDLE_MAX in front of its front, inside its lateral extent, not above it) that form
    clusters with no neighbour outside the zone and the assembly: handles of the part. Returns (handle_pts, obstacle_pts, info)."""
    c = body["centre"]; n_out = part["n_out"]; w = np.array([-n_out[1], n_out[0]]); Q = part["pts"]
    s_all = (frame_pts[:, :2] - c) @ n_out - part["face_off"]; t_all = (frame_pts[:, :2] - c) @ w
    sQ = (Q[:, :2] - c) @ n_out - part["face_off"]; tQ = (Q[:, :2] - c) @ w
    s_front = float(np.percentile(sQ, CONTAINER_PCT_HI)); t_lo, t_hi = float(np.percentile(tQ, CONTAINER_PCT_LO)), float(np.percentile(tQ, CONTAINER_PCT_HI))
    in_asm = ca_near(frame_pts, assembly_pts, CA_SAME_POINT)
    zone = (~in_asm) & (s_all > s_front - CA_FRONT_BAND) & (s_all <= s_front + CA_HANDLE_MAX) & (t_all >= t_lo) & (t_all <= t_hi) & (frame_pts[:, 2] <= part["z_hi"] + CA_CLEAR_MARGIN)
    others = frame_pts[(~in_asm) & (~zone)]
    handles = np.zeros(len(frame_pts), bool); info = {"n_zone": int(zone.sum()), "clusters": []}
    if int(zone.sum()) >= CA_PART_MIN_PTS // 4:
        Z = frame_pts[zone]; zi = np.where(zone)[0]
        g = np.floor((Z - Z.min(0)) / CA_CLUSTER_CELL).astype(int); occ = np.zeros(g.max(0) + 1, bool); occ[g[:, 0], g[:, 1], g[:, 2]] = True
        lab, n = _ndi.label(occ, structure=np.ones((3, 3, 3), bool)); L = lab[g[:, 0], g[:, 1], g[:, 2]]
        otree = _KDTree(others[:, :2]) if len(others) else None
        for k in range(1, n + 1):
            sel = L == k
            if int(sel.sum()) < CA_PART_MIN_PTS // 4:
                continue
            cont = 0
            if otree is not None:
                dn, _ = otree.query(Z[sel][:, :2], distance_upper_bound=CA_CELL); cont = int(np.isfinite(dn).sum())
            rec = {"n": int(sel.sum()), "continues_outside": cont, "z": [round(float(Z[sel][:, 2].min()), 3), round(float(Z[sel][:, 2].max()), 3)]}
            if cont == 0:
                handles[zi[sel]] = True; rec["handle"] = True
            info["clusters"].append(rec)
    obstacles = frame_pts[(~in_asm) & (~handles)]
    return frame_pts[handles], obstacles, info


def ca_ik_joints(pos, quat):
    """The public IK solution (anchored at the actual joints) as 7 joints, or None when the interface refuses the pose."""
    try:
        j = np.asarray(solve_ik([float(v) for v in pos], [float(v) for v in quat]), float).reshape(-1)
        return j[:7] if len(j) >= 7 and bool(np.all(np.isfinite(j[:7]))) else None
    except Exception:  # noqa: BLE001
        return None

def ca_route_cost(j_now, j_hi, j_pre):
    """Joint-space length (rad) of the approach + descent route under the public IK solutions anchored at the actual joints:
    |j_hi - j_now| + |j_pre - j_hi|. The interface executes the free move as one joint target and plans the linear descent as a chain
    anchored at the previous solution, so the shortest route is the one whose two solutions lie on the branch nearest the current
    configuration; it orders the candidates, the interface's own plan (ca_execute_push) admits them."""
    a = np.asarray(j_now, float).reshape(-1)[:7]; h = np.asarray(j_hi, float).reshape(-1)[:7]; p = np.asarray(j_pre, float).reshape(-1)[:7]
    return float(np.linalg.norm(h - a) + np.linalg.norm(p - h))

def ca_push_plan(part, body, floor_z, obst_all, obst_body, ik_probe, s_body_front=None, ik_solve=None, j_now=None, form=None, tool=False, asm_pts=None):   # C16I3: tool = commanded frame ao_tool_quat(model) + closing-position ranking; asm_pts = high-front fallback (None: the parent rule only)   # C15I4: form = "box" for a tracked sliding part (None: the footprint test below)
    """Push plan for the selected part (its points include its handles). obst_all: every visible point that is neither the assembly nor a
    handle (the approach must clear it); obst_body: every visible point that is not the part (the hand at the closing position must clear it).
    Every admissible candidate (hand tilt x lateral position x yaw flip; a panel's tilts are CA_PANEL_TILTS_DEG, a box's CA_PUSH_TILTS_DEG)
    is collected; with ik_solve (joints or None) and j_now (the actual joints) the candidates are ranked by ca_route_cost, the best is the
    plan and the others its "alternatives" (ca_execute_push probes them in that order); without a solver the first admissible is the plan.
    Returns (plan, info) or (None, info) when the approach is obstructed or unreachable."""
    c = body["centre"]; n_out = part["n_out"]; Q = part["pts"]; rect = part["rect"]; info = {"form": None}
    panel = (2 * rect["half_minor"] < CA_PANEL_THIN) if form is None else form == "panel"   # C15I4: form override (None = as before)
    if panel:
        d = rect["major"]; ends = [rect["centre"] - d * rect["half_major"], rect["centre"] + d * rect["half_major"]]
        dist = [float(np.linalg.norm(e - c)) for e in ends]; hinge = ends[int(np.argmin(dist))]
        if dist[0] > dist[1]:
            d = -d
        nrm = np.array([-d[1], d[0]])
        if float(nrm @ (c - rect["centre"])) < 0:
            nrm = -nrm
        R_len = 2 * rect["half_major"]; r0 = CA_PANEL_R0_FRAC * R_len
        r = (Q[:, :2] - hinge) @ d; qn = (Q[:, :2] - hinge) @ nrm
        near = np.abs(r - r0) <= 0.03
        if int(near.sum()) < CA_PART_MIN_PTS // 4:
            return None, dict(info, form="panel", why="no_points_at_push_point")
        s_out = float(np.percentile(qn[near], CONTAINER_PCT_LO)); q_lo = float(np.percentile(qn, CONTAINER_PCT_LO))
        thick = max(0.0, float(np.percentile(qn[near], CONTAINER_PCT_HI)) - s_out)   # panel thickness near the push point (the handle at the free end is not counted)
        r_lim = R_len
        if q_lo < s_out - CA_PUSH_GAP:   # something protrudes beyond the face (a handle): the sliding contact must stay short of it
            front = qn <= q_lo + CA_FRONT_BAND; r_lim = min(R_len, float(np.percentile(r[front], CONTAINER_PCT_LO)))
        r_lim = r_lim - CA_HAND["finger_half_wide"] - CA_CLEAR_MARGIN
        beta = np.radians(CA_PANEL_SWEEP_DEG); u = np.cos(beta) * nrm - np.sin(beta) * d
        disc = r_lim ** 2 - (r0 * np.cos(beta)) ** 2
        if disc <= 0:
            return None, dict(info, form="panel", why="no_sweep_travel")
        L = 0.9 * (r0 * np.sin(beta) + np.sqrt(disc))
        tip_xy = hinge + r0 * d + (s_out - CA_PUSH_GAP) * nrm
        z_tip = max(part["z_hi"] - CA_PANEL_TIP_BELOW, float(floor_z) + CA_FLOOR_CLEAR)
        # closing position check: the tip at (panel thickness + gap) before the body face it closes onto
        s_end = thick + CA_PUSH_GAP
        info.update(form="panel", hinge=hinge, d=d, normal=nrm, R=R_len, r0=r0, r_lim=r_lim, s_out=s_out, thickness=thick, beta_deg=CA_PANEL_SWEEP_DEG, theta_pred_deg=float(np.degrees(np.arctan2(L * np.cos(beta), r0 - L * np.sin(beta)))))
        cands_lat = [0.0]
    else:
        u = -np.asarray(n_out, float); w = np.array([-u[1], u[0]])
        s = (Q[:, :2] - c) @ n_out - part["face_off"]; s98 = float(np.percentile(s, CONTAINER_PCT_HI)); front = s >= s98 - CA_FRONT_BAND
        t = (Q[front, :2] - c) @ w; t_lo, t_hi = float(np.percentile(t, 5)), float(np.percentile(t, 95)); t_mid = 0.5 * (t_lo + t_hi)
        z_lo = float(np.percentile(Q[front, 2], CONTAINER_PCT_LO)); z_tip = max(z_lo - CA_TIP_BELOW, float(floor_z) + CA_FLOOR_CLEAR)
        L = s98 + CA_PUSH_GAP + CA_PUSH_EXTRA
        s_end = max(0.0, s98 - float(s_body_front if s_body_front is not None else s98))   # the tip when the part's body is flush with the face (its handle still protrudes)
        info.update(form="box", s_front=s98, s_body_front=s_body_front, t_range=[t_lo, t_hi], z_front=[z_lo, float(np.percentile(Q[front, 2], CONTAINER_PCT_HI))], n_front=int(front.sum()))
        half = t_hi - t_mid - CA_HAND["finger_half_wide"]
        cands_lat = [0.0] + [sgn * k * 0.01 for k in range(1, int(max(0.0, half) / 0.01) + 1) for sgn in (1, -1)]
        tip_xy = c + n_out * (part["face_off"] + s98 + CA_PUSH_GAP) + w * t_mid
    u = np.array([u[0], u[1]]) / np.linalg.norm(u); L = float(min(L, 0.45))
    tilts = CA_PANEL_TILTS_DEG if panel else CA_PUSH_TILTS_DEG   # a box's front rises above the tips: the tips must lead the body corner (28 deg or more); a panel's top is below the body's bottom at every tilt
    ranked = ik_solve is not None and j_now is not None
    tried = []; admissible = []
    for tilt in tilts:
        for lat in cands_lat:
            for flip in (False, True):
                R, q = ca_hand_frame(u, tilt, flip=flip); q = ao_tool_quat(R) if tool else q; off = ca_tip_offset(R, u); wv = np.array([-u[1], u[0]]) * lat   # C16I3: the commanded quaternion (C5I3) when tool
                tip = np.array([tip_xy[0] + wv[0], tip_xy[1] + wv[1], z_tip]); tcp = tip - off
                z_high = body["z_top"] + CA_HIGH_CLEAR - ca_hand_low_z(R)
                z_min_tip = float(floor_z) + CA_FLOOR_CLEAR - ca_hand_low_z(R) + off[2]   # the hand's lowest corner stays CA_FLOOR_CLEAR over the floor
                if tip[2] < z_min_tip:
                    tip[2] = z_min_tip; tcp = tip - off
                rec = {"tilt": tilt, "flip": flip, "lat": round(lat, 3), "z_tip": round(float(tip[2]), 3)}
                ok1, n1, near1 = ca_clearance(obst_all, tcp, R)
                if not ok1:
                    rec.update(why="pre_push_pose_obstructed", n=n1, nearest=near1); tried.append(rec); continue
                ok2, n2, near2, z2 = ca_column_clear(obst_all, tcp, z_high, R)
                if not ok2:
                    rec.update(why="descent_column_obstructed", n=n2, nearest=near2, z=z2); tried.append(rec); continue
                if panel:   # the closed panel lies along the closing face from the hinge end: the hand at the last push's end (tip thickness + gap outside the face at r0 along it)
                    n_c = -np.asarray(n_out, float); t_f = np.array([-n_c[1], n_c[0]]); d_c = t_f * (1.0 if float((c - hinge) @ t_f) >= 0 else -1.0)
                    u_c = np.cos(beta) * n_c - np.sin(beta) * d_c; u_c = u_c / np.linalg.norm(u_c)
                    R_c, _q_c = ca_hand_frame(u_c, tilt, flip=flip); off_c = ca_tip_offset(R_c, u_c)
                    tip_c = np.array([hinge[0] + r0 * d_c[0] - s_end * n_c[0], hinge[1] + r0 * d_c[1] - s_end * n_c[1], z_tip]); tcp_c = tip_c - off_c
                    ok3, n3, near3 = ca_clearance(obst_body, tcp_c, R_c); rec.update(closing_tip=np.round(tip_c, 4).tolist())
                    if not ok3:
                        rec.update(why="closing_pose_obstructed", n=n3, nearest=near3); tried.append(rec); continue
                    travel_end = None
                else:
                    travel_end = float(L - CA_PUSH_EXTRA - CA_PUSH_GAP - s_end)
                    if 0 < travel_end < L:
                        tcp_end = tcp + np.array([u[0], u[1], 0.0]) * travel_end
                        ok3, n3, near3 = ca_clearance(obst_body, tcp_end, R)
                        if not ok3:
                            rec.update(why="closing_pose_obstructed", n=n3, nearest=near3, travel_end=round(travel_end, 3)); tried.append(rec); continue
                    if tool:   # C16I3: the hand AT the closing position (tip at s_end, the parent's check stops CA_PUSH_GAP short); ranks, refuses nothing
                        d_fl = float(L - CA_PUSH_EXTRA - s_end); rec["flush_clear"] = bool(d_fl <= 0 or ca_clearance(obst_body, tcp + np.array([u[0], u[1], 0.0]) * d_fl, R)[0])
                high = [tcp[0], tcp[1], z_high]
                if ranked:
                    j_hi = ik_solve(high, q); j_pre = ik_solve(tcp, q); ik_hi = j_hi is not None; ik_pre = j_pre is not None
                else:
                    j_hi = j_pre = None; ik_hi = ik_probe(high, q); ik_pre = ik_probe(tcp, q)
                rec.update(ik_high=ik_hi, ik_pre=ik_pre)
                if not (ik_hi and ik_pre):
                    rec["why"] = "unreachable"; tried.append(rec); continue
                cost = ca_route_cost(j_now, j_hi, j_pre) if ranked else None
                rec["why"] = "ok"; rec["cost"] = None if cost is None else round(cost, 3); tried.append(rec)
                admissible.append({"u": u, "tip": tip, "tcp_pre": tcp, "z_high": float(z_high), "quat": q, "R": R, "tilt": tilt, "L": L, "flip": flip, "lat": lat, "z_tip": float(z_tip), "travel_end": travel_end, "closing_tip": rec.get("closing_tip"), "cost": cost, "flush_clear": rec.get("flush_clear"), **info})   # C16I3: flush_clear (None outside tool mode)
    if not admissible and not panel and asm_pts is not None:   # C16I3 (C14I3 rule): the parent rule admits no candidate for a box part -> high-front candidates
        hf, hf_info = ca_high_front_plans(part, body, floor_z, obst_all, obst_body, asm_pts, z_tip, ik_probe, ik_solve=ik_solve, j_now=j_now, tool=tool)
        info = dict(info, high_front={kk: vv for kk, vv in hf_info.items() if kk != "tried"})
        if hf:
            return dict(hf[0], tried=tried, alternatives=hf[1:], n_admissible=len(hf), **{kk: vv for kk, vv in info.items() if kk not in hf[0]}), info
    if not admissible:
        return None, dict(info, why="approach_blocked_or_unreachable", tried=tried, z_tip=float(z_tip), L=L, u=u.tolist())
    if ranked:
        admissible.sort(key=lambda a: a["cost"])   # stable: equal costs keep the enumeration order
    if tool:   # C16I3: closing-position-clear candidates first (stable: the cost order within each group)
        admissible.sort(key=lambda a: 0 if a.get("flush_clear") else 1)
    plan = dict(admissible[0], tried=tried, alternatives=admissible[1:], n_admissible=len(admissible))
    return plan, info



# ===== C16I3 manipulation_primitive: blocked box-part close (N16, slot chunk16_iter3) =====
# C14I3's high-front partial push (verbatim rule and constants) with the C5I3 commanded frame when tool=True: tip heights above the
# parent's (OP_Z_STEP steps, up to CA_TIP_BELOW under the part's top); local front = p98 of the part's points at or above the tip;
# pre-push pose and descent column clear of obstacles AND the assembly (CA_CLEAR_MARGIN); travel cut where the closed hand swept
# along the push (every OP_SWEEP_STEP) first meets a non-part point; at least CA_HF_MIN_TRAVEL; the public IK at the high and
# pre-push poses with the COMMANDED quaternion; the lowest admissible height wins, ranked by ca_route_cost.
CA_HF_MIN_TRAVEL = CA_PUSH_GAP + CA_PROGRESS_MIN   # m, 0.04: the shortest partial push (the pre-push gap plus the smallest motion read as progress; C14I3)


def ca_high_front_plans(part, body, floor_z, obst_all, obst_body, asm_pts, z_tip0, ik_probe, ik_solve=None, j_now=None, tool=False):
    """High-front partial-push candidates of a box part (C14I3 rule); tool=True commands ao_tool_quat(model frame). Returns (candidates, info)."""
    c = np.asarray(body["centre"], float); n_out = np.asarray(part["n_out"], float); Q = np.asarray(part["pts"], float); u = -n_out; w = np.array([-u[1], u[0]])
    s = (Q[:, :2] - c) @ n_out - part["face_off"]; u3 = np.array([u[0], u[1], 0.0])
    hard = np.concatenate([np.asarray(obst_all, float).reshape(-1, 3), np.asarray(asm_pts, float).reshape(-1, 3)], 0)
    ranked = ik_solve is not None and j_now is not None; tried = []; out = []
    z_top = float(part["z_hi"]) - CA_TIP_BELOW; n_z = int(np.floor((z_top - float(z_tip0)) / OP_Z_STEP + 1e-9))
    for kz in range(1, n_z + 1):
        z_t = float(z_tip0) + kz * OP_Z_STEP; above = Q[:, 2] >= z_t
        if int(above.sum()) < CA_PART_MIN_PTS // 4:
            tried.append({"z_tip": round(z_t, 3), "why": "no_front_points"}); continue
        s_loc = float(np.percentile(s[above], CONTAINER_PCT_HI)); front = above & (s >= s_loc - CA_FRONT_BAND)
        t = (Q[front, :2] - c) @ w; t_lo, t_hi = float(np.percentile(t, 5)), float(np.percentile(t, 95)); t_mid = 0.5 * (t_lo + t_hi)
        half = t_hi - t_mid - CA_HAND["finger_half_wide"]
        lats = [0.0] + [sgn * k * 0.01 for k in range(1, int(max(0.0, half) / 0.01) + 1) for sgn in (1, -1)]
        tip_xy = c + n_out * (part["face_off"] + s_loc + CA_PUSH_GAP) + w * t_mid
        L_full = float(min(s_loc + CA_PUSH_GAP + CA_PUSH_EXTRA, 0.45))
        for tilt in CA_PUSH_TILTS_DEG:
            for lat in lats:
                for flip in (False, True):
                    R, q = ca_hand_frame(u, tilt, flip=flip); q = ao_tool_quat(R) if tool else q; off = ca_tip_offset(R, u); wv = w * lat
                    tip = np.array([tip_xy[0] + wv[0], tip_xy[1] + wv[1], z_t]); tcp = tip - off
                    rec = {"z_tip": round(z_t, 3), "s_front": round(s_loc, 4), "tilt": tilt, "flip": flip, "lat": round(lat, 3)}
                    if tcp[2] + ca_hand_low_z(R) < float(floor_z) + CA_FLOOR_CLEAR:
                        rec["why"] = "hand_below_floor_clearance"; tried.append(rec); continue
                    z_high = body["z_top"] + CA_HIGH_CLEAR - ca_hand_low_z(R)
                    ok1, n1, near1 = ca_clearance(hard, tcp, R)
                    if not ok1:
                        rec.update(why="pre_push_pose_obstructed", n=n1, nearest=near1); tried.append(rec); continue
                    ok2, n2, near2, z2 = ca_column_clear(hard, tcp, z_high, R)
                    if not ok2:
                        rec.update(why="descent_column_obstructed", n=n2, nearest=near2, z=z2); tried.append(rec); continue
                    n_st = max(1, int(np.ceil(L_full / OP_SWEEP_STEP))); L = L_full
                    for i in range(1, n_st + 1):
                        sv = L_full * i / n_st
                        if not ca_clearance(obst_body, tcp + u3 * sv, R)[0]:
                            L = L_full * (i - 1) / n_st; break
                    partial = L < L_full - 1e-9; rec.update(L=round(L, 4), partial=partial)
                    if L < CA_HF_MIN_TRAVEL:
                        rec["why"] = "travel_obstructed"; tried.append(rec); continue
                    high = [tcp[0], tcp[1], z_high]
                    if ranked:
                        j_hi = ik_solve(high, q); j_pre = ik_solve(tcp, q); ik_hi = j_hi is not None; ik_pre = j_pre is not None
                    else:
                        j_hi = j_pre = None; ik_hi = ik_probe(high, q); ik_pre = ik_probe(tcp, q)
                    rec.update(ik_high=ik_hi, ik_pre=ik_pre)
                    if not (ik_hi and ik_pre):
                        rec["why"] = "unreachable"; tried.append(rec); continue
                    cost = ca_route_cost(j_now, j_hi, j_pre) if ranked else None
                    rec["why"] = "ok"; rec["cost"] = None if cost is None else round(cost, 3); tried.append(rec)
                    out.append({"u": u, "tip": tip, "tcp_pre": tcp, "z_high": float(z_high), "quat": q, "R": R, "tilt": tilt, "L": float(L), "flip": flip, "lat": lat, "z_tip": z_t,
                                "travel_end": None, "closing_tip": None, "cost": cost, "form": "box", "s_front": s_loc, "partial": bool(partial), "rule": "high_front", "tool_frame": bool(tool)})
        if out:
            break
    if ranked:
        out.sort(key=lambda a: a["cost"])
    return out, {"z_tip0": round(float(z_tip0), 4), "z_top": round(z_top, 4), "n_admissible": len(out), "tried": tried}
# ===== end C16I3 helper =====


def ca_measure(rgb, depth_img, K, E, prompts, side, floor_z, tcp_xy, ik_probe, k=0, ik_solve=None, j_now=None, carry=False):   # C15I4: carry = the part's contents travel with it (the place-and-close primitive; False = as before)
    """One measurement of the assembly from the current frame: body, parts, the part named by the side code (with its handles), the
    obstacle sets and the push plan (ranked by ca_route_cost when ik_solve and the actual joints j_now are given). Returns a record
    (printed as CLOSE_MEASURE by the caller); rec["plan"] is None when no push is possible."""
    rec = {"k": k, "side": side, "assembly": None, "body": None, "parts": [], "selected": None, "handles": None, "protrusion": None, "plan": None, "plan_info": None, "why": None}
    pts, ainfo = ca_assembly(rgb, depth_img, K, E, prompts, floor_z); rec["assembly"] = ainfo
    _CA_SEEN.clear(); _CA_SEEN["assembly"] = pts   # C15I4: geometry record for the place-and-close tracker (no output change)
    if pts is None:
        rec["why"] = "assembly_not_found"; return rec
    body = ca_body(pts)
    if body is None:
        rec["why"] = "no_body_top"; return rec
    rec["body"] = {"centre": np.round(body["centre"], 4).tolist(), "axis_deg": round(float(np.degrees(np.arctan2(body["axes"][0][1], body["axes"][0][0])) % 180.0), 1), "half": [round(float(h), 4) for h in body["half"]],
                   "z_top": round(float(body["z_top"]), 4), "n_top": body["n_top"], "n_cells": body["n_cells"], "n_regions": body["n_regions"]}
    comps = ca_parts(pts, body)
    rec["parts"] = [{"n": q["n"], "centroid": np.round(q["centroid"], 4).tolist(), "n_out": np.round(q["n_out"], 2).tolist(), "protrusion": round(float(q["protrusion"]), 4),
                     "extent": [round(2 * float(q["rect"]["half_major"]), 4), round(2 * float(q["rect"]["half_minor"]), 4)], "z": [round(q["z_lo"], 4), round(q["z_hi"], 4)]} for q in comps]
    part = ca_select_part(comps, side)
    if part is None:
        rec["why"] = "no_part" if comps else "no_protruding_part"; return rec
    F = ca_frame_raised(depth_img, K, E, floor_z, tcp_xy)
    hpts, obst_all, hinfo = ca_part_handles(part, body, F, pts); rec["handles"] = hinfo
    s_body_front = float(part["protrusion"])
    if len(hpts):
        Q = np.concatenate([part["pts"], hpts], 0); s = (Q[:, :2] - body["centre"]) @ part["n_out"] - part["face_off"]
        part = dict(part, pts=Q, n=int(len(Q)), protrusion=float(np.percentile(s, CONTAINER_PCT_HI)))
    obst_body = F[~ca_near(F, part["pts"], CA_SAME_POINT)]
    if carry:   # C15I4: what lies inside the drawn-out part's footprint and not above it slides in with it (not an obstacle at the closing position)
        cm = ca_carried(obst_body, body["centre"], part["n_out"], part["face_off"], part["pts"], s_body_front, part["z_hi"]); obst_body = obst_body[~cm]; rec["n_carried"] = int(cm.sum())
    rec["selected"] = {"n": part["n"], "centroid": np.round(part["centroid"], 4).tolist(), "n_out": np.round(part["n_out"], 2).tolist(), "face_off": round(part["face_off"], 4), "protrusion": round(float(part["protrusion"]), 4),
                       "body_front": round(s_body_front, 4), "minor": round(2 * float(part["rect"]["half_minor"]), 4), "major": round(2 * float(part["rect"]["half_major"]), 4), "n_handle_pts": int(len(hpts)), "n_obstacle_pts": int(len(obst_all))}
    rec["protrusion"] = float(part["protrusion"])
    _CA_SEEN.update(body=body, part=part, handles=hpts, s_body_front=s_body_front, obst_all=obst_all, obst_body=obst_body)   # C15I4: the selected part (handles merged) and its handle points   # C16I3: + the obstacle sets
    plan, pinfo = ca_push_plan(part, body, floor_z, obst_all, obst_body, ik_probe, s_body_front=s_body_front, ik_solve=ik_solve, j_now=j_now)
    rec["plan_info"] = {kk: (np.round(v, 4).tolist() if hasattr(v, "tolist") else v) for kk, v in pinfo.items()}; rec["plan"] = plan
    if plan is None:
        rec["why"] = pinfo.get("why")
    return rec



def ca_plan_summary(plan):
    return {"form": plan["form"], "tilt": plan["tilt"], "flip": plan["flip"], "lat": plan["lat"], "u": np.round(plan["u"], 3).tolist(), "tip": np.round(plan["tip"], 4).tolist(), "tcp_pre": np.round(plan["tcp_pre"], 4).tolist(),
            "z_high": round(plan["z_high"], 4), "L": round(plan["L"], 4), "travel_end": None if plan.get("travel_end") is None else round(float(plan["travel_end"]), 4), "closing_tip": plan.get("closing_tip"), "quat": np.round(plan["quat"], 4).tolist(),
            "theta_pred_deg": None if plan.get("theta_pred_deg") is None else round(float(plan["theta_pred_deg"]), 1), "n_tried": len(plan["tried"]),
            "cost": None if plan.get("cost") is None else round(float(plan["cost"]), 3), "n_admissible": plan.get("n_admissible"),
            "alternatives": [{"tilt": a["tilt"], "flip": a["flip"], "lat": a["lat"], "cost": None if a.get("cost") is None else round(float(a["cost"]), 3)} for a in (plan.get("alternatives") or [])]}



def ca_approach_budget(high, q, label):
    """Budget of the free move to a high pose from the interface's own plan (move_to_pose plan_only: the anchored IK target from the
    actual joints and its step estimate; no motion), by the existing H2 rule (1.25 x the planned steps, at least BUDGET_FREE, at most
    H_APPROACH_CAP). Returns (budget, record); budget None = the plan is refused or exceeds the cap (the candidate is skipped)."""
    r = move_to_pose(np.asarray(high, float), np.asarray(q, float), mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=label)
    steps = r.get("steps_planned") if r.get("detail") == "plan_only" else None
    rec = {"status": r.get("status"), "detail": r.get("detail"), "steps_planned": None if steps is None else int(steps), "jump": None if r.get("max_waypoint_step") is None else round(float(r["max_waypoint_step"]), 3)}
    if steps is None or int(steps) > H_APPROACH_CAP:
        return None, rec
    return int(min(H_APPROACH_CAP, max(BUDGET_FREE, int(np.ceil(1.25 * int(steps)))))), rec

def ca_probe_linear(target, q, corridor, budget, label):
    """The interface's own plan of a linear move from the actual joints (move_to_pose plan_only; no motion): (ok, record). ok = the chained
    IK plan exists and is continuous (the interface refuses ik_failed / discontinuous_path before any motion) and fits the budget."""
    r = move_to_pose(np.asarray(target, float), np.asarray(q, float), mode="linear", tol_pos=PRIM["reach_tol"], corridor=corridor, step_budget=budget, plan_only=True, label=label)
    ok = r.get("detail") == "plan_only" and not bool(r.get("plan_exceeds_budget"))
    return ok, {"status": r.get("status"), "detail": r.get("detail"), "steps_planned": r.get("steps_planned"), "max_step": None if r.get("max_waypoint_step") is None else round(float(r["max_waypoint_step"]), 3), "n_waypoints": r.get("n_waypoints")}

def ca_execute_push(plan, sg, k):
    """One push cycle through the existing contracts. The candidates (the ranked plan and its alternatives, at most CA_PROBE_MAX) are
    admitted one by one by the interface's own planner from the actual joints: the free move to the candidate's high pose gets its budget
    from the plan (H2 rule), the linear descent and the push line are probed (plan_only: a refused chain costs no motion) and a refused
    candidate is skipped for the next one, whose approach starts where the hand is. Then: linear descent, contact push, linear back-off and
    ascent, home. Returns a record with the stage reached, the push result summary, the probes and the control steps used."""
    rec = {"k": k, "stage": "above", "steps": 0, "push": None, "probes": [], "candidate": None}
    cands = ([plan] + list(plan.get("alternatives") or []))[:CA_PROBE_MAX]

    def used(r):
        return int((r or {}).get("steps_used") or 0) if isinstance(r, dict) else 0

    def home():
        h = goto_home_joint_position(); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(h); rec["home"] = h.get("status") if isinstance(h, dict) else "unreported"

    def ascend(q, z):
        p, _, _ = read_tcp(); r = move_to_pose(np.array([p[0], p[1], float(z)]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:push{k}_ascend"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r); rec["ascend"] = r.get("status"); return r

    for i, c in enumerate(cands):
        q = np.asarray(c["quat"], float); u3 = np.array([c["u"][0], c["u"][1], 0.0]); pre = np.asarray(c["tcp_pre"], float); high = np.array([pre[0], pre[1], float(c["z_high"])]); L = float(c["L"])
        pr = {"i": i, "tilt": c["tilt"], "flip": c["flip"], "lat": c["lat"], "cost": None if c.get("cost") is None else round(float(c["cost"]), 3)}
        budget, ap = ca_approach_budget(high, q, f"{sg}:push{k}_above_plan{i}"); pr["above_plan"] = dict(ap, budget=budget)
        if budget is None:
            pr["why"] = "approach_plan_refused"; rec["probes"].append(pr); continue
        rec["stage"] = "above"
        r0 = move_to_pose(high, q, mode="free", tol_pos=0.015, step_budget=budget, label=f"{sg}:push{k}_above"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r0); rec["above"] = r0.get("status")
        if r0.get("status") != "reached":
            pr["why"] = f"above_{r0.get('status')}"; rec["probes"].append(pr); home(); return rec
        ok1, dp = ca_probe_linear(pre, q, SEGMENT_CORRIDOR, BUDGET_LINEAR, f"{sg}:push{k}_descend_plan{i}"); pr["descend_plan"] = dp
        if not ok1:
            pr["why"] = "descent_plan_refused"; rec["probes"].append(pr); continue   # no motion: the next candidate's approach starts from this high pose
        rec["stage"] = "descend"
        r1 = move_to_pose(pre, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:push{k}_descend"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r1); rec["descend"] = r1.get("status")
        if r1.get("status") != "reached":
            pr["why"] = f"descend_{r1.get('status')}"; rec["probes"].append(pr); ascend(q, c["z_high"]); home(); return rec
        ok2, pp = ca_probe_linear(pre + u3 * L, q, PUSH_CORRIDOR, BUDGET_PUSH, f"{sg}:push{k}_push_plan{i}"); pr["push_plan"] = pp
        if not ok2:
            pr["why"] = "push_plan_refused"; rec["probes"].append(pr); r = ascend(q, c["z_high"])
            if r.get("status") != "reached":
                home()
            continue
        pr["why"] = "ok"; rec["probes"].append(pr); rec["candidate"] = {"i": i, "tilt": c["tilt"], "flip": c["flip"], "lat": c["lat"], "budget": budget}
        rec["stage"] = "push"
        _mark(ev="push_start", sg=sg, k=k, tcp=pre, u=u3, L=L)
        r2 = contact_move(pre + u3 * L, q, approach_axis=u3.tolist(), contact_tol_before=L, overshoot_after=0.0, start_pose=pre, start_tol=0.02, gripper_expect=None,
                          lateral_corridor=PUSH_CORRIDOR, step_budget=BUDGET_PUSH, label=f"{sg}:push{k}")
        _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r2)
        print("CONTACT_MOVE " + json.dumps({"label": f"{sg}:push{k}", "status": r2["status"], "detail": r2.get("detail"), "success": r2["success"], "steps": r2["steps_used"], "budget": r2["budget"], "wall_s": r2.get("wall_s"),
                                            "along_cm": round((r2.get("along_final") or 0) * 100, 2), "travel_cm": round((r2.get("travel") or 0) * 100, 2), "window_cm": [round(x * 100, 2) for x in (r2.get("window") or [])],
                                            "pos_err_cm": round(r2.get("pos_err", 9) * 100, 2), "evidence": r2.get("evidence"), "verifier": verify_contact(r2).get("status")}), flush=True)
        rec["push"] = {"status": r2["status"], "detail": r2.get("detail"), "along": None if r2.get("along_final") is None else float(r2["along_final"]), "travel": L, "steps": r2["steps_used"], **({"partial": True} if c.get("partial") else {})}   # C16I3: partial flag
        p_end = np.asarray(r2.get("final_tcp"), float) if r2.get("final_tcp") is not None else read_tcp()[0]
        _mark(ev="push_end", sg=sg, k=k, status=r2["status"], detail=r2.get("detail"), tcp=p_end, along=rec["push"]["along"])
        back = p_end - u3 * CA_BACKOFF
        r3 = move_to_pose(back, q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:push{k}_backoff"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r3); rec["backoff"] = r3.get("status")
        p_now = np.asarray(r3.get("final_tcp"), float) if r3.get("final_tcp") is not None else back
        r4 = move_to_pose(np.array([p_now[0], p_now[1], float(c["z_high"])]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:push{k}_ascend"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r4); rec["ascend"] = r4.get("status")
        home(); rec["stage"] = "done"
        return rec
    rec["stage"] = "probe"; rec["probe"] = "no_admissible_candidate"; home()
    return rec



def ca_verdict(last, prot_before, prot_now):
    """Verdict after a re-observation: closed (the push ended at a confirmed stop and the protrusion did not grow), push (no stop, the
    protrusion fell by CA_PROGRESS_MIN), or a stop reason. Returns (action, why)."""
    if last is None:
        return "push", "first_cycle"
    if prot_now is None or prot_before is None:
        return "stop", "part_not_re_observed"
    drop = float(prot_before) - float(prot_now); st = (last.get("push") or {}).get("status")
    if (last.get("push") or {}).get("partial"):   # C16I3 (C14I3): a partial push (travel cut before the closing position) is never a closing stop
        return ("push", "partial_push_progress") if drop >= CA_PROGRESS_MIN else ("stop", "no_progress")
    if st == "contact_established":
        return ("closed", "stop_confirmed") if drop >= -CA_PROGRESS_MIN else ("stop", "stop_but_protrusion_grew")
    if drop >= CA_PROGRESS_MIN:
        return "push", "progress_without_stop"
    return "stop", "no_progress"


# ===== C15I4 manipulation_primitive / task_driver: place-and-close (N15 group B, slot chunk15_iter4) =====
# (1) Re-finding (C14I1's rule). A clause "... and close it" names its part by reference only; the part phrase's masks cover the
#     body and not the drawn-out part (no component outside the body outline: part_not_found before motion). The referent IS the
#     previous clause's: run() stores its assembly vocabulary (la_assembly_prompts, C6I3) on the anaphoric subgoal as
#     referent_prompts and the prompt list is the subgoal's own list followed by it. A subgoal that spells its reference out is
#     unchanged.
# (2) Tracking (anaphoric part relations only: the close that follows a placement into the same part). The first measurement
#     fixes the part's window on the static body face: lateral range (CONTAINER_PCT_LO..HI of the part, +- CA_OUTLINE_MARGIN),
#     height range (the part's z_lo..z_hi +- CA_CLEAR_MARGIN) and the box of its handle points (+- CA_CELL). After a push, when
#     the segmentation shows no protruding component, the window is read from the raw public depth frame (ca_frame_raised):
#     points beyond the body outline margin (s > CA_OUTLINE_MARGIN, the body's own face lies within it) up to the previous
#     protrusion + CA_FRONT_BAND. Front wall = those points outside the handle box. Fewer than CA_PART_MIN_PTS // 4 front points
#     (the stack's smallest cluster) = flush -> closed (flush_confirmed). Otherwise the part is still out: protrusion = p98 of
#     the window points; it moved by CA_PROGRESS_MIN -> push again, planned on the tracked points as a box; no progress -> stop.
_CA_SEEN = {}   # the last ca_measure's geometry: assembly points, body, selected part (handles merged), handle points


def ca_relation_prompts(sg, td):
    """Prompt list of the assembly a part relation measures: the parent's rule, then the referent vocabulary of an anaphoric subgoal."""
    sup = la_phrase_in(td, "supports", sg.get("clause") or "")
    prompts = list(sg.get("part_prompts") or []) + (list((td.get("supports") or {}).get(sup) or []) if sup else []) + list(sg.get("support_prompts") or [])
    if sg.get("refers"):
        prompts += list(sg.get("referent_prompts") or [])
    return list(dict.fromkeys(prompts))


def ca_track_window(seen):
    """Window of the selected part on the static body face from the first measurement's geometry; None without a selected part."""
    part = seen.get("part"); body = seen.get("body")
    if part is None or body is None:
        return None
    c = np.asarray(body["centre"], float)[:2]; n = np.asarray(part["n_out"], float)[:2]; w = np.array([-n[1], n[0]])
    Q = np.asarray(part["pts"], float); t = (Q[:, :2] - c) @ w
    win = {"c": c, "n_out": n, "w": w, "face_off": float(part["face_off"]), "body": body,
           "t": (float(np.percentile(t, CONTAINER_PCT_LO)) - CA_OUTLINE_MARGIN, float(np.percentile(t, CONTAINER_PCT_HI)) + CA_OUTLINE_MARGIN),
           "z": (float(part["z_lo"]) - CA_CLEAR_MARGIN, float(part["z_hi"]) + CA_CLEAR_MARGIN), "handle": None}
    H = seen.get("handles")
    if H is not None and len(H) >= CA_PART_MIN_PTS // 4:
        H = np.asarray(H, float); th = (H[:, :2] - c) @ w
        win["handle"] = {"t": (float(th.min()) - CA_CELL, float(th.max()) + CA_CELL), "z": (float(H[:, 2].min()) - CA_CELL, float(H[:, 2].max()) + CA_CELL)}
    return win


def ca_tracked(F, win, s_prev):
    """The tracked part in the raw frame points F: window mask, flush flag, protrusion (p98 of the window, handles included) and the
    front wall's protrusion (handle box excluded). None when the frame has no point at all (nothing can be confirmed)."""
    if F is None or len(F) == 0 or win is None or s_prev is None:
        return None
    F = np.asarray(F, float); rel = F[:, :2] - win["c"]; s = rel @ win["n_out"] - win["face_off"]; t = rel @ win["w"]; z = F[:, 2]
    m = (t >= win["t"][0]) & (t <= win["t"][1]) & (z >= win["z"][0]) & (z <= win["z"][1]) & (s > CA_OUTLINE_MARGIN) & (s <= float(s_prev) + CA_FRONT_BAND)
    hm = np.zeros(len(F), bool)
    if win["handle"] is not None:
        h = win["handle"]; hm = m & (t >= h["t"][0]) & (t <= h["t"][1]) & (z >= h["z"][0]) & (z <= h["z"][1])
    front = m & ~hm; n_min = CA_PART_MIN_PTS // 4
    flush = int(front.sum()) < n_min
    prot = float(np.percentile(s[m], CONTAINER_PCT_HI)) if int(m.sum()) >= n_min else 0.0
    return {"mask": m, "flush": flush, "protrusion": prot, "front": (0.0 if flush else float(np.percentile(s[front], CONTAINER_PCT_HI))),
            "n_window": int(m.sum()), "n_front": int(front.sum()), "n_handle": int(hm.sum())}


def ca_carried(P, c, n_out, face_off, part_pts, s_front, z_top):
    """Mask of the points P carried by a sliding part: inside its horizontal footprint (beyond the body outline margin, short of the
    part's front s_front, within the lateral range CONTAINER_PCT_LO..HI of the part's points) and not above its top z_top."""
    P = np.asarray(P, float)
    if len(P) == 0:
        return np.zeros(0, bool)
    c = np.asarray(c, float)[:2]; n = np.asarray(n_out, float)[:2]; w = np.array([-n[1], n[0]]); tq = (np.asarray(part_pts, float)[:, :2] - c) @ w
    rel = P[:, :2] - c; s = rel @ n - float(face_off); t = rel @ w
    return (s > CA_OUTLINE_MARGIN) & (s < float(s_front)) & (t >= float(np.percentile(tq, CONTAINER_PCT_LO))) & (t <= float(np.percentile(tq, CONTAINER_PCT_HI))) & (P[:, 2] <= float(z_top))


def ca_tracked_verdict(tr, prot_prev):
    """Verdict on a tracked measurement: closed (flush), push (the part moved by CA_PROGRESS_MIN and is still out) or stop."""
    if tr["flush"]:
        return "closed", "flush_confirmed"
    if prot_prev is not None and float(prot_prev) - float(tr["protrusion"]) >= CA_PROGRESS_MIN:
        return "push", "tracked_progress"
    return "stop", "tracked_no_progress"


def ca_tracked_plan(F, win, tr, asm_pts, floor_z, ik_probe, ik_solve=None, j_now=None, tool=False):   # C16I3: tool = the blocked-part mode (commanded frame, closing-position ranking, high-front fallback)
    """Push plan on the tracked points as a box part of the static body (the parent's planner and checks; obstacles = every point
    that is not the tracked part, and for the approach also not the assembly). Returns (plan, info)."""
    F = np.asarray(F, float); m = tr["mask"]; Q = F[m]
    if len(Q) < CA_PART_MIN_PTS // 4:
        return None, {"why": "tracked_part_too_small", "n": int(len(Q))}
    cc, pa, pb, phu, phv = _ca_min_rect(Q[:, :2])
    part = {"pts": Q, "n": int(len(Q)), "centroid": Q.mean(0), "n_out": win["n_out"], "face_off": win["face_off"], "excess": 0.0, "protrusion": float(tr["protrusion"]),
            "rect": {"centre": cc, "major": pa, "minor": pb, "half_major": phu, "half_minor": phv},
            "z_lo": float(np.percentile(Q[:, 2], CONTAINER_PCT_LO)), "z_hi": float(np.percentile(Q[:, 2], CONTAINER_PCT_HI))}
    near_asm = ca_near(F, asm_pts, CA_SAME_POINT) if asm_pts is not None and len(asm_pts) else np.zeros(len(F), bool)
    carried = ca_carried(F, win["c"], win["n_out"], win["face_off"], Q, float(tr["front"]), part["z_hi"]) & ~m
    ob_body = F[(~m) & (~carried)]
    if tool:   # C16I3 (C14I5's rule): inside the part's window, the static face band up to the part's front is where the part lies when flush -> not an obstacle of the closing position
        rel = F[:, :2] - win["c"]; s_f = rel @ win["n_out"] - win["face_off"]; t_f = rel @ win["w"]
        own = (t_f >= win["t"][0]) & (t_f <= win["t"][1]) & (F[:, 2] >= win["z"][0]) & (F[:, 2] <= win["z"][1]) & (s_f > -CA_OUTLINE_MARGIN) & (s_f <= float(tr["protrusion"]) + CA_FRONT_BAND)
        ob_body = F[(~m) & (~carried) & (~own)]
    return ca_push_plan(part, win["body"], floor_z, F[(~m) & (~near_asm)], ob_body, ik_probe, s_body_front=float(tr["front"]), ik_solve=ik_solve, j_now=j_now, form="box",
                        tool=tool, asm_pts=((np.empty((0, 3)) if asm_pts is None else np.asarray(asm_pts, float)) if tool else None))   # C16I3
# ===== end C15I4 helpers =====


def run_close_articulated(sg, td, sg_label="sg1"):
    """Driver of the close_articulated relation: measure -> push cycle -> re-observe, bounded. Returns True when the part is claimed closed."""
    prompts = ca_relation_prompts(sg, td); side = sg.get("side")   # C15I4: + the referent vocabulary of an anaphoric subgoal
    track_on = bool(sg.get("refers")); win = None; tracked = None   # C15I4: the place-and-close primitive tracks the part it pushes
    blocked = False   # C16I3: the box part's parent rule admitted nothing at k = 0 -> commanded frame, high-front fallback, tracking
    _E1_LAST.update({"stop": False, "grasp_after": None, "hold": None})
    close_gripper(); gate_record("close", role="tool_config"); _PRIM_STATS["motion_calls"] += 1
    last = None; prot_prev = None; steps = 0; k = 0
    while True:
        obs = get_observation(); fr = mf_frame(obs); p_tcp, _ = tcp_from_obs(obs)
        floor_z = support_floor_z(fr["depth"], fr["K"], fr["E"]); _CG["floor_z"] = floor_z
        if floor_z is None:
            print("BINDING_STOP " + json.dumps({"status": "no_floor_estimate", "relation": sg["relation"]}), flush=True); return None
        rec = ca_measure(fr["rgb"], fr["depth"], fr["K"], fr["E"], prompts, side, floor_z, np.asarray(p_tcp, float)[:2], mf_ik_probe, k=k, ik_solve=ca_ik_joints, j_now=np.asarray(obs["robot_joint_pos"], float).reshape(-1)[:7], carry=track_on)   # C2I3: the public solutions + the actual joints rank the candidates   # C15I4: carry
        print("CLOSE_MEASURE " + _mfj({kk: v for kk, v in rec.items() if kk != "plan"}), flush=True)
        prot = rec["protrusion"]
        if k == 0 and prot is None:
            print("BINDING_STOP " + json.dumps({"status": "part_not_found", "relation": sg["relation"], "side": side, "why": rec["why"], "prompts": prompts}), flush=True); return None
        if track_on and k == 0:
            win = ca_track_window(_CA_SEEN)
        if k == 0 and rec["plan"] is None and rec["why"] == "approach_blocked_or_unreachable" and (rec.get("plan_info") or {}).get("form") == "box" and _CA_SEEN.get("part") is not None:   # C16I3: blocked box part
            blocked = True; win = ca_track_window(_CA_SEEN)
            plan_b, info_b = ca_push_plan(_CA_SEEN["part"], _CA_SEEN["body"], floor_z, _CA_SEEN["obst_all"], _CA_SEEN["obst_body"], mf_ik_probe, s_body_front=_CA_SEEN["s_body_front"], ik_solve=ca_ik_joints,
                                          j_now=np.asarray(obs["robot_joint_pos"], float).reshape(-1)[:7], tool=True, asm_pts=_CA_SEEN["assembly"])
            rec = dict(rec, plan=plan_b, why=(info_b.get("why") if plan_b is None else None), plan_info={kk: (np.round(v, 4).tolist() if hasattr(v, "tolist") else v) for kk, v in info_b.items()})
            print("CLOSE_BLOCKED " + _mfj({"sg": sg_label, "k": k, "why": rec["why"], "high_front": info_b.get("high_front"),
                                            "plan": None if plan_b is None else {"rule": plan_b.get("rule") or "parent", "partial": bool(plan_b.get("partial")), "flush_clear": plan_b.get("flush_clear")}}), flush=True)
        tracked = None; F_tr = None
        if blocked and last is not None:   # C16I3: a blocked part is read in its tracked window after every push
            F_tr = ca_frame_raised(fr["depth"], fr["K"], fr["E"], floor_z, np.asarray(p_tcp, float)[:2]); tracked = ca_tracked(F_tr, win, prot_prev)
            if tracked is None:
                print("CLOSE_STOP " + json.dumps({"sg": sg_label, "k": k, "why": "part_not_tracked", "steps_used": steps}), flush=True); return False
            print("CLOSE_TRACK " + _mfj({"sg": sg_label, "k": k, **{kk: v for kk, v in tracked.items() if kk != "mask"}, "protrusion_before": prot_prev}), flush=True)
        elif track_on and win is not None and last is not None and prot is None:   # C15I4: the part dropped out of the segmentation after a push
            F_tr = ca_frame_raised(fr["depth"], fr["K"], fr["E"], floor_z, np.asarray(p_tcp, float)[:2]); tracked = ca_tracked(F_tr, win, prot_prev)
            if tracked is not None:
                print("CLOSE_TRACK " + _mfj({"sg": sg_label, "k": k, **{kk: v for kk, v in tracked.items() if kk != "mask"}, "protrusion_before": prot_prev}), flush=True)
        if tracked is not None:
            action, why = ca_tracked_verdict(tracked, prot_prev); prot = tracked["protrusion"]
        else:
            action, why = ca_verdict(last, prot_prev, prot)
        if last is not None:
            print("CLOSE_VERIFY " + _mfj({"sg": sg_label, "k": k, "claim": "closed" if action == "closed" else "not_closed", "why": why, "protrusion_before": prot_prev, "protrusion_after": prot,
                                           "push": (last or {}).get("push"), "steps_used": steps}), flush=True)
            _mark(ev="close_verify", sg=sg_label, k=k, claim="closed" if action == "closed" else "not_closed", why=why)
        if action == "closed":
            return True
        if action == "stop":
            return False
        if k >= CA_CYCLES_MAX or steps > CA_STEPS_GUARD:
            print("CLOSE_STOP " + json.dumps({"sg": sg_label, "k": k, "why": "cycles_exhausted" if k >= CA_CYCLES_MAX else "step_guard", "steps_used": steps}), flush=True); return False
        plan = rec["plan"]
        if tracked is not None:   # C15I4: the still-out tracked part is planned as the box it is
            plan, tinfo = ca_tracked_plan(F_tr, win, tracked, _CA_SEEN.get("assembly"), floor_z, mf_ik_probe, ik_solve=ca_ik_joints, j_now=np.asarray(obs["robot_joint_pos"], float).reshape(-1)[:7], tool=blocked)   # C16I3: tool
            rec = dict(rec, plan_info={kk: (np.round(v, 4).tolist() if hasattr(v, "tolist") else v) for kk, v in tinfo.items()}, why=(tinfo.get("why") if plan is None else None))
        if plan is None:
            print("CLOSE_STOP " + _mfj({"sg": sg_label, "k": k, "why": rec["why"], "plan_info": rec["plan_info"]}), flush=True); return False
        print("CLOSE_PLAN " + _mfj({"sg": sg_label, "k": k, **ca_plan_summary(plan)}), flush=True)
        _mark(ev="close_plan", sg=sg_label, k=k, form=plan["form"], tip=plan["tip"], u=plan["u"], L=plan["L"])
        res = ca_execute_push(plan, sg_label, k); steps += int(res["steps"])
        print("CLOSE_PUSH " + _mfj({"sg": sg_label, **res}), flush=True)
        if res["stage"] != "done":
            print("CLOSE_STOP " + json.dumps({"sg": sg_label, "k": k, "why": f"motion_{res['stage']}_{res.get(res['stage'])}", "steps_used": steps}), flush=True); return False
        last = res; prot_prev = prot; k += 1
        for _ in range(3):
            get_observation()


LA_PART_RELATIONS = ("close_articulated", "actuate_on", "open_articulated", "actuate_off")   # C11I2: actuate_off (run_rotary) is a part relation too   # C5I1: open_articulated (run_open_articulated) is a part relation too   # relations run by run_close_articulated / run_actuate: a part (or a reference) and no object
# ===== C2I2 manipulation_primitive: actuate_on (round 20260924, chunk 2, iteration 2) =====
# Turning a lever-type part of a support (a switch): the lever is taken between the open pads and rotated about the part's axis.
# Public data only: the segmenter's masks of the adapter's part / support prompts, the depth frame, the public IK probe, the
# interface's motion contracts. The part is measured with the C2I1 rules (assembly, body outline, protruding components); its
# lever is the part's top band, its hub the base points around it (axis = circle fit, else the base rectangle centre). The wrist
# transmits the torque through the pads' normal forces on the lever's faces, so the turn is a grasp-and-rotate and not a fingertip
# push. Constants: hand geometry (robot), the interface's waypoint continuity limit, the stack's xy sigma; DIAGNOSIS.md section 4.
AC_TURN_STEP_DEG = 15.0       # deg, one rotation increment per linear move: the interface refuses a single-waypoint move whose joint step exceeds 1.5 x MAX_WAYPOINT_STEP = 0.45 rad; 15 deg = 0.26 rad of wrist rotation (30 deg = 0.52 rad was refused 10/10 in the probe)
AC_TURN_DEG = 90.0            # deg, a quarter turn per grasp cycle (6 increments); the part's stop is found by the re-observation loop, not by the turn size
AC_TURN_JOINT_MAX = 0.45      # rad, the interface's continuity limit, applied by the program to consecutive arc solutions of the public IK probe
AC_ROT_MIN_DEG = 20.0         # deg, the smallest lever rotation read as motion: atan(2 x xy sigma / (2 x finger_half_wide)) = 21 deg (the shortest lever the pads can hold)
AC_OPEN_HALF_WIDE = 0.04 + CA_HAND["finger_half_wide"]   # m, the open finger pair's half extent along the opening axis (8 cm opening + the finger's width)
AC_BODY_BANDS_MAX = 6         # bands of CA_TOP_BAND searched below the assembly's top for the body level (a bound: 12 cm)
AC_APPROACH_UP = 0.12         # m, the grasp path's approach offset above the pinch pose (attempt_grasp_verify)
AC_SENSES = (1.0, -1.0)       # rotation sense about +z tried first / second (a convention: the sense that admits motion is kept, verified by the re-observation)


def ac_body_levels(pts):
    """The highest planar level of the assembly: ca_body on the whole assembly (the C2I1 rule, identical result when it succeeds), else
    ca_body on the next CA_TOP_BAND band below the previous band, at most AC_BODY_BANDS_MAX bands. Returns (body, tried)."""
    pts = np.asarray(pts, float); z = pts[:, 2]; z_top = float(np.percentile(z, CONTAINER_PCT_HI)); body = ca_body(pts)
    tried = [{"band": 0, "top": round(z_top, 4), "n": int(len(pts)), "cells": None if body is None else body["n_cells"]}]; k = 1
    while body is None and k < AC_BODY_BANDS_MAX:
        hi = z_top - k * CA_TOP_BAND; band = pts[(z < hi) & (z >= hi - CA_TOP_BAND)]
        body = ca_body(band) if len(band) >= CA_MIN_TOP_CELLS else None
        tried.append({"band": k, "top": round(hi, 4), "n": int(len(band)), "cells": None if body is None else body["n_cells"]}); k += 1
    return body, tried


def ac_lever(part, asm_pts, body):
    """Lever geometry of a part: lever = the part's top band (min rectangle), base = the assembly points outside the body outline within
    the lever's length of it, below the band and outside the lever footprint (fragments of any size), axis = the circle fit of the base
    (rms <= CA_CLEAR_MARGIN / 2, radius within the base's half diagonal) else the base's rectangle centre, far end = the lever end
    farther from the axis. Returns (lever, None) or (None, why)."""
    Q = np.asarray(part["pts"], float); z_hi = float(np.percentile(Q[:, 2], CONTAINER_PCT_HI)); top = Q[Q[:, 2] >= z_hi - CA_TOP_BAND]
    if len(top) < CA_PART_MIN_PTS // 4:
        return None, "no_lever_top"
    c_t, a_t, b_t, hu, hv = _ca_min_rect(top[:, :2]); length = 2 * hu
    P = np.asarray(asm_pts, float); c = body["centre"]; a, b = body["axes"]; hbu, hbv = body["half"]
    rel = P[:, :2] - c; out = (np.abs(rel @ a) > hbu + CA_OUTLINE_MARGIN) | (np.abs(rel @ b) > hbv + CA_OUTLINE_MARGIN)
    lu = (P[:, :2] - c_t) @ a_t; lv = (P[:, :2] - c_t) @ b_t; in_foot = (np.abs(lu) <= hu + CA_CLEAR_MARGIN) & (np.abs(lv) <= hv + CA_CLEAR_MARGIN)
    reach = np.hypot(P[:, 0] - c_t[0], P[:, 1] - c_t[1]) <= length
    base = P[out & reach & (P[:, 2] < z_hi - CA_TOP_BAND) & (~in_foot)]
    qu = (Q[:, :2] - c_t) @ a_t; qv = (Q[:, :2] - c_t) @ b_t; lever_pts = Q[(np.abs(qu) <= hu + CA_CLEAR_MARGIN) & (np.abs(qv) <= hv + CA_CLEAR_MARGIN)]
    axis_src = "part_rect"; axis = np.asarray(part["rect"]["centre"], float); base_top = float(np.percentile(Q[:, 2], CONTAINER_PCT_LO)); fit = None
    if len(base) >= CA_PART_MIN_PTS // 4:
        base_top = float(np.percentile(base[:, 2], CONTAINER_PCT_HI)); cb, _ab, _bb, hbu2, hbv2 = _ca_min_rect(base[:, :2]); axis = cb; axis_src = "base_rect"
        half_diag = float(np.hypot(hbu2, hbv2)); fit = fit_circle_xy(base[:, :2])
        if fit is not None and fit[3] <= CA_CLEAR_MARGIN / 2 and 0.5 * half_diag <= fit[2] <= 1.5 * half_diag:
            axis = np.array([fit[0], fit[1]]); axis_src = "circle"
    ends = [c_t - a_t * hu, c_t + a_t * hu]; dist = [float(np.linalg.norm(e - axis)) for e in ends]
    far = ends[int(np.argmax(dist))]; near = ends[int(np.argmin(dist))]; R_len = max(dist)
    if R_len < 2 * CA_HAND["finger_half_thin"] + CA_CLEAR_MARGIN:
        return None, "lever_too_short"
    if z_hi - base_top < CA_CLEAR_MARGIN:
        return None, "lever_too_low"
    d = (far - axis) / R_len; r_near = float((near - axis) @ d)
    return {"top_pts": top, "lever_pts": lever_pts, "z_hi": z_hi, "centre": c_t, "axis_dir": a_t, "half": (float(hu), float(hv)), "base_top": base_top, "n_base": int(len(base)), "axis": axis, "axis_src": axis_src,
            "circle": None if fit is None else [round(float(v), 4) for v in fit], "far": far, "R": float(R_len), "r_near": r_near, "d": d, "phi_deg": float(np.degrees(np.arctan2(d[1], d[0])))}, None


def _ac_open_boxes():
    """The OPEN hand as two boxes in tool coordinates: the finger pair spread to the opening (x +- AC_OPEN_HALF_WIDE) and the hand body."""
    H = CA_HAND
    return [((-AC_OPEN_HALF_WIDE, AC_OPEN_HALF_WIDE), (-H["finger_half_thin"], H["finger_half_thin"]), (H["finger_base"], H["tip"])),
            ((-H["body_half_wide"], H["body_half_wide"]), (-H["body_half_thin"], H["body_half_thin"]), (H["body_top"], H["body_bottom"]))]


def ac_clearance(obst, tcp, R):
    """No obstacle point inside either OPEN-hand box grown by CA_CLEAR_MARGIN, with the hand at (tcp, R). Returns (ok, n_conflicts, nearest)."""
    if obst is None or len(obst) == 0:
        return True, 0, None
    P = (np.asarray(obst, float) - np.asarray(tcp, float)) @ R; n = 0; nearest = None; m = CA_CLEAR_MARGIN
    for (x0, x1), (y0, y1), (z0, z1) in _ac_open_boxes():
        inside = (P[:, 0] >= x0 - m) & (P[:, 0] <= x1 + m) & (P[:, 1] >= y0 - m) & (P[:, 1] <= y1 + m) & (P[:, 2] >= z0 - m) & (P[:, 2] <= z1 + m)
        k = int(inside.sum()); n += k
        if k and nearest is None:
            nearest = np.round(np.asarray(obst, float)[inside][0], 3).tolist()
    return n == 0, n, nearest


def ac_pinch_plan(lev, floor_z, obst, ik_probe):
    """Pinch pose over the lever: TCP at r_pinch along the lever direction from the axis (the pads centred between the hub and the far
    end), tips above the hub by the margin, palm above the lever top by the margin, tool axis vertical with the finger opening across
    the lever; the open hand clears the obstacles at the pose and along the descent column; the public IK solves the pose (both yaw
    senses tried). Returns (plan, info) or (None, info)."""
    H = CA_HAND; d = np.asarray(lev["d"], float); axis = np.asarray(lev["axis"], float)
    r_pinch = 0.5 * (lev["R"] + max(lev["r_near"], 0.0) + H["finger_half_thin"])
    z_tip = max(lev["base_top"] + CA_CLEAR_MARGIN + CA_TIP_BELOW, float(floor_z) + CA_FLOOR_CLEAR, lev["z_hi"] + CA_CLEAR_MARGIN + H["body_bottom"] - H["tip"])
    engage = lev["z_hi"] - z_tip
    info = {"r_pinch": float(r_pinch), "z_tip": float(z_tip), "engage": float(engage)}
    if engage < CA_CLEAR_MARGIN:
        return None, dict(info, why="lever_too_low_for_the_pads")
    tcp = np.array([axis[0] + d[0] * r_pinch, axis[1] + d[1] * r_pinch, z_tip + H["tip"]]); z_high = float(tcp[2] + AC_APPROACH_UP); tried = []
    for flip in (False, True):
        R, q = ca_hand_frame(d, 0.0, flip=flip); rec = {"flip": flip}
        ok1, n1, near1 = ac_clearance(obst, tcp, R)
        if not ok1:
            rec.update(why="pinch_pose_obstructed", n=n1, nearest=near1); tried.append(rec); continue
        ok2 = True
        for z in np.arange(z_high, float(tcp[2]) - 1e-9, -CA_DESCENT_STEP).tolist():
            okz, nz, nearz = ac_clearance(obst, [tcp[0], tcp[1], z], R)
            if not okz:
                ok2 = False; rec.update(why="descent_column_obstructed", n=nz, nearest=nearz, z=round(float(z), 3)); break
        if not ok2:
            tried.append(rec); continue
        ik_hi = ik_probe([tcp[0], tcp[1], z_high], q); ik_pin = ik_probe(tcp, q); rec.update(ik_high=ik_hi, ik_pinch=ik_pin)
        if not (ik_hi and ik_pin):
            rec["why"] = "unreachable"; tried.append(rec); continue
        rec["why"] = "ok"; tried.append(rec)
        return {"tcp": tcp, "z_high": z_high, "quat": q, "R": R, "flip": flip, "axis": axis, "d": d, "r_pinch": float(r_pinch), "z_tip": float(z_tip), "engage": float(engage), "tried": tried}, info
    return None, dict(info, why="pinch_blocked_or_unreachable", tried=tried)


def ac_arc(axis, tcp, quat, sense, total_deg=None, step_deg=None):
    """Waypoints of the turn: the TCP rotated about the vertical axis through `axis` by sense x k x step, the tool yawed by the same angle
    (k = 1..n). Returns [(pos, quat_wxyz), ...]."""
    total = AC_TURN_DEG if total_deg is None else float(total_deg); step = AC_TURN_STEP_DEG if step_deg is None else float(step_deg)
    n = max(1, int(round(total / step))); tcp = np.asarray(tcp, float); axis = np.asarray(axis, float)[:2]; q = np.asarray(quat, float)
    R0 = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); wps = []
    for k in range(1, n + 1):
        ang = float(sense) * np.radians(step * k); Rz = Rotation.from_rotvec([0.0, 0.0, ang]).as_matrix()
        p = np.array([*(axis + Rz[:2, :2] @ (tcp[:2] - axis)), tcp[2]]); qk = Rotation.from_matrix(Rz @ R0).as_quat()
        wps.append((p, np.array([qk[3], qk[0], qk[1], qk[2]])))
    return wps


def ac_probe_arc(wps, j_now, ik_solve):
    """Number of leading arc waypoints whose public IK solution exists and lies within AC_TURN_JOINT_MAX of the previous one (the first
    of the actual joints j_now). ik_solve(pos, quat) -> joints or None. The driver calls it one increment at a time from the actual joints."""
    prev = np.asarray(j_now, float).reshape(-1)[:7]; n_ok = 0; steps = []
    for p, q in wps:
        j = ik_solve(p, q)
        if j is None:
            break
        j = np.asarray(j, float).reshape(-1)[:7]; s = float(np.linalg.norm(j - prev)); steps.append(round(s, 3))
        if s > AC_TURN_JOINT_MAX:
            break
        prev = j; n_ok += 1
    return n_ok, steps


def ac_angle_diff(phi_now, phi_prev):
    """Signed angle phi_now - phi_prev wrapped to (-180, 180] deg."""
    return float((float(phi_now) - float(phi_prev) + 180.0) % 360.0 - 180.0)


def ac_verdict(last, phi_prev, phi_now, senses_tried):
    """Verdict after a re-observation: actuated (a rotation >= AC_ROT_MIN_DEG in the turn's sense followed by a move that ended short =
    the part's stop), turn (rotation, every move reached), switch (no rotation, the first move ended short, the other sense untried),
    or a stop reason. Returns (action, why, rotation_deg)."""
    if last is None:
        return "turn", "first_cycle", None
    if phi_now is None or phi_prev is None:
        return "stop", "part_not_re_observed", None
    rot = ac_angle_diff(phi_now, phi_prev); prog = rot * float(last["sense"]); moves = list(last.get("moves") or []); short = (not moves) or any(m != "reached" for m in moves)
    if prog >= AC_ROT_MIN_DEG:
        return ("actuated", "stop_after_rotation", rot) if short else ("turn", "rotation_without_stop", rot)
    if short and len(senses_tried) < len(AC_SENSES):
        return "switch", "no_rotation_at_stop", rot
    return "stop", "no_rotation", rot


def ac_measure(rgb, depth_img, K, E, prompts, side, floor_z, tcp_xy, ik_probe, k=0):
    """One measurement of the assembly from the current frame: body level, parts, the part named by the side code (none = the most
    protruding), its lever and axis, the obstacle set and the pinch plan. Returns a record (printed as ACT_MEASURE by the caller)."""
    rec = {"k": k, "side": side, "assembly": None, "body": None, "bands": None, "parts": [], "lever": None, "plan": None, "plan_info": None, "why": None}
    pts, ainfo = ca_assembly(rgb, depth_img, K, E, prompts, floor_z); rec["assembly"] = ainfo
    if pts is None:
        rec["why"] = "assembly_not_found"; return rec
    body, bands = ac_body_levels(pts); rec["bands"] = bands
    if body is None:
        rec["why"] = "no_body_level"; return rec
    rec["body"] = {"centre": np.round(body["centre"], 4).tolist(), "axis_deg": round(float(np.degrees(np.arctan2(body["axes"][0][1], body["axes"][0][0])) % 180.0), 1), "half": [round(float(h), 4) for h in body["half"]],
                   "z_top": round(float(body["z_top"]), 4), "n_top": body["n_top"], "n_cells": body["n_cells"], "n_regions": body["n_regions"]}
    comps = ca_parts(pts, body)
    rec["parts"] = [{"n": q["n"], "centroid": np.round(q["centroid"], 4).tolist(), "n_out": np.round(q["n_out"], 2).tolist(), "protrusion": round(float(q["protrusion"]), 4),
                     "extent": [round(2 * float(q["rect"]["half_major"]), 4), round(2 * float(q["rect"]["half_minor"]), 4)], "z": [round(q["z_lo"], 4), round(q["z_hi"], 4)]} for q in comps]
    part = ca_select_part(comps, side)
    if part is None:
        rec["why"] = "no_part" if comps else "no_protruding_part"; return rec
    lev, why = ac_lever(part, pts, body)
    if lev is None:
        rec["why"] = why; return rec
    rec["lever"] = {"axis": np.round(lev["axis"], 4).tolist(), "axis_src": lev["axis_src"], "circle": lev["circle"], "far": np.round(lev["far"], 4).tolist(), "R": round(lev["R"], 4), "r_near": round(lev["r_near"], 4), "phi_deg": round(lev["phi_deg"], 1),
                    "z_hi": round(lev["z_hi"], 4), "base_top": round(lev["base_top"], 4), "n_top": int(len(lev["top_pts"])), "n_base": lev["n_base"], "extent": [round(2 * lev["half"][0], 4), round(2 * lev["half"][1], 4)]}
    F = ca_frame_raised(depth_img, K, E, floor_z, tcp_xy); obst = F[~ca_near(F, lev["lever_pts"], CA_SAME_POINT)]
    plan, pinfo = ac_pinch_plan(lev, floor_z, obst, ik_probe)
    rec["plan_info"] = {kk: (np.round(v, 4).tolist() if hasattr(v, "tolist") else v) for kk, v in pinfo.items()}; rec["plan"] = plan; rec["n_obstacle_pts"] = int(len(obst))
    if plan is None:
        rec["why"] = pinfo.get("why")
    return rec


def ac_plan_summary(plan):
    return {"tcp": np.round(plan["tcp"], 4).tolist(), "z_high": round(plan["z_high"], 4), "quat": np.round(plan["quat"], 4).tolist(), "flip": plan["flip"], "axis": np.round(plan["axis"], 4).tolist(), "d": np.round(plan["d"], 3).tolist(),
            "r_pinch": round(plan["r_pinch"], 4), "z_tip": round(plan["z_tip"], 4), "engage": round(plan["engage"], 4), "n_tried": len(plan["tried"])}


def ac_execute_turn(plan, sense, sg, k, step_check=None, total_deg=None, reanchor=False, yaw_select=None):   # C17I4: yaw_select(tcp, quat) -> (quat, budget) replaces H2 A's choice (None: the parent)   # C11I2: step_check(obs, i) -> True ends the arc after increment i; total_deg = the stroke (None = AC_TURN_DEG); reanchor: increments after the first rotate the EXECUTED pose
    """One turn cycle through the existing contracts: open, free approach (yaw alternative and budget by planned steps), contact-aware
    descent, gated close + grasp verification, gated transport, the probed arc as linear moves (stopping at the first not reached),
    verify_hold, open, ascent, home. Returns a record with the stage reached, the moves' statuses and the control steps used."""
    tcp = np.asarray(plan["tcp"], float); high = np.array([tcp[0], tcp[1], plan["z_high"]]); rec = {"k": k, "sense": float(sense), "stage": "approach", "steps": 0, "moves": [], "n_planned": 0, "grasp": None, "hold": None}

    def used(r):
        return int((r or {}).get("steps_used") or 0) if isinstance(r, dict) else 0

    def home():
        h = goto_home_joint_position(); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(h); rec["home"] = h.get("status") if isinstance(h, dict) else "unreported"

    def ascend(q):
        p, _, _ = read_tcp(); r = move_to_pose(np.array([p[0], p[1], plan["z_high"]]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:turn{k}_ascend"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r); rec["ascend"] = r.get("status")

    open_gripper(); _MF["last_plan"] = None; _WG["pts"] = None; _WG["quat"] = None   # the width gate belongs to object grasps; a fixed lever is not one
    q, budget = h_select_approach(tcp, np.asarray(plan["quat"], float)) if yaw_select is None else yaw_select(tcp, np.asarray(plan["quat"], float))   # C17I4
    r0 = move_to_pose(high, q, mode="free", tol_pos=0.015, step_budget=budget, label=f"{sg}:turn{k}_approach"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r0); rec["approach"] = r0.get("status")
    if r0.get("status") != "reached":
        home(); return rec
    rec["stage"] = "descend"
    rg = mc_grasp_descend(tcp, high, q, f"{sg}:turn{k}_descent"); rec["steps"] += used(rg); rec["descend"] = rg.get("status")
    ok_close, why = gate_allow(rg, "close"); rec["close_allowed"] = ok_close
    if not ok_close:
        print(f"ACT_ABORT reason={why}", flush=True); ascend(q); home(); return rec
    rec["stage"] = "grasp"
    gv = mc_close_and_verify(rg, sg); rec["grasp"] = gv.get("status")
    if gv.get("status") != "verified":
        print(f"ACT_ABORT reason=grasp_not_verified: {gv.get('reason')}", flush=True); open_gripper(); gate_record("open", role="recovery_open"); ascend(q); home(); return rec
    ok_t, why_t = gate_allow(gv, "transport"); rec["transport_allowed"] = ok_t
    if not ok_t:
        print(f"ACT_ABORT reason=turn_refused: {why_t}", flush=True); open_gripper(); gate_record("open", role="recovery_open"); ascend(q); home(); return rec
    rec["stage"] = "turn"
    p_now, q_now, obs_now = read_tcp(); j_now = np.asarray(obs_now["robot_joint_pos"], float).reshape(-1)[:7]
    wps = ac_arc(plan["axis"], p_now, q_now, sense, total_deg=total_deg); rec["n_planned"] = len(wps); _p, _q = p_now, q_now; jsteps = []; cut = None
    # each increment is probed from the ACTUAL joints just before its move (the same anchored solve the interface makes for a
    # one-waypoint linear move): the arc ends at the first increment whose solution is missing or jumps
    for i, (p_i, q_i) in enumerate(wps):
        if reanchor and i > 0:   # C11I2: one increment from the executed pose (no drift accumulated from an axis estimate error)
            p_i, q_i = ac_arc(plan["axis"], _p, _q, sense, total_deg=AC_TURN_STEP_DEG)[0]
        n_ok, st = ac_probe_arc([(p_i, q_i)], j_now, lambda p, qq: _ik(p, qq)); jsteps += st
        if n_ok == 0:
            cut = {"i": i + 1, "why": "joint_jump" if st else "ik_failed", "step": st[0] if st else None}; break
        if i == 0:
            gate_record("transport"); _mark(ev="turn_start", sg=sg, k=k, sense=sense, n=len(wps))
        r = move_to_pose(p_i, q_i, mode="linear", tol_pos=PRIM["reach_tol"], corridor=PUSH_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:turn{k}_{i + 1}"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r); rec["moves"].append(r.get("status"))
        if r.get("status") != "reached":
            break
        _p, _q, obs_i = read_tcp(); j_now = np.asarray(obs_i["robot_joint_pos"], float).reshape(-1)[:7]
        if step_check is not None and step_check(obs_i, i + 1):   # C11I2: an observed condition ends the arc
            rec["check_stop"] = i + 1; break
    rec["arc"] = {"n_planned": len(wps), "n_moved": len(rec["moves"]), "joint_steps": jsteps, "cut": cut}
    print("ACT_ARC " + _mfj({"sg": sg, "k": k, "sense": sense, "tcp_now": np.round(p_now, 4).tolist(), **rec["arc"]}), flush=True)
    if not rec["moves"]:
        rec["stage"] = "turn_unreachable"; open_gripper(); gate_record("open", role="recovery_open"); ascend(q_now); home(); return rec
    hv = verify_hold(gv["after"], gripper_opening()); rec["hold"] = hv.get("status")
    p_end, q_end, _ = read_tcp(); _mark(ev="turn_end", sg=sg, k=k, moves=rec["moves"], hold=rec["hold"], tcp=p_end)
    open_gripper(); gate_record("open", role="tool_config")   # the part stays where it is: nothing is released
    ascend(q_end); home(); rec["stage"] = "done"
    return rec


def run_actuate(sg, td, sg_label="sg1"):
    """Driver of the actuate_on relation: measure -> turn cycle -> re-observe, bounded. Returns True when the part is claimed actuated."""
    sup = la_phrase_in(td, "supports", sg.get("clause") or "")
    part_prompts = list(sg.get("part_prompts") or []) or [p for lst in (td.get("articulated") or {}).values() for p in list(lst)]   # the clause's part, else every articulated entry of the descriptors (data)
    prompts = part_prompts + (list((td.get("supports") or {}).get(sup) or []) if sup else []) + list(sg.get("support_prompts") or [])
    prompts = list(dict.fromkeys(prompts)); side = sg.get("side")
    _E1_LAST.update({"stop": False, "grasp_after": None, "hold": None})
    last = None; phi_prev = None; steps = 0; k = 0; senses_tried = []; sense = AC_SENSES[0]
    while True:
        obs = get_observation(); fr = mf_frame(obs); p_tcp, _ = tcp_from_obs(obs)
        floor_z = support_floor_z(fr["depth"], fr["K"], fr["E"]); _CG["floor_z"] = floor_z
        if floor_z is None:
            print("BINDING_STOP " + json.dumps({"status": "no_floor_estimate", "relation": sg["relation"]}), flush=True); return None
        rec = ac_measure(fr["rgb"], fr["depth"], fr["K"], fr["E"], prompts, side, floor_z, np.asarray(p_tcp, float)[:2], mf_ik_probe, k=k)
        print("ACT_MEASURE " + _mfj({kk: v for kk, v in rec.items() if kk != "plan"}), flush=True)
        phi = None if rec["lever"] is None else float(rec["lever"]["phi_deg"])
        if k == 0 and phi is None:
            print("BINDING_STOP " + json.dumps({"status": "part_not_found", "relation": sg["relation"], "side": side, "why": rec["why"], "prompts": prompts}), flush=True); return None
        action, why, rot = ac_verdict(last, phi_prev, phi, senses_tried)
        if last is not None:
            print("ACT_VERIFY " + _mfj({"sg": sg_label, "k": k, "claim": "actuated" if action == "actuated" else "not_actuated", "why": why, "rotation_deg": None if rot is None else round(rot, 1), "sense": last["sense"], "phi_before": phi_prev, "phi_after": phi,
                                         "moves": last["moves"], "hold": last.get("hold"), "steps_used": steps}), flush=True)
            _mark(ev="act_verify", sg=sg_label, k=k, claim="actuated" if action == "actuated" else "not_actuated", why=why)
        if action == "actuated":
            return True
        if action == "stop":
            return False
        if action == "switch":
            sense = [s for s in AC_SENSES if s not in senses_tried][0]
        if k >= CA_CYCLES_MAX or steps > CA_STEPS_GUARD:
            print("ACT_STOP " + json.dumps({"sg": sg_label, "k": k, "why": "cycles_exhausted" if k >= CA_CYCLES_MAX else "step_guard", "steps_used": steps}), flush=True); return False
        plan = rec["plan"]
        if plan is None:
            print("ACT_STOP " + _mfj({"sg": sg_label, "k": k, "why": rec["why"], "plan_info": rec["plan_info"]}), flush=True); return False
        print("ACT_PLAN " + _mfj({"sg": sg_label, "k": k, "sense": sense, **ac_plan_summary(plan)}), flush=True)
        _mark(ev="act_plan", sg=sg_label, k=k, tcp=plan["tcp"], sense=sense)
        if sense not in senses_tried:
            senses_tried.append(sense)
        res = ac_execute_turn(plan, sense, sg_label, k); steps += int(res["steps"])
        print("ACT_TURN " + _mfj({"sg": sg_label, **res}), flush=True)
        if res["stage"] != "done":
            print("ACT_STOP " + json.dumps({"sg": sg_label, "k": k, "why": f"motion_{res['stage']}_{res.get(res['stage'])}", "steps_used": steps}), flush=True); return False
        last = res; phi_prev = phi; k += 1
        for _ in range(3):
            get_observation()


LA_ACTUATE_RELATIONS = ("actuate_on",)   # relations run by run_actuate: a reference (support / part) and no object
# ===== end C2I2 manipulation_primitive =====
# ===== end C2I1 manipulation_primitive =====


# ===== C11I2 manipulation_primitive: rotary actuation, both senses (N11 round, chunk 11, iteration 2; from C10I5) =====
# One driver (run_rotary) for actuate_on and actuate_off: turning a lever-type part of a support until the support's emitting
# indicator reads the target state (on: lit, off: dark). Public data only (segmenter masks of the descriptors' part / support
# prompts, rgb, depth, the public IK probe, the interface's motion contracts). From C10I5 (N10 chunk 10, iteration 5):
#  - the pinch (ao_pinch_plan) COMMANDS the hand model frame @ OP_TOOL_FROM_MODEL (C5I3 convention), so the pads close across the
#    lever; C2I2's ac_pinch_plan commands the model frame itself and the pads land along the lever (C10I3: 14/14 descents);
#  - the part's own mask joined to the support, the body without the part's points, the centre-pivot axis on the lever line, both
#    lever halves, lever-volume pixels not obstacles; the indicator read by a chroma test with a depth occlusion test.
# C11I2 changes (DIAGNOSIS.md):
#  (1) the assembly is anchored on the PART: the support mask adjacent to the best part mask, not the largest support mask (the
#      largest support-prompt mask can be another raised body; N10: no_protruding_part on 16/20 turn-on development starts);
#  (2) a turn is driven to the part's stop (or the stroke's end) and NOT ended at the first increment where the indicator changes:
#      the indicator changes inside the travel, short of the state the part reaches at its stop (C10I5: dark read 10/10, the task's
#      off state 0/10; the dial stayed 13-16 deg short of its rest);
#  (3) every increment after the first is re-anchored on the executed TCP pose: an absolute arc about an axis estimate off by e
#      drifts from the part's circle by 2 e sin(theta / 2) and reaches the reach tolerance after ~7 increments (C10I5 k1 pos_err
#      0.47 -> 0.97 cm over 7 increments, 4/10 ended `stalled` there); the per-increment drift is 2 e sin(7.5 deg) = 0.26 e;
#  (4) one state rule for both senses: lit fraction of the visible indicator pixels >= AO_LIT_MIN reads on, below reads off.
RT_TARGET = {"actuate_on": "on", "actuate_off": "off"}   # relation -> indicator state that completes it
RT_RELATIONS = tuple(RT_TARGET)   # relations run by run_rotary: a reference (support / part) and no object
AO_CHROMA_MIN = 0.5      # chroma of an emitting pixel: midway between the emitter (0.925) and the brightest non-emitting surface in view (table, max 0.173; part masks p99 <= 0.11) (C10I3)
AO_LIT_MIN = 0.02        # lit fraction of the visible indicator pixels at and above which it reads on: measured on 0.059-0.071 (C10I3/C10I5, 13 frames), off 0.000 (C10I5 10/10 dark frames); about a third of the lowest on value
AO_OCCLUDE = CA_CLEAR_MARGIN   # m, a pixel whose depth is nearer than at task start by more than the depth / xy sigma is occluded (hand, arm)
AO_VISIBLE_MIN = 0.5     # a verdict needs at least half of the indicator pixels visible
AO_PART_GAP = 2 * CA_CLUSTER_CELL   # m, the part belongs to a support mask whose points lie within two clustering cells of its points
AO_TURN_DEG = 180.0      # deg per grasp cycle (C10I3/C10I5 stroke): >= the measured stop-to-stop travel (~121 deg) + the hand's lead over the part at a stop (~18 deg), at the 15-deg increment; the turn is ended by the stop, the IK probe's cut or this stroke
AO_CIRCLE_RMS = CA_CLEAR_MARGIN   # m, a circle fit of a centre-pivot part's base is its rotation axis when its rms is within the depth / xy sigma
AO_RADIUS_TOL = 0.25     # the fitted hub radius may differ from the lever half length by a quarter (C10I3)


def ao_lit_map(rgb):
    """Boolean pixel map of emitting pixels: chroma (max - min channel) / 255 >= AO_CHROMA_MIN."""
    a = np.asarray(rgb, float)
    return (a.max(axis=2) - a.min(axis=2)) / 255.0 >= AO_CHROMA_MIN


def _ao_keep(pts, floor_z):
    pts = np.asarray(pts, float)
    return (pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] < C_WS[2][1]) & (pts[:, 2] > float(floor_z) + SUPPORT_RAISED_MIN)


def ao_candidates(rgb, depth_img, K, E, prompts, floor_z):
    """Masks of the prompts that pass ca_assembly's gate (top 15 per prompt, in-box majority, >= CA_PART_MIN_PTS raised points)."""
    out = []; rej = 0
    for prompt in prompts:
        for m in sorted(segment_sam3_text_prompt(rgb, prompt) or [], key=lambda d: -float(d["score"]))[:15]:
            mask = np.asarray(m["mask"]).astype(np.uint8); pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < CA_PART_MIN_PTS:
                rej += 1; continue
            pts = np.asarray(pts, float)
            inbox = (pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] < C_WS[2][1])
            keep = _ao_keep(pts, floor_z)
            if int(inbox.sum()) < 0.5 * len(pts) or int(keep.sum()) < CA_PART_MIN_PTS:
                rej += 1; continue
            out.append({"prompt": prompt, "score": float(m["score"]), "mask": mask, "pts": pts[keep], "n": int(keep.sum())})
        _PRIM_STATS["sam3_calls"] += 1
    return out, rej


def ao_assembly(rgb, depth_img, K, E, part_prompts, support_prompts, floor_z):
    """Support assembly anchored on its PART (C11I2): the part is the highest-scoring part-prompt mask that has a support-prompt mask
    within AO_PART_GAP of its points; the anchor is the largest support mask within AO_PART_GAP of that part, merged with the
    support masks overlapping it by SUPPORT_SAME_MASK (ca_assembly's merge rule). The support prompts can name a larger raised body
    elsewhere in view (its largest mask then is not the part's support); the part prompts name the part itself. When no
    part mask is adjacent to any support mask, the C10I5 rule is kept (largest support mask, no part). Returns
    (pts, support_mask, part_pts, info); (None, None, None, info) when no support mask passes the gate."""
    sup, rej = ao_candidates(rgb, depth_img, K, E, support_prompts, floor_z)
    if not sup:
        return None, None, None, {"n_support": 0, "rejected": rej}
    cands, _r = ao_candidates(rgb, depth_img, K, E, part_prompts, floor_z)
    trees = [_KDTree(c["pts"]) for c in sup]; best = None; adj = []; pc_info = []
    for c in sorted(cands, key=lambda x: -x["score"]):
        gaps = [float(np.min(t.query(c["pts"])[0])) for t in trees]; near = [j for j, g in enumerate(gaps) if g <= AO_PART_GAP]
        pc_info.append({"prompt": c["prompt"], "score": round(c["score"], 3), "n": c["n"], "gap": round(min(gaps), 4), "n_adjacent": len(near)})
        if best is None and near:
            best = c; adj = near
    anchor = max((sup[j] for j in adj), key=lambda c: c["n"]) if best is not None else max(sup, key=lambda c: c["n"])
    merged = [c for c in sup if c is anchor or mask_overlap_frac(c["mask"], anchor["mask"]) >= SUPPORT_SAME_MASK]
    smask = np.zeros_like(anchor["mask"])
    for c in merged:
        smask |= c["mask"]
    spts = np.asarray(mask_to_world_points(smask, depth_img, K, E), float); spts = spts[_ao_keep(spts, floor_z)]
    info = {"n_support": len(sup), "rejected": rej, "anchor": {"prompt": anchor["prompt"], "score": round(anchor["score"], 3), "n": anchor["n"], "rule": "part_adjacent" if best is not None else "largest"},
            "n_merged": len(merged), "n_support_pts": int(len(spts)), "part": None, "part_candidates": pc_info[:8]}
    if len(spts) < CA_PART_MIN_PTS:
        return None, None, None, dict(info, why="support_too_small")
    if best is None:
        return spts, smask, None, info
    info["part"] = {"prompt": best["prompt"], "score": round(best["score"], 3), "n": best["n"]}
    return np.concatenate([spts, best["pts"]]), smask, best["pts"], info


def _ao_about(lev, axis, src):
    """The lever record re-expressed about a new axis, exactly as ac_lever derives it: far end = the lever end farther from the axis,
    R its distance, d the unit direction, r_near the near end's offset along d. None when the lever is too short for the pads."""
    hu = float(lev["half"][0]); c_t = np.asarray(lev["centre"], float); a_t = np.asarray(lev["axis_dir"], float); axis = np.asarray(axis, float)
    ends = [c_t - a_t * hu, c_t + a_t * hu]; dist = [float(np.linalg.norm(e - axis)) for e in ends]
    i_far = int(np.argmax(dist)); far = ends[i_far]; near = ends[1 - i_far]; R_len = dist[i_far]   # the other end is the near one (also when both are equidistant)
    if R_len < 2 * CA_HAND["finger_half_thin"] + CA_CLEAR_MARGIN:
        return None
    d = (far - axis) / R_len
    return dict(lev, axis=axis, axis_src=src, far=far, R=float(R_len), r_near=float((near - axis) @ d), d=d, phi_deg=float(np.degrees(np.arctan2(d[1], d[0]))))


def ao_on_line(lev, p):
    """The point of the lever's centre line (its top band's rectangle centre + its long axis) nearest the xy point p."""
    c = np.asarray(lev["centre"], float); a = np.asarray(lev["axis_dir"], float); return c + a * float((np.asarray(p, float)[:2] - c) @ a)


def ao_in_lever(lev, pts):
    """Boolean mask of the points inside the lever's own volume: within its top band's rectangle grown by CA_CLEAR_MARGIN (the
    lever_pts rule of ac_lever) and not above its top by more than the margin. Such points are the lever itself (frame pixels at the
    mask's boundary), never an obstacle between the open pads: the lever occupies that volume."""
    P = np.asarray(pts, float)
    if len(P) == 0:
        return np.zeros(0, bool)
    c = np.asarray(lev["centre"], float); a = np.asarray(lev["axis_dir"], float); b = np.array([-a[1], a[0]]); hu, hv = lev["half"]
    u = (P[:, :2] - c) @ a; v = (P[:, :2] - c) @ b
    return (np.abs(u) <= hu + CA_CLEAR_MARGIN) & (np.abs(v) <= hv + CA_CLEAR_MARGIN) & (P[:, 2] <= float(lev["z_hi"]) + CA_CLEAR_MARGIN)


def ao_tool_quat(Rm):
    """wxyz quaternion of the COMMANDED tool frame for the hand model rotation Rm (C5I3: model frame @ OP_TOOL_FROM_MODEL; the
    robot's finger-opening axis, tool y, is the model's x)."""
    q = Rotation.from_matrix(np.asarray(Rm, float) @ OP_TOOL_FROM_MODEL).as_quat(); return np.array([q[3], q[0], q[1], q[2]])


def ao_pinch_plan(lev, floor_z, obst, ik_probe):
    """ac_pinch_plan's pinch (tips CA_TIP_BELOW + margin over the hub, palm a margin over the lever top, tool axis vertical, the open
    hand model clear of the obstacles at the pose and down the descent column, the public IK solving the high and pinch poses, both
    flips), with three differences: the COMMANDED quaternion is ao_tool_quat(model frame), so the finger-opening axis is across the
    lever; a centre-pivot lever (r_near < 0) offers both halves (far half first; per half r_pinch = (R_half + finger_half_thin) / 2,
    the C2I2 rule with the hub at the near end), an end-pivot lever only its far half (the C2I2 r_pinch); the TCP is on the line
    axis + d r (the axis lies on the lever's centre line for a centre-pivot lever, ao_lever). Returns (plan, info) or (None, info)."""
    H = CA_HAND; d0 = np.asarray(lev["d"], float); axis = np.asarray(lev["axis"], float)
    z_tip = max(lev["base_top"] + CA_CLEAR_MARGIN + CA_TIP_BELOW, float(floor_z) + CA_FLOOR_CLEAR, lev["z_hi"] + CA_CLEAR_MARGIN + H["body_bottom"] - H["tip"])
    engage = lev["z_hi"] - z_tip
    info = {"z_tip": float(z_tip), "engage": float(engage)}
    if engage < CA_CLEAR_MARGIN:
        return None, dict(info, why="lever_too_low_for_the_pads")
    if float(lev["r_near"]) < 0.0:
        halves = [(1.0, float(lev["R"])), (-1.0, -float(lev["r_near"]))]
    else:
        halves = [(1.0, None)]
    tried = []
    for sgn, R_half in halves:
        if R_half is None:
            r_pinch = 0.5 * (lev["R"] + max(lev["r_near"], 0.0) + H["finger_half_thin"])
        else:
            if R_half < 2 * H["finger_half_thin"] + CA_CLEAR_MARGIN:
                tried.append({"half": sgn, "why": "half_too_short_for_the_pads", "R_half": round(R_half, 4)}); continue
            r_pinch = 0.5 * (R_half + H["finger_half_thin"])
        d = sgn * d0
        tcp = np.array([axis[0] + d[0] * r_pinch, axis[1] + d[1] * r_pinch, z_tip + H["tip"]]); z_high = float(tcp[2] + AC_APPROACH_UP)
        for flip in (False, True):
            Rm, _q_model = ca_hand_frame(d, 0.0, flip=flip); q = ao_tool_quat(Rm); rec = {"half": sgn, "flip": flip, "r_pinch": round(float(r_pinch), 4)}
            ok1, n1, near1 = ac_clearance(obst, tcp, Rm)
            if not ok1:
                rec.update(why="pinch_pose_obstructed", n=n1, nearest=near1); tried.append(rec); continue
            ok2 = True
            for z in np.arange(z_high, float(tcp[2]) - 1e-9, -CA_DESCENT_STEP).tolist():
                okz, nz, nearz = ac_clearance(obst, [tcp[0], tcp[1], z], Rm)
                if not okz:
                    ok2 = False; rec.update(why="descent_column_obstructed", n=nz, nearest=nearz, z=round(float(z), 3)); break
            if not ok2:
                tried.append(rec); continue
            ik_hi = ik_probe([tcp[0], tcp[1], z_high], q); ik_pin = ik_probe(tcp, q); rec.update(ik_high=ik_hi, ik_pinch=ik_pin)
            if not (ik_hi and ik_pin):
                rec["why"] = "unreachable"; tried.append(rec); continue
            rec["why"] = "ok"; tried.append(rec)
            return {"tcp": tcp, "z_high": z_high, "quat": q, "R": Rm, "flip": flip, "half": sgn, "axis": axis, "d": d, "r_pinch": float(r_pinch), "z_tip": float(z_tip), "engage": float(engage), "tried": tried}, dict(info, r_pinch=float(r_pinch), half=sgn)
    return None, dict(info, why="pinch_blocked_or_unreachable", tried=tried)


def ao_lever(part, asm_pts, body):
    """ac_lever, with the rotation axis of a CENTRE-PIVOT part (the lever crosses its hub: ac_lever's r_near < 0, the near end lies
    beyond the axis) taken as (a) the circle fit of the base when its rms <= AO_CIRCLE_RMS and its radius is within AO_RADIUS_TOL of
    the lever's half length (the hub disc and the lever across it have the same extent), projected on the lever's centre line (C10I5:
    a lever across its hub passes through the hub centre), else (b) the lever's own centre (a lever across its hub is centred on it). An end-pivot lever (r_near >= 0) and ac_lever's strict circle are kept unchanged."""
    lev, why = ac_lever(part, asm_pts, body)
    if lev is None or lev["axis_src"] == "circle" or lev["r_near"] >= 0.0:
        return lev, why
    hu = float(lev["half"][0])
    if lev["circle"] is not None:
        cx, cy, r, rms = [float(v) for v in lev["circle"]]
        if rms <= AO_CIRCLE_RMS and abs(r - hu) <= AO_RADIUS_TOL * hu:
            out = _ao_about(lev, ao_on_line(lev, [cx, cy]), "circle_ao")
            if out is not None:
                return out, None
    out = _ao_about(lev, lev["centre"], "lever_centre")
    return (out, None) if out is not None else (lev, why)


def ao_named_part(comps, part_pts):
    """The part component that holds the part prompts' own mask (C11I2): the component with the most points within CA_SAME_POINT
    of the part mask's points (ties: the C2I1 most-protruding rule). The support masks can also cover other bodies resting on the
    support (they then join the assembly as protruding components); without a side code the named part is the one the part
    prompts segmented. None when no component holds any of those points."""
    best = None; best_n = 0
    for q in comps:
        n = int(ca_near(np.asarray(q["pts"], float), part_pts, CA_SAME_POINT).sum())
        if n > best_n or (n == best_n and n > 0 and float(q["protrusion"]) > float(best["protrusion"])):
            best, best_n = q, n
    return best


def ao_measure(rgb, depth_img, K, E, part_prompts, support_prompts, side, floor_z, tcp_xy, ik_probe, k=0):
    """ac_measure on the ao_assembly: body level, parts, the part named by the side code (none = the most protruding), its lever and
    axis, the obstacle set and the pinch plan (all C2I2 functions, unchanged). The record also carries the support mask."""
    rec = {"k": k, "side": side, "assembly": None, "body": None, "bands": None, "parts": [], "lever": None, "plan": None, "plan_info": None, "why": None, "support_mask": None}
    pts, smask, part_pts, ainfo = ao_assembly(rgb, depth_img, K, E, part_prompts, support_prompts, floor_z); rec["assembly"] = ainfo; rec["support_mask"] = smask
    if pts is None:
        rec["why"] = "assembly_not_found"; return rec
    body_pts = pts if part_pts is None else pts[~ca_near(pts, part_pts, CA_SAME_POINT)]   # the part's own points are not body (a part touching the body would join its outline)
    body, bands = ac_body_levels(body_pts) if len(body_pts) >= CA_MIN_TOP_CELLS else (None, []); rec["bands"] = bands
    if body is None:
        rec["why"] = "no_body_level"; return rec
    rec["body"] = {"centre": np.round(body["centre"], 4).tolist(), "axis_deg": round(float(np.degrees(np.arctan2(body["axes"][0][1], body["axes"][0][0])) % 180.0), 1), "half": [round(float(h), 4) for h in body["half"]],
                   "z_top": round(float(body["z_top"]), 4), "n_top": body["n_top"], "n_cells": body["n_cells"], "n_regions": body["n_regions"]}
    comps = ca_parts(pts, body)
    rec["parts"] = [{"n": q["n"], "centroid": np.round(q["centroid"], 4).tolist(), "n_out": np.round(q["n_out"], 2).tolist(), "protrusion": round(float(q["protrusion"]), 4),
                     "extent": [round(2 * float(q["rect"]["half_major"]), 4), round(2 * float(q["rect"]["half_minor"]), 4)], "z": [round(q["z_lo"], 4), round(q["z_hi"], 4)]} for q in comps]
    part = ca_select_part(comps, side) if (side is not None or part_pts is None) else ao_named_part(comps, part_pts)   # C11I2: no side code -> the component holding the part prompts' own mask, not the most protruding one
    if part is None:
        rec["why"] = "no_part" if comps else "no_protruding_part"; return rec
    lev, why = ao_lever(part, pts, body)
    if lev is None:
        rec["why"] = why; return rec
    rec["lever"] = {"axis": np.round(lev["axis"], 4).tolist(), "axis_src": lev["axis_src"], "circle": lev["circle"], "far": np.round(lev["far"], 4).tolist(), "R": round(lev["R"], 4), "r_near": round(lev["r_near"], 4), "phi_deg": round(lev["phi_deg"], 1),
                    "z_hi": round(lev["z_hi"], 4), "base_top": round(lev["base_top"], 4), "n_top": int(len(lev["top_pts"])), "n_base": lev["n_base"], "extent": [round(2 * lev["half"][0], 4), round(2 * lev["half"][1], 4)]}
    F = ca_frame_raised(depth_img, K, E, floor_z, tcp_xy); obst = F[~ca_near(F, lev["lever_pts"], CA_SAME_POINT) & ~ao_in_lever(lev, F)]   # C10I5: frame points inside the lever's own footprint are the lever (pixels its mask missed)
    plan, pinfo = ao_pinch_plan(lev, floor_z, obst, ik_probe)   # C10I5: pads across the lever (commanded tool frame), both halves
    rec["plan_info"] = {kk: (np.round(v, 4).tolist() if hasattr(v, "tolist") else v) for kk, v in pinfo.items()}; rec["plan"] = plan; rec["n_obstacle_pts"] = int(len(obst))
    if plan is None:
        rec["why"] = pinfo.get("why")
    return rec


def ao_indicator(rgb, depth_img, smask):
    """Task-start reference of the support's indicator: its pixel set and depth (the occlusion reference)."""
    S = np.asarray(smask) > 0; d = np.asarray(depth_img, float); d = d[:, :, 0] if d.ndim == 3 else d; lit = ao_lit_map(rgb)
    return {"S": S, "d0": d.copy(), "n": int(S.sum()), "lit0_frac": float(lit[S].mean()) if S.any() else 0.0}


def rt_state(ref, rgb, depth_img):
    """(state, info) of the support's emitting indicator in one frame: over the indicator pixels still visible (depth not nearer than
    at task start by more than AO_OCCLUDE), lit fraction >= AO_LIT_MIN -> "on", else "off". None = no verdict (fewer than
    AO_VISIBLE_MIN of the pixels visible: occlusion is never read as a state)."""
    d = np.asarray(depth_img, float); d = d[:, :, 0] if d.ndim == 3 else d
    vis = ref["S"] & np.isfinite(d) & (d > 0) & (d >= ref["d0"] - AO_OCCLUDE); nv = int(vis.sum())
    info = {"visible": round(nv / max(ref["n"], 1), 3)}
    if nv < AO_VISIBLE_MIN * ref["n"]:
        return None, dict(info, why="occluded")
    lit = float(ao_lit_map(rgb)[vis].mean()); info["lit"] = round(lit, 4)
    st = "on" if lit >= AO_LIT_MIN else "off"; info["why"] = st
    return st, info


def rt_next_sense(sense, last, state_changed):
    """Sense of the next cycle: reversed when the last turn kept the indicator unchanged and ended at an unreached increment (the
    part's stop in that sense) or could not start; kept otherwise (arc completed or cut by the IK probe)."""
    if last is None or state_changed:
        return sense
    moves = list(last.get("moves") or [])
    if (not moves) or any(m != "reached" for m in moves):
        return -float(sense)
    return sense


# ===== C17I4 manipulation_primitive: a rest-state turn is driven to the part's stop (N17 group N, slot chunk17_iter4) =====
# The emitting indicator of a support turns dark inside the lever's travel, short of the rest the part reaches at its stop
# (C11I2's finding; N17 development rows: dark at increment 8, the dial 10-11 deg short of its stop, the relation false 5/5).
# C11I2 drives a turn to the stop, but (a) H2 A picks the pinch's 180-deg yaw alternative by the approach's planned steps alone,
# and that alternative can leave the last arm joint too little travel in the turn's direction, so the public IK probe cuts the
# arc (joint_jump) before the stop; (b) the driver accepts the target state read after such a cut. For a REST target:
#  (1) the alternative whose last arm joint lies farther behind in the turn's direction is taken: per alternative, the public
#      IK of the high pose (anchored at the actual joints, as the approach's own solve) gives the joint value j; with the tool
#      axis vertical an increment of sense x a turns that joint by -sense x a, so lead = -sense x j; in a joint range symmetric
#      about zero the smaller lead leaves more travel. A tie, a missing solution or an unplannable alternative keeps the parent's choice;
#  (2) the target state read after a turn without an unreached increment (cut or stroke end: not at the stop) starts one more
#      cycle in the same sense within CA_CYCLES_MAX / CA_STEPS_GUARD; when the cycles end otherwise the claim is the parent's.
# No new numeric constant. Other relations and run_actuate run the parent code.
RT_REST = ("actuate_off",)   # relations whose target indicator state is the part's rest at the end of its travel


def rt_rest_pending(relation, last):
    """True when the relation's target is a rest state and the last turn ended without an unreached increment (every move
    reached: an IK cut or the stroke's end, not the part's stop)."""
    if relation not in RT_REST or last is None:
        return False
    moves = list(last.get("moves") or [])
    return bool(moves) and all(m == "reached" for m in moves)


def rt_wrist_lead(high, quat, sense):
    """-sense * j of the last arm joint for the pinch alternative `quat` (j from the public IK of the high pose, anchored at the
    actual joints like the approach's own solve). Every pinch plan has the tool axis vertical (ao_pinch_plan: tilt 0) and the
    last joint turns about the flange axis, parallel to the tool axis (down): an arc increment of sense x a about the world's
    up axis turns that joint by -sense x a (recorded: -0.18 rad per +15 deg increment, +0.19 per -15 deg). The travel remaining in the
    turn's direction in a joint range symmetric about zero is L - (-sense x j): the smaller value leaves more. None without a solution."""
    j_hi = _ik(high, quat)
    if j_hi is None:
        return None
    return float(-float(sense) * float(np.asarray(j_hi, float).reshape(-1)[6]))


def rt_select_turn_yaw(tcp, quat, plan, sense):
    """h_select_approach (the same plan_only probes, budget rule and APPROACH_PLAN record) with the alternative chosen by the
    wrist lead (rule 1); the parent's choice on a tie or missing data. Returns (quat, budget)."""
    tcp = np.asarray(tcp, float); ap = np.array([tcp[0], tcp[1], tcp[2] + AC_APPROACH_UP])   # H2 A's approach point (0.12 m above)
    cands = [np.asarray(quat, float), _flip_tool_z(np.asarray(quat, float))]
    plans = [move_to_pose(ap, q, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=f"sg1:approach_plan{k}") for k, q in enumerate(cands)]
    steps = [int(p["steps_planned"]) if (p.get("detail") == "plan_only" and p.get("steps_planned") is not None) else None for p in plans]
    ok = [(s, k) for k, s in enumerate(steps) if s is not None]
    k_par = min(ok)[1] if ok else None
    leads = [rt_wrist_lead(ap, q, sense) for q in cands]
    k_sel = k_par
    if k_par is not None and all(v is not None for v in leads) and leads[0] != leads[1]:
        k_lead = int(np.argmin(leads))
        if steps[k_lead] is not None:
            k_sel = k_lead
    budget = BUDGET_FREE if k_sel is None else int(min(H_APPROACH_CAP, max(BUDGET_FREE, int(np.ceil(1.25 * steps[k_sel])))))
    print("APPROACH_PLAN " + json.dumps({"planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": k_sel, "budget": budget}), flush=True)
    print("TURN_YAW " + _mfj({"sense": float(sense), "lead": [None if v is None else round(v, 3) for v in leads], "parent": k_par, "chosen": k_sel}), flush=True)
    return (cands[0] if k_sel is None else cands[k_sel]), budget
# ===== end C17I4 helpers =====


def run_rotary(sg, td, sg_label="sg1"):
    """Driver of actuate_on / actuate_off (C11I2): measure (part-anchored assembly, lever, ao_pinch_plan: pads across the lever) +
    indicator reference -> grasp-and-turn cycles -> re-observe, bounded by CA_CYCLES_MAX / CA_STEPS_GUARD. Every turn is driven in
    one sense until the part's stop (the first unreached increment) or the stroke's end, each increment re-anchored on the
    executed TCP pose. Returns True when the indicator reads the target state, False when the cycles end without it, None on a
    pre-motion stop."""
    target = RT_TARGET[sg["relation"]]
    sup = la_phrase_in(td, "supports", sg.get("clause") or "")
    part_prompts = list(sg.get("part_prompts") or []) or [p for lst in (td.get("articulated") or {}).values() for p in list(lst)]   # as run_actuate
    sup_prompts = list(dict.fromkeys((list((td.get("supports") or {}).get(sup) or []) if sup else []) + list(sg.get("support_prompts") or [])))
    part_prompts = list(dict.fromkeys(part_prompts)); side = sg.get("side")
    _E1_LAST.update({"stop": False, "grasp_after": None, "hold": None})
    ref = None; last = None; st_prev = None; steps = 0; k = 0; sense = AC_SENSES[0]
    while True:
        obs = get_observation(); fr = mf_frame(obs); p_tcp, _ = tcp_from_obs(obs)
        floor_z = support_floor_z(fr["depth"], fr["K"], fr["E"]); _CG["floor_z"] = floor_z
        if floor_z is None:
            print("BINDING_STOP " + json.dumps({"status": "no_floor_estimate", "relation": sg["relation"]}), flush=True); return None
        st = None
        if ref is not None:
            st, sinfo = rt_state(ref, fr["rgb"], fr["depth"])
            print("ACT_VERIFY " + _mfj({"sg": sg_label, "k": k, "relation": sg["relation"], "claim": "actuated" if st == target else "not_actuated", "state": st, "indicator": sinfo, "sense": last["sense"], "moves": last["moves"], "hold": last.get("hold"), "steps_used": steps}), flush=True)
            _mark(ev="act_verify", sg=sg_label, k=k, claim="actuated" if st == target else "not_actuated", why=sinfo.get("why"))
            rest = rt_rest_pending(sg["relation"], last) if st == target else False   # C17I4 (2): not yet at the part's stop
            if st == target and not rest:
                return True
            if rest:
                print("ACT_REST " + _mfj({"sg": sg_label, "k": k, "relation": sg["relation"], "why": "target_state_before_the_stop", "sense": last["sense"], "moves": last["moves"], "cut": (last.get("arc") or {}).get("cut"), "steps_used": steps}), flush=True)
        rec = ao_measure(fr["rgb"], fr["depth"], fr["K"], fr["E"], part_prompts, sup_prompts, side, floor_z, np.asarray(p_tcp, float)[:2], mf_ik_probe, k=k)
        print("ACT_MEASURE " + _mfj({kk: v for kk, v in rec.items() if kk not in ("plan", "support_mask")}), flush=True)
        if ref is None:
            if rec["support_mask"] is None:
                print("BINDING_STOP " + json.dumps({"status": "part_not_found", "relation": sg["relation"], "side": side, "why": rec["why"], "prompts": sup_prompts}), flush=True); return None
            ref = ao_indicator(fr["rgb"], fr["depth"], rec["support_mask"]); st, sinfo = rt_state(ref, fr["rgb"], fr["depth"])
            print("ACT_INDICATOR " + _mfj({"sg": sg_label, "relation": sg["relation"], "target": target, "n": ref["n"], "lit0_frac": round(ref["lit0_frac"], 4), "state0": st, "lit_min": AO_LIT_MIN}), flush=True)
            if st == target:   # the indicator already reads the target state at task start: no motion
                print("ACT_VERIFY " + _mfj({"sg": sg_label, "k": 0, "relation": sg["relation"], "claim": "actuated", "why": "target_state_at_start", "state": st}), flush=True); return True
            if rec["lever"] is None:
                print("BINDING_STOP " + json.dumps({"status": "part_not_found", "relation": sg["relation"], "side": side, "why": rec["why"], "prompts": part_prompts}), flush=True); return None
            st_prev = st
        rest_mode = st is not None and st == target   # C17I4: the target state was read (a later stop keeps the parent's True)
        changed = st is not None and st_prev is not None and st != st_prev
        new_sense = rt_next_sense(sense, last, changed)
        if last is not None and new_sense != sense and last.get("stage") == "turn_unreachable" and last.get("prev_unreachable"):
            print("ACT_STOP " + json.dumps({"sg": sg_label, "k": k, "why": "turn_unreachable_both_senses", "steps_used": steps}), flush=True); return False
        sense = new_sense
        if k >= CA_CYCLES_MAX or steps > CA_STEPS_GUARD:
            print("ACT_STOP " + json.dumps({"sg": sg_label, "k": k, "why": "cycles_exhausted" if k >= CA_CYCLES_MAX else "step_guard", "steps_used": steps}), flush=True); return rest_mode   # C17I4: parent False unless the target state was read
        plan = rec["plan"]
        if plan is None:
            print("ACT_STOP " + _mfj({"sg": sg_label, "k": k, "why": rec["why"], "plan_info": rec["plan_info"]}), flush=True); return rest_mode   # C17I4
        print("ACT_PLAN " + _mfj({"sg": sg_label, "k": k, "sense": sense, "relation": sg["relation"], "half": plan.get("half"), **ac_plan_summary(plan)}), flush=True)
        _mark(ev="act_plan", sg=sg_label, k=k, tcp=plan["tcp"], sense=sense)

        def step_check(obs_i, i, _ref=ref, _k=k):   # logged only: the turn goes on to the stop or the stroke's end
            f = mf_frame(obs_i); s_i, inf = rt_state(_ref, f["rgb"], f["depth"])
            print("ACT_STATE_STEP " + _mfj({"sg": sg_label, "k": _k, "i": i, "state": s_i, **inf}), flush=True)
            return False

        was_unreachable = last is not None and last.get("stage") == "turn_unreachable"
        yaw_sel = (lambda t, q, _p=plan, _s=sense: rt_select_turn_yaw(t, q, _p, _s)) if sg["relation"] in RT_REST else None   # C17I4 (1)
        res = ac_execute_turn(plan, sense, sg_label, k, step_check=step_check, total_deg=AO_TURN_DEG, reanchor=True, yaw_select=yaw_sel); steps += int(res["steps"])
        res["prev_unreachable"] = was_unreachable
        print("ACT_TURN " + _mfj({"sg": sg_label, **res}), flush=True)
        if res["stage"] not in ("done", "turn_unreachable"):
            print("ACT_STOP " + json.dumps({"sg": sg_label, "k": k, "why": f"motion_{res['stage']}_{res.get(res['stage'])}", "steps_used": steps}), flush=True); return rest_mode   # C17I4
        if st is not None:
            st_prev = st
        last = res; k += 1
        for _ in range(3):
            get_observation()
# ===== end C11I2 manipulation_primitive =====


# ===== C5I1 manipulation_primitive: open_articulated (round 20260924, chunk 5, iteration 1) =====
# Opening a sliding part by its handle: the handle is taken between the open pads with the tool axis horizontal into the
# part's face (the finger-opening axis vertical, so the pads close on the bar's top and bottom) and pulled straight out along the face
# normal until the part's stop. Public data only: the segmenter's masks of the adapter's part / support prompts (the C2I1 assembly and
# body), the depth frame, the public IK, the interface's motion contracts. A closed sliding part does not protrude beyond the body outline
# (its front lies in the body's face), so the part is found by its handle: the body's faces are measured on the whole raised frame (face
# plane = the median offset of the points within the outline band of that face), the handles of a face are the compact clusters of
# points standing CA_CLEAR_MARGIN .. CA_HANDLE_MAX in front of it inside its lateral extent (the C2I1 handle rule: no continuation
# outside that zone), the face is the one that carries handles, and the side code selects the handle (the C2I1 part selection rule).
# The grasp: fingertips CA_PUSH_GAP in front of the face (pads overlapping the handle's depth by >= GW_ENGAGE_MIN), TCP height in the
# band where both open pads straddle the handle by CA_CLEAR_MARGIN; the open hand (AC open boxes) clears every other visible point at
# the grasp, along the horizontal approach from the standoff and down the vertical column; the closed hand clears them along the pull;
# the public IK solves the high, standoff, grasp and pull-end poses; candidates (height x yaw flip) are ranked by the C2I3 route cost.
# C5I3: the hand model (CA_HAND, the AC open boxes) is written with x = the finger-opening axis, but the robot opens its fingers along
# the public tool y (finger 1, the one the public opening reads, on +y): the commanded frame is the model frame rotated -90 deg about
# the tool axis (OP_TOOL_FROM_MODEL), the model frame keeps serving every clearance check. The two fingers close independently, each at
# most to the TCP plane: off the handle's height centre only the finger on the centre's side meets the handle, so at equal height the
# flip with finger 1 toward the handle's centre is tried first.
# C5I4: the approach contact move reports "reached" anywhere inside its goal ball (tol_pos = PRIM reach_tol), i.e. up to reach_tol short
# of its commanded point, which can consume the pads' whole engagement over the handle. It is commanded op_overreach(engage) beyond the
# grasp point (tol_pos passed explicitly), so every reached arrival keeps at least OP_ENGAGE_KEEP of the planned engagement; the plan's
# approach sweep covers the overreach with the same open-hand model and margin.
# The pull: contact_move along the face normal (holding expected), commanded over the body's depth along it (a sliding part cannot travel
# farther than the body is deep) with the whole travel as the contact window: the part's stop ends it. Verified by re-observation: the
# handle's front moved out with the hand. Constants: hand geometry (robot), the stack's xy sigma / margins; DIAGNOSIS.md section 4.
OP_OPEN_HALF = GW_G["max_opening"] / 2      # m, the open pads' inner faces lie this far from the TCP along the opening axis (8 cm opening)
OP_Z_STEP = 0.005                            # m, TCP height candidates across the straddle band (half the xy sigma)
OP_TIP_GAP = CA_PUSH_GAP                     # m, fingertips this far in front of the measured face at the grasp (2 x xy sigma)
OP_STANDOFF = CA_PUSH_GAP                    # m, fingertips this far in front of the handle's front at the standoff pose
OP_SWEEP_STEP = CA_CLEAR_MARGIN              # m, spacing of the clearance checks along the horizontal approach and the pull (<= the margin)
OP_PULL_MAX = 0.45                           # m, the contact primitive's travel bound as used by the C2I1 push (L <= 0.45)
OP_APPROACH_MIN = PRIM["reach_tol"] / np.tan(np.radians(15.0 / 2))   # m, 0.076: the approach contact move starts where the descent arrived (within reach_tol); its direction to the grasp must stay within the contact primitive's approach-axis tolerance (15 deg), so the approach is at least reach_tol / tan(7.5 deg) long
OP_TOOL_FROM_MODEL = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # C5I3: commanded tool frame = model frame @ this (-90 deg about the tool axis): tool y (the Panda finger joints' slide axis; finger 1 on +y) = the model's finger-opening axis x, tool z = the model's z
OP_ENGAGE_KEEP = 0.5                         # C5I4: fraction of the planned pad engagement over the handle a reached approach must keep (the stack's pad convention: GW_ENGAGE_MIN = pad_engage / 2)
LA_OPEN_RELATIONS = ("open_articulated",)    # relations run by run_open_articulated: a part (or a reference) and no object


def op_hand_frame(u, flip=False):
    """Rotation of the hand MODEL (columns = model axes, the frame of CA_HAND and the AC open boxes: x = finger opening, vertical: up, or
    down when flipped; z horizontal along u, into the face; y = z x x, horizontal along the face) and the wxyz quaternion of the COMMANDED
    tool frame (C5I3): the model frame @ OP_TOOL_FROM_MODEL, whose y (the robot's finger-opening axis, finger 1 on +y) is the model's x."""
    u = np.array([float(u[0]), float(u[1]), 0.0]); u /= np.linalg.norm(u)
    xt = np.array([0.0, 0.0, -1.0 if flip else 1.0]); yt = np.cross(u, xt); Rm = np.stack([xt, yt, u], 1)
    q = Rotation.from_matrix(Rm @ OP_TOOL_FROM_MODEL).as_quat(); return Rm, np.array([q[3], q[0], q[1], q[2]])


def op_overreach(engage):
    """m (C5I4): how far beyond the grasp point the approach contact move is commanded. A reached contact move ends anywhere within its
    goal ball (tol_pos = PRIM reach_tol) of the commanded point, never beyond it (the last waypoint); commanding reach_tol - OP_ENGAGE_KEEP x
    engage beyond the grasp point keeps every reached arrival within OP_ENGAGE_KEEP x engage short of it, i.e. the pads keep at least
    (1 - OP_ENGAGE_KEEP) of the planned engagement over the handle. 0 when the engagement already covers the ball."""
    return max(0.0, float(PRIM["reach_tol"]) - OP_ENGAGE_KEEP * float(engage))


def op_box_low_z(Rm, boxes):
    """Lowest corner of the boxes (tool coordinates) relative to the TCP for the rotation Rm."""
    return min(float((Rm @ np.array([xv, yv, zv]))[2]) for (x0, x1), (y0, y1), (z0, z1) in boxes for xv in (x0, x1) for yv in (y0, y1) for zv in (z0, z1))


def op_faces(body, F, floor_z):
    """The body's four vertical faces measured on the raised frame points F: outward normal, lateral axis, lateral half extent, and the
    face plane offset (the median offset from the body centre of the points within CA_OUTLINE_MARGIN of that outline edge, inside its
    lateral extent and below the body top); None for a face with fewer than CA_PART_MIN_PTS such points (not seen)."""
    c = np.asarray(body["centre"], float); a, b = body["axes"]; hu, hv = body["half"]; out = []
    for n_out, half_n, half_t in ((a, hu, hv), (-a, hu, hv), (b, hv, hu), (-b, hv, hu)):
        n_out = np.asarray(n_out, float); w = np.array([-n_out[1], n_out[0]])
        s = (F[:, :2] - c) @ n_out; t = (F[:, :2] - c) @ w
        band = (np.abs(s - half_n) <= CA_OUTLINE_MARGIN) & (np.abs(t) <= half_t) & (F[:, 2] <= body["z_top"])
        rec = {"n_out": n_out, "w": w, "half_n": float(half_n), "half_t": float(half_t), "n_face": int(band.sum()), "s_face": None}
        if int(band.sum()) >= CA_PART_MIN_PTS:
            rec["s_face"] = float(np.median(s[band]))
        out.append(rec)
    return out


def op_handles(face, body, F):
    """Handles of a measured face: clusters (CA_CLUSTER_CELL, 26-connectivity, >= CA_PART_MIN_PTS // 4 points) of the frame points
    standing CA_CLEAR_MARGIN .. CA_HANDLE_MAX in front of the face plane inside its lateral extent and below the body top, with no
    point of the frame in front of the face but outside that zone within CA_CELL (xy) of them (the C2I1 handle rule: a body that
    continues beyond the zone is not a handle). Each: points, centroid, the front (p98 offset from the face), offset / lateral / height ranges."""
    c = np.asarray(body["centre"], float); n_out = face["n_out"]; w = face["w"]
    s = (F[:, :2] - c) @ n_out - face["s_face"]; t = (F[:, :2] - c) @ w
    front = s > CA_CLEAR_MARGIN
    zone = front & (s <= CA_HANDLE_MAX) & (np.abs(t) <= face["half_t"]) & (F[:, 2] <= body["z_top"])
    others = F[front & ~zone]; out = []
    if int(zone.sum()) < CA_PART_MIN_PTS // 4:
        return out
    Z = F[zone]; sZ = s[zone]; tZ = t[zone]
    g = np.floor((Z - Z.min(0)) / CA_CLUSTER_CELL).astype(int); occ = np.zeros(g.max(0) + 1, bool); occ[g[:, 0], g[:, 1], g[:, 2]] = True
    lab, n = _ndi.label(occ, structure=np.ones((3, 3, 3), bool)); L = lab[g[:, 0], g[:, 1], g[:, 2]]
    otree = _KDTree(others[:, :2]) if len(others) else None
    for k in range(1, n + 1):
        sel = L == k
        if int(sel.sum()) < CA_PART_MIN_PTS // 4:
            continue
        if otree is not None:
            dn, _ = otree.query(Z[sel][:, :2], distance_upper_bound=CA_CELL)
            if int(np.isfinite(dn).sum()):
                continue
        Q = Z[sel]; sQ = sZ[sel]; tQ = tZ[sel]
        out.append({"pts": Q, "n": int(sel.sum()), "centroid": Q.mean(0), "protrusion": float(np.percentile(sQ, CONTAINER_PCT_HI)), "s_lo": float(np.percentile(sQ, CONTAINER_PCT_LO)),
                    "t": (float(np.percentile(tQ, CONTAINER_PCT_LO)), float(np.percentile(tQ, CONTAINER_PCT_HI))), "z": (float(np.percentile(Q[:, 2], CONTAINER_PCT_LO)), float(np.percentile(Q[:, 2], CONTAINER_PCT_HI)))})
    return out


def op_sweep_clear(obst, tcp0, d, length, Rm, clear_fn):
    """The hand (clear_fn: ca_clearance = closed, ac_clearance = open) at tcp0 + d x s for s in [0, length] every OP_SWEEP_STEP is clear.
    Returns (ok, n, nearest, s)."""
    n_st = max(1, int(np.ceil(float(length) / OP_SWEEP_STEP)))
    for i in range(n_st + 1):
        sv = float(length) * i / n_st; ok, nc, near = clear_fn(obst, np.asarray(tcp0, float) + np.asarray(d, float) * sv, Rm)
        if not ok:
            return False, nc, near, round(sv, 3)
    return True, 0, None, None


def op_grasp_plan(face, handle, body, floor_z, obst, L, ik_solve=None, j_now=None, ik_probe=None):
    """Grasp-and-pull plan of a handle (candidates: TCP height across the straddle band x yaw flip), ranked by ca_route_cost when the
    public solutions and the actual joints are given. Returns (plan with alternatives, info) or (None, info)."""
    c = np.asarray(body["centre"], float); n_out = face["n_out"]; w = face["w"]; u = -n_out
    s_tip = face["s_face"] + OP_TIP_GAP; engage = handle["protrusion"] - OP_TIP_GAP
    t_mid = 0.5 * (handle["t"][0] + handle["t"][1]); z_lo, z_hi = handle["z"]
    z_min = z_hi + CA_CLEAR_MARGIN - OP_OPEN_HALF; z_max = z_lo - CA_CLEAR_MARGIN + OP_OPEN_HALF
    info = {"s_face": face["s_face"], "front": handle["protrusion"], "engage": float(engage), "t_mid": float(t_mid), "z_band": [float(z_min), float(z_max)], "L": float(L)}
    if engage < GW_ENGAGE_MIN:
        return None, dict(info, why="handle_too_shallow_for_the_pads")
    if z_min > z_max:
        return None, dict(info, why="handle_too_tall_for_the_opening")
    z_mid = 0.5 * (z_min + z_max); nz = int(np.floor((z_max - z_min) / 2 / OP_Z_STEP + 1e-9))
    zs = [z_mid] + [z_mid + sg * k * OP_Z_STEP for k in range(1, nz + 1) for sg in (1.0, -1.0)]
    xy = c + n_out * (s_tip + CA_HAND["tip"]) + w * t_mid; n3 = np.array([n_out[0], n_out[1], 0.0]); u3 = -n3
    back = max(engage + OP_STANDOFF, OP_APPROACH_MIN); ranked = ik_solve is not None and j_now is not None; tried = []; admissible = []
    over = op_overreach(engage)   # C5I4: the approach is commanded this far beyond the grasp point (the goal ball's far edge keeps half the engagement)
    open_boxes = _ac_open_boxes(); closed_boxes = _ca_hand_boxes()
    for z in zs:
        for flip in (False, True):
            Rm, q = op_hand_frame(u, flip); tcp = np.array([xy[0], xy[1], z]); stand = tcp + n3 * back
            rec = {"z": round(float(z), 4), "flip": flip}
            low = min(op_box_low_z(Rm, open_boxes), op_box_low_z(Rm, closed_boxes))
            if z + low < float(floor_z) + CA_FLOOR_CLEAR:
                rec["why"] = "hand_below_floor_clearance"; tried.append(rec); continue
            ok1, n1, near1, s1 = op_sweep_clear(obst, stand, u3, back + over, Rm, ac_clearance)   # C5I4: up to the commanded approach end
            if not ok1:
                rec.update(why="approach_obstructed", n=n1, nearest=near1, at=s1); tried.append(rec); continue
            z_high = float(body["z_top"]) + CA_HIGH_CLEAR - op_box_low_z(Rm, open_boxes)
            ok2 = True
            for zc in np.arange(z_high, float(stand[2]) - 1e-9, -CA_DESCENT_STEP).tolist():
                okz, nzc, nearz = ac_clearance(obst, [stand[0], stand[1], zc], Rm)
                if not okz:
                    ok2 = False; rec.update(why="descent_column_obstructed", n=nzc, nearest=nearz, z_col=round(float(zc), 3)); break
            if not ok2:
                tried.append(rec); continue
            ok3, n3c, near3, s3 = op_sweep_clear(obst, tcp, n3, L, Rm, ca_clearance)
            if not ok3:
                rec.update(why="pull_obstructed", n=n3c, nearest=near3, at=s3); tried.append(rec); continue
            high = np.array([stand[0], stand[1], z_high]); end = tcp + n3 * L
            if ranked:
                js = [ik_solve(p, q) for p in (high, stand, tcp, end)]; oks = [j is not None for j in js]
            else:
                js = [None] * 4; oks = [bool(ik_probe(p, q)) for p in (high, stand, tcp, end)] if ik_probe is not None else [True] * 4
            rec["ik"] = oks
            if not all(oks):
                rec["why"] = "unreachable"; tried.append(rec); continue
            cost = ca_route_cost(j_now, js[0], js[1]) if ranked else None
            rec["why"] = "ok"; rec["cost"] = None if cost is None else round(cost, 3); tried.append(rec)
            admissible.append({"u": u3[:2], "n_out": n_out, "tcp": tcp, "standoff": stand, "z_high": float(z_high), "end": end, "quat": q, "R": Rm, "flip": flip, "z": float(z), "L": float(L), "back": float(back), "over": float(over), "cost": cost, **info})
    if not admissible:
        return None, dict(info, why="grasp_blocked_or_unreachable", tried=tried)
    admissible.sort(key=lambda p: (int(round(abs(p["z"] - z_mid) / OP_Z_STEP)), int(float(p["R"][2, 0]) * (z_mid - p["z"]) < 0.0), p["cost"] if ranked else 0.0))   # the height nearest the band centre (pads straddling the handle symmetrically), then (C5I3) finger 1 (model +x) on the side of the handle's height centre (z_mid), then the C2I3 route cost
    return dict(admissible[0], tried=tried, alternatives=admissible[1:], n_admissible=len(admissible)), info


def op_front(F, body, face, handle, reach):
    """Front of the part at the handle: p98 offset from the face plane of the frame points within the handle's lateral range and height
    band (each grown by CA_CLEAR_MARGIN) standing CA_CLEAR_MARGIN .. reach in front of the face (None when fewer than CA_PART_MIN_PTS // 4)."""
    c = np.asarray(body["centre"], float); s = (F[:, :2] - c) @ face["n_out"] - face["s_face"]; t = (F[:, :2] - c) @ face["w"]
    m = (s > CA_CLEAR_MARGIN) & (s <= reach) & (t >= handle["t"][0] - CA_CLEAR_MARGIN) & (t <= handle["t"][1] + CA_CLEAR_MARGIN) & (F[:, 2] >= handle["z"][0] - CA_CLEAR_MARGIN) & (F[:, 2] <= handle["z"][1] + CA_CLEAR_MARGIN)
    return float(np.percentile(s[m], CONTAINER_PCT_HI)) if int(m.sum()) >= CA_PART_MIN_PTS // 4 else None


def op_measure(rgb, depth_img, K, E, prompts, side, floor_z, tcp_xy, ik_solve=None, j_now=None, ik_probe=None):
    """One measurement: assembly, body, faces, handles, the handle named by the side code, the obstacle set and the grasp-and-pull plan.
    Returns a record (printed as OPEN_MEASURE by the caller); rec["plan"] None when no plan."""
    rec = {"side": side, "assembly": None, "body": None, "faces": None, "handles": None, "selected": None, "plan": None, "plan_info": None, "why": None}
    pts, ainfo = ca_assembly(rgb, depth_img, K, E, prompts, floor_z); rec["assembly"] = ainfo
    if pts is None:
        rec["why"] = "assembly_not_found"; return rec
    body = ca_body(pts)
    if body is None:
        rec["why"] = "no_body_top"; return rec
    rec["body"] = {"centre": np.round(body["centre"], 4).tolist(), "axis_deg": round(float(np.degrees(np.arctan2(body["axes"][0][1], body["axes"][0][0])) % 180.0), 1), "half": [round(float(h), 4) for h in body["half"]], "z_top": round(float(body["z_top"]), 4)}
    F = ca_frame_raised(depth_img, K, E, floor_z, tcp_xy)
    faces = op_faces(body, F, floor_z); per = []
    for f in faces:
        f["handles"] = op_handles(f, body, F) if f["s_face"] is not None else []
        per.append({"n_out": np.round(f["n_out"], 3).tolist(), "n_face": f["n_face"], "s_face": None if f["s_face"] is None else round(f["s_face"], 4), "n_handles": len(f["handles"])})
    rec["faces"] = per
    cnt = sorted((len(f["handles"]) for f in faces), reverse=True)
    if cnt[0] == 0:
        rec["why"] = "no_handle"; return rec
    if len(cnt) > 1 and cnt[1] == cnt[0]:
        rec["why"] = "handle_face_ambiguous"; return rec
    face = max(faces, key=lambda f: len(f["handles"]))
    rec["handles"] = [{"n": h["n"], "centroid": np.round(h["centroid"], 4).tolist(), "front": round(h["protrusion"], 4), "t": np.round(h["t"], 4).tolist(), "z": np.round(h["z"], 4).tolist()} for h in face["handles"]]
    if side is None and len(face["handles"]) > 1:
        rec["why"] = "handle_ambiguous_without_side"; return rec
    h = ca_select_part(face["handles"], side)
    if h is None:
        rec["why"] = "no_handle_for_the_side"; return rec
    rec["selected"] = {"n": h["n"], "centroid": np.round(h["centroid"], 4).tolist(), "front": round(h["protrusion"], 4), "t": np.round(h["t"], 4).tolist(), "z": np.round(h["z"], 4).tolist()}
    obst = F[~ca_near(F, h["pts"], CA_SAME_POINT)]
    L = float(min(2.0 * face["half_n"], OP_PULL_MAX))
    plan, pinfo = op_grasp_plan(face, h, body, floor_z, obst, L, ik_solve=ik_solve, j_now=j_now, ik_probe=ik_probe)
    rec["plan_info"] = {kk: (np.round(v, 4).tolist() if hasattr(v, "tolist") else v) for kk, v in pinfo.items()}; rec["plan"] = plan
    rec["_face"] = face; rec["_handle"] = h; rec["_body"] = body
    if plan is None:
        rec["why"] = pinfo.get("why")
    return rec


def op_plan_summary(plan):
    return {"tcp": np.round(plan["tcp"], 4).tolist(), "standoff": np.round(plan["standoff"], 4).tolist(), "z_high": round(plan["z_high"], 4), "end": np.round(plan["end"], 4).tolist(), "quat": np.round(plan["quat"], 4).tolist(),
            "flip": plan["flip"], "L": round(plan["L"], 4), "back": round(plan["back"], 4), "engage": round(plan["engage"], 4), "z_band": [round(v, 4) for v in plan["z_band"]], "cost": None if plan.get("cost") is None else round(float(plan["cost"]), 3),
            "n_tried": len(plan["tried"]), "n_admissible": plan.get("n_admissible"), "alternatives": [{"z": round(a["z"], 4), "flip": a["flip"], "cost": None if a.get("cost") is None else round(float(a["cost"]), 3)} for a in (plan.get("alternatives") or [])]}


def op_execute(plan, sg, k):
    """One grasp-and-pull cycle through the existing contracts. Candidates (the plan and its alternatives, at most CA_PROBE_MAX) are
    admitted by the interface's own planner from the actual joints (the C2I3 probes: approach budget from the plan, linear moves probed
    plan_only). Then: open, free move to the high pose, linear descent to the standoff, contact approach along the tool axis (open
    gripper expected), gated close + grasp verification, gated pull (contact_move along the face normal, holding expected, the whole
    travel as the contact window), hold check, open, linear back-off along the face normal, ascent, home. Returns a record."""
    rec = {"k": k, "stage": "above", "steps": 0, "probes": [], "candidate": None, "grasp": None, "pull": None, "hold": None}
    cands = ([plan] + list(plan.get("alternatives") or []))[:CA_PROBE_MAX]

    def used(r):
        return int((r or {}).get("steps_used") or 0) if isinstance(r, dict) else 0

    def home():
        h = goto_home_joint_position(); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(h); rec["home"] = h.get("status") if isinstance(h, dict) else "unreported"

    def lin(target, q, label, key):
        r = move_to_pose(np.asarray(target, float), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=label); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r); rec[key] = r.get("status"); return r

    def leave(q, c, n3):   # open, back off along the face normal, ascend to the high pose, home
        open_gripper(); gate_record("open", role="recovery_open" if rec["stage"] != "pull" else "tool_config")
        p, _, _ = read_tcp(); lin(p + n3 * float(c["back"]), q, f"{sg}:open{k}_backoff", "backoff")
        p, _, _ = read_tcp(); lin(np.array([p[0], p[1], float(c["z_high"])]), q, f"{sg}:open{k}_ascend", "ascend"); home()

    open_gripper(); _MF["last_plan"] = None; _WG["pts"] = None; _WG["quat"] = None   # the width gate belongs to object grasps; a fixed part is not one
    for i, c in enumerate(cands):
        q = np.asarray(c["quat"], float); stand = np.asarray(c["standoff"], float); tcp = np.asarray(c["tcp"], float); high = np.array([stand[0], stand[1], float(c["z_high"])])
        n3 = np.array([c["n_out"][0], c["n_out"][1], 0.0]); pr = {"i": i, "z": round(c["z"], 4), "flip": c["flip"], "cost": None if c.get("cost") is None else round(float(c["cost"]), 3)}
        budget, ap = ca_approach_budget(high, q, f"{sg}:open{k}_above_plan{i}"); pr["above_plan"] = dict(ap, budget=budget)
        if budget is None:
            pr["why"] = "approach_plan_refused"; rec["probes"].append(pr); continue
        rec["stage"] = "above"
        r0 = move_to_pose(high, q, mode="free", tol_pos=0.015, step_budget=budget, label=f"{sg}:open{k}_above"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r0); rec["above"] = r0.get("status")
        if r0.get("status") != "reached":
            pr["why"] = f"above_{r0.get('status')}"; rec["probes"].append(pr); home(); return rec
        ok1, dp = ca_probe_linear(stand, q, SEGMENT_CORRIDOR, BUDGET_LINEAR, f"{sg}:open{k}_descend_plan{i}"); pr["descend_plan"] = dp
        if not ok1:
            pr["why"] = "descent_plan_refused"; rec["probes"].append(pr); continue   # no motion: the next candidate's approach starts from this high pose
        rec["stage"] = "descend"
        r1 = move_to_pose(stand, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:open{k}_descend"); _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r1); rec["descend"] = r1.get("status")
        if r1.get("status") != "reached":
            pr["why"] = f"descend_{r1.get('status')}"; rec["probes"].append(pr); lin(high, q, f"{sg}:open{k}_ascend", "ascend"); home(); return rec
        pr["why"] = "ok"; rec["probes"].append(pr); rec["candidate"] = {"i": i, "z": round(c["z"], 4), "flip": c["flip"], "budget": budget}
        rec["stage"] = "approach"
        ra = contact_move(tcp - n3 * float(c.get("over", 0.0)), q, approach_axis=(-n3).tolist(), contact_tol_before=GRIPPER_CONTACT_TOL, overshoot_after=0.0, start_pose=stand, start_tol=0.02, gripper_expect="open",
                          tol_pos=PRIM["reach_tol"], lateral_corridor=APPROACH_CORRIDOR, step_budget=BUDGET_CONTACT, label=f"{sg}:open{k}_approach")   # C5I4: commanded the overreach beyond the grasp point, the contract's goal tolerance explicit
        _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(ra); rec["approach"] = ra.get("status")
        print("CONTACT_MOVE " + json.dumps({"label": f"{sg}:open{k}_approach", "status": ra["status"], "detail": ra.get("detail"), "success": ra["success"], "steps": ra["steps_used"], "budget": ra["budget"],
                                            "along_cm": round((ra.get("along_final") or 0) * 100, 2), "travel_cm": round((ra.get("travel") or 0) * 100, 2), "pos_err_cm": round(ra.get("pos_err", 9) * 100, 2), "verifier": verify_contact(ra).get("status")}), flush=True)
        ok_close, why = gate_allow(ra, "close"); rec["close_allowed"] = ok_close
        if not ok_close:
            print(f"OPEN_ABORT reason={why}", flush=True); leave(q, c, n3); return rec
        rec["stage"] = "grasp"
        gv = mc_close_and_verify(ra, sg); rec["grasp"] = gv.get("status")
        if gv.get("status") != "verified":
            print(f"OPEN_ABORT reason=grasp_not_verified: {gv.get('reason')}", flush=True); leave(q, c, n3); return rec
        ok_t, why_t = gate_allow(gv, "transport"); rec["transport_allowed"] = ok_t
        if not ok_t:
            print(f"OPEN_ABORT reason=pull_refused: {why_t}", flush=True); leave(q, c, n3); return rec
        rec["stage"] = "pull"
        p_now, _, _ = read_tcp(); L = float(c["L"]); gate_record("transport"); _mark(ev="pull_start", sg=sg, k=k, tcp=p_now, n=n3, L=L)
        r2 = contact_move(p_now + n3 * L, q, approach_axis=n3.tolist(), contact_tol_before=L, overshoot_after=0.0, start_pose=p_now, start_tol=0.02, gripper_expect="holding",
                          lateral_corridor=PUSH_CORRIDOR, step_budget=BUDGET_PUSH, label=f"{sg}:open{k}_pull")
        _PRIM_STATS["motion_calls"] += 1; rec["steps"] += used(r2)
        print("CONTACT_MOVE " + json.dumps({"label": f"{sg}:open{k}_pull", "status": r2["status"], "detail": r2.get("detail"), "success": r2["success"], "steps": r2["steps_used"], "budget": r2["budget"], "wall_s": r2.get("wall_s"),
                                            "along_cm": round((r2.get("along_final") or 0) * 100, 2), "travel_cm": round((r2.get("travel") or 0) * 100, 2), "pos_err_cm": round(r2.get("pos_err", 9) * 100, 2), "evidence": r2.get("evidence"), "verifier": verify_contact(r2).get("status")}), flush=True)
        rec["pull"] = {"status": r2["status"], "detail": r2.get("detail"), "along": None if r2.get("along_final") is None else float(r2["along_final"]), "travel": L, "steps": r2["steps_used"]}
        hv = verify_hold(gv["after"], gripper_opening()); rec["hold"] = hv.get("status")
        p_end, _, _ = read_tcp(); _mark(ev="pull_end", sg=sg, k=k, status=r2["status"], tcp=p_end, along=rec["pull"]["along"], hold=rec["hold"])
        leave(q, c, n3); rec["stage"] = "done"
        return rec
    rec["stage"] = "probe"; rec["probe"] = "no_admissible_candidate"; home()
    return rec


def run_open_articulated(sg, td, sg_label="sg1"):
    """Driver of the open_articulated relation: measure -> one grasp-and-pull cycle -> re-observe. Returns True when the part's front
    moved out by at least CA_PROGRESS_MIN with the pull (claim opened), False otherwise, None when no part / plan was found."""
    sup = la_phrase_in(td, "supports", sg.get("clause") or "")
    prompts = list(sg.get("part_prompts") or []) + (list((td.get("supports") or {}).get(sup) or []) if sup else []) + list(sg.get("support_prompts") or [])
    prompts = list(dict.fromkeys(prompts)); side = sg.get("side")
    _E1_LAST.update({"stop": False, "grasp_after": None, "hold": None})
    obs = get_observation(); fr = mf_frame(obs); p_tcp, _ = tcp_from_obs(obs)
    floor_z = support_floor_z(fr["depth"], fr["K"], fr["E"]); _CG["floor_z"] = floor_z
    if floor_z is None:
        print("BINDING_STOP " + json.dumps({"status": "no_floor_estimate", "relation": sg["relation"]}), flush=True); return None
    rec = op_measure(fr["rgb"], fr["depth"], fr["K"], fr["E"], prompts, side, floor_z, np.asarray(p_tcp, float)[:2], ik_solve=ca_ik_joints, j_now=np.asarray(obs["robot_joint_pos"], float).reshape(-1)[:7], ik_probe=mf_ik_probe)
    print("OPEN_MEASURE " + _mfj({kk: v for kk, v in rec.items() if kk not in ("plan",) and not kk.startswith("_")}), flush=True)
    if rec["selected"] is None:
        print("BINDING_STOP " + json.dumps({"status": "part_not_found", "relation": sg["relation"], "side": side, "why": rec["why"], "prompts": prompts}), flush=True); return None
    plan = rec["plan"]
    if plan is None:
        print("OPEN_STOP " + _mfj({"sg": sg_label, "k": 0, "why": rec["why"], "plan_info": rec["plan_info"]}), flush=True); return None
    print("OPEN_PLAN " + _mfj({"sg": sg_label, "k": 0, **op_plan_summary(plan)}), flush=True)
    _mark(ev="open_plan", sg=sg_label, k=0, tcp=plan["tcp"], L=plan["L"])
    res = op_execute(plan, sg_label, 0)
    print("OPEN_PULL " + _mfj({"sg": sg_label, **res}), flush=True)
    if res["stage"] != "done":
        print("OPEN_STOP " + json.dumps({"sg": sg_label, "k": 0, "why": f"motion_{res['stage']}_{res.get(res['stage'])}", "steps_used": res["steps"]}), flush=True); return False
    for _ in range(3):
        get_observation()
    obs = get_observation(); fr = mf_frame(obs); p_tcp, _ = tcp_from_obs(obs)
    F = ca_frame_raised(fr["depth"], fr["K"], fr["E"], floor_z, np.asarray(p_tcp, float)[:2])
    front = op_front(F, rec["_body"], rec["_face"], rec["_handle"], float(plan["L"]) + CA_HANDLE_MAX)
    along = (res.get("pull") or {}).get("along"); growth = None if front is None else float(front) - float(rec["_handle"]["protrusion"])
    opened = growth is not None and along is not None and growth >= CA_PROGRESS_MIN and float(along) >= CA_PROGRESS_MIN
    print("OPEN_VERIFY " + _mfj({"sg": sg_label, "k": 0, "claim": "opened" if opened else "not_opened", "front_before": rec["_handle"]["protrusion"], "front_after": front, "growth": growth, "pull": res.get("pull"), "hold": res.get("hold"), "steps_used": res["steps"]}), flush=True)
    _mark(ev="open_verify", sg=sg_label, claim="opened" if opened else "not_opened", growth=growth)
    return bool(opened)
# ===== end C5I1 manipulation_primitive =====


# ===== C15I3 place_primitive: in_layer, horizontal insertion into a covered layer (N15 group C, slot C15I3; C15I5 release anchor; N16 group E, slot C16I4 carry settling + measured approach) =====
# Relation in_layer = put the object into a horizontal layer bounded below by a floor (the workspace floor or a board) and above by
# a cover (a board or the assembly's top), entered sideways through an open side of the assembly. Public depth + SAM3 only.
# 1. Measure (lp_measure, observe frame): the assembly (ca_assembly on the reference prompts) and its body outline (ca_body: the
#    largest top face, its minimum rectangle). Each outline side that faces the camera is inspected in a band LP_BAND deep behind
#    it (corners LP_CORNER excluded): the band's heights binned at CONTAINER_PLANE_BIN; a bin with >= LP_LEVEL_MIN_PTS points is a
#    structure level (floor, board, top). A closed side is a panel (every bin occupied); an open side shows levels with free runs
#    between them. The opening = the facing side with the tallest free run. Each free run is a layer: floor z_L = median height
#    of the lower level's band points (not below its run's top bin: a panel is not a floor); cover z_C = the carved underside: the height, at the cover's front edge, of the camera rays
#    to the lower level's points visible under the cover (a ray that reached them passed below the cover's underside; p99 of the
#    rays below the cover's visible top), capped by the cover's lowest visible point. A layer whose floor is not visible under its
#    cover is not measured (fail closed). Lateral opening = the widest gap between the structure points (posts, panels) seen at
#    heights inside the run; face = the outermost structure point of the band (p2 of the depth coordinate); depth = outline edge
#    to edge minus the face offset.
#    Side code: zmin = lowest layer, zmax = highest layer, None = the tallest layer (LA: "under" -> zmin, "on" -> zmax).
# 2. Before motion (lp_prefit): the object's standing height (p98 - p2 of its visible points) + 2 LP_MARGIN must fit the chosen
#    layer's free height, else LAYER_STOP (no motion). LP_MARGIN = CA_CLEAR_MARGIN (the program's hand / surface clearance).
# 3. After the verified lift (lp_plan): the held body is measured twice (measure_held_object, as P3); its in-hand offset is the
#    protrusion pinch's held body centre (gh_held_offset) when it applies, else the mean visible-axis offset. The hand is yawed about
#    the vertical so that this offset points into the opening (the body leads, the hand trails). The held points (gripper frame,
#    re-expressed at that orientation) give the body's extents along the insertion axis u and the lateral axis w, its top above the
#    TCP and its hang drop_used below it. TCP depth t along u from the face: the body centre at mid-depth, but the hand's boxes
#    (CA_HAND, all corners) stay LP_MARGIN outside the face and the body's far end LP_MARGIN before the back; the body centre must be
#    inside the face. Lateral: the body centred on the opening, LP_MARGIN inside its posts. Heights: TCP in [z_L + drop + LP_MARGIN,
#    z_C - LP_MARGIN - top of the held points that are inside the face]; the insertion height is the middle of that interval (at
#    least LP_MIN_TRAVEL above its bottom, the contact primitive's minimum travel); the lowest hand corner stays CA_FLOOR_CLEAR over
#    the floor. Swept checks against the observe frame's raised points (the bound object's points, the outline + LP_MARGIN, and the
#    arm strip removed): the vertical column at the pre-insert pose (body + hand from the transport height down) and the horizontal
#    sweep (pre-insert -> final), each with LP_MARGIN. The hand boxes are placed in the model frame whose x is the pad closing axis
#    (the commanded tool y, as gh_handle_plan / gw_hand_R use them).
# 4. Motion (lp_place): transport to the pre-insert pose above the opening's front (c_transport_height corridor rule), vertical
#    lowering (linear), horizontal insertion along u (linear, SEGMENT_CORRIDOR), the contact primitive straight down from the ACTUAL
#    TCP after the insertion (C15I5 lp_release_anchor, within the plan's clearances) to the release height
#    (mc_place_descend: z_L + drop + LP_MARGIN; contact within PLACE_CONTACT_TOL accepted), release through the gate, horizontal
#    retreat along -u by CA_BACKOFF, then safe_retreat (vertical rise, home). Any refusal stops with the object held and
#    safe_retreat. Reachable only from an in_layer subgoal (run_subgoal's first line).
# C16I4 (N16 group E): robustness across layouts.
# 5. Carry settling (lp_settle_carry): a body pinched across a protrusion pivots in the pads about the closing axis under the
#    transport's accelerations (recorded, truth for diagnosis only: 5.9 deg at the lift-pose measurement, 11-17 deg after the
#    transport), so the hang measured at the lift no longer bounds the body when it enters the layer. At the top of the pre-insert
#    column and again at the pre-insert pose (the robot static) the held body is measured twice; its pitch about the closing axis
#    relative to its resting attitude (the bound object's points before the grasp, in the carry frame of the grasp: e = horizontal
#    normal of the closing axis, c = closing axis, z up) is the rotation that best maps its top profile (CONTAINER_PCT_HI height per
#    LP_SLICE bin along e, vertical offset free) onto the resting profile (lp_pitch). Beyond the body's pitch tolerance (the angle that
#    moves its farthest measured point by LP_MARGIN) the hand turns about the closing axis through the TCP to bring the body back
#    (tool within LP_TILT_MAX of the vertical = GT_TILTS_DEG's bounded jaw-axis tilt; at most LP_LEVEL_TRIES turns, each re-measured),
#    and the insertion is re-planned from the latest measurement (its hang, extents and top; the in-hand offset of the lift turned by
#    the closing axis's yaw change). No measurement before a turn keeps the current plan; after a turn it stops with the object held.
# 6. Measured approach (lp_plan): the swept checks use the held body's footprint along u (lp_body_slices: per LP_SLICE slice the
#    lateral extent of the held points inside the measured hang, hidden slices bridged by their neighbours, a measured ring's chord
#    added; every slice keeps the full height [-hang, top]) instead of one bounding box, and when the centred mid-height approach is
#    blocked the approach column / insertion height is chosen among the measured alternatives: the lateral position inside the
#    opening's slack (body LP_MARGIN inside the posts) and the TCP height inside [z_lo, z_hi], in LP_MARGIN steps, nearest first.
#    No guard, margin, clearance, tolerance or budget is changed.
LP_RELATIONS = ("in_layer",)
LP_SIDES = (None, "zmin", "zmax")
LP_WORDS = {"under": "zmin", "on": "zmax"}   # the preposition of an in_layer clause (the template captures it as a word): below the cover of the lowest layer / on a cover = the highest layer
LP_BAND = 0.06            # m, depth of the inspected band behind a side: the lower level is seen up to 3.3-5.4 cm behind the cover's edge in the recorded frames (DIAGNOSIS 2)
LP_CORNER = 0.04          # m, band corners excluded from the level histogram: the posts are 1.2-1.4 cm wide inside a 0.5 cm outline pad (DIAGNOSIS 2)
LP_LEVEL_MIN_PTS = 20     # points per CONTAINER_PLANE_BIN bin that make a structure level: free bins hold 0 points, levels >= 400 (DIAGNOSIS 2)
LP_MARGIN = CA_CLEAR_MARGIN   # m, clearance of the body / hand from any measured structure (the program's hand-surface margin)
LP_MIN_TRAVEL = 0.005     # m, the contact primitive's minimum commanded travel (contact_move input contract)
LP_CARVE_PCT = 99         # percentile of the carved ray heights (robust maximum; DIAGNOSIS 2)
LP_SLICE = 2 * CONTAINER_PLANE_BIN   # m, C16I4: slice width of the held-body footprint and bin of the pitch profile (2 height-histogram bins; DIAGNOSIS C16I4 5)
LP_TILT_MAX = max(abs(v) for v in GT_TILTS_DEG)   # deg, C16I4: the program's bounded tool tilt about the jaw (closing) axis (C13I2 / ga_reachable_body_quat)
LP_PITCH_SPAN = 2.0 * LP_TILT_MAX   # deg, C16I4: pitch search range (+-30 deg: the recorded in-pad pivots are 5.5-17.3 deg)
LP_PITCH_STEP = 0.25      # deg, C16I4: pitch refinement step (the recorded estimation error is <= 0.5 deg, DIAGNOSIS C16I4 4)
LP_PITCH_MIN_BINS = 10    # C16I4: common profile bins required for a pitch estimate (10 cm of body; the recorded fits use 22-31 bins)
LP_LEVEL_TRIES = 2        # C16I4: hand turns per settling (the second corrects a pinch that lagged the first turn)


def _lp_runs(z, lo, hi):
    """Occupied / free runs of a height histogram (CONTAINER_PLANE_BIN bins from lo to hi): list of (occupied, z0, z1)."""
    edges = np.arange(float(lo), float(hi) + CONTAINER_PLANE_BIN, CONTAINER_PLANE_BIN)
    if len(edges) < 2:
        return []
    h, _ = np.histogram(np.asarray(z, float), bins=edges); occ = h >= LP_LEVEL_MIN_PTS; runs = []; k = 0
    while k < len(occ):
        j = k
        while j < len(occ) and occ[j] == occ[k]:
            j += 1
        runs.append((bool(occ[k]), float(edges[k]), float(edges[j]))); k = j
    return runs


def lp_frame_points(depth_img, K, E, keep_mask=None):
    """In-box frame points (C_WS) of the whole image, or of keep_mask's pixels."""
    d = np.asarray(depth_img); d = d[:, :, 0] if d.ndim == 3 else d
    m = np.ones(d.shape, np.uint8) if keep_mask is None else np.asarray(keep_mask).astype(np.uint8)
    pts = mask_to_world_points(m, d, K, E)
    if pts is None or len(pts) == 0:
        return np.empty((0, 3))
    pts = np.asarray(pts, float)
    k = (pts[:, 0] > C_WS[0][0]) & (pts[:, 0] < C_WS[0][1]) & (pts[:, 1] > C_WS[1][0]) & (pts[:, 1] < C_WS[1][1]) & (pts[:, 2] > C_WS[2][0]) & (pts[:, 2] < C_WS[2][1])
    return pts[k]


def lp_side_layers(F, c, n_out, h_n, w, h_w, z_top, floor_z, cam):
    """Levels, layers and opening of one outline side (rule above). Returns a dict (layers may be empty)."""
    n_out = np.asarray(n_out, float); w = np.asarray(w, float); c = np.asarray(c, float)
    d = h_n - (F[:, :2] - c) @ n_out; ww = (F[:, :2] - c) @ w; d_cam = h_n - float((np.asarray(cam, float)[:2] - c) @ n_out)
    band = (d > -LP_MARGIN) & (d < LP_BAND) & (F[:, 2] < z_top + CONTAINER_PLANE_BIN)
    core = band & (np.abs(ww) < h_w - LP_CORNER)
    runs = _lp_runs(F[core, 2], float(floor_z) - CONTAINER_PLANE_BIN, z_top + CONTAINER_PLANE_BIN)
    levels = [(a, b) for o, a, b in runs if o]
    rec = {"n_band": int(core.sum()), "levels": [[round(a, 4), round(b, 4)] for a, b in levels], "layers": []}
    if len(levels) < 2:
        return rec
    struct = band & (F[:, 2] >= levels[0][1])   # every band point above the lowest level: posts, boards, the top
    face_d = float(np.percentile(d[struct], CONTAINER_PCT_LO)) if int(struct.sum()) else 0.0
    rec.update(face_d=round(face_d, 4), depth=round(float(2.0 * h_n - face_d), 4), d_cam=round(d_cam, 4))
    for (l0, l1), (u0, u1) in zip(levels[:-1], levels[1:]):
        lay = {"run": [round(l1, 4), round(u0, 4)], "ok": False}
        Lb = core & (F[:, 2] >= l0) & (F[:, 2] < l1); Ub = band & (F[:, 2] >= u0) & (F[:, 2] < u1)
        if int(Lb.sum()) == 0 or int(Ub.sum()) == 0:
            lay["why"] = "level_unseen"; rec["layers"].append(lay); continue
        z_L = max(float(np.median(F[Lb, 2])), l1 - CONTAINER_PLANE_BIN); cover_vis = float(np.percentile(F[Ub, 2], CONTAINER_PCT_LO)); e_U = float(np.percentile(d[Ub], CONTAINER_PCT_LO))
        under = (F[:, 2] >= l0) & (F[:, 2] < l1) & (np.abs(ww) <= h_w) & (d > e_U + CONTAINER_PLANE_BIN) & (d < 2.0 * h_n)
        sfr = (d[under] - e_U) / (d[under] - d_cam)   # fraction of the way from the point to the camera at which the ray crosses the cover's edge plane
        zc = F[under, 2] + (float(cam[2]) - F[under, 2]) * sfr
        wc = ww[under] + (float((np.asarray(cam, float)[:2] - c) @ w) - ww[under]) * sfr
        zc = zc[(zc < cover_vis) & (np.abs(wc) <= h_w)]   # rays that crossed the edge plane inside the opening, below the cover's visible top
        lay.update(z_floor=round(z_L, 4), cover_vis=round(cover_vis, 4), cover_edge_d=round(e_U, 4), n_carved=int(len(zc)))
        if len(zc) < LP_LEVEL_MIN_PTS:
            lay["why"] = "floor_unseen_under_cover"; rec["layers"].append(lay); continue
        z_C = min(cover_vis, float(np.percentile(zc, LP_CARVE_PCT)))
        mid = band & (F[:, 2] > z_L + CONTAINER_PLANE_BIN) & (F[:, 2] < z_C - CONTAINER_PLANE_BIN) & (np.abs(ww) <= h_w + LP_MARGIN)
        ws = np.sort(np.concatenate([[-h_w], ww[mid], [h_w]])); gaps = np.diff(ws); k = int(np.argmax(gaps))
        lay.update(ok=True, z_cover=round(z_C, 4), height=round(z_C - z_L, 4), w_lo=round(float(ws[k]), 4), w_hi=round(float(ws[k + 1]), 4), n_posts=int(mid.sum()))
        rec["layers"].append(lay)
    return rec


def lp_measure(rgb, depth_img, K, E, prompts, floor_z, code):
    """Layer record of the reference assembly for the side code (rule above): ok, the chosen layer and the opening frame."""
    out = {"ok": False, "code": code}
    if floor_z is None:
        out["why"] = "no_floor"; return out
    pts, ainfo = ca_assembly(rgb, depth_img, K, E, list(prompts or []), floor_z)
    out["assembly"] = ainfo
    if pts is None or len(pts) == 0:
        out["why"] = "assembly_not_found"; return out
    body = ca_body(pts)
    if body is None:
        out["why"] = "no_top_face"; return out
    c = np.asarray(body["centre"], float); a, b = (np.asarray(v, float) for v in body["axes"]); hu, hv = (float(v) for v in body["half"]); z_top = float(body["z_top"])
    cam = np.asarray(E, float)[:3, 3]; F = lp_frame_points(depth_img, K, E)
    out.update(centre=[round(float(v), 4) for v in c], half=[round(hu, 4), round(hv, 4)], z_top=round(z_top, 4))
    sides = []
    for n_out, h_n, w, h_w in ((a, hu, b, hv), (-a, hu, b, hv), (b, hv, a, hu), (-b, hv, a, hu)):
        facing = float((cam[:2] - c) @ n_out) - h_n
        if facing <= 0:
            continue
        r = lp_side_layers(F, c, n_out, h_n, w, h_w, z_top, floor_z, cam)
        r.update(n_out=[round(float(v), 4) for v in n_out], w=[round(float(v), 4) for v in w], h_n=h_n, h_w=h_w)
        good = [l for l in r["layers"] if l.get("ok")]
        r["tallest"] = max([l["height"] for l in good], default=0.0)
        sides.append(r)
    out["sides"] = [{k: v for k, v in s.items()} for s in sides]
    if not sides or max(s["tallest"] for s in sides) <= 0.0:
        out["why"] = "no_open_side"; return out
    s = max(sides, key=lambda r: r["tallest"]); good = [l for l in s["layers"] if l.get("ok")]
    lay = good[0] if code == "zmin" else good[-1] if code == "zmax" else max(good, key=lambda l: l["height"]) if code is None else None
    if lay is None:
        out["why"] = "side_code_not_a_layer"; return out
    n_out = np.asarray(s["n_out"], float); w = np.asarray(s["w"], float); face_xy = c + n_out * (s["h_n"] - s["face_d"])
    out.update(ok=True, layer=lay, n_layers=len(good), half_nw=[round(float(s["h_n"]), 4), round(float(s["h_w"]), 4)], n_out=s["n_out"], u=[round(float(-v), 4) for v in n_out], w=s["w"], face_xy=[round(float(v), 4) for v in face_xy],
               depth=s["depth"], w_lo=lay["w_lo"], w_hi=lay["w_hi"], z_floor=lay["z_floor"], z_cover=lay["z_cover"], floor_z=round(float(floor_z), 4), outline_c=[round(float(v), 4) for v in c])
    return out


def lp_prefit(obj_pts, lay):
    """Before motion: the object's standing height + 2 LP_MARGIN must fit the layer's free height."""
    z = np.asarray(obj_pts, float)[:, 2]; h = float(np.percentile(z, CONTAINER_PCT_HI) - np.percentile(z, CONTAINER_PCT_LO))
    need = h + 2.0 * LP_MARGIN; free = float(lay["z_cover"]) - float(lay["z_floor"])
    return {"ok": bool(need <= free), "object_height": round(h, 4), "need": round(need, 4), "free": round(free, 4)}


def lp_obstacles(depth_img, K, E, floor_z, obj_mask, obj_pts, lay, tcp_obs):
    """Static obstacles of the observe frame: raised in-box points, without the robot's own points (inside the base -> TCP strip
    C_ARM_STRIP and above the observe pose's fingertips), the bound object's pixels / points (within LP_MARGIN) and the assembly's
    outline grown by LP_MARGIN (the layer model covers the assembly)."""
    keep = None if obj_mask is None else (np.asarray(obj_mask) == 0)
    P = lp_frame_points(depth_img, K, E, keep)
    P = P[P[:, 2] > float(floor_z) + SUPPORT_RAISED_MIN]
    if tcp_obs is not None and len(P):
        own = (_c_seg_dist(P[:, 0], P[:, 1], np.zeros(2), np.asarray(tcp_obs, float)[:2]) < C_ARM_STRIP) & (P[:, 2] > float(tcp_obs[2]) - CA_HAND["tip"] - LP_MARGIN)
        P = P[~own]
    if obj_pts is not None and len(obj_pts) and len(P):
        P = P[~ca_near(P, np.asarray(obj_pts, float), LP_MARGIN)]
    c = np.asarray(lay["outline_c"], float); n = np.asarray(lay["n_out"], float); w = np.asarray(lay["w"], float); h_n, h_w = (float(v) for v in lay["half_nw"])
    rel = P[:, :2] - c; inside = (np.abs(rel @ n) <= h_n + LP_MARGIN) & (np.abs(rel @ w) <= h_w + LP_MARGIN)
    return P[~inside]


def lp_hand_model_R(q):
    """Rotation of the hand model frame (CA_HAND / gw_hand_boxes: x = pad closing axis) for a commanded wxyz quaternion: the commanded
    tool y is the closing axis (make_topdown_quat(psi) with psi = closing angle + 90, as gh_handle_plan / ga_plan_grasp command it)."""
    Rc = _R(np.asarray(q, float)); return np.stack([Rc[:, 1], -Rc[:, 0], Rc[:, 2]], 1)


def lp_hand_extent(R, axis):
    """(min, max) of the closed hand's box corners (CA_HAND, model frame R) projected on a world axis."""
    vals = []
    for (x0, x1), (y0, y1), (z0, z1) in _ca_hand_boxes():
        for xv in (x0, x1):
            for yv in (y0, y1):
                for zv in (z0, z1):
                    vals.append(float(np.dot(R @ np.array([xv, yv, zv]), np.asarray(axis, float))))
    return min(vals), max(vals)


def lp_held(meas):
    """Held body from the lift measurements: (rel points in the gripper frame | None, offset_xy_base | None, info)."""
    ms = [m for m in (meas or []) if m is not None]
    if not ms:
        return None, None, {"why": "no_held_measurement"}
    off, hinfo = gh_held_offset(ms)
    if off is None:
        offs = [np.asarray(m["offset_xy_base"], float) for m in ms]
        off = np.mean(offs, 0); hinfo = dict(hinfo, how="visible_axis")
    m = max(ms, key=lambda q: q["n_pts"]); Rg = _R(np.asarray(m["tcp_q"], float))
    rel = (np.asarray(m["pts"], float) - np.asarray(m["tcp_p"], float)) @ Rg   # gripper-frame coordinates (rows: R^T (p - tcp))
    off_g = Rg.T @ np.array([float(off[0]), float(off[1]), 0.0])
    return rel, off_g, {**{k: v for k, v in hinfo.items() if k != "sections"}, "n_pts": int(len(rel))}


def lp_body_slices(a, bb, a_lo, a_hi, ring=None):
    """C16I4: footprint of the held body along the insertion axis u. [a_lo, a_hi] is cut into LP_SLICE slices; a slice's lateral extent
    (along w) is the CONTAINER_PCT_LO..HI range of the held points inside it (>= CONTAINER_SECTOR_MIN_PTS points); a slice with fewer
    points (a part hidden from the camera, e.g. the pinched section under the fingers) takes the union of its nearest measured
    neighbours on both sides; a measured ring body (a_c, b_c, r) adds its rim chord to every slice it crosses. Returns a list of
    [s0, s1, b_lo, b_hi] (None if no slice is measured)."""
    a = np.asarray(a, float); bb = np.asarray(bb, float)
    n = max(1, int(np.ceil((float(a_hi) - float(a_lo)) / LP_SLICE - 1e-9)))
    S = []; ext0 = []
    for k in range(n):
        s0 = float(a_lo) + k * LP_SLICE; s1 = min(s0 + LP_SLICE, float(a_hi)); m = (a >= s0) & (a < s1 + (1e-9 if k == n - 1 else 0.0))
        e = [float(np.percentile(bb[m], CONTAINER_PCT_LO)), float(np.percentile(bb[m], CONTAINER_PCT_HI))] if int(m.sum()) >= CONTAINER_SECTOR_MIN_PTS else None
        S.append([s0, s1]); ext0.append(e)
    idx = [k for k, e in enumerate(ext0) if e is not None]
    if not idx:
        return None
    out = []
    for k, (s0, s1) in enumerate(S):
        e = ext0[k]
        if e is None:
            nb = [j for j in idx if j < k][-1:] + [j for j in idx if j > k][:1]
            e = [min(ext0[j][0] for j in nb), max(ext0[j][1] for j in nb)]
        lo_, hi_ = e
        if ring is not None:
            a_c, b_c, r = (float(v) for v in ring); ac = min(max(a_c, s0), s1)
            if abs(ac - a_c) < r:
                h = float(np.sqrt(r * r - (ac - a_c) ** 2)); lo_, hi_ = min(lo_, b_c - h), max(hi_, b_c + h)
        out.append([s0, s1, lo_, hi_])
    return out


def lp_plan(lay, rel_g, off_g, q_hold, drop_used, obst, z_transport, r_ring=None, pref=None):
    """Insertion plan (rule above) from the layer record, the held body (gripper frame) and the obstacles. Pure geometry.
    C16I4: the swept checks use the held body's measured footprint (lp_body_slices) and the approach is chosen among measured
    candidates: the insertion height inside the layer's TCP interval and the lateral position inside the opening's slack, in steps of
    LP_MARGIN, nearest first (C15I3's centred mid-height approach is the first candidate; pref = (lateral, height) re-centres the order
    on a previous plan). Returns a dict: ok, why, q (wxyz), pre / fin TCP, z_ins, z_release, checks."""
    rec = {"ok": False}
    u = np.asarray(lay["u"], float); w = np.asarray(lay["w"], float); face = np.asarray(lay["face_xy"], float); c = np.asarray(lay["outline_c"], float)
    q0 = np.asarray(q_hold, float); off_w0 = _R(q0) @ np.asarray(off_g, float)
    delta = 0.0
    if float(np.hypot(off_w0[0], off_w0[1])) > PRIM["held_offset_max"]:   # a body held off its centre (beyond the program's centre-near hold bound) leads into the opening
        delta = float(np.degrees(np.arctan2(u[1], u[0]) - np.arctan2(off_w0[1], off_w0[0]))); delta = (delta + 180.0) % 360.0 - 180.0
    q = _yaw_about_vertical(q0, delta) if abs(delta) > 1e-9 else q0
    R = _R(q); Rm = lp_hand_model_R(q); P = np.asarray(rel_g, float) @ R.T; off = R @ np.asarray(off_g, float)
    a = P[:, :2] @ u; bb = P[:, :2] @ w
    a_min, a_max = float(np.percentile(a, CONTAINER_PCT_LO)), float(np.percentile(a, CONTAINER_PCT_HI)); b_min, b_max = float(np.percentile(bb, CONTAINER_PCT_LO)), float(np.percentile(bb, CONTAINER_PCT_HI))
    a_c, b_c = float(off[:2] @ u), float(off[:2] @ w)   # the held body's centre (in-hand offset) along u / w
    if r_ring is not None:   # a measured ring body: its rim circle bounds the body (the far rim is its leading end)
        a_min, a_max = min(a_min, a_c - float(r_ring)), max(a_max, a_c + float(r_ring)); b_min, b_max = min(b_min, b_c - float(r_ring)), max(b_max, b_c + float(r_ring))
    hu_lo, hu_hi = lp_hand_extent(Rm, np.array([u[0], u[1], 0.0])); hz_lo, _ = lp_hand_extent(Rm, np.array([0.0, 0.0, 1.0]))
    D = float(lay["depth"])
    t_max = min(-(hu_hi + LP_MARGIN), D - LP_MARGIN - a_max)   # C15I5: the deepest TCP depth the hand / far-end clearances allow (also bounds the release anchor)
    t = min(0.5 * D - a_c, t_max)
    rec.update(yaw_delta_deg=round(delta, 2), extent_u=[round(a_min, 4), round(a_max, 4)], extent_w=[round(b_min, 4), round(b_max, 4)], hand_u=[round(hu_lo, 4), round(hu_hi, 4)],
               tcp_depth=round(t, 4), tcp_depth_max=round(t_max, 4), body_centre_depth=round(t + a_c, 4), depth=round(D, 4), hand_z_lo=round(hz_lo, 4))
    if t + a_c <= 0.0:
        rec["why"] = "body_centre_outside_face"; return rec
    s0 = 0.5 * (float(lay["w_lo"]) + float(lay["w_hi"])) - b_c   # the body centre on the opening's lateral centre
    thru = (a + t) > -LP_MARGIN   # the visible points that pass the face plane
    if r_ring is not None or not int(thru.sum()):
        bl, bh = (b_c - float(r_ring), b_c + float(r_ring)) if r_ring is not None else (b_min, b_max)
    else:
        bl, bh = float(np.percentile(bb[thru], CONTAINER_PCT_LO)), float(np.percentile(bb[thru], CONTAINER_PCT_HI))
    if int(thru.sum()):
        bl, bh = min(bl, float(np.percentile(bb[thru], CONTAINER_PCT_LO))), max(bh, float(np.percentile(bb[thru], CONTAINER_PCT_HI)))
    s_lo, s_hi = float(lay["w_lo"]) + LP_MARGIN - bl, float(lay["w_hi"]) - LP_MARGIN - bh   # lateral positions that keep the body LP_MARGIN inside the posts
    rec["lateral"] = [round(s0 + bl, 4), round(s0 + bh, 4)]
    if s0 < s_lo - 1e-9 or s0 > s_hi + 1e-9:
        rec["why"] = "body_wider_than_opening"; return rec
    pre_t = -(a_max + LP_MARGIN)
    inside = (a + t) > 0.0
    top_in = float(np.percentile(P[inside, 2], CONTAINER_PCT_HI)) if int(inside.sum()) else float(np.percentile(P[:, 2], CONTAINER_PCT_HI))
    z_L = float(lay["z_floor"]); z_C = float(lay["z_cover"]); fl = max(z_L, float(lay["floor_z"]))
    lo = fl + float(drop_used) + LP_MARGIN; hi = z_C - LP_MARGIN - top_in
    rec.update(z_lo=round(lo, 4), z_hi=round(hi, 4), top_in=round(top_in, 4), drop=round(float(drop_used), 4))
    if hi < lo + LP_MIN_TRAVEL:
        rec["why"] = "held_body_does_not_fit_layer_height"; return rec
    z_mid = max(0.5 * (lo + hi), lo + LP_MIN_TRAVEL)
    if min(z_mid, lo) + hz_lo < float(lay["floor_z"]) + CA_FLOOR_CLEAR:   # C15I5: also at the release height (the lowest TCP of the sequence)
        rec["why"] = "hand_below_floor_clearance"; return rec
    # swept checks against the static obstacles (body footprint and hand boxes, LP_MARGIN)
    O = np.asarray(obst, float) if obst is not None and len(obst) else np.empty((0, 3))
    z_bot, z_top_rel = -float(drop_used), float(np.percentile(P[:, 2], CONTAINER_PCT_HI))
    body = (P[:, 2] >= z_bot - LP_MARGIN) & (P[:, 2] <= z_top_rel + LP_MARGIN)   # held points inside the measured hang (deeper points are not the body)
    S = lp_body_slices(a[body], bb[body], a_min, a_max, None if r_ring is None else (a_c, b_c, float(r_ring)))
    if S is None:
        S = [[a_min, a_max, b_min, b_max]]
    def body_hits(tcp_xy, z_lo_tcp, z_hi_tcp, a_hi_extra=0.0, box=False):
        if len(O) == 0:
            return 0
        r = O[:, :2] - np.asarray(tcp_xy, float); ra = r @ u; rb = r @ w
        zk = (O[:, 2] >= z_lo_tcp + z_bot - LP_MARGIN) & (O[:, 2] <= z_hi_tcp + z_top_rel + LP_MARGIN)
        if box:   # C15I3's single box (logged for comparison only)
            return int((zk & (ra >= a_min - LP_MARGIN) & (ra <= a_max + a_hi_extra + LP_MARGIN) & (rb >= b_min - LP_MARGIN) & (rb <= b_max + LP_MARGIN)).sum())
        m = np.zeros(len(O), bool)
        for x0, x1, y0, y1 in S:
            m |= zk & (ra >= x0 - LP_MARGIN) & (ra <= x1 + a_hi_extra + LP_MARGIN) & (rb >= y0 - LP_MARGIN) & (rb <= y1 + LP_MARGIN)
        return int(m.sum())
    def hand_hits(tcp_list):
        n = 0
        for p in tcp_list:
            ok, k, _ = ca_clearance(O, p, Rm)
            n += 0 if ok else k
        return n
    def check(s, z_ins, box=False):
        pre_xy = face + u * pre_t + w * s
        col = [np.array([pre_xy[0], pre_xy[1], z]) for z in list(np.arange(float(z_transport), z_ins, -CA_DESCENT_STEP)) + [z_ins]]
        sweep = [np.array([pre_xy[0], pre_xy[1], z_ins]) + np.array([u[0], u[1], 0.0]) * k for k in list(np.arange(0.0, t - pre_t, CA_DESCENT_STEP)) + [t - pre_t]]
        hh_c, hh_s = hand_hits(col), hand_hits(sweep)
        return body_hits(pre_xy, z_ins, max(float(z_transport), z_ins), box=box) + hh_c, body_hits(pre_xy, z_ins, z_ins, a_hi_extra=t - pre_t, box=box) + hh_s
    s_ref, z_ref = (s0, z_mid) if pref is None else (min(max(float(pref[0]), s_lo), s_hi), min(max(float(pref[1]), lo + LP_MIN_TRAVEL), hi))
    ni = int(np.ceil((s_hi - s_lo) / LP_MARGIN)) + 1; nj = int(np.ceil((hi - lo) / LP_MARGIN)) + 1
    cands = [(abs(i) + abs(j), abs(i), i, j) for i in range(-ni, ni + 1) for j in range(-nj, nj + 1)
             if s_lo - 1e-9 <= s_ref + i * LP_MARGIN <= s_hi + 1e-9 and lo + LP_MIN_TRAVEL - 1e-9 <= z_ref + j * LP_MARGIN <= hi + 1e-9]
    cands.sort()
    b_col, b_sw = check(s_ref, z_ref, box=True)
    rec.update(box_column_hits=b_col, box_sweep_hits=b_sw, n_slices=len(S), n_obstacles=int(len(O)), z_transport=round(float(z_transport), 4))
    first = None; chosen = None; n_tried = 0
    for _, _, i, j in cands:
        s, z_ins = s_ref + i * LP_MARGIN, z_ref + j * LP_MARGIN; n_col, n_sweep = check(s, z_ins); n_tried += 1
        if first is None:
            first = (n_col, n_sweep)
        if n_col == 0 and n_sweep == 0:
            chosen = (s, z_ins, i, j); break
    rec.update(n_candidates=len(cands), n_tried=n_tried, column_hits=first[0] if first else None, sweep_hits=first[1] if first else None)
    if chosen is None:
        rec["why"] = "pre_insert_column_blocked" if (first and first[0]) else "insertion_sweep_blocked"; return rec
    s, z_ins, i, j = chosen
    fin_xy = face + u * t + w * s; pre_xy = face + u * pre_t + w * s   # w-coordinates are relative to the outline centre, which lies on the face's normal
    rec.update(lateral=[round(s + bl, 4), round(s + bh, 4)], lateral_shift=round(s - s0, 4), alt_steps=[int(i), int(j)],
               pre=[round(float(pre_xy[0]), 4), round(float(pre_xy[1]), 4), round(z_ins, 4)], fin=[round(float(fin_xy[0]), 4), round(float(fin_xy[1]), 4), round(z_ins, 4)],
               z_ins=round(z_ins, 4), z_release=round(lo, 4), column_hits_chosen=0, sweep_hits_chosen=0, s_lat=round(s, 4),
               body_centre_xy=[round(float(v), 4) for v in (fin_xy + off[:2])])
    rec.update(ok=True, why=None, q=[float(v) for v in q])
    return rec


def lp_carry_frame(q):
    """C16I4: world-aligned carry frame of a hand orientation: columns e (horizontal, perpendicular to the pad closing axis), c (the
    closing axis, the commanded tool y, projected on the horizontal) and z (up); e = c x z (a proper rotation). None for a vertical closing axis."""
    Rc = _R(np.asarray(q, float)); cc = np.array([Rc[0, 1], Rc[1, 1], 0.0]); n = float(np.linalg.norm(cc))
    if n < 1e-6:
        return None
    cc /= n; z = np.array([0.0, 0.0, 1.0]); return np.stack([np.cross(cc, z), cc, z], 1)


def lp_carry_points(pts, origin, q):
    """Points in the carry frame of q, relative to origin (rows: C^T (p - origin)); None without a frame."""
    C = lp_carry_frame(q)
    return None if C is None or pts is None or len(pts) == 0 else (np.asarray(pts, float) - np.asarray(origin, float)) @ C


def _lp_profile(X):
    """Top profile of a body in the carry frame: CONTAINER_PCT_HI height per LP_SLICE bin along e (bins with >= CONTAINER_SECTOR_MIN_PTS points)."""
    b = np.floor(X[:, 0] / LP_SLICE).astype(int); out = {}
    for k in np.unique(b):
        m = b == k
        if int(m.sum()) >= CONTAINER_SECTOR_MIN_PTS:
            out[int(k)] = float(np.percentile(X[m, 2], CONTAINER_PCT_HI))
    return out


def lp_pitch(X_ref, X_new):
    """C16I4: the rotation phi (deg) about the carry frame's closing axis that maps the held body's top profile X_new onto the reference
    profile X_ref (vertical offset free; cost = median absolute deviation of the bin-wise height differences). Coarse 1-deg search over
    +-LP_PITCH_SPAN, refined at LP_PITCH_STEP. Returns (phi | None, info)."""
    if X_ref is None or X_new is None or len(X_ref) == 0 or len(X_new) == 0:
        return None, {"why": "no_points"}
    pr = _lp_profile(np.asarray(X_ref, float)); X_new = np.asarray(X_new, float)
    def cost(deg):
        Rr = Rotation.from_euler("y", float(deg), degrees=True).as_matrix(); pn = _lp_profile(X_new @ Rr.T); ks = sorted(set(pr) & set(pn))
        if len(ks) < LP_PITCH_MIN_BINS:
            return None, len(ks)
        d = np.array([pn[k] - pr[k] for k in ks]); return float(np.median(np.abs(d - np.median(d)))), len(ks)
    best = None
    for deg in np.arange(-LP_PITCH_SPAN, LP_PITCH_SPAN + 1e-9, 1.0):
        cst, nb = cost(deg)
        if cst is not None and (best is None or cst < best[1]):
            best = (float(deg), cst, nb)
    if best is None:
        return None, {"why": "too_few_common_bins", "n_ref_bins": len(pr)}
    d0 = best[0]
    for deg in np.arange(d0 - 1.0, d0 + 1.0 + 1e-9, LP_PITCH_STEP):
        cst, nb = cost(deg)
        if cst is not None and cst < best[1]:
            best = (float(deg), cst, nb)
    return best[0], {"phi_deg": round(best[0], 2), "mad_mm": round(1000.0 * best[1], 2), "n_bins": best[2]}


def lp_pitch_tol(X_new):
    """The pitch tolerance of a held body: the angle that moves its farthest measured point (CONTAINER_PCT_HI of |e|) by LP_MARGIN."""
    lever = float(np.percentile(np.abs(np.asarray(X_new, float)[:, 0]), CONTAINER_PCT_HI))
    return float(np.degrees(np.arcsin(min(1.0, LP_MARGIN / max(lever, LP_MARGIN))))), lever


def lp_tool_tilt(q):
    """Angle (deg) of the commanded tool z from the downward vertical."""
    return float(np.degrees(np.arccos(np.clip(-_R(np.asarray(q, float))[2, 2], -1.0, 1.0))))


def lp_pitched_quat(q, phi):
    """C16I4: q rotated by phi (deg) about its own horizontal closing axis through the TCP (gt_tilt_quat's axis), the rotation bounded so that
    the tool stays within LP_TILT_MAX of the vertical (the program's bounded jaw-axis tilt). Returns (q', phi_applied)."""
    C = lp_carry_frame(q)
    if C is None:
        return np.asarray(q, float), 0.0
    xq = Rotation.from_quat(np.asarray(q, float)[[1, 2, 3, 0]])
    for f in np.linspace(1.0, 0.0, int(np.ceil(abs(float(phi)) / LP_PITCH_STEP)) + 1):
        x = (Rotation.from_rotvec(C[:, 1] * np.radians(float(phi) * f)) * xq).as_quat(); qq = np.array([x[3], x[0], x[1], x[2]])
        if lp_tool_tilt(qq) <= LP_TILT_MAX + 1e-6:
            return qq, float(phi) * f
    return np.asarray(q, float), 0.0


def lp_offset_now(off_w_lift, q_lift, q_now):
    """C16I4: the held body's in-hand offset (gripper frame of q_now) after the hand has turned about the vertical from q_lift to q_now and the
    body has been brought back to its measured attitude: the lift-time horizontal offset turned by the closing axis's yaw change."""
    C0, C1 = lp_carry_frame(q_lift), lp_carry_frame(q_now)
    if C0 is None or C1 is None:
        return _R(np.asarray(q_now, float)).T @ np.array([float(off_w_lift[0]), float(off_w_lift[1]), 0.0])
    d = float(np.arctan2(C1[1, 1], C1[0, 1]) - np.arctan2(C0[1, 1], C0[0, 1])); cz, sz = np.cos(d), np.sin(d)
    o = np.array([cz * float(off_w_lift[0]) - sz * float(off_w_lift[1]), sz * float(off_w_lift[0]) + cz * float(off_w_lift[1]), 0.0])
    return _R(np.asarray(q_now, float)).T @ o


def lp_release_anchor(pl, lay, p_act):
    """C15I5: the release descent starts from the ACTUAL TCP after the horizontal insertion, straight down (the contact primitive's
    approach axis -z, AXIS_TOL_DEG unchanged). The insertion's goal check accepts any TCP within PRIM["reach_tol"] of the planned
    final pose, so the actual TCP may lie a few mm short of it; a descent commanded to the planned x-y would then leave the vertical
    by atan(offset / travel) (> 15 deg for a short descent). The anchor is accepted only when the actual pose keeps every clearance
    of the plan: TCP depth <= tcp_depth_max (hand LP_MARGIN outside the face, far end LP_MARGIN before the back), the body centre
    inside the face, the body LP_MARGIN inside the lateral posts, the TCP inside the planned height interval [z_lo, z_hi] and at
    least LP_MIN_TRAVEL above the release height. Pure geometry: returns {"ok", "why", "xy", "travel", "d_u", "d_w"}."""
    u = np.asarray(lay["u"], float); w = np.asarray(lay["w"], float); fin = np.asarray(pl["fin"], float); p = np.asarray(p_act, float)
    d = p[:2] - fin[:2]; d_u = float(d @ u); d_w = float(d @ w); travel = float(p[2]) - float(pl["z_release"])
    rec = {"ok": False, "why": None, "xy": [float(p[0]), float(p[1])], "travel": round(travel, 4), "d_u": round(d_u, 4), "d_w": round(d_w, 4)}
    if float(pl["tcp_depth"]) + d_u > float(pl["tcp_depth_max"]) + 1e-9:
        rec["why"] = "anchor_beyond_depth_clearance"; return rec
    if float(pl["body_centre_depth"]) + d_u <= 0.0:
        rec["why"] = "anchor_body_centre_outside_face"; return rec
    lat = np.asarray(pl["lateral"], float) + d_w
    if lat[0] < float(lay["w_lo"]) + LP_MARGIN or lat[1] > float(lay["w_hi"]) - LP_MARGIN:
        rec["why"] = "anchor_body_outside_opening"; return rec
    if not (float(pl["z_lo"]) <= float(p[2]) <= float(pl["z_hi"])):
        rec["why"] = "anchor_outside_layer_heights"; return rec
    if travel < LP_MIN_TRAVEL:
        rec["why"] = "anchor_descent_below_minimum_travel"; return rec
    rec.update(ok=True); return rec


def _lp_stop(sg, why, q, **kw):
    print("PLACE_RESULT " + json.dumps({"sg": sg, "status": "failed", "reason": "layer_" + str(why), **kw}), flush=True)
    safe_retreat(sg=sg, q_hold=q)
    return None


def lp_measure_carry(held_prompts, ref_X, sg, where):
    """C16I4: two held-body measurements at the current pose and the body's pitch about the closing axis relative to its resting attitude
    (ref_X: the bound object's points at rest in the carry frame of the grasp). Returns (meas, phi | None, tol | None, valid drops, info)."""
    meas = []; phis = []; info = {"sg": sg, "where": where}
    for k in range(2):
        obs = get_observation(); _c_stash_view(obs); m = measure_held_object(held_prompts, obs); meas.append(m)
        if m is not None and ref_X is not None:
            try:
                X = lp_carry_points(m["pts"], m["tcp_p"], m["tcp_q"]); ph, pinfo = lp_pitch(ref_X, X)
            except Exception as e:  # noqa: BLE001  (an estimation error: no pitch estimate from this frame)
                ph, pinfo = None, {"why": f"pitch_error: {type(e).__name__}: {e}"}
            info[f"fit{k}"] = pinfo
            if ph is not None:
                phis.append(ph); tol, lever = lp_pitch_tol(X); info.update(tol_deg=round(tol, 2), lever=round(lever, 4))
    ds = [m["drop_below_tcp"] for m in meas if m is not None and PRIM["drop_bounds"][0] <= m["drop_below_tcp"] <= PRIM["drop_bounds"][1]]
    phi = float(np.median(phis)) if phis else None
    info.update(n_meas=sum(m is not None for m in meas), n_valid_drop=len(ds), drop=round(float(np.median(ds)), 4) if ds else None, phi_deg=None if phi is None else round(phi, 2))
    tol = info.get("tol_deg")
    return meas, phi, tol, ds, info


def lp_settle_carry(lay, obst, held_prompts, q, z_col, ref_X, lift, sg, where, pref=None):
    """C16I4: measure the carry at the current (static) pose; when the body has pivoted about the pad closing axis by more than its pitch
    tolerance since it rested (the pinch slips under the transport's accelerations), turn the hand about that axis in place to bring it
    back (bounded tool tilt, at most LP_LEVEL_TRIES turns, each re-measured), then re-plan the insertion from the latest measurement.
    Returns (plan | None, q, why). No measurement before any turn: (None, q, None) = keep the current plan."""
    meas, phi, tol, ds, info = lp_measure_carry(held_prompts, ref_X, sg, where)
    print("LAYER_CARRY " + _mfj(info), flush=True)
    turned = 0
    while phi is not None and tol is not None and abs(phi) > tol and turned < LP_LEVEL_TRIES:
        p_now, _, _ = read_tcp(); q_t, ph_ap = lp_pitched_quat(q, phi)
        if abs(ph_ap) <= 1e-6:
            break
        if not mf_ik_probe(p_now, q_t):
            print("LAYER_IK " + _mfj({"sg": sg, "pose": f"level_{where}", "ok": False}), flush=True); break
        rt = move_to_pose(np.asarray(p_now, float), q_t, mode="linear", tol_pos=PRIM["reach_tol"], tol_ori_deg=float(tol), corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_level_{where}"); _PRIM_STATS["motion_calls"] += 1
        print("LAYER_MOVE " + _mfj({"sg": sg, "move": f"level_{where}", "status": rt["status"], "steps": rt.get("steps_used"), "phi_applied_deg": round(ph_ap, 2), "tool_tilt_deg": round(lp_tool_tilt(q_t), 2)}), flush=True)
        if not rt["reached"]:
            return None, q, f"level_{where}_" + str(rt["status"])
        q = q_t; turned += 1
        meas, phi, tol, ds, info = lp_measure_carry(held_prompts, ref_X, sg, where)
        print("LAYER_CARRY " + _mfj({**info, "after_turn": turned}), flush=True)
    try:
        rel, _, _ = lp_held(meas)
    except Exception:  # noqa: BLE001  (an unusable measurement)
        rel = None
    if rel is None or not ds:
        return (None, q, None) if turned == 0 else (None, q, f"held_body_unmeasured_{where}")
    off_g = lp_offset_now(lift["off_w"], lift["q"], q)
    try:
        pl = lp_plan(lay, rel, off_g, q, float(np.median(ds)), obst, float(z_col), r_ring=lift["r_ring"], pref=pref)
    except Exception as e:  # noqa: BLE001  (a planning error stops with the object held)
        pl = {"ok": False, "why": f"plan_error: {type(e).__name__}: {e}"}
    print("LAYER_PLAN " + _mfj({"sg": sg, "where": where, **pl}), flush=True)
    if not pl.get("ok"):
        return None, q, pl.get("why")
    return pl, np.asarray(pl["q"], float), None


def lp_place(lay, obst, held_prompts, gquat, sg="sg1", rest=None):
    """Held measurement -> plan -> transport, carry settling (C16I4), lowering, carry settling (C16I4), horizontal insertion, contact descent,
    release, horizontal retreat, rise. rest = (points, grasp TCP, grasp quaternion) of the bound object at rest (C16I4 carry reference)."""
    meas = []
    for k in range(2):
        obs = get_observation(); _c_stash_view(obs); m = measure_held_object(held_prompts, obs); meas.append(m)
        _mark(ev="held_measure", sg=sg, k=k, result=None if m is None else {"offset_xy_base_cm": (np.asarray(m["offset_xy_base"]) * 100).round(2), "drop_below_tcp_cm": round(m["drop_below_tcp"] * 100, 2), "n_pts": m["n_pts"]})
    ds = [m["drop_below_tcp"] for m in meas if m is not None and PRIM["drop_bounds"][0] <= m["drop_below_tcp"] <= PRIM["drop_bounds"][1]]
    rel, off_g, hinfo = lp_held(meas)
    print("LAYER_HELD " + _mfj({"sg": sg, "n_valid": len(ds), **hinfo}), flush=True)
    if rel is None or not ds:
        return _lp_stop(sg, "held_body_unmeasured", gquat)
    drop_used = float(np.median(ds)); p_now, q_now, _ = read_tcp()
    plan_ = _MF.get("last_plan") or {}; r_ring = ((plan_.get("profile_summary") or {}).get("r_top") if (plan_.get("mode") == "handle_pinch" and hinfo.get("applies") and hinfo.get("how") in ("held_ring", "planned_ring")) else None)
    off_w = _R(np.asarray(q_now, float)) @ off_g; obj_xy = np.asarray(p_now, float)[:2] + off_w[:2]
    m_ref = max([m for m in meas if m is not None], key=lambda q_: q_["n_pts"]); lift = {"off_w": (_R(np.asarray(m_ref["tcp_q"], float)) @ off_g)[:2], "q": np.asarray(m_ref["tcp_q"], float), "r_ring": r_ring}   # C16I4
    ref_X = None
    if rest is not None and rest[0] is not None and len(rest[0]):
        try:
            ref_X = lp_carry_points(rest[0], rest[1], rest[2])   # C16I4: the body's resting attitude in the carry frame of the grasp
        except Exception:  # noqa: BLE001  (no reference: no settling turn)
            ref_X = None
        if ref_X is not None:
            try:
                ph_l, pinfo_l = lp_pitch(ref_X, lp_carry_points(m_ref["pts"], m_ref["tcp_p"], m_ref["tcp_q"]))
            except Exception as e:  # noqa: BLE001  (logging only)
                pinfo_l = {"why": f"pitch_error: {type(e).__name__}: {e}"}
            print("LAYER_CARRY " + _mfj({"sg": sg, "where": "lift", **pinfo_l}), flush=True)
    try:
        pl0 = lp_plan(lay, rel, off_g, gquat, drop_used, obst, float(p_now[2]), r_ring=r_ring)
    except Exception as e:  # noqa: BLE001  (a planning error stops with the object held)
        pl0 = {"ok": False, "why": f"plan_error: {type(e).__name__}: {e}"}
    if not pl0.get("pre"):
        print("LAYER_PLAN " + _mfj({"sg": sg, **pl0}), flush=True); return _lp_stop(sg, pl0.get("why"), gquat)
    r_obj = 0.5 * float(np.hypot(pl0["extent_u"][1] - pl0["extent_u"][0], pl0["extent_w"][1] - pl0["extent_w"][0]))
    z_tr, _ci = c_transport_height(float(p_now[2]), float(lay["floor_z"]), obj_xy, np.asarray(pl0["pre"][:2], float) + (np.asarray(pl0["body_centre_xy"], float) - np.asarray(pl0["fin"][:2], float)), r_obj, drop_used, sg=sg)
    try:
        pl = lp_plan(lay, rel, off_g, gquat, drop_used, obst, max(float(z_tr), float(p_now[2])), r_ring=r_ring)
    except Exception as e:  # noqa: BLE001
        pl = {"ok": False, "why": f"plan_error: {type(e).__name__}: {e}"}
    print("LAYER_PLAN " + _mfj({"sg": sg, **pl}), flush=True)
    if not pl.get("ok"):
        return _lp_stop(sg, pl.get("why"), gquat)
    q = np.asarray(pl["q"], float); pre = np.asarray(pl["pre"], float); fin = np.asarray(pl["fin"], float); z_hi = float(pl["z_transport"])
    for lab, pos in (("pre_high", [pre[0], pre[1], z_hi]), ("pre_low", pre), ("fin", fin)):
        if not mf_ik_probe(pos, q):
            print("LAYER_IK " + _mfj({"sg": sg, "pose": lab, "ok": False}), flush=True); return _lp_stop(sg, "ik_" + lab, gquat)
    if z_hi > float(p_now[2]) + 0.005:
        rr = move_to_pose(np.array([p_now[0], p_now[1], z_hi]), np.asarray(q_now, float), mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:layer_raise"); _PRIM_STATS["motion_calls"] += 1
        if not rr["reached"]:
            return _lp_stop(sg, "raise_" + rr["status"], gquat)
    tr = move_to_pose(np.array([pre[0], pre[1], z_hi]), q, mode="linear", tol_pos=0.015, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_transport"); _PRIM_STATS["motion_calls"] += 1
    if tr["status"] == "unreachable":
        tr = move_to_pose(np.array([pre[0], pre[1], z_hi]), q, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:layer_transport_free"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "transport", "status": tr["status"], "steps": tr.get("steps_used")}), flush=True)
    if not tr["reached"]:
        return _lp_stop(sg, "transport_" + tr["status"], q)
    # C16I4: the carry at the top of the pre-insert column (static, clear of every structure): bring the pivoted body back to its resting
    # attitude and re-plan from that measurement (the column below and the insertion are checked for the body as it is now carried)
    pl1, q1, why = lp_settle_carry(lay, obst, held_prompts, q, z_hi, ref_X, lift, sg, "pre_high")
    if why:
        return _lp_stop(sg, why, q1)
    if pl1 is not None:
        pl, q = pl1, q1; pre = np.asarray(pl["pre"], float); fin = np.asarray(pl["fin"], float)
        for lab, pos in (("pre_high", [pre[0], pre[1], z_hi]), ("pre_low", pre), ("fin", fin)):
            if not mf_ik_probe(pos, q):
                print("LAYER_IK " + _mfj({"sg": sg, "pose": lab, "ok": False}), flush=True); return _lp_stop(sg, "ik_" + lab, q)
        p_b, _, _ = read_tcp()
        if float(np.hypot(*(np.asarray(p_b, float)[:2] - pre[:2]))) > 0.002:
            sh = move_to_pose(np.array([pre[0], pre[1], z_hi]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_column_shift"); _PRIM_STATS["motion_calls"] += 1
            print("LAYER_MOVE " + _mfj({"sg": sg, "move": "column_shift", "status": sh["status"], "steps": sh.get("steps_used")}), flush=True)
            if not sh["reached"]:
                return _lp_stop(sg, "column_shift_" + sh["status"], q)
    lw = move_to_pose(pre, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_lower"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "lower", "status": lw["status"], "steps": lw.get("steps_used")}), flush=True)
    if not lw["reached"]:
        return _lp_stop(sg, "lower_" + lw["status"], q)
    # C16I4: the carry at the pre-insert pose (the body may pivot again while lowering): settle and re-plan, preferring the current approach
    p_l, _, _ = read_tcp()
    pl2, q2, why = lp_settle_carry(lay, obst, held_prompts, q, max(float(p_l[2]), float(pre[2])), ref_X, lift, sg, "pre", pref=(pl.get("s_lat"), pl.get("z_ins")))
    if why:
        return _lp_stop(sg, why, q2)
    if pl2 is not None:
        pre2 = np.asarray(pl2["pre"], float)
        if float(np.hypot(*(pre2[:2] - pre[:2]))) > LP_MARGIN:   # the re-planned approach must start within the clearance margin of the pose reached
            return _lp_stop(sg, "pre_shift_beyond_margin", q2)
        pl, q = pl2, q2; pre = pre2; fin = np.asarray(pl["fin"], float)
        if not mf_ik_probe(fin, q):
            print("LAYER_IK " + _mfj({"sg": sg, "pose": "fin", "ok": False}), flush=True); return _lp_stop(sg, "ik_fin", q)
        if float(np.linalg.norm(np.asarray(p_l, float) - pre)) > 0.002:
            ad = move_to_pose(pre, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_pre_adjust"); _PRIM_STATS["motion_calls"] += 1
            print("LAYER_MOVE " + _mfj({"sg": sg, "move": "pre_adjust", "status": ad["status"], "steps": ad.get("steps_used")}), flush=True)
            if not ad["reached"]:
                return _lp_stop(sg, "pre_adjust_" + ad["status"], q)
    hv = verify_hold(_E1_LAST.get("grasp_after"), gripper_opening()) if _E1_LAST.get("grasp_after") is not None else {"status": "uncertain"}
    if hv.get("status") != "verified":
        return _lp_stop(sg, "hold_lost_before_insert", q, hold=hv.get("status"))
    pin = move_to_pose(fin, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, plan_only=True, label=f"{sg}:layer_insert_plan")
    if pin.get("detail") != "plan_only":
        print("LAYER_MOVE " + _mfj({"sg": sg, "move": "insert_plan", "status": pin["status"], "detail": pin.get("detail")}), flush=True); return _lp_stop(sg, "insert_unplannable", q)
    ins = move_to_pose(fin, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_insert"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "insert", "status": ins["status"], "detail": ins.get("detail"), "steps": ins.get("steps_used"), "lateral_cm": round(100.0 * float(ins.get("lateral_dev_max") or 0.0), 2)}), flush=True)
    if not ins["reached"]:
        p_b, _, _ = read_tcp(); move_to_pose(np.array([pre[0], pre[1], float(p_b[2])]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_withdraw"); _PRIM_STATS["motion_calls"] += 1
        return _lp_stop(sg, "insert_" + ins["status"], q)
    p_in = np.asarray(ins.get("final_tcp") if ins.get("final_tcp") is not None else read_tcp()[0], float)   # C15I5: the insertion's final TCP (the interface's FK of the public joints, the contact primitive's start frame)
    anc = lp_release_anchor(pl, lay, p_in)   # C15I5: the descent is anchored at the actual TCP (vertical), within the plan's clearances
    print("LAYER_ANCHOR " + _mfj({"sg": sg, **anc}), flush=True)
    if not anc["ok"]:
        move_to_pose(np.array([pre[0], pre[1], float(p_in[2])]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_withdraw"); _PRIM_STATS["motion_calls"] += 1
        return _lp_stop(sg, anc["why"], q)
    dsc = mc_place_descend(np.asarray(anc["xy"], float), q, float(pl["z_release"]), float(fin[2]), sg)
    _rd = dsc.get("mc_result"); ok_rel, why_rel = gate_allow(_rd, "release")
    if not ok_rel:
        _mark(ev="release_blocked", sg=sg, reason=why_rel)
        p_b, _, _ = read_tcp(); move_to_pose(np.array([pre[0], pre[1], float(p_b[2])]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_withdraw"); _PRIM_STATS["motion_calls"] += 1
        return _lp_stop(sg, "release_blocked", q, gate=why_rel)
    _mark(ev="release_start", sg=sg)
    open_gripper(); gate_record("release"); _PRIM_STATS["motion_calls"] += 1
    print("RELEASE_VERIFY " + _json.dumps({"sg": sg, **verify_release(gripper_opening())}), flush=True)
    for _ in range(5):
        get_observation()
    p_r, _, _ = read_tcp(); back = np.asarray(p_r, float) - np.array([float(lay["u"][0]), float(lay["u"][1]), 0.0]) * CA_BACKOFF
    rb = move_to_pose(back, q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_backoff"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "backoff", "status": rb["status"]}), flush=True)
    safe_retreat(sg=sg, q_hold=q)
    print("PLACE_RESULT " + _json.dumps({"sg": sg, "status": "released", "reason": "inserted_and_released", "relation": "in_layer"}), flush=True)
    return [float(v) for v in pl["body_centre_xy"]]


def lp_run_subgoal(pud_prompts, pl_select, support_prompts, object_side):
    """in_layer subgoal: observe, bind the object, measure the layer, pre-fit, grasp (the parent's strategy loop), lp_place."""
    _TD = task_descriptors()
    _E1_LAST.update({"stop": False, "grasp_after": None, "hold": None}); _CUR["prompts"] = list(pud_prompts); _MF["contract"] = None
    obs = get_observation(); _c_stash_view(obs)
    cam = obs["agentview"]; rgb = cam["images"]["rgb"]; depth = cam["images"]["depth"]; depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth; K = cam["intrinsics"]; E = cam["pose_mat"]
    pud_c, pud_pts, pud_mask = vb_localize_for_grasp(rgb, depth_img, K, E, pud_prompts, object_side)
    if pud_c is None:
        print("BINDING_STOP " + json.dumps({"status": _BIND["last"], "reason": "target not bound unambiguously; no motion"}), flush=True); return None
    floor_z = support_floor_z(depth_img, K, E)
    prompts = list(_TD.get("support_prompts") or []) if support_prompts is None else list(support_prompts)
    try:
        lay = lp_measure(rgb, depth_img, K, E, prompts, floor_z, pl_select)
    except Exception as e:  # noqa: BLE001  (a measurement error: no motion)
        lay = {"ok": False, "why": f"measure_error: {type(e).__name__}: {e}"}
    print("LAYER_MEASURE " + _mfj({k: v for k, v in lay.items() if k not in ("sides",)}), flush=True)
    print("LAYER_SIDES " + _mfj(lay.get("sides")), flush=True)
    if not lay.get("ok"):
        print("BINDING_STOP " + json.dumps({"status": "layer_not_found", "why": lay.get("why"), "select": pl_select}), flush=True); return None
    pf = lp_prefit(pud_pts, lay)
    print("LAYER_PREFIT " + _mfj(pf), flush=True)
    fpf = None   # C19I3 (C18I1's flat carry): the flat-carry prefit (None = the upright carry fits, lp_place as before)
    if not pf["ok"]:
        fpf = lpf_prefit(pud_pts, lay)
        print("LAYER_FLAT_PREFIT " + _mfj(fpf), flush=True)
        if not fpf["ok"]:
            print("LAYER_STOP " + json.dumps({"status": "object_taller_than_layer", **pf}), flush=True); return None
    p_obs, _ = tcp_from_obs(obs)
    obst = lp_obstacles(depth_img, K, E, floor_z, pud_mask, pud_pts, lay, np.asarray(p_obs, float))
    strategies = [{"use_topdown": True, "yaw": 0, "grasp_z": 0.015}, {"use_topdown": True, "yaw": 0, "grasp_z": 0.005}, {"use_topdown": False},
                  {"use_topdown": True, "yaw": 90, "grasp_z": 0.015}, {"use_topdown": True, "yaw": 0, "grasp_z": 0.010}, {"use_topdown": False}]
    _gplan = mf_plan_grasp(pud_pts, E, "sg1")
    if _gplan["mode"] != "infeasible":
        strategies[0] = {"use_topdown": True, "yaw": int(_gplan["yaw_order"][0]), "grasp_z": float(_gplan["target"][2]), "grasp_xy": [float(_gplan["target"][0]), float(_gplan["target"][1])]}
    grasp_ok = False; gpos = gquat = None; rest = None
    for i, strat in enumerate(strategies):
        obs = get_observation(); cam = obs["agentview"]; rgb = cam["images"]["rgb"]; depth = cam["images"]["depth"]; depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth; K = cam["intrinsics"]; E = cam["pose_mat"]
        pud_c2, pud_pts2, pud_mask2 = vb_localize_for_grasp(rgb, depth_img, K, E, pud_prompts, object_side)
        if pud_c2 is None:
            if gpos is not None and e1_hold_evidence():
                grasp_ok = True
            else:
                print("GRASP_ABORT reason=object_not_localized_no_hold_evidence", flush=True)
            break
        if pud_c2[0] < 0 or pud_c2[0] > 1.0:
            continue
        ok, gpos, gquat = attempt_grasp_verify(depth, K, pud_mask2, E, pud_c2, **strat)
        rest = (pud_pts2, None if gpos is None else np.asarray(gpos, float), None if gquat is None else np.asarray(gquat, float))   # C16I4: the body at rest before this grasp (carry reference)
        print(f"Attempt {i+1} ({'TD' if strat.get('use_topdown') else 'GN'} y={strat.get('yaw', '?')} z={strat.get('grasp_z', '?')}): {'OK' if ok else 'FAIL'}")
        if ok:
            grasp_ok = True; break
        if _E1_LAST.get("stop"):
            print("GRASP_ABORT reason=strategy_loop_stopped_after_lift_nonarrival", flush=True); break
    if not grasp_ok or gpos is None:
        print("All grasp attempts failed!"); return None
    if fpf is not None:   # C19I3 (C18I1's flat carry): the upright carry does not fit, the flat carry does
        xy = lpf_place(lay, obst, pud_prompts, gquat, fpf, sg="sg1")
    else:
        xy = lp_place(lay, obst, pud_prompts, gquat, sg="sg1", rest=rest if (rest is not None and rest[1] is not None and rest[2] is not None) else None)
    print("Done!")
    return None if xy is None else (float(xy[0]), float(xy[1]))
# ===== C19I3 place_primitive: in_layer lay-flat carry with a same-loading hold reference and the wrist's clearance (N19 group B2, slot chunk19_iter3) =====
# Carries C18I1's lay-flat carry (N18 chunk18_iter1, staged, not adopted) with the two defects its development rows exposed (DIAGNOSIS.md):
# A thin slab standing on its narrow end (standing height + 2 LP_MARGIN above the layer's free height, so lp_prefit refuses the
# upright carry) is inserted lying flat. The parent's top-down pinch closes across the slab's thin axis; the hand is first turned
# about the vertical so that the closing axis lies along the insertion axis u, then turned by 90 deg about the horizontal axis
# perpendicular to the closing axis (tool z -> u), in LP_TILT_MAX steps at the top of the pre-insert column. In the flat pose the hand's
# wide axis (CA_HAND body_half_wide) is vertical, so the hand stays outside the face (lp_plan's tcp_depth_max rule) and its lowest corner
# bounds the TCP from below (CA_FLOOR_CLEAR over the measured floor). The flat body is the lift measurement in the gripper frame (rigid
# pinch), extended by the closed opening and by the measurement taken after the turn. Everything after the plan is lp_place's sequence.
# C19I3 changes (DIAGNOSIS.md 3-5):
# (a) Hold reference. The public opening is ONE finger's position (contact_verifier: finger qpos / 0.04). Once the closing axis turns
#     towards the vertical, the held slab's weight shifts the finger pair along it (the pad gap is unchanged), so the one-finger reading
#     moves by up to 0.033 between the upright grasp and the flat pose (recorded on all three recorded turns; the slab stayed in the pads),
#     against verify_hold's HOLD_TOL 0.03 measured from the upright grasp. verify_hold is therefore applied, unchanged (HOLD_TOL, the
#     collapse floor, the LT2 min_opening), between readings taken under the same loading: against the grasp reference before the turn
#     (upright), between consecutive turn steps (a lost slab collapses the opening at that step), and, after the turn, against the
#     reading at the end of the turn (the flat-pose reference, used by the hold check before the insertion). After the turn the held
#     body must also be measured in the hand (measure_held_object), else it stops with the hand where it is.
# (b) Wrist clearance. lp_plan's swept checks cover the held body and the two hand boxes (CA_HAND); the arm's wrist housing behind the
#     hand (rigid to the hand) was not checked and stalled the lowering on a neighbouring standing body. In the flat plan the wrist is a
#     third hand box (tool z from the hand body's top back to the last arm joint, LPF_WRIST_BACK, grown by LPF_WRIST_R; half-width
#     LPF_WRIST_R across) inside lp_plan's column / sweep checks (ca_clearance, CA_CLEAR_MARGIN), and the release descent and the back-off
#     are checked for the wrist box at CA_DESCENT_STEP; a blocked candidate is skipped (the next closing-axis sign / flat yaw).
# (c) Candidate choice. C18I1 executed the first executable (sign, flat yaw); with the wrist box the centred approach can be blocked for
#     one yaw and not for another, so among the executable flat yaws of the first sign that has one, the plan with the smallest lateral
#     shift from the opening's centre (lp_plan's own nominal) is executed (ties: psi 0, +LP_TILT_MAX, -LP_TILT_MAX).
# Reachable only from lp_run_subgoal when lp_prefit refuses and lpf_prefit accepts. Margins LP_MARGIN, LP_MIN_TRAVEL, CA_FLOOR_CLEAR,
# CA_CLEAR_MARGIN, CA_HAND, HOLD_TOL and every budget unchanged.
# Candidate order: closing axis ending downward first (sign -1), then the flat yaw psi = 0, +-LP_TILT_MAX (the hand body and the wrist,
# which trail the TCP along -u, swing sideways past a body standing behind the hand; the slab still enters through the measured opening).
LPF_SIGNS = (-1.0, 1.0)
LPF_OPEN_FULL = 2.0 * (AC_OPEN_HALF_WIDE - CA_HAND["finger_half_wide"])   # m, the gripper opening at fraction 1.0 (8 cm)
LPF_WRIST_BACK = 0.207   # m, C19I3: the last arm joint (link 7 origin) behind the TCP along the tool axis: Panda flange 0.107 + the TCP 0.10 ahead of the hand base (motion interface TCP_OFFSET_HAND)
LPF_WRIST_R = 0.06       # m, C19I3: wrist housing half-width about the tool axis: p99 radial extent of the robot's own depth points around that axis (public frames, free-air poses; DIAGNOSIS 5)
_LPF_HOLD = {"ref": None}   # C19I3: the flat-pose opening reference (the reading at the end of the turn)


def lpf_thin(obj_pts):
    """Thin axis of a standing body from its top band (points within CONTAINER_PLANE_BIN of its CONTAINER_PCT_HI height):
    (thickness, horizontal thin axis, standing height) from the band's minimum-area rectangle; None if the band is too small."""
    P = np.asarray(obj_pts, float); z = P[:, 2]; top = float(np.percentile(z, CONTAINER_PCT_HI)); h = top - float(np.percentile(z, CONTAINER_PCT_LO))
    T = P[z >= top - CONTAINER_PLANE_BIN]
    if len(T) < CONTAINER_SECTOR_MIN_PTS:
        return None
    _c, _a, b, ha, hb = _ca_min_rect(T[:, :2])
    return 2.0 * float(hb), np.array([float(b[0]), float(b[1]), 0.0]), h


def lpf_tcp_interval(lay, half_t):
    """Flat carry: (lowest, highest) TCP of the insertion. Lowest = max(layer floor + half thickness + LP_MARGIN, measured floor +
    CA_FLOOR_CLEAR + the hand body's half width, which is vertical in this pose); highest = cover - LP_MARGIN - half thickness."""
    fl = max(float(lay["z_floor"]), float(lay["floor_z"]))
    lo_body = fl + float(half_t) + LP_MARGIN; lo_hand = float(lay["floor_z"]) + CA_FLOOR_CLEAR + CA_HAND["body_half_wide"]
    return max(lo_body, lo_hand), float(lay["z_cover"]) - LP_MARGIN - float(half_t), lo_hand


def lpf_prefit(obj_pts, lay):
    """Before motion, when the upright carry does not fit: the flat carry fits if the body's thickness + 2 LP_MARGIN fits the free
    height, the flat TCP interval (lpf_tcp_interval) holds LP_MIN_TRAVEL, and the standing height (the flat length along u) is shorter
    than the layer depth - LP_MARGIN."""
    th = lpf_thin(obj_pts)
    if th is None:
        return {"ok": False, "why": "thin_axis_unmeasured"}
    t, b, h = th; lo, hi, lo_hand = lpf_tcp_interval(lay, 0.5 * t)
    rec = {"thickness": round(t, 4), "standing_height": round(h, 4), "thin_axis": [round(float(b[0]), 3), round(float(b[1]), 3)], "tcp_lo": round(lo, 4), "tcp_hi": round(hi, 4), "tcp_lo_hand": round(lo_hand, 4),
           "free": round(float(lay["z_cover"]) - float(lay["z_floor"]), 4), "depth": lay.get("depth")}
    if t + 2.0 * LP_MARGIN > float(lay["z_cover"]) - float(lay["z_floor"]):
        return {**rec, "ok": False, "why": "thicker_than_layer"}
    if hi < lo + LP_MIN_TRAVEL:
        return {**rec, "ok": False, "why": "hand_does_not_fit_flat"}
    if h > float(lay["depth"]) - LP_MARGIN:
        return {**rec, "ok": False, "why": "longer_than_layer_depth"}
    return {**rec, "ok": True, "why": None}


def _lpf_q(R):
    x = Rotation.from_matrix(R).as_quat(); return np.array([x[3], x[0], x[1], x[2]])


def lpf_orientations(q_hold, u, sign, psi=0.0):
    """(q_up, q_flat, yaw_deg): q_up = q_hold turned about the vertical so that the closing axis (commanded tool y) points along
    sign * u_psi (u_psi = u turned by psi deg about the vertical); q_flat = the hand with tool z = u_psi and tool y = sign * z
    (q_up turned by 90 deg about the horizontal axis perpendicular to the closing axis: -z -> u_psi, u_psi -> z)."""
    Rh = _R(np.asarray(q_hold, float)); c = np.array([Rh[0, 1], Rh[1, 1]]); u2 = np.asarray(u, float)[:2] / float(np.linalg.norm(np.asarray(u, float)[:2]))
    cp, sp = np.cos(np.radians(float(psi))), np.sin(np.radians(float(psi))); u2 = np.array([cp * u2[0] - sp * u2[1], sp * u2[0] + cp * u2[1]])
    tgt = float(sign) * u2
    yaw = float(np.degrees(np.arctan2(tgt[1], tgt[0]) - np.arctan2(c[1], c[0]))); yaw = (yaw + 180.0) % 360.0 - 180.0
    q_up = _yaw_about_vertical(np.asarray(q_hold, float), yaw)
    z_t = np.array([u2[0], u2[1], 0.0]); y_t = np.array([0.0, 0.0, float(sign)]); x_t = np.cross(y_t, z_t)
    return q_up, _lpf_q(np.stack([x_t, y_t, z_t], 1)), yaw


def lpf_turn_steps(q_up, q_flat):
    """Intermediate orientations of the turn (slerp, LP_TILT_MAX per step; the last one = q_flat)."""
    a = Rotation.from_quat(np.asarray(q_up, float)[[1, 2, 3, 0]]); b = Rotation.from_quat(np.asarray(q_flat, float)[[1, 2, 3, 0]])
    ang = float(np.degrees((b * a.inv()).magnitude())); n = max(1, int(np.ceil(ang / LP_TILT_MAX - 1e-9)))
    from scipy.spatial.transform import Slerp
    s = Slerp([0.0, 1.0], Rotation.concatenate([a, b]))
    out = []
    for k in range(1, n + 1):
        x = s([k / n]).as_quat()[0]; out.append(np.array([x[3], x[0], x[1], x[2]]))
    return out


def lpf_body(rel_g, t_open, extra=None):
    """Flat body model in the gripper frame: the lift-measured points (rigid pinch) plus their copies at both pad faces along the
    closing axis (+-t_open/2: the pads touch both faces), plus extra (measured after the turn). Returns (points, centre offset, info)."""
    P = np.asarray(rel_g, float); faces = []
    for s in (-1.0, 1.0):
        F = P.copy(); F[:, 1] = s * 0.5 * float(t_open); faces.append(F)
    B = np.concatenate([P] + faces + ([np.asarray(extra, float)] if extra is not None and len(extra) else []), 0)
    lo = np.percentile(B, CONTAINER_PCT_LO, axis=0); hi = np.percentile(B, CONTAINER_PCT_HI, axis=0); c = 0.5 * (lo + hi)
    return B, c, {"extent_g": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(lo, hi)], "n": int(len(B))}


def lpf_wrist_box():
    """C19I3: the wrist housing as a box in the hand model frame (x = closing axis, z = tool axis): from the hand body's top back to the
    last arm joint, grown by LPF_WRIST_R along the tool axis; LPF_WRIST_R across."""
    return ((-LPF_WRIST_R, LPF_WRIST_R), (-LPF_WRIST_R, LPF_WRIST_R), (-(LPF_WRIST_BACK + LPF_WRIST_R), CA_HAND["body_top"]))


def lpf_wrist_clear(obst, tcp_list, q):
    """C19I3: (ok, n_conflicts, first blocked TCP) of the wrist box (CA_CLEAR_MARGIN) at the given TCPs, hand orientation q."""
    if obst is None or len(obst) == 0:
        return True, 0, None
    Rm = lp_hand_model_R(q); (x0, x1), (y0, y1), (z0, z1) = lpf_wrist_box(); m = CA_CLEAR_MARGIN; n = 0; first = None
    for p in tcp_list:
        P = (np.asarray(obst, float) - np.asarray(p, float)) @ Rm
        k = int(((P[:, 0] >= x0 - m) & (P[:, 0] <= x1 + m) & (P[:, 1] >= y0 - m) & (P[:, 1] <= y1 + m) & (P[:, 2] >= z0 - m) & (P[:, 2] <= z1 + m)).sum())
        if k and first is None:
            first = [round(float(v), 4) for v in p]
        n += k
    return n == 0, n, first


def lpf_release_path(pl, lay):
    """C19I3: the TCPs of the release descent (fin at z_ins down to z_release, CA_DESCENT_STEP) and of the back-off (CA_BACKOFF along -u)."""
    fin = np.asarray(pl["fin"], float); zr = float(pl["z_release"]); u = np.asarray(lay["u"], float)
    zs = list(np.arange(float(fin[2]), zr, -CA_DESCENT_STEP)) + [zr]
    out = [np.array([fin[0], fin[1], z]) for z in zs]
    return out + [np.array([fin[0] - u[0] * CA_BACKOFF, fin[1] - u[1] * CA_BACKOFF, zr])]


def lpf_plan(lay, B, c_g, q_flat, obst, z_transport, pref=None):
    """lp_plan on the flat body: drop = the body's depth below the TCP; the layer floor raised so that lp_plan's lowest TCP equals the
    flat interval's lowest TCP (the hand's lowest corner CA_FLOOR_CLEAR over the measured floor). C19I3: lp_plan's hand checks
    (ca_clearance over _ca_hand_boxes) include the wrist box while this plan is made (restored afterwards)."""
    P = np.asarray(B, float) @ _R(np.asarray(q_flat, float)).T
    drop = max(-float(np.percentile(P[:, 2], CONTAINER_PCT_LO)), 0.0); half_t = max(drop, float(np.percentile(P[:, 2], CONTAINER_PCT_HI)))
    lo, hi, lo_hand = lpf_tcp_interval(lay, half_t)
    lay_f = dict(lay); fl = max(float(lay["z_floor"]), float(lay["floor_z"]))
    if lo_hand > fl + drop + LP_MARGIN:
        lay_f["z_floor"] = lo_hand - drop - LP_MARGIN + 1e-6
    g = globals(); base = g["_ca_hand_boxes"]
    g["_ca_hand_boxes"] = lambda: list(base()) + [lpf_wrist_box()]
    try:
        pl = lp_plan(lay_f, B, c_g, q_flat, drop, obst, z_transport, None, pref=pref)
    finally:
        g["_ca_hand_boxes"] = base
    pl.update(flat=True, flat_drop=round(drop, 4), flat_tcp_lo_hand=round(lo_hand, 4))
    return pl


def lpf_hold(ref, now):
    """C19I3: verify_hold unchanged (HOLD_TOL, the collapse floor, the held grasp's LT2 min_opening as e1_hold_evidence passes it)
    between two readings taken under the same loading; uncertain without a reference."""
    if ref is None:
        return {"status": "uncertain", "reason": "no reference opening"}
    lmo = _E1_LAST.get("min_opening")
    return verify_hold(ref, now) if lmo is None else verify_hold(ref, now, min_opening=lmo)


def lpf_place(lay, obst, held_prompts, gquat, pf, sg="sg1"):
    """Lift measurement -> flat plan (both closing-axis signs, IK-probed) -> raise, transport (upright, yawed) -> turn in LP_TILT_MAX
    steps at the top of the pre-insert column -> hold check, measurement, re-plan -> lp_place's lowering / insertion / release."""
    meas = []
    for k in range(2):
        obs = get_observation(); _c_stash_view(obs); m = measure_held_object(held_prompts, obs); meas.append(m)
    ds = [m["drop_below_tcp"] for m in meas if m is not None and PRIM["drop_bounds"][0] <= m["drop_below_tcp"] <= PRIM["drop_bounds"][1]]
    rel, off_g, hinfo = lp_held(meas)
    g_open = gripper_opening(); t_open = float(g_open) * LPF_OPEN_FULL if g_open is not None else None
    print("LAYER_FLAT_HELD " + _mfj({"sg": sg, "n_valid": len(ds), "t_open": None if t_open is None else round(t_open, 4), "t_rest": pf.get("thickness"), **hinfo}), flush=True)
    if rel is None or not ds or t_open is None:
        return _lp_stop(sg, "flat_held_body_unmeasured", gquat)
    if lp_carry_frame(gquat) is None or abs(t_open - float(pf["thickness"])) > LP_MARGIN:   # the pads must close across the measured thin axis
        return _lp_stop(sg, "flat_grasp_not_across_thin_axis", gquat, t_open=round(t_open, 4))
    p_now, q_now, _ = read_tcp(); u = np.asarray(lay["u"], float)
    B, c_g, binfo = lpf_body(rel, t_open)
    m_ref = max([m for m in meas if m is not None], key=lambda q_: q_["n_pts"]); obj_xy = np.asarray(m_ref["axis_xy"], float)
    choice = None; tried = []; ok_c = []
    for sign, psi in [(s_, p_) for s_ in LPF_SIGNS for p_ in (0.0, LP_TILT_MAX, -LP_TILT_MAX)]:
        if ok_c and ok_c[0][0] != sign:   # C19I3: the first closing-axis sign with an executable candidate is kept
            break
        q_up, q_flat, yaw = lpf_orientations(q_now, u, sign, psi)
        try:
            pl0 = lpf_plan(lay, B, c_g, q_flat, obst, float(p_now[2]))
        except Exception as e:  # noqa: BLE001  (a planning error: this sign is not used)
            pl0 = {"ok": False, "why": f"plan_error: {type(e).__name__}: {e}"}
        if not pl0.get("pre"):
            tried.append({"sign": sign, "psi": psi, "why": pl0.get("why")}); continue
        z_tr, _ci = c_transport_height(float(p_now[2]), float(lay["floor_z"]), obj_xy, np.asarray(pl0["pre"][:2], float), 0.5 * float(np.ptp(np.asarray(rel, float)[:, 0])), float(np.median(ds)), sg=sg)
        z_hi = max(float(z_tr), float(p_now[2]))
        try:
            pl = lpf_plan(lay, B, c_g, q_flat, obst, z_hi)
        except Exception as e:  # noqa: BLE001
            pl = {"ok": False, "why": f"plan_error: {type(e).__name__}: {e}"}
        if not pl.get("ok"):
            tried.append({"sign": sign, "psi": psi, "why": pl.get("why")}); continue
        pre = np.asarray(pl["pre"], float); fin = np.asarray(pl["fin"], float); top = np.array([pre[0], pre[1], z_hi])
        steps = lpf_turn_steps(np.asarray(q_now, float), q_up) + lpf_turn_steps(q_up, q_flat)   # yaw in place, then the 90 deg turn
        probes = [("pre_high", top, np.asarray(q_now, float))] + [(f"turn{k}", top, qq) for k, qq in enumerate(steps)] + [("pre_low", pre, q_flat), ("fin", fin, q_flat), ("release", np.array([fin[0], fin[1], float(pl["z_release"])]), q_flat)]
        bad = [lab for lab, pos, qq in probes if not mf_ik_probe(pos, qq)]
        if bad:
            tried.append({"sign": sign, "psi": psi, "why": "ik_" + bad[0]}); continue
        w_ok, w_n, w_at = lpf_wrist_clear(obst, lpf_release_path(pl, lay), q_flat)   # C19I3: the wrist over the release descent and the back-off
        if not w_ok:
            tried.append({"sign": sign, "psi": psi, "why": "wrist_release_path_blocked", "n": w_n, "at": w_at}); continue
        ok_c.append((sign, q_up, q_flat, yaw, pl, z_hi, steps, psi))
    if ok_c:   # C19I3: among the flat yaws of that sign, the plan whose slab lies nearest the opening's lateral centre (lp_plan's own first preference; ties: psi 0, +, -)
        choice = min(ok_c, key=lambda c_: abs(float(c_[4].get("lateral_shift") or 0.0)))
    print("LAYER_FLAT_PLAN " + _mfj({"sg": sg, "tried": tried, "body": binfo, **({} if choice is None else {"sign": choice[0], "psi_deg": choice[7], "yaw_deg": round(choice[3], 2), "n_turn_steps": len(choice[6]), **choice[4]})}), flush=True)
    if choice is None:
        return _lp_stop(sg, "flat_" + str((tried[-1] if tried else {}).get("why")), gquat)
    sign, q_up, q_flat, yaw, pl, z_hi, steps, psi = choice; pre = np.asarray(pl["pre"], float)
    if z_hi > float(p_now[2]) + 0.005:
        rr = move_to_pose(np.array([p_now[0], p_now[1], z_hi]), np.asarray(q_now, float), mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:layer_raise"); _PRIM_STATS["motion_calls"] += 1
        if not rr["reached"]:
            return _lp_stop(sg, "raise_" + rr["status"], gquat)
    tr = move_to_pose(np.array([pre[0], pre[1], z_hi]), np.asarray(q_now, float), mode="linear", tol_pos=0.015, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_transport"); _PRIM_STATS["motion_calls"] += 1
    if tr["status"] == "unreachable":
        tr = move_to_pose(np.array([pre[0], pre[1], z_hi]), np.asarray(q_now, float), mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:layer_transport_free"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "transport", "status": tr["status"], "steps": tr.get("steps_used")}), flush=True)
    if not tr["reached"]:
        return _lp_stop(sg, "transport_" + tr["status"], q_now)
    q = np.asarray(q_now, float); _LPF_HOLD["ref"] = None
    g_prev = gripper_opening(); hv = lpf_hold(_E1_LAST.get("grasp_after"), g_prev)   # C19I3: upright, against the grasp reference (same loading)
    print("LAYER_FLAT_HOLD " + _mfj({"sg": sg, "where": "before_turn", "ref": _E1_LAST.get("grasp_after"), "now": g_prev, "status": hv.get("status")}), flush=True)
    if hv.get("status") != "verified":
        return _lp_stop(sg, "hold_lost_before_flat_turn", q, hold=hv.get("status"))
    for k, qq in enumerate(steps):   # the turn in place at the top of the pre-insert column (clear of every structure)
        p_b, _, _ = read_tcp()
        rt = move_to_pose(np.asarray(p_b, float), qq, mode="free", tol_pos=PRIM["reach_tol"], step_budget=BUDGET_FREE, label=f"{sg}:layer_flat_turn{k}"); _PRIM_STATS["motion_calls"] += 1
        print("LAYER_MOVE " + _mfj({"sg": sg, "move": f"flat_turn{k}", "status": rt["status"], "steps": rt.get("steps_used"), "tool_tilt_deg": round(lp_tool_tilt(qq), 2)}), flush=True)
        if not rt["reached"]:
            return _lp_stop(sg, f"flat_turn{k}_" + str(rt["status"]), q)
        q = qq
        g_now = gripper_opening(); hv = lpf_hold(g_prev, g_now)   # C19I3: consecutive turn steps (a lost slab collapses the opening at that step)
        print("LAYER_FLAT_HOLD " + _mfj({"sg": sg, "where": f"turn{k}", "ref": g_prev, "now": g_now, "status": hv.get("status")}), flush=True)
        if hv.get("status") != "verified":
            return _lp_stop(sg, f"hold_lost_in_flat_turn{k}", q, hold=hv.get("status"))
        g_prev = g_now
    _LPF_HOLD["ref"] = g_prev   # C19I3: the flat-pose reference
    extra = []; p_t, _, _ = read_tcp(); n_seen = 0
    for k in range(2):   # the flat body as now carried (its visible face; its other face lies t_open below / above along the closing axis)
        obs = get_observation(); _c_stash_view(obs); m = measure_held_object(held_prompts, obs)
        if m is not None:
            n_seen += 1
            X = (np.asarray(m["pts"], float) - np.asarray(m["tcp_p"], float)) @ _R(np.asarray(m["tcp_q"], float))
            s_y = float(_R(np.asarray(m["tcp_q"], float))[2, 1])   # world-vertical component of the closing axis (+-1 in the flat pose)
            X2 = X.copy(); X2[:, 1] = X2[:, 1] - s_y * float(t_open)   # the visible (upper) face moved down by the pad opening = the lower face
            extra += [X, X2]
    if n_seen == 0:   # C19I3: the held body must be seen in the hand after the turn
        return _lp_stop(sg, "flat_held_body_unseen_after_turn", q)
    B2, c2, binfo2 = lpf_body(rel, t_open, np.concatenate(extra, 0) if extra else None)
    try:
        pl2 = lpf_plan(lay, B2, c2, q_flat, obst, z_hi, pref=(pl.get("s_lat"), pl.get("z_ins")))
    except Exception as e:  # noqa: BLE001
        pl2 = {"ok": False, "why": f"plan_error: {type(e).__name__}: {e}"}
    print("LAYER_FLAT_PLAN " + _mfj({"sg": sg, "where": "after_turn", "n_meas_pts": int(sum(len(x) for x in extra) // 2), "body": binfo2, **pl2}), flush=True)
    if not pl2.get("ok"):
        return _lp_stop(sg, "flat_" + str(pl2.get("why")), q)
    w_ok, w_n, w_at = lpf_wrist_clear(obst, lpf_release_path(pl2, lay), q)   # C19I3
    if not w_ok:
        print("LAYER_FLAT_WRIST " + _mfj({"sg": sg, "where": "after_turn", "n": w_n, "at": w_at}), flush=True); return _lp_stop(sg, "flat_wrist_release_path_blocked", q)
    pl = pl2; pre = np.asarray(pl["pre"], float); fin = np.asarray(pl["fin"], float)
    for lab, pos in (("pre_high", [pre[0], pre[1], z_hi]), ("pre_low", pre), ("fin", fin)):
        if not mf_ik_probe(pos, q):
            print("LAYER_IK " + _mfj({"sg": sg, "pose": lab, "ok": False}), flush=True); return _lp_stop(sg, "ik_" + lab, q)
    if float(np.hypot(*(np.asarray(p_t, float)[:2] - pre[:2]))) > 0.002:
        sh = move_to_pose(np.array([pre[0], pre[1], z_hi]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_column_shift"); _PRIM_STATS["motion_calls"] += 1
        print("LAYER_MOVE " + _mfj({"sg": sg, "move": "column_shift", "status": sh["status"], "steps": sh.get("steps_used")}), flush=True)
        if not sh["reached"]:
            return _lp_stop(sg, "column_shift_" + sh["status"], q)
    return lpf_insert_release(lay, pl, q, sg)


def lpf_insert_release(lay, pl, q, sg):
    """lp_place's sequence from the top of the pre-insert column: lowering, hold check, planned linear insertion, anchored contact
    descent, gated release, back-off along -u, safe_retreat (unchanged rules)."""
    pre = np.asarray(pl["pre"], float); fin = np.asarray(pl["fin"], float)
    lw = move_to_pose(pre, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_lower"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "lower", "status": lw["status"], "steps": lw.get("steps_used")}), flush=True)
    if not lw["reached"]:
        return _lp_stop(sg, "lower_" + lw["status"], q)
    ref = _LPF_HOLD["ref"] if _LPF_HOLD["ref"] is not None else _E1_LAST.get("grasp_after")   # C19I3: the flat-pose reference (same loading)
    g_now = gripper_opening(); hv = lpf_hold(ref, g_now)
    print("LAYER_FLAT_HOLD " + _mfj({"sg": sg, "where": "before_insert", "ref": ref, "now": g_now, "status": hv.get("status")}), flush=True)
    if hv.get("status") != "verified":
        return _lp_stop(sg, "hold_lost_before_insert", q, hold=hv.get("status"))
    pin = move_to_pose(fin, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, plan_only=True, label=f"{sg}:layer_insert_plan")
    if pin.get("detail") != "plan_only":
        print("LAYER_MOVE " + _mfj({"sg": sg, "move": "insert_plan", "status": pin["status"], "detail": pin.get("detail")}), flush=True); return _lp_stop(sg, "insert_unplannable", q)
    ins = move_to_pose(fin, q, mode="linear", tol_pos=PRIM["reach_tol"], corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_insert"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "insert", "status": ins["status"], "detail": ins.get("detail"), "steps": ins.get("steps_used"), "lateral_cm": round(100.0 * float(ins.get("lateral_dev_max") or 0.0), 2)}), flush=True)
    if not ins["reached"]:
        p_b, _, _ = read_tcp(); move_to_pose(np.array([pre[0], pre[1], float(p_b[2])]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_withdraw"); _PRIM_STATS["motion_calls"] += 1
        return _lp_stop(sg, "insert_" + ins["status"], q)
    p_in = np.asarray(ins.get("final_tcp") if ins.get("final_tcp") is not None else read_tcp()[0], float)
    anc = lp_release_anchor(pl, lay, p_in)
    print("LAYER_ANCHOR " + _mfj({"sg": sg, **anc}), flush=True)
    if not anc["ok"]:
        move_to_pose(np.array([pre[0], pre[1], float(p_in[2])]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_withdraw"); _PRIM_STATS["motion_calls"] += 1
        return _lp_stop(sg, anc["why"], q)
    dsc = mc_place_descend(np.asarray(anc["xy"], float), q, float(pl["z_release"]), float(p_in[2]), sg)
    _rd = dsc.get("mc_result"); ok_rel, why_rel = gate_allow(_rd, "release")
    if not ok_rel:
        _mark(ev="release_blocked", sg=sg, reason=why_rel)
        p_b, _, _ = read_tcp(); move_to_pose(np.array([pre[0], pre[1], float(p_b[2])]), q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_TRANSPORT, label=f"{sg}:layer_withdraw"); _PRIM_STATS["motion_calls"] += 1
        return _lp_stop(sg, "release_blocked", q, gate=why_rel)
    _mark(ev="release_start", sg=sg)
    open_gripper(); gate_record("release"); _PRIM_STATS["motion_calls"] += 1
    print("RELEASE_VERIFY " + _json.dumps({"sg": sg, **verify_release(gripper_opening())}), flush=True)
    for _ in range(5):
        get_observation()
    p_r, _, _ = read_tcp(); back = np.asarray(p_r, float) - np.array([float(lay["u"][0]), float(lay["u"][1]), 0.0]) * CA_BACKOFF
    rb = move_to_pose(back, q, mode="linear", tol_pos=0.015, corridor=SEGMENT_CORRIDOR, step_budget=BUDGET_LINEAR, label=f"{sg}:layer_backoff"); _PRIM_STATS["motion_calls"] += 1
    print("LAYER_MOVE " + _mfj({"sg": sg, "move": "backoff", "status": rb["status"]}), flush=True)
    safe_retreat(sg=sg, q_hold=q)
    print("PLACE_RESULT " + _json.dumps({"sg": sg, "status": "released", "reason": "inserted_flat_and_released", "relation": "in_layer"}), flush=True)
    return [float(v) for v in pl["body_centre_xy"]]
# ===== end C19I3 =====
# ===== end C15I3 place_primitive (C15I5, C16I4) =====



# ===== C17I2 task_driver / language_adapter / binding: the carry order of a placed pair (N17 group M, slot chunk17_iter2) =====
# A clause "<relation> the A on the B" followed by an anaphora clause that names BOTH bodies with the plural pronoun ("... and
# place them in the C"; the adapter's anaphora template copies only the previous OBJECT, A, into the second subgoal) asks for
# the final state {A on B, the pair in C}. Executed in language order the program first puts A on B and then carries A alone
# to C: A leaves B (the first relation is undone) and B never reaches C. Recorded (DIAGNOSIS.md section 1, 14 development
# rows of the two tasks that have this form, N11, N13, N16): the first relation was placed in 14/14; the second subgoal then
# tried to pinch the upper body of the pair, which rests tilted 24-41 deg on the lower one (truth), and failed 14/14 (empty
# closes 0.008-0.041, holds lost at the lift, premature contacts, IK-box refusals); even a verified lift of A would have broken
# the first relation. The goal relations recorded by the runner were {A on B, B in C} in every row.
# Rule (sc_carry_order): when an "on" subgoal whose reference B is a carriable body (a phrase of the adapter's "objects"
# vocabulary, i.e. something the program can grasp, with a side code the object binding implements) is followed by a
# placement anaphora subgoal (refers = object, relation in SC_CARRY_RELATIONS) whose clause holds the plural pronoun, the two
# subgoals are executed as (1) B -> C (the anaphora's relation and reference; B bound by its own phrase and side code, exactly
# as an object), then (2) A on B, where B is the body subgoal (1) placed (support_track). The final state is the same set of
# relations the language asks for; the pair is never carried. Every other parse is returned unchanged (same list object).
# Support of (2) (sc_tracked_support): after (1) the side word no longer names B (it was moved), so B is found where (1)
# placed it: the instances of B's prompts (bind_instances, the C3I1 instance rule), each passing the unchanged K1 gate
# (support_candidate_ok: not the bound object by SUPPORT_SAME_MASK overlap, raised by SUPPORT_RISE_MIN, top <= SUPPORT_TOP_MAX
# over the floor), the nearest one whose OBB centre lies within its own footprint radius of the placed xy (the radius
# run_subgoal already measures for a support, pl_r: a body whose centre is farther than its own radius from the target does
# not cover the target, so it is not the body placed there). The 0.06 m OBB-centre search bound of localize_support_by_side is
# not used on this path: it filters an open search over the table, while here the identity is carried; a body resting on a
# container floor (measured 2.1 cm above the table, DIAGNOSIS.md section 3) has its centre raised by that floor. The K1 top
# bound still applies. No candidate -> support_not_found (no motion). If (1) did not place B, (2) does not move.
# No new numeric constant; no guard, threshold or budget change; no vb_* or adapter change.
SC_CARRY_RELATIONS = ("in", "on")   # placement relations run by run_subgoal that an object anaphora can name
SC_PLURAL = ("them",)               # the pronoun that names both bodies of the previous relation
_SC_XY = {}                         # subgoal index -> placed xy (run(); support_track of a later subgoal)


def sc_carry_order(td, sgs):
    """(1) B -> C, (2) A on the placed B for an "on" subgoal followed by a plural object-anaphora placement; else sgs unchanged."""
    if not sgs or len(sgs) < 2:
        return sgs
    objs = (td or {}).get("objects") or {}; out = list(sgs); i = 0; changed = False
    while i + 1 < len(out):
        a, b = out[i], out[i + 1]
        side = a.get("side")
        if (a.get("relation") == "on" and b.get("refers") == "object" and b.get("relation") in SC_CARRY_RELATIONS
                and a.get("support") in objs and a.get("support_prompts") and b.get("object") == a.get("object")
                and len(a.get("sides") or []) <= 1 and (side is None or str(side) in LA_OBJECT_SIDES)
                and any(t in SC_PLURAL for t in str(b.get("clause") or "").split())):
            carry = dict(b); carry.update(object=a["support"], object_prompts=list(a["support_prompts"]), object_side=side, refers=None, carry_of=i + 1)
            pair = dict(a); pair.update(sides=[], side=None, support_track=True)
            out[i:i + 2] = [carry, pair]; changed = True; i += 2
        else:
            i += 1
    return out if changed else sgs


def sc_tracked_support(rgb, depth_img, K, E, prompts, track_xy, exclude_mask):
    """Support of a pair subgoal: the K1-gated instance of the prompts nearest the placed xy, within its own footprint radius.
    Returns (OBB centre, points, mask) or (None, None, None). Logs SC_TRACK_SUPPORT (+ BIND_SUPPORT on acceptance)."""
    rec = {"track_xy": None if track_xy is None else [round(float(v), 4) for v in np.asarray(track_xy, float)[:2]], "prompts": list(prompts)}
    if track_xy is None:
        rec["why"] = "carry_not_placed"; print("SC_TRACK_SUPPORT " + json.dumps(rec), flush=True)
        return None, None, None
    floor_z = support_floor_z(depth_img, K, E); _CG["floor_z"] = floor_z
    t = np.asarray(track_xy, float)[:2]; best = None; seen = []
    for I in bind_instances(rgb, depth_img, K, E, list(prompts)):
        pts = np.asarray(I["pts"], float); mask = np.asarray(I["mask"]).astype(np.uint8)
        if len(pts) < SIDE_MIN_PTS:
            continue
        obb = get_oriented_bounding_box_from_3d_points(pts); c = np.asarray(obb["center"], float)
        r = max(float(obb["extent"][0]), float(obb["extent"][1])) / 2.0
        d = float(np.hypot(*(c[:2] - t)))
        ok, why, feat = support_candidate_ok(pts, mask, exclude_mask, floor_z, relation="on_support")
        seen.append({"center": [round(float(v), 4) for v in c], "dist": round(d, 4), "r": round(r, 4), "gate": why, "support": round(float(I["support"]), 3)})
        if ok and d <= r and (best is None or d < best[0]):
            best = (d, c, pts, mask)
    rec.update(floor_z=None if floor_z is None else round(float(floor_z), 4), n_instances=len(seen), instances=seen[:6])
    if best is None:
        rec["why"] = "no_instance_at_placed_xy"; print("SC_TRACK_SUPPORT " + json.dumps(rec, default=str), flush=True)
        return None, None, None
    rec.update(why="tracked", center=[round(float(v), 4) for v in best[1]], dist=round(best[0], 4))
    print("SC_TRACK_SUPPORT " + json.dumps(rec, default=str), flush=True)
    print("BIND_SUPPORT " + json.dumps({"prompt": list(prompts), "n_instances": len(seen), "select": None, "center": rec["center"], "rule": "placed_track"}), flush=True)
    return best[1], best[2], best[3]
# ===== end C17I2 =====


# ===== C1I1 language_adapter: shared-template language parsing (round 20260924, chunk 1, iteration 1) =====
# The adapter (task_descriptors(), DATA) carries the shared "language_templates" (regex + relation name; flags two_objects /
# two_instances), the "anaphora_templates" (clause forms that refer to the previous clause; "refers" = object | support | part),
# the "side_words" (word -> side code) and the vocabularies "objects" / "supports" / "articulated" (phrase -> prompt list).
# The runtime language (env.handle.task_language) is parsed the way the adapter's data view was produced: whole-sentence
# templates first (in data order), then clause splitting at " and " with the same templates plus the anaphora templates.
# One rule is added so that the parse is consistent with the vocabulary: a template match is accepted only when every captured
# phrase resolves in the adapter data (a side word, an object, a support or a part) or is a single word. An open group that
# swallowed several clauses ("<object> on the ... and put the <object>") is not a vocabulary phrase, so the whole-sentence pass
# rejects such a match and the clause pass yields one subgoal per clause, in language order. Nothing here is task-specific:
# relation names, side codes and phrases are values read from the data; the only relation set written in code is the one the
# placement stack implements today (LA_SUPPORTED -> run_subgoal with relation_geometry / localize_support_by_side).
import re as _la_re
LA_SUPPORTED = {"on": ("min", "max", None, "zmax", "mid", "xmin"), "beside": ("min", "max", "xmin"), "in": (None,), "in_part": ("min", "max", "xmin", "xmax"), "in_sliding": ("zmin", "zmax"), "close_articulated": (None, "min", "max", "xmin", "xmax", "zmin", "zmax"), "actuate_on": (None,), "actuate_off": (None,), "open_articulated": (None, "min", "max", "xmin", "xmax", "zmin", "zmax"), "in_layer": LP_SIDES}   # C15I3: in_layer (lp_run_subgoal) with the layer side codes   # C5I1: open_articulated (run_open_articulated) with the close_articulated side codes   # relation -> side codes run_subgoal implements (None = no side word); C1I2: "in" without a side code (containment); C2I1: close_articulated (run_close_articulated) with an axis-extreme side code or none; C2I2: actuate_on (run_actuate) without a side code
LA_LOOKUP = (("side", "side_words"), ("object", "objects"), ("support", "supports"), ("part", "articulated"))
LA_OBJECT_SIDES = ("min", "max", "xmin", "xmax", "zmin", "zmax", "mid")   # C3I1: side codes localize_target implements for an object instance (bind_select_by_side); any other object_side value is refused at dispatch
LA_CONJ = " and "


def la_normalise(text):
    """Lower case, single spaces, no trailing punctuation (the templates are anchored at the end of the text)."""
    return " ".join(str(text if text is not None else "").lower().split()).strip(" .!?;,")


def la_lookup(td, word):
    """One captured group against the adapter data -> (kind, phrase, value) with kind in side | object | support | part | word,
    or None for an empty group. The lookup order is LA_LOOKUP (a side word before a vocabulary phrase)."""
    w = la_normalise(word)
    if not w:
        return None
    for kind, key in LA_LOOKUP:
        table = (td or {}).get(key) or {}
        if w in table:
            return kind, w, table[w]
    return "word", w, None


def la_phrase_in(td, key, clause):
    """Longest phrase of td[key] that occurs as whole words inside the clause (None when none does)."""
    hits = [str(p) for p in ((td or {}).get(key) or {}) if _la_re.search(r"(?<![a-z])" + _la_re.escape(str(p).lower()) + r"(?![a-z])", clause)]
    return max(hits, key=len) if hits else None


def la_qualified(td, w):
    """A multi-word group of the form "<object phrase> <qualifier>" where the qualifier holds exactly one side word and no other
    vocabulary phrase nor a conjunction (an instance named by its position) -> (object phrase, side code); else None."""
    objs = (td or {}).get("objects") or {}; sw = (td or {}).get("side_words") or {}
    heads = sorted((str(p) for p in objs if w.startswith(str(p) + " ")), key=len)
    if not heads:
        return None
    head = heads[-1]; rest = w[len(head):].strip(); toks = rest.split()
    sides = [str(sw[t]) for t in toks if t in sw]
    if len(sides) != 1 or LA_CONJ.strip() in toks or any(la_phrase_in(td, key, rest) is not None for key in ("objects", "supports", "articulated")):
        return None
    return head, sides[0]


def la_subgoal(td, relation, groups, clause, instance=None):
    """Subgoal record from a template's captured groups: relation name, object phrase -> prompts (objects), reference phrase ->
    prompts (supports, or a second object phrase), part phrase -> prompts (articulated; captured, or spelled out in the clause),
    side codes in group order. C3I1: a side code qualifies the OBJECT (object_side, an instance choice localize_target implements)
    when it is (a) the single side-word token inside the object's own vocabulary phrase ("<side> <kind>"; the phrase's prompts
    are the data's), (b) the qualifier of a qualified object group ("<object> <at/on/in the side>", la_qualified), or (c) a
    template side slot before the reference slot other than the slot written immediately before it (the reference's own
    qualifier slot: "the (side) (reference)", "to the (side) of the (reference)"); every other side code is a reference side
    (sides / side): the slot before the reference, a side token of the reference phrase, the qualifier of a reference group, and
    every side slot of a template without a reference slot. Returns None when a multi-word group is not a vocabulary entry (the
    template swallowed more than one clause)."""
    sg = {"relation": str(relation), "clause": clause, "object": None, "object_prompts": None, "object_side": None, "support": None, "support_prompts": None,
          "part": None, "part_prompts": None, "side": None, "sides": [], "words": [], "instance": instance}
    sw = (td or {}).get("side_words") or {}
    slots = []   # per group: (kind, phrase, value, the phrase's own side code | None), or None for an empty group
    for g in groups:
        r = la_lookup(td, g)
        if r is None:
            slots.append(None); continue
        kind, w, val = r; own = None
        if kind == "word" and " " in w:
            q = la_qualified(td, w)
            if q is None:
                return None
            kind, w, val, own = "object", q[0], td["objects"][q[0]], q[1]   # (b)
        elif kind in ("object", "support", "part"):
            codes = [str(sw[t]) for t in w.split() if t in sw]   # (a): a side word inside the vocabulary phrase
            own = codes[0] if len(codes) == 1 else None
        slots.append((kind, w, val, own))
    ref = None; n_obj = 0
    for i, s in enumerate(slots):   # the reference slot: the first support / part group, or the second object group
        if s is None:
            continue
        if s[0] in ("support", "part") or (s[0] == "object" and n_obj == 1):
            ref = i; break
        if s[0] == "object":
            n_obj += 1
    for i, s in enumerate(slots):
        if s is None:
            continue
        kind, w, val, own = s
        if kind == "side":
            if ref is not None and i < ref - 1 and sg["object_side"] is None:   # (c)
                sg["object_side"] = str(val)
            else:
                sg["sides"].append(str(val))
        elif kind == "word":
            sg["words"].append(w)
        elif sg[kind] is None:
            sg[kind] = w; sg[kind + "_prompts"] = list(val)
            if own is not None:
                if kind == "object" and sg["object_side"] is None:
                    sg["object_side"] = own
                else:
                    sg["sides"].append(own)   # the qualifier of a reference selects the reference instance, like a side word of the template
        elif kind == "object" and sg["support"] is None:   # a second object phrase is the reference of the relation
            sg["support"] = w; sg["support_prompts"] = list(val)
            if own is not None:
                sg["sides"].append(own)
        else:
            sg["words"].append(w)
    if sg["part"] is None:
        p = la_phrase_in(td, "articulated", clause)   # a part the template spells out instead of capturing it
        if p is not None and p != sg["support"] and p != sg["object"]:
            sg["part"] = p; sg["part_prompts"] = list(td["articulated"][p])
    if sg["relation"] in LP_RELATIONS and not sg["sides"]:   # C15I3: the preposition of an in_layer clause names the layer (LP_WORDS); any other relation is unchanged
        _lw = [LP_WORDS[x] for x in sg["words"] if x in LP_WORDS]
        if len(_lw) == 1:
            sg["sides"].append(_lw[0])
    sg["side"] = sg["sides"][0] if sg["sides"] else None
    return sg


def la_literal_sides(td, m, sgs):
    """C6I1: side words written in the template's own text (outside every captured group; today only "on top of the (reference)")
    name a side of the relation's reference: their codes are appended to each subgoal's reference sides (sides / side), the way a side
    word captured in the reference's slot is. A match without such a word returns the subgoals unchanged."""
    sw = (td or {}).get("side_words") or {}; text = m.string[:m.end()]
    spans = [m.span(i) for i in range(1, (m.re.groups or 0) + 1) if m.span(i)[0] >= 0]
    lit = "".join(" " if any(a <= k < b for a, b in spans) else ch for k, ch in enumerate(text))
    codes = [str(sw[t]) for t in lit.split() if t in sw]
    for sg in sgs:
        for c in codes:
            if c not in sg["sides"]:
                sg["sides"].append(c)
        if codes:
            sg["side"] = sg["sides"][0] if sg["sides"] else None
    return sgs


def la_match(td, text, templates):
    """First template (data order) that matches the whole text with every captured phrase in the vocabulary -> subgoal list
    (two subgoals for the two_objects / two_instances flags), else None."""
    for t in templates:
        m = _la_re.match(str(t["regex"]), text)
        if not m:
            continue
        g = list(m.groups()); rel = t["relation"]
        if t.get("two_objects"):
            sgs = [la_subgoal(td, rel, [g[0]] + g[2:], text), la_subgoal(td, rel, [g[1]] + g[2:], text)]
        elif t.get("two_instances"):
            sgs = [la_subgoal(td, rel, g, text, instance=1), la_subgoal(td, rel, g, text, instance=2)]
        else:
            sgs = [la_subgoal(td, rel, g, text)]
        if all(s is not None for s in sgs):
            return la_literal_sides(td, m, sgs)   # C6I1: a side word of the template's own text is a side of the reference
    return None


def la_anaphora(td, clause, templates, prev):
    """Clause that refers to the previous subgoal ("... it"): the referent slot (object, or support / part, whichever the previous
    subgoal has) and its side codes are copied from the previous subgoal. None without a previous subgoal or a referent."""
    if prev is None:
        return None
    for t in templates:
        m = _la_re.match(str(t["regex"]), clause)
        if not m:
            continue
        sg = la_subgoal(td, t["relation"], list(m.groups()), clause)
        if sg is None:
            continue
        ref = str(t.get("refers") or "")
        order = {"object": ("object",), "support": ("support", "part"), "part": ("part", "support")}.get(ref, ())
        src = next((k for k in order if prev.get(k) is not None), None)
        if src is None:
            return None
        sg[src] = prev[src]; sg[src + "_prompts"] = list(prev[src + "_prompts"]); sg["refers"] = ref
        if src == "object":   # C3I1: the same instance
            sg["object_side"] = prev.get("object_side")
        if src != "object" and not sg["sides"]:
            sg["sides"] = list(prev["sides"]); sg["side"] = sg["sides"][0] if sg["sides"] else None
        return [sg]
    return None


def la_clauses(td, text, templates, anaphora, prev=None):
    """Clause pass: the earliest conjunction whose head parses (template, else anaphora) and whose remainder parses too."""
    if not text:
        return []
    cuts = [m.start() for m in _la_re.finditer(_la_re.escape(LA_CONJ), text)] + [len(text)]
    for c in cuts:
        sgs = la_match(td, text[:c], templates) or la_anaphora(td, text[:c], anaphora, prev)
        if not sgs:
            continue
        rest = la_clauses(td, text[c + len(LA_CONJ):], templates, anaphora, sgs[-1]) if c < len(text) else []
        if rest is not None:
            return sgs + rest
    return None


def la_parse(td, lang):
    """Runtime language -> (subgoal list in language order, mode). Whole-sentence templates first, then the clause pass."""
    text = la_normalise(lang); T = list((td or {}).get("language_templates") or []); A = list((td or {}).get("anaphora_templates") or [])
    sgs = la_match(td, text, T)
    if sgs:
        return sgs, "whole_sentence"
    sgs = la_clauses(td, text, T, A)
    sgs = sc_carry_order(td, sgs) if sgs else sgs   # C17I2: a plural anaphora after an on subgoal -> carry the reference first, then the pair
    return (sgs or []), ("clauses" if sgs else "unparsed")


def la_parse_old_form(td, lang):
    """The previous driver's parse (adapter fields subgoal_regex / side_select / relation / support_prompts) as subgoal records.
    Used only when the templates do not parse the language and the adapter still carries the old fields."""
    rx = (td or {}).get("subgoal_regex"); sel_map = (td or {}).get("side_select") or {}
    if not rx:
        return []
    out = []
    for obj, side in _la_re.findall(rx, str(lang).lower()):
        prompts = ((td or {}).get("objects") or {}).get(obj.strip()); sel = sel_map.get(side)
        if prompts is None or sel is None:
            return []
        out.append({"relation": str(td.get("relation") or "on"), "clause": la_normalise(lang), "object": obj.strip(), "object_prompts": list(prompts), "object_side": None,
                    "support": None, "support_prompts": list(td.get("support_prompts") or []), "part": None, "part_prompts": None,
                    "side": sel, "sides": [sel], "words": [], "instance": None, "old_form": True})
    return out


def la_reference_prompts(sg):
    """Prompts of the relation's reference: the support phrase, else the part phrase (None when the clause names neither)."""
    return sg["support_prompts"] if sg.get("support_prompts") is not None else sg.get("part_prompts")


def la_dispatch(sg):
    """(ok, why): whether run_subgoal implements this subgoal (LA_SUPPORTED relation with an implemented side code, an object
    and a reference with prompts). why in relation | side | object_side | vocabulary."""
    allowed = LA_SUPPORTED.get(sg["relation"])
    if allowed is None:
        return False, "relation"
    sides = list(sg.get("sides") or [])
    if len(sides) > 1 or (sides[0] if sides else None) not in allowed:
        return False, "side"
    if sg.get("object_side") is not None and str(sg["object_side"]) not in LA_OBJECT_SIDES:   # C3I1: an object instance chosen by its position is bound by bind_select_by_side; a code it does not implement is refused
        return False, "object_side"
    if sg["relation"] in LA_PART_RELATIONS:   # C2I1: a part-closing subgoal needs a reference (part or support prompts) and no object
        return (True, None) if la_reference_prompts(sg) is not None else (False, "vocabulary")
    if sg.get("object_prompts") is None or la_reference_prompts(sg) is None:
        return False, "vocabulary"
    return True, None


def la_summary(sg):
    return {k: sg.get(k) for k in ("relation", "object", "object_side", "support", "part", "side", "sides", "instance", "refers", "old_form", "support_track") if sg.get(k) is not None}   # C17I2: support_track
# ===== end C1I1 language_adapter =====


def run():
    _TD = task_descriptors()
    lang = str(env.handle.task_language)
    subgoals, mode = la_parse(_TD, lang)   # C1I1: shared templates (data) -> subgoals in language order
    if not subgoals and _TD.get("subgoal_regex"):
        subgoals = la_parse_old_form(_TD, lang); mode = "old_form_fallback" if subgoals else mode
    print("LANG_PARSE " + json.dumps({"language": lang, "mode": mode, "subgoals": [la_summary(s) for s in subgoals]}), flush=True)
    if any(s.get("support_track") for s in subgoals):   # C17I2
        print("SC_ORDER " + json.dumps({"order": [la_summary(s) for s in subgoals], "rule": "carry_reference_first"}), flush=True)
    if not subgoals:
        print("BINDING_STOP " + json.dumps({"status": "language_unparsed"}), flush=True); return
    for k, sg in enumerate(subgoals, 1):   # C1I1: every subgoal must be one the stack implements before any motion
        ok, why = la_dispatch(sg)
        if not ok:
            print("BINDING_STOP " + json.dumps({"status": "language_unparsed" if why == "vocabulary" else "relation_unsupported", "relation": sg["relation"], "side": sg["side"],
                                               "sides": sg["sides"], "why": why, "k": k, "n": len(subgoals), "implemented": {r: list(v) for r, v in LA_SUPPORTED.items()}}), flush=True)
            return
    vb_resolve_references(subgoals)   # VB1: task-start references, before any object moves
    placed = []
    for k, sg in enumerate(subgoals, 1):
        _VB["active"] = k; _VB["confirmed"] = None; _VB["pregrasp_img"] = None
        if k > 1:
            goto_home_joint_position()
            for _ in range(3):
                get_observation()
        if sg.get("refers") and sg["relation"] in LA_PART_RELATIONS:   # C15I4: the referent's assembly vocabulary (la_assembly_prompts over the previous subgoals)
            sg["referent_prompts"] = la_assembly_prompts(sg, _TD, subgoals[:k - 1])
        if sg["relation"] in LA_PART_RELATIONS:   # C2I1: articulated part closing (manipulation_primitive); returns no placement
            (run_rotary if sg["relation"] in RT_RELATIONS else run_actuate if sg["relation"] in LA_ACTUATE_RELATIONS else run_open_articulated if sg["relation"] in LA_OPEN_RELATIONS else run_close_articulated)(sg, _TD, sg_label=f"sg{k}"); xy = None   # C2I2: actuate_on -> run_actuate
        else:
            xy = run_subgoal(sg["object_prompts"], sg["side"], others_xy=tuple(placed), relation=sg["relation"], support_prompts=la_reference_prompts(sg), object_side=sg.get("object_side"), assembly_prompts=la_assembly_prompts(sg, _TD, subgoals[:k - 1]), support_track=({"xy": _SC_XY.get(k - 1)} if sg.get("support_track") else None))   # C17I2: support_track; C3I1; C6I3: the assembly vocabulary of the subgoal
        print("SUBGOAL_STATUS " + json.dumps({"k": k, "object": sg["object"], "side": sg["side"], "relation": sg["relation"], "placed": xy is not None}), flush=True)
        _SC_XY[k] = xy   # C17I2
        if xy is not None:
            placed.append(xy)

run()
