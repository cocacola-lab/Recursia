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
    pts = best["pts"]
    axis_xy, how = object_axis_center_xy(pts, cam_xy)
    out = {"prompt": best["prompt"], "score": best["score"], "n_pts": int(len(pts)),
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
def t_select_transport_p3(place_xy, meas, q, lift_z, sg="sg"):
    """Amendment 2: each orientation candidate is planned at ITS OWN compensated TCP (the held offset, measured once at the
    lift pose, re-expressed at that orientation by compensate_xy), because the IK margin at the compensated pose can be
    millimetres while the nominal pose is feasible. Returns (q, tcp_xy, compensation_info, chosen_name)."""
    q = np.asarray(q, float)
    cands = [("as_is", q), ("flip", _flip_tool_z(q)), ("yaw+90", _yaw_about_vertical(q, 90)), ("flip_yaw+90", _flip_tool_z(_yaw_about_vertical(q, 90))),
             ("yaw-90", _yaw_about_vertical(q, -90)), ("flip_yaw-90", _flip_tool_z(_yaw_about_vertical(q, -90)))]
    comp = [compensate_xy(np.asarray(place_xy, float), meas, qq) for _, qq in cands]
    plans = [move_to_pose(np.array([c[0][0], c[0][1], lift_z]), qq, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, plan_only=True, label=f"{sg}:transport_plan_{name}") for (name, qq), c in zip(cands, comp)]
    feas = [(int(p["steps_planned"]), k) for k, p in enumerate(plans) if p.get("detail") == "plan_only" and p.get("steps_planned") is not None]
    chosen = 0 if any(k == 0 for _, k in feas) else (min(feas)[1] if feas else 0)
    print("TRANSPORT_PLAN " + json.dumps({"sg": sg, "candidates": [c[0] for c in cands], "planned": [p.get("steps_planned") for p in plans], "detail": [p.get("detail") for p in plans], "chosen": cands[chosen][0], "at": "compensated_tcp"}), flush=True)
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
def p_slip_check(summary, tcp_xy, q_place, z_release, held_prompts, meas_lift, sg):
    xy_now = np.asarray(tcp_xy, float).copy(); hist = []
    ref = [m for m in (meas_lift or []) if m is not None]
    if not held_prompts or not ref:
        print("PLACE_SLIP " + _json.dumps({"sg": sg, "status": "no_reference"}), flush=True)
        return {"tcp_xy": xy_now, "history": hist, "status": "no_reference"}
    off_ref = np.mean([np.asarray(m["offset_xy_base"], float) for m in ref], axis=0); q_ref = np.asarray(ref[0]["tcp_q"], float)
    p0, _, _ = read_tcp(); tcp0 = np.asarray(p0[:2], float)   # Q1: executed corrections are the TCP displacement from here
    for k in range(P_SLIP_RETRIES + 1):
        m = measure_held_object(held_prompts, get_observation())
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

def prim_place(place_xy, q_place, lift_z, p0_descent_zs, support_z, drop_est, held_prompts=None, sg="sg1", settle_obs=5, alt_quat=None):
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
            q_place, tcp_xy, cinfo, _tsel = t_select_transport_p3(place_xy, meas, q_place, z_tr, sg=sg)   # M1 T (Amendment 2): orientation chosen at each candidate's compensated TCP
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
    if z_tr > float(lift_z) + 0.005:   # Q1 C: vertical lower to the descent start height above the target
        lres = move_to_pose([tcp_xy[0], tcp_xy[1], lift_z], q_place, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label=f"{sg}:transport_lower"); _PRIM_STATS["motion_calls"] += 1
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
        z_release = support_z + drop_used + PRIM["release_clearance"]
        summary["z_release"] = z_release; summary["drop_used"] = drop_used; summary["drop_source"] = drop_source
        summary["descend"] = mc_place_descend(tcp_xy, q_place, z_release, lift_z, sg)   # MC: contact-aware placement descent
        if summary["descend"].get("mc_result") is not None and (summary["descend"]["mc_result"] or {}).get("success"):
            summary["slip"] = p_slip_check(summary, tcp_xy, q_place, z_release, held_prompts, _meas_lift, sg)   # N1 P
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
    g1 = gripper_opening(); gv = verify_grasp(g0, g1)
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
    hv = verify_hold(ref, gripper_opening()) if ref is not None else {"status": "uncertain", "reason": "no reference opening"}
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
        if np.any(p < np.asarray(_G1_WS_LO) - 1e-6) or np.any(p > np.asarray(_G1_WS_HI) + 1e-6):   # 1e-6: floating tolerance so a candidate exactly at the box face is not filtered when the primitive would accept it
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
_FCFG = {'grasp_geometry': 'B+G2depth', 'arm_h': 'H2', 'arm_i': 'I2', 'arm_l': 'L1', 'width_gate': True, 'arm_m': 'M1', 'arm_n': 'N1', 'arm_q': 'Q1', 'arm_r': 'R1', 'arm_s': 'S1', 'arm_k': 'K2', 'placement_target': 'MC', 'enforce': False, 'corrections': False}
PLACE_TOUCHDOWN_UNDERSHOOT = 0.015   # m (identical to MG)
PLACE_RETRIES_TOTAL = 2              # bounded corrective actions per placement (module F)
_MF = {"contract": None, "last_plan": None, "rest_ref": None, "rest_src": None, "target_xy": None, "prof": None}
def _mfj(o):
    return json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
def mf_frame(obs):
    cam = obs["agentview"]; depth = cam["images"]["depth"]; _p_eef, _ = tcp_from_obs(obs)
    return {"rgb": cam["images"]["rgb"], "depth": depth[:, :, 0] if len(depth.shape) == 3 else depth, "K": cam["intrinsics"], "E": cam["pose_mat"], "cam_xy": np.asarray(cam["pose_mat"])[:2, 3], "robot": {"eef": np.asarray(_p_eef, float)}}
def mf_make_contract(relation, support_pts, obj_pts, prompts, E, region_dir=None, others_xy=(), label="", binding_base_z=None, target_xy=None):
    sup = pv_support_from_points(np.asarray(support_pts, float)); prof = gg_object_profile(np.asarray(obj_pts, float), np.asarray(E)[:2, 3]); _MF["prof"] = prof
    r_obj = float(max(prof["r_top"], prof["r_mid"])) if prof else 0.04
    c = pv_make_contract(relation, sup, list(prompts), r_obj, prof["height"] if prof else None, region_dir=region_dir, others_xy=others_xy, label=label)
    _MF["contract"] = c
    _MF["rest_ref"], _MF["rest_src"] = fg_rest_reference(relation, sup, binding_base_z)
    _MF["target_xy"] = np.asarray(sup["center_xy"], float) if target_xy is None else np.asarray(target_xy, float)
    print("MF_CONTRACT " + _mfj({"label": label, "relation": relation, "support": {"center_xy": sup["center_xy"], "r": sup["r"], "rest_z": sup["rest_z"], "table_z": sup["table_z"], "fit_ok": sup["fit_ok"]}, "object_r": r_obj, "object_h": prof["height"] if prof else None,
                                 "region": c.get("region"), "rest_ref": _MF["rest_ref"], "rest_src": _MF["rest_src"], "binding_base_z": binding_base_z, "target_xy": _MF["target_xy"]}), flush=True)
    return c
def mf_plan_grasp(obj_pts, E, label):
    plan = (fg_plan_grasp if _FCFG["grasp_geometry"] == "F" else gg_plan_grasp)(np.asarray(obj_pts, float), np.asarray(E)[:2, 3]); _MF["last_plan"] = plan
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


# ===== H1 B: task-agnostic multi-prompt target binding (runner-injected bind_object; fail-closed) =====
_BIND = {"last": None, "n": 0}
def localize_target(rgb, depth_img, K, E, prompts):
    r = bind_object(rgb, depth_img, K, E, list(prompts))
    _BIND["last"] = r["status"]; _BIND["n"] += 1
    print("BINDING " + json.dumps({"k": _BIND["n"], "status": r["status"], "support": round(float(r["support"]), 3),
                                   "rel_margin": None if r["rel_margin"] is None else round(float(r["rel_margin"]), 3),
                                   "instances": r["instances"][:3], "sam3_calls": r["sam3_calls"]}), flush=True)
    if r["status"] != "bound":
        return None, None, None
    pts = r["pts"]; obb = get_oriented_bounding_box_from_3d_points(pts)
    return obb["center"], pts, np.asarray(r["mask"]).astype(np.uint8)
# ===== end H1 B =====


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

    _feas = g1_descent_feasible(gpos, gpos[2] + 0.12)
    if not _feas["ok"]:   # G1 I2: infeasible candidate filtered BEFORE any motion (no open, no approach); the strategy loop continues
        print(f"GRASP_ABORT reason=candidate_infeasible: {_feas['why']}", flush=True)
        return False, gpos, gquat
    open_gripper()
    gquat, _abud = h_select_approach(gpos, gquat)
    _WG["quat"] = np.asarray(gquat, float)   # W: actual descent orientation (closing axis)
    ra = move_to_pose(np.array([gpos[0], gpos[1], gpos[2] + 0.12]), gquat, mode="free", tol_pos=0.015, step_budget=_abud, label="sg1:approach")
    if ra["status"] != "reached":
        print("GRASP_MOTION " + json.dumps({"approach": ra["status"], "descent": None, "detail": ra.get("detail"), "close": False, "why": "approach_nonarrival"}), flush=True)
        print(f"GRASP_ABORT reason=approach_nonarrival: {ra['status']}/{ra.get('detail')}", flush=True)
        goto_home_joint_position()
        return False, gpos, gquat
    rg = mc_grasp_descend(gpos, np.array([gpos[0], gpos[1], gpos[2] + 0.12]), gquat, "sg1:grasp_descent")   # MC: contact-aware grasp descent
    mf_record_grasp(rg)   # F: predicted vs actual contact height
    ok_close, why = gate_allow(rg, "close")
    print("GRASP_MOTION " + json.dumps({"approach": ra["status"], "descent": rg["status"], "detail": rg.get("detail"), "pos_err_cm": round(rg.get("pos_err", 9) * 100, 2), "lateral_cm": round(rg.get("lateral_max", 0) * 100, 2), "close": ok_close, "why": why}), flush=True)
    if not ok_close:
        print(f"GRASP_ABORT reason={why}", flush=True)
        move_to_pose(np.array([gpos[0], gpos[1], gpos[2] + 0.12]), gquat, mode="free", tol_pos=0.02, step_budget=BUDGET_FREE, label="sg1:backoff")
        goto_home_joint_position()
        return False, gpos, gquat
    gv = mc_close_and_verify(rg, "sg1")
    _E1_LAST.update({"grasp_after": None, "hold": None})
    if gv["status"] != "verified":
        print(f"GRASP_ABORT reason=grasp_not_verified: {gv.get('reason')}", flush=True)
        open_gripper(); gate_record("open", role="recovery_open")
        move_to_pose(np.array([gpos[0], gpos[1], gpos[2] + 0.12]), gquat, mode="free", tol_pos=0.02, step_budget=BUDGET_FREE, label="sg1:backoff")
        goto_home_joint_position()
        return False, gpos, gquat

    _okl, _whyl = e1_lift_allowed(gv)
    if not _okl:   # E1 G1: lift refused -> existing not-verified recovery path (open, backoff, home); no lift
        print(f"GRASP_ABORT reason=lift_refused: {_whyl}", flush=True)
        open_gripper(); gate_record("open", role="recovery_open")
        move_to_pose(np.array([gpos[0], gpos[1], gpos[2] + 0.12]), gquat, mode="free", tol_pos=0.02, step_budget=BUDGET_FREE, label="sg1:backoff")
        goto_home_joint_position()
        return False, gpos, gquat
    rl = move_to_pose(np.array([gpos[0], gpos[1], 0.25]), gquat, mode="free", tol_pos=0.015, step_budget=BUDGET_FREE, label="sg1:lift"); gate_record("lift")
    hv = verify_hold(gv["after"], gripper_opening())
    _E1_LAST.update({"grasp_after": gv["after"], "hold": hv["status"]})
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

    chk, _, _ = localize_target(rgb_c, di_c, cam_c["intrinsics"], cam_c["pose_mat"],
        list(_CUR["prompts"]))   # Amendment 3: the current target's prompts (Long descriptors carry several objects)   # I1 R: bound multi-prompt rule at the post-lift re-check (fail-closed binding never releases a verified hold)
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
def relation_geometry(td, select):
    """(contract relation, region direction | None) from the adapter data. Only the datum "beside" with a side selection of
    "min" / "max" yields the region form; any other value, or an absent datum, is the on-support relation (parent behaviour)."""
    rel = str((td or {}).get("relation") or "").strip().lower(); sel = str(select).strip().lower()
    if rel != "beside" or sel not in ("min", "max"):
        return "on_support", None
    d = np.zeros(2); d[SIDE_AXIS] = 1.0 if sel == "max" else -1.0
    return "beside_support", d
def place_target(relation, region_dir, support_c, support_pts, obj_pts, E):
    """Placement target xy for the relation. on_support: the support centre exactly as before. beside_support: outside the rim
    on the requested side (rule above). Returns (x, y, info)."""
    if relation != "beside_support" or region_dir is None:
        info = {"relation": relation, "rule": "support_centre", "target": [float(support_c[0]), float(support_c[1])]}
        print("PLACE_TARGET " + _mfj(info), flush=True)
        return float(support_c[0]), float(support_c[1]), info
    sup = pv_support_from_points(np.asarray(support_pts, float)); prof = gg_object_profile(np.asarray(obj_pts, float), np.asarray(E)[:2, 3])
    r_obj = float(max(prof["r_top"], prof["r_mid"])) if prof else 0.04   # identical to the contract's object radius (mf_make_contract)
    d = np.asarray(region_dir, float)[:2]; d = d / (float(np.linalg.norm(d)) + 1e-12)
    offset = float(sup["r"]) + r_obj + BESIDE_CLEARANCE
    c = np.asarray(sup["center_xy"], float)[:2]; nominal = c + d * offset
    x, y = mf_validate_beside_target(nominal, support_pts, obj_pts, E)   # shared feasible-region rule: bounded outward shift only
    info = {"relation": relation, "rule": "outside_rim_on_side", "dir": d, "support_center_xy": c, "r_support": float(sup["r"]), "fit_ok": sup.get("fit_ok"),
            "r_object": r_obj, "clearance": BESIDE_CLEARANCE, "offset": offset, "nominal": nominal, "target": [float(x), float(y)], "shift_m": float(np.hypot(float(x) - nominal[0], float(y) - nominal[1]))}
    print("PLACE_TARGET " + _mfj(info), flush=True)
    return float(x), float(y), info
# ===== end K2 place_primitive =====


def run_subgoal(pud_prompts, pl_select, others_xy=()):
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

    pud_c, pud_pts, pud_mask = localize_target(rgb, depth_img, K, E, pud_prompts)
    if pud_c is None:
        print("BINDING_STOP " + json.dumps({"status": _BIND["last"], "reason": "target not bound unambiguously; no motion"}), flush=True)
        return

    pl_c, pl_pts, _ = localize_support_by_side(rgb, depth_img, K, E, list(_TD["support_prompts"]), pl_select, exclude_mask=pud_mask)   # K1: the bound target is never its own support
    if pl_c is None:
        print("BINDING_STOP " + json.dumps({"status": "support_not_found", "select": pl_select}), flush=True)
        return
    pl_obb = get_oriented_bounding_box_from_3d_points(pl_pts)
    pl_r = max(pl_obb["extent"][0], pl_obb["extent"][1]) / 2.0
    table_z = pl_pts[:, 2].min()

    pud_base_z0 = object_base_z(pud_pts)
    print(f"PRIM_GEOM table_z={table_z:.4f} pud_base_z0={pud_base_z0:.4f}", flush=True)
    _rel, _rdir = relation_geometry(_TD, pl_select)   # K2: contract relation + region direction from the adapter DATA (relation datum, side selection)
    target_x, target_y, _tinfo = place_target(_rel, _rdir, pl_c, pl_pts, pud_pts, E)   # K2: on_support = support centre (unchanged); beside_support = outside the rim on the requested side
    print(f"Object: {pud_c}, Support: {pl_c}, Target: [{target_x:.3f}, {target_y:.3f}]")
    _MF["contract"] = mf_make_contract(_rel, pl_pts, pud_pts, pud_prompts, E, region_dir=_rdir, others_xy=tuple(others_xy), label="sg1", binding_base_z=pud_base_z0, target_xy=(target_x, target_y))   # K2: shadow contract in the relation's own form (region form for beside; never gates)
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

        pud_c2, _, pud_mask2 = localize_target(rgb, depth_img, K, E, pud_prompts)
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
    prim_place(np.array([target_x, target_y]), gquat, 0.25, [table_z + rz for rz in (0.15, 0.10, 0.06, 0.04)],
               float(table_z), drop_est, held_prompts=pud_prompts, sg="sg1", settle_obs=10)
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
def support_candidate_ok(pts, mask, exclude_mask, floor_z):
    """Geometry gate for one support candidate (public data only). Returns (ok, reason, features)."""
    feat = {"n_pts": int(len(pts))}
    if exclude_mask is not None:
        feat["overlap_bound"] = round(mask_overlap_frac(mask, exclude_mask), 3)
        if feat["overlap_bound"] >= SUPPORT_SAME_MASK:
            return False, "is_bound_object", feat
    if floor_z is not None:
        z = np.asarray(pts, float)[:, 2]
        feat["rise"] = round(float(np.median(z)) - float(floor_z), 4); feat["top"] = round(float(np.percentile(z, 98)) - float(floor_z), 4)
        if feat["rise"] < SUPPORT_RAISED_MIN:
            return False, "not_raised_over_floor", feat
        if feat["top"] > SUPPORT_TOP_MAX:
            return False, "too_tall_for_a_support", feat
    return True, "ok", feat
# ===== end K1 binding =====


def localize_support_by_side(rgb, depth_img, K, E, prompts, select, exclude_mask=None):
    """Support instance by side. Candidates = top masks of a support prompt inside the same candidate bounds as the parent
    two-support program that pass the K1 geometry gate (not the bound object; raised over the floor; low). Distinct instances are
    > 5 cm apart. The first prompt that yields >= 2 instances selects 'min' / 'max' along y (rule unchanged); when no prompt
    yields two, the single instance of the first prompt that yields one is the support (a lone instance is its own extreme)."""
    floor_z = support_floor_z(depth_img, K, E)
    single = None
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            continue
        cands = []; rejected = []
        for m in sorted(masks, key=lambda d: -float(d["score"]))[:15]:
            mask = m["mask"].astype(np.uint8); pts = mask_to_world_points(mask, depth_img, K, E)
            if pts is None or len(pts) < 10:
                continue
            obb = get_oriented_bounding_box_from_3d_points(pts); c = obb["center"]
            if 0.2 < c[0] < 0.8 and -0.5 < c[1] < 0.5 and -0.02 < c[2] < 0.06:
                ok, why, feat = support_candidate_ok(pts, mask, exclude_mask, floor_z)   # K1: geometry gate
                if ok:
                    cands.append((c, pts, mask))
                else:
                    rejected.append({"why": why, "score": round(float(m["score"]), 3), "center": [round(float(v), 4) for v in c], **feat})
        uniq = []
        for c in cands:
            if all(float(np.hypot(*(np.asarray(c[0][:2]) - np.asarray(u[0][:2])))) > 0.05 for u in uniq):
                uniq.append(c)
        print("SUPPORT_GATE " + json.dumps({"prompt": prompt, "floor_z": None if floor_z is None else round(floor_z, 4), "n_kept": len(cands), "n_instances": len(uniq), "rejected": rejected[:6]}), flush=True)
        if len(uniq) >= 2:
            uniq.sort(key=lambda x: float(x[0][1])); pick = uniq[0] if select == "min" else uniq[-1]
            print("BIND_SUPPORT " + json.dumps({"prompt": prompt, "n_instances": len(uniq), "select": select, "center": [round(float(v), 4) for v in pick[0]]}), flush=True)
            return pick
        if len(uniq) == 1 and single is None:
            single = (prompt, uniq[0])
    if single is not None:
        prompt, pick = single
        print("BIND_SUPPORT " + json.dumps({"prompt": prompt, "n_instances": 1, "select": select, "center": [round(float(v), 4) for v in pick[0]], "rule": "single_instance"}), flush=True)
        return pick
    return None, None, None


def run():
    import re as _re
    _TD = task_descriptors()
    lang = str(env.handle.task_language)
    subgoals = []
    for obj, side in _re.findall(_TD["subgoal_regex"], lang.lower()):
        prompts = _TD["objects"].get(obj.strip()); sel = _TD["side_select"].get(side)
        if prompts is None or sel is None:
            subgoals = []; break
        subgoals.append((obj.strip(), list(prompts), sel))
    print("LANG_PARSE " + json.dumps({"language": lang, "subgoals": [(o, s) for o, _, s in subgoals]}), flush=True)
    if not subgoals:
        print("BINDING_STOP " + json.dumps({"status": "language_unparsed"}), flush=True); return
    placed = []
    for k, (obj, prompts, sel) in enumerate(subgoals, 1):
        if k > 1:
            goto_home_joint_position()
            for _ in range(3):
                get_observation()
        xy = run_subgoal(prompts, sel, others_xy=tuple(placed))
        print("SUBGOAL_STATUS " + json.dumps({"k": k, "object": obj, "side": sel, "placed": xy is not None}), flush=True)
        if xy is not None:
            placed.append(xy)

run()
