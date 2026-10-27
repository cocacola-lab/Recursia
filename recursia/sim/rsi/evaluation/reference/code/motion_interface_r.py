"""General robot motion interface R (round rel_20260916_212500) — minimal general fixes over the frozen parent M2.

Public robot state only (7 arm joints + gripper opening, i.e. `robot0_joint_pos` / `robot0_gripper_qpos`, the same
quantities `get_observation()["robot_joint_pos"]` exposes) plus the Panda URDF. No object, contact, scene, task or seed
information enters any decision. See ../contract.md for the caller-facing contract.

Constants (robot/controller quantities; sources in the parent rounds):
  TCP_OFFSET_HAND    [0,0,-0.10] m   libero_reduced.py (requested point is 10 cm along tool z from panda_hand)
  IK_WORKSPACE_*                    libero_reduced.py public IK box
  SERVER_LIMIT_MARGIN 0.15 rad      IK server joint-limit margin
  V_EFF              0.0073 rad/step controller speed (stage-0 C2b) — used ONLY to schedule waypoint allocations
  SETTLE_STEPS       10             scheduling only
  CONV_TOL           0.01 rad       joint convergence tolerance of the production controller loop
  IK_TOL             3 mm / 2 deg   IK solution verification
  MAX_WAYPOINT_STEP  0.30 rad       joint-space waypoint spacing (discontinuity limit 1.5x)
  LINEAR_SPACING     0.02 m         Cartesian waypoint spacing in mode="linear"
  CHUNK_STEPS        10             feedback / constraint sampling granularity
  STALL_MIN_PROGRESS 0.02 rad       joint error must fall by this over 2 chunks, else `stalled`
  PASS_TOL           0.03 rad       intermediate linear waypoints are passed through (dev finding R2-1: converging every
                                    waypoint to 0.01 rad made R slower than the legacy command and lose a matched case)
  DEFAULT_BUDGET     120            legacy single-call cap (used when step_budget is None)
Changes vs the frozen M2 (code/frozen_parent/motion_interface.py):
  * no refusal from the plan estimate (`budget_insufficient` removed); the caller's budget is executed, never raised
  * `budget_exhausted` only when steps_used == budget; waypoint allocations are a schedule, the remainder flows on
  * arrival judged by the caller's pose tolerance every chunk on the last waypoint (no tighter joint criterion)
  * status set {reached, stalled, unreachable, budget_exhausted, constraint_violation, invalid_input, uncertain},
    `reached` boolean ≡ status == "reached"; sub-reasons in `detail`
  * input validation (`invalid_input`), q/-q equivalence, `plan_only`, stall evidence without contact inference
  * free mode commands the final joint target directly (legacy-equivalent controller command, no intermediate joint
    waypoints: dev finding R2-1, the 0.30 rad waypoints cost ~15% steps and lost F:yaw-60@120 that M1 reached)
Joint distance: all 7 Panda joints are limited revolute joints (range < 2*pi), so the joint-space distance is the plain
difference; there is no wrap-around. `joint_distance()` documents this and rejects out-of-limit targets.
"""
from __future__ import annotations

import json
import math
import numbers
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

TCP_OFFSET_HAND = np.array([0.0, 0.0, -0.10])
IK_WORKSPACE_LO = np.array([-0.1, -0.5, -0.035]); IK_WORKSPACE_HI = np.array([0.75, 0.5, 0.9])   # N15 amendment (user 2026-10-09: "allow changing lowest height"): TCP floor 0.005 -> -0.035 (tables measured up to ~4 cm below the old fingertip floor; contact stop unchanged). Policies must derive their own floor from the measured table.
SERVER_LIMIT_MARGIN = 0.15
V_EFF = 0.0073
SETTLE_STEPS = 10
CONV_TOL = 0.01
IK_TOL_POS = 0.003; IK_TOL_ORI_DEG = 2.0
MAX_WAYPOINT_STEP = 0.30
LINEAR_SPACING = 0.02
CHUNK_STEPS = 10
STALL_MIN_PROGRESS = 0.02
PASS_TOL = 0.03   # rad: intermediate linear-mode waypoints are passed through at this joint error (no deceleration stop)
DEFAULT_BUDGET = 120
STATUSES = ("reached", "stalled", "unreachable", "budget_exhausted", "constraint_violation", "invalid_input", "uncertain")
PRE_MOTION_DETAILS = ("out_of_workspace", "ik_failed", "discontinuous_path", "joint_limits")


def _R(q):
    w, x, y, z = [float(v) for v in q]; n = math.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_angle_deg(q1, q2):
    """Sign-invariant geodesic angle between two wxyz quaternions (q and -q are the same orientation)."""
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return float(np.degrees(2 * np.arccos(np.clip(d, 0.0, 1.0))))


def slerp(q0, q1, t):
    q0 = np.asarray(q0, float) / np.linalg.norm(q0); q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


# ---------------------------------------------------------------------------------------------- input validation
class InputError(ValueError):
    pass


def _vec(x, n, name):
    try:
        v = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise InputError(f"{name}: not numeric")
    if v.shape != (n,):
        v = v.reshape(-1)
        if v.shape != (n,):
            raise InputError(f"{name}: expected {n} numbers, got shape {np.shape(x)}")
    if not np.all(np.isfinite(v)):
        raise InputError(f"{name}: non-finite value")
    return v.copy()


def validate_budget(step_budget):
    """None -> DEFAULT_BUDGET; else an integer >= 1 (Python/numpy int, or float with integral value). Never raised."""
    if step_budget is None:
        return DEFAULT_BUDGET
    if isinstance(step_budget, bool) or not isinstance(step_budget, numbers.Real):
        raise InputError(f"step_budget: expected an integer >= 1, got {type(step_budget).__name__}")
    b = float(step_budget)
    if not math.isfinite(b):
        raise InputError("step_budget: non-finite")
    if b != math.floor(b):
        raise InputError(f"step_budget: non-integral value {step_budget}")
    if b < 1:
        raise InputError(f"step_budget: must be >= 1, got {step_budget}")
    return int(b)


def validate_tol(v, name):
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)) or float(v) <= 0:
        raise InputError(f"{name}: expected a finite number > 0, got {v!r}")
    return float(v)


def validate_quat(q):
    v = _vec(q, 4, "quaternion_wxyz"); n = float(np.linalg.norm(v))
    if n < 1e-6:
        raise InputError("quaternion_wxyz: zero norm")
    return v / n


class PandaModel:
    """URDF FK + local PyRoKi IK (CPU). Same description and 0.15 rad limit margin as the IK server."""

    def __init__(self):
        import pyroki as pk  # type: ignore
        from robot_descriptions.loaders.yourdfpy import load_robot_description  # type: ignore
        from cap.integrations.motion import pyroki_snippets as pks  # type: ignore
        self.pks = pks
        urdf = load_robot_description("panda_description")
        for j in urdf.robot.joints:
            if j.type == "revolute" and j.limit is not None and j.limit.lower is not None:
                j.limit.lower += SERVER_LIMIT_MARGIN; j.limit.upper -= SERVER_LIMIT_MARGIN
        self.robot = pk.Robot.from_urdf(urdf)
        self.hand_idx = self.robot.links.names.index("panda_hand")
        self.lo = np.array(self.robot.joints.lower_limits, float)[:7]; self.hi = np.array(self.robot.joints.upper_limits, float)[:7]
        assert np.all(self.hi - self.lo < 2 * np.pi), "Panda joints are limited; no wrap-around distance is needed"
        t0 = time.time(); self.fk_tcp(np.zeros(7)); self._ik_local(np.array([0.45, 0.0, 0.25]), np.array([0.0, 1.0, 0.0, 0.0]), np.zeros(7)); self.jit_s = time.time() - t0

    def fk_hand(self, j7):
        fk = np.array(self.robot.forward_kinematics(np.concatenate([np.asarray(j7, float).reshape(7), [0.02]])))
        return fk[self.hand_idx, 4:7].copy(), fk[self.hand_idx, 0:4].copy()

    def fk_tcp(self, j7):
        p, q = self.fk_hand(j7)
        return p - _R(q) @ TCP_OFFSET_HAND, q

    def residual(self, j7, pos, quat):
        p, q = self.fk_tcp(j7)
        return float(np.linalg.norm(p - np.asarray(pos, float))), quat_angle_deg(q, quat)

    def limit_violation(self, j7):
        j = np.asarray(j7, float); return float(np.max(np.maximum(self.lo - j, j - self.hi).clip(min=0.0)))

    def joint_distance(self, a, b):
        """Plain joint-space difference (all Panda joints are limited revolute joints: no periodic wrap)."""
        d = np.asarray(a, float) - np.asarray(b, float); return float(np.linalg.norm(d)), d

    def _ik_local(self, pos, quat, seed7, init=True):
        hand = np.asarray(pos, float) + _R(quat) @ TCP_OFFSET_HAND
        seed = np.concatenate([np.asarray(seed7, float), [0.02]])
        cfg = self.pks.solve_ik_vel_cost(robot=self.robot, target_link_name="panda_hand", target_wxyz=np.asarray(quat, float), target_position=hand,
                                         prev_cfg=seed, initial_cfg=seed if init else None)
        return np.asarray(cfg, float)[:7]

    def _ik_global(self, pos, quat):
        hand = np.asarray(pos, float) + _R(quat) @ TCP_OFFSET_HAND
        cfg = self.pks.solve_ik(robot=self.robot, target_link_name="panda_hand", target_wxyz=np.asarray(quat, float), target_position=hand)
        return np.asarray(cfg, float)[:7]

    def plan_ik(self, pos, quat, actual7):
        """Anchored IK with verification (no side effects). Returns dict(status ok|out_of_workspace|ik_failed, joints|None, residuals, jump, method)."""
        pos = np.asarray(pos, float).reshape(3); quat = np.asarray(quat, float).reshape(4); actual7 = np.asarray(actual7, float).reshape(7)
        if np.any(pos < IK_WORKSPACE_LO) or np.any(pos > IK_WORKSPACE_HI):
            return {"status": "out_of_workspace", "joints": None, "residual_pos": None, "residual_ori": None, "jump": None, "method": None}
        cands = []
        j = self._ik_local(pos, quat, actual7, init=True)
        j = self._ik_local(pos, quat, j, init=True)
        cands.append(("anchored", j))
        rp, ro = self.residual(j, pos, quat)
        if not (rp <= IK_TOL_POS and ro <= IK_TOL_ORI_DEG):
            cands.append(("default_init", self._ik_local(pos, quat, actual7, init=False)))
            cands.append(("global_hard_limits", self._ik_global(pos, quat)))
        best = None
        for name, jj in cands:
            rp, ro = self.residual(jj, pos, quat); ok = rp <= IK_TOL_POS and ro <= IK_TOL_ORI_DEG
            viol = self.limit_violation(jj)
            row = {"method": name, "joints": jj, "residual_pos": rp, "residual_ori": ro, "jump": float(np.linalg.norm(jj - actual7)), "limit_violation": viol, "ok": ok and viol < 1e-3}
            if row["ok"] and (best is None or row["jump"] < best["jump"]):
                best = row
        if best is None:
            worst = min(cands, key=lambda c: self.residual(c[1], pos, quat)[0])
            rp, ro = self.residual(worst[1], pos, quat)
            return {"status": "ik_failed", "joints": None, "residual_pos": rp, "residual_ori": ro, "jump": None, "method": None, "n_candidates": len(cands)}
        return {"status": "ok", **best, "n_candidates": len(cands)}


class MotionInterfaceR:
    """General motion interface: solve_ik drop-in (M1 semantics) + move_to_pose / move_to_joints with the contract of contract.md."""

    def __init__(self, low_env, log_path: Path | None = None, arm: str = "R"):
        self.low = low_env; self.arm = arm; self.model = PandaModel()
        self.log_f = open(log_path, "a") if log_path else None
        self.stats = {"ik_calls": 0, "ik_failed": 0, "ik_fallback": 0, "moves": 0, "sim_steps": 0, "reached": 0, "not_reached": 0, "status_counts": {}}
        self.last_result = None

    # ------------------------------------------------------------------ public robot state (proprioception only)
    def joints(self):
        obs = getattr(self.low, "_current_obs", None)
        if obs is None or "robot0_joint_pos" not in obs:
            return None
        j = np.asarray(obs["robot0_joint_pos"], float)
        if j.shape[0] < 7 or not np.all(np.isfinite(j[:7])):
            return None
        return j[:7].copy()

    def gripper_opening(self):
        return float(self.low._current_obs["robot0_gripper_qpos"][0] / self.low.gripper_metric_length)

    def read_robot_state(self):
        j = self.joints()
        if j is None:
            return {"status": "uncertain", "joints": None}
        p, q = self.model.fk_tcp(j)
        return {"status": "ok", "joints": j, "gripper_opening": self.gripper_opening(), "tcp_pos": p, "tcp_quat": q, "sim_step": int(self.low._sim_step_count)}

    def _log(self, rec):
        rec = {"t": round(time.time(), 3), "sim_step": int(getattr(self.low, "_sim_step_count", -1)), **rec}
        if self.log_f:
            self.log_f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); self.log_f.flush()

    def _count(self, res):
        self.stats["sim_steps"] += int(res.get("steps_used", 0)); self.stats["reached" if res["reached"] else "not_reached"] += 1
        self.stats["status_counts"][res["status"]] = self.stats["status_counts"].get(res["status"], 0) + 1
        self.last_result = res; self._log({"call": res.get("call", "move"), **res}); return res

    # ------------------------------------------------------------------ M1 drop-in
    def solve_ik(self, position, quaternion_wxyz):
        """Anchored at the ACTUAL joints, stateless, FK-verified. Raises RuntimeError (with the reason) instead of returning
        a substituted/clipped/unverified solution."""
        self.stats["ik_calls"] += 1
        try:
            pos = _vec(position, 3, "position"); quat = validate_quat(quaternion_wxyz)
        except InputError as e:
            self.stats["ik_failed"] += 1; raise RuntimeError(f"IK invalid_input: {e}")
        j0 = self.joints()
        if j0 is None:
            raise RuntimeError("IK uncertain: public joint observation missing")
        r = self.model.plan_ik(pos, quat, j0)
        self._log({"call": "solve_ik", "pos": pos, "quat": quat, **{k: v for k, v in r.items() if k != "joints"}, "joints": r.get("joints")})
        if r["status"] != "ok":
            self.stats["ik_failed"] += 1
            raise RuntimeError(f"IK {r['status']}: pos={np.round(pos, 4).tolist()} residual={r.get('residual_pos')} m / {r.get('residual_ori')} deg (no orientation substitution, no clipping)")
        if r["method"] != "anchored":
            self.stats["ik_fallback"] += 1
        return np.asarray(r["joints"], float)

    # ------------------------------------------------------------------ execution
    def _steps_needed(self, dq):
        return int(np.ceil(np.max(np.abs(dq)) / V_EFF)) + SETTLE_STEPS

    def _execute_waypoint(self, target7, alloc, corridor_fn=None, goal_fn=None, pass_tol=CONV_TOL):
        """Chunked move toward one joint waypoint, at most `alloc` control steps (every step counted from the simulator
        counter). Returns (event, steps_used, evidence). event ∈ converged | goal | constraint_violation | stalled |
        alloc_exhausted | uncertain."""
        used = 0; hist = []; lateral_max = 0.0; event = None; evidence = {}
        while used < alloc:
            k = min(CHUNK_STEPS, alloc - used); s0 = self.low._sim_step_count
            self.low.move_to_joints_blocking(target7, tolerance=CONV_TOL, max_steps=k)
            used += int(self.low._sim_step_count - s0)
            j = self.joints()
            if j is None:
                event = "uncertain"; break
            err = float(np.linalg.norm(j - target7)); hist.append(err)
            if corridor_fn is not None:
                lat = corridor_fn(j); lateral_max = max(lateral_max, lat)
                if lat > corridor_fn.limit:
                    event = "constraint_violation"; evidence = {"lateral_m": lat}; break
            if goal_fn is not None and goal_fn(j):
                event = "goal"; break
            if err < pass_tol:
                event = "converged"; break
            if len(hist) >= 3 and hist[-3] - hist[-1] < STALL_MIN_PROGRESS:
                event = "stalled"
                evidence = {"joint_err_rad": err, "progress_last_2_chunks_rad": hist[-3] - hist[-1], "joint_err_history": [round(h, 4) for h in hist[-6:]],
                            "commanded_minus_actual_rad": np.round(target7 - j, 4).tolist(), "note": "robot-motion evidence only; no inference about what stopped the motion"}
                break
        if event is None:
            event = "alloc_exhausted"
        return event, used, {"lateral_max": lateral_max, **evidence}

    def move_to_pose(self, position, quaternion_wxyz, mode="free", tol_pos=0.01, tol_ori_deg=5.0, step_budget=None, corridor=None, plan_only=False, label=""):
        """See contract.md. Returns a dict with status ∈ STATUSES, reached ≡ status == 'reached'."""
        t0 = time.time(); self.stats["moves"] += 1
        res = {"call": "move_to_pose", "label": label, "mode": mode, "budget": None, "steps_used": 0, "steps_planned": None, "plan_exceeds_budget": None, "n_waypoints": 0, "waypoints_completed": 0,
               "lateral_dev_max": 0.0, "constraint_check": None, "waypoint_events": [], "reached": False, "status": None, "detail": None, "reason": ""}

        def finish(status, detail=None, reason="", pos=None, quat=None, j0=None):
            j1 = self.joints()
            if j1 is not None:
                p1, q1 = self.model.fk_tcp(j1); res.update({"final_joints": j1, "final_tcp": p1, "final_quat": q1})
                if pos is not None:
                    res["pos_err"] = float(np.linalg.norm(p1 - pos)); res["ori_err_deg"] = quat_angle_deg(q1, quat)
                if j0 is not None:
                    res["total_jump"] = float(np.linalg.norm(j1 - j0))
            elif status not in ("invalid_input",):
                status, detail, reason = "uncertain", "joints_unobservable", "public joint observation missing"
            assert status in STATUSES, status
            res.update({"status": status, "detail": detail, "reason": reason, "reached": status == "reached", "wall_s": round(time.time() - t0, 3)})
            if status == "budget_exhausted":
                assert res["steps_used"] == res["budget"], (res["steps_used"], res["budget"])
            assert res["budget"] is None or res["steps_used"] <= res["budget"]
            return self._count(res)

        # ---- validate
        try:
            pos = _vec(position, 3, "position"); quat = validate_quat(quaternion_wxyz)
            tol_pos = validate_tol(tol_pos, "tol_pos"); tol_ori_deg = validate_tol(tol_ori_deg, "tol_ori_deg"); budget = validate_budget(step_budget)
            if corridor is not None:
                corridor = validate_tol(corridor, "corridor")
            if mode not in ("free", "linear"):
                raise InputError(f"mode: expected 'free' or 'linear', got {mode!r}")
        except InputError as e:
            res["budget"] = None
            return finish("invalid_input", "input", str(e))
        res.update({"requested_pos": pos, "requested_quat": quat, "tol_pos": tol_pos, "tol_ori_deg": tol_ori_deg, "budget": budget, "corridor": corridor})
        j0 = self.joints()
        if j0 is None:
            return finish("uncertain", "joints_unobservable", "public joint observation missing before motion")
        p0, q0 = self.model.fk_tcp(j0); res.update({"start_joints": j0, "start_tcp": p0})

        def goal_fn(j):
            rp, ro = self.model.residual(j, pos, quat); return rp <= tol_pos and ro <= tol_ori_deg

        # ---- plan (schedule only; never a refusal)
        if mode == "linear":
            dist = float(np.linalg.norm(pos - p0)); n = max(1, int(np.ceil(dist / LINEAR_SPACING)))
            wps = []; cur = j0.copy()
            for k in range(1, n + 1):
                t = k / n; pk_ = p0 + (pos - p0) * t; qk = slerp(q0, quat, t) if k < n else quat
                r = self.model.plan_ik(pk_, qk, cur)
                if r["status"] != "ok":
                    self.stats["ik_failed"] += 1
                    return finish("unreachable", r["status"], f"linear waypoint {k}/{n}: {r['status']} residual {r.get('residual_pos')} m / {r.get('residual_ori')} deg", pos, quat, j0)
                if r["method"] != "anchored":
                    self.stats["ik_fallback"] += 1
                wps.append(r["joints"]); cur = r["joints"]
            steps = [float(np.linalg.norm(wps[0] - j0))] + [float(np.linalg.norm(wps[i] - wps[i - 1])) for i in range(1, len(wps))]
            if max(steps) > MAX_WAYPOINT_STEP * 1.5:
                return finish("unreachable", "discontinuous_path", f"max joint step between {LINEAR_SPACING*100:.0f} cm waypoints {max(steps):.2f} rad > {MAX_WAYPOINT_STEP*1.5:.2f}", pos, quat, j0)
            line_dir = (pos - p0) / max(dist, 1e-9)
            if corridor is not None:
                def corridor_fn(j):
                    p, _ = self.model.fk_tcp(j); d = p - p0; return float(np.linalg.norm(d - line_dir * float(np.dot(d, line_dir))))
                corridor_fn.limit = float(corridor); res["constraint_check"] = f"sampled_every_{CHUNK_STEPS}_steps"
            else:
                corridor_fn = None
        else:
            r = self.model.plan_ik(pos, quat, j0)
            if r["status"] != "ok":
                self.stats["ik_failed"] += 1
                return finish("unreachable", r["status"], f"{r['status']}: residual {r.get('residual_pos')} m / {r.get('residual_ori')} deg", pos, quat, j0)
            if r["method"] != "anchored":
                self.stats["ik_fallback"] += 1
            jt = r["joints"]; total = float(np.linalg.norm(jt - j0))
            wps = [jt]; steps = [total]; corridor_fn = None   # free mode: direct joint target (same controller command as the legacy call), chunked monitoring
        needed = sum(self._steps_needed(wps[i] - (j0 if i == 0 else wps[i - 1])) for i in range(len(wps))) - SETTLE_STEPS * (len(wps) - 1)
        res.update({"n_waypoints": len(wps), "max_waypoint_step": float(max(steps)), "steps_planned": int(needed), "plan_exceeds_budget": bool(needed > budget), "target_joints": wps[-1]})
        if plan_only:
            return self._plan_only_result(res, pos, quat, j0, t0)
        if goal_fn(j0):   # already within tolerance: 0 steps, reached
            return finish("reached", "already_within_tolerance", "start pose already within the requested tolerance", pos, quat, j0)
        # ---- execute: allocations are a schedule; the remainder always flows to the following waypoints
        remaining = budget; refined = False; i = 0
        while i < len(wps) and remaining > 0:
            wp = wps[i]; prev = j0 if i == 0 else wps[i - 1]; last = i == len(wps) - 1
            alloc = remaining if last else min(remaining, self._steps_needed(wp - prev))
            ev, used, evd = self._execute_waypoint(wp, alloc, corridor_fn, goal_fn if last else None, pass_tol=CONV_TOL if last else PASS_TOL)
            remaining -= used; res["steps_used"] += used; res["lateral_dev_max"] = max(res["lateral_dev_max"], evd.get("lateral_max", 0.0))
            res["waypoint_events"].append({"i": i, "event": ev, "steps": used})
            if ev == "uncertain":
                return finish("uncertain", "joints_unobservable", "public joint observation lost during motion", pos, quat, j0)
            if ev == "constraint_violation":
                return finish("constraint_violation", "corridor", f"lateral deviation {evd['lateral_m']*1000:.1f} mm > corridor {corridor*1000:.1f} mm at waypoint {i+1}/{len(wps)} (sampled every {CHUNK_STEPS} steps)", pos, quat, j0)
            if ev == "stalled":
                res["motion_evidence"] = {k: v for k, v in evd.items() if k != "lateral_max"}
                return finish("stalled", "no_progress", f"no joint progress at waypoint {i+1}/{len(wps)} (joint err {evd['joint_err_rad']:.3f} rad, progress {evd['progress_last_2_chunks_rad']:.4f} rad / 2 chunks)", pos, quat, j0)
            if ev == "goal":
                res["waypoints_completed"] = i + 1
                return finish("reached", "within_tolerance", "", pos, quat, j0)
            res["waypoints_completed"] = i + 1
            if last and ev == "converged" and not goal_fn(self.joints()):
                if refined or remaining <= 0:
                    break
                # joint target reached but the pose is not within the caller's tolerance: one re-solve from the actual joints
                r = self.model.plan_ik(pos, quat, self.joints())
                if r["status"] != "ok":
                    break
                wps = wps + [r["joints"]]; refined = True; res["n_waypoints"] = len(wps); res["refined"] = True
            i += 1
        j1 = self.joints()
        if j1 is not None and goal_fn(j1):
            return finish("reached", "within_tolerance", "", pos, quat, j0)
        if res["steps_used"] >= budget:
            return finish("budget_exhausted", "budget", f"{budget} control steps spent; pose not within tolerance (waypoints completed {res['waypoints_completed']}/{len(wps)}; plan estimate {needed})", pos, quat, j0)
        return finish("unreachable", "precision_limit", "joint target converged but the pose residual exceeds the caller's tolerance after one re-solve (IK verification tolerance 3 mm / 2 deg)", pos, quat, j0)

    def _plan_only_result(self, res, pos, quat, j0, t0):
        p1, q1 = self.model.fk_tcp(j0)
        res.update({"plan_only": True, "status": "uncertain", "detail": "plan_only", "reason": f"plan only: ~{res['steps_planned']} steps, {res['n_waypoints']} waypoints; no motion", "reached": False,
                    "final_joints": j0, "final_tcp": p1, "final_quat": q1, "pos_err": float(np.linalg.norm(p1 - pos)), "ori_err_deg": quat_angle_deg(q1, quat), "total_jump": 0.0, "wall_s": round(time.time() - t0, 3)})
        return self._count(res)

    def move_to_joints(self, joints, step_budget=None, tol_joint=CONV_TOL):
        """Joint-space move with the same contract (reached = joint error < tol_joint from the public joints)."""
        t0 = time.time(); self.stats["moves"] += 1
        res = {"call": "move_to_joints", "label": "move_to_joints", "mode": "joints", "budget": None, "steps_used": 0, "steps_planned": None, "plan_exceeds_budget": None, "n_waypoints": 0, "waypoints_completed": 0, "waypoint_events": [], "reached": False}

        def finish(status, detail=None, reason="", jt=None, j0=None):
            j1 = self.joints()
            if j1 is not None:
                p1, q1 = self.model.fk_tcp(j1); res.update({"final_joints": j1, "final_tcp": p1, "final_quat": q1})
                if jt is not None:
                    res["joint_err"] = float(np.linalg.norm(j1 - jt))
                if j0 is not None:
                    res["total_jump"] = float(np.linalg.norm(j1 - j0))
            elif status != "invalid_input":
                status, detail, reason = "uncertain", "joints_unobservable", "public joint observation missing"
            assert status in STATUSES
            res.update({"status": status, "detail": detail, "reason": reason, "reached": status == "reached", "wall_s": round(time.time() - t0, 3)})
            if status == "budget_exhausted":
                assert res["steps_used"] == res["budget"]
            return self._count(res)

        try:
            jt = _vec(joints, 7, "joints"); budget = validate_budget(step_budget); tol_joint = validate_tol(tol_joint, "tol_joint")
        except InputError as e:
            return finish("invalid_input", "input", str(e))
        res.update({"target_joints": jt, "budget": budget, "tol_joint": tol_joint})
        if self.model.limit_violation(jt) > 1e-6:
            return finish("unreachable", "joint_limits", f"target joints violate the URDF limits (margin {SERVER_LIMIT_MARGIN} rad) by {self.model.limit_violation(jt):.3f} rad", jt)
        j0 = self.joints()
        if j0 is None:
            return finish("uncertain", "joints_unobservable", "public joint observation missing before motion")
        p0, _ = self.model.fk_tcp(j0); res.update({"start_joints": j0, "start_tcp": p0})
        total, _ = self.model.joint_distance(jt, j0); n = max(1, int(np.ceil(total / MAX_WAYPOINT_STEP)))
        wps = [j0 + (jt - j0) * k / n for k in range(1, n + 1)]
        needed = sum(self._steps_needed(wps[i] - (j0 if i == 0 else wps[i - 1])) for i in range(n)) - SETTLE_STEPS * (n - 1)
        res.update({"n_waypoints": n, "max_waypoint_step": total / n, "steps_planned": int(needed), "plan_exceeds_budget": bool(needed > budget)})
        goal_fn = lambda j: float(np.linalg.norm(j - jt)) < tol_joint  # noqa: E731
        if goal_fn(j0):
            return finish("reached", "already_within_tolerance", "", jt, j0)
        remaining = budget
        for i, wp in enumerate(wps):
            if remaining <= 0:
                break
            prev = j0 if i == 0 else wps[i - 1]; last = i == n - 1
            alloc = remaining if last else min(remaining, self._steps_needed(wp - prev))
            ev, used, evd = self._execute_waypoint(wp, alloc, None, goal_fn if last else None)
            remaining -= used; res["steps_used"] += used; res["waypoint_events"].append({"i": i, "event": ev, "steps": used})
            if ev == "uncertain":
                return finish("uncertain", "joints_unobservable", "public joint observation lost during motion", jt, j0)
            if ev == "stalled":
                res["motion_evidence"] = {k: v for k, v in evd.items() if k != "lateral_max"}
                return finish("stalled", "no_progress", f"no joint progress at waypoint {i+1}/{n} (joint err {evd['joint_err_rad']:.3f} rad)", jt, j0)
            res["waypoints_completed"] = i + 1
            if ev in ("goal", "converged") and last:
                break
        j1 = self.joints()
        if j1 is not None and goal_fn(j1):
            return finish("reached", "within_tolerance", "", jt, j0)
        if res["steps_used"] >= budget:
            return finish("budget_exhausted", "budget", f"{budget} control steps spent; joint error {float(np.linalg.norm(j1 - jt)):.4f} rad > {tol_joint}", jt, j0)
        return finish("unreachable", "precision_limit", "controller converged (< CONV_TOL) but the caller's joint tolerance is tighter", jt, j0)

    def last_motion_status(self):
        return self.last_result

    def close(self):
        self._log({"call": "stats", **self.stats, "jit_s": self.model.jit_s})
        if self.log_f:
            self.log_f.close()
