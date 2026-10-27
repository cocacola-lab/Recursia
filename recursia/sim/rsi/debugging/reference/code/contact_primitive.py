"""General contact-aware primitive MC (round contact_primitive_20260917_112613). See ../contract_mc.md.

Public robot state only (7 arm joints + gripper opening fraction) and the Panda URDF FK/IK shared with the frozen R module
(`motion_interface_r.PandaModel`). No object, contact-sensor, scene, task or seed information enters any decision. The
free-space interface R is not modified; contact mode is entered only through `contact_move`.

Constants (robot/controller/gripper quantities):
  CHUNK_STEPS 10        feedback granularity (same as R)
  V_EFF, SETTLE_STEPS   controller speed used only to schedule per-waypoint step allocations (R constants)
  SAG_REF 0.005 m       steady-state along-axis lag of the joint-position controller after convergence (parent rounds: 0.5-0.7 cm)
  SHORTFALL_MIN 0.008 m contact needs a commanded-minus-actual shortfall clearly above the sag
  PROGRESS_RATIO_MAX 0.4  a waypoint counts as a contact candidate when the actual advance is <= 40% of the commanded advance
  CANDIDATES_NEEDED 2   two consecutive candidate waypoints before a confirmation hold (at the command limit one candidate
                        with an axial shortfall >= SHORTFALL_MIN also triggers the hold: dev finding from setLR5_114151 table_press)
  HOLD_STEPS 10         confirmation hold (command held, TCP must stay within HOLD_STATIONARY = 0.002 m)
  ALIGN_MIN 0.6         along-axis component of the shortfall vector / its norm
  AXIS_TOL_DEG 15       approach_axis precondition
  GRIP_OPEN_MIN 0.8, GRIP_HOLD_RANGE (0.03, 0.9)  gripper opening fraction preconditions
"""
from __future__ import annotations

import json
import math
import numbers
import time
from pathlib import Path

import numpy as np

from motion_interface_r import (CHUNK_STEPS, CONV_TOL, IK_WORKSPACE_HI, IK_WORKSPACE_LO, SETTLE_STEPS, V_EFF, InputError, PandaModel, _vec,
                                quat_angle_deg, validate_budget, validate_quat, validate_tol)

SAG_REF = 0.005
SHORTFALL_MIN = 0.008
PROGRESS_RATIO_MAX = 0.4
CANDIDATES_NEEDED = 2
HOLD_STEPS = 10
HOLD_STATIONARY = 0.002
ALIGN_MIN = 0.6
AXIS_TOL_DEG = 15.0
GRIP_OPEN_MIN = 0.8
GRIP_HOLD_RANGE = (0.03, 0.9)
ALONG_CONV_TOL = 0.003       # a waypoint counts as executed when the actual along-axis coordinate is within 3 mm of the command
MAX_CONFIRM_FAILURES = 2
STATUSES = ("reached", "contact_established", "stalled", "uncertain", "budget_exhausted", "failed", "invalid_input")
SUCCESS_STATUSES = ("reached", "contact_established")
FAILED_DETAILS = ("precondition_start_pose", "precondition_axis", "precondition_gripper", "target_out_of_workspace", "ik_failed", "lateral_corridor",
                  "premature_contact", "late_contact")


def _finite_pos(v, name, lo, hi):
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)) or not (lo <= float(v) <= hi):
        raise InputError(f"{name}: expected a finite number in [{lo}, {hi}], got {v!r}")
    return float(v)


class ContactPrimitive:
    """Bounded contact-seeking straight-line motion with evidence-based contact detection (contract_mc.md)."""

    def __init__(self, low_env, model: PandaModel | None = None, log_path: Path | None = None, arm: str = "MC"):
        self.low = low_env; self.arm = arm; self.model = model if model is not None else PandaModel()
        self.log_f = open(log_path, "a") if log_path else None
        self.stats = {"calls": 0, "sim_steps": 0, "status_counts": {}, "success": 0, "non_success": 0}
        self.last_result = None

    # ------------------------------------------------------------------ public robot state
    def joints(self):
        obs = getattr(self.low, "_current_obs", None)
        if obs is None or "robot0_joint_pos" not in obs:
            return None
        j = np.asarray(obs["robot0_joint_pos"], float)
        if j.shape[0] < 7 or not np.all(np.isfinite(j[:7])):
            return None
        return j[:7].copy()

    def gripper_opening(self):
        obs = getattr(self.low, "_current_obs", None)
        if obs is None or "robot0_gripper_qpos" not in obs:
            return None
        v = float(np.asarray(obs["robot0_gripper_qpos"], float)[0] / self.low.gripper_metric_length)
        return v if math.isfinite(v) else None

    def _log(self, rec):
        if self.log_f:
            self.log_f.write(json.dumps({"t": round(time.time(), 3), **rec}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); self.log_f.flush()

    def _steps_needed(self, dq):
        return int(np.ceil(np.max(np.abs(dq)) / V_EFF)) + SETTLE_STEPS

    # ------------------------------------------------------------------ the primitive
    def contact_move(self, target_position, quaternion_wxyz, *, approach_axis=None, contact_tol_before=0.03, overshoot_after=0.01, start_pose=None, start_tol=0.02,
                     tol_pos=0.01, tol_ori_deg=5.0, gripper_expect=None, lateral_corridor=0.03, step_budget=240, wall_budget_s=60.0, increment=0.01, label=""):
        t0 = time.time(); self.stats["calls"] += 1
        res = {"call": "contact_move", "label": label, "phase": "precondition", "status": None, "detail": None, "reason": "", "success": False, "budget": None, "steps_used": 0,
               "wall_budget_s": None, "trace": [], "hold": None, "evidence": {}, "lateral_max": 0.0, "n_waypoints_executed": 0, "along_final": None}

        def finish(status, detail=None, reason="", target=None, quat=None):
            j1 = self.joints()
            if j1 is not None:
                p1, q1 = self.model.fk_tcp(j1); res.update({"final_joints": j1, "final_tcp": p1, "final_quat": q1, "gripper_opening_final": self.gripper_opening()})
                if target is not None:
                    res["pos_err"] = float(np.linalg.norm(p1 - target)); res["ori_err_deg"] = quat_angle_deg(q1, quat)
                if "start_tcp" in res and "u" in res:
                    res["along_final"] = float(np.dot(p1 - res["start_tcp"], res["u"]))
            elif status not in ("invalid_input",):
                status, detail, reason = "uncertain", "joints_unobservable", "public joint observation missing"
            assert status in STATUSES, status
            if status == "failed":
                assert detail in FAILED_DETAILS, detail
            res.update({"status": status, "detail": detail, "reason": reason, "success": status in SUCCESS_STATUSES, "wall_s": round(time.time() - t0, 3)})
            if status == "budget_exhausted" and detail == "steps":
                assert res["steps_used"] == res["budget"], (res["steps_used"], res["budget"])
            assert res["budget"] is None or res["steps_used"] <= res["budget"]
            if status == "contact_established":
                assert res["hold"] is not None and res["evidence"].get("confirmed") is True
            self.stats["sim_steps"] += res["steps_used"]; self.stats["success" if res["success"] else "non_success"] += 1
            self.stats["status_counts"][status] = self.stats["status_counts"].get(status, 0) + 1
            self.last_result = res; self._log({"sim_step": int(getattr(self.low, "_sim_step_count", -1)), **{k: v for k, v in res.items() if k != "trace"}, "n_trace": len(res["trace"])})
            return res

        # ---- input contract (no motion)
        try:
            target = _vec(target_position, 3, "target_position"); quat = validate_quat(quaternion_wxyz)
            tol_pos = validate_tol(tol_pos, "tol_pos"); tol_ori_deg = validate_tol(tol_ori_deg, "tol_ori_deg"); budget = validate_budget(step_budget)
            wall_budget_s = validate_tol(wall_budget_s, "wall_budget_s"); increment = _finite_pos(increment, "increment", 1e-4, 0.05)
            contact_tol_before = _finite_pos(contact_tol_before, "contact_tol_before", 0.0, 0.5); overshoot_after = _finite_pos(overshoot_after, "overshoot_after", 0.0, 0.1)
            start_tol = validate_tol(start_tol, "start_tol"); lateral_corridor = validate_tol(lateral_corridor, "lateral_corridor")
            if gripper_expect not in (None, "open", "holding"):
                raise InputError(f"gripper_expect: expected None|'open'|'holding', got {gripper_expect!r}")
            axis = None
            if approach_axis is not None:
                axis = _vec(approach_axis, 3, "approach_axis"); n = float(np.linalg.norm(axis))
                if n < 1e-6:
                    raise InputError("approach_axis: zero norm")
                axis = axis / n
            if start_pose is not None:
                start_pose = _vec(start_pose, 3, "start_pose")
        except InputError as e:
            return finish("invalid_input", "input", str(e))
        res.update({"budget": budget, "wall_budget_s": wall_budget_s, "target": target, "quat": quat, "tol_pos": tol_pos, "tol_ori_deg": tol_ori_deg, "lateral_corridor": lateral_corridor,
                    "contact_tol_before": contact_tol_before, "overshoot_after": overshoot_after, "increment": increment, "gripper_expect": gripper_expect})
        # ---- preconditions (no motion)
        j0 = self.joints()
        if j0 is None:
            return finish("uncertain", "joints_unobservable", "public joint observation missing before motion", target, quat)
        p0, q0 = self.model.fk_tcp(j0); res.update({"start_joints": j0, "start_tcp": p0, "start_quat": q0, "gripper_opening_start": self.gripper_opening()})
        travel = float(np.linalg.norm(target - p0))
        if not (0.005 <= travel <= 0.5):
            return finish("invalid_input", "input", f"travel {travel:.4f} m outside [0.005, 0.5]", target, quat)
        u = (target - p0) / travel; res.update({"u": u, "travel": travel, "window": [max(0.0, travel - contact_tol_before), travel + overshoot_after]})
        if start_pose is not None:
            d = float(np.linalg.norm(p0 - start_pose)); oe = quat_angle_deg(q0, quat)
            if d > start_tol or oe > tol_ori_deg:
                return finish("failed", "precondition_start_pose", f"actual TCP {d*100:.1f} cm / {oe:.1f} deg from the declared start pose (tol {start_tol*100:.1f} cm / {tol_ori_deg} deg)", target, quat)
        if axis is not None:
            ang = float(np.degrees(np.arccos(np.clip(np.dot(u, axis), -1.0, 1.0))))
            res["axis_angle_deg"] = ang
            if ang > AXIS_TOL_DEG:
                return finish("failed", "precondition_axis", f"direction to target deviates {ang:.1f} deg from the approach axis (tol {AXIS_TOL_DEG} deg)", target, quat)
        g = self.gripper_opening()
        if gripper_expect is not None:
            if g is None:
                return finish("uncertain", "gripper_unobservable", "gripper opening not observable", target, quat)
            if gripper_expect == "open" and g < GRIP_OPEN_MIN:
                return finish("failed", "precondition_gripper", f"gripper opening {g:.2f} < {GRIP_OPEN_MIN} (expected open)", target, quat)
            if gripper_expect == "holding" and not (GRIP_HOLD_RANGE[0] <= g <= GRIP_HOLD_RANGE[1]):
                return finish("failed", "precondition_gripper", f"gripper opening {g:.2f} outside {GRIP_HOLD_RANGE} (expected holding)", target, quat)
        end_pt = p0 + u * (travel + overshoot_after)
        for pt, nm in ((target, "target"), (end_pt, "overshoot end")):
            if np.any(pt < IK_WORKSPACE_LO) or np.any(pt > IK_WORKSPACE_HI):
                return finish("failed", "target_out_of_workspace", f"{nm} {np.round(pt, 3).tolist()} outside the public IK box", target, quat)

        # ---- search phase
        res["phase"] = "search"
        c_max = travel + overshoot_after; n_wp = int(np.ceil(c_max / increment - 1e-9))
        remaining = budget; cur_j = j0.copy(); a_prev = 0.0; c_prev = 0.0; candidates = 0; confirm_failures = 0; lateral_max = 0.0

        def measure():
            j = self.joints()
            if j is None:
                return None
            p, q = self.model.fk_tcp(j); d = p - p0; a = float(np.dot(d, u)); lat = float(np.linalg.norm(d - u * a))
            return {"j": j, "p": p, "q": q, "a": a, "lat": lat}

        def within_goal(m):
            return float(np.linalg.norm(m["p"] - target)) <= tol_pos and quat_angle_deg(m["q"], quat) <= tol_ori_deg

        def run_chunks(jt, alloc, c_cmd, goal_check=True):
            """Execute up to `alloc` control steps toward joint target jt in chunks; returns (event, used, last_measure)."""
            nonlocal lateral_max
            used = 0; m = None
            while used < alloc:
                k = min(CHUNK_STEPS, alloc - used); s0 = self.low._sim_step_count
                self.low.move_to_joints_blocking(jt, tolerance=CONV_TOL, max_steps=k)
                used += int(self.low._sim_step_count - s0)
                if time.time() - t0 > wall_budget_s:
                    return "wall", used, measure()
                m = measure()
                if m is None:
                    return "uncertain", used, None
                jerr = float(np.linalg.norm(m["j"] - jt)); lateral_max = max(lateral_max, m["lat"])
                res["trace"].append({"sim_step": int(self.low._sim_step_count), "c": round(c_cmd, 5), "a": round(m["a"], 5), "lat": round(m["lat"], 5), "jerr": round(jerr, 5), "p": np.round(m["p"], 5).tolist()})
                if m["lat"] > lateral_corridor:
                    return "corridor", used, m
                if goal_check and within_goal(m):
                    return "goal", used, m
                if abs(c_cmd - m["a"]) <= ALONG_CONV_TOL or jerr < CONV_TOL:
                    return "converged", used, m
            return "alloc", used, m

        def confirm_hold(jt, m, c):
            """Confirmation hold at the current command. Returns a final result (finish) or None when inconclusive (search continues)."""
            nonlocal remaining, lateral_max, confirm_failures, candidates, cur_j, a_prev
            res["phase"] = "confirm"
            hold_alloc = min(HOLD_STEPS, remaining); p_before = m["p"].copy(); s0 = self.low._sim_step_count
            self.low.move_to_joints_blocking(jt, tolerance=CONV_TOL, max_steps=hold_alloc)
            used = int(self.low._sim_step_count - s0); remaining -= used; res["steps_used"] = budget - remaining
            mh = measure()
            if mh is None:
                return finish("uncertain", "joints_unobservable", "public joint observation lost during the confirmation hold", target, quat)
            lateral_max = max(lateral_max, mh["lat"]); res["lateral_max"] = lateral_max
            p_cmd, _ = self.model.fk_tcp(jt); r_vec = p_cmd - mh["p"]; r_norm = float(np.linalg.norm(r_vec)); along = float(np.dot(r_vec, u))
            moved = float(np.linalg.norm(mh["p"] - p_before)); align = along / r_norm if r_norm > 1e-9 else 0.0
            hold = {"steps": used, "moved_m": round(moved, 5), "shortfall_vec": np.round(r_vec, 5).tolist(), "shortfall_along": round(along, 5), "align": round(align, 3), "a": round(mh["a"], 5),
                    "lat": round(mh["lat"], 5), "sim_step": int(self.low._sim_step_count), "p": np.round(mh["p"], 5).tolist(), "p_cmd": np.round(p_cmd, 5).tolist()}
            res["hold"] = hold; res["trace"].append({"sim_step": hold["sim_step"], "c": round(c, 5), "a": hold["a"], "lat": hold["lat"], "jerr": round(float(np.linalg.norm(mh["j"] - jt)), 5), "p": hold["p"], "hold": True})
            if mh["lat"] > lateral_corridor:
                return finish("failed", "lateral_corridor", f"lateral deviation {mh['lat']*1000:.1f} mm > corridor during the hold", target, quat)
            stationary = moved <= HOLD_STATIONARY; axial = along >= SHORTFALL_MIN and align >= ALIGN_MIN
            res["evidence"] = {"confirmed": bool(stationary and axial), "stationary": stationary, "axial": axial, "hold_moved_m": round(moved, 5), "shortfall_along_m": round(along, 5), "align": round(align, 3),
                               "candidate_waypoints": candidates, "along_at_contact": round(mh["a"], 5), "window": res["window"], "note": "public joints/FK only; no inference about what body was touched"}
            if stationary and axial:
                a_c = mh["a"]; lo, hi = res["window"]
                if a_c < lo:
                    return finish("failed", "premature_contact", f"axial contact confirmed at {a_c*100:.1f} cm, before the expected window [{lo*100:.1f}, {hi*100:.1f}] cm", target, quat)
                if a_c > hi + ALONG_CONV_TOL:
                    return finish("failed", "late_contact", f"axial contact confirmed at {a_c*100:.1f} cm, beyond the expected window", target, quat)
                return finish("contact_established", "axial_contact_confirmed", f"axial contact at {a_c*100:.1f} cm of {travel*100:.1f} cm (shortfall {along*100:.1f} cm, align {align:.2f}, hold motion {moved*1000:.1f} mm)", target, quat)
            if stationary and not axial:
                return finish("stalled", "misaligned_residual", f"robot stationary during the hold but the shortfall is not axial (along {along*100:.1f} cm, align {align:.2f})", target, quat)
            confirm_failures += 1; candidates = 0; res["phase"] = "search"; cur_j = mh["j"]; a_prev = mh["a"]
            if confirm_failures >= MAX_CONFIRM_FAILURES:
                return finish("uncertain", "inconsistent_evidence", f"{confirm_failures} confirmation holds showed motion ({moved*1000:.1f} mm) after low-progress waypoints", target, quat)
            return None

        for k in range(1, n_wp + 1):
            c = min(c_max, k * increment); pk = p0 + u * c
            r = self.model.plan_ik(pk, quat, cur_j)
            if r["status"] != "ok":
                res["steps_used"] = budget - remaining; res["lateral_max"] = lateral_max
                return finish("failed", "ik_failed", f"waypoint {k}/{n_wp} ({r['status']}): no verified IK solution", target, quat)
            jt = r["joints"]; alloc = min(remaining, self._steps_needed(jt - cur_j))
            if alloc <= 0:
                break
            ev, used, m = run_chunks(jt, alloc, c); remaining -= used; res["steps_used"] = budget - remaining; res["lateral_max"] = lateral_max; res["n_waypoints_executed"] = k
            if ev == "uncertain":
                return finish("uncertain", "joints_unobservable", "public joint observation lost during the search phase", target, quat)
            if ev == "wall":
                return finish("budget_exhausted", "wall_time", f"wall budget {wall_budget_s}s spent during the search phase", target, quat)
            if ev == "corridor":
                return finish("failed", "lateral_corridor", f"lateral deviation {m['lat']*1000:.1f} mm > corridor {lateral_corridor*1000:.1f} mm (sampled every {CHUNK_STEPS} steps)", target, quat)
            if ev == "goal":
                return finish("reached", "within_tolerance", "", target, quat)
            cur_j = m["j"]
            adv_cmd = c - c_prev; adv_act = m["a"] - a_prev; shortfall = c - m["a"]; ratio = adv_act / adv_cmd if adv_cmd > 1e-9 else 1.0
            res["trace"][-1].update({"wp": k, "ratio": round(ratio, 3), "shortfall": round(shortfall, 5)})
            candidate = ratio <= PROGRESS_RATIO_MAX and shortfall >= SHORTFALL_MIN
            candidates = candidates + 1 if candidate else 0
            a_prev = m["a"]; c_prev = c
            if candidates >= CANDIDATES_NEEDED:
                # ---- confirmation hold: keep the current command, the TCP must stay put and the shortfall must be axial
                if remaining <= 0:
                    break
                r_hold = confirm_hold(jt, m, c)
                if r_hold is not None:
                    return r_hold
            if remaining <= 0:
                break
        m = measure()
        if m is None:
            return finish("uncertain", "joints_unobservable", "public joint observation missing at the end of the search", target, quat)
        if within_goal(m):
            return finish("reached", "within_tolerance", "", target, quat)
        if remaining > 0 and candidates >= 1 and (c_prev - m["a"]) >= SHORTFALL_MIN:
            # the command limit was reached with a low-progress waypoint and an axial shortfall above the sag: one confirmation hold
            r_hold = confirm_hold(jt, m, c_prev)
            if r_hold is not None:
                return r_hold
            m = measure()
            if m is None:
                return finish("uncertain", "joints_unobservable", "public joint observation missing after the end-of-search hold", target, quat)
        if remaining <= 0:
            return finish("budget_exhausted", "steps", f"{budget} control steps spent; along {m['a']*100:.1f} cm of {travel*100:.1f} cm, no confirmed contact", target, quat)
        return finish("stalled", "no_contact_no_arrival", f"search reached the command limit {c_max*100:.1f} cm; TCP at {m['a']*100:.1f} cm, pos err {np.linalg.norm(m['p']-target)*100:.1f} cm; no confirmed contact", target, quat)

    def close(self):
        self._log({"call": "stats", **self.stats})
        if self.log_f:
            self.log_f.close()
