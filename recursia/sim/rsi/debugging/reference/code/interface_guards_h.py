"""Interface revision H (round binding_interface_20260922): goto_home continuation with a joint-distance budget.

Finding (posthoc_first_failure_20260922): 64/104 goto_home calls ended budget_exhausted after the production 120-step cap,
51 of them with joint 7 (wrist) as the largest residual; the next grasp attempt then started away from home and every later
precondition refused. The production command itself is fine, it just stops after 120 steps.

reported_home_h keeps the production command unchanged and re-issues it while it is making progress:
  budget   = clip(ceil(max_i |q_i - home_i| / V_EFF) + SETTLE_STEPS, HOME_MAX_STEPS, HOME_CAP_STEPS)
  loop     : call production home (<= 120 steps); stop when joint error < HOME_CONV_TOL (reached),
             steps_used >= budget (budget_exhausted; checked between production calls, so it may overrun by < 120 steps),
             or the error fell by < HOME_MIN_PROGRESS over one call (stalled).
Reporting format and statuses are those of interface_guards_g.reported_home, plus n_calls and budget_rule.
Constants: V_EFF / SETTLE_STEPS from motion_interface_r (controller speed used for scheduling), HOME_CAP_STEPS 600 (5x legacy),
HOME_MIN_PROGRESS 0.02 rad (= motion_interface_r.STALL_MIN_PROGRESS).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from interface_guards_g import HOME_CONV_TOL, HOME_MAX_STEPS, _write
from motion_interface_r import SETTLE_STEPS, STALL_MIN_PROGRESS, V_EFF

HOME_CAP_STEPS = 600
HOME_MIN_PROGRESS = STALL_MIN_PROGRESS


def home_budget(j0, home):
    need = math.ceil(float(np.max(np.abs(np.asarray(j0, float) - np.asarray(home, float)))) / V_EFF) + SETTLE_STEPS
    return int(min(HOME_CAP_STEPS, max(HOME_MAX_STEPS, need)))


def reported_home_h(fn, mi, log_path=None, printer=print, stats=None):
    stats = stats if stats is not None else {"calls": 0, "reached": 0, "not_reached": 0, "refused": 0, "sim_steps": 0, "status_counts": {}, "production_calls": 0}
    stats.setdefault("production_calls", 0)

    def finish(res):
        stats["calls"] += 1; stats["sim_steps"] += int(res.get("steps_used", 0)); stats["production_calls"] += int(res.get("n_calls", 0))
        stats["status_counts"][res["status"]] = stats["status_counts"].get(res["status"], 0) + 1
        if res["status"] == "failed":
            stats["refused"] += 1
        stats["reached" if res["reached"] else "not_reached"] += 1
        printer("HOME_MOTION " + json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), flush=True)
        _write(log_path, {"call": "goto_home", **res})
        return res

    def wrapped():
        t0 = time.time(); s0 = int(getattr(mi.low, "_sim_step_count", 0))
        home = getattr(mi.low, "home_joint_position", None)
        res = {"call": "goto_home", "label": "goto_home", "status": None, "detail": None, "reached": False, "steps_used": 0,
               "budget": None, "budget_rule": "joint_distance", "tol_joint": HOME_CONV_TOL, "executed": False, "n_calls": 0}
        if home is None:
            res.update({"status": "failed", "detail": "home_unavailable", "reason": "environment exposes no home configuration"}); return finish(res)
        home = np.asarray(home, float).reshape(7); res["target_joints"] = home
        viol = float(mi.model.limit_violation(home))
        if viol > 1e-6:
            res.update({"status": "failed", "detail": "joint_limits", "reason": f"home target violates the URDF limits by {viol:.3f} rad"}); return finish(res)
        j0 = mi.joints()
        if j0 is None:
            res.update({"status": "uncertain", "detail": "joints_unobservable", "reason": "public joint observation missing before motion"}); return finish(res)
        budget = home_budget(j0, home)
        res.update({"start_joints": j0, "joint_err_start": float(np.linalg.norm(j0 - home)), "budget": budget})
        prev = res["joint_err_start"]; err = prev; j1 = j0
        while True:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                res.update({"status": "uncertain", "detail": "exception", "reason": repr(e), "executed": True,
                            "steps_used": int(getattr(mi.low, "_sim_step_count", s0)) - s0}); finish(res); raise
            res["n_calls"] += 1; res["executed"] = True; res["steps_used"] = int(getattr(mi.low, "_sim_step_count", s0)) - s0
            j1 = mi.joints()
            if j1 is None:
                res.update({"status": "uncertain", "detail": "joints_unobservable", "reason": "public joint observation missing after motion"}); return finish(res)
            err = float(np.linalg.norm(j1 - home))
            if err < HOME_CONV_TOL:
                res.update({"status": "reached", "detail": "within_tolerance", "reached": True}); break
            if res["steps_used"] >= budget:
                res.update({"status": "budget_exhausted", "detail": "budget", "reason": f"{res['steps_used']} steps >= joint-distance budget {budget}; joint error {err:.4f} rad"}); break
            if prev - err < HOME_MIN_PROGRESS:
                res.update({"status": "stalled", "detail": "no_progress", "reason": f"joint error fell {prev - err:.4f} rad over one production call (< {HOME_MIN_PROGRESS})"}); break
            prev = err
        p1, q1 = mi.model.fk_tcp(j1)
        res.update({"final_joints": j1, "final_tcp": p1, "final_quat": q1, "joint_err": err, "wall_s": round(time.time() - t0, 3)})
        return finish(res)

    wrapped.__name__ = "goto_home_joint_position"; wrapped.stats = stats
    return wrapped
