"""Shared interface guards, revision G (round zeroshot_general_grasp_contract_20260918_OxvXZB).

Injected by the runner for arms G1 and G2 only; R1/R2 run the parent's injection byte-for-byte
(code/frozen_parent/interface_guards_parent.py). No object / task / scene / seed information enters any decision.

bounded_plan_grasp(fn, log_path)
    NARROW conversion (parent gap: the parent converted EVERY AssertionError). Only the production GraspNet empty-candidate
    signal -- `AssertionError` whose message is exactly NO_CANDIDATES_MSG (libero_reduced.py:318 / :345) -- is converted to the
    bounded value `(None, None)` + a `GRASP_CANDIDATES {"status":"failed","detail":"no_candidates"}` record. Any other
    AssertionError, and every other exception class, propagates unchanged.

reported_home(fn, mi, log_path)
    The production `goto_home_joint_position` is a joint move outside the general contract (parent report: "production
    goto_home_joint_position is a production joint move outside R"). This wrapper keeps the production motion command
    unchanged (same target = the environment's reset configuration, same blocking call, same 120-step cap) and REPORTS the
    actual termination from the public joints: status reached | budget_exhausted | stalled | failed(joint_limits) |
    uncertain, joint error, steps used. Prerequisite (target inside the URDF limits with the server margin) is checked before
    the motion; a violation refuses the move. Nothing is assumed: `reached` is true only when the public joint error is below
    the controller convergence tolerance. The result dict is returned to the caller (the parent returned None) and printed as
    `HOME_MOTION {...}`; stats are kept separately from the R interface stats so R2/G1 `moves` counts stay comparable.
"""
from __future__ import annotations

import json
import time

import numpy as np

NO_CANDIDATES_MSG = "No grasp candidates found"
HOME_MAX_STEPS = 120     # production move_to_joints_blocking default cap (cap/envs/simulators/libero.py)
HOME_CONV_TOL = 0.01     # production convergence tolerance (same as motion_interface_r.CONV_TOL)


def _write(log_path, rec):
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), **rec}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def is_no_candidates_error(e) -> bool:
    """True only for the exact known empty-candidate assertion of the production grasp planner."""
    return isinstance(e, AssertionError) and str(e) == NO_CANDIDATES_MSG


def bounded_plan_grasp(fn, log_path=None, printer=print):
    def wrapped(*args, **kwargs):
        t0 = time.time()
        try:
            res = fn(*args, **kwargs)
        except AssertionError as e:
            if not is_no_candidates_error(e):
                raise                                   # unrelated assertion: propagate unchanged (parent converted it)
            rec = {"call": "plan_grasp", "status": "failed", "detail": "no_candidates", "reason": str(e), "n_candidates": 0, "wall_s": round(time.time() - t0, 3)}
            printer("GRASP_CANDIDATES " + json.dumps(rec), flush=True); _write(log_path, rec)
            return None, None
        try:
            n = 0 if res is None or res[0] is None else len(res[0])
        except Exception:  # noqa: BLE001
            n = None
        rec = {"call": "plan_grasp", "status": "ok" if n else "failed", "detail": None if n else "no_candidates", "n_candidates": n, "wall_s": round(time.time() - t0, 3)}
        printer("GRASP_CANDIDATES " + json.dumps(rec), flush=True); _write(log_path, rec)
        return res
    wrapped.__name__ = getattr(fn, "__name__", "plan_grasp"); wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


def reported_home(fn, mi, log_path=None, printer=print, stats=None):
    """See module docstring. `mi` provides joints(), model.limit_violation(), model.fk_tcp(), low._sim_step_count, low.home_joint_position."""
    stats = stats if stats is not None else {"calls": 0, "reached": 0, "not_reached": 0, "refused": 0, "sim_steps": 0, "status_counts": {}}

    def finish(res):
        stats["calls"] += 1; stats["sim_steps"] += int(res.get("steps_used", 0))
        stats["status_counts"][res["status"]] = stats["status_counts"].get(res["status"], 0) + 1
        if res["status"] == "failed":
            stats["refused"] += 1
        stats["reached" if res["reached"] else "not_reached"] += 1
        printer("HOME_MOTION " + json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), flush=True); _write(log_path, {"call": "goto_home", **res})
        return res

    def wrapped():
        t0 = time.time(); s0 = int(getattr(mi.low, "_sim_step_count", 0))
        home = getattr(mi.low, "home_joint_position", None)
        res = {"call": "goto_home", "label": "goto_home", "status": None, "detail": None, "reached": False, "steps_used": 0, "budget": HOME_MAX_STEPS, "tol_joint": HOME_CONV_TOL, "executed": False}
        if home is None:
            res.update({"status": "failed", "detail": "home_unavailable", "reason": "environment exposes no home configuration"}); return finish(res)
        home = np.asarray(home, float).reshape(7); res["target_joints"] = home
        viol = float(mi.model.limit_violation(home))
        if viol > 1e-6:   # prerequisite: the target must be inside the URDF limits (server margin); refuse instead of moving
            res.update({"status": "failed", "detail": "joint_limits", "reason": f"home target violates the URDF limits by {viol:.3f} rad"}); return finish(res)
        j0 = mi.joints()
        if j0 is None:
            res.update({"status": "uncertain", "detail": "joints_unobservable", "reason": "public joint observation missing before motion"}); return finish(res)
        res["start_joints"] = j0; res["joint_err_start"] = float(np.linalg.norm(j0 - home))
        try:
            fn()                                        # unchanged production command
        except Exception as e:  # noqa: BLE001
            res.update({"status": "uncertain", "detail": "exception", "reason": repr(e), "executed": True, "steps_used": int(getattr(mi.low, "_sim_step_count", s0)) - s0}); finish(res); raise
        res["executed"] = True; res["steps_used"] = int(getattr(mi.low, "_sim_step_count", s0)) - s0
        j1 = mi.joints()
        if j1 is None:
            res.update({"status": "uncertain", "detail": "joints_unobservable", "reason": "public joint observation missing after motion"}); return finish(res)
        err = float(np.linalg.norm(j1 - home)); p1, q1 = mi.model.fk_tcp(j1)
        res.update({"final_joints": j1, "final_tcp": p1, "final_quat": q1, "joint_err": err, "wall_s": round(time.time() - t0, 3)})
        if err < HOME_CONV_TOL:
            res.update({"status": "reached", "detail": "within_tolerance", "reached": True})
        elif res["steps_used"] >= HOME_MAX_STEPS:
            res.update({"status": "budget_exhausted", "detail": "budget", "reason": f"{HOME_MAX_STEPS} control steps spent; joint error {err:.4f} rad >= {HOME_CONV_TOL}"})
        else:
            res.update({"status": "stalled", "detail": "no_progress", "reason": f"production loop ended after {res['steps_used']} steps with joint error {err:.4f} rad"})
        return finish(res)
    wrapped.__name__ = "goto_home_joint_position"; wrapped.stats = stats
    return wrapped
