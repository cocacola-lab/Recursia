"""Shared interface failure-propagation guards (round zeroshot_minimal_fusion_20260918_t7saYg).

Injected by the runner for arms E1 and E2 only; E0 runs the parent's injection byte-for-byte. No object / task / scene / seed
information enters any decision.

bounded_plan_grasp(fn, log_path)
    Wraps the production GraspNet `plan_grasp(depth, K, mask)` whose empty-candidate path raises
    `AssertionError("No grasp candidates found")` (libero_reduced.py:318). The parent round's MF pudding s54 attempt crashed
    on that assertion (sandbox_rc 1). The wrapper converts ONLY that failure class (AssertionError raised by the call) into
    the bounded value `(None, None)` -- the same shape the caller already tests (`grasp_poses is None or len(...) == 0`) --
    and logs a `GRASP_CANDIDATES {"status": "failed", "detail": "no_candidates"}` record. Every other exception propagates
    unchanged (narrow scope: no blanket exception conversion).
"""
from __future__ import annotations

import json
import time


def bounded_plan_grasp(fn, log_path=None, printer=print):
    def wrapped(*args, **kwargs):
        t0 = time.time()
        try:
            res = fn(*args, **kwargs)
        except AssertionError as e:   # the library's empty-candidate signal; nothing else is converted
            rec = {"call": "plan_grasp", "status": "failed", "detail": "no_candidates", "reason": str(e) or "AssertionError", "n_candidates": 0, "wall_s": round(time.time() - t0, 3)}
            printer("GRASP_CANDIDATES " + json.dumps(rec), flush=True)
            if log_path is not None:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"t": round(time.time(), 3), **rec}) + "\n")
            return None, None
        try:
            n = 0 if res is None or res[0] is None else len(res[0])
        except Exception:  # noqa: BLE001
            n = None
        rec = {"call": "plan_grasp", "status": "ok" if n else "failed", "detail": None if n else "no_candidates", "n_candidates": n, "wall_s": round(time.time() - t0, 3)}
        printer("GRASP_CANDIDATES " + json.dumps(rec), flush=True)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), **rec}) + "\n")
        return res
    wrapped.__name__ = getattr(fn, "__name__", "plan_grasp"); wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped
