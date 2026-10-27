"""Release gate for the shared placement module (round placement_geometry_20260917_122900).

A release is allowed only when BOTH hold:
  (1) the placement descent result is a success under the frozen MC gate (reached | contact_established) — decided by the
      unchanged `contact_gate.DependentActionGate` and passed in as `descent_ok`;
  (2) the independent placement verdict is `success`, and (3) there is touchdown evidence (descent `contact_established`) or
      no bounded corrective descent remains (the verifier cannot resolve a hover below its 0-2 cm base-bias band),
OR the pre-registered conservative policy for `uncertain` applies (all conditions, logged):
      verdict `uncertain` with `evidential_only` (no measured check failed or fell in an uncertain band) AND the descent
      ended in `contact_established` (physical touchdown evidence) AND the caller's bounded corrective retries are exhausted.
`failed` verdicts never release. Every decision is logged; `release_after_failed` must be 0 and every release on
`uncertain` is counted with its policy string. No object/scene/task parameter.
"""
from __future__ import annotations

import json
import time

POLICY_UNCERTAIN = "release_on_uncertain: evidential_only and descent_status==contact_established and retries_left==0"


class PlacementGate:
    def __init__(self, log_path=None):
        self.decisions = []; self.log_f = open(log_path, "a") if log_path else None
        self.counts = {"allow_success": 0, "release_on_uncertain": 0, "refused_failed": 0, "refused_uncertain": 0, "refused_descent": 0, "refused_no_verdict": 0, "refused_no_touchdown": 0}
        self.violations = []

    def allow_release(self, descent_ok, descent_status, verdict, retries_left, sim_step=None):
        policy = None
        # Amendment (continuation 2026-09-17 15:00): the gate consumes the PRE-RELEASE readiness claim (`ready | not_ready |
        # uncertain`, claim "ready_to_release"). A post-release placement status ("success") is not a valid input here.
        if not descent_ok:
            ok, why = False, f"refused: placement descent not a success ({descent_status})"; self.counts["refused_descent"] += 1
        elif not isinstance(verdict, dict) or verdict.get("status") not in ("ready", "not_ready", "uncertain") or verdict.get("claim", "ready_to_release") != "ready_to_release":
            ok, why = False, f"refused: no pre-release readiness verdict (got {None if not isinstance(verdict, dict) else verdict.get('status')})"; self.counts["refused_no_verdict"] += 1
        elif verdict["status"] == "ready" and descent_status != "contact_established" and int(retries_left) > 0:
            # the verifier cannot resolve a hover below its base-bias band (0-2 cm); without touchdown evidence a bounded corrective descent is required first
            ok, why = False, "refused: verdict ready but no touchdown evidence (descent %s) and a corrective descent is still available" % descent_status; self.counts["refused_no_touchdown"] += 1
        elif verdict["status"] == "ready":
            ok, why = True, "release readiness verified" + ("" if descent_status == "contact_established" else " (no touchdown evidence; corrective retries exhausted)"); self.counts["allow_success"] += 1
        elif verdict["status"] == "not_ready":
            ok, why = False, f"refused: not ready ({verdict.get('reason')})"; self.counts["refused_failed"] += 1
        else:
            cond = bool(verdict.get("evidential_only")) and descent_status == "contact_established" and int(retries_left) == 0
            if cond:
                ok, why, policy = True, f"release on uncertain under policy ({verdict.get('reason')})", POLICY_UNCERTAIN; self.counts["release_on_uncertain"] += 1
            else:
                ok, why = False, f"refused: placement uncertain ({verdict.get('reason')}); evidential_only={verdict.get('evidential_only')} descent={descent_status} retries_left={retries_left}"; self.counts["refused_uncertain"] += 1
        d = {"t": round(time.time(), 3), "sim_step": sim_step, "action": "release", "allow": ok, "why": why, "policy": policy, "verdict": (verdict or {}).get("status") if isinstance(verdict, dict) else None, "descent_status": descent_status, "retries_left": retries_left}
        self.decisions.append(d)
        if self.log_f:
            self.log_f.write(json.dumps(d) + "\n"); self.log_f.flush()
        return ok, why, policy

    def record_release(self, sim_step=None):
        prior = [d for d in self.decisions if d.get("action") == "release" and "allow" in d]
        if not prior or not prior[-1]["allow"]:
            self.violations.append({"sim_step": sim_step, "reason": "release executed without a preceding allow"})
        d = {"t": round(time.time(), 3), "sim_step": sim_step, "action": "release", "executed": True}
        self.decisions.append(d)
        if self.log_f:
            self.log_f.write(json.dumps(d) + "\n"); self.log_f.flush()

    def summary(self):
        return {**self.counts, "violations": len(self.violations), "decisions": len([d for d in self.decisions if "allow" in d])}

    def close(self):
        if self.log_f:
            self.log_f.close()
