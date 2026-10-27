"""Dependent-action gate (round contact_primitive_20260917_112613).

A dependent action (close after a contact/grasp descent, lift/transport after a grasp, release after a placement descent)
is allowed ONLY when the gating result is a success: `result["success"] is True` and `result["status"]` in the success set
(contact_move: reached | contact_established; verifiers: verified). Every decision is logged; refusals are counted. No
object/scene/task parameter.
"""
from __future__ import annotations

import json
import time

SUCCESS_STATUSES = ("reached", "contact_established", "verified")
DEPENDENT_ACTIONS = ("close", "lift", "transport", "release")


class DependentActionGate:
    def __init__(self, log_path=None):
        self.decisions = []; self.refusals = 0; self.violations = []
        self.log_f = open(log_path, "a") if log_path else None

    def allow(self, result, action: str, sim_step=None):
        if action not in DEPENDENT_ACTIONS:
            raise ValueError(f"unknown dependent action {action!r}")
        if result is None or not isinstance(result, dict):
            ok, why = False, "no_result"
        else:
            st = result.get("status"); succ = result.get("success")
            if succ is None:
                succ = st == "verified"   # verifier dicts carry no `success` flag; only `verified` counts
            ok = bool(succ is True and st in SUCCESS_STATUSES)
            why = f"{st}" if ok else f"refused: status={st} success={succ} detail={result.get('detail')}"
        d = {"t": round(time.time(), 3), "sim_step": sim_step, "action": action, "allow": ok, "why": why, "status": (result or {}).get("status") if isinstance(result, dict) else None}
        self.decisions.append(d)
        if not ok:
            self.refusals += 1
        if self.log_f:
            self.log_f.write(json.dumps(d) + "\n"); self.log_f.flush()
        return ok, why

    def gated(self, result, action, fn, sim_step=None):
        """Run fn() only if allowed. Returns (executed: bool, why, fn_return)."""
        ok, why = self.allow(result, action, sim_step)
        if not ok:
            return False, why, None
        return True, why, fn()

    def record_action(self, action, sim_step=None, role="dependent"):
        """Audit hook: the caller declares that a gripper action was executed. A dependent action without a preceding allow
        on the same action is a violation (kept for the offline check)."""
        if role == "dependent":
            prior = [d for d in self.decisions if d["action"] == action]
            if not prior or not prior[-1]["allow"]:
                self.violations.append({"action": action, "sim_step": sim_step, "reason": "dependent action executed without a preceding allow"})
        self.decisions.append({"t": round(time.time(), 3), "sim_step": sim_step, "action": action, "executed": True, "role": role})
        if self.log_f:
            self.log_f.write(json.dumps(self.decisions[-1]) + "\n"); self.log_f.flush()

    def summary(self):
        return {"decisions": len([d for d in self.decisions if "allow" in d]), "refusals": self.refusals, "violations": len(self.violations),
                "allowed_by_action": {a: sum(1 for d in self.decisions if d.get("action") == a and d.get("allow") is True) for a in DEPENDENT_ACTIONS}}

    def close(self):
        if self.log_f:
            self.log_f.close()
