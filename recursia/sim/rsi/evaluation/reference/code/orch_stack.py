"""N20O (interface v2): LLM orchestration INFERENCE METHOD over the frozen N20 RSI stack (explicit interface layer + interpreted
program boundary + one-shot generation). v2 refines v1 (N12O) with lessons from the N11-client reference implementation: generated code
is interpreted (orch_check.Interpreter), never exec'd; every contract-compatible catalog entry is retrieved; frozen-executor exceptions
are recorded and re-raised (baseline semantics); results have a fixed schema (refusals included); the episode summary is written even
when the 4000-step horizon interrupts orchestration.

Proposed extension (2026-10-07; v2 2026-10-08), not a description of earlier behaviour. The frozen stack is the N20 policy program
(programs/N18.py, sha256 STACK_SHA256). Its definitions are executed unchanged in the episode namespace; this module never
re-implements a skill, a verifier, a gate or a threshold. It only
  * wraps the complete subgoal executors (run_subgoal, run_close_articulated, run_open_articulated, run_actuate, run_rotary)
    with the same pre-call state handling as the baseline driver `run()` (VB reference activation, home between subgoals,
    others_xy, la_assembly_prompts in plan order);
  * reuses la_dispatch as the capability check (statically by orch_check, and again at runtime before every call);
  * reuses vb_resolve_references for task-start identity references (frozen before any object moves, as in run());
  * turns the executors' own tagged log lines into structured results, keeping "control completed", "enforced gate passed",
    "shadow verdict recorded" and "unknown" apart;
  * exposes only recovery actions the stack already has (re-observation, the production home move, one re-invocation of the
    same frozen executor when the evidence shows nothing is held).
The baseline (`run()` with la_parse) remains selectable as arm N20; this mode never falls back to it.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import orch_check as OC

STACK_ID = "N20"
STACK_SHA256 = "29b187dfaffb4e1bc3a9e8d866637c552453a36fbc2f00691c78dc065f2d7cee"
INTERFACE_VERSION = "rsi_n20_iface_v2_20261016"
PROMPT_VERSION = "orch_prompt_v2_20261008"
RESULT_TEMPLATE = {"status": None, "id": None, "failure_reason": None, "retry": False, "object_reference": None, "object_possibly_retained": None,
                   "placement_xy": None, "executor": None, "executor_return": None, "verification_status": "not_verified", "enforced_gates": [],
                   "stack_verifier_claims": [], "shadow_verdicts": [], "allowed_recovery_actions": [], "execution_evidence": {}}
GEN_MAX_TOKENS = 6000
LIMITS = {"stack_calls": 120, "executes": 8, "retries_per_subgoal": 1, "retries_total": 3, "reobserve": 20, "go_home": 8,
          "locate": 10, "log": 60, "program_ticks": 5000}
RECOVERY_ACTIONS = ("reobserve", "go_home", "retry_subgoal")


def executor_name(ns, relation):
    """run()'s exact dispatch, evaluated on the frozen tables (RT_RELATIONS first: in N12/N13/N18/N20 actuate_on AND actuate_off -> run_rotary)."""
    if relation not in ns["LA_PART_RELATIONS"]:
        return "run_subgoal"
    return ("run_rotary" if relation in ns["RT_RELATIONS"] else "run_actuate" if relation in ns["LA_ACTUATE_RELATIONS"]
            else "run_open_articulated" if relation in ns["LA_OPEN_RELATIONS"] else "run_close_articulated")

# Reasons whose evidence leaves the object possibly in the hand (the stack's own retreat-with-object paths).
RETAINED_PLACE_REASONS = ("release_blocked", "lift_", "transport_", "top_face_")
ENFORCED_VERIFIER_ABORTS = ("grasp_not_verified", "object_lost_after_lift", "lift_refused")
TAG_RE = re.compile(r"^([A-Z][A-Z0-9_]{2,})(?:\s+(.*))?$")


class _Tee(io.TextIOBase):
    def __init__(self, inner):
        self.inner, self.lines, self._buf = inner, [], ""

    def write(self, s):
        self.inner.write(s); self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1); self.lines.append(line)
        return len(s)

    def flush(self):
        self.inner.flush()

    def finish(self):
        if self._buf:
            self.lines.append(self._buf); self._buf = ""
        return self.lines


def parse_evidence(lines):
    """Tagged stack log lines -> [(tag, data)]. JSON payloads are parsed; `reason=...` payloads become {"reason": ...}."""
    out = []
    for line in lines:
        s = line.strip()
        if s == "All grasp attempts failed!":
            out.append(("ALL_GRASP_FAILED", {})); continue
        if s == "Done!":
            out.append(("EXECUTOR_DONE", {})); continue
        m = TAG_RE.match(s)
        if not m:
            continue
        tag, rest = m.group(1), (m.group(2) or "").strip()
        try:
            data = json.loads(rest) if rest else {}
            if not isinstance(data, dict):
                data = {"value": data}
        except json.JSONDecodeError:
            data = {"reason": rest[7:]} if rest.startswith("reason=") else {"text": rest[:300]}
        out.append((tag, data))
    return out


def _last(ev, tag):
    for t, d in reversed(ev):
        if t == tag:
            return d
    return None


def _all(ev, tag):
    return [d for t, d in ev if t == tag]


def classify_object_result(ev, xy):
    """run_subgoal evidence -> (status, failure_reason, retained). run_subgoal returns the placement xy after prim_place
    whatever prim_place's status, so the status comes from PLACE_RESULT, never from the return value alone."""
    stop = _last(ev, "BINDING_STOP"); place = _last(ev, "PLACE_RESULT"); gate = _last(ev, "VB_TRANSPORT_GATE")
    aborts = [d.get("reason", "") for d in _all(ev, "GRASP_ABORT")]
    if place is not None:
        if place.get("status") == "released":
            return "control_completed", None, False
        reason = str(place.get("reason"))
        kind = "verification_failed" if reason.startswith("release_blocked") else "execution_failed"
        return kind, reason, any(reason.startswith(p) for p in RETAINED_PLACE_REASONS)
    if xy is None and stop is not None:
        return "binding_failed", str(stop.get("status") or stop.get("reason")), False
    if xy is None and (_last(ev, "ALL_GRASP_FAILED") is not None or aborts):
        last = aborts[-1] if aborts else None
        retained = any(a.startswith("strategy_loop_stopped_after_lift_nonarrival") for a in aborts)
        if gate is not None and not gate.get("allow"):
            return "verification_failed", "transport_gate_refused", retained
        if last is not None and any(last.startswith(p) for p in ENFORCED_VERIFIER_ABORTS):
            return "verification_failed", last, retained
        return "execution_failed", last or "all_grasp_attempts_failed", retained
    return "unknown", "no_terminal_evidence", None


def classify_part_result(ev, ret):
    """Part executor evidence -> (status, failure_reason, retained). True = the executor's own verifier claimed the state."""
    stop = _last(ev, "BINDING_STOP")
    claims = [d for t, d in ev if t in ("CLOSE_VERIFY", "OPEN_VERIFY", "ACT_VERIFY")]
    stops = [d for t, d in ev if t in ("CLOSE_STOP", "OPEN_STOP", "ACT_STOP")]
    aborts = [d.get("reason", "") for t, d in ev if t in ("ACT_ABORT", "OPEN_ABORT")]
    if ret is True:
        return "control_completed", None, False
    if stop is not None and not claims and not stops:
        return "binding_failed", str(stop.get("status")), False
    if aborts:
        a = aborts[-1]
        return ("verification_failed" if a.startswith("grasp_not_verified") else "execution_failed"), a, False   # the abort paths open + home
    if stops:
        why = str(stops[-1].get("why"))
        if why.startswith("motion_"):
            return "execution_failed", why, None   # stopped mid-motion: hand state not established
        if claims and str(claims[-1].get("claim", "")).startswith("not_"):
            return "verification_failed", why, False
        return "execution_failed", why, False
    if claims and str(claims[-1].get("claim", "")).startswith("not_"):
        return "verification_failed", "stack_verifier_claim_" + str(claims[-1].get("claim")), False
    return "unknown", "no_terminal_evidence", None


def verification_record(ev, relation, part):
    """Verification evidence by semantics: enforced gates (they controlled execution), the executor's own verifier claim
    (used for control inside the executor), and shadow verdicts (logged only, never gate). None of these is task success."""
    gates = []
    for d in _all(ev, "VB_TRANSPORT_GATE"):
        gates.append({"gate": "vb_transport", "allow": bool(d.get("allow")), "identity": d.get("identity"), "retained": d.get("retained"), "moved_with_gripper": d.get("moved_with_gripper")})
    for d in _all(ev, "E1_GUARD"):
        gates.append({"gate": "e1_" + str(d.get("action")), "allow": bool(d.get("allow")), "why": d.get("why")})
    for d in _all(ev, "GRASP_WIDTH_GATE"):
        gates.append({"gate": "grasp_width", "allow": d.get("allow", d.get("ok")), "detail": {k: d.get(k) for k in ("status", "reason") if k in d}})
    for d in _all(ev, "VB_PREGRASP"):
        gates.append({"gate": "vb_pregrasp_identity", "allow": d.get("result") == "identity_verified", "result": d.get("result")})
    place = _last(ev, "PLACE_RESULT")
    if place is not None:
        gates.append({"gate": "release", "allow": place.get("status") == "released" or not str(place.get("reason", "")).startswith("release_blocked"),
                      "detail": place.get("reason")})
    shadow = []
    for t in ("MF_SHADOW_POST", "MF_SHADOW_PRE", "V1_SHADOW", "GRASP_WIDTH_SHADOW"):
        for d in _all(ev, t):
            shadow.append({"source": t, "status": d.get("status"), "claim": d.get("claim")})
    claims = [{"source": t, "claim": d.get("claim"), "why": d.get("why")} for t, d in ev if t in ("CLOSE_VERIFY", "OPEN_VERIFY", "ACT_VERIFY")]
    if part:
        status = ("stack_verifier_claim_positive" if claims and not str(claims[-1]["claim"]).startswith("not_")
                  else "stack_verifier_claim_negative" if claims else "not_verified")
    else:
        status = "shadow_recorded_only" if (place or {}).get("status") == "released" else ("gate_refused" if any(not g["allow"] for g in gates) else "not_verified")
    return {"verification_status": status, "enforced_gates": gates, "stack_verifier_claims": claims, "shadow_verdicts": shadow,
            "note": "Official task success is measured only by the benchmark evaluator; none of these fields is a success measurement."}


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:  # noqa: BLE001
        pass
    return str(o)


class RSIStack:
    """The only object a generated program can reach. Public methods = orch_check.STACK_METHODS; arguments and returns are plain
    JSON-like values. Internal state lives in a closure-free private dict; the checker rejects every underscore attribute."""

    def __init__(self, ns, td, lang, log_dir):
        self._ns, self._td, self._lang, self._dir = ns, td, lang, Path(log_dir)
        self._n = {k: 0 for k in LIMITS}; self._sg = {}; self._plan = None; self._placed = []; self._results = {}
        self._retries = {}; self._executed_any = False; self._obs_n = 0; self._finish = None; self._calls = []
        self._first_obs = None; self._stack_exception = None

    # ---- bookkeeping ----
    def _count(self, kind):
        self._n["stack_calls"] += 1
        if self._n["stack_calls"] > LIMITS["stack_calls"]:
            return False
        if kind in self._n:
            self._n[kind] += 1
            return self._n[kind] <= LIMITS[kind]
        return True

    def _record(self, method, args, result):
        rec = {"i": len(self._calls), "t": round(time.time(), 2), "sim_step": self._sim_step(), "method": method, "args": args, "result": result}
        self._calls.append(rec)
        with open(self._dir / "stack_calls.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=_jsonable) + "\n")
        print("ORCH_CALL " + json.dumps({"i": rec["i"], "method": method, "args": args, "status": (result or {}).get("status") if isinstance(result, dict) else None}, default=_jsonable), flush=True)
        return result

    def _sim_step(self):
        try:
            return int(self._ns["env"].low_level_env._sim_step_count)
        except Exception:  # noqa: BLE001
            return None

    def _refuse(self, method, args, why):
        """Every refusal carries the full result schema, so a program may index any result key without an error."""
        return self._record(method, args, dict(copy.deepcopy(RESULT_TEMPLATE), status="refused", failure_reason=why, id=(args or {}).get("id"),
                                               supported=False, why=why, references=[], unsupported=[], found=False, centre=None))

    def _codes(self):
        return dict(self._td.get("side_words") or {})

    # ---- vocabulary / capabilities ----
    def vocabulary(self):
        if not self._count("other"):
            return self._refuse("vocabulary", {}, "budget_stack_calls")
        v = {"objects": sorted((self._td.get("objects") or {})), "supports": sorted((self._td.get("supports") or {})),
             "articulated": sorted((self._td.get("articulated") or {})), "side_words": self._codes()}
        return self._record("vocabulary", {}, {"status": "ok", **v})

    def capabilities(self):
        if not self._count("other"):
            return self._refuse("capabilities", {}, "budget_stack_calls")
        return self._record("capabilities", {}, {"status": "ok", **capability_table(self._ns, self._td), "limits": dict(LIMITS)})

    # ---- observation ----
    def _robot(self, obs):
        ns = self._ns; p, q = ns["tcp_from_obs"](obs)
        try:
            g = float(ns["gripper_opening"]())
        except Exception:  # noqa: BLE001
            g = None
        return {"tcp_position": [round(float(x), 4) for x in p], "tcp_quat_wxyz": [round(float(x), 4) for x in q], "gripper_opening": g}

    def observe(self):
        if not self._count("reobserve"):
            return self._refuse("observe", {}, "budget_reobserve")
        obs = self._ns["get_observation"](); self._obs_n += 1
        rgb = obs["agentview"]["images"]["rgb"]
        return self._record("observe", {}, {"status": "ok", "obs_id": self._obs_n, "image_shape": list(rgb.shape[:2]), **self._robot(obs)})

    def robot_state(self):
        if not self._count("reobserve"):
            return self._refuse("robot_state", {}, "budget_reobserve")
        obs = self._ns["get_observation"](); self._obs_n += 1
        return self._record("robot_state", {}, {"status": "ok", "obs_id": self._obs_n, **self._robot(obs)})

    # ---- subgoals / capability check ----
    def subgoal(self, relation, object=None, reference=None, part=None, side=None, object_side=None, instance=None):   # noqa: A002
        args = {"relation": relation, "object": object, "reference": reference, "part": part, "side": side, "object_side": object_side, "instance": instance}
        if not self._count("other"):
            return self._refuse("subgoal", args, "budget_stack_calls")
        if self._plan is not None:
            return self._refuse("subgoal", args, "plan_already_frozen")
        rec, why = build_subgoal_record(self._td, self._lang, **args, part_relations=self._ns["LA_PART_RELATIONS"])
        self._n_sg = getattr(self, "_n_sg", 0) + 1; sid = f"sg{self._n_sg}"   # unique even when a subgoal is refused (not stored)
        if rec is None:
            out = {"status": "unsupported", "id": sid, "supported": False, "why": why, **args, "record": None}
        else:
            ok, w = self._ns["la_dispatch"](rec)
            out = {"status": "ok" if ok else "unsupported", "id": sid, "supported": bool(ok), "why": w, **args,
                   "record": {k: rec.get(k) for k in ("relation", "object", "object_side", "support", "part", "side", "sides")}}
            self._sg[sid] = rec
        return self._record("subgoal", args, out)

    def check(self, sg):
        sid = _sid(sg)
        if not self._count("other"):
            return self._refuse("check", {"id": sid}, "budget_stack_calls")
        rec = self._sg.get(sid)
        if rec is None:
            return self._record("check", {"id": sid}, {"status": "unsupported", "supported": False, "why": "unknown_subgoal"})
        ok, w = self._ns["la_dispatch"](rec)
        return self._record("check", {"id": sid}, {"status": "ok" if ok else "unsupported", "supported": bool(ok), "why": w,
                                                     "implemented": {r: list(v) for r, v in self._ns["LA_SUPPORTED"].items()}})

    def plan(self, subgoals):
        ids = [_sid(s) for s in list(subgoals)[:LIMITS["executes"]]]
        if not self._count("other"):
            return self._refuse("plan", {"ids": ids}, "budget_stack_calls")
        if self._plan is not None:
            return self._refuse("plan", {"ids": ids}, "plan_already_frozen")
        if not ids or len(set(ids)) != len(ids) or len(list(subgoals)) > LIMITS["executes"]:
            return self._refuse("plan", {"ids": ids}, "plan_must_list_1_to_%d_distinct_subgoals" % LIMITS["executes"])
        unsupported = []
        for sid in ids:
            rec = self._sg.get(sid)
            ok, w = (False, "unknown_subgoal") if rec is None else self._ns["la_dispatch"](rec)
            if not ok:
                unsupported.append({"id": sid, "why": w})
        if unsupported:   # capability check before any motion: refuse the whole plan, nothing is executed
            return self._record("plan", {"ids": ids}, {"status": "unsupported", "unsupported": unsupported, "references": [],
                                                       "implemented": {r: list(v) for r, v in self._ns["LA_SUPPORTED"].items()}})
        self._plan = ids
        recs = [self._sg[s] for s in ids]
        for i, r in enumerate(recs):   # N20 run() (C15I4): a part action on an assembly an EARLIER subgoal already used refers to it (as la_parse's "close it")
            if r["relation"] in self._ns["LA_PART_RELATIONS"] and not r.get("refers"):
                mine = {r.get("part"), r.get("support")} - {None}
                if any(mine & ({q.get("part"), q.get("support")} - {None}) for q in recs[:i]):
                    r["refers"] = "part"
        print("ORCH_PLAN " + json.dumps({"language": self._lang, "subgoals": [dict(id=s, **self._ns["la_summary"](r)) for s, r in zip(ids, recs)]}), flush=True)
        with self._capture() as tee:
            self._ns["vb_resolve_references"](recs)   # VB1 task-start references, before any object moves (as run())
        ev = parse_evidence(tee.lines)
        refs = [self.__ref(s) for s in ids]
        return self._record("plan", {"ids": ids}, {"status": "ok", "references": refs, "unsupported": [], "evidence": ev})

    def __ref(self, sid):
        k = self._plan.index(sid) + 1 if self._plan and sid in self._plan else None
        R = (self._ns["_VB"].get("refs") or {}).get(k) if k is not None else None
        return {"id": sid, "k": k, "handle": None if R is None else R.get("handle"), "status": "none" if R is None else R.get("status"),
                "phrase": None if R is None else R.get("phrase")}

    def reference(self, sg):
        sid = _sid(sg)
        if not self._count("other"):
            return self._refuse("reference", {"id": sid}, "budget_stack_calls")
        return self._record("reference", {"id": sid}, {"status": "ok", **self.__ref(sid)})

    def locate(self, phrase, object_side=None):
        """Text binding (localize_target) on a fresh observation; perception only, never an identity confirmation. The binding
        module's per-attempt state (_BIND reference / last, _VB phase) is restored afterwards so the executors see what they would
        have seen without this call."""
        args = {"phrase": phrase, "object_side": object_side}
        if not self._count("locate"):
            return self._refuse("locate", args, "budget_locate")
        prompts = (self._td.get("objects") or {}).get(phrase) or (self._td.get("supports") or {}).get(phrase) or (self._td.get("articulated") or {}).get(phrase)
        if prompts is None:
            return self._record("locate", args, {"status": "unsupported", "why": "vocabulary", "found": False, "centre": None})
        code = self._codes().get(object_side, object_side) if object_side is not None else None
        if code is not None and code not in self._ns["LA_OBJECT_SIDES"]:
            return self._record("locate", args, {"status": "unsupported", "why": "object_side", "found": False, "centre": None})
        ns = self._ns; saved = {k: ns["_BIND"].get(k) for k in ("reference", "last")}; ph = ns["_VB"].get("phase")
        ns["_VB"]["phase"] = None
        try:
            obs, rgb, depth_img, K, E = ns["vb_observe"](); self._obs_n += 1
            with self._capture():
                c, pts, mask = ns["localize_target"](rgb, depth_img, K, E, list(prompts), object_side=code)
            status = ns["_BIND"].get("last")
        finally:
            ns["_BIND"].update(saved); ns["_VB"]["phase"] = ph
        return self._record("locate", args, {"status": "ok", "found": c is not None, "binding_status": status, "obs_id": self._obs_n,
                                             "centre": None if c is None else [round(float(v), 4) for v in c], "semantics": "text_binding_only"})

    # ---- execution ----
    @contextlib.contextmanager
    def _capture(self):
        tee = _Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            yield tee
        tee.finish()

    def execute(self, sg):
        sid = _sid(sg); args = {"id": sid}
        if self._plan is None:
            return self._refuse("execute", args, "no_plan: call stack.plan([...]) first (identity references are frozen there)")
        if sid not in self._plan:
            return self._refuse("execute", args, "subgoal_not_in_plan")
        prev = self._results.get(sid)
        if prev is not None and prev["status"] == "control_completed":
            return self._refuse("execute", args, "already_control_completed")
        if prev is not None:   # a second execution is a retry: only through recover(..., "retry_subgoal")
            return self._refuse("execute", args, "already_executed: use stack.recover(result, 'retry_subgoal')")
        if not self._count("executes"):
            return self._refuse("execute", args, "budget_executes")
        return self.__execute(sid, retry=False)

    def __execute(self, sid, retry):
        ns = self._ns; rec = self._sg[sid]; k = self._plan.index(sid) + 1
        ok, why = ns["la_dispatch"](rec)   # runtime capability re-check before the call
        if not ok:
            r = dict(copy.deepcopy(RESULT_TEMPLATE), status="unsupported", id=sid, failure_reason=why)
            self._results[sid] = r
            return self._record("execute", {"id": sid, "retry": retry}, r)
        t0 = time.time(); s0 = self._sim_step()
        ns["_VB"]["active"] = k; ns["_VB"]["confirmed"] = None; ns["_VB"]["pregrasp_img"] = None   # = run()
        exc = None; part = rec["relation"] in ns["LA_PART_RELATIONS"]; ret = xy = None
        with self._capture() as tee:
            try:
                if self._executed_any:   # = run(): production home + 3 observations before every subgoal after the first
                    ns["goto_home_joint_position"]()
                    for _ in range(3):
                        ns["get_observation"]()
                self._executed_any = True
                if part and rec.get("refers"):   # = N20 run(): the referent's assembly vocabulary over the previous subgoals
                    rec["referent_prompts"] = ns["la_assembly_prompts"](rec, ns["task_descriptors"](), [self._sg[s] for s in self._plan[:k - 1]])
                if part:
                    fn = ns[executor_name(ns, rec["relation"])]
                    ret = fn(rec, ns["task_descriptors"](), sg_label=f"sg{k}"); xy = None
                else:
                    previous = [self._sg[s] for s in self._plan[:k - 1]]
                    ret = xy = ns["run_subgoal"](rec["object_prompts"], rec["side"], others_xy=tuple(self._placed), relation=rec["relation"],
                                                 support_prompts=ns["la_reference_prompts"](rec), object_side=rec.get("object_side"),
                                                 assembly_prompts=ns["la_assembly_prompts"](rec, ns["task_descriptors"](), previous),
                                                 support_track=None)   # N20 C17I2 carry tracking is only produced by la_parse's sc_carry_order; not reproduced (no Long-Pro instruction triggers it)
                print("SUBGOAL_STATUS " + json.dumps({"k": k, "object": rec["object"], "side": rec["side"], "relation": rec["relation"], "placed": xy is not None}), flush=True)
            except Exception as e:  # noqa: BLE001 - a frozen-executor exception: recorded below, then re-raised (baseline semantics)
                exc = e
        if exc is not None:
            import traceback
            r = dict(copy.deepcopy(RESULT_TEMPLATE), status="stack_exception", id=sid, retry=retry, failure_reason=repr(exc)[:300],
                     executor=executor_name(ns, rec["relation"]), execution_evidence={"tags": [t for t, _ in parse_evidence(tee.lines)][-60:],
                     "traceback": "".join(traceback.format_exception(exc))[-2000:]})
            self._results[sid] = r; self._stack_exception = repr(exc)[:300]
            with open(self._dir / f"evidence_{sid}{'_retry' if retry else ''}.json", "w") as fh:
                json.dump({"lines": tee.lines, "exception": r["execution_evidence"]["traceback"]}, fh, default=_jsonable)
            self._record("execute", {"id": sid, "retry": retry}, r)
            raise exc
        if xy is not None:
            self._placed.append(xy)   # = run(): every returned placement xy is an obstacle for later placements
        ev = parse_evidence(tee.lines)
        status, reason, retained = classify_part_result(ev, ret) if part else classify_object_result(ev, xy)
        ver = verification_record(ev, rec["relation"], part)
        allowed = ["reobserve", "go_home"]
        n_retry = self._retries.get(sid, 0)
        if status in ("binding_failed", "execution_failed", "verification_failed") and retained is False \
                and n_retry < LIMITS["retries_per_subgoal"] and sum(self._retries.values()) < LIMITS["retries_total"]:
            allowed.append("retry_subgoal")
        R = (ns["_VB"].get("refs") or {}).get(k)
        result = {"status": status, "id": sid, "failure_reason": reason, "retry": retry,
                  "object_reference": None if R is None else {"handle": R.get("handle"), "status": R.get("status"), "phrase": R.get("phrase")},
                  "object_possibly_retained": retained, "placement_xy": None if xy is None else [round(float(v), 4) for v in xy],
                  "executor": executor_name(ns, rec["relation"]),
                  "executor_return": ret if isinstance(ret, (bool, type(None))) else "xy",
                  **ver, "allowed_recovery_actions": allowed,
                  "execution_evidence": {"tags": [t for t, _ in ev][-60:], "terminal": [{"tag": t, **d} for t, d in ev if t in (
                      "BINDING_STOP", "PLACE_RESULT", "GRASP_ABORT", "ALL_GRASP_FAILED", "RELEASE_VERIFY", "CLOSE_STOP", "OPEN_STOP", "ACT_STOP",
                      "ACT_ABORT", "OPEN_ABORT", "SUBGOAL_STATUS")][-12:],
                      "sim_steps": None if s0 is None else (self._sim_step() or 0) - s0, "wall_s": round(time.time() - t0, 1)}}
        self._results[sid] = result
        with open(self._dir / f"evidence_{sid}{'_retry' if retry else ''}.json", "w") as fh:
            json.dump({"lines": tee.lines, "parsed": ev}, fh, default=_jsonable)
        return self._record("execute", {"id": sid, "retry": retry}, result)

    def recover(self, result, action):
        sid = result.get("id") if isinstance(result, dict) else _sid(result)
        args = {"id": sid, "action": action}
        if action not in RECOVERY_ACTIONS:
            return self._refuse("recover", args, "unsupported_recovery_action")
        last = self._results.get(sid)
        if action == "reobserve":
            return self.observe()
        if action == "go_home":
            if not self._count("go_home"):
                return self._refuse("recover", args, "budget_go_home")
            with self._capture():
                self._ns["goto_home_joint_position"]()
            return self._record("recover", args, {"status": "ok", "action": "go_home"})
        if last is None or "retry_subgoal" not in last.get("allowed_recovery_actions", ()):
            return self._refuse("recover", args, "retry_not_allowed_by_evidence")
        if not self._count("executes"):
            return self._refuse("recover", args, "budget_executes")
        self._retries[sid] = self._retries.get(sid, 0) + 1
        return self.__execute(sid, retry=True)

    # ---- misc ----
    def budget(self):
        return {"status": "ok", "used": dict(self._n), "limits": dict(LIMITS)}

    def log(self, message, data=None):
        if not self._count("log"):
            return {"status": "refused", "failure_reason": "budget_log"}
        print("ORCH_NOTE " + json.dumps({"message": str(message)[:300], "data": data}, default=str)[:2000], flush=True)
        return {"status": "ok"}

    def finish(self, status, reason=None):
        """The program's own declared stop. Recorded as a claim; task success is decided only by the benchmark evaluator."""
        self._finish = {"declared_status": str(status)[:60], "reason": None if reason is None else str(reason)[:300]}
        self._record("finish", self._finish, {"status": "ok"})
        return {"status": "ok"}


def _sid(sg):
    return sg.get("id") if isinstance(sg, dict) else str(sg)


PART_ARG_RELATIONS = ("in_sliding",)   # object relations whose records carry a part (la_parse: "in the <side> drawer of the <support>")


def build_subgoal_record(td, lang, relation, object=None, reference=None, part=None, side=None, object_side=None, instance=None, part_relations=()):   # noqa: A002
    """Interface arguments -> a subgoal record in la_subgoal's exact format (prompts from the frozen vocabulary only).
    Returns (record, None) or (None, why) with why in vocabulary | side | object_side | relation | reference_kind | part_not_allowed.
    Interface contract (stricter than la_dispatch, which only asks that SOME reference prompts exist): an object relation's
    reference is a support or an object phrase, the only record shapes the stack's parser produces and its executors were
    exercised on; an articulated part is never the container of an object relation (e.g. "in the microwave" -> relation in_hinged,
    unsupported), and a part argument is accepted only for in_sliding (and the part relations)."""
    td = td or {}; objs = td.get("objects") or {}; sups = td.get("supports") or {}; arts = td.get("articulated") or {}; sw = td.get("side_words") or {}
    if not isinstance(relation, str):
        return None, "relation"
    # clause: the executors read it only through la_phrase_in(td, "supports", clause) to add a support phrase's prompts; it is the
    # subgoal's own phrases (as la_parse's per-clause text would be), never the whole instruction (which names other subgoals' supports)
    clause = " ".join(str(p).lower() for p in (object, reference, part) if p is not None)
    sg = {"relation": relation, "clause": clause, "object": None, "object_prompts": None,
          "object_side": None, "support": None, "support_prompts": None, "part": None, "part_prompts": None, "side": None, "sides": [],
          "words": [], "instance": instance}
    if object is not None:
        if object not in objs:
            return None, "vocabulary"
        sg["object"] = object; sg["object_prompts"] = list(objs[object])
    is_part_rel = relation in tuple(part_relations)
    if not is_part_rel and reference is not None and reference not in sups and reference not in objs and reference in arts:
        return None, "reference_kind"
    if not is_part_rel and part is not None and relation not in PART_ARG_RELATIONS:
        return None, "part_not_allowed"
    if reference is not None:
        if reference in sups:
            sg["support"] = reference; sg["support_prompts"] = list(sups[reference])
        elif reference in objs and reference != object:
            sg["support"] = reference; sg["support_prompts"] = list(objs[reference])   # = la_subgoal: a second object phrase is the reference
        elif reference in arts:
            sg["part"] = reference; sg["part_prompts"] = list(arts[reference])
        else:
            return None, "vocabulary"
    if part is not None:
        if part not in arts:
            return None, "vocabulary"
        sg["part"] = part; sg["part_prompts"] = list(arts[part])
    if side is not None:
        code = sw.get(side, side)
        if relation == "in_layer" and side in {"under": "zmin", "on": "zmax"}:   # N20 LP_WORDS: the preposition names the layer
            code = {"under": "zmin", "on": "zmax"}[side]
        if code not in set(sw.values()):
            return None, "side"
        sg["sides"] = [str(code)]; sg["side"] = str(code)
    if object_side is not None:
        code = sw.get(object_side, object_side)
        if code not in set(sw.values()):
            return None, "object_side"
        sg["object_side"] = str(code)
    # = la_subgoal rule (a): a single side word inside a vocabulary phrase ("left moka pot") is that phrase's own side code; an object's
    # own code selects the object instance (object_side) unless one was given, a reference's / part's own code selects the reference
    for kind in ("object", "support", "part"):
        ph = sg[kind]
        if ph is None:
            continue
        codes = [str(sw[t]) for t in str(ph).split() if t in sw]
        own = codes[0] if len(codes) == 1 else None
        if own is None:
            continue
        if kind == "object":
            if sg["object_side"] is None:
                sg["object_side"] = own
        elif own not in sg["sides"]:
            sg["sides"].append(own)
    sg["side"] = sg["sides"][0] if sg["sides"] else None
    return sg, None


def capability_table(ns, td):
    sw = dict((td or {}).get("side_words") or {}); inv = {}
    for w, c in sw.items():
        inv.setdefault(c, []).append(w)
    rel = {}
    for r, codes in ns["LA_SUPPORTED"].items():
        rel[r] = {"side_words": sorted({w for c in codes if c is not None for w in inv.get(c, [])}), "no_side_ok": None in codes,
                  "executor": executor_name(ns, r), "needs": "reference (part or support)" if r in ns["LA_PART_RELATIONS"] else "object + reference"}
    return {"relations": rel, "object_side_words": sorted({w for c in ns["LA_OBJECT_SIDES"] for w in inv.get(c, [])}),
            "part_relations": list(ns["LA_PART_RELATIONS"]), "recovery_actions": list(RECOVERY_ACTIONS)}


# ---------------------------------------------------------------- execution boundary (part 2): interpreter, never exec
def execute_program(src, stack, context):
    """Interpret the (already checked) program: orch_check.Interpreter over the validated `run` node. Returns {"status": returned |
    runtime_error | budget_exceeded, ...}. A frozen-executor exception (stack._stack_exception) and the episode horizon propagate."""
    fn, v = OC.parse_and_validate(src)
    if fn is None or v.errors:
        return {"status": "rejected", "detail": v.errors[:5]}
    trace = []
    it = OC.Interpreter(fn, stack, trace=lambda kind, **kw: trace.append(dict(kind=kind, **kw)) if len(trace) < 500 else None)
    t0 = time.time()
    try:
        ret = it.run(context)
        return {"status": "returned", "return": ret if isinstance(ret, (str, int, float, bool, type(None))) else str(ret)[:200], "steps": it.steps, "calls": it.calls,
                "branches": trace, "wall_s": round(time.time() - t0, 1)}
    except OC.ProgramBudgetExceeded as e:
        return {"status": "budget_exceeded", "detail": str(e), "steps": it.steps, "calls": it.calls, "branches": trace, "wall_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        if stack._stack_exception is not None:
            raise   # frozen-stack exception: baseline semantics (the episode ends as the frozen driver's would)
        return {"status": "runtime_error", "detail": repr(e)[:500], "steps": it.steps, "calls": it.calls, "branches": trace, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- catalog retrieval + generation
def load_catalog(path):
    cat = json.loads(Path(path).read_text())
    assert cat["stack"]["sha256"] == STACK_SHA256 and cat["interface_version"] == INTERFACE_VERSION, "catalog / stack version mismatch"
    return cat


def retrieve(cat, td, lang):
    """Deterministic, version-pinned retrieval (v2, after the N11-client reference design): EVERY entry of this small catalog whose stack hash equals
    the loaded stack's and whose required vocabulary kinds are present in the instruction's frozen vocabulary. No lexical trigger
    filter, so wording can never hide a compatible capability."""
    kinds = {k for k in ("objects", "supports", "articulated") if (td or {}).get(k)}
    picked = []; skipped = []
    for e in cat["entries"]:
        if e.get("stack_sha256") != STACK_SHA256:
            skipped.append({"id": e["id"], "why": "other_stack_version"}); continue
        need = set(e.get("requires_vocabulary") or ())
        if not e.get("always") and not need <= kinds:
            skipped.append({"id": e["id"], "why": "contract_vocabulary_missing", "missing": sorted(need - kinds)}); continue
        picked.append(e)
    return picked, skipped


GEN_SCHEMA = {"type": "object", "properties": {"plan_notes": {"type": "string"}, "program": {"type": "string"}},
              "required": ["plan_notes", "program"], "additionalProperties": False}


def build_prompt(template, lang, vocab, caps, entries):
    def fmt(e):
        keys = ("id", "name", "signature", "purpose", "supported", "contract", "arguments", "preconditions", "completion", "failure_states", "handoff",
                "recovery", "dependencies", "example", "implementation")
        return json.dumps({k: e[k] for k in keys if k in e}, indent=1)
    return (template.replace("{{INSTRUCTION}}", lang).replace("{{VOCABULARY}}", json.dumps(vocab, indent=1))
            .replace("{{CAPABILITIES}}", json.dumps(caps, indent=1)).replace("{{CATALOG}}", "\n".join(fmt(e) for e in entries)))


def orch_main(ns, *, diag_dir, query_llm, catalog_path, prompt_path, code_hashes):
    """Episode entry for arm N20O (called from the program's last line with the policy namespace). One generation, no rewrite, no
    fallback. The summary is written on every exit path, including the episode horizon and frozen-executor exceptions (re-raised)."""
    out = Path(diag_dir) / "orch"; out.mkdir(parents=True, exist_ok=True)
    td = ns["task_descriptors"](); lang = str(ns["env"].handle.task_language)
    summary = {"arm_mode": "llm_orchestration", "method": "inference method only; the learned stack is " + STACK_ID,
               "stack": {"id": STACK_ID, "sha256": STACK_SHA256}, "interface_version": INTERFACE_VERSION,
               "checker_version": OC.CHECKER_VERSION, "prompt_version": PROMPT_VERSION, "code_hashes": code_hashes, "language": lang,
               "fallback_to_baseline": False, "llm_rewrite_on_failure": False, "outcome": None}
    holder = {"stack": None}

    def fill():
        st = holder["stack"]
        if st is None:
            return
        summary.update(stop=st._finish, n_stack_calls=len(st._calls), plan=st._plan, retries=dict(st._retries), stack_exception=st._stack_exception,
                       budget_used=dict(st._n, retries_total=sum(st._retries.values())),
                       results={k: {f: r.get(f) for f in ("status", "failure_reason", "verification_status", "object_possibly_retained", "retry")} for k, r in st._results.items()})

    def write():
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))

    def done(outcome, **kw):
        fill(); summary.update(outcome=outcome, **kw); write()
        print("ORCH_DONE " + json.dumps({"outcome": outcome, **{k: summary.get(k) for k in ("program_sha256", "stop", "n_stack_calls")}}, default=_jsonable), flush=True)
        return summary

    try:
        return _orch_body(ns, td, lang, out, summary, holder, done, query_llm, catalog_path, prompt_path)
    except BaseException as e:   # horizon / stack exception: record, then propagate unchanged
        if summary.get("outcome") is None:
            st = holder["stack"]
            kind = ("episode_timeout_during_orchestration" if type(e).__name__ == "EpisodeHorizonReached"
                    else "stack_exception" if (st is not None and st._stack_exception is not None) else "orchestration_exception")
            fill(); summary.update(outcome=kind, exception=repr(e)[:300]); write()
            print("ORCH_DONE " + json.dumps({"outcome": kind}), flush=True)
        raise


def _orch_body(ns, td, lang, out, summary, holder, done, query_llm, catalog_path, prompt_path):
    cat = load_catalog(catalog_path); summary["catalog"] = {"path": str(catalog_path), "sha256": hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(), "version": cat["catalog_version"]}
    template = Path(prompt_path).read_text(); summary["prompt_sha256"] = hashlib.sha256(template.encode()).hexdigest()
    stack = RSIStack(ns, td, lang, out); holder["stack"] = stack
    vocab = {"objects": sorted(td.get("objects") or {}), "supports": sorted(td.get("supports") or {}), "articulated": sorted(td.get("articulated") or {}),
             "side_words": dict(td.get("side_words") or {})}
    caps = capability_table(ns, td)
    entries, skipped = retrieve(cat, td, lang)
    obs0 = ns["get_observation"](); rgb0 = obs0["agentview"]["images"]["rgb"]
    robot0 = stack._robot(obs0)
    prompt = build_prompt(template, lang, vocab, caps, entries)
    gen_ctx = {"model": "claude-opus-5-5 (proxy-fixed)", "effort": "low (proxy-fixed)", "max_tokens": GEN_MAX_TOKENS, "temperature": "provider default (not set; not seed-controlled)",
               "n_generations": 1, "retrieved": [e["id"] for e in entries], "skipped": skipped, "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "initial_observation": {"image": "orch/initial_rgb.png", "robot": robot0, "sim_step": stack._sim_step()}}
    try:
        from PIL import Image
        import numpy as np
        Image.fromarray(np.asarray(rgb0, dtype=np.uint8)).save(out / "initial_rgb.png")
    except Exception as e:  # noqa: BLE001
        gen_ctx["initial_observation"]["image_error"] = repr(e)
    (out / "generation_context.json").write_text(json.dumps(gen_ctx, indent=1, default=_jsonable))
    summary["generation"] = {k: gen_ctx[k] for k in ("model", "effort", "max_tokens", "temperature", "n_generations", "retrieved", "prompt_sha256")}
    resp = query_llm([rgb0], prompt, GEN_SCHEMA, tag="orch_generate", max_tokens=GEN_MAX_TOKENS)
    (out / "generation_response.json").write_text(json.dumps(resp, indent=1, default=_jsonable))
    summary["generation"].update({k: resp.get(k) for k in ("status", "cost_usd", "usage", "id", "latency_s")})
    if resp.get("status") != "ok" or not isinstance((resp.get("data") or {}).get("program"), str) or not resp["data"]["program"].strip():
        return done("generation_error", detail=resp.get("status"))
    src = resp["data"]["program"]; psha = hashlib.sha256(src.encode()).hexdigest()
    (out / "program.py").write_text(src); summary["program_sha256"] = psha; summary["plan_notes"] = str(resp["data"].get("plan_notes"))[:2000]
    print("ORCH_PROGRAM " + json.dumps({"sha256": psha, "n_chars": len(src)}), flush=True)
    rep = OC.check_program(src, vocab=vocab, supported=ns["LA_SUPPORTED"], object_sides=ns["LA_OBJECT_SIDES"], part_relations=ns["LA_PART_RELATIONS"],
                           recovery_actions=RECOVERY_ACTIONS)
    (out / "check_report.json").write_text(json.dumps(rep, indent=1, default=str))
    summary["check"] = {"ok": rep["ok"], "n_errors": len(rep["errors"]), "kinds": sorted({e["kind"] for e in rep["errors"]})}
    print("ORCH_CHECK " + json.dumps(summary["check"]), flush=True)
    if not rep["ok"]:
        cap_only = all(e["kind"].startswith("capability") for e in rep["errors"])
        return done("capability_unsupported" if cap_only else "program_rejected", check_errors=rep["errors"][:20])
    context = {"instruction": lang, "initial_observation": {"obs_id": 0, "image_shape": list(rgb0.shape[:2]), **robot0},
               "vocabulary": vocab, "capabilities": caps}
    res = execute_program(src, stack, copy.deepcopy(context))
    summary["program_run"] = res
    plan_refused = any(c["method"] == "plan" and (c["result"] or {}).get("status") == "unsupported" for c in stack._calls)
    if res["status"] == "runtime_error":
        return done("program_runtime_error")
    if res["status"] == "budget_exceeded":
        return done("program_budget_exceeded")
    if stack._plan is None:
        return done("capability_unsupported" if plan_refused else "no_plan_executed")
    st = [r["status"] for r in stack._results.values()]
    if len(stack._results) == len(stack._plan) and all(s == "control_completed" for s in st):
        return done("all_controls_completed")
    order = ("unsupported", "binding_failed", "verification_failed", "execution_failed", "unknown")
    first = next((s for s in order if s in st), "not_all_executed")
    return done({"unsupported": "capability_unsupported"}.get(first, first))
