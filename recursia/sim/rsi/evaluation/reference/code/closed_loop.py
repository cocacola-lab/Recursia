"""N20C: CLOSED-LOOP LLM orchestration over the frozen N20 stack (inference method only; the learned stack is N20).

Separate experimental mode, kept apart from the one-shot N20O method; the original spec requires any iterative mode to be separate.
The LLM never writes code. It returns typed JSON only:
  1. one PLAN (subgoals from the frozen vocabulary). One re-plan is allowed when the frozen capability check refuses the plan before
     any motion; this is capability feedback, not target-outcome feedback;
  2. after every executed subgoal, a DECISION chosen from the action list the harness computed as allowed;
  3. VISUAL CHECK calls (the initial + current frames and a relation sentence -> yes / no / unsure), made after every object subgoal
     that reaches control_completed and once before stopping.
Actions, all on frozen N20 executors through the v2 interface (orch_stack.RSIStack):
  execute(sg)  first execution of a planned subgoal                                           (always allowed while pending)
  retry(sg)    one re-invocation after a failure, when the evidence shows nothing is held      (orch_stack retry rule + step rule)
  replace(sg)  one re-invocation after control_completed when the VISUAL CHECK says "no"       (only if the calibration gate passed + step rule)
  stop
Step rule: retry / replace need remaining steps >= STEP_COST * (1 + pending subgoals). STEP_COST = 1109 is the p75 of final_sim_step
over 426 successful single-goal LIBERO-90 development episodes (N20 round), so a recovery never starves later subgoals by design.
N20's MF placement verifier stays in shadow mode: in the 100 N20O episodes its "failed" verdict was right in only 11 of 23 cases."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import orch_stack as OS

CL_VERSION = "n20c_closed_loop_v1_20261016"
PLAN_PROMPT, DECIDE_PROMPT, VERIFY_PROMPT = "cl_plan_v1.md", "cl_decide_v1.md", "cl_verify_v1.md"
VERIFY_PROMPT_VERSION = "cl_verify_v1_20261008"
HORIZON = 4000
STEP_COST = 1109
LIMITS = {"llm_calls": 24, "decisions": 10, "checks": 10, "replace_per_subgoal": 1, "replace_total": 2, "invalid_decisions": 1}
REPLACE_ENABLED = True   # calibration gate PASSED 2026-10-08 (calibration/result.json): precision of "no" 46/49 = 0.939 >= 0.90, n_no 49 >= 20; recall 0.72

NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
PLAN_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["plan_notes", "subgoals"], "properties": {
    "plan_notes": {"type": "string"},
    "subgoals": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                 "required": ["relation", "object", "reference", "part", "side", "object_side", "instance"],
                 "properties": {"relation": {"type": "string"}, "object": NULLABLE_STR, "reference": NULLABLE_STR, "part": NULLABLE_STR,
                                "side": NULLABLE_STR, "object_side": NULLABLE_STR, "instance": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}}}
DECIDE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["action", "subgoal", "reason"], "properties": {
    "action": {"type": "string", "enum": ["execute", "retry", "replace", "stop"]}, "subgoal": NULLABLE_STR, "reason": {"type": "string"}}}
VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdict", "evidence"], "properties": {
    "verdict": {"type": "string", "enum": ["yes", "no", "unsure"]}, "evidence": {"type": "string"}}}

SIDE_TEXT = {"min": "left", "max": "right", "xmin": "front", "xmax": "back", "zmin": "bottom", "zmax": "top", "mid": "middle"}


def relation_clause(rec):
    """Public relation sentence from a subgoal record (phrases from the frozen vocabulary; no simulator names)."""
    r = rec.get("relation"); obj = rec.get("object"); ref = rec.get("support") or rec.get("part"); side = SIDE_TEXT.get(rec.get("side"), rec.get("side"))
    osd = SIDE_TEXT.get(rec.get("object_side"), rec.get("object_side"))
    o = f"the {osd + ' ' if osd and osd not in str(obj) else ''}{obj}"
    if r == "on":
        return f"{o} is resting on top of the {side + ' ' if side and side != 'top' else ''}{ref}" if side != "top" else f"{o} is resting on the top face of the {ref}"
    if r == "in":
        return f"{o} is inside the {ref}"
    if r == "beside":
        return f"{o} is on the table to the {side} of the {ref}, not on it"
    if r == "in_part":
        return f"{o} is inside the {side} compartment of the {ref}"
    if r == "in_sliding":
        return f"{o} is inside the {side} {rec.get('part')} of the {rec.get('support')}"
    if r == "close_articulated":
        return f"the {side + ' ' if side else ''}{rec.get('part') or ref} is fully closed"
    if r == "open_articulated":
        return f"the {side + ' ' if side else ''}{rec.get('part') or ref} is open"
    if r == "actuate_on":
        return f"the {ref} is turned on"
    if r == "actuate_off":
        return f"the {ref} is turned off"
    return f"{o} satisfies '{r}' with the {ref}"


def visual_check(query_llm, template, images, instruction, clause, tag="cl_verify"):
    """One structured relation check -> {"verdict": yes|no|unsure, "evidence", "status", "cost_usd"}; abstention -> 'unsure'."""
    prompt = template.replace("{{INSTRUCTION}}", instruction).replace("{{CLAUSE}}", clause)
    resp = query_llm(images, prompt, VERIFY_SCHEMA, tag=tag, max_tokens=800)
    d = resp.get("data") if resp.get("status") == "ok" else None
    v = (d or {}).get("verdict") if isinstance(d, dict) else None
    return {"verdict": v if v in ("yes", "no", "unsure") else "unsure", "evidence": str((d or {}).get("evidence", ""))[:400] if isinstance(d, dict) else "",
            "status": resp.get("status"), "cost_usd": resp.get("cost_usd"), "usage": resp.get("usage")}


class ClosedLoopStack(OS.RSIStack):
    """RSIStack + replace(): one re-invocation of the frozen executor for a control_completed object subgoal, with the subgoal's own
    previous placement removed from others_xy (it is the object being moved), the frozen VB reference unchanged."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._replaced = {}; self._placed_by = {}

    def _sim_step(self):
        env = self._ns.get("env")   # the namespace's env IS the low-level env (v1/v2 read env.low_level_env and recorded None)
        for obj in (env, getattr(env, "low_level_env", None)):
            v = getattr(obj, "_sim_step_count", None)
            if v is not None:
                return int(v)
        return None

    def replace(self, sid):
        last = self._results.get(sid); rec = self._sg.get(sid)
        if last is None or last["status"] != "control_completed" or rec is None or rec["relation"] in self._ns["LA_PART_RELATIONS"]:
            return self._refuse("replace", {"id": sid}, "replace_needs_a_control_completed_object_subgoal")
        if self._replaced.get(sid, 0) >= LIMITS["replace_per_subgoal"] or sum(self._replaced.values()) >= LIMITS["replace_total"]:
            return self._refuse("replace", {"id": sid}, "budget_replace")
        if not self._count("executes"):
            return self._refuse("replace", {"id": sid}, "budget_executes")
        self._replaced[sid] = self._replaced.get(sid, 0) + 1
        own = last.get("placement_xy")
        if own is not None:   # the object's own previous placement is not an obstacle for its re-placement
            for i, xy in enumerate(self._placed):
                if [round(float(v), 4) for v in xy] == own:
                    self._placed.pop(i); break
        self._results.pop(sid, None)
        r = self._RSIStack__execute(sid, retry=True)
        r["replace"] = True
        return r


def _rgb(ns):
    obs = ns["get_observation"]()
    return obs["agentview"]["images"]["rgb"]


def orch_closed_loop(ns, *, diag_dir, query_llm, catalog_path, prompt_dir, code_hashes, replace_enabled=None):
    out = Path(diag_dir) / "orch"; out.mkdir(parents=True, exist_ok=True)
    td = ns["task_descriptors"](); lang = str(ns["env"].handle.task_language)
    replace_enabled = bool(REPLACE_ENABLED) if replace_enabled is None else bool(replace_enabled)
    summary = {"arm_mode": "closed_loop_llm_orchestration", "method": "inference method only; the learned stack is " + OS.STACK_ID, "cl_version": CL_VERSION,
               "stack": {"id": OS.STACK_ID, "sha256": OS.STACK_SHA256}, "interface_version": OS.INTERFACE_VERSION, "code_hashes": code_hashes, "language": lang,
               "replace_enabled": replace_enabled, "step_cost": STEP_COST, "fallback_to_baseline": False, "outcome": None, "llm_calls": [], "decisions": [], "checks": []}
    holder = {"stack": None}
    prompts = {n: (Path(prompt_dir) / n).read_text() for n in (PLAN_PROMPT, DECIDE_PROMPT, VERIFY_PROMPT)}
    summary["prompt_sha256"] = {n: hashlib.sha256(t.encode()).hexdigest() for n, t in prompts.items()}

    def write():
        st = holder["stack"]
        if st is not None:
            summary.update(n_stack_calls=len(st._calls), plan=st._plan, retries=dict(st._retries), replaced=dict(st._replaced), stack_exception=st._stack_exception,
                           results={k: {f: r.get(f) for f in ("status", "failure_reason", "verification_status", "object_possibly_retained", "retry", "replace")} for k, r in st._results.items()},
                           final_sim_step=st._sim_step())
        summary["cost_usd"] = round(sum(float(c.get("cost_usd") or 0) for c in summary["llm_calls"]), 5)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=OS._jsonable))

    def done(outcome, **kw):
        summary.update(outcome=outcome, **kw); write()
        print("ORCH_DONE " + json.dumps({"outcome": outcome, "n_llm_calls": len(summary["llm_calls"]), "cost_usd": summary.get("cost_usd")}, default=str), flush=True)
        return summary

    try:
        return _loop(ns, td, lang, out, summary, holder, done, write, query_llm, catalog_path, prompts, replace_enabled)
    except BaseException as e:
        if summary.get("outcome") is None:
            st = holder["stack"]
            kind = ("episode_timeout_during_orchestration" if type(e).__name__ == "EpisodeHorizonReached"
                    else "stack_exception" if (st is not None and st._stack_exception is not None) else "orchestration_exception")
            summary.update(outcome=kind, exception=repr(e)[:300]); write()
            print("ORCH_DONE " + json.dumps({"outcome": kind}), flush=True)
        raise


def _loop(ns, td, lang, out, summary, holder, done, write, query_llm, catalog_path, prompts, replace_enabled):
    cat = OS.load_catalog(catalog_path)
    stack = ClosedLoopStack(ns, td, lang, out); holder["stack"] = stack
    vocab = {"objects": sorted(td.get("objects") or {}), "supports": sorted(td.get("supports") or {}), "articulated": sorted(td.get("articulated") or {}),
             "side_words": dict(td.get("side_words") or {})}
    caps = OS.capability_table(ns, td)
    entries, _ = OS.retrieve(cat, td, lang)
    entries = [e for e in entries if e["id"] not in ("core.program", "core.budget_log_finish")]   # program-format entries do not apply to typed actions
    rgb0 = _rgb(ns)
    try:
        from PIL import Image
        import numpy as np
        Image.fromarray(np.asarray(rgb0, dtype=np.uint8)).save(out / "initial_rgb.png")
    except Exception:  # noqa: BLE001
        pass

    def call(images, prompt, schema, tag, max_tokens):
        if len(summary["llm_calls"]) >= LIMITS["llm_calls"]:
            return {"status": "budget_exhausted_local"}
        resp = query_llm(images, prompt, schema, tag=tag, max_tokens=max_tokens)
        rec = {"tag": tag, "status": resp.get("status"), "cost_usd": resp.get("cost_usd"), "usage": resp.get("usage"), "id": resp.get("id"),
               "latency_s": resp.get("latency_s"), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "sim_step": stack._sim_step()}
        summary["llm_calls"].append(rec)
        with open(out / "llm_calls.jsonl", "a") as fh:
            fh.write(json.dumps(dict(rec, prompt=prompt, data=resp.get("data")), default=str) + "\n")
        return resp

    # ---- 1. plan (one capability re-plan allowed, before any motion)
    base_plan_prompt = OS.build_prompt(prompts[PLAN_PROMPT], lang, vocab, caps, entries)
    feedback = ""; ids = None
    for attempt in range(2):
        resp = call([rgb0], base_plan_prompt + feedback, PLAN_SCHEMA, f"cl_plan_{attempt}", 3000)
        if resp.get("status") != "ok" or not isinstance((resp.get("data") or {}).get("subgoals"), list):
            return done("generation_error", detail=resp.get("status"))
        data = resp["data"]; summary.setdefault("plan_attempts", []).append(data)
        subs = data["subgoals"][:OS.LIMITS["executes"]]
        made = [stack.subgoal(**{k: s.get(k) for k in ("relation", "object", "reference", "part", "side", "object_side", "instance")}) for s in subs]
        bad = [m for m in made if m.get("status") != "ok"]
        if not subs:
            bad = [{"why": "empty_plan"}]
        if not bad:
            p = stack.plan([m["id"] for m in made])
            if p["status"] == "ok":
                ids = [m["id"] for m in made]; break
            bad = p.get("unsupported") or [{"why": p.get("failure_reason")}]
        if attempt == 0:
            feedback = ("\n\nYour previous plan was REFUSED by the frozen capability check before any motion: "
                        + json.dumps([{k: b.get(k) for k in ("relation", "object", "reference", "part", "side", "why")} for b in bad], default=str)
                        + "\nReturn a corrected plan using only implemented relations / side words / vocabulary, or an EMPTY subgoal list if the "
                          "instruction cannot be implemented by this stack. Never substitute an unrelated capability.")
            stack = ClosedLoopStack(ns, td, lang, out); holder["stack"] = stack   # fresh interface state: nothing has moved
    if ids is None:
        return done("capability_unsupported")
    recs = {s: stack._sg[s] for s in ids}
    clauses = {s: relation_clause(recs[s]) for s in ids}
    state = {s: {"status": "pending", "attempts": 0, "replaced": 0, "check": None, "last_failure": None, "retained": None} for s in ids}
    last = None; final_check_done = False

    def steps_remaining():
        v = stack._sim_step()
        return None if v is None else HORIZON - v

    def pending_after(sid):
        return sum(1 for s in ids if s != sid and state[s]["status"] == "pending")

    def recovery_ok(sid):
        rem = steps_remaining()
        return rem is None or rem >= STEP_COST * (1 + pending_after(sid))

    def allowed():
        acts = [{"action": "execute", "subgoal": s} for s in ids if state[s]["status"] == "pending"]
        for s in ids:
            r = stack._results.get(s)
            if r is None or not recovery_ok(s):
                continue
            if r["status"] in ("binding_failed", "execution_failed", "verification_failed") and "retry_subgoal" in r.get("allowed_recovery_actions", ()):
                acts.append({"action": "retry", "subgoal": s})
            if (replace_enabled and r["status"] == "control_completed" and recs[s]["relation"] not in ns["LA_PART_RELATIONS"] and state[s]["check"] == "no"
                    and stack._replaced.get(s, 0) < LIMITS["replace_per_subgoal"] and sum(stack._replaced.values()) < LIMITS["replace_total"]):
                acts.append({"action": "replace", "subgoal": s})
        acts.append({"action": "stop", "subgoal": None})
        return acts

    def check(sid):
        if len(summary["checks"]) >= LIMITS["checks"]:
            return
        rgb = _rgb(ns)
        v = visual_check(lambda *a, **k: call(*a, **k), prompts[VERIFY_PROMPT], [rgb0, rgb], lang, clauses[sid], tag=f"cl_verify_{sid}")
        state[sid]["check"] = v["verdict"]
        summary["checks"].append({"subgoal": sid, "clause": clauses[sid], "verdict": v["verdict"], "evidence": v["evidence"], "sim_step": stack._sim_step()})
        print("CL_CHECK " + json.dumps({"subgoal": sid, "verdict": v["verdict"]}), flush=True)

    def absorb(sid, r):
        state[sid]["attempts"] += 1; state[sid]["status"] = r["status"]; state[sid]["last_failure"] = r.get("failure_reason")
        state[sid]["retained"] = r.get("object_possibly_retained"); state[sid]["check"] = None
        if r.get("replace"):
            state[sid]["replaced"] += 1
        if r["status"] == "control_completed" and recs[sid]["relation"] not in ns["LA_PART_RELATIONS"]:
            check(sid)

    # ---- 2. decision loop
    for n in range(LIMITS["decisions"]):
        acts = allowed()
        view = {"instruction": lang, "steps_used": stack._sim_step(), "steps_remaining": steps_remaining(), "step_cost_estimate_per_subgoal": STEP_COST,
                "subgoals": [{"id": s, "relation_sentence": clauses[s], **state[s]} for s in ids],
                "last_result": None if last is None else {k: last.get(k) for k in ("id", "status", "failure_reason", "object_possibly_retained", "verification_status", "replace", "retry")},
                "allowed_actions": acts, "replace_enabled": replace_enabled}
        prompt = prompts[DECIDE_PROMPT].replace("{{STATE}}", json.dumps(view, indent=1, default=str))
        rgb = _rgb(ns)
        dec = None
        for k in range(1 + LIMITS["invalid_decisions"]):
            resp = call([rgb0, rgb], prompt, DECIDE_SCHEMA, f"cl_decide_{n}_{k}", 1500)
            if resp.get("status") != "ok":
                return done("decision_error", detail=resp.get("status"))
            d = resp["data"]; cand = {"action": d.get("action"), "subgoal": d.get("subgoal") if d.get("action") != "stop" else None}
            if cand in acts:
                dec = dict(cand, reason=str(d.get("reason"))[:400]); break
            prompt = prompt + "\n\nYour previous answer " + json.dumps(cand) + " is not in allowed_actions. Choose exactly one entry of allowed_actions."
        if dec is None:
            return done("invalid_decision")
        summary["decisions"].append(dict(dec, n=n, sim_step=stack._sim_step()))
        print("CL_DECISION " + json.dumps(dec), flush=True)
        if dec["action"] == "stop":
            done_objs = [s for s in ids if state[s]["status"] == "control_completed" and recs[s]["relation"] not in ns["LA_PART_RELATIONS"]]
            if replace_enabled and not final_check_done and done_objs and sum(stack._replaced.values()) < LIMITS["replace_total"]:
                final_check_done = True   # a later placement can disturb an earlier one: re-check every placed object once before stopping
                for s in done_objs:
                    check(s)
                if any(state[s]["check"] == "no" for s in done_objs):
                    continue
            stack.finish("stop", dec["reason"])
            break
        sid = dec["subgoal"]
        if dec["action"] == "execute":
            last = stack.execute(sid)
        elif dec["action"] == "retry":
            last = stack.recover(stack._results[sid], "retry_subgoal")
        else:
            last = stack.replace(sid)
        absorb(sid, last)
    else:
        stack.finish("decision_budget", "decision budget exhausted")
    st = [stack._results.get(s, {}).get("status", "not_executed") for s in ids]
    if all(s == "control_completed" for s in st):
        return done("all_controls_completed")
    order = ("unsupported", "binding_failed", "verification_failed", "execution_failed", "unknown", "not_executed")
    return done(next((s for s in order if s in st), "incomplete"))
