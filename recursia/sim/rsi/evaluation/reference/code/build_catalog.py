"""Build the versioned interface catalog for the frozen N20 stack (catalog/rsi_n20_interface_catalog_v2.json).
Every entry names the frozen functions it wraps, with the sha256 of each function's source in programs/N20.py and the hash
of the whole stack program; the wrapper module's hash is recorded too. Run: .venv-libero/bin/python3 code/build_catalog.py"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import orch_offline as O
import orch_stack as OS

CATALOG_VERSION = "rsi_n20_catalog_v2_20261016"


_NS = None


def sides_for(relation):
    global _NS
    if _NS is None:
        _NS = O.load_namespace("put the bowl on the plate")[0]
    sw = O.descriptors_for("put the bowl on the plate")["side_words"]; inv = {}
    for w, c in sw.items():
        inv.setdefault(c, []).append(w)
    return sorted({w for c in _NS["LA_SUPPORTED"][relation] if c is not None for w in inv.get(c, [])}) + ([None] if None in _NS["LA_SUPPORTED"][relation] else [])


def fn_hashes(names):
    src = O.stack_source(); tree = ast.parse(src); out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()
    missing = set(names) - set(out)
    assert not missing, missing
    return out


COMMON = {"stack_sha256": OS.STACK_SHA256}
OBJ_FAIL = ["binding_failed (BINDING_STOP: target not bound unambiguously / support_not_found / region_direction_undefined; VB identity_unknown) - pre-motion",
            "execution_failed (approach / transport / lift non-arrival, infeasible candidate, all grasp strategies failed)",
            "verification_failed (enforced gate refused: grasp_not_verified, object_lost_after_lift, lift_refused, VB transport gate, release_blocked)",
            "unknown (no terminal evidence line)"]
OBJ_HANDOFF = ("Inside the executor (frozen, enforced): VB pre-grasp identity against the task-start reference; grasp contact verification; "
               "E1 lift / transport guards; VB transport gate = identity verified AND hold verified AND object rose with the gripper; "
               "MC release gate (descent reached or contact established). Placement verifiers MF/V1 run in SHADOW mode: their verdicts "
               "are reported as shadow_verdicts and never gate.")
OBJ_RECOVERY = ("reobserve, go_home always; retry_subgoal once per subgoal (3 per episode) only when the evidence shows the object is not "
                "retained (object_possibly_retained == False) and the status is a failure. Never after control_completed.")


def obj_entry(eid, relation, purpose, sides, triggers, example, requires=("objects",), extra=None):
    e = {"id": eid, "name": f"stack.execute(<{relation} subgoal>)", "kind": "executor", "relations": [relation],
         "signature": f"stack.subgoal('{relation}', object=<object phrase>, reference=<support/object phrase>, side=<side word|None>, object_side=<side word|None>) -> subgoal; stack.execute(subgoal) -> result",
         "purpose": purpose, "supported": {"relation": relation, "side_words_or_none": sides},
         "arguments": {"object": "phrase from vocabulary.objects (prompts come from the frozen adapter lexicon)",
                       "reference": "phrase from vocabulary.supports (or a second object phrase as the reference)",
                       "side": "side word selecting the reference instance / region (must be implemented for this relation)",
                       "object_side": "side word selecting the object instance (left/right/front/back/top/bottom/middle)"},
         "preconditions": ["the subgoal is in the frozen plan (stack.plan) - task-start identity references were taken there",
                           "stack.check(subgoal).supported is True (la_dispatch)", "the hand is empty (the wrapper homes before every subgoal after the first)"],
         "completion": "status == 'control_completed' means PLACE_RESULT released (object released at the target and the hand retreated). It is NOT verified task success: verification_status is 'shadow_recorded_only'.",
         "failure_states": OBJ_FAIL, "handoff": OBJ_HANDOFF, "recovery": OBJ_RECOVERY,
         "dependencies": ["stack.plan", "VB task-start reference", "SAM3 / GraspNet / PyRoki services", "previous placements (others_xy) in execution order"],
         "example": example, "requires_vocabulary": list(requires), "triggers": triggers,
         "implementation": {"wrapper": "orch_stack.RSIStack.execute -> run_subgoal", "frozen_functions": fn_hashes(["run_subgoal", "prim_place", "attempt_grasp_verify", "vb_localize_for_grasp", "vb_post_grasp", "la_dispatch", "la_reference_prompts", "la_assembly_prompts", "place_target", "relation_geometry"] + (["lp_run_subgoal"] if relation == "in_layer" else [])), **COMMON}}
    e.update(extra or {})
    return e


def part_entry(eid, relation, executor, purpose, sides, triggers, example, claim):
    return {"id": eid, "name": f"stack.execute(<{relation} subgoal>)", "kind": "executor", "relations": [relation],
            "signature": f"stack.subgoal('{relation}', reference=<part or support phrase>, part=<part phrase|None>, side=<side word|None>) -> subgoal; stack.execute(subgoal) -> result",
            "purpose": purpose, "supported": {"relation": relation, "side_words_or_none": sides},
            "arguments": {"reference": "phrase from vocabulary.articulated (a part) or vocabulary.supports", "part": "phrase from vocabulary.articulated", "side": "side word selecting the part (where implemented)"},
            "preconditions": ["the subgoal is in the frozen plan", "stack.check(subgoal).supported is True"],
            "completion": f"status == 'control_completed' means the executor returned True: its own perception verifier claimed '{claim}' (verification_status 'stack_verifier_claim_positive'). This claim controlled the executor's loop; it is not official task success.",
            "failure_states": ["binding_failed (BINDING_STOP part_not_found / no_floor_estimate)", "execution_failed (*_STOP motion_* / plan not found / cycles exhausted)",
                               f"verification_failed (stack verifier claim not_{claim}, grasp_not_verified)", "unknown"],
            "handoff": "Executor-internal: bounded measure -> act -> re-observe cycles with its own verdict; abort paths open the gripper and home.",
            "recovery": "reobserve, go_home; retry_subgoal once only when object_possibly_retained == False (a stop in mid-motion leaves the hand state unknown: no retry).",
            "dependencies": ["stack.plan"], "example": example, "requires_vocabulary": ["articulated"] if relation != "actuate_on" else [], "triggers": triggers,
            "implementation": {"wrapper": f"orch_stack.RSIStack.execute -> {executor}", "frozen_functions": fn_hashes([executor, "la_dispatch"]), **COMMON}}


def build():
    core = [
        {"id": "core.program", "always": True, "name": "def run(context, stack)", "kind": "contract",
         "signature": "def run(context, stack): ...   # the only entry point; context = {instruction, initial_observation, vocabulary, capabilities}",
         "purpose": "One-shot task orchestration program, INTERPRETED (never exec'd). It may branch on structured results and loop within fixed bounds; it never controls the robot directly.",
         "supported": {"statements": "name = value; if/elif/else; for x in range(<int literal 0..8>) or for x in <held list of <= 16 items>; break; continue; return; pass",
                       "expressions": "str/int/float/bool/None literals, lists, tuples, dicts with string-literal keys, indexing value[key], and/or/not, comparisons (== != < <= > >= in, not in, is), x if c else y, + on two strings / two lists / two integers (use str(x) for others; failure_reason may be None)",
                       "calls": "stack.<method>(...) for the methods below, len(x), str(x); nothing else (no attribute reads such as d.get or list.append)",
                       "stack_methods": list(OS.OC.STACK_METHODS)},
         "preconditions": ["exactly one top-level `def run(context, stack):`; no imports, helpers, classes, while, try, with, raise, comprehensions, lambdas, augmented assignment",
                           "no names starting with _; `stack` only as stack.<method>(...); results are read by indexing (every result has a fixed set of keys)"],
         "completion": "returning from run() ends orchestration; stack.finish(status, reason) records the program's own declared stop (a claim, not success)",
         "failure_states": ["program_rejected (static check)", "program_runtime_error", "program_budget_exceeded"],
         "handoff": "-", "recovery": "-", "dependencies": [],
         "example": "def run(context, stack):\n    sg = stack.subgoal('in', object='alphabet soup', reference='basket')\n    p = stack.plan([sg])\n    if p['status'] != 'ok':\n        stack.finish('unsupported', str(p['unsupported']))\n        return\n    r = stack.execute(sg)\n    if r['status'] != 'control_completed' and 'retry_subgoal' in r['allowed_recovery_actions']:\n        r = stack.recover(r, 'retry_subgoal')\n    stack.finish(r['status'], str(r['failure_reason']))",
         "implementation": {"wrapper": "orch_check.check_program + orch_check.Interpreter (via orch_stack.execute_program)", **COMMON}},
        {"id": "core.subgoal_check_plan", "always": True, "name": "stack.subgoal / stack.check / stack.plan", "kind": "binding+capability",
         "signature": "stack.subgoal(relation, object=None, reference=None, part=None, side=None, object_side=None, instance=None) -> {id, supported, why, ...}; stack.check(sg) -> {supported, why, implemented}; stack.plan([sg, ...]) -> {status, references | unsupported}",
         "purpose": "Declare subgoals from vocabulary phrases (no free-text prompts); capability check = the frozen la_dispatch; plan() freezes the ordered subgoal list ONCE and resolves the VB task-start identity reference of every object subgoal before any object moves.",
         "supported": "relations and side words: see context['capabilities']; phrases: context['vocabulary']",
         "arguments": {"relation": "one of capabilities.relations", "side": "a side word (left/right/front/back/top/bottom/middle) implemented for the relation, or None"},
         "preconditions": ["subgoal() only before plan()", "plan() exactly once, with 1-8 distinct subgoals, all supported; any unsupported subgoal refuses the whole plan (nothing moves)"],
         "completion": "plan status 'ok' with references[{id, handle, status: frozen|unresolved|none}]; an 'unresolved' reference makes that subgoal's execute end binding_failed (identity_unknown) without motion",
         "failure_states": ["unsupported (why: relation | side | object_side | vocabulary | reference_kind | part_not_allowed)", "refused (plan_already_frozen, budget)"],
         "contract": "Interface contract stricter than la_dispatch: an object relation's reference must be a support or object phrase (never an articulated part such as a microwave: in_hinged is not implemented); a part argument only for in_sliding.",
         "handoff": "the frozen reference is what the executor's pre-grasp and post-grasp identity gates compare against",
         "recovery": "-", "dependencies": [],
         "example": "a = stack.subgoal('on', object='white mug', reference='plate', side='left')\nb = stack.subgoal('on', object='yellow and white mug', reference='plate', side='right')\np = stack.plan([a, b])",
         "implementation": {"wrapper": "orch_stack.RSIStack.subgoal/check/plan + build_subgoal_record", "frozen_functions": fn_hashes(["la_dispatch", "la_summary", "vb_resolve_references", "localize_target", "vb_point_localize"]), **COMMON}},
        {"id": "core.execute_result", "always": True, "name": "stack.execute(sg) / stack.recover(result, action)", "kind": "execution+recovery",
         "signature": "stack.execute(sg) -> result; stack.recover(result, 'reobserve' | 'go_home' | 'retry_subgoal') -> result",
         "purpose": "Run the complete frozen subgoal executor for a planned subgoal and return structured evidence.",
         "supported": {"result_fields": ["status: control_completed | binding_failed | execution_failed | verification_failed | unsupported | unknown | refused",
                                         "failure_reason", "object_reference {handle, status, phrase}", "object_possibly_retained (True / False / None=unknown)",
                                         "verification_status: shadow_recorded_only | stack_verifier_claim_positive | stack_verifier_claim_negative | gate_refused | not_verified",
                                         "enforced_gates [...]", "shadow_verdicts [...] (logged only, never gated)", "stack_verifier_claims [...]",
                                         "allowed_recovery_actions", "execution_evidence {terminal lines, sim_steps, wall_s}", "placement_xy"]},
         "preconditions": ["sg in the plan; each subgoal executes once; a second attempt only via recover(result, 'retry_subgoal') when allowed_recovery_actions lists it"],
         "completion": "control_completed is control completion, never task success (the benchmark evaluator decides success)",
         "failure_states": ["refused: no_plan / subgoal_not_in_plan / already_control_completed / already_executed / budget_*"],
         "handoff": "the wrapper reproduces the baseline driver: production home + 3 observations before every subgoal after the first; previously returned placements are obstacles (others_xy)",
         "recovery": OBJ_RECOVERY, "dependencies": ["stack.plan"],
         "example": "r = stack.execute(a)\nif r['status'] != 'control_completed' and 'retry_subgoal' in r['allowed_recovery_actions']:\n    r = stack.recover(r, 'retry_subgoal')",
         "implementation": {"wrapper": "orch_stack.RSIStack.execute/recover", "frozen_functions": fn_hashes(["run_subgoal", "safe_retreat", "prim_place"]), **COMMON}},
        {"id": "core.observe_locate", "always": True, "name": "stack.observe / stack.robot_state / stack.locate / stack.reference", "kind": "observation",
         "signature": "stack.observe() -> {obs_id, image_shape, tcp_position, tcp_quat_wxyz, gripper_opening}; stack.locate(phrase, object_side=None) -> {found, centre, binding_status}; stack.reference(sg) -> {handle, status}",
         "purpose": "Fresh public observation summaries and perception-only text binding (localize_target) for branching, e.g. whether an object is still visible or where it is now.",
         "supported": "locate: vocabulary phrases only; result is text binding, NOT an identity confirmation (identity gates live inside execute)",
         "preconditions": ["budgets: reobserve 20, locate 10"], "completion": "status ok", "failure_states": ["unsupported (vocabulary / object_side)", "refused (budget)"],
         "handoff": "-", "recovery": "-", "dependencies": [],
         "example": "s = stack.locate('moka pot')\nif not s['found']:\n    stack.finish('stopped', 'moka pot not visible')",
         "implementation": {"wrapper": "orch_stack.RSIStack.observe/locate", "frozen_functions": fn_hashes(["localize_target", "tcp_from_obs", "vb_observe"]), **COMMON}},
        {"id": "core.budget_log_finish", "always": True, "name": "stack.budget / stack.log / stack.finish", "kind": "bookkeeping",
         "signature": "stack.budget() -> {used, limits}; stack.log(message, data=None); stack.finish(status, reason=None)",
         "purpose": "Inspect remaining budgets, leave notes in the episode log, declare the program's stop reason (a claim, not success).",
         "supported": {"limits": dict(OS.LIMITS)}, "preconditions": [], "completion": "-", "failure_states": ["refused (budget_log)"], "handoff": "-", "recovery": "-", "dependencies": [],
         "example": "stack.finish('stopped', 'subgoal 1 binding failed after one retry')",
         "implementation": {"wrapper": "orch_stack.RSIStack.budget/log/finish", **COMMON}},
    ]
    execs = [
        obj_entry("exec.on", "on", "Pick the object and place it ON a support (support centre; top face with side 'top'; plate instance chosen by side).",
                  ["left", "right", "front", "middle", "top", None], ["on", "onto", "on top of"],
                  "sg = stack.subgoal('on', object='moka pot', reference='stove')", requires=("objects", "supports")),
        obj_entry("exec.beside", "beside", "Pick the object and place it beside the reference on the given side (outside the reference footprint).",
                  ["left", "right", "front"], ["left of", "right of", "front of", "next to", "beside"],
                  "sg = stack.subgoal('beside', object='chocolate pudding', reference='plate', side='right')"),
        obj_entry("exec.in", "in", "Pick the object and place it IN a container support (interior centre, container gate on the support).",
                  [None], ["in", "into", "inside"], "sg = stack.subgoal('in', object='butter', reference='basket')"),
        obj_entry("exec.in_part", "in_part", "Pick the object and place it in a bay / compartment of a support selected by side (e.g. back compartment of the caddy).",
                  ["left", "right", "front", "back"], ["compartment", "section"], "sg = stack.subgoal('in_part', object='book', reference='caddy', side='back')", requires=("objects", "supports")),
        obj_entry("exec.in_sliding", "in_sliding", "Pick the object and place it in an OPEN sliding part (drawer) of an assembly, level chosen by side (top / bottom).",
                  ["top", "bottom"], ["drawer"], "sg = stack.subgoal('in_sliding', object='black bowl', reference='cabinet', part='drawer', side='bottom')", requires=("objects", "articulated")),
        obj_entry("exec.in_layer", "in_layer", "Pick the object and place it into a layer of a layered support (e.g. a shelf / rack): side 'bottom' (or the preposition 'under') = the lowest layer below the cover, 'top' (or 'on') = on the cover / highest layer; N18 runs it through run_subgoal -> lp_run_subgoal.",
                  ["top", "bottom", None], ["layer", "shelf", "rack"], "sg = stack.subgoal('in_layer', object='wine bottle', reference='wine rack', side='top')", requires=("objects", "supports")),
        part_entry("exec.close_articulated", "close_articulated", "run_close_articulated", "Close an articulated part (drawer / door) by bounded push cycles.",
                   [None, "left", "right", "front", "back", "top", "bottom"], ["close", "shut"], "sg = stack.subgoal('close_articulated', part='drawer', side='bottom')", "closed"),
        part_entry("exec.open_articulated", "open_articulated", "run_open_articulated", "Open an articulated part by one grasp-and-pull cycle.",
                   [None, "left", "right", "front", "back", "top", "bottom"], ["open"], "sg = stack.subgoal('open_articulated', part='drawer', side='top')", "opened"),
        part_entry("exec.actuate_on", "actuate_on", "run_rotary", "Turn a control on (e.g. the stove knob) by rotary actuation with an indicator-state check (N12/N13 dispatch actuate_on to run_rotary).",
                   [None], ["turn on", "switch on"], "sg = stack.subgoal('actuate_on', reference='stove')", "actuated"),
        part_entry("exec.actuate_off", "actuate_off", "run_rotary", "Turn a control off by rotary actuation with an indicator-state check.",
                   [None], ["turn off", "switch off"], "sg = stack.subgoal('actuate_off', reference='stove')", "actuated"),
    ]
    for e in execs:   # supported side words come from the frozen LA_SUPPORTED table, not from this file
        e["supported"]["side_words_or_none"] = sides_for(e["relations"][0])
    for e in core + execs:
        e.setdefault("stack_sha256", OS.STACK_SHA256)
    ns = O.load_namespace("put the bowl on the plate")[0]
    for e in execs:   # the executor named in each entry is the one run()'s dispatch selects
        assert e["implementation"]["wrapper"].endswith(OS.executor_name(ns, e["relations"][0])), e["id"]
    wrapper = {n: hashlib.sha256((O.HERE / n).read_bytes()).hexdigest() for n in ("orch_stack.py", "orch_check.py")}
    return {"catalog_version": CATALOG_VERSION, "interface_version": OS.INTERFACE_VERSION,
            "stack": {"id": OS.STACK_ID, "program": "programs/N20.py", "sha256": OS.STACK_SHA256, "lineage": "N20 = marker-only snapshot of C19I3 (rsi_libero90_N19_round_20261013; N18 -> C19I5 (N19) -> C19I3)"},
            "wrapper_sha256": wrapper, "retrieval": "orch_stack.retrieve (v2): every entry whose stack hash equals the loaded stack's and whose required vocabulary kinds are present; no lexical trigger filter",
            "entries": core + execs}


if __name__ == "__main__":
    out = O.EXP / "catalog" / "rsi_n20_interface_catalog_v2.json"
    cat = build(); out.write_text(json.dumps(cat, indent=1) + "\n")
    print(out, len(cat["entries"]), hashlib.sha256(out.read_bytes()).hexdigest())
