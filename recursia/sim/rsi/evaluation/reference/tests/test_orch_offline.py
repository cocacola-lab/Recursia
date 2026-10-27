"""Offline validation of the N18O orchestration patch (interface v2, interpreter boundary) (no simulator, no robot, no model calls).
Run: cd <patch dir> && ../../.venv-libero/bin/python3 -m pytest -q tests/test_orch_offline.py
The frozen N12 definitions are loaded unchanged (orch_offline.load_namespace); only robot I/O and the executors' insides are
stubbed by spies that record every call and then print canned tagged evidence."""
from __future__ import annotations

import ast
import copy
import glob
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "code")); sys.path.insert(0, str(EXP / "examples"))
import orch_check as OC  # noqa: E402
import orch_offline as O  # noqa: E402
import orch_stack as OS  # noqa: E402
import rule_equivalent as RE  # noqa: E402

LANGS = O.runtime_languages()
CATALOG = EXP / "catalog" / "rsi_n20_interface_catalog_v2.json"
EXECUTORS = ("run_subgoal", "run_close_articulated", "run_open_articulated", "run_actuate", "run_rotary")


def _canon(x):
    if isinstance(x, np.ndarray):
        return [_canon(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _obs():
    return {"agentview": {"images": {"rgb": np.zeros((64, 64, 3), np.uint8), "depth": np.ones((64, 64, 1))}, "intrinsics": np.eye(3), "pose_mat": np.eye(4)},
            "robot_cartesian_pos": np.array([0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]), "robot_joint_pos": np.zeros(9)}


SG_KEYS = ("relation", "object", "object_prompts", "object_side", "support", "support_prompts", "part", "part_prompts", "side", "sides", "instance")
N18_KEYS = ("refers", "referent_prompts", "support_track")   # N18 run() semantics, compared for executor calls


def _proj(ns, td, sg, clause=True):
    d = {k: _canon(sg.get(k)) for k in SG_KEYS}
    if clause:   # the clause is read only by la_assembly_prompts and the four part executors, through la_phrase_in(td, "supports", clause)
        d["clause_support"] = ns["la_phrase_in"](td, "supports", sg.get("clause") or "")
        d.update({k: bool(sg.get(k)) if k == "refers" else _canon(sg.get(k)) for k in N18_KEYS})   # refers: its truth value is what the executors read
    return d


def instrumented(lang, behaviour=None):
    """Frozen namespace + spies. behaviour(name, k_call) -> (lines to print, return value)."""
    calls = []
    api = {"get_observation": lambda: (calls.append(("get_observation",)), _obs())[1],
           "goto_home_joint_position": lambda: calls.append(("goto_home_joint_position",)), "gripper_opening": lambda: 0.08}
    ns, td = O.load_namespace(lang, api)
    count = {"n": 0}

    def default(name, n):
        if name == "run_subgoal":
            return (['PLACE_RESULT {"sg": "sg1", "status": "released", "reason": "arrived_and_released", "shadow_post": "uncertain"}',
                     'MF_SHADOW_POST {"sg": "sg1", "status": "uncertain", "claim": null}', "Done!"], (0.40 + 0.05 * n, 0.10))
        return (['CLOSE_VERIFY {"sg": "sg1", "k": 1, "claim": "closed"}'], True)
    beh = behaviour or default
    for name in EXECUTORS:
        orig = ns[name]

        def spy(*a, _name=name, **k):
            count["n"] += 1
            sg = a[0] if _name != "run_subgoal" else None
            calls.append((_name, _canon(a[1:]) if sg is not None else _canon(a), _canon({kk: v for kk, v in k.items()}),
                          None if sg is None else _proj(ns, td, sg), {"vb_active": ns["_VB"]["active"]}))
            lines, ret = beh(_name, count["n"])
            for ln in lines:
                print(ln)
            return ret
        spy.original = orig
        ns[name] = spy
    orig_vb = ns["vb_resolve_references"]

    def vb_spy(subgoals):
        calls.append(("vb_resolve_references", [_proj(ns, td, s, clause=False) for s in subgoals]))   # vb_resolve_references never reads the clause
        return None
    vb_spy.original = orig_vb
    ns["vb_resolve_references"] = vb_spy
    return ns, td, calls


def run_program(ns, td, lang, src, tmp_path):
    rep = OC.check_program(src, vocab={"objects": sorted(td["objects"]), "supports": sorted(td["supports"]), "articulated": sorted(td["articulated"]),
                                       "side_words": td["side_words"]}, supported=ns["LA_SUPPORTED"], object_sides=ns["LA_OBJECT_SIDES"], part_relations=ns["LA_PART_RELATIONS"])
    if not rep["ok"]:
        return rep, None, None
    st = OS.RSIStack(ns, td, lang, tmp_path)
    res = OS.execute_program(src, st, {"instruction": lang})
    return rep, res, st


# ---------------------------------------------------------------------------------------------------- 1. runtime branching
def _scenario(kind):
    def beh(name, n):
        if name != "run_subgoal":
            return (['CLOSE_VERIFY {"sg": "sg2", "k": 1, "claim": "closed"}'], True)
        if kind == "nominal":
            return (['PLACE_RESULT {"sg": "sg1", "status": "released", "reason": "arrived_and_released", "shadow_post": "uncertain"}'], (0.4, 0.1))
        if kind == "binding_then_ok":
            if n == 1:
                return (['BINDING_STOP {"status": "ambiguous", "reason": "target not bound unambiguously; no motion"}'], None)
            return (['PLACE_RESULT {"sg": "sg1", "status": "released", "reason": "arrived_and_released", "shadow_post": "pass"}'], (0.4, 0.1))
        if kind == "retained":
            return (["GRASP_VERIFY {}", 'PLACE_RESULT {"sg": "sg1", "status": "failed", "reason": "release_blocked: descent_not_reached"}', "Done!"], (0.4, 0.1))
        raise AssertionError(kind)
    return beh


def test_same_program_takes_different_branches(tmp_path):
    lang = LANGS["long3_swap"]; src = (EXP / "examples" / "long3_drawer_then_close.py").read_text()
    seqs = {}
    for kind in ("nominal", "binding_then_ok", "retained"):
        ns, td, calls = instrumented(lang, _scenario(kind))
        d = tmp_path / kind; d.mkdir()
        rep, res, st = run_program(ns, td, lang, src, d)
        assert rep["ok"], rep["errors"]; assert res["status"] == "returned", res
        seqs[kind] = ([c["method"] + ("(retry)" if (c["args"] or {}).get("retry") else "") + ":" + str((c["args"] or {}).get("action", (c["result"] or {}).get("status"))) for c in st._calls], st._finish,
                      [c[0] for c in calls if c[0] in EXECUTORS + ("goto_home_joint_position",)])
    nominal, retry, retained = seqs["nominal"], seqs["binding_then_ok"], seqs["retained"]
    assert nominal[2] == ["run_subgoal", "goto_home_joint_position", "run_close_articulated"]
    assert retry[2] == ["run_subgoal", "goto_home_joint_position", "run_subgoal", "goto_home_joint_position", "run_close_articulated"]   # one bounded retry
    assert "execute(retry):control_completed" in retry[0] and "execute:binding_failed" in retry[0]
    assert retained[2] == ["run_subgoal", "goto_home_joint_position"] and retained[1]["declared_status"] == "stopped"   # no retry, no close
    assert nominal[1]["declared_status"] == "controls_done" and len({tuple(s[0]) for s in seqs.values()}) == 3
    (tmp_path / "branches.json").write_text(json.dumps(seqs, indent=1, default=str))


def test_structured_result_semantics(tmp_path):
    ns, td, calls = instrumented(LANGS["long3_swap"], _scenario("retained"))
    rep, res, st = run_program(ns, td, LANGS["long3_swap"], (EXP / "examples" / "long3_drawer_then_close.py").read_text(), tmp_path)
    r = st._results["sg1"]
    assert r["status"] == "verification_failed" and r["failure_reason"].startswith("release_blocked")
    assert r["object_possibly_retained"] is True and "retry_subgoal" not in r["allowed_recovery_actions"]
    assert r["verification_status"] == "gate_refused"
    ns2, td2, _ = instrumented(LANGS["long3_swap"], _scenario("nominal"))
    rep2, res2, st2 = run_program(ns2, td2, LANGS["long3_swap"], (EXP / "examples" / "long3_drawer_then_close.py").read_text(), tmp_path / "n" if (tmp_path / "n").mkdir() is None else None)
    r1, r2 = st2._results["sg1"], st2._results["sg2"]
    assert r1["status"] == "control_completed" and r1["verification_status"] == "shadow_recorded_only"   # released != verified
    assert r2["status"] == "control_completed" and r2["verification_status"] == "stack_verifier_claim_positive"


# ---------------------------------------------------------------------------------------------------- 2. rejection
BAD = {
    "import": "import os\ndef run(context, stack):\n    os.system('x')\n",
    "open_file": "def run(context, stack):\n    open('/etc/passwd').read()\n",
    "while": "def run(context, stack):\n    while True:\n        stack.observe()\n",
    "range_var": "def run(context, stack):\n    n = 5\n    for i in range(n):\n        stack.observe()\n",
    "range_big": "def run(context, stack):\n    for i in range(1000):\n        stack.observe()\n",
    "comprehension_big": "def run(context, stack):\n    x = [i for i in range(100000)]\n",
    "recursion": "def f(stack):\n    return f(stack)\ndef run(context, stack):\n    f(stack)\n",
    "private": "def run(context, stack):\n    stack._ns['move_to_pose']([0,0,0])\n",
    "dunder": "def run(context, stack):\n    stack.__class__\n",
    "getattr": "def run(context, stack):\n    getattr(stack, 'execute')\n",
    "robot_api": "def run(context, stack):\n    move_to_pose([0.5, 0, 0.3], [0, 1, 0, 0])\n",
    "observation_api": "def run(context, stack):\n    get_observation()\n",
    "stack_assign": "def run(context, stack):\n    stack.execute = None\n",
    "unknown_method": "def run(context, stack):\n    stack.move_to_pose([0, 0, 0])\n",
    "bare_except": "def run(context, stack):\n    try:\n        stack.observe()\n    except:\n        pass\n",
    "grow_loop": "def run(context, stack):\n    xs = [1]\n    for x in xs:\n        xs.append(x)\n",
    "top_level_code": "print('x')\ndef run(context, stack):\n    pass\n",
    "eval": "def run(context, stack):\n    eval('1')\n",
    "unsupported_relation": "def run(context, stack):\n    s = stack.subgoal('in_hinged', object='black bowl', part='drawer')\n",
    "unsupported_side": "def run(context, stack):\n    s = stack.subgoal('on', object='black bowl', reference='cabinet', side='back')\n",
    "unknown_phrase": "def run(context, stack):\n    s = stack.subgoal('in', object='banana', reference='basket')\n",
    "no_entry": "def main(context, stack):\n    pass\n",
    "part_as_container": "def run(context, stack):\n    s = stack.subgoal('in', object='black bowl', reference='drawer')\n",
    "dict_get": "def run(context, stack):\n    r = stack.observe()\n    x = r.get('status')\n",
    "aug_assign": "def run(context, stack):\n    n = 0\n    n += 1\n",
    "stack_escape": "def run(context, stack):\n    s = stack\n",
    "bad_recovery": "def run(context, stack):\n    r = stack.observe()\n    stack.recover(r, 'teleport')\n",
    "helper": "def run(context, stack):\n    pass\ndef helper(stack):\n    pass\n",
}
EXPECT = {"import": "top_level", "open_file": "unauthorized_call", "while": "forbidden_construct", "range_var": "unbounded_loop",
          "range_big": "unbounded_loop", "comprehension_big": "unbounded_loop", "recursion": "top_level", "private": "attribute",
          "dunder": "attribute", "getattr": "unauthorized_call", "robot_api": "unauthorized_call", "observation_api": "unauthorized_call",
          "stack_assign": "assignment_target", "unknown_method": "stack_method", "bare_except": "forbidden_construct", "grow_loop": "attribute",
          "top_level_code": "top_level", "eval": "unauthorized_call", "dict_get": "attribute", "aug_assign": "forbidden_construct",
          "stack_escape": "stack_reference_escape", "bad_recovery": "capability_recovery", "helper": "top_level", "unsupported_relation": "capability_relation", "unsupported_side": "capability_side",
          "unknown_phrase": "capability_vocabulary", "no_entry": "entry",
          "part_as_container": "capability_reference_kind"}


@pytest.mark.parametrize("name", sorted(BAD))
def test_static_rejection(name):
    ns, td, _ = instrumented(LANGS["long3_swap"])
    rep = OC.check_program(BAD[name], vocab={"objects": sorted(td["objects"]), "supports": sorted(td["supports"]), "articulated": sorted(td["articulated"]),
                                             "side_words": td["side_words"]}, supported=ns["LA_SUPPORTED"], object_sides=ns["LA_OBJECT_SIDES"], part_relations=ns["LA_PART_RELATIONS"],
                           recovery_actions=OS.RECOVERY_ACTIONS)
    assert not rep["ok"] and EXPECT[name] in {e["kind"] for e in rep["errors"]}, rep["errors"]


def test_interpreter_budget_and_no_exec(tmp_path):
    ns, td, calls = instrumented(LANGS["long3_swap"])
    src = ("def run(context, stack):\n    n = 0\n    for a in range(8):\n        for b in range(8):\n            for c in range(8):\n"
           "                for d in range(8):\n                    n = 'x'\n    return n\n")
    rep, res, st = run_program(ns, td, LANGS["long3_swap"], src, tmp_path)
    assert rep["ok"] and res["status"] == "budget_exceeded"
    st2 = OS.RSIStack(ns, td, LANGS["long3_swap"], tmp_path)   # the interpreter refuses unvalidated source outright: nothing is exec'd
    assert OS.execute_program("import os\ndef run(context, stack):\n    return 1\n", st2, {})["status"] == "rejected"
    assert OS.execute_program("def run(context, stack):\n    return open('/etc/hostname')\n", st2, {})["status"] == "rejected"
    r = OS.execute_program("def run(context, stack):\n    o = stack.observe()\n    return o['no_such_key']\n", st2, {})
    assert r["status"] == "runtime_error" and "no_such_key" in r["detail"]   # a program error ends orchestration, never the episode
    import orch_check as _oc
    assert "exec(" not in Path(_oc.__file__).read_text().split("class Interpreter")[1]


def test_runtime_capability_and_protocol_refusals(tmp_path):
    ns, td, calls = instrumented(LANGS["long9_swap"])
    src = ("def run(context, stack):\n    rel = 'in_' + 'hinged'\n    a = stack.subgoal(rel, object='yellow and white mug', reference='microwave')\n"
           "    b = stack.subgoal('close_articulated', part='microwave')\n    p = stack.plan([a, b])\n    r = stack.execute(b)\n"
           "    stack.finish(p['status'], r['failure_reason'])\n")
    rep, res, st = run_program(ns, td, LANGS["long9_swap"], src, tmp_path)
    assert rep["ok"]   # the relation is a runtime value: the static check cannot see it ...
    plan = next(c for c in st._calls if c["method"] == "plan")
    assert plan["result"]["status"] == "unsupported" and plan["result"]["unsupported"][0]["why"] == "unknown_subgoal"   # ... the runtime check refused subgoal a
    assert st._calls[0]["result"]["status"] == "unsupported" and st._calls[0]["result"]["why"] in ("relation", "reference_kind")
    assert not [c for c in calls if c[0] in EXECUTORS + ("vb_resolve_references",)]   # nothing moved, no references taken
    assert st._finish["reason"].startswith("no_plan")
    ns, td, calls = instrumented(LANGS["long9_swap"])
    src = ("def run(context, stack):\n    rel = 'i' + 'n'\n    a = stack.subgoal(rel, object='white mug', reference='microwave')\n"
           "    stack.finish(a['status'], a['why'])\n")
    rep, res, st = run_program(ns, td, LANGS["long9_swap"], src, tmp_path)
    assert rep["ok"] and st._finish == {"declared_status": "unsupported", "reason": "reference_kind"}
    ns, td, calls = instrumented(LANGS["long2_swap"])
    src = ("def run(context, stack):\n    a = stack.subgoal('on', object='moka pot', reference='stove')\n    p = stack.plan([a])\n"
           "    r1 = stack.execute(a)\n    r2 = stack.execute(a)\n    r3 = stack.recover(r1, 'retry_subgoal')\n    r4 = stack.recover(r1, 'tele' + 'port')\n"
           "    stack.finish('x', r2['failure_reason'] + '|' + r3['failure_reason'] + '|' + r4['failure_reason'])\n")
    rep, res, st = run_program(ns, td, LANGS["long2_swap"], src, tmp_path)
    assert st._finish["reason"] == "already_control_completed|retry_not_allowed_by_evidence|unsupported_recovery_action"
    assert [c[0] for c in calls if c[0] in EXECUTORS] == ["run_subgoal"]


# ---------------------------------------------------------------------------------------------------- 3. frozen implementation
def test_interfaces_invoke_frozen_n20(tmp_path):
    src = O.stack_source(); tree = ast.parse(src)
    lines = {n.name: n.lineno for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert hashlib.sha256(O.N12_PATH.read_bytes()).hexdigest() == OS.STACK_SHA256
    ns, td, calls = instrumented(LANGS["long2_swap"])
    for name in EXECUTORS + ("vb_resolve_references",):
        f = ns[name].original
        assert f.__code__.co_filename == str(O.N12_PATH) and f.__code__.co_firstlineno == lines[name], name
    for name in ("la_dispatch", "la_reference_prompts", "la_assembly_prompts", "localize_target", "tcp_from_obs", "la_summary"):
        assert ns[name].__code__.co_filename == str(O.N12_PATH) and ns[name].__code__.co_firstlineno == lines[name]
    cat = json.loads(CATALOG.read_text())
    for e in cat["entries"]:
        assert e["stack_sha256"] == OS.STACK_SHA256
        for fn, h in (e["implementation"].get("frozen_functions") or {}).items():
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
            assert hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest() == h, (e["id"], fn)
    prog = (EXP / "programs" / "N20C.py").read_text()
    stack_part = prog[: prog.index("\n\n# ===== N20C orchestration entry")] + "\n\nrun()\n"
    assert hashlib.sha256(stack_part.encode()).hexdigest() == OS.STACK_SHA256


# ---------------------------------------------------------------------------------------------------- 4. behaviour preservation
@pytest.mark.parametrize("short", sorted(LANGS))
def test_rule_equivalent_program_matches_baseline_driver(short, tmp_path, capsys):
    lang = LANGS[short]
    ns_b, td_b, calls_b = instrumented(lang)
    ns_b["run"]()   # baseline driver: la_parse -> la_dispatch -> vb_resolve_references -> executors
    ns_o, td_o, calls_o = instrumented(lang)
    sgs, mode = ns_o["la_parse"](td_o, lang)
    src = RE.program_for(sgs, td_o)
    rep, res, st = run_program(ns_o, td_o, lang, src, tmp_path)
    if not rep["ok"]:   # the static check refuses what la_dispatch refuses; the baseline must then have moved nothing either
        assert {e["kind"] for e in rep["errors"]} <= {"capability_relation", "capability_side", "capability_object_side", "capability_vocabulary"}
        assert not [c for c in calls_b if c[0] in EXECUTORS + ("vb_resolve_references", "goto_home_joint_position")]
        return
    assert res["status"] == "returned"
    keep = EXECUTORS + ("vb_resolve_references", "goto_home_joint_position", "get_observation")
    assert [c for c in calls_b if c[0] in keep] == [c for c in calls_o if c[0] in keep]
    for k in ("_VB", "_E1_LAST", "_CUR", "_BIND"):   # shared episode state left identical by both drivers
        assert json.dumps(_canon(ns_b[k]), sort_keys=True, default=repr) == json.dumps(_canon(ns_o[k]), sort_keys=True, default=repr), k


def test_wrappers_keep_gates_and_shadow_settings(tmp_path):
    lang = LANGS["long6_swap"]
    ns, td, calls = instrumented(lang)

    def snap():
        out = {}
        for k, v in ns.items():
            if (k.isupper() and not k.startswith("_")) or k == "_FCFG":   # configuration: thresholds, tables, modes (episode state excluded)
                try:
                    out[k] = json.dumps(_canon(v), sort_keys=True, default=repr)
                except Exception:  # noqa: BLE001
                    out[k] = repr(v)
        return out
    before = snap()
    sgs, _ = ns["la_parse"](td, lang)
    rep, res, st = run_program(ns, td, lang, RE.program_for(sgs, td), tmp_path)
    assert res["status"] == "returned"
    after = snap()
    assert before == after
    assert ns["PRIM"]["shadow_verifier"] is True and ns["_FCFG"]["enforce"] is False and ns["_FCFG"]["corrections"] is False and ns["VB_VETO_TEXT_BINDING"] is False
    for k in ("run_subgoal", "prim_place", "mf_post_release", "vb_post_grasp", "attempt_grasp_verify"):
        if k not in EXECUTORS:
            assert ns[k].__code__.co_filename == str(O.N12_PATH)


def test_real_n12_logs_classify_without_upgrading_shadow(tmp_path):
    """Recorded stdout of the 100 real N18 interim episodes, cut at SUBGOAL_STATUS, through the object-subgoal classifier."""
    base = EXP.parent / "rsi_libero90_N18_interim_long100_20261012/stage_interim_N18_long100/attempts"
    rows = []
    for d in sorted(glob.glob(str(base / "*/N18/*/seed_*"))):
        m = json.loads((Path(d) / "attempt_manifest.json").read_text())
        so = glob.glob(d + "/artifacts/trial_*/stdout.txt")
        if not so:
            continue
        seg = []
        for line in Path(so[0]).read_text().splitlines():
            seg.append(line)
            if line.startswith("SUBGOAL_STATUS "):
                s = json.loads(line.split(" ", 1)[1])
                if s["relation"] not in ("close_articulated", "open_articulated", "actuate_on", "actuate_off"):
                    ev = OS.parse_evidence(seg)
                    status, reason, retained = OS.classify_object_result(ev, (0.0, 0.0) if s["placed"] else None)
                    ver = OS.verification_record(ev, s["relation"], False)
                    rows.append({"task": m["task_short"], "seed": m["seed"], "k": s["k"], "official": m["official_task_completed"], "status": status,
                                 "reason": reason, "retained": retained, "verification_status": ver["verification_status"],
                                 "shadow": [v["status"] for v in ver["shadow_verdicts"] if v["source"] == "MF_SHADOW_POST"]})
                seg = []
    assert len(rows) > 50
    assert all(r["verification_status"] != "verified" for r in rows)
    assert all(r["verification_status"] == "shadow_recorded_only" for r in rows if r["status"] == "control_completed")
    out = EXP / "tests" / "real_log_classification.json"
    from collections import Counter
    summ = {"n_object_subgoals": len(rows), "status": Counter(r["status"] for r in rows), "verification_status": Counter(r["verification_status"] for r in rows),
            "control_completed_in_failed_episodes": sum(1 for r in rows if r["status"] == "control_completed" and r["official"] == 0),
            "unknown": [r for r in rows if r["status"] == "unknown"][:10], "rows": rows}
    out.write_text(json.dumps(summ, indent=1, default=str))
    assert summ["control_completed_in_failed_episodes"] > 0   # control completion is not task success


# ---------------------------------------------------------------------------------------------------- 5. retrieval / prompt
def test_retrieval_is_version_pinned_and_capability_aware():
    cat = OS.load_catalog(CATALOG); template = (EXP / "prompts" / "orch_prompt_v2.md").read_text(); out = {}
    for short, lang in sorted(LANGS.items()):
        ns, td = O.load_namespace(lang)
        picked, skipped = OS.retrieve(cat, td, lang)
        ids = [e["id"] for e in picked]
        assert all(e["stack_sha256"] == OS.STACK_SHA256 for e in picked)
        assert {"core.program", "core.subgoal_check_plan", "core.execute_result"} <= set(ids)
        sgs, _ = ns["la_parse"](td, lang)
        for sg in sgs:   # every relation the baseline parse uses (and the stack implements) is retrieved
            if ns["la_dispatch"](sg)[0]:
                assert any(sg["relation"] in e.get("relations", ()) for e in picked), (short, sg["relation"], ids)
        prompt = OS.build_prompt(template, lang, {"objects": sorted(td["objects"])}, OS.capability_table(ns, td), picked)
        out[short] = {"retrieved": ids, "skipped": [s["id"] + ":" + s["why"] for s in skipped], "prompt_chars": len(prompt)}
    bad = copy.deepcopy(cat); bad["entries"][-1]["stack_sha256"] = "0" * 64
    picked, skipped = OS.retrieve(bad, O.descriptors_for(LANGS["long2_swap"]), LANGS["long2_swap"])
    assert any(s["why"] == "other_stack_version" for s in skipped)
    bad["stack"]["sha256"] = "0" * 64; p = Path("/tmp/_bad_catalog.json"); p.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        OS.load_catalog(p)
    (EXP / "tests" / "retrieval_report.json").write_text(json.dumps(out, indent=1))


def test_orch_main_offline_generation_paths(tmp_path):
    """orch_main end to end with a scripted generator: generation error, rejected program, and an accepted program. No fallback."""
    lang = LANGS["long2_swap"]
    good = ("def run(context, stack):\n    a = stack.subgoal('actuate_on', reference='stove')\n    b = stack.subgoal('on', object='moka pot', reference='stove')\n"
            "    p = stack.plan([a, b])\n    if p['status'] != 'ok':\n        stack.finish('unsupported', 'plan')\n        return\n"
            "    for sg in [a, b]:\n        r = stack.execute(sg)\n        if r['status'] != 'control_completed' and 'retry_subgoal' in r['allowed_recovery_actions']:\n"
            "            r = stack.recover(r, 'retry_subgoal')\n    stack.finish('done', 'ok')\n")
    cases = {"error": {"status": "error", "detail": "connection"}, "rejected": {"status": "ok", "data": {"plan_notes": "x", "program": "import os\ndef run(context, stack):\n    pass\n"}},
             "accepted": {"status": "ok", "data": {"plan_notes": "turn on then place", "program": good}, "cost_usd": 0.05, "usage": {"input_tokens": 1}}}
    outcomes = {}
    for name, resp in cases.items():
        ns, td, calls = instrumented(lang)
        d = tmp_path / name; d.mkdir()
        s = OS.orch_main(ns, diag_dir=d, query_llm=lambda *a, _r=resp, **k: dict(_r), catalog_path=CATALOG, prompt_path=EXP / "prompts" / "orch_prompt_v2.md", code_hashes={})
        outcomes[name] = (s["outcome"], [c[0] for c in calls if c[0] in EXECUTORS])
        assert (d / "orch" / "summary.json").is_file() and (d / "orch" / "generation_context.json").is_file()
        assert s["fallback_to_baseline"] is False
    assert outcomes["error"] == ("generation_error", []) and outcomes["rejected"] == ("program_rejected", [])
    assert outcomes["accepted"] == ("all_controls_completed", ["run_rotary", "run_subgoal"])   # N12 dispatch: actuate_on -> run_rotary


def test_phrase_side_words_follow_la_subgoal():
    """A side word inside a vocabulary phrase is that phrase's own side code (la_subgoal rule (a)), with no explicit side argument."""
    lang = LANGS["long8_task"]; ns, td = O.load_namespace(lang)
    rec, why = OS.build_subgoal_record(td, lang, "on", object="left moka pot", reference="stove", part_relations=ns["LA_PART_RELATIONS"])
    base = ns["la_parse"](td, lang)[0][0]
    assert why is None and {k: rec[k] for k in SG_KEYS} == {k: base[k] for k in SG_KEYS}
    assert rec["object_side"] == "min" and ns["la_dispatch"](rec) == (True, None)


class _Horizon(BaseException):
    pass


_Horizon.__name__ = "EpisodeHorizonReached"


@pytest.mark.parametrize("kind", ["horizon", "stack_exception"])
def test_summary_written_when_execution_is_interrupted(kind, tmp_path):
    """v2 fix of N12O defect 1/3: the episode summary is written on the horizon and on a frozen-executor exception, which propagate."""
    lang = LANGS["long2_swap"]

    def beh(name, n):
        if name == "run_subgoal":
            if kind == "horizon":
                raise _Horizon()
            raise TypeError("'NoneType' object is not subscriptable")
        return (['ACT_VERIFY {"sg": "sg1", "k": 1, "claim": "actuated"}'], True)
    ns, td, calls = instrumented(lang, beh)
    good = ("def run(context, stack):\n    a = stack.subgoal('actuate_on', reference='stove')\n    b = stack.subgoal('on', object='moka pot', reference='stove')\n"
            "    p = stack.plan([a, b])\n    for sg in [a, b]:\n        r = stack.execute(sg)\n    stack.finish('done', 'x')\n")
    resp = {"status": "ok", "data": {"plan_notes": "x", "program": good}, "cost_usd": 0.05}
    with pytest.raises(BaseException) as ei:
        OS.orch_main(ns, diag_dir=tmp_path, query_llm=lambda *a, **k: dict(resp), catalog_path=CATALOG, prompt_path=EXP / "prompts" / "orch_prompt_v2.md", code_hashes={})
    s = json.loads((tmp_path / "orch" / "summary.json").read_text())
    if kind == "horizon":
        assert type(ei.value).__name__ == "EpisodeHorizonReached" and s["outcome"] == "episode_timeout_during_orchestration"
    else:
        assert isinstance(ei.value, TypeError) and s["outcome"] == "stack_exception" and s["results"]["sg2"]["status"] == "stack_exception"
    assert s["results"]["sg1"]["status"] == "control_completed" and s["program_sha256"]


def test_no_long_pro_instruction_uses_carry_tracking():
    """N18's C17I2 carry ordering (support_track) is not reproduced by the interface; confirm la_parse never produces it here."""
    for short, lang in LANGS.items():
        ns, td = O.load_namespace(lang)
        sgs, _ = ns["la_parse"](td, lang)
        assert not any(s.get("support_track") for s in sgs), short
