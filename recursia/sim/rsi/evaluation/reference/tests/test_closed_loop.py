"""Offline tests of the N18C closed-loop driver (frozen N20 namespace with spied executors; scripted mock LLM; no simulator, no model)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "code")); sys.path.insert(0, str(EXP / "tests")); sys.path.insert(0, str(EXP / "examples"))
import closed_loop as CL  # noqa: E402
import test_orch_offline as T  # noqa: E402

LANG2 = T.LANGS["long0_swap"]   # put both the alphabet soup and the tomato sauce in the basket
PLAN2 = [{"relation": "in", "object": "alphabet soup", "reference": "basket", "part": None, "side": None, "object_side": None, "instance": None},
         {"relation": "in", "object": "tomato sauce", "reference": "basket", "part": None, "side": None, "object_side": None, "instance": None}]


def state_of(prompt):
    a = prompt.index("Current state (JSON):\n") + len("Current state (JSON):\n"); b = prompt.index("\n\nMeaning of the fields")
    return json.loads(prompt[a:b])


class MockLLM:
    """tag-dispatched scripted answers; decide() picks the first allowed action matching the preference list."""

    def __init__(self, plans, verdicts=None, prefer=("replace", "retry", "execute", "stop"), decide=None, fail_decide=False):
        self.plans = list(plans); self.verdicts = list(verdicts or []); self.prefer = prefer; self.decide_fn = decide; self.fail_decide = fail_decide
        self.calls = []; self.states = []

    def __call__(self, images, prompt, schema, tag="", max_tokens=0):
        self.calls.append(tag)
        if tag.startswith("cl_plan"):
            return {"status": "ok", "data": {"plan_notes": "x", "subgoals": self.plans.pop(0)}, "cost_usd": 0.07}
        if tag.startswith("cl_verify"):
            return {"status": "ok", "data": {"verdict": self.verdicts.pop(0) if self.verdicts else "yes", "evidence": "e"}, "cost_usd": 0.02}
        if tag.startswith("cl_decide"):
            if self.fail_decide:
                return {"status": "error", "detail": "connection"}
            st = state_of(prompt); self.states.append(st)
            if self.decide_fn:
                return {"status": "ok", "data": self.decide_fn(st), "cost_usd": 0.03}
            for p in self.prefer:
                for a in st["allowed_actions"]:
                    if a["action"] == p:
                        return {"status": "ok", "data": {"action": a["action"], "subgoal": a["subgoal"], "reason": "r"}, "cost_usd": 0.03}
        raise AssertionError(tag)


def place_ok(name, n):
    if name == "run_subgoal":
        return (['PLACE_RESULT {"sg": "sg1", "status": "released", "reason": "arrived_and_released", "shadow_post": "uncertain"}'], (0.40 + 0.05 * n, 0.10))
    return (['CLOSE_VERIFY {"sg": "sg1", "k": 1, "claim": "closed"}'], True)


def run(tmp_path, llm, beh=None, lang=LANG2, replace=False, step0=None, step_per_call=0):
    ns, td, calls = T.instrumented(lang, beh or place_ok)
    if step0 is not None:   # a step counter on the namespace env, advanced by every executor call
        ns["env"]._sim_step_count = step0
        for name in T.EXECUTORS:
            f = ns[name]

            def adv(*a, _f=f, **k):
                ns["env"]._sim_step_count += step_per_call
                return _f(*a, **k)
            adv.original = f.original; ns[name] = adv
    s = CL.orch_closed_loop(ns, diag_dir=tmp_path, query_llm=llm, catalog_path=T.CATALOG, prompt_dir=EXP / "prompts", code_hashes={}, replace_enabled=replace)
    return s, [c[0] for c in calls if c[0] in T.EXECUTORS + ("goto_home_joint_position",)], calls, ns


def test_nominal_closed_loop(tmp_path):
    llm = MockLLM([PLAN2])
    s, ex, _, _ = run(tmp_path, llm)
    assert s["outcome"] == "all_controls_completed" and ex == ["run_subgoal", "goto_home_joint_position", "run_subgoal"]
    assert [d["action"] for d in s["decisions"]] == ["execute", "execute", "stop"]
    assert sum(t.startswith("cl_verify") for t in llm.calls) == 2 and (tmp_path / "orch" / "summary.json").is_file()
    assert s["fallback_to_baseline"] is False and s["method"].startswith("inference method only")


def test_replace_never_offered_when_gate_off(tmp_path):
    llm = MockLLM([PLAN2], verdicts=["no", "no"])
    s, ex, _, _ = run(tmp_path, llm, replace=False)
    assert all(a["action"] != "replace" for st in llm.states for a in st["allowed_actions"])
    assert ex.count("run_subgoal") == 2 and s["outcome"] == "all_controls_completed"


def test_replace_when_gate_on_excludes_own_placement(tmp_path):
    llm = MockLLM([PLAN2], verdicts=["no", "yes", "yes", "yes"])
    s, ex, calls, _ = run(tmp_path, llm, replace=True)
    rs = [c for c in calls if c[0] == "run_subgoal"]
    assert [d["action"] for d in s["decisions"]][:3] == ["execute", "replace", "execute"]
    first_xy, replace_call = rs[0], rs[1]
    assert replace_call[2]["others_xy"] == [] and replace_call[4]["vb_active"] == 1   # own previous spot is not an obstacle; same frozen reference
    assert s["replaced"] == {"sg1": 1} and s["results"]["sg1"]["replace"] is True
    assert sum(1 for st in llm.states for a in st["allowed_actions"] if a["action"] == "replace" and a["subgoal"] == "sg1") == 1   # at most once


def test_step_rule_blocks_recovery_that_would_starve_later_subgoals(tmp_path):
    def beh(name, n):
        if n == 1:
            return (['BINDING_STOP {"status": "ambiguous", "reason": "no motion"}'], None)
        return place_ok(name, n)
    llm = MockLLM([PLAN2], prefer=("retry", "execute", "stop"))
    s, ex, _, _ = run(tmp_path, llm, beh=beh, step0=1800)    # after sg1: 2200 left < 1109 * 2 = 2218 -> no retry while sg2 pending
    assert all(a["action"] != "retry" for a in llm.states[1]["allowed_actions"])
    llm2 = MockLLM([PLAN2], prefer=("retry", "execute", "stop"))
    s2, ex2, _, _ = run(tmp_path / "b", llm2, beh=beh, step0=0)
    assert any(a["action"] == "retry" for a in llm2.states[1]["allowed_actions"]) and s2["decisions"][1]["action"] == "retry"


def test_invalid_decision_is_reasked_once_then_stops(tmp_path):
    llm = MockLLM([PLAN2], decide=lambda st: {"action": "replace", "subgoal": "sg9", "reason": "x"})
    s, ex, _, _ = run(tmp_path, llm)
    assert s["outcome"] == "invalid_decision" and ex == [] and sum(t.startswith("cl_decide") for t in llm.calls) == 2


def test_capability_replan_then_unsupported_without_motion(tmp_path):
    lang = T.LANGS["long9_swap"]
    bad = [{"relation": "in", "object": "yellow and white mug", "reference": "microwave", "part": None, "side": None, "object_side": None, "instance": None},
           {"relation": "close_articulated", "object": None, "reference": None, "part": "microwave", "side": None, "object_side": None, "instance": None}]
    llm = MockLLM([bad, []])
    s, ex, calls, _ = run(tmp_path, llm, lang=lang)
    assert s["outcome"] == "capability_unsupported" and ex == [] and len(s["plan_attempts"]) == 2
    assert not [c for c in calls if c[0] == "vb_resolve_references"]


def test_decision_error_stops(tmp_path):
    s, ex, _, _ = run(tmp_path, MockLLM([PLAN2], fail_decide=True))
    assert s["outcome"] == "decision_error" and ex == []


def test_horizon_writes_summary(tmp_path):
    def beh(name, n):
        raise T._Horizon()
    with pytest.raises(BaseException):
        run(tmp_path, MockLLM([PLAN2]), beh=beh)
    s = json.loads((tmp_path / "orch" / "summary.json").read_text())
    assert s["outcome"] == "episode_timeout_during_orchestration" and s["decisions"][0]["action"] == "execute"


def test_final_recheck_before_stop_can_trigger_replace(tmp_path):
    llm = MockLLM([PLAN2], verdicts=["yes", "yes", "no", "yes", "yes"], prefer=("replace", "execute", "stop"))
    s, ex, _, _ = run(tmp_path, llm, replace=True)
    acts = [d["action"] for d in s["decisions"]]
    assert acts[:3] == ["execute", "execute", "stop"] and "replace" in acts and ex.count("run_subgoal") == 3
    assert [c["subgoal"] for c in s["checks"]][:4] == ["sg1", "sg2", "sg1", "sg2"]


def test_relation_clauses():
    assert CL.relation_clause({"relation": "in", "object": "butter", "support": "basket"}) == "the butter is inside the basket"
    assert CL.relation_clause({"relation": "beside", "object": "chocolate pudding", "support": "plate", "side": "max"}) == "the chocolate pudding is on the table to the right of the plate, not on it"
    assert CL.relation_clause({"relation": "in_sliding", "object": "black bowl", "support": "cabinet", "part": "drawer", "side": "zmin"}) == "the black bowl is inside the bottom drawer of the cabinet"
