"""Generic RSI harness tests (no simulator, no GPU, no proxy)."""
import copy
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]
REF = HARNESS.parent / "reference"
sys.path.insert(0, str(HARNESS))
import rsi_lib as L  # noqa: E402

TASKS = json.loads((REF / "task_manifest.json").read_text())["tasks"]
GROUP5 = ["s07_pick_up_the_alphabet_soup_an", "s09_put_the_black_bowl_in_the_bo", "s12_pick_up_the_book_on_the_righ",
          "s13_open_the_top_drawer_of_the_c", "s16_pick_up_the_milk_and_put_it"]


def targets(groups, k=1):
    return {"iteration": k, "first_pass_order": list(groups),
            "groups": {g: {"tasks": ts, "failure_classes": ["grasp"], "mechanism_allowed": True, "vb_changes_allowed": False, "pairs": []}
                       for g, ts in groups.items()}}


# ----------------------------------------------------------------------------- invariants and validation
def test_reference_manifest_is_source_only():
    L.assert_no_longpro_manifest(TASKS)
    assert len(L.libero90_tasks(TASKS)) == 90
    assert len(L.synthetic_tasks(TASKS, "synthetic_dev")) == 10 and len(L.synthetic_tasks(TASKS, "synthetic_regression")) == 8
    assert all(t in TASKS for t in L.PANEL_11)


def test_seed_pools_disjoint_and_exclude_eval():
    assert not set(L.DEV_POOL) & set(L.HELD_POOL)
    assert not (set(L.DEV_POOL) | set(L.HELD_POOL)) & set(L.EVAL_SEEDS)


def test_default_and_warm_strategies_valid():
    for s in (L.default_strategy_s1("N0"), L.warm_strategy_n17_n20("N0")):
        assert L.validate_strategy(s, TASKS, 1, "N0") == []


@pytest.mark.parametrize("mutate,expect", [
    (lambda s: s["seeds"].__setitem__("dev", [1, 2, 3]), "seeds.dev"),
    (lambda s: s["seeds"].__setitem__("selection", [86, 87]), "seeds.selection"),
    (lambda s: s["debug_data_roles"]["proposer_visible"].append("selection"), "proposer_visible"),
    (lambda s: s["gate"].__setitem__("timeout_rate_max", 0.05), "timeout_rate_max"),
    (lambda s: s["gate"]["outcome"].__setitem__("selection_net_min", 0), "selection_net_min"),
    (lambda s: s["pairing"].__setitem__("mode", "none"), "pairing.mode"),
    (lambda s: s.__setitem__("parent", "N3"), "parent"),
])
def test_strategy_violations(mutate, expect):
    s = L.default_strategy_s1("N0"); mutate(s)
    errs = L.validate_strategy(s, TASKS, 1, "N0")
    assert any(expect in e for e in errs), errs


def test_targets_validation():
    s = L.default_strategy_s1("N0")
    assert L.validate_targets(targets({"A": GROUP5}), s, TASKS, 1) == []
    assert L.validate_targets(targets({"A": GROUP5 + ["s01_pick_up_the_cream_cheese_box"]}), s, TASKS, 1)   # > 5 tasks
    assert L.validate_targets(targets({"A": ["syn01_kitchen4_on_then_in"]}), s, TASKS, 1)                  # synthetic is not a target
    w = L.warm_strategy_n17_n20("N0")
    assert L.validate_targets(targets({"A": GROUP5[:2], "B": GROUP5[1:3]}), w, TASKS, 1)                  # overlap


# ----------------------------------------------------------------------------- plans
def test_s1_rows_match_the_seed_spec():
    s = L.default_strategy_s1("N0"); st = L.new_state()
    rows = L.arm_rows("N0", s, targets({"A": GROUP5}), "A", TASKS, st)
    from collections import Counter
    c = Counter(r["role"] for r in rows)
    assert c == {"dev": 50, "selection": 50, "synthetic_dev": 30, "synthetic_selection": 30, "ordinary_regression": 33, "synthetic_regression": 16}
    assert {r["seed"] for r in rows if r["role"] == "dev"} == set(range(86, 96))
    assert {r["seed"] for r in rows if r["role"] == "selection"} == set(range(76, 86))


def test_panel_task_in_group_is_swapped_out():
    s = L.default_strategy_s1("N0")
    rows = L.arm_rows("N0", s, targets({"A": ["s01_pick_up_the_cream_cheese_box"]}), "A", TASKS, L.new_state())
    assert not [r for r in rows if r["role"] == "ordinary_regression" and r["task_short"] == "s01_pick_up_the_cream_cheese_box"]
    assert sum(r["role"] == "ordinary_regression" for r in rows) == 30


def test_guard_panel_uses_accepted_tasks():
    w = L.warm_strategy_n17_n20("N0"); st = L.new_state()
    st["accepted"].append({"arm": "C1I1", "tasks": ["s55_pick_up_the_butter_and_put_i", "s07_pick_up_the_alphabet_soup_an"]})
    rows = L.arm_rows("N0", w, targets({"A": ["s07_pick_up_the_alphabet_soup_an"]}), "A", TASKS, st)
    assert {r["task_short"] for r in rows if r["role"] == "guard"} == {"s55_pick_up_the_butter_and_put_i"}


def test_eval_seed_rows_refused():
    with pytest.raises(L.HarnessError):
        L.assert_rows_source_only([{"arm": "N0", "task_short": GROUP5[0], "seed": 7, "role": "dev"}], TASKS)


# ----------------------------------------------------------------------------- compare
def blk(spec):
    return {t: {s: {"official": o, "state_idx": (s - 1) % 50, "first_failure": "x"} for s, o in v.items()} for t, v in spec.items()}


def test_gains_never_cancel_losses():
    p = blk({"a": {1: 0, 2: 0, 3: 1}, "b": {1: 1, 2: 1}})
    c = blk({"a": {1: 1, 2: 1, 3: 1}, "b": {1: 0, 2: 0}})
    r = L.compare(p, c, cap=5, per_task_cap=1)
    assert len(r["gained"]) == 2 and len(r["lost"]) == 2 and not r["ok"]   # b lost 2 > per-task cap
    r = L.compare(p, c, cap=1, per_task_cap=2)
    assert not r["ok"]                                                       # 2 losses > role cap 1


def test_exemption_removes_counted_loss():
    p = blk({"b": {1: 1}}); c = blk({"b": {1: 0}})
    assert not L.compare(p, c, 0, 1)["ok"]
    r = L.compare(p, c, 0, 1, exempt=frozenset({("b", 1)}))
    assert r["ok"] and r["exempt"] == [("b", 1)]


# ----------------------------------------------------------------------------- end-to-end gate on a fake round
def fake_round(tmp_path, strategy, groups, parent="N0"):
    R = tmp_path / "R"
    for d in ("programs", "results", "proposals", "logs", "iter_1"):
        (R / d).mkdir(parents=True)
    (R / "task_manifest.json").write_text(json.dumps({"tasks": TASKS}))
    (R / "authorization.json").write_text((HARNESS / "authorization.template.json").read_text())
    rnd = L.Round(R); rnd.save_state(L.new_state(parent))
    L.write_json(R / "iter_1/strategy.json", strategy); L.write_json(R / "iter_1/targets.json", targets(groups))
    (R / "iter_1/STRATEGY_UPDATE.md").write_text("S_1, initial strategy.\n")
    L.freeze_strategy(rnd, 1)
    return rnd


def put_phase(rnd, phase, kind, rows, arm, parent, group, outcome):
    d = rnd.results(phase); (d / "evaluator_only").mkdir(parents=True)
    out = [dict(r, official=outcome(r), state_idx=(r["seed"] - 1) % 50, first_failure="" if outcome(r) else "grasp_descent:failed",
                stop=None, key_lines=[], episode_dir=f"stage_{phase}/x") for r in rows]
    (d / "evaluator_only/rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    L.write_json(d / "integrity.json", {"eligible": True})
    L.write_json(rnd.p(f"plan_{phase}_roles.json"), {"phase": phase, "kind": kind, "iteration": 1, "arm": arm, "parent": parent, "group": group, "rows": rows})


def put_candidate_files(rnd, phase, arm, parent, group, mode="outcome"):
    d = rnd.p("proposals", phase); d.mkdir(parents=True)
    L.write_json(d / "predeclared.json", {"arm": arm, "parent": parent, "group": group, "mode": mode, "named_tasks": GROUP5[:2],
                                          "failure_classes": ["grasp_descent"]})
    L.write_json(d / "build.json", {"iteration": 1, "arm": arm, "parent": parent, "group": group, "runtime_markers": [], "program_norm_sha": "x"})


def test_gate_row_store_pass_and_panel_loss(tmp_path):
    w = L.warm_strategy_n17_n20("N0"); w["debug_data_roles"]["guard_panel"]["enabled"] = False
    w["debug_data_roles"]["evaluator_only"].remove("guard"); w["gate"]["coverage_gate_before_staging"] = False
    rnd = fake_round(tmp_path, w, {"A": GROUP5[:2]})
    st = rnd.state(); s, t = rnd.strategy(1), rnd.targets(1)
    prow = L.arm_rows("N0", s, t, "A", TASKS, st)
    put_phase(rnd, "refresh1_N0_A", "refresh", prow, "N0", "N0", "A", lambda r: int(r["role"] not in ("dev", "selection")))
    crow = L.arm_rows("C1I1", s, t, "A", TASKS, st)
    gain = lambda r: int(r["role"] not in ("dev", "selection") or r["seed"] in (86, 87, 76))   # dev +4, selection +2
    put_phase(rnd, "chunk1_iter1", "candidate", crow, "C1I1", "N0", "A", gain)
    put_candidate_files(rnd, "chunk1_iter1", "C1I1", "N0", "A")
    rep = L.evaluate_gate(rnd, "chunk1_iter1")
    assert rep["passes_numeric_gates"] and rep["accepted_by"] == "outcome" and rep["group_dev_net"] == 4 and rep["group_selection_net"] == 2
    # same candidate, but one ordinary-regression loss on top of the gains -> rejected (gains never cancel losses)
    lose = lambda r: 0 if (r["role"] == "ordinary_regression" and r["seed"] == 76 and r["task_short"] == L.PANEL_11[0]) else gain(r)
    put_phase(rnd, "chunk1_iter2", "candidate", L.arm_rows("C1I2", s, t, "A", TASKS, st), "C1I2", "N0", "A", lose)
    put_candidate_files(rnd, "chunk1_iter2", "C1I2", "N0", "A")
    w2 = copy.deepcopy(rnd.strategy(1)); assert w2["gate"]["loss_caps"]["ordinary_regression"] == 1
    rep2 = L.evaluate_gate(rnd, "chunk1_iter2")
    assert rep2["passes_numeric_gates"]   # 1 loss == cap 1 is allowed
    assert rep2["roles"]["ordinary_regression"]["lost"] == [(L.PANEL_11[0], 76)] or rep2["roles"]["ordinary_regression"]["lost"] == [[L.PANEL_11[0], 76]]
    with pytest.raises(L.HarnessError):
        L.evaluate_gate(rnd, "chunk1_iter2")   # gate.json is write-once


def test_gate_full_rerun_pairs_within_the_stage(tmp_path):
    s1 = L.default_strategy_s1("N0")
    rnd = fake_round(tmp_path, s1, {"A": GROUP5[:2]})
    st = rnd.state(); s, t = rnd.strategy(1), rnd.targets(1)
    rows = L.arm_rows("N0", s, t, "A", TASKS, st) + L.arm_rows("C1I1", s, t, "A", TASKS, st)
    out = lambda r: int(r["role"] not in ("dev", "selection") or (r["arm"] == "C1I1" and r["seed"] in (86, 87, 76)))
    put_phase(rnd, "chunk1_iter1", "candidate", rows, "C1I1", "N0", "A", out)
    put_candidate_files(rnd, "chunk1_iter1", "C1I1", "N0", "A")
    rep = L.evaluate_gate(rnd, "chunk1_iter1")
    assert rep["pairing"] == "full_rerun" and rep["passes_numeric_gates"] and rep["group_dev_net"] == 4


def test_mechanism_gate(tmp_path):
    s1 = L.default_strategy_s1("N0")
    rnd = fake_round(tmp_path, s1, {"A": GROUP5[:2]})
    st = rnd.state(); s, t = rnd.strategy(1), rnd.targets(1)
    rows = L.arm_rows("N0", s, t, "A", TASKS, st) + L.arm_rows("C1I1", s, t, "A", TASKS, st)
    put_phase(rnd, "chunk1_iter1", "candidate", rows, "C1I1", "N0", "A", lambda r: int(r["role"] not in ("dev", "selection")))
    # candidate: same outcomes, but its failures moved to a later class
    d = rnd.results("chunk1_iter1") / "evaluator_only/rows.jsonl"
    rr = [json.loads(l) for l in d.read_text().splitlines()]
    for r in rr:
        if r["arm"] == "C1I1" and not r["official"]:
            r["first_failure"] = "place_descent:failed"
    d.write_text("".join(json.dumps(r) + "\n" for r in rr))
    put_candidate_files(rnd, "chunk1_iter1", "C1I1", "N0", "A", mode="mechanism")
    rep = L.evaluate_gate(rnd, "chunk1_iter1")
    assert rep["mechanism_ok"] and rep["accepted_by"] == "mechanism" and rep["passes_numeric_gates"]


def test_infra_rows_replace_and_block(tmp_path):
    rnd = fake_round(tmp_path, L.default_strategy_s1("N0"), {"A": GROUP5[:1]})
    rows = [{"arm": "N0", "task_short": GROUP5[0], "seed": s, "role": "dev"} for s in (86, 87)]
    put_phase(rnd, "baseline1_A", "baseline", rows, "N0", "N0", "A", lambda r: 0)
    L.write_json(rnd.results("baseline1_A") / "vlm_infra_episodes.json", {"episodes": [{"arm": "N0", "task_short": GROUP5[0], "seed": 87}]})
    with pytest.raises(L.HarnessError):
        L.phase_rows(rnd, "baseline1_A")
    assert L.row_store(rnd, "N0") == {}   # pending infra rerun: not in any store
    put_phase(rnd, "baseline1_A_infra1", "infra", rows[1:], "N0", "N0", "A", lambda r: 1)
    pr = L.phase_rows(rnd, "baseline1_A")
    assert pr[("N0", GROUP5[0], 87)]["official"] == 1 and pr[("N0", GROUP5[0], 87)]["infra_rerun"]


def test_strategy_is_frozen(tmp_path):
    rnd = fake_round(tmp_path, L.default_strategy_s1("N0"), {"A": GROUP5[:1]})
    p = rnd.iter_dir(1) / "strategy.json"; s = json.loads(p.read_text()); s["gate"]["loss_caps"]["ordinary_regression"] = 5
    p.write_text(json.dumps(s))
    with pytest.raises(L.HarnessError):
        rnd.strategy(1)


# ----------------------------------------------------------------------------- slots, build, runner
def test_slot_rules():
    w = L.warm_strategy_n17_n20("N0"); st = L.new_state(); st["slots"]["1"] = []
    assert L.slot_check(st, w, 1, 1, 0, "A") is False
    st["slots"]["1"].append({"j": 1, "rev": 0, "group": "A", "retry": False})
    assert L.slot_check(st, w, 1, 1, 1, "A") is True             # same-slot revision
    with pytest.raises(L.HarnessError):
        L.slot_check(st, w, 1, 1, 0, "A")                        # slot reused
    with pytest.raises(L.HarnessError):
        L.slot_check(st, L.default_strategy_s1("N0"), 1, 1, 1, "A")   # revisions disabled in S_1
    with pytest.raises(L.HarnessError):
        L.slot_check(st, w, 1, 7, 0, "B")                        # > max_slots


def test_apply_proposal_rules(tmp_path):
    parent = (REF / "programs/N20.py").read_text()
    anchor = "def la_dispatch("
    assert parent.count(anchor) == 1
    good = tmp_path / "good.py"
    good.write_text(f"MODULE = 'language_adapter'\nTOUCHED_MODULES = ['language_adapter']\nFAILURE_CLASS = 'binding:'\n"
                    f"INSERTS = [({anchor!r}, '# rsi test insert\\n')]\nRUNTIME_MARKERS = ['RSI_TEST']\n")
    forbidden = ("bowl", "plate"); modules = ("language_adapter", "binding", "place_primitive")
    prog, info = L.apply_proposal(parent, good, "one_module", forbidden, modules, "C1I1")
    assert "'arm_90': 'C1I1'," in prog and "# rsi test insert" in prog and info["runtime_markers"] == ["RSI_TEST"]
    assert L.normalized_sha(prog) != L.normalized_sha(parent)
    bad = tmp_path / "bad.py"
    bad.write_text(f"MODULE = 'language_adapter'\nFAILURE_CLASS = 'x'\nINSERTS = [({anchor!r}, 'target = bowl\\n')]\n")
    with pytest.raises(L.HarnessError, match="forbidden"):
        L.apply_proposal(parent, bad, "one_module", forbidden, modules, "C1I1")
    two = tmp_path / "two.py"
    two.write_text(f"TOUCHED_MODULES = ['language_adapter', 'binding']\nFAILURE_CLASS = 'x'\nINSERTS = [({anchor!r}, '# a\\n')]\n")
    with pytest.raises(L.HarnessError, match="one_module"):
        L.apply_proposal(parent, two, "one_module", forbidden, modules, "C1I1")
    prog2, _ = L.apply_proposal(parent, two, "primitive_across_modules", forbidden, modules, "C1I2")
    assert "'arm_90': 'C1I2'," in prog2


def test_runner_patch_compiles_and_accepts_generic_arms(tmp_path):
    src = (REF / "code/iso_runner_90.py").read_text()
    out = tmp_path / "iso_runner_90.py"; out.write_text(L.patch_runner(src))
    py_compile.compile(str(out), doraise=True)
    text = out.read_text()
    start = text.index("import re as _rsi_re"); end = text.index("ARMS90 = _RSIArmMap(lambda a: a)") + len("ARMS90 = _RSIArmMap(lambda a: a)")
    ns = {}
    exec(text[start:end] + "\nARM_BASE = _RSIArmMap(lambda a: 'MGG')\nPROGRAM_MARKER = _RSIArmMap(lambda a: f\"'arm_k': 'K2', 'arm_90': '{a}',\")\n", ns)
    for arm in ("N0", "N12", "C3I2", "C3I2r1"):
        assert arm in ns["ARMS90"] and ns["ARM_BASE"][arm] == "MGG" and ns["PROGRAM_MARKER"].get(arm).endswith(f"'{arm}',")
    for arm in ("N20C", "R1", "long0", ""):
        assert arm not in ns["ARMS90"] and ns["PROGRAM_MARKER"].get(arm) is None
    assert "choices=list(ARM_BASE)" not in text and "type=_rsi_arm" in text


def test_k2_has_no_language_adapter_and_n20_does():
    rows, _ = L.coverage(REF / "programs/K2.py", TASKS)
    assert rows is None
    rows, sup = L.coverage(REF / "programs/N20.py", TASKS)
    assert len(rows) == 90 and sup


def test_cli_init_builds_a_clean_round(tmp_path):
    R = tmp_path / "round"
    env = dict(os.environ, RSI_SKIP_INIT_TESTS="1")
    r = subprocess.run([sys.executable, str(HARNESS / "rsi.py"), "--round", str(R), "init", "--authorization", str(HARNESS / "authorization.template.json")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    n0 = (R / "programs/N0.py").read_text()
    assert n0.count("'arm_90': 'N0',") == 1
    assert n0.replace("'arm_90': 'N0',", "'arm_90': 'A0',") == (REF / "programs/A0.py").read_text()   # default N0 = A0: clean start
    assert json.loads((R / "authorization.json").read_text())["interface"] == "original"
    assert not (R / "code/n19.py").exists() and (R / "code/rsi.py").exists() and (R / "code/posthoc_first_failure.py").exists()
    runner = (R / "code/iso_runner_90.py").read_text()
    assert "type=_rsi_arm" in runner and 'if os.environ.get("RSI_INTERFACE", "rsi") != "original":' in runner
    assert 'manifest.update(interface=os.environ.get("RSI_INTERFACE", "rsi")' in runner
    L.verify_hash_manifest(R, R / "prepared_inputs_sha256.txt")
    assert json.loads((R / "state.json").read_text())["current"] == "N0"
    s1 = json.loads((R / "iter_1/strategy_template.json").read_text())
    assert L.validate_strategy(s1, TASKS, 1, "N0") == []
    hits = subprocess.run(["grep", "-rIlE", "libero_10|LIBERO-Long", str(R), "--include=*.json", "--include=*.txt"], capture_output=True, text=True).stdout
    assert hits == "", hits


def test_two_step_freeze_for_inventory_first(tmp_path):
    R = tmp_path / "R"
    for d in ("programs", "results", "proposals", "logs", "iter_1"):
        (R / d).mkdir(parents=True)
    (R / "task_manifest.json").write_text(json.dumps({"tasks": TASKS}))
    rnd = L.Round(R); rnd.save_state(L.new_state("N0"))
    L.write_json(R / "iter_1/strategy.json", L.default_strategy_s1("N0"))
    (R / "iter_1/STRATEGY_UPDATE.md").write_text("S_1\n")
    with pytest.raises(L.HarnessError, match="inventory"):
        L.freeze_strategy(rnd, 1)                                   # S_1 runs no inventory: targets required
    w = L.warm_strategy_n17_n20("N0"); L.write_json(R / "iter_1/strategy.json", w)
    s, t = L.freeze_strategy(rnd, 1)
    assert t is None and rnd.strategy(1)["pairing"]["mode"] == "row_store"
    assert L.inventory_rows("N0", s, TASKS)[0]["role"] == "inventory" and len(L.inventory_rows("N0", s, TASKS)) == 180
    with pytest.raises(L.HarnessError, match="not frozen"):
        rnd.targets(1)
    L.write_json(R / "iter_1/targets.json", targets({"A": GROUP5[:2], "B": GROUP5[2:4]}))
    L.freeze_strategy(rnd, 1, targets_only=True)
    assert list(rnd.targets(1)["groups"]) == ["A", "B"]
    with pytest.raises(L.HarnessError):
        L.freeze_strategy(rnd, 1, targets_only=True)               # targets are frozen once


def fake_episode(rnd, phase, arm, task, seed, official, vlm_error=False, timeout=False, stop_only=False, silent=False):
    import hashlib
    ep = Path(f"stage_{phase}/attempts/{arm}__{task}__s{seed}__abcd1234/{arm}/{task}/seed_{seed:02d}")
    d = rnd.p(ep); (d / "diag").mkdir(parents=True); (d / "artifacts/trial").mkdir(parents=True)
    (d / "policy.py").write_bytes(rnd.program(arm).read_bytes())
    for n in ("events.jsonl", "recorder_meta.json") + (() if stop_only else ("samples.npz",)):
        (d / "diag" / n).write_text("x")
    if stop_only:
        (d / "artifacts/trial/stdout.txt").write_text("" if silent else 'LANG_PARSE {"mode": "unparsed_clause"}\nBINDING_STOP {"status": "language_unparsed"}\n')
    else:
        (d / "artifacts/trial/stdout.txt").write_text('BINDING {"status": "bound"}\nSUBGOAL_STATUS {}\n' + ("EPISODE_TIMEOUT horizon=4000\n" if timeout else ""))
    if vlm_error:
        (d / "diag/vlm_calls.jsonl").write_text(json.dumps({"status": "error"}) + "\n")
    td = {"objects": [task]}
    m = {"arm": arm, "task_short": task, "seed": seed, "phase": phase, "policy_sha256": L.digest(rnd.program(arm)),
         "adapter_files_sha256": {n: L.digest(rnd.p("code", n)) for n in ("adapter_v3.py", "adapter_lexicon_v3.json")},
         "task_descriptors": td, "descriptor_sha256": hashlib.sha256(json.dumps(td, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
         "official_task_completed": official, "n_init_states": 50, "state_idx": (seed - 1) % 50, "sandbox_rc": 0, "recorder_errors": [],
         "failure_class": "episode_timeout" if timeout else None, "truncated": timeout, "final_sim_step": 4000 if timeout else (10 if stop_only else 900),
         "n_samples": 0 if stop_only else 500}
    (d / "attempt_manifest.json").write_text(json.dumps(m))
    return {"phase": phase, "resolved": "True", "episode_dir": str(ep), "task": task, "seed": str(seed), "official": str(official),
            "crashed_traceback": "False", "first_failure_truth_corrected": "" if official else ("none_logged" if stop_only else "grasp_descent:failed")}


def test_analyze_phase_and_infra_flags(tmp_path):
    import csv
    rnd = fake_round(tmp_path, L.default_strategy_s1("N0"), {"A": GROUP5[:1]})
    (rnd.p("code")).mkdir()
    for n in ("adapter_v3.py", "adapter_lexicon_v3.json"):
        (rnd.p("code", n)).write_bytes((REF / "code" / n).read_bytes())
    rnd.program("N0").write_text("# N0\n")
    rows = [{"arm": "N0", "task_short": GROUP5[0], "seed": s, "role": "dev"} for s in (86, 87)] + \
           [{"arm": "N0", "task_short": GROUP5[0], "seed": 76, "role": "selection"}]
    meta = {"phase": "baseline1_N0_A", "kind": "baseline", "iteration": 1, "arm": "N0", "parent": "N0", "group": "A", "rows": rows,
            "timeout_rate_max": 0.02}
    L.write_plan(rnd, "baseline1_N0_A", meta)
    post = [fake_episode(rnd, "baseline1_N0_A", "N0", GROUP5[0], 86, 1), fake_episode(rnd, "baseline1_N0_A", "N0", GROUP5[0], 87, 0, vlm_error=True),
            fake_episode(rnd, "baseline1_N0_A", "N0", GROUP5[0], 76, 0)]
    (rnd.p("results/posthoc")).mkdir(parents=True)
    with open(rnd.p("results/posthoc/episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(post[0])); w.writeheader(); w.writerows(post)
    with open(rnd.p("attempts_log_baseline1_N0_A.csv"), "w") as fh:
        fh.write("rc\n0\n0\n0\n")
    rnd.p("frozen_sha256_baseline1_N0_A.txt").write_text(L.hash_manifest(rnd.root, [rnd.program("N0")]))
    integ = L.analyze_phase(rnd, "baseline1_N0_A")
    assert integ["eligible"], integ["errors"]
    a = json.loads((rnd.results("baseline1_N0_A") / "analysis.json").read_text())
    assert a["totals"]["N0"] == {"dev": "1/2"} and "selection" not in json.dumps(a["proposer_visible"])   # selection never in proposer files
    att = (rnd.results("baseline1_N0_A") / "attribution.md").read_text()
    assert "seed 86" in att and "seed 76" not in att
    sel = json.loads((rnd.results("baseline1_N0_A") / "evaluator_only/selection.json").read_text())
    assert sel["N0"]["selection"][GROUP5[0]]["official"] == "0/1"
    infra = L.flag_infra(rnd, "baseline1_N0_A")
    assert [(e["task_short"], e["seed"], e["role"]) for e in infra["episodes"]] == [(GROUP5[0], 87, "dev")]
    # a pre-motion stop (no samples) is valid; a silent zero-motion episode is not
    post.append(fake_episode(rnd, "baseline1_N0_A", "N0", GROUP5[0], 88, 0, stop_only=True))
    post.append(fake_episode(rnd, "baseline1_N0_A", "N0", GROUP5[0], 89, 0, stop_only=True, silent=True))
    rows += [{"arm": "N0", "task_short": GROUP5[0], "seed": s, "role": "dev"} for s in (88, 89)]
    rnd.p("plan_baseline1_N0_A_roles.json").write_text(json.dumps(dict(meta, rows=rows)))
    with open(rnd.p("results/posthoc/episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(post[0])); w.writeheader(); w.writerows(post)
    with open(rnd.p("attempts_log_baseline1_N0_A.csv"), "w") as fh:
        fh.write("rc\n" + "0\n" * 5)
    integ = L.analyze_phase(rnd, "baseline1_N0_A")
    assert integ["errors"] == [f"trajectory: ('N0', '{GROUP5[0]}', 89)"], integ["errors"]
    rr = {r["seed"]: r for r in L.read_jsonl(rnd.results("baseline1_N0_A") / "evaluator_only/rows.jsonl")}
    assert rr[88]["pre_motion_stop"] and rr[88]["first_failure"] == "stop:language_unparsed"
    assert any(l.startswith("BINDING ") for l in rr[86]["key_lines"])
    # a tampered program after the stage makes the phase ineligible
    rnd.program("N0").write_text("# N0 changed\n")
    (rnd.results("baseline1_N0_A") / "integrity.json").unlink()
    assert not L.analyze_phase(rnd, "baseline1_N0_A")["eligible"]



@pytest.mark.parametrize("name", ["A0", "S1", "K2", "N20"])
def test_start_programs_get_one_n0_marker(name):
    src = (REF / "programs" / f"{name}.py").read_text()
    assert L.digest(REF / "programs" / f"{name}.py") in (REF / "programs" / L.START_PROGRAMS[name]).read_text()
    n0 = L.add_marker(src, "N0")
    assert len(L.MARK_RE.findall(n0)) == 1 and "'arm_90': 'N0'," in n0
    assert L.set_marker(n0, "C1I1").count("'arm_90': 'C1I1',") == 1
    if name in ("A0", "N20"):
        assert n0.replace("'arm_90': 'N0',", f"'arm_90': '{name}',") == src
    else:
        assert n0.replace(" 'arm_90': 'N0',", "", 1) == src


def test_s1_is_k0_without_its_marker():
    # K0 = S1 + marker (subset round); K1/K2 are the two RSI iterations a S1 start skips
    s1 = (REF / "programs/S1.py").read_text(); k2 = (REF / "programs/K2.py").read_text()
    assert "'arm_k'" not in s1 and "'arm_k': 'K2'," in k2 and s1 != k2.replace(" 'arm_k': 'K2',", "")



def test_a0_is_a_clean_original_api_program():
    import importlib.util
    src = (REF / "programs/A0.py").read_text()
    spec = importlib.util.spec_from_file_location("ap", REF / "code/apply_proposal_90.py"); ap = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap)
    import re
    assert not [w for w in ap.FORBIDDEN if re.search(r"(?<![a-z_])" + w + r"(?![a-z_])", src.lower())]   # no object/scene/colour literal
    assert not [p for p in L.FORBIDDEN_API if re.search(p, src)]                                          # no simulator truth
    harness_tools = ("move_to_pose", "verify_grasp", "verify_hold", "bind_object", "task_descriptors(", "query_vlm", "contact_move",
                     "gate_allow", "gg_plan_grasp", "pv_make_contract", "fg_plan_grasp", "read_robot_state")
    assert not [t for t in harness_tools if t in src]                                                    # original interface only
    assert "'arm_k'" not in src and "'arm_s'" not in src                                                  # not derived from S1/K2
    rows, _ = L.coverage(REF / "programs/A0.py", TASKS)
    assert sum(r["status"] == "dispatch_ok" for r in rows.values()) == 62
    ns = L.load_policy_ns(REF / "programs/A0.py")
    sg, mode = ns["la_parse"]({}, "put the yellow and white mug on the plate")
    assert mode == "grammar_v0" and sg[0]["object"] == "yellow and white mug" and sg[0]["support"] == "plate"
    sg, _ = ns["la_parse"]({}, "pick up the book on the shelf and place it in the box")
    assert sg == [{"verb": "place", "object": "book on the shelf", "relation": "in", "side": "-", "support": "box"}]
    assert ns["la_parse"]({}, "turn on the lamp")[0] == []


def test_patch_rules_depend_on_interface_and_ban_truth_apis(tmp_path):
    parent = (REF / "programs/A0.py").read_text(); anchor = "def la_dispatch("
    side = tmp_path / "side.py"
    side.write_text(f"MODULE = 'language_adapter'\nFAILURE_CLASS = 'x'\nINSERTS = [({anchor!r}, 'SIDES = (\"left\", \"right\")\\n')]\n")
    forbidden = ("left", "right", "bowl"); modules = ("language_adapter",)
    with pytest.raises(L.HarnessError, match="forbidden literal"):
        L.apply_proposal(parent, side, "one_module", forbidden, modules, "C1I1", interface="rsi")
    prog, _ = L.apply_proposal(parent, side, "one_module", forbidden, modules, "C1I1", interface="original")
    assert "'arm_90': 'C1I1'," in prog
    cheat = tmp_path / "cheat.py"
    cheat.write_text(f"MODULE = 'language_adapter'\nFAILURE_CLASS = 'x'\nINSERTS = [({anchor!r}, 'P = env.handle.env.sim.data.body_xpos\\n')]\n")
    with pytest.raises(L.HarnessError, match="simulator-truth"):
        L.apply_proposal(parent, cheat, "one_module", forbidden, modules, "C1I1", interface="original")


def test_auth_interface_rules():
    sys.path.insert(0, str(HARNESS)); import rsi
    a = json.loads((HARNESS / "authorization.template.json").read_text())
    rsi.validate_auth(a)
    for bad in ({"interface": "none"}, {"start_program": "S1", "interface": "original"}):
        with pytest.raises(L.HarnessError):
            rsi.validate_auth(dict(a, **bad))
