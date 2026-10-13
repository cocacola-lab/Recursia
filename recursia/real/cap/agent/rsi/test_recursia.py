"""Hardware-free checks of the physical paired-panel protocol."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .campaign import Experiment, RolloutAbort
from .capabilities import validate_policy
from .generator import decode_response
from .recursia import RecursiaProtocol
from .store import MethodStore, initialize, read_json, write_json
from .versions import EvolutionGit

V0 = "def run_task(task):\n    print('v0')\n"
V1 = "def run_task(task):\n    print('v1')\n"
V2 = "def run_task(task):\n    print('v2')\n"


class Model:
    def __init__(self, codes=(V0, V1, V2)):
        self.codes = iter(codes)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        answer = {"analysis": "Fix the diagnosed placement failure using task descriptors.", "code": next(self.codes)}
        if kwargs.get("schema"):
            answer["failure_class"] = "placement"
        return answer


def make_experiment(tmp_path, codes=(V0, V1, V2), evaluator=None, *, dry_run=False):
    tasks = read_json(Path(__file__).resolve().parents[3] / "config/rsi_tasks.json")["tasks"]
    root = initialize(tmp_path / "campaign", tasks)
    store = MethodStore(root, "recursia")
    model = Model(codes)
    calls = []

    def rollout(code, task, episode, folder):
        calls.append((code, task["id"], episode))
        assert not read_json(folder / "execution_mode.json")["fixed_policy"]
        return {"success": "v1" in code, "execution_model_calls": 0}

    exp = Experiment(store, model, evaluator or rollout, dry_run=dry_run)
    return store, model, exp, RecursiaProtocol(store, exp), calls


def panel(exp):
    return [exp.source([f"S{index}"]) for index in range(1, 7)]


def test_fixed_baseline_is_generated_once_and_resume_keeps_identical_module(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    exp.source(["S1"])
    assert protocol.status()["pending_tasks"] == ["S2", "S3", "S4", "S5", "S6"]
    for index in range(2, 7):
        Experiment(store, Model(()), exp.evaluator).source([f"S{index}"])
    assert len(model.prompts) == 1 and len(calls) == 6
    assert all(code == V0 + "\nrun_task(TASK)\n" for code, _, _ in calls)
    assert store.knowledge == V0 and protocol.status()["phase"] == "review_pending"
    assert "T1" not in model.prompts[0] and "first put the small basin" not in model.prompts[0]


def test_full_protocol_accepts_improvement_rejects_regression_then_runs_three_targets(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    peers = {method: MethodStore(store.campaign, method).knowledge for method in ("sd-only", "aspire")}
    panel(exp)
    assert protocol.propose()["candidate"] == "candidate_1"
    assert store.knowledge == V0
    panel(exp)
    assert store.knowledge == V0  # wait for explicit review, including human scores
    assert protocol.review()["accepted"]
    assert store.knowledge == V1
    assert protocol.propose()["candidate"] == "candidate_2"
    panel(exp)
    assert not protocol.review()["accepted"]
    assert store.knowledge == V1
    assert store.state["source_rollouts"] == len(calls) == 18
    assert len(model.prompts) == 3
    with pytest.raises(ValueError, match="budget"):
        protocol.propose()
    frozen = store.freeze()
    before = store.state_path.read_bytes(), store.knowledge, EvolutionGit(store).head()
    for task in ("T1", "T2", "T3"):
        result = exp.evaluate(task)
        assert result["programs_generated"] == result["execution_model_calls"] == 0
        assert "layout" not in result and result["success"]
    assert len(calls) == 21 and len(model.prompts) == 3
    assert not (store.root / "evaluation_panel.json").exists()
    assert store.state == {**json.loads(before[0]), "frozen": False}
    assert before[1:] == (store.knowledge, EvolutionGit(store).head())
    assert store.freeze()["sha256"] == frozen["sha256"]
    assert all(MethodStore(store.campaign, method).knowledge == text for method, text in peers.items())
    assert "put both the mug and the milk carton" not in model.prompts[1]
    history = EvolutionGit(store).history()
    assert len(history) == 3  # empty, N0, accepted C1; rejected C2 never imported


def test_incomplete_panel_duplicate_task_and_early_freeze_do_not_start_more_motion(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    exp.source(["S1"])
    with pytest.raises(ValueError, match="already ran"):
        exp.source(["S1"])
    with pytest.raises(ValueError, match="all source tasks"):
        protocol.propose()
    with pytest.raises(ValueError, match="all source tasks"):
        store.freeze()
    assert len(calls) == len(model.prompts) == 1


def test_manual_scores_are_applied_before_review_and_sealed_afterwards(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    panel(exp)
    protocol.score("S1", success=True, note="video shows object released inside destination")
    protocol.propose()
    panel(exp)
    protocol.score("S1", success=False, failure_class="placement", progress=2)
    decision = protocol.review()
    assert not decision["accepted"] and decision["regressed_tasks"] == ["S1"]
    assert store.knowledge == V0
    with pytest.raises(ValueError, match="already reviewed"):
        protocol.score("S1", success=True)


def test_failure_stage_acceptance_requires_named_class_eliminated_on_all_affected_cases():
    old = {"S1": {"success": False, "failure_class": "placement", "progress": 2},
           "S2": {"success": False, "failure_class": "placement", "progress": 2}}
    new = {key: {"success": False, "failure_class": "release", "progress": 3} for key in old}
    assert RecursiaProtocol.verdict(old, new, "placement")["accepted"]
    assert not RecursiaProtocol.verdict(old, new)["accepted"]
    new["S2"]["failure_class"] = "placement"
    assert not RecursiaProtocol.verdict(old, new, "placement")["accepted"]


def test_invalid_proposals_use_candidate_budget_without_motion_or_official_update(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path, (V0, "open_gripper()", V0))
    panel(exp)
    with pytest.raises(ValueError, match="run_task"):
        protocol.propose()
    with pytest.raises(ValueError, match="byte-identical"):
        protocol.propose()
    assert len(calls) == 6 and store.knowledge == V0
    assert protocol.status()["candidates_proposed"] == 2
    assert not read_json(store.root / "source/chunk_001/candidate_1/offline_checks.json")["passed"]
    store.freeze()


@pytest.mark.parametrize("invalid", ["def run_task(task)\n    pass", "def run_task(task):\n    get_object_pose('mug')",
                                   "def run_task(task):\n    pass\ndef bad(a=open_gripper()):\n    pass"])
def test_invalid_bootstrap_never_executes_robot(tmp_path, invalid):
    store, model, exp, protocol, calls = make_experiment(tmp_path, (invalid,))
    with pytest.raises(ValueError):
        exp.source(["S1"])
    assert not calls and store.state["source_rollouts"] == 0


@pytest.mark.parametrize("failure", ["capture", "generation", "validation"])
def test_bootstrap_retry_preserves_failed_attempt_and_captures_fresh_scene(tmp_path, failure):
    from .test_simulation_flow import FakeRuntime
    store, _, exp, protocol, calls = make_experiment(tmp_path)
    scenes = []

    class Observer:
        def observe(self, task, folder):
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "evidence.txt").write_text(str(len(scenes)))
            scenes.append(folder)
            if len(scenes) == 1 and failure == "capture":
                raise RuntimeError("camera unavailable")
            return FakeRuntime(folder).capture_feedback()

        def __call__(self, *args):
            return original_rollout(*args)

    class RetryModel:
        calls = 0

        def generate(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1 and failure == "generation":
                raise RuntimeError("provider unavailable")
            return {"code": "open_gripper()" if self.calls == 1 and failure == "validation" else V0}

    original_rollout = exp.evaluator
    exp.evaluator, exp.generator = Observer(), RetryModel()
    with pytest.raises((RuntimeError, ValueError)):
        exp.source(["S1"])
    original = protocol.folder / "bootstrap"
    evidence = {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
    assert not protocol.path.exists() and not calls and store.state["source_rollouts"] == 0
    exp.source(["S1"])
    assert len(scenes) == 2 and scenes[0] != scenes[1]
    assert scenes[1].parent.name.startswith("bootstrap_retry_")
    assert evidence == {p.relative_to(original): p.read_bytes() for p in original.rglob("*") if p.is_file()}
    assert len(calls) == store.state["source_rollouts"] == 1 and store.knowledge == V0
    bootstrap_event = next(e for e in store.state["source_events"] if e["kind"] == "bootstrap")
    assert bootstrap_event["path"] == str(scenes[1].parent)


def test_legacy_campaign_is_readable_but_cannot_mix_old_and_new_learning(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    meta = read_json(store.campaign / "campaign.json")
    meta.pop("recursia_protocol")
    write_json(store.campaign / "campaign.json", meta)
    exp = Experiment(MethodStore(store.campaign, "recursia"), model, exp.evaluator)
    with pytest.raises(ValueError, match="fresh campaign"):
        exp.source(["S1"])
    assert not calls and not model.prompts


def test_existing_learning_and_edited_round_archives_are_not_silently_adopted(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    exp.source(["S1"])
    (protocol.folder / "baseline.py").write_text(V2)
    with pytest.raises(ValueError, match="hash changed"):
        exp.source(["S2"])
    assert len(calls) == 1


def test_unknown_worker_failure_is_recorded_without_replaying_on_resume(tmp_path):
    def failing(*args):
        return {"success": False, "unknown_worker_failure": True, "error": "worker died"}
    store, model, exp, protocol, calls = make_experiment(tmp_path, evaluator=failing)
    with pytest.raises(RolloutAbort):
        exp.source(["S1"])
    assert protocol.load()["rounds"]["baseline"]["records"]["S1"]["unknown_worker_failure"]
    with pytest.raises(ValueError, match="already ran"):
        exp.source(["S1"])
    with pytest.raises(ValueError, match="cannot be scored"):
        protocol.score("S1", success=True)
    assert store.state["source_rollouts"] == 1


def test_dry_run_cannot_be_accepted_as_improvement_or_evaluated_physically(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path, dry_run=True)
    panel(exp)
    protocol.propose()
    panel(exp)
    with pytest.raises(ValueError, match="cannot be scored"):
        protocol.score("S1", success=True)
    assert protocol.review()["reason"] == "preview_only"
    assert store.knowledge == V0


def test_frozen_protocol_and_manual_target_scores_preserve_learning(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    panel(exp)
    store.freeze()
    before = store.state_path.read_bytes(), store.knowledge, protocol.path.read_bytes()
    result = exp.evaluate("T1")
    protocol.score_target("T1", success=True, note="manual video check")
    assert read_json(Path(result["run_dir"]) / "manual_score.json")["scored_by"] == "human"
    assert read_json(Path(result["run_dir"]) / "eval_result.json")["success"]
    assert store.state == {**json.loads(before[0]), "frozen": False}
    assert before[1:] == (store.knowledge, protocol.path.read_bytes())
    assert store.freeze()
    state = read_json(protocol.path)
    state["rounds"]["baseline"]["records"]["S1"]["success"] = True
    write_json(protocol.path, state)
    with pytest.raises(ValueError, match="protocol evidence changed"):
        store.snapshot()


def test_target_without_layout_can_repeat_and_ignores_old_exhausted_panel(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    panel(exp)
    old_panel = {"T1": {key: {"status": "completed", "success": False}
                        for key in ("eval01", "eval02", "eval03")}}
    write_json(store.root / "evaluation_panel.json", old_panel)
    # Finalizing source evidence on the first freeze is separate from target
    # execution; compare learning after that existing source transition.
    with store.evaluation_freeze():
        pass
    before = store.state, store.knowledge, protocol.path.read_bytes()
    first = exp.evaluate("T1")
    second = exp.evaluate("T1")
    assert first["run_dir"] != second["run_dir"]
    assert "layout" not in first and "layout" not in second
    assert len(calls) == 8 and len(model.prompts) == 1
    assert before == (store.state, store.knowledge, protocol.path.read_bytes())
    assert read_json(store.root / "evaluation_panel.json") == old_panel


def test_target_score_without_layout_uses_latest_run_and_cli_routes_to_target(tmp_path, capsys):
    from .cli import main
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    panel(exp)
    first, second = exp.evaluate("T1"), exp.evaluate("T1")
    before = store.state, store.knowledge, protocol.path.read_bytes()
    assert main(["score", "--campaign", str(store.campaign), "--method", "recursia",
                 "--task", "T1", "--success", "--note", "operator checked"]) == 0
    score = read_json(Path(second["run_dir"]) / "manual_score.json")
    assert score["success"] and score["automatic_success"] is False
    assert not (Path(first["run_dir"]) / "manual_score.json").exists()
    assert not read_json(Path(first["run_dir"]) / "eval_result.json")["success"]
    assert before == (store.state, store.knowledge, protocol.path.read_bytes())


def test_target_score_does_not_fall_back_to_earlier_run_after_interrupt(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    panel(exp)
    completed = exp.evaluate("T1")
    unfinished = Path(completed["run_dir"]).parent / "T1_999999999999999999999"
    unfinished.mkdir()
    with pytest.raises(ValueError, match="latest target execution did not complete"):
        protocol.score_target("T1", success=True)
    assert not (Path(completed["run_dir"]) / "manual_score.json").exists()


def test_explicit_legacy_layout_still_resolves_historical_scores(tmp_path):
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    panel(exp)
    exp.evaluate("T1", layout="eval01")
    protocol.score_target("T1", "eval01", success=True)
    assert read_json(store.root / "evaluation_panel.json")["T1"]["eval01"]["success"]
    with pytest.raises(ValueError, match="unused layout"):
        exp.evaluate("T1", layout="eval01")


@pytest.mark.parametrize("split", ["source", "target"])
@pytest.mark.parametrize("stopped", ["raise RuntimeError('grasp uncertain')", "print('rollout stopped')"])
def test_recursia_trials_repair_after_stop_without_persisting_online_code(tmp_path, split, stopped):
    from .cli import run_worker
    from .test_simulation_flow import FakeRuntime, Decisions
    from cap.saved_scripts.fr3_placement.test_placement import station
    module = "def run_task(task):\n    " + stopped + "\n"
    store, _, exp, _, _ = make_experiment(tmp_path, (module,))
    if split == "target":
        panel(exp)
        store.freeze()
    before = store.state_path.read_bytes(), store.knowledge, EvolutionGit(store).head()
    cfg = station.__wrapped__(tmp_path)
    model = Decisions(["REGENERATE\n```python\nopen_gripper()\n```", "FINISH"])

    def rollout(code, task, episode, folder):
        runtime = FakeRuntime(folder)
        args = SimpleNamespace(code=folder / "code.py", task_file=folder / "task.json",
            config=cfg.path, run_dir=folder, output=folder / "result.json", dry_run=False,
            snapshot=None, progress_log=None, context_file=folder / "context.md",
            fixed_policy=read_json(folder / "execution_mode.json")["fixed_policy"])
        return run_worker(args, runtime_factory=lambda *a, **kw: runtime, decision_generator=model)

    exp.evaluator = rollout
    packet = exp.source(["S1"]) if split == "source" else exp.evaluate("T1", layout="eval01")
    if split == "source":
        execution = read_json(Path(packet["tasks"][0]["path"]) / "execution/result.json")
        assert packet["tasks"][0]["model_finished"]
        assert store.state["source_model_calls"] == 3  # bootstrap plus two feedback calls
    else:
        execution = packet["execution"]
        assert packet["layout"] == "eval01" and packet["model_finished"]
        assert store.state == {**json.loads(before[0]), "frozen": False}
        assert before[1:] == (store.knowledge, EvolutionGit(store).head())
    assert execution["success"] and execution["num_regenerations"] == 1
    assert execution["execution_model_calls"] == len(model.calls) == 2
    assert execution["workflow"] == packet["workflow"] == "recursia_multi_turn.v1"
    assert not execution["fixed_policy"] and store.knowledge == module
    assert "open_gripper()" in execution["executed_code"]
    assert "grasp uncertain" in model.calls[0][0] or "rollout stopped" in model.calls[0][0]
    assert "there is no external model" not in model.calls[0][0]


def test_fixed_worker_never_calls_action_model_even_on_code_error(tmp_path):
    from .test_simulation_flow import FakeRuntime, Decisions
    from .trial_adapter import execute_trial
    runtime = FakeRuntime(tmp_path)
    model = Decisions([])
    result, verification = execute_trial(runtime, {}, "raise RuntimeError('failed')", model,
                                        context="fixed", timeout_s=1, fixed_policy=True)
    assert not model.calls and not runtime.trace and result["num_code_blocks"] == 1
    assert result["num_regenerations"] == 0 and "failed" in result["info_step"]["stderr"]


@pytest.mark.parametrize("code", ["open_gripper()", "raise RuntimeError('failed')"])
def test_fixed_flag_reaches_worker_without_action_feedback_or_regeneration(tmp_path, code):
    from .cli import run_worker
    from .test_rsi import TASKS
    from .test_simulation_flow import FakeRuntime, Decisions
    from cap.saved_scripts.fr3_placement.test_placement import station
    cfg = station.__wrapped__(tmp_path)
    code_file, task_file = tmp_path / "code.py", tmp_path / "task.json"
    code_file.write_text(code)
    write_json(task_file, TASKS[0])
    runtime, model = FakeRuntime(tmp_path / "execution"), Decisions([])
    args = SimpleNamespace(code=code_file, task_file=task_file, config=cfg.path, run_dir=runtime.run_dir,
                           output=tmp_path / "result.json", dry_run=False, snapshot=None, progress_log=None,
                           fixed_policy=True)
    result = run_worker(args, runtime_factory=lambda *a, **kw: runtime, decision_generator=model)
    assert result["fixed_policy"] and not result["model_finished"]
    assert result["execution_model_calls"] == result["num_regenerations"] == 0
    assert not model.calls and runtime.closed
    assert result["policy_finished"] is (code == "open_gripper()")


def test_candidate_response_decoder_preserves_failure_class():
    import json
    schema = {"properties": {key: {} for key in ("analysis", "code", "failure_class")},
              "required": ["analysis", "code", "failure_class"]}
    value = {"analysis": "diagnosis", "code": V1, "failure_class": "placement"}
    assert decode_response(json.dumps(value), schema) == value
    assert decode_response(json.dumps({"structured_output": value}), schema) == value


def test_propose_and_score_cli_are_offline_and_do_not_require_motion_permission(tmp_path, monkeypatch):
    from . import cli
    store, model, exp, protocol, calls = make_experiment(tmp_path)
    panel(exp)
    monkeypatch.delenv("OPENFORGE_ALLOW_PHYSICAL_MOTION", raising=False)
    monkeypatch.setattr(cli, "IsolatedGenerator", lambda *args: model)
    prefix = ["--campaign", str(store.campaign), "--method", "recursia"]
    assert cli.main(["score", *prefix, "--task", "S1", "--failure", "--note", "manual check"]) == 0
    assert cli.main(["propose", *prefix]) == 0
    assert len(calls) == 6 and protocol.status()["active_round"] == "candidate_1"
