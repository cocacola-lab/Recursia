"""Paired chunk scheduling with fake scenes, models and robot executions."""
from pathlib import Path
import json

import pytest

from .campaign import Experiment, RolloutAbort
from .recursia import recursia_protocol
from .store import MethodStore, digest, initialize, read_json, write_json
from .test_recursia import Model, V0, V1, V2


def setup_chunked(tmp_path, *, codes=None, evaluate=None):
    tasks = read_json(Path(__file__).resolve().parents[3] / "config/rsi_tasks.json")["tasks"]
    root = initialize(tmp_path / "campaign", tasks, chunk_size=2)
    store = MethodStore(root, "recursia")
    model = Model(codes or (V0, V1, V2, V2, V0, V2, V0))
    calls, preparations = [], []

    def rollout(code, task, episode, folder):
        calls.append((task["id"], episode, code, folder))
        assert not read_json(folder / "execution_mode.json")["fixed_policy"]
        assert folder.is_relative_to(store.root)
        return evaluate(code, task, episode, folder) if evaluate else {
            "success": "v1" in code, "hardware_fault": False, "execution_model_calls": 1,
            "model_finished": True, "executed_code": code + "\nprint('local repair')"}

    exp = Experiment(store, model, rollout)
    def prepare(chunk, round_, task):
        preparations.append((chunk, round_, task["id"]))
    return store, model, exp, calls, preparations, prepare


def test_three_pairs_run_all_candidates_stop_at_boundary_and_inherit_accepted_stack(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    assert store.meta["recursia_protocol"]["chunks"] == [["S1", "S2"], ["S3", "S4"], ["S5", "S6"]]
    assert store.meta["recursia_protocol"]["online_action_correction"]
    peer = MethodStore(store.campaign, "sd-only").state_path.read_bytes()
    for number, pair in enumerate((("S1", "S2"), ("S3", "S4"), ("S5", "S6")), 1):
        protocol = recursia_protocol(store, exp)
        before = len(calls)
        result = protocol.run_chunk(number, prepare=prepare)
        assert result["completed"] and result["progress"]["phase"] == "complete"
        assert len(calls) == before + 6 and [c[0] for c in calls[before:]] == list(pair) * 3
        assert [p[2] for p in prompts[before:]] == list(pair) * 3
        assert store.knowledge == V1
        assert (store.root / f"source/chunk_{number:03d}/N{number}.py").read_text() == V1
        if number > 1:
            assert calls[before][2] == V1 + "\nrun_task(TASK)\n"
        with pytest.raises(ValueError, match="already completed"):
            protocol.run_chunk(number, prepare=prepare)
        assert len(calls) == before + 6
    assert len(model.prompts) == 7 and store.state["source_rollouts"] == 18
    assert store.state["source_model_calls"] == 25  # seven generation + eighteen execution feedback calls
    assert MethodStore(store.campaign, "sd-only").state_path.read_bytes() == peer
    frozen = store.freeze()
    assert frozen["sha256"] == digest(V1) and len(frozen["chunk_state_sha256"]) == 3
    before = store.state_path.read_bytes()
    for task in ("T1", "T2", "T3"):
        target = exp.evaluate(task)
        assert "layout" not in target and target["success"]
    assert not (store.root / "evaluation_panel.json").exists()
    assert store.state == {**json.loads(before), "frozen": False}
    store.freeze()
    child = store.root / "source/chunk_001/chunk.json"
    child.write_text(child.read_text() + "\n")
    with pytest.raises(ValueError, match="chunk evidence changed"):
        store.snapshot()


def test_chunk_order_task_selection_and_early_freeze_fail_before_execution(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    protocol = recursia_protocol(store, exp)
    for number in (0, 2, 4):
        with pytest.raises(ValueError):
            protocol.run_chunk(number, prepare=prepare)
    with pytest.raises(ValueError, match="--chunk"):
        exp.source(["S1"])
    with pytest.raises(ValueError, match="complete all"):
        store.freeze()
    assert not calls and not prompts and not model.prompts


def test_manual_reset_interruption_resumes_pending_task_without_replaying_completed_one(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    def interrupted(chunk, round_, task):
        if task["id"] == "S2":
            raise KeyboardInterrupt()
        prepare(chunk, round_, task)
    with pytest.raises(KeyboardInterrupt):
        recursia_protocol(store, exp).run_chunk(1, prepare=interrupted)
    assert [c[0] for c in calls] == ["S1"]
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare)
    assert result["completed"] and [c[0] for c in calls] == ["S1", "S2"] * 3
    assert len(model.prompts) == 3


@pytest.mark.parametrize("fault", ["hardware_fault", "unknown_worker_failure", "camera"])
def test_noninteractive_fault_stops_then_interactive_resume_requires_a_decision(tmp_path, fault):
    failed = False
    def evaluate(*args):
        nonlocal failed
        if not failed:
            failed = True
            return {"success": False, fault: True,
                    "error": "terminated episode: camera feedback unavailable: disconnected" if fault == "camera" else "fault"}
        return {"success": True, "model_finished": True}
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, evaluate=evaluate)
    with pytest.raises(RolloutAbort):
        recursia_protocol(store, exp).run_chunk(1, prepare=prepare)
    assert len(calls) == len(prompts) == len(model.prompts) == 1
    with pytest.raises(RolloutAbort, match="--resume"):
        recursia_protocol(store, exp).run_chunk(1, prepare=prepare)
    assert len(calls) == len(prompts) == 1
    decisions = []
    def decide(chunk, round_, task, record):
        decisions.append(task["id"])
        if len(decisions) == 1:
            assert task["id"] == "S1" and len(calls) == len(model.prompts) == 1
        return 1
    assert recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=decide)["completed"]
    assert [c[0] for c in calls] == ["S1", "S2"] * 3


@pytest.mark.parametrize("fault", ["hardware_fault", "unknown_worker_failure", "camera"])
def test_second_task_fault_waits_for_zero_and_retries_only_second_task(tmp_path, fault):
    failed = False
    def evaluate(code, task, *args):
        nonlocal failed
        if task["id"] == "S2" and not failed:
            failed = True
            return {"success": False, fault: True,
                    "error": "terminated episode: camera feedback unavailable: disconnected" if fault == "camera" else "fault"}
        return {"success": True, "model_finished": True}
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, evaluate=evaluate)
    def decide(chunk, round_, task, record):
        if round_ == "baseline" and task["id"] == "S2" and record["attempt"] == 1:
            assert len(calls) == len(prompts) == 2 and len(model.prompts) == 1
            assert recursia_protocol(store, exp)._faulted(record)
            return 0
        return 1
    assert recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=decide)["completed"]
    assert [c[0] for c in calls] == ["S1", "S2", "S2"] + ["S1", "S2"] * 2
    state = read_json(store.root / "source/chunk_001/chunk.json")
    baseline = state["rounds"]["baseline"]
    assert baseline["records"]["S1"]["attempt"] == 1
    assert baseline["records"]["S2"]["attempt"] == 2
    assert "S2" not in baseline.get("attempt_history", {})
    assert not calls[1][3].exists()


def test_restart_after_fault_at_task_end_resumes_second_task_choice_without_resume_flag(tmp_path):
    failed = False
    def evaluate(code, task, *args):
        nonlocal failed
        if task["id"] == "S2" and not failed:
            failed = True
            return {"success": False, "hardware_fault": True, "error": "controller fault"}
        return {"success": True, "model_finished": True}
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, evaluate=evaluate)
    def interrupted(chunk, round_, task, record):
        if task["id"] == "S2":
            assert record["hardware_fault"]
            raise KeyboardInterrupt()
        return 1
    with pytest.raises(KeyboardInterrupt):
        recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=interrupted)
    choices = []
    def decide(chunk, round_, task, record):
        choices.append((round_, task["id"], record["attempt"]))
        if len(choices) == 1:
            assert len(calls) == 2 and len(model.prompts) == 1 and record["hardware_fault"]
            return 0
        return 1
    assert recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=decide)["completed"]
    assert choices[0] == ("baseline", "S2", 1)
    assert [c[0] for c in calls] == ["S1", "S2", "S2"] + ["S1", "S2"] * 2


def test_abort_before_execution_result_does_not_become_a_task_end_choice(tmp_path, monkeypatch):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    def abort(*args, **kwargs):
        raise RolloutAbort("setup/budget unavailable")
    monkeypatch.setattr(exp, "_rollout", abort)
    def decide(*args):
        pytest.fail("no completed execution result to decide on")
    with pytest.raises(RolloutAbort, match="setup/budget"):
        recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=decide)


def test_invalid_candidate_is_archived_then_remaining_candidate_finishes_chunk(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, codes=(V0, "open_gripper()", V1))
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare)
    assert result["completed"] and len(calls) == len(prompts) == 4
    assert store.knowledge == V1 and result["progress"]["accepted_candidates"] == ["candidate_2"]
    assert not read_json(store.root / "source/chunk_001/candidate_1/offline_checks.json")["passed"]
    assert not (store.root / "source/chunk_001/candidate_1/S1").exists()


def test_rejected_candidates_regenerate_without_baseline_replay_and_preserve_old_evidence(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, codes=(V0, "open_gripper()", "close_gripper()", V1, V2))
    protocol = recursia_protocol(store, exp)
    assert protocol.run_chunk(1, prepare=prepare)["completed"]
    folder = store.root / "source/chunk_001"
    before = store.state
    baseline = (folder / "baseline.py").read_bytes()
    baseline_records = read_json(folder / "chunk.json")["rounds"]["baseline"]
    original = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()
                and (p.name in {"chunk.json", "chunk_result.json", "snapshot.json", "N1.py"}
                     or p.parent.name in {"candidate_1", "candidate_2"})}
    assert len(calls) == 2 and len(model.prompts) == 3
    result = protocol.run_chunk(1, prepare=prepare, retry_rejected_candidates=True)
    assert result["completed"] and len(calls) == 6 and len(model.prompts) == 5
    assert [c[1] for c in calls] == ["baseline"] * 2 + ["candidate_1"] * 2 + ["candidate_2"] * 2
    state = read_json(folder / "chunk.json")
    assert state["rounds"]["baseline"] == baseline_records
    assert (folder / "baseline.py").read_bytes() == baseline
    recovery = state["proposal_recoveries"][0]
    archive = store.root / recovery["archive"]
    for name, content in original.items():
        assert (archive / name).read_bytes() == content
    assert state["candidates_proposed"] == 2 and store.knowledge == V1
    assert (folder / "N1.py").read_text() == V1
    assert (archive / "N1.py").read_text() == V0
    assert store.state["source_rollouts"] == before["source_rollouts"] + 4
    assert store.state["source_model_calls"] == before["source_model_calls"] + 6


def test_recovery_preserves_accepted_candidate_and_only_regenerates_trailing_rejection(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, codes=(V0, V1, "open_gripper()", V2))
    protocol = recursia_protocol(store, exp)
    protocol.run_chunk(1, prepare=prepare)
    folder = store.root / "source/chunk_001"
    accepted = (folder / "candidate_1/review.json").read_bytes()
    assert store.knowledge == V1 and len(calls) == 4
    protocol.run_chunk(1, prepare=prepare, retry_rejected_candidates=True)
    assert len(calls) == 6 and [c[1] for c in calls[-2:]] == ["candidate_2"] * 2
    assert calls[-2][2] == V2 + "\nrun_task(TASK)\n"
    assert (folder / "candidate_1/review.json").read_bytes() == accepted
    assert read_json(folder / "chunk.json")["proposal_recoveries"][0]["rejected_candidates"] == ["candidate_2"]


def test_recovery_does_not_reopen_completed_validated_rounds_or_new_chunks(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    protocol = recursia_protocol(store, exp)
    with pytest.raises(ValueError, match="start this chunk"):
        protocol.run_chunk(1, prepare=prepare, retry_rejected_candidates=True)
    assert not calls and not prompts and not model.prompts
    protocol.run_chunk(1, prepare=prepare)
    path = store.root / "source/chunk_001/chunk.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="no trailing offline-rejected"):
        protocol.run_chunk(1, prepare=prepare, retry_rejected_candidates=True)
    assert path.read_bytes() == before and len(calls) == 6


def test_recovery_resumes_pending_validation_without_reopening_it_again(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, codes=(V0, "open_gripper()", "close_gripper()", V1, V2))
    protocol = recursia_protocol(store, exp)
    protocol.run_chunk(1, prepare=prepare)
    def interrupted(chunk, round_, task):
        assert round_ == "candidate_1" and task["id"] == "S1"
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        protocol.run_chunk(1, prepare=interrupted, retry_rejected_candidates=True)
    assert len(calls) == 2 and len(model.prompts) == 4
    protocol.run_chunk(1, prepare=prepare, retry_rejected_candidates=True)
    assert len(calls) == 6 and len(model.prompts) == 5
    state = read_json(store.root / "source/chunk_001/chunk.json")
    assert len(state["proposal_recoveries"]) == 1


def test_cli_recovery_flag_regenerates_candidates_and_only_prompts_for_validation(tmp_path, monkeypatch):
    from . import cli
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, codes=(V0, "open_gripper()", "close_gripper()", V1, V2))
    recursia_protocol(store, exp).run_chunk(1, prepare=prepare)
    monkeypatch.setenv("OPENFORGE_ALLOW_PHYSICAL_MOTION", "1")
    monkeypatch.setattr(cli, "IsolatedGenerator", lambda *a: model)
    monkeypatch.setattr(cli, "PhysicalEvaluator", lambda *a: exp.evaluator)
    inputs = []
    def enter(prompt):
        inputs.append(prompt)
        assert " / baseline / " not in prompt
        if "Score the actual task outcome" in prompt:
            return "s"
        return "1" if "Enter 1" in prompt else ""
    monkeypatch.setattr("builtins.input", enter)
    assert cli.main(["source", "--campaign", str(store.campaign), "--method", "recursia", "--chunk", "1",
                     "--retry-rejected-candidates"]) == 0
    assert len(inputs) == 12 and len(calls) == 6


def test_completed_chunk_state_tampering_blocks_next_chunk(tmp_path):
    store, _, exp, _, _, prepare = setup_chunked(tmp_path)
    recursia_protocol(store, exp).run_chunk(1, prepare=prepare)
    p = store.root / "source/chunk_001/chunk.json"
    state = read_json(p)
    state["phase"] = "executing"
    write_json(p, state)
    with pytest.raises(ValueError):
        recursia_protocol(store, exp).run_chunk(2, prepare=prepare)


def test_cli_chunk_runs_to_completion_with_six_manual_resets_and_no_following_chunk(tmp_path, monkeypatch, capsys):
    from . import cli
    store, model, exp, calls, prompts, _ = setup_chunked(tmp_path)
    monkeypatch.setenv("OPENFORGE_ALLOW_PHYSICAL_MOTION", "1")
    monkeypatch.setattr(cli, "IsolatedGenerator", lambda *a: model)
    monkeypatch.setattr(cli, "PhysicalEvaluator", lambda *a: exp.evaluator)
    def enter(prompt):
        if "Score the actual task outcome" in prompt:
            return "s"
        if "Enter 1" in prompt:
            return "1"
        assert len(calls) == len(prompts)
        if not prompts:
            assert not model.prompts  # first layout is ready before bootstrap generation
        prompts.append(prompt)
        return ""
    monkeypatch.setattr("builtins.input", enter)
    assert cli.main(["source", "--campaign", str(store.campaign), "--method", "recursia", "--chunk", "1"]) == 0
    assert len(calls) == len(prompts) == 6 and [c[0] for c in calls] == ["S1", "S2"] * 3
    assert not (store.root / "source/chunk_002").exists()
    assert "Chunk 1 completed" in capsys.readouterr().out


def test_manual_scores_override_automatic_failures_and_drive_candidate_acceptance(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, evaluate=lambda *args: {
        "success": False, "model_finished": True, "verification": {"success": False, "error": "SAM3 miss"}})
    judgements = []
    def judge(chunk, round_, task, record):
        assert record["scored_by"] == "evaluator" and record["success"] is False
        passed = round_ != "baseline" or task["id"] == "S1"
        judgements.append((round_, task["id"], passed))
        return passed
    def decide(chunk, round_, task, record):
        assert record["scored_by"] == "human"
        return 1
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=judge, decide=decide)
    assert result["completed"] and len(judgements) == 6 and store.knowledge == V1
    state = read_json(store.root / "source/chunk_001/chunk.json")
    assert state["rounds"]["candidate_1"]["accepted"]
    assert state["rounds"]["candidate_1"]["gained_tasks"] == ["S2"]
    assert state["reference"]["S1"]["success"] and state["reference"]["S2"]["success"]
    assert [t["success"] for t in result["tasks"]] == [True, False, True, True, True, True]
    for (round_, task_id, passed), call in zip(judgements, calls):
        saved = read_json(call[3] / "result.json")
        manual = read_json(call[3] / "manual_score.json")
        summary = read_json(call[3].parent / "source_result.json")["tasks"][0]
        assert saved["success"] is passed and saved["automatic_success"] is False
        assert saved["verification"]["error"] == "SAM3 miss"
        assert manual["success"] is passed and manual["scored_by"] == "human"
        assert summary["success"] is passed and summary["scored_by"] == "human"
    assert '"scored_by": "human"' in model.prompts[1]


def test_manual_failure_overrides_automatic_success_and_blocks_regressing_candidate(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, evaluate=lambda *args: {
        "success": True, "model_finished": True})
    def judge(chunk, round_, task, record):
        return not (round_ == "candidate_1" and task["id"] == "S1")
    recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=judge, decide=lambda *args: 1)
    state = read_json(store.root / "source/chunk_001/chunk.json")
    assert state["rounds"]["candidate_1"]["reason"] == "regression"
    assert not state["rounds"]["candidate_1"]["accepted"] and store.knowledge == V0
    rejected_task = read_json(calls[2][3] / "result.json")
    assert rejected_task["automatic_success"] is True and rejected_task["success"] is False


def test_human_score_preserves_fault_diagnostics_and_dry_preview_cannot_validate(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, evaluate=lambda *args: {
        "success": False, "hardware_fault": True, "error": "controller fault"})
    recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=lambda *args: True, decide=lambda *args: 1)
    saved = read_json(calls[0][3] / "result.json")
    assert saved["scored_by"] == "human" and saved["success"] is True
    assert saved["hardware_fault"] and saved["forced_finish"] and saved["error"] == "controller fault"
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path / "preview")
    exp.dry_run = True
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=lambda *args: True, decide=lambda *args: 1)
    assert result["completed"] and not result["progress"]["accepted_candidates"]
    assert all(not task["success"] for task in result["tasks"])


@pytest.mark.parametrize("stage", ["judge", "decide"])
def test_manual_score_interruption_resumes_without_reexecution_or_reasking_saved_score(tmp_path, stage):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    def judge(chunk, round_, task, record):
        if stage == "judge" and task["id"] == "S2":
            raise KeyboardInterrupt()
        return round_ != "baseline" or task["id"] == "S1"
    def decide(chunk, round_, task, record):
        if stage == "decide" and task["id"] == "S2":
            raise KeyboardInterrupt()
        return 1
    with pytest.raises(KeyboardInterrupt):
        recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=judge, decide=decide)
    assert len(calls) == 2 and len(model.prompts) == 1
    scores, choices = [], []
    def resume_judge(chunk, round_, task, record):
        scores.append((round_, task["id"]))
        return round_ != "baseline" or task["id"] == "S1"
    def resume_decide(chunk, round_, task, record):
        choices.append((round_, task["id"]))
        return 1
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=resume_judge, decide=resume_decide)
    assert result["completed"] and len(calls) == 6
    assert choices[0] == ("baseline", "S2")
    assert scores[0] == (("baseline", "S2") if stage == "judge" else ("candidate_1", "S1"))


def test_zero_discards_manual_score_and_retries_with_fresh_judgement(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    scores = []
    def judge(chunk, round_, task, record):
        scores.append((round_, task["id"], record["attempt"]))
        return round_ != "baseline" or task["id"] == "S1"
    def decide(chunk, round_, task, record):
        return 0 if round_ == "baseline" and task["id"] == "S1" and record["attempt"] == 1 else 1
    recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=judge, decide=decide)
    assert scores[:2] == [("baseline", "S1", 1), ("baseline", "S1", 2)]
    assert not (calls[0][3] / "manual_score.json").exists()
    assert (calls[1][3] / "manual_score.json").is_file()


def test_upgrade_collects_missing_manual_scores_for_completed_active_tasks_without_replay(tmp_path, monkeypatch):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    protocol = recursia_protocol(store, exp)
    def interrupted():
        raise KeyboardInterrupt()
    monkeypatch.setattr(protocol, "review", interrupted)
    with pytest.raises(KeyboardInterrupt):
        protocol.run_chunk(1, prepare=prepare)
    assert len(calls) == 2
    grades = []
    def judge(chunk, round_, task, record):
        grades.append((round_, task["id"]))
        return round_ != "baseline" or task["id"] == "S1"
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare, judge=judge, decide=lambda *args: 1)
    assert result["completed"] and len(calls) == 6
    assert grades[:2] == [("baseline", "S1"), ("baseline", "S2")]
    assert store.knowledge == V1


def test_cli_manual_score_reprompts_invalid_input_before_continue(tmp_path, monkeypatch, capsys):
    from . import cli
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    monkeypatch.setenv("OPENFORGE_ALLOW_PHYSICAL_MOTION", "1")
    monkeypatch.setattr(cli, "IsolatedGenerator", lambda *args: model)
    monkeypatch.setattr(cli, "PhysicalEvaluator", lambda *args: exp.evaluator)
    seen = []
    invalid = True
    def enter(prompt):
        nonlocal invalid
        if "Score the actual task outcome" in prompt:
            if invalid:
                invalid = False
                return "wrong"
            seen.append("score")
            return "S"
        if "Enter 1" in prompt:
            assert seen[-1] == "score"
            seen.append("continue")
            return "1"
        return ""
    monkeypatch.setattr("builtins.input", enter)
    assert cli.main(["source", "--campaign", str(store.campaign), "--method", "recursia", "--chunk", "1"]) == 0
    assert seen == ["score", "continue"] * 6
    assert "Enter s (success) or f (failure)." in capsys.readouterr().out


def test_cli_init_defaults_to_pairs_and_rejects_single_task_without_motion(tmp_path, monkeypatch):
    from . import cli
    root = tmp_path / "fresh"
    assert cli.main(["init", "--campaign", str(root)]) == 0
    store = MethodStore(root, "recursia")
    assert store.meta["recursia_protocol"]["chunks"] == [["S1", "S2"], ["S3", "S4"], ["S5", "S6"]]
    monkeypatch.setenv("OPENFORGE_ALLOW_PHYSICAL_MOTION", "1")
    monkeypatch.setattr(cli, "IsolatedGenerator", lambda *a: None)
    assert cli.main(["source", "--campaign", str(root), "--method", "recursia", "--task", "S1"]) == 2
    assert store.state["source_rollouts"] == 0 and not (store.root / "source").exists()


@pytest.mark.parametrize("task_id", ["S1", "S2"])
@pytest.mark.parametrize("round_name", ["baseline", "candidate_1"])
def test_zero_deletes_current_attempt_and_excludes_it_from_future_feedback(tmp_path, task_id, round_name):
    marker = "DISCARDED_ATTEMPT_SHOULD_NOT_REACH_PROPOSER"
    def evaluate(code, task, episode, folder):
        return {"success": "v1" in code, "model_finished": True,
                "stdout": marker if episode == round_name and task["id"] == task_id and folder.parent.name == task_id else "retained result"}
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path, evaluate=evaluate)
    repeated = False
    def decide(chunk, round_, task, record):
        nonlocal repeated
        if round_ == round_name and task["id"] == task_id and not repeated:
            assert len(model.prompts) == (1 if round_name == "baseline" else 2)
            repeated = True
            return 0
        return 1
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=decide)
    expected = ["S1", "S1", "S2"] if task_id == "S1" else ["S1", "S2", "S2"]
    expected = expected + ["S1", "S2"] * 2 if round_name == "baseline" else ["S1", "S2"] + expected + ["S1", "S2"]
    assert [c[0] for c in calls] == expected
    assert result["completed"] and len(calls) == len(prompts) == store.state["source_rollouts"] == 7
    assert len(model.prompts) == 3 and store.knowledge == V1
    state = read_json(store.root / "source/chunk_001/chunk.json")
    latest = state["rounds"][round_name]["records"][task_id]
    assert latest["attempt"] == 2 and latest["operator_action"] == "continue"
    assert "attempt_002" in latest["path"]
    assert task_id not in state["rounds"][round_name].get("attempt_history", {})
    removed = next(c[3] for c in calls if c[0] == task_id and c[1] == round_name)
    assert not removed.exists()
    assert (Path(latest["path"]) / "result.json").is_file()
    assert (Path(latest["path"]) / "code.py").read_text() == (V0 if round_name == "baseline" else V1) + "\nrun_task(TASK)\n"
    assert marker not in model.prompts[-1] and "attempt_002" in model.prompts[-1]
    assert all(event.get("path") != str(removed) for event in store.state["source_events"])
    assert len(result["tasks"]) == 6
    for task in result["tasks"]:
        assert (Path(task["path"]) / "execution/result.json").is_file()
    for call in calls:
        if call[3] != removed:
            assert call[3].exists()


def test_zero_deletes_before_reset_and_resume_runs_only_pending_task(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    def decide(chunk, round_, task, record):
        return 0 if round_ == "baseline" and task["id"] == "S2" and record["attempt"] == 1 else 1
    def interrupted(chunk, round_, task):
        if round_ == "baseline" and task["id"] == "S2" and len(calls) == 2:
            assert not calls[1][3].exists()
            raise KeyboardInterrupt()
        prepare(chunk, round_, task)
    with pytest.raises(KeyboardInterrupt):
        recursia_protocol(store, exp).run_chunk(1, prepare=interrupted, decide=decide)
    state = read_json(store.root / "source/chunk_001/chunk.json")
    assert state["rounds"]["baseline"]["next_attempts"]["S2"] == 2
    assert "S2" not in state["rounds"]["baseline"]["records"]
    assert store.state["source_rollouts"] == 2 and len(model.prompts) == 1
    assert recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=decide)["completed"]
    assert [c[0] for c in calls] == ["S1", "S2", "S2"] + ["S1", "S2"] * 2


def test_saved_zero_choice_finishes_interrupted_cleanup_and_multiple_retries_keep_only_latest(tmp_path, monkeypatch):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    protocol = recursia_protocol(store, exp)
    def decide(chunk, round_, task, record):
        return 0 if round_ == "baseline" and task["id"] == "S1" and record["attempt"] < 3 else 1
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(protocol, "_discard_task_attempt", interrupted)
    with pytest.raises(KeyboardInterrupt):
        protocol.run_chunk(1, prepare=prepare, decide=decide)
    assert len(calls) == 1 and calls[0][3].exists()
    result = recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=decide)
    assert result["completed"] and len(calls) == store.state["source_rollouts"] == 8
    assert not calls[0][3].exists() and not calls[1][3].exists() and calls[2][3].exists()
    state = read_json(store.root / "source/chunk_001/chunk.json")
    assert state["rounds"]["baseline"]["records"]["S1"]["attempt"] == 3
    assert "S1" not in state["rounds"]["baseline"].get("attempt_history", {})


def test_interrupt_at_last_task_choice_resumes_choice_before_proposal_without_reexecution(tmp_path):
    store, model, exp, calls, prompts, prepare = setup_chunked(tmp_path)
    def interrupted(chunk, round_, task, record):
        if task["id"] == "S2":
            raise KeyboardInterrupt()
        return 1
    with pytest.raises(KeyboardInterrupt):
        recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=interrupted)
    assert len(calls) == 2 and len(model.prompts) == 1
    decisions = []
    def proceed(chunk, round_, task, record):
        decisions.append((round_, task["id"]))
        if len(decisions) == 1:
            assert len(calls) == 2 and len(model.prompts) == 1
        return 1
    assert recursia_protocol(store, exp).run_chunk(1, prepare=prepare, decide=proceed)["completed"]
    assert decisions[0] == ("baseline", "S2") and len(calls) == 6 and len(model.prompts) == 3
