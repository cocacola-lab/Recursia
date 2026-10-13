"""RSI evidence, source-only topic merging, no-op auditing and frozen transfer."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .campaign import Experiment
from .generator import IsolatedGenerator, decode_response
from .learning import FINDINGS_SCHEMA, UPDATE_SCHEMA
from .promotion import Promotion
from .store import MethodStore, initialize, read_json, write_json, digest
from .test_rsi import TASKS, rollout


PATTERN = """# Grasp verification

Trigger: the Hand reports closure but the object did not move.

```python
observation = get_observation()
print(observation['views']['front']['images']['rgb'].shape)
```

Use fresh observation to distinguish closure from pickup; an untested new grasp
is a hypothesis. Evidence: the current source execution tool trace.
"""


class Learner:
    def __init__(self, updates=(), *, reason="Merge the observed failure pattern."):
        self.updates, self.reason = list(updates), reason
        self.calls = []

    def generate(self, prompt, *, schema=None, images=(), **kwargs):
        self.calls.append((prompt, schema, list(images)))
        if schema is None:
            return {"analysis": "initial", "code": "print('failure evidence')"}
        if "findings" in schema["properties"]:
            return {"analysis": "observed source failure", "findings":
                    "# Findings\nClosure did not prove pickup. Task incomplete; no validated new grasp. "
                    "Inspect the object in the source execution trace."}
        return {"analysis": "merge, not duplicate", "updates": self.updates, "reason": self.reason}


def own_store(tmp_path):
    return MethodStore(initialize(tmp_path / "campaign", TASKS), "aspire")


def ledger(store):
    return [json.loads(line) for line in (store.root / "skill_promotions.jsonl").read_text().splitlines()]


def test_failed_source_teaches_a_pattern_without_success_or_function_gate(tmp_path):
    own = own_store(tmp_path)
    learner = Learner([{"file": "grasp.md", "content": PATTERN}])
    robot_calls = []
    def failed(code, task, episode, folder):
        robot_calls.append(episode)
        return {"success": False, "task_completed": False, "model_finished": True,
                "error": None, "executed_code": code,
                "tool_trace": [{"tool": "close_gripper", "ok": True, "result": {"success": True}}]}
    result = Experiment(own, learner, failed).source(["S1"])
    assert robot_calls == ["source"] and len(learner.calls) == 3
    assert not result["tasks"][0]["success"] and result["tasks"][0]["updated"]
    assert own.skill_documents() == {"grasp.md": PATTERN}
    assert not (own.root / "skill_library.json").exists()
    assert own.state["source_model_calls"] == 3
    folder = Path(result["tasks"][0]["path"])
    assert "Task incomplete" in (folder / "findings.md").read_text()
    record = ledger(own)[0]
    assert record["status"] == "completed" and not record["no_op"]
    assert record["changed_skill_files"] == ["skills/grasp.md"]
    promotion = own.root / "skill_promotions" / record["promotion_id"]
    assert (promotion / "after/skills/grasp.md").read_text() == PATTERN
    assert "+# Grasp verification" in (promotion / "changes.patch").read_text()
    assert read_json(promotion / "record.json") == record
    for peer in ("sd-only", "recursia"):
        assert not (own.campaign / peer / "skills").exists()
        assert not (own.campaign / peer / "skill_promotions.jsonl").exists()


def test_merge_existing_topic_keeps_unmentioned_topics_and_before_snapshot(tmp_path):
    own = own_store(tmp_path)
    skills = own.root / "skills"
    skills.mkdir()
    (skills / "grasp.md").write_text("# Existing grasp pattern\n")
    (skills / "transport.md").write_text("# Transport\nKeep this lesson.\n")
    merged = "# Existing grasp pattern\n" + PATTERN
    learner = Learner([{"file": "grasp.md", "content": merged}])
    result = Experiment(own, learner, rollout).source(["S1"])
    assert "Existing grasp pattern" in learner.calls[2][0]
    assert own.skill_documents()["grasp.md"] == merged
    assert own.skill_documents()["transport.md"] == "# Transport\nKeep this lesson.\n"
    assert not result["tasks"][0]["learning_errors"]
    record = ledger(own)[0]
    assert record["changed_skill_files"] == ["skills/grasp.md"]
    before = own.root / "skill_promotions" / record["promotion_id"] / "before/skills/grasp.md"
    assert before.read_text() == "# Existing grasp pattern\n"
    assert own.state["knowledge_sha256"] == digest(own.context_knowledge())


def test_intentional_noop_preserves_library_and_appends_distinct_records(tmp_path):
    own = own_store(tmp_path)
    learner = Learner([], reason="No generalizable change beyond existing knowledge.")
    e = Experiment(own, learner, rollout)
    one = e.source(["S1"]); two = e.source(["S2"])
    records = ledger(own)
    assert len(records) == 2 and records[0]["promotion_id"] != records[1]["promotion_id"]
    assert all(r["no_op"] and r["status"] == "completed" and r["reason"] for r in records)
    assert not one["tasks"][0]["updated"] and not two["tasks"][0]["updated"]
    assert own.knowledge == "" and own.skill_documents() == {}
    assert own.state["source_rollouts"] == 2


def test_frozen_target_reads_source_topics_but_does_not_reflect_or_modify_ledger(tmp_path):
    own = own_store(tmp_path)
    learner = Learner([{"file": "grasp.md", "content": PATTERN}])
    Experiment(own, learner, rollout).source(["S1"])
    frozen = own.freeze()
    assert PATTERN in frozen["content"]
    assert frozen["skill_document_sha256"] == {"grasp.md": digest(PATTERN)}
    before = {str(p.relative_to(own.root)): p.read_bytes() for p in own.root.rglob("*") if p.is_file()}
    target_learner = Learner([{"file": "grasp.md", "content": "TARGET_LEAK"}])
    result = Experiment(own, target_learner, rollout).evaluate("T1")
    assert len(target_learner.calls) == 1 and PATTERN in target_learner.calls[0][0]
    context = (Path(result["run_dir"]) / "execution/context.md").read_text()
    assert PATTERN in context
    assert all((own.root / name).read_bytes() == contents for name, contents in before.items()
               if name not in ("state.json", "frozen.json"))
    assert own.state == {**json.loads(before["state.json"]), "frozen": False}
    assert read_json(Path(result["run_dir"]) / "snapshot.json") == frozen
    assert own.skill_documents()["grasp.md"] == PATTERN


def test_next_source_reuses_topics_without_reporting_an_unchanged_library_as_updated(tmp_path):
    own = own_store(tmp_path)
    Experiment(own, Learner([{"file": "grasp.md", "content": PATTERN}]), rollout).source(["S1"])
    learner = Learner([], reason="Existing pattern already covers this execution.")
    result = Experiment(own, learner, rollout).source(["S2"])
    assert PATTERN in learner.calls[0][0]
    folder = Path(result["tasks"][0]["path"])
    assert PATTERN in (folder / "execution/context.md").read_text()
    assert not result["tasks"][0]["updated"]
    assert ledger(own)[-1]["no_op"]


def test_old_frozen_snapshot_without_topic_hashes_remains_readable(tmp_path):
    own = own_store(tmp_path)
    Experiment(own, Learner(), rollout).source(["S1"])
    frozen = own.freeze()
    frozen.pop("skill_document_sha256")
    write_json(own.root / "frozen.json", frozen)
    assert own.snapshot() == frozen


def test_interrupted_multi_file_update_rolls_back_and_keeps_source_findings(tmp_path, monkeypatch):
    from . import promotion as module
    own = own_store(tmp_path)
    folder = own.root / "skills"
    folder.mkdir()
    (folder / "grasp.md").write_text("existing grasp lesson")
    replace = module.os.replace
    def interrupted(source, target):
        if Path(target) == folder / "transport.md":
            raise OSError("interrupted second skill write")
        return replace(source, target)
    monkeypatch.setattr(module.os, "replace", interrupted)
    learner = Learner([{"file": "grasp.md", "content": PATTERN},
                       {"file": "transport.md", "content": "new transport lesson"}])
    result = Experiment(own, learner, rollout).source(["S1"])
    assert own.skill_documents() == {"grasp.md": "existing grasp lesson"}
    assert {p.name for p in folder.iterdir()} == {"grasp.md"}
    record = ledger(own)[0]
    assert record["status"] == "failed" and record["no_op"] and record["findings_sha256"]
    assert "interrupted second skill write" in result["tasks"][0]["learning_errors"][0]


@pytest.mark.parametrize("dangling", [False, True])
def test_topic_directory_cannot_redirect_learning_to_a_peer(tmp_path, dangling):
    own = own_store(tmp_path)
    peer = own.campaign / "sd-only/skills"
    if not dangling:
        peer.mkdir()
        (peer / "grasp.md").write_text("peer lesson")
    (own.root / "skills").symlink_to(peer, target_is_directory=True)
    with pytest.raises(ValueError, match="belong to this method"):
        own.context_knowledge()
    if not dangling:
        assert (peer / "grasp.md").read_text() == "peer lesson"


@pytest.mark.parametrize("mutation", ["edit", "remove", "add"])
def test_frozen_skill_document_mutations_are_detected(tmp_path, mutation):
    own = own_store(tmp_path)
    Experiment(own, Learner([{"file": "grasp.md", "content": PATTERN}]), rollout).source(["S1"])
    own.freeze()
    path = own.root / "skills/grasp.md"
    if mutation == "edit":
        path.write_text("changed")
    elif mutation == "remove":
        path.unlink()
    else:
        (path.parent / "new.md").write_text("new target knowledge")
    with pytest.raises(ValueError, match="hash changed"):
        own.snapshot()


@pytest.mark.parametrize("name", ["../sd-only/skills.md", "/tmp/leak.md", "grasp.py", "nested/grasp.md"])
def test_bad_update_paths_do_not_write_even_an_earlier_valid_update(tmp_path, name):
    own = own_store(tmp_path)
    learner = Learner([{"file": "grasp.md", "content": PATTERN}, {"file": name, "content": "bad"}])
    result = Experiment(own, learner, rollout).source(["S1"])
    assert own.skill_documents() == {}
    assert result["tasks"][0]["learning_errors"]
    assert ledger(own)[0]["status"] == "failed" and ledger(own)[0]["no_op"]
    assert own.state["source_rollouts"] == 1


def test_bad_or_interrupted_authoring_preserves_findings_and_records_failure(tmp_path):
    own = own_store(tmp_path)
    own.meta["model_call_cap"] = 2  # initial + findings; updater cannot run
    learner = Learner([{"file": "grasp.md", "content": PATTERN}])
    result = Experiment(own, learner, rollout).source(["S1"])
    assert len(learner.calls) == 2 and own.state["source_model_calls"] == 2
    assert result["tasks"][0]["learning_errors"]
    record = ledger(own)[0]
    assert record["status"] == "failed" and record["findings_sha256"]
    assert "budget exhausted" in record["reason"] and record["no_op"]
    assert own.skill_documents() == {} and own.state["source_rollouts"] == 1


def test_dry_run_never_promotes_preview_findings(tmp_path):
    own = own_store(tmp_path)
    learner = Learner([{"file": "grasp.md", "content": PATTERN}])
    result = Experiment(own, learner, rollout, dry_run=True).source(["S1"])
    assert len(learner.calls) == 1 and not result["tasks"][0]["updated"]
    assert not (own.root / "skill_promotions.jsonl").exists()
    assert own.skill_documents() == {}


def test_reflection_reuses_last_recorded_views_and_contains_actual_repair_evidence(tmp_path):
    own = own_store(tmp_path)
    learner = Learner()
    def failed(code, task, episode, folder):
        for i in (1, 2):
            turn = folder / "turns" / f"turn_{i:03d}"
            write_json(turn / "step.json", {"code": "print('repair')", "info_step": {
                "stdout": "last measured geometry", "stderr": "ValueError: failed pickup" if i == 1 else ""},
                "decision_prompt": "DO_NOT_REPEAT_FULL_CONTEXT"})
            write_json(turn / "response.json", {"analysis": "repair cause", "response": "FINISH"})
            views = {}
            for view in ("front", "wrist"):
                image = turn / (view + ".png")
                image.write_bytes(b"fake recorded image")
                views[view] = {"rgb_path": str(image), "captured_at": str(i), "frame": view}
            write_json(turn / "observation.json", {"views": views})
        return {"success": False, "task_completed": False, "model_finished": True, "executed_code": code}
    Experiment(own, learner, failed).source(["S1"])
    prompt, _, images = learner.calls[1]
    assert "failed pickup" in prompt and "repair cause" in prompt and "last measured geometry" in prompt
    assert "DO_NOT_REPEAT_FULL_CONTEXT" not in prompt
    assert len(images) == 2 and all("turn_002" in str(image) for image in images)
    assert not learner.calls[2][2]  # skill author needs evidence, not a new camera capture


def test_promotion_snapshots_and_patch_verify_and_detect_corruption(tmp_path):
    own = own_store(tmp_path)
    source = own.root / "source/run/S1"
    source.mkdir(parents=True)
    promotion = Promotion(own, "S1", source)
    promotion.apply([{"file": "grasp.md", "content": "# No trailing newline"}])
    record = promotion.finish(reason="Reusable source lesson.")
    assert record["patch_sha256"]
    assert "No newline at end of file" in (promotion.folder / "changes.patch").read_text()
    promotion.verify()
    (promotion.folder / "after/skills/grasp.md").write_text("corrupt")
    with pytest.raises(ValueError, match="snapshot hash"):
        promotion.verify()


@pytest.mark.parametrize("schema,value", [
    (FINDINGS_SCHEMA, {"analysis": "observed", "findings": "source findings"}),
    (UPDATE_SCHEMA, {"analysis": "merge", "updates": [{"file": "grasp.md", "content": PATTERN}], "reason": "source evidence"}),
])
@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_real_transport_decodes_findings_and_skill_updates(tmp_path, monkeypatch, schema, value, provider):
    def process(command, **kwargs):
        if provider == "codex":
            Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(value))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"structured_output": value}), stderr="")
    monkeypatch.setattr("cap.agent.rsi.generator.subprocess.run", process)
    monkeypatch.setattr("cap.agent.rsi.generator._resolve_codex_executable", lambda: "mock-codex")
    assert IsolatedGenerator(provider).generate("source learning only", timeout_s=5, schema=schema) == value


@pytest.mark.parametrize("schema,value", [
    (FINDINGS_SCHEMA, {"analysis": "ok", "findings": []}),
    (UPDATE_SCHEMA, {"analysis": "ok", "updates": "grasp.md", "reason": "bad"}),
    (UPDATE_SCHEMA, {"analysis": "ok", "updates": [{"file": "grasp.md", "content": 1}], "reason": "bad"}),
])
def test_learning_decoder_rejects_wrong_field_types(schema, value):
    with pytest.raises(ValueError):
        decode_response(json.dumps(value), schema)
