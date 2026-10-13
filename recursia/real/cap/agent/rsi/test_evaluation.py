"""Visual placement scoring stays independent of actions and geometric gates."""
import copy
import json
from types import SimpleNamespace

import pytest

from cap.saved_scripts.fr3_placement.test_placement import station
from .evaluation import is_cup_plate_task, PLACEMENT_SCHEMA
from .generator import IsolatedGenerator, decode_response
from .observation import image_paths
from .runtime import FiveToolRuntime
from .store import read_json
from .test_observation import configure_wrist, frame
from .trial_adapter import ModelBudget


TASK = {"instruction": "put the blue mug on the right plate", "objects": [
    {"name": "blue mug", "prompt": "blue mug"},
    {"name": "right plate", "prompt": "plate", "side": "right"}],
    "goals": [{"relation": "on", "object": "blue mug", "target": "right plate"}]}


def verdict(task=TASK, status="SUCCESS"):
    return {"summary": "Current scene evidence", "goals": [
        {"object": g["object"], "target": g["target"], "status": status,
         "evidence": "Cup is released on the right plate" if status == "SUCCESS" else "Not confirmed"}
        for g in task["goals"]]}


class NoSam:
    def segment_all(self, *args):
        pytest.fail("visual placement scoring must not call SAM3 or fit bounding boxes")


def runtime(station, tmp_path, vision):
    configure_wrist(station)
    return FiveToolRuntime(station, tmp_path / "evaluate", task=TASK, sam3=NoSam(),
        arm_factory=lambda cfg: pytest.fail("visual judge must not require gripper width or FCI"),
        camera_fn=frame, vision_generator=vision)


@pytest.mark.parametrize("status,success", [("SUCCESS", True), ("FAILURE", False), ("UNSURE", False)])
def test_visual_status_uses_same_feedback_images_without_geometry(station, tmp_path, status, success):
    seen = []
    def judge(observation, task, **kwargs):
        assert task == TASK
        assert all(path.is_file() for path in image_paths(observation))
        seen.append(observation)
        return verdict(status=status)
    instance = runtime(station, tmp_path, SimpleNamespace(evaluate_placement=judge))
    feedback = instance.capture_feedback()
    instance.camera_fn = lambda serial: pytest.fail("judge recaptured an already fresh feedback frame")
    result = instance.evaluate_goal()
    assert seen == [feedback]
    assert result["status"] == status and result["success"] is success
    assert result["goals"][0]["evidence"] and result["method"] == "vlm_reward"
    assert instance.arm is None and instance.vision_model_calls == 1
    folder = image_paths(feedback)[0].parent.parent
    assert read_json(folder / "placement_verification.json") == result
    assert "right plate" in (folder / "placement_verification_prompt.txt").read_text()
    instance.close()


def test_repeated_direct_evaluation_gets_new_views(station, tmp_path):
    instance = runtime(station, tmp_path, SimpleNamespace(evaluate_placement=lambda *a, **kw: verdict()))
    first, second = instance.evaluate_goal(), instance.evaluate_goal()
    assert first["snapshot"] != second["snapshot"]
    assert first["success"] and second["success"]
    instance.close()


def test_relative_run_directory_saves_verdict(station, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    instance = runtime(station, tmp_path, SimpleNamespace(evaluate_placement=lambda *a, **kw: verdict()))
    instance.run_dir = instance.run_dir.relative_to(tmp_path)
    result = instance.evaluate_goal()
    assert result["success"]
    assert read_json(instance.run_dir / "verification.json") == result
    instance.close()


@pytest.mark.parametrize("response", [None, {"summary": "", "goals": []},
    {"summary": "", "goals": [{"object": "blue mug", "target": "left plate", "status": "SUCCESS", "evidence": ""}]}])
def test_missing_or_wrong_goal_verdict_is_unsure(station, tmp_path, response):
    instance = runtime(station, tmp_path, SimpleNamespace(evaluate_placement=lambda *a, **kw: response))
    result = instance.evaluate_goal()
    assert not result["success"] and result["status"] == "UNSURE"
    assert "error" in result
    instance.close()


def test_judge_failure_is_recorded_without_fallback_or_motion(station, tmp_path):
    def failed(*args, **kwargs):
        raise TimeoutError("visual judge timeout")
    instance = runtime(station, tmp_path, SimpleNamespace(evaluate_placement=failed))
    result = instance.evaluate_goal()
    assert result["status"] == "UNSURE" and not result["success"]
    assert "timeout" in result["error"] and not instance.physical_motion_commanded
    instance.close()


def test_all_cups_must_be_on_their_own_plates(station, tmp_path):
    instance = runtime(station, tmp_path, None)
    task = copy.deepcopy(TASK)
    task["objects"] += [{"name": "white mug", "prompt": "white mug"},
                        {"name": "left plate", "prompt": "plate", "side": "left"}]
    task["goals"].append({"relation": "on", "object": "white mug", "target": "left plate"})
    instance.task = task
    response = verdict(task)
    response["goals"][1].update(status="FAILURE", evidence="White mug is on the wrong plate")
    instance.vision_generator = SimpleNamespace(evaluate_placement=lambda *a, **kw: response)
    result = instance.evaluate_goal()
    assert result["status"] == "FAILURE" and not result["success"]
    assert [g["success"] for g in result["goals"]] == [True, False]
    instance.close()


def test_dry_run_visual_success_is_not_physical_success(station, tmp_path):
    instance = runtime(station, tmp_path, SimpleNamespace(evaluate_placement=lambda *a, **kw: verdict()))
    for view, serial in [("front", "mock-camera"), ("wrist", "mock-wrist")]:
        frame(serial).save(tmp_path / "snapshot" / view)
    instance.dry_run, instance.snapshot = True, tmp_path / "snapshot"
    instance.camera_fn = lambda serial: pytest.fail("dry run opened a camera")
    result = instance.evaluate_goal()
    assert result["status"] == "SUCCESS" and not result["success"]
    instance.close()


def test_other_relations_and_support_objects_keep_existing_evaluator():
    assert is_cup_plate_task(TASK)
    task = copy.deepcopy(TASK)
    task["goals"][0]["relation"] = "in"
    assert not is_cup_plate_task(task)
    task["goals"][0].update(relation="on", object="small basin")
    assert not is_cup_plate_task(task)


def test_isolated_judge_gets_images_and_task_only(tmp_path, monkeypatch):
    generator = IsolatedGenerator("codex")
    observation = {"views": {view: {"rgb_path": str(tmp_path / f"{view}.png")}
                             for view in ("front", "wrist")}}
    def generate(prompt, **kwargs):
        assert "FRONT image" in prompt and "handle may extend" in prompt
        assert "skills" not in prompt and "executed" not in prompt and str(tmp_path) not in prompt
        assert kwargs["images"] == image_paths(observation)
        assert kwargs["schema"] == PLACEMENT_SCHEMA
        return verdict()
    monkeypatch.setattr(generator, "generate", generate)
    assert generator.evaluate_placement(observation, TASK, timeout_s=5) == verdict()
    assert decode_response(json.dumps({"structured_output": verdict()}), PLACEMENT_SCHEMA) == verdict()


def test_judging_counts_as_visual_not_action_calls(tmp_path):
    generator = SimpleNamespace(evaluate_placement=lambda *a, **kw: verdict())
    budget = ModelBudget(generator, 1, tmp_path)
    assert budget.evaluate_placement("pixels", TASK) == verdict()
    assert budget.calls == 1 and budget.action_calls == 0
    with pytest.raises(RuntimeError, match="budget"):
        budget.evaluate_placement("pixels", TASK)
