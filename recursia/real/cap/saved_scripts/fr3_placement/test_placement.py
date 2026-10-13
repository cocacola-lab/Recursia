"""Offline task tests: no FCI connection, camera pipeline or SAM3 server."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from cap.agent.code_validation import validate_code
from cap.saved_scripts.fr3_placement.config import load_config
from cap.saved_scripts.fr3_placement.perception import describe_object, grasp_candidate, placement_geometry
from cap.saved_scripts.fr3_placement.runtime import DryArm, PhysicalFault, PlacementRuntime
from cap.saved_scripts.fr3_runtime.camera import Capture


@pytest.fixture
def station(tmp_path):
    root = tmp_path / "station"
    root.mkdir()
    hardware = root / "hardware.yaml"
    hardware.write_text(yaml.safe_dump({"hardware": {"franka": {"ip": "mock"},
                                                  "cameras": {"ext": {"serial": "mock-camera"}}}}))
    (root / "safety.yaml").write_text(yaml.safe_dump({"safety": {
        "cart_workspace_min_xyz": [0.1, -0.6, 0.02], "cart_workspace_max_xyz": [0.9, 0.6, 1.1]}}))
    station_path = root / "station.yaml"
    station_path.write_text(yaml.safe_dump({"calibration_id": "synthetic", "external_camera": {
        "serial": "mock-camera", "optical_frame": "color",
        "T_base_camera_optical": [[1, 0, 0, 0.4], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]]}}))
    path = root / "task.yaml"
    path.write_text(yaml.safe_dump({"schema": "aspire.fr3_eraser_place.v1",
                                   "hardware_config": str(hardware), "station_model": str(station_path)}))
    return load_config(path)


def scene():
    rgb = np.full((240, 320, 3), 240, dtype=np.uint8)
    board = np.zeros((240, 320), dtype=bool)
    board[60:180, 80:240] = True
    eraser = np.zeros_like(board)
    eraser[110:125, 150:180] = True
    depth = np.full(board.shape, 0.53, dtype=np.float32)
    depth[board] = 0.50
    depth[eraser] = 0.48
    capture = Capture(rgb, depth, np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1]]),
                      "mock-camera", "synthetic")
    return capture, board, eraser


class FakeSam:
    def segment_all(self, rgb, prompt):
        _, board, eraser = scene()
        return [(board if prompt == "whiteboard" else eraser, 0.99)]


class FakeVideo:
    def __init__(self, *args): pass
    def start(self): return self
    def check(self): pass
    def stop(self): pass


class FakeArm(DryArm):
    def __init__(self, cfg):
        super().__init__(cfg)
        def move(width, speed):
            self.width, self.held = width, False
            return True
        def grasp(width, speed, force, inner, outer):
            self.width, self.held = width, True
            return True
        self.gripper = SimpleNamespace(max_width=0.076, move=move, grasp=grasp)


def runtime(cfg, root, **kwargs):
    return PlacementRuntime(cfg, root, camera_fn=lambda serial: scene()[0], sam3=FakeSam(),
                            arm_factory=FakeArm, recorder_factory=FakeVideo, **kwargs)


def test_config_has_station_settings_without_wipe_or_ink_inputs(station):
    assert station.relative_dynamics == (0.05, 0.05, 0.04)
    assert station.grasp_epsilon_m == 0.010
    assert not hasattr(station, "wipe_quat_xyzw")
    assert not hasattr(station, "ink_gray_threshold")


def test_perception_and_support_estimate_without_ink(station):
    capture, board_mask, eraser_mask = scene()
    board = describe_object(capture, board_mask, station, name="whiteboard", score=1)
    eraser = describe_object(capture, eraser_mask, station, name="whiteboard eraser", score=1)
    assert eraser["bottom_source"] == "observed_support_plane"
    assert np.isclose(eraser["bottom_position_base_m"][2], 0.50, atol=0.002)
    assert np.isclose(eraser["extent_m"][2], 0.02, atol=0.002)
    evidence = placement_geometry(board, eraser)
    assert evidence["on_board"]
    shifted = {**eraser, "corners_base_m": (np.array(eraser["corners_base_m"]) + [0.3, 0, 0]).tolist()}
    assert not placement_geometry(board, shifted)["on_board"]
    unsupported = {**eraser, "bottom_position_base_m": None}
    assert not placement_geometry(board, unsupported)["on_board"]


def test_grasp_candidate_chooses_nearest_equivalent_yaw(station):
    capture, _, mask = scene()
    obj = describe_object(capture, mask, station, name="whiteboard eraser", score=1)
    first = grasp_candidate(obj, [1, 0, 0, 0], np.zeros(3))
    assert np.allclose(first["position_base_m"], obj["grasp_point_base_m"])
    q = np.array(first["quaternion_xyzw"])
    again = grasp_candidate(obj, q, np.zeros(3))
    assert (Rotation.from_quat(q).inv() * Rotation.from_quat(again["quaternion_xyzw"])).magnitude() < 1e-6
    assert 0.005 <= first["width_m"] <= 0.075


def test_dry_run_never_connects_and_cannot_claim_physical_success(station, tmp_path):
    def forbidden(*args): raise AssertionError("physical arm must not connect")
    rt = runtime(station, tmp_path / "run", dry_run=True)
    rt.arm_factory = forbidden
    rt.detect_object("whiteboard")
    grasp = rt.sample_grasp_pose()
    rt.goto_pose(grasp["position_base_m"], grasp["quaternion_xyzw"], label="grasp")
    assert rt.close_gripper(grasp["width_m"])["success"]
    assert rt.get_held_object()["offset_source"] == "RGB-D support estimate at grasp"
    rt.open_gripper()
    check = rt.verify_placement()
    assert check["on_board"] and not check["success"]
    assert not rt.physical_motion_commanded
    rt.close()


def test_ik_joint_runtime_binds_solved_endpoint_before_moving(station, tmp_path, monkeypatch):
    cfg = replace(station, pose_control="ik_joint")
    rt = PlacementRuntime(cfg, tmp_path / "joint-run", arm_factory=lambda _cfg: DryArm(cfg))
    rt.objects["whiteboard"] = {"observed": True}
    rt._start_video = lambda: None
    q = [0, -0.7, 0, -2, 0, 1.5, 0.7]
    seen = []
    def solve(_cfg, board, state, step):
        assert board == {"observed": True}
        seen.append((state, dict(step)))
        return {"target_q": q}
    monkeypatch.setattr("cap.saved_scripts.fr3_placement.runtime.solve_joint_pose", solve)
    rt.goto_pose([0.4, 0, 0.5], [1, 0, 0, 0])
    events = [json.loads(line) for line in (rt.run_dir / "trace.jsonl").read_text().splitlines()]
    assert events[-1]["target"]["joint_target_q"] == q
    assert len(seen) == 1
    def fail(*args):
        raise ValueError("IK failed")
    monkeypatch.setattr("cap.saved_scripts.fr3_placement.runtime.solve_joint_pose", fail)
    with pytest.raises(ValueError, match="IK failed"):
        rt.goto_pose([0.5, 0, 0.5], [1, 0, 0, 0])
    assert rt.move_count == 1


def test_pose_control_config_rejects_unknown_backend(station):
    raw = yaml.safe_load(station.path.read_text())
    raw["motion"] = {"pose_control": "unknown"}
    station.path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="pose_control"):
        load_config(station.path)


def test_release_and_goal_verification_are_independent_tools(station, tmp_path, monkeypatch):
    monkeypatch.setattr("cap.saved_scripts.fr3_placement.runtime.preview_phase", lambda *args: {"mock": True})
    rt = runtime(station, tmp_path / "run")
    assert rt.verify_placement()["success"]
    candidate = rt.sample_grasp_pose()
    rt.close_gripper(candidate["width_m"])
    assert rt.goal_evidence is None  # A previous success is invalidated by a new action.
    assert rt.get_held_object() is not None
    assert not rt.verify_placement()["success"]  # Tool still held despite visual overlap.
    rt.open_gripper()
    assert rt.verify_placement()["success"]
    assert rt.get_held_object() is None
    rt.close()


def test_grasp_failure_returns_feedback_without_throwing(station, tmp_path):
    rt = runtime(station, tmp_path / "run")
    arm = rt._ensure_arm()
    arm.gripper.grasp = lambda *args: False
    result = rt.close_gripper(0.04)
    assert not result["success"] and not result["hand_is_grasped"]
    assert rt.get_held_object() is None
    rt.close()


def test_thin_contact_width_reaches_hand_unchanged(station, tmp_path):
    rt = runtime(station, tmp_path / "thin")
    result = rt.close_gripper(.0039721765)
    assert result["success"]
    assert rt.arm.width == .0039721765
    rt.close()


def test_close_rejects_width_larger_than_actual_hand_without_grasp_call(station, tmp_path):
    rt = runtime(station, tmp_path / "wide")
    arm = rt._ensure_arm()
    arm.gripper.grasp = lambda *args: pytest.fail("unsupported width reached hand")
    with pytest.raises(ValueError, match="measured hand opening"):
        rt.close_gripper(.077)
    rt.close()


def test_new_task_can_start_with_held_eraser_without_opening(station, tmp_path):
    rt = runtime(station, tmp_path / "run")
    arm = rt._ensure_arm()
    arm.held, arm.width = True, 0.044
    held = rt.get_held_object()
    assert "provisional" in held["offset_source"]
    assert held["bottom_in_ee_m"] == station.fallback_bottom_in_ee_m.tolist()
    assert arm.held and arm.width == 0.044
    rt.close()


def test_fault_stops_even_when_policy_can_retry(station, tmp_path):
    rt = runtime(station, tmp_path / "run")
    rt._ensure_arm().state = lambda: {"has_errors": True, "current_errors": "controller_fault"}
    with pytest.raises(PhysicalFault, match="recovery required"):
        rt.goto_pose([0.4, 0, 0.3], [1, 0, 0, 0])
    assert rt.hardware_fault
    rt.close()






@pytest.mark.parametrize("code", ['import os', 'open("x")', 'np.save("x", [1])'])
def test_policy_does_not_get_file_process_tools(code):
    with pytest.raises(ValueError): validate_code(code)






def test_explicit_local_safety_file_replaces_legacy_sibling(station):
    raw = yaml.safe_load(station.path.read_text())
    raw['safety_config'] = 'safety.local.yaml'
    station.path.write_text(yaml.safe_dump(raw))
    (station.path.parent / 'safety.local.yaml').write_text(yaml.safe_dump({'safety': {
        'cart_workspace_min_xyz': [0.2, -0.4, 0.1],
        'cart_workspace_max_xyz': [0.8, 0.4, 0.9]}}))
    (station.path.parent / 'safety.yaml').unlink()
    cfg = load_config(station.path)
    np.testing.assert_array_equal(cfg.workspace_min_m, [0.2, -0.4, 0.1])
    np.testing.assert_array_equal(cfg.workspace_max_m, [0.8, 0.4, 0.9])


def test_return_pose_uses_manipulation_config_dynamics(station):
    from tools.fr3_return_pose import _hardware

    raw = yaml.safe_load(station.path.read_text())
    raw['motion'] = {'relative_dynamics': [0.2, 0.05, 0.04]}
    station.path.write_text(yaml.safe_dump(raw))
    # No camera calibration, safety file, or teleoperation config is needed.
    Path(raw['station_model']).unlink()
    (station.path.parent / 'safety.yaml').unlink()
    raw['hardware_config'] = station.hardware_config_path.name
    station.path.write_text(yaml.safe_dump(raw))
    assert _hardware(station.path) == ('mock', (0.2, 0.05, 0.04))
    raw['motion']['relative_dynamics'] = [float('nan'), 0.05, 0.04]
    station.path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='relative dynamics'):
        _hardware(station.path)
