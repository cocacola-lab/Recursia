"""Independent ASPIRE primitives: sensing has no arm state, movement no SAM3."""
from dataclasses import replace
import numpy as np

from cap.saved_scripts.fr3_placement.test_placement import (
    station, scene, FakeSam, FakeArm, FakeVideo,
)
from .runtime import FiveToolRuntime

TASK = {"objects": [{"name": "mug", "prompt": "mug"}], "goals": []}


class GraspNet:
    def predict(self, capture, mask):
        assert np.any(mask)
        return {"grasps": np.eye(4)[None], "scores": np.array([.9]),
                "widths": np.array([.004]), "contact_pts": np.zeros((1,3))}


def test_sample_without_robot_connection_or_object_volume(station, tmp_path, monkeypatch):
    def no_arm(_):
        raise AssertionError("grasp sensing connected to robot")
    runtime = FiveToolRuntime(station, tmp_path/'sample', task=TASK, arm_factory=no_arm,
                             sam3=FakeSam(), camera_fn=lambda _: scene()[0], graspnet=GraspNet())
    monkeypatch.setattr(runtime, "_detect", lambda _: (_ for _ in ()).throw(
        AssertionError("grasp sensing required a visible bounding volume")))
    position, quaternion = runtime.sample_grasp_pose('mug')
    assert runtime.arm is None
    assert position.shape == (3,) and quaternion.shape == (4,)
    assert not runtime.physical_motion_commanded
    runtime.close()


def test_sampling_allowed_when_closed_hand_reports_grasped(station, tmp_path):
    runtime = FiveToolRuntime(station, tmp_path/'closed', task=TASK, dry_run=True,
                             sam3=FakeSam(), camera_fn=lambda _: scene()[0], graspnet=GraspNet())
    runtime.close_gripper()
    assert runtime.arm.held and runtime.arm.width == 0
    runtime.arm.state = lambda: (_ for _ in ()).throw(AssertionError("sampling read closed Hand"))
    assert len(runtime.sample_grasp_pose('mug')) == 2
    runtime.close()


def test_move_without_camera_segmentation_or_board(station, tmp_path, monkeypatch):
    cfg = replace(station, pose_control="ik_joint")
    def no_camera(_):
        raise AssertionError("motion captured the camera")
    class NoSam:
        def segment_all(self, *args):
            raise AssertionError("motion called SAM3")
    calls = []
    def solve(config, board, state, step):
        assert board is None
        calls.append(step)
        return {"target_q": state["q"]}
    monkeypatch.setattr("cap.saved_scripts.fr3_placement.runtime.solve_joint_pose", solve)
    runtime = FiveToolRuntime(cfg, tmp_path/'move', task=TASK, camera_fn=no_camera,
                             sam3=NoSam(), arm_factory=FakeArm, recorder_factory=FakeVideo)
    assert runtime.goto_pose([.45, 0, .25], [1, 0, 0, 0]) == {"success": True}
    assert calls[0]["stage"] == "agent_move"
    assert not runtime.arm.monitor_grasp_state
    runtime.close()


def test_close_is_independent_force_command_not_object_verification(station, tmp_path):
    runtime = FiveToolRuntime(station, tmp_path/'close', task=TASK, arm_factory=FakeArm,
                             recorder_factory=FakeVideo)
    arm, calls = runtime._ensure_arm(), []
    arm.gripper.grasp = lambda *args: calls.append(args) or True
    assert runtime.close_gripper() == {"success": True}
    assert calls == [(0., station.grasp_speed_m_s, station.grasp_force_n,
                      station.grasp_epsilon_m, arm.gripper.max_width)]
    assert not arm.held  # Command success does not fabricate an attachment.
    assert not (runtime.run_dir/'held_object.json').exists()
    runtime.close()
