"""IK and joint-command checks without connecting to hardware."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap.saved_scripts.fr3_runtime.kinematics import _load_fr3_kinematics, solve_joint_pose
from cap.saved_scripts.fr3_runtime.robot import Fr3WipeRobot, MotionResult


def test_ik_joint_reaches_pose_using_live_tcp_and_rejects_unreachable_goal():
    pytest.importorskip("mujoco")
    cfg = SimpleNamespace()
    cls, model = _load_fr3_kinematics(cfg)
    arm = cls(model, limit_margin_rad=0.052)
    q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("z", -np.pi/4).as_matrix()
    tcp[2, 3] = 0.1034
    actual = arm.fk(q, tcp)
    target = actual.copy()
    target[0, 3] += 0.1
    step = {"stage": "test", "ee_xyz_base_m": target[:3, 3].tolist(),
            "ee_quat_xyzw": Rotation.from_matrix(target[:3, :3]).as_quat().tolist()}
    board = {"corners_base_m": [[0.8, -0.05, 0], [0.9, -0.05, 0],
                                [0.9, 0.05, 0], [0.8, 0.05, 0]],
             "normal_base": [0, 0, 1], "width_m": 0.1, "height_m": 0.1}
    state = {"q": q.tolist(), "ee_pose_base": actual.tolist()}
    result = solve_joint_pose(cfg, board, state, step)
    predicted = arm.fk(np.array(result["target_q"]), tcp)
    assert np.linalg.norm(predicted[:3, 3] - target[:3, 3]) < 0.0005
    assert (Rotation.from_matrix(predicted[:3, :3]).inv()
            * Rotation.from_matrix(target[:3, :3])).magnitude() < 0.002
    assert np.allclose(result["tcp_from_live_state"], tcp)
    assert result["joint_path_preview"]["samples"] == 121
    with pytest.raises(ValueError, match="IK failed"):
        solve_joint_pose(cfg, board, state, {**step, "ee_xyz_base_m": [3, 0, 0.5]})


def test_pose_executor_uses_one_joint_motion_and_keeps_pose_verification(monkeypatch):
    q = [0, -0.7, 0, -2, 0, 1.5, 0.7]
    pose = np.eye(4)
    pose[:3, 3] = [0.3, 0, 0.5]
    state = {"q": q, "ee_pose_base": pose.tolist()}
    cfg = SimpleNamespace(pose_control="ik_joint", workspace_min_m=np.array([0, -1, 0]),
                          workspace_max_m=np.ones(3), force_limit_n=12, grasp_force_limit_n=20)
    calls = []
    fake_franky = SimpleNamespace(JointMotion=lambda target: ("JointMotion", target))
    arm = Fr3WipeRobot(cfg, None, None, fake_franky)
    monkeypatch.setattr(arm, "_ready", lambda: state)
    monkeypatch.setattr(arm, "state", lambda: state)
    def run(motion, **kwargs):
        calls.append((motion, kwargs))
        target_pose = pose.copy()
        target_pose[0, 3] = 0.4
        state["ee_pose_base"] = target_pose.tolist()
        return MotionResult(kwargs["stage"], [0.3, 0, 0.5], [0.4, 0, 0.5], 0, 1)
    monkeypatch.setattr(arm, "_run_motion", run)
    step = {"stage": "agent_move", "ee_xyz_base_m": [0.4, 0, 0.5],
            "ee_quat_xyzw": [0, 0, 0, 1], "joint_target_q": q}
    assert arm.move_pose(step).after_xyz_m == [0.4, 0, 0.5]
    assert len(calls) == 1 and calls[0][0] == ("JointMotion", q)
    assert calls[0][1]["force_limit_n"] == 12
    with pytest.raises(ValueError, match="seven-joint target"):
        arm.move_pose({**step, "joint_target_q": [0]*6})
    with pytest.raises(RuntimeError, match="joint margin"):
        arm.move_pose({**step, "joint_target_q": [0]*7})
    assert len(calls) == 1
    # A joint command finishing does not bypass the original EE pose check.
    with pytest.raises(RuntimeError, match="EE position error"):
        arm.move_pose({**step, "ee_xyz_base_m": [0.5, 0, 0.5]})


def test_return_pose_preview_uses_bundled_arm_model():
    pytest.importorskip('mujoco')
    from robot.franka.kinematics import ArmKinematics, tcp_from_xyz_rpy
    from tools.fr3_return_pose import DEFAULT_Q_RAD, _preview

    arm = ArmKinematics()
    q = np.asarray(DEFAULT_Q_RAD)
    tcp = tcp_from_xyz_rpy([0, 0, 0.1034], [0, 0, -np.pi/4])
    ee = arm.fk(q, tcp)
    result = _preview(q, q, ee, ee)
    assert result['joint_path_samples'] == 2
    assert result['saved_pose_model_position_error_m'] < 1e-12
    assert result['saved_pose_model_angle_error_deg'] < 1e-10
