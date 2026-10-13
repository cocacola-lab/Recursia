"""FR3 reachability and arm/board collision preview using the bundled FR3 MJCF.

The live FCI state supplies the IK seed and the TCP transform. This check is
read-only and is run before a physical phase's first motor command.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .config import WhiteboardConfig
from robot.franka.kinematics import default_model_path as arm_model_path


def _load_fr3_kinematics(cfg: WhiteboardConfig):
    from robot.franka.kinematics import ArmKinematics, default_model_path

    return ArmKinematics, default_model_path()


def _model_with_board(cfg: WhiteboardConfig, board: dict[str, Any] | None):
    import mujoco

    ArmKinematics, source = _load_fr3_kinematics(cfg)
    if board is None:
        # Generic pose control needs the robot model, not a perceived board.
        return ArmKinematics(xml_path=source, limit_margin_rad=0.052), mujoco
    corners = np.asarray(board["corners_base_m"], dtype=float)
    u = corners[1] - corners[0]
    u /= np.linalg.norm(u)
    n = np.asarray(board["normal_base"], dtype=float)
    v = np.cross(n, u)
    rotation = np.column_stack([u, v, n])
    if np.linalg.det(rotation) < 0.99:
        raise ValueError("board axes are not right-handed")
    center = corners.mean(axis=0) - n * 0.005
    quat = Rotation.from_matrix(rotation).as_quat()  # scipy XYZW -> MJCF WXYZ
    wxyz = quat[[3, 0, 1, 2]]
    values = lambda a: " ".join(f"{float(x):.9g}" for x in a)
    geom = (f'<geom name="observed_whiteboard" type="box" pos="{values(center)}" '
            f'quat="{values(wxyz)}" size="{values([board["width_m"]/2, board["height_m"]/2, 0.005])}" '
            'contype="1" conaffinity="1" margin="0.01"/>')
    xml = source.read_text(encoding="utf-8").replace("<worldbody>", "<worldbody>\n    " + geom, 1)
    with tempfile.TemporaryDirectory(prefix="fr3-board-model-") as directory:
        folder = Path(directory)
        (folder / "fr3.xml").write_text(xml, encoding="utf-8")
        (folder / "assets").symlink_to(source.parent / "assets", target_is_directory=True)
        arm = ArmKinematics(xml_path=folder / "fr3.xml", limit_margin_rad=0.052)
    return arm, mujoco


def _pose(step: dict[str, Any]) -> np.ndarray:
    target = np.eye(4)
    target[:3, 3] = step["ee_xyz_base_m"]
    target[:3, :3] = Rotation.from_quat(step["ee_quat_xyzw"]).as_matrix()
    return target


def _steps_for_phase(plan: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    if phase == "grasp":
        return plan["staging"] + plan["grasp"]
    if phase == "contact-probe":
        return plan["transfer"] + [plan["wipe"][0]] + plan["approach"] + [plan["wipe"][1]] + list(reversed(plan["approach"][:-1])) + [plan["wipe"][0]]
    if phase == "wipe":
        return plan["approach"] + plan["wipe"][1:] + plan["retreat"]
    raise ValueError(f"unknown phase {phase}")


def preview_phase(cfg: WhiteboardConfig, plan: dict[str, Any], phase: str,
                  state: dict[str, Any]) -> dict[str, Any]:
    """Check every commanded segment with <=15 mm / 5 degree IK samples."""
    arm, mujoco = _model_with_board(cfg, plan["board"])
    q = np.asarray(state["q"], dtype=float)
    actual = np.asarray(state["ee_pose_base"], dtype=float)
    if q.shape != (7,) or actual.shape != (4, 4) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(actual)):
        raise ValueError("invalid current FR3 q or EE pose for kinematic preview")
    tcp = arm.calibrate_tcp(q, actual)
    if np.linalg.norm(tcp[:3, 3]) > 0.25:
        raise ValueError("FCI EE-to-FR3-model transform exceeds 25 cm")
    model = arm.model
    data = arm.data
    board_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "observed_whiteboard")
    checked = 0
    max_pos_error = 0.0
    max_rot_error = 0.0

    def check_contacts(stage: str) -> None:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if (geom1 == board_geom or geom2 == board_geom or
                    (model.geom_contype[geom1] and model.geom_contype[geom2])) and contact.dist < 0.01:
                left = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                right = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                raise ValueError(f"{stage} arm collision/clearance failed: {left} vs {right} ({contact.dist:.3f} m)")

    check_contacts("start")
    previous = actual
    for step in _steps_for_phase(plan, phase):
        target = _pose(step)
        distance = np.linalg.norm(target[:3, 3] - previous[:3, 3])
        angle = (Rotation.from_matrix(previous[:3, :3]).inv() *
                 Rotation.from_matrix(target[:3, :3])).magnitude()
        samples = max(1, math.ceil(distance / 0.015), math.ceil(angle / np.deg2rad(5)))
        slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([previous[:3, :3], target[:3, :3]])))
        for fraction in np.linspace(0, 1, samples + 1)[1:]:
            goal = np.eye(4)
            goal[:3, 3] = previous[:3, 3] * (1 - fraction) + target[:3, 3] * fraction
            goal[:3, :3] = slerp([fraction]).as_matrix()[0]
            solution = arm.solve_ik(q, goal, tcp=tcp, max_iters=35)
            if not solution.converged:
                raise ValueError(f"{step['stage']} FR3 IK failed at sample {checked + 1}: "
                                 f"{solution.pos_err_m:.3f} m, {np.rad2deg(solution.rot_err_rad):.1f} deg")
            q = solution.q
            max_pos_error = max(max_pos_error, solution.pos_err_m)
            max_rot_error = max(max_rot_error, solution.rot_err_rad)
            check_contacts(step["stage"])
            checked += 1
        previous = target
    return {"schema": "aspire.fr3_whiteboard.kinematic_preview.v1",
            "phase": phase, "plan_id": plan["plan_id"], "samples": checked,
            "max_position_error_m": max_pos_error, "max_rotation_error_rad": max_rot_error,
            "model": str(arm_model_path()),
            "checked": "FR3 arm self-collision and observed board; other scene objects and held eraser geometry are not modeled"}


def preview_joint_path(cfg: WhiteboardConfig, board: dict[str, Any] | None,
                       start_q: np.ndarray, target_q: np.ndarray,
                       *, samples: int = 120) -> dict[str, Any]:
    """Check a sampled joint interpolation against limits, self, and board."""
    arm, mujoco = _model_with_board(cfg, board)
    start = np.asarray(start_q, dtype=float)
    target = np.asarray(target_q, dtype=float)
    if start.shape != (7,) or target.shape != (7,) or not (
            np.all(np.isfinite(start)) and np.all(np.isfinite(target))):
        raise ValueError("joint preview requires two finite 7-DOF configurations")
    if np.any(target < arm.q_min) or np.any(target > arm.q_max):
        raise ValueError("joint preview target is outside guarded model limits")
    board_geom = mujoco.mj_name2id(
        arm.model, mujoco.mjtObj.mjOBJ_GEOM, "observed_whiteboard")
    minimum_distance = float("inf")
    for index, fraction in enumerate(np.linspace(0, 1, samples + 1)):
        arm.data.qpos[:] = start * (1 - fraction) + target * fraction
        mujoco.mj_forward(arm.model, arm.data)
        for contact_index in range(arm.data.ncon):
            contact = arm.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            minimum_distance = min(minimum_distance, float(contact.dist))
            if (geom1 == board_geom or geom2 == board_geom or
                    (arm.model.geom_contype[geom1] and arm.model.geom_contype[geom2])) and contact.dist < 0.01:
                left = mujoco.mj_id2name(arm.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                right = mujoco.mj_id2name(arm.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                raise ValueError(
                    f"home path collision at sample {index}: {left} vs {right} "
                    f"({contact.dist:.3f} m)")
    return {"schema": "aspire.fr3_whiteboard.joint_preview.v1",
            "samples": samples + 1, "target_q": target.tolist(),
            "minimum_reported_contact_distance_m": (
                None if not np.isfinite(minimum_distance) else minimum_distance),
            "checked": ("sampled joint interpolation, guarded limits, FR3 self-collision" +
                        (" and conservative board box" if board is not None else ""))}


def solve_joint_pose(cfg: WhiteboardConfig, board: dict[str, Any] | None,
                     state: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    """Use FR3 DLS IK with the live joint seed and calibrated TCP.

    Only the endpoint becomes a motor command. The joint-path samples below
    are collision previews, not segmented movement commands.
    """
    arm, _ = _model_with_board(cfg, board)
    start = np.asarray(state["q"], dtype=float)
    actual = np.asarray(state["ee_pose_base"], dtype=float)
    target = _pose(step)
    if start.shape != (7,) or actual.shape != (4, 4) or not (
            np.all(np.isfinite(start)) and np.all(np.isfinite(actual))
            and np.all(np.isfinite(target))):
        raise ValueError("invalid measured state or target for FR3 IK")
    tcp = arm.calibrate_tcp(start, actual)
    if np.linalg.norm(tcp[:3, 3]) > 0.25:
        raise ValueError("FCI EE-to-FR3-model transform exceeds 25 cm")
    # A complete task endpoint needs a larger iteration budget than small
    # streaming updates, with the same DLS algorithm.
    solution = arm.solve_ik(start, target, tcp=tcp, max_iters=100,
                            tol_pos_m=0.0005, tol_rot_rad=0.002)
    if not solution.converged:
        raise ValueError(f"{step['stage']} FR3 IK failed: {solution.pos_err_m:.4f} m, "
                         f"{np.rad2deg(solution.rot_err_rad):.2f} deg")
    path = preview_joint_path(cfg, board, start, solution.q)
    return {"schema": "aspire.fr3_placement.ik_joint.v1", "stage": step["stage"],
            "solver": "FR3 ArmKinematics.solve_ik (DLS)",
            "seed_q": start.tolist(), "target_q": solution.q.tolist(),
            "tcp_from_live_state": tcp.tolist(), "iterations": solution.iters,
            "position_error_m": solution.pos_err_m,
            "rotation_error_rad": solution.rot_err_rad,
            "joint_path_preview": path, "motor_command": "one franky.JointMotion"}
