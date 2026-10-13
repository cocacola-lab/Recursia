#!/usr/bin/env python3
"""Save an FR3 joint pose, preview a return, or return to a target pose.

By default, return to the Franka example configuration:
J1..J7 = [0, -45, 0, -135, 0, 90, 45] degrees, then open the gripper.
Use --pose to return to a saved bookmark instead.
Wait 3 seconds before execution; use --delay-s to change or disable the wait.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cap.saved_scripts.fr3_runtime.hardware import active_fci_clients  # noqa: E402
from robot.franka import Fr3Client  # noqa: E402
from robot.franka.fr3 import FR3_Q_MAX, FR3_Q_MIN, LIMIT_MARGIN_RAD  # noqa: E402

DEFAULT_POSE = ROOT / "config/return_pose.local.json"
DEFAULT_Q_RAD = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4,
                 0.0, math.pi / 2, math.pi / 4)
DEFAULT_CONFIG = ROOT / "config/rsi_fr3_station.local.yaml"
STILL_RAD_S = 0.02
GRIPPER_SPEED_M_S = 0.2
GRIPPER_TOLERANCE_M = 0.003


def _hardware(config: Path) -> tuple[str, tuple[float, float, float]]:
    # Returning to joint angles does not require camera calibration.
    config = config.expanduser().resolve()
    raw = yaml.safe_load(config.read_text())
    hardware_path = Path(raw["hardware_config"]).expanduser()
    if not hardware_path.is_absolute():
        hardware_path = config.parent / hardware_path
    hardware = yaml.safe_load(hardware_path.read_text())["hardware"]
    ip = hardware["franka"]["ip"]
    factors = tuple(float(value) for value in raw.get("motion", {}).get(
        "relative_dynamics", [0.20, 0.05, 0.04]))
    if not isinstance(ip, str) or not ip.strip() or len(factors) != 3 or not all(
            math.isfinite(value) and 0 < value <= 1 for value in factors):
        raise ValueError("invalid robot IP or motion relative dynamics")
    return ip, factors


def _check_q(q: object, name: str) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain seven finite joint angles")
    if np.any(value <= FR3_Q_MIN + LIMIT_MARGIN_RAD) or np.any(
            value >= FR3_Q_MAX - LIMIT_MARGIN_RAD):
        raise ValueError(f"{name} reaches the guarded FR3 joint limits")
    return value


def _load_pose(path: Path | None, robot_ip: str) -> tuple[dict, np.ndarray, np.ndarray | None]:
    if path is None:
        return ({"captured_at": None, "source": "Franka official example joint configuration"},
                _check_q(DEFAULT_Q_RAD, "default q"), None)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema") != "aspire.fr3.pose_bookmark.v1":
        raise ValueError("unrecognized FR3 pose bookmark")
    if saved.get("robot_ip") != robot_ip:
        raise ValueError("bookmark belongs to a different robot IP")
    q = _check_q(saved.get("q_rad"), "saved q")
    ee = np.asarray(saved.get("T_base_ee"), dtype=np.float64)
    if ee.shape != (4, 4) or not np.all(np.isfinite(ee)) or not np.allclose(
            ee[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(
                ee[:3, :3].T @ ee[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError("bookmark has an invalid O_T_EE transform")
    return saved, q, ee


def _check_fci_free() -> None:
    clients = active_fci_clients()
    if clients:
        raise RuntimeError(f"another FR3 FCI bridge is running: {[item['pid'] for item in clients]}")


def capture(path: Path, robot_ip: str) -> dict:
    _check_fci_free()
    with Fr3Client(robot_ip, enforce_realtime=False) as arm:
        state = arm.read_state()
    _check_q(state.q, "live q")
    if float(np.max(np.abs(state.dq))) > STILL_RAD_S:
        raise RuntimeError("FR3 is moving; wait for it to stop before saving")
    saved = {"schema": "aspire.fr3.pose_bookmark.v1",
             "captured_at": datetime.now().astimezone().isoformat(),
             "robot_ip": robot_ip, "source": "FR3 FCI O_T_EE",
             "q_rad": state.q.tolist(), "T_base_ee": state.ee.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(saved, file, indent=2)
        file.write("\n")
    return {"saved_pose": str(path.resolve()), "captured_at": saved["captured_at"],
            "q_rad": saved["q_rad"], "ee_xyz_base_m": state.ee[:3, 3].tolist(),
            "physical_motion_commanded": False}


def _preview(start_q: np.ndarray, target_q: np.ndarray, live_ee: np.ndarray,
             saved_ee: np.ndarray | None) -> dict:
    """Check a joint interpolation against the FR3 model and its own body."""
    import mujoco

    from robot.franka.kinematics import ArmKinematics, default_model_path

    model = default_model_path()
    arm = ArmKinematics(xml_path=model, limit_margin_rad=LIMIT_MARGIN_RAD)
    tcp = arm.calibrate_tcp(start_q, live_ee)
    predicted = arm.fk(target_q, tcp=tcp)
    xyz_error = angle_error = None
    if saved_ee is not None:
        xyz_error = float(np.linalg.norm(predicted[:3, 3] - saved_ee[:3, 3]))
        angle_error = float(Rotation.from_matrix(
            predicted[:3, :3] @ saved_ee[:3, :3].T).magnitude())
        if xyz_error > 0.015 or angle_error > math.radians(5):
            raise ValueError("saved EE pose does not match the current robot TCP/model "
                             f"({xyz_error:.3f} m, {math.degrees(angle_error):.1f} deg)")
    samples = max(1, math.ceil(float(np.max(np.abs(target_q - start_q))) / 0.025))
    for index, fraction in enumerate(np.linspace(0, 1, samples + 1)):
        q = start_q * (1 - fraction) + target_q * fraction
        if np.any(q < arm.q_min) or np.any(q > arm.q_max):
            raise ValueError(f"joint limit along return path at sample {index}")
        arm.data.qpos[:] = q
        mujoco.mj_forward(arm.model, arm.data)
        for contact_index in range(arm.data.ncon):
            contact = arm.data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if (arm.model.geom_contype[first] and arm.model.geom_contype[second]
                    and contact.dist < 0.01):
                raise ValueError(f"model self-collision along return path at sample {index}")
    return {"joint_path_samples": samples + 1,
            "target_ee_xyz_base_m": predicted[:3, 3].tolist(),
            "saved_pose_model_position_error_m": xyz_error,
            "saved_pose_model_angle_error_deg": (math.degrees(angle_error)
                                                 if angle_error is not None else None),
            "checked": "joint limits and modeled FR3 self-collision; scene objects are not modeled"}


def return_pose(path: Path | None, robot_ip: str,
                factors: tuple[float, float, float], *, preview_only: bool,
                timeout_s: float) -> dict:
    saved, target, saved_ee = _load_pose(path, robot_ip)
    if not preview_only and os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION") != "1":
        raise PermissionError("restore requires OPENFORGE_ALLOW_PHYSICAL_MOTION=1")
    _check_fci_free()
    import franky

    robot = franky.Robot(
        robot_ip, relative_dynamics_factor=franky.RelativeDynamicsFactor(*factors),
        realtime_config=(franky.RealtimeConfig.Ignore if preview_only
                         else franky.RealtimeConfig.Enforce))
    result = {"schema": "aspire.fr3.pose_return_result.v1",
              "bookmark": str(path.resolve()) if path is not None else None,
              "source": saved.get("source"), "captured_at": saved["captured_at"],
              "robot_ip": robot_ip, "target_q_rad": target.tolist(),
              "physical_motion_commanded": False, "gripper_motion_commanded": False,
              "success": False}
    try:
        before = robot.state
        start = _check_q(before.q, "live q")
        dq = np.asarray(before.dq, dtype=np.float64)
        if dq.shape != (7,) or not np.all(np.isfinite(dq)) or np.max(np.abs(dq)) > STILL_RAD_S:
            raise RuntimeError("FR3 is moving; wait until it is stationary")
        if robot.has_errors:
            raise RuntimeError(f"FR3 reports errors: {before.current_errors}")
        live_ee = np.asarray(before.O_T_EE.matrix, dtype=np.float64).reshape(4, 4)
        result["before_q_rad"] = start.tolist()
        result["before_ee_xyz_base_m"] = live_ee[:3, 3].tolist()
        result["max_joint_travel_rad"] = float(np.max(np.abs(target - start)))
        result["preview"] = _preview(start, target, live_ee, saved_ee)
        if preview_only:
            result["success"] = True
            return result
        if result["max_joint_travel_rad"] > 0.005:
            started = time.monotonic()
            result["physical_motion_commanded"] = True
            robot.move(franky.JointMotion(target.tolist()), asynchronous=True)
            while not robot.poll_motion():
                if time.monotonic() - started > timeout_s:
                    raise TimeoutError(f"return motion exceeded {timeout_s:.0f} s")
                if robot.has_errors:
                    raise RuntimeError("FR3 reported an error during return motion")
                time.sleep(0.02)
            if not robot.join_motion(timeout=1.0):
                raise TimeoutError("return motion did not complete")
            result["elapsed_s"] = time.monotonic() - started
        else:
            result["already_at_target"] = True
        after = robot.state
        q_after = np.asarray(after.q, dtype=np.float64)
        joint_error = float(np.max(np.abs(q_after - target)))
        result.update({"after_q_rad": q_after.tolist(),
                       "after_ee_xyz_base_m": np.asarray(after.O_T_EE.matrix, dtype=float)[:3, 3].tolist(),
                       "max_joint_error_rad": joint_error})
        if robot.has_errors or joint_error > 0.01:
            raise RuntimeError(f"FR3 did not reach target joints: max error {joint_error:.4f} rad")
        gripper = franky.Gripper(robot_ip)
        hand_before = gripper.state
        open_width = float(gripper.max_width)
        result["before_hand_width_m"] = float(hand_before.width)
        result["target_hand_width_m"] = open_width
        result["physical_motion_commanded"] = True
        result["gripper_motion_commanded"] = True
        if not gripper.move(open_width, GRIPPER_SPEED_M_S):
            raise RuntimeError("Franka Hand open command failed")
        hand_after = gripper.state
        result["after_hand_width_m"] = float(hand_after.width)
        if abs(float(hand_after.width) - open_width) > GRIPPER_TOLERANCE_M:
            raise RuntimeError("Franka Hand did not reach its fully open width")
        result["success"] = True
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if robot.is_in_control:
            robot.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "preview", "restore"))
    parser.add_argument("--pose", type=Path,
                        help="saved bookmark JSON; preview/restore default to the Franka example "
                             f"pose; capture defaults to {DEFAULT_POSE}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="local manipulation configuration")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--delay-s", type=float, default=3.0,
                        help="wait this many seconds before execution (default: 3; 0: no wait)")
    parser.add_argument("--result", type=Path,
                        help="JSON result file; restore defaults to a timestamped log")
    args = parser.parse_args()
    try:
        if not math.isfinite(args.delay_s) or args.delay_s < 0:
            raise ValueError("--delay-s must be nonnegative and finite")
        if args.command != "capture" and (
                not math.isfinite(args.timeout_s) or args.timeout_s <= 0):
            raise ValueError("--timeout-s must be positive and finite")
        if args.delay_s > 0:
            print(f"Waiting {args.delay_s:g} s before {args.command}; Ctrl+C to cancel.",
                  file=sys.stderr, flush=True)
            time.sleep(args.delay_s)
        ip, factors = _hardware(args.config)
        if args.command == "capture":
            output = capture(args.pose or DEFAULT_POSE, ip)
        else:
            output = return_pose(args.pose, ip, factors,
                                 preview_only=args.command == "preview",
                                 timeout_s=args.timeout_s)
        result_path = args.result
        if result_path is None and args.command == "restore":
            stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f")
            result_path = ROOT / "logs/fr3_pose_returns" / f"{stamp}.json"
        if result_path:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            output["result_path"] = str(result_path.resolve())
            with result_path.open("x", encoding="utf-8") as file:
                json.dump(output, file, indent=2)
                file.write("\n")
        print(json.dumps(output, indent=2))
        return 0 if output.get("success", True) else 2
    except (OSError, ValueError, KeyError, RuntimeError, PermissionError,
            ImportError, TimeoutError) as exc:
        print(f"FR3 pose return stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
