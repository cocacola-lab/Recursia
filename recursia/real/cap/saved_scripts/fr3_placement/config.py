"""Station and control settings; no object poses or task route in YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from cap.saved_scripts.fr3_runtime.geometry import validate_transform


@dataclass(frozen=True)
class PlacementConfig:
    path: Path
    hardware_config_path: Path
    robot_ip: str
    external_serial: str
    camera_to_base: np.ndarray
    calibration_id: str
    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    segmentation_url: str
    board_prompt: str
    eraser_prompt: str
    grasp_speed_m_s: float
    grasp_force_n: float
    grasp_epsilon_m: float
    force_limit_n: float
    grasp_force_limit_n: float
    grasp_ee_offset_from_target_base_m: np.ndarray
    fallback_bottom_in_ee_m: np.ndarray
    relative_dynamics: tuple[float, float, float]
    grasp_width_m: float = 0.04
    graspnet_url: str = "http://127.0.0.1:8115"
    graspnet_to_ee: np.ndarray = field(default_factory=lambda: np.array(
        [[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., .1034], [0., 0., 0., 1.]]))
    pose_control: str = "cartesian"


def load_config(path: str | Path) -> PlacementConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if raw.get("schema") not in {"aspire.fr3_eraser_place.v1", "aspire.fr3_manipulation.v1"}:
        raise ValueError("expected FR3 placement or manipulation schema")

    def resolve(value):
        p = Path(value).expanduser()
        return (path.parent / p).resolve() if not p.is_absolute() else p.resolve()

    def vector(value, name):
        arr = np.asarray(value, dtype=float)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite 3-vector")
        return arr

    def number(value, name, low, high):
        value = float(value)
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    hardware_path = resolve(raw["hardware_config"])
    hardware = yaml.safe_load(hardware_path.read_text())["hardware"]
    station = yaml.safe_load(resolve(raw["station_model"]).read_text())
    camera = station["external_camera"]
    serial = str(hardware["cameras"]["ext"]["serial"])
    if str(camera["serial"]) != serial or camera.get("optical_frame") != "color":
        raise ValueError("station must describe the configured external color optical frame")
    safety_path = resolve(raw["safety_config"]) if "safety_config" in raw else hardware_path.parent / "safety.yaml"
    safety = yaml.safe_load(safety_path.read_text())["safety"]
    motion, hand = raw.get("motion", {}), raw.get("gripper", {})
    perception, tool = raw.get("perception", {}), raw.get("tool", {})
    lo = vector(motion.get("workspace_min_m") or safety["cart_workspace_min_xyz"], "workspace_min_m")
    hi = vector(motion.get("workspace_max_m") or safety["cart_workspace_max_xyz"], "workspace_max_m")
    if np.any(lo >= hi):
        raise ValueError("workspace bounds are inverted")
    dynamics = tuple(number(v, "relative_dynamics", 0.001, 1.0)
                     for v in motion.get("relative_dynamics", [0.05, 0.05, 0.04]))
    if len(dynamics) != 3:
        raise ValueError("relative_dynamics needs velocity, acceleration, jerk")
    pose_control = str(motion.get("pose_control", "cartesian"))
    if pose_control not in {"cartesian", "ik_joint"}:
        raise ValueError("pose_control must be cartesian or ik_joint")
    return PlacementConfig(
        path, hardware_path, str(hardware["franka"]["ip"]), serial,
        validate_transform(camera["T_base_camera_optical"]), str(station["calibration_id"]),
        lo, hi, str(perception.get("sam3_url", "http://127.0.0.1:6767")),
        str(perception.get("board_prompt", "whiteboard")),
        str(perception.get("eraser_prompt", "whiteboard eraser")),
        number(hand.get("speed_m_s", 0.2), "gripper speed", 0.005, 0.2),
        number(hand.get("force_n", 15), "gripper force", 5, 40),
        number(hand.get("epsilon_m", 0.01), "grasp epsilon", 0.001, 0.02),
        number(motion.get("force_limit_n", 12), "force limit", 3, 40),
        number(motion.get("grasp_force_limit_n", 20), "grasp force limit", 3, 40),
        vector(tool.get("grasp_ee_offset_from_target_base_m", [0, 0, 0]), "grasp offset"),
        vector(tool.get("fallback_bottom_in_ee_m", [0, 0, 0.0158]), "bottom offset"), dynamics,
        graspnet_url=str(perception.get("contact_graspnet_url", "http://127.0.0.1:8115")),
        graspnet_to_ee=validate_transform(tool.get("T_contact_graspnet_ee",
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, .1034], [0, 0, 0, 1]])),
        pose_control=pose_control)
