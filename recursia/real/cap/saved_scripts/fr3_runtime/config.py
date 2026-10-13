"""Validated configuration for one FR3 whiteboard station."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .geometry import validate_transform


def _array(raw: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return arr


def _positive(raw: Any, name: str, low: float, high: float) -> float:
    val = float(raw)
    if not np.isfinite(val) or not low <= val <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return val


@dataclass(frozen=True)
class WhiteboardConfig:
    path: Path
    hardware_config_path: Path
    station_model_path: Path | None
    robot_ip: str
    external_serial: str
    wrist_serial: str | None
    camera_to_base: np.ndarray | None
    calibration_id: str | None
    board_corners_uv: np.ndarray | None
    eraser_polygon_uv: np.ndarray | None
    grasp_point_uv: np.ndarray | None
    grasp_quat_xyzw: np.ndarray | None
    grasp_ee_offset_from_target_base_m: np.ndarray | None
    wipe_quat_xyzw: np.ndarray | None
    pad_center_in_ee_m: np.ndarray | None
    pad_normal_in_ee: np.ndarray | None
    pad_width_m: float | None
    pad_length_m: float | None
    grasp_width_m: float | None
    grasp_force_n: float
    grasp_speed_m_s: float
    lift_m: float
    clearance_m: float
    contact_indent_m: float
    force_limit_n: float
    grasp_force_limit_n: float
    ink_gray_threshold: int
    min_ink_reduction: float
    workspace_min_m: np.ndarray | None
    workspace_max_m: np.ndarray | None
    staging_ee_poses: tuple[dict[str, list[float]], ...]
    pad_length_axis_in_ee: np.ndarray | None = None
    min_contact_force_n: float = 0.3
    scene_mode: str = "manual"
    segmentation_url: str = "http://127.0.0.1:6767"
    board_prompt: str = "whiteboard"
    eraser_prompt: str = "whiteboard eraser"
    eraser_body_width_m: float | None = None
    eraser_body_length_m: float | None = None

    def missing_for_plan(self) -> list[str]:
        required = {
            "camera_to_base": self.camera_to_base,
            "calibration_id": self.calibration_id,
            "board_corners_uv": self.board_corners_uv,
            "eraser_polygon_uv": self.eraser_polygon_uv,
            "grasp_point_uv": self.grasp_point_uv,
            "grasp_quat_xyzw": self.grasp_quat_xyzw,
            "grasp_ee_offset_from_target_base_m": self.grasp_ee_offset_from_target_base_m,
            "wipe_quat_xyzw": self.wipe_quat_xyzw,
            "pad_center_in_ee_m": self.pad_center_in_ee_m,
            "pad_normal_in_ee": self.pad_normal_in_ee,
            "pad_length_axis_in_ee": self.pad_length_axis_in_ee,
            "pad_width_m": self.pad_width_m,
            "pad_length_m": self.pad_length_m,
            "grasp_width_m": self.grasp_width_m,
            "workspace_min_m": self.workspace_min_m,
            "workspace_max_m": self.workspace_max_m,
        }
        missing = [key for key, value in required.items()
                   if value is None or (isinstance(value, str) and not value)]
        if self.scene_mode == "auto":
            missing = [name for name in missing if name not in {
                "board_corners_uv", "eraser_polygon_uv", "grasp_point_uv",
                "grasp_quat_xyzw", "wipe_quat_xyzw", "grasp_width_m",
                "pad_width_m", "pad_length_m"}]
        return missing

    def require_plan(self) -> None:
        missing = self.missing_for_plan()
        # The detector/planner must resolve these before observe() is called.
        if self.scene_mode == "auto":
            missing += [name for name in ("board_corners_uv", "eraser_polygon_uv",
                                       "grasp_point_uv", "grasp_quat_xyzw",
                                       "wipe_quat_xyzw", "grasp_width_m")
                        if getattr(self, name) is None]
            missing += [name for name in ("pad_width_m", "pad_length_m")
                        if getattr(self, name) is None]
        if missing:
            raise ValueError("whiteboard plan inputs incomplete (perception/planning/tool/station): " + ", ".join(missing))


def load_config(path: str | Path) -> WhiteboardConfig:
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "aspire.fr3_whiteboard.v1":
        raise ValueError("expected schema aspire.fr3_whiteboard.v1")
    hw_path = Path(str(data.get("hardware_config", ""))).expanduser()
    if not hw_path.is_absolute():
        hw_path = path.parent / hw_path
    hw_path = hw_path.resolve()
    hw = yaml.safe_load(hw_path.read_text(encoding="utf-8"))["hardware"]
    if hw.get("gripper", {}).get("driver", "franka_hand") != "franka_hand":
        raise ValueError("FR3 whiteboard requires hardware.gripper.driver=franka_hand")
    ip = str(hw["franka"]["ip"])
    ext = str(hw["cameras"]["ext"]["serial"])
    wrist = hw.get("cameras", {}).get("wrist", {}).get("serial")
    station_path = None
    station_camera: dict[str, Any] = {}
    station_calibration_id = None
    if data.get("station_model"):
        station_path = Path(str(data["station_model"])).expanduser()
        if not station_path.is_absolute():
            station_path = path.parent / station_path
        station_path = station_path.resolve()
        station = yaml.safe_load(station_path.read_text(encoding="utf-8"))
        if not isinstance(station, dict) or station.get("schema") != "aspire.fr3_station.v1":
            raise ValueError("expected station model schema aspire.fr3_station.v1")
        station_camera = station.get("external_camera", {})
        if not isinstance(station_camera, dict):
            raise ValueError("station external_camera must be a mapping")
        if str(station_camera.get("serial", "")) != ext:
            raise ValueError("station external camera serial differs from hardware config")
        if station_camera.get("optical_frame") != "color":
            raise ValueError("station external camera must use the color optical frame")
        station_calibration_id = station.get("calibration_id")
    camera = data.get("camera", {})
    scene = data.get("scene", {})
    mode = scene.get("mode", "manual")
    if mode not in {"manual", "auto"}:
        raise ValueError("scene.mode must be manual or auto")
    segmentation_url = str(scene.get("segmentation_url", "http://127.0.0.1:6767"))
    board_prompt = str(scene.get("board_prompt", "whiteboard")).strip()
    eraser_prompt = str(scene.get("eraser_prompt", "whiteboard eraser")).strip()
    if not board_prompt or not eraser_prompt:
        raise ValueError("scene prompts must be nonempty")
    tool = data.get("tool", {})
    motion = data.get("motion", {})
    transform = camera.get("T_base_camera_optical")
    if transform is None:
        transform = station_camera.get("T_base_camera_optical")
    if transform is not None:
        transform = validate_transform(transform)
    calibration_id = camera.get("calibration_id") or station_calibration_id
    corners = scene.get("board_corners_uv")
    eraser = scene.get("eraser_polygon_uv")
    grasp_px = scene.get("grasp_point_uv")
    q_grasp = tool.get("grasp_quat_xyzw")
    grasp_offset = tool.get("grasp_ee_offset_from_target_base_m")
    q_wipe = tool.get("wipe_quat_xyzw")
    offset = tool.get("pad_center_in_ee_m")
    pad_normal = tool.get("pad_normal_in_ee")
    pad_axis = tool.get("pad_length_axis_in_ee")
    if pad_axis is not None:
        pad_axis = _array(pad_axis, (3,), "pad_length_axis_in_ee")
        if abs(np.linalg.norm(pad_axis) - 1.0) > 0.01:
            raise ValueError("pad_length_axis_in_ee must be a unit vector")
    wmin = motion.get("workspace_min_m")
    wmax = motion.get("workspace_max_m")
    if corners is not None:
        corners = _array(corners, (4, 2), "board_corners_uv")
    if eraser is not None:
        eraser = np.asarray(eraser, dtype=np.float64)
        if eraser.ndim != 2 or eraser.shape[1] != 2 or len(eraser) < 3 or not np.all(np.isfinite(eraser)):
            raise ValueError("eraser_polygon_uv needs at least 3 finite pixels")
    if grasp_px is not None:
        grasp_px = _array(grasp_px, (2,), "grasp_point_uv")
    if q_grasp is not None:
        q_grasp = _array(q_grasp, (4,), "grasp_quat_xyzw")
    if q_wipe is not None:
        q_wipe = _array(q_wipe, (4,), "wipe_quat_xyzw")
    for name, quat in (("grasp_quat_xyzw", q_grasp), ("wipe_quat_xyzw", q_wipe)):
        if quat is not None and abs(np.linalg.norm(quat) - 1.0) > 0.01:
            raise ValueError(f"{name} is not a unit quaternion")
    if offset is not None:
        offset = _array(offset, (3,), "pad_center_in_ee_m")
    if pad_normal is not None:
        pad_normal = _array(pad_normal, (3,), "pad_normal_in_ee")
        if abs(np.linalg.norm(pad_normal) - 1.0) > 0.01:
            raise ValueError("pad_normal_in_ee must be a unit vector")
        if pad_axis is not None and abs(float(pad_axis @ pad_normal)) > 0.01:
            raise ValueError("pad length axis must be perpendicular to pad normal")
    if grasp_offset is not None:
        grasp_offset = _array(grasp_offset, (3,), "grasp_ee_offset_from_target_base_m")
        if np.linalg.norm(grasp_offset) > 0.20:
            raise ValueError("grasp EE offset exceeds 0.20 m")
    if wmin is not None:
        wmin = _array(wmin, (3,), "workspace_min_m")
    if wmax is not None:
        wmax = _array(wmax, (3,), "workspace_max_m")
    # Reuse the FR3 station's authoritative Cartesian bounds when the task
    # does not specify narrower limits. These are EE bounds, not collision
    # clearance or proof that a target is reachable.
    if wmin is None or wmax is None:
        safety_path = hw_path.parent / "safety.yaml"
        if safety_path.is_file():
            safety = yaml.safe_load(safety_path.read_text(encoding="utf-8"))["safety"]
            if wmin is None:
                wmin = _array(safety["cart_workspace_min_xyz"], (3,), "station workspace min")
            if wmax is None:
                wmax = _array(safety["cart_workspace_max_xyz"], (3,), "station workspace max")
    if wmin is not None and wmax is not None and np.any(wmax <= wmin):
        raise ValueError("workspace bounds are inverted")
    staging = []
    for index, pose in enumerate(motion.get("staging_ee_poses", [])):
        xyz = _array(pose["ee_xyz_base_m"], (3,), f"staging[{index}].xyz")
        quat = _array(pose["ee_quat_xyzw"], (4,), f"staging[{index}].quat")
        if abs(np.linalg.norm(quat) - 1.0) > 0.01:
            raise ValueError(f"staging[{index}] quaternion is not unit length")
        staging.append({"ee_xyz_base_m": xyz.tolist(), "ee_quat_xyzw": quat.tolist()})
    threshold = int(scene.get("ink_gray_threshold", 120))
    if threshold < 20 or threshold > 220:
        raise ValueError("ink_gray_threshold must be 20..220")
    reduction = float(scene.get("min_ink_reduction", 0.5))
    if not 0.0 < reduction <= 1.0:
        raise ValueError("min_ink_reduction must be (0,1]")
    indent = _positive(motion.get("contact_indent_m", 0.002), "contact_indent_m", 0.0, 0.004)
    force_limit = _positive(motion.get("force_limit_n", 8), "force_limit_n", 3, 12)
    grasp_force_limit = _positive(motion.get("grasp_force_limit_n", 8), "grasp_force_limit_n", 3, 20)
    min_contact = _positive(motion.get("min_contact_force_n", 0.3), "min_contact_force_n", 0.1, 3)
    if min_contact >= force_limit:
        raise ValueError("min_contact_force_n must be below force_limit_n")
    return WhiteboardConfig(
        path=path, hardware_config_path=hw_path, station_model_path=station_path, robot_ip=ip,
        external_serial=ext, wrist_serial=str(wrist) if wrist else None,
        camera_to_base=transform, calibration_id=calibration_id,
        board_corners_uv=corners, eraser_polygon_uv=eraser, grasp_point_uv=grasp_px,
        grasp_quat_xyzw=q_grasp, grasp_ee_offset_from_target_base_m=grasp_offset,
        wipe_quat_xyzw=q_wipe, pad_center_in_ee_m=offset,
        pad_normal_in_ee=pad_normal,
        pad_width_m=None if tool.get("pad_width_m") is None else _positive(tool["pad_width_m"], "pad_width_m", 0.01, 0.20),
        pad_length_m=None if tool.get("pad_length_m") is None else _positive(tool["pad_length_m"], "pad_length_m", 0.01, 0.20),
        grasp_width_m=None if tool.get("grasp_width_m") is None else _positive(tool["grasp_width_m"], "grasp_width_m", 0.001, 0.079),
        grasp_force_n=_positive(tool.get("grasp_force_n", 15), "grasp_force_n", 5, 40),
        grasp_speed_m_s=_positive(tool.get("grasp_speed_m_s", 0.2), "grasp_speed_m_s", 0.005, 0.2),
        lift_m=_positive(motion.get("lift_m", 0.08), "lift_m", 0.04, 0.20),
        clearance_m=_positive(motion.get("clearance_m", 0.08), "clearance_m", 0.04, 0.20),
        contact_indent_m=indent,
        force_limit_n=force_limit, grasp_force_limit_n=grasp_force_limit,
        min_contact_force_n=min_contact,
        pad_length_axis_in_ee=pad_axis,
        ink_gray_threshold=threshold, min_ink_reduction=reduction,
        workspace_min_m=wmin, workspace_max_m=wmax,
        staging_ee_poses=tuple(staging),
        scene_mode=mode, segmentation_url=segmentation_url,
        board_prompt=board_prompt, eraser_prompt=eraser_prompt,
    )
