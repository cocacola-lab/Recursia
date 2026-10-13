"""Observation -> grasp/wipe plan -> evidence, with no hardware side effects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import Capture
from .config import WhiteboardConfig
from .geometry import (BoardGeometry, deproject_pixels, estimate_board, ink_mask,
                       points_from_mask, polygon_mask, serpentine)


def execution_contract(cfg: WhiteboardConfig) -> dict[str, Any]:
    """Bind runtime settings as well as waypoint geometry to the plan ID."""
    from dataclasses import fields

    ignored = {"path", "hardware_config_path", "station_model_path"}
    return {field.name: (value.tolist() if isinstance(value, np.ndarray) else value)
            for field in fields(cfg) if field.name not in ignored
            for value in [getattr(cfg, field.name)]}


def capture_digest(capture: Capture) -> str:
    digest = hashlib.sha256()
    for value in (capture.rgb, capture.depth_m, capture.intrinsics, capture.distortion_coeffs):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Scene:
    board: BoardGeometry
    board_mask: np.ndarray
    eraser_mask: np.ndarray
    ink_mask: np.ndarray
    eraser_target_base_m: np.ndarray
    ink_points_base_m: np.ndarray
    ink_pixel_count: int
    eraser_on_board: bool


def observe(capture: Capture, cfg: WhiteboardConfig, *, eraser_present: bool = True,
            held_eraser_target_base_m: np.ndarray | None = None) -> Scene:
    cfg.require_plan()
    if capture.serial != cfg.external_serial:
        raise ValueError("observation came from the wrong RealSense camera")
    shape = capture.depth_m.shape
    board_mask = polygon_mask(shape, cfg.board_corners_uv)
    eraser_mask = (polygon_mask(shape, cfg.eraser_polygon_uv) if eraser_present
                   else np.zeros(shape, dtype=bool))
    overlap = (float(np.count_nonzero(board_mask & eraser_mask) / np.count_nonzero(eraser_mask))
               if eraser_present else 0.0)
    if eraser_present:
        u, v = np.rint(cfg.grasp_point_uv).astype(int)
        if not (0 <= u < shape[1] and 0 <= v < shape[0]) or not eraser_mask[v, u]:
            raise ValueError("grasp pixel must lie inside the annotated eraser")
    elif held_eraser_target_base_m is None:
        raise ValueError("a held-eraser observation needs the original grasp target")
    board = estimate_board(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                           cfg.board_corners_uv, board_mask,
                           eraser_mask if eraser_present else None,
                           distortion=capture.distortion_coeffs)
    target = (deproject_pixels(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                               cfg.grasp_point_uv.reshape(1, 2), patch_radius=3,
                               distortion=capture.distortion_coeffs)[0]
              if eraser_present else np.asarray(held_eraser_target_base_m, dtype=np.float64))
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("invalid held eraser grasp target")
    ink = ink_mask(capture.rgb, board_mask, eraser_mask if eraser_present else None,
                   gray_threshold=cfg.ink_gray_threshold)
    count = int(np.count_nonzero(ink))
    if count < 10:
        raise ValueError("no reliable dark writing inside the whiteboard ROI")
    points = points_from_mask(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                              ink, stride=1, distortion=capture.distortion_coeffs,
                              min_points=10)
    # Depth from ink can be sparse; reject points far from the live board plane.
    distance = np.abs(board.local(points)[:, 2])
    points = points[distance < 0.012]
    if len(points) < 10:
        raise ValueError("dark pixels have no depth support on the board plane")
    # Once depth establishes the surface, intersect every ink pixel with that
    # measured plane, including thin strokes missed by the depth subsampling.
    ys, xs = np.nonzero(ink)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    rays_xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), capture.intrinsics,
                                 capture.distortion_coeffs).reshape(-1, 2)
    rays = np.column_stack([rays_xy, np.ones(len(pixels))]) @ cfg.camera_to_base[:3, :3].T
    origin = cfg.camera_to_base[:3, 3]
    denominator = rays @ board.normal
    if np.any(np.abs(denominator) < 0.1):
        raise ValueError("ink ray is nearly parallel to board")
    distance = float((board.origin - origin) @ board.normal) / denominator
    if np.any((distance <= 0.1) | (distance >= 2.0)):
        raise ValueError("ink ray is outside camera depth range")
    points = origin + rays * distance[:, None]
    return Scene(board, board_mask, eraser_mask, ink, target, points, count,
                 eraser_on_board=overlap > 0.25)


def _pose(xyz: np.ndarray, quat_xyzw: np.ndarray, label: str) -> dict[str, Any]:
    return {"stage": label, "ee_xyz_base_m": np.asarray(xyz).tolist(),
            "ee_quat_xyzw": np.asarray(quat_xyzw).tolist()}


def plan_scene(scene: Scene, cfg: WhiteboardConfig, capture: Capture) -> dict[str, Any]:
    """All targets are live-observation-derived; no simulator coordinates."""
    normal = scene.board.normal
    grasp = scene.eraser_target_base_m + cfg.grasp_ee_offset_from_target_base_m
    pregrasp = grasp + normal * cfg.clearance_m
    lift = grasp + normal * cfg.lift_m
    # Pad dimensions are defined in the board's u/v axes. A wipe EE pose is
    # obtained by subtracting the rotated measured EE->pad-centre offset.
    wipe_rotation = Rotation.from_quat(cfg.wipe_quat_xyzw).as_matrix()
    facing = float((wipe_rotation @ cfg.pad_normal_in_ee) @ (-normal))
    if facing < 0.999:
        raise ValueError(f"eraser pad does not face the board (alignment {facing:.3f})")
    if cfg.pad_length_axis_in_ee is None:
        raise ValueError("pad_length_axis_in_ee must be measured")
    alignment = abs(float((wipe_rotation @ cfg.pad_length_axis_in_ee) @ scene.board.axis_u))
    if alignment < 0.999:
        raise ValueError("pad long edge must align with board u axis within 2.5 degrees")
    pad_offset_base = wipe_rotation @ cfg.pad_center_in_ee_m
    points = serpentine(scene.board, scene.ink_points_base_m,
                         pad_width_m=cfg.pad_width_m, pad_length_m=cfg.pad_length_m,
                         margin_m=0.008, clearance_m=cfg.clearance_m,
                         body_width_m=cfg.eraser_body_width_m,
                         body_length_m=cfg.eraser_body_length_m)
    wipe: list[dict[str, Any]] = []
    for waypoint in points:
        pad = np.asarray(waypoint["pad_center_base_m"], dtype=np.float64)
        stage = str(waypoint["stage"])
        if stage == "wipe":
            pad = pad - normal * cfg.contact_indent_m
        ee = pad - pad_offset_base
        item = _pose(ee, cfg.wipe_quat_xyzw, stage)
        item["pad_center_base_m"] = pad.tolist()
        if stage == "wipe":
            item["board_normal_base"] = normal.tolist()
        if "row" in waypoint:
            item["row"] = int(waypoint["row"])
            item["index"] = int(waypoint["index"])
        wipe.append(item)
    first_contact_xyz = np.asarray(wipe[1]["ee_xyz_base_m"])
    first_hover_xyz = np.asarray(wipe[0]["ee_xyz_base_m"])
    # One direct descent to the contact standoff, followed by the contact pose.
    approach = [_pose(first_contact_xyz + normal * 0.012,
                      cfg.wipe_quat_xyzw, "approach")]
    grasp_steps = [
        _pose(pregrasp, cfg.grasp_quat_xyzw, "pregrasp"),
        _pose(grasp, cfg.grasp_quat_xyzw, "grasp"),
        _pose(lift, cfg.grasp_quat_xyzw, "lift"),
    ]
    # Keep clearance and direction changes, without artificial distance/angle
    # subdivisions. Franky generates the trajectory to each endpoint.
    hover_xyz = np.asarray(wipe[0]["ee_xyz_base_m"])
    safe_z = max(float(lift[2]), float(hover_xyz[2]))
    safe_from = lift.copy()
    safe_from[2] = safe_z
    safe_to = hover_xyz.copy()
    safe_to[2] = safe_z
    transfer: list[dict[str, Any]] = []
    if np.linalg.norm(safe_from - lift) > 1e-4:
        transfer.append(_pose(safe_from, cfg.grasp_quat_xyzw, "transfer_rise"))
    angle = (Rotation.from_quat(cfg.grasp_quat_xyzw).inv()
             * Rotation.from_quat(cfg.wipe_quat_xyzw)).magnitude()
    if angle > 1e-4:
        transfer.append(_pose(safe_from, cfg.wipe_quat_xyzw, "transfer_orient"))
    if np.linalg.norm(safe_to - safe_from) > 1e-4:
        transfer.append(_pose(safe_to, cfg.wipe_quat_xyzw, "transfer_cross"))
    if np.linalg.norm(safe_to - hover_xyz) > 1e-4:
        transfer.append(_pose(hover_xyz, cfg.wipe_quat_xyzw, "transfer_lower"))
    # After wiping, return above the board before re-observing so the eraser
    # does not occlude the dark-ink mask. Retrace the saved transfer route.
    last_hover_xyz = np.asarray(wipe[-1]["ee_xyz_base_m"])
    retreat = []
    return_height = max(float(last_hover_xyz[2]), float(first_hover_xyz[2]))
    raised = last_hover_xyz.copy()
    raised[2] = return_height
    if np.linalg.norm(raised - last_hover_xyz) > 1e-4:
        retreat.append(_pose(raised, cfg.wipe_quat_xyzw, "retreat_rise"))
    if np.linalg.norm(first_hover_xyz - raised) > 1e-4:
        retreat.append(_pose(first_hover_xyz, cfg.wipe_quat_xyzw, "retreat_cross"))
    retreat.extend({**step, "stage": "retreat_" + step["stage"]}
                   for step in reversed([grasp_steps[-1]] + transfer))
    staging = [_pose(np.asarray(pose["ee_xyz_base_m"]),
                     np.asarray(pose["ee_quat_xyzw"]), "staging")
               for pose in cfg.staging_ee_poses]
    all_poses = staging + grasp_steps + transfer + approach + wipe + retreat
    lo, hi = cfg.workspace_min_m, cfg.workspace_max_m
    for item in all_poses:
        xyz = np.asarray(item["ee_xyz_base_m"])
        if np.any(xyz < lo) or np.any(xyz > hi):
            raise ValueError(f"{item['stage']} EE target outside configured workspace bounds: {xyz}")
    content: dict[str, Any] = {
        "schema": "aspire.fr3_whiteboard.plan.v1",
        "execution_contract": execution_contract(cfg),
        "capture_digest": capture_digest(capture),
        "calibration_id": cfg.calibration_id,
        "camera_serial": capture.serial,
        "capture_time": capture.captured_at,
        "board": {
            "corners_base_m": scene.board.corners_base.tolist(),
            "normal_base": normal.tolist(),
            "width_m": scene.board.width_m,
            "height_m": scene.board.height_m,
            "plane_rms_m": scene.board.plane_rms_m,
        },
        "ink_pixel_count_before": scene.ink_pixel_count,
        "eraser_target_base_m": scene.eraser_target_base_m.tolist(),
        "eraser_on_board": scene.eraser_on_board,
        "staging": staging,
        "grasp": grasp_steps,
        "transfer": transfer,
        "approach": approach,
        "wipe": wipe,
        "retreat": retreat,
        "limits": {"force_limit_n": cfg.force_limit_n,
                   "grasp_force_limit_n": cfg.grasp_force_limit_n,
                   "min_contact_force_n": cfg.min_contact_force_n,
                   "contact_indent_m": cfg.contact_indent_m,
                   "workspace_min_m": lo.tolist(), "workspace_max_m": hi.tolist()},
    }
    return seal_plan(content)


def seal_plan(content: dict[str, Any]) -> dict[str, Any]:
    """Bind a previewed plan ID to every field that affects execution."""
    content = {key: value for key, value in content.items() if key != "plan_id"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    content["plan_id"] = hashlib.sha256(encoded).hexdigest()[:16]
    return content


def save_plan(plan: dict[str, Any], scene: Scene, capture: Capture,
              cfg: WhiteboardConfig, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing.get("plan_id") == plan.get("plan_id") and
                seal_plan(existing)["plan_id"] == existing["plan_id"]):
            return path
        raise FileExistsError(f"a different plan already exists; use a fresh run directory: {path}")
    overlay = cv2.cvtColor(capture.rgb, cv2.COLOR_RGB2BGR)
    cv2.polylines(overlay, [np.rint(cfg.board_corners_uv).astype(np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(overlay, [np.rint(cfg.eraser_polygon_uv).astype(np.int32)], True, (255, 0, 255), 2)
    overlay[scene.ink_mask] = (0, 0, 255)
    # Reproject path through the configured external camera transform.
    camera_from_base = np.linalg.inv(cfg.camera_to_base)
    uv = []
    for point in plan["wipe"]:
        if point["stage"] != "wipe":
            continue
        xyz = np.asarray(point["pad_center_base_m"])
        camera = camera_from_base[:3, :3] @ xyz + camera_from_base[:3, 3]
        if camera[2] <= 0:
            continue
        projected, _ = cv2.projectPoints(camera.reshape(1, 3), np.zeros(3),
                                          np.zeros(3), capture.intrinsics,
                                          capture.distortion_coeffs)
        px = projected.reshape(2)
        uv.append((int(round(px[0])), int(round(px[1]))))
    if len(uv) > 1:
        cv2.polylines(overlay, [np.asarray(uv, np.int32)], False, (255, 255, 0), 2)
    if not cv2.imwrite(str(folder / "plan_overlay.png"), overlay):
        raise RuntimeError("could not write plan overlay")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    return path


def evaluate_after(before: Scene, after_capture: Capture, cfg: WhiteboardConfig) -> dict[str, Any]:
    if after_capture.serial != cfg.external_serial or after_capture.rgb.shape[:2] != before.board_mask.shape:
        raise ValueError("after image is from the wrong camera or resolution")
    after_board = estimate_board(after_capture.depth_m, after_capture.intrinsics,
                                  cfg.camera_to_base, cfg.board_corners_uv,
                                  before.board_mask,
                                  distortion=after_capture.distortion_coeffs)
    drift = np.max(np.linalg.norm(after_board.corners_base - before.board.corners_base, axis=1))
    # Retain depth-fit variation as diagnostic data, not a visual-score gate.
    # Reuse the same board ROI but do not exclude the original eraser location;
    # it should be empty after the tool is lifted. Mask it only if occupied.
    after_ink = ink_mask(after_capture.rgb, before.board_mask, None,
                         gray_threshold=cfg.ink_gray_threshold)
    before_count = before.ink_pixel_count
    after_count = int(np.count_nonzero(after_ink))
    reduction = 1.0 - after_count / max(1, before_count)
    count, _, stats, _ = cv2.connectedComponentsWithStats(after_ink.astype(np.uint8), 8)
    residuals = [{"bbox_uvwh": row[:4].tolist(), "dark_pixels": int(row[4])}
                 for row in stats[1:count] if row[4] >= 4]
    residuals.sort(key=lambda item: item["dark_pixels"], reverse=True)
    return {"schema": "aspire.fr3_whiteboard.visual_verification.v1",
            "before_dark_pixels": before_count, "after_dark_pixels": after_count,
            "dark_pixel_reduction": reduction,
            "board_corner_drift_m": float(drift),
            "residual_regions": residuals,
            "threshold_used": cfg.min_ink_reduction,
            "visual_pass": bool(reduction >= cfg.min_ink_reduction),
            "caveat": "Dark-pixel reduction is a proxy; inspect before/after images for actual cleanliness."}
