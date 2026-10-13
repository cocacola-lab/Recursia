"""SAM3 and aligned RGB-D object geometry, without ink or wiping rules."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cap.saved_scripts.fr3_runtime.geometry import deproject_pixels, points_from_mask


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the dominant observed plane; do not assume a horizontal base plane."""
    points = np.asarray(points, dtype=float)
    if len(points) < 20:
        raise ValueError("too few valid depth points for a surface")
    sample = points[::max(1, len(points) // 2000)]
    rng = np.random.default_rng(0)
    best = np.ones(len(sample), dtype=bool)
    count = 0
    for _ in range(80):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-8:
            continue
        inside = np.abs((sample - a) @ (n / length)) < 0.008
        if inside.sum() > count:
            count, best = int(inside.sum()), inside
    inliers = sample[best]
    origin = np.median(inliers, axis=0)
    _, _, vt = np.linalg.svd(inliers - origin, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    selected = np.abs((points - origin) @ normal) < 0.012
    return origin, normal, points[selected]


def describe_object(capture, mask: np.ndarray, cfg, *, name: str, score: float) -> dict:
    points = points_from_mask(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                              mask, stride=2, distortion=capture.distortion_coeffs)
    origin, normal, planar = fit_plane(points)
    # A stable tangent frame is just a coordinate basis, not a required hand yaw.
    x = np.array([1.0, 0.0, 0.0])
    if abs(x @ normal) > 0.9:
        x = np.array([0.0, 1.0, 0.0])
    x = x - (x @ normal) * normal
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    local = np.column_stack([(planar - origin) @ x, (planar - origin) @ y])
    rectangle = cv2.minAreaRect(local.astype(np.float32))
    box2 = cv2.boxPoints(rectangle).astype(float)
    corners = origin + box2[:, :1] * x + box2[:, 1:] * y
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) < 0.005:
        raise ValueError(f"{name} has a degenerate depth footprint")
    axis = edges[int(np.argmax(lengths))] / max(lengths)
    if axis @ x < 0:
        axis = -axis
    other = np.cross(normal, axis)
    center = corners.mean(axis=0)
    short, long = float(min(lengths)), float(max(lengths))
    grasp_point = None
    if name == "whiteboard eraser":
        interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        if not interior.any():
            interior = mask.astype(np.uint8)
        distances = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
        row, column = np.unravel_index(np.argmax(distances), distances.shape)
        # Use depth at an actual interior pixel for grasping. A bounding-box
        # centre fitted to noisy depth is not necessarily on the object top.
        grasp_point = deproject_pixels(
            capture.depth_m, capture.intrinsics, cfg.camera_to_base,
            np.array([[column, row]]), patch_radius=3,
            distortion=capture.distortion_coeffs)[0].tolist()
    support = support_normal_result = None
    thickness = float(np.ptp((points - origin) @ normal))
    # The visible ring around an object gives a support estimate independent
    # of the board's yaw or the simulated tool dimensions.
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~mask
    ring_points = points_from_mask(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                                   ring, stride=2, distortion=capture.distortion_coeffs)
    if len(ring_points) >= 20:
        support_origin, support_normal, _ = fit_plane(ring_points)
        signed_height = float((center - support_origin) @ support_normal)
        if 0.002 <= signed_height <= 0.12 and support_normal @ normal > 0.9:
            support = center - support_normal * signed_height
            support_normal_result = support_normal.tolist()
            thickness = max(thickness, signed_height)
    return {
        "name": name, "score": score, "position_base_m": center.tolist(),
        "grasp_point_base_m": grasp_point,
        "quaternion_xyzw": Rotation.from_matrix(np.column_stack([axis, other, normal])).as_quat().tolist(),
        "normal_base": normal.tolist(), "long_axis_base": axis.tolist(),
        "extent_m": [long, short, thickness],
        "corners_base_m": corners.tolist(), "width_m": float(lengths[0]),
        "height_m": float(lengths[1]), "valid_depth_points": len(points),
        "bottom_position_base_m": None if support is None else support.tolist(),
        "bottom_source": "observed_support_plane" if support is not None else "unavailable",
        "support_normal_base": support_normal_result,
        "capture_time": capture.captured_at,
    }


def grasp_candidate(obj: dict, current_quaternion_xyzw, offset_base_m) -> dict:
    z = -np.asarray(obj.get("support_normal_base") or obj["normal_base"])
    long_axis = np.asarray(obj["long_axis_base"])
    long_axis = long_axis - (long_axis @ z) * z
    long_axis /= np.linalg.norm(long_axis)
    rotations = []
    for sign in (1, -1):
        x = sign * long_axis
        rotations.append(Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])))
    current = Rotation.from_quat(current_quaternion_xyzw)
    chosen = min(rotations, key=lambda r: (current.inv() * r).magnitude())
    width = float(obj["extent_m"][1]) - 0.004
    if not 0.005 <= width <= 0.075:
        raise ValueError(f"observed eraser grasp width {width:.3f} m does not fit the hand")
    point = obj.get("grasp_point_base_m") or obj["position_base_m"]
    return {"position_base_m": (np.asarray(point) + offset_base_m).tolist(),
            "quaternion_xyzw": chosen.as_quat().tolist(), "width_m": width,
            "object": obj, "note": "candidate only; the agent chooses approach/lift/placement"}


def placement_geometry(board: dict, eraser: dict, *, height_tolerance_m: float = 0.025) -> dict:
    """Goal evidence: observed object footprint on the observed support plane."""
    corners = np.asarray(board["corners_base_m"])
    u = corners[1] - corners[0]
    v = corners[3] - corners[0]
    u_length, v_length = np.linalg.norm(u), np.linalg.norm(v)
    u, v = u / u_length, v / v_length
    footprint = np.asarray(eraser["corners_base_m"]) - corners[0]
    uv = np.column_stack([footprint @ u, footprint @ v])
    inside = bool(np.all(uv >= -0.005) and
                  np.all(uv <= [u_length + 0.005, v_length + 0.005]))
    normal = np.asarray(board["normal_base"])
    top_height = float((np.asarray(eraser["position_base_m"])
                        - np.asarray(board["position_base_m"])) @ normal)
    bottom = eraser.get("bottom_position_base_m")
    bottom_height = None if bottom is None else float(
        (np.asarray(bottom) - np.asarray(board["position_base_m"])) @ normal)
    supported = (bottom_height is not None and abs(bottom_height) <= height_tolerance_m
                 and 0.002 <= top_height <= 0.12)
    return {"on_board": bool(inside and supported), "footprint_inside_board": inside,
            "top_height_above_board_m": top_height, "bottom_height_above_board_m": bottom_height,
            "support_observed": bottom is not None}
