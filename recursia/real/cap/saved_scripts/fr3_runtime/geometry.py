"""Camera geometry and bounded coverage plans for a horizontal whiteboard.

All positions are metres in the FR3 base frame. Pixel coordinates are in the
RealSense colour image after depth-to-colour alignment. Nothing here opens a
camera or an FCI connection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_BOARD_PLANE_INLIER_FRACTION = 0.30


@dataclass(frozen=True)
class BoardGeometry:
    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    normal: np.ndarray
    width_m: float
    height_m: float
    corners_base: np.ndarray
    plane_rms_m: float

    def point(self, u_m: float, v_m: float, normal_m: float = 0.0) -> np.ndarray:
        return self.origin + u_m * self.axis_u + v_m * self.axis_v + normal_m * self.normal

    def local(self, points: np.ndarray) -> np.ndarray:
        delta = np.asarray(points) - self.origin
        return np.stack([delta @ self.axis_u, delta @ self.axis_v, delta @ self.normal], axis=-1)


def validate_transform(transform: object) -> np.ndarray:
    mat = np.asarray(transform, dtype=np.float64)
    if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
        raise ValueError("camera-to-base transform must be a finite 4x4 matrix")
    if not np.allclose(mat[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("invalid homogeneous transform bottom row")
    rot = mat[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=0.01) or np.linalg.det(rot) < 0.99:
        raise ValueError("camera-to-base rotation is not orthonormal")
    return mat


def polygon_mask(shape: tuple[int, int], vertices: object) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("pixel polygon requires at least three finite (u,v) vertices")
    height, width = shape
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        raise ValueError("polygon lies outside the colour image")
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    if np.count_nonzero(mask) < 100:
        raise ValueError("polygon is too small")
    return mask.astype(bool)


def deproject_pixels(depth_m: np.ndarray, intrinsics: object, transform: object,
                     pixels_uv: np.ndarray, *, patch_radius: int = 2,
                     distortion: object | None = None) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    t = validate_transform(transform)
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    coeffs = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    if depth.ndim != 2 or k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("invalid aligned depth or camera intrinsics")
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, coeffs).reshape(-1, 2)
    points = []
    for (u, v), (nx, ny) in zip(pixels, normalized):
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
            raise ValueError("pixel outside aligned depth image")
        crop = depth[max(0, y - patch_radius):y + patch_radius + 1,
                     max(0, x - patch_radius):x + patch_radius + 1]
        valid = crop[np.isfinite(crop) & (crop > 0.1) & (crop < 2.0)]
        if len(valid) < 3:
            raise ValueError(f"no reliable depth near pixel ({u:.0f},{v:.0f})")
        z = float(np.median(valid))
        camera = np.array([nx * z, ny * z, z])
        points.append(t[:3, :3] @ camera + t[:3, 3])
    return np.asarray(points)


def points_from_mask(depth_m: np.ndarray, intrinsics: object, transform: object,
                     mask: np.ndarray, *, stride: int = 3,
                     distortion: object | None = None,
                     min_points: int = 20) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    t = validate_transform(transform)
    if mask.shape != depth.shape or stride < 1 or min_points < 1:
        raise ValueError("mask/depth shape mismatch or invalid stride")
    sampled = np.zeros_like(mask, dtype=bool)
    sampled[::stride, ::stride] = np.asarray(mask[::stride, ::stride], dtype=bool)
    valid = sampled & np.isfinite(depth) & (depth > 0.1) & (depth < 2.0)
    ys, xs = np.nonzero(valid)
    if len(xs) < min_points:
        raise ValueError("too few valid depth points in mask")
    z = depth[ys, xs]
    coeffs = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    normalized = cv2.undistortPoints(
        np.stack([xs, ys], axis=1).astype(np.float64).reshape(-1, 1, 2),
        k, coeffs).reshape(-1, 2)
    camera = np.column_stack([normalized[:, 0] * z, normalized[:, 1] * z, z])
    return camera @ t[:3, :3].T + t[:3, 3]


def estimate_board(depth_m: np.ndarray, intrinsics: object, transform: object,
                   corners_uv: object, board_mask: np.ndarray,
                   eraser_mask: np.ndarray | None = None,
                   distortion: object | None = None,
                   validate_rectangle: bool = True) -> BoardGeometry:
    """Fit the board plane from live depth, excluding the tool and border."""
    corners_uv = np.asarray(corners_uv, dtype=np.float64)
    if corners_uv.shape != (4, 2):
        raise ValueError("board corners must be TL, TR, BR, BL in the image")
    if not cv2.isContourConvex(corners_uv.astype(np.float32)):
        raise ValueError("board corners must form a convex ordered quadrilateral")
    interior = cv2.erode(board_mask.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
    if eraser_mask is not None:
        interior &= ~cv2.dilate(eraser_mask.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    points = points_from_mask(depth_m, intrinsics, transform, interior,
                              distortion=distortion)
    # Bright whiteboard patches can produce grossly wrong RealSense depths.
    # A least-squares seed is biased by those points, so first find the
    # dominant near-horizontal plane with deterministic 3-point RANSAC.
    rng = np.random.default_rng(0)
    best_keep: np.ndarray | None = None
    best_count = 0
    for _ in range(600):
        tri = points[rng.choice(len(points), 3, replace=False)]
        candidate = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = float(np.linalg.norm(candidate))
        if length < 1e-6:
            continue
        candidate /= length
        if abs(candidate[2]) < 0.94:
            continue
        inliers = np.abs((points - tri[0]) @ candidate) < 0.005
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count, best_keep = count, inliers
    min_inliers = max(100, int(MIN_BOARD_PLANE_INLIER_FRACTION * len(points)))
    if best_keep is None or best_count < min_inliers:
        raise ValueError("board depth has no dominant horizontal plane")
    x = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    keep = best_keep
    for _ in range(4):
        coeff, *_ = np.linalg.lstsq(x[keep], points[keep, 2], rcond=None)
        residual = points[:, 2] - x @ coeff
        keep = np.abs(residual) < 0.006
        if np.count_nonzero(keep) < min_inliers:
            raise ValueError("board depth is not a stable plane")
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    normal = np.array([-coeff[0], -coeff[1], 1.0])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.94 or rms > 0.006:
        raise ValueError("board is not sufficiently flat and horizontal")
    # Intersect annotated corner rays with the fitted board plane. Border
    # pixels frequently show table depth instead of whiteboard depth.
    t = validate_transform(transform)
    k = np.asarray(intrinsics, dtype=np.float64)
    coeffs = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    normalized_corners = cv2.undistortPoints(
        corners_uv.reshape(-1, 1, 2), k, coeffs).reshape(-1, 2)
    corners = []
    for nx, ny in normalized_corners:
        ray_camera = np.array([nx, ny, 1.0])
        ray_base = t[:3, :3] @ ray_camera
        ray_origin = t[:3, 3]
        denom = ray_base[2] - coeff[0] * ray_base[0] - coeff[1] * ray_base[1]
        if abs(denom) < 0.1:
            raise ValueError("board corner ray is nearly parallel to board")
        distance = (coeff[0] * ray_origin[0] + coeff[1] * ray_origin[1]
                    + coeff[2] - ray_origin[2]) / denom
        if not 0.1 < distance < 2.0:
            raise ValueError("board corner is outside camera depth range")
        corners.append(ray_origin + distance * ray_base)
    corners = np.asarray(corners)
    axis_u = corners[1] - corners[0]
    axis_u -= axis_u @ normal * normal
    width = float(np.linalg.norm(axis_u))
    if width < 0.08:
        raise ValueError("implausible board dimensions")
    axis_u /= width
    axis_v = corners[3] - corners[0]
    axis_v -= axis_v @ normal * normal + axis_v @ axis_u * axis_u
    height = float(np.linalg.norm(axis_v))
    if not (0.08 <= width <= 1.2 and 0.08 <= height <= 1.2):
        raise ValueError("implausible board dimensions")
    axis_v /= height
    expected = corners[0] + width * axis_u + height * axis_v
    corner_residual = float(np.linalg.norm(corners[2] - expected))
    if validate_rectangle and corner_residual > max(0.025, 0.15 * min(width, height)):
        raise ValueError(f"board corners are inconsistent with a rectangle ({corner_residual:.3f} m)")
    if np.dot(corners[2] - corners[0], axis_u) < width * 0.75:
        raise ValueError("board corners are not consistently ordered")
    return BoardGeometry(corners[0], axis_u, axis_v, normal, width, height, corners, rms)


def ink_mask(rgb: np.ndarray, board_mask: np.ndarray, eraser_mask: np.ndarray | None,
             *, gray_threshold: int = 120, border_px: int = 20) -> np.ndarray:
    """Conservative dark-ink proxy; verification is visual evidence, not proof."""
    if rgb.ndim != 3 or rgb.shape[:2] != board_mask.shape or rgb.shape[2] != 3:
        raise ValueError("RGB and board mask dimensions differ")
    if not 0 <= gray_threshold <= 255 or border_px < 0:
        raise ValueError("invalid ink threshold or border")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inside = cv2.erode(board_mask.astype(np.uint8),
                       np.ones((2 * border_px + 1, 2 * border_px + 1), np.uint8)) > 0
    if eraser_mask is not None:
        inside &= ~cv2.dilate(eraser_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    return inside & (gray < gray_threshold)


def serpentine(board: BoardGeometry, target_points_base: np.ndarray,
               *, pad_width_m: float, pad_length_m: float, margin_m: float,
               row_overlap: float = 0.35,
               clearance_m: float = 0.08,
               body_width_m: float | None = None,
               body_length_m: float | None = None) -> list[dict[str, object]]:
    """Plan pad-centre hover/contact/retract waypoints over detected ink."""
    if not (0.01 <= pad_width_m <= 0.20 and 0.01 <= pad_length_m <= 0.20):
        raise ValueError("pad coverage footprint is not available")
    if body_width_m is not None and not pad_width_m <= body_width_m <= 0.25:
        raise ValueError("eraser body width does not bound the pad")
    if body_length_m is not None and not pad_length_m <= body_length_m <= 0.25:
        raise ValueError("eraser body length does not bound the pad")
    if not (0.0 <= margin_m <= 0.08 and 0.0 <= row_overlap < 0.8):
        raise ValueError("invalid board margin or row overlap")
    if not 0.03 <= clearance_m <= 0.20:
        raise ValueError("invalid clearance")
    local = board.local(np.asarray(target_points_base).reshape(-1, 3))
    if len(local) < 10 or not np.all(np.isfinite(local)):
        raise ValueError("too few valid ink points to plan wiping")
    half_u = (body_length_m or pad_length_m) / 2 + margin_m
    half_v = (body_width_m or pad_width_m) / 2 + margin_m
    if 2 * half_u > board.width_m or 2 * half_v > board.height_m:
        raise ValueError("eraser footprint does not fit inside the board margin")
    lower_reach = np.array([half_u - pad_length_m / 2, half_v - pad_width_m / 2])
    upper_reach = np.array([board.width_m - half_u + pad_length_m / 2,
                            board.height_m - half_v + pad_width_m / 2])
    # Do not discard sparse strokes as percentile outliers. A thin line or
    # dot can be covered by one pad placement even with zero extent on an axis.
    low, high = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
    if np.any(low < lower_reach) or np.any(high > upper_reach):
        raise ValueError("ink target is too close to the board edge for conservative eraser coverage")
    u0, u1 = np.clip([low[0], high[0]], half_u, board.width_m - half_u)
    v0, v1 = np.clip([low[1], high[1]], half_v, board.height_m - half_v)
    pitch = min(0.025, pad_width_m * (1.0 - row_overlap))
    rows = max(1, int(np.ceil((v1 - v0) / pitch)) + 1)
    if rows > 30:
        raise ValueError("too many wipe rows")
    waypoints: list[dict[str, object]] = []
    for row, v in enumerate(np.linspace(v0, v1, rows)):
        start, stop = (u0, u1) if row % 2 == 0 else (u1, u0)
        # Only the row endpoints are commanded; the Cartesian controller
        # interpolates each sweep. Retain row starts to avoid diagonal turns.
        for idx, u in enumerate((start, stop)):
            waypoints.append({"stage": "wipe", "row": row, "index": idx,
                              "pad_center_base_m": board.point(float(u), float(v)).tolist()})
    if len(waypoints) > 500:
        raise ValueError("wipe plan exceeds 500 contact waypoints; split the task")
    first = np.asarray(waypoints[0]["pad_center_base_m"])
    last = np.asarray(waypoints[-1]["pad_center_base_m"])
    return ([{"stage": "hover", "pad_center_base_m": (first + board.normal * clearance_m).tolist()}]
            + waypoints
            + [{"stage": "retract", "pad_center_base_m": (last + board.normal * clearance_m).tolist()}])
