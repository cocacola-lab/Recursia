"""Runtime whiteboard/eraser detection from a saved RGB-D frame.

The segmentation backend is ASPIRE's existing SAM3 /segment_all service.
This module does not open a camera, robot, or model server; failure to find a
well-formed target is a hard planning error.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from .camera import Capture
from .config import WhiteboardConfig
from .geometry import estimate_board, polygon_mask
from .geometry import points_from_mask
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SceneDetection:
    board_corners_uv: np.ndarray
    eraser_polygon_uv: np.ndarray
    grasp_point_uv: np.ndarray
    board_score: float
    eraser_score: float
    board_plane_rms_m: float
    eraser_on_board: bool

    def as_dict(self) -> dict[str, Any]:
        return {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in vars(self).items()}


class Sam3Client:
    def __init__(self, url: str, *, timeout_s: float = 60.0):
        if not url.startswith("http://127.0.0.1:") and not url.startswith("http://localhost:"):
            raise ValueError("FR3 scene segmentation requires a localhost SAM3 endpoint")
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def segment_all(self, rgb: np.ndarray, prompt: str) -> list[tuple[np.ndarray, float]]:
        buffer = io.BytesIO()
        np.save(buffer, rgb)
        payload = json.dumps({"text": prompt,
                              "image_b64": base64.b64encode(buffer.getvalue()).decode(),
                              "score_threshold": 0.3}).encode()
        request = urllib.request.Request(self.url + "/segment_all", data=payload,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            data = json.loads(response.read())
        detections = []
        for item in data["detections"]:
            mask = np.load(io.BytesIO(base64.b64decode(item["mask_b64"])), allow_pickle=False)
            if mask.shape != rgb.shape[:2]:
                raise ValueError("SAM3 returned a mask with the wrong image dimensions")
            detections.append((mask.astype(bool), float(item["score"])))
        return detections


def detect_board(capture: Capture, cfg: WhiteboardConfig,
                 client: Sam3Client | None = None) -> tuple[np.ndarray, float]:
    """Find the visible board independently after the eraser has been held."""
    client = client or Sam3Client(cfg.segmentation_url)
    candidates = []
    for mask, score in client.segment_all(capture.rgb, cfg.board_prompt):
        if score < 0.5:
            continue
        try:
            corners, _ = _board_rectangle(capture, cfg, mask, None)
            candidates.append((score, corners))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("no whiteboard detection passed geometry and depth checks")
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
        raise ValueError("multiple similar whiteboards; scene is ambiguous")
    return candidates[0][1], candidates[0][0]


def _quad(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("whiteboard segmentation is empty")
    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    area = float(cv2.contourArea(hull))
    if area < 4000 or area > mask.size * 0.65:
        raise ValueError("whiteboard mask has implausible size")
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for fraction in (0.01, 0.015, 0.02, 0.03, 0.04):
        candidate = cv2.approxPolyDP(hull, perimeter * fraction, True).reshape(-1, 2)
        if len(candidate) == 4 and cv2.contourArea(candidate.astype(np.float32)) >= area * 0.85:
            quad = candidate.astype(float)
            break
    if quad is None:
        # This is only the seed for plane fitting. The final rectangle is
        # recovered in 3-D from the depth-supported semantic mask below.
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
    if np.any(quad[:, 0] < 3) or np.any(quad[:, 0] > mask.shape[1] - 4) or np.any(quad[:, 1] < 3) or np.any(quad[:, 1] > mask.shape[0] - 4):
        raise ValueError("whiteboard is clipped by the camera frame")
    top = quad[np.argsort(quad[:, 1])[:2]]
    bottom = quad[np.argsort(quad[:, 1])[2:]]
    ordered = np.array([top[np.argmin(top[:, 0])], top[np.argmax(top[:, 0])],
                        bottom[np.argmax(bottom[:, 0])], bottom[np.argmin(bottom[:, 0])]])
    if not cv2.isContourConvex(ordered.astype(np.float32)):
        raise ValueError("whiteboard corners are not a convex quadrilateral")
    return ordered


def _eraser_polygon(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("eraser segmentation is empty")
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 100 or area > mask.size * 0.15:
        raise ValueError("eraser mask has implausible size")
    polygon = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    if not eroded.any():
        raise ValueError("eraser mask is too narrow for a grasp target")
    distances = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(distances), distances.shape)
    return polygon, np.array([x, y], dtype=float)


def _board_rectangle(capture: Capture, cfg: WhiteboardConfig,
                     board_mask: np.ndarray, eraser_mask: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Estimate a rectangular board ROI on the depth-supported surface."""
    rough = _quad(board_mask)
    provisional = estimate_board(capture.depth_m, capture.intrinsics,
                                 cfg.camera_to_base, rough, board_mask, eraser_mask,
                                 distortion=capture.distortion_coeffs,
                                 validate_rectangle=False)
    usable = board_mask.copy()
    if eraser_mask is not None:
        usable &= ~cv2.dilate(eraser_mask.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    points = points_from_mask(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                              usable, stride=3, distortion=capture.distortion_coeffs)
    points = points[np.abs(provisional.local(points)[:, 2]) < 0.008]
    if len(points) < 150:
        raise ValueError("board mask has too little plane-supported depth")
    rectangle = cv2.minAreaRect(points[:, :2].astype(np.float32))
    if min(rectangle[1]) < 0.08 or max(rectangle[1]) > 1.2:
        raise ValueError("board plane rectangle has implausible dimensions")
    camera_from_base = np.linalg.inv(cfg.camera_to_base)
    height, width = board_mask.shape
    rectangle_center = np.asarray(rectangle[0], dtype=float)
    outer_box = cv2.boxPoints(rectangle).astype(float)
    corners = None
    roi = None
    # Corner protectors and an eraser can make a rectangular board's semantic
    # mask hexagonal.  The minimum-area rectangle around that mask then extends
    # beyond the observed surface.  Select the largest concentric rectangle
    # that remains in-frame and is supported by the segmentation instead of
    # rejecting an otherwise sound plane fit.
    for scale in np.arange(1.0, 0.64, -0.025):
        base = np.zeros((4, 3), dtype=float)
        base[:, :2] = rectangle_center + (outer_box - rectangle_center) * scale
        normal = provisional.normal
        base[:, 2] = (provisional.origin[2] -
                      ((base[:, :2] - provisional.origin[:2]) @ normal[:2]) / normal[2])
        camera_points = base @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
        if np.any(camera_points[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3),
                                          capture.intrinsics, capture.distortion_coeffs)
        uv = projected.reshape(4, 2)
        top = uv[np.argsort(uv[:, 1])[:2]]
        bottom = uv[np.argsort(uv[:, 1])[2:]]
        candidate = np.array([top[np.argmin(top[:, 0])], top[np.argmax(top[:, 0])],
                              bottom[np.argmax(bottom[:, 0])], bottom[np.argmin(bottom[:, 0])]])
        if (np.any(candidate[:, 0] < 3) or np.any(candidate[:, 0] > width - 4)
                or np.any(candidate[:, 1] < 3) or np.any(candidate[:, 1] > height - 4)):
            continue
        candidate_roi = polygon_mask(board_mask.shape, candidate)
        support = np.count_nonzero(candidate_roi & board_mask) / np.count_nonzero(candidate_roi)
        if support >= 0.65:
            corners, roi = candidate, candidate_roi
            break
    if corners is None or roi is None:
        raise ValueError("no depth-supported board rectangle fits inside the observed mask")
    verified = estimate_board(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                              corners, roi, eraser_mask,
                              distortion=capture.distortion_coeffs)
    return corners, verified.plane_rms_m


def detect_scene(capture: Capture, cfg: WhiteboardConfig,
                 client: Sam3Client | None = None) -> SceneDetection:
    if cfg.camera_to_base is None or capture.serial != cfg.external_serial:
        raise ValueError("scene detection requires the configured external camera pose")
    client = client or Sam3Client(cfg.segmentation_url)
    boards = client.segment_all(capture.rgb, cfg.board_prompt)
    erasers = client.segment_all(capture.rgb, cfg.eraser_prompt)
    if not boards or not erasers:
        raise ValueError("SAM3 did not find both a whiteboard and an eraser")
    candidates = []
    rejections = []
    for board_mask, board_score in boards:
        if board_score < 0.5:
            continue
        for eraser_mask, eraser_score in erasers:
            if eraser_score < 0.5:
                continue
            try:
                eraser, grasp = _eraser_polygon(eraser_mask)
                corners, rms = _board_rectangle(capture, cfg, board_mask, eraser_mask)
                roi = polygon_mask(capture.depth_m.shape, corners)
                overlap = np.count_nonzero(roi & eraser_mask) / np.count_nonzero(eraser_mask)
                candidates.append((board_score + eraser_score, SceneDetection(
                    corners, eraser, grasp, board_score, eraser_score,
                    rms, overlap > 0.25)))
            except ValueError as exc:
                rejections.append(str(exc))
                continue
    if not candidates:
        detail = f": {rejections[0]}" if rejections else ""
        raise ValueError("no SAM3 board/eraser pair passed geometry and depth checks" + detail)
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
        raise ValueError("multiple similar board/eraser pairs; scene is ambiguous")
    return candidates[0][1]


def resolve_scene(capture: Capture, cfg: WhiteboardConfig,
                  client: Sam3Client | None = None) -> tuple[WhiteboardConfig, SceneDetection]:
    """Derive scene pixels and pose proposals; only called before preview."""
    detection = detect_scene(capture, cfg, client)
    corners = detection.board_corners_uv
    board_mask = polygon_mask(capture.depth_m.shape, corners)
    eraser_mask = polygon_mask(capture.depth_m.shape, detection.eraser_polygon_uv)
    board = estimate_board(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                           corners, board_mask, eraser_mask,
                           distortion=capture.distortion_coeffs)
    if cfg.pad_normal_in_ee is None or cfg.pad_length_axis_in_ee is None:
        raise ValueError("measured pad axes are needed to propose a wipe orientation")
    source_x = cfg.pad_length_axis_in_ee
    source_y = cfg.pad_normal_in_ee
    source = np.column_stack([source_x, source_y, np.cross(source_x, source_y)])
    target_x, target_y = board.axis_u, -board.normal
    target = np.column_stack([target_x, target_y, np.cross(target_x, target_y)])
    wipe_quat = Rotation.from_matrix(target @ source.T).as_quat()
    points = points_from_mask(capture.depth_m, capture.intrinsics, cfg.camera_to_base,
                              eraser_mask, stride=2,
                              distortion=capture.distortion_coeffs)
    local = board.local(points)
    if len(local) < 50:
        raise ValueError("eraser has too little valid depth for a footprint estimate")
    center = np.median(local[:, :2], axis=0)
    keep = np.linalg.norm(local[:, :2] - center, axis=1) < 0.15
    if np.count_nonzero(keep) < 50:
        raise ValueError("eraser depth footprint is ambiguous")
    rectangle = cv2.minAreaRect(local[keep, :2].astype(np.float32))
    short, long = sorted(float(size) for size in rectangle[1])
    if short < 0.015 or long > 0.20:
        raise ValueError("observed eraser footprint has implausible dimensions")
    box = cv2.boxPoints(rectangle).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    short_axis_local = edges[int(np.argmin(np.linalg.norm(edges, axis=1)))]
    short_axis_local /= np.linalg.norm(short_axis_local)
    grasp_y = short_axis_local[0] * board.axis_u + short_axis_local[1] * board.axis_v
    # Franka Hand closes along EE Y. Its closing axis must match the short
    # edge whose length becomes grasp_width_m, even when the eraser is rotated.
    grasp_z = -board.normal
    grasp_x = np.cross(grasp_y, grasp_z)
    # The two signs have identical finger geometry. Keep EE X toward the
    # board's top-to-bottom direction, matching the observed hand approach.
    if grasp_x @ board.axis_v < 0:
        grasp_x = -grasp_x
        grasp_y = -grasp_y
    grasp_quat = Rotation.from_matrix(np.column_stack([grasp_x, grasp_y, grasp_z])).as_quat()
    width = short - 0.004
    if not 0.005 <= width <= 0.075:
        raise ValueError(f"observed eraser grasp width {width:.3f} m is not plausible")
    # The visible shell bounds where the held eraser may contact the board.
    # Half-size contact dimensions are a deliberately conservative coverage
    # estimate; after a real wipe, the visual verifier checks the result.
    pad_width = max(0.01, short * 0.5)
    pad_length = max(0.01, long * 0.5)
    resolved = replace(cfg,
                       board_corners_uv=cfg.board_corners_uv if cfg.board_corners_uv is not None else corners,
                       eraser_polygon_uv=cfg.eraser_polygon_uv if cfg.eraser_polygon_uv is not None else detection.eraser_polygon_uv,
                       grasp_point_uv=cfg.grasp_point_uv if cfg.grasp_point_uv is not None else detection.grasp_point_uv,
                       grasp_quat_xyzw=cfg.grasp_quat_xyzw if cfg.grasp_quat_xyzw is not None else grasp_quat,
                       wipe_quat_xyzw=cfg.wipe_quat_xyzw if cfg.wipe_quat_xyzw is not None else wipe_quat,
                       grasp_width_m=cfg.grasp_width_m if cfg.grasp_width_m is not None else width,
                       pad_width_m=cfg.pad_width_m if cfg.pad_width_m is not None else pad_width,
                       pad_length_m=cfg.pad_length_m if cfg.pad_length_m is not None else pad_length,
                       eraser_body_width_m=short + 0.01,
                       eraser_body_length_m=long + 0.01)
    resolved.require_plan()
    return resolved, detection
