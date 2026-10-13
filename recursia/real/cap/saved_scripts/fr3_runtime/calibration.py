"""Offline extrinsic solve from measured base-frame fiducials and clicked pixels."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import validate_transform


def solve_external_extrinsic(intrinsics: np.ndarray,
                             distortion_coeffs: np.ndarray,
                             correspondences: list[dict[str, Any]]) -> dict[str, Any]:
    """Return T_base_camera_optical; never treats simulator coordinates as real.

    Each correspondence must be an independently measured base-frame 3-D point
    and its pixel in the same fixed external camera. Six non-coplanar points
    are required to avoid a deceptively plausible planar pose solution.
    """
    if len(correspondences) < 6:
        raise ValueError("at least six 3-D/2-D correspondences are required")
    world = np.asarray([row["base_xyz_m"] for row in correspondences], dtype=np.float64)
    image = np.asarray([row["pixel_uv"] for row in correspondences], dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    coeffs = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)
    if world.shape != (len(correspondences), 3) or image.shape != (len(correspondences), 2):
        raise ValueError("invalid correspondence shape")
    if not np.all(np.isfinite(world)) or not np.all(np.isfinite(image)) or k.shape != (3, 3):
        raise ValueError("correspondences or intrinsics contain invalid values")
    singular = np.linalg.svd(world - world.mean(axis=0), compute_uv=False)
    if singular[-1] < 0.01:
        raise ValueError("fiducial points are coplanar or nearly so")
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        world, image, k, coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0, iterationsCount=500, confidence=0.999)
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError("camera calibration did not find six consistent points")
    r_cam_base, _ = cv2.Rodrigues(rvec)
    t_cam_base = tvec.reshape(3)
    base_camera = np.eye(4)
    base_camera[:3, :3] = r_cam_base.T
    base_camera[:3, 3] = -r_cam_base.T @ t_cam_base
    validate_transform(base_camera)
    projections, _ = cv2.projectPoints(world, rvec, tvec, k, coeffs)
    errors = np.linalg.norm(projections.reshape(-1, 2) - image, axis=1)
    rms = float(np.sqrt(np.mean(errors[inliers.ravel()] ** 2)))
    if rms > 2.0:
        raise ValueError(f"reprojection RMS {rms:.2f} px is too high")
    return {"T_base_camera_optical": base_camera.tolist(),
            "reprojection_rms_px": rms,
            "max_reprojection_px": float(np.max(errors)),
            "inlier_count": int(len(inliers)), "total_count": int(len(world)),
            "warning": "Review against held-out measured 3-D points before enabling motion."}


def aruco_corners_marker_m(side_m: float) -> np.ndarray:
    """Marker corners in OpenCV's TL, TR, BR, BL order; local +Y points up."""
    if not np.isfinite(side_m) or not 0.02 <= side_m <= 0.20:
        raise ValueError("marker black-square side must be 0.02..0.20 m")
    half = side_m / 2.0
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], dtype=np.float64)


def _pose_from_vector(vector: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    pose[:3, 3] = vector[3:6]
    return pose


def _pose_vector(pose: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]


def solve_handeye_external(intrinsics: np.ndarray, distortion_coeffs: np.ndarray,
                           marker_side_m: float, samples: list[dict[str, Any]],
                           initial_base_camera: np.ndarray) -> dict[str, Any]:
    """Jointly fit fixed camera pose and unknown rigid EE-to-marker mount.

    Each sample supplies a stationary FR3 O_T_EE pose and four ArUco corners.
    Every fifth sample is held out while fitting, then all samples determine
    the final transform after the held-out reprojection check passes.
    """
    if len(samples) < 12:
        raise ValueError("collect at least 12 stationary marker poses")
    k = np.asarray(intrinsics, dtype=np.float64)
    distortion = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)
    if k.shape != (3, 3) or not np.all(np.isfinite(k)) or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("invalid camera intrinsics")
    if distortion.shape not in ((4,), (5,), (8,)) or not np.all(np.isfinite(distortion)):
        raise ValueError("invalid OpenCV distortion coefficients")
    initial = validate_transform(initial_base_camera)
    marker_points = aruco_corners_marker_m(marker_side_m)
    ee_poses = np.asarray([validate_transform(row["ee_pose_base"]) for row in samples])
    image_points = np.asarray([row["corners_uv"] for row in samples], dtype=np.float64)
    if image_points.shape != (len(samples), 4, 2) or not np.all(np.isfinite(image_points)):
        raise ValueError("each sample needs four finite marker corners")
    translations = ee_poses[:, :3, 3]
    position_singular = np.linalg.svd(translations - translations.mean(axis=0),
                                      compute_uv=False)
    relative_rotation = (Rotation.from_matrix(ee_poses[0, :3, :3]).inv()
                         * Rotation.from_matrix(ee_poses[:, :3, :3])).as_rotvec()
    rotation_singular = np.linalg.svd(relative_rotation, compute_uv=False)
    if position_singular[-1] < 0.04 or rotation_singular[1] < 0.25:
        raise ValueError("marker poses need 3-D position spread and wrist tilts about two axes")

    mount_estimates = []
    for ee, corners in zip(ee_poses, image_points):
        ok, rvec, tvec = cv2.solvePnP(marker_points, corners, k, distortion,
                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            raise ValueError("marker PnP failed for one sample")
        camera_marker = np.eye(4)
        camera_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
        camera_marker[:3, 3] = tvec.ravel()
        mount_estimates.append(np.linalg.inv(ee) @ initial @ camera_marker)
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_matrix(
        np.asarray([pose[:3, :3] for pose in mount_estimates])).mean().as_matrix()
    mount[:3, 3] = np.median([pose[:3, 3] for pose in mount_estimates], axis=0)
    x0 = np.r_[_pose_vector(initial), _pose_vector(mount)]

    def errors(vector: np.ndarray, indices: np.ndarray) -> np.ndarray:
        camera_base = np.linalg.inv(_pose_from_vector(vector[:6]))
        ee_marker = _pose_from_vector(vector[6:])
        residuals = []
        for index in indices:
            camera_marker = camera_base @ ee_poses[index] @ ee_marker
            points_camera = marker_points @ camera_marker[:3, :3].T + camera_marker[:3, 3]
            if np.any(points_camera[:, 2] <= 0.05):
                residuals.append(np.full((4, 2), 1000.0))
                continue
            projected, _ = cv2.projectPoints(points_camera, np.zeros(3), np.zeros(3),
                                             k, distortion)
            residuals.append(projected.reshape(4, 2) - image_points[index])
        return np.asarray(residuals).ravel()

    indices = np.arange(len(samples))
    held = indices[4::5]
    train = np.setdiff1d(indices, held)
    fit = least_squares(errors, x0, args=(train,), loss="soft_l1", f_scale=2.0,
                        max_nfev=2000)
    if not fit.success:
        raise ValueError(f"camera hand-eye solve failed: {fit.message}")
    held_errors = errors(fit.x, held).reshape(-1, 2)
    held_rms = float(np.sqrt(np.mean(np.sum(held_errors ** 2, axis=1))))
    held_max = float(np.max(np.linalg.norm(held_errors, axis=1)))
    if held_rms > 3.0 or held_max > 6.0:
        raise ValueError(f"held-out marker reprojection too high: RMS {held_rms:.2f} px, "
                         f"max {held_max:.2f} px; recapture varied poses or check marker size")
    final = least_squares(errors, fit.x, args=(indices,), loss="soft_l1", f_scale=2.0,
                          max_nfev=2000)
    if not final.success:
        raise ValueError(f"final camera hand-eye solve failed: {final.message}")
    all_errors = errors(final.x, indices).reshape(len(samples), 4, 2)
    per_sample = np.sqrt(np.mean(np.sum(all_errors ** 2, axis=2), axis=1))
    all_rms = float(np.sqrt(np.mean(per_sample ** 2)))
    all_max = float(np.max(np.linalg.norm(all_errors, axis=2)))
    if all_rms > 3.0 or all_max > 6.0:
        raise ValueError(f"marker reprojection too high: RMS {all_rms:.2f} px, "
                         f"max {all_max:.2f} px; review per-sample marker detection")
    base_camera = validate_transform(_pose_from_vector(final.x[:6]))
    ee_marker = validate_transform(_pose_from_vector(final.x[6:]))
    if np.linalg.norm(ee_marker[:3, 3]) > 0.5:
        raise ValueError("solved marker mount is over 0.5 m from the EE; check pose diversity")
    return {"T_base_camera_optical": base_camera.tolist(),
            "T_ee_marker": ee_marker.tolist(),
            "sample_count": len(samples), "held_out_count": len(held),
            "held_out_rms_px": held_rms, "held_out_max_px": held_max,
            "all_rms_px": all_rms, "all_max_px": all_max,
            "per_sample_rms_px": per_sample.tolist(),
            "position_spread_singular_m": position_singular.tolist(),
            "rotation_spread_singular_rad": rotation_singular.tolist(),
            "initial_to_solved_translation_m": float(np.linalg.norm(
                base_camera[:3, 3] - initial[:3, 3]))}
