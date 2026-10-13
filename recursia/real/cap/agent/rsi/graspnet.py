"""Contact-GraspNet service client and explicit optical-to-Franka TCP conversion."""
import base64
import io
import json
import urllib.request

import numpy as np

from cap.saved_scripts.fr3_runtime.geometry import points_from_mask, validate_transform


def encode_array(value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode_array(value):
    return np.load(io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False)


def default_grasp_to_ee():
    # Network x = finger separation, z = approach; Franka Hand TCP y = separation.
    # Network origin is the hand base, 103.4 mm behind its finger contact baseline.
    transform = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.],
                          [0., 0., 1., .1034], [0., 0., 0., 1.]])
    return transform


def select_grasp(grasps, scores, widths, camera_to_base, grasp_to_ee, *, max_width_m=.08):
    grasps, scores, widths = np.asarray(grasps), np.asarray(scores), np.asarray(widths)
    count = len(scores)
    if count == 0:
        raise ValueError("Contact-GraspNet returned no grasps for the segmented object")
    if grasps.shape != (count, 4, 4) or scores.shape != (count,) or widths.shape != (count,):
        raise ValueError("Contact-GraspNet candidate shapes do not agree")
    if not all(np.isfinite(a).all() for a in (grasps, scores, widths)):
        raise ValueError("Contact-GraspNet returned non-finite candidates")
    max_width_m = float(max_width_m)
    if not np.isfinite(max_width_m) or max_width_m <= 0:
        raise ValueError("invalid maximum hand opening")
    # Thin surfaces such as cup rims are valid network predictions. There is
    # no 5 mm minimum contact width; use the Hand's measured maximum opening.
    usable = np.flatnonzero((widths >= 0) & (widths <= max_width_m))
    if not len(usable):
        raise ValueError(f"Contact-GraspNet widths {widths.min():.6f}..{widths.max():.6f} m "
                         f"do not fit the hand opening 0..{max_width_m:.6f} m")
    index = int(usable[np.argmax(scores[usable])])
    transform = (validate_transform(camera_to_base) @ validate_transform(grasps[index])
                 @ validate_transform(grasp_to_ee))
    return transform, float(widths[index]), index


class ContactGraspNetClient:
    def __init__(self, url="http://127.0.0.1:8115", timeout_s=120):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def predict(self, capture, mask):
        # Undistort using the same RealSense calibration as object localization.
        # Send camera-optical point clouds; no robot/world coordinates enter the network.
        full = points_from_mask(capture.depth_m, capture.intrinsics, np.eye(4),
                                np.ones_like(mask, dtype=bool), stride=2,
                                distortion=capture.distortion_coeffs)
        segment = points_from_mask(capture.depth_m, capture.intrinsics, np.eye(4),
                                   mask, stride=1, distortion=capture.distortion_coeffs)
        if len(segment) < 20:
            raise ValueError("insufficient target depth for Contact-GraspNet")
        body = {"pc_full_base64": encode_array(full.astype(np.float32)),
                "pc_segment_base64": encode_array(segment.astype(np.float32)),
                "forward_passes": 1}
        request = urllib.request.Request(self.url + "/plan_point_clouds", json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.load(response)
        if result.get("model") != "contact_graspnet_pytorch" or result.get("frame") != "camera_optical":
            raise ValueError("unexpected Contact-GraspNet model or output frame")
        return {key: decode_array(result[key + "_base64"])
                for key in ("grasps", "scores", "widths", "contact_pts")}
