"""Two named, timestamped camera snapshots; never connect to the robot."""
from pathlib import Path

import numpy as np
import yaml

from cap.saved_scripts.fr3_runtime.camera import Capture, capture_realsense
from .store import write_json


def capture_observation(cfg, folder, *, snapshot=None, camera_fn=capture_realsense):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    cameras = yaml.safe_load(cfg.hardware_config_path.read_text())["hardware"]["cameras"]
    serials = {"front": cfg.external_serial, "wrist": str(cameras["wrist"]["serial"])}
    if serials["front"] == serials["wrist"]:
        raise ValueError("front and wrist cameras must have distinct serials")
    if snapshot is not None:
        # A front-only recording is not evidence of a wrist view.
        if not all((Path(snapshot) / view / "meta.json").is_file() for view in serials):
            raise ValueError("dual-view dry-run requires front/ and wrist/ RGB-D snapshots")
    views = {}
    for name, serial in serials.items():
        if snapshot is not None:
            frame = Capture.load(Path(snapshot) / name)
        elif camera_fn is capture_realsense:
            # Observation can show an occluded/too-close view. Metric localization
            # retains its existing depth quality checks in get_object_pose().
            frame = camera_fn(serial, min_valid_depth_fraction=0.0)
        else:
            frame = camera_fn(serial)
        if frame.serial != serial:
            raise ValueError(f"{name} camera serial mismatch")
        frame.save(folder / name)
        views[name] = {"serial": serial, "captured_at": frame.captured_at,
                       "rgb_path": str((folder / name / "rgb.png").resolve()),
                       "depth_path": str((folder / name / "depth_m.npy").resolve()),
                       "intrinsics": frame.intrinsics.tolist(),
                       "depth_valid_fraction": float(np.mean(frame.depth_m > 0)),
                       "frame": f"{name} color optical",
                       "T_base_camera_optical": cfg.camera_to_base.tolist() if name == "front" else None}
    packet = {"schema": "aspire.rsi.observation.v1", "views": views,
              "capture_mode": "sequential snapshots, not synchronized video",
              "physical_motion_commanded": False}
    write_json(folder / "observation.json", packet)
    return packet


def image_paths(observation):
    return [Path(observation["views"][view]["rgb_path"]) for view in ("front", "wrist")]


def public_observation(observation):
    """No run paths, task IDs or previous experiment data enter the model."""
    return {"views": [{"name": name, "captured_at": observation["views"][name]["captured_at"],
                       "frame": observation["views"][name]["frame"]}
                      for name in ("front", "wrist")],
            "capture_mode": observation["capture_mode"]}


def array_observation(folder, packet):
    views = {}
    for name in ("front", "wrist"):
        frame = Capture.load(Path(folder) / name)
        views[name] = {"images": {"rgb": frame.rgb, "depth": frame.depth_m},
                       "intrinsics": frame.intrinsics, **packet["views"][name]}
    return {"views": views, "capture_mode": packet["capture_mode"]}
