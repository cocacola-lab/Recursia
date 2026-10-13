"""Read-only, aligned RealSense captures for FR3 task evidence."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# Use the RealSense profile for this station. Its driver requests
# 640x480 BGR8 + Z16 at 30 fps; 15 fps gave misleading depth timeouts here.
CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def _require_usb3(device: object, rs: object, serial: str) -> str:
    """Apply the same USB transport requirement as the camera driver."""
    usb = (device.get_info(rs.camera_info.usb_type_descriptor)
           if device.supports(rs.camera_info.usb_type_descriptor) else "unknown")
    if not str(usb).startswith("3"):
        raise RuntimeError(f"camera {serial} negotiated USB {usb}; replug on a USB3 port")
    return str(usb)


@dataclass(frozen=True)
class Capture:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    serial: str
    captured_at: str
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    color_frame_number: int | None = None
    depth_frame_number: int | None = None
    color_timestamp_ms: float | None = None
    depth_timestamp_ms: float | None = None
    usb_type: str | None = None
    depth_scale_m: float | None = None

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        if not cv2.imwrite(str(folder / "rgb.png"), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("could not write RGB image")
        np.save(folder / "depth_m.npy", self.depth_m)
        (folder / "meta.json").write_text(json.dumps({
            "serial": self.serial, "captured_at": self.captured_at,
            "intrinsics": self.intrinsics.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
            "depth_shape": list(self.depth_m.shape),
            "color_frame_number": self.color_frame_number,
            "depth_frame_number": self.depth_frame_number,
            "color_timestamp_ms": self.color_timestamp_ms,
            "depth_timestamp_ms": self.depth_timestamp_ms,
            "usb_type": self.usb_type,
            "depth_scale_m": self.depth_scale_m,
            "profile": {"width": CAMERA_WIDTH, "height": CAMERA_HEIGHT, "fps": CAMERA_FPS},
        }, indent=2) + "\n", encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> Capture:
        folder = Path(folder)
        bgr = cv2.imread(str(folder / "rgb.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("snapshot has no rgb.png")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        depth = np.load(folder / "depth_m.npy", allow_pickle=False)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if depth.shape != rgb.shape[:2]:
            raise ValueError("snapshot RGB/depth size mismatch")
        return cls(rgb, depth, np.asarray(meta["intrinsics"], dtype=np.float64),
                   str(meta["serial"]), str(meta["captured_at"]),
                   np.asarray(meta.get("distortion_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64),
                   meta.get("color_frame_number"), meta.get("depth_frame_number"),
                   meta.get("color_timestamp_ms"), meta.get("depth_timestamp_ms"),
                   meta.get("usb_type"), meta.get("depth_scale_m"))


def capture_realsense(serial: str, *, warmup_frames: int = 20,
                      min_valid_depth_fraction: float = 0.5) -> Capture:
    """Open only the named camera; never connect to FCI or issue robot motion."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    profile = pipeline.start(cfg)
    try:
        device = profile.get_device()
        actual = device.get_info(rs.camera_info.serial_number)
        if actual != serial:
            raise RuntimeError(f"RealSense serial mismatch: expected {serial}, got {actual}")
        usb = _require_usb3(device, rs, serial)
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)
        scale = device.first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(max(5, warmup_frames)):
            frames = align.process(pipeline.wait_for_frames(5000))
            if not frames.get_color_frame() or not frames.get_depth_frame():
                raise RuntimeError("missing aligned RGB/depth frame during warmup")
        assert frames is not None
        colour, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not colour or not depth:
            raise RuntimeError("missing aligned RGB/depth frame")
        intr = colour.profile.as_video_stream_profile().intrinsics
        raw_coeffs = np.asarray(intr.coeffs[:5], dtype=np.float64)
        if intr.model in (rs.distortion.none, rs.distortion.inverse_brown_conrady):
            # This station's external D435i reports inverse Brown with all
            # coefficients zero. Nonzero inverse coefficients are not a
            # forward Brown model and must not enter OpenCV undistortPoints.
            if np.max(np.abs(raw_coeffs)) > 1e-9:
                raise RuntimeError("nonzero inverse Brown coefficients need a RealSense-specific deprojection")
            distortion = np.zeros(5, dtype=np.float64)
        elif intr.model in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
            distortion = raw_coeffs
        else:
            raise RuntimeError(f"unsupported RealSense colour distortion model: {intr.model}")
        k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
        rgb = cv2.cvtColor(np.asanyarray(colour.get_data()), cv2.COLOR_BGR2RGB)
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * float(scale)
        if float(np.mean((depth_m > 0.1) & (depth_m < 2.0))) < min_valid_depth_fraction:
            raise RuntimeError(f"aligned depth has fewer than {min_valid_depth_fraction:.0%} valid pixels")
        return Capture(rgb, depth_m, k, actual, time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       distortion, int(colour.get_frame_number()), int(depth.get_frame_number()),
                       float(colour.get_timestamp()), float(depth.get_timestamp()), usb, float(scale))
    finally:
        pipeline.stop()


class ColorVideoRecorder:
    """External-camera evidence for a physical phase; no robot connection."""

    def __init__(self, serial: str, path: str | Path):
        self.serial = serial
        self.path = Path(path)
        self._pipeline = None
        self._writer = None
        self._thread = None
        self._stop = threading.Event()
        self._first = threading.Event()
        self._error: BaseException | None = None
        self.frames = 0

    def start(self) -> ColorVideoRecorder:
        import pyrealsense2 as rs

        self.path.parent.mkdir(parents=True, exist_ok=True)
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
        profile = pipeline.start(cfg)
        try:
            actual = profile.get_device().get_info(rs.camera_info.serial_number)
            if actual != self.serial:
                raise RuntimeError("recording camera serial mismatch")
            _require_usb3(profile.get_device(), rs, self.serial)
        except BaseException:
            pipeline.stop()
            raise
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), float(CAMERA_FPS),
                                 (CAMERA_WIDTH, CAMERA_HEIGHT))
        if not writer.isOpened():
            pipeline.stop()
            raise RuntimeError("could not start task video recorder")
        self._pipeline, self._writer = pipeline, writer

        def loop() -> None:
            try:
                while not self._stop.is_set():
                    frames = pipeline.wait_for_frames(5000)
                    colour = frames.get_color_frame()
                    if not colour:
                        continue
                    writer.write(np.asanyarray(colour.get_data()))
                    self.frames += 1
                    self._first.set()
            except BaseException as exc:
                if not self._stop.is_set():
                    self._error = exc
                self._first.set()

        self._thread = threading.Thread(target=loop, name="fr3-whiteboard-video", daemon=True)
        self._thread.start()
        if not self._first.wait(7.0) or self._error is not None or self.frames == 0:
            self.stop()
            raise RuntimeError(f"external camera video did not produce a frame: {self._error}")
        return self

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"task video recorder failed: {self._error}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> ColorVideoRecorder:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()
