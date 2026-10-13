#!/usr/bin/env python3
"""Read-only FR3 hand-eye capture for the fixed external RealSense camera.

Attach one printed ArUco marker rigidly to the Hand or wrist. The solver
estimates both camera-to-base and marker-to-EE; no marker mount measurement is
needed. The operator moves the arm in Desk between separate sample commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cap.saved_scripts.fr3_runtime.calibration import (aruco_corners_marker_m,
                                                       solve_handeye_external)  # noqa: E402
from cap.saved_scripts.fr3_runtime.camera import capture_realsense  # noqa: E402
from cap.saved_scripts.fr3_placement.config import load_config  # noqa: E402
from robot.franka import Fr3Client  # noqa: E402

MARKER_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_NAME = "DICT_4X4_50"


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_numbered_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        folder = parent / f"{number:04d}"
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise RuntimeError(f"no free numbered directory under {parent}")


def write_marker_svg(path: Path, *, marker_id: int, side_mm: float) -> None:
    aruco_corners_marker_m(side_mm / 1000.0)
    if not 0 <= marker_id < 50:
        raise ValueError("DICT_4X4_50 marker ID must be 0..49")
    modules = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY), marker_id, 6)
    page_mm = side_mm * 8.0 / 6.0  # one white module of margin on every side
    squares = [f'<rect x="{x + 1}" y="{y + 1}" width="1" height="1"/>'
               for y in range(6) for x in range(6) if modules[y, x] == 0]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_mm:g}mm" '
           f'height="{page_mm:g}mm" viewBox="0 0 8 8">'
           '<rect width="8" height="8" fill="white"/>'
           '<g fill="black">' + "".join(squares) + '</g></svg>\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"marker file exists: {path}")
    path.write_text(svg, encoding="utf-8")


def _marker_corners(rgb: np.ndarray, marker_id: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY), params)
    corners, ids, _ = detector.detectMarkers(gray)
    matches = ([corner.reshape(4, 2) for corner, found_id in zip(corners, ids.ravel())
               if int(found_id) == marker_id] if ids is not None else [])
    if len(matches) != 1:
        raise ValueError(f"expected one visible ArUco ID {marker_id}, found {len(matches)}")
    result = np.asarray(matches[0], dtype=np.float64)
    edges = np.linalg.norm(np.roll(result, -1, axis=0) - result, axis=1)
    if float(np.min(edges)) < 35.0:
        raise ValueError("marker is too small in the image; move it closer to the camera")
    h, w = gray.shape
    if np.any(result[:, 0] < 8) or np.any(result[:, 0] > w - 9) or np.any(
            result[:, 1] < 8) or np.any(result[:, 1] > h - 9):
        raise ValueError("marker is clipped by the camera image")
    return result


def collect(config_path: Path, session: Path, *, marker_id: int, side_mm: float) -> dict:
    aruco_corners_marker_m(side_mm / 1000.0)
    if not 0 <= marker_id < 50:
        raise ValueError("DICT_4X4_50 marker ID must be 0..49")
    session = session.expanduser().resolve()
    cfg = load_config(config_path)
    task_yaml = yaml.safe_load(cfg.path.read_text(encoding="utf-8"))
    if (task_yaml.get("camera") or {}).get("T_base_camera_optical") is not None:
        raise ValueError("clear the task-local camera transform override before station calibration")
    station = Path(task_yaml["station_model"]).expanduser()
    if not station.is_absolute():
        station = cfg.path.parent / station
    station = station.resolve()
    if not station.is_file() or cfg.camera_to_base is None:
        raise ValueError("task config needs a station model with an initial camera pose")
    if not session.exists():
        session.mkdir(parents=True)
    metadata_path = session / "session.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != "aspire.fr3.camera_handeye_session.v1":
            raise ValueError("invalid FR3 hand-eye session")
        for name, expected in (("robot_ip", cfg.robot_ip),
                               ("camera_serial", cfg.external_serial),
                               ("marker_id", marker_id), ("marker_side_mm", side_mm)):
            if metadata[name] != expected:
                raise ValueError(f"session {name} differs from this collection command")
        if metadata["station_sha256"] != _digest(station):
            raise RuntimeError("station model changed during collection; start a new session")
    elif any(session.iterdir()):
        raise FileExistsError("session directory is nonempty but has no session.json")
    else:
        metadata = {"schema": "aspire.fr3.camera_handeye_session.v1",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "config_path": str(config_path.resolve()),
                    "station_path": str(station), "station_sha256": _digest(station),
                    "robot_ip": cfg.robot_ip, "camera_serial": cfg.external_serial,
                    "marker_dictionary": MARKER_NAME, "marker_id": marker_id,
                    "marker_side_mm": side_mm,
                    "initial_T_base_camera_optical": cfg.camera_to_base.tolist()}
        _json(metadata_path, metadata)

    with Fr3Client(cfg.robot_ip, enforce_realtime=False) as arm:
        before = arm.read_state()
        if float(np.max(np.abs(before.dq))) > 0.02:
            raise RuntimeError("FR3 is moving; stop it before sampling")
        capture = capture_realsense(cfg.external_serial)
        after = arm.read_state()
    if float(np.max(np.abs(after.dq))) > 0.02 or float(np.max(np.abs(after.q - before.q))) > 0.005:
        raise RuntimeError("FR3 moved during camera capture; sample was discarded")
    translation_shift = float(np.linalg.norm(after.ee[:3, 3] - before.ee[:3, 3]))
    rotation_shift = float((Rotation.from_matrix(before.ee[:3, :3]).inv()
                            * Rotation.from_matrix(after.ee[:3, :3])).magnitude())
    if translation_shift > 0.002 or rotation_shift > np.deg2rad(1.0):
        raise RuntimeError("FR3 pose changed during camera capture; sample was discarded")
    corners = _marker_corners(capture.rgb, marker_id)
    if "intrinsics" in metadata:
        if not np.allclose(capture.intrinsics, metadata["intrinsics"], atol=0.5) or not np.allclose(
                capture.distortion_coeffs, metadata["distortion_coeffs"], atol=1e-5):
            raise RuntimeError("camera intrinsics changed during the session")
    else:
        metadata["intrinsics"] = capture.intrinsics.tolist()
        metadata["distortion_coeffs"] = capture.distortion_coeffs.tolist()
        _json(metadata_path, metadata)

    folder = _new_numbered_dir(session / "samples")
    capture.save(folder / "capture")
    sample = {"schema": "aspire.fr3.camera_handeye_sample.v1",
              "captured_at": capture.captured_at,
              "ee_pose_base": after.ee.tolist(), "q": after.q.tolist(),
              "corners_uv": corners.tolist(),
              "capture": str(folder / "capture")}
    _json(folder / "sample.json", sample)
    overlay = cv2.cvtColor(capture.rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(overlay, [corners.astype(np.float32).reshape(1, 4, 2)],
                                   np.asarray([[marker_id]], dtype=np.int32))
    if not cv2.imwrite(str(folder / "marker_overlay.png"), overlay):
        raise RuntimeError("could not write marker overlay")
    return {"sample": str(folder), "sample_count": len(list((session / "samples").glob("*/sample.json"))),
            "marker_overlay": str(folder / "marker_overlay.png"),
            "next": "Move the stationary Hand to a different position and wrist tilt, then sample again."}


def solve(session: Path) -> dict:
    session = session.expanduser().resolve()
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "aspire.fr3.camera_handeye_session.v1":
        raise ValueError("invalid FR3 hand-eye session")
    paths = sorted((session / "samples").glob("*/sample.json"))
    if len(paths) < 12:
        raise ValueError(f"need at least 12 valid samples; found {len(paths)}")
    samples = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = solve_handeye_external(
        np.asarray(metadata["intrinsics"]), np.asarray(metadata["distortion_coeffs"]),
        float(metadata["marker_side_mm"]) / 1000.0, samples,
        np.asarray(metadata["initial_T_base_camera_optical"]))
    station_path = Path(metadata["station_path"])
    if not station_path.is_file():
        raise FileNotFoundError(station_path)
    station = yaml.safe_load(station_path.read_text(encoding="utf-8"))
    if station.get("schema") != "aspire.fr3_station.v1" or str(
            station["external_camera"]["serial"]) != metadata["camera_serial"]:
        raise ValueError("station model identity differs from collected samples")
    calibration_id = "aruco-handeye-" + time.strftime("%Y%m%dT%H%M%S")
    station["calibration_id"] = calibration_id
    station["external_camera"]["T_base_camera_optical"] = result["T_base_camera_optical"]
    folder = _new_numbered_dir(session / "solutions")
    candidate = folder / "fr3_station.calibrated.yaml"
    candidate.write_text(yaml.safe_dump(station, sort_keys=False), encoding="utf-8")
    result.update({"schema": "aspire.fr3.camera_handeye_result.v1",
                   "calibration_id": calibration_id,
                   "camera_serial": metadata["camera_serial"],
                   "marker_dictionary": metadata["marker_dictionary"],
                   "marker_id": metadata["marker_id"],
                   "marker_side_mm": metadata["marker_side_mm"],
                   "station_candidate": str(candidate), "station_path": str(station_path),
                   "station_sha256_at_collection": metadata["station_sha256"],
                   "station_applied": False,
                   "solution_dir": str(folder)})
    _json(folder / "result.json", result)
    return result


def apply_result(result_path: Path) -> dict:
    result_path = result_path.expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != "aspire.fr3.camera_handeye_result.v1":
        raise ValueError("invalid FR3 camera calibration result")
    if result.get("station_applied"):
        raise ValueError("this calibration result has already been applied")
    if result["held_out_rms_px"] > 3.0 or result["held_out_max_px"] > 6.0:
        raise ValueError("held-out reprojection check did not pass")
    station_path = Path(result["station_path"])
    candidate = Path(result["station_candidate"])
    if _digest(station_path) != result["station_sha256_at_collection"]:
        raise RuntimeError("station model changed since collection; refusing to overwrite it")
    proposed = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if proposed.get("schema") != "aspire.fr3_station.v1" or str(
            proposed["external_camera"]["serial"]) != result["camera_serial"]:
        raise ValueError("station candidate identity differs from the result")
    backup = result_path.parent / "fr3_station.before.yaml"
    if backup.exists():
        raise FileExistsError(f"backup exists: {backup}")
    backup.write_bytes(station_path.read_bytes())
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=station_path.parent,
                                     prefix=".fr3_station_calibrated_", delete=False) as stream:
        stream.write(candidate.read_text(encoding="utf-8"))
        temp_path = Path(stream.name)
    os.replace(temp_path, station_path)
    result["station_applied"] = True
    result["station_backup"] = str(backup)
    _json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    marker = commands.add_parser("marker", help="create a 1:1 printable ArUco SVG")
    marker.add_argument("--out", type=Path, required=True)
    marker.add_argument("--id", type=int, default=0)
    marker.add_argument("--side-mm", type=float, default=60.0,
                        help="black outer-square side length, not the white page")
    sample = commands.add_parser("sample", help="read FR3 state and capture one marker view; no motion")
    sample.add_argument("--config", type=Path, required=True)
    sample.add_argument("--session", type=Path, required=True)
    sample.add_argument("--id", type=int, default=0)
    sample.add_argument("--side-mm", type=float, default=60.0,
                        help="measured black outer-square side length")
    solve_cmd = commands.add_parser("solve", help="fit with held-out frames; write a station candidate")
    solve_cmd.add_argument("--session", type=Path, required=True)
    apply_cmd = commands.add_parser("apply", help="back up and install one reviewed calibration result")
    apply_cmd.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "marker":
            write_marker_svg(args.out, marker_id=args.id, side_mm=args.side_mm)
            output = {"marker_svg": str(args.out), "id": args.id,
                      "black_square_side_mm": args.side_mm,
                      "print": "Print at 100% scale and measure the black square with a ruler."}
        elif args.command == "sample":
            output = collect(args.config, args.session, marker_id=args.id, side_mm=args.side_mm)
        elif args.command == "solve":
            output = solve(args.session)
        else:
            output = apply_result(args.result)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as exc:
        print(f"FR3 camera calibration stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
