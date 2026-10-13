#!/usr/bin/env python3
"""Bounded FR3/Hand/RGB-D checks; never move, home, recover or unlock."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def service_health(url, name):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as response:
            health = json.load(response)
        ready = health.get("model_loaded") is True
        if name in {"anygrasp", "contact_graspnet"}:
            ready = ready and health.get("mock") is not True and health.get("safe_for_robot_motion") is not False
        if name == "contact_graspnet":
            ready = ready and health.get("model") == "contact_graspnet_pytorch"
        return {"ok": ready, "url": url, "health": health,
                "error": None if ready else "service has not confirmed a loaded, real model"}
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def robot_probe(ip, *, franky_module=None):
    import numpy as np
    from cap.saved_scripts.fr3_runtime.hardware import active_fci_clients
    from robot.franka.fr3 import FR3_Q_MIN, FR3_Q_MAX
    clients = active_fci_clients()
    if clients:
        return {"ok": False, "error": "another FCI bridge owns the connection", "fci_clients": clients}
    if franky_module is None:
        network = {}
        for name, port in (("desk", 443), ("fci", 1337), ("hand", 1338)):
            try:
                with socket.create_connection((ip, port), timeout=1.5):
                    network[name] = {"ok": True, "port": port}
            except OSError as exc:
                network[name] = {"ok": False, "port": port, "error": str(exc)}
        if not network["fci"]["ok"]:
            route = subprocess.run(["ip", "route", "get", ip], capture_output=True, text=True, timeout=2)
            return {"ok": False, "robot_ip": ip, "network": network, "route": route.stdout.strip(),
                    "error": "FCI TCP port is unreachable; check robot power, cable, local route and FCI activation"}
        import franky as franky_module
    robot = franky_module.Robot(ip, realtime_config=franky_module.RealtimeConfig.Ignore)
    before = robot.state
    time.sleep(0.15)
    state = robot.state
    q, dq = np.asarray(state.q), np.asarray(state.dq)
    pose = np.asarray(state.O_T_EE.matrix)
    finite = (q.shape == (7,) and dq.shape == (7,) and pose.shape == (4, 4)
              and np.isfinite(q).all() and np.isfinite(dq).all() and np.isfinite(pose).all())
    errors = bool(robot.has_errors)
    mode = str(state.robot_mode)
    idle = mode.lower().split(".")[-1] == "idle"
    in_limits = bool(finite and np.all(q >= FR3_Q_MIN) and np.all(q <= FR3_Q_MAX))
    stationary = bool(finite and np.max(np.abs(dq)) < 0.02)
    result = {"ok": bool(finite and in_limits and stationary and idle and not errors), "robot_ip": ip,
              "robot_mode": mode, "has_errors": errors, "stationary": stationary,
              "state_finite": bool(finite), "within_joint_limits": in_limits,
              "q_rad": q.tolist(), "dq_rad_s": dq.tolist(), "T_base_ee": pose.tolist(),
              "current_errors": str(state.current_errors), "last_motion_errors": str(state.last_motion_errors),
              "robot_time_s": float(state.time.to_sec()),
              "state_time_advanced": bool(state.time.to_sec() > before.time.to_sec())}
    # Release this arm connection before probing the hand; no stop command.
    del robot
    try:
        hand = franky_module.Gripper(ip).state
        width, maximum = float(hand.width), float(hand.max_width)
        result["hand"] = {"ok": bool(np.isfinite(width) and 0 <= width <= maximum + 0.001 and maximum > 0),
                          "width_m": width, "max_width_m": maximum, "is_grasped": bool(hand.is_grasped),
                          "temperature_c": int(hand.temperature)}
        result["ok"] = result["ok"] and result["hand"]["ok"]
    except Exception as exc:
        result["hand"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["ok"] = False
    if not result["ok"]:
        result["error"] = "robot/hand state is not ready; see mode, errors, joints and hand fields"
    return result


def camera_probe(serial):
    import numpy as np
    from cap.saved_scripts.fr3_runtime.camera import capture_realsense
    image = capture_realsense(serial, warmup_frames=20)
    return {"ok": True, "serial": serial, "usb_type": image.usb_type,
            "shape": list(image.depth_m.shape), "valid_depth_fraction": float(np.mean(image.depth_m > 0)),
            "color_frame_number": image.color_frame_number, "depth_frame_number": image.depth_frame_number}


def bounded_probe(kind, value, timeout):
    with tempfile.TemporaryDirectory(prefix="aspire-fr3-check-") as temporary:
        output = Path(temporary) / "probe.json"
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--probe", kind,
                                      "--value", value, "--output", str(output)],
                                     capture_output=True, text=True, timeout=timeout, cwd=ROOT)
            if output.is_file():
                return json.loads(output.read_text())
            return {"ok": False, "error": f"probe exited {process.returncode}", "stderr": process.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"{kind} probe timed out after {timeout}s; no motion commanded"}


def check(hardware_path, *, sam3_url, anygrasp_url=None, grasp_url=None,
          grasp_backend="contact_graspnet", cameras=True):
    import yaml
    hardware = yaml.safe_load(hardware_path.read_text())["hardware"]
    if anygrasp_url is not None:
        grasp_backend, grasp_url = "anygrasp", anygrasp_url
    grasp_url = grasp_url or "http://127.0.0.1:8115"
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    result = {"schema": "aspire.fr3.self_check.v1", "checked_at": now.isoformat(),
              "physical_motion_commanded": False,
              "services": {name: service_health(url, name) for name, url in
                           (("sam3", sam3_url), (grasp_backend, grasp_url))},
              "robot": bounded_probe("robot", hardware["franka"]["ip"], 15), "cameras": {},
              "realtime_priority_hard_limit": resource.getrlimit(resource.RLIMIT_RTPRIO)[1]}
    for key, command in (("network_route", ["ip", "route", "get", hardware["franka"]["ip"]]),
                         ("network_links", ["ip", "-brief", "address"])):
        try:
            result[key] = subprocess.run(command, capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = str(exc)
    if cameras:
        for name in ("ext", "wrist"):
            serial = hardware["cameras"].get(name, {}).get("serial")
            result["cameras"][name] = bounded_probe("camera", str(serial), 35) if serial else {
                "ok": False, "error": "camera serial not configured"}
    result["hardware_ok"] = result["robot"]["ok"] and all(c["ok"] for c in result["cameras"].values())
    result["ok"] = (result["hardware_ok"] and all(s["ok"] for s in result["services"].values())
                    and result["realtime_priority_hard_limit"] >= 80)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "config/hardware.local.yaml")
    parser.add_argument("--sam3-url", default="http://127.0.0.1:6767")
    parser.add_argument("--contact-graspnet-url", default="http://127.0.0.1:8115")
    parser.add_argument("--anygrasp-url", help="check legacy AnyGrasp instead of Contact-GraspNet")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", choices=["robot", "camera"], help=argparse.SUPPRESS)
    parser.add_argument("--value", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        result = ((robot_probe(args.value) if args.probe == "robot" else camera_probe(args.value))
                  if args.probe else check(args.hardware_config, sam3_url=args.sam3_url,
                                          anygrasp_url=args.anygrasp_url,
                                          grasp_url=args.contact_graspnet_url, cameras=not args.no_cameras))
    except Exception as exc:
        result = {"ok": False, "physical_motion_commanded": False, "error": f"{type(exc).__name__}: {exc}"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not args.probe:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
