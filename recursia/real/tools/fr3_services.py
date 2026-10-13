#!/usr/bin/env python3
"""Start SAM3 and Contact-GraspNet, then inspect FR3 without motion."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from fr3_self_check import check, service_health

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "logs/fr3_services"


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f")


def start_ticks(pid):
    try:
        # /proc comm can contain spaces or parentheses; starttime is field 22.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def owned(record):
    return bool(record.get("start_ticks") and start_ticks(record["pid"]) == record["start_ticks"])


def anygrasp_command():
    if os.environ.get("ANYGRASP_SERVER_COMMAND"):
        return ["bash", "-lc", os.environ["ANYGRASP_SERVER_COMMAND"]], []
    prefix = Path(os.environ.get("ASPIRE_ANYGRASP_DIR", str(Path.home() / ".local/share/aspire/anygrasp")))
    python = prefix / "venv/bin/python"
    server = prefix / "serve_anygrasp.py"
    checkpoint = Path(os.environ.get("ANYGRASP_CHECKPOINT", str(prefix / "checkpoint_detection.tar"))).expanduser()
    license_dir = Path(os.environ.get("ANYGRASP_LICENSE_DIR", str(prefix / "sdk/grasp_detection/license"))).expanduser()
    missing = [str(path) for path in (python, server, checkpoint, license_dir / "licenseCfg.json") if not path.is_file()]
    if not any(license_dir.glob("*.lic")):
        missing.append(f"{license_dir}/*.lic (per-machine AnyGrasp license)")
    return [str(python), str(server), "--checkpoint", str(checkpoint), "--license-dir", str(license_dir),
            "--host", "127.0.0.1", "--port", "8122"], missing


def contact_graspnet_command():
    prefix = Path(os.environ.get("ASPIRE_CONTACT_GRASPNET_DIR",
                               str(Path.home() / ".local/share/aspire/contact-graspnet")))
    python, source = prefix / "venv/bin/python", prefix / "source"
    required = (python, source / "checkpoints/contact_graspnet/checkpoints/model.pt",
                source / "checkpoints/contact_graspnet/config.yaml")
    return [str(python), str(ROOT / "tools/vision/serve_contact_graspnet.py"),
            "--source", str(source), "--host", "127.0.0.1", "--port", "8115"], [
                str(p) for p in required if not p.is_file()]


def launch(name, url, command, timeout):
    health = service_health(url, name)
    if health["ok"]:
        print(f"[{name}] ready; reusing {url}", flush=True)
        return health
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1):
            return {"ok": False, "error": f"port {parsed.port} is occupied by a service that is not ready; see /health"}
    except OSError:
        pass
    log = STATE / f"{name}_{stamp()}.log"
    print(f"[{name}] starting; log={log}", flush=True)
    with log.open("ab") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "start_ticks": start_ticks(process.pid), "command": command, "log": str(log)}
    (STATE / f"{name}.json").write_text(json.dumps(record, indent=2))
    end, update = time.monotonic() + timeout, time.monotonic() + 25
    while time.monotonic() < end:
        health = service_health(url, name)
        if health["ok"]:
            print(f"[{name}] model ready at {url}", flush=True)
            return {**health, "log": str(log)}
        if process.poll() is not None:
            return {"ok": False, "error": f"service exited {process.returncode}", "log": str(log)}
        if time.monotonic() >= update:
            print(f"[{name}] loading; see {log}", flush=True)
            update = time.monotonic() + 25
        time.sleep(1)
    return {"ok": False, "error": f"readiness timeout after {timeout}s; process left running", "log": str(log)}


def stop(name):
    path = STATE / f"{name}.json"
    if not path.exists():
        print(f"[{name}] no owned process; existing external service left running")
        return
    record = json.loads(path.read_text())
    if owned(record):
        os.killpg(record["pid"], signal.SIGTERM)
        print(f"[{name}] terminated owned PID {record['pid']}")
    else:
        print(f"[{name}] recorded PID has exited or changed; no process signalled")
    path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "check", "stop"], nargs="?", default="start")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--grasp-backend", choices=["contact_graspnet", "anygrasp"], default="contact_graspnet")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "config/hardware.local.yaml")
    args = parser.parse_args()
    if args.timeout_s < 1:
        parser.error("timeout must be positive")
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another FR3 service launcher is running", file=sys.stderr)
            return 2
        if args.action == "stop":
            for name in ("contact_graspnet", "anygrasp", "sam3"):
                stop(name)
            return 0
        backend = args.grasp_backend
        urls = {"sam3": "http://127.0.0.1:6767", backend: (
            "http://127.0.0.1:8115" if backend == "contact_graspnet" else "http://127.0.0.1:8122")}
        if args.action == "status":
            result = {name: service_health(url, name) for name, url in urls.items()}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if all(s["ok"] for s in result.values()) else 2
        startup = {}
        if args.action == "start":
            startup["sam3"] = launch("sam3", urls["sam3"], ["bash", "tools/fr3_sam3.sh", "start"], args.timeout_s)
            existing = service_health(urls[backend], backend)
            command, missing = contact_graspnet_command() if backend == "contact_graspnet" else anygrasp_command()
            startup[backend] = existing if existing["ok"] else (
                {"ok": False, "error": f"{backend} installation/assets are incomplete", "missing": missing}
                if missing else launch(backend, urls[backend], command, args.timeout_s))
            for name, value in startup.items():
                if not value["ok"]:
                    print(f"[{name}] not ready: {json.dumps(value, ensure_ascii=False)}", flush=True)
        print("[FR3] checking robot, hand and cameras; no motion commands", flush=True)
        result = check(args.hardware_config, sam3_url=urls["sam3"],
                       grasp_url=urls[backend], grasp_backend=backend, cameras=not args.no_cameras)
        result["startup"] = startup
        output = STATE / f"self_check_{stamp()}.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"[FR3] report={output}")
        return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
