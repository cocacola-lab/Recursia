# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
from .fr3_console import AgentConsole

def _run_worker_live(command: list[str], folder: Path, timeout_s: int,
                     console: AgentConsole) -> subprocess.CompletedProcess:
    """Tail compact tool events; keep SDK stdout/stderr in artifact files."""
    progress = folder / "progress.jsonl"
    progress.touch()
    command = command + ["--progress-log", str(progress)]
    stdout = folder / "worker.stdout.log"
    stderr = folder / "worker.stderr.log"
    started = time.monotonic()
    timed_out = False
    pending = ""
    with stdout.open("w") as out, stderr.open("w") as err, progress.open() as events:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                   stdout=out, stderr=err, env=os.environ.copy())
        try:
            while True:
                pending += events.read()
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    if line.strip():
                        console.progress(json.loads(line))
                if process.poll() is not None:
                    # Drain once after exit; all writer file descriptors are closed.
                    for line in (pending + events.read()).splitlines():
                        if line.strip():
                            console.progress(json.loads(line))
                    break
                if time.monotonic() - started > timeout_s:
                    timed_out = True
                    process.kill()
                    process.wait()
                    break
                time.sleep(0.1)
        except BaseException:
            process.kill()
            process.wait()
            raise
    captured_out = stdout.read_text(errors="replace")
    captured_err = stderr.read_text(errors="replace")
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout_s, output=captured_out,
                                        stderr=captured_err)
    return subprocess.CompletedProcess(command, process.returncode, captured_out, captured_err)
