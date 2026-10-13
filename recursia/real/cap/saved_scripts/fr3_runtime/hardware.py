"""Read-only checks shared by the FR3 doctor and direct FCI executor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def active_fci_clients() -> list[dict[str, Any]]:
    """Find FCI bridge processes that may own the single FCI arm session."""
    clients: list[dict[str, Any]] = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(cmdline.parent.name)
            if pid == os.getpid():
                continue
            command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if ("franka_bridge_node" in command
                    or ("ros2 launch fi_bringup" in command and "collect.launch.py" in command)):
                clients.append({"pid": pid, "command": command[:160]})
        except (OSError, ValueError):
            continue
    return clients
