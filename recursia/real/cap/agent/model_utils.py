# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json, os, shutil
from pathlib import Path
from typing import Any

def _decode_claude_output(stdout: str) -> dict[str, Any]:
    """Accept Claude Code's structured-output variants without guessing code."""
    outer = json.loads(stdout)
    value: Any = outer
    if isinstance(outer, dict):
        if isinstance(outer.get("structured_output"), dict):
            value = outer["structured_output"]
        elif "result" in outer:
            value = outer["result"]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("```json") and stripped.endswith("```"):
            stripped = stripped[7:-3].strip()
        value = json.loads(stripped)
    if not isinstance(value, dict) or not isinstance(value.get("code"), str):
        raise ValueError("Claude response has no structured code field")
    return {"analysis": str(value.get("analysis", "")), "code": value["code"]}

def _resolve_codex_executable(executable: str = "codex") -> str:
    """Find Codex in both ordinary shells and the desktop app installation."""
    requested = os.environ.get("ASPIRE_CODEX_EXECUTABLE", executable)
    resolved = shutil.which(requested)
    if resolved:
        return resolved
    candidates = [
        Path(requested).expanduser(),
        Path("/usr/lib/chatgpt/resources/codex"),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    raise FileNotFoundError(
        "Codex CLI was not found. Put 'codex' on PATH or set "
        "ASPIRE_CODEX_EXECUTABLE to its executable path."
    )
