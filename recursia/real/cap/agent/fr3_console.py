"""Human-readable progress for the FR3 runner; no robot-control dependencies."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO


class AgentConsole:
    def __init__(self, *, stream: TextIO | None = None, quiet: bool = False,
                 verbose: bool = False):
        self.stream = stream if stream is not None else sys.stdout
        self.quiet = quiet
        self.verbose = verbose
        self.started = time.monotonic()
        self.move_label = None
        self.move_count = 0

    def line(self, message: str, *, final: bool = False) -> None:
        if not self.quiet or final:
            elapsed = time.monotonic() - self.started
            print(f"[run_agent +{elapsed:6.1f}s] {message}", file=self.stream, flush=True)

    def progress(self, event: dict[str, Any]) -> None:
        tool = event.get("tool", "")
        status = event.get("status")
        detail = event.get("detail") or {}
        if status == "error":
            self.line(f"ERROR {tool}: {self._text(event.get('error'))}")
            return
        if tool == "print":
            self.line(f"Agent: {self._text(detail.get('text'))}")
            return
        if tool == "goto_pose":
            label = detail.get("label", "agent_move")
            if status == "started":
                changed = label != self.move_label
                self.move_label = label
                if changed or self.verbose:
                    self.line(f"→ {label}  target={detail.get('position')}")
            elif status == "done":
                self.move_count += 1
                if self.verbose or label != "wipe" or self.move_count % 10 == 0:
                    self.line(f"✓ {label}  {event.get('elapsed_s', 0):.2f}s  "
                              f"actual={detail.get('after_xyz_m')}")
            return
        if tool in {"get_robot_state", "whiteboard_audit", "whiteboard_status", "get_saved_plan"} and not self.verbose:
            return
        if status == "started":
            extra = detail.get("prompt") or detail.get("stage") or ""
            self.line(f"→ {tool} {extra}".rstrip())
        elif status == "done":
            extra = ""
            if tool == "segment_sam3_text_prompt":
                extra = f"masks={detail.get('count', 0)} pixels={detail.get('pixels', [])}"
            elif tool == "plan_whiteboard":
                extra = f"ink={detail.get('ink_pixels')} wipe_waypoints={detail.get('wipe_waypoints')}"
            elif tool == "close_gripper":
                width = detail.get("hand_width_m")
                width_text = f"{width * 1000:.1f} mm" if isinstance(width, (float, int)) else "unknown"
                extra = f"success={detail.get('success')} held={detail.get('hand_is_grasped')} width={width_text}"
            elif tool == "evaluate_wipe":
                ratio = detail.get("ink_pixel_reduction", 0)
                extra = f"success={detail.get('success')} ink_reduction={ratio:.1%}"
            marker = "!" if detail.get("success") is False else "✓"
            self.line(f"{marker} {tool}  {event.get('elapsed_s', 0):.2f}s  {extra}".rstrip())

    @staticmethod
    def _text(value: Any) -> str:
        return " ".join(str(value or "").split())[:600]

    def finish(self, result: dict[str, Any]) -> None:
        latest = (result.get("history") or [{}])[-1]
        explanation = (latest.get("display_error") or latest.get("error") or
                       (latest.get("declared_result") or {}).get("summary") or result.get("stop_reason"))
        outcome = 'SUCCESS' if result['success'] else 'FAILED'
        if result.get("dry_run"):
            outcome = "DRY RUN COMPLETE" if result.get("stop_reason") == "dry_run_complete" else "DRY RUN FAILED"
        self.line(f"{outcome}  goal={result['goal']}  "
                  f"attempts={result.get('attempts_this_invocation', 0)}  "
                  f"physical_motion_commanded={result.get('physical_motion_commanded', False)}", final=True)
        self.line(self._text(explanation), final=True)
        self.line(f"Stop: {result.get('stop_reason')}", final=True)
        root = Path(result["run_dir"])
        self.line(f"Output: {root}", final=True)
        self.line(f"Result: {root / 'agent_result.json'}", final=True)


def emit_progress(path: Path | None, **event: Any) -> None:
    if path is not None:
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
