"""Ephemeral, tool-less model calls: no shared conversations or repository context."""
import json
import base64
from pathlib import Path
import shutil
import subprocess
import tempfile

from cap.agent.model_utils import _decode_claude_output, _resolve_codex_executable
from cap.agent.model_defaults import DEFAULT_CODEX_MODEL, DEFAULT_REASONING_EFFORT

DISABLED_FEATURES = ("shell_tool", "unified_exec", "apps", "plugins", "remote_plugin", "multi_agent",
                     "multi_agent_v2", "browser_use", "browser_use_external", "computer_use", "code_mode",
                     "view_image", "image_generation", "sleep_tool", "goals", "worktrees", "memories",
                     "skill_search", "workspace_dependencies", "hooks", "code_mode_host", "in_app_browser",
                     "in_app_chat", "in_app_local_automation")

def configured_model():
    """Use the project default, independent of the desktop chat's selection."""
    return DEFAULT_CODEX_MODEL

def codex_command(executable, root, model=None, reasoning_effort=DEFAULT_REASONING_EFFORT, images=()):
    cmd = [executable, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--json",
           "--sandbox", "read-only", "--skip-git-repo-check", "--cd", str(root),
           "--output-schema", str(root / "schema.json"), "--output-last-message", str(root / "response.json"),
           "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0",
           "-c", 'developer_instructions=""', "--enable", "skip_host_skill_discovery",
           "--enable", "fast_mode", "-c", 'service_tier="fast"',
           "-c", f'model_reasoning_effort="{reasoning_effort}"']
    for name in DISABLED_FEATURES:
        cmd += ["--disable", name]
    cmd += ["--model", model or DEFAULT_CODEX_MODEL]
    for path in images:
        cmd += ["--image", str(path)]
    return cmd + ["-"]

def claude_command(root, schema, model=None):
    cmd = ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
           "--tools", "", "--permission-mode", "dontAsk", "--no-session-persistence",
           "--setting-sources", "", "--strict-mcp-config", "--mcp-config", "{}",
           "--disable-slash-commands", "--system-prompt", "Return the requested structured answer using only supplied context."]
    if model:
        cmd += ["--model", model]
    return cmd


def decode_response(text, schema):
    if "failure_class" in schema["properties"]:
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("structured_output", value.get("result", value))
        if isinstance(value, str):
            value = json.loads(value)
        if (not isinstance(value, dict) or set(value) != set(schema["required"])
                or not all(isinstance(field, str) for field in value.values())):
            raise ValueError("invalid Recursia candidate response")
        return value
    if "code" in schema["properties"]:
        return _decode_claude_output(text)
    value = json.loads(text)
    if isinstance(value, dict):
        value = value.get("structured_output", value.get("result", value))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        raise ValueError("model returned fields inconsistent with the requested response schema")
    if "response" in schema["properties"]:
        if not isinstance(value["analysis"], str) or not isinstance(value["response"], str):
            raise ValueError("model returned invalid multi-turn feedback fields")
        return value
    if "findings" in schema["properties"]:
        if not isinstance(value["analysis"], str) or not isinstance(value["findings"], str):
            raise ValueError("model returned invalid source findings fields")
        return value
    if "updates" in schema["properties"]:
        if (not isinstance(value["analysis"], str) or not isinstance(value["reason"], str)
                or not isinstance(value["updates"], list)):
            raise ValueError("model returned invalid skill update fields")
        for update in value["updates"]:
            if (not isinstance(update, dict) or set(update) != {"file", "content"}
                    or not all(isinstance(update[k], str) for k in ("file", "content"))):
                raise ValueError("model returned an invalid skill document replacement")
        return value
    if "goals" in schema["properties"]:
        from .evaluation import validate_placement_response
        return validate_placement_response(value)
    raise ValueError("unsupported model response schema")

class IsolatedGenerator:
    def __init__(self, provider, model=None, reasoning_effort=DEFAULT_REASONING_EFFORT):
        if provider not in {"codex", "claude"}:
            raise ValueError("RSI provider must be codex or claude")
        self.provider, self.model = provider, model
        self.reasoning_effort = reasoning_effort

    def generate(self, prompt, *, timeout_s, images=(), schema=None):
        with tempfile.TemporaryDirectory(prefix="aspire-rsi-model-") as temporary:
            root = Path(temporary)
            schema = schema or {"type": "object", "properties": {"analysis": {"type": "string"}, "code": {"type": "string"}},
                      "required": ["analysis", "code"], "additionalProperties": False}
            attached = []
            for i, path in enumerate(images):
                copy = root / f"view_{i+1}.png"
                shutil.copyfile(path, copy)
                attached.append(copy)
            if self.provider == "claude":
                command = claude_command(root, schema, self.model)
                payload = prompt
                if attached:
                    command += ["--input-format", "stream-json"]
                    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                               "data": base64.b64encode(p.read_bytes()).decode()}} for p in attached]
                    content.append({"type": "text", "text": prompt})
                    payload = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
                proc = subprocess.run(command, input=payload, cwd=root,
                                      capture_output=True, text=True, timeout=timeout_s)
                if proc.returncode:
                    raise RuntimeError(f"Claude generator exited {proc.returncode}: {(proc.stderr or proc.stdout)[-1500:]}")
                return decode_response(proc.stdout, schema)
            (root / "schema.json").write_text(json.dumps(schema))
            proc = subprocess.run(codex_command(_resolve_codex_executable(), root, self.model, self.reasoning_effort,
                                               attached), input=prompt,
                                  cwd=root, capture_output=True, text=True, timeout=timeout_s)
            if proc.returncode:
                raise RuntimeError(f"Codex generator exited {proc.returncode}: {(proc.stderr or proc.stdout)[-1500:]}")
            for line in proc.stdout.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("item", {}).get("type")
                if kind and kind not in {"reasoning", "agent_message", "error"}:
                    raise RuntimeError(f"tool-less model unexpectedly emitted {kind}; response rejected")
            return decode_response((root / "response.json").read_text(), schema)

    def evaluate_placement(self, observation, task, *, timeout_s):
        """Score current pixels in a separate call without policy or skill history."""
        from .evaluation import PLACEMENT_SCHEMA, placement_prompt, validate_placement_response
        from .observation import image_paths
        result = self.generate(placement_prompt(task), timeout_s=timeout_s,
                               images=image_paths(observation), schema=PLACEMENT_SCHEMA)
        return validate_placement_response(result, task)
