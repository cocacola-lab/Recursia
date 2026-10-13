"""Per-method stores and immutable content-addressed evaluation snapshots."""
import contextlib
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from cap.agent.model_defaults import DEFAULT_CODEX_MODEL, DEFAULT_REASONING_EFFORT

METHODS = ("recursia", "aspire", "sd-only")

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)

def read_json(path):
    return json.loads(Path(path).read_text())

def initial_knowledge(method):
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    return ("def run_task(task):\n    raise RuntimeError('Recursia stack has not learned a program yet')\n"
            if method == "recursia" else "")

def initial_state(method):
    return {"method": method, "frozen": False,
            "knowledge_sha256": digest(initial_knowledge(method)), "source_events": [], "regression_cases": [],
            "source_model_calls": 0, "source_vision_model_calls": 0, "source_rollouts": 0}

class MethodStore:
    def __init__(self, campaign, method):
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        self.campaign = Path(campaign).resolve()
        self.root = self.campaign / method
        self.method = method
        self.meta = read_json(self.campaign / "campaign.json")
        self.state_path = self.root / "state.json"
        self.knowledge_path = self.root / ("stack.py" if method == "recursia" else "skills.md")
        if not self.state_path.exists():
            raise ValueError("campaign method is not initialized; create a fresh campaign with the current method names")

    @property
    def state(self):
        return read_json(self.state_path)

    @property
    def knowledge(self):
        return self.knowledge_path.read_text()

    def skill_documents(self):
        from .promotion import documents
        return documents(self.root)

    def context_knowledge(self, content=None):
        """Legacy/function knowledge plus this method's source-learned topics."""
        parts = [self.knowledge if content is None else content]
        parts += [f"# Source skill file: skills/{name}\n{content}"
                  for name, content in self.skill_documents().items()]
        return "\n\n".join(part for part in parts if part)

    @contextlib.contextmanager
    def locked(self):
        with (self.root / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("this method already has an active source/eval operation") from exc
            yield

    def save_learning(self, content, event):
        state = self.state
        if state["frozen"]:
            raise ValueError("method is frozen; use a fresh campaign for further source learning")
        self.knowledge_path.write_text(content)
        state["source_events"].append(event)
        state["knowledge_sha256"] = digest(self.context_knowledge())
        write_json(self.state_path, state)

    def skill_code(self):
        library = self.root / "skill_library.json"
        if not library.is_file():
            return ""
        return "\n\n".join(skill["code"] for skill in read_json(library)["skills"].values() if skill["promoted"])

    def freeze(self):
        state = self.state
        if state["frozen"]:
            return self.snapshot()
        if self.method == "recursia" and self.meta.get("recursia_protocol"):
            from .recursia import recursia_protocol
            recursia_protocol(self).finish()
            state = self.state
        if not state["source_events"]:
            raise ValueError("no source learning recorded; do not label an empty campaign as trained")
        from .versions import EvolutionGit
        version = EvolutionGit(self).checkpoint('Freeze source learning')
        content = self.context_knowledge()
        packet = {"schema": "aspire.rsi.snapshot.v1", "method": self.method, "content": content,
                  "agent_git_commit": version['head'],
                  "sha256": digest(content), "source_events_sha256": digest(json.dumps(state["source_events"], sort_keys=True)),
                  "campaign_sha256": digest(json.dumps(read_json(self.campaign / "campaign.json"), sort_keys=True)),
                  "campaign_id": self.meta["campaign_id"], "provider": self.meta["provider"],
                  "model": self.meta["model"], "tools": self.meta["tools"],
                  "reasoning_effort": self.meta.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
                  "source_task_ids": sorted({t for e in state["source_events"]
                                             for t in e.get("task_ids", [e.get("task_id")]) if t}),
                  "contains_preview_learning": any(e.get("dry_run") for e in state["source_events"])}
        library = self.root / "skill_library.json"
        if library.exists():
            packet["skill_library_sha256"] = digest(library.read_text())
        packet["skill_document_sha256"] = {name: digest(text) for name, text in self.skill_documents().items()}
        chunk = self.root / "recursia_chunk.json"
        if self.method == "recursia" and chunk.is_file():
            packet["recursia_chunk_sha256"] = digest(chunk.read_text())
            packet["chunk_state_sha256"] = {str(p.relative_to(self.root)): digest(p.read_text())
                                             for p in sorted((self.root / "source").glob("chunk_*/chunk.json"))}
        write_json(self.root / "frozen.json", packet)
        state["frozen"] = True
        state["knowledge_sha256"] = packet["sha256"]
        write_json(self.state_path, state)
        return packet

    def snapshot(self):
        packet = read_json(self.root / "frozen.json")
        if packet["method"] != self.method or packet["campaign_id"] != self.meta["campaign_id"]:
            raise ValueError("snapshot belongs to a different method/campaign")
        if digest(packet["content"]) != packet["sha256"] or digest(self.context_knowledge()) != packet["sha256"]:
            raise ValueError("frozen knowledge hash changed")
        if packet.get("skill_document_sha256", {}) != {name: digest(text) for name, text in self.skill_documents().items()}:
            raise ValueError("frozen skill document hash changed")
        library = self.root / "skill_library.json"
        if packet.get("skill_library_sha256") != (digest(library.read_text()) if library.exists() else None):
            raise ValueError("frozen skill library hash changed")
        state = self.state
        if not state["frozen"] or state["knowledge_sha256"] != packet["sha256"]:
            raise ValueError("frozen snapshot and learning state disagree")
        if digest(json.dumps(state["source_events"], sort_keys=True)) != packet["source_events_sha256"]:
            raise ValueError("frozen source history hash changed")
        if digest(json.dumps(read_json(self.campaign / "campaign.json"), sort_keys=True)) != packet["campaign_sha256"]:
            raise ValueError("frozen campaign configuration hash changed")
        if "recursia_chunk_sha256" in packet:
            if digest((self.root / "recursia_chunk.json").read_text()) != packet["recursia_chunk_sha256"]:
                raise ValueError("frozen Recursia protocol evidence changed")
        if "chunk_state_sha256" in packet:
            actual = {str(p.relative_to(self.root)): digest(p.read_text())
                      for p in sorted((self.root / "source").glob("chunk_*/chunk.json"))}
            if actual != packet["chunk_state_sha256"]:
                raise ValueError("frozen chunk evidence changed")
        return packet

    @contextlib.contextmanager
    def evaluation_freeze(self):
        """Pin source knowledge for one target run, then unlock source learning."""
        self.freeze()
        try:
            yield self.snapshot()
        finally:
            frozen = self.root / "frozen.json"
            archive = self.root / "frozen_history" / f"{time.time_ns()}.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            os.replace(frozen, archive)
            state = self.state
            state["frozen"] = False
            write_json(self.state_path, state)

def initialize(root, tasks, *, provider="codex", model=None,
               model_call_cap=None, rollout_cap=None, multi_turn_limit=10, chunk_size=None):
    from .prompts import TOOLS
    if provider == "codex" and model is None:
        model = DEFAULT_CODEX_MODEL
    root = Path(root).expanduser().resolve()
    source_ids = [task["id"] for task in tasks if task["split"] == "source"]
    protocol = {"schema": "physical_recursia_chunk.v1", "max_candidates": 2,
                "target_layouts": ["eval01", "eval02", "eval03"], "online_action_correction": True}
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("chunk size must be positive")
        protocol.update(schema="physical_recursia_chunks.v2",
                        chunks=[source_ids[i:i + chunk_size] for i in range(0, len(source_ids), chunk_size)])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "campaign.json", {"schema": "aspire.rsi.campaign.v1", "campaign_id": root.name,
               "tasks": tasks, "provider": provider, "model": model,
               "reasoning_effort": DEFAULT_REASONING_EFFORT if provider == "codex" else None,
               "workflow": "simulation_multi_turn.v1", "multi_turn_limit": multi_turn_limit,
               "execution_schedule": "single_episode.v1",
               "recursia_protocol": protocol,
               "model_call_cap": model_call_cap, "rollout_cap": rollout_cap,
               "tools": list(TOOLS), "scope": "LIBERO-inspired real-robot transfer, not official simulator scores"})
    for method in METHODS:
        folder = root / method
        folder.mkdir(mode=0o700)
        content = initial_knowledge(method)
        filename = "stack.py" if method == "recursia" else "skills.md"
        (folder / filename).write_text(content)
        write_json(folder / "state.json", initial_state(method))
    return root
