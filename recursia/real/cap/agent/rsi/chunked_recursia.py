"""Run one paired source chunk through baseline and all candidate validations."""
from pathlib import Path
import shutil
import time

from .campaign import RolloutAbort
from .recursia import RecursiaProtocol
from .store import digest, read_json, write_json
from .versions import EvolutionGit


class ChunkedRecursiaProtocol(RecursiaProtocol):
    @staticmethod
    def _faulted(record):
        return (record.get("hardware_fault") or record.get("unknown_worker_failure")
                or record.get("scored_by") == "interrupted"
                or "terminated episode: camera feedback unavailable" in (record.get("error") or ""))

    def __init__(self, store, experiment=None):
        super().__init__(store, experiment)
        self.index_path = self.path
        self.chunks = self.config["chunks"]
        flat = [task for chunk in self.chunks for task in chunk]
        if flat != self.source_ids or any(not chunk for chunk in self.chunks):
            raise ValueError("chunk manifest must contain every source task once, in order")
        index = read_json(self.index_path) if self.index_path.exists() else {}
        self._select(index.get("active_chunk", 1))

    def _select(self, number):
        if not isinstance(number, int) or not 1 <= number <= len(self.chunks):
            raise ValueError(f"chunk must be between 1 and {len(self.chunks)}")
        self.number = number
        self.source_ids = self.chunks[number - 1]
        self.folder = self.store.root / "source" / f"chunk_{number:03d}"
        self.path = self.folder / "chunk.json"

    def _index(self):
        return read_json(self.index_path) if self.index_path.exists() else {
            "schema": "physical_recursia_chunks.v2", "chunks": self.chunks, "active_chunk": 1,
            "official_sha256": digest(self.store.knowledge), "records": {}}

    def _sync(self):
        if not self.path.exists():
            return
        state, index = read_json(self.path), self._index()
        index["records"][str(self.number)] = {"phase": state["phase"],
            "path": str(self.path.relative_to(self.store.root)), "sha256": digest(self.path.read_text())}
        index.update(active_chunk=self.number, official_sha256=digest(self.store.knowledge))
        write_json(self.index_path, index)

    def load(self):
        index = self._index()
        if index["chunks"] != self.chunks or index["official_sha256"] != digest(self.store.knowledge):
            raise ValueError("chunk history and official module disagree; use a fresh campaign after rollback/manual edits")
        state = super().load()
        recorded = index["records"].get(str(self.number))
        if recorded and recorded["sha256"] != digest(self.path.read_text()):
            raise ValueError("chunk evidence changed outside the scheduler")
        return state

    def status(self):
        index = self._index()
        completed = [int(key) for key, record in index["records"].items() if record["phase"] == "complete"]
        state = read_json(self.path) if self.path.exists() else None
        round_ = state["rounds"].get(state["active_round"]) if state else None
        pending = [task for task in self.source_ids if not round_ or task not in round_["records"]]
        return {"protocol": "physical_recursia_chunks.v2", "chunk": self.number,
            "tasks": self.source_ids, "phase": state["phase"] if state else "not_started",
            "active_round": state["active_round"] if state else None,
            "pending_tasks": pending, "completed_chunks": sorted(completed),
            "candidates_proposed": state["candidates_proposed"] if state else 0,
            "accepted_candidates": [name for name, item in state["rounds"].items() if item.get("accepted")] if state else [],
            "next": f"source --chunk {self.number + 1}" if state and state["phase"] == "complete"
                    and self.number < len(self.chunks) else "freeze / eval" if len(completed) == len(self.chunks)
                    else f"source --chunk {self.number}"}

    def _bootstrap(self, task):
        if self.number == 1:
            state = super()._bootstrap(task)
            state["chunk"] = self.number
            write_json(self.path, state)
        else:
            code = self.store.knowledge
            self.experiment._validate_source(code)
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "baseline.py").write_text(code)
            state = {"schema": "physical_recursia_chunk.v1", "chunk": self.number,
                "phase": "executing", "source_ids": self.source_ids, "official_sha256": digest(code),
                "active_round": "baseline", "candidates_proposed": 0, "reference": {},
                "inherited_from_chunk": self.number - 1,
                "rounds": {"baseline": {"module_sha256": digest(code), "records": {}, "reviewed": False}}}
            write_json(self.path, state)
            self.store.save_learning(code, {"kind": "chunk_baseline", "chunk": self.number,
                "task_ids": [], "updated": False, "dry_run": self.experiment.dry_run,
                "path": str(self.folder)})
        self._sync()
        return state

    def source(self, task_ids):
        raise ValueError("paired chunk scheduler requires source --chunk N, not --task")

    def review(self):
        try:
            return super().review()
        finally:
            self._sync()

    def propose(self):
        try:
            return super().propose()
        finally:
            self._sync()

    def score(self, *args, **kwargs):
        try:
            return super().score(*args, **kwargs)
        finally:
            self._sync()

    def _retry_rejected_candidates(self, state):
        """Reopen only trailing offline rejections, preserving executed rounds."""
        if state["phase"] not in {"ready", "complete"}:
            return state  # resume an already reopened validation normally
        names = []
        for index in range(state["candidates_proposed"], 0, -1):
            name = f"candidate_{index}"
            check = self.folder / name / "offline_checks.json"
            if name in state["rounds"] or not check.is_file() or read_json(check).get("passed") is not False:
                break
            names.append(name)
        if not names:
            raise ValueError("no trailing offline-rejected candidates to retry; validated rounds are preserved")
        if any(not round_.get("reviewed") for round_ in state["rounds"].values()):
            raise ValueError("finish the current source validation before retrying rejected proposals")
        archive = self.folder / "offline_rejections" / f"retry_{time.time_ns()}"
        archive.mkdir(parents=True, exist_ok=False)
        write_json(archive / "chunk.json", state)
        write_json(archive / "recursia_chunk.json", self._index())
        for name in reversed(names):
            (self.folder / name).rename(archive / name)
        for name in ("chunk_result.json", "snapshot.json", f"N{self.number}.py"):
            path = self.folder / name
            if path.exists():
                path.rename(archive / name)
        state.setdefault("proposal_recoveries", []).append({
            "archive": str(archive.relative_to(self.store.root)),
            "rejected_candidates": list(reversed(names)),
            "candidates_proposed_before": state["candidates_proposed"],
            "reason": "operator requested candidate regeneration after offline rejection"})
        state.update(phase="ready", candidates_proposed=state["candidates_proposed"] - len(names))
        write_json(self.path, state)
        self._sync()
        self.experiment.line(f"Chunk {self.number}: archived offline rejections at {archive}; "
                             "resuming candidate generation without replaying baseline")
        return state

    def _discard_task_attempt(self, name, task_id):
        """Input 0 deletes this task's attempts, without altering other rounds."""
        state = self.load()
        round_ = state["rounds"][name]
        record = round_["records"][task_id]
        if round_.get("reviewed") or record.get("operator_action") != "retry":
            raise ValueError("only the current unreviewed task selected for retry may be deleted")
        folder = self.folder / name / task_id
        if folder.resolve() != folder or not folder.is_relative_to(self.folder):
            raise ValueError("discarded task directory must stay inside the current chunk")
        # The saved retry action allows cleanup to finish on the next launch
        # if interrupted here. Deletion happens before the reset prompt.
        if folder.exists():
            shutil.rmtree(folder)
        method_state = self.store.state
        method_state["source_events"] = [event for event in method_state["source_events"]
            if not (event.get("kind") == "panel_execution" and event.get("path")
                    and Path(event["path"]).is_relative_to(folder))]
        write_json(self.store.state_path, method_state)
        round_.setdefault("next_attempts", {})[task_id] = record.get("attempt", 1) + 1
        del round_["records"][task_id]
        round_.get("attempt_history", {}).pop(task_id, None)
        if "score_history" in round_:
            round_["score_history"] = [score for score in round_["score_history"] if score.get("task_id") != task_id]
        state["phase"] = "executing"
        write_json(self.path, state)
        self._sync()
        self.experiment.line(f"Deleted {name}/{task_id} attempt records; only this task will restart")
        return folder

    def _judge_task(self, name, task_id, record, judge):
        success = judge(self.number, name, self.tasks[task_id], record)
        if not isinstance(success, bool):
            raise ValueError("manual task score must be True (success) or False (failure)")
        state = self.load()
        round_ = state["rounds"][name]
        if round_.get("reviewed"):
            raise ValueError("cannot change a reviewed round")
        score = {"success": success, "automatic_success": bool(record["success"]),
                 "scored_by": "human", "note": "operator task-end goal judgement",
                 "dry_run": bool(record.get("dry_run"))}
        # Keep controller/worker errors as diagnostics. The operator judges the
        # observed task outcome; a dry preview still cannot validate a candidate.
        record.update(automatic_success=score["automatic_success"], human_success=success,
                      success=bool(success and not score["dry_run"]), scored_by="human", note=score["note"])
        round_["records"][task_id] = record
        write_json(self.path, state)
        self._sync()
        if record.get("path"):
            execution = Path(record["path"])
            write_json(execution / "manual_score.json", score)
            result_path = execution / "result.json"
            if result_path.is_file():
                result = read_json(result_path)
                result.update(automatic_success=score["automatic_success"], human_success=success,
                              scored_by="human", success=record["success"], task_completed=record["success"])
                write_json(result_path, result)
            summary_path = execution.parent / "source_result.json"
            if summary_path.is_file():
                summary = read_json(summary_path)
                for task in summary["tasks"]:
                    task.update(success=record["success"], validated=record["success"],
                                automatic_success=score["automatic_success"], scored_by="human")
                write_json(summary_path, summary)
            method_state = self.store.state
            for event in method_state["source_events"]:
                if event.get("kind") == "panel_execution" and event.get("path") == str(execution):
                    event.update(success=record["success"], automatic_success=score["automatic_success"], scored_by="human")
            write_json(self.store.state_path, method_state)
        return record

    def run_chunk(self, number, *, prepare, decide=None, judge=None, resume=False, retry_rejected_candidates=False):
        self._mutable()
        self._select(number)
        index = self._index()
        if index["chunks"] != self.chunks or index["official_sha256"] != digest(self.store.knowledge):
            raise ValueError("chunk history and official module disagree")
        for key, record in index["records"].items():
            path = self.store.root / "source" / f"chunk_{int(key):03d}" / "chunk.json"
            if not path.is_file() or digest(path.read_text()) != record["sha256"]:
                raise ValueError("archived chunk evidence changed; do not continue with mixed results")
        for previous in range(1, number):
            if index["records"].get(str(previous), {}).get("phase") != "complete":
                raise ValueError(f"finish chunk {previous} before starting chunk {number}")
        later = [int(key) for key in index["records"] if int(key) > number]
        if later:
            raise ValueError("earlier chunks are archived; do not replay them after advancing")
        first_prepared = None
        if self.path.exists():
            state = self.load()
            if retry_rejected_candidates:
                state = self._retry_rejected_candidates(state)
            if state["phase"] == "complete":
                raise ValueError("this chunk already completed; choose the next chunk")
            faults = [r for round_ in state["rounds"].values() for r in round_["records"].values() if self._faulted(r)]
            # Interactive recovery resumes at the saved task-end decision,
            # never by automatically replaying an interrupted motion.
            if faults and not resume and decide is None:
                raise RolloutAbort("chunk interrupted/controller fault; recover the robot then use --resume; consumed tasks are not replayed")
        else:
            if retry_rejected_candidates:
                raise ValueError("start this chunk before retrying rejected candidates")
            prepare(self.number, "baseline", self.tasks[self.source_ids[0]])
            first_prepared = self.source_ids[0]
            self._bootstrap(self.tasks[self.source_ids[0]])
        executions = []
        while True:
            state = self.load()
            phase = state["phase"]
            if phase == "complete":
                break
            if phase in {"executing", "review_pending"}:
                name = state["active_round"]
                for task_id in self.source_ids:
                    while True:
                        record = self.load()["rounds"][name]["records"].get(task_id)
                        if record and record.get("operator_action") == "continue":
                            # On upgrading a paused round, collect missing human
                            # grades for its completed tasks without replaying them.
                            if judge is not None and record.get("scored_by") != "human":
                                self._judge_task(name, task_id, record, judge)
                            break
                        retry = bool(record and record.get("operator_action") == "retry")
                        if retry:
                            discarded = self._discard_task_attempt(name, task_id)
                            executions = [item for item in executions if not Path(item["path"]).is_relative_to(discarded)]
                            continue
                        if record is None:
                            if first_prepared != task_id or name != "baseline":
                                prepare(self.number, name, self.tasks[task_id])
                            first_prepared = None
                            expected_attempt = self.load()["rounds"][name].get("next_attempts", {}).get(task_id, 1)
                            try:
                                packet = super().source([task_id])
                            except RolloutAbort:
                                # Worker failures can raise after saving a complete
                                # failure record. Present that record to the operator;
                                # budget/setup errors and Ctrl+C still exit normally.
                                saved = read_json(self.path)["rounds"][name]["records"].get(task_id)
                                if (decide is None or not saved or not self._faulted(saved)
                                        or saved.get("attempt") != expected_attempt
                                        or saved.get("scored_by") != "evaluator"
                                        or not (Path(saved["path"]) / "result.json").is_file()):
                                    raise
                                packet = read_json(Path(saved["path"]).parent / "source_result.json")
                            finally:
                                self._sync()
                            executions.extend(packet["tasks"])
                            record = self.load()["rounds"][name]["records"][task_id]
                        if self._faulted(record) and decide is None:
                            raise RolloutAbort(f"chunk stopped at {task_id}: controller/camera/worker fault; operator decision required")
                        if judge is not None and record.get("scored_by") != "human":
                            record = self._judge_task(name, task_id, record, judge)
                        for item in executions:
                            if record.get("path") and item["path"] == str(Path(record["path"]).parent):
                                item.update(success=record["success"], validated=record["success"],
                                            scored_by=record.get("scored_by"))
                        choice = decide(self.number, name, self.tasks[task_id], record) if decide else 1
                        if choice not in (0, 1):
                            raise ValueError("task-end decision must be 1 (continue) or 0 (retry current task)")
                        state = self.load()
                        state["rounds"][name]["records"][task_id]["operator_action"] = "retry" if choice == 0 else "continue"
                        if choice == 0:
                            state["phase"] = "executing"
                        write_json(self.path, state)
                        self._sync()
                        if choice == 0:
                            discarded = self._discard_task_attempt(name, task_id)
                            executions = [item for item in executions if not Path(item["path"]).is_relative_to(discarded)]
                self.review()
            elif phase == "ready":
                self.experiment.line(f"Chunk {self.number}: generating candidate {state['candidates_proposed'] + 1}")
                try:
                    self.propose()
                except ValueError as exc:
                    # Invalid candidate consumes its slot and keeps its evidence.
                    if self.load()["phase"] not in {"ready", "complete"}:
                        raise
                    self.experiment.line(f"Candidate rejected by offline checks: {exc}")
            else:
                raise ValueError("unknown chunk phase: " + phase)
        code = self.store.knowledge
        snapshot = self.folder / f"N{self.number}.py"
        if snapshot.exists() and snapshot.read_text() != code:
            raise ValueError("completed chunk snapshot changed")
        snapshot.write_text(code)
        write_json(self.folder / "snapshot.json", {"chunk": self.number, "sha256": digest(code),
            "agent_git_commit": EvolutionGit(self.store).head(), "path": str(snapshot)})
        packet = {"method": "recursia", "workflow": "recursia_multi_turn.v1",
            "source_schedule": "physical_recursia_chunks.v2", "chunk": self.number, "completed": True,
            "tasks": executions, "progress": self.status(), "run_dir": str(self.folder)}
        write_json(self.folder / "chunk_result.json", packet)
        self.experiment.line(f"Chunk {self.number} completed; accepted={packet['progress']['accepted_candidates']}; stopping")
        return packet

    def finish(self):
        index = self._index()
        if len(index["records"]) != len(self.chunks) or any(r["phase"] != "complete" for r in index["records"].values()):
            raise ValueError("complete all source chunks before freezing")
        for record in index["records"].values():
            path = self.store.root / record["path"]
            if digest(path.read_text()) != record["sha256"]:
                raise ValueError("chunk evidence changed before freezing")
        if index["official_sha256"] != digest(self.store.knowledge):
            raise ValueError("official module changed before freezing")
