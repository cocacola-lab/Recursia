"""One physical source chunk: fixed baseline, two proposals, paired validation.

Each source command executes exactly one task. Scene resets remain external;
propose/review/score/freeze never start a robot execution. Candidate code stays
in its archive until a complete source panel passes the acceptance rule.
"""
import difflib
import json
import time
from pathlib import Path

from .prompts import recursia_bootstrap_prompt, recursia_candidate_prompt
from .store import digest, initial_knowledge, read_json, write_json
from .versions import EvolutionGit


def recursia_protocol(store, experiment=None):
    if store.meta.get("recursia_protocol", {}).get("schema") == "physical_recursia_chunks.v2":
        from .chunked_recursia import ChunkedRecursiaProtocol
        return ChunkedRecursiaProtocol(store, experiment)
    return RecursiaProtocol(store, experiment)


class RecursiaProtocol:
    def __init__(self, store, experiment=None):
        if store.method != "recursia":
            raise ValueError("chunk protocol is only for recursia")
        self.store, self.experiment = store, experiment
        self.path = store.root / "recursia_chunk.json"
        self.folder = store.root / "source" / "chunk_001"
        self.tasks = {t["id"]: t for t in store.meta["tasks"]}
        self.source_ids = [t["id"] for t in store.meta["tasks"] if t["split"] == "source"]
        self.config = store.meta.get("recursia_protocol")

    def _mutable(self):
        if not self.config:
            raise ValueError("legacy recursia campaign: initialize a fresh campaign for the chunk protocol")
        if self.store.state["frozen"]:
            raise ValueError("source learning is locked after freeze")

    def load(self):
        if not self.path.is_file():
            raise ValueError("Recursia chunk not started; run each source task before proposing or freezing")
        state = read_json(self.path)
        if state["source_ids"] != self.source_ids or state["official_sha256"] != digest(self.store.knowledge):
            raise ValueError("Recursia chunk and official module disagree; use a fresh campaign after rollback/manual edits")
        return state

    def status(self):
        if not self.path.exists():
            return {"protocol": "physical_recursia_chunk.v1", "phase": "not_started", "pending_tasks": self.source_ids,
                    "next": "source --task " + self.source_ids[0]}
        state = self.load()
        name = state["active_round"]
        round_ = state["rounds"].get(name)
        pending = [task for task in self.source_ids if round_ and task not in round_["records"]]
        return {"protocol": state["schema"], "phase": state["phase"], "active_round": name,
                "pending_tasks": pending, "candidates_proposed": state["candidates_proposed"],
                "accepted_candidates": [name for name, item in state["rounds"].items() if item.get("accepted")],
                "next": "source --task " + pending[0] if pending else
                        "review / propose / freeze" if state["phase"] != "complete" else "freeze / eval"}

    def _bootstrap(self, task):
        if self.store.state["source_events"] or self.store.knowledge != initial_knowledge("recursia"):
            raise ValueError("Recursia chunk requires fresh knowledge; existing learning is preserved, use a fresh campaign")
        exp = self.experiment
        versions = EvolutionGit(self.store)
        versions.checkpoint("Initial empty Recursia stack")
        bootstrap = self.folder / "bootstrap"
        if bootstrap.exists():
            # Failed capture/generation/validation leaves evidence. A retry
            # needs fresh frames and must not overwrite that attempt.
            bootstrap = self.folder / f"bootstrap_retry_{time.time_ns()}"
        code = exp._generate(bootstrap, recursia_bootstrap_prompt(
            [self.tasks[key] for key in self.source_ids]), source=True, observation_task=task)
        exp._validate_source(code)
        self.store.save_learning(code, {"kind": "bootstrap", "task_ids": [], "updated": True,
            "dry_run": exp.dry_run, "validation": "offline_structure_only", "path": str(bootstrap)})
        versions.checkpoint("N0: initial Recursia stack without source execution feedback")
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "baseline.py").write_text(code)
        state = {"schema": "physical_recursia_chunk.v1", "phase": "executing", "source_ids": self.source_ids,
                 "official_sha256": digest(code), "active_round": "baseline", "candidates_proposed": 0,
                 "reference": {}, "rounds": {"baseline": {"module_sha256": digest(code), "records": {},
                                                          "reviewed": False}}}
        write_json(self.path, state)
        return state

    def module(self, name):
        return (self.folder / "baseline.py" if name == "baseline" else self.folder / name / "stack.py").read_text()

    def source(self, task_ids, *, retry=False):
        self._mutable()
        if len(task_ids) != 1:
            raise ValueError("one source task per command; reset the scene externally between tasks")
        task = self.tasks[task_ids[0]]
        if task["split"] != "source":
            raise ValueError("source learning may use only source task IDs")
        state = self.load() if self.path.exists() else self._bootstrap(task)
        if state["phase"] != "executing" and not (retry and state["phase"] == "review_pending"):
            raise ValueError("source panel finished; use review/propose/freeze, not another source rollout")
        name = state["active_round"]
        round_ = state["rounds"][name]
        if retry and (task["id"] not in round_["records"] or round_["reviewed"]):
            raise ValueError("retry requires an existing task attempt in an unreviewed round")
        if task["id"] in round_["records"] and not retry:
            raise ValueError("this task already ran in this round; no automatic repeat or overwrite")
        code = self.module(name)
        if digest(code) != round_["module_sha256"]:
            raise ValueError("archived round module hash changed")
        self.experiment._validate_source(code)
        cap = self.store.meta.get("rollout_cap")
        if cap is not None and self.store.state["source_rollouts"] >= cap:
            raise RuntimeError("source rollout budget exhausted")
        folder = self.folder / name / task["id"]
        attempt = round_.get("next_attempts", {}).pop(task["id"], 1)
        if retry:
            history = round_.setdefault("attempt_history", {}).setdefault(task["id"], [])
            history.append(dict(round_["records"][task["id"]]))
            attempt = len(history) + 1
        if attempt > 1:
            folder = folder / f"attempt_{attempt:03d}"
        # Reserve before motion. An interrupted/unknown attempt is never silently
        # replayed on resume; it remains a consumed, unsuccessful case.
        round_["records"][task["id"]] = {"success": False, "scored_by": "interrupted",
                                          "error": "attempt reserved; execution result pending", "attempt": attempt}
        write_json(self.path, state)
        error = None
        try:
            result = self.experiment._rollout(code + "\nrun_task(TASK)\n", task, name,
                folder / "execution", source=True, knowledge=code, fixed_policy=False)
        except BaseException as exc:
            error = exc
            result_path = folder / "execution" / "result.json"
            result = read_json(result_path) if result_path.is_file() else {
                "success": False, "error": f"{type(exc).__name__}: {exc}", "unknown_worker_failure": True}
        record = {"success": bool(result.get("success")), "scored_by": "evaluator", "progress": None,
                  "failure_class": None, "path": str(folder / "execution"),
                  "hardware_fault": bool(result.get("hardware_fault")),
                  "unknown_worker_failure": bool(result.get("unknown_worker_failure")),
                  "dry_run": bool(self.experiment.dry_run or result.get("dry_run")), "error": result.get("error"),
                  "workflow": result.get("workflow"), "attempt": attempt}
        round_["records"][task["id"]] = record
        if len(round_["records"]) == len(self.source_ids):
            state["phase"] = "review_pending"
        write_json(self.path, state)
        self.store.save_learning(self.store.knowledge, {"kind": "panel_execution", "task_id": task["id"],
            "round": name, "success": record["success"], "updated": False, "dry_run": record["dry_run"],
            "path": str(folder / "execution"), "source_schedule": state["schema"]})
        summary = {"task_id": task["id"], "success": record["success"], "updated": False,
                   "model_finished": bool(result.get("model_finished")), "forced_finish": bool(result.get("forced_finish")),
                   "stop_reason": result.get("stop_reason"), "validated": record["success"],
                   "attempt": attempt, "path": str(folder)}
        packet = {"method": "recursia", "workflow": "recursia_multi_turn.v1", "source_schedule": state["schema"], "round": name,
                  "tasks": [summary], "run_dir": str(folder), "progress": self.status(),
                  "agent_version": {"head": EvolutionGit(self.store).head(), "changed": False}}
        write_json(folder / "source_result.json", packet)
        if error is not None:
            raise error
        return packet

    def score(self, task_id, *, success, progress=None, failure_class=None, note=""):
        self._mutable()
        state = self.load()
        round_ = state["rounds"][state["active_round"]]
        if round_["reviewed"]:
            raise ValueError("round already reviewed; its acceptance evidence cannot be changed")
        record = round_["records"][task_id]
        if success and any(record.get(key) for key in ("dry_run", "hardware_fault", "unknown_worker_failure")):
            raise ValueError("preview/interrupted/controller-fault execution cannot be scored as success")
        if progress is not None and progress < 0:
            raise ValueError("progress must be a nonnegative, consistently defined stage index")
        round_.setdefault("score_history", []).append({"task_id": task_id, "previous": dict(record),
            "success": success, "progress": progress, "failure_class": failure_class, "note": note})
        record.update(success=success, scored_by="human", progress=progress,
                      failure_class=None if success else failure_class, note=note)
        write_json(self.path, state)
        return {"round": state["active_round"], "task_id": task_id, "score": record}

    @staticmethod
    def verdict(reference, records, failure_class=None):
        regressions = [key for key, old in reference.items() if old["success"] and not records[key]["success"]]
        gains = [key for key, old in reference.items() if not old["success"] and records[key]["success"]]
        affected = [key for key, old in reference.items() if not old["success"]
                    and failure_class and old.get("failure_class") == failure_class]
        advanced = bool(affected) and all(records[key]["success"] or (
            records[key].get("failure_class") and records[key]["failure_class"] != failure_class
            and reference[key].get("progress") is not None and records[key].get("progress") is not None
            and records[key]["progress"] > reference[key]["progress"]) for key in affected)
        return {"accepted": not regressions and bool(gains or advanced), "gained_tasks": gains,
                "regressed_tasks": regressions, "failure_class_eliminated": failure_class if advanced else None,
                "reason": "regression" if regressions else "success_gain" if gains else
                          "failure_stage_advanced" if advanced else "no_measured_improvement"}

    def review(self):
        self._mutable()
        state = self.load()
        name = state["active_round"]
        round_ = state["rounds"][name]
        if round_["reviewed"]:
            return round_
        if set(round_["records"]) != set(self.source_ids):
            raise ValueError("all source tasks must run once before review or another proposal")
        if name == "baseline":
            decision = {"accepted": False, "reason": "baseline_reference", "gained_tasks": [], "regressed_tasks": []}
            state["reference"] = round_["records"]
        else:
            if round_["parent_sha256"] != state["official_sha256"]:
                raise ValueError("candidate is not based on the current official module")
            decision = self.verdict(state["reference"], round_["records"], round_.get("failure_class"))
            # Preview runs cannot establish a measured improvement.
            if any(r.get("dry_run") for r in round_["records"].values()):
                decision.update(accepted=False, reason="preview_only")
            if decision["accepted"]:
                code = self.module(name)
                if digest(code) != round_["module_sha256"]:
                    raise ValueError("candidate module hash changed")
                from .capabilities import validate_policy
                validate_policy(code, stack=True)
                self.store.save_learning(code, {"kind": "accepted_candidate", "task_ids": self.source_ids,
                    "round": name, "updated": True, "validated": True, "dry_run": False,
                    "knowledge_before_sha256": state["official_sha256"], "knowledge_after_sha256": digest(code),
                    "path": str(self.folder / name), "decision": decision})
                state["official_sha256"] = digest(code)
                state["reference"] = round_["records"]
                EvolutionGit(self.store).checkpoint("Accepted paired source candidate " + name)
        round_.update(reviewed=True, **decision)
        state["phase"] = "complete" if state["candidates_proposed"] >= self.config["max_candidates"] else "ready"
        write_json(self.path, state)
        write_json(self.folder / name / "review.json", decision)
        return {"round": name, **decision, "progress": self.status()}

    def propose(self):
        self._mutable()
        state = self.load()
        if state["phase"] in ("executing", "review_pending"):
            self.review()
            state = self.load()
        if state["phase"] == "complete" or state["candidates_proposed"] >= self.config["max_candidates"]:
            raise ValueError("candidate budget exhausted; freeze the final official version")
        index = state["candidates_proposed"] + 1
        name = f"candidate_{index}"
        folder = self.folder / name
        # Count attempted proposals even if generation/offline validation fails.
        state["candidates_proposed"] = index
        write_json(self.path, state)
        current = self.store.knowledge
        feedback = []
        for old_name, old_round in state["rounds"].items():
            results = {}
            for key, record in old_round["records"].items():
                path = Path(record.get("path", "")) / "result.json"
                results[key] = {"score": record, "execution": read_json(path) if path.is_file() else {}}
                results[key]["previous_attempts"] = []
                for old in old_round.get("attempt_history", {}).get(key, []):
                    old_path = Path(old.get("path", "")) / "result.json"
                    results[key]["previous_attempts"].append({"score": old,
                        "execution": read_json(old_path) if old_path.is_file() else {}})
            feedback.append({"round": old_name, "accepted": old_round.get("accepted"), "results": results})
        try:
            response = self.experiment._request(folder, recursia_candidate_prompt(
                [self.tasks[key] for key in self.source_ids], current, feedback), source=True, schema={
                    "type": "object", "properties": {"analysis": {"type": "string"}, "code": {"type": "string"},
                        "failure_class": {"type": "string"}}, "required": ["analysis", "code", "failure_class"],
                    "additionalProperties": False})
            diagnosis = response["analysis"]
            if not isinstance(diagnosis, str) or not diagnosis.strip():
                raise ValueError("candidate requires a failure diagnosis")
            (folder / "DIAGNOSIS.md").write_text(diagnosis)
            code = response["code"]
            self.experiment._validate_source(code)
            if code == current:
                raise ValueError("candidate is byte-identical to the current module")
            (folder / "stack.py").write_text(code)
            (folder / "changes.patch").write_text("".join(difflib.unified_diff(
                current.splitlines(True), code.splitlines(True), fromfile="official/stack.py", tofile=name + "/stack.py")))
            round_ = {"module_sha256": digest(code), "parent_sha256": digest(current), "records": {},
                      "reviewed": False, "failure_class": response["failure_class"] or None}
            state["rounds"][name] = round_
            state.update(active_round=name, phase="executing")
            write_json(folder / "offline_checks.json", {"passed": True, "module_sha256": digest(code)})
        except Exception as exc:
            state["phase"] = "complete" if index >= self.config["max_candidates"] else "ready"
            write_json(folder / "offline_checks.json", {"passed": False, "error": f"{type(exc).__name__}: {exc}"})
            write_json(self.path, state)
            raise
        write_json(self.path, state)
        return {"candidate": name, "official_updated": False, "path": str(folder), "progress": self.status()}

    def finish(self):
        self._mutable()
        state = self.load()
        if state["phase"] in ("executing", "review_pending"):
            self.review()
            state = self.load()
        state["phase"] = "complete"
        write_json(self.path, state)

    def evaluate(self, task_id, *, layout=None):
        task = self.tasks[task_id]
        if task["split"] != "target":
            raise ValueError("frozen evaluation requires a target task")
        self.store.snapshot()
        if layout is None:
            # One independent target episode, matching sd-only/ASPIRE. Keep
            # explicit layouts only for compatibility with historical panels.
            return self.experiment._evaluate(task_id, fixed_policy=False)
        path = self.store.root / "evaluation_panel.json"
        panel = read_json(path) if path.is_file() else {}
        layouts = self.config["target_layouts"]
        records = panel.setdefault(task_id, {})
        if layout not in layouts or layout in records:
            raise ValueError("target requires an unused layout: " + ",".join(layouts))
        records[layout] = {"success": False, "status": "started"}
        write_json(path, panel)
        packet = self.experiment._evaluate(task_id, fixed_policy=False, layout=layout)
        records[layout] = {"success": packet["success"], "status": "completed", "run_dir": packet["run_dir"],
                           "snapshot_sha256": packet["snapshot_sha256"]}
        write_json(path, panel)
        return packet

    def score_target(self, task_id, layout=None, *, success, note=""):
        if self.tasks[task_id]["split"] != "target":
            raise ValueError("target scoring requires a target task")
        if layout is None:
            runs = list((self.store.root / "eval").glob(task_id + "_*"))
            if not runs:
                raise ValueError("no target execution to score")
            run = max(runs, key=lambda p: int(p.name.rsplit("_", 1)[1]))
            if not (run / "eval_result.json").is_file():
                raise ValueError("latest target execution did not complete; no earlier run was scored")
            result = read_json(run / "eval_result.json")
            score_path = run / "manual_score.json"
            record = read_json(score_path) if score_path.is_file() else {
                "success": result["success"], "automatic_success": result["success"],
                "status": "completed", "run_dir": str(run), "snapshot_sha256": result["snapshot_sha256"]}
        else:
            path = self.store.root / "evaluation_panel.json"
            panel = read_json(path)
            record = panel[task_id][layout]
            result = read_json(Path(record["run_dir"]) / "eval_result.json")
        if record["status"] != "completed":
            raise ValueError("target execution did not complete; keep the interrupted attempt recorded")
        snapshot_path = Path(record["run_dir"]) / "snapshot.json"
        if snapshot_path.is_file():
            snapshot = read_json(snapshot_path)
            if (snapshot["method"] != self.store.method or snapshot["campaign_id"] != self.store.meta["campaign_id"]
                    or digest(snapshot["content"]) != result["snapshot_sha256"]
                    or snapshot["sha256"] != record["snapshot_sha256"]):
                raise ValueError("target score snapshot and evaluation disagree")
        else:
            # Older trials predate per-run snapshots and used a persistent freeze.
            self.store.snapshot()
        execution = result["execution"]
        if success and (result.get("dry_run") or execution.get("hardware_fault") or execution.get("unknown_worker_failure")):
            raise ValueError("preview/interrupted/controller-fault execution cannot be scored as success")
        record.setdefault("score_history", []).append({"previous_success": record["success"], "success": success, "note": note})
        record.update(success=success, scored_by="human", note=note)
        if layout is None:
            write_json(score_path, record)
            result.setdefault("automatic_success", result["success"])
            result.update(success=success, human_success=success, scored_by="human", note=note)
            write_json(run / "eval_result.json", result)
            return {"task_id": task_id, "run_dir": str(run), "score": record}
        write_json(path, panel)
        return {"task_id": task_id, "layout": layout, "score": record}
