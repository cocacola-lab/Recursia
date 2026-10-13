"""Three learning mechanisms sharing a six-tool executor and frozen evaluation."""
import json
import time
from pathlib import Path

from .capabilities import validate_policy
from .prompts import policy_prompt, summary_prompt, execution_context
from .store import digest, write_json, read_json
from .simulation.library import SkillLibrary
from .learning import evolve
from .promotion import Promotion

class RolloutAbort(RuntimeError):
    pass

class Experiment:
    def __init__(self, store, generator, evaluator, *, timeout_s=300, console=None, dry_run=False, judge=None):
        self.store, self.generator, self.evaluator = store, generator, evaluator
        self.timeout_s, self.console, self.dry_run = timeout_s, console, dry_run
        self.judge = judge
        self.tasks = {t["id"]: t for t in store.meta["tasks"]}

    def line(self, text):
        if self.console:
            self.console.line(text)

    def _request(self, folder, prompt, *, source, observation_task=None, schema=None, images=()):
        folder.mkdir(parents=True, exist_ok=True)
        images = list(images)
        if observation_task is not None and hasattr(self.evaluator, "observe"):
            from .observation import image_paths, public_observation
            self.line("Observing current front and wrist views before generation")
            observation = self.evaluator.observe(observation_task, folder / "scene")
            images = image_paths(observation)
            prompt += "\nCurrent scene: attached image 1=front, image 2=wrist.\n" + json.dumps(public_observation(observation))
        (folder / "prompt.md").write_text(prompt)
        if schema is not None:
            write_json(folder / "schema.json", schema)
        if source:
            state = self.store.state
            cap = self.store.meta.get("model_call_cap")
            if cap is not None and state["source_model_calls"] + state.get("source_vision_model_calls", 0) >= cap:
                raise RolloutAbort("source model-call budget exhausted")
            state["source_model_calls"] += 1
            write_json(self.store.state_path, state)
        self.line(f"{self.store.method}: generating {folder.name}")
        response = self.generator.generate(prompt, timeout_s=self.timeout_s,
            **({"images": images} if images else {}), **({"schema": schema} if schema is not None else {}))
        write_json(folder / "response.json", response)
        return response

    def _generate(self, folder, prompt, *, source, observation_task=None):
        response = self._request(folder, prompt, source=source, observation_task=observation_task)
        if not isinstance(response.get("code"), str):
            raise ValueError("model response requires a code string")
        return response["code"]

    def _rollout(self, code, task, episode, folder, *, source, knowledge=None, fixed_policy=False):
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "code.py").write_text(code)
        write_json(folder / "task.json", task)
        (folder / "context.md").write_text(execution_context(task, self.store.context_knowledge() if knowledge is None else knowledge,
                                                          fixed_policy=fixed_policy))
        (folder / "skill_code.py").write_text(self.store.skill_code())
        write_json(folder / "execution_mode.json", {"fixed_policy": fixed_policy})
        if source:
            state = self.store.state
            cap = self.store.meta.get("rollout_cap")
            if cap is not None and state["source_rollouts"] >= cap:
                raise RolloutAbort("source rollout budget exhausted")
            state["source_rollouts"] += 1
            write_json(self.store.state_path, state)
        if hasattr(self.evaluator, "set_vision_budget"):
            remaining = self.store.meta.get("model_call_cap")
            if source and remaining is not None:
                state = self.store.state
                remaining -= state["source_model_calls"] + state.get("source_vision_model_calls", 0)
            self.evaluator.set_vision_budget(None if remaining is None else max(0, remaining))
        self.line(f"{task['id']} {episode}: {task['instruction']}")
        result = self.evaluator(code, task, episode, folder)
        if source:
            state = self.store.state
            state["source_model_calls"] += result.get("execution_model_calls", 0)
            state["source_vision_model_calls"] = state.get("source_vision_model_calls", 0) + result.get("vision_model_calls", 0)
            write_json(self.store.state_path, state)
        # Dry command previews are never successes and never validate a skill.
        result["workflow"] = ("fixed_recursia.v1" if fixed_policy else
                              "recursia_multi_turn.v1" if self.store.method == "recursia" else
                              "simulation_multi_turn.v1")
        if fixed_policy:
            result["model_finished"] = False
            if result.get("execution_model_calls", 0):
                raise RolloutAbort("fixed Recursia stack unexpectedly generated online action code")
        result["success"] = bool(result.get("success") and not result.get("dry_run") and not self.dry_run)
        if result.get("hardware_fault"):
            result.update(success=False, task_completed=False, model_finished=False,
                          forced_finish=True, stop_reason="hardware_fault")
            fault_file = folder / "robot_fault.json"
            if fault_file.is_file():
                try:
                    result["fault_snapshot"] = read_json(fault_file)
                except (OSError, ValueError) as exc:
                    result["fault_snapshot_error"] = str(exc)
            self.line("FINISH (runtime): hardware fault; no further motion")
        write_json(folder / "result.json", result)
        if self.store.method == "sd-only" and self.judge is not None and not self.dry_run and not result.get("dry_run"):
            success = self.judge(task, result)
            if not isinstance(success, bool):
                raise ValueError("manual task score must be True (success) or False (failure)")
            score = {"success": success, "automatic_success": bool(result["success"]),
                     "automatic_task_completed": bool(result.get("task_completed")),
                     "scored_by": "human", "note": "operator task-end goal judgement"}
            write_json(folder / "manual_score.json", score)
            result.update(automatic_success=score["automatic_success"],
                          automatic_task_completed=score["automatic_task_completed"],
                          human_success=success, scored_by="human", success=success, task_completed=success)
            write_json(folder / "result.json", result)
        if result.get("unknown_worker_failure") and result.get("scored_by") != "human":
            raise RolloutAbort(result.get("error") or "worker/controller fault requires operator recovery")
        return result

    def source(self, task_ids):
        if self.store.method == "recursia":
            from .recursia import recursia_protocol
            return recursia_protocol(self.store, self).source(task_ids)
        from .versions import EvolutionGit
        if self.store.state['frozen']:
            raise ValueError('source learning is locked after freeze')
        if len(task_ids) != 1:
            raise ValueError('one source task per command; FINISH ends the run')
        task = self.tasks[task_ids[0]]
        if task['split'] != 'source':
            raise ValueError('source learning may use only source task IDs')
        versions = EvolutionGit(self.store)
        versions.checkpoint('Before source ' + task['id'])
        try:
            packet = self._source(task_ids)
        finally:
            version = versions.checkpoint('Source ' + task['id'] + ': reusable learning')
        packet['agent_version'] = version
        write_json(Path(packet['run_dir']) / 'source_result.json', packet)
        return packet

    def _source(self, task_ids):
        if self.store.state["frozen"]:
            raise ValueError("source learning is locked after freeze")
        if len(task_ids) != 1:
            raise ValueError("one source task per command; FINISH ends the run")
        task = self.tasks[task_ids[0]]
        if task["split"] != "source":
            raise ValueError("source learning may use only source task IDs")
        root = self.store.root / "source" / str(time.time_ns())
        folder = root / task["id"]
        folder.mkdir(parents=True)
        knowledge = self.store.knowledge
        context = self.store.context_knowledge()
        library = SkillLibrary(self.store.root / "skill_library.json") if self.store.method == "aspire" else None
        before = digest(context)
        library_before = self._library_digest(library)
        code = ""
        try:
            code = self._generate(folder / "generation",
                policy_prompt(task, context, method=self.store.method), source=True, observation_task=task)
            result = self._rollout(code, task, "source", folder / "execution", source=True, knowledge=context)
            result["initial_code"] = code
        except RolloutAbort:
            raise
        except Exception as exc:
            result = {"success": False, "task_completed": False, "initial_code": code,
                      "error": f"{type(exc).__name__}: {exc}"}
        completed = bool(result.get("task_completed", (result.get("verification") or {}).get("success", False))
                         and not result.get("dry_run") and not self.dry_run)
        new_skills, learning_error = [], None
        promotion = (Promotion(self.store, task["id"], folder)
                     if library is not None and not self.dry_run and not result.get("dry_run") else None)
        if library is not None and completed:
            try:
                final_code = result.get("final_code") or result.get("executed_code") or code
                new_skills = library.extract_from_code(final_code, task_name=task["id"])
                library.get_promoted_skills()
                if library.skills or library.path.exists():
                    library.save()
                if any(skill.promoted for skill in library.skills.values()):
                    knowledge = library.get_skill_docs()
            except Exception as exc:
                learning_error = f"{type(exc).__name__}: {exc}"
        library_after = self._library_digest(library)
        after = digest(self.store.context_knowledge(knowledge))
        event = {"kind": "execution", "task_id": task["id"], "method": self.store.method,
                 "success": bool(result["success"]), "task_completed": completed, "validated": completed,
                 "model_finished": bool(result.get("model_finished", False)),
                 "forced_finish": bool(result.get("forced_finish", False)),
                 "stop_reason": result.get("stop_reason"),
                 "updated": before != after or library_before != library_after,
                 "knowledge_before_sha256": before, "knowledge_after_sha256": after,
                 "library_before_sha256": library_before, "library_after_sha256": library_after,
                 "new_skills": new_skills, "dry_run": self.dry_run, "learning_error": learning_error,
                 "path": str(folder), "source_schedule": "single_episode.v1"}
        if result.get("scored_by") == "human":
            event.update(scored_by="human", automatic_success=result["automatic_success"],
                         human_success=result["human_success"])
        self.store.save_learning(knowledge, event)
        write_json(folder / "learning.json", event)
        events = [event]
        write_json(folder / "findings.json", {"status": "execution_finished", "execution": result, "learning": events})
        if promotion is not None:
            reason, findings_path, error = "No reusable source pattern was promoted.", None, None
            try:
                reason, findings_path = evolve(self.store, task, folder, result,
                    lambda directory, prompt, **kwargs: self._request(directory, prompt, source=True, **kwargs), promotion)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                reason = "Source skill authoring failed: " + error
                if (folder / "findings.md").is_file():
                    findings_path = folder / "findings.md"
            record = promotion.finish(reason=reason, status="failed" if error else "completed", findings_path=findings_path)
            event = {"kind": "skill_promotion", "task_id": task["id"], "method": "aspire",
                     "promotion_id": record["promotion_id"], "updated": not record["no_op"],
                     "learning_error": error, "reason": reason, "status": record["status"],
                     "changed_skill_files": record["changed_skill_files"],
                     "findings_path": record["findings_path"], "no_op": record["no_op"],
                     "dry_run": False, "path": str(promotion.folder), "source_schedule": "single_episode.v1"}
            self.store.save_learning(self.store.knowledge, event)
            write_json(folder / "skill_learning.json", event)
            events.append(event)
        # Read-only source learning never invokes the executor again. FINISH,
        # failure or an execution limit cannot schedule another physical run.
        if self.store.method == "sd-only":
            before, error = digest(knowledge), None
            try:
                prompt = summary_prompt("sd-only", task, result, knowledge)
                proposed = self._generate(folder / "summary", prompt, source=True)
                knowledge = proposed
            except RolloutAbort as exc:
                if not result.get("forced_finish"):
                    raise
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            event = {"kind": "summary", "task_id": task["id"], "method": self.store.method,
                     "updated": before != digest(knowledge), "learning_error": error,
                     "knowledge_before_sha256": before, "knowledge_after_sha256": digest(knowledge),
                     "dry_run": self.dry_run, "path": str(folder / "summary"),
                     "source_schedule": "single_episode.v1"}
            self.store.save_learning(knowledge, event)
            write_json(folder / "summary" / "learning.json", event)
            events.append(event)
        write_json(folder / "findings.json", {"status": "completed", "execution": result, "learning": events})
        summary = {"task_id": task["id"], "method": self.store.method,
                   "success": bool(result["success"]), "validated": completed,
                   "model_finished": bool(result.get("model_finished", False)),
                   "forced_finish": bool(result.get("forced_finish", False)),
                   "stop_reason": result.get("stop_reason"),
                   "updated": any(e["updated"] for e in events), "new_skills": new_skills,
                   "skill_promotions": [e for e in events if e["kind"] == "skill_promotion"],
                   "learning_errors": [e["learning_error"] for e in events if e.get("learning_error")],
                   "path": str(folder)}
        if result.get("scored_by") == "human":
            summary.update(scored_by="human", automatic_success=result["automatic_success"],
                           human_success=result["human_success"])
        packet = {"method": self.store.method, "workflow": "simulation_multi_turn.v1",
                  "source_schedule": "single_episode.v1", "tasks": [summary], "run_dir": str(root)}
        write_json(root / "source_result.json", packet)
        return packet

    def _validate_source(self, code):
        if self.store.method == "recursia":
            forbidden = [d["name"] for t in self.tasks.values() for d in t["objects"]]
            forbidden += list(self.tasks)
            forbidden += ["red", "white", "yellow", "cream", "tomato", "chocolate", "mug", "plate", "basket",
                          "pudding", "cheese", "sauce", "living_room", "scene", "black", "brown", "blue",
                          "green", "orange", "pink", "purple", "grey", "gray"]
            try:
                validate_policy(code, stack=True, forbidden_literals=forbidden)
            except SyntaxError as exc:
                raise ValueError(f"Recursia stack syntax error: {exc}") from exc

    @staticmethod
    def _library_digest(library):
        if library is None:
            return None
        data = json.loads(library.path.read_text()) if library.path.exists() else {"skills": {}}
        return digest(json.dumps(data, sort_keys=True))

    def evaluate(self, task_id, *, layout=None):
        if self.tasks[task_id]["split"] != "target":
            raise ValueError("frozen evaluation requires a target task")
        with self.store.evaluation_freeze():
            if self.store.method == "recursia" and self.store.meta.get("recursia_protocol"):
                from .recursia import recursia_protocol
                return recursia_protocol(self.store, self).evaluate(task_id, layout=layout)
            return self._evaluate(task_id)

    def _evaluate(self, task_id, *, fixed_policy=False, layout=None):
        task = self.tasks[task_id]
        if task["split"] != "target":
            raise ValueError("frozen evaluation requires a target task")
        frozen = self.store.snapshot()
        if fixed_policy or self.store.method == "recursia":
            self._validate_source(frozen["content"])
        if frozen["contains_preview_learning"] and not self.dry_run:
            raise ValueError("preview learning is not a trained physical snapshot; use a fresh campaign")
        root = self.store.root / "eval" / f"{task_id}_{time.time_ns()}"
        root.mkdir(parents=True)
        write_json(root / "snapshot.json", frozen)
        state_bytes = self.store.state_path.read_bytes()
        try:
            generated = int(self.store.method != "recursia")
            if generated:
                code = self._generate(root / "generation", policy_prompt(task, frozen["content"],
                    target=True, method=self.store.method), source=False, observation_task=task)
            else:
                code = frozen["content"] + "\nrun_task(TASK)\n"
            execution = self._rollout(code, task, "target", root / "execution", source=False,
                                      knowledge=frozen["content"], fixed_policy=fixed_policy)
            packet = {"method": self.store.method, "workflow": execution["workflow"],
                      "execution_schedule": "single_episode.v1", "task_id": task_id,
                      "snapshot_sha256": frozen["sha256"], "programs_generated": generated,
                      "policy_paths": [str(root / "execution/code.py")],
                      "vision_model_calls": execution.get("vision_model_calls", 0),
                      "execution_model_calls": execution.get("execution_model_calls", 0),
                      "success": execution["success"], "model_finished": bool(execution.get("model_finished", False)),
                      "forced_finish": bool(execution.get("forced_finish", False)),
                      "stop_reason": execution.get("stop_reason"),
                      "execution": execution, "dry_run": self.dry_run, "run_dir": str(root)}
            if execution.get("scored_by") == "human":
                packet.update(scored_by="human", automatic_success=execution["automatic_success"],
                              human_success=execution["human_success"])
            if self.store.method == "recursia" and layout is not None:
                packet["layout"] = layout
            write_json(root / "eval_result.json", packet)
            return packet
        finally:
            self.store.snapshot()
            if self.store.state_path.read_bytes() != state_bytes:
                raise RuntimeError("target evaluation modified source learning state")
