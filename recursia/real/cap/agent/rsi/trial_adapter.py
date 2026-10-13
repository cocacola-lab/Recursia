"""Real RGB-D/model bindings for the copied simulation trial and executor."""
from pathlib import Path

from cap.agent.fr3_console import emit_progress
from .capabilities import skill_names
from .observation import image_paths
from .prompts import multi_turn_prompt
from .runtime import PhysicalFault
from .simulation.executor import CodeExecutor
from .simulation.trial import _parse_multi_turn_decision, run_trial
from .store import write_json


class ModelBudget:
    """Count action/vision calls together; an optional limit bounds both."""
    def __init__(self, generator, limit, folder):
        self.generator, self.limit, self.folder = generator, limit, Path(folder)
        self.calls = self.action_calls = 0

    def _charge(self, *, action):
        if self.limit is not None and self.calls >= self.limit:
            raise RuntimeError("trial model-call budget exhausted")
        self.calls += 1
        self.action_calls += int(action)
        write_json(self.folder / "model_calls.json", {"calls": self.calls, "action_calls": self.action_calls})

    def generate(self, *args, **kwargs):
        self._charge(action=True)
        return self.generator.generate(*args, **kwargs)

    def evaluate_placement(self, *args, **kwargs):
        self._charge(action=False)
        return self.generator.evaluate_placement(*args, **kwargs)


class Fr3CodeEnvironment:
    """Persistent Python variables and fresh images after each executed block."""
    def __init__(self, runtime, task):
        self.runtime = runtime
        def printer(*values, **kwargs):
            print(*values, **kwargs)
            emit_progress(runtime.progress_path, tool="print", status="done",
                          detail={"text": " ".join(map(str, values))[:1000]})
        self.executor = CodeExecutor(task, runtime.namespace(), printer=printer)
        self.verification = None
        self.injection_error = None

    def step(self, code):
        start = len(self.runtime.trace)
        if self.injection_error:
            result = {"ok": False, "stdout": "", "stderr": self.injection_error,
                      "execution_status": "not_executed"}
            self.injection_error = None
        else:
            result = self.executor.run(code)
        error = result["stderr"]
        # A generated try/except cannot clear a controller fault recorded by a tool.
        fault = self.runtime.hardware_fault
        if fault:
            error += "\nterminated episode: physical controller fault\n"
        info = {"sandbox_rc": 0 if result["ok"] else 1, "stdout": result["stdout"],
                "stderr": error, "tool_trace": self.runtime.trace[start:],
                "execution_status": result["execution_status"]}
        collisions = [record for record in self.runtime.trace[start:] if record.get("recoverable_collision")]
        if collisions:
            info["recoverable_collisions"] = collisions
            # A policy may catch the exception. Still show the failure in the
            # following model feedback, without forcing FINISH.
            info["stderr"] += "\nRecoverable collision feedback: " + "\n".join(r["error"] for r in collisions) + "\n"
        if fault:
            info["stop_reason"] = "hardware_fault"
            return None, 0., True, False, info
        try:
            observation = self.runtime.capture_feedback()
        except Exception as exc:
            info["stderr"] += f"\nterminated episode: camera feedback unavailable: {exc}\n"
            return None, 0., True, False, info
        # Match sim.step: reward is measured after each step, independently of
        # the model's FINISH. Perception failure is recorded, not a motion gate.
        try:
            self.verification = self.runtime.evaluate_goal()
        except PhysicalFault as exc:
            self.runtime.hardware_fault = True
            info["stderr"] += f"\nterminated episode: {exc}\n"
            info["stop_reason"] = "hardware_fault"
            return observation, 0., True, False, info
        except Exception as exc:
            self.verification = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            if getattr(self.runtime, "collision_pending", False):
                info["stderr"] += f"\nController collision reflex requires Recovery: {exc}\n"
        info["task_completed"] = bool(self.verification.get("success"))
        return observation, float(info["task_completed"]), False, False, info


def execute_trial(runtime, task, code, generator, *, context, timeout_s, limit=10, skill_code="", fixed_policy=False):
    env = Fr3CodeEnvironment(runtime, task)
    if skill_code:
        # Import only validated definitions, never replay source actions.
        try:
            skill_names(skill_code)
            result = env.executor.run(skill_code)
            if not result["ok"]:
                env.injection_error = "source skill injection failed:\n" + result["stderr"]
        except Exception as exc:
            env.injection_error = f"source skill injection failed: {type(exc).__name__}: {exc}\n"
    folder = Path(runtime.run_dir)

    def decide(observation, executed, info, turns):
        if fixed_policy:
            return "finish", None, "Fixed module execution ended; no online action generation.", ""
        prompt = multi_turn_prompt(context, turns, info, observation)
        index = len(turns)
        turn_dir = folder / "turns" / f"turn_{index:03d}"
        write_json(turn_dir / "observation.json", observation)
        (turn_dir / "prompt.md").write_text(prompt)
        emit_progress(runtime.progress_path, tool="model_feedback", status="started",
                      detail={"turn": index, "stage": f"turn {index}: current images + stdout/stderr"})
        schema = {"type": "object", "properties": {"analysis": {"type": "string"},
                  "response": {"type": "string"}}, "required": ["analysis", "response"],
                  "additionalProperties": False}
        response = generator.generate(prompt, timeout_s=timeout_s, images=image_paths(observation), schema=schema)
        write_json(turn_dir / "response.json", response)
        content = response["response"].strip()
        if not content.startswith(("REGENERATE", "FINISH")):
            raise ValueError("multi-turn response must start with REGENERATE or FINISH")
        decision, new_code = _parse_multi_turn_decision("FINISH" if content.startswith("FINISH") else content)
        emit_progress(runtime.progress_path, tool="model_feedback", status="done",
                      detail={"turn": index, "decision": decision, "text": response["analysis"][:1000]})
        emit_progress(runtime.progress_path, tool="print", status="done",
                      detail={"text": f"Turn {index}: {decision.upper()} — {response['analysis'][:1000]}"})
        return decision, new_code, response["analysis"], prompt

    def save_turn(turn, responses):
        turn_dir = folder / "turns" / f"turn_{turn['index']:03d}"
        write_json(turn_dir / "step.json", turn)
        (turn_dir / "code.py").write_text(turn["code"])
        if responses is not None:
            write_json(folder / "all_responses.json", responses)

    result = run_trial(env, code, decide, limit=limit, on_turn=save_turn)
    write_json(folder / "trial.json", result)
    (folder / "executed_code.py").write_text(result["executed_code"])
    (folder / "final_code.py").write_text(result["final_code"])
    return result, env.verification
