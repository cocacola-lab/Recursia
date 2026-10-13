"""Initialize, learn source tasks, freeze and evaluate three isolated RSI methods."""
import argparse
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from dataclasses import asdict

from cap.agent.fr3_console import AgentConsole, emit_progress
from cap.agent.model_defaults import DEFAULT_REASONING_EFFORT
from cap.agent.worker_utils import _run_worker_live
from cap.saved_scripts.fr3_placement.config import load_config
from .campaign import Experiment, RolloutAbort
from .generator import IsolatedGenerator, configured_model
from .simulation.trial import MULTITURN_LIMIT
from .trial_adapter import execute_trial, ModelBudget
from .prompts import TOOLS, policy_prompt, public_task
from .runtime import FiveToolRuntime, PhysicalFault, _json_value
from .observation import capture_observation
from .store import METHODS, MethodStore, initialize, read_json, write_json

REAL_ROOT = Path(__file__).resolve().parents[3]
TARGET_MAX_TURNS = 20

def manual_task_judge(task, result, console):
    if result.get("hardware_fault") or result.get("unknown_worker_failure"):
        console.line("The task ended after a hardware/worker fault. Errors remain recorded; manual scoring does not recover hardware or issue actions.")
    while True:
        try:
            answer = input(f"{task['id']} finished: {task['instruction']}\n"
                           "Score the actual task outcome: enter s for success or f for failure: ").strip().lower()
        except EOFError as exc:
            raise RolloutAbort("task result saved without a manual score; no source summary or final evaluation recorded") from exc
        if answer in {"s", "f"}:
            return answer == "s"
        console.line("Enter s (success) or f (failure).")

def validate_tasks(tasks):
    seen = set()
    for task in tasks:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task["id"]) or task["id"] in seen:
            raise ValueError("task IDs must be unique path-safe identifiers")
        seen.add(task["id"])
        if task["split"] not in {"source", "target"} or not task.get("instruction"):
            raise ValueError("each task needs source/target split and an instruction")
        names = [o["name"] for o in task["objects"]]
        if len(names) != len(set(names)) or not names:
            raise ValueError("object names must be unique within a task")
        for goal in task["goals"]:
            if goal["relation"] not in {"on", "in", "right_of"} or any(goal[k] not in names for k in ("object", "target")):
                raise ValueError("goal requires a supported relation and declared objects")
        if not task["goals"]:
            raise ValueError("empty task goal")
    if not any(t["split"] == "source" for t in tasks) or not any(t["split"] == "target" for t in tasks):
        raise ValueError("manifest requires source and target tasks")
    return tasks

def run_worker(args, *, runtime_factory=FiveToolRuntime, decision_generator=None):
    # Initial code is validated inside CodeExecutor, so failures reach the
    # same observation/REGENERATE loop as errors in later blocks.
    code = args.code.read_text()
    library_file = getattr(args, "library_file", None)
    skill_code = library_file.read_text() if library_file else ""
    task = read_json(args.task_file)
    cfg = load_config(args.config)
    generator = decision_generator or IsolatedGenerator(getattr(args, "vision_provider", "codex"),
        getattr(args, "vision_model", None), getattr(args, "reasoning_effort", DEFAULT_REASONING_EFFORT))
    budget = ModelBudget(generator, getattr(args, "vision_call_limit", None), args.run_dir)
    vision_args = {}
    if runtime_factory is FiveToolRuntime:
        vision_args = {"vision_generator": budget,
                       "vision_timeout_s": getattr(args, "vision_timeout_s", 300),
                       "vision_call_limit": getattr(args, "vision_call_limit", None)}
    runtime = runtime_factory(cfg, args.run_dir, task=task, dry_run=args.dry_run,
                              snapshot=args.snapshot, progress_path=args.progress_log, **vision_args)
    write_json(args.run_dir / "runtime_config.json", _json_value(asdict(cfg)))
    output, error, verification, trial_result = io.StringIO(), None, None, None
    context_file = getattr(args, "context_file", None)
    context = context_file.read_text() if context_file else policy_prompt(task)
    try:
        with contextlib.redirect_stdout(output):
            trial_result, verification = execute_trial(runtime, public_task(task), code, budget,
                context=context, timeout_s=getattr(args, "vision_timeout_s", 300),
                limit=getattr(args, "turn_limit", MULTITURN_LIMIT),
                skill_code=skill_code, fixed_policy=getattr(args, "fixed_policy", False))
        error = trial_result["model_error"]
        if trial_result["terminated"]:
            error = trial_result["info_step"]["stderr"]
        elif trial_result["truncated"]:
            error = error or "multi-turn execution limit reached"
        elif trial_result["info_step"]["sandbox_rc"] != 0:
            error = error or trial_result["info_step"]["stderr"]
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, PhysicalFault):
            runtime.hardware_fault = True
        traceback.print_exc(file=output)
    packet = {"success": bool(verification and verification["success"] and error is None
                              and not runtime.hardware_fault and not args.dry_run),
              "task_completed": bool(trial_result and trial_result["info_step"].get("task_completed")
                                     and not runtime.hardware_fault and not args.dry_run),
              "execution_success": bool(trial_result and trial_result["info_step"]["sandbox_rc"] == 0),
              "verification": verification, "error": error, "stdout": output.getvalue(),
              "tool_trace": runtime.trace, "hardware_fault": runtime.hardware_fault,
              "vision_model_calls": budget.calls - budget.action_calls,
              "execution_model_calls": budget.action_calls,
              "executed_code": trial_result["executed_code"] if trial_result else "",
              "final_code": trial_result["final_code"] if trial_result else "",
              "model_finished": bool(trial_result and trial_result["num_finishes"] and not getattr(args, "fixed_policy", False)),
              "fixed_policy": getattr(args, "fixed_policy", False),
              "policy_finished": bool(getattr(args, "fixed_policy", False) and trial_result
                                      and not trial_result["terminated"] and trial_result["info_step"]["sandbox_rc"] == 0),
              "num_regenerations": trial_result["num_regenerations"] if trial_result else 0,
              "num_code_blocks": trial_result["num_code_blocks"] if trial_result else 0,
              "physical_motion_commanded": runtime.physical_motion_commanded, "dry_run": args.dry_run}
    packet.update(collision_faults=getattr(runtime, "collision_faults", []),
                  pending_controller_recovery=getattr(runtime, "collision_pending", False))
    packet.update(forced_finish=bool(runtime.hardware_fault),
                  stop_reason="hardware_fault" if runtime.hardware_fault else None,
                  model_finished=bool(packet["model_finished"] and not runtime.hardware_fault))
    write_json(args.output, packet)
    try:
        runtime.close()
    except Exception as exc:
        packet.update(success=False, unknown_worker_failure=True, cleanup_error=str(exc))
        write_json(args.output, packet)
    return packet

class PhysicalEvaluator:
    def __init__(self, args, console, meta=None):
        self.args, self.console = args, console
        self.meta = meta or {}
        self.vision_call_limit = self.meta.get("model_call_cap")

    def set_vision_budget(self, remaining):
        self.vision_call_limit = remaining

    def observe(self, task, folder):
        lock_path = REAL_ROOT / "logs" / ".rsi_hardware.lock"
        lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RolloutAbort("another RSI method currently owns the robot/camera") from exc
            command = [sys.executable, "-m", "cap.agent.rsi.cli", "observe", "--config", str(self.args.config),
                       "--run-dir", str(folder)]
            if self.args.dry_run:
                command += ["--dry-run", "--snapshot", str(self.args.snapshot)]
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=70, cwd=REAL_ROOT)
            except subprocess.TimeoutExpired as exc:
                raise RolloutAbort("front/wrist capture timed out; no robot motion requested") from exc
            if process.returncode:
                raise RolloutAbort(f"front/wrist capture failed: {process.stderr[-1200:]}")
            return read_json(folder / "observation.json")

    def __call__(self, code, task, trial, folder):
        if not self.args.dry_run and os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION") != "1":
            raise RolloutAbort("physical source/eval requires OPENFORGE_ALLOW_PHYSICAL_MOTION=1")
        lock_path = REAL_ROOT / "logs" / ".rsi_hardware.lock"
        lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RolloutAbort("another RSI method currently owns the robot/camera") from exc
            command = [sys.executable, "-m", "cap.agent.rsi.cli", "worker", "--config", str(self.args.config),
                       "--code", str(folder / "code.py"), "--task-file", str(folder / "task.json"),
                       "--run-dir", str(folder), "--output", str(folder / "result.json")]
            # The simulation loop's limit is a zero-based block index, not a count.
            turn_limit = (TARGET_MAX_TURNS - 1 if task.get("split") == "target" else
                          self.meta.get("multi_turn_limit", MULTITURN_LIMIT))
            command += ["--vision-provider", self.meta.get("provider", "codex"),
                        "--vision-timeout-s", str(self.args.timeout_s),
                        "--turn-limit", str(turn_limit),
                        "--reasoning-effort", self.meta.get("reasoning_effort") or DEFAULT_REASONING_EFFORT]
            if self.vision_call_limit is not None:
                command += ["--vision-call-limit", str(self.vision_call_limit)]
            if (folder / "context.md").is_file():
                command += ["--context-file", str(folder / "context.md")]
            if (folder / "skill_code.py").is_file():
                command += ["--library-file", str(folder / "skill_code.py")]
            if (folder / "execution_mode.json").is_file() and read_json(folder / "execution_mode.json")["fixed_policy"]:
                command += ["--fixed-policy"]
            if self.meta.get("model"):
                command += ["--vision-model", self.meta["model"]]
            if self.args.dry_run:
                command += ["--dry-run", "--snapshot", str(self.args.snapshot)]
            def charges():
                path = folder / "model_calls.json"
                packet = read_json(path) if path.is_file() else {"calls": 0, "action_calls": 0}
                return {"execution_model_calls": packet["action_calls"],
                        "vision_model_calls": packet["calls"] - packet["action_calls"]}
            try:
                process = _run_worker_live(command, folder, self.args.worker_timeout_s, self.console)
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "worker timeout", "unknown_worker_failure": True,
                        **charges()}
            if not (folder / "result.json").exists():
                return {"success": False, "error": f"worker exited {process.returncode} without a result",
                        "unknown_worker_failure": True, "stderr": process.stderr[-1000:],
                        **charges()}
            result = read_json(folder / "result.json")
            if process.returncode not in (0, 2):
                result.update(success=False, unknown_worker_failure=True, error=f"worker exit {process.returncode}")
            return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="create three independent empty stores; no model or hardware call")
    init.add_argument("--campaign", type=Path, required=True)
    init.add_argument("--manifest", type=Path, default=REAL_ROOT / "config/rsi_tasks.json")
    init.add_argument("--provider", choices=["codex", "claude"], default="codex")
    init.add_argument("--model")
    init.add_argument("--multi-turn-limit", type=int, default=MULTITURN_LIMIT)
    init.add_argument("--model-call-cap", type=int, help="optional cumulative model-call cap; default unlimited")
    init.add_argument("--rollout-cap", type=int, help="optional cumulative source rollout cap; default unlimited")
    for command in ("source", "eval", "freeze", "status", "prompt", "propose", "review", "score"):
        child = sub.add_parser(command)
        child.add_argument("--campaign", type=Path, required=True)
        child.add_argument("--method", choices=METHODS, required=True)
        if command != "prompt":
            child.add_argument("--json", action="store_true", help="print full result JSON instead of a compact summary")
        if command == "source":
            selection = child.add_mutually_exclusive_group(required=True)
            selection.add_argument("--task", "--tasks", dest="task", help="one source ID for sd-only/aspire or legacy recursia")
            selection.add_argument("--chunk", type=int, help="recursia chunk: 1=S1/S2, 2=S3/S4, 3=S5/S6")
            child.add_argument("--resume", action="store_true", help="resume saved chunk progress; task retries still require input 0")
            child.add_argument("--retry-rejected-candidates", action="store_true",
                               help="archive trailing offline-rejected proposals and regenerate them without replaying baseline")
        if command in {"eval", "prompt"}:
            child.add_argument("--task", required=True)
        if command == "eval":
            child.add_argument("--layout", choices=["eval01", "eval02", "eval03"], help=argparse.SUPPRESS)
        if command == "propose":
            child.add_argument("--timeout-s", type=int, default=300)
        if command == "score":
            child.add_argument("--task", required=True)
            child.add_argument("--layout", choices=["eval01", "eval02", "eval03"], help=argparse.SUPPRESS)
            choice = child.add_mutually_exclusive_group(required=True)
            choice.add_argument("--success", action="store_true")
            choice.add_argument("--failure", action="store_true")
            child.add_argument("--progress", type=int, help="optional common failure-stage index, higher means later")
            child.add_argument("--failure-class", help="optional failure attribution for stage-improvement acceptance")
            child.add_argument("--note", default="", help="human score evidence, e.g. video timestamp")
        if command in {"source", "eval"}:
            child.add_argument("--config", type=Path, default=REAL_ROOT / "config/rsi_fr3_station.example.yaml")
            child.add_argument("--dry-run", action="store_true")
            child.add_argument("--snapshot", type=Path, help="saved RGB-D, required for dry-run")
            # Legacy flag remains accepted; all rollouts now run without stdin confirmation.
            child.add_argument("--noninteractive", action="store_true", help=argparse.SUPPRESS)
            child.add_argument("--timeout-s", type=int, default=300)
            child.add_argument("--worker-timeout-s", type=int, default=3600)
    observe = sub.add_parser("observe", help="capture front/wrist RGB-D without robot or model calls")
    observe.add_argument("--config", type=Path, required=True)
    observe.add_argument("--run-dir", type=Path, required=True)
    observe.add_argument("--dry-run", action="store_true")
    observe.add_argument("--snapshot", type=Path)
    worker = sub.add_parser("worker", help="internal six-tool execution worker")
    for flag in ("config", "code", "task-file", "run-dir", "output"):
        worker.add_argument("--"+flag, type=Path, required=True)
    worker.add_argument("--dry-run", action="store_true")
    worker.add_argument("--snapshot", type=Path)
    worker.add_argument("--progress-log", type=Path)
    worker.add_argument("--vision-provider", choices=["codex", "claude"], default="codex")
    worker.add_argument("--vision-model")
    worker.add_argument("--context-file", type=Path)
    worker.add_argument("--library-file", type=Path)
    worker.add_argument("--turn-limit", type=int, default=MULTITURN_LIMIT)
    worker.add_argument("--fixed-policy", action="store_true", help="execute module once without online action generation")
    worker.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    worker.add_argument("--vision-timeout-s", type=int, default=300)
    worker.add_argument("--vision-call-limit", type=int, help="optional shared action/vision call cap; default unlimited")
    args = parser.parse_args(argv)
    for key in ("campaign", "manifest", "config", "snapshot", "code", "task_file", "run_dir", "output", "progress_log", "context_file", "library_file"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    try:
        if args.command == "init":
            if any(cap is not None and cap < 1 for cap in (args.model_call_cap, args.rollout_cap)):
                parser.error("budgets must be positive")
            if args.multi_turn_limit < 0:
                parser.error("multi-turn limit must be nonnegative")
            tasks = validate_tasks(read_json(args.manifest)["tasks"])
            model = args.model or (configured_model() if args.provider == "codex" else None)
            root = initialize(args.campaign, tasks, provider=args.provider, model=model,
                              model_call_cap=args.model_call_cap, rollout_cap=args.rollout_cap,
                              multi_turn_limit=args.multi_turn_limit, chunk_size=2)
            print(f"Created {root}; methods={','.join(METHODS)}; model={model or 'provider default'}; tools={','.join(TOOLS)}")
            return 0
        if getattr(args, "snapshot", None) and not getattr(args, "dry_run", False):
            parser.error("saved snapshot is only available in dry-run")
        if getattr(args, "dry_run", False) and not args.snapshot:
            parser.error("dry-run requires --snapshot and never opens a live camera")
        if args.command == "observe":
            capture_observation(load_config(args.config), args.run_dir,
                                snapshot=args.snapshot if args.dry_run else None)
            return 0
        if args.command == "worker":
            if args.turn_limit < 0 or (args.vision_call_limit is not None and args.vision_call_limit < 0) or args.vision_timeout_s < 1:
                parser.error("invalid worker budgets")
            result = run_worker(args)
            return 0 if result["success"] else 2
        store = MethodStore(args.campaign, args.method)
        with store.locked():
            if args.command == "status":
                state = store.state
                summary = {k: state[k] for k in (
                    "method", "frozen", "knowledge_sha256", "source_model_calls", "source_rollouts")}
                summary["source_vision_model_calls"] = state.get("source_vision_model_calls", 0)
                if store.method == "recursia" and store.meta.get("recursia_protocol"):
                    from .recursia import recursia_protocol
                    summary["chunk"] = recursia_protocol(store).status()
                    if (store.root / "evaluation_panel.json").is_file():
                        summary["target_panel"] = read_json(store.root / "evaluation_panel.json")
                    if args.json:
                        state = {**state, **summary}
                print(json.dumps(state if args.json else summary,
                    ensure_ascii=False, indent=2))
                return 0
            if args.command == "freeze":
                frozen = store.freeze()
                print(json.dumps(frozen if args.json else {k: frozen[k] for k in (
                    "method", "sha256", "source_task_ids", "contains_preview_learning")}, ensure_ascii=False, indent=2))
                return 0
            if args.command == "prompt":
                task = next(t for t in store.meta["tasks"] if t["id"] == args.task)
                if store.method == "recursia":
                    from .prompts import recursia_bootstrap_prompt, API
                    print(API + "\nFrozen module:\n" + store.knowledge if task["split"] == "target" else
                          recursia_bootstrap_prompt([t for t in store.meta["tasks"] if t["split"] == "source"]))
                    return 0
                print(policy_prompt(task, store.knowledge, target=task["split"] == "target", method=store.method))
                return 0
            if args.command in {"propose", "review", "score"}:
                from .recursia import recursia_protocol
                experiment = None
                if args.command == "propose":
                    if args.timeout_s < 1:
                        parser.error("timeout must be positive")
                    experiment = Experiment(store, IsolatedGenerator(store.meta["provider"], store.meta["model"],
                        store.meta.get("reasoning_effort", DEFAULT_REASONING_EFFORT)), None, timeout_s=args.timeout_s,
                        console=AgentConsole())
                protocol = recursia_protocol(store, experiment)
                result = (protocol.propose() if args.command == "propose" else protocol.review() if args.command == "review" else
                          protocol.score_target(args.task, args.layout, success=args.success, note=args.note)
                          if protocol.tasks[args.task]["split"] == "target" else
                          protocol.score(args.task, success=args.success, progress=args.progress,
                                         failure_class=args.failure_class, note=args.note))
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
            if min(args.timeout_s, args.worker_timeout_s) < 1:
                parser.error("timeouts must be positive")
            chunk_run = args.command == "source" and args.chunk is not None
            if chunk_run and (store.method != "recursia" or
                              store.meta.get("recursia_protocol", {}).get("schema") != "physical_recursia_chunks.v2"):
                parser.error("--chunk requires recursia in a new paired-chunk campaign; initialize a fresh campaign")
            if args.command == "source" and args.resume and not chunk_run:
                parser.error("--resume is only available with --chunk")
            if args.command == "source" and args.retry_rejected_candidates and not chunk_run:
                parser.error("--retry-rejected-candidates is only available with --chunk")
            if not args.dry_run and os.environ.get("OPENFORGE_ALLOW_PHYSICAL_MOTION") != "1":
                parser.error("source/eval requires OPENFORGE_ALLOW_PHYSICAL_MOTION=1; init/prompt/status/freeze do not")
            console = AgentConsole()
            console.line(f"{args.method} {args.command}; independent workspace={store.root}")
            if store.meta["tools"] != list(TOOLS):
                raise ValueError("campaign tool interface is outdated; initialize a fresh six-tool campaign")
            console.line("Recursia panel: archived module → execute → fresh images/output → REGENERATE/FINISH; updates require paired validation"
                         if store.method == "recursia" else
                         "Simulation multi-turn loop: execute → fresh images/output → REGENERATE/FINISH")
            console.line(f"Provider={store.meta['provider']} model={store.meta['model'] or 'provider default'}; six primitives with front/wrist vision")
            if store.meta['provider'] == 'codex':
                console.line("Codex Fast mode enabled; reasoning effort remains "
                             + store.meta.get('reasoning_effort', DEFAULT_REASONING_EFFORT))
            experiment = Experiment(store, IsolatedGenerator(store.meta["provider"], store.meta["model"],
                                    store.meta.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
                                    PhysicalEvaluator(args, console, store.meta), timeout_s=args.timeout_s,
                                    console=console, dry_run=args.dry_run,
                                    judge=(lambda task, result: manual_task_judge(task, result, console))
                                          if store.method == "sd-only" and not args.dry_run else None)
            if store.method == "sd-only" and not args.dry_run:
                console.line("Task-end score: manual s=success / f=failure; source summary uses the human result")
            console.line("One chunk: baseline → candidates → paired validation → review; manual success/failure after each task, then 1=continue / 0=retry"
                         if chunk_run else "One task execution; REGENERATE continues, FINISH ends; no automatic reruns")
            if args.command == "eval":
                console.line(f"Target: at most {TARGET_MAX_TURNS} turns; automatically freeze source skills for this run; unfreeze on exit")
            if chunk_run:
                from .recursia import recursia_protocol
                def prepare(number, round_name, task):
                    try:
                        input(f"Chunk {number} / {round_name} / {task['id']}: {task['instruction']}\n"
                              "Reset the robot and objects to the initial layout for this task. Press Enter to execute (Ctrl+C to abort): ")
                    except EOFError as exc:
                        raise RolloutAbort("manual scene reset requires terminal input; no task started") from exc
                def judge(number, round_name, task, record):
                    while True:
                        try:
                            answer = input(f"Chunk {number} / {round_name} / {task['id']} / attempt {record.get('attempt', 1)} finished\n"
                                           "Score the actual task outcome: enter s for success or f for failure: ").strip().lower()
                        except EOFError as exc:
                            raise RolloutAbort("task result saved; rerun this chunk command to provide the pending manual score") from exc
                        if answer in {"s", "f"}:
                            return answer == "s"
                        console.line("Enter s (success) or f (failure).")
                def decide(number, round_name, task, record):
                    if record.get("hardware_fault"):
                        console.line("The task ended after a robot fault. Complete controller Recovery and reset the scene before retrying or continuing.")
                    elif record.get("unknown_worker_failure") or "terminated episode: camera feedback unavailable" in (record.get("error") or ""):
                        console.line("The task ended after a camera/worker fault. Restore the affected device or process before continuing or retrying.")
                    while True:
                        try:
                            answer = input(f"Chunk {number} / {round_name} / {task['id']} / attempt {record.get('attempt', 1)} "
                                           f"finished, success={record['success']}\n"
                                           "Enter 1 to continue or 0 to delete this attempt and retry the current task (without restarting the chunk): ").strip()
                        except EOFError as exc:
                            raise RolloutAbort("task result saved; rerun this chunk command to resume the pending 1/0 choice") from exc
                        if answer in {"0", "1"}:
                            return int(answer)
                        console.line("Enter 1 or 0.")
                result = recursia_protocol(store, experiment).run_chunk(args.chunk, prepare=prepare,
                    decide=decide, judge=judge, resume=args.resume, retry_rejected_candidates=args.retry_rejected_candidates)
            else:
                result = (experiment.source(args.task.split(",")) if args.command == "source"
                          else experiment.evaluate(args.task, layout=args.layout))
            if args.command == "eval":
                console.line("Source skills unfrozen; target snapshot saved with the evaluation")
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif args.command == "eval":
                completion = f"FINISH={result['model_finished']}"
                console.line(f"Evaluation: success={result['success']}; {completion}; "
                             f"new programs={result['programs_generated']}; snapshot={result['snapshot_sha256'][:12]}")
                console.line(f"Result: {result['run_dir']}/eval_result.json")
            else:
                task_result = result['tasks'][0] if result['tasks'] else None
                if chunk_run:
                    console.line(f"Source chunk {result['chunk']}: completed={result['completed']}; {len(result['tasks'])} task executions this invocation")
                elif store.method == "recursia":
                    console.line(f"Source {result['round']}: success={task_result['success']}; official module unchanged")
                else:
                    console.line(f"Source: success={task_result['success']}; FINISH={task_result['model_finished']}; "
                                 f"skills updated={task_result['updated']}")
                console.line(f"Source model calls={store.state['source_model_calls']}; rollouts={store.state['source_rollouts']}")
                console.line(f"Output: {store.root}")
                if result.get("progress"):
                    console.line("Chunk: " + json.dumps(result["progress"], ensure_ascii=False))
            return 0
    except (ValueError, KeyError, StopIteration, FileNotFoundError, FileExistsError, RolloutAbort, RuntimeError) as exc:
        print(f"RSI stopped: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
