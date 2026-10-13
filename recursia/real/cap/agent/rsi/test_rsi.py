"""No hardware/model calls: method isolation, held-out lockout and learning decisions."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .cli import validate_tasks, run_worker, PhysicalEvaluator
from .campaign import Experiment, RolloutAbort
from .generator import codex_command, claude_command, DISABLED_FEATURES
from .capabilities import validate_policy
from .simulation.executor import CodeExecutor
from .prompts import TOOLS, policy_prompt
from .store import initialize, MethodStore, read_json, write_json
from .runtime import FiveToolRuntime, describe_volume

def execute_policy(code, task, tools, *, printer=print):
    validate_policy(code)
    result = CodeExecutor(task, tools).run(code)
    if not result["ok"]:
        raise ValueError(result["stderr"])


TASKS = [
    {"id": "S1", "split": "source", "instruction": "put item on support",
     "objects": [{"name": "item"}, {"name": "support"}],
     "goals": [{"relation": "on", "object": "item", "target": "support"}]},
    {"id": "S2", "split": "source", "instruction": "put another item on support",
     "objects": [{"name": "item"}, {"name": "support"}],
     "goals": [{"relation": "on", "object": "item", "target": "support"}]},
    {"id": "T1", "split": "target", "instruction": "target instruction",
     "objects": [{"name": "item"}, {"name": "support"}],
     "goals": [{"relation": "on", "object": "item", "target": "support"}]},
    {"id": "T2", "split": "target", "instruction": "another target instruction",
     "objects": [{"name": "item"}, {"name": "support"}],
     "goals": [{"relation": "on", "object": "item", "target": "support"}]},
]

class Generator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []
        self.learning_prompts = []
    def generate(self, prompt, *, timeout_s, schema=None, images=()):
        if schema is not None:
            if "failure_class" in schema["properties"]:
                self.prompts.append(prompt)
                return {"analysis": "test diagnosis", "code": next(self.responses), "failure_class": ""}
            self.learning_prompts.append(prompt)
            if "findings" in schema["properties"]:
                return {"analysis": "test evidence", "findings": "No new transferable lesson in this fixture."}
            return {"analysis": "test no-op", "updates": [], "reason": "No new transferable lesson."}
        self.prompts.append(prompt)
        return {"analysis": "test", "code": next(self.responses)}

def store(tmp_path, method, *, iterations=2, trials=1):
    root = initialize(tmp_path / "campaign", TASKS)
    # Simulate an existing campaign with obsolete scheduling fields.
    meta = read_json(root / "campaign.json")
    meta.update(iterations=iterations, trials=trials, minimum_gain=1)
    write_json(root / "campaign.json", meta)
    return MethodStore(root, method)

def rollout(code, task, trial, folder):
    success = "open_gripper()" in code
    return {"success": success, "task_completed": success, "error": None, "tool_trace": [], "dry_run": False}

def recursia_baseline(s, code="def run_task(task):\n    open_gripper()\n"):
    experiment = Experiment(s, Generator([code]), rollout)
    for task in s.meta["tasks"]:
        if task["split"] == "source":
            experiment.source([task["id"]])

def test_exactly_six_tool_namespace():
    tools = {name: (lambda *a: None) for name in TOOLS}
    execute_policy("open_gripper()", TASKS[0], tools)
    with pytest.raises(ValueError, match="exactly"):
        execute_policy("open_gripper()", TASKS[0], {**tools, "get_robot_state": lambda: None})

@pytest.mark.parametrize("extra", ["get_robot_state", "verify_placement", "detect_object", "set_agent_result", "env", "APIS", "capture_scene"])
def test_hidden_capabilities_rejected(extra):
    with pytest.raises(ValueError):
        validate_policy(extra + "()")


@pytest.mark.parametrize("parameter", ["item", "candidate"])
def test_lambda_candidate_sorting_passes_recursia_check_and_executes(parameter):
    code = ("def run_task(task):\n"
            "    candidates = [(2, 'second'), (1, 'first')]\n"
            f"    candidates.sort(key=lambda {parameter}: {parameter}[0])\n"
            "    print(candidates[0][1])\n")
    validate_policy(code, stack=True)
    result = CodeExecutor(TASKS[0], {name: lambda *a: None for name in TOOLS}).run(code + "run_task(TASK)\n")
    assert result["ok"] and result["stdout"].strip() == "first"


def test_lambda_parameters_cover_argument_kinds_and_nested_closures():
    code = "f = lambda x, /, y=1, *args, z=2, **kwargs: (lambda inner: x + y + z + args[0] + kwargs['n'] + inner)(3)\n"
    validate_policy(code)
    result = CodeExecutor(TASKS[0], {name: lambda *a: None for name in TOOLS}).run(code + "print(f(4, 5, 6, z=7, n=8))\n")
    assert result["ok"] and result["stdout"].strip() == "33"


@pytest.mark.parametrize("code, missing", [
    ("f = lambda item: get_robot_state()", "get_robot_state"),
    ("f = lambda get_robot_state: get_robot_state()\nget_robot_state()", "get_robot_state"),
])
def test_lambda_does_not_expose_parameters_or_hidden_capabilities(code, missing):
    with pytest.raises(ValueError, match=f"undeclared policy capability: {missing}"):
        validate_policy(code)


@pytest.mark.parametrize("code", [
    "f = lambda item: item[0]\nprint(item)",
    "f = lambda item: item[0]\ng = lambda other: item[0]\ng(None)",
    "f = lambda item=item: item",
    "f = lambda *, item=item: item",
])
def test_missing_data_name_is_a_python_error_and_lambda_parameters_do_not_leak(code):
    result = CodeExecutor(TASKS[0], {name: lambda *a: None for name in TOOLS}).run(code)
    assert result["execution_status"] == "execution_error"
    assert "NameError" in result["stderr"]
    assert "undeclared policy capability" not in result["stderr"]


def test_globals_queries_current_episode_and_retains_successful_helper_motion_state():
    moves = []
    tools = {name: lambda *a: None for name in TOOLS}
    tools["goto_pose"] = lambda p, q: moves.append((p, q)) or {"success": True}
    executor = CodeExecutor(TASKS[0], tools)
    assert executor.run("""def move_and_save(p, q):
    global last_commanded_pose
    result = goto_pose(p, q)
    if result['success']:
        last_commanded_pose = (p, q)
""")["ok"]
    missing = executor.run("""present = 'last_commanded_pose' in globals()
try:
    saved = last_commanded_pose
except NameError:
    saved = None
""")
    assert missing["ok"] and executor._exec_globals["present"] is False
    assert executor._exec_globals["saved"] is None
    # The target name is created dynamically by a retained helper, rather than
    # assigned in this block. The whole block must reach Python execution.
    result = executor.run("move_and_save([.4, .1, .2], [0., 1., 0., 0.])\n"
                          "saved = last_commanded_pose\nqueried = globals()['last_commanded_pose']")
    assert result["ok"] and len(moves) == 1
    assert executor._exec_globals["saved"] == executor._exec_globals["queried"] == moves[0]
    other = CodeExecutor(TASKS[0], tools)
    assert other.run("saved = globals().get('last_commanded_pose')")["ok"]
    assert other._exec_globals["saved"] is None


def test_globals_is_read_only_and_excludes_host_state_and_function_locals():
    tools = {name: lambda *a: None for name in TOOLS}
    executor = CodeExecutor(TASKS[0], tools)
    assert executor.run("""def local_only():
    high = [1, 2, 3]
    return globals().get('high')
local_value = local_only()
keys = list(globals())
""")["ok"]
    assert executor._exec_globals["local_value"] is None
    assert "__builtins__" not in executor._exec_globals["keys"]
    assert "runtime" not in executor._exec_globals["keys"]
    rejected = executor.run("globals()['open_gripper'] = lambda: None")
    assert not rejected["ok"] and "TypeError" in rejected["stderr"]
    assert executor._exec_globals["open_gripper"] is tools["open_gripper"]
    error = executor.run("globals()['get_robot_state']()")
    assert not error["ok"] and "KeyError" in error["stderr"]


@pytest.mark.parametrize("fault_flag", [None, False, True])
def test_exception_wrapper_can_read_optional_hardware_fault_attribute(fault_flag):
    def failed_hand():
        error = RuntimeError("hand failed")
        if fault_flag is not None:
            error.hardware_fault = fault_flag
        raise error
    tools = {name: (lambda *a: None) for name in TOOLS}
    tools["close_gripper"] = failed_hand
    executor = CodeExecutor(TASKS[0], tools)
    code = """state = {'terminal_fault': False}
def call_tool(fn):
    try:
        return fn()
    except Exception as exc:
        state['terminal_fault'] = bool(getattr(exc, 'hardware_fault', False))
        raise
call_tool(close_gripper)
"""
    result = executor.run(code)
    assert result["execution_status"] == "execution_error"
    assert "RuntimeError: hand failed" in result["stderr"]
    assert "unavailable policy name" not in result["stderr"]
    assert executor._exec_globals["state"]["terminal_fault"] is bool(fault_flag)

def test_generic_stack_and_object_literal_guards():
    validate_policy("def run_task(task):\n    get_object_pose(task['objects'][0]['name'])\n", stack=True, forbidden_literals=["red", "mug"])
    with pytest.raises(ValueError, match="literal"):
        validate_policy("def run_task(task):\n    get_object_pose('red mug')\n", stack=True, forbidden_literals=["red", "mug"])
    with pytest.raises(ValueError, match="top-level"):
        validate_policy("def run_task(task):\n    pass\nopen_gripper()\n", stack=True)

@pytest.mark.parametrize("body", [
    '"""Place the mug using observed scene geometry."""\n    pass',
    "print('Scene geometry unavailable; grasp planning stopped.')",
    "print(f\"Scene geometry unavailable for {task['objects'][0]['name']}\")",
    "raise RuntimeError('Scene geometry unavailable; grasp planning stopped.')",
    "raise ValueError(f\"Missing scene geometry for {task['objects'][0]['name']}\")",
    "return {'success': False, 'reason': 'Scene geometry unavailable; grasp planning stopped.'}",
    "return {'message': 'Mug detection failed', 'note': 'Scene occluded'}",
    "print({'reason': 'Scene geometry unavailable'})",
])
def test_recursia_allows_diagnostic_text_without_task_specialization(body):
    validate_policy("def run_task(task):\n    " + body + "\n", stack=True,
                    forbidden_literals=["scene", "mug"])


@pytest.mark.parametrize("body", [
    "get_object_pose('mug')",
    "name = 'mug'\n    get_object_pose(name)",
    "if task['objects'][0]['name'] == 'mug':\n        open_gripper()",
    "if task['id'] == 'S2':\n        open_gripper()",
    "print(get_object_pose('mug'))",
    "print(f\"Scene geometry: {get_object_pose('mug')}\")",
    "return {'reason': get_object_pose('mug')}",
    "raise RuntimeError(get_object_pose('mug'))",
    "return {'reason': f\"Scene geometry: {get_object_pose('mug')}\"}",
    "return {'object': 'mug'}",
])
def test_recursia_diagnostics_do_not_exempt_hardcoded_tool_or_branch_inputs(body):
    with pytest.raises(ValueError, match=r"literal at line \d+"):
        validate_policy("def run_task(task):\n    " + body + "\n", stack=True,
                        forbidden_literals=["scene", "mug", "S2"])

def test_recursia_allows_messages_forwarded_through_diagnostic_helpers():
    code = """def report(reason, released=False):
    return {'success': False, 'released': released, 'reason': reason}
def outer(message):
    return report(reason=message)
def run_task(task):
    return outer('Scene geometry unavailable; grasp planning stopped.')
"""
    validate_policy(code, stack=True, forbidden_literals=["scene", "mug"])


@pytest.mark.parametrize("use", [
    "get_object_pose(reason)",
    "if reason == task['objects'][0]['name']:\n        open_gripper()",
])
def test_recursia_message_helpers_still_check_parameters_used_for_actions(use):
    code = ("def report(reason):\n    " + use
            + "\n    return {'reason': reason}\ndef run_task(task):\n    return report('mug')\n")
    with pytest.raises(ValueError, match="literal"):
        validate_policy(code, stack=True, forbidden_literals=["mug"])

def test_stores_start_empty_without_previous_eraser_seeds(tmp_path):
    s = store(tmp_path, "aspire")
    assert s.knowledge == MethodStore(s.campaign, "sd-only").knowledge == ""
    s.save_learning("aspire only", {"task_id": "S1"})
    assert MethodStore(s.campaign, "sd-only").knowledge == ""
    assert "aspire only" not in MethodStore(s.campaign, "recursia").knowledge

def test_source_rejects_target_before_calling_model(tmp_path):
    s = store(tmp_path, "aspire")
    g = Generator([])
    with pytest.raises(ValueError, match="only source"):
        Experiment(s, g, rollout).source(["T1"])
    assert not g.prompts

def test_aspire_extracts_completed_task_functions_without_another_execution(tmp_path):
    s = store(tmp_path, "aspire", trials=3)
    code = "def release():\n    open_gripper()\nrelease()\n"
    g = Generator([code])
    result = Experiment(s, g, rollout).source(["S1"])
    assert result["tasks"][0]["validated"] and s.state["source_rollouts"] == 1
    library = read_json(s.root / "skill_library.json")["skills"]
    assert library["release"]["occurrences"] == 1 and not library["release"]["promoted"]
    assert len(g.prompts) == 1


def test_aspire_does_not_retry_or_promote_failed_program(tmp_path):
    s = store(tmp_path, "aspire", iterations=5, trials=3)
    g = Generator(["print('failure')"])
    Experiment(s, g, rollout).source(["S1"])
    assert s.knowledge == "" and len(g.prompts) == 1 and s.state["source_rollouts"] == 1


def test_sd_only_summarizes_once_without_another_execution(tmp_path):
    s = store(tmp_path, "sd-only", iterations=5, trials=3)
    g = Generator(["print('failure')", "SD_ONLY_LESSON"])
    Experiment(s, g, rollout).source(["S1"])
    assert s.knowledge == "SD_ONLY_LESSON" and len(g.prompts) == 2
    assert [e["kind"] for e in s.state["source_events"]] == ["execution", "summary"]
    assert s.state["source_rollouts"] == 1


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("method", ["aspire", "sd-only", "recursia"])
def test_finish_or_failure_never_schedules_another_execution(tmp_path, success, method):
    s = store(tmp_path, method, iterations=5, trials=3)
    code = "open_gripper()" if success else "print('failure')"
    responses = (["def run_task(task):\n    " + code, "def run_task(task):\n    " + code] if method == "recursia"
                 else [code, "SKILLS"] if method == "sd-only" else [code])
    g, calls = Generator(responses), []
    def execute(code, task, episode, folder):
        calls.append((task["id"], episode))
        return {**rollout(code, task, episode, folder), "model_finished": True}
    result = Experiment(s, g, execute).source(["S1"])
    assert len(calls) == s.state["source_rollouts"] == 1
    assert result["tasks"][0]["success"] is success
    if method == "recursia":
        assert result["tasks"][0]["model_finished"]
        assert result["source_schedule"] == "physical_recursia_chunk.v1"
        assert len(g.prompts) == 1 and result["round"] == "baseline"
        return
    assert result["tasks"][0]["model_finished"]
    assert result["source_schedule"] == "single_episode.v1"
    findings = read_json(Path(result["tasks"][0]["path"]) / "findings.json")
    assert findings["status"] == "completed" and "execution" in findings
    assert "trials" not in findings


@pytest.mark.parametrize("completed", [True, False])
def test_task_completion_is_distinct_from_error_free_execution(tmp_path, completed):
    s = store(tmp_path, "aspire")
    g = Generator(["print('source')"])
    packet = {"success": False, "task_completed": completed, "error": "feedback model failed",
              "execution_success": True, "final_code": "def release():\n    open_gripper()\nrelease()",
              "executed_code": "pass"}
    result = Experiment(s, g, lambda *args: packet.copy()).source(["S1"])
    assert result["tasks"][0]["validated"] is completed
    assert not result["tasks"][0]["success"]
    assert (s.root / "skill_library.json").exists() is completed


def test_old_worker_verification_signal_can_extract_skills(tmp_path):
    s = store(tmp_path, "aspire")
    code = "def release():\n    open_gripper()\nrelease()"
    def old_packet(*args):
        return {"success": False, "verification": {"success": True}, "executed_code": code}
    Experiment(s, Generator([code]), old_packet).source(["S1"])
    assert s.state["source_events"][0]["task_completed"]
    assert read_json(s.root / "skill_library.json")["skills"]["release"]["occurrences"] == 1


def test_no_skill_update_reported_merely_because_library_already_exists(tmp_path):
    s = store(tmp_path, "aspire")
    code = "def release():\n    open_gripper()\nrelease()"
    Experiment(s, Generator([code]), rollout).source(["S1"])
    before = (s.root / "skill_library.json").read_bytes()
    result = Experiment(s, Generator(["print('failed')"]), rollout).source(["S1"])
    assert not result["tasks"][0]["updated"]
    assert (s.root / "skill_library.json").read_bytes() == before


def test_empty_library_creation_is_not_a_skill_update(tmp_path):
    s = store(tmp_path, "aspire")
    result = Experiment(s, Generator(["open_gripper()"]), rollout).source(["S1"])
    assert not result["tasks"][0]["updated"]


def test_source_execution_checkpoint_survives_summary_budget_exhaustion(tmp_path):
    s = store(tmp_path, "sd-only")
    s.meta["model_call_cap"] = 1
    with pytest.raises(RolloutAbort, match="budget"):
        Experiment(s, Generator(["open_gripper()"]), rollout).source(["S1"])
    events = s.state["source_events"]
    assert len(events) == 1 and events[0]["task_completed"]
    folder = Path(events[0]["path"])
    assert read_json(folder / "learning.json") == events[0]
    assert read_json(folder / "findings.json")["status"] == "execution_finished"
    assert not (folder.parent / "source_result.json").exists()


@pytest.mark.parametrize("method", ["aspire", "sd-only", "recursia"])
def test_multiple_source_tasks_rejected_before_any_model_or_robot_call(tmp_path, method):
    s = store(tmp_path, method)
    g = Generator([])
    with pytest.raises(ValueError, match="one source task"):
        Experiment(s, g, lambda *a: pytest.fail("robot execution")).source(["S1", "S2"])
    assert not g.prompts and s.state["source_rollouts"] == 0


def test_sd_only_summary_uses_only_current_execution_and_own_skills(tmp_path):
    s = store(tmp_path, "sd-only")
    g = Generator(["print('first')", "LOCAL_SKILLS", "print('second')", "MORE_SKILLS"])
    e = Experiment(s, g, rollout)
    e.source(["S1"]); e.source(["S2"])
    assert s.state["source_rollouts"] == 2 and len(g.prompts) == 4
    assert "first" in g.prompts[1] and "second" in g.prompts[3]
    assert "first" not in g.prompts[3] and "LOCAL_SKILLS" in g.prompts[2]
    assert s.knowledge == "MORE_SKILLS"


@pytest.mark.parametrize("split", ["source", "target"])
@pytest.mark.parametrize("automatic,human", [(False, True), (True, False)])
def test_sd_only_human_score_controls_summary_and_final_result(tmp_path, split, automatic, human):
    s = store(tmp_path, "sd-only")
    if split == "target":
        s.save_learning("SOURCE_SKILLS", {"task_id": "S1"})
    before = s.state_path.read_bytes()
    g = Generator(["pass", "HUMAN_GRADED_SKILLS"])
    calls = []
    def evaluate(*args):
        calls.append("execute")
        return {"success": automatic, "task_completed": automatic, "model_finished": True,
                "verification": {"success": automatic}, "error": None}
    def judge(task, result):
        assert len(g.prompts) == 1  # score before any summary
        assert result["success"] is automatic
        calls.append("judge")
        return human
    e = Experiment(s, g, evaluate, judge=judge)
    packet = e.source(["S1"]) if split == "source" else e.evaluate("T1")
    summary = packet["tasks"][0] if split == "source" else packet
    assert summary["success"] is human and summary["automatic_success"] is automatic
    assert summary["scored_by"] == "human" and calls == ["execute", "judge"]
    folder = Path(summary["path"] if split == "source" else packet["run_dir"]) / "execution"
    result = read_json(folder / "result.json")
    assert result["success"] is human and result["task_completed"] is human
    assert result["verification"]["success"] is automatic
    assert read_json(folder / "manual_score.json")["success"] is human
    if split == "source":
        data = json.loads(g.prompts[1].split("\n")[-1])
        assert data["source_execution"]["success"] is human
        assert "Final task success is the human score" in g.prompts[1]
        assert s.state["source_events"][0]["success"] is human
        assert s.state["source_events"][0]["scored_by"] == "human"
        assert s.knowledge == "HUMAN_GRADED_SKILLS"
    else:
        assert len(g.prompts) == 1 and s.state_path.read_bytes() == before
        assert packet["execution"]["success"] is human


@pytest.mark.parametrize("fault", ["hardware_fault", "unknown_worker_failure"])
def test_sd_only_manual_fault_score_preserves_diagnostics_and_still_summarizes(tmp_path, fault):
    s = store(tmp_path, "sd-only")
    g = Generator(["pass", "READ_ONLY_LESSON"])
    result = {"success": False, "task_completed": False, "error": "saved fault", fault: True}
    packet = Experiment(s, g, lambda *a: result, judge=lambda *a: True).source(["S1"])
    assert packet["tasks"][0]["success"] and s.state["source_rollouts"] == 1
    assert s.knowledge == "READ_ONLY_LESSON" and len(g.prompts) == 2
    assert result[fault] and result["error"] == "saved fault"
    assert not result["automatic_success"] and result["human_success"]


def test_sd_only_dry_run_does_not_request_human_score(tmp_path):
    s = store(tmp_path, "sd-only")
    result = Experiment(s, Generator(["pass", "PREVIEW"]),
                        lambda *a: {"success": True, "task_completed": True}, dry_run=True,
                        judge=lambda *a: pytest.fail("preview requested a score")).source(["S1"])
    assert not result["tasks"][0]["success"]


def test_sd_only_unsubmitted_manual_score_does_not_learn(tmp_path):
    s = store(tmp_path, "sd-only")
    g = Generator(["pass"])
    def interrupted(*args):
        raise RolloutAbort("no manual score")
    with pytest.raises(RolloutAbort, match="no manual score"):
        Experiment(s, g, rollout, judge=interrupted).source(["S1"])
    assert not s.state["source_events"] and s.knowledge == ""
    assert len(g.prompts) == 1
    executions = list((s.root / "source").glob("*/S1/execution/result.json"))
    assert len(executions) == 1 and not read_json(executions[0]).get("scored_by")


def test_manual_task_judge_reprompts_invalid_input_and_reports_closed_stdin(monkeypatch):
    from .cli import manual_task_judge
    console = SimpleNamespace(line=lambda *a: None)
    answers = iter(["1", "", " S "])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert manual_task_judge(TASKS[0], {}, console) is True
    def eof(*args):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(RolloutAbort, match="without a manual score"):
        manual_task_judge(TASKS[0], {}, console)


@pytest.mark.parametrize("split", ["source", "eval"])
def test_sd_only_cli_automatically_requests_task_end_human_score(tmp_path, monkeypatch, split):
    from . import cli
    s = store(tmp_path, "sd-only")
    if split == "eval":
        s.save_learning("SOURCE_SKILLS", {"task_id": "S1"})
    g = Generator(["pass", "HUMAN_SKILLS"])
    monkeypatch.setenv("OPENFORGE_ALLOW_PHYSICAL_MOTION", "1")
    monkeypatch.setattr(cli, "IsolatedGenerator", lambda *a: g)
    monkeypatch.setattr(cli, "PhysicalEvaluator", lambda *a: rollout)
    prompts = []
    def answer(prompt):
        assert len(g.prompts) == 1
        prompts.append(prompt)
        return "s"
    monkeypatch.setattr("builtins.input", answer)
    assert cli.main([split, "--campaign", str(s.campaign), "--method", "sd-only",
                     "--task", "S1" if split == "source" else "T1"]) == 0
    assert len(prompts) == 1 and "enter s for success" in prompts[0]
    path = next((s.root / ("source" if split == "source" else "eval")).glob(
        "*/S1/execution/manual_score.json" if split == "source" else "*/execution/manual_score.json"))
    assert read_json(path)["success"]


def test_recursia_records_baseline_without_saving_online_repair_code(tmp_path):
    s = store(tmp_path, "recursia")
    module = "def run_task(task):\n    open_gripper()\n"
    repaired = "def run_task(task):\n    open_gripper()\n    print('repaired')\n"
    g, calls = Generator([module, repaired]), []
    def evaluate(code, task, episode, folder):
        calls.append(code)
        return {"success": True, "task_completed": True, "model_finished": True,
                "executed_code": code + "\nprint('online correction')"}
    result = Experiment(s, g, evaluate).source(["S1"])
    assert len(calls) == 1 and calls[0] == module + "\nrun_task(TASK)\n"
    assert len(g.prompts) == 1 and s.knowledge == module
    assert s.state["source_events"][-1]["kind"] == "panel_execution"
    assert not result["tasks"][0]["updated"]


def test_invalid_recursia_bootstrap_preserves_previous_module_without_execution(tmp_path):
    s = store(tmp_path, "recursia")
    before = s.knowledge
    g = Generator(["open_gripper()"])
    with pytest.raises(ValueError, match="run_task"):
        Experiment(s, g, rollout).source(["S1"])
    assert s.state["source_rollouts"] == 0 and s.knowledge == before

@pytest.mark.parametrize("method", ["aspire", "sd-only"])
def test_target_fresh_generation_keeps_source_frozen_and_histories_isolated(tmp_path, method):
    s = store(tmp_path, method)
    s.save_learning("FROZEN_SOURCE_ONLY", {"task_id": "S1"})
    before = s.state_path.read_bytes()
    MethodStore(s.campaign, "recursia").save_learning("OTHER_METHOD_SECRET", {"task_id": "S1"})
    g = Generator(["print('TARGET_FAILURE_SECRET')"] * 5)
    e = Experiment(s, g, rollout)
    result = e.evaluate("T1")
    assert not result["success"] and result["programs_generated"] == 1
    assert len(g.prompts) == 1 and s.state_path.read_bytes() == before
    repeated = e.evaluate("T1")
    assert repeated["programs_generated"] == 1 and len(g.prompts) == 2
    e.evaluate("T2")
    assert all("TARGET_FAILURE_SECRET" not in p and "OTHER_METHOD_SECRET" not in p for p in g.prompts)
    assert s.knowledge == "FROZEN_SOURCE_ONLY"
    assert not s.state["frozen"]
    assert read_json(Path(result["run_dir"]) / "snapshot.json")["content"] == "FROZEN_SOURCE_ONLY"
    assert len(list((s.root / "frozen_history").glob("*.json"))) == 3


@pytest.mark.parametrize("error", [RuntimeError("generation failed"), KeyboardInterrupt()])
def test_target_unfreezes_after_exception_or_interrupt_and_source_can_resume(tmp_path, error):
    s = store(tmp_path, "sd-only")
    s.save_learning("source skills", {"task_id": "S1"})
    before = s.state_path.read_bytes()
    class FailingGenerator:
        def generate(self, *args, **kwargs):
            assert s.state["frozen"]
            with pytest.raises(ValueError, match="frozen"):
                s.save_learning("target leak", {"task_id": "T1"})
            raise error
    with pytest.raises(type(error)):
        Experiment(s, FailingGenerator(), rollout).evaluate("T1")
    assert s.state_path.read_bytes() == before
    assert not (s.root / "frozen.json").exists()
    assert len(list((s.root / "frozen_history").glob("*.json"))) == 1
    s.save_learning("next source skills", {"task_id": "S2"})
    assert s.knowledge == "next source skills"


def test_target_also_releases_an_existing_manual_freeze(tmp_path):
    s = store(tmp_path, "sd-only")
    s.save_learning("source skills", {"task_id": "S1"})
    snapshot = s.freeze()
    result = Experiment(s, Generator(["open_gripper()"]), rollout).evaluate("T1")
    assert result["snapshot_sha256"] == snapshot["sha256"]
    assert not s.state["frozen"]
    assert read_json(Path(result["run_dir"]) / "snapshot.json") == snapshot


def test_recursia_target_executes_frozen_program_without_proposer(tmp_path):
    s = store(tmp_path, "recursia")
    recursia_baseline(s)
    assert s.freeze()["source_task_ids"] == ["S1", "S2"]
    g = Generator([])
    result = Experiment(s, g, rollout).evaluate("T1")
    assert result["success"] and not g.prompts

def test_snapshot_tampering_is_detected(tmp_path):
    s = store(tmp_path, "sd-only")
    s.save_learning("original", {"task_id": "S1"})
    s.freeze()
    s.knowledge_path.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        s.snapshot()

def test_failed_target_generation_does_not_poison_next_fresh_trial(tmp_path):
    s = store(tmp_path, "sd-only")
    s.save_learning("source only", {"task_id": "S1"})
    s.freeze()
    g = Generator(["get_robot_state()", "open_gripper()"])
    def checked_rollout(code, task, episode, folder):
        tools = {name: lambda *a: None for name in TOOLS}
        execution = CodeExecutor(task, tools).run(code)
        return {"success": execution["ok"], "error": execution["stderr"]}
    e = Experiment(s, g, checked_rollout)
    assert not e.evaluate("T1")["success"]
    assert e.evaluate("T1")["success"]
    assert len(g.prompts) == 2


def test_frozen_skill_library_tampering_is_detected(tmp_path):
    s = store(tmp_path, "aspire")
    s.save_learning("source only", {"task_id": "S1"})
    write_json(s.root / "skill_library.json", {"skills": {}})
    s.freeze()
    write_json(s.root / "skill_library.json", {"skills": {"tampered": {}}})
    with pytest.raises(ValueError, match="skill library hash"):
        s.snapshot()


def test_dry_run_does_not_validate_or_promote_skills(tmp_path):
    s = store(tmp_path, "aspire", iterations=1)
    Experiment(s, Generator(["open_gripper()"]), rollout, dry_run=True).source(["S1"])
    assert s.knowledge == "" and not s.state["source_events"][0]["validated"]
    s.freeze()
    with pytest.raises(ValueError, match="preview"):
        Experiment(s, Generator([]), rollout).evaluate("T1")

@pytest.mark.parametrize("method", ["aspire", "sd-only"])
def test_worker_fault_finishes_motion_then_learns_without_another_execution(tmp_path, method):
    s = store(tmp_path, method)
    summary = "def run_task(task):\n    open_gripper()" if method == "recursia" else "Recorded failure lesson."
    g = Generator(["open_gripper()", summary])
    calls = []
    def fault(code, task, episode, folder):
        calls.append(episode)
        write_json(folder / "robot_fault.json", {"current_errors": ["self_collision_avoidance_violation"]})
        return {"success": False, "hardware_fault": True, "error": "recovery required"}
    result = Experiment(s, g, fault).source(["S1"])
    packet = result["tasks"][0]
    assert calls == ["source"] and packet["forced_finish"] and not packet["model_finished"]
    assert packet["stop_reason"] == "hardware_fault" and not packet["success"]
    prompts = g.learning_prompts if method == "aspire" else g.prompts[1:]
    assert prompts and "self_collision_avoidance_violation" in prompts[0]
    assert s.state["source_rollouts"] == 1 and not packet["validated"]
    assert (Path(result["run_dir"]) / "source_result.json").is_file()


@pytest.mark.parametrize("method", ["aspire", "sd-only", "recursia"])
def test_target_fault_finishes_without_learning_or_another_execution(tmp_path, method):
    s = store(tmp_path, method)
    if method == "recursia":
        recursia_baseline(s)
    else:
        s.save_learning("frozen source", {"task_id": "S1"})
    s.freeze()
    before = {str(p.relative_to(s.root)): p.read_bytes() for p in s.root.rglob("*") if p.is_file()}
    g = Generator(["open_gripper()"])
    calls = []
    def fault(*args):
        calls.append("target")
        return {"success": False, "hardware_fault": True, "error": "recovery required"}
    result = Experiment(s, g, fault).evaluate("T1")
    assert result["forced_finish"] and not result["model_finished"] and not result["success"]
    assert calls == ["target"] and not g.learning_prompts
    assert len(g.prompts) == int(method != "recursia")
    assert all((s.root / name).read_bytes() == contents for name, contents in before.items()
               if name not in ("state.json", "frozen.json"))
    assert s.state == {**json.loads(before["state.json"]), "frozen": False}


def test_unknown_worker_failure_still_aborts_without_model_learning(tmp_path):
    s = store(tmp_path, "aspire")
    g = Generator(["open_gripper()"])
    with pytest.raises(RolloutAbort):
        Experiment(s, g, lambda *args: {"success": False, "unknown_worker_failure": True}).source(["S1"])
    assert len(g.prompts) == 1 and not g.learning_prompts


@pytest.mark.parametrize("method", ["aspire", "sd-only"])
def test_fault_finish_with_exhausted_budget_preserves_final_result_and_learning_error(tmp_path, method):
    s = store(tmp_path, method)
    s.meta['model_call_cap'] = 1
    g = Generator(['open_gripper()'])
    result = Experiment(s, g, lambda *args: {'success': False, 'hardware_fault': True,
                                           'error': 'recovery required'}).source(['S1'])
    assert result['tasks'][0]['forced_finish'] and result['tasks'][0]['learning_errors']
    assert s.state['source_model_calls'] == 1 and s.state['source_rollouts'] == 1
    assert not g.learning_prompts and len(g.prompts) == 1
    assert (Path(result['run_dir']) / 'source_result.json').is_file()

def test_source_budgets_are_enforced(tmp_path):
    s = store(tmp_path, "aspire")
    s.meta["model_call_cap"] = 150
    state = s.state
    state["source_model_calls"] = s.meta["model_call_cap"]
    write_json(s.state_path, state)
    g = Generator([])
    with pytest.raises(RolloutAbort, match="budget"):
        Experiment(s, g, rollout).source(["S1"])
    assert not g.prompts


@pytest.mark.parametrize("method", ["sd-only", "aspire"])
def test_default_source_has_no_cumulative_caps_and_keeps_counts(tmp_path, method):
    s = store(tmp_path, method)
    assert s.meta["model_call_cap"] is None and s.meta["rollout_cap"] is None
    state = s.state
    state.update(source_model_calls=150, source_vision_model_calls=150, source_rollouts=150)
    write_json(s.state_path, state)
    class Evaluator:
        def set_vision_budget(self, remaining):
            assert remaining is None
        def __call__(self, *args):
            return {**rollout(*args), "execution_model_calls": 2, "vision_model_calls": 3}
    g = Generator(["open_gripper()", "# No new lesson"])
    result = Experiment(s, g, Evaluator()).source(["S1"])
    assert result["tasks"][0]["success"]
    assert s.state["source_rollouts"] == 151
    assert s.state["source_model_calls"] > 152
    assert s.state["source_vision_model_calls"] == 153


def test_unlimited_model_budget_counts_action_and_vision(tmp_path):
    from .trial_adapter import ModelBudget
    raw = SimpleNamespace(generate=lambda: "action",
                          evaluate_placement=lambda: "judge")
    budget = ModelBudget(raw, None, tmp_path)
    budget.calls = 150
    assert budget.generate() == "action"
    assert budget.evaluate_placement() == "judge"
    assert budget.evaluate_placement() == "judge"
    assert read_json(tmp_path / "model_calls.json") == {"calls": 153, "action_calls": 1}


def test_new_campaign_has_no_repeat_settings(tmp_path):
    root = initialize(tmp_path / "campaign", TASKS)
    meta = read_json(root / "campaign.json")
    assert meta["execution_schedule"] == "single_episode.v1"
    assert not {"trials", "iterations", "minimum_gain"} & set(meta)


@pytest.mark.parametrize("command", ["init", "eval"])
def test_cli_no_longer_accepts_trial_count(tmp_path, command):
    from .cli import main
    arguments = [command, "--campaign", str(tmp_path / "campaign"), "--trials", "3"]
    if command == "eval":
        arguments += ["--method", "sd-only", "--task", "T1"]
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2

def test_manifest_and_minimal_prompt():
    root = Path(__file__).resolve().parents[3]
    tasks = validate_tasks(read_json(root / "config/rsi_tasks.json")["tasks"])
    assert sum(t["split"] == "source" for t in tasks) == 6
    assert sum(t["split"] == "target" for t in tasks) == 3
    text = policy_prompt(tasks[0])
    assert "confirmed_eraser_hold" not in text and "bottom_in_ee" not in text
    assert "get_robot_state" not in text and "verify_placement" not in text
    assert len(text) < 3000

def test_codex_generator_disables_host_tools_and_persistence(tmp_path):
    cmd = codex_command("codex", tmp_path, "fixed-model")
    assert "--ephemeral" in cmd and "--ignore-user-config" in cmd and cmd[-1] == "-"
    for feature in DISABLED_FEATURES:
        assert cmd[cmd.index(feature)-1] == "--disable"
    assert "resume" not in cmd
    assert "project_doc_max_bytes=0" in cmd
    assert 'service_tier="fast"' in cmd
    assert cmd[cmd.index('fast_mode')-1] == '--enable'
    assert 'model_reasoning_effort="medium"' in cmd

def test_claude_generator_disables_tools_mcp_and_user_skills(tmp_path):
    cmd = claude_command(tmp_path, {}, "fixed-model")
    assert cmd[cmd.index("--tools")+1] == ""
    assert cmd[cmd.index("--setting-sources")+1] == ""
    assert "--strict-mcp-config" in cmd and "--disable-slash-commands" in cmd
    assert "--no-session-persistence" in cmd

def test_primitive_tools_cannot_be_replaced():
    for code in ("def open_gripper():\n    pass\n", "goto_pose = print\n"):
        with pytest.raises(ValueError, match="primitive"):
            validate_policy(code)

def test_recursia_proposal_receives_complete_panel_execution_feedback(tmp_path):
    s = store(tmp_path, "recursia")
    g = Generator(["def run_task(task):\n    get_object_pose(task['objects'][0]['name'])\n", "def run_task(task):\n    open_gripper()\n"])
    def failed(*args):
        return {"success": False, "error": "ValueError: no object detected", "model_finished": True}
    experiment = Experiment(s, g, failed)
    experiment.source(["S1"])
    experiment.source(["S2"])
    from .recursia import RecursiaProtocol
    RecursiaProtocol(s, experiment).propose()
    assert "no object detected" in g.prompts[1]
    assert "get_object_pose(task['objects'][0]['name'])" in g.prompts[1]

def test_frozen_source_history_cannot_change(tmp_path):
    s = store(tmp_path, "aspire")
    s.save_learning("skills", {"task_id": "S1"})
    s.freeze()
    state = s.state
    state["source_events"].append({"task_id": "S2"})
    write_json(s.state_path, state)
    with pytest.raises(ValueError, match="history hash"):
        s.snapshot()

def test_runtime_wraps_perception_width_and_gripper_without_extra_tools(tmp_path):
    from cap.saved_scripts.fr3_placement.test_placement import station as station_fixture, scene, FakeSam
    cfg = station_fixture.__wrapped__(tmp_path)
    class FakeGraspNet:
        def predict(self, capture, mask):
            grasp = np.eye(4)
            grasp[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
            grasp[:3, 3] = [.5, 0, .2034]
            return {"grasps": grasp[None], "scores": np.array([.9]),
                    "widths": np.array([.04]), "contact_pts": np.array([[.5, 0, .1]])}
    runtime = FiveToolRuntime(cfg, tmp_path / "tools", task=TASKS[0], dry_run=True,
                             camera_fn=lambda serial: scene()[0], sam3=FakeSam(), graspnet=FakeGraspNet())
    tools = runtime.namespace()
    assert len(tools["get_object_pose"]("alternate description")) == 3
    position, quaternion = tools["sample_grasp_pose"]("item")
    assert tools["goto_pose"](position, quaternion) == {"success": True}
    assert tools["close_gripper"]()["success"]
    assert runtime.arm.held
    assert tools["open_gripper"]()["success"]
    assert not runtime.arm.held
    assert {t["tool"] for t in runtime.trace} <= set(TOOLS)
    runtime.close()

def test_worker_uses_hidden_evaluator_without_declaration_tool(tmp_path):
    from cap.saved_scripts.fr3_placement.test_placement import station as station_fixture
    # Reuse the hardware-free station fixture's constructor, not any host config.
    cfg = station_fixture.__wrapped__(tmp_path)
    code, task_file = tmp_path / "policy.py", tmp_path / "task.json"
    code.write_text("open_gripper()\n")
    write_json(task_file, TASKS[0])
    class FakeRuntime:
        hardware_fault = physical_motion_commanded = False
        trace = []
        def __init__(self, cfg, folder, **kwargs):
            self.run_dir, self.progress_path = folder, None
        def capture_feedback(self):
            return {"views": {v: {"rgb_path":str(tmp_path / (v+".png")), "captured_at":"now", "frame":v}
                    for v in ("front", "wrist")}, "capture_mode":"test"}
        def namespace(self): return {name: lambda *args: {"success": True} for name in TOOLS}
        def evaluate_goal(self): return {"success": True, "fresh": True}
        def close(self): pass
    args = SimpleNamespace(code=code, task_file=task_file, config=cfg.path, run_dir=tmp_path / "worker",
                           output=tmp_path / "result.json", dry_run=False, snapshot=None, progress_log=None)
    generator = SimpleNamespace(generate=lambda *a, **kw: {"analysis":"done", "response":"FINISH"})
    assert run_worker(args, runtime_factory=FakeRuntime, decision_generator=generator)["success"]

@pytest.mark.parametrize("trial", ["dev01", "dev02", "eval01", "target"])
@pytest.mark.parametrize("fixed_policy", [False, True])
def test_physical_rollout_starts_without_stdin_confirmation(tmp_path, monkeypatch, trial, fixed_policy):
    from . import cli
    monkeypatch.setattr(cli, "REAL_ROOT", tmp_path)
    monkeypatch.setenv("OPENFORGE_ALLOW_PHYSICAL_MOTION", "1")
    def reject_input(*args, **kwargs):
        raise AssertionError("rollout must not wait for manual confirmation")
    monkeypatch.setattr("builtins.input", reject_input)
    folder = tmp_path / "rollout"
    folder.mkdir()
    write_json(folder / "execution_mode.json", {"fixed_policy": fixed_policy})
    calls = []
    def worker(command, run_dir, timeout, console):
        calls.append(command)
        write_json(run_dir / "result.json", {"success": True, "vision_model_calls": 0})
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(cli, "_run_worker_live", worker)
    # No noninteractive argument: default execution must work even with closed stdin.
    args = SimpleNamespace(dry_run=False, config=tmp_path / "station.yaml",
                           timeout_s=300, worker_timeout_s=900)
    task = TASKS[2] if trial == "target" else TASKS[0]
    result = PhysicalEvaluator(args, None)("open_gripper()", task, trial, folder)
    assert result["success"] and len(calls) == 1
    assert "worker" in calls[0]
    assert "--vision-call-limit" not in calls[0]
    assert calls[0][calls[0].index("--turn-limit") + 1] == ("19" if trial == "target" else "10")
    assert ("--fixed-policy" in calls[0]) is fixed_policy

def test_volume_geometry_handles_non_planar_object():
    from cap.saved_scripts.fr3_placement.test_placement import scene
    capture, _, mask = scene()
    cfg = SimpleNamespace(camera_to_base=np.eye(4))
    obj = describe_volume(capture, mask, cfg, "test object")
    assert len(obj["extent_m"]) == 3 and obj["valid_depth_points"] >= 20

def test_runtime_namespace_is_only_six_names(tmp_path):
    # Construct without __init__; namespace itself must not access hardware.
    instance = FiveToolRuntime.__new__(FiveToolRuntime)
    assert set(instance.namespace()) == set(TOOLS)

def test_evaluator_does_not_count_one_detection_as_two_objects(tmp_path):
    from cap.saved_scripts.fr3_placement.test_placement import station as station_fixture, scene, FakeSam, FakeArm
    cfg = station_fixture.__wrapped__(tmp_path)
    task = {"objects": [{"name": name} for name in ("item A", "item B", "support")],
            "goals": [{"relation": "on", "object": name, "target": "support"} for name in ("item A", "item B")]}
    runtime = FiveToolRuntime(cfg, tmp_path / "evaluate", task=task,
                             camera_fn=lambda serial: scene()[0], sam3=FakeSam(), arm_factory=FakeArm)
    result = runtime.evaluate_goal()
    assert result["ambiguous_object_masks"]
    assert not result["success"]
    runtime.close()
