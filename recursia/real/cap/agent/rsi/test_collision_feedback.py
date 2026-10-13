"""Collision feedback using the actual RSI runtime with fake arm/cameras."""
from types import SimpleNamespace

import numpy as np
import pytest

from cap.saved_scripts.fr3_placement.test_placement import station, FakeArm, FakeVideo
from .runtime import FiveToolRuntime, RecoverableCollision, collision_fault
from .test_rsi import TASKS
from .test_simulation_flow import FakeRuntime, Decisions
from .trial_adapter import execute_trial
from .cli import run_worker


@pytest.mark.parametrize("codes,expected", [
    (["joint_reflex"], True),
    (["cartesian_reflex"], True),
    (["self_collision_avoidance_violation"], True),
    (["joint_reflex", "communication_constraints_violation"], False),
    (["power_limit_violation"], False),
    ([], False),
    ("unknown", False),
])
def test_collision_classifier_requires_specific_evidence(codes, expected):
    assert collision_fault({"current_errors": codes, "last_motion_errors": []}) is expected


def test_empty_error_lists_require_reflex_and_collision_flags():
    state = {"current_errors": "[]", "last_motion_errors": "[]", "robot_mode": "RobotMode.Reflex",
             "joint_collision": [0, 0, 0, 2, 0, 0, 0]}
    assert collision_fault(state)
    assert not collision_fault({**state, "robot_mode": "RobotMode.Idle"})
    assert not collision_fault({**state, "joint_collision": [0] * 7})
    assert not collision_fault({**state, "current_errors": '["communication_constraints_violation"]'})


def runtime_with_collision(cfg, folder, monkeypatch, *, kind="joint_reflex"):
    class Arm(FakeArm):
        errors = []
        def __init__(self, cfg):
            super().__init__(cfg)
            self.commands = []
        def state(self):
            return {**super().state(), "has_errors": bool(self.errors),
                    "current_errors": self.errors, "last_motion_errors": self.errors,
                    "robot_mode": "RobotMode.Reflex" if self.errors else "RobotMode.Idle"}
        def move_pose(self, step):
            self.commands.append(step)
            if len(self.commands) == 1:
                self.errors = [] if kind == "force" else [kind]
                error = "external force change 31.0 N exceeded 30.0 N" if kind == "force" else "FR3 error during agent_move"
                self.diagnostics({"event": "failed", "failure_state": self.state(), "error": error})
                raise RuntimeError(error)
            return super().move_pose(step)
    arm = Arm(cfg)
    rt = FiveToolRuntime(cfg, folder, task=TASKS[0], arm_factory=lambda *a: arm,
                        recorder_factory=FakeVideo)
    rt.captures, rt.camera_error = 0, False
    monkeypatch.setattr(rt, "capture_feedback", lambda: FakeRuntime.capture_feedback(rt))
    monkeypatch.setattr(rt, "evaluate_goal", lambda: {"success": len(arm.commands) > 1})
    monkeypatch.setattr("cap.saved_scripts.fr3_placement.runtime.solve_joint_pose",
                        lambda *a: {"target_q": [0] * 7})
    return rt, arm


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("catch", [False, True])
def test_collision_reaches_model_and_only_new_command_runs_after_recovery(station, tmp_path, monkeypatch, restore, catch):
    rt, arm = runtime_with_collision(station, tmp_path, monkeypatch)
    class Model(Decisions):
        def generate(self, prompt, **kwargs):
            if not self.calls:
                assert "RecoverableCollision" in prompt
                assert len(arm.commands) == 1 and not rt.hardware_fault
                if restore:
                    arm.errors = []  # represent operator Desk Recovery; no runtime recovery command
            return super().generate(prompt, **kwargs)
    model = Model(['REGENERATE\n```python\ngoto_pose(p + np.array([0,0,.05]), q)\n```', 'FINISH'])
    initial = "p=np.array([.4,.1,.2]); q=np.array([0.,1.,0.,0.])\n"
    initial += ("try:\n    goto_pose(p,q)\nexcept Exception:\n    pass\n" if catch else "goto_pose(p,q)\n")
    result, verification = execute_trial(rt, TASKS[0], initial, model, context="six tools", timeout_s=5)
    assert not result['terminated'] and not result['forced_finish'] and not rt.hardware_fault
    assert len(model.calls) == 2 and rt.captures == 2
    assert len(arm.commands) == (2 if restore else 1)
    assert rt.collision_pending is (not restore)
    assert verification['success'] is restore
    if restore:
        assert arm.commands[1]['ee_xyz_base_m'] == [.4, .1, .25]
    else:
        assert "Recovery is required" in model.calls[1][0]
    assert rt.collision_faults and (tmp_path/'robot_fault.json').exists()
    assert result['turns'][0]['info_step']['recoverable_collisions']


def test_force_threshold_stop_reaches_model_without_controller_recovery(station, tmp_path, monkeypatch):
    rt, arm = runtime_with_collision(station, tmp_path, monkeypatch, kind="force")
    model = Decisions(['REGENERATE\n```python\ngoto_pose(np.array([.4,.1,.25]),np.array([0.,1.,0.,0.]))\n```', 'FINISH'])
    result, _ = execute_trial(rt, TASKS[0], 'goto_pose(np.array([.4,.1,.2]),np.array([0.,1.,0.,0.]))',
                              model, context="six tools", timeout_s=5)
    assert not result['terminated'] and len(arm.commands) == 2
    assert not rt.hardware_fault and not rt.collision_pending
    assert '31.0 N exceeded 30.0 N' in model.calls[0][0]


def test_noncollision_controller_error_still_forces_finish(station, tmp_path, monkeypatch):
    rt, arm = runtime_with_collision(station, tmp_path, monkeypatch, kind="communication_constraints_violation")
    model = Decisions([])
    result, _ = execute_trial(rt, TASKS[0], 'goto_pose(np.array([.4,.1,.2]),np.array([0.,1.,0.,0.]))',
                              model, context="six tools", timeout_s=5)
    assert result['terminated'] and result['forced_finish'] and rt.hardware_fault
    assert not model.calls and len(arm.commands) == 1


def test_worker_collision_keeps_feedback_and_records_pending_state(station, tmp_path, monkeypatch):
    rt, arm = runtime_with_collision(station, tmp_path/'execution', monkeypatch)
    code, task = tmp_path/'code.py', tmp_path/'task.json'
    code.write_text('goto_pose(np.array([.4,.1,.2]),np.array([0.,1.,0.,0.]))')
    from .store import write_json
    write_json(task, TASKS[0])
    args = SimpleNamespace(code=code, task_file=task, config=station.path, run_dir=rt.run_dir,
                           output=rt.run_dir/'result.json', dry_run=False, snapshot=None, progress_log=None)
    model = Decisions(['FINISH'])
    packet = run_worker(args, runtime_factory=lambda *a, **kw: rt, decision_generator=model)
    assert not packet['hardware_fault'] and not packet['forced_finish']
    assert packet['model_finished'] and packet['pending_controller_recovery']
    assert packet['collision_faults'] and len(model.calls) == 1 and len(arm.commands) == 1

