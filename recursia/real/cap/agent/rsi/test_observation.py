"""Visual inputs are delivered to the model; observations cannot update target policies."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from cap.saved_scripts.fr3_placement.test_placement import station, scene, FakeSam, FakeArm
from cap.saved_scripts.fr3_runtime.camera import Capture
from .campaign import Experiment, RolloutAbort
from .generator import IsolatedGenerator, codex_command, decode_response
from .observation import capture_observation, image_paths, public_observation
from .simulation.executor import CodeExecutor
from .capabilities import validate_policy
from .prompts import TOOLS
from .runtime import FiveToolRuntime
from .store import initialize, MethodStore, read_json, write_json
from .test_rsi import TASKS, rollout


def execute_policy(code, task, tools, *, printer=print):
    validate_policy(code)
    result = CodeExecutor(task, tools).run(code)
    if not result["ok"]:
        raise ValueError(result["stderr"])


def configure_wrist(cfg):
    data = yaml.safe_load(cfg.hardware_config_path.read_text())
    data['hardware']['cameras']['wrist'] = {'serial': 'mock-wrist'}
    cfg.hardware_config_path.write_text(yaml.safe_dump(data))


def frame(serial):
    capture = scene()[0]
    return Capture(capture.rgb.copy(), capture.depth_m.copy(), capture.intrinsics.copy(), serial, 'synthetic')


def test_dual_capture_names_and_serials_are_not_interchangeable(station, tmp_path):
    configure_wrist(station)
    called = []
    def camera(serial):
        called.append(serial)
        return frame(serial)
    packet = capture_observation(station, tmp_path/'views', camera_fn=camera)
    assert called == ['mock-camera', 'mock-wrist']
    assert [p.parent.name for p in image_paths(packet)] == ['front', 'wrist']
    assert packet['views']['wrist']['T_base_camera_optical'] is None
    assert str(tmp_path) not in json.dumps(public_observation(packet))
    assert not packet['physical_motion_commanded']


def test_wrong_serial_does_not_become_second_camera(station, tmp_path):
    configure_wrist(station)
    with pytest.raises(ValueError, match='serial mismatch'):
        capture_observation(station, tmp_path/'views', camera_fn=lambda serial: frame('mock-camera'))


def test_front_only_snapshot_cannot_fake_wrist(station, tmp_path):
    configure_wrist(station)
    frame('mock-camera').save(tmp_path/'front_only')
    with pytest.raises(ValueError, match='front/ and wrist/'):
        capture_observation(station, tmp_path/'views', snapshot=tmp_path/'front_only',
                            camera_fn=lambda serial: pytest.fail('dry-run opened a camera'))


def test_dual_dry_run_never_opens_live_camera(station, tmp_path):
    configure_wrist(station)
    for view, serial in [('front','mock-camera'), ('wrist','mock-wrist')]:
        frame(serial).save(tmp_path/'snapshot'/view)
    packet = capture_observation(station, tmp_path/'views', snapshot=tmp_path/'snapshot',
                                 camera_fn=lambda serial: pytest.fail('dry-run opened a camera'))
    assert len(image_paths(packet)) == 2


def test_codex_receives_actual_image_files_and_medium_effort(tmp_path):
    paths = [tmp_path/'front.png', tmp_path/'wrist.png']
    for path in paths:
        path.write_bytes(b'FAKE_IMAGE_BYTES')
    seen = []
    def process(command, **kwargs):
        files = [Path(command[i+1]) for i,arg in enumerate(command) if arg == '--image']
        assert len(files) == 2 and all(p.read_bytes() == b'FAKE_IMAGE_BYTES' for p in files)
        assert all(p.parent == Path(kwargs['cwd']) for p in files)
        assert 'model_reasoning_effort="medium"' in command
        assert 'service_tier="fast"' in command
        assert command[command.index('fast_mode')-1] == '--enable'
        seen.append(kwargs['input'])
        Path(command[command.index('--output-last-message')+1]).write_text('{"analysis":"visible", "code":"pass"}')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    with patch('cap.agent.rsi.generator._resolve_codex_executable', return_value='codex'), \
         patch('cap.agent.rsi.generator.subprocess.run', side_effect=process):
        result = IsolatedGenerator('codex').generate('current views', timeout_s=5, images=paths)
    assert result['code'] == 'pass' and seen == ['current views']


def test_claude_receives_images_without_enabling_shell_tools(tmp_path):
    path = tmp_path/'frame.png'
    path.write_bytes(b'IMAGE_BYTES')
    def process(command, **kwargs):
        assert command[command.index('--tools')+1] == ''
        assert command[command.index('--input-format')+1] == 'stream-json'
        message = json.loads(kwargs['input'])['message']
        assert message['content'][0]['source']['media_type'] == 'image/png'
        assert message['content'][0]['source']['data'] == 'SU1BR0VfQllURVM='
        return SimpleNamespace(returncode=0, stdout='{"structured_output":{"analysis":"seen","code":"pass"}}', stderr='')
    with patch('cap.agent.rsi.generator.subprocess.run', side_effect=process):
        assert IsolatedGenerator('claude').generate('view', timeout_s=5, images=[path])['code'] == 'pass'


@pytest.mark.parametrize('hardware_fault', [False, True])
def test_runtime_observation_returns_arrays_without_model_or_arm(station, tmp_path, hardware_fault):
    configure_wrist(station)
    class Vision:
        def observe(self, packet, objects, **kwargs):
            pytest.fail('get_observation invoked a visual model')
    runtime = FiveToolRuntime(station, tmp_path/'run', task=TASKS[0], camera_fn=frame,
        vision_generator=Vision(), sam3=FakeSam(), arm_factory=lambda cfg: pytest.fail('observation connected to arm'))
    runtime.hardware_fault = hardware_fault
    result = runtime.namespace()['get_observation']()
    assert result['views']['front']['images']['rgb'].shape == (240,320,3)
    assert result['views']['wrist']['images']['depth'].shape == (240,320)
    assert set(result) == {'views', 'capture_mode'}
    assert runtime.vision_model_calls == 0
    assert not list(runtime.run_dir.rglob('vision.json'))
    assert 'images' not in runtime.trace[-1]['result']['views']['front']
    assert not runtime.physical_motion_commanded
    assert runtime.hardware_fault == hardware_fault
    runtime.close()


@pytest.mark.parametrize('has_model', [False, True])
def test_observation_does_not_require_model_or_model_budget(station, tmp_path, has_model):
    configure_wrist(station)
    vision = SimpleNamespace(observe=lambda *a,**k: pytest.fail('budget exhausted'))
    runtime = FiveToolRuntime(station, tmp_path/'run', task=TASKS[0], camera_fn=frame,
        vision_generator=vision if has_model else None, vision_call_limit=0, sam3=FakeSam())
    assert runtime.get_observation()['views']['front']['images']['rgb'].shape == (240,320,3)
    assert runtime.vision_model_calls == 0
    runtime.close()


@pytest.mark.parametrize('method', ['aspire','sd-only','recursia'])
def test_initial_images_reach_every_method_generator(tmp_path, method):
    root = initialize(tmp_path/'campaign', TASKS)
    store = MethodStore(root, method)
    observed, generated = [], []
    class Evaluator:
        def observe(self, task, folder):
            observed.append(task['id'])
            return {'views': {v:{'rgb_path':str(tmp_path/f'{v}.png'),'captured_at':'now','frame':v}
                             for v in ('front','wrist')}, 'capture_mode':'sequential'}
    class Generator:
        def generate(self, prompt, *, images, **kwargs):
            generated.append(images)
            assert str(tmp_path) not in prompt
            return {'code':'pass'}
    code = Experiment(store, Generator(), Evaluator())._generate(tmp_path/'generate', 'task only',
        source=True, observation_task=TASKS[0])
    assert code == 'pass' and observed == ['S1'] and len(generated[0]) == 2


def test_target_current_observation_keeps_learning_frozen(tmp_path):
    root = initialize(tmp_path/'campaign', TASKS)
    store = MethodStore(root,'sd-only')
    store.save_learning('source skills', {'task_id':'S1'})
    before = store.state_path.read_bytes()
    calls = []
    class Generator:
        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            return {'code':'obs = get_observation()\nprint(obs["capture_mode"])'}
    def evaluate(code, task, trial, folder):
        tools = {name:(lambda: None) for name in TOOLS}
        tools['get_observation'] = lambda: {'views':{}, 'capture_mode':'sequential'}
        execute_policy(code, task, tools, printer=lambda *a: None)
        return {'success':False, 'error':None, 'vision_model_calls':0}
    experiment = Experiment(store, Generator(), evaluate)
    result = experiment.evaluate('T1')
    repeated = experiment.evaluate('T1')
    assert len(calls) == 2 and repeated['programs_generated'] == 1
    assert result['vision_model_calls'] == 0
    assert store.state_path.read_bytes() == before and store.knowledge == 'source skills'


def test_source_visual_calls_are_counted_against_model_budget(tmp_path):
    root = initialize(tmp_path/'campaign', TASKS, model_call_cap=2)
    store = MethodStore(root,'sd-only')
    def evaluate(*args):
        return {'success':False, 'vision_model_calls':2}
    e = Experiment(store, None, evaluate)
    e._rollout('pass',TASKS[0],'dev01',tmp_path/'rollout',source=True)
    assert store.state['source_vision_model_calls'] == 2
    with pytest.raises(RolloutAbort,match='budget'):
        e._generate(tmp_path/'generate','summary',source=True)
