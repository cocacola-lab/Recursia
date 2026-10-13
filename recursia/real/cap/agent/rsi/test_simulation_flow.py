"""Feedback-loop regressions with fake hardware and real saved RGB pixels."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2

from .capabilities import validate_policy, skill_names
from .campaign import Experiment
from .cli import run_worker, TARGET_MAX_TURNS
from .prompts import TOOLS, SIM_MULTI_TURN_PROMPT, execution_context
from .runtime import PhysicalFault
from .simulation.executor import CodeExecutor
from .simulation.library import SkillLibrary
from .simulation.trial import run_trial
from .store import MethodStore, initialize, read_json, write_json
from .test_rsi import TASKS, store
from .trial_adapter import execute_trial, ModelBudget


class FakeRuntime:
    def __init__(self, folder, *, fault=False, camera_error=False, verifier_error=False):
        self.run_dir, self.progress_path = Path(folder), None
        self.trace = []
        self.hardware_fault = self.physical_motion_commanded = False
        self.moves, self.captures, self.verifications = [], 0, 0
        self.released = False
        self.fault, self.camera_error, self.verifier_error = fault, camera_error, verifier_error
        self.closed = False

    def namespace(self):
        def call(name):
            def tool(*args):
                self.trace.append({'tool': name, 'ok': True, 'result': {'success': True}})
                if name == 'goto_pose':
                    self.physical_motion_commanded = True
                    self.moves.append(args[0].copy())
                    if self.fault:
                        self.hardware_fault = True
                        raise PhysicalFault('controller reflex')
                if name == 'open_gripper':
                    self.released = True
                return {'success': True}
            return tool
        return {name: call(name) for name in TOOLS}

    def capture_feedback(self):
        if self.camera_error:
            raise RuntimeError('camera disconnected')
        self.captures += 1
        folder = self.run_dir / 'views' / str(self.captures)
        folder.mkdir(parents=True)
        views = {}
        for view in ('front', 'wrist'):
            path = folder / (view+'.png')
            assert cv2.imwrite(str(path), np.full((12,12,3), self.captures, dtype=np.uint8))
            views[view] = {'rgb_path':str(path), 'captured_at':str(self.captures), 'frame':view}
        return {'views':views, 'capture_mode':'synthetic'}

    def evaluate_goal(self):
        self.verifications += 1
        if self.verifier_error:
            raise ValueError('metric depth ambiguous')
        return {'success':self.released}

    def close(self):
        self.closed = True


class Decisions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, *, images, **kwargs):
        self.calls.append((prompt, images))
        assert len(images) == 2 and all(Path(p).is_file() for p in images)
        return {'analysis':'fake visual decision', 'response':next(self.responses)}


def test_copied_simulation_sources_match_recorded_provenance():
    import hashlib
    real = Path(__file__).resolve().parents[3]
    folder = real / 'cap/agent/rsi/simulation'
    manifest = read_json(folder/'SOURCE.json')
    for name, digest in manifest['standalone_checksums'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    tree = ast.parse((folder/'trial.py').read_text())
    for name, digest in manifest['standalone_parser_ast'].items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest() == digest


def test_runtime_error_is_repaired_in_same_scene_without_replaying_motion(tmp_path):
    runtime = FakeRuntime(tmp_path)
    model = Decisions(['REGENERATE\n```python\ngoto_pose(p + [.0,.0,.1], q)\nopen_gripper()\n```','FINISH'])
    initial = "p=np.array([.4,.1,.2]); q=np.array([0.,1.,0.,0.])\ngoto_pose(p,q)\nraise ValueError('pickup not verified')"
    result, verified = execute_trial(runtime,TASKS[0],initial,model,
        context=execution_context(TASKS[0],''),timeout_s=5)
    assert len(runtime.moves) == 2  # first command was NOT replayed
    assert np.allclose(runtime.moves[1], [.4,.1,.3])  # p,q survive exception
    assert verified['success'] and result['num_regenerations'] == 1
    assert result['num_code_blocks'] == 2 and result['num_finishes'] == 1
    assert 'ValueError: pickup not verified' in model.calls[0][0]
    assert result['turns'][0]['info_step']['execution_status'] == 'execution_error'
    assert 'EXECUTION ERROR: execution started; earlier statements and definitions may remain active.' in model.calls[0][0]
    assert result['turns'][1]['info_step']['execution_status'] == 'executed'
    assert 'EXECUTED SUCCESSFULLY' in model.calls[1][0]
    assert '"tool": "goto_pose"' in model.calls[0][0]
    for index, (_, paths) in enumerate(model.calls,1):
        assert cv2.imread(str(paths[0]))[0,0,0] == index
    assert read_json(tmp_path/'turns/turn_001/response.json')['response'].startswith('REGENERATE')
    assert (tmp_path/'executed_code.py').is_file()


def test_rejected_definitions_remain_marked_in_later_feedback(tmp_path):
    runtime = FakeRuntime(tmp_path)
    initial = "def helper():\n    open_gripper()\nhelper()\nprint(locals())"
    class Repair:
        def __init__(self):
            self.prompts = []
        def generate(self, prompt, **kwargs):
            self.prompts.append(prompt)
            assert 'The following code blocks were just executed:' not in prompt
            assert '# Code block 1\n# REJECTED BEFORE EXECUTION' in prompt
            assert 'this block created no variables or functions.' in prompt
            if len(self.prompts) == 1:
                assert not runtime.released
                response = "REGENERATE\n```python\ndef helper():\n    open_gripper()\n```"
            elif len(self.prompts) == 2:
                assert '# Code block 2\n# EXECUTED SUCCESSFULLY' in prompt
                assert not runtime.released
                response = "REGENERATE\n```python\nhelper()\n```"
            else:
                assert runtime.released
                response = 'FINISH'
            return {'analysis': 'repair rejected definition', 'response': response}
    model = Repair()
    result, verified = execute_trial(runtime, TASKS[0], initial, model,
        context=execution_context(TASKS[0], ''), timeout_s=5)
    assert result['num_code_blocks'] == 3 and verified['success']
    assert [t['info_step']['execution_status'] for t in result['turns']] == [
        'validation_rejected', 'executed', 'executed']
    assert read_json(tmp_path/'turns/turn_001/step.json')['info_step']['execution_status'] == 'validation_rejected'


@pytest.mark.parametrize('bad_statement', ['print(locals())', 'break'])
def test_validation_or_compile_rejection_executes_no_definitions(bad_statement):
    tools = {name: lambda *a: None for name in TOOLS}
    executor = CodeExecutor(TASKS[0], tools)
    rejected = executor.run("def helper():\n    return 42\n" + bad_statement)
    assert rejected['execution_status'] == 'validation_rejected' and not rejected['ok']
    missing = executor.run('helper()')
    assert 'undeclared policy capability: helper' in missing['stderr']


def test_same_variable_names_are_not_leaked_to_next_trial():
    tools = {name: lambda *a: None for name in TOOLS}
    one = CodeExecutor(TASKS[0],tools)
    assert one.run('p=np.array([1,2,3])')['ok']
    assert one.run('print(p.sum())')['stdout'].strip() == '6'
    assert not CodeExecutor(TASKS[0],tools).run('print(p)')['ok']


def test_handled_physical_fault_never_reaches_model_regeneration(tmp_path):
    runtime = FakeRuntime(tmp_path,fault=True)
    model = Decisions([])
    code = 'try:\n    goto_pose(np.zeros(3),np.array([0,1,0,0]))\nexcept Exception:\n    pass'
    result, verified = execute_trial(runtime,TASKS[0],code,model,context='task',timeout_s=5)
    assert result['terminated'] and runtime.hardware_fault
    assert not model.calls and not runtime.captures and verified is None
    assert result['num_finishes'] == 1 and result['forced_finish']
    assert result['all_responses'][-1]['origin'] == 'runtime'
    assert read_json(tmp_path/'turns/turn_001/step.json')['decision_origin'] == 'runtime'


def test_missing_camera_feedback_ends_trial_without_more_motion(tmp_path):
    runtime = FakeRuntime(tmp_path,camera_error=True)
    model = Decisions([])
    result,_ = execute_trial(runtime,TASKS[0],'pass',model,context='task',timeout_s=5)
    assert result['terminated'] and 'camera disconnected' in result['info_step']['stderr']
    assert not model.calls
    assert not result['forced_finish']


def test_worker_fault_closes_hardware_and_reports_runtime_finish_not_model_finish(tmp_path):
    from cap.saved_scripts.fr3_placement.test_placement import station
    cfg = station.__wrapped__(tmp_path)
    task_file, code_file = tmp_path/'task.json', tmp_path/'code.py'
    write_json(task_file, TASKS[0])
    code_file.write_text('goto_pose(np.array([.4,.1,.2]),np.array([0.,1.,0.,0.]))\nopen_gripper()')
    runtime = FakeRuntime(tmp_path/'execution', fault=True)
    args = SimpleNamespace(code=code_file, task_file=task_file, config=cfg.path,
        run_dir=runtime.run_dir, output=runtime.run_dir/'result.json', dry_run=False,
        snapshot=None, progress_log=None, vision_timeout_s=5, vision_call_limit=4, turn_limit=10)
    model = Decisions([])
    packet = run_worker(args, runtime_factory=lambda *a, **kw: runtime, decision_generator=model)
    assert packet['hardware_fault'] and packet['forced_finish'] and not packet['model_finished']
    assert packet['stop_reason'] == 'hardware_fault' and not packet['success']
    assert packet['execution_model_calls'] == 0 and len(runtime.moves) == 1
    assert runtime.closed and not runtime.released and not runtime.captures


def test_latched_fault_allows_observation_but_blocks_other_tools_even_if_program_catches_exception(monkeypatch):
    from .runtime import FiveToolRuntime
    runtime = FiveToolRuntime.__new__(FiveToolRuntime)
    runtime.hardware_fault, runtime.progress_path, runtime.trace = False, None, []
    events, dispatched = [], []
    monkeypatch.setattr(runtime, '_event', lambda **event: events.append(event))
    def fault(*args):
        dispatched.append('goto_pose')
        raise PhysicalFault('controller self-collision protection')
    monkeypatch.setattr(runtime, 'goto_pose', fault)
    monkeypatch.setattr(runtime, 'get_observation', lambda: {'views': {},
                                                           'capture_mode': 'synthetic'})
    for name in TOOLS:
        if name not in ('goto_pose', 'get_observation'):
            monkeypatch.setattr(runtime, name, lambda *args: pytest.fail('non-observation tool dispatched after fault'))
    tools = runtime.namespace()
    with pytest.raises(PhysicalFault):
        tools['goto_pose']([.4,.1,.2], [0.,1.,0.,0.])
    assert tools['get_observation']()['capture_mode'] == 'synthetic'
    for name in TOOLS:
        if name != 'get_observation':
            with pytest.raises(PhysicalFault, match='only get_observation'):
                tools[name]()
    assert dispatched == ['goto_pose']
    assert runtime.hardware_fault and not runtime.trace[0]['ok']
    assert 'self-collision' in runtime.trace[0]['error'] and events[0] == runtime.trace[0]
    assert runtime.trace[-1]['tool'] == 'get_observation' and runtime.trace[-1]['ok']


def test_finish_is_not_an_evaluator_success_and_metric_failure_is_not_motion_gate(tmp_path):
    runtime = FakeRuntime(tmp_path,verifier_error=True)
    model = Decisions(['REGENERATE\n```python\nopen_gripper()\n```','FINISH'])
    result,verified = execute_trial(runtime,TASKS[0],'pass',model,context='task',timeout_s=5)
    assert runtime.released and result['num_finishes'] == 1
    assert not verified['success'] and 'metric depth ambiguous' in verified['error']
    assert not result['terminated']


def test_turn_budget_bounds_repeated_regeneration(tmp_path):
    runtime = FakeRuntime(tmp_path)
    model = Decisions(['REGENERATE\n```python\npass\n```']*20)
    result,_ = execute_trial(runtime,TASKS[0],'pass',model,context='task',timeout_s=5,limit=2)
    assert result['truncated'] and result['num_code_blocks'] == 3
    assert len(model.calls) == 3 and runtime.captures == 3


def test_target_budget_allows_exactly_twenty_turns(tmp_path):
    runtime = FakeRuntime(tmp_path)
    model = Decisions(['REGENERATE\n```python\npass\n```'] * 21)
    result, _ = execute_trial(runtime, TASKS[2], 'pass', model, context='task',
                              timeout_s=5, limit=TARGET_MAX_TURNS - 1)
    assert result['truncated'] and result['num_code_blocks'] == 20
    assert len(model.calls) == runtime.captures == 20
    assert not (tmp_path / 'turns/turn_021').exists()


def test_visual_and_action_queries_share_one_budget(tmp_path):
    calls=[]
    raw=SimpleNamespace(evaluate_placement=lambda *a,**kw:calls.append('vision'),
                        generate=lambda *a,**kw:calls.append('action'))
    model=ModelBudget(raw,2,tmp_path)
    model.evaluate_placement('pixels')
    model.generate('feedback')
    with pytest.raises(RuntimeError,match='budget'):
        model.generate('over budget')
    assert calls==['vision','action'] and model.action_calls==1
    assert read_json(tmp_path/'model_calls.json')['calls']==2


def test_successful_function_promotion_and_namespace_injection(tmp_path):
    library=SkillLibrary(tmp_path/'library.json')
    code='def release():\n    open_gripper()\n'
    library.extract_from_code(code,'S1'); library.extract_from_code(code,'S2')
    assert library.get_promoted_skills()=={'release':code.rstrip()}
    library.save()
    validate_policy('release()',known_names=skill_names(code))
    runtime=FakeRuntime(tmp_path/'run')
    result,verified=execute_trial(runtime,TASKS[0],'release()',Decisions(['FINISH']),
        context='learned release helper',timeout_s=5,skill_code=code)
    assert verified['success'] and result['num_code_blocks']==1
    with pytest.raises(ValueError,match='only functions'):
        skill_names(code+'open_gripper()\n')
    with pytest.raises(ValueError,match='injection time'):
        skill_names('def side_effect(arg=open_gripper()):\n    pass\n')


def test_successful_trials_update_copied_library_only_in_aspire_group(tmp_path):
    root=initialize(tmp_path/'campaign',TASKS)
    own=MethodStore(root,'aspire')
    class Initial:
        def generate(self,*args,**kwargs):
            if kwargs.get('schema'):
                if 'findings' in kwargs['schema']['properties']:
                    return {'analysis':'no lesson', 'findings':'No new generalizable lesson.'}
                return {'analysis':'no-op','updates':[],'reason':'No new generalizable lesson.'}
            return {'code':'def release():\n    open_gripper()\nrelease()'}
    def evaluate(code,*args):
        return {'success':True,'task_completed':True,'executed_code':code,'execution_model_calls':1}
    experiment=Experiment(own,Initial(),evaluate)
    experiment.source(['S1'])
    experiment.source(['S2'])
    library=read_json(own.root/'skill_library.json')['skills']
    assert library['release']['promoted'] and library['release']['occurrences']==2
    assert own.state['source_model_calls']==8  # initial + decision + findings + skill update per source
    assert 'release' in own.skill_code()
    assert not (root/'sd-only/skill_library.json').exists()
    assert MethodStore(root,'sd-only').knowledge==''


def test_worker_packet_reports_repaired_failure_as_success(tmp_path):
    from cap.saved_scripts.fr3_placement.test_placement import station
    cfg=station.__wrapped__(tmp_path)
    task_file,code_file=tmp_path/'task.json',tmp_path/'code.py'
    write_json(task_file,TASKS[0])
    code_file.write_text("p=np.array([.4,.1,.2]); q=np.array([0.,1.,0.,0.])\ngoto_pose(p,q)\nraise ValueError('need another observation')")
    runtime=FakeRuntime(tmp_path/'execution')
    args=SimpleNamespace(code=code_file,task_file=task_file,config=cfg.path,
        run_dir=runtime.run_dir,output=runtime.run_dir/'result.json',dry_run=False,
        snapshot=None,progress_log=None,vision_timeout_s=5,vision_call_limit=4,turn_limit=10)
    model=Decisions(['REGENERATE\n```python\ngoto_pose(p+[0,0,.1],q)\nopen_gripper()\n```','FINISH'])
    packet=run_worker(args,runtime_factory=lambda *a,**kw:runtime,decision_generator=model)
    assert packet['success'] and packet['error'] is None
    assert packet['task_completed'] and packet['execution_success'] and packet['model_finished']
    assert 'open_gripper' in packet['final_code']
    assert packet['execution_model_calls']==2 and packet['vision_model_calls']==0
    assert packet['num_regenerations']==1 and packet['num_code_blocks']==2
    assert len(runtime.moves)==2 and runtime.closed
    assert read_json(args.output)['success']


@pytest.mark.parametrize('method', ['aspire', 'sd-only'])
def test_campaign_runs_regenerate_then_finish_once_with_real_worker_loop(tmp_path, method):
    from cap.saved_scripts.fr3_placement.test_placement import station
    cfg=station.__wrapped__(tmp_path)
    own=store(tmp_path,method,iterations=5,trials=3)
    module="def run_task(task):\n    print('start')"
    repaired="def run_task(task):\n    open_gripper()"
    class Initial:
        calls=0
        def generate(self,*args,**kwargs):
            self.calls+=1
            if kwargs.get('schema'):
                if 'findings' in kwargs['schema']['properties']:
                    return {'analysis':'no lesson','findings':'No new generalizable lesson.'}
                return {'analysis':'no-op','updates':[],'reason':'No new generalizable lesson.'}
            return {'code': (module if self.calls==1 else repaired) if method=='recursia'
                    else "print('start')" if self.calls==1 else 'source skills'}
    initial=Initial()
    feedback=Decisions(['REGENERATE\n```python\nopen_gripper()\n```','FINISH'])
    runtimes=[]
    def evaluate(code, task, episode, folder):
        runtime=FakeRuntime(folder)
        runtimes.append(runtime)
        args=SimpleNamespace(code=folder/'code.py',task_file=folder/'task.json',config=cfg.path,
            run_dir=folder,output=folder/'result.json',dry_run=False,snapshot=None,
            progress_log=None,vision_timeout_s=5,vision_call_limit=6,turn_limit=10)
        return run_worker(args,runtime_factory=lambda *a,**kw:runtime,decision_generator=feedback)
    result=Experiment(own,initial,evaluate).source(['S1'])
    assert len(runtimes)==1 and runtimes[0].closed
    assert [t['tool'] for t in runtimes[0].trace]==['open_gripper']
    assert len(feedback.calls)==2 and result['tasks'][0]['model_finished']
    assert own.state['source_rollouts']==1
    assert initial.calls==(3 if method=='aspire' else 2)  # later calls only learn, never move


@pytest.mark.parametrize('camera_disconnect', [False, True])
def test_worker_completion_uses_latest_step_independently_of_model_error(tmp_path, camera_disconnect):
    from cap.saved_scripts.fr3_placement.test_placement import station
    cfg=station.__wrapped__(tmp_path)
    task_file,code_file=tmp_path/'task.json',tmp_path/'code.py'
    write_json(task_file,TASKS[0])
    code_file.write_text('open_gripper()')
    runtime=FakeRuntime(tmp_path/'execution')
    args=SimpleNamespace(code=code_file,task_file=task_file,config=cfg.path,
        run_dir=runtime.run_dir,output=runtime.run_dir/'result.json',dry_run=False,
        snapshot=None,progress_log=None,vision_timeout_s=5,vision_call_limit=4,turn_limit=10)
    class FailingFeedback:
        def generate(self, *a, **kw):
            if camera_disconnect:
                runtime.camera_error=True
                return {'analysis':'continue', 'response':'REGENERATE\n```python\npass\n```'}
            raise RuntimeError('model unavailable after task completion')
    packet=run_worker(args,runtime_factory=lambda *a,**kw:runtime,decision_generator=FailingFeedback())
    assert packet['execution_success'] and not packet['success']
    assert packet['task_completed'] is (not camera_disconnect)
    assert packet['verification']['success']  # prior metric cannot override a missing latest observation


def test_invalid_regenerated_capability_is_returned_to_model_without_execution(tmp_path):
    runtime=FakeRuntime(tmp_path)
    model=Decisions(['REGENERATE\n```python\nget_robot_state()\n```',
                     'REGENERATE\n```python\nopen_gripper()\n```','FINISH'])
    result,verified=execute_trial(runtime,TASKS[0],'pass',model,context='six tools',timeout_s=5)
    assert verified['success'] and result['num_regenerations']==2
    assert 'undeclared policy capability: get_robot_state' in model.calls[1][0]
    assert [t['tool'] for t in runtime.trace]==['open_gripper']


@pytest.mark.parametrize('code,expected', [
    ('open_gripper()\nif True print(1)', 'SyntaxError'),
    ('open_gripper()\nget_robot_state()', 'undeclared policy capability'),
    ('import os\nopen_gripper()', 'ValueError'),
    ('def open_gripper():\n    pass', 'policy cannot redefine'),
    ('x=y\ny=1', 'NameError'),
])
def test_initial_code_errors_reach_worker_feedback_before_any_robot_action(tmp_path, code, expected):
    from cap.saved_scripts.fr3_placement.test_placement import station
    cfg=station.__wrapped__(tmp_path)
    task_file,code_file=tmp_path/'task.json',tmp_path/'code.py'
    write_json(task_file,TASKS[0]); code_file.write_text(code)
    runtime=FakeRuntime(tmp_path/'execution')
    args=SimpleNamespace(code=code_file,task_file=task_file,config=cfg.path,
        run_dir=runtime.run_dir,output=runtime.run_dir/'result.json',dry_run=False,
        snapshot=None,progress_log=None,vision_timeout_s=5,vision_call_limit=4,turn_limit=10)
    class Model(Decisions):
        def generate(self,*a,**kw):
            if not self.calls:
                assert expected in a[0] and runtime.trace==[]
            return super().generate(*a,**kw)
    model=Model(['REGENERATE\n```python\nopen_gripper()\n```','FINISH'])
    packet=run_worker(args,runtime_factory=lambda *a,**kw:runtime,decision_generator=model)
    assert packet['success'] and packet['model_finished']
    assert packet['num_regenerations']==1 and len(model.calls)==2
    assert [t['tool'] for t in runtime.trace]==['open_gripper']
    assert runtime.closed


def test_invalid_source_skill_injection_is_feedback_not_motion_or_terminal_error(tmp_path):
    runtime=FakeRuntime(tmp_path)
    model=Decisions(['REGENERATE\n```python\nopen_gripper()\n```','FINISH'])
    result,verified=execute_trial(runtime,TASKS[0],'release()',model,context='source helper',timeout_s=5,
        skill_code='def release(arg=open_gripper()):\n    pass')
    assert 'source skill injection failed' in model.calls[0][0]
    assert verified['success'] and result['num_regenerations']==1
    assert [t['tool'] for t in runtime.trace]==['open_gripper']  # no default-argument side effect


@pytest.mark.parametrize('method', ['aspire','sd-only'])
@pytest.mark.parametrize('split', ['source','target'])
def test_campaign_initial_syntax_error_can_be_repaired_in_online_methods_and_splits(tmp_path, method, split):
    from cap.saved_scripts.fr3_placement.test_placement import station
    cfg=station.__wrapped__(tmp_path)
    own=store(tmp_path,method)
    bad="def run_task(task)\n    pass"
    if split=='target':
        own.save_learning(bad if method=='recursia' else 'source skills',{'task_id':'S1'})
        own.freeze()
    class Initial:
        calls=0
        def generate(self,*a,**kw):
            self.calls+=1
            schema=kw.get('schema')
            if schema:
                if 'findings' in schema['properties']:
                    return {'analysis':'observed syntax repair','findings':'The initial syntax error was repaired.'}
                return {'analysis':'no-op','updates':[],'reason':'No new generalizable pattern.'}
            return {'code':bad if self.calls==1 else 'def run_task(task):\n    open_gripper()' if method=='recursia' else 'source lesson'}
    feedback=Decisions(['REGENERATE\n```python\nopen_gripper()\n```','FINISH'])
    runtimes=[]
    def evaluate(code,task,episode,folder):
        runtime=FakeRuntime(folder); runtimes.append(runtime)
        args=SimpleNamespace(code=folder/'code.py',task_file=folder/'task.json',config=cfg.path,
            run_dir=folder,output=folder/'result.json',dry_run=False,snapshot=None,progress_log=None,
            vision_timeout_s=5,vision_call_limit=6,turn_limit=10)
        return run_worker(args,runtime_factory=lambda *a,**kw:runtime,decision_generator=feedback)
    e=Experiment(own,Initial(),evaluate)
    result=e.source(['S1']) if split=='source' else e.evaluate('T1')
    assert 'SyntaxError' in feedback.calls[0][0]
    assert len(runtimes)==1 and runtimes[0].closed and len(feedback.calls)==2
    assert [t['tool'] for t in runtimes[0].trace]==['open_gripper']
    assert result['tasks'][0]['model_finished'] if split=='source' else result['model_finished']


@pytest.mark.parametrize('provider', ['codex', 'claude'])
def test_feedback_transport_decodes_action_response_and_continues_trial(tmp_path, monkeypatch, provider):
    # Exercise the real decoder and transport, not a fake generate() method.
    import json
    from .generator import IsolatedGenerator
    responses=iter(['REGENERATE\n```python\ngoto_pose(p+[0,0,.1],q)\nopen_gripper()\n```','FINISH'])
    calls=[]
    def process(command, **kwargs):
        payload={'analysis':'actual transport decoding', 'response':next(responses)}
        calls.append(payload)
        if provider=='codex':
            schema=json.loads((Path(kwargs['cwd'])/'schema.json').read_text())
            assert schema['required']==['analysis','response']
            Path(command[command.index('--output-last-message')+1]).write_text(json.dumps(payload))
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        return SimpleNamespace(returncode=0,stdout=json.dumps({'structured_output':payload}),stderr='')
    monkeypatch.setattr('cap.agent.rsi.generator.subprocess.run',process)
    monkeypatch.setattr('cap.agent.rsi.generator._resolve_codex_executable',lambda:'mock-codex')
    runtime=FakeRuntime(tmp_path/'run')
    result,verified=execute_trial(runtime,TASKS[0],
        'p=np.array([.4,.1,.2]); q=np.array([0.,1.,0.,0.])',
        IsolatedGenerator(provider),context='same-trial feedback',timeout_s=5)
    assert verified['success'] and result['num_regenerations']==1
    assert result['model_error'] is None and len(calls)==2
    assert len(runtime.moves)==1 and np.allclose(runtime.moves[0],[.4,.1,.3])


@pytest.mark.parametrize('value', [{'analysis':'ok','response':None}, {'analysis':3,'response':'FINISH'}])
def test_feedback_decoder_rejects_wrong_field_types(value):
    import json
    from .generator import decode_response
    schema={'properties':{'analysis':{'type':'string'},'response':{'type':'string'}},
            'required':['analysis','response']}
    with pytest.raises(ValueError,match='multi-turn feedback fields'):
        decode_response(json.dumps(value),schema)
