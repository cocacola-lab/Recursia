"""Real local Git, fake models only: version isolation, rollback and preserved evidence."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from .campaign import Experiment
from .promotion import Promotion, library_files
from .store import MethodStore, digest, initialize, read_json, write_json
from .test_rsi import TASKS, Generator, rollout
from .test_learning import Learner, PATTERN
from .versions import EvolutionGit, manage_versions


def own(tmp_path, method='sd-only'):
    return MethodStore(initialize(tmp_path / 'campaign', TASKS), method)


@pytest.mark.parametrize('method', ['sd-only', 'aspire', 'recursia'])
def test_rollback_restores_learning_only_preserves_budgets_logs_and_other_agents(tmp_path, method):
    store = own(tmp_path, method)
    versions = EvolutionGit(store)
    first = versions.checkpoint('initial')['head']
    store.save_learning('version one', {'task_id': 'S1'})
    (store.root / 'skills').mkdir()
    (store.root / 'skills/grasp.md').write_text('grasp one')
    write_json(store.root / 'skill_library.json', {'skills': {}})
    expected = library_files(store)
    second = versions.checkpoint('one')['head']
    store.save_learning('version two', {'task_id': 'S2'})
    (store.root / 'skills/grasp.md').write_text('grasp two')
    (store.root / 'skills/transport.md').write_text('new transport')
    (store.root / 'source').mkdir()
    log = store.root / 'source/evidence.txt'
    log.write_text('immutable run evidence')
    state = store.state
    state.update(source_model_calls=9, source_vision_model_calls=3, source_rollouts=2)
    write_json(store.state_path, state)
    third = versions.checkpoint('two')['head']
    peers = {str(p): p.read_bytes() for method2 in ('sd-only', 'aspire', 'recursia') if method2 != method
             for p in (store.campaign / method2).rglob('*') if p.is_file()}
    result = manage_versions(store.campaign, method)
    assert result['before'] == third and result['after'] == second
    assert library_files(store) == expected and versions.head() == second
    assert store.state['knowledge_sha256'] == digest(store.context_knowledge())
    assert store.state['source_model_calls'] == 9 and store.state['source_vision_model_calls'] == 3
    assert store.state['source_rollouts'] == 2 and len(store.state['source_events']) == 3
    assert log.read_text() == 'immutable run evidence'
    assert all(Path(path).read_bytes() == contents for path, contents in peers.items())
    assert third.encode() in versions.git('show-ref')  # discarded version remains recoverable
    assert manage_versions(store.campaign, method)['after'] == first


def test_frozen_rollback_archives_snapshot_and_unfreezes_without_rewriting_eval_logs(tmp_path):
    store = own(tmp_path)
    versions = EvolutionGit(store)
    versions.checkpoint('initial')
    store.save_learning('source one', {'task_id': 'S1'})
    expected = versions.checkpoint('one')['head']
    store.save_learning('source two', {'task_id': 'S2'})
    versions.checkpoint('two')
    frozen = store.freeze()
    assert frozen['agent_git_commit'] == versions.head()
    frozen_bytes = (store.root / 'frozen.json').read_bytes()
    write_json(store.root / 'eval/result.json', {'snapshot_sha256': frozen['sha256']})
    result = manage_versions(store.campaign, 'sd-only')
    assert result['unfrozen'] and not store.state['frozen'] and result['after'] == expected
    assert not (store.root / 'frozen.json').exists()
    assert Path(result['archived_frozen']).read_bytes() == frozen_bytes
    assert read_json(store.root / 'eval/result.json')['snapshot_sha256'] == frozen['sha256']
    assert store.freeze()['content'] == 'source one'


def test_dry_run_writes_nothing_and_running_agent_cannot_be_rolled_back(tmp_path):
    store = own(tmp_path)
    versions = EvolutionGit(store)
    versions.checkpoint('initial')
    store.save_learning('one', {'task_id': 'S1'})
    versions.checkpoint('one')
    with store.locked():
        with pytest.raises(RuntimeError, match='active'):
            manage_versions(store.campaign, 'sd-only')
    before = {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob('*') if p.is_file()}
    result = manage_versions(store.campaign, 'sd-only', dry_run=True)
    assert result['dry_run'] and result['changed_files'] == ['skills.md']
    assert before == {str(p.relative_to(store.root)): p.read_bytes() for p in store.root.rglob('*') if p.is_file()}


def test_manual_changes_are_saved_before_rollback_and_initial_version_has_no_parent(tmp_path):
    store = own(tmp_path)
    versions = EvolutionGit(store)
    initial = versions.checkpoint('initial')['head']
    with pytest.raises(ValueError, match='only one'):
        manage_versions(store.campaign, 'sd-only')
    store.knowledge_path.write_text('manual learned content')
    result = manage_versions(store.campaign, 'sd-only')
    assert result['after'] == initial and store.knowledge == ''
    assert versions.files(result['before'])['skills.md'] == 'manual learned content'


@pytest.mark.parametrize('method', ['sd-only', 'recursia'])
def test_imports_saved_hash_verified_summary_and_ignores_unsaved_proposal(tmp_path, method):
    store = own(tmp_path, method)
    folder = store.root / 'source/run/S1/summary'
    write_json(folder / 'response.json', {'code': 'recorded reusable knowledge'})
    before = digest(store.knowledge)
    store.save_learning('recorded reusable knowledge', {'kind': 'summary', 'task_id': 'S1',
        'path': str(folder), 'learning_error': None, 'knowledge_before_sha256': before,
        'knowledge_after_sha256': digest('recorded reusable knowledge')})
    bad = store.root / 'source/run2/S1/summary'
    write_json(bad / 'response.json', {'code': 'unsaved proposal'})
    store.save_learning(store.knowledge, {'kind': 'summary', 'path': str(bad), 'learning_error': None,
                                       'knowledge_after_sha256': digest('different actual content')})
    result = manage_versions(store.campaign, method, action='init')
    assert len(result['versions']) == 2
    assert manage_versions(store.campaign, method)['changed_files'] == [store.knowledge_path.name]
    assert store.knowledge != 'recorded reusable knowledge'


def test_imports_aspire_promotion_snapshots_and_can_restore_before_first_update(tmp_path):
    store = own(tmp_path, 'aspire')
    folder = store.root / 'source/run/S1'
    folder.mkdir(parents=True)
    promotion = Promotion(store, 'S1', folder)
    promotion.apply([{'file': 'grasp.md', 'content': PATTERN}])
    promotion.finish(reason='source lesson')
    store.save_learning(store.knowledge, {'task_id': 'S1'})
    result = manage_versions(store.campaign, 'aspire', action='init')
    assert len(result['versions']) == 2
    manage_versions(store.campaign, 'aspire')
    assert store.skill_documents() == {}
    assert (promotion.folder / 'after/skills/grasp.md').read_text() == PATTERN


@pytest.mark.parametrize('method', ['aspire', 'sd-only', 'recursia'])
def test_source_automatically_records_one_version_per_changed_learning_and_target_is_readonly(tmp_path, method):
    store = own(tmp_path, method)
    model = (Learner([{'file': 'grasp.md', 'content': PATTERN}]) if method == 'aspire' else
             Generator(['open_gripper()', 'learned summary'] if method == 'sd-only' else
                       ['def run_task(task):\n    open_gripper()', 'def run_task(task):\n    open_gripper()']))
    packet = Experiment(store, model, rollout).source(['S1'])
    versions = EvolutionGit(store)
    assert packet['agent_version']['head'] == versions.head()
    assert len(versions.history()) == 2
    # Repeat identical learned contents: no duplicate evolution version.
    model = (Learner([], reason='unchanged lesson') if method == 'aspire' else
             Generator(['open_gripper()', 'learned summary'] if method == 'sd-only' else
                       ['def run_task(task):\n    open_gripper()', 'def run_task(task):\n    open_gripper()']))
    packet = Experiment(store, model, rollout).source(['S2'])
    assert not packet['agent_version']['changed'] and len(versions.history()) == 2
    store.freeze()
    before = {str(p.relative_to(versions.repo)): p.read_bytes() for p in versions.repo.rglob('*') if p.is_file()}
    Experiment(store, Generator(['open_gripper()']), rollout).evaluate('T1')
    assert before == {str(p.relative_to(versions.repo)): p.read_bytes() for p in versions.repo.rglob('*') if p.is_file()}


def test_failed_restore_recovers_files_state_and_git_head(tmp_path, monkeypatch):
    from . import versions as module
    store = own(tmp_path)
    versions = EvolutionGit(store)
    versions.checkpoint('initial')
    store.save_learning('one', {'task_id': 'S1'})
    (store.root / 'skills').mkdir()
    (store.root / 'skills/grasp.md').write_text('one')
    versions.checkpoint('one')
    store.save_learning('two', {'task_id': 'S2'})
    (store.root / 'skills/grasp.md').write_text('two')
    head = versions.checkpoint('two')['head']
    state, files = store.state_path.read_bytes(), library_files(store)
    replace = module.os.replace
    failures = []
    def interrupted(source, target):
        if Path(target) == store.root / 'skills/grasp.md' and not failures:
            failures.append(True)
            raise OSError('interrupted restore')
        return replace(source, target)
    monkeypatch.setattr(module.os, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        manage_versions(store.campaign, 'sd-only')
    assert library_files(store) == files and store.state_path.read_bytes() == state
    assert versions.head() == head


def test_rollback_script_preview_and_real_rollback_use_selected_agent_only(tmp_path):
    store = own(tmp_path)
    versions = EvolutionGit(store)
    initial = versions.checkpoint('initial')['head']
    store.save_learning('one', {'task_id': 'S1'})
    latest = versions.checkpoint('one')['head']
    command = [sys.executable, str(Path(__file__).resolve().parents[3] / 'tools/rsi_rollback_agent.py'),
               '--campaign', str(store.campaign), '--agent', 'sd-only']
    preview = subprocess.run(command + ['--dry-run'], capture_output=True, text=True, check=True)
    assert json.loads(preview.stdout)['after'] == initial and versions.head() == latest
    restored = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(restored.stdout)['after'] == initial and store.knowledge == ''
    assert not (store.campaign / 'aspire/.evolution.git').exists()
