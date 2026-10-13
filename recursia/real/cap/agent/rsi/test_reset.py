"""Workspace cleanup cannot erase another method, follow links or race active learning."""
from pathlib import Path
import subprocess
import sys

import pytest

from .reset import clear_method
from .store import METHODS, MethodStore, initialize, initial_knowledge, initial_state, read_json, write_json
from .test_rsi import TASKS


def contents(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('method', METHODS)
def test_clear_only_selected_method_and_allow_source_learning_again(tmp_path, method):
    campaign = initialize(tmp_path / 'campaign', TASKS)
    selected = MethodStore(campaign, method)
    for name in METHODS:
        store = MethodStore(campaign, name)
        if name == 'recursia':
            from .test_rsi import recursia_baseline
            recursia_baseline(store)
        else:
            store.save_learning('learned ' + name, {'task_id': 'S1'})
        store.freeze()
        for relative in ('source/run/code.py', 'target_programs/T1/code.py', 'eval/T1/result.json', 'extra.txt'):
            p = store.root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('old ' + name)
    state = selected.state
    state.update(source_model_calls=5, source_vision_model_calls=3, source_rollouts=9,
                 regression_cases=[{'task_id': 'S1', 'trial': 'dev01'}])
    write_json(selected.state_path, state)
    other = {name: contents(campaign / name) for name in METHODS if name != method}
    config = (campaign / 'campaign.json').read_bytes()
    result = clear_method(campaign, method)
    assert result['reset']
    assert selected.knowledge == initial_knowledge(method)
    assert selected.state == initial_state(method)
    assert {p.name for p in selected.root.iterdir()} == {'.lock', 'state.json', selected.knowledge_path.name}
    assert (campaign / 'campaign.json').read_bytes() == config
    assert all(contents(campaign / name) == data for name, data in other.items())
    selected.save_learning('new source skill', {'task_id': 'S1'})
    assert selected.state['source_events'] == [{'task_id': 'S1'}]


def test_preview_preserves_existing_outputs_and_learning(tmp_path):
    campaign = initialize(tmp_path / 'campaign', TASKS)
    store = MethodStore(campaign, 'sd-only')
    store.save_learning('keep skill', {'task_id': 'S1'})
    before = contents(campaign)
    result = clear_method(campaign, 'sd-only', dry_run=True)
    assert not result['reset'] and result['dry_run']
    assert 'skills.md' in result['removed_entries']
    assert {k: v for k, v in contents(campaign).items() if not k.endswith('/.lock')} == before


def test_active_method_lock_blocks_clear(tmp_path):
    campaign = initialize(tmp_path / 'campaign', TASKS)
    store = MethodStore(campaign, 'sd-only')
    with store.locked():
        before = contents(campaign)
        script = 'from cap.agent.rsi.reset import clear_method; import sys; clear_method(sys.argv[1], "sd-only")'
        proc = subprocess.run([sys.executable, '-c', script, str(campaign)],
                              cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
        assert proc.returncode != 0 and 'active source/eval' in proc.stderr
        assert contents(campaign) == before


def test_method_directory_symlink_is_rejected(tmp_path):
    campaign = initialize(tmp_path / 'campaign', TASKS)
    old = campaign / 'sd-only'
    old.rename(campaign / 'unused')
    old.symlink_to(campaign / 'aspire', target_is_directory=True)
    before = contents(campaign / 'aspire')
    with pytest.raises(ValueError, match='real directory'):
        clear_method(campaign, 'sd-only')
    assert contents(campaign / 'aspire') == before


def test_child_symlinks_are_removed_without_touching_targets(tmp_path):
    campaign = initialize(tmp_path / 'campaign', TASKS)
    external = tmp_path / 'external'
    external.mkdir()
    protected = external / 'keep.txt'
    protected.write_text('must survive')
    (campaign / 'sd-only' / 'external_link').symlink_to(external, target_is_directory=True)
    (campaign / 'sd-only' / 'skills.md').unlink()
    (campaign / 'sd-only' / 'skills.md').symlink_to(protected)
    clear_method(campaign, 'sd-only')
    assert protected.read_text() == 'must survive'
    assert not (campaign / 'sd-only' / 'external_link').exists()
    assert not (campaign / 'sd-only' / 'skills.md').is_symlink()


def test_unknown_agent_is_rejected_before_any_deletion(tmp_path):
    campaign = initialize(tmp_path / 'campaign', TASKS)
    before = contents(campaign)
    with pytest.raises(ValueError, match='unknown method'):
        clear_method(campaign, '../aspire')
    assert contents(campaign) == before
