"""Independent Git histories for reusable RSI knowledge; never rewind run budgets."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .promotion import library_files
from .store import MethodStore, digest, initial_knowledge, read_json, write_json


class EvolutionGit:
    def __init__(self, store):
        self.store = store
        self.root = store.root
        self.repo = self.root / '.evolution.git'
        if self.root.is_symlink() or self.root.resolve() != self.root:
            raise ValueError('agent workspace must be a real directory inside the campaign')
        for path in (self.repo, store.state_path, self.root / '.lock', self.root / 'frozen_history'):
            if path.is_symlink():
                raise ValueError('agent version files must not be symlinks')

    def git(self, *args, input=None, index=None, check=True):
        env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        if index is not None:
            env['GIT_INDEX_FILE'] = str(index)
        command = ['git', '--git-dir=' + str(self.repo), '-c', 'core.hooksPath=/dev/null',
                   '-c', 'user.name=ASPIRE RSI', '-c', 'user.email=rsi@localhost', *args]
        result = subprocess.run(command, input=input, capture_output=True, env=env, cwd=self.root)
        if check and result.returncode:
            raise RuntimeError('agent Git: ' + result.stderr.decode(errors='replace').strip())
        return result.stdout

    def head(self):
        return self.git('rev-parse', '--verify', 'refs/heads/current', check=False).decode().strip() or None

    def _validate_files(self, files):
        primary = self.store.knowledge_path.name
        if primary not in files:
            raise ValueError('agent version is missing its knowledge file')
        for name, content in files.items():
            if name not in (primary, 'skill_library.json') and not re.fullmatch(r'skills/[A-Za-z0-9_-]+\.md', name):
                raise ValueError('Git version contains a non-learning file')
            if not isinstance(content, str):
                raise ValueError('agent Git versions contain only UTF-8 learning files')

    def files(self, revision):
        result = {}
        for entry in self.git('ls-tree', '-rz', '--full-tree', revision).split(b'\0'):
            if not entry:
                continue
            header, name = entry.split(b'\t', 1)
            mode, kind, oid = header.split()
            if mode != b'100644' or kind != b'blob':
                raise ValueError('agent Git version contains a link or unsupported file')
            result[name.decode()] = self.git('cat-file', 'blob', oid.decode()).decode()
        self._validate_files(result)
        return result

    def _commit(self, files, message):
        self._validate_files(files)
        previous = self.head()
        with tempfile.TemporaryDirectory(prefix='rsi-git-index-') as folder:
            index = Path(folder) / 'index'
            self.git('read-tree', '--empty', index=index)
            for name, content in sorted(files.items()):
                oid = self.git('hash-object', '-w', '--stdin', input=content.encode()).decode().strip()
                self.git('update-index', '--add', '--cacheinfo', '100644,' + oid + ',' + name, index=index)
            tree = self.git('write-tree', index=index).decode().strip()
        if previous and tree == self.git('rev-parse', previous + '^{tree}').decode().strip():
            return {'head': previous, 'previous': previous, 'changed': False}
        parents = ['-p', previous] if previous else []
        commit = self.git('commit-tree', tree, *parents, input=(message + '\n').encode()).decode().strip()
        self.git('update-ref', 'refs/heads/current', commit, previous or '0' * 40)
        return {'head': commit, 'previous': previous, 'changed': True}

    def _import_history(self):
        """Only import saved, hash-verified source updates, never failed proposals."""
        primary = self.store.knowledge_path.name
        if self.store.method in ('sd-only', 'recursia'):
            for event in self.store.state['source_events']:
                if event.get('kind') != 'summary' or event.get('learning_error') or not event.get('path'):
                    continue
                folder = Path(event['path']).resolve()
                if not folder.is_relative_to(self.root / 'source') or not (folder / 'response.json').is_file():
                    continue
                content = read_json(folder / 'response.json').get('code')
                if not isinstance(content, str) or digest(content) != event.get('knowledge_after_sha256'):
                    continue
                initial = initial_knowledge(self.store.method)
                if self.head() is None and digest(initial) == event.get('knowledge_before_sha256'):
                    self._commit({primary: initial}, 'Initial knowledge before recorded source learning')
                self._commit({primary: content}, 'Import source ' + str(event.get('task_id')) + ': ' + str(folder.relative_to(self.root)))
        else:
            for record_file in sorted((self.root / 'skill_promotions').glob('*/record.json')):
                if record_file.resolve() != record_file:
                    raise ValueError('skill promotion archive must belong to this agent')
                record = read_json(record_file)
                if record.get('method') != self.store.method or not record.get('source_path', '').startswith('source/'):
                    continue
                for stage in ('before', 'after'):
                    files = {}
                    for name, checksum in record['skill_sha256_' + stage].items():
                        # Validate names before reading a model-authored archive path.
                        self._validate_files({primary: '', name: ''})
                        path = record_file.parent / stage / name
                        if path.is_symlink() or not path.resolve().is_relative_to(record_file.parent.resolve()):
                            raise ValueError('skill snapshot must stay in its source archive')
                        content = path.read_text()
                        if digest(content) != checksum:
                            raise ValueError('cannot import changed source skill snapshot')
                        files[name] = content
                    self._commit(files, 'Import source ' + record['source_path'] + ': ' + stage)

    def ensure(self):
        if not self.repo.exists():
            self.git('init', '--bare', '--initial-branch=current', str(self.repo))
        if self.git('rev-parse', '--is-bare-repository').strip() != b'true':
            raise ValueError('agent evolution repository must be bare')
        if self.head() is None:
            self._import_history()

    def checkpoint(self, message):
        self.ensure()
        return self._commit(library_files(self.store), message)

    def history(self):
        if not self.repo.is_dir() or not self.head():
            return []
        raw = self.git('log', '--format=%H%x00%s', 'refs/heads/current').decode()
        return [dict(zip(('commit', 'message'), line.split('\0', 1))) for line in raw.splitlines()]

    def _write_files(self, files):
        self._validate_files(files)
        current = library_files(self.store)
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
                stream.write(content)
                temporary = stream.name
            try:
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        for name in current.keys() - files.keys():
            (self.root / name).unlink()

    def rollback(self, *, dry_run=False):
        if dry_run:
            if not self.head():
                raise ValueError('no Git version yet; use --init-only first')
            if library_files(self.store) != self.files(self.head()):
                raise ValueError('unrecorded learning changes; real rollback will checkpoint these first')
        else:
            self.checkpoint('Checkpoint current learning before requested rollback')
        previous = self.head()
        target = self.git('rev-parse', '--verify', previous + '^', check=False).decode().strip()
        if not target:
            raise ValueError('this agent has only one recorded version; no previous version exists')
        before, after = library_files(self.store), self.files(target)
        changed = [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]
        packet = {'agent': self.store.method, 'git_dir': str(self.repo), 'before': previous, 'after': target,
                  'changed_files': changed, 'dry_run': dry_run,
                  'unfrozen': bool(self.store.state['frozen']), 'logs_and_budgets_preserved': True}
        if dry_run:
            return packet
        state_bytes = self.store.state_path.read_bytes()
        frozen = self.root / 'frozen.json'
        if frozen.is_symlink():
            raise ValueError('frozen snapshot must not be a symlink')
        archive = None
        self.git('update-ref', 'refs/archive/' + str(time.time_ns()), previous)
        try:
            self._write_files(after)
            if frozen.is_file():
                archive = self.root / 'frozen_history' / (str(time.time_ns()) + '.json')
                archive.parent.mkdir(exist_ok=True)
                os.replace(frozen, archive)
            state = self.store.state
            state['frozen'] = False
            write_json(self.store.state_path, state)
            event = {'kind': 'git_rollback', 'from_commit': previous, 'to_commit': target,
                     'changed_files': changed, 'knowledge_before_sha256': digest(self._context(before)),
                     'knowledge_after_sha256': digest(self.store.context_knowledge()),
                     'archived_frozen': str(archive.relative_to(self.root)) if archive else None,
                     'dry_run': False}
            self.store.save_learning(after[self.store.knowledge_path.name], event)
            self.git('update-ref', 'refs/heads/current', target, previous)
        except Exception:
            self._write_files(before)
            self.store.state_path.write_bytes(state_bytes)
            if archive and archive.is_file():
                os.replace(archive, frozen)
            raise
        packet['archived_frozen'] = str(archive) if archive else None
        return packet

    def _context(self, files):
        parts = [files[self.store.knowledge_path.name]]
        parts += [f'# Source skill file: {name}\n{files[name]}' for name in sorted(files) if name.startswith('skills/')]
        return '\n\n'.join(part for part in parts if part)


def manage_versions(campaign, method, *, action='rollback', dry_run=False):
    if action not in ('rollback', 'init', 'list'):
        raise ValueError('unknown agent version operation')
    store = MethodStore(campaign, method)
    versions = EvolutionGit(store)
    if action == 'list':
        return {'agent': method, 'git_dir': str(versions.repo), 'versions': versions.history()}
    with store.locked():
        if action == 'init':
            packet = versions.checkpoint('Adopt current agent learning')
            return {'agent': method, 'git_dir': str(versions.repo), **packet, 'versions': versions.history()}
        return versions.rollback(dry_run=dry_run)
