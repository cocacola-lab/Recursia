"""Register the frozen source-generated suite in each isolated eval process."""
import hashlib
from pathlib import Path

SUITE='rsi_source_synthetic'


def register_synthetic(manifest, experiment_dir):
    from libero import benchmark
    exp=Path(experiment_dir)
    tasks=sorted((k,v) for k,v in manifest['tasks'].items() if v['suite']==SUITE)
    assert len(tasks)==18
    folder=exp/'synthetic/tasks'
    rows={}
    for short,v in tasks:
        stem=v['task']; bddl=folder/(stem+'.bddl'); init=folder/(stem+'.pruned_init')
        assert bddl.is_file() and init.is_file()
        assert hashlib.sha256(bddl.read_bytes()).hexdigest()==v['bddl_sha256']
        assert hashlib.sha256(init.read_bytes()).hexdigest()==v['init_sha256']
        rows[stem]=benchmark.Task(name=stem, language=v['language'],problem='Libero',
                                  problem_folder=str(folder),bddl_file=bddl.name,init_states_file=init.name)
    benchmark.task_maps[SUITE]=rows
    class RSI_SOURCE_SYNTHETIC(benchmark.Benchmark):
        def __init__(self,task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name=SUITE
            self._make_benchmark()
    benchmark.register_benchmark(RSI_SOURCE_SYNTHETIC)
    return {stem:i for i,stem in enumerate(rows)}
