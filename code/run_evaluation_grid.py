"""Frozen, held-out evaluation. Run only after all training jobs finish."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'dynamic_threshold'
CODE=Path(__file__).resolve().parent

def run(job):
    label,args=job
    folder=OUT/'logs'/'evaluation';folder.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with (folder/f'{label}.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,*args],stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps({'job':label,'exit':result.returncode,'seconds':round(time.time()-started,2)}),flush=True)
    if result.returncode:raise RuntimeError(f'Failed {label}; see evaluation log')

def freeze():
    analysis_freeze=OUT/'analysis_script_freeze.json'
    if not analysis_freeze.exists():
        raise RuntimeError('Freeze the statistical analysis with analyze_threshold_experiment.py --freeze before testing.')
    analysis=json.loads(analysis_freeze.read_text(encoding='utf-8'))
    for key,path in (
        ('analysis_script_sha256',CODE/'analyze_threshold_experiment.py'),
        ('validation_selection_sha256',OUT/'validation_selection.json'),
        ('protocol_sha256',OUT/'protocol_audit.md'),
    ):
        if not path.exists() or analysis.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f'Frozen analysis input changed or is missing: {path.name}')
    expected=[OUT/'checkpoints'/f'{mode}_{seed}.pt' for mode in ('belief','history','openloop') for seed in range(10)]
    expected += [OUT/'logs'/f'{mode}_{seed}.json' for mode in ('belief','history','openloop') for seed in range(10)]
    expected += [OUT/'schedule_selection.json',CODE/'analyze_threshold_experiment.py']
    for path in expected:
        if not path.exists():raise RuntimeError(f'Not ready: {path}')
    files=expected+list(CODE.glob('*.py'))+[OUT/'validation_selection.json',OUT/'implementation_addendum.json',OUT/'protocol_audit.md',analysis_freeze]
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'worlds_per_type':100,
       'test_seed':800021,'hashes':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path=OUT/'pretest_freeze.json'
    if path.exists():raise RuntimeError('Pretest manifest already exists; do not overwrite frozen test provenance')
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':
    freeze()
    import numpy as np
    from schedule_baselines import run_schedule
    from run_threshold_experiment import save_result
    selected=json.loads((OUT/'schedule_selection.json').read_text(encoding='utf-8'))
    result=run_schedule(selected['selected']['actions'],np.repeat(np.arange(16),100),800021)
    save_result(OUT/'data'/'test'/'schedule_best.npz',result)
    script=str(CODE/'run_threshold_experiment.py')
    suffix=['--split','test','--worlds','100','--stream-seed','800021']
    jobs=[('fixed',[script,'baselines',*suffix])]
    jobs += [(f'{mode}_{seed}',[script,'evaluate','--mode',mode,'--seed',str(seed),*suffix])
             for seed in range(10) for mode in ('belief','history','openloop')]
    jobs += [(mode,[script,'simple','--mode',mode,*suffix]) for mode in
             ('ucb','bayes_greedy','fete_fixed','oracle_fixed','oracle_unconstrained')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run,jobs))
    (OUT/'main_evaluation_complete.json').write_text(json.dumps({'jobs':len(jobs),'utc':datetime.now(timezone.utc).isoformat()}),encoding='utf-8')
