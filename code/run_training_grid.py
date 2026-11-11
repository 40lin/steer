"""Run the predeclared ten seeds for all three recurrent policy inputs."""
import concurrent.futures
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'dynamic_threshold'

def worker(job):
    mode,seed=job
    log=OUT/'logs'/f'{mode}_{seed}.stdout.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('run_threshold_experiment.py')),
            'train','--mode',mode,'--seed',str(seed),'--steps','2000000'],
            stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    message={'mode':mode,'seed':seed,'exit':result.returncode,'seconds':round(time.time()-started,2)}
    print(json.dumps(message),flush=True)
    if result.returncode:
        raise RuntimeError(f'Failed training {mode}/{seed}; see {log}')
    return message

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6,
                        help='Concurrent training processes (default: 6)')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    if (OUT/'pretest_freeze.json').exists():
        raise RuntimeError('This experiment is frozen; do not overwrite its checkpoints. '
                           'Use a fresh checkout/output directory for a new run.')
    if not (OUT/'validation_selection.json').exists():
        raise RuntimeError('Run the validation fixed-threshold baselines first.')
    jobs=[(mode,seed) for seed in range(10) for mode in ('belief','history','openloop')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(worker,jobs))
    (OUT/'training_grid.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
