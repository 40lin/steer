"""Run the V2 committed-capacity-price iteration without changing V1 outputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'v2_capacity_billing'
CODE=ROOT/'code'
for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(variable,'1')

def call(script,*arguments):
    subprocess.run([sys.executable,str(CODE/script),*map(str,arguments)],cwd=ROOT,check=True)

def worker(seed):
    directory=OUT/'logs';directory.mkdir(exist_ok=True)
    with (directory/f'dynamic_{seed}.stdout.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,str(CODE/'run_capacity_billing.py'),'train','--seed',str(seed)],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps({'training_seed':seed,'exit':result.returncode}),flush=True)
    if result.returncode:raise RuntimeError(f'Training failed; see seed {seed} log')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','test','sensitivity','all'],default='all',nargs='?')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    if args.stage in ('train','all'):
        if (OUT/'pretest_freeze.json').exists():raise RuntimeError('V2 test already frozen; use a fresh study')
        if not (OUT/'selection.json').exists():call('run_capacity_billing.py','prepare')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(worker,range(10)))
    if args.stage in ('test','all'):
        # Both protocols must exist before the first held-out test is opened.
        # An existing matching protocol is verified; a frozen test itself
        # cannot be resumed or silently overwritten by the next command.
        call('analyze_capacity_billing.py','--freeze')
        call('capacity_billing_attacks.py','prepare')
        call('run_capacity_billing.py','test')
        call('analyze_capacity_billing.py','--run')
        call('capacity_billing_attacks.py','run')
    if args.stage in ('sensitivity','all'):call('run_capacity_billing.py','price-grid')
