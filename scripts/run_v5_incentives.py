"""Run V5 fixed-allocation report-conditioned incentive experiments in a fresh output directory."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/v5_conditional_incentive'
CODE=ROOT/'code'
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('PYTHONIOENCODING','utf-8')

def call(script,*args):
    subprocess.run([sys.executable,str(CODE/script),*map(str,args)],cwd=ROOT,check=True)

def worker(seed):
    directory=OUT/'logs';directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'train_{seed}.stdout.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,str(CODE/'run_conditional_incentive.py'),'train','--seed',str(seed)],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps({'training_seed':seed,'exit':result.returncode}),flush=True)
    if result.returncode:raise RuntimeError(f'Seed {seed} failed: inspect logs before resuming.')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('check','train','test','all'),default='all',nargs='?')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    if args.stage in ('check','all'):
        call('conditional_incentive_market.py','--self-test')
        call('run_conditional_incentive.py','self-test')
        call('conditional_incentive_stress.py','self-test')
        call('analyze_conditional_incentive.py','--self-test')
        call('evaluate_fullpool_incentives.py','self-test')
    if args.stage=='all':
        # Complete fresh-study reproduction includes the independent stage
        # audit before validation/learning. Its own freeze guard rejects reuse.
        call('conditional_incentive_stress.py','all')
    if args.stage in ('train','all'):
        if (OUT/'pretest_freeze.json').exists():raise RuntimeError('Test already frozen; use a fresh checkout.')
        if not (OUT/'selection.json').exists():call('run_conditional_incentive.py','prepare')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(worker,range(10)))
    if args.stage in ('test','all'):
        call('analyze_conditional_incentive.py','--freeze')
        call('run_conditional_incentive.py','test')
        call('analyze_conditional_incentive.py','--run')
        call('evaluate_fullpool_incentives.py','all')
