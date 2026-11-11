"""Predeclared threshold constraints on frozen v1 platforms; no retraining.

This diagnostic tests whether a floor or irreversible menu changes repair the
observed manipulation failure without changing prices or physical allocation.
The executed action, rather than the unconstrained neural proposal, feeds the
platform's next public observation. Original v1 files and results are untouched.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from run_threshold_experiment import Policy,run_episodes,save_result
from analyze_threshold_experiment import crossed_bootstrap,estimate,write_json,write_csv
import adaptive_coalition_attack as attack

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'v2_threshold_constraints'
CHECKPOINTS=ROOT/'experiments'/'dynamic_threshold'/'checkpoints'
MODES=('original','floor','floor_down_only','floor_up_only')
CASES=('exact','canonical','tabular')
SEEDS=tuple(range(10))
TEST_STREAM=420041
VALIDATION_STREAM=420031


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ConstrainedPolicy(nn.Module):
    """A deterministic action wrapper; no access to any true model state."""
    def __init__(self,policy,mode):
        super().__init__();self.policy=policy;self.mode=mode;self.last=None;self.steps=0
        if mode not in MODES:raise ValueError(mode)

    def forward(self,obs,hidden=None):
        logits,value,hidden=self.policy(obs,hidden)
        if self.mode=='original':return logits,value,hidden
        assert logits.shape[0]==1,'Wrapper evaluates one public decision block at a time'
        action=logits[0].argmax(-1).clamp(min=1)
        if self.mode=='floor_down_only' and self.last is not None:
            action=torch.minimum(action,self.last)
        if self.mode=='floor_up_only':
            action=torch.ones_like(action) if self.last is None else torch.maximum(action,self.last)
        self.last=action.clone();self.steps+=1
        constrained=torch.full_like(logits,-20.)
        constrained[0,torch.arange(len(action)),action]=20.
        return constrained,value,hidden


def load_policy(seed,mode):
    frozen=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    path=CHECKPOINTS/f'belief_{seed}.pt'
    if digest(path)!=frozen['checkpoint_sha256'][str(seed)]:
        raise RuntimeError('Original frozen checkpoint changed')
    state=torch.load(path,weights_only=False)
    policy=Policy();policy.load_state_dict(state['state_dict']);policy.eval()
    return ConstrainedPolicy(policy,mode).eval()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    value={'stage':'frozen_before_new_tests','created_utc':datetime.now(timezone.utc).isoformat(),
        'modes':MODES,'platform_seeds':SEEDS,'cases':CASES,
        'canonical_and_tabular_worlds_per_fixed_type':100,
        'exact_worlds_per_four_physical_markets':200,
        'validation_stream':VALIDATION_STREAM,'test_stream':TEST_STREAM,
        'policy_retraining':False,'prices_changed':False,'rebates':0,
        'default_comparison':'paired fixed rho=.5 on the same new stream',
        'action_rules':{'floor':'clamp proposed local action to at least1',
         'floor_down_only':'first proposal clamped to>=1; later min(previous,clamped proposal)',
         'floor_up_only':'first action1; later max(previous,clamped proposal)'},
        'attacks':{'modes':['floor'],'platform_seeds':[0,4,9],
          'templates':89,'validation_worlds':8,'test_worlds':64,
          'selection':'full horizon coalition utility; deterministic index tie',
          'validation_stream_base':520031,'test_stream_base':520041},
        'bootstrap':{'replicates':10000,'seed':1420041,'fixed_strata':True,
          'pairing':'crossed training-seed and within-type world resampling'},
        'checkpoint_sha256':{str(s):digest(CHECKPOINTS/f'belief_{s}.pt') for s in SEEDS},
        'source_sha256':{p.name:digest(p) for p in [Path(__file__),
            ROOT/'code'/'run_threshold_experiment.py',ROOT/'code'/'steering_env.py',
            ROOT/'code'/'belief_baselines.py',ROOT/'code'/'adaptive_coalition_attack.py']}}
    value=json.loads(json.dumps(value))
    path=OUT/'protocol.json'
    if path.exists():
        prior=json.loads(path.read_text(encoding='utf-8'))
        for key in value:
            if key!='created_utc' and prior[key]!=value[key]:
                raise RuntimeError('Existing protocol differs; preserve old results')
        return prior
    write_json(path,value)
    return value


def evaluate(case,mode,seed):
    ids=np.repeat(np.array([2,6,10,14]),200) if case=='exact' else np.repeat(np.arange(16),100)
    kwargs={'response':'exact_grand'} if case=='exact' else {'learner_mode':case}
    if mode=='fixed':
        result=run_episodes('fixed',ids,TEST_STREAM,fixed_action=1,**kwargs)
        name='fixed'
    else:
        policy=load_policy(seed,mode)
        result=run_episodes('belief',ids,TEST_STREAM,policy=policy,**kwargs)
        name=f'{mode}_{seed}'
    save_result(OUT/'data'/case/(name+'.npz'),result)
    print(json.dumps({'case':case,'mode':mode,'seed':seed,
                     'revenue':float(result['totals']['revenue'].mean()),
                     'volume':float(result['totals']['volume'].mean())}),flush=True)


def child_job(case,mode,seed):
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'{case}_{mode}_{seed}.log'
    command=[sys.executable,str(Path(__file__).resolve()),'evaluate','--case',case,'--mode',mode,'--seed',str(seed),
             '--checkpoint-dir',str(CHECKPOINTS)]
    started=time.monotonic()
    with path.open('w',encoding='utf-8') as handle:
        completed=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps({'job':[case,mode,seed],'exit':completed.returncode,
        'seconds':round(time.monotonic()-started,2)}),flush=True)
    if completed.returncode:raise RuntimeError(path.read_text(encoding='utf-8'))


def run_all(workers):
    prepare()
    jobs=[]
    for case in CASES:
        jobs.append((case,'fixed',-1))
        jobs.extend((case,mode,seed) for seed in SEEDS for mode in MODES)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(child_job,*job) for job in jobs]):future.result()


def summarize():
    rows=[];seed_rows=[];all_results={}
    for ci,case in enumerate(CASES):
        arrays={}
        for mode in ('fixed',*MODES):
            paths=[OUT/'data'/case/'fixed.npz'] if mode=='fixed' else [OUT/'data'/case/f'{mode}_{s}.npz' for s in SEEDS]
            samples=[]
            for path in paths:
                with np.load(path) as data:
                    m=data['metrics'];acts=data['actions'];phase=data['phase_totals']
                    assert np.isfinite(m).all()
                    np.testing.assert_allclose(phase.sum(1),m[:,:2],atol=1e-8)
                    assert np.all(acts[20:]==acts[19])
                    if mode.startswith('floor'):assert np.all(acts>=1)
                    if mode=='floor_down_only':assert np.all(np.diff(acts,axis=0)<=0)
                    if mode=='floor_up_only':assert np.all(np.diff(acts,axis=0)>=0) and np.all(acts[0]==1)
                    samples.append(m.reshape((4,200,8) if case=='exact' else (16,100,8)))
            arrays[mode]=np.stack(samples)
        boots=crossed_bootstrap(arrays,10000,1420041+ci)
        groups={'all':np.arange(arrays['fixed'].shape[1])}
        if case!='exact':
            groups['coalition']=np.array([i for i in range(16) if (i//2)%2])
            groups['independent']=np.array([i for i in range(16) if not (i//2)%2])
        case_rows=[]
        for group,index in groups.items():
            for mode in arrays:
                point=arrays[mode][:,index].mean(axis=(0,1,2))
                reference=arrays['fixed'][:,index].mean(axis=(0,1,2))
                boot=boots[mode][:,index].mean(axis=1);base=boots['fixed'][:,index].mean(axis=1)
                delta=estimate(point[0]-reference[0],boot[:,0]-base[:,0])
                jobs=estimate(point[2]-reference[2],boot[:,2]-base[:,2])
                row={'case':case,'group':group,'mode':mode,'revenue':point[0],
                     'volume':point[1],'completed_jobs':point[2],
                     'underreport_rate_among_active':point[6]/point[5],
                     'revenue_delta_vs_fixed':delta['estimate'],
                     'revenue_delta_ci_low':delta['ci95_low'],'revenue_delta_ci_high':delta['ci95_high'],
                     'jobs_delta_vs_fixed':jobs['estimate'],
                     'jobs_delta_ci_low':jobs['ci95_low'],'jobs_delta_ci_high':jobs['ci95_high']}
                rows.append(row);case_rows.append(row)
                if mode!='fixed':
                    for seed in SEEDS:
                        sp=arrays[mode][seed,index].mean(axis=(0,1))
                        seed_rows.append({'case':case,'group':group,'mode':mode,'seed':seed,
                            'revenue':sp[0],'delta_vs_fixed':sp[0]-reference[0],
                            'completed_jobs':sp[2],'volume':sp[1]})
        all_results[case]=case_rows
        print(json.dumps({'summary_case':case,'all_group':case_rows[:5]}),flush=True)
    write_json(OUT/'summary.json',{'protocol_sha256':digest(OUT/'protocol.json'),
         'scope':'All frozen constraints reported without outcome-based selection; exploratory follow-up.',
         'results':all_results})
    write_csv(OUT/'summary.csv',rows);write_csv(OUT/'seed_results.csv',seed_rows)


def attacks():
    prepare();templates=attack.search_space();all_seeds=[]
    for seed in (0,4,9):
        rows=[]
        for market in range(4):
            vs=520031+seed*10000+market*101;ts=520041+seed*10000+market*101
            validation=attack.simulate(load_policy(seed,'floor'),market,templates,8,vs)
            choice=attack.choose_by_utility(validation['metrics'])
            chosen=templates[choice]
            result=attack.simulate(load_policy(seed,'floor'),market,(templates[0],chosen),64,ts)
            reference,selected=result['metrics']
            row={'market':'ABCD'[market],'selected_template':chosen.name,
                 'attack_revenue':float(selected[:,0].mean()),'myopic_revenue':float(reference[:,0].mean()),
                 'coalition_utility_gain':attack.paired_summary(selected[:,1],reference[:,1],ts+17),
                 'revenue_delta':attack.paired_summary(selected[:,0],reference[:,0],ts+31),
                 'attack_jobs':float(selected[:,4].mean()),'myopic_jobs':float(reference[:,4].mean())}
            rows.append(row)
            np.savez_compressed(OUT/f'attack_floor_{seed}_{market}.npz',validation_metrics=validation['metrics'],
                selection=np.array(choice),test_metrics=result['metrics'],test_actions=result['actions'])
            print(json.dumps({'attack_seed':seed,**row}),flush=True)
        summary={'seed':seed,'mode':'floor','markets':rows,
                 'mean_attack_revenue':float(np.mean([r['attack_revenue'] for r in rows])),
                 'mean_myopic_revenue':float(np.mean([r['myopic_revenue'] for r in rows]))}
        write_json(OUT/f'attack_floor_{seed}.json',summary);all_seeds.append(summary)
    write_json(OUT/'attack_summary.json',{'selection':'89-template utility-only validation search',
        'scope':'restricted attack class lower bound; three predeclared seeds', 'results':all_seeds})


def self_test():
    torch.manual_seed(173)
    policy=Policy().eval();ids=np.array([2,6,10,14])
    for case in ('canonical','tabular'):
        a=run_episodes('belief',ids,9917,policy=policy,learner_mode=case)
        b=run_episodes('belief',ids,9917,policy=ConstrainedPolicy(policy,'original'),learner_mode=case)
        np.testing.assert_array_equal(a['actions'],b['actions'])
        for metric in a['totals']:np.testing.assert_array_equal(a['totals'][metric],b['totals'][metric])
    for mode in MODES[1:]:
        wrapped=ConstrainedPolicy(policy,mode)
        result=run_episodes('belief',ids,117,policy=wrapped,response='exact_grand')
        actions=result['actions'];assert (actions>=1).all()
        if mode=='floor_down_only':assert (np.diff(actions,axis=0)<=0).all()
        if mode=='floor_up_only':assert (actions[0]==1).all() and (np.diff(actions,axis=0)>=0).all()
    print('PASS: original wrapper identity, exact response replay and three action constraints.')


def main():
    global CHECKPOINTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','self-test','run','evaluate','summarize','attacks'])
    parser.add_argument('--case',choices=CASES,default='canonical')
    parser.add_argument('--mode',choices=('fixed',*MODES),default='floor')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--checkpoint-dir',type=Path,default=CHECKPOINTS,
        help='Directory containing the ten V1 belief_0.pt ... belief_9.pt checkpoints')
    args=parser.parse_args()
    CHECKPOINTS=args.checkpoint_dir.expanduser().resolve()
    if args.command=='prepare':prepare()
    elif args.command=='self-test':self_test()
    elif args.command=='run':run_all(args.workers)
    elif args.command=='evaluate':evaluate(args.case,args.mode,args.seed)
    elif args.command=='summarize':summarize()
    else:attacks()


if __name__=='__main__':main()
