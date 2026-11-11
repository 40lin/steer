"""Independent strong-coalition replay for frozen V5 incentive policies.

All buyers may pool VMs across DIFFERENT reports and transfer arbitrary money.
The coalition is imposed as a stress case: its existence is not an outcome.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import torch
import run_conditional_incentive as runner
import conditional_incentive_stress as stress
from analyze_threshold_experiment import crossed_bootstrap,estimate

ROOT=Path(__file__).resolve().parents[1]
OUT=runner.OUT/'fullpool_replay'
STREAM=850061
BOOTSTRAP_SEED=850071
WORLDS=100
METRICS=tuple(stress.FULLPOOL_METRICS)


def simulate(physical,stream=STREAM,*,budget=50,policy=None,controller=None):
    n=len(physical);remaining=np.full(n,float(budget));previous=np.zeros(n,dtype=int)
    observations=[];running=np.zeros((n,3));hidden=None
    rng=np.random.default_rng(stream);uniforms=rng.random((400,n))
    phases=np.zeros((n,2,len(METRICS)));actions=[]
    for macro in range(40):
        mask=runner.budget_mask(remaining)
        if policy is not None:
            obs=runner.public_features(previous,observations,remaining,budget,macro,running,macro*10)
            with torch.no_grad():
                logits,_,hidden=policy(torch.from_numpy(obs[None]),hidden)
                selected=logits[0].masked_fill(~torch.from_numpy(mask),-1e9).argmax(-1).numpy()
        else:
            rule=controller or runner.candidates()[0]
            action=runner.scheduled_action(rule,macro)
            selected=np.full(n,action,dtype=int)
            allowed=mask[:,action]
            if rule['pacing']=='uniform':
                allowed &= budget-remaining+10*runner.BONUSES[action].sum()<=budget*(macro+1)/20+1e-9
            selected[~allowed]=0
        if macro>=20:selected[:]=0
        assert mask[np.arange(n),selected].all()
        observations=[]
        for within in range(10):
            public,metrics=stress.fullpool_stage_step(physical,selected,uniforms[macro*10+within])
            assert set(public)=={'reports','allocated_capacities','payments','completed','bonuses'}
            np.testing.assert_allclose(metrics['cost'],public['bonuses'].sum(1),atol=1e-10)
            np.testing.assert_allclose(metrics['utility']+metrics['net_revenue'],metrics['value'],atol=1e-10)
            remaining-=metrics['cost'];running+=public['completed'];observations.append(public)
            phases[:,int(macro>=20)]+=np.column_stack([metrics[key] for key in METRICS])
        assert remaining.min()>=-1e-8
        actions.append(selected.copy());previous=selected
    return {'physical_ids':physical,'stream_seed':np.array(stream),'metric_names':np.array(METRICS),
        'budget':np.array(budget),'metrics':phases.sum(1),'phases':phases,'actions':np.stack(actions),
        'remaining_budget':remaining}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)


def self_test():
    ids=np.arange(12);zero=simulate(ids,850003,budget=0)
    np.testing.assert_allclose(zero['metrics'][:,METRICS.index('net_revenue')],1040.,atol=1e-8)
    np.testing.assert_array_equal(zero['metrics'][:,METRICS.index('jobs')],800)
    rewarded=simulate(ids,850003,budget=50,controller={'kind':'fixed','action':34,'pacing':'front'})
    np.testing.assert_allclose(rewarded['phases'][:,1],zero['phases'][:,1],atol=1e-8)
    assert (rewarded['actions'][20:]==0).all()
    assert rewarded['metrics'][:,METRICS.index('cost')].max()<=50+1e-8
    print('PASS: full-pool strong BR, public-only replay, exact zero-reward receipts/jobs, budget and withdrawal identity.')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    required=[Path(__file__),ROOT/'code/run_conditional_incentive.py',
        ROOT/'code/conditional_incentive_market.py',ROOT/'code/conditional_incentive_stress.py',
        ROOT/'code/incentive_market.py',ROOT/'code/incentive_stress.py',
        ROOT/'code/validate_micro_model.py',ROOT/'code/analyze_threshold_experiment.py',
        runner.OUT/'selection.json',runner.OUT/'pretest_freeze.json']
    required += [runner.OUT/'checkpoints'/f'dynamic_{seed}.pt' for seed in range(10)]
    for path in required:
        if not path.exists():raise FileNotFoundError(path)
    # Preserve the original selection freeze, rather than legitimizing changed
    # checkpoints by hashing their new contents in this later stress protocol.
    main_freeze=json.loads((runner.OUT/'pretest_freeze.json').read_text(encoding='utf-8'))
    for relative,expected in main_freeze['hashes'].items():
        if runner.digest(ROOT/relative)!=expected:
            raise RuntimeError(f'Main V5 freeze no longer matches: {relative}')
    if (OUT/'protocol.json').exists():raise RuntimeError('Full-pool replay already frozen.')
    frozen={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_fullpool_policy_replay',
        'hashes':{p.relative_to(ROOT).as_posix():runner.digest(p) for p in required},
        'physical_types':12,'worlds_per_type':WORLDS,'stream':STREAM,'budget':50,
        'no_reward_reference_budget':0,'main_pretest_manifest_verified':True,
        'bootstrap_replicates':10000,'bootstrap_seed':BOOTSTRAP_SEED,
        'methods':['none','validation_selected_fixed','all10_selected_dynamic'],
        'comparison':'Paired 400-round and phase receipts/cost/jobs/underreports; no formation claim for external coalition.',
        'selection':'All V5 checkpoints and fixed rule frozen on original validation before this stronger stress test.',
        'posterior_information':'Exactly the public-feature whitelist of the main policy.',
        'theory':'Full-pool exact expected net receipts <=2.6 per round under any nonnegative bonus; not a pointwise sample-path cap.',
        'test_or_checkpoint_reselection':False}
    runner.write_json(OUT/'protocol.json',frozen)


def run():
    frozen=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for name,sha in frozen['hashes'].items():
        if runner.digest(ROOT/name)!=sha:raise RuntimeError(f'Frozen source changed: {name}')
    if (OUT/'data').exists():raise RuntimeError('Replay data already exists; do not overwrite.')
    ids=np.repeat(np.arange(12),WORLDS)
    selection=json.loads((runner.OUT/'selection.json').read_text(encoding='utf-8'))
    rule=selection['by_budget']['50']['candidate']
    results={'none':simulate(ids,budget=0),'fixed':simulate(ids,controller=rule)}
    np.testing.assert_allclose(results['fixed']['phases'][:,1],
                               results['none']['phases'][:,1],atol=1e-8,rtol=0)
    save(OUT/'data/none.npz',results['none']);save(OUT/'data/fixed.npz',results['fixed'])
    dynamic=[];choices=[]
    for seed in range(10):
        kwargs,checkpoint=runner.load_controller(seed)
        data=simulate(ids,**kwargs);dynamic.append(data)
        np.testing.assert_allclose(data['phases'][:,1],results['none']['phases'][:,1],
                                   atol=1e-8,rtol=0)
        assert (data['actions'][20:]==0).all()
        choices.append({'seed':seed,'kind':checkpoint['kind'],'iteration':checkpoint['iteration']})
        save(OUT/'data'/f'dynamic_{seed}.npz',data)
        print(json.dumps({'fullpool_seed':seed,'net_revenue':float(data['metrics'][:,METRICS.index('net_revenue')].mean()),
            'cost':float(data['metrics'][:,METRICS.index('cost')].mean())}),flush=True)
    arrays={key:np.concatenate((data['metrics'][:,None],data['phases']),axis=1).reshape(1,12,WORLDS,-1)
        for key,data in results.items()}
    arrays['dynamic']=np.stack([np.concatenate((d['metrics'][:,None],d['phases']),axis=1).reshape(12,WORLDS,-1)
        for d in dynamic])
    draws=crossed_bootstrap(arrays,10000,BOOTSTRAP_SEED,chunk_size=64)
    methods={};comparisons={};k=len(METRICS)
    for name,array in arrays.items():
        point=array.mean((0,1,2)).reshape(3,k)
        boot=draws[name].mean(1).reshape(10000,3,k)
        methods[name]={phase:{metric:estimate(point[p,col],boot[:,p,col]) for col,metric in enumerate(METRICS)}
            for p,phase in enumerate(('full','control','withdraw'))}
        for p,phase in enumerate(('full','control','withdraw')):
            u=METRICS.index('underreport_count');active=METRICS.index('active')
            methods[name][phase]['underreport_active_rate']=estimate(point[p,u]/point[p,active],boot[:,p,u]/boot[:,p,active])
    for reference in ('none','fixed'):
        diff=(arrays['dynamic'].mean((0,1,2))-arrays[reference].mean((0,1,2))).reshape(3,k)
        boot=(draws['dynamic']-draws[reference]).mean(1).reshape(10000,3,k)
        comparisons[reference]={phase:{metric:estimate(diff[p,col],boot[:,p,col]) for col,metric in enumerate(METRICS)}
            for p,phase in enumerate(('full','control','withdraw'))}
    runner.write_json(OUT/'summary.json',{'status':'complete','scope':frozen['comparison'],
        'protocol_sha256':runner.digest(OUT/'protocol.json'),'choices':choices,'methods':methods,'dynamic_vs':comparisons,
        'receipt_bound_is_expected_not_sample_path':True,'coalition_formation_measured':False,
        'data_sha256':{p.name:runner.digest(p) for p in (OUT/'data').glob('*.npz')}})
    print('Full-pool frozen-policy replay complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('self-test','prepare','run','all'))
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='all':self_test();prepare();run()
    elif args.command=='self-test':self_test()
    elif args.command=='prepare':prepare()
    else:run()
