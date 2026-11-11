"""Independent forward-looking attack check for committed capacity billing.

The 89 report templates are selected by coalition total utility on validation,
then evaluated on fresh streams. A uniform no-op belief tracker reproduces the
history-only actor's input exactly; no private platform type information enters.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import adaptive_coalition_attack as attacks
import run_capacity_billing as billing
from capacity_billing_env import CapacityMarket,pricing_tables
from steering_env import VALUES

ROOT=Path(__file__).resolve().parents[1]
PLATFORM=ROOT/'experiments'/'v2_capacity_billing'
OUT=PLATFORM/'attacks'
SEEDS=(0,4,9)
VALIDATION_STREAM=710031
TEST_STREAM=710041


class UniformTracker:
    """Matches mode='history': all sixteen model feature slots stay uniform."""
    def __init__(self,batch_size,**kwargs):self.batch_size=batch_size
    @property
    def posterior(self):return np.full((self.batch_size,16),1/16)
    def observe(self,*args):return np.ones(self.batch_size)


class ConstantPolicy(nn.Module):
    def __init__(self,action):
        super().__init__();self.action=int(action)
        if not 0<=self.action<4:raise ValueError('Unknown fixed fallback')
    def forward(self,obs,hidden=None):
        logits=torch.full((*obs.shape[:2],4),-20.,device=obs.device)
        logits[:,:,self.action]=20.
        return logits,torch.zeros(obs.shape[:2],device=obs.device),hidden


def configure_attack_environment():
    attacks.BatchMarket=CapacityMarket
    attacks.TABLES=pricing_tables(.78,1.)
    attacks.EXACT_REPORT=attacks.exact_report_table()
    attacks.BeliefTracker=UniformTracker


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    value={'candidate_stage':'frozen_before_attack_evaluation',
        'platform_seeds':list(SEEDS),'price':.78,'capacity_weight':1.,
        'price_committed_all_rounds':True,'horizon':400,'control_rounds':200,
        'control_block_rounds':10,'actor_inputs':'public history with uniform posterior slots',
        'dynamic_action_menu':[.5,1.],'fixed_fallback_actions_allowed':[0,1,2,3],
        'selection':'maximum mean full-horizon coalition utility, lowest template index for ties',
        'templates':[asdict(t) for t in attacks.search_space()],
        'validation_worlds_per_market_template':8,'test_worlds_per_market_selected_attack':64,
        'validation_seed_base':VALIDATION_STREAM,'test_seed_base':TEST_STREAM,
        'stream_formula':'base + seed*10000 + market*101',
        'comparisons':'selected template vs exact myopic BR, paired streams within each policy and market',
        'scope':'restricted 89-template empirical check, not a search over all dynamic strategies',
        'independent_stage_bound':'finite-catalogue bound checked separately; no scope beyond the four markets',
        'source_sha256':{p.name:sha(p) for p in [Path(__file__),
            ROOT/'code'/'adaptive_coalition_attack.py',ROOT/'code'/'capacity_billing_env.py']}}
    path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise RuntimeError('Attack protocol/source changed; preserve this frozen search')
    else:write(path,value)
    return value


def verify_platform(seed):
    path=PLATFORM/'pretest_freeze.json'
    if not path.exists():raise FileNotFoundError('Platform has not been frozen before test')
    frozen=json.loads(path.read_text(encoding='utf-8'))
    hashes={k.replace('\\','/'):v for k,v in frozen['hashes'].items()}
    checkpoint=PLATFORM/'checkpoints'/f'dynamic_{seed}.pt'
    for file in (checkpoint,ROOT/'code'/'run_capacity_billing.py',ROOT/'code'/'capacity_billing_env.py'):
        key=file.relative_to(ROOT).as_posix()
        if hashes.get(key)!=sha(file):raise RuntimeError(f'Frozen platform file mismatch: {key}')
    log=json.loads((PLATFORM/'logs'/f'dynamic_{seed}.json').read_text(encoding='utf-8'))
    if not log or log[-1]['iteration']!=39:raise RuntimeError('Training is incomplete')
    data=torch.load(checkpoint,weights_only=False)
    return data,sha(checkpoint),sha(path)


def policy_from(checkpoint):
    if checkpoint['kind']=='fixed':return ConstantPolicy(checkpoint['action']).eval()
    if checkpoint['kind']!='dynamic':raise ValueError('Unknown checkpoint kind')
    result=billing.SafePolicy();result.load_state_dict(checkpoint['state_dict']);return result.eval()


def finite_stage_bound():
    """Enumerate every report/matching; not an empirical training assertion."""
    table=pricing_tables(.78,1.)
    n=table['counts'];mask=np.arange(table['revenue'].shape[-1])[None,None,:]<n[...,None]
    rows=[]
    for market in range(4):
        utility=table['utility'][market,1].sum(-1)
        mean=(utility*mask).sum(-1)/n
        upper=VALUES[market].sum()-3.9
        maxima=mean.max(-1)
        np.testing.assert_allclose(maxima,upper,atol=1e-12)
        report=attacks.EXACT_REPORT[market]
        for action,joint in enumerate(report):
            k=table['counts'][action,joint]
            np.testing.assert_allclose(table['revenue'][action,joint,:k],3.9,atol=1e-12)
            np.testing.assert_array_equal(table['served'][market,1,action,joint,:k].sum(-1),2)
        rows.append({'market':'ABCD'[market],'maximum_stage_utility':float(upper),
           'maximum_400_round_utility':float(400*upper),'myopic_revenue_400':1560.,
           'stage_maxima_all_five_numerical_thresholds':maxima.tolist(),
           'all_jobs_completed':True})
    return {'scope':'Only the fixed four-market catalogue, p=.78, whole-VM pooling, two accounts, no outside transfers/subsidies',
            'argument':'For every current threshold, full job completion attains the same maximum coalition stage utility. Summing the stage bounds yields a 400-round upper bound achieved by exact BR regardless of threshold history.',
            'does_not_imply':'Truthful individual reports, no cooperation, or a guarantee for larger/different markets.',
            'markets':rows}


def self_test():
    configure_attack_environment()
    torch.manual_seed(81231)
    for policy in (billing.SafePolicy().eval(),ConstantPolicy(0),ConstantPolicy(1)):
        result=attacks.simulate(policy,2,(attacks.search_space()[0],),3,81331)
        reference=billing.run('history',np.full(3,10),81331,policy=policy,response='exact_grand')
        for column,key in ((0,'revenue'),(1,'total_user_utility'),(2,'volume'),
                           (3,'completed_value'),(4,'completed_jobs')):
            np.testing.assert_allclose(result['metrics'][0,:,column],reference['totals'][key],atol=1e-9)
        np.testing.assert_array_equal(result['actions'][:,0],reference['actions'])
    fake=attacks.Template('selftest_fake','emulate_model',400,fake_model=7)
    policy=billing.SafePolicy().eval()
    result=attacks.simulate(policy,0,(fake,),3,81431)
    reference=billing.run('history',np.full(3,7),81431,policy=policy,response='learning')
    np.testing.assert_allclose(result['metrics'][0,:,0],reference['totals']['revenue'],atol=1e-9)
    np.testing.assert_array_equal(result['actions'][:,0],reference['actions'])
    finite_stage_bound()
    assert np.all(UniformTracker(3).posterior==1/16)
    print('PASS: capacity-price attack replay, history-only actor inputs, fixed fallback including action0, fake-model replay, and finite-catalogue stage bounds.',flush=True)


def run_seed(seed):
    prepare();configure_attack_environment()
    checkpoint,checkpoint_sha,freeze_sha=verify_platform(seed)
    templates=attacks.search_space();rows=[]
    for market in range(4):
        vs=VALIDATION_STREAM+seed*10000+101*market;ts=TEST_STREAM+seed*10000+101*market
        validation=attacks.simulate(policy_from(checkpoint),market,templates,8,vs)
        choice=attacks.choose_by_utility(validation['metrics'])
        selection={'seed':seed,'market':'ABCD'[market],'template_index':choice,
            'template':asdict(templates[choice]),'validation_utility':validation['metrics'][:,:,1].mean(1).tolist(),
            'checkpoint_sha256':checkpoint_sha,'test_stream_not_opened':True}
        write(OUT/f'selection_{seed}_{market}.json',selection)
        result=attacks.simulate(policy_from(checkpoint),market,(templates[0],templates[choice]),64,ts)
        reference,selected=result['metrics']
        row={'market':'ABCD'[market],'selected_template':templates[choice].name,
            'selected_template_index':choice,'attack_revenue':float(selected[:,0].mean()),
            'myopic_revenue':float(reference[:,0].mean()),
            'attack_utility':float(selected[:,1].mean()),'myopic_utility':float(reference[:,1].mean()),
            'coalition_utility_gain':attacks.paired_summary(selected[:,1],reference[:,1],ts+17),
            'platform_revenue_change':attacks.paired_summary(selected[:,0],reference[:,0],ts+31),
            'attack_jobs':float(selected[:,4].mean()),'myopic_jobs':float(reference[:,4].mean()),
            'observed_local_actions':sorted(map(int,np.unique(result['actions'])))}
        rows.append(row)
        np.savez_compressed(OUT/f'trajectories_{seed}_{market}.npz',validation_metrics=validation['metrics'],
           selection=np.array(choice),test_metrics=result['metrics'],test_actions=result['actions'])
        print(json.dumps({'seed':seed,**row}),flush=True)
    summary={'seed':seed,'kind':checkpoint['kind'],'fixed_action':checkpoint.get('action'),
        'selected_iteration':checkpoint.get('selected_iteration'),
        'checkpoint_sha256':checkpoint_sha,'pretest_freeze_sha256':freeze_sha,
        'attack_protocol_sha256':sha(OUT/'protocol.json'),'markets':rows,
        'mean_attack_revenue':float(np.mean([r['attack_revenue'] for r in rows])),
        'mean_myopic_revenue':float(np.mean([r['myopic_revenue'] for r in rows])),
        'mean_coalition_utility_gain':float(np.mean([r['coalition_utility_gain']['mean'] for r in rows]))}
    write(OUT/f'summary_{seed}.json',summary)
    return summary


def run_all(wait=False,deadline_minutes=180):
    prepare()
    deadline=time.monotonic()+deadline_minutes*60
    while not (PLATFORM/'pretest_freeze.json').exists():
        if not wait:raise RuntimeError('Waiting requires explicit --wait')
        if time.monotonic()>=deadline:raise TimeoutError('No frozen capacity-billing platform')
        time.sleep(5)
    configure_attack_environment()
    results=[run_seed(seed) for seed in SEEDS]
    value={'status':'complete','price':.78,'capacity_weight':1.,
        'predeclared_platform_seeds':list(SEEDS),'validation_worlds':8,'test_worlds':64,
        'template_count':89,'results':results,'finite_catalogue_stage_bound':finite_stage_bound(),
        'limitations':['The empirical search covers 89 fixed templates, not all dynamic strategies.',
          'The separate stage bound covers only the four specified markets and fixed posted price.',
          'Zero-width intervals can arise from deterministic histories and do not cover model/type uncertainty.',
          'If the validation-selected platform is a fixed fallback, this is not evidence of successful learned steering.']}
    write(OUT/'summary.json',value)
    print(json.dumps({'status':'complete','mean_revenue':float(np.mean([r['mean_attack_revenue'] for r in results])),
        'mean_utility_gain':float(np.mean([r['mean_coalition_utility_gain'] for r in results]))}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','self-test','run'])
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--deadline-minutes',type=float,default=180)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='self-test':self_test()
    else:run_all(args.wait,args.deadline_minutes)


if __name__=='__main__':main()
