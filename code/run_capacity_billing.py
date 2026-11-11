"""V2: committed capacity billing, public-history steering, strict service tests.

All V1 code and data remain unchanged. The primary posted price (.78) and safe
dynamic menu (.5, 1) are fixed before validation/test. Completion labels enter
offline simulator training rewards only; the deployed actor sees V1 public data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import run_threshold_experiment as v1
from capacity_billing_env import CapacityMarket,CapacityBeliefTracker,pricing_tables

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'v2_capacity_billing'
PRICE=.78
VALIDATION_SEED=700031
TEST_SEED=700041
SEEDS=tuple(range(10))
PRICES=(.65,.68,.70,.72,.75,.78,.79,.80,.82)
METRICS=v1.METRICS
original_run=v1.run_episodes


class SafePolicy(v1.Policy):
    def forward(self,obs,hidden=None):
        logits,values,hidden=super().forward(obs,hidden)
        mask=torch.tensor([True,False,True,False],device=logits.device)
        return logits.masked_fill(mask,-1e9),values,hidden


class TracedMarket(CapacityMarket):
    def __init__(self,*args,**kwargs):
        self.service_history=[]
        super().__init__(*args,**kwargs)
    def _realize_and_update(self,*args,**kwargs):
        obs,metrics=super()._realize_and_update(*args,**kwargs)
        self.service_history.append(np.column_stack((metrics['completed_jobs'],metrics['completed_value'])))
        return obs,metrics


def run(mode,ids,seed,*,policy=None,fixed_action=None,price=PRICE,alpha=1.,
        learner='canonical',response='learning',training=False,policy_seed=0,
        service_weights=(0.,0.),targets=(0.,0.)):
    created=[]
    def factory(model_ids,**kwargs):
        env=TracedMarket(model_ids,unit_price=price,capacity_weight=alpha,**kwargs)
        created.append(env)
        return env
    v1.BatchMarket=factory
    v1.BeliefTracker=lambda size,**kw:CapacityBeliefTracker(size,unit_price=price,capacity_weight=alpha,**kw)
    v1.TABLES=pricing_tables(price,alpha)
    result=original_run(mode,ids,seed,policy=policy,fixed_action=fixed_action,
        learner_mode=learner,response=response,collect_training=training,
        stochastic_policy=training,policy_seed=policy_seed)
    assert len(created)==1
    service=np.stack(created[0].service_history).reshape(40,10,len(ids),2).sum(axis=1)
    result['service_trajectories']=service
    if training:
        result['training']['reward'] += (service_weights[0]*(service[:,:,0]/20-targets[0]/800)
            +service_weights[1]*(service[:,:,1]/140-targets[1]/5600))
    return result


def save(path,result):
    v1.save_result(path,result)
    # Keep the standard eight metrics/phase arrays so prior analysis can read
    # them. Extra block service labels are only needed while training.


def ids_for(response,worlds=100):
    return np.repeat([2,6,10,14] if response=='exact_grand' else np.arange(16),worlds)


def mean_metrics(result):
    return {key:float(value.mean()) for key,value in result['totals'].items()}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),
        'primary_price':PRICE,'capacity_weight':1.,'price_fixed_all_400_rounds':True,
        'dynamic_menu_rho':[.5,1.],'fixed_comparators':[0,.5,2/3,1.],
        'training_seeds':SEEDS,'training_steps_each':1996800,'validation_seed':VALIDATION_SEED,
        'test_seed':TEST_SEED,'worlds_per_type':100,'primary_response':'canonical learning',
        'service_labels':'completed_jobs and completed_value; offline training/evaluation only',
        'service_targets':'no mean loss relative to V1 fixed rho=.5, price=1, alpha=.2',
        'checkpoint_selection':'maximize revenue among J/V-feasible candidates; include same-mechanism validation-best fixed fallback; if no feasible candidate maximize minimum normalized service slack, then revenue',
        'price_grid_exploratory':PRICES,'revenue_attribution':'new mechanism versus V1; steering versus best new fixed, reported separately',
        'strong_coalition_scope':'four fixed markets, indivisible jobs, whole VM pooling, posted price independent of reports/history',
        'all_seeds_mandatory':True,'rebate_budget_spent':0}
    path=OUT/'protocol.json'
    if path.exists():raise RuntimeError('Preparation already exists; do not overwrite this iteration')
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    ids=ids_for('learning')
    old=run('fixed',ids,VALIDATION_SEED,fixed_action=1,price=1,alpha=.2)
    save(OUT/'data'/'validation'/'old_fixed.npz',old)
    target=(old['totals']['completed_jobs'].mean(),old['totals']['completed_value'].mean())
    records=[]
    for action in range(4):
        result=run('fixed',ids,VALIDATION_SEED,fixed_action=action)
        save(OUT/'data'/'validation'/f'fixed_{action}.npz',result)
        records.append({'action':action,**mean_metrics(result)})
    # All four static rules remain comparators, even though dynamic actions use
    # a smaller menu justified by the independent-player equilibrium audit.
    feasible=[r for r in records if r['completed_jobs']>=target[0] and r['completed_value']>=target[1]]
    best=max(feasible if feasible else records,key=lambda r:r['revenue'])
    selection={'primary_price':PRICE,'targets_jobs_value':target,'old_fixed':mean_metrics(old),
        'fixed_records':records,'best_fixed_action':best['action'],
        'best_fixed_feasible':bool(feasible),'validation_seed':VALIDATION_SEED}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(selection),flush=True)


def rank(metrics,targets):
    slack=min((metrics['completed_jobs']-targets[0])/800,
              (metrics['completed_value']-targets[1])/5600)
    feasible=slack>=-1e-12
    return (int(feasible),metrics['revenue'] if feasible else slack,metrics['revenue'])


def train(seed):
    if (OUT/'pretest_freeze.json').exists():raise RuntimeError('Experiment already frozen')
    selected=json.loads((OUT/'selection.json').read_text())
    targets=selected['targets_jobs_value']
    torch.manual_seed(501001+seed)
    policy=SafePolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    rng=np.random.default_rng(601001+seed);batch=128
    champion=selected['fixed_records'][selected['best_fixed_action']]
    best_rank=rank(champion,targets)
    checkpoint={'kind':'fixed','action':selected['best_fixed_action'],'seed':seed,
                'selected_iteration':0,'validation':champion,'targets':targets}
    folder=OUT/'checkpoints';folder.mkdir(exist_ok=True)
    path=folder/f'dynamic_{seed}.pt';torch.save(checkpoint,path)
    weights=np.array([1.,1.]);logs=[];start=time.perf_counter()
    for iteration in range(39):
        ids=np.tile(np.arange(16),8);rng.shuffle(ids)
        roll=run('history',ids,int(rng.integers(1,2**31)),policy=policy,
            training=True,policy_seed=seed*10000+iteration,
            service_weights=weights,targets=targets)
        t=roll['training'];values=t['value'];reward=t['reward']
        advantage=np.zeros_like(reward);last=np.zeros(batch)
        for k in reversed(range(40)):
            future=values[k+1] if k<39 else 0
            delta=reward[k]+future-values[k]
            last=delta+.95*last;advantage[k]=last
        returns=advantage+values
        reference=advantage[:20];advantage=(advantage-reference.mean())/(reference.std()+1e-8)
        obs=torch.from_numpy(t['obs']);actions=torch.from_numpy(t['action'])
        oldlog=torch.from_numpy(t['logprob']).float();adv=torch.from_numpy(advantage).float()
        ret=torch.from_numpy(returns).float()
        for _ in range(10):
            order=rng.permutation(batch)
            for lo in range(0,batch,32):
                ix=order[lo:lo+32];logits,value,_=policy(obs[:,ix])
                dist=Categorical(logits=logits[:20]);log=dist.log_prob(actions[:20,ix])
                ratio=(log-oldlog[:20,ix]).exp();a=adv[:20,ix]
                loss=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
                loss+=.25*((value-ret[:,ix])**2).mean()-.01*dist.entropy().mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
        metrics=mean_metrics(roll)
        weights=np.clip(weights+.5*np.array([(targets[0]-metrics['completed_jobs'])/800,
                                             (targets[1]-metrics['completed_value'])/5600]),0,20)
        log={'iteration':iteration+1,'market_steps':(iteration+1)*128*400,
             'train':metrics,'service_weights':weights.tolist(),'elapsed_seconds':time.perf_counter()-start}
        if (iteration+1)%5==0 or iteration==38:
            validation=mean_metrics(run('history',ids_for('learning'),VALIDATION_SEED,policy=policy))
            log['validation']=validation
            candidate_rank=rank(validation,targets)
            if candidate_rank>best_rank:
                best_rank=candidate_rank
                checkpoint={'kind':'dynamic','state_dict':policy.state_dict(),'seed':seed,
                    'selected_iteration':iteration+1,'validation':validation,'targets':targets}
                torch.save(checkpoint,path)
            print(json.dumps({'seed':seed,**log}),flush=True)
        logs.append(log)
    logpath=OUT/'logs'/f'dynamic_{seed}.json';logpath.parent.mkdir(exist_ok=True)
    logpath.write_text(json.dumps(logs,indent=2),encoding='utf-8')


def freeze_test():
    path=OUT/'pretest_freeze.json'
    if path.exists():raise RuntimeError('Test already frozen; do not overwrite')
    required=[OUT/'checkpoints'/f'dynamic_{s}.pt' for s in SEEDS]
    required += [OUT/'logs'/f'dynamic_{s}.json' for s in SEEDS]
    required += [Path(__file__),ROOT/'code'/'capacity_billing_env.py',ROOT/'code'/'analyze_capacity_billing.py',
                 ROOT/'code'/'capacity_billing_attacks.py',OUT/'analysis_freeze.json',
                 OUT/'protocol.json',OUT/'selection.json']
    for p in required:
        if not p.exists():raise RuntimeError(f'Missing: {p}')
    document={'created_utc':datetime.now(timezone.utc).isoformat(),
              'hashes':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in required}}
    path.write_text(json.dumps(document,indent=2),encoding='utf-8')


def test():
    freeze_test()
    for case,learner,response in [('canonical','canonical','learning'),('tabular','tabular','learning'),
                                  ('exact','canonical','exact_grand')]:
        ids=ids_for(response);directory=OUT/'data'/case
        save(directory/'old_fixed.npz',run('fixed',ids,TEST_SEED,fixed_action=1,
             price=1,alpha=.2,learner=learner,response=response))
        for action in range(4):
            save(directory/f'fixed_{action}.npz',run('fixed',ids,TEST_SEED,
                 fixed_action=action,learner=learner,response=response))
        for seed in SEEDS:
            checkpoint=torch.load(OUT/'checkpoints'/f'dynamic_{seed}.pt',weights_only=False)
            if checkpoint['kind']=='fixed':
                result=run('fixed',ids,TEST_SEED,fixed_action=checkpoint['action'],learner=learner,response=response)
            else:
                policy=SafePolicy();policy.load_state_dict(checkpoint['state_dict']);policy.eval()
                result=run('history',ids,TEST_SEED,policy=policy,learner=learner,response=response)
            save(directory/f'dynamic_{seed}.npz',result)
            print(json.dumps({'case':case,'seed':seed,'kind':checkpoint['kind'],**mean_metrics(result)}),flush=True)


def price_grid():
    # Exploratory sensitivity; never changes PRICE or the selected checkpoints.
    rows=[]
    for learner in ('canonical','tabular'):
        ids=ids_for('learning')
        for price in PRICES:
            for action in range(4):
                r=run('fixed',ids,700081,fixed_action=action,price=price,learner=learner)
                rows.append({'price':price,'action':action,'learner':learner,**mean_metrics(r)})
    (OUT/'price_sensitivity.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Exploratory price sensitivity complete',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','train','test','price-grid'])
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='train':train(args.seed)
    elif args.command=='test':test()
    else:price_grid()


if __name__=='__main__':main()
