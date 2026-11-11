"""V5 report-conditioned completion incentives, keeping allocation and bills fixed.

This independent runner is derived from the frozen V4 runner. PPO, reward
normalization, budget enforcement, withdrawal, validation ranking and evaluation
schema are unchanged. V5 adds 24 publicly report-conditioned rewards, 64 static
schedules and independent random streams. It never reads V4 experiment results.
"""
from __future__ import annotations
import os
for _variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_variable,'1')
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
import conditional_incentive_market as market

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'v5_conditional_incentive'
HORIZON=400
CONTROL=200
BLOCK=10
SEEDS=tuple(range(10))
BUDGETS=(0,10,25,50,100)
MAIN_BUDGET=50
VALIDATION_SEED=850031
TEST_SEED=850041
VALIDATION_WORLDS=50
TEST_WORLDS=100
OBS_DIM=71
BONUSES=np.asarray(market.BONUS_ACTIONS,dtype=float)
N_ACTIONS=len(BONUSES)
METRICS=('gross_revenue','cost','net_revenue','jobs','value','utility',
    'underreport_count','severity','overreport_count','active','join_count',
    'formed_members','harmful_members','successful_collusion_members',
    'trade_attempt','trade_success','trade_gains','failed_jobs')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8',newline='\n')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_prepared_sources():
    """Do not mix validation outcomes with silently changed simulator sources."""
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for relative,expected in protocol['source_sha256'].items():
        if digest(ROOT/relative)!=expected:
            raise RuntimeError(f'Prepared-study source changed: {relative}; use a fresh study or explicit amendment.')


def verify_design_freeze():
    frozen=json.loads((ROOT/'docs/v5_conditional_protocol.freeze.json').read_text(encoding='utf-8'))
    for relative,expected in frozen['files'].items():
        if digest(ROOT/relative)!=expected:
            raise RuntimeError(f'Design document changed after its freeze: {relative}')


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(OBS_DIM,64),nn.Tanh())
        self.gru=nn.GRU(64,64)
        self.actor=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,N_ACTIONS))
        self.critic=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,math.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        nn.init.orthogonal_(self.critic[-1].weight,1.)

    def forward(self,obs,hidden=None):
        z,hidden=self.gru(self.encoder(obs),hidden)
        return self.actor(z),self.critic(z).squeeze(-1),hidden


def candidates():
    # Preserve the original 39 candidates, then append the 64 V5 additions.
    result=[{'name':'none','kind':'fixed','action':0,'pacing':'front'}]
    for action in range(1,16):
        for pacing in ('front','uniform'):
            result.append({'name':f'fixed_{action}_{pacing}','kind':'fixed','action':action,'pacing':pacing})
    for level in range(4):
        for pacing in ('front','uniform'):
            result.append({'name':f'rotation_{level}_{pacing}','kind':'rotation','level':level,'pacing':pacing})
    for action in range(16,40):
        for pacing in ('front','uniform'):
            result.append({'name':f'fixed_{action}_{pacing}','kind':'fixed','action':action,'pacing':pacing})
    for threshold in (2,3):
        for level in range(4):
            for pacing in ('front','uniform'):
                result.append({'name':f'rotation_r{threshold}_{level}_{pacing}',
                    'kind':'rotation','threshold':threshold,'level':level,'pacing':pacing})
    assert len(result)==103 and len({x['name'] for x in result})==103
    return result


def scheduled_action(rule,macro):
    if rule['kind']=='fixed':return rule['action']
    base={0:1,2:16,3:28}[rule.get('threshold',0)]
    return base+(macro%3)*4+rule['level']


def ids_for(case,worlds):
    ids=np.arange(0,24,2) if case=='exact_transferable' else np.arange(24)
    return np.repeat(ids,worlds)


def public_features(previous,block_obs,remaining,budget,macro,running_completion,rounds):
    """40 advertised-action indicators and the unchanged 31 public features."""
    n=len(remaining);out=np.zeros((n,OBS_DIM),dtype=np.float32);offset=N_ACTIONS
    out[np.arange(n),previous]=1.
    if block_obs:
        reports=np.stack([o['reports'] for o in block_obs],axis=1)
        for buyer in range(3):
            for report in range(4):out[:,offset+buyer*4+report]=(reports[:,:,buyer]==report).mean(1)
        out[:,offset+12:offset+15]=np.mean([o['allocated_capacities'] for o in block_obs],axis=0)/3
        out[:,offset+15:offset+18]=np.mean([o['payments'] for o in block_obs],axis=0)/3
        out[:,offset+18:offset+21]=np.mean([o['completed'] for o in block_obs],axis=0)
        out[:,offset+21:offset+24]=np.mean([o['bonuses'] for o in block_obs],axis=0)/2
    out[:,offset+24]=remaining/100
    out[:,offset+25]=budget/100
    out[:,offset+26]=macro/40
    out[:,offset+27]=float(macro<20)
    out[:,offset+28:offset+31]=running_completion/max(rounds,1)
    return out


def budget_mask(remaining):
    return BLOCK*BONUSES.sum(1)[None,:] <= remaining[:,None]+1e-9


def run(ids,stream_seed,*,budget=MAIN_BUDGET,controller=None,policy=None,
        training=False,policy_seed=0,case='bounded'):
    ids=np.asarray(ids,dtype=int);n=len(ids)
    trade_mode='transferable' if case in ('transferable','exact_transferable') else ('none' if case=='no_trade' else 'bounded')
    env=market.Market(ids,trade_mode=trade_mode)
    rng=np.random.default_rng(stream_seed)
    uniforms=rng.random((HORIZON,n,4))
    remaining=np.full(n,float(budget));previous=np.zeros(n,dtype=int)
    block_obs=[];running_completion=np.zeros((n,3));hidden=None
    trajectories=[];actions=[];features=[];values=[];logs=[];masks=[]
    buyer_names=('completed','payments','bonuses','active','underreport','overreport')
    buyer_phases=np.zeros((n,2,3,len(buyer_names)))
    if training:torch.manual_seed(policy_seed)
    for macro in range(40):
        obs=public_features(previous,block_obs,remaining,budget,macro,running_completion,macro*BLOCK)
        feasible=budget_mask(remaining)
        if policy is not None:
            with torch.no_grad():
                logits,value,hidden=policy(torch.from_numpy(obs[None]),hidden)
                masked=logits[0].masked_fill(~torch.from_numpy(feasible),-1e9)
                dist=Categorical(logits=masked)
                chosen=dist.sample() if training and macro<20 else masked.argmax(-1)
                if macro>=20:chosen=torch.zeros_like(chosen)
                log=dist.log_prob(chosen).numpy();value=value[0].numpy()
            chosen=chosen.numpy()
        else:
            rule=controller or candidates()[0]
            action=scheduled_action(rule,macro)
            chosen=np.full(n,action,dtype=int)
            allowed=feasible[:,action]
            if rule['pacing']=='uniform':
                released=float(budget)*(macro+1)/20
                allowed &= (budget-remaining+BLOCK*BONUSES[action].sum() <= released+1e-9)
            chosen[~allowed]=0
            if macro>=20:chosen[:]=0
            value=np.zeros(n);log=np.zeros(n)
        assert feasible[np.arange(n),chosen].all()
        assert macro<20 or (chosen==0).all()
        block_obs=[];block=np.zeros((n,len(METRICS)))
        for step in range(BLOCK):
            forced=None
            if case=='exact_transferable':
                from conditional_incentive_stress import exact_best_actions
                forced=exact_best_actions(ids//2,chosen,trade_mode='transferable')
            public,metrics=env.step(chosen,uniforms[macro*BLOCK+step],buyer_actions=forced)
            assert set(public)=={'reports','allocated_capacities','payments','completed','bonuses'}
            block_obs.append(public)
            running_completion+=public['completed']
            reports=np.asarray(public['reports'])
            demand=market.DEMANDS[ids]
            buyer_phases[:,int(macro>=20)]+=np.stack((public['completed'],public['payments'],
                public['bonuses'],reports>0,(reports>0)&(reports<demand),reports>demand),axis=-1)
            cost=np.asarray(metrics['cost'])
            np.testing.assert_allclose(cost,np.asarray(public['bonuses']).sum(1),atol=1e-10)
            remaining-=cost
            block+=np.column_stack([metrics[key] for key in METRICS])
        assert remaining.min()>=-1e-8
        trajectories.append(block);actions.append(chosen.copy());features.append(obs)
        values.append(value);logs.append(log);masks.append(feasible);previous=chosen
    trajectories=np.stack(trajectories)
    result={'models':ids,'stream_seed':np.array(stream_seed),'budget':np.array(budget),
        'metric_names':np.array(METRICS),'metrics':trajectories.sum(0),
        'phases':np.stack([trajectories[:20].sum(0),trajectories[20:].sum(0)],axis=1),
        'trajectories':trajectories,'actions':np.stack(actions),'remaining_budget':remaining,
        'buyer_metric_names':np.array(buyer_names),'buyer_phases':buyer_phases}
    if training:
        u=trajectories[:,:,METRICS.index('underreport_count')]/30
        c=trajectories[:,:,METRICS.index('successful_collusion_members')]/30
        jobs=trajectories[:,:,METRICS.index('jobs')]/20
        weight=np.r_[np.full(20,2.),np.full(20,10.)][:,None]
        jweight=np.r_[np.full(20,1.),np.full(20,5.)][:,None]
        reward=(-weight*(u+c)+jweight*jobs)/20
        reward-=trajectories[:,:,METRICS.index('cost')]/50
        result['training']={'obs':np.stack(features),'action':np.stack(actions),
            'value':np.stack(values),'logprob':np.stack(logs),'mask':np.stack(masks),
            'reward':reward}
    return result


def save(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{k:v for k,v in result.items() if k!='training'})


def summary(result):
    metrics=result['metrics'];withdraw=result['phases'][:,1]
    ix={k:i for i,k in enumerate(METRICS)}
    return {'cost':float(metrics[:,ix['cost']].mean()),
        'net_revenue':float(metrics[:,ix['net_revenue']].mean()),
        'gross_revenue':float(metrics[:,ix['gross_revenue']].mean()),
        'jobs':float(metrics[:,ix['jobs']].mean()),
        'withdraw_jobs':float(withdraw[:,ix['jobs']].mean()),
        'withdraw_underreport':float(withdraw[:,ix['underreport_count']].sum()/max(withdraw[:,ix['active']].sum(),1)),
        'withdraw_collusion':float(withdraw[:,ix['successful_collusion_members']].mean()/600),
        'withdraw_coordination':float(withdraw[:,ix['harmful_members']].mean()/600)}


def rank(record,reference):
    targets=(.8*reference['withdraw_underreport'],.8*reference['withdraw_collusion'])
    feasible=(record['withdraw_underreport']<=targets[0]+1e-12 and
        record['withdraw_collusion']<=targets[1]+1e-12 and
        record['jobs']>=reference['jobs']-1e-12 and
        record['withdraw_jobs']>=reference['withdraw_jobs']-1e-12)
    ratios=(record['withdraw_underreport']/max(targets[0],1e-12),
        record['withdraw_collusion']/max(targets[1],1e-12),
        reference['jobs']/max(record['jobs'],1e-12),
        reference['withdraw_jobs']/max(record['withdraw_jobs'],1e-12))
    if feasible:return (1,-record['cost'],-sum(ratios[:2]),record['net_revenue'])
    return (0,-max(ratios),-record['cost'],0.)


def prepare():
    if (OUT/'selection.json').exists():raise RuntimeError('Validation already exists; use a fresh study.')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),
        'experiment':'v5_fixed_allocation_report_conditioned_completion_incentives','allocation':'A0, maximum original VM capacity, rho=0, uniform ties',
        'base_billing':'.2*c+.8*report; never controlled by the platform',
        'bonus_actions':BONUSES.tolist(),'reward_thresholds':market.REWARD_THRESHOLDS.tolist(),
        'base_bonus_ids':market.BASE_BONUS_IDS.tolist(),'target_buyers':market.TARGET_BUYERS.tolist(),
        'reward_condition':'Current public report meets advertised threshold AND own final task completes.',
        'relation_to_v4':'Separate extension and random streams; no direct cross-stream subtraction. The first 16 rewards and original 39 schedules are retained.',
        'sampling_and_update':'One buyer draw and one Q update per step; effective V4 reward selected from current public reports before voluntary trade.',
        'shared_immutable_economics':'incentive_market.py and validate_micro_model.py; no V4 experiment data are read or changed.',
        'horizon':400,'control_rounds':200,'withdraw_rounds':200,'block':10,
        'main_budget':50,'budget_migration':list(BUDGETS),'budget_rule':'Worst-case full-block redemption must fit available balance; no scaling.',
        'validation_seed':VALIDATION_SEED,'test_seed':TEST_SEED,'validation_worlds':VALIDATION_WORLDS,
        'test_worlds':TEST_WORLDS,'models':24,'seeds':list(SEEDS),'iterations':20,
        'training_steps_per_seed':1024000,'training_batch':128,'ppo_epochs':5,
        'checkpoint_selection':'Minimum actual cost among point-feasible U,C relative20% reductions and full/withdraw J non-loss; otherwise smallest worst normalized violation.',
        'fixed_candidates':candidates(),'primary_case':'bounded','stress_cases':['transferable','exact_transferable','no_trade'],
        'primary_comparison':'B50 dynamic versus no reward; incremental effect versus validation-best fixed/scheduled reward.',
        'budget_migration_checkpoint':'Same selected B50 checkpoint; fixed candidates selected per budget on validation.',
        'deployment_inputs':['reports','initial allocation','original payments','completed','own bonus payments','time','budget'],
        'private_labels':'demands,values,coalition membership and Q are simulator training/evaluation only',
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in
            [Path(__file__),ROOT/'code/conditional_incentive_market.py',
             ROOT/'code/incentive_market.py',ROOT/'code/validate_micro_model.py']}}
    write_json(OUT/'protocol.json',protocol)
    ids=ids_for('bounded',VALIDATION_WORLDS)
    baseline=run(ids,VALIDATION_SEED,budget=0)
    save(OUT/'validation/none.npz',baseline);reference=summary(baseline)
    selections={};records=[]
    for budget in BUDGETS:
        options=[]
        for number,rule in enumerate(candidates()):
            if budget==0 and number>0:continue
            result=baseline if rule['name']=='none' else run(ids,VALIDATION_SEED,budget=budget,controller=rule)
            row={'budget':budget,'candidate':rule,**summary(result)}
            options.append(row);records.append(row)
        best=max(options,key=lambda row:rank(row,reference))
        selections[str(budget)]=best
        print(json.dumps({'budget':budget,'selected':best,'feasible':bool(rank(best,reference)[0])}),flush=True)
    write_json(OUT/'selection.json',{'reference':reference,'by_budget':selections,'records':records})


def train(seed):
    if (OUT/'pretest_freeze.json').exists():raise RuntimeError('Frozen experiment cannot be retrained.')
    verify_prepared_sources()
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    reference=selection['reference'];best=selection['by_budget']['50']
    best_rank=rank(best,reference)
    path=OUT/'checkpoints'/f'dynamic_{seed}.pt';path.parent.mkdir(exist_ok=True)
    torch.manual_seed(855001+seed);rng=np.random.default_rng(856001+seed)
    policy=Policy();optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    torch.save({'kind':'scheduled','controller':best['candidate'],'seed':seed,'iteration':0,
        'validation':best,'reference':reference},path)
    history=[];started=time.perf_counter()
    for iteration in range(20):
        ids=rng.integers(0,24,size=128)
        roll=run(ids,int(rng.integers(1,2**31)),policy=policy,training=True,
            policy_seed=seed*10000+iteration)
        t=roll['training'];reward=t['reward'];value=t['value']
        advantage=np.zeros_like(reward);last=np.zeros(128)
        for k in reversed(range(40)):
            delta=reward[k]+(value[k+1] if k<39 else 0)-value[k]
            last=delta+.95*last;advantage[k]=last
        returns=advantage+value
        advantage=(advantage-advantage[:20].mean())/(advantage[:20].std()+1e-8)
        obs=torch.from_numpy(t['obs']);act=torch.from_numpy(t['action'])
        oldlog=torch.from_numpy(t['logprob']).float();adv=torch.from_numpy(advantage).float()
        ret=torch.from_numpy(returns).float();mask=torch.from_numpy(t['mask'])
        for _ in range(5):
            order=rng.permutation(128)
            for lo in range(0,128,32):
                ix=order[lo:lo+32];logits,pred,_=policy(obs[:,ix])
                dist=Categorical(logits=logits[:20].masked_fill(~mask[:20,ix],-1e9))
                log=dist.log_prob(act[:20,ix]);ratio=(log-oldlog[:20,ix]).exp();a=adv[:20,ix]
                loss=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
                loss+=.25*((pred-ret[:,ix])**2).mean()-.01*dist.entropy().mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
        row={'iteration':iteration+1,'market_steps':(iteration+1)*128*400,
            'train':summary(roll),'elapsed_seconds':time.perf_counter()-started}
        if (iteration+1)%5==0:
            validation=summary(run(ids_for('bounded',VALIDATION_WORLDS),VALIDATION_SEED,policy=policy))
            row['validation']=validation
            if rank(validation,reference)>best_rank:
                best_rank=rank(validation,reference)
                torch.save({'kind':'dynamic','state_dict':policy.state_dict(),'seed':seed,
                    'iteration':iteration+1,'validation':validation,'reference':reference},path)
            print(json.dumps({'seed':seed,**row}),flush=True)
        history.append(row)
    write_json(OUT/'logs'/f'dynamic_{seed}.json',history)


def load_controller(seed):
    checkpoint=torch.load(OUT/'checkpoints'/f'dynamic_{seed}.pt',weights_only=False)
    if checkpoint['kind']=='scheduled':return {'controller':checkpoint['controller']},checkpoint
    policy=Policy();policy.load_state_dict(checkpoint['state_dict']);policy.eval()
    return {'policy':policy},checkpoint


def freeze_test():
    if (OUT/'pretest_freeze.json').exists():raise RuntimeError('Test already frozen. Use a fresh study.')
    verify_prepared_sources()
    verify_design_freeze()
    required=[OUT/'protocol.json',OUT/'selection.json',OUT/'analysis_freeze.json']
    required += [OUT/'checkpoints'/f'dynamic_{s}.pt' for s in SEEDS]
    required += [OUT/'logs'/f'dynamic_{s}.json' for s in SEEDS]
    required += [ROOT/'code'/name for name in ('run_conditional_incentive.py','conditional_incentive_market.py',
        'conditional_incentive_stress.py','analyze_conditional_incentive.py',
        'incentive_market.py','incentive_stress.py','analyze_threshold_experiment.py',
        'validate_micro_model.py')]
    required += [ROOT/'docs'/name for name in ('v5_conditional_protocol.md','v5_conditional_protocol.freeze.json')]
    required += [ROOT/'scripts/run_v5_incentives.py']
    for p in required:
        if not p.exists():raise FileNotFoundError(p)
    for seed in SEEDS:
        log=json.loads((OUT/'logs'/f'dynamic_{seed}.json').read_text(encoding='utf-8'))
        assert len(log)==20 and log[-1]['market_steps']==1024000
    write_json(OUT/'pretest_freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'hashes':{p.relative_to(ROOT).as_posix():digest(p) for p in required}})


def test():
    freeze_test()
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    for case in ('bounded','transferable','exact_transferable','no_trade'):
        ids=ids_for(case,TEST_WORLDS)
        baseline=run(ids,TEST_SEED,budget=0,case=case)
        save(OUT/'test'/case/'none.npz',baseline)
        budgets=BUDGETS if case=='bounded' else (50,)
        for budget in budgets:
            rule=selection['by_budget'][str(budget)]['candidate']
            result=run(ids,TEST_SEED,budget=budget,controller=rule,case=case)
            save(OUT/'test'/case/f'fixed_b{budget}.npz',result)
            for seed in SEEDS:
                kwargs,checkpoint=load_controller(seed)
                result=run(ids,TEST_SEED,budget=budget,case=case,**kwargs)
                save(OUT/'test'/case/f'dynamic_b{budget}_s{seed}.npz',result)
                print(json.dumps({'case':case,'budget':budget,'seed':seed,
                    'kind':checkpoint['kind'],**summary(result)}),flush=True)


def self_test():
    torch.set_num_threads(1)
    assert len(candidates())==103 and BONUSES.shape==(40,3) and OBS_DIM==N_ACTIONS+31
    empty=public_features(np.zeros(2,dtype=int),[],np.array([50.,0.]),50,0,np.zeros((2,3)),0)
    assert empty.shape==(2,OBS_DIM)
    assert budget_mask(np.array([0.]))[0,0] and budget_mask(np.array([0.])).sum()==1
    ids=np.array([0,7,12,23]);rule=candidates()[-1]
    a=run(ids,12317,budget=10,controller=rule)
    b=run(ids,12317,budget=10,controller=rule)
    np.testing.assert_array_equal(a['metrics'],b['metrics'])
    assert np.all(a['actions'][20:]==0)
    assert a['metrics'][:,METRICS.index('cost')].max()<=10+1e-10
    np.testing.assert_allclose(a['phases'].sum(1),a['metrics'],atol=1e-8)
    np.testing.assert_allclose(a['metrics'][:,2]+a['metrics'][:,5],a['metrics'][:,4],atol=1e-8)
    torch.manual_seed(43);policy=Policy()
    z=run(ids,731,budget=0,policy=policy)
    q=run(ids,731,budget=0)
    np.testing.assert_array_equal(z['metrics'],q['metrics'])
    # Every conditional rotation addresses the intended current account and
    # threshold, without changing the conservative reservation amount.
    for rule in candidates():
        for macro in range(3):
            action=scheduled_action(rule,macro)
            assert 0<=action<40
            if rule['kind']=='rotation' and rule.get('threshold',0):
                assert market.TARGET_BUYERS[action]==macro
                assert market.REWARD_THRESHOLDS[action]==rule['threshold']
    # Sequence recomputation must reproduce the sampling-time PPO probability;
    # withdrawal actions are deterministic and excluded from actor loss.
    rollout=run(ids,85731,budget=10,policy=policy,training=True,policy_seed=85732)
    t=rollout['training']
    with torch.no_grad():
        logits,pred,_=policy(torch.from_numpy(t['obs']))
        distribution=Categorical(logits=logits[:20].masked_fill(~torch.from_numpy(t['mask'][:20]),-1e9))
        replay_log=distribution.log_prob(torch.from_numpy(t['action'][:20])).numpy()
    np.testing.assert_allclose(replay_log,t['logprob'][:20],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(pred.numpy(),t['value'],rtol=1e-6,atol=1e-6)
    assert np.all(t['action'][20:]==0)
    print('PASS: 103 schedules, 71 public features, conservative budget cap, deterministic replay, '
          'withdrawal, phase/cash accounting, zero-budget identity, conditional rotations and PPO replay.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','test','self-test'))
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='prepare':prepare()
    elif args.command=='train':train(args.seed)
    elif args.command=='test':test()
    else:self_test()


if __name__=='__main__':main()
