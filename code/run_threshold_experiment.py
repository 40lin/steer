"""Actual threshold experiments: public-history recurrent PPO and strong controls.

No rebates. Canonical economic eligibility features are primary. A threshold is
chosen every 10 rounds for 200 rounds, then its final value is held for 200.
The actor never receives model IDs, private rewards, learner weights or RNG IDs.
All true hidden information below is confined to environment generation/evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from steering_env import BatchMarket, PRIMARY_ACTION_INDICES, RHO_VALUES, TABLES, JOINT_REPORTS
from belief_baselines import BeliefTracker

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'dynamic_threshold'
BLOCK = 10
CONTROL = 200
HOLD = 200
HORIZON = CONTROL + HOLD
MACROS = HORIZON // BLOCK
CONTROL_MACROS = CONTROL // BLOCK
OBS_DIM = 36
METRICS = ['revenue','volume','completed_jobs','completed_value','total_user_utility',
           'active_buyers','underreport_count_active','overreport_count']
PRIMARY = np.array(PRIMARY_ACTION_INDICES)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM,64), nn.Tanh())
        self.gru = nn.GRU(64,64)
        self.actor = nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,4))
        self.critic = nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        nn.init.orthogonal_(self.critic[-1].weight,1.)

    def forward(self,obs,hidden=None):
        z, hidden = self.gru(self.encoder(obs),hidden)
        return self.actor(z),self.critic(z).squeeze(-1),hidden


def public_features(action, observations, next_macro, posterior=None, mode='belief'):
    b=len(action)
    result=np.zeros((b,OBS_DIM),np.float32)
    result[:,18]=next_macro/CONTROL_MACROS
    result[:,19]=float(next_macro<CONTROL_MACROS)
    result[:,20:]=1/16 if posterior is None else posterior
    if mode=='openloop':
        result[:,20:]=1/16
        return result
    if not observations:
        return result
    result[np.arange(b),action]=1
    reports=np.stack([o['reports'] for o in observations],axis=1)
    for i in range(2):
        for r in range(4):
            result[:,4+i*4+r]=(reports[:,:,i]==r).mean(axis=1)
    result[:,12:14]=np.mean([o['allocated_capacities'] for o in observations],axis=0)/3
    result[:,14:16]=np.mean([o['payments'] for o in observations],axis=0)/3
    result[:,16]=np.mean([o['revenue'] for o in observations],axis=0)/5
    result[:,17]=np.mean([o['volume'] for o in observations],axis=0)/5
    return result


def exact_response(env, actions, u):
    """Evaluation-only myopic exact grand-coalition response; pessimistic ties."""
    mid=env._market
    counts=TABLES['counts'][actions]
    slots=np.arange(TABLES['owners'].shape[2])[None,None,:]
    mask=slots<counts[:,:,None]
    utility=TABLES['utility'][mid[:,None,None],1,actions[:,None,None],
                              np.arange(16)[None,:,None],slots].sum(axis=-1)
    revenue=TABLES['revenue'][actions]
    eu=(utility*mask).sum(axis=-1)/counts
    er=(revenue*mask).sum(axis=-1)/counts
    winners=eu>=eu.max(axis=-1,keepdims=True)-1e-10
    pessimistic=np.where(winners,er,np.inf)
    bestrev=pessimistic.min(axis=-1,keepdims=True)
    eligible=winners&(er<=bestrev+1e-10)
    # Deterministic report tie-breaking, recorded as part of this attack model.
    joint=eligible.argmax(axis=-1)
    k=np.floor(u[:,2]*TABLES['counts'][actions,joint]).astype(int)
    phi,qi,qg=env._predictions(actions)
    return env._realize_and_update(actions,joint,k,phi,qi,qg)


def run_episodes(mode,model_ids,stream_seed,policy=None,fixed_action=None,
                 learner_mode='canonical',stochastic_policy=False,policy_seed=0,
                 collect_training=False,service_lambda=0.,target_fraction=0.,
                 response='learning'):
    model_ids=np.asarray(model_ids,int)
    b=len(model_ids)
    env=BatchMarket(model_ids,learner_mode=learner_mode)
    use_belief=mode in ('belief','bayes_greedy','fete_fixed')
    tracker=BeliefTracker(b,learner_mode=learner_mode) if use_belief else None
    rng=np.random.default_rng(stream_seed)
    torch.manual_seed(policy_seed)
    totals={k:np.zeros(b) for k in METRICS}
    phase_totals=np.zeros((b,2,2)) # phase, revenue/volume
    trajectories=np.zeros((MACROS,b,2),np.float32)
    action_history=np.zeros((MACROS,b),np.int8)
    obs_np=public_features(np.zeros(b,int),[],0,
                           tracker.posterior if tracker else None,mode)
    hidden=None
    last_action=np.zeros(b,int)
    training={key:[] for key in ('obs','action','logprob','value','reward')}
    # A simple nonstationary feedback baseline, fit from observed block revenue.
    bandit_sums=np.zeros((b,4)); bandit_counts=np.zeros((b,4))
    for macro in range(MACROS):
        active=macro<CONTROL_MACROS
        x=torch.from_numpy(obs_np)
        if policy is not None:
            with torch.no_grad():
                logits,values,hidden=policy(x[None],hidden)
                distribution=Categorical(logits=logits[0])
                sampled=distribution.sample() if stochastic_policy else logits[0].argmax(-1)
                if active:
                    action=sampled.numpy()
                else:
                    action=last_action.copy()
                lp=distribution.log_prob(torch.from_numpy(action)).numpy()
                value=values[0].numpy()
        else:
            lp=np.zeros(b);value=np.zeros(b)
            if not active:
                action=last_action.copy()
            elif mode=='fixed':
                action=np.broadcast_to(fixed_action,(b,)).copy().astype(int)
            elif mode in ('bayes_greedy','fete_fixed'):
                env_a=np.asarray(tracker.recommend(mode),int)
                action=np.searchsorted(PRIMARY,env_a)
            elif mode=='ucb':
                if macro<4:
                    action=np.full(b,macro)
                else:
                    means=bandit_sums/np.maximum(bandit_counts,1)
                    bonus=.2*np.sqrt(np.log(macro+1)/np.maximum(bandit_counts,1))
                    action=(means+(bonus if macro<CONTROL_MACROS-1 else 0)).argmax(-1)
            else:
                raise ValueError(mode)
        if active:
            last_action=action.copy()
        env_actions=PRIMARY[action]
        observed=[]
        rev=np.zeros(b);vol=np.zeros(b)
        for _ in range(BLOCK):
            u=rng.random((b,3))
            if response=='learning':
                observation,metrics=env.step(env_actions,u)
            elif response=='exact_grand':
                observation,metrics=exact_response(env,env_actions,u)
            else:
                raise ValueError(response)
            # During evaluation no later action can change after CONTROL;
            # retain the end-of-control posterior instead of updating an unused
            # hypothesis bank for another 200 (or sensitivity 1000) rounds.
            # Training still supplies full history to the hold-phase critic.
            if tracker is not None and (active or collect_training):
                tracker.observe(env_actions,observation['reports'],observation['owners'])
            observed.append(observation)
            rev+=observation['revenue'];vol+=observation['volume']
            for key in METRICS:
                totals[key]+=observation[key] if key in observation else metrics[key]
        bandit_sums[np.arange(b),action]+=rev/(BLOCK*5)
        bandit_counts[np.arange(b),action]+=1
        phase_totals[:,int(not active),0]+=rev
        phase_totals[:,int(not active),1]+=vol
        trajectories[macro,:,0]=rev/BLOCK
        trajectories[macro,:,1]=vol/BLOCK
        action_history[macro]=action
        if collect_training:
            training['obs'].append(obs_np.copy())
            training['action'].append(action)
            training['logprob'].append(lp)
            training['value'].append(value)
            training['reward'].append(rev/(BLOCK*5)+service_lambda*(vol/(BLOCK*5)-target_fraction))
        obs_np=public_features(action,observed,macro+1,
                               tracker.posterior if tracker else None,mode)
    result={'totals':totals,'phase_totals':phase_totals,'trajectories':trajectories,
            'actions':action_history,'models':model_ids}
    if tracker is not None:
        result['posterior']=tracker.posterior.copy()
    if collect_training:
        result['training']={k:np.stack(v) for k,v in training.items()}
    return result


def save_result(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,models=result['models'],metric_names=np.array(METRICS),
        metrics=np.stack([result['totals'][k] for k in METRICS],axis=-1),
        phase_totals=result['phase_totals'],trajectories=result['trajectories'],
        actions=result['actions'],**({'posterior':result['posterior']} if 'posterior' in result else {}))


def baseline_split(split,worlds,seed,learner_mode='canonical',response='learning'):
    if split=='validation' and (OUT/'pretest_freeze.json').exists():
        raise RuntimeError('This experiment is frozen; validation selection must not be overwritten.')
    ids=np.repeat(np.arange(16),worlds)
    if response=='exact_grand':
        ids=np.repeat(np.array([2,3,6,7,10,11,14,15]),worlds)
    directory=OUT/'data'/split
    records=[]
    for a in range(4):
        r=run_episodes('fixed',ids,seed,fixed_action=a,learner_mode=learner_mode,response=response)
        save_result(directory/f'fixed_{a}.npz',r)
        records.append({'action':a,'rho':float(RHO_VALUES[PRIMARY[a]]),
                        'revenue':float(r['totals']['revenue'].mean()),
                        'volume':float(r['totals']['volume'].mean())})
    if split=='validation':
        q0=records[0]['volume'];target=.95*q0
        feasible=[r for r in records if r['volume']>=target]
        best=max(feasible,key=lambda x:x['revenue'])
        unconstrained=max(records,key=lambda x:x['revenue'])
        matrices=np.stack([np.load(directory/f'fixed_{a}.npz')['metrics'] for a in range(4)])
        bymodel=matrices.reshape(4,16,worlds,len(METRICS)).mean(axis=2)
        permitted=bymodel[:,:,1]>=.95*bymodel[0,:,1][None,:]
        oracle=np.where(permitted,bymodel[:,:,0],-np.inf).argmax(axis=0)
        metadata={'fixed_records':records,'best_fixed_action':best['action'],
            'best_unconstrained_action':unconstrained['action'],
            'target_volume':target,'target_volume_fraction_per_step':target/(HORIZON*5),
            'original_volume':q0,'oracle_per_type_actions':oracle.tolist(),
            'oracle_unconstrained_actions':bymodel[:,:,0].argmax(axis=0).tolist(),
            'worlds_per_model':worlds,'stream_seed':seed}
        (OUT/'validation_selection.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'split':split,'fixed':records}),flush=True)
    return records


def validate_policy(policy,mode,seed,worlds=100):
    ids=np.repeat(np.arange(16),worlds)
    r=run_episodes(mode,ids,300017,policy=policy,policy_seed=seed)
    return float(r['totals']['revenue'].mean()),float(r['totals']['volume'].mean())


def train(mode,seed,steps=2_000_000,batch=128,epochs=10,validation_worlds=100):
    assert mode in ('belief','history','openloop')
    if (OUT/'pretest_freeze.json').exists():
        raise RuntimeError('This experiment is frozen; use a fresh run before training new checkpoints.')
    OUT.mkdir(parents=True,exist_ok=True)
    selection=json.loads((OUT/'validation_selection.json').read_text(encoding='utf-8'))
    target=selection['target_volume_fraction_per_step']
    target_volume=selection['target_volume']
    torch.manual_seed(seed+70001)
    policy=Policy();optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    generator=np.random.default_rng(seed+900001)
    iterations=max(1,steps//(HORIZON*batch))
    lam=0.;best_rank=None;logs=[];start=time.perf_counter()
    folder=OUT/'checkpoints';folder.mkdir(exist_ok=True)
    path=folder/f'{mode}_{seed}.pt'
    for iteration in range(iterations):
        ids=np.tile(np.arange(16),math.ceil(batch/16))[:batch].copy();generator.shuffle(ids)
        roll=run_episodes(mode,ids,int(generator.integers(1,2**31)),policy,
            stochastic_policy=True,policy_seed=seed*10000+iteration,
            collect_training=True,service_lambda=lam,target_fraction=target)
        t=roll['training'];values=t['value'];rewards=t['reward']
        advantage=np.zeros_like(rewards);last=np.zeros(batch)
        for k in reversed(range(MACROS)):
            future=values[k+1] if k+1<MACROS else 0
            delta=rewards[k]+future-values[k]
            last=delta+.95*last
            advantage[k]=last
        returns=advantage+values
        active_adv=advantage[:CONTROL_MACROS]
        advantage=(advantage-active_adv.mean())/(active_adv.std()+1e-8)
        obs=torch.from_numpy(t['obs']);act=torch.from_numpy(t['action'])
        oldlog=torch.from_numpy(t['logprob']).float()
        adv=torch.from_numpy(advantage).float();ret=torch.from_numpy(returns).float()
        policy.train()
        losses=[]
        for epoch in range(epochs):
            order=generator.permutation(batch)
            for lo in range(0,batch,32):
                indices=order[lo:lo+32]
                logits,value,_=policy(obs[:,indices])
                dist=Categorical(logits=logits[:CONTROL_MACROS])
                log=dist.log_prob(act[:CONTROL_MACROS,indices])
                ratio=(log-oldlog[:CONTROL_MACROS,indices]).exp()
                a=adv[:CONTROL_MACROS,indices]
                actor_loss=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
                value_loss=.5*((value-ret[:,indices])**2).mean()
                loss=actor_loss+.5*value_loss-.01*dist.entropy().mean()
                optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
                losses.append(float(loss.detach()))
        volume=float(roll['totals']['volume'].mean())
        lam=max(0.,min(10.,lam+.05*(target-volume/(HORIZON*5))))
        log={'iteration':iteration+1,'actual_market_steps':(iteration+1)*batch*HORIZON,
             'train_revenue':float(roll['totals']['revenue'].mean()),'train_volume':volume,
             'service_lambda':lam,'loss':float(np.mean(losses)),
             'elapsed_seconds':time.perf_counter()-start}
        if (iteration+1)%5==0 or iteration+1==iterations:
            policy.eval()
            vr,vq=validate_policy(policy,mode,seed,worlds=validation_worlds)
            feasible=vq>=target_volume
            rank=(int(feasible),vr if feasible else vq,vr)
            log.update(validation_revenue=vr,validation_volume=vq,validation_feasible=feasible)
            if best_rank is None or rank>best_rank:
                best_rank=rank
                torch.save({'state_dict':policy.state_dict(),'mode':mode,'seed':seed,
                    'selected_iteration':iteration+1,'validation_revenue':vr,'validation_volume':vq,
                    'target_volume':target_volume,'canonical_primary':True,
                    'actual_market_steps':(iteration+1)*batch*HORIZON},path)
        logs.append(log)
        if 'validation_revenue' in log or iteration==0:
            print(json.dumps({'mode':mode,'seed':seed,**log}),flush=True)
    logpath=OUT/'logs'/f'{mode}_{seed}.json';logpath.parent.mkdir(exist_ok=True)
    logpath.write_text(json.dumps(logs,indent=2),encoding='utf-8')
    return path


def evaluate_saved(mode,seed,split='test',worlds=100,stream_seed=800021,
                   learner_mode='canonical',response='learning'):
    checkpoint=torch.load(OUT/'checkpoints'/f'{mode}_{seed}.pt',weights_only=False)
    policy=Policy();policy.load_state_dict(checkpoint['state_dict']);policy.eval()
    ids=np.repeat(np.arange(16),worlds)
    if response=='exact_grand':
        ids=np.repeat(np.array([2,3,6,7,10,11,14,15]),worlds)
    result=run_episodes(mode,ids,stream_seed,policy=policy,policy_seed=seed,
                        learner_mode=learner_mode,response=response)
    save_result(OUT/'data'/split/f'{mode}_{seed}.npz',result)
    print(json.dumps({'evaluation':split,'mode':mode,'seed':seed,
         **{k:float(v.mean()) for k,v in result['totals'].items()}}),flush=True)


def evaluate_baseline_policy(mode,split='test',worlds=100,stream_seed=800021,
                             learner_mode='canonical',response='learning'):
    ids=np.repeat(np.arange(16),worlds)
    if response=='exact_grand':ids=np.repeat(np.array([2,3,6,7,10,11,14,15]),worlds)
    if mode in ('oracle_fixed','oracle_unconstrained'):
        sel=json.loads((OUT/'validation_selection.json').read_text())
        field='oracle_per_type_actions' if mode=='oracle_fixed' else 'oracle_unconstrained_actions'
        actions=np.array(sel[field])[ids]
        result=run_episodes('fixed',ids,stream_seed,fixed_action=actions,
                            learner_mode=learner_mode,response=response)
    else:
        result=run_episodes(mode,ids,stream_seed,learner_mode=learner_mode,response=response)
    save_result(OUT/'data'/split/f'{mode}.npz',result)
    print(json.dumps({'evaluation':split,'mode':mode,
        **{k:float(v.mean()) for k,v in result['totals'].items()}}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['baselines','train','evaluate','simple'])
    parser.add_argument('--mode',default='belief');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=2_000_000)
    parser.add_argument('--split',default='validation')
    parser.add_argument('--worlds',type=int,default=100)
    parser.add_argument('--stream-seed',type=int,default=None)
    parser.add_argument('--learner-mode',default='canonical')
    parser.add_argument('--response',default='learning')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stream_seed is None:
        args.stream_seed=300017 if args.split=='validation' else 800021
    if args.split!='validation' and args.command!='train' and args.stream_seed==300017:
        raise ValueError('Validation stream cannot be reused for a test split')
    if args.command=='baselines':baseline_split(args.split,args.worlds,args.stream_seed,args.learner_mode,args.response)
    elif args.command=='train':train(args.mode,args.seed,args.steps)
    elif args.command=='evaluate':evaluate_saved(args.mode,args.seed,args.split,args.worlds,args.stream_seed,args.learner_mode,args.response)
    else:evaluate_baseline_policy(args.mode,args.split,args.worlds,args.stream_seed,args.learner_mode,args.response)


if __name__=='__main__':main()
