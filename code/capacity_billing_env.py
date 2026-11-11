"""V2 pricing experiment; V1 allocation, learning and sharing are unchanged.

The posted price is fixed before the episode. With capacity_weight=1 the bill
depends only on the VM actually allocated, never on its requested minimum.
Reward normalization stays at the V1 monetary units (3 / 6), so changing prices
does not secretly change the learners' exploration temperature.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from steering_env import BatchMarket, TABLES, JOINT_REPORTS, VALUES, MODEL_COUNT
from belief_baselines import BeliefTracker


@lru_cache(None)
def pricing_tables(unit_price=.78, capacity_weight=1.):
    if not (unit_price>0 and 0<=capacity_weight<=1):
        raise ValueError('Invalid posted price or capacity weight')
    result=dict(TABLES)
    if unit_price==1 and capacity_weight==.2:
        return result
    allocated=TABLES['allocation']
    reports=JOINT_REPORTS[None,:,None,:]
    pay=unit_price*(capacity_weight*allocated+(1-capacity_weight)*reports)*(allocated>0)
    result['payment']=pay
    result['revenue']=pay.sum(-1)
    result['utility']=TABLES['served']*VALUES[:,None,None,None,None,:]-pay[None,None]
    for name in ('payment','revenue','utility'):result[name].setflags(write=False)
    return result


class CapacityMarket(BatchMarket):
    def __init__(self,model_ids,seed=0,learner_mode='canonical',unit_price=.78,capacity_weight=1.):
        self.unit_price=float(unit_price)
        self.capacity_weight=float(capacity_weight)
        self.tables=pricing_tables(self.unit_price,self.capacity_weight)
        super().__init__(model_ids,seed=seed,learner_mode=learner_mode)

    def _realize_and_update(self,a,joint,outcome,phi,qi,qg):
        observation,metrics=super()._realize_and_update(a,joint,outcome,phi,qi,qg)
        old=observation['payments']
        new=self.tables['payment'][a,joint,outcome].copy()
        correction=old-new
        rows=np.flatnonzero(self._coalition==0)
        if len(rows):
            reports=observation['reports'][rows]
            self._w_independent[rows[:,None],np.arange(2)[None,:],reports] += (
                self._eta[rows,None,None]*correction[rows,:,None]/3.*phi[rows,None,:])
        rows=np.flatnonzero(self._coalition==1)
        if len(rows):
            self._w_grand[rows,joint[rows]] += (
                self._eta[rows,None]*correction[rows].sum(-1)[:,None]/6.*phi[rows])
        observation['payments']=new
        observation['revenue']=new.sum(-1)
        metrics['utilities_before_transfers']+=correction
        metrics['total_user_utility']+=correction.sum(-1)
        # This measures evasion relative to the advertised tariff for the same
        # physical allocation, not whether a report equals a private demand.
        metrics['same_capacity_payment_gap']=(self.unit_price*observation['volume']
                                             -observation['revenue'])
        self._last_obs={key:value.copy() if isinstance(value,np.ndarray) else value
                        for key,value in observation.items()}
        return observation,metrics

    def clone(self):
        return CapacityMarket(self._model_ids,learner_mode=self.learner_mode,
            unit_price=self.unit_price,capacity_weight=self.capacity_weight).set_state(self.get_state())


class CapacityBeliefTracker(BeliefTracker):
    def __init__(self,batch_size,learner_mode='canonical',unit_price=.78,capacity_weight=1.,**kwargs):
        self.unit_price=unit_price;self.capacity_weight=capacity_weight
        super().__init__(batch_size,learner_mode=learner_mode,**kwargs)

    def reset(self):
        super().reset()
        self._candidates=CapacityMarket(np.tile(np.arange(MODEL_COUNT),self.batch_size),
            learner_mode=self.learner_mode,unit_price=self.unit_price,capacity_weight=self.capacity_weight)
        return self

    def predict_revenue(self):
        table=pricing_tables(self.unit_price,self.capacity_weight)
        mask=np.arange(table['revenue'].shape[-1])[None,None,:]<table['counts'][...,None]
        expected=(table['revenue']*mask).sum(-1)/table['counts']
        return self._predict_table(expected)


def self_test():
    rng=np.random.default_rng(9041)
    legacy=BatchMarket(np.arange(16));copy=CapacityMarket(np.arange(16),unit_price=1,capacity_weight=.2)
    for _ in range(50):
        actions=rng.choice([0,1,2,4],16);u=rng.random((16,3))
        a,ma=legacy.step(actions,u);b,mb=copy.step(actions,u)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        for key in ma:np.testing.assert_array_equal(ma[key],mb[key])
    env=CapacityMarket(np.arange(16));tracker=CapacityBeliefTracker(16)
    for _ in range(50):
        actions=rng.choice([0,1,2,4],16)
        obs,metrics=env.step(actions,rng.random((16,3)))
        np.testing.assert_allclose(obs['payments'],.78*obs['allocated_capacities'],atol=1e-12)
        np.testing.assert_allclose(metrics['total_user_utility']+obs['revenue'],metrics['completed_value'],atol=1e-12)
        tracker.observe(actions,obs['reports'],obs['owners'])
    candidates=tracker._candidates
    for i in range(16):
        np.testing.assert_allclose(candidates._w_grand[i*16+i],env._w_grand[i],atol=1e-12)
        np.testing.assert_allclose(candidates._w_independent[i*16+i],env._w_independent[i],atol=1e-12)
    clone=env.clone();u=rng.random((16,3));actions=np.ones(16,dtype=int)
    np.testing.assert_array_equal(env.step(actions,u)[0]['revenue'],clone.step(actions,u)[0]['revenue'])
    print('PASS: V1 exact replay; capacity billing; cash conservation; candidate replay; clone; fixed monetary normalization.')


if __name__=='__main__':self_test()
