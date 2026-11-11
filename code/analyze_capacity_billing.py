"""Frozen V2 analysis: revenue improvement with no loss in jobs or value.

All ten trained/fallback seeds are retained. Reference choice uses validation
only. A shared crossed bootstrap resamples training seeds and within-type
worlds; the finite type catalogue itself is not resampled. Exact responses use
four markets, without duplicate learning-rate labels.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_variable, '1')
import numpy as np
from analyze_threshold_experiment import crossed_bootstrap, estimate, write_json, write_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'v2_capacity_billing'
METHODS=('old_fixed','fixed_0','fixed_1','fixed_2','fixed_3','dynamic')
CASES=('canonical','tabular','exact')
SEEDS=tuple(range(10))
METRICS=('revenue','volume','completed_jobs','completed_value','total_user_utility',
         'active_buyers','underreport_count_active','overreport_count')
REPLICATES=10_000
BOOTSTRAP_SEED=2026111301
ROUNDING_TOLERANCE=1e-9


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    required=[OUT/'protocol.json',OUT/'selection.json']
    for path in required:
        if not path.exists():raise FileNotFoundError(f'Prepare validation first: {path}')
    document={'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'analysis_frozen_before_main_test',
        'script_sha256':digest(__file__),
        'input_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in required},
        'replicates':REPLICATES,'bootstrap_seed_base':BOOTSTRAP_SEED,
        'mandatory_seeds':SEEDS,'mandatory_cases':CASES,'methods':METHODS,
        'primary_endpoints':['net receipts','completed jobs','completed job value'],
        'strict_joint_rule':'R lower95 > 0, J lower95 >= 0, V lower95 >= 0',
        'rounding_tolerance_only':ROUNDING_TOLERANCE,
        'reference_selection':'selection.json, from validation; no test reselection',
        'attribution':['new fixed versus old fixed','dynamic versus old fixed','dynamic versus new fixed'],
        'sampling':'paired crossed training-seed and within-fixed-type world bootstrap; types not resampled',
        'test_data_read_to_freeze':False,
        'other_methods_cases_groups':'All disaggregated effects reported; only primary canonical overall contrasts have the prespecified main interpretation. Stress and subgroup claims remain conditional/explanatory.'}
    destination=OUT/'analysis_freeze.json'
    if destination.exists():
        previous=json.loads(destination.read_text(encoding='utf-8'))
        if previous['script_sha256']!=document['script_sha256'] or previous['input_sha256']!=document['input_sha256']:
            raise RuntimeError('Existing analysis freeze differs; append an amendment rather than silently replacing it.')
        return previous
    if any((OUT/'data'/case).exists() for case in CASES):
        raise RuntimeError('Test directory already exists: cannot create a before-test freeze retrospectively.')
    write_json(destination,document)
    print('Frozen V2 analysis:',document['script_sha256'],flush=True)
    return document


def load_case(case):
    expected_types=np.array([2,6,10,14]) if case=='exact' else np.arange(16)
    arrays={};hashes={};reference_models=None;reference_worlds=None
    for method in METHODS:
        names=[f'dynamic_{seed}.npz' for seed in SEEDS] if method=='dynamic' else [f'{method}.npz']
        pieces=[]
        for name in names:
            path=OUT/'data'/case/name
            with np.load(path,allow_pickle=False) as data:
                assert tuple(str(k) for k in data['metric_names'])==METRICS,(path,'metric names')
                model_ids=data['models'].astype(int);values=data['metrics'].astype(float)
                phase=data['phase_totals'].astype(float);actions=data['actions']
                trajectories=data['trajectories'].astype(float)
            assert len(model_ids)%len(expected_types)==0,(path,'unequal strata')
            worlds=len(model_ids)//len(expected_types)
            np.testing.assert_array_equal(model_ids,np.repeat(expected_types,worlds))
            if reference_models is None:
                reference_models=model_ids;reference_worlds=worlds
            np.testing.assert_array_equal(model_ids,reference_models)
            assert worlds==reference_worlds
            assert values.shape==(len(model_ids),len(METRICS))
            assert actions.shape==(40,len(model_ids))
            assert trajectories.shape==(40,len(model_ids),2)
            assert phase.shape==(len(model_ids),2,2)
            assert np.isfinite(values).all()
            np.testing.assert_allclose(phase.sum(axis=1),values[:,:2],atol=2e-8,rtol=1e-10)
            np.testing.assert_allclose(trajectories.sum(axis=0)*10,values[:,:2],atol=.002,rtol=2e-6)
            np.testing.assert_allclose(values[:,4]+values[:,0],values[:,3],atol=2e-8,rtol=1e-10)
            np.testing.assert_array_equal(actions[20:],np.broadcast_to(actions[19],actions[20:].shape))
            assert np.all((values[:,2]>=0)&(values[:,2]<=800))
            pieces.append(values.reshape(len(expected_types),worlds,len(METRICS)))
            hashes[path.relative_to(ROOT).as_posix()]=digest(path)
        arrays[method]=np.stack(pieces)
    return arrays,expected_types,hashes


def groups_for(types):
    groups={'all':np.arange(len(types)),
            'coalition':np.flatnonzero((types//2)%2==1)}
    independent=np.flatnonzero((types//2)%2==0)
    if len(independent):groups['independent']=independent
    for market in range(4):
        groups[f'market_{"ABCD"[market]}']=np.flatnonzero(types//4==market)
    return {name:ids for name,ids in groups.items() if len(ids)}


def comparison(method,reference,indices,points,bootstraps):
    point=points[method][indices].mean(0)-points[reference][indices].mean(0)
    draws=bootstraps[method][:,indices].mean(1)-bootstraps[reference][:,indices].mean(1)
    endpoint={METRICS[column]:estimate(point[column],draws[:,column]) for column in (0,2,3,1,4)}
    base=points[reference][indices,0].mean()
    base_draws=bootstraps[reference][:,indices,0].mean(1)
    endpoint['revenue_gain_percent']=estimate(100*point[0]/base,100*draws[:,0]/base_draws)
    passes={'revenue':endpoint['revenue']['one_sided95_low']>ROUNDING_TOLERANCE,
        'completed_jobs':endpoint['completed_jobs']['one_sided95_low']>=-ROUNDING_TOLERANCE,
        'completed_value':endpoint['completed_value']['one_sided95_low']>=-ROUNDING_TOLERANCE}
    return {'method':method,'reference':reference,'endpoints':endpoint,
            'criteria':passes,'joint_pass':all(passes.values())}


def analyze():
    frozen=json.loads((OUT/'analysis_freeze.json').read_text(encoding='utf-8'))
    if digest(__file__)!=frozen['script_sha256']:
        raise RuntimeError('Analysis source changed after freeze')
    for name,checksum in frozen['input_sha256'].items():
        if digest(ROOT/name)!=checksum:raise RuntimeError(f'Frozen input changed: {name}')
    pretest=json.loads((OUT/'pretest_freeze.json').read_text(encoding='utf-8'))
    frozen_script=pretest['hashes'].get(Path(__file__).relative_to(ROOT).as_posix())
    if frozen_script!=digest(__file__):raise RuntimeError('Execution manifest does not match analysis source')
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    selected=f'fixed_{selection["best_fixed_action"]}'
    rows=[];contrasts=[];seed_rows=[];type_rows=[];summaries={};all_hashes={}
    for case_index,case in enumerate(CASES):
        arrays,types,hashes=load_case(case);all_hashes.update(hashes)
        boots=crossed_bootstrap(arrays,replicates=REPLICATES,seed=BOOTSTRAP_SEED+case_index)
        points={name:values.mean(axis=(0,2)) for name,values in arrays.items()}
        groups=groups_for(types)
        case_summary={'fixed_type_ids':types.tolist(),'worlds_per_type':arrays['old_fixed'].shape[2],
                      'selected_new_fixed':selected,'methods':{},'comparisons':{}}
        for method in METHODS:
            method_summary={}
            for group,indices in groups.items():
                p=points[method][indices].mean(0);b=boots[method][:,indices].mean(1)
                stats={name:estimate(p[col],b[:,col]) for col,name in enumerate(METRICS)}
                stats['underreport_rate_among_active']=estimate(p[6]/p[5],b[:,6]/b[:,5])
                stats['exit_rate']=estimate(1-p[5]/800,1-b[:,5]/800)
                method_summary[group]=stats
                for metric,record in stats.items():
                    rows.append({'case':case,'method':method,'group':group,'metric':metric,**record})
                for seed in range(arrays[method].shape[0]):
                    sample=arrays[method][seed,indices].mean(axis=(0,1))
                    seed_rows.append({'case':case,'method':method,'training_seed':seed if method=='dynamic' else 'not_trained',
                        'group':group,**{name:float(sample[col]) for col,name in enumerate(METRICS)}})
            case_summary['methods'][method]=method_summary
            for position,model_id in enumerate(types):
                for column,name in enumerate(METRICS):
                    type_rows.append({'case':case,'method':method,'model_id':int(model_id),
                                      'metric':name,**estimate(points[method][position,column],boots[method][:,position,column])})
        for method,reference,label in ((selected,'old_fixed','mechanism'),
                                       ('dynamic','old_fixed','total'),
                                       ('dynamic',selected,'steering')):
            case_summary['comparisons'][label]={}
            for group,indices in groups.items():
                record=comparison(method,reference,indices,points,boots)
                case_summary['comparisons'][label][group]=record
                for metric,stat in record['endpoints'].items():
                    contrasts.append({'case':case,'attribution':label,'method':method,'reference':reference,
                        'group':group,'metric':metric,'joint_R_J_V_pass':record['joint_pass'],**stat})
        summaries[case]=case_summary
        print(json.dumps({'case':case,'total':case_summary['comparisons']['total']['all'],
                          'steering':case_summary['comparisons']['steering']['all']}),flush=True)
    destination=OUT/'analysis';destination.mkdir(exist_ok=True)
    for name,data in [('method_summary.csv',rows),('comparisons.csv',contrasts),
                      ('seed_results.csv',seed_rows),('type_results.csv',type_rows)]:
        write_csv(destination/name,data)
    result={'status':'complete','analysis_frozen':frozen,'input_sha256':all_hashes,
            'selection':selection,'cases':summaries,
            'scope':'Finite-catalogue conditional simulation evidence. Capacity billing does not identify truthful individual demand or guarantee arbitrary private types.',
            'receipt_identity':'Price is committed and positive; for capacity_weight=1 each payment equals posted_price times realized leased capacity, independent of reports conditional on that capacity.',
            'exact_theorem_reference':'code/v2_mechanism_audit.py; 0<p<.8, specified four markets and fully coordinated coalition. Dynamic and fixed thresholds then have identical receipts and completed jobs.'}
    write_json(destination/'summary.json',result)
    write_json(destination/'primary.json',{'canonical':summaries['canonical']['comparisons'],
        'same_mechanism_fixed_selection_feasible':selection['best_fixed_feasible'],
        'exact':summaries['exact']['comparisons'],
        'tabular':summaries['tabular']['comparisons']})
    print('V2 analysis written:',destination,flush=True)


def self_test():
    # Exact shifts preserve paired noise; zero service changes must not be
    # misclassified through machine-rounding error as service deterioration.
    rng=np.random.default_rng(672)
    old=rng.uniform(1,2,size=(1,4,15,8))
    new=np.repeat(old,10,axis=0)
    new[:,:,:,0]+=2
    arrays={'old':old,'new':new}
    boots=crossed_bootstrap(arrays,replicates=1000,seed=19)
    points={key:value.mean(axis=(0,2)) for key,value in arrays.items()}
    result=comparison('new','old',np.arange(4),points,boots)
    assert result['joint_pass']
    np.testing.assert_allclose(result['endpoints']['revenue']['estimate'],2,atol=1e-12)
    assert not comparison('old','old',np.arange(4),points,boots)['joint_pass']
    impaired=new.copy();impaired[:,:,:,2]-=.1
    arrays['impaired']=impaired
    boots=crossed_bootstrap(arrays,replicates=1000,seed=19)
    points={key:value.mean(axis=(0,2)) for key,value in arrays.items()}
    assert not comparison('impaired','old',np.arange(4),points,boots)['joint_pass']
    groups=groups_for(np.array([2,6,10,14]))
    assert 'independent' not in groups and len(groups['all'])==4
    print('PASS: exact paired shifts, zero-service tolerance, strict revenue criterion, service failure, four-market exact grouping; no experimental test data read.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    if args.freeze:freeze()
    if args.run:analyze()
    if not any((args.self_test,args.freeze,args.run)):parser.print_help()
