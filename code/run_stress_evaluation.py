"""Out-of-training response/long-hold sensitivity; frozen platform weights."""
import argparse
import json
import run_threshold_experiment as experiment

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('case',choices=['exact','tabular','raw','longhold'])
    parser.add_argument('--seed',type=int,default=-1)
    args=parser.parse_args()
    if not (experiment.OUT/'pretest_freeze.json').exists():
        raise RuntimeError('Freeze all checkpoints and analysis before stress evaluation')
    kwargs={'split':'stress_'+args.case,'worlds':100,'stream_seed':1800021}
    if args.case=='exact':kwargs['response']='exact_grand'
    if args.case in ('tabular','raw'):kwargs['learner_mode']=args.case
    if args.case=='longhold':
        experiment.HOLD=1000;experiment.HORIZON=1200;experiment.MACROS=120
    if args.seed<0:
        experiment.baseline_split(kwargs['split'],kwargs['worlds'],kwargs['stream_seed'],
            kwargs.get('learner_mode','canonical'),kwargs.get('response','learning'))
    else:
        experiment.evaluate_saved('belief',args.seed,**kwargs)
