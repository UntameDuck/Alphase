import argparse
import json
from pathlib import Path
from .data import extract_archives,inspect,save_json
from .config import load_config

def main():
    p=argparse.ArgumentParser(prog='alphase');sub=p.add_subparsers(dest='command',required=True)
    def cmd(name,data=False,output=False):
        a=sub.add_parser(name)
        if data:a.add_argument('--data',required=True)
        if output:a.add_argument('--output',required=True)
        return a
    a=cmd('prepare',True);a.add_argument('--archives',required=True)
    a=cmd('inspect',True,True);a.add_argument('--official-counts',action='store_true')
    a=cmd('morphology',True,True)
    a=cmd('train',True,True);a.add_argument('--config',required=True);a.add_argument('--resume');a.add_argument('--steps',type=int);a.add_argument('--auto-tune',action='store_true')
    a=cmd('tune',output=True);a.add_argument('--config',required=True);a.add_argument('--patches',type=int,nargs='+',default=[256,512]);a.add_argument('--batches',type=int,nargs='+',default=[2,4,8])
    a=cmd('benchmark',True,True);a.add_argument('--config',required=True);a.add_argument('--warmup',type=int,default=2);a.add_argument('--batches',type=int,default=10)
    for name in ['evaluate','infer','repeat-check']:
        a=cmd(name,True,True);a.add_argument('--checkpoint',nargs='+',required=True);a.add_argument('--device',default='auto');a.add_argument('--weights',type=float,nargs='+');a.add_argument('--tta',action='store_true')
        a.add_argument('--tta-mode',choices=['none','flip','d4'])
        if name=='evaluate':
            a.add_argument('--absent-policy',choices=['exclude','zero','one']);a.add_argument('--cache',action='store_true');a.add_argument('--tolerance',type=int,default=2);a.add_argument('--small-area',type=int,default=64)
        else:a.add_argument('--contract')
    a=cmd('experiment-plan',output=True);a.add_argument('--config',required=True);a.add_argument('--ids',nargs='+',default=['A','D','E','F','G']);a.add_argument('--seeds',type=int,nargs='+',default=[42]);a.add_argument('--allow-pretrained',action='store_true')
    a=cmd('experiments',True,True);a.add_argument('--plan',required=True);a.add_argument('--steps',type=int)
    a=cmd('compare',output=True);a.add_argument('--plan',required=True);a.add_argument('--runs',required=True)
    a=cmd('ensemble-search',output=True);a.add_argument('--caches',nargs='+',required=True);a.add_argument('--denominator',type=int,default=4);a.add_argument('--absent-policy',default='exclude',choices=['exclude','zero','one'])
    a=cmd('postprocess',output=True);a.add_argument('--cache',required=True);a.add_argument('--min-area',type=int,default=8);a.add_argument('--confidence',type=float,default=.6);a.add_argument('--absent-policy',default='exclude')
    a=cmd('validate-output',True,True);a.add_argument('--predictions',required=True);a.add_argument('--contract')
    a=cmd('package',True,True);a.add_argument('--predictions',required=True);a.add_argument('--checkpoint',nargs='+',required=True);a.add_argument('--report',required=True);a.add_argument('--receipt',required=True);a.add_argument('--contract',required=True);a.add_argument('--weights',type=float,nargs='+');a.add_argument('--tta',action='store_true')
    a.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    a.add_argument('--tta-mode',choices=['none','flip','d4'])
    a=cmd('verify-package',True,True);a.add_argument('--archive',required=True);a.add_argument('--python');a.add_argument('--device',default='cpu')
    a=cmd('report-draft',output=True);a.add_argument('--data-report',required=True);a.add_argument('--evaluation',required=True);a.add_argument('--comparison',required=True)
    a=cmd('budget',output=True);a.add_argument('--run',required=True);a.add_argument('--hourly-rate',type=float,required=True)
    args=p.parse_args();name=args.command
    if hasattr(args,'tta_mode') and args.tta_mode is not None:
        if args.tta and args.tta_mode!='flip':p.error('--tta means flip; do not combine with a different --tta-mode')
        args.tta=args.tta_mode
    if name=='prepare':extract_archives(args.archives,args.data);return
    if name=='inspect':
        r=inspect(args.data,{'train':70,'valid':20,'test':10} if args.official_counts else None);save_json(args.output,r);print({k:v['count'] for k,v in r['splits'].items()});return
    if name=='morphology':
        from .morphology import audit_train
        r=audit_train(args.data);save_json(args.output,r);print({'split':r['split'],'image_count':r['image_count']});return
    if name=='budget':
        from .budget import estimate
        r=estimate(args.run,args.hourly_rate,args.output)
    elif name=='report-draft':
        from .reporting import draft
        r=draft(args.data_report,args.evaluation,args.comparison,args.output)
    elif name in ['experiment-plan','experiments','compare']:
        from . import experiments
        if name=='experiment-plan':r=experiments.plan(load_config(args.config),args.output,args.seeds,args.ids,args.allow_pretrained)
        elif name=='experiments':r=experiments.run(args.plan,args.data,args.output,args.steps)
        else:r=experiments.compare(args.plan,args.runs,args.output)
    elif name=='ensemble-search':
        from .ensemble import search
        r=search(args.caches,args.output,args.denominator,args.absent_policy)
    elif name=='postprocess':
        from .postprocess import evaluate_cache
        r=evaluate_cache(args.cache,args.output,args.min_area,args.confidence,args.absent_policy)
    elif name in ['validate-output','package','verify-package']:
        from . import submission
        if name=='validate-output':r=submission.validate_predictions(args.data,args.predictions,submission.load_contract(args.contract))
        elif name=='package':r=submission.build_package(args.data,args.predictions,args.checkpoint,args.report,args.receipt,args.output,args.contract,args.weights,args.tta,args.device)
        else:
            kw={'python':args.python} if args.python else {}
            r=submission.verify_package(args.archive,args.data,device=args.device,**kw)
        if name!='package':save_json(args.output,r)
    else:
        from . import engine
        if name=='train':
            cfg=load_config(args.config)
            if args.steps is not None and args.steps<1:raise ValueError('--steps must be positive')
            if args.auto_tune:
                if args.resume:raise ValueError('Do not change memory settings when resuming')
                from .resources import tune
                cfg,probe=tune(cfg);save_json(Path(args.output)/'resource_probe.json',probe)
            engine.train(args.data,cfg,args.output,args.resume,args.steps);return
        elif name=='benchmark':
            from .resources import benchmark
            r=benchmark(args.data,load_config(args.config),args.warmup,args.batches);save_json(args.output,r)
        elif name=='tune':
            from .resources import tune
            cfg,r=tune(load_config(args.config),args.patches,args.batches);save_json(args.output,cfg);save_json(str(args.output)+'.probe.json',r)
        elif name=='evaluate':
            predictor=engine.Predictor(args.checkpoint,args.device,args.weights,args.tta)
            policies={cfg['absent_policy'] for _,cfg in predictor.loaded}
            if args.absent_policy is None and len(policies)!=1:raise ValueError('Ensemble checkpoints have different absent policies')
            absent=args.absent_policy or policies.pop()
            r=engine.evaluate(predictor,args.data,args.output,absent,args.tolerance,args.small_area,args.cache)
        elif name=='infer':engine.infer(args.data,args.checkpoint,args.output,args.device,args.weights,args.tta,args.contract);return
        else:
            from .submission import reproducibility_check
            r=reproducibility_check(args.data,args.checkpoint,args.output,args.device,args.weights,args.tta,args.contract)
    print(json.dumps({k:v for k,v in r.items() if k in ['status','selected','image_macro_miou','best','suggested_variant','count','zip']},ensure_ascii=False))

if __name__=='__main__':main()
