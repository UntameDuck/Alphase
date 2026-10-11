from pathlib import Path
import json
import csv
import subprocess
import sys
from .config import DEFAULT,load_config,resolved_config
from .data import save_json,sha256,pairs,read_rgb,decode_mask
from .metrics import paired_bootstrap
from .provenance import validate_evaluation,array_digest

PARENTS={'F':'E','G':'F','O':'F','L':'F','S':'F'}
BUDGET_KEYS=['patch_size','batch_size','patches_per_image','epochs','patience','lr','weight_decay',
    'brightness','absent_policy','pretrained','encoder_lr','warmup_epochs','min_epochs','accumulation_steps',
    'ema_decay','freeze_encoder_bn','brightness_probability','scale_range','dice_reduction','inference_amp',
    'amp','amp_dtype','overlap','normalization_mean','normalization_std']

VARIANTS={
 'A':dict(architecture='unet',encoder='resnet34',ce_weight=1.,dice_weight=0.),
 'B':dict(architecture='unet',encoder='timm-efficientnet-b3',ce_weight=1.,dice_weight=0.),
 'C':dict(architecture='unetplusplus',encoder='resnet34',ce_weight=1.,dice_weight=0.),
 'D':dict(architecture='unetplusplus',encoder='timm-efficientnet-b3',ce_weight=1.,dice_weight=0.),
 'E':{},'F':dict(sampler='phase_boundary'),'G':dict(sampler='phase_boundary',boundary_weight=.05),
 'CN':dict(architecture='unet',encoder='tu-convnext_tiny.fb_in1k',sampler='phase_boundary'),
 'O':dict(sampler='phase_boundary',phase_sampling='component'),
 'L':dict(sampler='phase_boundary',lovasz_weight=.2),
 'S':dict(sampler='phase_boundary',decoder_attention='scse'),
 'DL':dict(architecture='deeplabv3plus',encoder='resnet50'),
 'Focal':dict(loss_kind='focal_dice'),'Tversky':dict(loss_kind='ce_tversky'),
 'Random':dict(pretrained=False),'Pretrained':dict(pretrained=True)}

def plan(base,output,seeds=(42,),ids=('A','D','E','F','G'),allow_pretrained=False):
    if not seeds or not ids or len(set(seeds))!=len(seeds) or len(set(ids))!=len(ids):raise ValueError('Empty or duplicate experiment identifiers')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);runs=[]
    if any(output.iterdir()):raise ValueError('Use an empty plan directory')
    base=resolved_config(base)
    unknown=set(ids)-set(VARIANTS)
    if unknown:raise ValueError(f'Unknown variants: {sorted(unknown)}')
    canonical=dict(architecture='unetplusplus',encoder='timm-efficientnet-b3',ce_weight=.5,dice_weight=.5,
        loss_kind='ce_dice',class_weights=None,boundary_weight=0.,rare_probability=0.,sampler='uniform',
        phase_sampling='pixel',lovasz_weight=0.,decoder_attention=None)
    for ident in ids:
        if ident not in VARIANTS:raise ValueError(f'Unknown variant {ident}')
        for seed in seeds:
            cfg=resolved_config(base|canonical|VARIANTS[ident]|{'seed':seed})
            if cfg['pretrained'] and not allow_pretrained:raise ValueError('Pretrained use must be explicitly allowed after checking competition rules')
            if cfg['pretrained']:cfg['pretrained_permission']=True
            name=f'{ident}_s{seed}';config=output/(name+'.json');save_json(config,cfg)
            runs.append({'id':ident,'seed':seed,'config':config.name,'output':name})
    p={'runs':runs,'inference_variants':['single','flip_tta','validation_weight_search'],
        'run_tta':False,'run_ensemble':False,'minimum_gain':.005,'minimum_replication_seeds':3,
        'note':'TTA and ensemble search are opt-in after screening. E is uniform Core; F adds phase/boundary sampling; G adds classwise boundary loss; CN is the ConvNeXt/U-Net challenger. Inference comparisons and seed replication require matched conditions.'}
    save_json(output/'plan.json',p);return p

def run(plan_path,data,output,steps=None):
    if steps is not None and (type(steps) is not int or steps<1):raise ValueError('steps must be positive')
    path=Path(plan_path);p=json.loads(path.read_text(encoding='utf-8'));out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for item in p['runs']:
        dest=out/item['output'];config=path.parent/item['config']
        args=[sys.executable,'-m','alphase','train','--data',str(data),'--config',str(config),'--output',str(dest)]
        if (dest/'last.pt').exists():args+=['--resume',str(dest/'last.pt')]
        if steps is not None:args+=['--steps',str(steps)]
        subprocess.run(args,check=True)
        def evaluate_once(folder,mode):
            dest_eval=dest/folder
            if (dest_eval/'metrics.json').exists():
                m=validate_evaluation(dest_eval,require_cache=True)
                expected={ip.name:(array_digest(read_rgb(ip)),array_digest(decode_mask(read_rgb(mp)))) for ip,mp in pairs(data,'valid')}
                observed={r['image']:(r['input_sha256'],r['target_sha256']) for r in m['images']}
                if (m['checkpoint_sha256']!=[sha256(dest/'best.pt')] or m['configs']!=[load_config(config)] or
                    m['tta']!=mode or observed!=expected):
                    raise ValueError('Existing evaluation is stale; use a new run/evaluation directory')
                return
            subprocess.run([sys.executable,'-m','alphase','evaluate','--data',str(data),'--checkpoint',str(dest/'best.pt'),
                '--output',str(dest_eval),'--cache','--tta-mode',mode],check=True)
        if p.get('run_tta',True):
            evaluate_once('evaluation_tta','flip')
        evaluate_once('evaluation','none')
    result=compare(plan_path,out,out/'comparison')
    # Per-seed ensembles use the same validation set; no held-out-test tuning.
    from .ensemble import search
    for seed in sorted({r['seed'] for r in result['runs']}) if p.get('run_ensemble',False) else []:
        singles=[r for r in result['runs'] if r['seed']==seed and r['id'] in result['eligible_variants'] and not r['id'].endswith('_TTA')]
        top=sorted(singles,key=lambda r:r['miou'],reverse=True)[:3]
        if len(top)>1:
            search([out/f'{r["id"]}_s{seed}'/'evaluation' for r in top],out/'comparison'/f'ensemble_s{seed}.json')
    return result

def compare(plan_path,runs,output):
    path=Path(plan_path);p=json.loads(path.read_text(encoding='utf-8'));rows=[];metrics={};config={};missing=[]
    for item in p['runs']:
        folder=Path(runs)/item['output'];mp=folder/'evaluation'/'metrics.json'
        if not mp.exists():missing.append(item['output']);continue
        m=validate_evaluation(mp.parent);c=load_config(path.parent/item['config']);key=(item['id'],item['seed']);metrics[key]=m;config[key]=c
        if m['configs']!=[c] or m['weights']!=[1.] or m['tta']!='none':raise ValueError('Measured single-model config/inference differs from the plan')
        rows.append({'id':item['id'],'seed':item['seed'],'miou':m['image_macro_miou'],**{f'class_{i}_iou':v for i,v in enumerate(m['class_macro_iou'])}})
        tta_path=folder/'evaluation_tta'/'metrics.json'
        if tta_path.exists():
            tm=validate_evaluation(tta_path.parent)
            if (tm['tta'] not in ['flip','d4'] or any(tm[k]!=m[k] for k in
                ['configs','checkpoint_sha256','weights','source_fingerprint','training_runs'])):
                raise ValueError('TTA must use the exact same checkpoint, config, weights and completed run')
            tk=(item['id']+'_TTA',item['seed']);metrics[tk]=tm;config[tk]=c
            rows.append({'id':tk[0],'seed':item['seed'],'miou':tm['image_macro_miou'],**{f'class_{i}_iou':v for i,v in enumerate(tm['class_macro_iou'])}})
    comparisons=[]
    if len({m['absent_policy'] for m in metrics.values()})>1:raise ValueError('Do not compare different absent-class policies')
    if len({m['source_fingerprint'] for m in metrics.values()})>1:raise ValueError('Evaluation source versions differ')
    reference=None
    for m in metrics.values():
        identities={r['image']:(r['input_sha256'],r['target_sha256']) for r in m['images']}
        if reference is not None and identities!=reference:raise ValueError('Validation image sets or target hashes differ')
        reference=identities
    for seed in sorted({r['seed'] for r in rows}):
        pairs_to_compare=[('A','B'),('A','C'),('A','D'),('D','E'),('E','F'),('F','G'),('Random','Pretrained')]
        pairs_to_compare += [('F',child) for child in ['O','L','S']]
        pairs_to_compare += [(item['id'],item['id']+'_TTA') for item in p['runs'] if item['seed']==seed]
        for a,b in pairs_to_compare:
            if (a,seed) in metrics and (b,seed) in metrics:
                ma=metrics[a,seed];mb=metrics[b,seed]
                different=[k for k in BUDGET_KEYS if config[a,seed][k]!=config[b,seed][k] and not (a=='Random' and b=='Pretrained' and k=='pretrained')]
                comparisons.append({'before':a,'after':b,'seed':seed,'budget_mismatch':different,
                    **paired_bootstrap(ma['images'],mb['images'],seed)})
    grouped={}
    for ident in {r['id'] for r in rows}:
        rs=[r for r in rows if r['id']==ident];vals=[r['miou'] for r in rs]
        import numpy as np
        grouped[ident]={'mean_miou':float(np.mean(vals)),'std_across_seeds':float(np.std(vals)),'seed_count':len(vals),'seeds':sorted(r['seed'] for r in rs)}
    effects=[]
    for seed in sorted({r['seed'] for r in rows}):
        if all((k,seed) in metrics for k in 'ABCD') and not any(
            config[k,seed][field]!=config['A',seed][field] for k in 'BCD' for field in BUDGET_KEYS):
            a,b,c,d=[metrics[k,seed]['image_macro_miou'] for k in 'ABCD']
            effects.append({'seed':seed,'encoder_effect':b-a,'decoder_effect':c-a,'interaction':(d-c)-(b-a)})
    # Suggest highest mean among configurations. No inference about unseen test performance.
    expected_seeds=sorted({item['seed'] for item in p['runs']})
    eligible={k for k,v in grouped.items() if v['seeds']==expected_seeds};gates=[];exclusions={}
    anchor=next((name for name in ['E','D','A','Random'] if name in grouped),min(grouped,default=None))
    for ident in sorted(grouped):
        reasons=[];keys=[key for key in metrics if key[0]==ident]
        if ident not in eligible:reasons.append('missing_planned_seeds')
        recipes=[{k:v for k,v in config[key].items() if k not in ['seed','model_name']} for key in keys]
        if any(c!=recipes[0] for c in recipes[1:]):reasons.append('recipe_changes_across_seeds')
        for key in keys:
            records=metrics[key].get('training_runs')
            if not records or any(r is None or not r.get('completed') or r.get('steps') is not None for r in records):
                reasons.append('smoke_incomplete_or_untracked_training')
            reference_key=(anchor,key[1])
            if reference_key not in config:reasons.append('missing_budget_control')
            if reference_key in config and any(config[key][k]!=config[reference_key][k] for k in BUDGET_KEYS
                if not ({ident,anchor}=={'Random','Pretrained'} and k=='pretrained')):
                reasons.append('unmatched_training_budget')
        if reasons:eligible.discard(ident);exclusions[ident]=sorted(set(reasons))
    for child,parent in list(PARENTS.items())+[(k+'_TTA',k) for k in sorted(grouped) if not k.endswith('_TTA')]:
        if child not in grouped:continue
        if parent not in grouped:
            eligible.discard(child);exclusions.setdefault(child,[]).append('missing_parent_control');continue
        shared=[seed for ident,seed in metrics if ident==parent and (child,seed) in metrics]
        decline=[]
        for seed in shared:
            a=metrics[parent,seed]['class_macro_iou'];b=metrics[child,seed]['class_macro_iou']
            decline.extend([b[i]-a[i] for i in range(4) if a[i] is not None and b[i] is not None])
        gain=grouped[child]['mean_miou']-grouped[parent]['mean_miou']
        wins=sum(metrics[child,s]['image_macro_miou']>metrics[parent,s]['image_macro_miou'] for s in shared)
        required_wins=len(expected_seeds)//2+1
        allowed={'G':{'boundary_weight'},'F':{'sampler'},'O':{'phase_sampling'},'L':{'lovasz_weight'},'S':{'decoder_attention'}}.get(child,set())
        matched=all(not any(config[parent,s][k]!=config[child,s][k] for k in DEFAULT if k not in allowed|{'model_name'}) for s in shared)
        passed=parent in eligible and child in eligible and bool(shared) and matched and wins>=required_wins and gain>=p.get('minimum_gain',.005) and all(d>=-.02 for d in decline)
        if not passed:eligible.discard(child)
        gates.append({'parent':parent,'child':child,'eligible':passed,'matched_conditions':matched,'positive_seeds':wins,'required_positive_seeds':required_wins,'class_decline_tolerance':.02,'minimum_gain':p.get('minimum_gain',.005),
            'note':'Heuristic gate, not independent evidence of test performance.'})
    selected=sorted(eligible,key=lambda k:(-grouped[k]['mean_miou'],k))[0] if eligible else None
    result={'runs':rows,'missing':missing,'groups':grouped,'comparisons':comparisons,'factor_effects':effects,'selection_gates':gates,'suggested_variant':selected,
        'eligible_variants':sorted(eligible),'exclusions':exclusions,
        'selection_status':('no_eligible_candidate' if selected is None else
            ('screen_only' if len(expected_seeds)<p.get('minimum_replication_seeds',3) else 'replicated_candidate')),
        'note':'Validation-based selection only. Replication does not remove repeated-selection bias.'}
    output=Path(output);output.mkdir(parents=True,exist_ok=True);save_json(output/'comparison.json',result)
    if rows:
        with (output/'comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ALPHASE experiment comparison','','| Variant | Seed | mIoU |','|---|---:|---:|']
    lines += [f'| {r["id"]} | {r["seed"]} | {r["miou"]:.4f} |' for r in sorted(rows,key=lambda r:r['miou'],reverse=True)]
    lines += ['',f'Missing runs: {", ".join(missing) or "none"}',f'Suggested variant: {selected or "not measured"}',
        '', 'See comparison.json for per-class IoU, paired bootstrap intervals, budget mismatches and factor effects.']
    (output/'comparison.md').write_text('\n'.join(lines),encoding='utf-8');return result
