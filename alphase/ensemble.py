from pathlib import Path
import json
import numpy as np
from .metrics import image_iou,diagnostics,summarize,paired_bootstrap
from .data import save_json
from .provenance import validate_evaluation,array_digest
from math import comb

def simplex(n,total):
    if n==1:yield [total];return
    for first in range(total+1):
        for tail in simplex(n-1,total-first):yield [first]+tail

def search(caches,output,denominator=4,absent='exclude',max_candidates=2000):
    if not caches or not 1<=denominator<=20:raise ValueError('Invalid search size')
    if comb(denominator+len(caches)-1,len(caches)-1)>max_candidates:raise ValueError('Search exceeds max_candidates')
    folders=[Path(p) for p in caches];files=[{p.name:p for p in f.glob('*.npz')} for f in folders]
    if not files[0] or any(set(f)!=set(files[0]) for f in files[1:]):raise ValueError('Cache image sets differ or empty')
    metas=[validate_evaluation(f,require_cache=True) for f in folders]
    if any(len(m['configs'])!=1 for m in metas):raise ValueError('Search requires single-checkpoint caches; evaluate nested ensembles explicitly')
    identities=[{r['image']:(r['input_sha256'],r['target_sha256']) for r in m['images']} for m in metas]
    if any(identity!=identities[0] for identity in identities[1:]):raise ValueError('Cache input/target identities differ')
    if any(m['absent_policy']!=absent for m in metas):raise ValueError('Cached metric policy differs')
    if len({m.get('tta',False) for m in metas})>1:raise ValueError('Mixing TTA and non-TTA caches is unsupported; evaluate caches with one common TTA mode')
    loaded=[]
    for name in sorted(files[0]):
        arrays=[];target=None
        for index,f in enumerate(files):
            with np.load(f[name],allow_pickle=False) as z:
                p=z['probabilities'];y=z['target']
                row=next(r for r in metas[index]['images'] if Path(r['image']).stem+'.npz'==name)
                if (str(z['prediction_fingerprint'])!=metas[index]['prediction_fingerprint'] or
                    str(z['input_sha256'])!=row['input_sha256'] or str(z['target_sha256'])!=row['target_sha256'] or
                    array_digest(y)!=row['target_sha256']):raise ValueError('Cache provenance mismatch')
            if p.shape!=(4,*y.shape) or not np.isfinite(p).all() or np.min(p)<0 or not np.allclose(p.sum(0),1,atol=1e-5):raise ValueError('Invalid cached probabilities')
            if target is not None and not np.array_equal(y,target):raise ValueError('Cache targets differ')
            target=y;arrays.append(p)
        loaded.append((name,np.stack(arrays),target))
    candidates=[];best=None
    for parts in simplex(len(folders),denominator):
        if len(candidates)>=max_candidates:raise ValueError('Search exceeds max_candidates; reduce denominator/models')
        weights=np.array(parts,dtype=np.float64)/denominator;scores=[]
        for _,probs,target in loaded:
            pred=np.tensordot(weights,probs,axes=(0,0)).astype(np.float32).argmax(0)
            scores.append(image_iou(pred,target,absent)[0])
        row={'weights':weights.tolist(),'miou':float(np.mean(scores))};candidates.append(row)
        # Deterministic tie: prefer fewer nonzero weights, then earlier checkpoint.
        key=(row['miou'],-np.count_nonzero(weights),tuple(weights.tolist()))
        if best is None or key>best[0]:best=(key,row)
    best=best[1];rows=[]
    for name,probs,target in loaded:
        pred=np.tensordot(best['weights'],probs,axes=(0,0)).astype(np.float32).argmax(0);r=diagnostics(pred,target,absent);r['image']=name.replace('.npz','.png');rows.append(r)
    baselines=[]
    for i in range(len(folders)):
        rs=[{'image':name.replace('.npz','.png'),'miou':image_iou(ps[i].argmax(0),y,absent)[0]} for name,ps,y in loaded]
        baselines.append({'cache':str(folders[i]),'miou':float(np.mean([r['miou'] for r in rs])), 'comparison':paired_bootstrap(rs,rows)})
    result={'best':best,'candidates':candidates,'baselines':baselines,'validation':summarize(rows)|{'images':rows},
        'caches':list(map(str,folders)),'absent_policy':absent,
        'inference':{'checkpoints':[m.get('checkpoints') for m in metas],'tta':[m.get('tta') for m in metas]},
        'note':'Weights selected on validation data. Improvement is selection-biased; recheck across seeds or independent data.'}
    save_json(output,result);return result
