"""Optional confidence-gated removal of isolated tiny predictions. Never default-on."""
import numpy as np
from scipy import ndimage as ndi

def clean(prob,min_area=8,confidence=.6,classes=(0,1,2)):
    if prob.shape[0]!=4 or min_area<1 or not 0<=confidence<=1:raise ValueError('Invalid postprocessing settings')
    pred=prob.argmax(0).astype(np.uint8);out=pred.copy()
    for c in classes:
        if c not in range(4):raise ValueError('Invalid class')
        labels,n=ndi.label(pred==c,structure=np.ones((3,3),int));sizes=np.bincount(labels.ravel())
        for ident in range(1,n+1):
            mask=labels==ident
            if sizes[ident]>=min_area or prob[c,mask].mean()>=confidence:continue
            ring=ndi.binary_dilation(mask,structure=np.ones((3,3),bool))&~mask
            neighbors=[k for k in np.unique(pred[ring]) if k!=c]
            if not neighbors:continue
            # Assign each pixel the most probable class present in its outer ring.
            selected=np.array(neighbors);out[mask]=selected[prob[selected][:,mask].argmax(0)]
    return out

def evaluate_cache(folder,output,min_area=8,confidence=.6,absent='exclude'):
    from pathlib import Path
    from .data import save_json
    from .metrics import diagnostics,summarize,paired_bootstrap,image_iou
    from .provenance import validate_evaluation
    metadata=validate_evaluation(folder,require_cache=True)
    if metadata['absent_policy']!=absent:raise ValueError('Postprocessing metric policy differs from cached evaluation')
    rows=[];before=[]
    for path in sorted(Path(folder).glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:prob=z['probabilities'];target=z['target']
        pred=clean(prob,min_area,confidence);r=diagnostics(pred,target,absent);r['image']=path.stem+'.png';rows.append(r)
        before.append({'image':r['image'],'miou':image_iou(prob.argmax(0),target,absent)[0]})
    if not rows:raise ValueError('No probability caches')
    result=summarize(rows)|{'images':rows,'source_prediction_fingerprint':metadata['prediction_fingerprint'],'settings':{'min_area':min_area,'confidence':confidence},
        'comparison':paired_bootstrap(before,rows),'note':'Experimental postprocessing; no guarantee it preserves real small phases.'}
    save_json(output,result);return result
