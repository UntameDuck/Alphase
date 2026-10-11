import numpy as np
from scipy import ndimage as ndi

def confusion(pred,target):
    if pred.shape!=target.shape:raise ValueError('Prediction and target shapes differ')
    if not np.isin(pred,range(4)).all() or not np.isin(target,range(4)).all():raise ValueError('Labels must be 0..3')
    return np.bincount((target.astype(np.int64)*4+pred).ravel(),minlength=16).reshape(4,4)

def image_iou(pred,target,absent='exclude'):
    cm=confusion(pred,target);tp=np.diag(cm);union=cm.sum(0)+cm.sum(1)-tp
    scores=np.divide(tp,union,out=np.full(4,np.nan),where=union>0)
    if absent=='zero':scores[union==0]=0
    elif absent=='one':scores[union==0]=1
    elif absent!='exclude':raise ValueError(absent)
    return float(np.nanmean(scores)),scores,cm

def boundary(binary):
    # Inner 8-neighbor boundary; outer image border is not a phase boundary.
    return binary & ~ndi.binary_erosion(binary,structure=np.ones((3,3)),border_value=1)

def boundary_f1(pred,target,tolerance=2):
    if tolerance<0:raise ValueError('tolerance must be nonnegative')
    pb=boundary(pred);tb=boundary(target);npb=int(pb.sum());ntb=int(tb.sum())
    if npb==0 and ntb==0:return None
    if npb==0 or ntb==0:return 0.
    if tolerance:
        # Euclidean pixel distance, not a square dilation neighborhood.
        near_t=ndi.distance_transform_edt(~tb)<=tolerance
        near_p=ndi.distance_transform_edt(~pb)<=tolerance
    else:near_t=tb;near_p=pb
    precision=float((pb&near_t).sum()/npb);recall=float((tb&near_p).sum()/ntb)
    return 2*precision*recall/(precision+recall) if precision+recall else 0.

def small_components(pred,target,max_area=64):
    labels,n=ndi.label(target,structure=np.ones((3,3),int));areas=np.bincount(labels.ravel())
    small_ids=np.where((areas<=max_area)&(areas>0))[0];small_ids=small_ids[small_ids!=0]
    small=np.isin(labels,small_ids);pixel_count=int(small.sum())
    objects=[];covered=np.bincount(labels[pred].ravel(),minlength=n+1)
    for ident in small_ids:
        hits=int(covered[ident]);area=int(areas[ident])
        objects.append({'area':area,'covered_pixels':hits,'recall':hits/area})
    # This is GT-small-component recall, not instance IoU or whole-class IoU.
    fp_labels,fp_n=ndi.label(pred,structure=np.ones((3,3),int));fp_areas=np.bincount(fp_labels.ravel())
    matched=np.bincount(fp_labels[target].ravel(),minlength=fp_n+1)
    unmatched=int(((fp_areas[1:]<=max_area)&(matched[1:]==0)).sum())
    return {'gt_small_count':len(objects),'gt_small_pixels':pixel_count,
        'gt_small_pixel_recall':float((small&pred).sum()/pixel_count) if pixel_count else None,
        'gt_small_object_mean_recall':float(np.mean([o['recall'] for o in objects])) if objects else None,
        'unmatched_small_prediction_count':unmatched,'objects':objects}

def diagnostics(pred,target,absent='exclude',tolerance=2,max_area=64):
    score,scores,cm=image_iou(pred,target,absent);rows=[]
    for c in range(4):
        tp=int(cm[c,c]);fp=int(cm[:,c].sum()-tp);fn=int(cm[c,:].sum()-tp)
        rows.append({'class_id':c,'iou':None if np.isnan(scores[c]) else float(scores[c]),'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'target_area':int(cm[c,:].sum()),'predicted_area':int(cm[:,c].sum()),
            'target_area_fraction':float(cm[c,:].sum()/cm.sum()),'predicted_area_fraction':float(cm[:,c].sum()/cm.sum()),
            'area_fraction_absolute_error':float(abs(cm[c,:].sum()-cm[:,c].sum())/cm.sum()),
            'boundary_f1':boundary_f1(pred==c,target==c,tolerance),
            'small_components':small_components(pred==c,target==c,max_area)})
    return {'miou':score,'classes':rows,'confusion_matrix':cm.tolist()}

def summarize(rows):
    def avg(values):
        v=[x for x in values if x is not None];return float(np.mean(v)) if v else None
    return {'image_macro_miou':avg([r['miou'] for r in rows]),
        'class_macro_iou':[avg([r['classes'][c]['iou'] for r in rows]) for c in range(4)],
        'class_macro_boundary_f1':[avg([r['classes'][c]['boundary_f1'] for r in rows]) for c in range(4)],
        'class_macro_area_fraction_mae':[avg([r['classes'][c].get('area_fraction_absolute_error') for r in rows]) for c in range(4)],
        'confusion_matrix':np.sum([r['confusion_matrix'] for r in rows],axis=0).tolist(),
        'per_image_miou_std':float(np.std([r['miou'] for r in rows])), 'image_count':len(rows)}

def paired_bootstrap(before,after,seed=42,iterations=2000):
    a={r['image']:r['miou'] for r in before};b={r['image']:r['miou'] for r in after}
    if set(a)!=set(b):raise ValueError('Paired comparison requires exactly the same images')
    delta=np.array([b[k]-a[k] for k in sorted(a)])
    rng=np.random.default_rng(seed);means=delta[rng.integers(0,len(delta),(iterations,len(delta)))].mean(1)
    return {'mean_delta':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
        'improved_images':int((delta>0).sum()),'worsened_images':int((delta<0).sum()),'image_count':len(delta),
        'note':'Exploratory paired image bootstrap; does not account for repeated model selection or correlated sample groups.'}
