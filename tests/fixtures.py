"""Explicit synthetic provenance, never presented as measured competition scores."""
from pathlib import Path
import numpy as np
from alphase.config import resolved_config
from alphase.data import save_json,sha256
from alphase.provenance import prediction_fingerprint,array_digest
from alphase.metrics import image_iou


def metric_record(cfg,score=.5,tta='none',checkpoint='synthetic_checkpoint',steps=None,completed=True):
    r={'schema_version':4,'complete':True,'absent_policy':'exclude','image_macro_miou':score,
        'class_macro_iou':[score]*4,'configs':[resolved_config(cfg)],'weights':[1.],
        'checkpoint_sha256':[checkpoint],'checkpoints':[checkpoint+'.pt'],'tta':tta,
        'source_fingerprint':'synthetic_source','training_runs':[{'completed':completed,'steps':steps,'run_id':'synthetic'}],
        'images':[{'image':'x_image.png','miou':score,'input_sha256':'synthetic_input','target_sha256':'synthetic_target'}]}
    r['prediction_fingerprint']=prediction_fingerprint(r)
    return r


def write_cache(folder,prob,target,cfg):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    score,scores,_=image_iou(prob.argmax(0),target)
    r=metric_record(cfg,score,checkpoint=folder.name)
    r['class_macro_iou']=[None if np.isnan(x) else float(x) for x in scores]
    r['images'][0]['target_sha256']=array_digest(target)
    path=folder/'x_image.npz'
    np.savez_compressed(path,probabilities=prob,target=target,prediction_fingerprint=r['prediction_fingerprint'],
        input_sha256=r['images'][0]['input_sha256'],target_sha256=array_digest(target))
    r['cache_manifest']={path.name:sha256(path)};save_json(folder/'metrics.json',r)
    return r
