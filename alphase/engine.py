from pathlib import Path
import csv
import json
import random
import time
import platform
import hashlib
import importlib.metadata
import math
import uuid
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from .data import pairs,read_rgb,decode_mask,save_json,PALETTE,CLASSES,data_fingerprint,sha256
from .metrics import diagnostics,summarize,image_iou
from .visuals import panel,report_html
from .config import resolved_config
from .models import build_model,normalized,amp_settings,ModelEMA,make_optimizer
from .sampling import PatchDataset
from .losses import loss_value
from .inference import predict,positions,tta_mode
from .provenance import array_digest,source_fingerprint,prediction_fingerprint

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def device_for(name):
    if name=='auto':name='cuda' if torch.cuda.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    return torch.device(name)

def environment(device):
    names=['torch','torchvision','segmentation-models-pytorch','timm','numpy','Pillow','scipy']
    versions={}
    for n in names:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]=None
    return {'python':platform.python_version(),'versions':versions,'device':str(device),'cuda_build':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'gpu_total_bytes':torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None}

def load_model(checkpoint,device):
    # Only load trusted project checkpoints (pickle payload).
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=resolved_config(state['config']);model=build_model(cfg,download=False)
    model.load_state_dict(state.get('eval_model',state['model']));return model.to(device).eval(),cfg

def normalized_weights(weights,n):
    w=np.ones(n,np.float64) if weights is None else np.asarray(weights,np.float64)
    if w.shape!=(n,) or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('Weights must be finite, nonnegative, correct length, positive sum')
    return w/w.sum()

class Predictor:
    """CPU-resident ensemble, at most one model on GPU at a time."""
    def __init__(self,checkpoints,device_name='auto',weights=None,tta=False):
        self.device=device_for(device_name);self.weights=normalized_weights(weights,len(checkpoints));self.tta=tta_mode(tta);self.checkpoints=[str(Path(p).resolve()) for p in checkpoints]
        self.checkpoint_sha256=[sha256(p) for p in self.checkpoints]
        self.loaded=[load_model(p,torch.device('cpu')) for p in checkpoints]
        if self.checkpoint_sha256 != [sha256(p) for p in self.checkpoints]:raise ValueError('Checkpoint changed while loading')
        self.training_runs=[torch.load(p,map_location='cpu',weights_only=False).get('training_run') for p in self.checkpoints]

    def metadata(self):
        result={'checkpoints':self.checkpoints,'checkpoint_sha256':self.checkpoint_sha256,
            'configs':[cfg for _,cfg in self.loaded],'weights':self.weights.tolist(),'tta':self.tta,
            'source_fingerprint':source_fingerprint(),'training_runs':self.training_runs}
        result['prediction_fingerprint']=prediction_fingerprint(result)
        return result
    def __call__(self,rgb):
        acc=None
        for weight,(model,cfg) in zip(self.weights,self.loaded):
            if weight==0:continue
            model.to(self.device)
            try:prob=predict(model,rgb,self.device,cfg['patch_size'],cfg['overlap'],self.tta,cfg['inference_batch_size'],cfg)
            finally:
                if self.device.type=='cuda':model.cpu()
            acc=prob*weight if acc is None else acc+prob*weight
        # One explicit rounding contract for evaluation, caches and deployment.
        return acc.astype(np.float32)

def evaluate(predictor,root,output=None,absent='exclude',tolerance=2,small_area=64,cache=False,detailed=True):
    output=Path(output) if output else None
    if output:
        output.mkdir(parents=True,exist_ok=True)
        if any(output.iterdir()):raise ValueError('Use a new empty evaluation directory; caches are immutable')
    metadata=predictor.metadata() if hasattr(predictor,'metadata') else {}
    cache_manifest={}
    rows=[];tick=time.time()
    for ip,mp in pairs(root,'valid'):
        rgb=read_rgb(ip);target=decode_mask(read_rgb(mp));prob=predictor(rgb);pred=prob.argmax(0).astype(np.uint8)
        if detailed:r=diagnostics(pred,target,absent,tolerance,small_area)
        else:
            score,scores,cm=image_iou(pred,target,absent)
            r={'miou':score,'confusion_matrix':cm.tolist(),'classes':[{'iou':None if np.isnan(s) else float(s),'boundary_f1':None} for s in scores]}
        r['image']=ip.name;r['mean_confidence']=float(prob.max(0).mean())
        r['target_sha256']=array_digest(target);r['input_sha256']=array_digest(rgb)
        cm=np.asarray(r['confusion_matrix']);tp=np.diag(cm);union=cm.sum(0)+cm.sum(1)-tp
        ious=np.divide(tp,union,out=np.zeros(4,dtype=float),where=union>0)
        r['alternative_miou_policies']={'exclude':float(ious[union>0].mean()),'zero':float(ious.mean()),'one':float((ious.sum()+(union==0).sum())/4)}
        if output:
            mask_name=ip.name.replace('_image.png','_mask.png');Image.fromarray(PALETTE[pred]).save(output/mask_name)
            r['panel']=ip.stem+'_panel.png';panel(output/r['panel'],rgb,target,pred,prob)
            if cache:
                cache_path=output/(ip.stem+'.npz')
                np.savez_compressed(cache_path,probabilities=prob.astype(np.float32),target=target,
                    prediction_fingerprint=metadata.get('prediction_fingerprint','untracked'),
                    input_sha256=r['input_sha256'],target_sha256=r['target_sha256'])
                cache_manifest[cache_path.name]=sha256(cache_path)
        rows.append(r)
    result=summarize(rows)|{'images':rows,'absent_policy':absent,'boundary_tolerance':tolerance,'small_area':small_area,'detailed':detailed,'seconds':time.time()-tick}
    result['alternative_miou_policies']={p:float(np.mean([r['alternative_miou_policies'][p] for r in rows])) for p in ['exclude','zero','one']}
    train_prefixes={ip.name.split('_')[0] for ip,_ in pairs(root,'train')}
    groups={'train_prefix_seen':[],'train_prefix_unseen':[]}
    for row in rows:groups['train_prefix_seen' if row['image'].split('_')[0] in train_prefixes else 'train_prefix_unseen'].append(row['miou'])
    result['prefix_diagnostics']={key:{'count':len(values),'miou':float(np.mean(values)) if values else None} for key,values in groups.items()}
    result['prefix_note']='Filename prefix is only a proxy, not a confirmed specimen identity; official split is unchanged.'
    result|=metadata|{'schema_version':4,'complete':True,'cache_manifest':cache_manifest}
    if output:
        report_html(output,result)
        save_json(output/'metrics.json',result)
    return result

def atomic_save(path,state):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)

def export_run(output,state,preserve_identical=False):
    """Derived files can always be rebuilt from the committed, self-contained last.pt."""
    training_run={'run_id':state['run_id'],'steps':state['steps'],'completed':state['completed'],
        'stop_reason':state['stop_reason'],'completed_epochs':state['epoch']+1,'best_epoch':state['best_epoch']+1}
    best={'format_version':4,'config':state['config'],'model':state['best_eval_model'],
        'data_fingerprint':state['data_fingerprint'],'source_fingerprint':state['source_fingerprint'],
        'training_run':training_run}
    for name in ['best.pt','inference.pt']:
        path=output/name;identical=False
        if preserve_identical and path.exists():
            try:
                existing=torch.load(path,map_location='cpu',weights_only=False)
                identical=(all(existing.get(k)==v for k,v in best.items() if k!='model') and
                    set(existing.get('model',{}))==set(best['model']) and
                    all(torch.equal(existing['model'][k],v) for k,v in best['model'].items()))
            except (RuntimeError,EOFError,ValueError,pickle.UnpicklingError):identical=False
        if not identical:atomic_save(path,best)
    save_json(output/'best_metrics.json',state['best_metrics'])
    save_json(output/'run_status.json',training_run)
    history=output/'history.csv';temp=history.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(state['history'][0]));writer.writeheader();writer.writerows(state['history'])
    temp.replace(history)

def train(root,cfg,output,resume=None,steps=None):
    cfg=resolved_config(cfg)
    if cfg['pretrained'] and not cfg['pretrained_permission']:raise ValueError('Pretrained permission is not confirmed')
    if steps is not None and (type(steps) is not int or steps<1):raise ValueError('steps must be positive')
    seed_all(cfg['seed']);device=device_for(cfg['device']);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not resume and any(p.name!='resource_probe.json' for p in output.iterdir()):raise ValueError('Use a new empty run directory or --resume')
    fingerprint=data_fingerprint(root)
    source_hash=source_fingerprint()
    state=torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    if state and state.get('format_version')!=4:raise ValueError('Legacy checkpoint is inference-only in v0.4; start a new run')
    if state and 'optimizer' not in state:raise ValueError('Resume requires last.pt, not inference-only weights')
    if state and (state['config']!=cfg or state.get('data_fingerprint')!=fingerprint):raise ValueError('Resume config or dataset changed')
    if state and state.get('source_fingerprint')!=source_hash:raise ValueError('Resume source code changed')
    if state and state['steps']!=steps:raise ValueError('Resume step budget changed; keep smoke and full runs separate')
    if state and output.resolve()!=Path(resume).resolve().parent and any(output.iterdir()):
        raise ValueError('A forked resume requires an empty destination')
    if state and (output/'last.pt').exists() and sha256(output/'last.pt')!=sha256(resume):
        raise ValueError('Refusing to overwrite a different run or rewind an existing run')
    save_json(output/'config.json',cfg);save_json(output/'environment.json',environment(device))
    if state:
        if output.resolve()!=Path(resume).resolve().parent:atomic_save(output/'last.pt',state)
        export_run(output,state,preserve_identical=True)
        if state['completed']:return
    model=build_model(cfg,download=not bool(resume)).to(device);opt=make_optimizer(model,cfg)
    def factor(epoch):
        warmup=cfg['warmup_epochs']
        if epoch<warmup:return (epoch+1)/warmup
        return .01+.99*.5*(1+math.cos(math.pi*min(epoch-warmup,cfg['epochs']-warmup)/max(1,cfg['epochs']-warmup)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    amp,dtype=amp_settings(cfg,device);scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    ema=ModelEMA(model,cfg['ema_decay']) if cfg['ema_decay'] else None
    start=0;best=-1.;stale=0;best_eval=None;best_epoch=-1;best_metrics=None;history_rows=[];run_id=str(uuid.uuid4())
    if state:
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        if ema:
            ema.module.load_state_dict(state['eval_model']);ema.updates=state['ema_updates']
        start=state['epoch']+1;best=state['best'];stale=state['stale']
        best_eval=state['best_eval_model'];best_epoch=state['best_epoch'];best_metrics=state['best_metrics'];history_rows=state['history'];run_id=state['run_id']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    dataset=PatchDataset(root,cfg)
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    for epoch in range(start,cfg['epochs']):
        tick=time.time();model.train();dataset.set_epoch(epoch);total=0.;n=0;exposure=np.zeros(4,np.int64)
        if cfg['freeze_encoder_bn'] and cfg['pretrained']:
            for module in model.encoder.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        batches=min(len(loader),steps) if steps is not None else len(loader)
        if batches<1:raise ValueError('No training steps')
        accumulation=cfg['accumulation_steps'];opt.zero_grad(set_to_none=True)
        updates=0;skipped=0
        for i,(x,y) in enumerate(loader):
            if i>=batches:break
            exposure+=np.bincount(y[y!=255].numpy(),minlength=4);x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,enabled=amp,dtype=dtype):loss=loss_value(model(x),y,cfg)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            group_start=(i//accumulation)*accumulation
            group_end=min(group_start+accumulation,batches)
            group_samples=min(group_end*cfg['batch_size'],len(dataset))-group_start*cfg['batch_size']
            scaler.scale(loss*(len(x)/group_samples)).backward()
            if i+1==group_end:
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=not scaler.is_enabled())
                old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()>=old_scale:
                    updates+=1
                    if ema:ema.update(model)
                else:skipped+=1
                opt.zero_grad(set_to_none=True)
            total+=loss.item()*len(x);n+=len(x)
        if n==0:raise ValueError('No training steps')
        evaluation_model=ema.module if ema else model
        def single(rgb):return predict(evaluation_model,rgb,device,cfg['patch_size'],cfg['overlap'],False,cfg['inference_batch_size'],cfg)
        result=evaluate(single,root,absent=cfg['absent_policy'],detailed=False);score=result['image_macro_miou'];improved=score>best
        if improved:
            best=score;stale=0;best_epoch=epoch;best_metrics=result
            best_eval={k:v.detach().cpu().clone() for k,v in evaluation_model.state_dict().items()}
        else:stale+=1
        scheduler.step()
        row={'epoch':epoch+1,'loss':total/n,'val_image_macro_miou':score,'seconds':time.time()-tick,
            'optimizer_steps':updates,'skipped_amp_steps':skipped,'encoder_lr_next':opt.param_groups[0]['lr'],
            'decoder_lr_next':opt.param_groups[1]['lr'],'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
            **{f'class_{c}_iou':result['class_macro_iou'][c] for c in range(4)},
            **{f'class_{c}_sampled_pixels':int(exposure[c]) for c in range(4)}}
        history_rows.append(row)
        stopped=epoch+1>=cfg['min_epochs'] and stale>=cfg['patience']
        completed=stopped or epoch+1==cfg['epochs']
        state={'format_version':4,'run_id':run_id,'steps':steps,'completed':completed,
            'stop_reason':'early_stopping' if stopped else ('max_epochs' if completed else None),
            'history':history_rows,'best_eval_model':best_eval,'best_epoch':best_epoch,'best_metrics':best_metrics,
            'config':cfg,'data_fingerprint':fingerprint,'source_fingerprint':source_hash,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'eval_model':evaluation_model.state_dict(),'ema_updates':ema.updates if ema else 0,
            'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'stale':stale,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        # Commit once. best/history/export are derived, recoverable artifacts.
        atomic_save(output/'last.pt',state)
        export_run(output,state)
        print(json.dumps(row),flush=True)
        if completed:break

def infer(root,checkpoints,output,device='auto',weights=None,tta=False,contract=None):
    from .submission import load_contract,validate_predictions,output_name
    c=load_contract(contract);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('*.png')):raise ValueError('Use an empty prediction directory')
    predictor=Predictor(checkpoints,device,weights,tta);rows=[]
    for ip,_ in pairs(root,'test'):
        rgb=read_rgb(ip);pred=predictor(rgb).argmax(0).astype(np.uint8)
        filename=output_name(ip.name,c)
        encoded=PALETTE[pred]
        if c['mode']=='RGBA':encoded=np.concatenate([encoded,np.full((*pred.shape,1),255,np.uint8)],-1)
        Image.fromarray(encoded).save(output/filename)
        rows.append({'input':ip.name,'input_sha256':array_digest(rgb),'output':filename,'output_sha256':sha256(output/filename),'height':rgb.shape[0],'width':rgb.shape[1]})
    save_json(output/'manifest.json',predictor.metadata()|{'schema_version':4,'model':'ALPHASE','contract':c,'images':rows})
    validate_predictions(root,output,c)
