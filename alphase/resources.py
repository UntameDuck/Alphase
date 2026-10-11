"""Opt-in empirical training memory probe; never claims CPU probe is GPU validation."""
import gc
import time
import torch
from .engine import device_for,build_model,loss_value,environment,seed_all
from .data import save_json
from .models import amp_settings,ModelEMA,make_optimizer
from .config import resolved_config

def tune(cfg,patches=(256,512),batches=(2,4,8),safety_fraction=.85):
    cfg=resolved_config(cfg)
    if not 0<safety_fraction<1:raise ValueError('safety_fraction must be in (0,1)')
    device=device_for(cfg['device']);record={'environment':environment(device),'trials':[],'selected':None}
    if device.type!='cuda':
        record['status']='skipped_no_cuda';record['note']='No GPU present. Config unchanged; no CPU-based VRAM guess.'
        return cfg.copy(),record
    if any(p<32 or p%32 for p in patches) or any(b<1 for b in batches):raise ValueError('Invalid probe candidates')
    for p in sorted(set(patches),reverse=True):
        for b in sorted(set(batches),reverse=True):
            torch.cuda.empty_cache();seed_all(cfg['seed']);tick=time.time();row={'patch_size':p,'batch_size':b};model=x=y=opt=loss=ema=None
            try:
                torch.cuda.reset_peak_memory_stats(device)
                model=build_model(cfg,download=False).to(device);model.train()
                opt=make_optimizer(model,cfg)
                ema=ModelEMA(model,cfg['ema_decay']) if cfg.get('ema_decay',0) else None
                amp,dtype=amp_settings(cfg,device);scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
                x=torch.randn(b,3,p,p,device=device);y=torch.randint(0,4,(b,p,p),device=device)
                for _ in range(2):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',enabled=amp,dtype=dtype):loss=loss_value(model(x),y,cfg)
                    scaler.scale(loss).backward();scaler.step(opt);scaler.update()
                    if ema:ema.update(model)
                torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved(device);free,total=torch.cuda.mem_get_info(device)
                other_used=total-free-torch.cuda.memory_reserved(device)
                row|={'peak_reserved_bytes':peak,'total_bytes':total,'other_used_bytes':max(0,other_used),'seconds':time.time()-tick}
                row['status']='pass' if peak+max(0,other_used)<=total*safety_fraction else 'insufficient_margin'
                if row['status']=='pass' and record['selected'] is None:record['selected']={'patch_size':p,'batch_size':b}
            except torch.cuda.OutOfMemoryError:row['status']='out_of_memory'
            finally:
                model=x=y=opt=loss=ema=None;gc.collect();torch.cuda.empty_cache()
            record['trials'].append(row)
            if record['selected']:break
        if record['selected']:break
    if not record['selected']:raise RuntimeError('No candidate fits GPU memory with safety margin; reduce patch/batch')
    updated=cfg.copy();updated.update(record['selected']);updated['overlap']=min(updated['overlap'],updated['patch_size']//4)
    record['status']='selected';record['note']='Two-step synthetic probe including AdamW states. Real data/runtime memory can still differ.'
    return updated,record


def benchmark(root,cfg,warmup=2,batches=10):
    """Opt-in GPU timing on real train crops; no saved weights or accuracy claim."""
    from torch.utils.data import DataLoader
    from .sampling import PatchDataset
    from .data import pairs,read_rgb
    from .inference import predict
    if warmup<0 or batches<1:raise ValueError('Invalid benchmark lengths')
    cfg=resolved_config(cfg);device=device_for(cfg['device'])
    record={'environment':environment(device),'config':cfg,'trained_competition_model':False}
    if device.type!='cuda':return record|{'status':'skipped_no_cuda','note':'CPU timing is not a GPU cost estimate.'}
    seed_all(cfg['seed']);model=build_model(cfg,download=False).to(device)
    ema=ModelEMA(model,cfg['ema_decay']) if cfg['ema_decay'] else None
    opt=make_optimizer(model,cfg);enabled,dtype=amp_settings(cfg,device)
    scaler=torch.amp.GradScaler('cuda',enabled=enabled and dtype==torch.float16)
    loader=DataLoader(PatchDataset(root,cfg),batch_size=cfg['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
    if warmup+batches>len(loader):raise ValueError('Benchmark exceeds one epoch; reduce warmup/batches')
    iterator=iter(loader);times=[];torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup+batches):
        torch.cuda.synchronize();tick=time.perf_counter();x,y=next(iterator)
        model.train();x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',enabled=enabled,dtype=dtype):loss=loss_value(model(x),y,cfg)
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt);scaler.update()
        if ema:ema.update(model)
        torch.cuda.synchronize()
        if i>=warmup:times.append(time.perf_counter()-tick)
    eval_model=ema.module if ema else model
    tick=time.perf_counter()
    for ip,_ in pairs(root,'valid'):
        predict(eval_model,read_rgb(ip),device,cfg['patch_size'],cfg['overlap'],'none',cfg['inference_batch_size'],cfg)
    torch.cuda.synchronize();validation_seconds=time.perf_counter()-tick
    import numpy as np
    return record|{'status':'measured','warmup_batches':warmup,'measured_batches':batches,
        'median_batch_seconds':float(np.median(times)),'p90_batch_seconds':float(np.quantile(times,.9)),
        'estimated_epoch_train_seconds':float(np.median(times)*len(loader)),
        'validation_inference_seconds':validation_seconds,'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
        'note':'Random-init timing only; optimizer step on every microbatch gives conservative accumulation overhead. Excludes startup, metric computation, checkpoint I/O, TTA, storage and network. Re-estimate using real epoch logs.'}
