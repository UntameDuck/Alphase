"""Bounded-batch, full-resolution probability blending and invertible D4 TTA."""
import numpy as np
import torch
from .config import resolved_config
from .models import normalized,amp_settings


def positions(n,p,stride):
    if min(n,p,stride)<1:raise ValueError('Invalid tile extent')
    seq=list(range(0,max(n-p,0)+1,stride));end=max(n-p,0)
    if seq[-1]!=end:seq.append(end)
    return seq


def tta_mode(value):
    if value is True:return 'flip'
    if value is False or value is None:return 'none'
    if value not in ['none','flip','d4']:raise ValueError('TTA must be none, flip, or d4')
    return value


def transforms(mode):
    mode=tta_mode(mode)
    if mode=='none':return [(0,False)]
    if mode=='flip':return [(0,False),(0,True),(2,False),(2,True)]
    return [(k,flip) for k in range(4) for flip in [False,True]]


def transform(x,k,flip,inverse=False):
    if inverse:
        if flip:x=torch.flip(x,[-1])
        return torch.rot90(x,-k,[-2,-1])
    x=torch.rot90(x,k,[-2,-1])
    return torch.flip(x,[-1]) if flip else x


@torch.inference_mode()
def predict(model,rgb,device,patch=256,overlap=64,tta=False,batch_size=1,cfg=None):
    cfg=resolved_config(cfg or {})
    if patch<32 or patch%32 or not 0<=overlap<patch or batch_size<1:raise ValueError('Invalid tile configuration')
    model.eval();h,w=rgb.shape[:2]
    x=np.pad(rgb,((0,max(0,patch-h)),(0,max(0,patch-w)),(0,0)),mode='edge')
    H,W=x.shape[:2];acc=np.zeros((4,H,W),np.float32);norm=np.zeros((H,W),np.float32)
    axis=np.maximum(np.hanning(patch),.05).astype(np.float32);weight=axis[:,None]*axis[None,:]
    coords=[(top,left) for top in positions(H,patch,patch-overlap) for left in positions(W,patch,patch-overlap)]
    ops=transforms(tta);enabled,dtype=amp_settings(cfg,device,inference=True)
    for start in range(0,len(coords),batch_size):
        chunk=coords[start:start+batch_size]
        tiles=torch.stack([normalized(x[top:top+patch,left:left+patch],cfg) for top,left in chunk]).to(device)
        probs=torch.zeros((len(chunk),4,patch,patch),device=device,dtype=torch.float32)
        for k,flip in ops:
            with torch.autocast(device.type,enabled=enabled,dtype=dtype):out=model(transform(tiles,k,flip))
            probs.add_(transform(out.float().softmax(1),k,flip,inverse=True))
        arrays=(probs/len(ops)).cpu().numpy()
        for pr,(top,left) in zip(arrays,chunk):
            acc[:,top:top+patch,left:left+patch]+=pr*weight
            norm[top:top+patch,left:left+patch]+=weight
    if np.any(norm<=0) or not np.isfinite(acc).all():raise RuntimeError('Uncovered pixels or nonfinite prediction')
    return (acc/norm[None])[:,:h,:w]
