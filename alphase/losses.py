"""FP32 overlap/boundary arithmetic, including under mixed precision."""
import torch
import torch.nn.functional as F
from .config import resolved_config


def lovasz_image_loss(prob,target,ignore=255):
    """Per-image, present-class Lovasz-Softmax surrogate (Berman et al., CVPR 2018).

    Independent implementation of sorted errors dotted with Jaccard increments.
    CE remains responsible for false positives on GT-absent classes. No top-k
    approximation, downsampling, or detach of the prediction gradient.
    """
    image_losses=[]
    for p,y in zip(prob.float(),target):
        valid=y!=ignore;y=y[valid];p=p.permute(1,2,0)[valid]
        if not y.numel():raise ValueError('All-ignore crop')
        classes=[]
        for c in y.unique():
            foreground=(y==c).float();errors=(foreground-p[:,int(c)]).abs()
            errors,order=torch.sort(errors,descending=True,stable=True)
            sorted_fg=foreground[order];count=sorted_fg.sum()
            intersection=count-sorted_fg.cumsum(0)
            union=count+(1-sorted_fg).cumsum(0)
            jaccard=1-intersection/union
            gradient=torch.cat([jaccard[:1],jaccard[1:]-jaccard[:-1]])
            classes.append(torch.dot(errors,gradient))
        image_losses.append(torch.stack(classes).mean())
    return torch.stack(image_losses).mean()


def classwise_boundary_loss(prob,gt,valid):
    invalid=F.pad((~valid).float()[:,None],(1,1,1,1),value=1)
    safe=F.max_pool2d(invalid,3,1,0)==0
    if not safe.any():return prob.sum()*0
    def edge(a):return F.max_pool2d(a,3,1,1)+F.max_pool2d(-a,3,1,1)
    pred=edge(prob);target=edge(gt)
    # No amax across classes: swapping the phase on a boundary must be penalized.
    dims=(2,3);p=pred*safe;g=target*safe
    present=g.sum(dims)>0
    loss=1-(2*(p*g).sum(dims)+1e-6)/(p.sum(dims)+g.sum(dims)+1e-6)
    return loss[present].mean() if present.any() else p.mean()


def loss_value(logits,y,cfg):
    cfg=resolved_config(cfg);valid=y!=255
    if not valid.any():raise ValueError('All-ignore crop')
    if not torch.all(valid.flatten(1).any(1)):raise ValueError('Batch contains an all-ignore crop')
    z=logits.float();weights=cfg['class_weights']
    weights=torch.tensor(weights,device=z.device,dtype=torch.float32) if weights else None
    ce_map=F.cross_entropy(z,y,ignore_index=255,weight=weights,reduction='none')
    if cfg['loss_kind']=='focal_dice':
        logp=F.log_softmax(z,1).gather(1,y.clamp(0,3)[:,None])[:,0]
        ce_map=-(1-logp.exp()).pow(cfg['focal_gamma'])*logp
        if weights is not None:ce_map=ce_map*weights[y.clamp(0,3)]
    # Equal crop weight even when padding contributes different numbers of valid pixels.
    ce=((ce_map*valid).sum((1,2))/valid.sum((1,2))).mean()
    prob=z.softmax(1);gt=F.one_hot(y.clamp(0,3),4).permute(0,3,1,2).float()
    q=prob*valid[:,None];g=gt*valid[:,None]
    dims=(0,2,3) if cfg['dice_reduction']=='batch' else (2,3)
    tp=(q*g).sum(dims)
    if cfg['loss_kind']=='ce_tversky':
        fp=(q*(1-g)).sum(dims);fn=((1-q)*g).sum(dims)
        overlap=1-(tp+1e-6)/(tp+cfg['tversky_alpha']*fp+cfg['tversky_beta']*fn+1e-6)
    else:overlap=1-(2*tp+1e-6)/(q.sum(dims)+g.sum(dims)+1e-6)
    if cfg['dice_reduction']=='image_present':
        present=g.sum(dims)>0
        overlap=((overlap*present).sum(1)/present.sum(1).clamp_min(1)).mean()
    else:overlap=overlap.mean()
    loss=cfg['ce_weight']*ce+cfg['dice_weight']*overlap
    if cfg['lovasz_weight']:loss=loss+cfg['lovasz_weight']*lovasz_image_loss(prob,y)
    if cfg['boundary_weight']:
        if cfg['boundary_kind']=='classwise':boundary=classwise_boundary_loss(prob,g,valid)
        else:
            def bd(a):return (F.max_pool2d(a,3,1,1)+F.max_pool2d(-a,3,1,1)).amax(1)
            invalid=F.pad((~valid).float()[:,None],(1,1,1,1),value=1)
            safe=F.max_pool2d(invalid,3,1,0)[:,0]==0
            boundary=(bd(prob)-bd(g)).abs()[safe].mean() if safe.any() else z.sum()*0
        loss=loss+cfg['boundary_weight']*boundary
    return loss
