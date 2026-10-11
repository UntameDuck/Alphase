"""Model construction and preprocessing. Never download weights during inference."""
import copy
import numpy as np
import torch
import segmentation_models_pytorch as smp
from .config import resolved_config


def build_model(cfg,download=True):
    cfg=resolved_config(cfg)
    if cfg['pretrained'] and download and not cfg['pretrained_permission']:
        raise ValueError('Confirm competition permission before downloading pretrained weights')
    constructors={'unet':smp.Unet,'unetplusplus':smp.UnetPlusPlus,'deeplabv3plus':smp.DeepLabV3Plus}
    options={'decoder_attention_type':cfg['decoder_attention']} if cfg['architecture']!='deeplabv3plus' else {}
    return constructors[cfg['architecture']](encoder_name=cfg['encoder'],
        encoder_weights='imagenet' if cfg['pretrained'] and download else None,
        in_channels=3,classes=4,activation=None,**options)


def normalized(rgb,cfg=None):
    cfg=resolved_config(cfg or {})
    x=rgb.astype(np.float32)/255.
    x=(x-np.asarray(cfg['normalization_mean'],np.float32))/np.asarray(cfg['normalization_std'],np.float32)
    return torch.from_numpy(x.transpose(2,0,1).copy())


def amp_settings(cfg,device,inference=False):
    enabled=device.type=='cuda' and cfg['amp'] and (not inference or cfg['inference_amp'])
    dtype=torch.float16
    if enabled:
        supported=torch.cuda.is_bf16_supported()
        if cfg['amp_dtype']=='bf16' and not supported:raise ValueError('Requested BF16 is unsupported')
        if cfg['amp_dtype']=='bf16' or (cfg['amp_dtype']=='auto' and supported):dtype=torch.bfloat16
    return enabled,dtype


class ModelEMA:
    """Update after successful optimizer steps; copy BN buffers, average parameters."""
    def __init__(self,model,decay):
        self.module=copy.deepcopy(model).eval().requires_grad_(False)
        self.decay=decay
        self.updates=0

    @torch.no_grad()
    def update(self,model):
        self.updates+=1
        decay=min(self.decay,(1+self.updates)/(10+self.updates))
        for target,source in zip(self.module.parameters(),model.parameters()):
            target.lerp_(source.detach(),1-decay)
        for target,source in zip(self.module.buffers(),model.buffers()):
            target.copy_(source)


def make_optimizer(model,cfg):
    encoder_ids={id(p) for p in model.encoder.parameters()}
    groups=[{'params':list(model.encoder.parameters()),'lr':cfg['encoder_lr'] or cfg['lr']},
            {'params':[p for p in model.parameters() if id(p) not in encoder_ids],'lr':cfg['lr']}]
    return torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
