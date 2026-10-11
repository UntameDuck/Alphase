import json
import math
from pathlib import Path

DEFAULT=dict(model_name='ALPHASE',architecture='unetplusplus',encoder='timm-efficientnet-b3',pretrained=False,
    patch_size=256,overlap=64,batch_size=4,patches_per_image=8,epochs=60,patience=12,lr=.0003,weight_decay=.0001,
    ce_weight=.5,dice_weight=.5,boundary_weight=0.,loss_kind='ce_dice',focal_gamma=2.,tversky_alpha=.3,tversky_beta=.7,
    class_weights=None,rare_probability=0.,rare_class=0,brightness=True,absent_policy='exclude',seed=42,device='auto',amp=True)

# Old configurations retain their old training recipe; v0.3 recipes opt in explicitly.
DEFAULT.update(amp_dtype='auto',encoder_lr=None,warmup_epochs=0,min_epochs=1,
    accumulation_steps=1,ema_decay=0.,freeze_encoder_bn=False,pretrained_permission=False,
    sampler='legacy',phase_probability=.25,boundary_probability=.15,phase_classes=[0,2],
    cache_images=False,brightness_probability=1.,scale_range=[1.,1.],
    dice_reduction='batch',boundary_kind='classwise',inference_batch_size=1,
    inference_amp=False,normalization_mean=[.485,.456,.406],normalization_std=[.229,.224,.225])
DEFAULT.update(decoder_attention=None,lovasz_weight=0.,phase_sampling='pixel',component_min_area=4)

def resolved_config(values):
    cfg=DEFAULT|values
    validate(cfg)
    return cfg

def load_config(path):
    return resolved_config(json.loads(Path(path).read_text(encoding='utf-8')))

def validate(c):
    c=DEFAULT|c
    unknown=set(c)-set(DEFAULT)
    if unknown:raise ValueError(f'Unknown config keys {sorted(unknown)}')
    for k,v in c.items():
        if isinstance(v,float) and not math.isfinite(v):raise ValueError(k+' must be finite')
    for k in ['patch_size','overlap','batch_size','patches_per_image','epochs','patience','seed',
              'warmup_epochs','min_epochs','accumulation_steps','inference_batch_size','rare_class','component_min_area']:
        if type(c[k]) is not int:raise ValueError(k+' must be an integer')
    for k in ['pretrained','brightness','amp','freeze_encoder_bn','pretrained_permission','cache_images','inference_amp']:
        if type(c[k]) is not bool:raise ValueError(k+' must be boolean')
    if c['architecture'] not in ['unet','unetplusplus','deeplabv3plus']:raise ValueError('Unknown architecture')
    if c['decoder_attention'] not in [None,'scse']:raise ValueError('Invalid decoder attention')
    if c['decoder_attention'] and c['architecture']=='deeplabv3plus':raise ValueError('Decoder attention requires a U-Net family model')
    if c['lovasz_weight']<0 or not math.isfinite(c['lovasz_weight']):raise ValueError('Invalid Lovasz weight')
    if c['phase_sampling'] not in ['pixel','component'] or c['component_min_area']<1:raise ValueError('Invalid component sampling')
    if not 0<=c['seed']<2**32:raise ValueError('Seed must fit NumPy uint32')
    if c['patch_size']<32 or c['patch_size']%32:raise ValueError('patch_size must be a positive multiple of 32')
    if not 0<=c['overlap']<c['patch_size']:raise ValueError('Invalid overlap')
    for k in ['batch_size','patches_per_image','epochs','patience']:
        if c[k]<1:raise ValueError(k+' must be positive')
    if not 0<=c['rare_probability']<=1 or c['rare_class'] not in range(4):raise ValueError('Invalid rare sampler')
    if c['absent_policy'] not in ['exclude','zero','one']:raise ValueError('Invalid absent policy')
    if c['device'] not in ['auto','cpu','cuda']:raise ValueError('Invalid device')
    if c['lr']<=0 or c['weight_decay']<0:raise ValueError('Invalid optimizer settings')
    if any(c[k]<0 for k in ['ce_weight','dice_weight','boundary_weight']) or c['ce_weight']+c['dice_weight']<=0:raise ValueError('Invalid loss weights')
    if c['loss_kind'] not in ['ce_dice','focal_dice','ce_tversky']:raise ValueError('Invalid loss kind')
    if c['focal_gamma']<0 or c['tversky_alpha']<0 or c['tversky_beta']<0 or c['tversky_alpha']+c['tversky_beta']==0:raise ValueError('Invalid loss parameters')
    if c['class_weights'] is not None and (len(c['class_weights'])!=4 or min(c['class_weights'])<=0):raise ValueError('Invalid class weights')
    if c['class_weights'] is not None and not all(math.isfinite(x) for x in c['class_weights']):raise ValueError('Nonfinite class weights')
    if c['encoder_lr'] is not None and (not math.isfinite(c['encoder_lr']) or c['encoder_lr']<=0):raise ValueError('Invalid encoder_lr')
    if not 0<=c['warmup_epochs']<c['epochs'] or not 1<=c['min_epochs']<=c['epochs']:raise ValueError('Invalid epoch schedule')
    if c['accumulation_steps']<1 or c['inference_batch_size']<1:raise ValueError('Invalid accumulation/inference batch')
    if not 0<=c['ema_decay']<1:raise ValueError('Invalid EMA decay')
    if c['amp_dtype'] not in ['auto','bf16','fp16']:raise ValueError('Invalid AMP dtype')
    if c['sampler'] not in ['legacy','uniform','phase_boundary']:raise ValueError('Invalid sampler')
    if not (0<=c['phase_probability']<=1 and 0<=c['boundary_probability']<=1 and c['phase_probability']+c['boundary_probability']<=1):raise ValueError('Invalid sampler probabilities')
    if not c['phase_classes'] or len(set(c['phase_classes']))!=len(c['phase_classes']) or any(type(x) is not int or x not in range(4) for x in c['phase_classes']):raise ValueError('Invalid phase classes')
    if not 0<=c['brightness_probability']<=1:raise ValueError('Invalid brightness probability')
    if len(c['scale_range'])!=2 or not all(math.isfinite(x) for x in c['scale_range']) or not 0<c['scale_range'][0]<=c['scale_range'][1]<=2:raise ValueError('Invalid scale range')
    if c['dice_reduction'] not in ['batch','image_present']:raise ValueError('Invalid Dice reduction')
    if c['boundary_kind'] not in ['legacy','classwise']:raise ValueError('Invalid boundary kind')
    for key in ['normalization_mean','normalization_std']:
        if len(c[key])!=3 or not all(math.isfinite(x) for x in c[key]):raise ValueError('Invalid normalization')
    if min(c['normalization_std'])<=0:raise ValueError('Normalization std must be positive')
