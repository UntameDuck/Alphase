"""Image-uniform patch exposure; labels are read from train only."""
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .config import resolved_config
from .data import pairs,read_rgb,decode_mask
from .models import normalized
from scipy import ndimage as ndi


def component_coordinates(mask,class_id,min_area=4):
    """Flat pixel coordinates grouped by eligible 8-connected component.

    Stores phase pixels once, not a full label image per cached class. Tiny
    components still participate in uniform/pixel sampling and all losses.
    """
    labels,n=ndi.label(mask==class_id,np.ones((3,3),int))
    sizes=np.bincount(labels.ravel(),minlength=n+1)
    ids=np.flatnonzero(sizes>=min_area);ids=ids[ids!=0]
    if not len(ids):return []
    coords=np.flatnonzero(np.isin(labels,ids)).astype(np.int32)
    component_ids=labels.ravel()[coords];order=np.argsort(component_ids,kind='stable')
    coords=coords[order];cuts=np.flatnonzero(np.diff(component_ids[order]))+1
    return np.split(coords,cuts)


def phase_boundary(mask):
    out=np.zeros(mask.shape,bool)
    valid=mask!=255
    d=(mask[1:]!=mask[:-1]) & valid[1:] & valid[:-1]
    out[1:]|=d;out[:-1]|=d
    d=(mask[:,1:]!=mask[:,:-1]) & valid[:,1:] & valid[:,:-1]
    out[:,1:]|=d;out[:,:-1]|=d
    return out


class PatchDataset(Dataset):
    def __init__(self,root,cfg):
        self.items=pairs(root,'train');self.cfg=resolved_config(cfg);self.cache={};self.component_cache={};self.epoch=0

    def set_epoch(self,epoch):self.epoch=epoch

    def __len__(self):return len(self.items)*self.cfg['patches_per_image']

    def _read(self,ident):
        if ident in self.cache:return self.cache[ident]
        ip,mp=self.items[ident];x=read_rgb(ip);y=decode_mask(read_rgb(mp))
        if x.shape[:2]!=y.shape:raise ValueError('Image/mask shape mismatch')
        coordinates={c:np.flatnonzero(y==c).astype(np.int32) for c in self.cfg['phase_classes']}
        edges=np.flatnonzero(phase_boundary(y)).astype(np.int32)
        value=(x,y,coordinates,edges)
        if self.cfg['cache_images']:self.cache[ident]=value
        return value

    def _components(self,ident,y,c):
        key=(ident,c,self.cfg['component_min_area'])
        if key not in self.component_cache:
            groups=component_coordinates(y,c,self.cfg['component_min_area'])
            if self.cfg['cache_images']:self.component_cache[key]=groups
            return groups
        return self.component_cache[key]

    def __getitem__(self,index):
        cfg=self.cfg
        # Independent of loader order/workers; exact epoch-boundary resume.
        rng=random.Random((cfg['seed']<<40)+(self.epoch<<20)+index)
        x,y,coordinates,edges=self._read(index//cfg['patches_per_image'])
        h0,w0=y.shape;p=cfg['patch_size']
        # Crop a variable field of view, then resize once; masks always nearest-neighbor.
        size=max(1,round(p*rng.uniform(*cfg['scale_range'])))
        dh=max(0,size-h0);dw=max(0,size-w0)
        if dh or dw:
            x=np.pad(x,((0,dh),(0,dw),(0,0)),mode='edge')
            y=np.pad(y,((0,dh),(0,dw)),constant_values=255)
        h,w=y.shape;top=rng.randint(0,h-size);left=rng.randint(0,w-size)
        selected=None
        if cfg['sampler']=='phase_boundary':
            draw=rng.random()
            if draw<cfg['phase_probability']:
                present=[c for c,coords in coordinates.items() if len(coords)]
                if present:
                    c=rng.choice(present);coords=coordinates[c]
                    if cfg['phase_sampling']=='component':
                        groups=self._components(index//cfg['patches_per_image'],y[:h0,:w0],c)
                        if groups:coords=groups[rng.randrange(len(groups))]
                    selected=int(coords[rng.randrange(len(coords))])
            elif draw<cfg['phase_probability']+cfg['boundary_probability'] and len(edges):
                selected=int(edges[rng.randrange(len(edges))])
        elif cfg['sampler']=='legacy' and rng.random()<cfg['rare_probability']:
            coords=np.flatnonzero(y[:h0,:w0]==cfg['rare_class'])
            if len(coords):selected=int(coords[rng.randrange(len(coords))])
        if selected is not None:
            cy,cx=divmod(selected,w0)
            top=min(max(cy-rng.randrange(size),0),h-size)
            left=min(max(cx-rng.randrange(size),0),w-size)
        x=x[top:top+size,left:left+size];y=y[top:top+size,left:left+size]
        if size!=p:
            x=np.asarray(Image.fromarray(x).resize((p,p),Image.Resampling.BILINEAR))
            y=np.asarray(Image.fromarray(y).resize((p,p),Image.Resampling.NEAREST))
        k=rng.randrange(4);x=np.rot90(x,k);y=np.rot90(y,k)
        if rng.random()<.5:x=np.flip(x,1);y=np.flip(y,1)
        if cfg['brightness'] and rng.random()<cfg['brightness_probability']:
            x=np.clip(x.astype(np.float32)*rng.uniform(.9,1.1)+rng.uniform(-8,8),0,255).astype(np.uint8)
        return normalized(x,cfg),torch.from_numpy(y.astype(np.int64).copy())
