from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
from PIL import Image

CLASSES = ['Primary Si', 'Al3Ni', 'Eutectic Si', 'Al']
PALETTE = np.array([[76,178,76],[25,76,153],[204,204,204],[153,127,76]],np.uint8)

def read_rgb(path):
    with Image.open(path) as im:
        if im.mode=='RGBA' and np.asarray(im.getchannel('A')).min()!=255:
            raise ValueError(f'Nonopaque alpha: {path}')
        return np.array(im.convert('RGB'))

def decode_mask(rgb):
    out=np.full(rgb.shape[:2],255,np.uint8)
    for i,color in enumerate(PALETTE):out[np.all(rgb==color,axis=-1)]=i
    if np.any(out==255):raise ValueError(f'Unknown mask colors: {np.unique(rgb[out==255],axis=0)[:20].tolist()}')
    return out

def pairs(root,split):
    if split not in ['train','valid','test']:raise ValueError(split)
    base=Path(root)/split;images=sorted((base/'images').glob('*_image.png'))
    if not images:raise ValueError(f'No images in {base}')
    rows=[]
    for ip in images:
        mp=base/'masks'/ip.name.replace('_image.png','_mask.png')
        if split!='test' and not mp.exists():raise ValueError(f'Missing mask {mp}')
        rows.append((ip,mp if split!='test' else None))
    if split!='test' and set((base/'masks').glob('*.png'))!={m for _,m in rows}:raise ValueError('Orphan mask files')
    return rows

def extract_archives(source,target):
    target=Path(target).resolve();target.mkdir(parents=True,exist_ok=True)
    for split in ['train','valid','test']:
        with zipfile.ZipFile(Path(source)/f'{split}.zip') as z:
            for item in z.infolist():
                dest=(target/item.filename).resolve()
                if not dest.is_relative_to(target) or item.filename.split('/')[0]!=split:
                    raise ValueError(f'Unsafe ZIP entry {item.filename}')
                if item.external_attr>>16 & 0o170000 == 0o120000:raise ValueError('ZIP symlinks are not allowed')
            z.extractall(target)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def inspect(root,expected=None):
    report={'classes':CLASSES,'palette':PALETTE.tolist(),'splits':{}};hashes={};groups={}
    for split in ['train','valid','test']:
        records=[];counts=np.zeros(4,np.int64);seen=set()
        for ip,mp in pairs(root,split):
            x=read_rgb(ip);digest=hashlib.sha256(x.tobytes()+str(x.shape).encode()).hexdigest()
            if digest in seen:raise ValueError(f'Duplicate image within {split}: {ip.name}')
            seen.add(digest)
            row={'image':ip.name,'height':x.shape[0],'width':x.shape[1],'sha256_rgb':digest}
            if mp:
                y=decode_mask(read_rgb(mp))
                if y.shape!=x.shape[:2]:raise ValueError(f'Shape mismatch {ip.name}')
                c=np.bincount(y.ravel(),minlength=4);counts+=c;row['class_pixels']=c.tolist();row['mask_sha256']=sha256(mp)
            records.append(row)
        if expected and len(records)!=expected[split]:raise ValueError(f'Unexpected {split} count {len(records)}')
        hashes[split]=seen;groups[split]={r['image'].split('_')[0] for r in records}
        report['splits'][split]={'count':len(records),'class_pixels':counts.tolist(),'records':records}
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        if hashes[a]&hashes[b]:raise ValueError(f'Exact duplicate across {a}/{b}')
    report['group_overlap_train_valid']=sorted(groups['train']&groups['valid'])
    return report

def data_fingerprint(root):
    r=inspect(root)
    return hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
