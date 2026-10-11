"""Train-only morphology evidence in pixels, never inferred physical units."""
import numpy as np
from scipy import ndimage as ndi
from .data import pairs,read_rgb,decode_mask,sha256,CLASSES


def audit_train(root):
    areas={c:[] for c in range(4)};images=[]
    for ip,mp in pairs(root,'train'):
        rgb=read_rgb(ip);target=decode_mask(read_rgb(mp))
        if rgb.shape[:2]!=target.shape:raise ValueError('Image/mask shape mismatch')
        counts=[]
        for c in range(4):
            labels,n=ndi.label(target==c,np.ones((3,3),int))
            sizes=np.bincount(labels.ravel(),minlength=n+1)[1:]
            areas[c].extend(sizes.tolist());counts.append(n)
        images.append({'image':ip.name,'input_sha256':sha256(ip),'mask_sha256':sha256(mp),
            'shape':list(target.shape),'component_counts':counts})
    rows=[]
    for c in range(4):
        a=np.asarray(areas[c],dtype=np.int64);total=int(a.sum())
        rows.append({'class_id':c,'phase':CLASSES[c],'component_count':len(a),'pixel_count':total,
            'area_quantiles':dict(zip(['min','p25','p50','p75','p90','p99','max'],np.quantile(a,[0,.25,.5,.75,.9,.99,1]).tolist())) if len(a) else None,
            'component_fraction_le64':float((a<=64).mean()) if len(a) else None,
            'phase_pixel_fraction_in_le64':float(a[a<=64].sum()/total) if total else None,
            'component_fraction_below4':float((a<4).mean()) if len(a) else None})
    return {'split':'train','image_count':len(images),'connectivity':8,'classes':rows,'images':images,
        'note':'Connected components are label regions, not independently verified physical particles. Pixel area is not physical area. No validation/test statistics used to fit sampling.'}
