"""Exact, data-grounded PNG panels and static HTML reports."""
from pathlib import Path
import html
import json
import numpy as np
from PIL import Image,ImageDraw
from .data import PALETTE,CLASSES

def panel(path,rgb,target,pred,prob):
    h,w=pred.shape;error=np.zeros((h,w,3),np.uint8);error[pred!=target]=[255,70,70]
    confidence=prob.max(0);uncertainty=1-confidence
    heat=np.stack([uncertainty*255,np.zeros_like(uncertainty),confidence*255],-1).astype(np.uint8)
    overlay=(.55*rgb+.45*PALETTE[pred]).astype(np.uint8)
    blocks=[('Input',rgb),('Ground truth',PALETTE[target]),('Prediction',PALETTE[pred]),('Overlay',overlay),('Error (red)',error),('Uncertainty',heat)]
    scale=min(1.,480/w);pw=max(1,round(w*scale));ph=max(1,round(h*scale));canvas=Image.new('RGB',(pw*3,(ph+30)*2),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,a) in enumerate(blocks):
        x=(i%3)*pw;y=(i//3)*(ph+30);draw.text((x+8,y+8),label,fill='black')
        canvas.paste(Image.fromarray(a).resize((pw,ph),Image.Resampling.NEAREST),(x,y+30))
    canvas.save(path)

def report_html(folder,result):
    folder=Path(folder);rows=sorted(result['images'],key=lambda r:r['miou'])
    headers=''.join(f'<th>{html.escape(c)}</th>' for c in CLASSES)
    body=[]
    for r in rows:
        vals=''.join(f'<td>{c["iou"]:.4f}</td>' if c['iou'] is not None else '<td>absent</td>' for c in r['classes'])
        file=r.get('panel');img=f'<a href="{html.escape(file,quote=True)}"><img loading="lazy" src="{html.escape(file,quote=True)}"></a>' if file else ''
        body.append(f'<tr><td>{html.escape(r["image"])}</td><td>{r["miou"]:.4f}</td>{vals}</tr><tr><td colspan="6">{img}</td></tr>')
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ALPHASE evaluation</title>
<style>body{{font:15px system-ui;max-width:1450px;margin:32px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border:1px solid #ddd;text-align:left}}img{{width:100%;max-width:1440px}}pre{{white-space:pre-wrap}}</style>
<h1>ALPHASE validation</h1><p>Image macro mIoU: {result['image_macro_miou']:.4f}. Absent policy: {html.escape(result['absent_policy'])}. Worst images first.</p>
<p>Diagnostic boundaries: Euclidean tolerance {result['boundary_tolerance']} pixels. Small GT components: area ≤ {result['small_area']} pixels. Confidence is uncalibrated.</p>
<table><tr><th>Image</th><th>mIoU</th>{headers}</tr>{''.join(body)}</table></html>'''
    (folder/'report.html').write_text(text,encoding='utf-8')
