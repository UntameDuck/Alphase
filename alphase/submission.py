"""Local dry-run submission validation and deterministic packaging; no upload."""
from pathlib import Path
import json
import zipfile
import tempfile
import shutil
import subprocess
import sys
import re
from PIL import Image
import numpy as np
from .data import pairs,decode_mask,sha256,save_json
from .provenance import source_fingerprint,array_digest

DEFAULT_CONTRACT={'expected_count':10,'input_suffix':'_image.png','output_suffix':'_mask.png','mode':'RGB',
    'filename_map':None,
    'absent_policy_confirmed':False,'pretrained_allowed_confirmed':False,'output_format_confirmed':False}

def load_contract(path=None):
    c=DEFAULT_CONTRACT|(json.loads(Path(path).read_text(encoding='utf-8')) if path else {})
    if set(c)-set(DEFAULT_CONTRACT):raise ValueError('Unknown output contract fields')
    for key in ['absent_policy_confirmed','pretrained_allowed_confirmed','output_format_confirmed']:
        if type(c[key]) is not bool:raise ValueError('Contract confirmation flags must be booleans')
    if type(c['expected_count']) is not int:raise ValueError('Expected count must be an integer')
    if c['mode'] not in ['RGB','RGBA'] or c['expected_count']<1:raise ValueError('Invalid output contract')
    if not c['input_suffix'].endswith('.png') or not c['output_suffix'].endswith('.png'):raise ValueError('Only PNG contracts supported')
    if any('/' in c[k] or '\\' in c[k] for k in ['input_suffix','output_suffix']):raise ValueError('Suffix must not contain a path')
    return c

def output_name(input_name,c):
    mapping=c.get('filename_map')
    if mapping is not None:
        if input_name not in mapping:raise ValueError('Missing output filename mapping')
        name=mapping[input_name]
    else:
        if not input_name.endswith(c['input_suffix']):raise ValueError('Input suffix does not match')
        name=input_name[:-len(c['input_suffix'])]+c['output_suffix']
    if Path(name).name!=name or '\\' in name or not name.endswith('.png'):raise ValueError('Unsafe output filename')
    return name

def validate_predictions(root,predictions,contract=None):
    c=DEFAULT_CONTRACT|(contract or {});rows=pairs(root,'test');folder=Path(predictions)
    expected={output_name(ip.name,c):ip for ip,_ in rows}
    if len(expected)!=len(rows):raise ValueError('Output filename collision')
    if len(expected)!=c['expected_count']:raise ValueError(f'Expected {c["expected_count"]} test images, got {len(expected)}')
    actual={p.name for p in folder.glob('*.png')}
    if actual!=set(expected):raise ValueError(f'Prediction file mismatch missing={set(expected)-actual}, extra={actual-set(expected)}')
    files=[]
    for name,ip in sorted(expected.items()):
        pp=folder/name
        with Image.open(ip) as x,Image.open(pp) as p:
            if p.format!='PNG' or p.mode!=c['mode'] or p.size!=x.size:raise ValueError(f'Wrong mode/size/format: {name}')
            if p.mode=='RGBA' and np.asarray(p.getchannel('A')).min()!=255:raise ValueError('Output alpha must be opaque')
            decode_mask(np.asarray(p.convert('RGB')))
            files.append({'filename':name,'width':p.width,'height':p.height,'sha256':sha256(pp)})
    return {'status':'pass','count':len(files),'files':files,'contract':c,
        'format_status':'confirmed' if c['output_format_confirmed'] else 'provisional_contract'}

def reproducibility_check(root,checkpoints,existing,device='auto',weights=None,tta=False,contract=None):
    from .engine import infer
    c=load_contract(contract);first=validate_predictions(root,existing,c)
    with tempfile.TemporaryDirectory(prefix='alphase_reproduce_') as d:
        infer(root,checkpoints,d,device,weights,tta,contract);second=validate_predictions(root,d,c)
    hashes1={x['filename']:x['sha256'] for x in first['files']};hashes2={x['filename']:x['sha256'] for x in second['files']}
    if hashes1!=hashes2:raise ValueError('Repeated inference PNG hashes differ')
    return {'status':'pass','identical_png_hashes':True,'count':first['count'],
        'note':'Same-environment repeat, not a clean-environment installation test.'}

def build_package(root,predictions,checkpoints,report,receipt,output,contract=None,weights=None,tta=False,device='auto'):
    c=load_contract(contract)
    if not c['output_format_confirmed'] or not c['absent_policy_confirmed']:
        raise ValueError('Confirm official output format and absent-class policy in contract before final packaging')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',receipt):raise ValueError('Invalid receipt filename')
    output=Path(output).resolve()
    if output.name!=receipt+'.zip':raise ValueError('Output filename must equal receipt.zip')
    if output.exists():raise ValueError('Submission ZIP already exists; choose a new path')
    report=Path(report)
    if report.suffix.lower()!='.pdf':raise ValueError('Package requires the final report as PDF')
    from pypdf import PdfReader
    pages=len(PdfReader(report).pages)
    if not 1<=pages<=10:raise ValueError('Report must be 1..10 pages')
    validated=validate_predictions(root,predictions,c)
    # Export only inference payloads; never optimizer states, training labels or test input data.
    import torch
    from .engine import normalized_weights,tta_mode
    w=normalized_weights(weights,len(checkpoints)).tolist();source=Path(__file__).resolve().parents[1]
    manifest_path=Path(predictions)/'manifest.json'
    if not manifest_path.exists():raise ValueError('Missing inference manifest; regenerate predictions')
    recorded=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (recorded.get('schema_version')!=4 or recorded.get('checkpoint_sha256')!=[sha256(p) for p in checkpoints] or
        recorded.get('weights')!=w or recorded.get('tta')!=tta_mode(tta) or recorded.get('contract')!=c or
        recorded.get('source_fingerprint')!=source_fingerprint()):
        raise ValueError('Prediction manifest does not match package checkpoint/weights/TTA/contract/source')
    actual={r['filename']:r['sha256'] for r in validated['files']}
    if {r['output']:r['output_sha256'] for r in recorded['images']}!=actual:raise ValueError('Prediction output hashes changed')
    from .data import read_rgb
    if {r['input']:r['input_sha256'] for r in recorded['images']}!={ip.name:array_digest(read_rgb(ip)) for ip,_ in pairs(root,'test')}:
        raise ValueError('Prediction input hashes changed')
    # Do not call a format-only ZIP a passing model submission.
    reproduction=reproducibility_check(root,checkpoints,predictions,device,weights,tta,contract)
    with tempfile.TemporaryDirectory(prefix='alphase_package_') as temp:
        base=Path(temp);dest=base/'source';dest.mkdir()
        for folder,pattern in [('alphase','*.py'),('configs','*.json')]:
            (dest/folder).mkdir()
            for p in (source/folder).glob(pattern):shutil.copy2(p,dest/folder/p.name)
        for name in ['README.md','requirements.txt','environment.yaml','pyproject.toml']:
            if (source/name).exists():shutil.copy2(source/name,dest/name)
        if (source/'docs').exists():shutil.copytree(source/'docs',dest/'docs')
        model_paths=[]
        for i,path in enumerate(checkpoints):
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state['config']['pretrained'] and not c['pretrained_allowed_confirmed']:raise ValueError('Pretrained permission not confirmed')
            model_paths.append(f'model_{i}.pt');torch.save({'model':state.get('eval_model',state['model']),'config':state['config']},dest/model_paths[-1])
        save_json(dest/'inference_spec.json',{'checkpoints':model_paths,'weights':w,'tta':tta,'contract':c})
        wrapper='''#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from alphase.engine import infer
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='auto');a=p.parse_args()
base=Path(__file__).resolve().parent;s=json.loads((base/'inference_spec.json').read_text(encoding='utf-8'))
(base/'contract.json').write_text(json.dumps(s['contract']))
infer(a.data,[str(base/x) for x in s['checkpoints']],a.output,a.device,s['weights'],s['tta'],base/'contract.json')
'''
        (dest/'run_inference.py').write_text(wrapper);(dest/'run_inference.sh').write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nexec python run_inference.py "$@"\n')
        (dest/'run_inference.sh').chmod(0o755)
        predictions_zip=base/'test_predictions.zip'
        with zipfile.ZipFile(predictions_zip,'w',zipfile.ZIP_DEFLATED) as z:
            for f in validated['files']:z.write(Path(predictions)/f['filename'],f['filename'])
        (base/'report').mkdir();shutil.copy2(report,base/'report'/'modeling_report.pdf')
        save_json(base/'validation.json',validated|{'report_pages':pages,'reproduction':reproduction})
        manifest={str(p.relative_to(base)):sha256(p) for p in base.rglob('*') if p.is_file()}
        save_json(base/'SHA256.json',manifest);output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(base.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(base).as_posix())
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise ValueError('ZIP integrity failure')
    return {'status':'pass','zip':str(output),'sha256':sha256(output),'report_pages':pages,'predictions':validated['count'],
        'repeated_inference_verified':True,
        'note':'Same-environment inference verified. Run verify-package in the target clean environment; no submission performed.'}

def verify_package(archive,data,python=sys.executable,device='cpu'):
    with tempfile.TemporaryDirectory(prefix='alphase_package_verify_') as tmp:
        root=Path(tmp).resolve()
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (root/item.filename).resolve().is_relative_to(root):raise ValueError('Unsafe archive path')
            z.extractall(root)
        manifest=json.loads((root/'SHA256.json').read_text(encoding='utf-8'))
        for name,digest in manifest.items():
            if not (root/name).resolve().is_relative_to(root) or sha256(root/name)!=digest:raise ValueError('Package file hash mismatch')
        spec=json.loads((root/'source'/'inference_spec.json').read_text(encoding='utf-8'));expected=root/'expected';expected.mkdir()
        with zipfile.ZipFile(root/'test_predictions.zip') as z:
            for name in z.namelist():
                if Path(name).name!=name:raise ValueError('Unexpected prediction ZIP path')
            z.extractall(expected)
        actual=root/'regenerated'
        # Caller can supply a clean virtualenv interpreter. Do not inherit test-only PYTHONPATH.
        import os
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        subprocess.run([str(python),str(root/'source'/'run_inference.py'),'--data',str(Path(data).resolve()),
            '--output',str(actual),'--device',device],cwd=root/'source',env=env,check=True)
        a=validate_predictions(data,expected,spec['contract']);b=validate_predictions(data,actual,spec['contract'])
        if {r['filename']:r['sha256'] for r in a['files']}!={r['filename']:r['sha256'] for r in b['files']}:raise ValueError('Regenerated PNG hashes differ')
        return {'status':'pass','count':a['count'],'interpreter':str(python),'identical_png_hashes':True,
            'note':'Package re-execution. Clean-environment claim requires caller-provided clean interpreter and dependencies.'}
