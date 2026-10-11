"""Content-addressed inference/evaluation records; filenames alone are not identity."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import sha256


SPEC_KEYS = ('checkpoint_sha256', 'configs', 'weights', 'tta', 'source_fingerprint')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()+str(array.shape).encode()).hexdigest()


def source_fingerprint():
    files = sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(''.join(p.name+':'+sha256(p)+'\n' for p in files).encode()).hexdigest()


def prediction_fingerprint(record):
    return digest({key: record[key] for key in SPEC_KEYS})


def validate_evaluation(folder, require_cache=False):
    folder = Path(folder)
    record = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    if record.get('schema_version') != 4 or not record.get('complete'):
        raise ValueError('Unverified/legacy evaluation: re-evaluate into a new directory')
    if any(key not in record for key in SPEC_KEYS):
        raise ValueError('Evaluation inference provenance is missing')
    if record.get('prediction_fingerprint') != prediction_fingerprint(record):
        raise ValueError('Evaluation inference provenance changed')
    n = len(record['configs'])
    if not n or len(record['checkpoint_sha256']) != n or len(record['weights']) != n:
        raise ValueError('Evaluation checkpoint metadata lengths differ')
    weights = np.asarray(record['weights'])
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid evaluation ensemble weights')
    if record['tta'] not in ('none', 'flip', 'd4'):
        raise ValueError('Invalid recorded TTA mode')
    rows = record['images']
    if not rows or len({row['image'] for row in rows}) != len(rows):
        raise ValueError('Missing/duplicate evaluation images')
    for row in rows:
        if not row.get('input_sha256') or not row.get('target_sha256'):
            raise ValueError('Evaluation input/target fingerprints are missing')
        if not np.isfinite(row['miou']) or not 0 <= row['miou'] <= 1:
            raise ValueError('Invalid evaluation score')
    if not np.isclose(record['image_macro_miou'], np.mean([r['miou'] for r in rows])):
        raise ValueError('Evaluation aggregate disagrees with per-image scores')
    if require_cache:
        manifest = record.get('cache_manifest', {})
        expected = {Path(row['image']).stem+'.npz' for row in rows}
        if set(manifest) != expected or {p.name for p in folder.glob('*.npz')} != expected:
            raise ValueError('Evaluation cache manifest differs from evaluated images')
        if any(sha256(folder/name) != manifest[name] for name in expected):
            raise ValueError('Evaluation cache content hash mismatch')
    return record
