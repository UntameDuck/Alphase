from pathlib import Path
import csv
import json
import numpy as np
from .data import save_json

def estimate(run,hourly_rate,output):
    if not np.isfinite(hourly_rate) or hourly_rate<0:raise ValueError('Hourly rate must be finite and nonnegative')
    run=Path(run);cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    with (run/'history.csv').open() as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No measured epoch times')
    seconds=np.array([float(r['seconds']) for r in rows]);last=int(rows[-1]['epoch'])
    if not np.isfinite(seconds).all() or np.any(seconds<0):raise ValueError('Invalid measured epoch times')
    status_path=run/'run_status.json'
    status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    completed=status.get('completed') is True and status.get('completed_epochs')==last
    remaining=0 if completed else max(0,cfg['epochs']-last)
    result={'measured_epochs':len(rows),'mean_seconds_per_epoch':float(seconds.mean()),'measured_seconds':float(seconds.sum()),
        'completed':completed,'remaining_max_epochs':remaining,'estimated_remaining_hours':float(remaining*seconds.mean()/3600),
        'hourly_rate_usd':hourly_rate,'estimated_remaining_compute_usd':float(remaining*seconds.mean()/3600*hourly_rate),
        'note':'Compute/validation epoch extrapolation; excludes checkpoint/export I/O, startup, storage, bandwidth, other experiments and final inference. Use wall-clock benchmark and overhead reserve for total spending. Early stopping may reduce usage.'}
    save_json(output,result);return result
