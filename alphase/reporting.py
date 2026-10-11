"""Generate a source-grounded editable report draft; never fabricate missing scores."""
from pathlib import Path
import json
from .data import CLASSES

def draft(data_report,evaluation,comparison,output):
    data=json.loads(Path(data_report).read_text(encoding='utf-8'));ev=json.loads(Path(evaluation).read_text(encoding='utf-8'));comp=json.loads(Path(comparison).read_text(encoding='utf-8'))
    selected=ev.get('configs',[])
    architecture='; '.join(f"{c['architecture']} / {c['encoder']} / pretrained={c['pretrained']} / patch={c['patch_size']} / EMA={c.get('ema_decay',0)}" for c in selected)
    architecture=architecture or 'Checkpoint config metadata missing: supply and verify the selected architecture manually.'
    pix=data['splits']['train']['class_pixels'];total=sum(pix)
    lines=['# ALPHASE — KoMaP modeling report draft','',
        'This is an editable draft, not a final submission report. Review claims, figures and the organizer template before exporting PDF.',
        '', '## 1. Problem and objective','Four-phase semantic segmentation of aluminum-alloy optical microstructures.',
        '', '## 2. Data and quality checks',f'Train/valid/test: {data["splits"]["train"]["count"]}/{data["splits"]["valid"]["count"]}/{data["splits"]["test"]["count"]}.',
        '', '| Phase | Train pixels | Train fraction |','|---|---:|---:|']
    lines += [f'| {name} | {pix[i]} | {pix[i]/total:.4%} |' for i,name in enumerate(CLASSES)] if total else ['No train distribution available.']
    lines += ['',f'Cross-split group identifiers: {data.get("group_overlap_train_valid",[])}. Official image split retained; investigate sample relationship.',
        '', '## 3. Architecture',architecture,
        f"Inference weights: {ev.get('weights','not recorded')}; TTA: {ev.get('tta','not recorded')}. Existing architectures are credited; no new-backbone claim.",
        '', '## 4. Training','Patch sampling, synchronized rotation/flips, CE/overlap loss, AdamW, cosine schedule and checkpoint-based early stopping. Cite the actual config and environment.',
        '', '## 5. Evaluation',f'Image-wise four-class mIoU, then mean across images. Absent policy used: {ev["absent_policy"]}. Confirm this with the organizer.',
        '', '## 6. Measured validation results',f'Measured image macro mIoU: {ev["image_macro_miou"]:.6f}.',
        '', '| Phase | Image macro IoU | Boundary F1 |','|---|---:|---:|']
    def fmt(x):return f'{x:.6f}' if x is not None else 'not measured / absent'
    lines += [f'| {name} | {fmt(ev["class_macro_iou"][i])} | {fmt(ev["class_macro_boundary_f1"][i])} |' for i,name in enumerate(CLASSES)]
    lines += ['', '### 2D phase-area application check',
        'Image-mean absolute error of predicted 2D area fractions (fraction units, not physical volume or modulus):',
        '; '.join(f'{name}: {fmt(value)}' for name,value in zip(CLASSES,ev.get('class_macro_area_fraction_mae',[None]*4))),
        'Use failure panels and a domain expert review before quantitative materials decisions.']
    lines += ['', '## 7. Ablation and contribution','| Variant | Seed | Validation mIoU |','|---|---:|---:|']
    lines += [f'| {r["id"]} | {r["seed"]} | {r["miou"]:.6f} |' for r in comp['runs']]
    lines += ['', 'Missing experiments: '+(', '.join(comp['missing']) or 'none'),
        'Selection status: '+comp.get('selection_status','not recorded')+'. Unmeasured improvements are hypotheses.',
        'Discuss encoder, decoder, sampler and loss effects only where matched experiments exist. Bootstrap intervals are exploratory.',
        '', '## 8. Error analysis',f'Boundary tolerance: {ev["boundary_tolerance"]} pixels; small-object area: {ev["small_area"]} pixels.',
        'Small-object metrics are GT component recall, not instance IoU. Add the worst-image panels from report.html.',
        '', '## 9. Limitations and applications','Small dataset, correlated sample groups, acquisition variation, uncalibrated confidence and validation selection bias. Expert review remains necessary.',
        '', '## 10. Reproduction and conclusion','List final checkpoint hashes, config, dependency lock, inference commands and package verification result. Do not insert unmeasured test accuracy.',
        '', 'References: KoMaP official data/metric description; U-Net++; EfficientNet; ConvNeXt; SMP; Lovasz-Softmax (Berman et al., CVPR 2018); scSE (Roy et al., MICCAI 2018). Cite only techniques actually used in the selected model.',
        '', 'Rubric evidence: accuracy 50, originality/new technology 20, data analysis 15, application potential 10, task understanding 5. These are official rubric weights, not scores awarded to this project. See https://komap.ai/front/competition2026/award_info .']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines),encoding='utf-8')
    return {'status':'draft_written','path':str(output)}
