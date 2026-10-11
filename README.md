# ALPHASE v0.4 — KoMaP 경쟁 설계·구현

**현재 우선 기준: [v4 업그레이드·실험 프로토콜](docs/UPGRADE_V4.md).** 아래 v3/v2 설명에서 상충하는 재개·캐시·추천 동작보다 우선합니다.

**다음 작업을 시작할 때:** [단계별 행동 지침](docs/ACTION_GUIDE.md) → [실행 체크리스트](docs/EXECUTION_CHECKLIST.md). 실제 승인·비용·명령·결과는 [기록 양식](docs/EXECUTION_RECORD_TEMPLATE.md)을 복사해 남기고, 실패·중단 시 [복구 지침](docs/RECOVERY_GUIDE.md)을 따릅니다. 현재 다음 단계는 로컬 기준선 확인과 규정 확정이며, 문서 작성은 GPU 지출·학습·제출 승인이 아닙니다.

- 추가 후보: O(component sampling), L(per-image Lovasz), S(scSE decoder). 기본에 일괄 적용하지 않고 F와 단일 변경 비교합니다.
- 완료 run 재학습 방지, self-contained `last.pt` 복구, immutable 평가·cache 출처 검증, 공정 조건·다수 seed 채택 게이트를 구현했습니다.
- 재개는 v4 `last.pt`만 지원합니다. `best.pt`/`inference.pt`는 추론 전용이며 구버전 metrics/cache는 새 폴더에서 재평가합니다.
- `package`는 manifest 대조와 반복 추론을 필수 실행합니다. `--device cpu|cuda|auto`로 장비를 지정합니다.
- morphology: `python -m alphase morphology --data ../data --output runs/train_morphology_v4.json`.
- GPU 지출·본학습·최종 성능·수상 입증은 아직 하지 않았습니다.

## v3 설계 및 기존 사용법

**현재 기준은 [설계 문서](docs/README.md)입니다.** 기존 기본 설정은 비교/호환용으로 보존했습니다.

- 주력: `configs/champion.json` — B3/U-Net++, 512 crop, phase/boundary sampler, EMA, 누적 학습.
- 도전 모델: `configs/challenger.json` — ConvNeXt-Tiny/U-Net. 아직 검증 점수로 우열을 측정하지 않았습니다.
- [모델 설계](docs/MODEL_DESIGN.md), [폴더·파일별 문서](docs/FILE_MAP.md), [실행·비용·마감](docs/RUNBOOK.md), [최신 검증](docs/VERIFICATION.md).
- GPU 대여/지출/본학습/대회 제출은 미수행. 사전학습 허용·공식 제출 계약 확인이 필요합니다.
- `pretrained=true` 학습은 `pretrained_permission=true`도 필요합니다. 확인된 경우에만 두 값을 설정하세요.
- v0.2 checkpoint는 추론만 호환합니다. v0.3 재개는 config/data/source fingerprint가 같은 v3 checkpoint만 허용합니다.
- TTA는 `--tta-mode none|flip|d4`; 기존 `--tta`는 flip입니다. `benchmark`는 GPU timing용이며 성능 시험이 아닙니다.
- `SOURCE_SHA256.json`은 현재 소스 manifest, workspace 상위 `ALPHASE_code.zip`은 이전 v0.2 archive입니다.

```bash
python -m unittest discover -s tests -v
python -m alphase experiment-plan --config configs/champion.json --ids E F CN --seeds 42 --output plans/screen_v3
# 아래 학습은 GPU 승인·규정·자원 probe 후에만 실행
python -m alphase experiments --data ../data --plan plans/screen_v3/plan.json --output runs/screen_v3
```

## 이전 v0.2 사용 설명과 호환 명령

2026 KoMaP 알루미늄 합금 OM 이미지 4상 분할 코드. 실제 데이터 전체 본학습과 GPU 실행은
수행하지 않았습니다. 데이터·학습 가중치는 코드 ZIP에 포함하지 않습니다.

## 빠른 시작

GPU용 PyTorch/torchvision이 설치된 Vast.ai PyTorch+Jupyter 또는 Colab GPU 환경을 사용하세요.
RTX 5090은 Blackwell 지원 빌드가 필요합니다. 알려진 설치 조합 예시:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m alphase prepare --archives /path/to/zips --data data
python -m alphase inspect --data data --official-counts --output runs/data_report.json
python -m alphase train --data data --config configs/smoke.json --output runs/smoke --steps 1
```

smoke.json은 작은 CPU 기능 검사 설정입니다. 실제 GPU smoke test를 할 때 config의
device를 `cuda` 또는 `auto`로 변경합니다. Python 3.10 이상. GPU 드라이버와 빌드 호환은
실제 GPU에서 확인해야 합니다. 이미 호환 PyTorch가 있으면 첫 설치 명령은 생략합니다.
`python -m pip freeze > environment-lock.txt`로 실제 GPU 환경을 고정하세요.

## 데이터 계약

```text
data/train/images/*_image.png
data/train/masks/*_mask.png
data/valid/images/*_image.png
data/valid/masks/*_mask.png
data/test/images/*_image.png
```

이전 첨부 데이터 검사: train 70쌍, valid 20쌍, test 10장. 이미지/마스크는 opaque RGBA,
해상도는 다양하고 각 이미지·마스크의 크기는 일치했습니다. 학습은 원본에서 crop,
검증·추론은 타일 확률을 합성해 원본 크기로 복원합니다. 검증 이미지를 학습하지 않습니다.
Unknown 색상·비투명 alpha·누락·크기 불일치·세트 간 정확 중복은 검사에서 거부합니다.
A번호가 공식 train/valid 양쪽에 있어도 공식 분할을 유지하고 보고합니다.

| ID | Phase | RGB |
|---|---|---|
|0|Primary Si|76,178,76|
|1|Al3Ni|25,76,153|
|2|Eutectic Si|204,204,204|
|3|Al|153,127,76|

## 학습·재개

```bash
python -m alphase train --data data --config configs/baseline.json --output runs/e0
python -m alphase train --data data --config configs/core.json --output runs/core
python -m alphase train --data data --config configs/core.json --output runs/core --resume runs/core/last.pt
```

Core = EfficientNet-B3/U-Net++; baseline = ResNet34/U-Net/CE; Edge = Core + rare sampler +
경계 손실. soft Dice/CE, Focal+Dice, CE+Tversky와 class_weights를 config로 설정합니다.
Edge 기본 boundary_weight=0.05; 향상은 미검증입니다. 최소 손실·증강·optimizer 설정을
JSON에 저장하며 best/last 체크포인트, inference-only 가중치, config, 환경, CSV 로그를 생성합니다.
CSV에 실제 클래스별 학습 노출 픽셀 수도 기록합니다. 학습 중 검증은 mIoU 중심으로 수행하고,
경계/소형 성분 상세 분석은 별도 evaluate에서 수행해 매 epoch 부담을 줄입니다.

pretrained는 기본 false입니다. 외부 사전학습 허용 확인 후 true로 바꾸세요.
num_workers=0으로 patch RNG 재개를 지원합니다. config·데이터 fingerprint가 달라지면
재개를 거부합니다. GPU 간 bitwise 동일성을 보장하지 않습니다.

## GPU 메모리 설정 탐색

```bash
python -m alphase tune --config configs/core.json --patches 256 512 --batches 2 4 8 --output configs/core_gpu.json
python -m alphase train --data data --config configs/core_gpu.json --output runs/core
# 단일 실험에서만 편리한 자동 탐색
python -m alphase train --data data --config configs/core.json --output runs/core_auto --auto-tune
```

AdamW 상태·forward/backward를 포함한 두 step GPU probe를 실행하고 메모리 여유를 남긴
후 patch/batch를 고릅니다. 실데이터 OOM 방지를 보장하지는 않습니다. CPU에서는 명시적으로
skip하고 config를 유지합니다. **공정한 ablation에서는 동일 patch/batch를 모든 설정에 사용**하세요.
모델마다 auto-tune을 따로 적용하면 구조와 학습 조건 효과가 섞입니다. 재개 중 auto-tune은 금지됩니다.

## 실험 자동화 — A–G와 H 추론 비교

```bash
python -m alphase experiment-plan --config configs/core.json --output plans/main --ids A B C D E F G DL --seeds 42 43
python -m alphase experiments --data data --plan plans/main/plan.json --output runs/ablation
python -m alphase compare --plan plans/main/plan.json --runs runs/ablation --output runs/comparison
```

| ID | 변경 |
|---|---|
|A|ResNet34/U-Net/CE|
|B|EfficientNet-B3/U-Net/CE|
|C|ResNet34/U-Net++/CE|
|D|EfficientNet-B3/U-Net++/CE|
|E|D + 0.5CE+0.5Dice|
|F|v0.3: E + phase/boundary sampler|
|G|v0.3: F + classwise boundary_weight=0.05|
|CN|v0.3: ConvNeXt-Tiny/U-Net + phase/boundary sampler|
|DL|DeepLabV3+/ResNet50|
|Focal / Tversky|목적함수 대안|
|Random / Pretrained|동일 Core 구조의 초기화 대조|

희귀 클래스는 configs의 rare_class로 지정하며 train 분포만 보고 설정합니다. 기본은 Primary Si
ID 0입니다. 가장 적은 픽셀 클래스와 반드시 같지 않으므로 EDA 후 수정하세요.
Pretrained 실험은 규정 확인 후 `--allow-pretrained`로만 plan에 추가합니다.
동일 base config·seed·최대 epoch budget을 유지합니다. Early stopping 실제 epoch는 달라질 수 있습니다.
실험 runner는 학습·상세 평가·확률 캐시를 실행합니다. v0.3에서는 run_tta와 run_ensemble을
기본 false로 두고 screen 후 선택된 후보에만 TTA/앙상블을 명시적으로 실행합니다.

comparison CSV/JSON/Markdown은 이미지별 paired bootstrap, 클래스별 IoU, seed 평균/편차,
encoder B−A, decoder C−A, interaction (D−C)−(B−A)를 기록합니다. 빈 클래스 정책과 GT hash가
다른 결과는 비교하지 않습니다. 예정된 모든 seed가 측정된 후보만 추천합니다.
F/G/TTA는 parent보다 mIoU가 최소 0.005 높고 어떤 클래스도 0.02 이상 악화되지 않을 때 추천 후보가 됩니다.
이 규칙은 의사결정 보조용 가설이며 자동 성능 보증이 아닙니다. 검증 선택 편향을 고려하세요.

## 상세 평가·시각화

```bash
python -m alphase evaluate --data data --checkpoint runs/core/best.pt --output runs/core/evaluation --cache
python -m alphase evaluate --data data --checkpoint runs/core/best.pt --output runs/core/tta --cache --tta
```

metrics.json, RGB 예측, PNG 비교 패널, report.html, 선택적 float32 확률 npz를 생성합니다.
지표는 이미지별 mIoU 평균이며, 클래스별 FP/FN·정밀도·재현율·면적, Boundary F1, 작은 GT 조직
재현율을 제공합니다. Boundary tolerance는 기본 유클리드 거리 2픽셀, 8-neighbor inner boundary이며
이미지 외곽은 제외합니다. 작은 성분은 8-connected, 기본 ≤64픽셀입니다. **작은 성분 재현율을
instance IoU로 부르지 않습니다.** confidence는 보정되지 않은 softmax입니다.

공식 absent-class 처리는 미확정입니다. exclude(임시), zero, one을 `--absent-policy`로 선택하고,
각 이미지의 세 정책 점수를 모두 기록합니다. absent는 GT·예측 모두 없는 union=0 클래스만 해당합니다.

## 앙상블 평가·가중치 탐색

```bash
python -m alphase evaluate --data data --checkpoint runs/core/best.pt runs/e0/best.pt --weights 0.75 0.25 --output runs/ensemble --tta
python -m alphase ensemble-search --caches runs/core/evaluation runs/e0/evaluation --denominator 4 --output runs/ensemble_search.json
```

CPU에 모델을 보관하고 GPU에는 한 모델씩 올립니다. CPU RAM에는 모든 가중치와 캐시를 위한 공간이
필요합니다. search는 동일 validation 이미지·GT·정책·TTA 조건을 확인하고 simplex grid에서 가중치를
선택합니다. 기본 0.25 간격; 2000 후보 제한. 검색한 검증 점수는 선택 편향이 있어 독립 검증이 아닙니다.
선택된 best.weights와 cache metrics.json의 checkpoint 경로로 infer를 실행하세요.

## 실험적 후처리

```bash
python -m alphase postprocess --cache runs/core/evaluation --min-area 8 --confidence 0.6 --output runs/postprocess.json
```

작고 낮은 확신도의 고립 성분을 주변 클래스 중 확률이 높은 상으로 재배정하는 **검증용 실험**입니다.
기본 추론에는 적용하지 않습니다. 실제 작은 조직을 제거할 수 있으므로 효과 확인 전 제출에 사용하지 마세요.

## 테스트 출력·검사

```bash
python -m alphase infer --data data --checkpoint runs/core/inference.pt --output predictions/core --contract configs/submission_contract.json
python -m alphase validate-output --data data --predictions predictions/core --contract configs/submission_contract.json --output runs/output_check.json
python -m alphase repeat-check --data data --checkpoint runs/core/inference.pt --output predictions/core --contract configs/submission_contract.json
```

동일 checkpoint/weights/TTA/contract로 repeat-check하세요. 파일 개수·이름·크기·RGB/RGBA·색상·alpha·해시를
확인합니다. 기본 expected_count=10, RGB, *_mask.png는 **내부 임시 계약**입니다. 공식 형식 확인 후
contract를 수정합니다. filename_map으로 공식 파일명을 각각 지정할 수 있습니다.

## 제출 패키지 / 재실행

```bash
python -m alphase package --data data --predictions predictions/core --checkpoint runs/core/inference.pt --report final_report.pdf --receipt YOUR_RECEIPT --output submissions/YOUR_RECEIPT.zip --contract configs/submission_contract.json
python -m alphase verify-package --data data --archive submissions/YOUR_RECEIPT.zip --python /path/to/clean-venv/bin/python --device cuda --output runs/package_check.json
```

package는 실제 최종 PDF 1~10쪽, 10개 출력, 확정 계약을 검사합니다. contract의 output_format_confirmed,
absent_policy_confirmed가 true여야 하며 pretrained 사용 시 pretrained_allowed_confirmed도 true여야 합니다.
**확인하지 않은 내용을 true로 바꾸지 마세요.** PDF 보고서, 소스·환경 YAML·추론 실행 스크립트·가중치,
test_predictions.zip, SHA256 manifest를 receipt.zip으로 묶습니다. optimizer·train/valid GT·원본 데이터는
제외합니다. 중첩 ZIP·실행 파일 요구·체크포인트 포함 허용은 운영 자료와 최종 대조해야 합니다.
verify-package는 압축 해제·해시 검증·소스에서 추론 재실행·PNG hash 일치를 확인합니다.
깨끗한 환경 설치는 사용자가 제공하는 별도 interpreter에서 실행하세요. 업로드/제출은 하지 않습니다.

## 보고서 초안

```bash
python -m alphase report-draft --data-report runs/data_report.json --evaluation runs/core/evaluation/metrics.json --comparison runs/comparison/comparison.json --output runs/modeling_report_draft.md
```

실측 수치를 반영한 10개 항목의 편집용 Markdown 초안입니다. 최종 10쪽 PDF를 자동 보장하지 않습니다.
템플릿·그림·선택된 모델 설정을 검토하고 최종 보고서를 별도로 편집합니다.

## 파일 / 검증 범위

`ALPHASE_Colab.ipynb`에 업로드·GPU 점검·Drive 저장·학습·평가·출력 명령을 포함했습니다.
`IMPLEMENTATION_STATUS.md`는 설계 대비 구현 상태, `VALIDATION.md`는 실제 검증과 미검증 범위입니다.
checkpoint는 직접 생성한 신뢰할 수 있는 파일만 로드합니다(torch pickle 사용).

근거: https://komap.ai/front/competition2026/process1 · https://smp.readthedocs.io/en/latest/models.html ·
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html · https://pytorch.org/get-started/previous-versions/

## 측정 기반 예산 추정

```bash
python -m alphase budget --run runs/core --hourly-rate 0.432 --output runs/budget.json
```

현재 서비스 가격을 hourly-rate로 입력합니다. 측정한 epoch 시간을 외삽하며 저장·전송·다른 실험 비용은 포함하지 않습니다.
