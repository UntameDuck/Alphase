# 다음 단계 실행 지침 — ALPHASE v0.4

작성 기준: 2026-10-11 KST. 이 문서는 **앞으로 수행할 작업**이다. 명령이 적혀 있다고 실행·승인·성공한 것으로 간주하지 않는다. 이번 문서 작성에서는 모델 코드·학습 설정을 변경하거나 GPU를 사용하지 않았다.

설계 판단은 [v4 설계](UPGRADE_V4.md), 실제 완료 증거는 [검증 기록](VERIFICATION.md), 단계별 상태는 [실행 체크리스트](EXECUTION_CHECKLIST.md)를 따른다. 실행할 때는 [기록 양식](EXECUTION_RECORD_TEMPLATE.md)을 복사해 채운다. 다른 실행 예시와 충돌하면 이 문서의 승인 경계·명령 경로·순서를 우선한다.

## 0. 현재 위치와 절대 경계

- 완료: v4 구현, CPU 테스트 53개, 같은 환경 패키지 재추론, 원본 PNG 190개 무결성, train-only 형태 분석.
- 미완료: 공식 미확정 규정, GPU 지출 승인, 실제 CUDA 512/AMP 검증, 대회 본학습, 실측 validation 성능, 독립 clean-GPU 재현, 제출.
- $300은 사용자가 알려 준 보유 크레딧이다. 재확인된 잔액·승인된 지출 한도·예상 총비용이 아니다.
- 원본 `../data`, `../train.zip`, `../valid.zip`, `../test.zip`, 이전 `../ALPHASE_code.zip`을 수정·교체하지 않는다.
- 공식 train70/valid20/test10 분할 유지. test 정답 수집·test 기반 tuning·pseudo-label·90장 재학습은 이 절차에 없다.
- 초기 후보는 A/E/F/CN, 조건부 O/L/S/G다. 새로운 구조나 조합의 무제한 탐색을 시작하지 않는다.
- 임대·다운로드 비용·학습·외부 업로드·인스턴스 제거는 해당 단계의 승인/확인이 있어야 한다. 규정 문의 메일 발송도 사용자의 별도 승인이 필요하다.

**지금의 첫 행동:** G0의 로컬 상태를 확인하고 G1의 규정 확인표를 채운다. 아직 `train`, `tune`, `benchmark`, `experiments`, 임대 버튼을 실행하지 않는다.

## 공통 실행 규칙

모든 CLI 예시는 `ALPHASE` 폴더가 작업 디렉터리이고 데이터가 `../data`에 있는 경우다. 로컬은 `E:\Competition\코맵\ALPHASE`, 원격은 사용자가 승인한 업로드 위치에서 같은 상대 구조를 만든다. 원격 경로를 로컬 경로라고 오인하지 않는다.

- Windows에서는 `.\activate.ps1`로 로컬 환경을 활성화한다. 이 스크립트는 현재 쉘의 환경 변수를 바꾸므로 해당 프로젝트용 터미널에서 사용한다.
- 원격 Linux에서는 새 GPU 전용 venv를 활성화한다. 로컬 Windows `.venv`를 복사해 쓰지 않는다.
- 아래 `text` 코드 블록은 값·파일명 설명이며 쉘 명령이 아니다. JSON 블록은 명시적으로 '변경 필드'라고 적힌 경우 전체 설정이 아니다.
- `configs/runtime/`, `plans/`, `runs/competition_v4/`, `runs/gpu_smoke_v4/`, `artifacts/`는 **향후 생성할 경로**다. 현재 존재한다고 가정하지 않는다. 사용자가 정한 실행 ID로 폴더를 만들고 오래된 결과를 덮어쓰지 않는다.
- 새 plan/evaluation/prediction 폴더는 비어 있어야 한다. 이미 존재하면 상태를 확인하고 문서의 복구 절차를 따른다. 자동 삭제하지 않는다.
- 명령 하나가 실패하면 다음 명령으로 넘어가지 않는다. exit code, traceback, 마지막 완료 파일을 기록한다.
- 본학습 시작 후 `alphase/*.py`, config, 데이터, steps를 고정한다. v4 재개는 source/data/config/steps가 같아야 한다. 코드 수정을 했다면 기존 환경을 보존하고 새로운 실험 계열로 분리한다.

## G0. 로컬 기준선 고정 — 비용 없음

### 행동

1. [검증 기록](VERIFICATION.md)의 날짜·코드 버전·남은 제약을 읽는다.
2. 새 실행 기록 문서를 만들고 현재 시각, 담당자, 다음 단계, 코드 버전, manifest 위치를 적는다. 템플릿의 '미확인'을 실제 확인 없이 '완료'로 바꾸지 않는다.
3. `SOURCE_SHA256.json`을 대조한다. 차이가 있으면 정상 변경인지 먼저 조사하고 승인 없이 원본 파일을 복원하거나 manifest만 갱신해 숨기지 않는다.
4. 코드/의존성 변경이 없다면 기존 53개 PASS 기록을 근거로 삼을 수 있다. 변경 또는 새 환경이 있으면 아래 CPU 회귀 검사를 실행한다. 합성 fixture의 optimizer 실행은 실제 대회 데이터 학습과 다르다.

```powershell
Set-Location 'E:\Competition\코맵\ALPHASE'
.\activate.ps1
$env:PYTHONDONTWRITEBYTECODE='1'
$env:HF_HUB_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:ALPHASE_VERIFY_PYTHON=(Resolve-Path .venv/Scripts/python.exe).Path
python -m unittest discover -s tests -v
```

이 offline 설정은 이 터미널에 남는다. 나중에 허용된 pretrained 다운로드를 해야 하면 별도 다운로드 환경에서만 명시적으로 해제하고 파일·출처·해시를 기록한다. 규정 확인 없이 해제해 학습을 시작하지 않는다.

### 통과 기준 / 산출물

원본 보존, 변경 파일 설명 가능, 필요한 회귀 검사 PASS. 실행 기록에 테스트 명령·결과와 source manifest SHA를 첨부한다. 실패하면 [R1](RECOVERY_GUIDE.md#r1-로컬-테스트설정-실패)을 따른다.

## G1. 규정 4종 확정 — 유료 자원 사용 전

[데이터·제출 계약](DATA_AND_EVALUATION.md)의 미확정 항목을 [기록 양식](EXECUTION_RECORD_TEMPLATE.md)에 채운다. 공개 설명만으로 부족하면 참가자 자료·공식 서식·운영자 답변이 필요하다. 공식 사이트 링크가 있다는 사실만으로 허용을 추정하지 않는다.

| 확인 대상 | 확인할 질문 | 반영 위치 | 미확정이면 |
|---|---|---|---|
| 외부 pretrained | ImageNet 가중치, 출처·라이선스, 외부 데이터, ensemble 허용 범위 | runtime base의 `pretrained`, `pretrained_permission`; 제출 계약 | false 유지. 랜덤 초기화 실험도 별도 승인한 범위에서만 |
| mIoU absent 처리 | GT와 예측 모두 없는 상의 union=0을 exclude/zero/one 중 어떻게 처리하는가? | runtime base의 `absent_policy` 및 제출 계약 | 최종 비교·성능 확정 보류. 진단 실행은 임시 정책임을 표시하고 별도 승인 |
| 출력 계약 | PNG/RGB/RGBA, 파일명·매핑, 원본 해상도, ZIP 구조 | runtime 제출 계약의 mode/suffix/map/count | 임시 출력은 내부 시험용이며 최종 package 금지 |
| 실행·제출물 | Python 스크립트가 실행 파일로 인정되는가? checkpoint·용량·중첩 ZIP·보고서 요구는? | 패키지 구성·실행 안내·보고서 | package가 성공해도 제출 형식 승인으로 간주하지 않음 |

### 행동

1. 항목마다 출처 URL/첨부파일, 조회 시각(KST), 근거 문구·결론·확인자를 기록한다. 계정 비밀번호·토큰은 문서에 쓰지 않는다.
2. 허용이 확인된 경우에만 runtime 설정을 별도로 만든다. 저장된 기본 `configs/champion.json`과 `configs/submission_contract.json`은 보존한다.
3. 외부 pretrained를 사용하기로 승인했다면 `pretrained=true`와 `pretrained_permission=true`를 함께 적용한다. 공식 허용과 우리 실험에서 실제 사용 여부는 다른 항목이다.
4. 제출 계약의 confirmation flag는 검증을 통과하려고 임의로 켜지 않는다. 특히 `absent_policy_confirmed=true`만으로 checkpoint의 실제 정책이 바뀌지 않는다. checkpoint config 및 평가 metadata도 대조한다.
5. 공식 근거가 기존 이해와 다르면 설정·코드를 수정하고 회귀 검사를 다시 한 후 비용 계획으로 넘어간다.

### 통과 기준

채택할 pretrained/metric 정책이 명시되고, 최종 제출 제약 및 남은 불확실성이 기록돼 있다. 미확정 상태에서 진행할 예외가 필요하면 사용자가 진단 범위만 따로 승인해야 한다. 본 지침은 예외 승인이 아니다.

## G2. 유료 진단 범위 승인 — 임대 전에 중단하여 확인

### 행동

1. 당일 GPU 매물의 GPU/VRAM, 시간당 compute, 디스크 용량·요금 단위, 송수신 비용, 최소 과금 조건, 호스트 가용성·신뢰 지표를 읽기 전용으로 확인한다.
2. 결제 화면에서 credit 잔액과 적용 조건을 확인한다. 자동 충전·추가 결제를 켜지 않는다.
3. 먼저 **설치·probe·benchmark·짧은 실제-data smoke만** 포함하는 진단 단계의 달러/시간 상한을 제안한다. 본학습은 G4의 실측 후 따로 승인받는다.
4. 사용자가 승인한 금액·시간·대상·중단 조건을 기록한다. 보유 $300 전액을 한도로 자동 간주하지 않는다.
5. 긴 실행의 감시 담당과 청구 확인 주기를 정한다. CLI에 자동 비용 상한·자동 인스턴스 종료 기능은 없다. 호스트 기능이 있다면 실제 지원·설정을 확인해야 한다.

### 통과 기준 / 산출물

진단 단계의 명시적 지출 승인과 상한, 청구 시작/중단 방식, 백업·종료 계획이 모두 있어야 한다. 그 전에는 인스턴스를 만들지 않는다.

## G3. 승인된 GPU 환경 구축·무결성 검사

### 행동

1. 승인된 GPU/가격/디스크 조건을 마지막으로 대조하고 인스턴스를 만든다. 생성 시각과 실제 청구 시작 시각을 기록한다.
2. 원격 폴더 구조를 `.../ALPHASE`와 `.../data`로 맞춘다. 전송은 승인된 호스트로만 한다. 소스·설정·문서·필수 데이터만 전송하고 `.venv`, `.downloads`, 캐시, 개인 자격증명, 불필요한 archive는 제외한다.
3. CUDA driver와 호환되는 torch/torchvision을 **먼저** 설치한다. 설치 시점의 공식 PyTorch 안내와 GPU image 버전을 확인하고 정확한 설치 명령·버전을 기록한다. 이 문서는 확인되지 않은 CUDA wheel 주소를 고정하지 않는다.
4. GPU venv에서 나머지 의존성과 프로젝트를 설치한다. 설치 전후 torch/torchvision/CUDA 버전이 의도치 않게 바뀌지 않았는지 확인한다.

```bash
nvidia-smi
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m alphase --help
python -m alphase inspect --data ../data --official-counts --output runs/preflight_v4/data_report.json
```

5. 소스는 manifest와 대조하고 원격 데이터 보고서를 로컬 `runs/data_report.json`과 비교한다. train70/valid20/test10, palette, shape, 입력/GT 해시가 같아야 한다. GPU dependency lock은 새로 기록하며 Windows CPU lock을 GPU 설치 명세로 쓰지 않는다.
6. G0의 CPU 회귀 검사를 원격 새 환경에서도 수행한다. `ALPHASE_VERIFY_PYTHON`은 Windows 경로가 아닌 **원격 venv Python의 절대 경로**로 설정한다. GPU 설치 후 CPU 검사 PASS는 실제 CUDA PASS가 아니다.

### 통과 기준

CUDA 실제 장치 확인, 설치 오류 없음, 소스/데이터 일치, 테스트 PASS, 승인 잔여 비용 확인. 실패 시 추가 유료 실험을 시작하지 말고 [R2](RECOVERY_GUIDE.md#r2-cuda설치-실패)를 따른다.

## G4. CUDA smoke·측정 → 본학습 예산 확정

### G4-A. 공통 자원 조건 선택

우선 아래 probe는 **합성 텐서의 optimizer step을 실행**한다. GPU 비용이 들고 학습 메모리만 검사한다. 평가·캐시·패키지까지 안전하다는 뜻은 아니다.

```bash
python -m alphase tune --config configs/champion.json --patches 512 --batches 4 2 --output configs/runtime/core_probe.json
python -m alphase tune --config configs/challenger.json --patches 512 --batches 4 2 --output configs/runtime/cn_probe.json
```

- `.probe.json`의 selected/status/peak/margin을 확인한다. 현재 기본 두 파일은 pretrained=false이며 위 probe가 pretrained 실험 허가를 의미하지 않는다.
- probe가 모델마다 고른 설정을 그대로 공정 비교에 사용하지 않는다. 모든 A/E/F/CN에 맞는 patch·batch·inference batch를 정하고 공통 base로 기록한다.
- `tune`은 accumulation을 자동 조정하지 않는다. batch4→2이면 effective batch8을 유지하도록 accumulation2→4를 **runtime base에서** 조정한다. 단, BatchNorm의 microbatch 동작까지 동일해지는 것은 아니다.
- 512가 안 되면 256/64 후보를 별도 비교 계열로 승인한다. 256 결과와 512 결과를 순수 모델 구조 차이로 비교하지 않는다.
- `configs/champion.json`을 토대로 G1 정책과 공통 자원 설정을 적용한 **완전한 설정 파일** `configs/runtime/base_v4.json`을 편집해 저장한다. 기존 설정을 덮어쓰지 않는다. 이 파일은 현재 자동 생성돼 있지 않다.

### G4-B. 별도 2-epoch 실제-data smoke

`--steps 2`만 붙이면 80 epoch 설정은 그대로다. smoke base는 runtime base를 복사해 아래 필드만 변경한 완전한 파일 `configs/runtime/smoke_v4.json`으로 만든다. 비용 승인은 실제 데이터의 짧은 optimizer 실행까지 포함해야 한다.

```json
{
  "epochs": 2,
  "warmup_epochs": 0,
  "min_epochs": 1,
  "patience": 2
}
```

아래 계획 생성은 학습하지 않는다. 학습 명령은 G2 승인 이후에만 실행한다. pretrained=true일 때만 G1 근거를 확인하고 `experiment-plan`에 `--allow-pretrained`를 추가한다. 이 규칙은 뒤의 모든 계획 생성 명령에 동일하게 적용한다.

```bash
python -m alphase experiment-plan --config configs/runtime/smoke_v4.json --ids A E F CN --seeds 42 --output plans/gpu_smoke_v4
python -m alphase experiments --data ../data --plan plans/gpu_smoke_v4/plan.json --output runs/gpu_smoke_v4 --steps 2
```

- 이 실행도 매 epoch 전체 validation과 마지막 상세 evaluation/cache를 수행한다. 단순 2-batch만큼의 시간으로 예상하지 않는다.
- 각 run의 finite loss, optimizer_steps>0, skipped_amp_steps, 원본 크기 추론, VRAM, last/best/inference/history/status, 보고서·cache를 확인한다. smoke의 낮은 mIoU 자체를 모델 탈락 사유로 쓰지 않는다.
- 동일 명령을 다시 실행해 완료 run이 추가 학습 없이 종료하고 checkpoint·evaluation SHA가 유지되는지 확인한다. 저장·재개 확인을 위해 source/config/steps를 바꾸지 않는다.
- 비교 결과는 `no_eligible_candidate`여야 한다. smoke 점수는 본학습 순위나 보고서의 대회 성능으로 쓰지 않는다.

### G4-C. 본학습 설정의 시간 측정

```bash
python -m alphase experiment-plan --config configs/runtime/base_v4.json --ids A E F CN --seeds 42 --output plans/screen_v4
python -m alphase benchmark --data ../data --config plans/screen_v4/E_s42.json --warmup 2 --batches 10 --output runs/preflight_v4/benchmark_E.json
python -m alphase benchmark --data ../data --config plans/screen_v4/CN_s42.json --warmup 2 --batches 10 --output runs/preflight_v4/benchmark_CN.json
```

필요하면 A/F 및 신규 O/L/S도 같은 방식으로 측정한다. benchmark는 **실제 train crop에서 랜덤 초기화 모델을 임시 optimizer update**하지만 checkpoint나 성능 점수를 저장하지 않는다. pretrained 수렴·정확도 시험이 아니다. 매 microbatch optimizer update라 실제 accumulation 시간과 차이가 있고, sampler·metric·checkpoint I/O·설치·전송 비용을 모두 포함하지 않는다. 전체 wall-clock과 초기 본학습 epoch로 다시 추정한다.

### 비용·시간 계산 및 통과 기준

```text
T_i = epochs_i × (epoch_train_seconds_i + validation_seconds_i + 측정된 추가 overhead_i)
T_total = Σ T_i + 설치 + 상세 평가 + TTA/ensemble + package/repeat + clean 재실행 + 백업
C_total = compute 요금(T_total) + 디스크 요금 + 송수신 + 기타 확인된 청구
예산안 = C_total에 최소 25% 운영 여유 반영 (미측정 항목은 별도 표시)
```

early stopping을 기대해 상한을 줄이지 않는다. 3seed 부모 대조군 비용, 신규 후보 probe, 최종 검증 시간도 포함한다. GPU의 시간 한도가 마감 전에 들어오는지 비용과 별도로 확인한다.

**G5로 넘어가기 전에** 실측 표·최대 예상 비용·단계별 한도를 사용자에게 제시하고 본학습 승인을 받는다. 예산·시간이 넘으면 seed 대조군을 몰래 생략하지 말고 후보 수를 줄인다. CPU 테스트 시간으로 GPU 비용을 추정하지 않는다.

## G5. 첫 실제 기준 성능 — A → E → F → CN

G4의 `plans/screen_v4`를 검토하고 같은 source/runtime base를 고정한다. 이 단계는 `--steps` 없이 실행한다. smoke checkpoint를 재개하거나 GPU smoke 폴더에 쓰지 않는다.

```bash
python -m alphase experiments --data ../data --plan plans/screen_v4/plan.json --output runs/competition_v4
```

runner는 계획 순서대로 학습하고 완료 평가·비교를 수행한다. `run_tta=false`, `run_ensemble=false`를 확인한다. 모든 후보가 끝나기 전에도 담당자가 비용·발산·학습 진행을 점검한다. 자동 임대 종료나 비용 차단 기능은 없다.

### 각 모델에서 확인할 것

1. `config.json`, `environment.json`이 동결 설정과 맞는가? pretrained 허가·실제 사용, absent policy, patch/batch/accum, seed를 대조한다.
2. `history.csv`의 loss/val score가 유한한가? optimizer step이 실행되는가? AMP skip이 반복되거나 메모리가 계속 늘어나는가?
3. `run_status.json.completed=true`이고 stop_reason이 early_stopping 또는 max_epochs인가? `steps=null`인가? 미완료 run을 완료로 수동 편집하지 않는다.
4. `evaluation/metrics.json`의 source/checkpoint/config/data 출처가 맞는가? `report.html`에서 클래스 붕괴, 경계·작은 Si·worst case를 확인한다.
5. 실제 wall-clock·청구액·디스크 사용량을 기록하고 G4 예산을 갱신한다. 다음 명령은 **한 run의 남은 계산비만** 추정하며 I/O·저장·후속 실험을 포함하지 않는다. 시간당 가격은 직접 확인한 숫자로 치환한다.

```text
python -m alphase budget --run runs/competition_v4/E_s42 --hourly-rate ACTUAL_USD_PER_HOUR --output runs/preflight_v4/budget_E.json
```

### 통과 기준

`runs/competition_v4/comparison/comparison.json`에 실제 scores·missing·eligible_variants·exclusions·selection_gates가 있다. 한 seed는 screen_only다. suggested_variant를 무조건 최종 모델로 승인하지 않는다. F가 E를 통과하지 못하면 G/O/L/S를 자동 추가하지 않는다. 기준 결과와 실패 원인을 기록한 뒤 G6 여부를 결정한다.

## G6. 조건부 개선 실험 — 효과와 시간에 근거해 실행

F가 E 대비 gate를 통과하고 비용·시간이 남은 경우에만 진행한다. 먼저 O(component)와 L(Lovasz)를 비교하고 S(scSE)는 추가 여유, G(boundary)는 경계 오류 근거가 있을 때만 추가한다. 총 단일 seed 신규 recipe 상한은 A/E/F/CN/O/L/S/G 8개다.

```bash
python -m alphase experiment-plan --config configs/runtime/base_v4.json --ids E F O L --seeds 42 --output plans/extensions_v4
```

신규 O/L 설정으로 G4의 probe·benchmark 및 **별도 smoke plan**을 먼저 통과한다. Lovasz 정렬 메모리와 component sampler의 CPU RAM/준비 시간은 Core probe만으로 보증되지 않는다. 새 후보 때문에 batch/patch를 바꾸면 기존 결과와 같은 비교가 아니므로 공통 조건의 새 계열을 설계한다.

```bash
python -m alphase experiments --data ../data --plan plans/extensions_v4/plan.json --output runs/competition_v4
```

같은 E/F는 source/config/steps·입력/평가 출처가 일치하면 완료 run을 재사용한다. candidate JSON을 base로 줘도 canonical plan이 해당 변경을 초기화하므로 반드시 O/L/S ID로 지정한다.

### 채택 규칙

- 부모·자식 모두 계획 seed를 완료하고 budget/metric/source/GT가 같아야 한다.
- 부모 E→F, F→O/L/S/G의 **단일 변경** 비교. seed 평균 mIoU +0.005 이상, 각 seed/상 하락 0.02 이내, 과반수 seed에서 양수여야 한다.
- small-component recall뿐 아니라 FP·상별 IoU·전체 mIoU를 함께 본다. 한 보조 지표가 좋다고 채택하지 않는다.
- 한 seed 통과는 반복할 후보 선정일 뿐이다. 조합 후보는 최대 2개 변경, 별도 사전 기록·부모 대조·실행 승인 필요. 현재 자동 조합 탐색 기능은 없다.

## G7. 반복 seed — 부모 대조군을 함께 확보

상위 최대 2 recipe를 고르고 **그 주장에 필요한 부모 대조군**을 포함한다. 예: O와 CN이 유망하면 E/F/O/CN을 seeds42/123/2026으로 반복한다. 아래는 이 상황의 예시이며 실제 승자가 다르면 ID를 바꾼다.

```bash
python -m alphase experiment-plan --config configs/runtime/base_v4.json --ids E F O CN --seeds 42 123 2026 --output plans/replicate_v4
python -m alphase experiments --data ../data --plan plans/replicate_v4/plan.json --output runs/competition_v4
python -m alphase compare --plan plans/replicate_v4/plan.json --runs runs/competition_v4 --output runs/review_v4/replication
```

E/CN만 비교한다면 E/CN 3seed, F/CN을 비교한다면 E/F/CN 3seed가 필요하다. seed42 결과 재사용에는 완전 동일 설정이 필요하다. 이미 생성한 plan을 수정하지 말고 다른 이름의 새 plan으로 후보 변경을 기록한다.

### 통과 기준

missing 없음, 각 후보의 비교 예산·recipe 일치, 부모 대조군 적격, 3seed 분산·과반수 개선·상별 하락 검사. 최종 기록에는 평균·편차·개별 seed 점수를 모두 적는다. validation20장의 반복 사용 편향과 시편 상관은 남는다. `replicated_candidate`는 독립 검증 성공·수상권 성능이라는 뜻이 아니다.

## G8. 추론 전략 고정 — none / flip / D4 / 제한 ensemble

1. 채택한 recipe·seed·checkpoint SHA를 먼저 고정하고 none 평가를 보존한다.
2. 추가 TTA 계산 시간을 확인한다. flip은 코드상 4변환, D4는 8변환이며 실제 전체 소요 시간이 정확히 그 배수라고 단정하지 않는다.
3. 아래 F는 **경로 예시**다. 실제 적격 모델로 바꾼다. `evaluation_tta`가 없는 경우에만 선택한 한 모드를 기록한다. 동일 checkpoint의 `best.pt`를 사용해 기존 single 평가와 SHA를 맞춘다. `inference.pt`는 텐서가 같아도 파일 SHA가 다를 수 있다.

```bash
python -m alphase evaluate --data ../data --checkpoint runs/competition_v4/F_s42/best.pt --device cuda --tta-mode d4 --output runs/competition_v4/F_s42/evaluation_tta --cache
```

4. `compare`가 자동으로 읽는 TTA 폴더는 `evaluation_tta` 하나다. flip과 D4를 둘 다 시험한다면 다른 이름의 별도 평가 폴더에 보관하고 자동 비교에 둘 다 들어간다고 착각하지 않는다. 최종 TTA 채택은 모든 비교 seed에 같은 모드·동일 부모 checkpoint를 사용한 기록으로 판단한다.
5. ensemble은 적격 checkpoint 최대3개, 우선 균등 가중치로 제한한다. 새 evaluation 폴더에서 같은 입력/GT/policy/source·TTA로 단일 모델과 비교한다. 연속 weight 탐색과 postprocess는 기본 off다.
6. 최종 추론 명세에 checkpoint **순서**, 각 SHA, weights, tta-mode, config/정규화, source/dependency, contract SHA, validation 결과 경로를 고정한다. 명세를 바꾸면 test 생성부터 package 검증까지 다시 수행한다.

**주의:** ensemble/TTA 선택 역시 같은 validation20장의 선택 결과다. test 점수 추정치로 쓰지 않는다. 차이가 불확실하면 측정된 성능·실행 비용·재현성을 함께 보고 단순한 구성을 선택한다.

## G9. 최종 valid / test / 보고서 생성

G1 출력 계약이 확정된 후 `configs/runtime/submission_v4.json`을 만든다. 원본 임시 계약은 보존한다. 다음 예시는 F_s42 단일 / tta=none이며 실제 G8 명세와 다르면 **아래 전 명령을 함께** 수정한다. ensemble이면 같은 checkpoint 순서와 weights를 모두 전달한다.

```bash
python -m alphase evaluate --data ../data --checkpoint runs/competition_v4/F_s42/best.pt --device cuda --tta-mode none --output runs/final_v4/validation --cache
python -m alphase infer --data ../data --checkpoint runs/competition_v4/F_s42/best.pt --device cuda --tta-mode none --contract configs/runtime/submission_v4.json --output runs/final_v4/predictions
python -m alphase validate-output --data ../data --predictions runs/final_v4/predictions --contract configs/runtime/submission_v4.json --output runs/final_v4/output_validation.json
python -m alphase repeat-check --data ../data --checkpoint runs/competition_v4/F_s42/best.pt --device cuda --tta-mode none --contract configs/runtime/submission_v4.json --output runs/final_v4/predictions
python -m alphase report-draft --data-report runs/preflight_v4/data_report.json --evaluation runs/final_v4/validation/metrics.json --comparison runs/review_v4/replication/comparison.json --output artifacts/report_draft_v4.md
```

- `repeat-check --output`은 새 JSON 출력 위치가 아니라 **이미 존재하는 예측 폴더**다. 성공 결과는 stdout에 나오므로 실행 기록에 남긴다.
- 출력 10장, 원본 크기, RGB/RGBA·opaque alpha, palette, 파일명/충돌, manifest 입력·출력 SHA 확인. 예측 PNG를 그림 편집기로 수정하지 않는다.
- report-draft는 편집용 Markdown이며 최종 한국어 보고서/PDF가 아니다. 실측 표·도판·인용·오류 사례·예산·규정·재현 환경을 사람이 검토해 공식 서식과 페이지 제한에 맞춘다.
- 2D 면적 분율 MAE를 물성 예측 정확도로 부르지 않는다. 독창성 기여는 데이터 관측→가설→공정 ablation→효과·한계로 설명한다. 사용하지 않은 기법을 최종 모델 구성으로 쓰지 않는다.
- 보고서의 checkpoint/config/seed/metric/TTA/ensemble 설명이 G8 명세·metrics와 일치해야 한다. synthetic/smoke 점수와 공식 참고값을 실제 모델 성능으로 섞지 않는다.

### 통과 기준

최종 추론 명세, validation 근거, 확인된 contract, test 출력·반복 hash, 검토된 최종 PDF가 모두 일치한다. 여기까지는 제출한 것이 아니다.

## G10. 패키지·독립 clean-GPU 재현

아래 `ACTUAL_RECEIPT`는 **실제 접수번호로 치환해야 하는 자리표시자**다. 접수번호를 받기 전에는 package하지 않는다. PDF 경로도 실제 검토 완료 파일로 바꾼다. G9과 같은 checkpoint/weights/TTA/device/contract를 유지한다.

```bash
python -m alphase package --data ../data --predictions runs/final_v4/predictions --checkpoint runs/competition_v4/F_s42/best.pt --device cuda --tta-mode none --contract configs/runtime/submission_v4.json --report artifacts/modeling_report.pdf --receipt ACTUAL_RECEIPT --output artifacts/ACTUAL_RECEIPT.zip
```

package는 모델을 다시 추론하므로 시간·비용이 추가된다. manifest/hash·PNG·PDF·규정 flag를 검사하지만 '실행 파일 인정'이나 공식 업로드 성공을 확인하지 않는다. 생성 ZIP은 덮어쓰지 않는다.

별도 새 GPU 환경에 선언된 dependency를 설치하고 아래를 실행한다. 깨끗한 환경의 Python 경로를 `--python`으로 지정하거나 그 환경에서 직접 실행한다. 같은 venv의 별도 subprocess를 clean install이라고 부르지 않는다. 새로운 유료 환경이 필요하면 비용 승인이 먼저다.

```bash
python -m alphase verify-package --archive artifacts/ACTUAL_RECEIPT.zip --data ../data --device cuda --output artifacts/verify_clean_gpu.json
```

패키지 재실행의 정확한 PNG SHA 일치가 통과 조건이다. 다른 GPU/backend의 수치 차이로 실패하면 '거의 같다'고 무시하지 말고 [R7](RECOVERY_GUIDE.md#r7-추론패키지-해시-불일치)을 따른다. 기준 환경·추론 명세를 재검토하고 변경했다면 valid/test/package를 다시 만든다.

### 수동 QA

- 공식 접수번호와 ZIP 이름, 파일 용량·내부 구조·실행 파일 인정·환경 YAML이 요구에 맞는가?
- 보고서가 공식 페이지 계산 방식의 한도 내이며 수치·모델·근거가 모두 일치하는가?
- test 10장과 필요한 모델·소스가 있으며 train/valid GT·optimizer·비밀키·불필요한 데이터가 없는가?
- 제출 ZIP SHA, clean 검증 결과, 최종 보고서, checkpoint·source·dependency를 별도 보관했는가?

## G11. 제출·접수 확인·GPU 정리 — 별도 승인

1. 모델 동결은 10-14 KST, 내부 제출 목표는 10-15 18:00 KST로 잡는다. 실제 실행 시 공식 최신 마감·서버 시각을 재확인한다. 제출 직전에는 새 후보 학습을 시작하지 않는다.
2. 사용자에게 최종 ZIP/보고서/검증 결과/잔여 위험을 보여주고 **제출 승인**을 받는다. 학습 승인은 공식 제출 승인과 다르다.
3. 승인된 공식 페이지에만 업로드한다. 업로드/제출 완료 시각, 접수번호, 성공 화면·확인 문구, 제출 파일 SHA를 기록한다. 성공을 확인하지 못했다면 제출 완료로 표시하지 않는다.
4. GPU 종료 전에 source/config/environment/history/status/last/best/inference, evaluation, 최종 예측·ZIP·보고서·청구 기록을 승인된 저장소로 백업한다. 로컬에서 파일 존재·크기·SHA 및 checkpoint 로딩 가능 여부를 확인한다.
5. 인스턴스 ID·볼륨·백업을 확인하고 승인된 종료 방식으로 정리한다. stop/destroy 각각의 저장 비용·데이터 보존 동작은 당시 서비스 정책과 화면에서 확인한다. 유일한 checkpoint가 있는 볼륨을 제거하지 않는다.
6. 실제 총비용과 잔여 credit, 청구 종료 여부를 확인한다. 후속 발표·재현에 필요한 파일 위치를 기록한다.

## 단계 종료 때 반드시 남길 한 문단

```text
단계/시각/담당자:
한 일과 실제 명령:
통과 또는 실패 근거(파일·SHA·지표·로그):
지출/잔여 승인 한도/다음 실행 예상 시간:
아직 모르는 사항 및 실행하지 않은 일:
다음 행동 한 가지와 필요한 승인:
```

성능이 불충분하면 설계 수정이 다시 필요할 수 있다. 다만 실측 없이 후보를 계속 늘리거나 실패를 감추는 대신, [복구·중단 지침](RECOVERY_GUIDE.md)에 따라 확인된 단순 모델·명확한 비교 근거·정상 제출을 보존한다.
