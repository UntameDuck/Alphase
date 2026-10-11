# v0.4 검증 기록 — 2026-10-11 KST

현재 설계는 [v4 기준](UPGRADE_V4.md)이다. 아래 v3 검증은 당시 이력으로 보존한다. 테스트 성공과 실제 대회 성능을 구분한다.

## 후속 운영 문서 검사 — 2026-10-11 KST

- 단계별 행동 지침·체크리스트·실행 기록 양식·복구 지침 4개 추가. 현재 docs 폴더의 Markdown 문서는 31개다.
- 신규 문서의 CLI 예시 25개는 현재 argparse로 **파싱만** 검사했다. `--help` 예시도 확인했다. 실제 train/tune/benchmark/experiments/infer/package를 실행한 검사가 아니다.
- 2-epoch smoke 변경 필드를 champion 설정과 합쳐 설정 validation 통과 확인. 실제 runtime 설정 파일은 만들거나 수정하지 않았다.
- 로컬 문서 링크·절 앵커·코드 블록 닫힘 검사 통과. README·설계 목차·RUNBOOK·파일 지도에서 새 지침을 연결했다.
- 기존 source manifest와 대조해 코드20개·테스트7개·설정11개, 총38개 파일이 변경되지 않았음을 확인했다. 이 문서 작업을 이유로 기존 53개 학습 회귀 suite를 재실행하지는 않았다.
- GPU 임대·지출·본학습·외부 제출·Notion 변경 없음. 기존 Notion 동기화는 아래 v4 구현 검증 당시의 이력이다.

## 이번 버전에서 확인한 결과

- Python 3.12 / torch 2.7.1+cpu / torchvision 0.22.1+cpu / SMP 0.5.0 / timm 1.0.15, HF offline 환경에서 **53개 고유 테스트 PASS**. `ALPHASE_VERIFY_PYTHON`을 현재 venv로 지정한 전체 suite이며 119.101초가 걸렸다. 이는 테스트 시간이지 학습 비용 벤치마크가 아니다.
- 신규 Lovasz를 SMP reference와 값·gradient 대조. 완전 예측·ignore·이미지 동등 가중 검사. scSE + 손실의 CPU forward/backward/reload 검사. 성분 균등 선택·crop 정렬·train-only morphology 및 2D 면적 분율 MAE 손계산 검사.
- 합성 6 crop / batch2 / accumulation2 = 3 microbatch의 부분 누적 포함, epoch당 2회 optimizer update. 연속 실행과 중간 epoch 재개의 raw/EMA/optimizer/torch RNG/scheduler 일치 및 EMA 4회 update 확인.
- 조기 종료 재개 no-op, 새 폴더의 last/best/inference/history/status 복원, last commit 이후 best export 실패 복구, 파생 best 손상 복구, 무관한 목적지 보존, steps 변경 거부.
- stale evaluation 폴더·변경 cache/metadata·서로 다른 TTA checkpoint 거부. 불일치 budget·미완료/smoke·누락 seed/부모·한 seed에만 집중된 개선은 채택 제외.
- 합성 fixture에서 실제 subprocess `experiments → train → evaluate → compare`를 두 번 실행. 완료 run 재호출 시 last/best/metrics SHA가 불변이고 smoke는 추천 제외됨을 확인.
- package의 manifest/모델/TTA/hash 대조와 필수 반복 추론 PASS. palette는 정상이나 한 pixel이 달라진 PNG 및 잘못된 TTA 거부. 패키지 압축 해제 후 **같은 dependency 환경의 별도 Python 프로세스** 추론에서 PNG hash 일치. 독립 clean install 검증은 아니다.
- 조기 종료 완료 상태/history 일치 시 남은 해당 run 비용 0. NaN/Inf/음수 가격 거부. 실제 GPU 시간·Vast 비용은 미측정.
- 실제 train 70장 morphology를 생성했다. Eutectic Si 성분 63,393개, 면적 중앙값35 pixel, 64 pixel 이하69.10%. 원본/마스크 SHA를 함께 기록했다. 연결 성분을 물리적 입자로 동일시하지 않는다.
- 실제 train의 두 개 128×128 crop에서 Core/O/L/S 출력 `[2,4,128,128]`, finite loss 및 존재하는 gradient finite. **optimizer step 0회, checkpoint 저장 없음**. Core 13,625,228 / scSE 13,715,677 parameters. 성능 시험이 아니다.
- Python 소스·테스트 27개의 AST, recipe10개와 제출 계약1개, 문서27개의 내부 링크 및 코드 모듈20개의 문서 coverage 통과. CLI morphology 및 package/verify-package 경로 확인.
- 실제 데이터 inspect 결과는 기존 `runs/data_report.json`과 동일. PNG **190개 모두 원본 train/valid/test ZIP의 PNG byte SHA와 일치**. 원본 데이터·archive는 수정하지 않았다.
- Notion [v4 설계](https://app.notion.com/p/3f5e1c573e7581928b48c9bbf1bda770) 생성 및 프로젝트/기술/v3 이력/파일별 계약 갱신 후 재조회 확인. 기존 하위 페이지를 보존했다.

재현 명령(PowerShell, ALPHASE 폴더):

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:HF_HUB_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:ALPHASE_VERIFY_PYTHON=(Resolve-Path .venv/Scripts/python.exe).Path
& .venv/Scripts/python.exe -m unittest discover -s tests -v
```

## 남은 확인

실제 CUDA/512/AMP/VRAM·시간, 허용된 pretrained 학습, 대회 validation 성능, 총 GPU 비용, 독립 clean-GPU 설치·재실행, 미확정 공식 제출 규정. 후보 O/L/S는 개선 가설이며 기본에 일괄 활성화하지 않는다. CPU PASS나 3seed gate는 수상 또는 test 일반화의 증거가 아니다. GPU 임대·크레딧 지출·본학습·제출은 하지 않았다.

---

# v0.3 검증 이력 — 2026-10-10 KST

## 완료한 검사

- 로컬 원본 train70/valid20/test10 입력과 90 GT mask 재검사. saved `runs/data_report.json`과 동일. mask palette·opaque alpha·쌍·shape·정확 중복 검사 통과.
- Python 3.12 / torch 2.7.1+cpu / torchvision 0.22.1+cpu / SMP 0.5.0 / timm 1.0.15 환경.
- `python -m unittest discover -s tests -v`: **31개 CPU 테스트 PASS**. HF offline 설정으로도 통과. 기존 21개에 경쟁 recipe 회귀 10개 추가.
- 추가로 `ALPHASE_VERIFY_PYTHON`에 현재 venv의 절대 Python 경로를 지정하여 workflow 6개 재실행 PASS. source 압축 해제→별도 프로세스 추론→예측 PNG hash 동일 확인을 포함한다. 6개는 위 31개의 부분집합이며 37개 고유 테스트가 아니다.
- 실제 train 이미지에서 두 개의 128×128 crop을 각 Core/CN에 입력: 출력 `[2,4,128,128]`, finite loss, 모든 존재하는 gradient finite. optimizer step 0회, checkpoint 저장 없음. 실제 성능 시험이 아닌 수치 검사.
- 모델 파라미터 실측: Core 13,625,228 / CN 31,929,188.
- 합성 integration의 EMA/연속 vs epoch 재개·checkpoint 로딩·infer/package 일관성 확인. 합성 점수는 대회 점수로 공개하지 않는다.
- CLI `--help`에서 benchmark와 기존 경로가 노출되는지 확인.
- 최종본에서 Python AST·모든 recipe config·문서 24개의 내부 링크·18개 코드 모듈의 문서 coverage 검사 통과.
- 원본 PNG 190개 및 ZIP 4개를 최초 SHA-256과 대조하여 **194개 모두 변경 없음** 확인.
- Notion에 v3 설계와 파일별 계약 페이지 생성, 기존 프로젝트/기술 페이지의 현재 기준·상태 갱신, 읽기 재확인 완료.

초기 테스트에서 Windows 기본 cp949로 UTF-8 보고서를 읽는 오류가 발견되어 프로젝트의 JSON/Markdown 읽기를 UTF-8로 명시했다. 초기 preprocessing 정보 확인 중 SMP가 메타데이터 config를 조회했으나 사전학습 모델 가중치를 다운로드하거나 사용한 학습은 하지 않았다. 이후 테스트는 해당 메타데이터 조회도 mock하여 offline 검증했다.

## 확인된 개선과 제한

| 발견 사항 | 조치 | 남은 검증 |
|---|---|---|
| rare_class=0 고정 | train 관측에 맞춘 phase 0/2 sampler와 boundary branch | 실제 E→F mIoU |
| class 채널을 합친 경계 | classwise soft-boundary Dice | F→G 필요성 |
| 문서512 vs 코드256 | v3 512/128 recipe 추가, 기존 설정 보존 | GPU VRAM/시간 |
| 단일 LR/no warmup | encoder/decoder LR, warmup, EMA, 누적 | pretrained 허용 및 실제 효과 |
| 검증 tile batch1 | batch 가능·D4 역변환 | 실제 GPU inference AMP/속도 |
| base 설정의 ablation 오염 | canonical 비교 설정과 자동 expensive search off | 최종 multi-seed 선택 |
| 고정 Core 보고서 문구 | 평가 metadata의 실제 모델 구성 사용 | 최종 보고서 수동 편집 |
| 구환경 검증 문구 혼재 | 최신 문서·이력 분리 | GPU lock/clean install |

## 하지 않은 일

- GPU 임대, 크레딧 지출, 서비스 설정 변경.
- GPU CUDA/AMP/512 batch fitting, pretrained weights 미세조정, 전체 데이터 본학습.
- 측정된 대회 validation mIoU·test mIoU 주장, 우승 보장.
- 공식 제출, 원본 데이터/원본 archive 변경.
- 독립 GPU 환경에서의 설치·재실행 검증. 같은 dependency set의 subprocess 검증과 구별한다.

## 후속 게이트

규정 확인 → 지출 승인 → CUDA 설치/자원 probe → 실데이터 GPU smoke와 benchmark → 단계별 학습 → 반복 seed와 제한 TTA/ensemble → 제출 계약·패키지·보고서 검증.

검증 범위는 파일별 [계약](FILE_MAP.md)과 함께 읽는다. 한 CPU fixture의 성공이 모든 batch/해상도/GPU에서의 동작이나 유효한 외부 제출 형식을 보증하지 않는다.
