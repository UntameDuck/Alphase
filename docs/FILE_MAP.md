# 폴더·파일 설계 지도

v4 추가·변경 기준: [UPGRADE_V4](UPGRADE_V4.md). 추가 모듈은 [provenance](modules/provenance.py.md), [morphology](modules/morphology.py.md)다. `configs/candidate_component.json`, `candidate_lovasz.json`, `candidate_scse.json`은 F와 단일 변경 비교용 설정이다. `tests/fixtures.py`는 합성 출처 fixture, `test_algorithms.py`는 신규 알고리즘, `test_reliability.py`는 중단·출처·채택 회귀를 담당한다. `runs/train_morphology_v4.json`은 train-only 관측이며 학습 점수가 아니다.

## 실제 구조와 경계

```text
코맵/
├─ data/                    원본 train70 / valid20 / test10 (변경 금지)
├─ train.zip, valid.zip, test.zip  원본 배포 archive (보존)
├─ ALPHASE_code.zip         이전 v0.2 snapshot (현재 코드 아님)
├─ .python/, .tools/        로컬 Python/도구 runtime
├─ .downloads/, .uv-cache/  설치 캐시 (학습 데이터/제출물 아님)
├─ .tmp/                   임시 작업 영역
└─ ALPHASE/
   ├─ alphase/             모델·학습·평가·제출 코드
   ├─ configs/             실행 계약·recipe
   ├─ tests/               CPU 회귀·통합 검사
   ├─ docs/                현재 설계 기준과 파일별 계약
   │  └─ assets/           README 전용 시각 자산
   ├─ runs/                데이터 무결성·train-only morphology 관측
   ├─ .venv/               Windows CPU 검증 환경 (제출하지 않음)
   └─ alphase.egg-info/    설치 메타데이터 (소스 기준 아님)
```

raw PNG 190장이나 제3자 패키지 내부 파일마다 중복 설계 문서를 만들지는 않는다. 데이터 파일은 [데이터 계약](DATA_AND_EVALUATION.md), 실행 환경은 [운영 계약](RUNBOOK.md)이 통제한다. 아래 목록은 직접 관리하는 코드·설정·테스트·지원 파일 전체를 포함한다.

## alphase — 파일별 독립 문서

| 파일 | 역할 | 계약 |
|---|---|---|
| `__init__.py` | 패키지 버전 | [문서](modules/__init__.py.md) |
| `__main__.py` | CLI | [문서](modules/__main__.py.md) |
| `config.py` | 타입·수치·recipe 검증 | [문서](modules/config.py.md) |
| `data.py` | 라벨·분할·무결성 | [문서](modules/data.py.md) |
| `models.py` | factory·EMA·optimizer·AMP | [문서](modules/models.py.md) |
| `sampling.py` | train crop·동기화 증강 | [문서](modules/sampling.py.md) |
| `losses.py` | 상·경계 손실 | [문서](modules/losses.py.md) |
| `inference.py` | tile batch·D4·확률 합성 | [문서](modules/inference.py.md) |
| `engine.py` | 학습·평가·재개 | [문서](modules/engine.py.md) |
| `metrics.py` | 공식 목적·진단 지표 | [문서](modules/metrics.py.md) |
| `provenance.py` | 평가·추론·cache 출처 검증 | [문서](modules/provenance.py.md) |
| `morphology.py` | train-only 연결 성분 크기 분석 | [문서](modules/morphology.py.md) |
| `experiments.py` | 단계 실험·비교 | [문서](modules/experiments.py.md) |
| `ensemble.py` | 선택적 ensemble 검색 | [문서](modules/ensemble.py.md) |
| `postprocess.py` | 비기본 후처리 연구 | [문서](modules/postprocess.py.md) |
| `resources.py` | GPU probe·benchmark | [문서](modules/resources.py.md) |
| `budget.py` | 실측 epoch 비용 외삽 | [문서](modules/budget.py.md) |
| `visuals.py` | 오류 시각화 | [문서](modules/visuals.py.md) |
| `reporting.py` | 측정 기반 보고서 초안 | [문서](modules/reporting.py.md) |
| `submission.py` | 계약 검증·패키지 재실행 | [문서](modules/submission.py.md) |

## configs — 파일별 의도

| 파일 | 용도·제약 |
|---|---|
| `champion.json` | v3 주력 Core recipe. 512/128, EMA·누적·phase sampler. 아직 최고점 측정 없음. CUDA 명시. |
| `challenger.json` | 동일 training protocol의 CN/U-Net. backbone/decoder 비교는 두 요소가 함께 바뀌는 recipe 비교다. |
| `candidate_component.json` | F와 phase-pixel/component 균등 sampling을 비교하는 O 후보. |
| `candidate_lovasz.json` | F에 per-image Lovasz 0.2만 추가하는 L 후보. |
| `candidate_scse.json` | F의 decoder에 scSE만 추가하는 S 후보. |
| `core.json` | 기존 256/60 epoch Core. v3 최고 설정이라고 부르지 않는다. |
| `baseline.json` | 기존 R34/U-Net/CE. v3 protocol 비교는 champion base의 canonical A plan을 쓴다. |
| `edge.json` | 이전 rare_class=0/rare_probability=.3 설정. v3 phase sampler G와 다르다. 이력/호환용. |
| `deeplab.json` | 기존 DeepLabV3+/R50. 이번 기본 screen에서는 우선순위 낮음. |
| `smoke.json` | CPU64/2epoch 기능 확인만. 여기의 점수를 성능으로 보고하지 않는다. |
| `submission_contract.json` | 미확정 공식 사항을 false로 유지한 임시 RGB/파일명 계약. 확인 근거 없이 수정 금지. |

`config.py` DEFAULT와 합친 resolved config가 checkpoint에 저장된다. 새 실행 설정은 기존 파일을 계속 덮어쓰지 말고 run/plan 단위로 고정한다.

## tests — 파일별 검증 책임

- `test_engine.py`: Core/기존 factory, loss ignore gradient, tiling/TTA, weights, CPU tune skip.
- `test_metrics.py`: palette, 손계산 IoU/absent/global 차이, 경계 거리, 작은 성분 recall, 후처리·bootstrap.
- `test_workflow.py`: 합성 train/resume/EMA/evaluate/infer/package, filename/ZIP 보안, ensemble/계획/보고서/예산. 별도 interpreter 검증은 환경변수로 opt-in.
- `test_competition.py`: CN, strict config, permission gate, D4 역변환, classwise boundary, crop 정렬/재현/부재, EMA loading, 오염 없는 계획, CPU benchmark skip.
- `test_algorithms.py`: Lovasz reference 값·gradient, scSE 복원, 성분 균등 선택, morphology, 면적 분율 MAE, 잘못된 후보 설정.
- `test_reliability.py`: 완료·중단·손상 복구, 새 폴더 재개, 출처·TTA·budget·다수 seed gate, smoke 제외, 완료 비용 0.
- `fixtures.py`: 실제 대회 점수와 구분되는 합성 평가·cache provenance fixture.

합성 및 CPU 테스트를 CUDA benchmark나 실제 mIoU 시험이라고 바꾸어 부르지 않는다.

## 지원 파일·기록

| 파일/폴더 | 계약 |
|---|---|
| `README.md` | 첫 진입 문서. 프로젝트 개요, 모델 설계, 실행 흐름, 상태와 문서 탐색을 제공. |
| `IMPLEMENTATION_STATUS.md` | v4 구현과 미검증 구분을 상단에 기록. 과거 표는 이력. |
| `VALIDATION.md` | 과거 실행환경의 기록. 최신 결과는 docs/VERIFICATION.md. |
| `pyproject.toml` | 패키지·Python 요구·entry point·SMP/timm pin. |
| `requirements.txt` | 공통 의존성. CUDA torch는 host에 맞게 먼저 설치. |
| `environment.yaml` | 환경 골격. 자체만으로 GPU torch 조합이 완전히 고정되지 않음. |
| `environment-lock.txt` | Windows CPU 실측 lock. 절대 editable path 포함하므로 다른 host 재현용으로 복사 설치 금지. |
| `activate.ps1` | workspace local runtime/cache/temp 활성화. 공유 데이터는 변경하지 않음. |
| `ALPHASE_Colab.ipynb` | v0.2 Colab 절차 이력. 실제 서비스 미검증; v3는 RUNBOOK의 CLI/recipe 사용. |
| `SOURCE_SHA256.json` | 현재 직접 관리 source/config/test/docs manifest. 원본 zip은 별개 이력. |
| `runs/data_report.json` | 데이터 무결성·분포 결과. 학습 성능 결과가 아님. |
| `runs/train_morphology_v4.json` | train 70장의 연결 성분 통계 및 입력·라벨 파일 SHA. |
| `docs/UPGRADE_V4.md` | v4 알고리즘 후보·공정 선택·중단 복구·심사 대응의 최신 기준. |
| `docs/README.md` | 설계 우선순위·진행 상태·문서 탐색. |
| `docs/MODEL_DESIGN.md` | 확정된 후보·학습·추론·채택 정책. |
| `docs/DATA_AND_EVALUATION.md` | 공식/관측/미확정 구분. |
| `docs/RUNBOOK.md` | GPU 승인 후 실행, 비용, 마감. |
| `docs/ACTION_GUIDE.md` | G0–G11 단계별 선행 조건·승인·명령·통과 기준·산출물. 다음 실행의 주 지침. |
| `docs/EXECUTION_CHECKLIST.md` | 현재 미실행 상태와 단계별 확인 목록. 증거 없이 완료 표시 금지. |
| `docs/EXECUTION_RECORD_TEMPLATE.md` | 규정 근거·지출 승인·설정 동결·실험/비용·최종 추론·제출·백업 기록 양식. |
| `docs/RECOVERY_GUIDE.md` | 설치/OOM/중단/cache/불공정 비교/재현 실패/비용·제출 위험의 복구·중단 기준. |
| `docs/FILE_MAP.md` | 이 파일; 문서 coverage의 기준. |
| `docs/VERIFICATION.md` | 이번 검증 결과와 잔여 리스크. |
| `docs/assets/readme-banner.png` | README 상단 ALPHASE 워드마크 배너. 모델·성능 정보는 포함하지 않는 장식 자산. |

## 책임 분리와 변경 원칙

`data → sampling → models/losses → engine → inference/metrics → experiments → submission/reporting`.

순환 의존을 늘리지 않는다. engine은 기존 public API의 thin re-export를 유지하여 tests/CLI 호환성을 보존한다. sampling은 validation split을 모르고, loss는 원본 파일을 읽지 않으며, inference는 정답을 읽지 않는다. resources는 장비 측정만 하고 외부 인스턴스를 생성하지 않는다.
