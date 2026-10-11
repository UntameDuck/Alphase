# 실패·중단·재개 행동 지침

적용: ALPHASE v0.4. 정상 순서는 [행동 지침](ACTION_GUIDE.md), 증거 기록은 [양식](EXECUTION_RECORD_TEMPLATE.md)을 사용한다. 이 문서의 예시 명령은 해당 파일·설정·승인을 먼저 확인한 후 사용한다.

## 공통 원칙

1. 새로운 학습·평가를 겹쳐 실행하지 않는다. 기존 프로세스가 계속 실행 중인지 먼저 확인한다.
2. traceback, config/environment/history/status, 마지막 유효 checkpoint와 평가 폴더를 보존한다. 실패 사실을 지우기 위해 파일을 덮어쓰거나 confirmation flag를 켜지 않는다.
3. 같은 source/config/data/steps를 유지하는 재개인지, 설정이 바뀌는 새로운 실험인지 구분한다.
4. 새 학습이 필요하면 다른 출력 루트·run ID와 새로운 비용 계획을 사용한다. 원본 checkpoint는 보존한다.
5. 코드에 비용 상한 차단·자동 호스트 종료는 없다. 승인 한도·시간을 넘기기 전에 감시 담당자가 중단·백업·청구 정리를 수행한다.

## R1. 로컬 테스트·설정 실패

**징후:** import error, config validation, 단위 테스트 실패, source manifest mismatch.

- 확인: 활성 Python·cwd·dependency·변경 파일을 먼저 확인한다. 다른 프로젝트 venv나 Windows CPU lock을 GPU에 복사한 것은 아닌지 본다.
- 행동: 오류 재현에 필요한 가장 작은 테스트부터 검사한다. 수정했다면 관련 테스트와 전체 회귀를 다시 실행하고 설계/계약/manifest를 함께 갱신한다.
- 금지: 실패한 테스트 삭제·skip으로 성공을 만들기, 임의 library upgrade로 checkpoint 호환성을 깨기, 변경 이유 없이 manifest만 갱신하기.
- 재개 조건: 원인·수정·검증 결과 기록. source가 바뀌면 이전 training last.pt의 재개가 거부되는 것이 정상이다.

## R2. CUDA·설치 실패

**징후:** CUDA unavailable, driver/library mismatch, CPU-only torch, dependency conflict.

- 확인: `nvidia-smi`, Python 경로, torch/torchvision/CUDA 버전, GPU 노출 여부, `pip check`.
- 행동: 실제 driver에 맞는 공식 GPU runtime/torch 조합을 확인한다. 설치 변경 비용도 승인 진단 한도에 포함한다. 새 환경 설치 내용을 기록한다.
- 금지: CPU로 자동 fallback해 얻은 시간으로 GPU 비용을 산정하기, CPU skip을 GPU PASS로 기록하기.
- 재개 조건: CUDA 장치 확인→CPU 회귀→실제 CUDA probe/smoke. 설치만 성공했다고 본학습으로 넘어가지 않는다.

## R3. OOM·비유한 loss·AMP 반복 skip

**징후:** training 또는 validation의 OutOfMemoryError, NaN/Inf loss/gradient, 반복적인 skipped_amp_steps.

- 확인: 어느 단계에서 실패했는지, 동시 프로세스·예약 메모리, input/patch/batch, inference batch, inference AMP, Lovasz 정렬, cache/RAM을 구분한다.
- 행동: 아직 full run 전이면 공통 batch를 줄이고 accumulation으로 effective batch를 맞춘 새 설정을 만든다. 모든 비교 모델이 같은 조건을 통과하도록 다시 probe/smoke한다.
- 학습 중 실패: 유효한 last.pt와 로그를 보존한다. device/config/AMP/LR/batch 등을 바꾸면 기존 run의 동일 조건 재개가 아니므로 새 실험 계열이 필요하다. 무관한 GPU 프로세스의 제거 등 환경 복구만으로 같은 조건을 회복할 수 있는지 먼저 확인한다.
- 금지: NaN을 무시하고 진행, AMP skip이 계속되는데 학습 완료로 해석, 모델마다 다른 해상도/학습량의 점수를 순수 구조 효과로 비교.
- 재개 조건: finite update, 실제 evaluation 메모리까지 통과, 새 예산/시간과 source/config 정책 기록.

## R4. 프로세스·호스트 중단

`last.pt`는 **완료 epoch 경계**의 재개 checkpoint다. 중간 batch까지 이어지는 것이 아니다. epoch 중단 시 마지막 저장 이후의 epoch 작업을 다시 수행할 수 있으며 그 비용도 든다.

### 같은 폴더 재개

다음은 E_s42 full run 예시다. 원래 실행과 같은 plan config를 사용하고 `--steps`를 추가하지 않는다.

```bash
python -m alphase train --data ../data --config plans/screen_v4/E_s42.json --output runs/competition_v4/E_s42 --resume runs/competition_v4/E_s42/last.pt
```

- 원래가 smoke steps2였다면 해당 smoke config와 폴더에서 `--steps 2`까지 같아야 한다. steps를 생략하거나 0으로 설정하지 않는다.
- completed=true인 run은 추가 학습하지 않고 best/inference/history/status를 복구한다. 같은 파생 checkpoint는 hash를 유지한다.
- last commit 뒤 best 저장이 중단된 경우 유효 last로 파생 파일을 복구할 수 있다. 유일한 last 자체가 손상됐으면 자동 복구를 보장하지 않는다. 검증된 백업 또는 새 학습이 필요하다.
- 다른 미완료 프로세스가 같은 폴더를 사용 중이면 동시에 재개하지 않는다.

### 새 빈 폴더 재개

```bash
python -m alphase train --data ../data --config plans/screen_v4/E_s42.json --output runs/recovered_v4/E_s42 --resume runs/competition_v4/E_s42/last.pt
```

목적지는 비어 있어야 한다. fork 후 last/best/inference/history/status를 확인하고 어느 run 루트를 비교에 사용할지 기록한다. 새 폴더 복구가 기존 plan의 output 경로를 자동으로 바꾸지는 않는다. 필요하면 별도 plan·runs 경로를 명시적으로 재구성하고 중복 실험으로 집계하지 않는다.

### 호스트 소실

원격 디스크가 사라지면 atomic checkpoint만으로 복구할 수 없다. 사전에 로컬/승인 저장소에 백업한 source/config/dependency/data/last를 사용한다. 호스트 변경 후 GPU·환경 차이를 검사하고 같은 조건의 재현성을 다시 확인한다. 사용자가 비용을 승인하기 전 대체 호스트를 임대하지 않는다.

## R5. 평가 cache가 남았거나 출처가 맞지 않음

**징후:** empty evaluation directory, stale evaluation, provenance/cache SHA, checkpoint/TTA mismatch.

- 완료 run 재호출인데 checkpoint hash가 달라졌다면 다른 가중치 파일·추론 명세·코드로 바뀌었는지 먼저 확인한다. `best.pt`와 `inference.pt`의 텐서가 같아도 파일 SHA는 다를 수 있다.
- 부분 생성된 `evaluation` 폴더는 자동 삭제하지 않는다. 폴더를 새 보존 이름으로 **복사한 뒤** 파일 목록·해시를 확인하고, 새 평가 폴더에 재평가한다. 자동 runner는 `evaluation`이라는 고정 하위 폴더를 쓰므로 계속 사용하려면 해당 부분 폴더를 안전한 이름으로 이동하는 별도 정리 절차가 필요하다.
- 정리 전에 원본·목적지 절대 경로, 작업 폴더 안의 정확한 대상, 해당 프로세스 종료를 확인한다. broad glob/recursive 삭제를 쓰지 않는다. 사용자 파일과 충돌하거나 이전 유효 결과를 제거해야 한다면 먼저 확인을 받는다.
- code/source가 바뀐 구버전 metrics/cache는 metadata만 수정해 재사용하지 않는다. 새 빈 폴더에서 실제로 재평가한다.
- pretrained 허가·absent 정책이 바뀌었다면 checkpoint의 학습/선택 조건과 어떤 비교가 유효한지 다시 검토한다. run 이름만 같은 것은 재사용 근거가 아니다.

재개 조건: 입력·GT·config·checkpoint·weights·TTA·source가 일치하는 새 평가 완료. `metrics.json`과 모든 cache manifest가 검증돼야 한다.

## R6. 비교에서 후보가 제외됨

| 상태/이유 | 행동 |
|---|---|
| `smoke_incomplete_or_untracked_training` | steps-null full run, completed, 실제 training provenance 확인. smoke metadata를 고치지 않음 |
| `missing_planned_seeds` | 누락 seed를 같은 recipe로 수행하거나 실제 계획을 새로 수립. 빠진 seed를 평균에서 몰래 제외하지 않음 |
| `missing_budget_control` / `missing_parent_control` | 공통 budget 기준·부모를 확보. F→O만 보고 E→F의 채택 근거를 생략하지 않음 |
| `unmatched_training_budget` / `recipe_changes_across_seeds` | 다른 epoch/patch/batch/LR/정책 등 변경을 확인. 같게 다시 실험하거나 별도 비인과 비교로 표시 |
| selection gate false | 평균 이득·과반수 seed·각 상 하락·정확한 단일 변경·부모 적격 여부를 하나씩 확인 |
| `screen_only` | 기능 오류가 아님. 반복 seed 전 후보 screening 단계 |
| `no_eligible_candidate` | 최고 모델 없음. 원인 해결 전 자동 suggested_variant를 만들어 넣지 않음 |

비교 코드의 보수적 규칙을 완화해야 한다면 사전 이유·영향·사용자 결정을 기록하고 새 분석으로 구분한다. 이미 본 결과를 통과시키려고 기준을 사후 변경했다면 그 사실을 숨기지 않는다.

## R7. 추론·패키지 해시 불일치

- 먼저 checkpoint 순서·SHA, 정규화된 weights, TTA 모드, input/contract/source hash, PNG 편집 여부를 확인한다.
- 같은 환경 repeat-check 실패는 다른 옵션·가중치·불안정 실행의 신호다. 기존 prediction을 보존하고 새 폴더에서 같은 최종 명세로 다시 생성한다.
- GPU/driver/library가 달라진 clean 환경에서는 argmax 경계의 수치 차이도 가능하다. PNG를 강제로 복사해 일치시키거나 검사를 꺼서 PASS로 만들지 않는다.
- 재현 가능한 기준 환경과 추론 정밀도를 검토한다. source/config/추론 명세 변경이 필요하면 valid 재평가→test 생성→report 수치 대조→package→clean 검증을 다시 수행한다. 검증하지 않은 deterministic 보장을 문서에 쓰지 않는다.
- receipt.zip이 이미 있으면 덮어쓰지 않는다. 이전 ZIP의 hash를 보존하고 새 artifact 하위 폴더에 같은 실제 접수번호 이름으로 새 버전을 생성할 수 있다.
- package PASS라도 공식 실행 파일 인정·ZIP 구조·용량·서식·제출 성공은 별도 수동 확인이 필요하다.

## R8. 비용 또는 마감 한도 접근

1. 남은 승인 한도와 현재 청구, 실행 중 run의 예상 완료/백업 시간을 즉시 확인한다.
2. 신규 모델·추가 TTA·weight 탐색을 중단한다. 다음 epoch를 무조건 실행하지 않는다. 정상 종료 또는 마지막 유효 checkpoint 보존 방식으로 멈춘다.
3. 확보된 적격 단일 모델과 검증 증거를 우선한다. 비교가 불완전하면 그 한계를 명시하고 수상권/최고 성능을 주장하지 않는다.
4. 필요한 부모 대조군·clean 재현·제출 QA 시간을 빼면서 후보 수를 유지하지 않는다. 후보 수 축소나 추가 비용/시간은 사용자에게 선택을 요청한다.
5. 코드 프로세스가 멈춘 것과 GPU 청구가 멈춘 것은 다르다. 승인된 방식으로 백업·호스트 종료·청구 상태를 확인한다.

## R9. 제출 실패 또는 백업 불완전

- 업로드 요청/진행 화면만 있고 접수 성공을 확인하지 못하면 완료 표시를 하지 않는다. 공식 화면의 오류·시각·대상 파일 hash를 기록하고 동일 파일의 중복 제출 여부를 먼저 확인한다.
- 파일 교체가 필요하면 사용자에게 변경 내용·새 hash를 제시한다. 허가 없이 다른 결과로 제출하지 않는다.
- 백업 파일의 크기·SHA·로드 검사 실패 시 원격 유일본을 지우지 않는다. 먼저 백업을 재검증한다.
- stop/destroy 전 정확한 인스턴스/볼륨 ID·저장 비용·복구 가능 범위를 확인한다. 삭제가 필요하면 승인 범위와 대상을 확인하고 수행 후 무엇이 제거됐으며 복구 가능한지 기록한다.

## 금지하는 빠른 우회

- `steps=0`으로 smoke를 표현하기.
- 끝난 run의 epochs·config를 바꾸고 기존 last에 이어붙이기.
- best.pt를 학습 재개용 last처럼 사용하기.
- 캐시의 checkpoint/hash/점수 metadata만 고쳐 맞추기.
- 공식 확인 대신 contract flag만 true로 만들기.
- 서로 다른 조건의 최고값만 뽑거나 seed가 나쁜 결과를 숨기기.
- 원본 데이터/기존 실험/유일한 checkpoint를 broad 삭제로 정리하기.
- 본학습·지출·제출·외부 연락을 문서 작성 요청에 포함된 승인으로 간주하기.
