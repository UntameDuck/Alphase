# 단계별 실행 체크리스트

기준: 2026-10-11 KST / ALPHASE v0.4. 상세 행동·명령은 [ACTION_GUIDE](ACTION_GUIDE.md), 기록 필드는 [기록 양식](EXECUTION_RECORD_TEMPLATE.md), 실패 대응은 [복구 지침](RECOVERY_GUIDE.md)을 따른다.

**이 체크리스트는 앞으로의 GPU 실행을 관리하는 문서다.** 문서를 작성한 것과 해당 작업을 완료한 것은 다르다. 체크할 때 반드시 시각·담당자·증거 경로를 실행 기록에 남긴다. 이전 CPU 검사 성공을 GPU 단계의 완료 근거로 대체하지 않는다.

## 현재 다음 행동

**G0 로컬 상태 확인 → G1 규정 확인표 작성. GPU 임대는 아직 하지 않는다.**

| 단계 | 현재 상태 | 넘어가기 위한 핵심 증거 | 별도 승인 |
|---|---|---|---|
| G0 기준선 | 기존 v4 검증 있음 / 새 실행 확인 대기 | source·데이터 보존 및 필요한 회귀 검사 | 원격 업로드는 별도 |
| G1 규정 | 확인 필요 | pretrained/metric/output/executable 근거 | 문의 발송·예외 진행 |
| G2 진단 예산 | 미승인 | 요금·최대 금액/시간·중단 기준 | GPU 임대·유료 진단 |
| G3 GPU 환경 | 미실행 | CUDA·dependency·원격 데이터 일치 | G2 범위 내 |
| G4 smoke·측정 | 미실행 | 모델별 CUDA 검사·wall time·비용표 | 실제-data 진단 포함 |
| G5 기준 학습 | 미실행 | A/E/F/CN의 완료 run·정식 평가 | 본학습 비용 |
| G6 개선 후보 | 조건부 대기 | F→E 이득·새 후보 자원 검사 | 승인 범위 확대 시 |
| G7 반복 seed | 미실행 | 후보+부모의 3seed·공정 조건 | 비용·시간 범위 내 |
| G8 추론 동결 | 미실행 | checkpoint/weights/TTA/contract 명세 | 추가 추론 비용 범위 |
| G9 최종 산출물 | 미실행 | valid·test10·반복 추론·최종 보고서 | 승인 범위 내 |
| G10 재현 QA | 미실행 | ZIP·독립 clean-GPU 재실행·수동 QA | 새 유료 환경 필요 시 |
| G11 제출·정리 | 미실행 | 접수 성공·백업·최종 청구 확인 | 공식 제출·볼륨 제거 |

## G0 — 로컬 기준선

- [ ] 실행 ID·KST 시각·담당자·코드 버전 기록.
- [ ] source manifest 대조, 변경 파일과 이유 확인.
- [ ] 원본 데이터/ZIP 보존 확인.
- [ ] 기존 테스트 증거를 사용할지, 새 환경/수정으로 회귀 재실행할지 결정.
- [ ] 필요한 검사 PASS, 실패 또는 미검사 항목 표시.

## G1 — 규정 확정

- [ ] pretrained·외부 데이터·ensemble 허용 범위의 공식 근거 보관.
- [ ] union=0 처리와 metric 값 범위(0..1 / % 표기) 확인.
- [ ] 10개 출력의 파일명·해상도·encoding·palette·ZIP 계약 확인.
- [ ] 실행 파일 인정·checkpoint·용량·보고서 서식 확인.
- [ ] 미확정 항목이 있으면 최종 결정 보류 또는 사용자 승인된 진단 예외 기록.
- [ ] runtime config/contract는 새 파일로 작성하고 원본 기본 설정 보존.

## G2 — 진단 비용 승인

- [ ] 잔여 credit와 실제 매물/저장/전송 가격·요금 단위 확인.
- [ ] 진단 대상 모델·최대 달러·시간·중단 조건 제시.
- [ ] 사용자 승인 시각과 정확한 허용 범위 기록.
- [ ] 실제-data benchmark/smoke 포함 여부 명시.
- [ ] 감시 담당·청구 확인 주기·백업/종료 방법 확정.
- [ ] 자동 충전·무승인 유료 작업이 활성화되지 않음 확인.

## G3 — 원격 환경

- [ ] 승인한 GPU/VRAM/가격/인스턴스 ID 대조.
- [ ] 허가된 호스트로 필요한 파일만 전송, 비밀키·캐시·로컬 venv 제외.
- [ ] CUDA와 torch/torchvision 확인, 의존성 검사 PASS.
- [ ] 원격 source manifest·train70/valid20/test10·입력/GT 해시 일치.
- [ ] 독립 GPU dependency 버전/설치 명령 보관.
- [ ] 새 환경 CPU 회귀 PASS. CUDA 시험과 구분.

## G4 — GPU 진단·비용 실측

- [ ] 각 모델의 probe와 실제 평가 peak 검사.
- [ ] 공통 patch/batch/accumulation/inference batch 고정.
- [ ] 별도 2-epoch smoke config와 별도 run 루트 사용.
- [ ] `--steps`는 epoch별 제한이며 총 epoch 제한이 아님 확인.
- [ ] finite loss, optimizer update, AMP skip, 원본 크기 추론 점검.
- [ ] 완료 run 재호출 no-op와 checkpoint/evaluation 해시 유지.
- [ ] smoke가 성능 추천에서 제외되는지 확인.
- [ ] 본학습 base의 benchmark와 wall time·storage/전송 측정.
- [ ] 상위 후보+부모 3seed·추론·재현·백업·운영 여유 비용 계산.
- [ ] 본학습 한도와 시간표에 대해 사용자 승인 획득.

## G5 — A/E/F/CN 기준 실험

- [ ] `--steps` 없는 full run, smoke와 다른 폴더 사용.
- [ ] pretrained/metric/source/data/공통 training budget 고정.
- [ ] 학습 중 청구·loss/gradient·AMP·디스크·시간 감시.
- [ ] 모든 run의 completed/stop_reason/steps와 best epoch 검토.
- [ ] 공식 방식 full-resolution validation 및 클래스별 IoU 확인.
- [ ] report.html의 worst case·클래스 붕괴·작은 Si·경계 확인.
- [ ] comparison의 missing/exclusions/gates/selection_status 검토.
- [ ] F→E 결과와 CN 비교에 따라 다음 후보를 기록하고 비용 갱신.

## G6 — O/L/S/G 선택적 추가

- [ ] F의 채택 근거와 남은 비용·시간 확인.
- [ ] O/L 우선, S/G 필요성 및 미실행 이유까지 사전 기록.
- [ ] 새 후보의 CUDA probe·실제-data smoke 완료.
- [ ] 부모 F/E와 동일 조건, canonical ID로 plan 생성.
- [ ] 신규 recipe 총8개 상한 준수. 조합은 별도 승인/대조 실험.
- [ ] mIoU·상별 손실·작은 성분 recall와 FP를 함께 분석.

## G7 — seed 반복

- [ ] 상위 최대2 recipe와 필요한 부모 대조군을 모두 계획에 포함.
- [ ] seeds42/123/2026의 동일 recipe/budget 확인.
- [ ] 완료 seed42의 source/config/steps 해시가 같을 때만 재사용.
- [ ] 누락 seed·미완료·smoke가 없는지 확인.
- [ ] 평균 +0.005, 과반수 양수, 각 seed/상 하락 0.02 이내 gate 확인.
- [ ] seed 분산·paired 이미지 차이·실패 패널·선택 편향 기록.
- [ ] 반복 validation을 독립 test 검증이라고 부르지 않음 확인.

## G8 — 추론 명세 동결

- [ ] 단일/ensemble 후보와 추가 계산비를 제한.
- [ ] TTA는 정확히 같은 부모 best.pt, 같은 seed·source·policy로 비교.
- [ ] `evaluation_tta` 한 폴더가 flip/D4 모두를 자동 비교하지 않음을 확인.
- [ ] 최대3개 균등 ensemble부터 검사, 자동 weight sweep/postprocess off.
- [ ] 최종 checkpoint 순서·SHA·weights·TTA·config·contract·환경 고정.

## G9 — 최종 예측·보고서

- [ ] 최종 명세로 별도 폴더의 valid 평가 생성.
- [ ] 확정 contract로 test10 출력 생성·형식 검사 PASS.
- [ ] repeat-check의 output은 기존 prediction 폴더이며 PNG 해시 일치.
- [ ] 보고서 초안을 최종 한국어/공식 서식으로 편집·검토.
- [ ] 표·그래프·구조·seed·추론 설명·기여와 한계가 실제 증거와 일치.
- [ ] 테스트/참고/가상 점수를 실측 대회 성능으로 쓰지 않음 확인.

## G10 — 최종 QA

- [ ] 실제 접수번호·최종 PDF로 package 생성. 재추론 시간 확보.
- [ ] 별도 clean-GPU에서 압축 해제→재실행→PNG SHA 일치.
- [ ] 공식 요구와 ZIP 내부 구조·실행 파일·YAML·용량 대조.
- [ ] GT/optimizer/비밀키/불필요 데이터 미포함 확인.
- [ ] 최종 ZIP SHA·검증 결과·보고서·checkpoint·소스 별도 보관.

## G11 — 제출과 종료

- [ ] 공식 최신 마감 확인. 사용자 제출 승인 기록.
- [ ] 업로드 후 접수 성공 화면·시각·번호·파일 SHA 보관.
- [ ] 모든 필요한 파일을 승인된 위치로 백업하고 해시/로드 검사.
- [ ] 종료 대상 인스턴스·볼륨을 정확히 확인하고 승인된 방식으로 정리.
- [ ] 실제 총비용·잔여 credit·청구 종료 여부 확인.
- [ ] 다음 담당자에게 파일 위치·남은 위험·재현 명령 인계.

## 이번 실행의 마지막 상태

```text
현재 단계: 미기입 (문서 작성 시점에는 G0/G1 준비)
마지막 성공 명령/시각:
완료 근거 경로:
실행 중 프로세스/인스턴스:
누적 지출 / 승인 잔여 한도:
막힌 항목 / 필요한 사용자 선택:
다음 행동 한 가지:
```
