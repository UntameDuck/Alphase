# 실행 기록 양식 — 복사 후 실제 근거로 작성

이 파일은 **미기입 템플릿**이다. 실제 기록은 예를 들어 `docs/records/실행ID.md`처럼 별도 문서로 만든다. 자리표시자를 그대로 승인·결과·비용으로 사용하지 않는다. 비밀번호, SSH private key, API token, 전체 결제 수단 정보는 기록하지 않는다.

[행동 지침](ACTION_GUIDE.md) · [체크리스트](EXECUTION_CHECKLIST.md) · [복구 지침](RECOVERY_GUIDE.md)

## 1. 실행 식별

| 필드 | 값 |
|---|---|
| 실행 ID / 담당자 | 미기입 |
| 시작·마지막 갱신 KST | 미기입 |
| 코드 버전 / source fingerprint | 미기입 |
| SOURCE_SHA256.json SHA | 미기입 |
| 로컬·원격 작업 경로 | 미기입 |
| GPU / VRAM / 인스턴스 ID | 미기입 |
| Python / torch / torchvision / CUDA / driver | 미기입 |
| GPU dependency lock / 설치 명령 기록 경로 | 미기입 |
| data_report 및 data fingerprint | 미기입 |
| 현재 단계 / 다음 행동 | 미기입 |

## 2. 규정 확인 — 근거가 없으면 미확인

| 항목 | 공식 URL/첨부/답변 | 조회 시각 | 확인한 내용 | 결론·반영 파일·검토자 |
|---|---|---|---|---|
| pretrained·외부 데이터·라이선스 | 미기입 | 미기입 | 미기입 | 미확인 |
| ensemble 허용 범위 | 미기입 | 미기입 | 미기입 | 미확인 |
| union=0 / 공식 집계 | 미기입 | 미기입 | 미기입 | 미확인 |
| mask encoding/색상/shape/파일명 | 미기입 | 미기입 | 미기입 | 미확인 |
| 실행 파일·checkpoint·용량·ZIP | 미기입 | 미기입 | 미기입 | 미확인 |
| 보고서 서식·페이지 수·마감 | 미기입 | 미기입 | 미기입 | 미확인 |

규정 미확정 상태에서 허용받은 진단 예외가 있다면: 허용 범위 / 사용자 승인 근거 / 임시 정책 / 해당 결과를 최종 결과에서 제외할 방법을 적는다. 공란은 예외 허가가 아니다.

## 3. 승인 기록 — 단계별로 따로 작성

```text
승인 대상: 유료 GPU 진단 / 실제 본학습 / 추가 실험 / 별도 clean 환경 / 제출 / 인스턴스·볼륨 제거
요청한 작업과 제외한 작업:
대상 인스턴스·서비스·출력/업로드 대상:
가격 확인 시각과 compute·storage·송수신 요금:
최대 허용 달러 / 시간:
중단 조건과 감시 담당:
사용자의 실제 승인 내용·시각·근거:
현재까지 사용액 / 남은 승인 한도:
```

보유 credit $300과 승인 지출 한도는 구별한다. 침묵, 로그인 상태, 코드 작성 요청, 이전 단계 승인은 다음 단계의 결제·제출 승인이 아니다.

## 4. 설정 동결 및 변경 이력

| 항목 | 확정 값 / 경로 / SHA |
|---|---|
| runtime base / pretrained 실제 사용·허가 | 미기입 |
| absent policy / 정규화 | 미기입 |
| patch / overlap / batch / accumulation / effective batch | 미기입 |
| inference batch / AMP·dtype / inference AMP | 미기입 |
| patches per image / epochs / warmup / min epochs / patience | 미기입 |
| LR / encoder LR / loss / sampler / EMA / seed | 미기입 |
| smoke config / steps / 별도 smoke 출력 루트 | 미기입 |
| screen / extension / replication plan 경로·SHA | 미기입 |
| 정식 run 루트 / source freeze 시각 | 미기입 |

```text
변경 전 → 변경 후:
변경 이유와 실측 근거:
다른 항목도 함께 바뀌었는가:
기존 run을 재개할 수 있는가 / 새 비교 계열이 필요한가:
변경 승인 및 새 regression 결과:
```

## 5. 자원·비용 계산표

단위는 정확히 기록한다. `GB-month` 같은 가격을 `GB-hour`로 오인하지 않는다. 미측정 칸에 0을 넣지 말고 '미측정'으로 둔다. 예약 시간·청구 계산·최소 과금은 당시 서비스 기준을 확인한다.

| 대상 recipe/단계 | run 수·seed | 최대 epoch | 실측 epoch wall 초 | 추가 setup/eval/TTA/repeat 시간 | compute 가격·예상액 | storage/전송·기타 | 근거 |
|---|---|---|---|---|---|---|---|
| 진단 smoke·benchmark | 미기입 | 미기입 | 미측정 | 미측정 | 미측정 | 미측정 | 미기입 |
| screen A/E/F/CN | 미기입 | 미기입 | 미측정 | 미측정 | 미측정 | 미측정 | 미기입 |
| 조건부 O/L/S/G | 미기입 | 미기입 | 미측정 | 미측정 | 미측정 | 미측정 | 미기입 |
| 상위 후보+부모 반복 seed | 미기입 | 미기입 | 미측정 | 미측정 | 미측정 | 미측정 | 미기입 |
| 최종 추론·package·clean 재현 | 미기입 | 해당 없음 | 해당 없음 | 미측정 | 미측정 | 미측정 | 미기입 |
| 백업·종료·기타 | 미기입 | 해당 없음 | 해당 없음 | 미측정 | 미측정 | 미측정 | 미기입 |

```text
중복 재사용 run을 제외한 고유 run 목록:
compute 예상 합계:
storage·송수신·기타 예상 합계:
최소 25% 운영 여유와 적용 대상:
미측정·미포함 항목:
총 상한 / 실제 누적 청구 / 승인 잔여:
예상 종료 시각과 제출 QA에 남기는 시간:
최종 비용 산정자·시각·승인 근거:
```

## 6. 실험별 기록 — 각 ID/seed마다 반복

```text
ID / seed / parent control:
사전 가설과 단일 변경 필드:
config 경로·SHA / source·data fingerprint:
실제 명령 / cwd / interpreter:
시작·종료 KST / wall time / 청구 증분:
실행 종류: smoke(steps=값) / full(steps=null)
completed / stop_reason / completed_epochs / best_epoch:
best.pt SHA / metrics 경로·prediction_fingerprint:
image-macro mIoU (0..1):
상별 IoU [Primary Si, Al3Ni, Eutectic Si, Al]:
Boundary F1 / 작은 GT 성분 recall / FP / 면적 분율 MAE:
최대 VRAM·RAM·디스크 / AMP skip / 학습 특이사항:
실패 이미지·panel 경로:
부모 대비 차이 / budget mismatch / gate / missing:
채택·보류·탈락 및 이유:
후속 변경을 하기 전에 확인할 것:
```

## 7. 선택 결정 — 데이터와 해석을 분리

| 비교 | matched config/data/source/budget | seed별 차이 | 평균 차이·분산 | 과반수 개선 | 최대 상별 하락 | gate | 최종 판단·근거 |
|---|---|---|---|---|---|---|---|
| E→F | 미확인 | 미측정 | 미측정 | 미확인 | 미측정 | 미확인 | 미결정 |
| F→O/L/S/G 중 실제 실행한 것 | 미확인 | 미측정 | 미측정 | 미확인 | 미측정 | 미확인 | 미결정 |
| 최상위 단일→TTA/ensemble | 미확인 | 미측정 | 미측정 | 미확인 | 미측정 | 미확인 | 미결정 |

선택 상태: 미측정 / no_eligible_candidate / screen_only / replicated_candidate.

validation20장의 선택 편향, 그룹 상관, 부족한 반복, 누락 대조군, 실제 test 일반화에 관한 한계를 적는다. CN 비교는 encoder·decoder가 함께 달라지는 recipe 비교이며 순수 encoder 효과가 아니다.

## 8. 최종 추론·제출 명세

```text
최종 checkpoint 목록(순서 유지) 및 각각의 SHA:
weights(입력 값과 정규화된 값) / tta-mode:
config·normalization·source fingerprint:
device·GPU·driver·Python·dependency:
확인된 contract 경로·SHA 및 규정 근거:
최종 validation metrics·prediction fingerprint:
prediction manifest / 10개 PNG SHA 목록:
repeat-check 명령·결과·시각:
보고서 원본 / 최종 PDF 경로·SHA·쪽 수:
실제 접수번호 / 최종 ZIP 경로·SHA:
독립 clean-GPU verify-package 환경·명령·결과:
공식 제출 승인 / 업로드 시각 / 접수 성공 근거:
```

## 9. 장애·백업·인계 기록

```text
발생 단계/시각/오류 원문:
영향받은 run과 마지막 유효 last.pt:
원인 확인 전 보존한 파일·로그:
원인과 적용한 복구 절차(R 번호):
변경된 source/config/data/환경 유무:
재실행 명령·성공 기준·실제 결과:
원격→로컬 백업 목록·크기·SHA·로드 검사:
종료 대상 인스턴스/볼륨·승인·시각·청구 확인:
최종 비용 및 남은 credit:
미해결 문제·다음 행동·담당자:
```
