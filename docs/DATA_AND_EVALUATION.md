# 데이터·평가·제출 계약

## 실물 검사 — 2026-10-10

로컬 전체 190개 PNG를 다시 읽었다. `runs/data_report.json`의 내용과 재검사 결과는 동일했다. 입력과 mask는 opaque RGBA이고 train/valid의 크기·파일 쌍·정확한 RGB 중복·4색 라벨 검사를 통과했다. test에는 입력 10장만 있으며 GT mask는 없다.

| split | 입력 | GT | 총 유효 픽셀 |
|---|---:|---:|---:|
| train | 70 | 70 | 68,184,888 |
| valid | 20 | 20 | 15,757,260 |
| test | 10 | 0 | 평가 GT 미제공 |

| ID | 상 | RGB | train pixel % | train/valid 미존재 이미지 |
|---|---|---|---:|---|
| 0 | Primary Si | 76,178,76 | 9.4283 | 1 / 1 |
| 1 | Al3Ni | 25,76,153 | 26.3143 | 0 / 0 |
| 2 | Eutectic Si | 204,204,204 | 6.9769 | 0 / 0 |
| 3 | Al | 153,127,76 | 57.2806 | 0 / 0 |

ID 0 부재: train `A57_step1_sample3_image.png`, valid `A71_step2_sample1_image.png`. valid에서 union=0 규칙이 실제 모델 선택에 영향을 줄 수 있으므로 반드시 확인한다.

해상도는 `(H,W)` 기준 480×675, 875×1230, 909×1278, 1024×1440이며 valid에 780×1096도 있다. 크기 차이를 없애기 위한 전체 이미지 강제 축소는 하지 않는다.

train/valid에서 `A125,A153,A181,A191,A40,A55,A7` 접두어가 겹친다. **동일 시편이라는 확증도, 데이터 누수 확정도 아니다.** 공식 split은 유지하고 `prefix_diagnostics`로 seen/unseen proxy 집단의 count·mIoU를 별도 보고한다. 실제 시편 ID 확보 전 임의로 데이터를 재분할하지 않는다.

## 공식 점수

이미지마다 네 상의 IoU를 계산해 mIoU를 구하고 평가 10장에 대해 평균한다. pooled confusion matrix의 global mIoU를 주지표로 쓰지 않는다. [공식 데이터·지표](https://komap.ai/front/competition2026/process1)

`IoU(c,i)=TP/(TP+FP+FN)`, `score=mean_i(mean_c IoU(c,i))`. 단, union=0인 상의 공식 처리는 공개 설명만으로 확정하지 못했다. 코드에는 exclude/zero/one 3정책을 모두 기록하고 임시 주지표는 exclude다. GT에 없더라도 예측에 있으면 union>0이므로 제외하지 않는다.

공식 페이지의 약 75%는 **공식 참고값**이며 우리 모델의 측정값이나 우승 기준이 아니다. 목표 성능 숫자를 근거 없이 약속하지 않는다.

## 데이터 누수 방지

- `sampling.py`는 train의 image/mask만 읽는다. val/test 통계로 sampler나 정규화를 fit하지 않는다.
- `evaluate`는 valid만 점수화하고 test 정답을 찾지 않는다. test는 최종 예측/형식 검사만 한다.
- augmentation은 image와 GT에 같은 좌표 변환. raw mask RGB 값을 자연색 입력과 혼동하지 않는다.
- 데이터 fingerprint에는 검증 및 test 입력 무결성도 포함되지만 이는 해시 검사이지 test 학습이 아니다.
- 단순 file hash와 픽셀 중복 검사는 유사 이미지·시편 중복까지 보증하지 않는다.

## 보조 분석

상별 image-macro IoU, FP/FN, 면적 비율 오차, Boundary F1(유클리드 2 pixel), GT 작은 성분 recall(8-connected, 64 pixel 이하), worst-image 패널. 물리적 배율이 없으므로 2/64 pixel을 물리 길이·면적으로 해석하지 않는다. 작은 성분 recall은 instance IoU가 아니고 confidence는 보정되지 않은 softmax다.

보고서는 평균 점수뿐 아니라 실패 이미지, 희소 상 손실, 경계 과분할, 배율/해상도 차이, 접두어 집단 차이를 보여준다. 예측 상 면적비는 2D 영상의 면적 분율이며 추가 가정 없이 체적 분율·탄성률을 직접 예측했다고 주장하지 않는다.

## 제출 계약과 게이트

공식 요구: 10장 이내 최종 공모안(hwp/docx, PDF 가능), 소스·실행 파일·개발 환경 YAML, test masking 결과 10장의 ZIP. 이 산출물을 `접수번호.zip`에 묶으며 마감은 2026-10-15 24시다. [공식 제출 안내](https://komap.ai/front/competition2026/apply_info)

내부 기본 계약의 RGB PNG, `_image.png→_mask.png`는 **아직 공식 확인 전 임시값**이다. 원본 shape·palette·alpha·개수·충돌·unknown 색상을 검사한다. `output_format_confirmed`, `absent_policy_confirmed`, pretrained 사용 시 `pretrained_allowed_confirmed`가 실제 확인 없이 true가 되어서는 안 된다.

Python 실행 스크립트가 공식 ‘실행 파일’로 인정되는지, checkpoint 포함 및 용량 제한, 세부 파일명은 운영 규정/담당자 확인이 필요하다. 패키징 구현 완료가 제출 형식 승인 또는 실제 제출을 뜻하지 않는다.

모델 최종 선택 뒤 같은 checkpoint/weights/TTA/contract로 valid 재평가 → test 10장 생성 → 반복 추론 → 패키지 재실행 → 보고서 수치/manifest 대조. 새로운 독립 GPU 환경의 설치 검증은 로컬 same-dependencies 재실행과 구별한다.
