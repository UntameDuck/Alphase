# ALPHASE v4 — 데이터 근거·모델 개선·수상 대응 설계

기준: 2026-10-11 KST / 코드 0.4.0. 이 문서가 v3 문서의 상충하는 재개·캐시·추천 규칙보다 우선한다. **구현·수치 검증과 성능 입증을 구분한다. 실제 학습 점수나 수상 가능성을 측정한 문서는 아니다.**

## 1. 승부할 대상과 증거

공식 심사표를 재확인했다. 정확도 50, 독창성·신기술 20, 데이터 분석 15, 활용 가능성 10, 과제 이해도 5점이다. 이 점수는 프로젝트가 획득한 점수가 아니라 배점이다. [공식 심사표](https://komap.ai/front/competition2026/award_info)

| 심사 항목 | 구현·설계 대응 | 최종 보고서에 필요한 실측 증거 |
|---|---|---|
| 정확도 50 | 원본 해상도 4상 분할, 이미지별 mIoU, EMA, 제한 TTA/ensemble | 동일 조건 multi-seed 점수, 상별 IoU, 실패 이미지 |
| 독창성 20 | 데이터의 조직 크기에 따른 sampling 설계와 보존 효과 분석 | F→O 단일 변경 실험, 작은 조직 recall·FP 동시 비교 |
| 데이터 분석 15 | train-only morphology, 중복·색상·shape·group proxy 진단 | 연결 성분 크기 분포와 실제 샘플 노출·오류 관계 |
| 활용 10 | 상별 2D 면적 분율 MAE와 오류 패널 | 전문가 검토 가능한 정량 오차와 실패 사례 |
| 이해도 5 | 공식 image-wise mIoU와 소량 데이터 제약에 맞춘 선택 규칙 | split 유지·규정 확인·재현 패키지 |

기존 scSE/Lovasz/encoder를 우리가 발명했다고 쓰지 않는다. 우리의 기여 후보는 **이 소재 영상의 관측된 분포에 맞춘 설계·실험·해석의 연결**이다. 새롭다는 평가나 성능 향상은 아직 보장하지 않는다.

## 2. 실제 train에서 확인한 개선 근거

재현 명령: `python -m alphase morphology --data ../data --output runs/train_morphology_v4.json`.
70장 train만 사용한다. valid/test의 크기 분포로 sampler를 fit하지 않는다. [실측 JSON](../runs/train_morphology_v4.json)

| 항목 | Primary Si (0) | Eutectic Si (2) |
|---|---:|---:|
| 8-connected 성분 수 | 3,015 | 63,393 |
| 성분 면적 중앙값 | 912 pixel | 35 pixel |
| 64 pixel 이하 성분 비중 | 3.18% | 69.10% |
| 위 성분이 차지하는 해당 상의 픽셀 비중 | 0.0148% | 23.40% |
| 4 pixel 미만 성분 비중 | 1.63% | 3.39% |

**해석:** 기존 phase-pixel sampling은 같은 상 안에서 면적이 큰 성분을 더 자주 선택한다. 작은 Eutectic Si가 많으므로 성분 균등 선택을 비교할 근거가 있다. 이것이 전체 mIoU를 높인다는 결론은 아니다. 과도한 작은 성분 노출은 FP를 늘릴 수 있다.

8-connected 라벨 성분은 물리적으로 독립된 입자와 같다고 보장하지 않는다. 촬영 배율 정보 없이 pixel 면적을 물리적 면적·체적·탄성률로 해석하지 않는다.

## 3. 모델·알고리즘 후보 확정

기준 F는 `champion.json`의 B3/U-Net++, 512/128, batch4×accum2, image-present CE/Dice, phase/boundary sampler, EMA다. 파일 이름 champion은 측정된 우승 모델이라는 뜻이 아니다.

| ID | 부모 | 단일 변경 | 실행 설정 | 가설 / 포기 조건 |
|---|---|---|---|---|
| E | 기준 | uniform sampler | plan의 E | 복잡도 없는 Core 대조군 |
| F | E | phase/boundary sampling | champion.json | E 대비 이득 없으면 E 유지 |
| O | F | phase branch 안에서 component 균등 선택 | candidate_component.json | 작은 Si recall 개선, FP·mIoU 악화 시 포기 |
| L | F | +0.2 per-image Lovasz-Softmax | candidate_lovasz.json | mIoU 개선이 정렬 연산 비용에 미달하면 포기 |
| S | F | decoder scSE | candidate_scse.json | 특징 재가중이 도움 없으면 기존 decoder 유지 |
| G | F | +0.05 classwise boundary loss | plan의 G | 경계 진단에서 필요할 때만 실행 |
| CN | 독립 recipe | ConvNeXt-Tiny/U-Net | challenger.json | 구조 두 요소가 함께 바뀌므로 encoder 효과로 단정 금지 |

O/L/S는 **독립 가설**이다. 전부 켠 조합을 기본 모델로 만들지 않는다. 개별 효과를 통과한 변경 중 최대 2개만 결합 후보로 사전 기록하고, 조합의 추가 효과를 다시 확인한다. 자동 조합 탐색은 구현하지 않았다.

### O — 성분 균등 phase sampling

전체 crop의 60% uniform / 25% phase / 15% boundary 비율은 유지한다. phase branch에서 이미지에 있는 0/2를 균등 선택 → 4 pixel 이상인 해당 상의 연결 성분을 균등 선택 → 성분 내부 pixel 균등 선택 → 그 pixel을 포함하는 crop을 선택한다. 적격 성분이 없으면 기존 phase-pixel로 돌아간다.

4 pixel 미만 성분을 지우거나 loss에서 제외하지 않는다. uniform·boundary·주변 crop으로 계속 학습된다. 4는 극소 성분만을 반복 표적화하지 않기 위한 보수적 초기값이지 최적화된 threshold가 아니다. resize 이후 한 pixel의 생존까지 보장하는 sampler는 아니므로 작은 조직 recall을 실제로 검사한다.

### L — per-image Lovasz-Softmax

손실은 `0.5 CE + 0.5 image-present Dice + 0.2 Lovasz`다. 이미지별/GT-present 상별로 정렬한 오차와 Jaccard 증가량을 계산한다. ignore=255는 제외하고 crop을 동일 가중한다. GT-absent 상의 FP는 CE가 억제한다. FP32 계산이며 top-k 근사나 라벨 축소는 하지 않는다.

IoU 최적화를 위한 알려진 대리 손실을 도입한 것이다. crop mIoU와 원본 image-mIoU가 같다는 주장이나 KoMaP에서의 향상 보장은 없다. 512 입력의 정렬 시간·VRAM은 GPU에서 측정해야 한다. [Lovasz-Softmax 논문](https://arxiv.org/abs/1705.08790)

### S — decoder scSE

SMP의 검증 가능한 `decoder_attention_type='scse'`를 U-Net 계열에 연결했다. 새로운 encoder나 사전학습 출처를 추가하지 않는다. Core 13,625,228 → scSE 13,715,677 parameters: 90,449개, 약 0.66% 증가. 연산 시간·활성화 메모리도 0.66% 증가한다는 뜻은 아니다.

S는 작은 조직이나 경계의 특징을 더 유용하게 재가중할 수 있다는 가설이다. [scSE 논문](https://arxiv.org/abs/1803.02579), [SMP 0.5.0 모델 문서](https://smp.readthedocs.io/en/v0.5.0/models.html)

## 4. 재현성 수정 사항

| 이전 위험 | v4 동작 |
|---|---|
| 조기 종료 run이 다시 학습됨 | `completed/stop_reason` 저장, 완료 run은 학습 없이 export 복구만 |
| last 저장 후 best 저장 실패 | self-contained last에 최고 EMA·best epoch·metrics·전체 history 포함; best/history는 복구 가능한 파생 파일 |
| 다른 폴더 재개 시 best 없음 | 새 폴더에 last/best/inference/history/status 모두 복구; 비어 있지 않은 타 run 목적지 거부 |
| 재개 시 steps 변경·steps=0 무제한 실행 | steps를 checkpoint에 고정, 0/음수 거부, smoke 결과는 추천 제외 |
| 완료 run 재저장으로 hash 변경 | 동일 payload는 재저장하지 않아 기존 평가의 checkpoint hash 유지 |
| stale evaluation/cache 혼합 | 비어 있지 않은 평가 폴더 거부; 파일별 cache SHA·추론 fingerprint·입력/GT hash 대조 |
| 다른 checkpoint를 TTA로 비교 | 동일 config/checkpoint/weights/source/run, 실제 flip/d4 모드 확인 |
| 불공정한 학습 조건도 추천 | budget mismatch, seed별 recipe 변경, 누락 seed·부모 대조군, 미완료 run 제외 |
| 한 seed의 큰 개선만으로 채택 | 다수 seed에서 양수인 경우만 증분 변경 채택; 3개 미만은 screen_only |
| 잘못된 PNG와 모델을 함께 패키징 | manifest/해시/계약 일치 + 같은 환경 반복 추론을 package 안에서 필수 실행 |

v2/v3 가중치는 기본 필드를 보완해 추론할 수 있지만 **학습 재개는 v4 last.pt만 지원**한다. best.pt는 추론 전용이다. v3 cache/metrics는 새 빈 폴더에서 재평가해야 한다. 원본 데이터와 이전 archive는 보존한다.

원자적 rename은 정상 파일시스템의 프로세스 중단 복구 수단이며, 디스크 고장이나 호스트 소실의 백업이 아니다. `last.pt`는 최고 가중치와 history도 포함해 더 커지므로 디스크·업로드 여유를 확보한다. 매 epoch `seconds`는 학습/검증 계산 구간이며 checkpoint/export I/O는 별도 비용이다.

## 5. 제한된 실험 순서

1. 규정·지출 승인·CUDA 설치·모델별 probe/benchmark를 먼저 통과한다. 지금은 실행하지 않았다.
2. 동일 조건의 A baseline, E/F/CN seed42로 시작한다. 짧은 smoke는 기능·발산 진단이며 성능 추천에 쓰지 않는다.
3. F가 E를 통과하고 실측 시간 여유가 있으면 O/L을 우선, S는 추가 후보로 검사한다. G는 경계 실패가 확인될 때만 추가한다. 단일 seed 신규 후보는 총 8 recipe(A/E/F/CN/O/L/S/G)를 넘기지 않는다.
4. 상위 최대 2 recipe와 **그 결론에 필요한 부모 대조군**을 seeds42/123/2026로 반복한다. O의 효과를 주장하려면 F, F의 효과를 주장하려면 E를 생략하지 않는다. 추가 control 비용까지 승인된 예산에 포함한다.
5. 증분 채택 기준: 같은 config/data/source/budget, seed 평균 +0.005 이상, 각 seed/상 IoU 하락 0.02 이내, 다수 seed 개선. 이는 내부 보수적 규칙이지 통계적 유의성 보장이 아니다.
6. 최종 후보 고정 후 none/flip/D4, 단일 및 최대 3개 균등 ensemble만 비교한다. 결과가 비슷하면 단순한 모델을 우선한다. 연속 weight 탐색은 기본 off.
7. 모델 확정 후 test 생성 → package의 반복 추론 → 압축 해제 clean-GPU verify-package → 보고서 수치/패키지 hash 대조.

plan은 canonical E/F를 복원하므로 candidate config를 base로 넘겨도 O/L/S의 단일 변경은 **ID**로 명시해야 한다. 같은 run 루트에서 상위 plan을 확장하면 이미 완료된 일치 run은 추가 학습 없이 재사용한다. 부분 생성된 evaluation 폴더는 자동 삭제하지 않으므로 보존 후 새 목적지로 실행한다.

```bash
# 계획 생성만 수행; 아래 experiments는 승인된 GPU 환경에서만 실행
python -m alphase experiment-plan --config configs/champion.json --ids A E F CN --seeds 42 --output plans/screen_v4
python -m alphase experiments --data ../data --plan plans/screen_v4/plan.json --output runs/competition_v4

# 1단계 결과와 시간 여유가 이를 정당화할 때만 다음 plan을 생성/실행
python -m alphase experiment-plan --config configs/champion.json --ids E F O L S --seeds 42 --output plans/extensions_v4
python -m alphase experiments --data ../data --plan plans/extensions_v4/plan.json --output runs/competition_v4
```

현재 20장 validation의 반복 사용 편향은 남는다. seed 반복은 독립 데이터 검증이 아니다. 실제 시편 ID가 확보되면 train 내부 group-aware 검증을 별도로 설계한다. 확인되지 않은 A번호만으로 임의 재분할하거나 test를 tuning하지 않는다.

## 6. 수상 대응 산출물과 중단 기준

- 비교표에는 model/config/hash, seed, 완료 epoch, 실제 GPU 시간, 전체·상별 mIoU, 작은 성분 recall/FP, 경계 F1, 2D 면적 분율 MAE를 기록한다.
- 이미지별 paired 차이와 worst-case 패널을 포함하고, 일반화·독창성 주장을 측정 범위 안으로 제한한다.
- Lovasz/scSE의 기존 연구 출처를 명시하고, 해당 대회에서의 적용·검증을 기여로 설명한다.
- pretrained 허용, union=0 처리, 출력 파일 규격·실행 파일 인정은 아직 확정하지 않았다. 확인 없이 contract flag를 켜지 않는다.
- 새 독립 GPU 환경 설치, 512 CUDA/AMP, 실측 validation 성능, 총 비용은 남은 게이트다.
- 코드 오류·비유한 loss·출처 불일치·미확정 규정이면 해당 분기를 중단한다. 효과가 없거나 시간을 넘기면 확보된 단순한 검증 모델로 돌아간다.
- 10-14 KST 모델 동결, 10-15 18:00 KST 내부 제출 목표는 유지한다. 새 후보보다 유효 제출과 재현성에 우선순위를 둔다.

이 설계의 완료는 **실험 가능한 경쟁 구성을 준비했다**는 뜻이다. 수상권 성능의 입증은 실제 동일 조건 학습 결과와 심사에서만 가능하다.
