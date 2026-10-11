# ALPHASE 경쟁 모델 설계 v3

v4 추가 후보·채택 규칙은 [업그레이드 기준](UPGRADE_V4.md)을 우선한다. scSE는 S 후보에만 적용하며 기존 Core/CN 기본 구조는 유지한다. 다수 seed 개선 조건은 v4에서 자동 검사한다.

## 목표와 설계 판단

2026 KoMaP의 4상 OM 분할은 거대한 새 네트워크를 처음부터 만드는 것보다 작은 학습 세트의 세부 조직을 보존하고 이미지별 mIoU를 안정화하는 것이 우선이다. 아래 설계는 데이터 크기, 코드 상태, 남은 일정에 근거한 **우리의 판단**이며 해당 대회의 최적 모델이라는 실험적 결론은 아니다.

공식 심사에는 정확도뿐 아니라 독창성·분석·활용·이해도가 포함된다. 기존 backbone을 새 발명이라고 주장하지 않는다. 우리의 기여 후보는 상·경계 중심 학습과 상별/경계별 오류 분석이며, 비교 실험으로 효과를 입증해야 한다. [공식 심사](https://komap.ai/front/competition2026/award_info)

## 1. 모델 구성

| 후보 | 정확한 구현 | 역할 | 파라미터 실측 |
|---|---|---|---:|
| Core | SMP 0.5.0 `UnetPlusPlus`, `timm-efficientnet-b3` | 고해상도 세부 조직·다중 단계 skip 기반 주력 | 13,625,228 |
| CN | SMP 0.5.0 `Unet`, `tu-convnext_tiny.fb_in1k` | 다른 특징 추출 방식의 비교·앙상블 후보 | 31,929,188 |
| A | SMP `Unet`, `resnet34`, CE | 구현·성능 기준선 | 별도 측정 |
| G | Core + classwise boundary loss | 경계 오류를 줄이는 선택적 확장 | Core와 동일 |

입력 `[B,3,P,P]`, 출력 `[B,4,P,P]` logits. 인코더 깊이 5, U-Net 계열 decoder 기본 채널 `(256,128,64,32,16)`, attention·추가 분류 head 없음. SMP의 U-Net++ 구현을 사용하며 논문의 deep supervision을 구현했다고 주장하지 않는다. ConvNeXt는 1/4부터 시작하는 feature hierarchy를 SMP의 universal encoder가 연결한다. 이 조합의 forward/backward는 로컬에서 확인했다.

Core의 `timm-` 이름은 폐기 예정 API이지만 현재 버전에서 확인된 기존 가중치 호환을 유지한다. SMP/timm을 임의 업그레이드하거나 encoder를 `tu-` 이름으로 치환하면 별도 호환 검증이 필요하다. [SMP v0.5.0](https://github.com/qubvel-org/segmentation_models.pytorch/releases/tag/v0.5.0), [universal encoder](https://smp.readthedocs.io/en/v0.5.0/encoders_timm.html)

근거: [U-Net++ 원 논문](https://arxiv.org/abs/1807.10165), [EfficientNet 원 논문](https://arxiv.org/abs/1905.11946), [ConvNeXt 원 논문](https://arxiv.org/abs/2201.03545). 해당 논문의 다른 데이터셋 성능을 KoMaP 예상 성능으로 전용하지 않는다.

## 2. 초기화와 규정 분기

성능 우선 경로는 **허용이 확인된 ImageNet 사전학습 encoder**를 미세조정하는 것이다. 현재 공지에서 외부 사전학습 허용을 확정하지 못했으므로 배포 설정은 `pretrained=false`, `pretrained_permission=false`다. 직접 train 진입에서도 확인 없이 다운로드되지 않도록 차단한다.

허용 근거를 확보하면 실행용 config 복사본에 두 값을 true로 하고 `encoder_lr=0.0001`, decoder `lr=0.0003`을 사용한다. CN은 이름에 `fb_in1k`를 명시하여 암묵적으로 ImageNet-22K 가중치를 선택하지 않는다. 허용되지 않거나 확인할 수 없으면 random initialization 경로를 유지하고 기준선과 비교한다. 이 경우 같은 성능을 보장하지 않는다.

## 3. 데이터·패치 학습

- 원본 이미지별 8 crop으로 이미지 노출 횟수를 동일하게 유지한다. 기본 70×8=560 patches/epoch.
- 공간 크기 512, batch 4, accumulation 2 → 최대 8 crop의 gradient 누적. 메모리 실측 후 필요하면 batch 2 + accumulation 4. BN은 누적 batch 크기를 공유하지 않는다.
- 패치 샘플링: uniform 60%, phase-centered 25%, boundary-centered 15%.
- phase-centered는 해당 이미지에 존재하는 ID 0/2 중 균등 선택한다. train pixel 비율은 ID 2가 6.98%, ID 0이 9.43%다. ID 0 하나를 자동으로 최희소 상이라고 취급하지 않는다.
- 경계는 서로 다른 유효 라벨의 4-neighbor 접합부에서 선택한다. 영상 외곽이나 ignore padding을 조직 경계로 만들지 않는다. 지정 상/경계가 없으면 uniform으로 돌아간다.
- crop field of view 0.9–1.1배 후 512로 resize. RGB는 bilinear, 라벨은 nearest. 과한 scale/elastic 변형은 미세 조직을 왜곡할 수 있어 제외한다.
- D4 회전·반사 동기화. 밝기 증강은 확률 0.3, gain 0.9–1.1, offset −8..8. hue 변화·강한 blur·CutMix·GAN 증강은 기본 제외.
- 작은 이미지는 RGB edge padding, mask 255 ignore. 라벨과 padding이 loss/metric에 섞이지 않는다.
- 정규화 mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`를 config에 기록한다. 승인된 사전학습 출처와 불일치하면 수정 후 다시 비교한다.

샘플 선택은 `(seed, epoch, crop index)`로 결정하므로 loader 순서와 무관하게 재현 가능하다. 이미지/라벨/좌표 캐시는 train만 사용한다. 약 수백 MB 이상의 RAM이 필요하며 실제 host RAM을 확인한다.

## 4. 손실

Core/F/CN 기본: `0.5 × CE + 0.5 × Dice`.

CE는 각 crop의 유효 픽셀 평균을 낸 뒤 crop 평균. Dice는 crop별 **GT에 존재하는 상**을 동일 가중하고 crop별 평균을 낸다. 없는 상의 false positive는 CE가 억제한다. 이는 이미지별 공식 지표를 고려한 대리 목적함수이지 공식 mIoU와 수학적으로 같지는 않다.

G만 `+ 0.05 × classwise boundary Dice`. 3×3 soft morphological gradient를 상별로 유지한다. 기존 `amax` 방식은 상 채널을 합쳐 경계의 위치는 같지만 상의 종류가 다른 오류를 구별하지 못했으므로 개선했다. ignore 주변 1픽셀과 영상 외곽은 제외한다. **G는 검증 전 채택하지 않는다.**

loss의 softmax·overlap·boundary 계산은 FP32. Focal/Tversky/CE class weights는 기존 대안으로 유지하되 동시에 여러 요소를 바꾸지 않는다. inverse-frequency class weights와 과도한 oversampling을 중복 적용하지 않는다.

## 5. 학습 안정화

- AdamW, weight decay 1e-4. random encoder는 3e-4, 승인된 pretrained encoder는 1e-4, decoder는 3e-4.
- 최대 80 epochs, 5 epoch 선형 warmup + cosine decay(최소 비율 0.01), patience 12, 최소 15 epochs.
- CUDA AMP auto: BF16 지원 시 BF16, 그 외 FP16+GradScaler. CPU는 FP32.
- gradient norm 1.0. FP16 overflow는 update를 건너뛰고 EMA도 갱신하지 않는다.
- EMA decay 0.99 상한 + 초기 warmup decay. parameter는 평균, BN buffer는 현재 모델에서 복사. 학습 raw 가중치와 평가 EMA 가중치를 체크포인트에 모두 저장한다.
- pretrained encoder BN running statistics는 freeze. 학습 가능한 affine parameter는 유지한다. random 초기화에서는 BN freeze를 하지 않는다.
- 매 epoch 전체 validation을 원본 크기로 평가. early stop/최고 checkpoint는 동일한 EMA 평가 정책을 사용한다.
- config/data/source fingerprint가 달라지면 재개를 거부한다. epoch 경계 재개를 지원하며 중간 microbatch 재개는 지원하지 않는다.

## 6. 추론과 최종 모델 선택

512 tile / overlap 128 / Hann 가중 확률 합성. 패딩 제거 후 원래 `(H,W)`로 복원한다. argmax는 타일별로 하지 않고 합성 후 한 번만 한다. batched tile 추론은 기본 4, inference AMP는 기본 false로 수치 기준을 유지한다.

TTA 후보는 `none`, 4개 flip, 8개 D4다. 변환별 확률을 역변환 후 평균한다. D4가 무조건 유리하다고 가정하지 않는다.

최종 후보 제한:

1. 가장 좋은 단일 checkpoint.
2. 같은 recipe의 3 seed 균등 확률 평균.
3. Core 계열과 CN의 1:1 확률 평균 또는 검증된 최대 3-checkpoint 균등 평균.

처음부터 연속 가중치 대탐색·classwise threshold 최적화·많은 ensemble 조합을 돌리지 않는다. 20장에 대한 선택 편향을 줄이기 위해 후보 수와 채택 규칙을 사전 고정한다. 격자 검색 도구는 연구용으로 남기되 자동 runner에서는 껐다.

## 7. 선택 기준과 포기 조건

단일 seed는 screen일 뿐이다. 상위 최대 2 recipe를 seeds 42/123/2026에서 비교한다. 복잡도 추가는 평균 image-mIoU +0.005(0.5 percentage point) 이상, 상별 하락 0.02 이하, 가능하면 3 seed 중 2개 이상 양수일 때 채택한다. 마지막 seed 일관성은 현재 수동 검토다. 작은 차이는 단순하고 빠른 모델을 고른다. bootstrap CI는 탐색적 정보이며 통계적 유의성이나 독립 검증을 주장하지 않는다.

상위 recipe 재학습도 동일 validation을 계속 보므로 seed 반복만으로 선택 편향이 제거되지는 않는다. 시편 메타데이터가 확보되면 train 내부 group-aware 검증을 추가할 수 있지만 현재 구현은 공식 split 유지 + 접두어 진단이다.

거대 foundation model 전체 미세조정, 처음부터 새 transformer 학습, GAN 라벨 합성, pseudo-label test 학습, 무조건적 작은 성분 제거, 무제한 하이퍼파라미터 sweep은 이번 기본 전략에서 제외한다. 전부 나쁘다는 뜻이 아니라 현재 근거·시간·재현 비용에 비해 우선순위가 낮다는 결정이다.
