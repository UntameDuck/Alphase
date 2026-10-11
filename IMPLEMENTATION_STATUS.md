# 설계 반영 상태 — v0.4.0

후속 운영 문서: [단계별 행동 지침](docs/ACTION_GUIDE.md), [체크리스트](docs/EXECUTION_CHECKLIST.md), [기록 양식](docs/EXECUTION_RECORD_TEMPLATE.md), [복구 지침](docs/RECOVERY_GUIDE.md)을 추가했다. 이는 향후 실행 지침이며 본학습·GPU 사용·제출을 실행하거나 승인한 기록이 아니다. 아래 테스트·문서 수는 당시 검증 기록이다.

2026-10-11 KST: CPU 테스트 53개, 패키지 별도 프로세스 재추론, 완료 실험 재호출 해시 불변 검사 PASS. 20개 모듈 계약·27개 설계 문서·Notion v4 반영. 원본 PNG 190개는 배포 ZIP과 일치한다. 상세 조건은 [검증 기록](docs/VERIFICATION.md)에 기록했다.

최신 [v4 변경 기준](docs/UPGRADE_V4.md): component sampling, per-image Lovasz, scSE 후보를 독립 설정으로 구현했다. self-contained last/복구, 완료 run 재학습 차단, cache/체크포인트 출처 검증, 공정 budget·다수 seed gate, package 필수 반복 추론을 보강했다. 실제 성능 향상과 수상 가능성은 미입증이다. 아래 v3/v2 기록은 이력이다.

2026-10-10: 사용자 승인에 따라 문서 전용에서 구현까지 진행했습니다.
최신 [설계](docs/README.md), [파일별 계약](docs/FILE_MAP.md), [검증 결과](docs/VERIFICATION.md)가 아래 과거 기록보다 우선합니다.

v0.3 변경: ConvNeXt-Tiny/U-Net 후보, models/sampling/losses/inference 분리, phase 0/2 및 경계 sampler,
image-present Dice/classwise boundary loss, EMA, warmup/차등 LR/gradient accumulation/BF16 선택,
batched tile·D4, 오염 없는 canonical 실험 계획, source-aware resume, GPU benchmark, UTF-8 보고서 처리.
EMA는 평가/최종 checkpoint/제출 패키지에서 동일하게 사용합니다. source/data/config 변경 시 재개는 거부합니다.
GPU·실제 성능·최종 제출 규정은 여전히 미검증입니다. 구현 완료를 성능 입증으로 해석하지 마세요.

## v0.2.0 이전 구현 기록

현재 환경에서 코드로 구현할 수 있는 주요 항목을 확장했습니다. 코드 구현 완료와 실제 경쟁 성능 검증은 별개입니다.

| 설계 항목 | 구현 | 실제 검증 |
|---|---|---|
|ALPHASE 명칭·Core EfficientNet-B3/U-Net++|완료|CPU forward/backward·checkpoint 복원|
|U-Net/ResNet34, U-Net++/ResNet34, EfficientNet-B3/U-Net, DeepLabV3+/ResNet50|factory·config·실험 plan 완료|모두 CPU forward/backward·출력 shape 검사 PASS|
|CE+Dice, Focal+Dice, CE+Tversky, class weights|완료|loss gradient·ignore 검증|
|로그 기반 soft-boundary 보조 손실|완료|CPU gradient 검증; 성능 미측정|
|uniform/rare patch sampling, 동기화 증강|완료|합성 데이터 학습·재개; 클래스 노출 로깅|
|원본 단위 분할·RGB 라벨·ignore padding|완료|자동 검사·수동 지표 대조·합성 integration|
|원본 크기 overlap probability blending|완료|홀수/작은 크기·TTA 일치|
|이미지별 4-class mIoU 평균|완료|수동 계산·absent 세 정책·macro/global 차이|
|A–D encoder/decoder factorial, E/F/G incremental ablation|완료|계획·budget·factor 계산·추천 gate 검사|
|다중 seed·CSV/JSON/Markdown 결과표·paired bootstrap|완료|seed/GT/policy 일치 검사·결과 비교|
|H flip TTA·앙상블 평가·simplex 가중치 탐색|완료|동일-model 앙상블·알려진 최적 가중치·CLI TTA|
|Boundary F1·FP/FN·면적·작은 조직 recall|완료|합성 도형의 허용거리·재현율·false positive 검사|
|비교 패널·HTML validation report|완료|합성 결과에서 파일 생성 검증|
|작은 낮은 confidence 성분의 후처리 후보|검증용 cache 실험 완료|변환 규칙 검사; 기본 추론 비활성|
|GPU patch/batch empirical tuning·AMP|코드 완료|CPU skip 분기 검증; CUDA 분기 미실행|
|학습 재개·RNG·config/data fingerprint|완료|CPU 연속학습과 재개 가중치 bitwise 일치|
|best/last/inference-only checkpoint·환경·노출 로그|완료|integration 저장·재로딩|
|출력 10장·파일명·shape·RGB/RGBA·색상·hash 검사|완료|합성 fixture의 출력·오류 검사; 공식10장 결과 미생성|
|receipt ZIP·PDF ≤10쪽·소스·YAML·추론 스크립트·가중치|완료|합성 fixture와 1쪽 test PDF의 패키징|
|압축 해제·소스 추론 재실행·PNG hash 비교|완료|별도 interpreter에서 same-dependencies 재실행|
|측정된 epoch 시간 기반 비용 추정|완료|합성 실행 history로 계산 검증|
|실측 결과 기반 10개 항목 보고서 초안|완료|Markdown 생성 검사; 최종 제출 보고서 아님|
|Vast.ai/Colab 실행 문서·Colab 노트북|작성 완료|노트북 JSON·Python 셀 구문 검사; 실제 서비스 실행 미검증|

## 외부 환경·사실 확인이 남은 항목

- RTX 5090/T4 등 실제 GPU에서 CUDA/AMP, GPU memory probe, 설치 호환, 속도·VRAM 확인.
- 원본 train/valid 전체 학습, 검증 mIoU, seed별 최종 구조·loss·sampler·TTA·앙상블 선정.
- 외부 ImageNet 사전학습 허용 규정. 기본 pretrained=false 유지.
- 공식 union=0 클래스 처리, 제출 mask 파일명·RGB/RGBA·규격.
- 공식 최종 ZIP/실행 파일/체크포인트 요구, 최종 실측 보고서와 제출 확인.
- 깨끗한 GPU 환경에서 의존성을 새로 설치한 뒤 package 검증. 현재 검증은 동일 dependency set을 사용한 별도 interpreter.

완료되지 않은 사실을 코드에서 임의로 확정하지 않습니다. 성능 수치와 최종 보고서는 실제 학습 결과가 생성된 후 채워집니다.
