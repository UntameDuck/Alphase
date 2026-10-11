# ALPHASE 검증 기록

최신 기록은 [v0.4 검증](docs/VERIFICATION.md)을 따릅니다. 아래 v0.2 내용은 **이전 환경에서의 이력**입니다.
2026-10-10 현재 workspace에는 실제 train/valid/test 데이터가 있으며 전체 무결성을 재확인했습니다.
아래 '첨부 파일이 남아 있지 않다'는 문장은 현재 환경에 대한 설명이 아닙니다.

## v0.2 검증 기록 (보존)

## 수행

- Python 3.12, torch 2.7.1+cpu, torchvision 0.22.1+cpu, SMP 0.5.0, timm 1.0.15,
  huggingface-hub 0.30.2, SciPy 1.17.0, pypdf 5.4.0 환경에서 테스트 **21개 PASS**
  (전체 20개 + 추가 모델 factory 검사; 변경된 engine 테스트 7개 재실행 PASS).
- Core(B3/U-Net++) forward/backward·출력 shape·checkpoint 복원 출력 일치.
- ResNet34/U-Net, EfficientNet-B3/U-Net, ResNet34/U-Net++, ResNet50/DeepLabV3+ 모두 CPU forward/backward·shape 검사 PASS.
- CE/Dice, Focal/Dice, CE/Tversky와 boundary auxiliary loss의 gradient·ignore 처리.
- 정답 4색 매핑·미지 색 거부, 수동 IoU, absent policy, 이미지 macro 지표.
- 경계 허용 거리, 작은 GT component recall, 작은 false-positive 성분 개수.
- 원본 크기 tiled probability, overlap·flip TTA, 작은/홀수 크기 처리.
- 합성 데이터 2 epoch 학습 후 epoch 0 checkpoint에서 재개: 최종 CPU 가중치 bitwise 동일.
- 단일/동일-model 앙상블 평가·HTML/PNG/npz 생성·검증용 simplex weight search 최적 사례.
- A/B/C/D factorial effect와 E/F/G 선택 gate, planned seed budget·미실험 표시.
- 합성 예측 PNG 개수/shape/mode/palette 검증·반복 추론 hash 비교.
- 1쪽 test PDF + inference-only weights + source/env/script + prediction ZIP 패키징.
- 별도 Python interpreter(동일 dependency set)에서 패키지 압축 해제·추론 재실행·PNG hash 일치.
- 실제 CLI experiment-plan → experiments → train/evaluate/TTA/compare를 **합성 fixture**로 실행.
- CLI tune의 no-CUDA skip, 보고서 초안, 비용 추정, Python 구문 컴파일.
- Colab 노트북 구조·Python 셀 구문 검사. 실제 Colab 브라우저 실행은 미수행.

## 구분해야 할 이전 검증

v0.1에서는 사용자가 첨부한 실제 70/20/10 ZIP 검사와 train의 256×256 두 crop Core
forward/backward를 수행했습니다. 이번 세션의 작업 공간에는 그 첨부 파일이 남아 있지 않아
새 기능은 합성 fixture로 검증했습니다. 이번 버전에서 원본 데이터 전체 학습·재검사는 수행하지 않았습니다.
합성 테스트에서 나온 mIoU는 대회 성능이 아닙니다.

## 제한

GPU 및 CUDA/AMP 분기는 실제 GPU에서 확인해야 합니다. CPU의 /proc 및 NNPACK 경고는
제한된 실행 환경에서 발생했으며 테스트는 완료됐습니다. test_engine/workflow 본 프로세스는
NNPACK을 끄고 검증했지만 재실행 subprocess에는 경고가 있을 수 있습니다.

사전학습 규정, 공식 빈 클래스 처리, 출력 계약, 최종 제출 요구는 미확정입니다.
default 계약은 검사용 내부 출발점이며 확인 플래그가 false이면 final package를 거부합니다.

## 재검증

```bash
python -m unittest discover -s tests -v
# package 재실행까지 test할 때 (새 interpreter에 필요한 dependencies가 설치되어 있어야 함)
ALPHASE_VERIFY_PYTHON=/absolute/path/to/python python -m unittest discover -s tests -v
```
