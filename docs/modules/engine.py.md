# engine.py — 학습·평가·checkpoint 조정

대상: [소스](../../alphase/engine.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

분리된 모듈을 조립하고 best/last/inference, history, 상세 지표, test mask를 생성한다.

## 불변 조건·위험

raw 학습 가중치와 EMA 평가 가중치를 구분. v4 last.pt만 config/data/source/steps 일치 시 재개한다. last에 최고 EMA·best epoch·metrics·전체 history를 포함해 먼저 원자적으로 commit하고 best/inference/history/status를 파생한다. 완료 run은 재학습하지 않으며 동일한 파생 checkpoint는 재저장하지 않는다. 새 폴더 재개는 빈 목적지만 허용한다.

평가는 빈 폴더에만 생성하고 입력/GT·checkpoint·config·TTA·cache 해시를 기록한다. metrics.json이 마지막 완료 표식이다. Predictor와 cache ensemble은 최종 확률 float32 반올림을 공유한다. 체크포인트는 신뢰할 수 있는 로컬 파일만 로드한다.

## 완료 기준·검증

연속 vs epoch 재개 raw/EMA/optimizer/RNG/scheduler 동일, 중간 best export 실패·손상 복구, 완료 run no-op·새 폴더 복원, stale 평가 폴더 거부. CPU 검사는 실제 CUDA를 대신하지 않는다.

회귀 검사: `tests/test_workflow.py / tests/test_reliability.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
