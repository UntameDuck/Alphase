# resources.py — GPU 자원·시간 측정

대상: [소스](../../alphase/resources.py). 설계 v3 / 코드 v0.3.0 / 2026-10-10.

## 책임과 인터페이스

tune은 optimizer/EMA를 포함한 메모리 probe; benchmark는 실제 crop train step 및 전체 valid 입력 추론 timing.

## 불변 조건·위험

CPU는 GPU 검증이라고 주장하지 않음. random-init benchmark는 품질 평가가 아님. 실제 학습·평가 peak와 재확인 필요.

## 완료 기준·검증

CPU skip 회귀 통과; CUDA/AMP/VRAM/timing은 아직 미검증.

회귀 검사: `tests/test_competition.py / tests/test_engine.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
