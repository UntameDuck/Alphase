# inference.py — 원본 해상도 추론

대상: [소스](../../alphase/inference.py). 설계 v3 / 코드 v0.3.0 / 2026-10-10.

## 책임과 인터페이스

타일 위치, D4 transform/inverse, 제한 batch의 softmax, Hann blending, crop 복원을 담당한다.

## 불변 조건·위험

norm>0, finite 확률, 모든 원본 pixel 덮기. 하드 마스크가 아니라 확률 합성. none/flip/d4를 저장한다.

## 완료 기준·검증

비정사각·작은·홀수 입력, batch/tile 경계, D4 역변환 일치.

회귀 검사: `tests/test_competition.py / tests/test_engine.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
