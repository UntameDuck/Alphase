# ensemble.py — 확률 앙상블 탐색

대상: [소스](../../alphase/ensemble.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

v4 단일 checkpoint cache의 출처·파일 SHA·embedded fingerprint·입력/GT·policy·TTA를 검증하고 simplex 후보를 점수화한다. 중첩 ensemble cache는 거부하고 후보 수 상한은 cache 로드 전에 확인한다. Predictor와 같은 최종 float32 반올림을 적용한다.

## 불변 조건·위험

도구는 유지하되 v4 기본은 소수의 균등 ensemble을 evaluate에서 비교한다. 20장 연속 weight 탐색 결과는 선택 편향이 있다.

## 완료 기준·검증

확률 유효성, 같은 이미지/GT, 알려진 최적 조합. cache의 CPU RAM 규모를 확인.

회귀 검사: `tests/test_workflow.py / tests/test_reliability.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
