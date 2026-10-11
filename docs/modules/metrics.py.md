# metrics.py — 공식 목적·진단

대상: [소스](../../alphase/metrics.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

image-wise macro IoU, 상별 FP/FN, Boundary F1, 작은 GT 성분 recall, paired bootstrap, 이미지별 상의 2D 면적 분율 절대오차 및 image-macro MAE.

## 불변 조건·위험

union=0과 GT-only absent를 구분. bootstrap은 선택 편향·상관 시편을 보정하지 않는다. 2D 면적 분율은 체적·물성 예측이 아니며 GT와 pred의 상별 면적비 차이다.

## 완료 기준·검증

손계산 혼동행렬, macro/global 차이, absent 정책, boundary tolerance 검사.

회귀 검사: `tests/test_metrics.py / tests/test_algorithms.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
