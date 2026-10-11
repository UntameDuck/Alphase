# __init__.py — 패키지 버전

대상: [소스](../../alphase/__init__.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

코드 버전 0.4.0을 노출한다.

## 불변 조건·위험

모델 성능 버전과 패키지 구현 버전을 혼동하지 않는다.

## 완료 기준·검증

pyproject.toml과 버전 일치.

회귀 검사: `tests/test_competition.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
