# visuals.py — 오류 패널·HTML

대상: [소스](../../alphase/visuals.py). 설계 v3 / 코드 v0.3.0 / 2026-10-10.

## 책임과 인터페이스

원본/GT/예측/overlay/error/uncertainty PNG와 worst-first HTML을 만든다.

## 불변 조건·위험

GT와 예측 palette 동일. confidence는 미보정. 출력 그림은 원본 데이터를 덮어쓰지 않는다.

## 완료 기준·검증

실제 지표와 image ID 일치, HTML escaping, 패널 생성.

회귀 검사: `tests/test_workflow.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
