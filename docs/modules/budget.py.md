# budget.py — 측정 후 비용 외삽

대상: [소스](../../alphase/budget.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

history.csv epoch seconds로 해당 run의 남은 max epoch 연산비를 추정한다.

## 불변 조건·위험

시간당 가격은 사용자 입력 현재값. checkpoint/export I/O·저장·전송·시작·다른 실험·최종추론은 제외. 전체 $300 예산 추정과 구별.

## 완료 기준·검증

빈 history·음수/비유한 rate·잘못된 시간 거부. history 마지막 epoch와 run_status의 완료 epoch가 일치하는 완료 run은 조기 종료라도 남은 비용 0. GPU 측정 전 비용 확정 금지.

회귀 검사: `tests/test_workflow.py / tests/test_reliability.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
