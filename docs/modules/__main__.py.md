# __main__.py — CLI·작업 경계

대상: [소스](../../alphase/__main__.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

prepare/inspect/morphology/train/tune/benchmark/evaluate/infer/experiment-plan/experiments/compare/ensemble-search/postprocess/report-draft/budget/package/verify-package를 연결한다.

## 불변 조건·위험

GPU benchmark는 opt-in. --tta-mode none/flip/d4와 legacy --tta를 명확히 구분한다. package --device는 필수 반복 추론의 장비를 선택한다. 자동 임대·업로드 없음.

## 완료 기준·검증

잘못된 옵션을 거부하고 evaluate→infer→package에 TTA/weights/contract를 그대로 전달.

회귀 검사: `tests/test_workflow.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
