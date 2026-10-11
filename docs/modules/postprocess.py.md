# postprocess.py — 실험적 작은 성분 정리

대상: [소스](../../alphase/postprocess.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

confidence-gated 성분 제거를 cache에서만 실험한다.

## 불변 조건·위험

v4 cache SHA·출처와 absent policy 일치 확인 후 실행한다. 제출 경로 기본 off. 작은 실제 Eutectic Si를 지울 위험. 기본 infer/package와 미연결이므로 별도 검증 없이 사용 금지.

## 완료 기준·검증

낮은 확신도 고립 성분과 높은 확신도 성분 구별. 개선 입증 없이 채택하지 않음.

회귀 검사: `tests/test_metrics.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
