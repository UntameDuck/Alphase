# reporting.py — 증거 기반 보고서 초안

대상: [소스](../../alphase/reporting.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

data_report/evaluation/comparison의 실측 수치를 Markdown 초안에 모은다.

## 불변 조건·위험

측정 metadata의 architecture/encoder/EMA/TTA/ensemble weights로 설명을 만든다. 최종 config와 사람이 대조해야 하며 최종 PDF/템플릿 완성은 아니다.

## 완료 기준·검증

미실험·selection_status 표시, 상별 면적 분율 MAE, 공식 배점과 획득 점수 구분, 알려진 기법 출처, test 점수 날조 없음, UTF-8 저장/읽기.

회귀 검사: `tests/test_workflow.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
