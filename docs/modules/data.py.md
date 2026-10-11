# data.py — 입출력·라벨·무결성

대상: [소스](../../alphase/data.py). 설계 v3 / 코드 v0.3.0 / 2026-10-10.

## 책임과 인터페이스

pairs/read_rgb/decode_mask/inspect/data_fingerprint. RGBA→RGB 후 exact palette를 0..3으로 매핑한다.

## 불변 조건·위험

unknown 색·nonopaque alpha·쌍 누락·크기 불일치·정확 중복·ZIP traversal 거부. test GT를 추측하지 않는다.

## 완료 기준·검증

실제 70/20/10 검사와 saved data_report 일치; 원본 데이터를 수정하지 않음.

회귀 검사: `tests/test_metrics.py / tests/test_workflow.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
