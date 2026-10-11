# config.py — 설정 계약

대상: [소스](../../alphase/config.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

DEFAULT + JSON을 검증하고 resolved config를 반환한다. 기존 recipe는 유지하고 v4의 decoder_attention/scSE, lovasz_weight, phase_sampling/component_min_area는 명시적 opt-in이다. seed는 uint32 범위, min_epochs는 epochs 이하로 제한한다.

## 불변 조건·위험

NaN/Inf, 정수·bool 혼용, 잘못된 schedule/sampler/정규화 값 거부. pretrained_permission은 사실 확인 없이 켜지 않는다.

## 완료 기준·검증

동일 config의 재개 및 잘못된 값의 조기 실패. champion/challenger의 새 필드가 누락 없이 처리됨.

회귀 검사: `tests/test_competition.py / tests/test_algorithms.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
