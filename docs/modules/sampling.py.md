# sampling.py — train 전용 crop

대상: [소스](../../alphase/sampling.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

이미지당 고정 crop 수, 60/25/15 sampler, D4·완만한 scale·밝기 증강, 선택적 RAM cache.

## 불변 조건·위험

phase 0/2 중 존재하는 상만 선택; empty→uniform. component 모드는 phase branch 안에서 8-connected 적격 성분을 균등 선택한 뒤 내부 pixel을 선택한다. 기본 최소 면적은 4 pixel이며 해당 성분이 없으면 기존 phase-pixel 선택으로 돌아간다. 작은 라벨을 지우거나 loss에서 제외하지 않는다. mask resize nearest, pad=255. RNG=(seed,epoch,index).

## 완료 기준·검증

RGB/GT 좌표 일치, epoch 재현·변화, absent fallback, 캐시 원본 불변.

회귀 검사: `tests/test_competition.py / tests/test_algorithms.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
