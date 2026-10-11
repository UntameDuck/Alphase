# models.py — 모델·AMP·EMA·optimizer

대상: [소스](../../alphase/models.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

SMP factory, config 정규화, auto BF16/FP16, 두 optimizer 그룹, ModelEMA를 소유한다. decoder_attention='scse'는 U-Net/Unet++에만 전달한다.

## 불변 조건·위험

download=False는 encoder_weights=None. 사전학습 다운로드는 permission 확인. EMA는 성공한 step 이후만 update; buffer는 복사.

## 완료 기준·검증

Core/CN 출력 [B,4,H,W], finite backward, EMA checkpoint 재로딩, optimizer parameter 중복 없음.

회귀 검사: `tests/test_competition.py / tests/test_engine.py / tests/test_algorithms.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
