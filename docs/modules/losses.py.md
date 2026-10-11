# losses.py — 상별 목적함수

대상: [소스](../../alphase/losses.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

CE/Dice/Focal/Tversky, classwise soft-boundary, 선택적 per-image Lovasz-Softmax 보조 손실을 계산한다. Lovasz는 GT-present 상별 정렬 오차·Jaccard 증가량을 사용하고 image 평균한다. 후보 L은 0.5 CE + 0.5 Dice + 0.2 Lovasz다.

## 불변 조건·위험

FP32 arithmetic, ignore 제외, image_present Dice. classwise 경계는 amax로 상 정체성을 지우지 않는다. 모든 픽셀이 ignore인 crop은 실패.

## 완료 기준·검증

finite gradient, ignore 영역 gradient 0, 상이 뒤바뀐 경계에 페널티. Lovasz 값·gradient는 SMP reference와 대조하고 perfect/ignore/image 동등 가중 사례를 검사한다. 512 CUDA 정렬 비용은 미측정.

회귀 검사: `tests/test_competition.py / tests/test_engine.py / tests/test_algorithms.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
