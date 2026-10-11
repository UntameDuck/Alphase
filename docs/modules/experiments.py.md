# experiments.py — 공정한 비교·선택

대상: [소스](../../alphase/experiments.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

canonical A/B/C/D/E/F/G/O/L/S/CN/DL 등 계획, subprocess 학습, 동일 seed 비교 및 결과표.

## 불변 조건·위험

base의 sampler/loss/Lovasz/attention 오염을 reset. TTA/ensemble 자동 실행 off. 증분 변경은 동일 조건 부모 대조군·평균 이득 0.005·각 seed/상 IoU 하락 한도 0.02·과반수 seed 개선을 자동 검사한다.

미완료·steps 제한 smoke·seed 누락·대조군 누락·불일치 budget·seed별 recipe 변경은 추천에서 제외한다. 후보가 없으면 no_eligible_candidate, 3seed 미만은 screen_only다. TTA는 같은 checkpoint/config/weights/source/run의 flip/d4만 허용한다. 완료 run 재호출은 학습 no-op 후 일치 cache를 재사용하며 부분 평가 폴더를 삭제하지 않는다.

## 완료 기준·검증

오염된 base에서도 대조군 복원, 출처/config/GT/policy 정합성, 누락 seed 표시.

회귀 검사: `tests/test_competition.py / tests/test_workflow.py / tests/test_reliability.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
