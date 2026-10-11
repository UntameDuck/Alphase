# submission.py — 출력 검증·패키지·재실행

대상: [소스](../../alphase/submission.py). 설계 v4 / 코드 v0.4.0 / 2026-10-11.

## 책임과 인터페이스

10 PNG contract 검증, inference-only EMA 가중치 export, source/env/script/PDF/ZIP 및 hash 검증.

## 불변 조건·위험

공식 확인 flag가 없으면 final package 거부하며 문자열을 bool로 인정하지 않는다. manifest의 checkpoint/weights/TTA/source/contract/input/output 해시를 대조한 뒤 같은 환경 반복 추론을 package 내부에서 필수 실행한다. GT/optimizer 제외. trusted checkpoint만 load. 업로드 없음.

## 완료 기준·검증

합성 package 재생성 일치, palette-valid 변조 PNG 및 다른 TTA 거부. 별도 interpreter라도 같은 dependencies이면 clean install 검증이라 부르지 않음.

회귀 검사: `tests/test_workflow.py`. 구체적인 실행 결과는 [검증 기록](../VERIFICATION.md)을 따른다.

## 연결 문서

[전체 모델 설계](../MODEL_DESIGN.md) · [데이터/평가 계약](../DATA_AND_EVALUATION.md) · [실행 절차](../RUNBOOK.md). 본 문서는 구현 계약이며 실측 대회 성능을 주장하지 않는다.
