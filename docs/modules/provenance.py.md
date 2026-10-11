# provenance.py — 추론·평가 출처 계약

대상: [소스](../../alphase/provenance.py). v0.4.0.

checkpoint SHA/config/weights/TTA/source를 prediction fingerprint로 묶는다. 완료된 schema4 평가만 비교·앙상블에 사용한다. 입력/GT identity, 이미지 중복, 합계 점수, cache 파일 SHA를 확인한다. 이는 우발적 결과 혼합 방지이며 악의적 서명 위조를 막는 인증 시스템이 아니다.

추론 가중치 파일 변경 시 새 평가 폴더를 사용한다. 사전 버전 metrics/cache는 묵시적으로 신뢰하지 않는다. 회귀: `test_reliability.py`, `test_workflow.py`. [v4 기준](../UPGRADE_V4.md)
