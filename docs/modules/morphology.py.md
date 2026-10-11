# morphology.py — train-only 조직 크기 분석

대상: [소스](../../alphase/morphology.py). v0.4.0.

`audit_train(root)`는 train image/mask만 읽고 8-connected 상별 성분 수·pixel 면적 분위수·64 pixel 이하 수/면적 비중·4 pixel 미만 성분 비중을 JSON으로 반환한다. 입력/라벨 파일 SHA와 이미지별 성분 수를 기록한다. CLI `morphology`로 재생성한다.

connected region은 물리 입자 ID가 아니다. 배율 없는 pixel을 물리량으로 바꾸지 않는다. valid/test를 sampler fitting에 사용하지 않는다. 회귀: `test_algorithms.py`. [v4 기준](../UPGRADE_V4.md)
