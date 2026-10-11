# 단계별 실행·예산·동결 계획

**실행용 상세 지침은 [ACTION_GUIDE](ACTION_GUIDE.md)가 우선한다.** 이 파일은 전략·비용의 요약이다. 단계별 명령·선행 승인·통과 기준·산출물·실패 대응은 상세 지침, 실제 상태는 [체크리스트](EXECUTION_CHECKLIST.md), 증거는 [기록 양식](EXECUTION_RECORD_TEMPLATE.md)에 남긴다. 아래 짧은 예시만 보고 GPU를 임대하거나 전체 실험을 실행하지 않는다.

v4 실행·재개·추천 계약은 [최신 업그레이드](UPGRADE_V4.md)를 우선한다. 재개는 self-contained v4 last.pt만 가능하며 best.pt는 추론 전용이다. 평가 폴더는 비어 있어야 하고 package에서 반복 추론이 추가 실행된다. epoch seconds는 checkpoint/export I/O를 제외하므로 wall time과 운영 여유를 별도로 계산한다. 반복 단계에는 상위 후보뿐 아니라 필요한 부모 대조군을 포함한다.

아래는 **향후 실행 절차**다. 이번 작업에서 GPU 임대나 본학습은 실행하지 않았다. 모든 예시는 `ALPHASE`가 cwd이고 데이터가 `../data`인 경우다. Windows 활성화는 `activate.ps1`; Linux는 설치된 CUDA 환경의 Python을 쓴다.

## G0 — 지출 전 확인

1. 사전학습 허용, absent policy, mask encoding/파일명, 실행 파일 요구를 확인하고 근거를 보관한다.
2. GPU 가격·VRAM·디스크·네트워크 비용·host 신뢰도를 당일 확인하고 **사용자에게 지출 승인을 받는다**.
3. CUDA torch와 GPU driver를 실제 host에서 검증한다. 현재 `environment-lock.txt`는 Windows CPU 환경 기록이며 GPU lock이 아니다.
4. 모델별 자원 probe 후 비교 실험은 동일 patch·effective batch로 고정한다. `tune`의 모델별 자동 선택 결과를 그대로 비교하면 조건이 섞인다.

```bash
python -m alphase tune --config configs/champion.json --patches 512 --batches 4 2 --output configs/champion_gpu.json
python -m alphase benchmark --data ../data --config configs/champion_gpu.json --output runs/champion_benchmark.json
```

probe는 synthetic 2-step + optimizer + EMA 메모리 검사다. benchmark는 실제 crop timing과 전체 validation 입력 추론 시간만 측정하며 점수는 산출하지 않는다. 둘 다 GPU가 없으면 명시적으로 skip/오류 처리한다. **CPU로 GPU 시간·비용을 추정하지 않는다.** inference AMP=false의 평가 peak가 학습보다 커질 수도 있으므로 실제 benchmark 및 smoke가 필요하다.

batch를 4→2로 바꾸면 accumulation을 2→4로 맞춘다. 512가 불가능하면 양쪽 모델 모두 256/64로 fallback한 별도 실험군을 만든다. 256과 512는 같은 학습량/연산량이 아니므로 같은 표에서 순수 구조 효과라고 주장하지 않는다.

## G1 — 순차 실험

1. CPU tests와 CUDA 실제 한-batch smoke, format QA. 이때 만든 config는 실제 학습 config와 별도 저장한다.
2. A baseline 한 seed. 이어 E(uniform Core), F(phase/boundary sampler), CN을 seed 42로 screen.
3. F가 E를 개선하면 O(component)/L(Lovasz)를 우선 검사하고 S(scSE)는 여유가 있을 때, G(boundary)는 경계 진단상 필요할 때 추가한다. 개선하지 않으면 E를 유지한다. 초기 신규 recipe 총8개를 넘기지 않는다.
4. 상위 최대 2 recipe와 필요한 부모 대조군에 123, 2026 seed를 추가한다. 채택된 sampler/경계/EMA/증강을 묶음 효과가 아닌 개별 원인이라고 주장하려면 별도 대조군을 둔다. 같은 source/config/steps/budget의 완료 run만 재사용하고 smoke는 추천 제외한다.
5. 후보를 고정하고 none/flip/D4 및 제한된 균등 ensemble을 평가. 동일 검증 이미지/정책/GT hash/출처를 유지한다.

```bash
python -m alphase experiment-plan --config configs/champion.json --ids A E F CN --seeds 42 --output plans/screen_v4
# 규정상 pretrained 허용이 확인된 실행용 config에서만 --allow-pretrained 추가
python -m alphase experiments --data ../data --plan plans/screen_v4/plan.json --output runs/competition_v4
```

`plan`은 E/A/D 등을 canonical recipe로 재설정해 base의 Edge 설정이 대조군에 섞이는 것을 막는다. 이 때문에 `champion.json`을 줘도 E는 sampler=uniform, F는 phase_boundary다. `run_tta=false`, `run_ensemble=false` 기본으로 전체 후보에 자동으로 비싼 후처리 실험을 붙이지 않는다. 서로 다른 단계의 plan이 겹치면 기존 run을 덮어쓰지 않는다.

seed 반복 결과 평균·편차, +0.5pp 게이트, 클래스별 최대 −2pp, 개선 이미지 수·worst case·추론 시간으로 결정한다. 코드의 추천은 보조 도구이며 다른 budget이나 누락된 seed를 무시하고 ‘최고’로 단정하지 않는다.

```bash
python -m alphase evaluate --data ../data --checkpoint runs/competition_v4/F_s42/best.pt --device cuda --tta-mode none --output runs/final_eval_none --cache
python -m alphase evaluate --data ../data --checkpoint runs/competition_v4/F_s42/best.pt --device cuda --tta-mode d4 --output runs/final_eval_d4 --cache
```

앙상블은 같은 선택 과정을 통과한 checkpoint만 `--checkpoint`에 나열하고 고정된 `--weights 1 1` 또는 `1 1 1`을 쓴다. evaluate/infer/package/repeat-check 모두 같은 `--tta-mode`를 사용한다. legacy `--tta`는 4-flip을 뜻한다.

위 F는 실제 최종 승자가 아니라 경로 예시다. `final_eval_none/d4`는 수동 검토용 폴더이며 자동 `compare`의 `evaluation/evaluation_tta`와 다르다. 정확한 TTA 비교·채택·최종 명세는 상세 지침 G8을 따른다.

## 예산은 실측으로 갱신

현재 $300 크레딧은 사용자 진술이며 이번에 billing 잔액을 재검증하지 않았다. 과거 대화의 $12–25/$25–40/$50는 벤치마크 없는 계획 추정이며 승인된 지출 한도나 보장 비용으로 쓰지 않는다.

`C = GPU시간×시간당가격 + 디스크 GB·시간 비용 + 송수신 비용`.

`GPU시간 = Σ(실험별 실제 epoch 수×실측 epoch 초)/3600 + 설치/재시작/최종평가/TTA/패키지 재실행 시간`.

초기 budget은 early stopping 없이 max epoch로 잡고 최소 25% 운영 여유를 둔다. **아래는 시세나 예측이 아니라** 계산 이해를 위한 가상 시나리오(10 runs×80 epochs, $0.50/h 가정, GPU 연산만)다.

| epoch 평균(가정) | 학습+epoch 검증 시간 | 연산비 | 25% 여유 포함 |
|---:|---:|---:|---:|
| 1분 | 13.33h | $6.67 | $8.33 |
| 3분 | 40h | $20 | $25 |
| 10분 | 133.33h | $66.67 | $83.33 |

실제 비용뿐 아니라 남은 시간에 맞아야 한다. full run 하나의 `history.csv`가 생기면 `budget` 명령으로 갱신하되 이 명령은 해당 run의 남은 epoch 비용만 포함한다. 여러 모델·TTA·storage 비용은 별도 합산한다. Vast.ai의 stop 후 디스크 비용 발생 여부와 destroy 시 데이터 소실을 확인하고 checkpoint를 로컬로 백업한 후 종료한다. [Vast billing 공식 문서](https://github.com/vast-ai/docs/blob/main/guides/reference/faq/billing.mdx)

## 마감·보고서

2026-10-14 모델/추론 설정 동결, 10-15 18:00 KST 내부 제출 목표. 공식 마감 10-15 24시보다 일찍 완료한다. 10-15에는 새 모델 추가를 하지 않는다. 학습이 실패하면 확보된 최선의 검증된 단일 checkpoint를 제출 대상으로 사용한다.

10쪽 보고서: 문제·공식 지표 / 데이터·라벨 / 모델 / 학습 / 주결과 / matched ablation / 상별·경계 분석 / 실패 사례 / 2D 면적분율 활용과 한계 / 재현·규정·결론. 실제 수치가 없는 표는 미측정으로 남긴다. 상대 비교에서 구현되지 않은 기법이나 독립 검증을 했다고 쓰지 않는다.

90장(train+valid) 재학습은 기본 제외한다. 독립 OOF 근거 없이 공식 validation을 학습에 합치면 최종 모델의 비교 검증 근거를 잃는다. 그 분기는 별도 프로토콜 승인과 규정 확인 후에만 고려한다.
