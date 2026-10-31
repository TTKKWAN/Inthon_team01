# 절 분할 개선(clause_split) 실험 정리 (2026-10-31)

설명을 "사건 하나 = 조각 하나"로 더 잘게, 시간순으로 나누는 규칙 기반 분할기를 만들고, v2에서 기존 `split_clauses`와 바꿔 A/B로 비교했다.
**결론: af_full에서 개선이 없어 기본값으로 채택하지 않는다.** 분할기는 옵션(`--clause-lexicon`)으로만 남긴다.
실험은 새 분할 학습 시드 1개(2026)가 끝난 시점에 중단했다. 기존 분할은 시드 3개(2026, 1, 2)를 완료했다.

## 1. 새 분할 규칙 (`clause_split.py`)

`re`와 Train 설명에서 코드로 뽑은 동사 목록 JSON만 쓴다(spaCy·NLTK 없음, 채점 환경에서 동작). 완성 문장은 코드에 없다.

| 규칙 | 기존 `split_clauses` | 새 `clause_split` |
|---|---|---|
| 문장, `then`류, `;`, `, after which`, `, before` | 나눔 | 나눔 + 주어 복원 |
| 쉼표 없는 `before ~ing` (설명의 15%) | 안 나눔 | 나눔, ~ing → 현재형 |
| 문장 첫 `After/Once ~, Y` / `Before X, Y` | 안 나눔 | 나눔, Before는 Y → X로 순서 바꿈 |
| 문장 중간 `X after ~ing` | 안 나눔 | 나눔, 시간순으로 바꿈 |
| `until` | 안 나눔 | 나눔 |
| `followed by` + 명사 | 동사 없는 조각 | 앞 동사 복사 ("display Y" / "Y returns") |
| 동사 나열 `, lowers it` / `and brings up` | 안 나눔 | 나눔 (주어-동사 수 일치, 왼쪽에도 동사, 명사형 동사는 뒤에 목적어 필요) |
| 주어가 바뀌는 절 `, and they …` | 안 나눔 | 나눔, 그 절 주어로 다시 처리 |
| 분사구 `, standing them …` | 안 나눔 | 나눔, 현재형, 동시 사건 |
| `while` | 안 나눔 | 상태면 상태 조각, 아니면 동시 사건 |
| `one after another` / `in turn` 나열 | 안 나눔 | 항목별 조각으로 펼침 |
| 문장 앞 표지어 Then/Next/Finally | 남음 | 제거 |
| 상태 서술 (`remain`, `is empty`) | 사건과 섞임 | 상태 조각, 맨 뒤 |
| 최대 조각 수 | 8 | 12 |

동사 목록: `python clause_split.py --data-dir ../data/Train --splits validation/splits.json --out cache/clause_lexicon.json` (학습 split 3,544개에서 동사 171개).

## 2. 분할 자체의 지표 (`compare_splits.py`, Train 4,170개)

| 지표 | 기존 | 새 |
|---|---|---|
| 설명당 조각 수 | 3.63 | 6.70 |
| 조각 길이 (단어, 중앙값 / p90 / 최대) | 17 / 24 / 48 | 9 / 15 / 35 |
| 동사 2개 이상 든 조각 (근사) | 65% | 20% |
| 동사 없는 조각 | 3.1% | 3.2% |
| SigLIP 토큰 최대 | 57 | 45 |
| 오답에서 바뀐 부분이 차지하는 단어 비율, val_full antonym / spatial / object / drop_corrupt | .32 / .33 / .43 / .47 | .17 / .19 / .27 / .30 |
| 같은 지표, val_short antonym / spatial | .92 / .93 | .52 / .57 |
| order 오답이 정답 조각의 순서만 바뀐 형태인 비율 | 86% | 78% |

텍스트 캐시: 기존 `texts.pt` 텍스트당 절 2.93개, 새 `texts_v3.pt` 5.19개 (텍스트 95,048개 동일, MPNet 전체문장 벡터 동일).

## 3. 학습 없는 SigLIP 점수 (`ab_compare.py --zero-shot`, 기존 → 새, 같은 문제 짝지어 비교)

| 방식 | rule_full | af_full | nn_full | rule_short | af_short |
|---|---|---|---|---|---|
| zs_mean (eval_nnval 정의) | .436 → .409* | .340 → .317* | .747 → .696* | .450 → .484* | .461 → .475 |
| zs_cmean (절별 최고 프레임의 평균) | .472 → .444* | .350 → .339 | .784 → .743* | .593 → .585 | .498 → .528* |

`*` = bootstrap 95% CI가 0을 넘지 않음. zs_cmin(가장 약한 절)은 동점이 많아 무작위 이하라 지표로 못 쓴다.
기존 분할에서도 zs_cmean이 팀 zs_mean보다 대체로 높다(nn_full .784 vs .747, rule_short .593 vs .450).

## 4. 학습 A/B (v2, `config_ab.json`, 시드 2026끼리만 비교)

| 검증 | 기존 → 새 | 차이 [95% CI] |
|---|---|---|
| rule_full | .842 → .844 | +.002 [−.018, +.021] |
| **af_full** | **.653 → .655** | **+.002 [−.024, +.028]** |
| nn_full | .707 → .683 | −.023 [−.050, +.003] |
| rule_short | .879 → .888 | +.009 [−.007, +.025] |
| af_short (probe .69라 신뢰 안 함) | .782 → .817 | +.036 [+.015, +.057] |
| nn_short | .686 → .686 | +.000 |

af_full 유형별 통과율: antonym .76 → .81, object .78 → .81, append_unrelated .84 → .79, drop_corrupt .90 → .85, 나머지 ±.02 이내.
학습 중 선택 지표(rule 평균): 기존 .8605 / .8591 / .8554, 새 .8661 (시드 2026).

## 5. 해석

- 한 단어만 바뀐 오답(antonym, object)은 짧은 조각에서 더 잘 드러나지만, 덧붙거나 빠진 사건(append_unrelated, drop_corrupt)은 잘게 쪼갠 조각 사이에 묻혀 서로 상쇄된다.
- SigLIP 조각 평균이 바뀌어 장면 구분(nn)은 학습 전후 모두 약간 나빠졌다.
- NOTES_leaderboard 기준 af_full이 그대로면 LB도 오르지 않을 가능성이 크다. 텍스트 처리는 지금 병목이 아니다.

## 6. 같이 나온 것

| 항목 | 내용 |
|---|---|
| order 오답의 문장 지름길 | 순서를 바꾸며 주어-동사 수 일치가 깨진 오답이 있다 ("The person places …, then hold up …"). 영상 없이 문법으로 구별 가능 |
| `train_seed` 옵션 | 학습 초기화·샘플링만 바꾸고 split과 검증 문제는 `seed`로 고정 |
| `max_clauses` 옵션 | 절 순서 임베딩 크기 (기본 8) |
| MPS 지원 | `pick_device()`; MPS에서 인코더 fp16 |
| `ab_compare.py` | 여러 시드 run을 rule/af/nn에서 문제별로 짝지어 bootstrap 비교, 학습 없는 SigLIP 점수 포함 |
| `zeroshot/` | SigLIP zero-shot 제출물 (학습 없음, eval_nnval `siglip_zs` 정의). `python zeroshot/export_zs.py --siglip models/siglip --out <dir>` → 359 MiB ZIP, sentencepiece 없이 오프라인 실행 확인. 과적합 확인용 LB 제출 후보 |
| tk_s 도구 | `ablation_textonly.py`, `eval_nnval.py` + `train_v2.nn_rows`를 가져와 CUDA 외 장치에서도 돌게 함 |

## 7. 재현

```bash
python clause_split.py --data-dir ../data/Train --splits validation/splits.json --out cache/clause_lexicon.json
python prepare_features.py --config config_ab.json --part frames
python prepare_features.py --config config_ab.json --part texts
python prepare_features.py --config config_ab.json --part texts --texts-suffix _v3 --clause-lexicon cache/clause_lexicon.json
python ablation_textonly.py --text full --save outputs/textonly_full.pt
python ablation_textonly.py --text mpnet --save outputs/textonly_mpnet.pt
python eval_nnval.py --mlp --v2            # validation/val_af_*.json, val_nn_*.json 생성
python train_v2.py --config config_ab.json --set train_seed=2026 --output-dir outputs/ab_old_s2026
python train_v2.py --config config_ab.json --texts-suffix _v3 --set train_seed=2026 --output-dir outputs/ab_new_s2026
python ab_compare.py --config config_ab.json --sets rule,af,nn --zero-shot \
  --arm old:texts.pt:outputs/ab_old_s2026 --arm new:texts_v3.pt:outputs/ab_new_s2026
```

M3 Pro(MPS) 기준: 프레임 캐시 23분, 텍스트 캐시 두 개 동시 약 55분, 학습 run당 약 8분.

## 8. 한계와 남은 일

- 새 분할 학습은 시드 1개뿐이다. 시드 간 흔들림(약 .003)이 af_full CI 폭(±.025)보다 훨씬 작아 결론은 바뀌지 않을 것으로 본다.
- 새 분할로 제출하려면 `export_v2.py`가 `clause_split.py`와 동사 목록 JSON을 ZIP에 넣고 추론 설정으로 넘겨야 한다(미구현).
- `max_clauses=12`로 학습한 체크포인트는 이 옵션이 없는 `model_v2`에서 로드되지 않는다.
- 이 브랜치(cdj)는 f27b50a 기준이다. tk_s·Sangjun과 합치려면 `model_v2.py`, `train_v2.py`를 맞춰야 한다.
- 분할기에서 남은 오류: 분사 뒤 ~ing 나열, as/when 절, 부대상황(`with its cord lying …`), 대명사 `it`.
