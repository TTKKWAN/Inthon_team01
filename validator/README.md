# validator: 로컬 검증기

## ⛔ 최우선: 로컬 최고 기록 미만이면 제출하지 않는다

리더보드 제출 기회는 몇 번 남지 않았다. 이 규칙은 그 기회를 아끼려는 것이다.

- 기준은 **이 검증기의 최고 기록**이다. 리더보드 점수가 아니다.
- **현재 최고: full 0.710 / short 0.821**
  - 측정: 보정 검증기(`calibrated/`, `weights_*.json`) 반 B(홀수 번호 에피소드)
  - 방법: span + 0.3·SigLIP(편향 제거) + 0.3·공간 헤드(편향 제거). 제출본 `diff_probe/submissions/video_judge_patch.zip`, 2026-11-01
- 새 후보도 **같은 조건**에서 잰다. 셀과 비율을 같게 두고, 설정은 반 A로 고르고, 점수는 반 B로 보고한다.
- full과 short **둘 다** 최고 기록 이상일 때만 제출한다.
- 영상을 쓰는 방법은 **다른 에피소드 영상 대조**도 통과해야 한다. 다른 영상을 넣었을 때 점수가 떨어지지 않으면, 최고 기록을 넘어도 제출하지 않는다.
- 최고 기록 미만이거나 같은 조건에서 재지 않았으면 제출하지 않는다. 리더보드로 가설만 확인하는 탐색용 제출도 하지 않는다.
- 학습 모델은 이 검증기에서 리더보드보다 높게 나온다(아래 §1 표). 이 기준은 "내면 안 되는 것"을 거르는 용도이고, 리더보드 향상을 보장하지 않는다.
- 더 높은 기록이 나오면 이 숫자와 방법을 갱신한다.

> **2026-11-01 리더보드 결과:** 위 기록의 제출본(`video_judge_patch`)이 리더보드에서 **0.4952**로, span 단독(0.5661)보다 낮았다.
> 이 검증기는 영상을 쓰는 방법의 리더보드 점수를 맞히지 못한다(아래 §1 표, §3). 기준을 다시 정하기 전까지는 영상·학습 방법을 이 점수만으로 제출하지 않는다.

리더보드 제출 없이 방법을 비교하려고 만든 검증 세트와 채점 코드를 모은 폴더다. 데이터는 모두
`baseline_v2/validation/splits.json`의 **검증 에피소드 626개**(Train, 학습에 안 씀)에서 나왔다.

| 층 | 위치 | 용도 |
|---|---|---|
| **보정 검증기** (메인) | `calibrated/`, `calibrate.py` | 오답 구조를 리더보드에 맞춘 세트. 방법 간 비교 |
| 개별 세트 | `sets/` | 구조·생성기별 세트. 진단과 강건성 확인 |
| 영상 판단기 진단 | `judges/` | 쌍 정확도, span 결합, 생성기 바뀜, 편향 제거 ablation |
| 채점 | `eval_subs.py` | 제출 폴더를 서버와 같은 방식(`Submission.predict`)으로 채점 |

> **`parts/`, `inputs/`, `sets/claude_*.json`은 Claude가 쓴 문장이다. 평가에만 쓰고 학습·증강에는 절대 쓰지 않는다.**
> 대회 규칙 제3조 6항·8항 위반이다. 이 세트로 만든 파생 세트도 마찬가지다.

## 빠른 사용

```bash
# 영상을 안 보는 방법 (GPU, 이미지 불필요)
python validator/eval_subs.py --sets validator/calibrated/full_*.json --centrality --out res.json
python validator/calibrate.py --combine res.json validator/calibrated/weights_full.json

# 제출 폴더 하나 (zip을 푼 폴더). 이미지 경로는 --data-dir (기본 data/Train)
python validator/eval_subs.py --sets validator/calibrated/full_*.json \
    --sub NAME=path/to/unzipped_submission --data-dir data/Train --device cuda:0 --out res.json
python validator/calibrate.py --combine res.json validator/calibrated/weights_full.json
```

`short_*.json` / `weights_short.json`으로 바꾸면 첫 문장 버전이다. 선지 순서는 에피소드마다 고정 시드(`--seed 0`)로 섞는다.

## 1. 보정 검증기 (`calibrated/`)

오답 4개를 어떻게 조합하느냐에 따라 셀 5개를 만들고, **영상을 안 보는 두 제출의 리더보드 점수가 재현되도록** 셀 비율을 맞췄다.

| 셀 | 오답 4개 | 출처 |
|---|---|---|
| `single` | 오답마다 한 곳씩 수정 | Qwen 한 곳 수정(A′) |
| `ab_ab_c` | {a, b, a+b, c}: 수정 하나를 오답 2개가 공유 | A′를 `diff_probe/sim_valsets.merge_edits`로 합침 |
| `ab_ab_ac` | {a, b, a+b, a+c}: 수정 하나를 오답 3개가 공유 | 〃 |
| `share4` | {a, a+b, a+c, a+d}: 수정 하나를 오답 4개 모두 공유 | `sets/qwen_share3_*` |
| `para_negs` | 정답을 다르게 쓴 문장(Qwen)을 고친 오답 | Qwen paraphrase(B′) |

- 버전: `full`(설명 전체), `short`(첫 문장). test 설명 길이를 몰라서 둘 다 본다.
- 맞춘 값: byte 거리 방식(hyuwooma) **0.520**, span 거리 방식 **0.566** (`calibrate.py`의 `LB`).
- 고른 비율 (`weights_*.json`): full = ab_ab_c 0.35 + ab_ab_ac 0.55 + share4 0.10, short = ab_ab_c 0.35 + ab_ab_ac 0.65.
  맞는 혼합이 여러 개(full 58개, short 10개)라 하나로 정해지지 않는다. `calibrate.py --view full`이 범위를 출력한다.
- 쓰는 법: 설정(가중치·epoch)은 **반 A(짝수 번호 에피소드)**로 고르고 **반 B(홀수)**로 보고한다. 영상을 쓰는 방법은 **다른 에피소드 영상을 넣은 대조**도 함께 본다(영상 덕이면 점수가 떨어져야 한다).

**한계.** 맞춘 것은 오답 구조뿐이다. 영상(PerceptionTest)과 오답 생성기(Qwen)는 test와 다르다.
학습 모델은 이 검증기에서 리더보드보다 높게 나왔다(`results_cal/`, 아래 표). **순위 비교용이지 리더보드 예측치가 아니다.**

| 제출 | 리더보드 | 검증기 추정 (full) |
|---|---|---|
| starter | 0.350 | 0.488 |
| marked_v1 | 0.413 | 0.447 |
| minimal | 0.426 | 0.466 |
| marked_v2 | 0.432 | 0.402 |
| baseline_v2 (int8) | 0.439 | 0.573 |
| video_judge_patch (span + SigLIP + 공간 헤드) | **0.4952** | 0.710 |
| byte 거리 (hyuwooma) | 0.520 | 0.528 (맞춘 값) |
| span 거리 | 0.566 | 0.563 (맞춘 값) |

학습 모델 점수는 셀마다 합친 오답(a+b) 수가 늘수록 단조롭게 오른다(starter 0.42 → 0.54). 코드로 합친 문장이 흔적을 남겼거나, 거르기 쉬운 오답이 늘어난 효과로 본다.

## 2. 개별 세트 (`sets/`)

| 세트 | 오답 | 출처 |
|---|---|---|
| `claude_short` | 첫 문장 정답의 한 곳 수정 4개 | **Claude가 씀** (`parts/`). 평가 전용 |
| `claude_ab_ac` | 위 수정으로 만든 {a, b, a+b, a+c} | `parts/` + 코드 |
| `qwen_full/short` | Qwen 한 곳 수정 4개 | HF `merged/bank_qval.json` |
| `qwen_share3_full/short` | {a, a+b, a+c, a+d} | HF `qwen/qwen_negatives_from_gt.json` |
| `para_negs_*`, `para_same_*`, `para_share3_*` | paraphrase 기반 변형 | HF `qwen/qwen_paraphrases_and_negatives.json` |
| `af_full` | 규칙 오답 | `baseline_v2/validation/val_af_full.json` 복사 |

Claude 세트는 **오답 생성기가 바뀌어도 버티는지** 보는 데 쓴다(학습 오답은 Qwen).

## 3. 영상 판단기 진단 (`judges/`)

제출본(`diff_probe/video_judge/`, span + SigLIP + 학습 헤드)을 만들 때 쓴 평가다. GPU와 서버 캐시가 필요하다.

| 스크립트 | 내용 |
|---|---|
| `siglip_rerank.py` | span + SigLIP zero-shot 정렬 점수, 보정 검증기에서 λ를 반 A로 고르고 반 B로 보고 |
| `rerank_eval.py` | 판단기별 쌍 정확도(정답 vs 한 곳 수정, 수정 종류별) + span 결합. 참조 영상 64개 평균을 빼는 글 편향 제거 포함 |
| `rerank_eval_patch.py` | 위와 같고 공간 헤드(`diff_probe/patch_head.py`)용 |
| `robust_eval.py` | Claude 세트 vs 같은 구조의 Qwen 셀, 편향 제거 on/off ablation |

필요한 것 (모두 서버 `~/Inthon_tk` 기준, git에 없음):
- `diff_probe/`의 모듈: `pair_head.py`, `patch_head.py`, `siglip_pairs.py`, `centrality.py` (학습 코드와 같이 쓰므로 그 자리에 둔다)
- 특징 캐시: `baseline_v2/cache/frames.pt` (`baseline_v2/prepare_features.py`), `baseline_v2/cache/patches4.pt` (`diff_probe/cache_patches.py`)
- 헤드 체크포인트: `diff_probe/runs/pair_head*/`, `diff_probe/runs/patch_head*/` (`pair_head.py`, `patch_head.py`로 학습, A100 수 분)
- Qwen 오답 원본: `diff_probe/runs/qwen_negatives_from_gt.json` (아래 HF 파일)

주요 결과 (보정 검증기 반 B, full / short):

| 구성 | 점수 | 다른 영상 대조 |
|---|---|---|
| span 단독 | 0.576 / 0.560 | 같음 |
| + SigLIP (편향 제거) | 0.631 / 0.753 | 0.52 / 0.49 |
| + SigLIP + 공간 헤드 (제출본 `video_judge_patch`) | 0.710 / 0.821 | 0.50 / 0.45 |

생성기가 바뀌면(Claude 세트, 공유 구조) 이득이 줄어든다: 0.767(Qwen) → 0.635(Claude). 편향 제거는 SigLIP에 일관되게 도움(+0.00~0.03),
공간 헤드에는 잡음 범위(±0.015)지만 다른 영상 대조 점수를 낮춘다.

**리더보드 결과 (2026-11-01): `video_judge_patch` 0.4952**, span 단독 0.5661보다 0.07 낮다.
- 이 검증기 셀에서 영상 판단기 대신 무작위 점수를 넣고 같은 공식(span + 0.3·z + 0.3·z)으로 시뮬레이션했다.
  판단기가 찍기 수준(쌍 정확도 0.50)이면 0.47, 0.58이면 0.57이 나온다. 0.4952는 test에서 쌍 정확도 **약 0.52**에 해당한다. 영상 신호가 test에서는 거의 없었다는 뜻이다.
- 로컬에서도 판단기가 약한 수정은 동작(action)과 순서(order)다. SigLIP은 0.52~0.58, 헤드는 0.57~0.73이다. 반면 이 검증기의 Qwen 오답에는 판단기가 강한 사물·속성·동작 추가 수정이 60~70% 섞여 있다(full 59%, short 69%).
  test 오답이 TemporalBench처럼 동작과 순서 위주라면 로컬 점수가 부풀려진 이유가 설명된다. 다만 이는 아직 확인되지 않은 가설이다.

## 4. 데이터 출처와 재생성

- 검증 분할: `baseline_v2/validation/splits.json` (valid 626 / train 3,544)
- Qwen 오답: HF `SANGchu-03/INThon_Augmented` @ `cbb24e162e35128742da011eb41aa289b007ee09`,
  `Qwen/Qwen3-4B-Instruct-2507` (리비전 `cdbee75f17c01a7cc42f958dc650907174af0554`), fp16, greedy.
- 파일 위치 (git 제외):
  - `qwen_valsets.py`는 HF 저장소를 받은 폴더 그대로(`<dir>/qwen/qwen_negatives_from_gt.json`, `<dir>/qwen/qwen_paraphrases_and_negatives.json`)를 읽는다.
  - `build_sets.py`는 `validator/hf/`에 평평하게 둔 `qwen_negatives_from_gt.json`, `qwen_paraphrases_and_negatives.json`, `bank_qval.json`(HF `merged/`)을 읽는다.
- 다시 만들기:

```bash
python diff_probe/qwen_valsets.py --hf-dir <HF 저장소 폴더> --out diff_probe/valsets_qwen.json  # single / ab_ab_c / ab_ab_ac / para_negs
python validator/build_sets.py                                                         # sets/
python validator/calibrate.py --view full  --out validator/calibrated                  # 셀 복사 + weights_full.json
python validator/calibrate.py --view short --out validator/calibrated
```

## 5. 이전 결과 (2026-10-31): 어떤 세트도 리더보드 순서를 재현하지 못함

| set | starter | marked_v1 | minimal | baseline_v2 int8 | centrality byte | span |
|---|---|---|---|---|---|---|
| af_full | 0.352 | 0.883 | 0.407 | 0.635 | 0.996 | 0.985 |
| claude_short | 0.409 | 0.803 | 0.427 | 0.500 | 0.998 | 0.970 |
| claude_ab_ac | 0.496 | 0.320 | 0.501 | 0.612 | 0.348 | 0.345 |
| qwen_full | 0.408 | 0.748 | 0.389 | 0.488 | 0.994 | 0.972 |
| qwen_short | 0.410 | 0.855 | 0.560 | 0.561 | 0.996 | 0.991 |
| qwen_share3_full | 0.541 | 0.357 | 0.545 | 0.651 | 0.219 | 0.227 |
| qwen_share3_short | 0.520 | 0.443 | 0.709 | 0.733 | 0.211 | 0.153 |
| para_negs_full | 0.435 | 0.565 | 0.313 | 0.716 | 0.501 | 0.214 |
| para_negs_short | 0.438 | 0.510 | 0.552 | 0.680 | 0.580 | 0.333 |
| para_same_full | 0.412 | 0.736 | 0.406 | 0.496 | 0.974 | 0.940 |
| para_same_short | 0.462 | 0.831 | 0.539 | 0.577 | 0.981 | 0.962 |
| para_share3_full | 0.531 | 0.311 | 0.570 | 0.599 | 0.242 | 0.221 |
| para_share3_short | 0.532 | 0.362 | 0.669 | 0.722 | 0.245 | 0.155 |

- 한 곳 수정 세트(af, claude_short, qwen, para_same): 거리 방식 0.97~1.0, marked_v1 0.74~0.88.
- 공유 수정 세트(share3, ab_ac): 거리 방식 0.22~0.35, marked_v1 0.31~0.44.
- para_negs에서는 baseline_v2(0.72)가 정답만 Train 원문 문체인 점을 알아봐서 이긴다.
- `mix_order.py`: 한 곳 수정 셀과 공유 셀을 섞으면 리더보드 순서가 para_full p=0.57 한 점에서만 맞는다. 그때도 starter가 0.47~0.50이라 리더보드(0.35)와 멀다.
