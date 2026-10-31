# baseline_v2: 규칙 기반 hard negative + 공통 검증 세트

스타터킷 모델을 **한 줄도 안 바꾸고** 학습 오답만 규칙 기반 hard negative로 바꾸면, 같은 검증 세트에서 0.467 → **0.728**이 된다(`minimal_submission`).
모델까지 바꾼 v2(SigLIP + 절 단위 순서 인식 scorer)는 **0.861**이다.

**현재 최고 리더보드: v2_matcher = 0.4485** (minimal 0.4260, 찍기 0.375). 아래 "리더보드로 본 테스트 세트"와 "v2_matcher 재현"부터 보면 된다.

## 리더보드로 본 테스트 세트

| 제출물 | 리더보드 (public) | 찍기 대비 | 자체 검증 | 엉뚱한 영상 검증 | 영상 몫 |
| --- | --- | --- | --- | --- | --- |
| minimal_submission | 0.4260 | +0.051 | 0.728 | 0.584 | 0.144 |
| v2_matcher_submission | **0.4485** | **+0.074** | 0.853 | 0.598 | 0.255 |

검증 숫자는 예전 오답 묶음 기준 full/short 평균이다(`results/eval_runs_matcher_oldbank.json`). public은 약 573문제라 표준오차가 약 ±0.0125이고, 두 제출의 차이 +0.0225는 그 약 1.8배다.

두 점만으로 추측할 수 있는 것:

1. **검증에서 얻은 이득 중 약 15%만 테스트로 넘어간다.** "리더보드 − 찍기"를 "검증 − 찍기"로 나누면 두 모델 모두 0.14~0.15다. 우리 검증은 테스트 난이도를 6~7배 과대평가한다.
2. **넘어가는 몫은 영상 몫(진짜 영상 − 엉뚱한 영상)을 따라간다.** "리더보드 − 찍기"를 영상 몫으로 나누면 0.35 / 0.29다. 텍스트 꼼수로 얻은 점수는 테스트에서 거의 사라지고, 영상을 보고 얻은 점수의 약 1/3이 남는다.
3. **테스트 오답에는 우리 생성기의 흔적이 없다.** 규칙 기반 오답에서만 통하는 문장 흔적(비문법, 템플릿, 어색한 단어 조합)으로는 점수를 못 얻는다.
4. **테스트는 우리 오답과 다른 방식, 또는 더 미세한 차이일 가능성이 크다.** 두 모델 모두 검증에서 1위 비율이 40~70%인데, 리더보드 0.43~0.45는 1위 비율이 대략 20~25%라는 뜻이다(나머지가 대부분 2~3위라고 가정).
5. **영상 쪽 개선은 실제로 효과가 있다.** v2_matcher는 영상 이해(절 시간 정렬, 동작 라벨) 쪽 개선이고, 리더보드도 minimal보다 올랐다. 방향은 맞지만 이득이 크게 줄어든다.

그래서 앞으로는 **검증 점수보다 "엉뚱한 영상 점수가 낮고 영상 몫이 큰" 모델**을 고르고, 오답은 문장 흔적이 적게 만들자. `llm_negatives.py`(로컬 LLM 오답)와 `llm_paraphrase.py`(정답 바꿔 쓰기)가 그 시도이고, 아직 학습에는 반영하지 않았다.

## v2_matcher 재현

v2_matcher = v2 scorer + debias + **학습형 클립 매처(out-of-fold, 추가 방식)** + 버그 수정 오답 묶음.

| 구성 | 내용 | 파일 |
| --- | --- | --- |
| 오답 묶음 | `negatives.py` 규칙 기반 9종. `left`(동사 leave) 교체 버그와 `lower`(동사)→`upper` 버그를 고친 버전 | `data/negatives.json` |
| 클립 매처 | 프레임 특징 + 앞 프레임과의 차이 → 1D conv로 짧은 클립 벡터, 절과 코사인 유사도. 텍스트로는 못 푸는 3가지로 학습: 절 → 자기 영상 찾기, 절 시간 순서, `episode.json` action 라벨 63종(`manifest.json`의 frame_id로 이미지에 연결) | `model_v2.py` `ClipMatcher`, `train_align.py` |
| out-of-fold | 학습 영상을 3조각으로 나눠 "그 조각을 안 본 매처" 3개 + 전체 매처 1개. scorer 학습 때는 각 영상에 그 영상을 안 본 매처 신호를, 추론에는 전체 매처를 쓴다. 매처가 학습 영상에서만 잘 맞아(순서 93% vs 검증 37%) scorer가 과신하던 문제를 막는다 | `matcher/folds.json` |
| scorer 연결 | 기존 SigLIP 정렬 신호 3개 옆에 매처 신호 3개를 **추가**(교체하면 0.845 → 0.77~0.80으로 하락). 매처는 scorer 학습 중 고정 | `train_v2.py --set matcher_dir=...` |
| debias | 학습 때만 텍스트 전용 보조 모델(product-of-experts) | `--set debias_weight=1.0` |

```bash
cd baseline_v2
python negatives.py --data-dir ../data/Train --out data/negatives.json        # 커밋된 파일과 같은 내용
python prepare_features.py --part all                                         # frames.pt, texts.pt
python train_align.py --output-dir outputs/align_pre                          # 전체 매처 -> matcher.pt
python train_align.py --output-dir outputs/align_pre --folds 3 --exclude-fold 0
python train_align.py --output-dir outputs/align_pre --folds 3 --exclude-fold 1
python train_align.py --output-dir outputs/align_pre --folds 3 --exclude-fold 2
python train_v2.py --output-dir outputs/matcher_oof --set debias_weight=1.0 matcher_dir=outputs/align_pre
python export_v2.py --checkpoint outputs/matcher_oof/best.pt --output-dir outputs/v2_matcher_submission
python check_deps.py outputs/v2_matcher_submission.zip                        # 허용 패키지만 쓰는지
python eval_submission.py outputs/v2_matcher_submission.zip                   # ZIP 실측 채점
```

- 매처 학습은 각 40 epoch, 5070 Ti에서 1~2분, scorer는 약 1분이다.
- `matcher/folds.json`은 `train_align.py --folds 3`이 seed로 다시 만드는 것과 같다(확인용으로 커밋).
- ZIP 743.8 MiB, 추론 1,910개 예상 약 6분(5070 Ti), 허용 패키지만 사용.
- 결과 기록: `results/history_matcher_*.json`(oof / frozen / tuned), `results/history_align_pre_full.json`(매처 사전학습), `results/leaderboard.json`.

### 큰 ZIP 옮기기 (`make_bundle.py`, `rebuild_zip.py`)

ZIP의 약 700MB는 공개 인코더다. `make_bundle.py`가 그 부분을 빼고(텐서별 SHA-256과 HF 커밋을 기록) 10~40MB 묶음을 만들고, 받는 쪽에서 `rebuild_zip.py`가 Hugging Face에서 받아 fp16으로 바꾼 뒤 모든 텐서를 대조해서 같은 ZIP을 만든다. torch 없이 `numpy safetensors huggingface_hub`만 있으면 된다.

점수는 대회식이다: 정답 1위 1점, 2위 0.5, 3위 0.25, 4위 0.125, 5위 0. 무작위로 찍으면 0.375.

## 결과 요약

같은 626개 문제를 각 제출물의 `Submission.predict`로 풀었다(`eval_submission.py`, `results/eval_submissions.json`).

| 제출물 | 모델 | 학습 오답 | full | short | 평균 | full 1위 비율 |
| --- | --- | --- | --- | --- | --- | --- |
| 스타터킷 baseline (전체 Train 학습) | 프레임 평균 DINOv2 + MPNet → MLP | random 1개, BCE | 0.455 | 0.480 | 0.467 | 18.2% |
| **minimal_submission** | 스타터킷과 동일 | hard 8 + random 3, listwise CE | 0.654 | 0.803 | **0.728** | 40.3% |
| v2_submission | DINOv2 + SigLIP + MPNet, 순서 인식 scorer | 위와 같음 + 시간 역순 영상 | 0.844 | 0.879 | 0.861 | 71.4% |

- 스타터킷 baseline은 자기 검증(정답 vs random 오답 1개)에서 pair accuracy 95%가 나오지만, 같은 장면을 조금 바꾼 오답 앞에서는 찍기보다 조금 나은 수준이다.
- **리더보드: minimal_submission = 0.4260** (public, 추론 3분 5초), **v2_matcher = 0.4485**. 자체 검증보다 훨씬 낮다. 원인은 아래 "텍스트 꼼수".

## 텍스트 꼼수: 검증은 높은데 리더보드가 낮은 이유

같은 검증 문제에 **엉뚱한 에피소드의 영상**을 넣어도 점수가 찍기(0.375)로 떨어지지 않는다. 모델이 영상보다 "규칙으로 손댄 문장인지"를 보고 고르고 있다는 뜻이다. 테스트 오답에는 우리 생성기의 흔적이 없으니 이 몫이 사라진다.

| 모델 | full (진짜 영상) | full (엉뚱한 영상) | 영상 몫 | short (진짜) | short (엉뚱) | 영상 몫 |
| --- | --- | --- | --- | --- | --- | --- |
| minimal | 0.653 | 0.525 | +0.13 | 0.803 | 0.643 | +0.16 |
| v2 (grd_base) | 0.844 | 0.630 | +0.21 | 0.879 | 0.708 | +0.17 |
| v2 + debias (grd_debias) | 0.830 | **0.500** | **+0.33** | 0.869 | **0.531** | **+0.34** |
| v2 + 텍스트→영상 loss (grd_t2v) | 0.834 | 0.472 | +0.36 | 0.880 | 0.711 | +0.17 |
| v2 + 둘 다 (grd_both) | 0.825 | 0.473 | +0.35 | 0.876 | 0.604 | +0.27 |
| v3 bank + debias | 0.829 | 0.496 | +0.33 | 0.859 | 0.526 | +0.33 |

- `train_v2.py`가 매 epoch `full_shuf_lb`, `short_shuf_lb`(엉뚱한 영상 점수)와 `*_gap`(영상 몫)을 같이 기록한다. **모델은 영상 몫을 보고 고르자.**
- **debias** (`--set debias_weight=1.0`): 학습 때만 텍스트 전용 보조 모델을 붙여(product-of-experts) 문장만으로 풀리는 몫을 가져가게 한다. 추론에는 쓰지 않는다.
- **텍스트→영상 loss** (`--set t2v_weight=1.0`): 설명 하나에 영상 8개를 주고 맞는 영상을 고르게 한다.
- **v3 bank** (`negatives.py --variants`): 모든 문장을 각각 짧은 정답으로 쓰고, 뜻이 같은 rewrite(진행형, 동의어, 주어, 접속어, 문장 합치기)를 정답 변형과 오답 절반에 적용한다. 이 검증에서는 개선이 없었다(검증 오답이 v1 생성기 그대로라 문체 강건성은 드러나지 않음).
- 원본: `results/eval_runs_grounding.json`, `results/history_grd_*.json`, `results/history_v3_*.json`

## 허용 패키지 확인 (`check_deps.py`)

제출 ZIP을 새 프로세스에서 로딩·추론하고, 실제로 불려 온 패키지를 채점 환경 목록과 비교한다.

```bash
python check_deps.py outputs/minimal_submission.zip outputs/v2_debias_submission.zip
```

- **v2의 SigLIP 토크나이저(`SiglipTokenizer`)는 `sentencepiece`가 필요한데, 채점 환경에 없다.** `model_v2.py`의 `load_siglip_tokenizer`로 SigLIP `tokenizer.json`(fast 토크나이저)을 쓰도록 바꿨다. Train 절 8,740개에서 토큰이 같고 텍스트 특징 차이가 0이라 재학습은 필요 없다.
- 주의: fast 토크나이저는 기본으로 `attention_mask`도 넘기는데, SigLIP은 mask 없이 학습돼서 넣으면 결과가 달라진다(검증 0.849 → 0.461). `model_input_names=["input_ids"]`로 막아 두었다.
- 이 수정 전에 export한 v2 ZIP은 서버에서 로딩 오류가 날 수 있으니 다시 export해서 쓴다.

## minimal_submission이 스타터킷과 다른 점

제출 ZIP을 파일 단위로 비교하면 **`weights/mlp.pt`만 다르다.**

- `submission.py`, `baseline_model.py`, DINOv2·MPNet `model.safetensors`, 토크나이저: 바이트 단위로 동일하다.
- 설정 JSON: 내용이 같다. `models/*/config.json`의 `transformers_version` 표기만 4.57.6이다.
- 학습만 다르다: 정답 1개 + `negatives.py`의 hard negative 8개 + 다른 에피소드 정답 3개를 한 묶음으로 넣고 listwise cross-entropy를 쓴다. 40 epoch 중 검증 점수가 가장 높은 38 epoch를 썼다.

## 폴더 구성

```text
baseline_v2/
├── negatives.py          # 규칙 기반 오답 생성기 (seed 2026)
├── data/negatives.json   # negatives.py로 생성한 오답 묶음 (4,170 에피소드, 86,713개)
├── validation/           # 공통 검증 세트
│   ├── splits.json       #   train 3,544 / valid 626 에피소드 ID
│   ├── val_full.json     #   valid 626개 × (정답 1 + 서로 다른 유형 오답 4), 정답 = 전체 설명
│   └── val_short.json    #   같은 에피소드, 정답 = 첫 문장
├── prepare_features.py   # 동결 인코더 특징 캐시 (frames.pt, texts.pt)
├── ablation_mlp.py       # 스타터킷 MLP를 캐시 위에서 학습 (--mode random | hard)
├── train_v2.py, model_v2.py, submission.py, export_v2.py   # v2 모델
├── eval_runs.py          # 학습한 run들을 같은 검증 세트로 채점 (캐시 사용)
├── eval_submission.py    # 제출 ZIP을 실제 추론 경로로 채점
├── check_deps.py         # 제출 ZIP이 채점 환경에 있는 패키지만 쓰는지 확인
├── train_align.py        # 학습형 클립 매처 사전학습 (--folds/--exclude-fold로 out-of-fold)
├── matcher/folds.json    # out-of-fold 조각 배정
├── make_bundle.py, rebuild_zip.py   # 큰 제출 ZIP을 작은 묶음으로 옮기기
├── llm_negatives.py      # (실험 중) 로컬 LLM Qwen3-4B로 자연스러운 오답 생성
├── llm_paraphrase.py     # (실험 중) 정답 바꿔 쓰기 + 그 문장에서 오답 생성
├── export_val_set.py     # validation/ 파일을 다시 만드는 스크립트
├── download_siglip.py
├── results/              # 아래 "결과 파일"
└── t4_timing/            # T4 추론 시간 측정 (Colab 노트북 포함)
```

## minimal_submission 재현

저장소 루트에 `data/Train/`이 있다고 가정한다(루트 README의 다운로드 방법).

```bash
cd starterkit && python download_models.py && cd ..        # DINOv2, MPNet
cd baseline_v2
python download_siglip.py                                    # 특징 캐시가 SigLIP도 함께 만든다
python negatives.py --data-dir ../data/Train --out data/negatives.json   # 이미 있는 파일과 동일하게 재생성됨
python prepare_features.py --part all                        # cache/frames.pt, cache/texts.pt (5070 Ti 기준 약 5분)
python ablation_mlp.py --mode hard --save outputs/minimal_hardneg/best.pt
cd ../starterkit
python export.py --checkpoint ../baseline_v2/outputs/minimal_hardneg/best.pt --output-dir ../baseline_v2/outputs/minimal_submission
python validate.py --submission-dir ../baseline_v2/outputs/minimal_submission --device auto
python package.py --submission-dir ../baseline_v2/outputs/minimal_submission --output ../baseline_v2/outputs/minimal_submission.zip
```

- minimal은 캐시 특징 중 DINOv2 CLS(앞 768차원)와 MPNet 전체 설명(앞 768차원)만 쓴다. 스타터킷 추론과 같은 특징이다.
- 원래 실행에서는 스타터킷 `sample_submission/models`의 fp16 인코더로 특징을 뽑았다. `download_models.py`의 원본(fp32)을 쓰면 수치가 아주 조금 다를 수 있다.
- 비교용 스타터킷 방식: `python ablation_mlp.py --mode random` → 0.482(캐시 특징 기준). 실제 `train.py`로 학습해도 0.467로 비슷하다.

## 오답 생성 (`negatives.py`)

정답 설명을 규칙으로 살짝 바꾼다. 생성형 모델이나 사람·AI가 쓴 문장은 쓰지 않는다. 시드가 고정이라 항상 같은 결과가 나온다.

| 유형 | 규칙 | 개수 |
| --- | --- | --- |
| antonym | 동사 하나를 반대말로 (lifts → lowers, buttons → unbuttons) | 12,476 |
| spatial | 위치어 하나를 반대로 (left ↔ right, in front of ↔ behind) | 7,048 |
| attribute | 색·개수 단어 하나를 다른 값으로 | 7,324 |
| object | 물체 명사 하나를 Train에서 찾은 다른 물체로 (단·복수 유지) | 16,448 |
| order | 사건 절 두 개를 맞바꾸거나 전체를 뒤집음 (Finally 같은 표지는 다시 붙임) | 9,189 |
| append_unrelated | 정답 뒤에 다른 에피소드의 문장을 붙임 | 8,340 |
| append_plausible | 정답 뒤에 이 장면 물체로 만든 동작을 붙임 | 16,680 |
| replace_clause | 절 하나를 다른 에피소드의 절로 교체 | 5,051 |
| drop_corrupt | 절 하나를 빼고 다른 곳을 틀리게 바꿈 (더 짧은 오답) | 4,157 |

- 정답이 2문장 이상이면 **첫 문장만으로 된 짧은 정답 그룹**도 만들고, 그 문장에서 오답을 따로 만든다(짧은 테스트 설명 대비).
- 코드에 직접 쓴 것은 일반 영어 단어 표뿐이다: 반대 동사·위치어 쌍, 색·숫자 목록, 덧붙일 동사 목록. 물체 어휘(238개)는 Train 설명에서 뽑는다.
- **팀 검토 필요**: `append_plausible`은 `"Then they {verb} the {obj}."` 같은 빈칸 템플릿에 단어를 채워 문장을 만든다. 완성 문장을 하드코딩한 것은 아니지만, Rule.md 제3조 8항 해석이 애매하면 이 유형을 빼거나 다른 에피소드 절을 붙이는 방식으로 바꾸자. leave-one-out 결과로는 이 유형을 빼면 해당 통과율이 100% → 89%로 떨어진다.

## 검증 세트 (`validation/`)

- **에피소드**: Train 4,170개를 seed 2026으로 섞어 15%(626개)를 검증으로 뗀다. 학습에는 전혀 쓰지 않는다. 스타터킷의 80:20 분할과 같은 순서로 섞기 때문에 이 626개는 스타터킷 검증 834개 안에 모두 들어 있다(스타터킷 학습과도 겹치지 않음).
- **문제**: 에피소드마다 정답 1개 + 서로 다른 유형의 생성 오답 4개(유형이 4개보다 적으면 다른 에피소드 정답으로 채움). `val_full.json`은 전체 설명, `val_short.json`은 첫 문장이 정답이다. 파일 안의 `choices[0]`이 항상 정답이니 모델에 줄 때는 섞어야 한다.
- **해석 주의**: 오답 생성기가 학습과 같아서 실제 테스트보다 높게 나올 가능성이 크다. 실험끼리 비교하는 용도로 쓴다.

## 결과 파일 (`results/`)

| 파일 | 내용 |
| --- | --- |
| `eval_submissions.json` | 위 요약표 원본: 제출물별 full/short 점수와 정답 순위 분포 |
| `eval_runs.json` | v2 변형 4개(base, frame-drop, 4배 오답 묶음, 둘 다)를 같은 검증으로 채점. 모두 0.85~0.86으로 차이 없음 |
| `history_exp_*.json` | 위 run들의 epoch별 학습 loss와 검증 지표 |
| `loto_results.json` | leave-one-type-out: 유형 하나를 학습에서 빼고 그 유형 통과율 측정 |
| `minimal_training_info.json` | minimal_submission의 `training_info.json` |
| `eval_runs_grounding.json` | debias / 텍스트→영상 / v3 run들의 진짜 영상·엉뚱한 영상 점수 |
| `history_grd_*.json`, `history_v3_*.json` | 위 run들의 epoch별 기록 |

leave-one-type-out 요약(처음 보는 유형에 대한 일반화):

| 학습에서 뺀 유형 | 학습했을 때 | 처음 볼 때 |
| --- | --- | --- |
| drop_corrupt | 91.3% | 64.1% |
| antonym | 88.4% | 67.9% |
| append_unrelated | 96.6% | 80.1% |
| append_plausible | 100% | 89.4% |
| spatial | 78.9% | 69.5% |
| order | 95.7% | 88.6% |
| attribute | 86.0% | 82.0% |
| object | 91.1% | 88.5% |
| replace_clause | 95.1% | 94.1% |

오답 개수를 4배로 늘려도 점수가 오르지 않았고, 유형을 빼면 크게 떨어진다. 오답의 **개수보다 유형과 표현의 다양성**이 중요하다.

## 추론 시간 (`t4_timing/`)

RTX 5070 Ti에서 Train 60개 에피소드(평균 15.4프레임)를 이미지 로딩 포함으로 측정했다.

| ZIP | 에피소드당 | 1,910개 예상 |
| --- | --- | --- |
| minimal (349 MiB) | 0.115 s | 3.7분 |
| v2 (714 MiB) | 0.184 s | 5.9분 |

T4는 3~5배 느릴 수 있어서 v2는 Colab T4로 실측이 필요하다. `t4_timing_colab.ipynb`를 쓴다(샘플 ZIP과 제출 ZIP은 용량 때문에 git에 없다).
