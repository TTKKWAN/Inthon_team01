# baseline_v2: 규칙 기반 hard negative + 공통 검증 세트

스타터킷 모델을 **한 줄도 안 바꾸고** 학습 오답만 규칙 기반 hard negative로 바꾸면, 같은 검증 세트에서 0.467 → **0.728**이 된다(`minimal_submission`).
모델까지 바꾼 v2(SigLIP + 절 단위 순서 인식 scorer)는 **0.861**이다.

점수는 대회식이다: 정답 1위 1점, 2위 0.5, 3위 0.25, 4위 0.125, 5위 0. 무작위로 찍으면 0.375.

## 결과 요약

같은 626개 문제를 각 제출물의 `Submission.predict`로 풀었다(`eval_submission.py`, `results/eval_submissions.json`).

| 제출물 | 모델 | 학습 오답 | full | short | 평균 | full 1위 비율 |
| --- | --- | --- | --- | --- | --- | --- |
| 스타터킷 baseline (전체 Train 학습) | 프레임 평균 DINOv2 + MPNet → MLP | random 1개, BCE | 0.455 | 0.480 | 0.467 | 18.2% |
| **minimal_submission** | 스타터킷과 동일 | hard 8 + random 3, listwise CE | 0.654 | 0.803 | **0.728** | 40.3% |
| v2_submission | DINOv2 + SigLIP + MPNet, 순서 인식 scorer | 위와 같음 + 시간 역순 영상 | 0.844 | 0.879 | 0.861 | 71.4% |

- 스타터킷 baseline은 자기 검증(정답 vs random 오답 1개)에서 pair accuracy 95%가 나오지만, 같은 장면을 조금 바꾼 오답 앞에서는 찍기보다 조금 나은 수준이다.
- **리더보드 점수는 아직 없다.** 검증 오답이 학습 오답과 같은 생성기에서 나오므로 실제 테스트 점수는 이보다 낮을 수 있다.

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
