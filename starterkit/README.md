# Inthon baseline starter kit

원본 Train만 사용하는 학습·검증·제출 예제입니다. Python 3.11과 허용된
`facebook/dinov2-base`, `sentence-transformers/all-mpnet-base-v2`를 사용합니다.

## 방법론

1. 에피소드의 모든 이미지를 각각 DINOv2로 인코딩합니다. 프레임별 CLS 특징을
   **mean pooling**하여 768차원 이미지 시퀀스 특징을 얻습니다.
2. 원본 설명을 MPNet으로 인코딩하고 padding을 제외한 token mean pooling으로
   768차원 텍스트 특징을 얻습니다.
3. 두 특징을 **concat**하고 `Linear(1536, 256) → ReLU → Dropout → Linear(256, 1)`을
   적용합니다. Sigmoid 출력은 각 영상–설명 쌍에 대한 0–1 범위의 스칼라입니다.
4. 매 epoch에 각 학습 에피소드의 정답 설명 한 개(label 1)와, **다른 학습 에피소드에서
   무작위로 고른 원문 설명 한 개**(label 0)를 사용합니다. 자기 설명과 동일한 설명은
   negative에서 제외합니다. 이미지·텍스트 생성이나 새로운 에피소드 파일 작성은 하지 않습니다.
5. 두 인코더는 동결하고 MLP만 학습합니다. 수치 안정성을 위해 학습에서는 sigmoid 전
   logits에 `BCEWithLogitsLoss`를 적용하고, 추론에서는 sigmoid 점수를 반환합니다.

이미지는 숫자 순서로 읽고, 고정 resize/center crop/정규화만 적용합니다.
랜덤 이미지 변환, 프레임 축소, 설명 변형, teacher 예측은 사용하지 않습니다.
mean pooling은 순서에 불변하므로 이 baseline 자체에는 시간 순서 추론 모듈이 없습니다.
Sigmoid 값은 학습한 matching 점수이며 확률 보정이 보장되는 값은 아닙니다.

## 설치와 데이터

아래 명령은 `starterkit/`에서 실행합니다. T4에서는 CUDA를 지원하는 PyTorch 환경을
사용하세요. 이미 PyTorch가 설치된 Colab에서는 기존 CUDA 환경과의 호환성을 먼저 확인합니다.

```bash
python -m pip install -r requirements.txt
python download_models.py
```

이미 로컬에 **동일한 허용 모델**이 있으면 다운로드 대신 복사할 수 있습니다.

```bash
python download_models.py --source-root /path/to/model_snapshots
```

이 경로 아래에 `dinov2/`, `mpnet/`의 config, safetensors, processor/tokenizer 파일이
있어야 합니다. 기본 설정은 `config.json`이며 경로는 이 설정 파일의 위치를 기준으로 해석합니다.
CLI로 전달하는 경로는 현재 작업 디렉토리 기준입니다.

데이터 형식은 다음과 같습니다. `description.json`이나 오답 annotation은 요구하지 않습니다.

```text
data/Train/
  episode_1/
    description.txt
    images/1.jpg, 2.jpg, ..., 10.jpg, ...
  episode_2/
    ...
```

## 학습과 로컬 평가

```bash
python train.py --device auto
python evaluate.py --checkpoint outputs/baseline/best.pt
```

- 기본적으로 전체 Train을 에피소드 단위 80:20으로 분할합니다. seed는 2026입니다.
- `outputs/baseline/splits.json`에 학습·검증 ID를 기록합니다. 프레임 단위로 나누지 않습니다.
- negative는 각 split 내부에서만 뽑습니다. 학습 negative는 epoch마다 다시 뽑고,
  검증 negative는 고정합니다. epoch당 학습 에피소드 수가 N이면 학습 쌍은 정확히 2N개입니다.
- 동결한 인코더 특징을 `features.pt`에 저장하고 MLP를 학습합니다.
  데이터·인코더 파일이 바뀌면 기존 캐시 재사용을 거부합니다.
- 검증 BCE가 가장 낮은 checkpoint를 `best.pt`에 저장합니다. 기존 실험을 덮어쓰지 않으므로
  재실행할 때는 `--output-dir outputs/다른이름`을 지정합니다.
- GPU가 없으면 `--device cpu`로 실행할 수 있습니다. 전체 특징 추출에는 시간이 걸립니다.

먼저 짧게 동작만 확인하려면 다음 명령을 사용합니다.

```bash
python train.py --max-episodes 12 --epochs 3 --output-dir outputs/smoke
python evaluate.py --checkpoint outputs/smoke/best.pt
```

출력 지표는 두 쌍에 대한 `binary_accuracy`, 정답 설명 점수가 random negative보다 높은 비율인
`pair_accuracy`, 그리고 BCE입니다. 동점은 pair 정답으로 세지 않습니다.
이 지표는 **리더보드 객관식 정확도가 아닙니다**. 다른 영상의 설명도 일부 내용이 맞을 수 있고,
random negative는 사건 순서만 바꾼 어려운 오답을 대표하지 못합니다.
학습·테스트의 출처도 다르므로 이 검증 성능만으로 일반화 성능을 보장하지 않습니다.

## 제출물 생성

```bash
python export.py --checkpoint outputs/baseline/best.pt
python validate.py --submission-dir sample_submission --device auto
python package.py
```

export 대상은 기본적으로 `sample_submission/`입니다. 기존 결과가 있으면 새
`--output-dir`을 사용하세요. package의 기존 ZIP 교체는 `--force`로 명시합니다.

```text
sample_submission/
  submission.py
  submission.json              # {"api_version": 2}
  baseline_model.py
  inference_config.json
  training_info.json
  weights/mlp.pt
  models/dinov2/                # 인코더 가중치 + 이미지 전처리기
  models/mpnet/                 # 인코더 가중치 + 토크나이저
```

ZIP의 최상위에 `submission.py`가 있어야 합니다. ZIP 안에 `sample_submission/`이라는
상위 폴더를 한 번 더 넣지 않습니다. 학습 데이터·특징 캐시·학습 스크립트는 추론 ZIP에
들어가지 않으며, 재현 검수용으로는 이 starter kit와 학습 설정·로직을 별도 보관합니다.

동결 인코더는 제출 용량을 줄이기 위해 float16으로 저장합니다. CUDA 추론은 float16,
CPU 추론은 float32로 실행하며, 저장 정밀도 변환에 따른 작은 수치 차이가 생길 수 있습니다.

서버는 아래 클래스 인터페이스를 지원합니다.

```python
class Submission:
    def __init__(self, artifact_dir: str, device: str):
        ...  # 로컬 모델과 MLP를 한 번만 로딩

    def predict(self, images, choices) -> list[str]:
        ...  # 점수가 높은 순서의 모든 선지 ID
```

`images`는 시간순 RGB PIL 이미지 리스트, `choices`는 `{"id": str, "text": str}`의
리스트입니다. 선지 수나 ID를 A–E로 고정하지 않습니다. `Submission.score()`는 선지별
0–1 점수를 계산하고, `predict()`는 이를 정렬한 ID 리스트를 반환합니다. 평가 에피소드 간
파라미터·통계·검색 메모리를 갱신하지 않으며 인터넷이나 외부 캐시에 의존하지 않습니다.

운영진 저장소에서는 실제 서버 코드로 ZIP 계약과 추론을 함께 확인할 수 있습니다.

```bash
python validate.py --zip sample_submission.zip --server-dir ../server --device cpu
```

이 검사는 원본 Train에서 가져온 입력을 임시 디렉토리에 준비하고, 빈 HF 캐시와 별도 CWD에서
서버 `runner/execute.py`를 실행합니다. Test 데이터는 읽지 않습니다.
현재 서버의 런타임·업로드 제한과 모델 제출을 위한 설정은
[SERVER_COMPATIBILITY.md](SERVER_COMPATIBILITY.md)에 정리했습니다.

## 수정 위치

| 파일 | 참가자 수정 지점 |
| --- | --- |
| `config.json` | 배치 크기, 학습률, MLP 크기, epoch 등 |
| `baseline_model.py` | 이미지 집계, 텍스트 인코딩, concat MLP 구조 |
| `dataset.py` | 데이터 로딩과 negative sampling |
| `train.py` | 학습 로직과 손실 |
| `submission.py` | 선지별 점수와 순위 반환 |

모델·전처리를 바꾸면 export와 추론에도 같은 변경이 반영되어야 합니다.
동결을 해제한 인코더를 학습하려면 특징 캐시 방식도 함께 바꾸어야 합니다.
규정은 운영진의 `documents/rules.md`가 우선합니다.

## 동봉된 샘플 가중치

이 작업에서 생성한 `sample_submission/`은 **원본 Train 12개 중 학습 10개·검증 2개,
3 epoch 실행으로 만든 기능 확인용 가중치**입니다. 전체 Train 학습 성능을 나타내지 않습니다.
실제 사용하려면 전체 Train으로 학습한 뒤 새로운 디렉토리로 export하고 제출하세요.

학습 결과와 provenance는 `outputs/smoke/`, 제출물의 학습 정보는
`sample_submission/training_info.json`에 있습니다.
