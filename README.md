# Inthon team01

Inthon 데이터톤 **Multimodal Reasoning for Image Sequences** 팀 저장소.
연속된 이미지 시퀀스를 보고, description 선지 5개(정답 1개, 오답 4개)의 순위를 매기는 과제다(예: `B>D>E>C>A`).

> **규정은 [Rule.md](Rule.md)를 먼저 읽는다.** 발표자료와 운영진 공지를 합친 최종본이다.
> 대회 발표자료 원본은 [데이터톤.pdf](데이터톤.pdf)에 있다.

## 저장소 구성

```text
.
├── README.md
├── Rule.md                 # 최종 규정 (원 규정 + 발표자료 + 운영진 공지)
├── CLAUDE.md               # Claude Code 작업 규칙 요약
├── 데이터톤.pdf             # 대회 발표자료
├── .claude/                # Claude Code 규정 위반 방지 훅 (rule_guard.py)
├── scripts/
│   └── download_data.py    # Hugging Face에서 Train data 다운로드
├── starterkit/             # 운영진 baseline (DINOv2 + MPNet + MLP)
└── data/Train/             # ← 다운로드 후 생기는 위치 (git에 올리지 않음)
```

## 1. 데이터셋 다운로드

데이터셋은 git에 넣지 않고 Hugging Face에 있다: [TTaekwan/Inthon_team01_public](https://huggingface.co/datasets/TTaekwan/Inthon_team01_public)

| 파일 | 내용 |
| --- | --- |
| `Train.tar` (약 3.96GB) | `Train/` 폴더 전체: 4,170 episodes, 이미지 55,726장 |
| `Train.tar.sha256` | 무결성 확인용 SHA256 |

### 방법 A: 스크립트 (권장)

저장소 루트에서 실행한다. SHA256을 확인한 뒤 `data/Train/`에 압축을 푼다. starterkit의 기본 경로(`config.json`의 `"data_dir": "../data/Train"`)와 같다.

```bash
git clone https://github.com/TTKKWAN/Inthon_team01.git
cd Inthon_team01
pip install huggingface_hub
python scripts/download_data.py
# 다른 위치에 받으려면: python scripts/download_data.py --out-dir /content/data
```

### 방법 B: Python 코드로 직접

```python
import tarfile
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="TTaekwan/Inthon_team01_public",
    filename="Train.tar",
    repo_type="dataset",
)
with tarfile.open(path) as tar:
    tar.extractall("data")   # -> data/Train/episode_N/...
```

### 방법 C: CLI

```bash
pip install huggingface_hub
hf download TTaekwan/Inthon_team01_public Train.tar --repo-type dataset --local-dir .
mkdir -p data && tar -xf Train.tar -C data && rm Train.tar
```

저장소는 public이라 토큰 없이 받을 수 있다. 업로드할 때만 쓰기 권한 토큰이 필요하다. **토큰을 코드나 커밋에 넣지 않는다.** `export HF_TOKEN=...`처럼 환경변수로 쓴다.

### 데이터 형식

```text
data/Train/
  episode_1/
    description.txt        # 정답 description 1개 (오답 선지는 없음)
    episode.json           # 원본 메타데이터
    images/1.jpg, 2.jpg, ...   # 숫자 순서 = 시간 순서 (8~20장)
  ...
  episode_id_map.csv       # 에피소드 번호 ↔ 원본 split / 영상 ID
  manifest.json, README.md, *_report.json
```

Train에는 정답 description만 있다. 오답(hard negative)은 증강 코드로 직접 만들어야 하고, 그 방법은 [Rule.md](Rule.md) 제3조 제8항(hand-crafted description 금지)을 따른다.

## 2. Baseline 실행 (starterkit)

자세한 내용은 [starterkit/README.md](starterkit/README.md)에 있다.

```bash
cd starterkit
pip install -r requirements.txt
python download_models.py              # 허용 모델(dinov2-base, all-mpnet-base-v2)을 models/에 받음
python train.py --device auto
python evaluate.py --checkpoint outputs/baseline/best.pt
python export.py --checkpoint outputs/baseline/best.pt --output-dir sample_submission_new
python validate.py --submission-dir sample_submission_new --device auto
python package.py                       # 제출용 zip 생성
```

모델 가중치(`models/`, `weights/`)와 zip 파일은 용량 때문에 git에 넣지 않았다. 위 명령으로 각자 다시 만든다.

## 3. 꼭 지킬 규정 요약

전문은 [Rule.md](Rule.md).

- **사전학습 모델은 4개만 허용**: `facebook/dinov2-base`, `sentence-transformers/all-mpnet-base-v2`, `google/siglip-base-patch16-224`, `HuggingFaceTB/SmolVLM-256M-Instruct`. 다른 pretrained 가중치(ImageNet weights, `pretrained=True` 등)를 학습·추론에 쓰면 그 제출은 오류 처리된다.
- **제출 zip은 1GB 이하**, 전체 test 추론은 **15분 이내**. 채점 환경은 인터넷이 없다. 패키지 버전은 고정이다(Python 3.12.3, torch 2.8.0 등).
- **평가 중 학습·보정 금지**: TTA/TTT, 다른 테스트 에피소드의 입력·예측·통계 사용(transductive)은 금지다.
- **데이터**: 운영진 Train data와 허용된 증강 데이터만 쓴다. 생성형 모델 증강은 T4(16GB)에서 재현 가능할 때만 허용한다. Claude·Codex 같은 외부 AI 서비스로 데이터를 만들면 안 된다.
- **hand-crafted description 금지**: 오답 문장은 규칙 기반 변환 코드나 로컬 생성형 모델 출력으로만 만든다. 증강 데이터는 생성 코드·시드·모델 버전과 함께 보관한다.
- **증류 금지**: teacher 모델의 예측·확률·logits·순위를 레이블이나 학습 목표로 쓰지 않는다.
- **재현성**: 증강부터 학습까지 전부 스크립트로 남기고, T4 1대에서 24시간 안에 재현 가능해야 한다.
