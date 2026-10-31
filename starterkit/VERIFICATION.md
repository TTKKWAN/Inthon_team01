# 실행 검증 기록

2026-10-29에 수행했습니다. 원본 `data/Train`만 읽었고 `data/Test`는 사용하지 않았습니다.
서버 코드·운영 설정과 원본 데이터는 변경하지 않았습니다.

## 학습과 평가

```bash
/home/aikusrv04/anaconda3/bin/python starterkit/train.py \
  --image-model-dir verification/models/dinov2 \
  --text-model-dir verification/models/mpnet \
  --device cpu --max-episodes 12 --epochs 3 \
  --output-dir starterkit/outputs/smoke

/home/aikusrv04/anaconda3/bin/python starterkit/evaluate.py \
  --checkpoint starterkit/outputs/smoke/best.pt --device cpu
```

- 실제 Train 4,170개 중 seed 2026으로 12개 선택, 학습 10개·검증 2개.
- 각 epoch의 학습 쌍은 정답 10개·random negative 10개.
- 3 epoch 실행 완료. 검증 BCE 기준 선택된 checkpoint는 epoch 1.
- checkpoint를 다시 불러온 검증 결과가 학습 중 기록과 일치:
  BCE `0.6996325254440308`, binary accuracy `0.5`, pair accuracy `0.5`.
- 이 수치는 기능 확인용 작은 표본의 결과이며 전체 학습이나 리더보드 성능이 아닙니다.

## 코드 검사

```bash
/home/aikusrv04/anaconda3/bin/python -m pytest -q starterkit/tests
server/.venv/bin/ruff check starterkit --output-format concise
/home/aikusrv04/anaconda3/bin/python -m compileall -q starterkit
```

테스트 **7개 통과**, Ruff와 Python 컴파일 검사 통과.
검사 범위는 정확한 1:1 레이블, 자기 설명·동일 설명 제외, split 내부 negative,
seed 재현과 epoch별 재추출, 숫자순 이미지 정렬, 마지막 이미지 배치까지 포함한 평균,
텍스트 padding 제외, 스칼라 sigmoid 출력과 BCE에 의한 MLP 갱신입니다.

## 제출물과 서버 runner

```bash
/home/aikusrv04/anaconda3/bin/python starterkit/export.py \
  --checkpoint starterkit/outputs/smoke/best.pt
/home/aikusrv04/anaconda3/bin/python starterkit/validate.py --device cpu
/home/aikusrv04/anaconda3/bin/python starterkit/package.py
/home/aikusrv04/anaconda3/bin/python starterkit/validate.py \
  --zip starterkit/sample_submission.zip --server-dir server --device cpu
```

- 직접 추론: 선지마다 0–1 점수 한 개, 5개/2개 선지, 임의 문자열 ID, 반복 호출 통과.
- 실제 `server/contracts/archive.py`: API v2 ZIP 검사·안전한 압축 해제 통과.
- 실제 `server/runner/execute.py`: Train 2개 에피소드 모두 유효한 ranking 반환.
- 새 임시 CWD와 빈 HF 캐시, offline 환경, `python -I`로 runner 실행.
- ZIP은 모델과 토크나이저를 포함하고, Train/Test·학습 데이터·특징 캐시·pycache를 포함하지 않음.
- export된 `submission.py`, `baseline_model.py`는 학습 코드 쪽 원본과 바이트 단위 일치.

| 항목 | 값 |
| --- | --- |
| ZIP | `sample_submission.zip` |
| 압축 크기 | 365,781,282 bytes, 약 348.8 MiB |
| 압축 해제 크기 | 394,711,685 bytes, 약 376.4 MiB |
| ZIP 파일 수 | 17 |
| SHA256 | `8a3c7222bde9bd99a01a1928087169229188c2c489f05e70fca19d9b49934646` |

## 운영 환경과의 구분

검증은 ML 패키지가 설치된 로컬 CPU 환경에서 실제 서버 runner를 실행한 것입니다.
현재 Pillow 전용 `cpu-reference-v2` Docker 이미지에서 실행한 것은 아닙니다.
현재 서버의 기본 업로드 한도 256 MiB도 이 ZIP보다 작습니다.
배포 전에 필요한 공통 ML 런타임과 용량 설정은
[SERVER_COMPATIBILITY.md](SERVER_COMPATIBILITY.md)에 명시했습니다.

GPU 드라이버를 사용할 수 없어 T4의 실행 시간·메모리, 전체 학습 24시간과
전체 Test 추론 15분 충족 여부는 검증하지 않았습니다.
