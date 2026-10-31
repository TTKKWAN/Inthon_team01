# 서버 호환성

## 확인한 제출 계약

- `server/contracts/archive.py`: ZIP 최상위 `submission.py` 필수.
- `submission.json`은 선택이며 포함 시 `{"api_version": 2}`만 사용.
- 기본 압축 해제 한도 1 GiB, 파일 수 10,000개 이하.
- `server/runner/submission.py`: `Submission(artifact_dir, device)`를 한 번 초기화한 후
  문제마다 `predict(images, choices)` 호출.
- `server/runner/execute.py`: RGB 이미지와 정답 정보 없는 선지를 전달하고,
  모든 선지 ID가 한 번씩 있는 `list[str]`를 검사.

이 starter kit는 위 계약에 맞춰 export합니다. 모델·토크나이저·전처리기·MLP를
모두 ZIP 내부에서 읽습니다. `validate.py --zip ... --server-dir ../server`로
서버의 안전한 ZIP 검사·압축 해제와 실제 runner 실행을 확인할 수 있습니다.

## 운영 환경에서 필요한 설정

현재 저장소의 `cpu-reference-v2`는 **Pillow만 설치한 CPU 참조 런타임**입니다.
`server/runtime/requirements.lock.txt`에 PyTorch와 Transformers가 없으므로,
기존 immutable 이미지에 이 신경망 제출물을 그대로 실행할 수는 없습니다.
제출 ZIP에 requirements를 넣어도 서버가 참가자 패키지를 설치하지 않습니다.

모델을 사용하는 공통 채점 이미지는 starter kit의 `requirements.txt`에 명시한
PyTorch / Transformers / NumPy / Pillow / safetensors를 제공해야 합니다.
GPU 이미지는 NVIDIA T4를 지원하는 CUDA/PyTorch 조합으로 운영진이 빌드·검증하고,
서버·이미지·워커의 runtime ID와 immutable image digest를 함께 맞춰야 합니다.
이 작업에서는 기존 서버 런타임 설정이나 운영 이미지를 변경하지 않았습니다.

인코더를 float16으로 저장해도 전체 제출 ZIP은 현재 서버 기본 업로드 제한인
256 MiB보다 큽니다. 운영 환경과 reverse proxy에 충분한 용량을 허용해야 합니다.
예를 들어 서버 설정은 다음과 같이 맞출 수 있습니다.

```text
MAX_UPLOAD_BYTES=1073741824
MAX_EXTRACTED_BYTES=1073741824
EXECUTION_TIMEOUT_SECONDS=900
```

압축 해제 한도는 기존 1 GiB와 동일합니다. 실행 제한 900초는 참가 규정의 15분입니다.
현재 서버 기본 실행 제한은 1800초이므로 운영 설정에서 규정과 맞춰야 합니다.
reverse proxy의 요청 크기 제한과 업로드 시간 제한도 실제 제출 크기에 맞춰야 합니다.

## 검증 범위

- 학습: 실제 Train의 소규모 부분집합으로 실행.
- 제출 코드: 실제 서버 archive contract와 CPU runner로 실행.
- 실행 환경: 로컬 Python 3.11.7, PyTorch 2.5.1+cu121, Transformers 4.57.3,
  NumPy 1.26.4, Pillow 10.2.0. 설치용 명세와 서버의 Pillow 버전은 12.3.0.
- GPU 드라이버가 제공되지 않아 T4 속도·메모리, 전체 Test 15분, 전체 학습 24시간은
  측정하지 않았습니다. 기존 Pillow 전용 Docker 이미지에서의 실행도 통과한 것으로 표시하지 않습니다.

실제 실행 결과는 [VERIFICATION.md](VERIFICATION.md)에 기록합니다.
