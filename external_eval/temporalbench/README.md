# TemporalBench: 평가 전용 외부 검증

`microsoft/TemporalBench` @ `7e36b12` (gated, MSR license; HF 계정 TTaekwan이 2026-11-01 동의).
**학습·증강·오답 선택·참조 영상·제출 zip에 절대 쓰지 않는다.** 설정이나 제출본을 고르는 검증에는 쓴다(사용자 결정, 2026-11-01).
데이터(`raw/`, `episodes/`, `sets/`, 데이터 카드)는 git에서 뺐다. 서버 `~/Inthon_tk/external_eval/temporalbench/`에 있다.

| 스크립트 | 내용 |
|---|---|
| `build.py` | `temporalbench_short_qa.json`(정답 + 한 곳 수정 오답, 수정 종류 표시)과 짧은 영상 2,179개로 5지선다 셀을 만든다. 영상마다 12프레임, 긴 변 640px. 셀: `single`(오답 4개 그대로), `ab_ab_c`, `ab_ab_ac`, `share4`(오답을 `merge_edits`로 합침). `--workers`로 병렬 추출 |
| `score.py` | 제출본 `video_judge_patch`, `video_judge_siglip`을 채점한다. 영상 인코딩과 문장 사전값을 캐시하고, 처음 `--check`개는 실제 `Submission.predict`와 순위를 대조한다. 다른 영상 대조 포함 |
| `report.py` | 셀별, 공유 수정 종류별, 출처별 표와 쌍 정확도 (CPU) |

```bash
python external_eval/temporalbench/build.py --workers 12          # CPU, 약 15분 (20코어)
python external_eval/temporalbench/score.py --patch diff_probe/runs/vjp_unzip \
    --siglip-sub diff_probe/runs/vjs_unzip --out external_eval/temporalbench/results/score.json   # A100 약 12분
python external_eval/temporalbench/report.py external_eval/temporalbench/results/score.json
```

셀 크기: single 1,255 / ab_ab_c 1,763 / ab_ab_ac 1,665 / share4 1,049 (영상 1,840개, 3개는 디코딩 실패).

## 결과 (2026-11-01)

| 구성 | single | ab_ab_c | ab_ab_ac | share4 | 보정 비율 혼합 | 혼합, 다른 영상 |
|---|---|---|---|---|---|---|
| span 단독 | 0.932 | 0.831 | 0.314 | 0.202 | 0.484 | - |
| + 0.3 SigLIP | 0.835 | 0.679 | 0.430 | 0.261 | 0.501 | 0.476 |
| + 0.3 공간 헤드 | 0.814 | 0.613 | 0.388 | 0.232 | 0.451 | 0.452 |
| 제출본 (0.3, 0.3) | 0.774 | 0.619 | 0.424 | 0.285 | 0.478 | 0.459 |

쌍 정확도(single 셀, 5,020쌍): SigLIP 0.561 (다른 영상 0.538), 공간 헤드 0.496 (0.491).
리더보드(제출본 0.4952 < span 0.5661)와 같은 방향이다.
