"""Time a submission ZIP the way the server runs it, to check the 15-minute limit on a T4.

  python t4_timing.py --zip v2_submission.zip --sample t4_timing_sample.zip

What it does
  1. Extracts the ZIP to a temp dir and builds Submission(artifact_dir, "cuda") once
     (timed separately: model loading).
  2. For every sample episode: opens its frames from disk as RGB PIL images (in numeric
     order) and calls predict(images, choices) with 5 choices (ids only, no answer).
     Each episode is timed end to end (image decode + predict), GPU synchronized.
  3. Repeats the sample `--passes` times and extrapolates to `--test-episodes` (1,910).

The sample's 60 Train episodes are 20 each with 19-20, 14-18 and 8-13 frames, so the
report also gives seconds per frame for when test episodes are longer or shorter.
The 5 choices are the GT + 4 rule-based negatives; the leaderboard-style score printed at
the end is only a check that the ZIP works, not a performance estimate.
"""

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
import zipfile
from pathlib import Path

RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]
LIMIT_SECONDS = 15 * 60


def load_episodes(sample_dir):
    episodes = []
    for folder in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
        frames = sorted((folder / "images").iterdir(), key=lambda p: int(p.stem))
        choices = json.loads((folder / "choices.json").read_text(encoding="utf-8"))
        episodes.append((folder.name, frames, choices))
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, required=True, help="submission ZIP")
    parser.add_argument("--sample", type=Path, required=True,
                        help="t4_timing_sample.zip (or its extracted folder)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--test-episodes", type=int, default=1910)
    args = parser.parse_args()

    import torch
    from PIL import Image

    work = Path(tempfile.mkdtemp(prefix="t4_timing_"))
    artifact = work / "submission"
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(artifact)
    if args.sample.is_dir():
        sample_dir = args.sample / "sample" if (args.sample / "sample").is_dir() else args.sample
    else:
        with zipfile.ZipFile(args.sample) as z:
            z.extractall(work)
        sample_dir = work / "sample"
    episodes = load_episodes(sample_dir)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"GPU: {gpu} | Python {platform.python_version()} | torch {torch.__version__}")
    if "T4" not in gpu:
        print("WARNING: not a T4, the numbers below do not answer the T4 question.")

    sys.path.insert(0, str(artifact))
    from submission import Submission

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    t0 = time.perf_counter()
    sub = Submission(str(artifact), args.device)
    sync()
    init_s = time.perf_counter() - t0

    def run(frames, choices):
        images = []
        for p in frames:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        ranking = sub.predict(images, [{"id": c["id"], "text": c["text"]} for c in choices])
        sync()
        return ranking

    run(*episodes[0][1:])  # warm-up (CUDA kernels, allocator)
    per_episode, per_frame, score = [], [], []
    for _ in range(args.passes):
        for _, frames, choices in episodes:
            t = time.perf_counter()
            ranking = run(frames, choices)
            dt = time.perf_counter() - t
            per_episode.append(dt)
            per_frame.append(dt / len(frames))
            gt = next(c["id"] for c in choices if c.get("gt"))
            score.append(RANK_SCORE[ranking.index(gt)])

    mean = statistics.mean(per_episode)
    p95 = sorted(per_episode)[int(0.95 * len(per_episode)) - 1]
    frames_mean = statistics.mean(len(f) for _, f, _ in episodes)
    total = init_s + mean * args.test_episodes
    worst = init_s + p95 * args.test_episodes
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"ZIP: {args.zip.name} ({args.zip.stat().st_size / 2**20:.0f} MiB)")
    print(f"model load: {init_s:.1f} s | peak GPU memory: {peak:.2f} GiB")
    print(f"per episode: mean {mean:.3f} s, p95 {p95:.3f} s "
          f"(sample mean {frames_mean:.1f} frames, {statistics.mean(per_frame) * 1000:.1f} ms/frame)")
    print(f"estimated {args.test_episodes} episodes: {total / 60:.1f} min "
          f"(if every episode were p95: {worst / 60:.1f} min) | limit 15 min")
    verdict = "OK" if worst < LIMIT_SECONDS * 0.7 else ("TIGHT" if total < LIMIT_SECONDS else "TOO SLOW")
    print(f"verdict: {verdict} (OK = even the p95 estimate is under 70% of the limit)")
    print(f"sanity score on sample (leaderboard style): {statistics.mean(score):.3f}")


if __name__ == "__main__":
    main()
