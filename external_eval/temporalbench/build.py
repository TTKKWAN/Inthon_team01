"""TemporalBench short videos as Inthon-style 5-choice episodes. EVALUATION ONLY.

Never train, augment, pick negatives, tune a head or choose a weight on this data
(Rule.md §3-7). Results are reported for fixed configs only.

Input (raw/, out of git): temporalbench_short_qa.json and short_video/ from
microsoft/TemporalBench @ 7e36b12. Each binary QA item pairs a video's caption with one
edited caption and labels the edit ("category").

Output:
  episodes/<id>/images/1.jpg..12.jpg  12 evenly spaced frames, long side 640 (like Train)
  sets/<cell>.json                    5-choice rows in the validator/eval_subs.py format
Cells (same structure as validator/calibrated/, edits are TemporalBench's, combined by rule):
  single    GT + {a, b, c, d}          4 TemporalBench negatives as given
  ab_ab_c   GT + {a, b, a+b, c}
  ab_ab_ac  GT + {a, b, a+b, a+c}
  share4    GT + {a, a+b, a+c, a+d}
"a+b" applies both word-level edits to GT (diff_probe/sim_valsets.merge_edits).

    python external_eval/temporalbench/build.py
"""

import argparse
import collections
import json
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "diff_probe"))
from sim_valsets import merge_edits  # noqa: E402

N_FRAMES = 12
LONG_SIDE = 640
QA = re.compile(r"\nA\. (.*?)\s*\nB\. (.*?)\s*\nAnswer", re.S)


def load_videos(qa_path):
    """video_name -> {"dataset", "pos", "negs": [(category, text)]}"""
    videos = {}
    for item in json.loads(qa_path.read_text(encoding="utf-8")):
        a, b = (" ".join(t.split()) for t in QA.search(item["question"]).groups())
        pos, neg = (a, b) if item["GT"] == "A" else (b, a)
        v = videos.setdefault(item["video_name"], {"dataset": item["dataset"], "pos": collections.Counter(),
                                                   "negs": {}})
        v["pos"][pos] += 1
        v["negs"].setdefault(neg, item["category"])
    for v in videos.values():
        pos = v["pos"].most_common(1)[0][0]
        v["pos"] = pos
        v["negs"] = [(k, t) for t, k in v["negs"].items() if t.lower() != pos.lower()]
    return videos


def episode_id(video_name):
    return "__".join(Path(video_name).with_suffix("").parts[1:])


def extract_frames(video, out_dir):
    # Two sequential passes (count, then keep 12): seeking is unreliable on some codecs.
    cap = cv2.VideoCapture(str(video))
    total = 0
    while cap.grab():
        total += 1
    cap.release()
    if total < N_FRAMES:
        return False
    idx = [round(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)]
    cap, frames, i = cv2.VideoCapture(str(video)), [], 0
    while len(frames) < N_FRAMES and cap.grab():
        while len(frames) < N_FRAMES and idx[len(frames)] == i:
            frames.append(cap.retrieve()[1])
        i += 1
    cap.release()
    if len(frames) < N_FRAMES:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, f in enumerate(frames, 1):
        h, w = f.shape[:2]
        s = LONG_SIDE / max(h, w)
        f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{n}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def build_cell(cell, pos, negs, rng):
    """Returns [(kind, text)] for the 4 negatives, or None if this video cannot fill the cell."""
    need = {"single": 4, "ab_ab_c": 3, "ab_ab_ac": 3, "share4": 4}[cell]
    if len(negs) < need:
        return None
    for _ in range(20):
        a, b, c, *rest = rng.sample(negs, need)
        if cell == "single":
            out = [a, b, c, rest[0]]
        else:
            ab = merge_edits(pos, a[1], b[1])
            ac = merge_edits(pos, a[1], c[1])
            ad = merge_edits(pos, a[1], rest[0][1]) if rest else None
            plus = lambda x, y, t: (f"{x[0]}+{y[0]}", t) if t else None  # noqa: E731
            out = {"ab_ab_c": [a, b, plus(a, b, ab), c],
                   "ab_ab_ac": [a, b, plus(a, b, ab), plus(a, c, ac)],
                   "share4": [a, plus(a, b, ab), plus(a, c, ac), plus(a, rest[0], ad) if rest else None]}[cell]
        if all(out) and len({t.lower() for _, t in out} | {pos.lower()}) == 5:
            return out
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "raw")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-extract", action="store_true",
                    help="only use videos whose frames already exist; write sets_partial/")
    args = ap.parse_args()

    videos = load_videos(args.raw / "temporalbench_short_qa.json")
    sets = {c: [] for c in ("single", "ab_ab_c", "ab_ab_ac", "share4")}
    stats = collections.Counter()
    built = {}
    for name in sorted(videos):
        v = videos[name]
        ep = episode_id(name)
        rows = {}
        for cell in sets:
            negs = build_cell(cell, v["pos"], v["negs"], random.Random(f"{args.seed}:{cell}:{name}"))
            if negs:
                rows[cell] = {"episode": ep, "video": name, "dataset": v["dataset"], "cell": cell,
                              "shared_kind": negs[0][0] if cell in ("ab_ab_ac", "share4") else None,
                              "choices": [{"type": "pos", "text": v["pos"]}]
                              + [{"type": "neg", "kind": k, "text": t} for k, t in negs]}
        if not rows:
            stats["no_cell"] += 1
            continue
        built[name] = rows
    todo = [(args.raw / name, args.out / "episodes" / episode_id(name) / "images") for name in built]
    todo = [t for t in todo if not (t[1] / f"{N_FRAMES}.jpg").exists()]
    if args.no_extract:
        bad = {str(v) for v, _ in todo}
    else:
        with Pool(args.workers) as pool:
            bad = {str(v) for (v, _), ok in zip(todo, pool.starmap(extract_frames, todo)) if not ok}
    for name, rows in built.items():
        if str(args.raw / name) in bad:
            stats["bad_video"] += 1
            continue
        for cell, row in rows.items():
            sets[cell].append(row)
    set_dir = args.out / ("sets_partial" if args.no_extract else "sets")
    set_dir.mkdir(parents=True, exist_ok=True)
    for cell, rows in sets.items():
        (set_dir / f"{cell}.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        kinds = collections.Counter(c["kind"] for r in rows for c in r["choices"][1:])
        print(f"{cell:9s} n={len(rows):5d}", dict(kinds.most_common()))
    print(dict(stats), "videos", len(videos))


if __name__ == "__main__":
    main()
