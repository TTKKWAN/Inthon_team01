"""Score packaged submissions and model-free centrality on 5-choice validation sets.

A set is a JSON list of {"episode", "choices": [{"type", "text"}, ...]} with the positive's
type "pos". Choices are shuffled per episode with a fixed seed and passed to
Submission.predict(images, choices) the way the server does.

    python validator/eval_subs.py --sets validator/sets/*.json \
        --sub starter=baseline_v2/outputs/starter_submission --out validator/results/starter.json
    python validator/eval_subs.py --sets validator/sets/*.json --centrality   # no GPU, no images
"""

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", Path(path).name)]


def load_images(folder):
    from PIL import Image
    paths = sorted((p for p in (folder / "images").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}), key=natural_key)
    return [Image.open(p).convert("RGB") for p in paths]


def load_submission(folder, device):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("participant_submission", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Submission(str(folder), device)


def shuffled(row, seed):
    choices = [{"id": "ABCDE"[i], "text": c["text"], "pos": c["type"] == "pos"}
               for i, c in enumerate(row["choices"])]
    random.Random(f"{seed}:{row['episode']}").shuffle(choices)
    return choices, next(c["id"] for c in choices if c["pos"])


def centrality_rankers():
    sys.path.insert(0, str(ROOT / "diff_probe"))
    from centrality import SCORERS

    def ranker(fn):
        def rank(images, choices):
            scores = fn([c["text"] for c in choices])
            order = sorted(range(len(choices)), key=lambda i: -scores[i])
            return [choices[i]["id"] for i in order]
        return rank
    return {f"central_{k}": ranker(fn) for k, fn in SCORERS.items()}


def run(predict, rows, data_dir, seed, need_images):
    total, top1 = 0.0, 0
    for row in rows:
        choices, truth = shuffled(row, seed)
        images = load_images(data_dir / row["episode"]) if need_images else None
        order = predict(images, [{"id": c["id"], "text": c["text"]} for c in choices])
        rank = order.index(truth)
        total += RANK_SCORE[rank]
        top1 += rank == 0
    return round(total / len(rows), 4), round(top1 / len(rows), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", type=Path, required=True)
    ap.add_argument("--sub", help="NAME=SUBMISSION_DIR")
    ap.add_argument("--centrality", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/Train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sets = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in args.sets}
    if args.centrality:
        predictors, need_images = centrality_rankers(), False
    else:
        name, folder = args.sub.split("=", 1)
        sub = load_submission(Path(folder).resolve(), args.device)
        predictors, need_images = {name: sub.predict}, True

    results = {}
    for name, predict in predictors.items():
        results[name] = {}
        for set_name, rows in sets.items():
            t0 = time.time()
            score, top1 = run(predict, rows, args.data_dir, args.seed, need_images)
            results[name][set_name] = {"score": score, "top1": top1, "n": len(rows)}
            print(f"{name:16s} {set_name:20s} score={score:.4f} top1={top1:.3f} "
                  f"n={len(rows)} {time.time() - t0:.0f}s", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
