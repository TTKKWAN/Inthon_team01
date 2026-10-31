"""Score submission ZIPs (or extracted folders) through their own Submission.predict on the
shared 5-way validation set, so any model — starter kit, minimal, v2 — is compared on the
exact same questions and the exact inference path the server runs.

Questions are the same as train_v2 / eval_runs (valid split of outputs/exp_base, seeds
2027 / 2028, 1 GT + 4 negatives of distinct types), built from texts.pt; frames are read
from Train as RGB PIL images in numeric order.

  python eval_submission.py outputs/minimal_submission.zip outputs/v2_submission.zip
"""

import argparse
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from train_v2 import RANK_SCORE, valid_rows

ROOT = Path(__file__).resolve().parent


def load_submission(path):
    path = Path(path)
    if path.suffix == ".zip":
        folder = Path(tempfile.mkdtemp(prefix="eval_sub_"))
        with zipfile.ZipFile(path) as z:
            z.extractall(folder)
        path = folder
    # Each submission ships its own submission.py / model modules: import them fresh.
    for name in ["submission", "baseline_model", "model_v2"]:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path))
    try:
        from submission import Submission
        return Submission(str(path), "cuda:0" if torch.cuda.is_available() else "cpu")
    finally:
        sys.path.remove(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submissions", nargs="+")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cache = args.config.resolve().parent / cfg["cache_dir"]
    t = torch.load(cache / "texts.pt", map_location="cpu", weights_only=False)
    data = SimpleNamespace(groups=t["groups"])
    ids = t["episode_ids"]
    index = {e: i for i, e in enumerate(ids)}
    split = json.loads((ROOT / "validation/splits.json").read_text(encoding="utf-8"))
    valid = sorted(index[e] for e in split["valid"])
    sets = {"full": valid_rows(data, valid, 0, cfg["seed"] + 1),
            "short": valid_rows(data, valid, 1, cfg["seed"] + 2)}
    train_dir = args.config.resolve().parent / cfg["data_dir"]

    def frames(i):
        folder = train_dir / ids[i] / "images"
        images = []
        for p in sorted(folder.iterdir(), key=lambda p: int(p.stem)):
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        return images

    results = []
    for sub_path in args.submissions:
        sub = load_submission(sub_path)
        res = {"submission": Path(sub_path).name}
        t0 = time.perf_counter()
        for name, (rows, _) in sets.items():
            ranks = []
            for row in rows:
                video = row[0][0]
                choices = [{"id": f"c{j}", "text": t["texts"][c[2]]} for j, c in enumerate(row)]
                ranks.append(sub.predict(frames(video), choices).index("c0"))
            n = len(ranks)
            res[f"{name}_lb"] = sum(RANK_SCORE[r] for r in ranks) / n
            res[f"{name}_ranks"] = [round(ranks.count(k) / n, 3) for k in range(5)]
        res["mean_lb"] = (res["full_lb"] + res["short_lb"]) / 2
        res["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(res)
        print(json.dumps(res), flush=True)
        del sub
        torch.cuda.empty_cache()
    (ROOT / "outputs" / "eval_submissions.json").write_text(json.dumps(results, indent=2),
                                                           encoding="utf-8")


if __name__ == "__main__":
    main()
