"""Write the shared 5-way validation set to readable JSON, so anyone can inspect or reuse it.

Same questions as train_v2 / eval_runs / eval_submission: valid episodes of the seed-2026
split, 1 GT + 4 negatives of distinct types from negatives.json (seeds 2027 full, 2028
short). Every choice keeps its negative type; choice 0 is always the GT here, so shuffle
before showing them to a model.

  python export_val_set.py --out validation/
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from train_v2 import valid_rows

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--splits", type=Path, default=ROOT / "validation/splits.json")
    parser.add_argument("--out", type=Path, default=ROOT / "validation")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    t = torch.load(args.config.resolve().parent / cfg["cache_dir"] / "texts.pt",
                   map_location="cpu", weights_only=False)
    ids = t["episode_ids"]
    index = {e: i for i, e in enumerate(ids)}
    split = json.loads(args.splits.read_text(encoding="utf-8"))
    valid = sorted(index[e] for e in split["valid"])
    data = SimpleNamespace(groups=t["groups"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "splits.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    for name, group, seed in [("full", 0, cfg["seed"] + 1), ("short", 1, cfg["seed"] + 2)]:
        rows, kinds = valid_rows(data, valid, group, seed)
        questions = [{"episode": ids[row[0][0]],
                      "choices": [{"type": k, "text": t["texts"][c[2]]}
                                  for c, k in zip(row, kind)]}
                     for row, kind in zip(rows, kinds)]
        (args.out / f"val_{name}.json").write_text(
            json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"val_{name}.json: {len(questions)} questions")


if __name__ == "__main__":
    main()
