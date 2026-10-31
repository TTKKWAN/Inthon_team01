"""Score several trained runs on ONE shared validation set, so runs trained on
different negative banks are comparable.

The validation episodes come from each run's splits.json (all runs use seed 2026, so
they match) and the 5-way questions are built from --texts (default: the x1 bank).

  python eval_runs.py outputs/exp_base outputs/exp_x4 ...
"""

import argparse
import json
from pathlib import Path

import torch

from model_v2 import OrderAwareScorer
from train_v2 import Data, evaluate, valid_rows

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--texts", default="texts.pt")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cache = args.config.resolve().parent / cfg["cache_dir"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = Data(cache / "frames.pt", cache / args.texts, device)
    index = {e: i for i, e in enumerate(data.ids)}

    splits = None
    results = []
    for run in args.runs:
        valid_ids = json.loads((run / "splits.json").read_text(encoding="utf-8"))["valid"]
        if splits is None:
            splits = valid_ids
            valid = sorted(index[e] for e in valid_ids)
            sets = {"full": valid_rows(data, valid, 0, cfg["seed"] + 1),
                    "short": valid_rows(data, valid, 1, cfg["seed"] + 2)}
        elif valid_ids != splits:
            raise ValueError(f"{run} used a different validation split")
        ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model = OrderAwareScorer(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = {"run": run.name, "epoch": ckpt["epoch"]}
        for name, (rows, kinds) in sets.items():
            metrics.update(evaluate(model, data, rows, kinds, name))
        metrics["mean_lb"] = (metrics["full_lb"] + metrics["short_lb"]) / 2
        results.append(metrics)

    cols = ["mean_lb", "full_lb", "short_lb", "full_acc", "full_vs_spatial",
            "full_vs_attribute", "full_vs_antonym", "full_vs_order", "full_vs_drop_corrupt"]
    print("run".ljust(22) + "ep  " + "  ".join(c.replace("full_vs_", "vs_")[:10].rjust(10)
                                               for c in cols))
    for m in results:
        print(m["run"].ljust(22) + str(m["epoch"]).rjust(2) + "  "
              + "  ".join(f"{m[c]:10.4f}" for c in cols))
    (ROOT / "outputs" / "eval_runs.json").write_text(json.dumps(results, indent=2),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
