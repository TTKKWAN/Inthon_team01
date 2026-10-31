"""Starter-kit model on our cached features, to see which change matters most.

Model = the starter kit's: mean over frames of DINOv2 CLS (768) ++ MPNet mean-pooled full
description (768) -> Linear(1536, 256) -> ReLU -> Dropout -> Linear(256, 1).
These are the first 768 dims of our cached frame / full-text features.

  --mode random : starter recipe, 1 GT + 1 random other description, BCE
  --mode hard   : same model, our listwise recipe (GT + 8 hard + 3 random, CE)

Scored on the same 5-way validation as eval_runs.py (1 / 0.5 / 0.25 / 0.125 / 0).
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from train_v2 import RANK_SCORE, Data, valid_rows

ROOT = Path(__file__).resolve().parent


class StarterMLP(nn.Module):
    def __init__(self, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1536, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, video, text):
        return self.net(torch.cat([video, text], -1)).squeeze(-1)


def features(data):
    video = torch.stack([data.frames[i, :n, :768].float().mean(0)
                         for i, n in enumerate(data.lengths)])
    return video, data.full[:, :768].float()


def lb_score(model, video, text, rows):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for row in rows:
            v = video[[c[0] for c in row]]
            t = text[[c[2] for c in row]]
            s = model(v, t)
            total += RANK_SCORE[int((s[1:] >= s[0]).sum())]
    return total / len(rows)


def save_starter_checkpoint(path, state, best, mode, cfg):
    """format_version 1 checkpoint that the starter kit's export.py accepts unchanged."""
    models = (ROOT / cfg["image_model_dir"]).resolve(), (ROOT / cfg["text_model_dir"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "model_config": {"image_dim": 768, "text_dim": 768, "hidden_dim": 256, "dropout": 0.1},
        "state_dict": state,
        "config": {"image_model_dir": str(models[0]), "text_model_dir": str(models[1]),
                   "image_batch_size": 16, "text_batch_size": 32, "max_text_length": 384,
                   "cpu_threads": 4},
        "epoch": best[1]["epoch"],
        "metrics": {"val_mean_lb": round(best[0], 4), "val_full_lb": round(best[1]["full"], 4),
                    "val_short_lb": round(best[1]["short"], 4)},
        "provenance": {
            "image_model_id": "facebook/dinov2-base",
            "text_model_id": "sentence-transformers/all-mpnet-base-v2",
            "negative_sampling": f"{mode}: GT + 8 rule-based hard negatives (negatives.py) "
                                 "+ 3 random other GTs, listwise cross-entropy"
                                 if mode == "hard" else "1 random other GT, BCE",
        },
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["random", "hard"], required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save", type=Path,
                        help="write the best model as a starter-kit checkpoint (export.py input)")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cache = ROOT / cfg["cache_dir"]
    data = Data(cache / "frames.pt", cache / "texts.pt", "cuda:0")
    video, text = features(data)
    idx = {e: i for i, e in enumerate(data.ids)}
    split = json.loads((ROOT / "validation/splits.json").read_text(encoding="utf-8"))
    train = sorted(idx[e] for e in split["train"])
    valid = sorted(idx[e] for e in split["valid"])
    vsets = {"full": valid_rows(data, valid, 0, cfg["seed"] + 1)[0],
             "short": valid_rows(data, valid, 1, cfg["seed"] + 2)[0]}

    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    model = StarterMLP().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best = (-1, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train[:]
        rng.shuffle(perm)
        for s in range(0, len(perm), 64):
            idx_b = perm[s:s + 64]
            if args.mode == "random":
                # Starter recipe: own full GT (1) vs one other episode's full GT (0).
                neg = [rng.choice([j for j in rng.sample(train, 2) if j != i]) for i in idx_b]
                v = video[idx_b + idx_b]
                t = text[[data.groups[i][0]["pos"] for i in idx_b]
                         + [data.groups[j][0]["pos"] for j in neg]]
                y = torch.tensor([1.0] * len(idx_b) + [0.0] * len(idx_b), device="cuda:0")
                loss = nn.functional.binary_cross_entropy_with_logits(model(v, t), y)
            else:
                from train_v2 import train_rows
                rows = train_rows(data, idx_b, train, {**cfg, "reverse_video_negative": False},
                                  rng)
                v = video[[c[0] for r in rows for c in r]]
                t = text[[c[2] for r in rows for c in r]]
                logits = model(v, t).view(len(rows), -1)
                loss = nn.functional.cross_entropy(
                    logits, torch.zeros(len(rows), dtype=torch.long, device="cuda:0"))
            opt.zero_grad()
            loss.backward()
            opt.step()
        scores = {k: lb_score(model, video, text, r) for k, r in vsets.items()}
        mean = (scores["full"] + scores["short"]) / 2
        if mean > best[0]:
            best = (mean, {"epoch": epoch, **scores})
            # Starter kit names the Sequential "network"; keys otherwise match.
            best_state = {k.replace("net.", "network.", 1): v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if args.save:
        save_starter_checkpoint(args.save, best_state, best, args.mode, cfg)
    print(json.dumps({"mode": args.mode, "best_mean_lb": round(best[0], 4),
                      **{k: round(v, 4) if isinstance(v, float) else v
                         for k, v in best[1].items()}}))


if __name__ == "__main__":
    main()
