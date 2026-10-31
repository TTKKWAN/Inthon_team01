"""Text-only control: can the negatives be spotted without looking at the video?

Same recipe as `ablation_mlp.py --mode hard` (GT + 8 rule-based hard + 3 random, listwise CE,
same 5-way validation), but the scorer only sees the candidate text feature:
  --text mpnet : MPNet mean-pooled full description (768), the text half of minimal
  --text full  : MPNet ++ mean SigLIP over clauses (1536), v2's full-text feature
If this scores far above random (0.375), the validation set is solvable from wording alone.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from train_v2 import RANK_SCORE, Data, train_rows, valid_rows

ROOT = Path(__file__).resolve().parent


class TextMLP(nn.Module):
    def __init__(self, dim, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, text):
        return self.net(text).squeeze(-1)


def lb_score(model, text, rows):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for row in rows:
            s = model(text[[c[2] for c in row]])
            total += RANK_SCORE[int((s[1:] >= s[0]).sum())]
    return total / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", choices=["mpnet", "full"], default="mpnet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save", type=Path, help="write the best state_dict here")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cache = ROOT / cfg["cache_dir"]
    data = Data(cache / "frames.pt", cache / "texts.pt", "cuda:0")
    text = data.full.float()
    if args.text == "mpnet":
        text = text[:, :768]
    idx = {e: i for i, e in enumerate(data.ids)}
    split = json.loads((ROOT / "validation/splits.json").read_text(encoding="utf-8"))
    train = sorted(idx[e] for e in split["train"])
    valid = sorted(idx[e] for e in split["valid"])
    vsets = {"full": valid_rows(data, valid, 0, cfg["seed"] + 1)[0],
             "short": valid_rows(data, valid, 1, cfg["seed"] + 2)[0]}

    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    model = TextMLP(text.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best = (-1, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train[:]
        rng.shuffle(perm)
        for s in range(0, len(perm), 64):
            rows = train_rows(data, perm[s:s + 64], train,
                              {**cfg, "reverse_video_negative": False}, rng)
            logits = model(text[[c[2] for r in rows for c in r]]).view(len(rows), -1)
            loss = nn.functional.cross_entropy(
                logits, torch.zeros(len(rows), dtype=torch.long, device="cuda:0"))
            opt.zero_grad()
            loss.backward()
            opt.step()
        scores = {k: lb_score(model, text, r) for k, r in vsets.items()}
        mean = (scores["full"] + scores["short"]) / 2
        if mean > best[0]:
            best = (mean, {"epoch": epoch, **scores})
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"text": args.text, "dim": text.shape[1], "state_dict": best_state}, args.save)
    print(json.dumps({"text": args.text, "best_mean_lb": round(best[0], 4),
                      **{k: round(v, 4) if isinstance(v, float) else v
                         for k, v in best[1].items()}}))


if __name__ == "__main__":
    main()
