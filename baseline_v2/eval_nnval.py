"""Score models on three 5-way validations of the same 626 valid episodes (full + short each):

  rule : valid_rows — 4 rule-based negatives of distinct types (the original shared val).
         Solvable from wording alone: a text-only scorer gets 0.78, above minimal's 0.72,
         while minimal scored 0.426 on the public leaderboard.
  nn   : nn_rows — negatives are the real GTs of the 4 most MPNet-similar other valid
         episodes. Natural wording, so it measures video-text matching (scene level).
  af   : rule-based negatives of the episode, keeping the 4 that the selector text-only model
         (--selector) finds most GT-like. Subtle edits whose wording does not give them away,
         closer to the test's "slightly different" negatives. Biased against the selector by
         construction, so the shortcut check is the separate --probe text-only model.
         The selector only picks validation questions; nothing here is used for training.

Models: probe/selector text-only MLPs (no video), siglip_zs (no training: mean over frames of
SigLIP cos(frame, text)), starter-kit MLPs (ablation_mlp.py / starterkit train.py), and any
number of train_v2 checkpoints.
Writes validation/val_{nn,af}_{full,short}.json (choice 0 = GT) and outputs/eval_nnval.json.

  python eval_nnval.py --v2 base=outputs/v2_a100/best.pt nn4=outputs/v2_nn4/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ablation_mlp import StarterMLP
from ablation_textonly import TextMLP
from model_v2 import SIGLIP_DIM, OrderAwareScorer
from train_v2 import RANK_SCORE, Data, nn_rows, score, valid_rows

ROOT = Path(__file__).resolve().parent


def load_text_only(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TextMLP(ckpt["dim"]).cuda().eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["dim"]


def af_rows(data, valid, group_index, selector, dim):
    """GT + the 4 rule-based negatives the selector text-only model scores highest."""
    rows = []
    for i in valid:
        if group_index >= len(data.groups[i]):
            continue
        group = data.groups[i][group_index]
        negs = sorted({t for _, t in group["negs"]})
        with torch.no_grad():
            s = selector(data.full[negs, :dim].float())
        top = [negs[j] for j in s.topk(min(4, len(negs))).indices.tolist()]
        if len(top) == 4:
            rows.append([(i, False, group["pos"])] + [(i, False, t) for t in top])
    return rows


def rank_score(logits):
    """logits [R, 5], GT at 0 -> mean contest score."""
    ranks = (logits[:, 1:] >= logits[:, :1]).sum(1)
    return sum(RANK_SCORE[r] for r in ranks.tolist()) / len(ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlp", nargs="*", default=["minimal=outputs/minimal_hardneg/best.pt"],
                        metavar="NAME=CHECKPOINT",
                        help="starter-kit format MLP checkpoints (ablation_mlp.py or starterkit train.py)")
    parser.add_argument("--selector", type=Path, default=ROOT / "outputs/textonly_full.pt")
    parser.add_argument("--probe", type=Path, default=ROOT / "outputs/textonly_mpnet.pt")
    parser.add_argument("--v2", nargs="*", default=["base=outputs/v2_a100/best.pt"],
                        metavar="NAME=CHECKPOINT")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    cache = ROOT / cfg["cache_dir"]
    data = Data(cache / "frames.pt", cache / "texts.pt", "cuda:0")
    idx = {e: i for i, e in enumerate(data.ids)}
    split = json.loads((ROOT / "validation/splits.json").read_text(encoding="utf-8"))
    valid = sorted(idx[e] for e in split["valid"])
    selector, sel_dim = load_text_only(args.selector)
    probe, probe_dim = load_text_only(args.probe)

    sets = {}
    for view, g in [("full", 0), ("short", 1)]:
        sets[f"rule_{view}"] = valid_rows(data, valid, g, cfg["seed"] + 1 + g)[0]
        sets[f"nn_{view}"] = nn_rows(data, valid, g)[0]
        sets[f"af_{view}"] = af_rows(data, valid, g, selector, sel_dim)
    for name, rows in sets.items():
        if name.startswith(("nn_", "af_")):
            questions = [{"episode": data.ids[row[0][0]],
                          "choices": [{"type": "pos" if j == 0 else name.split("_")[0],
                                       "text": data.texts[c[2]]} for j, c in enumerate(row)]}
                         for row in rows]
            (ROOT / f"validation/val_{name}.json").write_text(
                json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in sets.items()})

    video_cls = torch.stack([data.frames[i, :n, :768].float().mean(0)
                             for i, n in enumerate(data.lengths)])

    def flat(rows):
        return [c for r in rows for c in r]

    def text_scorer(model, dim):
        return lambda rows: model(
            data.full[[c[2] for c in flat(rows)], :dim].float()).view(len(rows), -1)

    def siglip_zs(rows):
        out = []
        for row in rows:
            i = row[0][0]
            frames = F.normalize(data.frames[i, :data.lengths[i], -SIGLIP_DIM:].float(), dim=-1)
            texts = F.normalize(data.full[[c[2] for c in row], -SIGLIP_DIM:].float(), dim=-1)
            out.append((texts @ frames.T).mean(1))
        return torch.stack(out)

    def mlp_scorer(path):
        model = StarterMLP().cuda().eval()
        state = torch.load(ROOT / path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict({k.replace("network.", "net.", 1): v for k, v in state.items()})
        return lambda rows: model(
            video_cls[[c[0] for c in flat(rows)]],
            data.full[[c[2] for c in flat(rows)], :768].float()).view(len(rows), -1)

    def v2_scorer(path):
        ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        model = OrderAwareScorer(**ckpt["model_config"]).cuda().eval()
        model.load_state_dict(ckpt["state_dict"])
        return lambda rows: torch.cat([score(model, data, rows[s:s + 128])
                                       for s in range(0, len(rows), 128)])

    scorers = {
        "probe_text": text_scorer(probe, probe_dim),
        "select_text": text_scorer(selector, sel_dim),
        "siglip_zs": siglip_zs,
    }
    for item in args.mlp:
        name, path = item.split("=", 1)
        scorers[name] = mlp_scorer(path)
    for item in args.v2:
        name, path = item.split("=", 1)
        scorers[f"v2_{name}"] = v2_scorer(path)

    results = {}
    with torch.no_grad():
        for model_name, fn in scorers.items():
            r = {s: round(rank_score(fn(rows).float().cpu()), 4) for s, rows in sets.items()}
            for kind in ("rule", "nn", "af"):
                r[kind] = round((r[f"{kind}_full"] + r[f"{kind}_short"]) / 2, 4)
            results[model_name] = r
    print(f"\n{'model':12s} |   rule |     nn |     af | (full/short each)")
    for m, r in results.items():
        print(f"{m:12s} | {r['rule']:.4f} | {r['nn']:.4f} | {r['af']:.4f} | "
              + "  ".join(f"{k}={r[k]:.3f}" for k in sets))
    print("random = 0.375 | minimal public LB = 0.4260")
    (ROOT / "outputs/eval_nnval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
