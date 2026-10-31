"""Paired A/B comparison of train_v2 runs that differ only in their texts cache.

Question sets (same five texts per question in every arm; only the cached text features
differ between arms):
  rule_{full,short} : valid_rows, the original shared validation (solvable from wording)
  af_{full,short}   : validation/val_af_*.json from eval_nnval.py (text-only-hard rule edits)
  nn_{full,short}   : validation/val_nn_*.json from eval_nnval.py (GTs of similar episodes)
Per-question scores are averaged over an arm's runs (seeds); the arm difference is
bootstrapped over questions. --zero-shot also scores every arm without training, from
SigLIP cosine similarity: zs_mean (mean over frames, as in eval_nnval), zs_cmean / zs_cmin
(per clause: best-matching frame; then mean / weakest clause).

  python ab_compare.py --config config_ab.json --sets rule,af,nn --zero-shot \
      --arm old:texts.pt:outputs/ab_old_s1,outputs/ab_old_s2 \
      --arm new:texts_v3.pt:outputs/ab_new_s1,outputs/ab_new_s2
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from model_v2 import SIGLIP_DIM, OrderAwareScorer, pick_device
from train_v2 import RANK_SCORE, Data, score, valid_rows

ROOT = Path(__file__).resolve().parent
VIEWS = (("full", 0), ("short", 1))


def result(logits):
    """(leaderboard score, [positive beats negative j]); ties count against the positive."""
    rank = int((logits[1:] >= logits[0]).sum())
    return RANK_SCORE[rank], [bool(logits[0] > x) for x in logits[1:]]


def load_sets(data, cfg, valid_ids, names):
    index = {e: i for i, e in enumerate(data.ids)}
    valid = sorted(index[e] for e in valid_ids)
    text_index = {t: k for k, t in enumerate(data.texts)}
    sets = {}
    for view, g in VIEWS:
        if "rule" in names:
            sets[f"rule_{view}"] = valid_rows(data, valid, g, cfg["seed"] + 1 + g)
        for kind in ("af", "nn"):
            path = ROOT / f"validation/val_{kind}_{view}.json"
            if kind not in names or not path.exists():
                continue
            rows, kinds = [], []
            for q in json.loads(path.read_text(encoding="utf-8")):
                i = index[q["episode"]]
                tids = [text_index[c["text"]] for c in q["choices"]]
                negs = data.groups[i][g]["negs"] if g < len(data.groups[i]) else []
                type_of = {t: k for k, t in negs}
                rows.append([(i, False, t) for t in tids])
                kinds.append(["pos"] + [type_of.get(t, kind) for t in tids[1:]])
            sets[f"{kind}_{view}"] = (rows, kinds)
    return sets


@torch.no_grad()
def trained_results(model, data, rows, batch=128):
    model.eval()
    return [result(logits) for s in range(0, len(rows), batch)
            for logits in score(model, data, rows[s:s + batch]).cpu()]


@torch.no_grad()
def zero_shot_results(data, frames, rows):
    out = {"zs_mean": [], "zs_cmean": [], "zs_cmin": []}
    for row in rows:
        i = row[0][0]
        v = F.normalize(frames[i, :data.lengths[i], -SIGLIP_DIM:].float(), dim=-1)
        per = {k: [] for k in out}
        for _, _, t in row:
            full = F.normalize(data.full[t, -SIGLIP_DIM:].float().cpu(), dim=-1)
            c = data.clause_flat[data.clause_off[t]:data.clause_off[t + 1], -SIGLIP_DIM:]
            best = (F.normalize(c.float().cpu(), dim=-1) @ v.T).max(1).values
            per["zs_mean"].append((v @ full).mean())
            per["zs_cmean"].append(best.mean())
            per["zs_cmin"].append(best.min())
        for k in out:
            out[k].append(result(torch.stack(per[k])))
    return out


def summarize(results, kinds):
    """results: list over runs of per-question (score, beats) -> per-question mean, pass rates."""
    n = len(results[0])
    mean_q = [statistics.fmean(r[i][0] for r in results) for i in range(n)]
    passed, seen = defaultdict(float), defaultdict(int)
    for i, kind in enumerate(kinds):
        for j, k in enumerate(kind[1:]):
            seen[k] += 1
            passed[k] += statistics.fmean(r[i][1][j] for r in results)
    return mean_q, {k: passed[k] / seen[k] for k in sorted(seen)}


def paired(a, b, n=10000, seed=0):
    d = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(n))
    return {"diff": statistics.fmean(d), "ci95": [boots[int(0.025 * n)], boots[int(0.975 * n) - 1]],
            "p_le_0": sum(x <= 0 for x in boots) / n,
            "better": sum(x > 1e-9 for x in d), "worse": sum(x < -1e-9 for x in d)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--splits", type=Path, default=ROOT / "validation/splits.json")
    parser.add_argument("--arm", action="append", required=True,
                        help="name:texts_cache[:run_dir,run_dir...] (no runs = zero-shot only)")
    parser.add_argument("--sets", default="rule", help="comma list of rule, af, nn")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ab_compare.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cache = args.config.resolve().parent / cfg["cache_dir"]
    device = pick_device()
    valid_ids = json.loads(args.splits.read_text(encoding="utf-8"))["valid"]
    names = set(args.sets.split(","))

    arms, signature = {}, None
    for spec in args.arm:
        name, texts, *runs = spec.split(":", 2)
        runs = [Path(r) for r in runs[0].split(",")] if runs and runs[0] else []
        data = Data(cache / "frames.pt", cache / texts, device)
        sets = load_sets(data, cfg, valid_ids, names)
        sig = {s: [[data.texts[c[2]] for c in row] for row in rows] for s, (rows, _) in sets.items()}
        if signature is None:
            signature = sig
        elif sig != signature:
            raise ValueError(f"arm {name} sees different questions")
        arm = {"kinds": {s: k for s, (_, k) in sets.items()}, "scorers": {}}
        if runs:
            per_run = defaultdict(list)
            arm["runs"] = []
            for run in runs:
                split = json.loads((run / "splits.json").read_text(encoding="utf-8"))["valid"]
                if split != valid_ids:
                    raise ValueError(f"{run} used a different validation split")
                ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
                model = OrderAwareScorer(**ckpt["model_config"]).to(device)
                model.load_state_dict(ckpt["state_dict"])
                arm["runs"].append({"run": run.name, "epoch": ckpt["epoch"]})
                for s, (rows, _) in sets.items():
                    per_run[s].append(trained_results(model, data, rows))
            arm["scorers"]["trained"] = dict(per_run)
        if args.zero_shot:
            frames = data.frames.cpu()
            for s, (rows, _) in sets.items():
                for k, res in zero_shot_results(data, frames, rows).items():
                    arm["scorers"].setdefault(k, {})[s] = [res]
        arms[name] = arm
        del data
        if device == "mps":
            torch.mps.empty_cache()

    names_ = list(arms)
    base = names_[0]
    report = {"questions": {s: len(v) for s, v in signature.items()}, "arms": {}, "diff": {}}
    print("questions:", report["questions"])
    for name, arm in arms.items():
        report["arms"][name] = {"runs": arm.get("runs", []), "scores": {}}
        for scorer, by_set in arm["scorers"].items():
            for s, results in by_set.items():
                mean_q, pass_rate = summarize(results, arm["kinds"][s])
                report["arms"][name]["scores"][f"{scorer}/{s}"] = {
                    "lb": statistics.fmean(mean_q), "per_run": [statistics.fmean(x[0] for x in r)
                                                                 for r in results],
                    "pass_rate": pass_rate, "mean_q": mean_q}
    for other in names_[1:]:
        print(f"\n{other} vs {base}: mean leaderboard-style score, paired diff [95% CI], "
              f"questions better/worse")
        for key, a in report["arms"][base]["scores"].items():
            b = report["arms"][other]["scores"].get(key)
            if not b:
                continue
            d = paired(a["mean_q"], b["mean_q"])
            report["diff"][f"{other}-{base}/{key}"] = d
            runs = (f"  runs {base} {['%.3f' % x for x in a['per_run']]} {other} "
                    f"{['%.3f' % x for x in b['per_run']]}" if len(a["per_run"]) > 1 else "")
            print(f"  {key:22s} {a['lb']:.4f} -> {b['lb']:.4f}  {d['diff']:+.4f} "
                  f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  {d['better']}/{d['worse']}{runs}")
        for key, a in report["arms"][base]["scores"].items():
            b = report["arms"][other]["scores"].get(key)
            if b and key.startswith("trained/") and key.endswith(("af_full", "rule_full")):
                print(f"  pass rate {key}: " + ", ".join(
                    f"{k} {a['pass_rate'][k]:.2f}->{b['pass_rate'][k]:.2f}"
                    for k in a["pass_rate"] if k in b["pass_rate"]))
    for arm in report["arms"].values():
        for v in arm["scores"].values():
            v.pop("mean_q")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
