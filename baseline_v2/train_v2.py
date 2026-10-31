"""Listwise training of OrderAwareScorer on cached features + the rule-based negative bank.

For each training video, one group (full GT, or with `short_group_prob` the first-sentence
GT) is drawn and the candidate list is
  [group positive] + `hard_negatives` edits of it (from negatives.py)
  + `random_negatives` positives of other episodes
  + the positive paired with time-reversed frames (optional)
and cross-entropy pushes the positive (index 0) to the top.

Validation mirrors the leaderboard: 1 positive + 4 hard negatives of distinct types,
scored with 1 / 0.5 / 0.25 / 0.125 / 0 by the rank of the positive. Reported for the
full GT ("full_*") and first-sentence GT ("short_*"), plus per-type pass rates
(share of negatives of that type ranked below the positive).
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from model_v2 import OrderAwareScorer

ROOT = Path(__file__).resolve().parent
RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]


class Data:
    def __init__(self, frames_path, texts_path, device):
        f = torch.load(frames_path, map_location="cpu", weights_only=False)
        t = torch.load(texts_path, map_location="cpu", weights_only=False)
        if f["episode_ids"] != t["episode_ids"]:
            raise ValueError("frames.pt and texts.pt were built for different episodes.")
        off = f["frame_offsets"]
        n = len(off) - 1
        lengths = off[1:] - off[:-1]
        tmax = int(lengths.max())
        frames = torch.zeros(n, tmax, f["frames"].shape[1], dtype=torch.float16)
        for i in range(n):
            frames[i, :lengths[i]] = f["frames"][off[i]:off[i + 1]]
        self.device = device
        self.ids = f["episode_ids"]
        self.lengths = lengths.tolist()
        self.frames = frames.to(device)
        self.full = t["full"].to(device)
        self.clause_flat = t["clause_flat"].to(device)
        self.clause_off = t["clause_offsets"].to(device)
        self.groups = t["groups"]
        self.texts = t["texts"]

    def batch(self, rows, views=None):
        """rows: list of lists of (video, reversed, text id) -> model inputs.
        views: optional per-row list of kept frame indices (frame-drop augmentation);
        every candidate of a row sees the same frames."""
        flat, order = [], []
        for r, row in enumerate(rows):
            for c in row:
                kept = views[r] if views and views[r] else list(range(self.lengths[c[0]]))
                flat.append(c)
                order.append(kept[::-1] if c[1] else kept)
        tmax = max(len(o) for o in order)
        tidx = torch.tensor([o + [o[-1]] * (tmax - len(o)) for o in order], device=self.device)
        fmask = torch.tensor([[True] * len(o) + [False] * (tmax - len(o)) for o in order],
                             device=self.device)
        v = torch.tensor([c[0] for c in flat], device=self.device)
        tid = torch.tensor([c[2] for c in flat], device=self.device)
        frames = self.frames[v].gather(1, tidx.unsqueeze(-1).expand(-1, -1,
                                                                     self.frames.shape[2]))
        start, end = self.clause_off[tid], self.clause_off[tid + 1]
        n = end - start
        k = int(n.max())
        pos = start.unsqueeze(1) + torch.arange(k, device=self.device)
        cmask = torch.arange(k, device=self.device).unsqueeze(0) < n.unsqueeze(1)
        clauses = self.clause_flat[pos.clamp(max=len(self.clause_flat) - 1)]
        clauses = clauses * cmask.unsqueeze(-1)
        return (frames.float(), fmask, self.full[tid].float(), clauses.float(), cmask)


def score(model, data, rows, views=None):
    return model(*data.batch(rows, views)).view(len(rows), -1)


def frame_views(data, idx, cfg, rng):
    """Frame-drop augmentation: with `frame_drop_prob`, keep a random ordered subset of
    at least `frame_keep_min` of the frames (and >= 4), so the scorer does not rely on
    one exact frame count / spacing. None = all frames."""
    views = []
    for i in idx:
        n = data.lengths[i]
        if rng.random() >= cfg.get("frame_drop_prob", 0.0) or n <= 4:
            views.append(None)
            continue
        keep = rng.randint(max(4, round(n * cfg.get("frame_keep_min", 0.5))), n - 1)
        views.append(sorted(rng.sample(range(n), keep)))
    return views


def train_rows(data, idx, pool, cfg, rng):
    rows = []
    for i in idx:
        groups = data.groups[i]
        # Short view: any single-sentence group (v3 banks have one per sentence).
        g = (rng.randrange(1, len(groups))
             if len(groups) > 1 and rng.random() < cfg["short_group_prob"] else 0)
        group = groups[g]
        # Meaning-preserving rewrite of the positive (v3 banks), so edits are not a tell.
        pos = group["pos"]
        if group.get("variants") and rng.random() < cfg.get("pos_variant_prob", 0.0):
            pos = rng.choice(group["variants"])
        # Leave-one-type-out: types in `exclude_types` are never trained on, only validated,
        # to see whether the scorer rejects a kind of negative it has never seen.
        pool_negs = [n for n in group["negs"] if n[0] not in cfg.get("exclude_types", [])]
        negs = rng.sample(pool_negs, min(cfg["hard_negatives"], len(pool_negs)))
        row = [(i, False, pos)] + [(i, False, t) for _, t in negs]
        need = cfg["hard_negatives"] + cfg["random_negatives"] - len(negs)
        while need:
            j = rng.choice(pool)
            if j != i:
                other = data.groups[j]
                row.append((i, False, other[min(g, len(other) - 1)]["pos"]))
                need -= 1
        if cfg["reverse_video_negative"]:
            row.append((i, True, pos))
        rows.append(row)
    return rows


def valid_rows(data, valid, group_index, seed):
    """1 positive + 4 negatives of distinct types (random-other fallback)."""
    rng = random.Random(seed)
    rows, kinds = [], []
    for i in valid:
        if group_index >= len(data.groups[i]):
            continue
        group = data.groups[i][group_index]
        by_type = defaultdict(list)
        for k, t in group["negs"]:
            by_type[k].append(t)
        types = rng.sample(sorted(by_type), min(4, len(by_type)))
        row, kind = [(i, False, group["pos"])], ["pos"]
        for k in types:
            row.append((i, False, rng.choice(by_type[k])))
            kind.append(k)
        while len(row) < 5:
            j = rng.choice(valid)
            if j != i:
                row.append((i, False, data.groups[j][0]["pos"]))
                kind.append("random")
        rows.append(row)
        kinds.append(kind)
    return rows, kinds


def shuffle_videos(rows, valid, seed):
    """Same questions, but every candidate is paired with ANOTHER episode's video.
    A model that really uses the video drops to chance (0.375); what it keeps is
    what it solves from the text alone (artifacts of our negative generator)."""
    rng = random.Random(seed)
    perm = valid[:]
    while any(a == b for a, b in zip(valid, perm)):
        rng.shuffle(perm)
    other = dict(zip(valid, perm))
    return [[(other[c[0]], c[1], c[2]) for c in row] for row in rows]


def text_to_video_rows(data, idx, pool, n_videos, rng):
    """Text -> video ranking: one GT text, its own video first + other videos.
    Text artifacts cannot solve this direction, so it pushes the scorer to use the video."""
    rows = []
    for i in idx:
        others = [j for j in rng.sample(pool, n_videos + 1) if j != i][:n_videos]
        text = data.groups[i][0]["pos"]
        rows.append([(i, False, text)] + [(j, False, text) for j in others])
    return rows


class TextBias(nn.Module):
    """Text-only scorer used only during training (product-of-experts debiasing).
    It absorbs whatever can be solved from the text alone; the main scorer is trained on
    logits + bias.detach(), so it only has to learn the part that needs the video.
    Not exported: inference uses the main scorer alone."""

    def __init__(self, text_dim=1536, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(text_dim + 1), nn.Linear(text_dim + 1, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, full_text, clause_mask):
        n = clause_mask.float().sum(1, keepdim=True) / 8
        return self.net(torch.cat([full_text, n], -1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, data, rows, kinds, prefix, batch=128):
    model.eval()
    acc = lb = 0.0
    passed, seen = defaultdict(int), defaultdict(int)
    for s in range(0, len(rows), batch):
        logits = score(model, data, rows[s:s + batch]).cpu()
        for row_logits, kind in zip(logits, kinds[s:s + batch]):
            rank = int((row_logits[1:] >= row_logits[0]).sum())
            acc += rank == 0
            lb += RANK_SCORE[rank]
            for j, k in enumerate(kind[1:], start=1):
                seen[k] += 1
                passed[k] += int(row_logits[0] > row_logits[j])
    n = max(len(rows), 1)
    out = {f"{prefix}_acc": acc / n, f"{prefix}_lb": lb / n}
    out.update({f"{prefix}_vs_{k}": round(passed[k] / seen[k], 4) for k in sorted(seen)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--suffix", default="", help="cache suffix, e.g. _smoke")
    parser.add_argument("--texts-suffix", help="texts cache suffix if different, e.g. _x4")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. frame_drop_prob=0.5")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--run-name", help="wandb run name (default: output dir name)")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    if args.epochs:
        cfg["epochs"] = args.epochs
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = json.loads(value)
    texts_suffix = args.suffix if args.texts_suffix is None else args.texts_suffix
    cfg["texts_cache"] = f"texts{texts_suffix}.pt"
    out = args.output_dir or base / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(f"Use a new --output-dir: {out}")
    torch.manual_seed(cfg["seed"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cache = base / cfg["cache_dir"]
    data = Data(cache / f"frames{args.suffix}.pt", cache / cfg["texts_cache"], device)

    n = len(data.ids)
    order = list(range(n))
    random.Random(cfg["seed"]).shuffle(order)
    n_valid = round(n * cfg["validation_fraction"])
    valid, train = sorted(order[:n_valid]), sorted(order[n_valid:])
    (out / "splits.json").write_text(json.dumps(
        {"train": [data.ids[i] for i in train], "valid": [data.ids[i] for i in valid]}),
        encoding="utf-8")
    val_sets = {"full": valid_rows(data, valid, 0, cfg["seed"] + 1),
                "short": valid_rows(data, valid, 1, cfg["seed"] + 2)}
    for name in ["full", "short"]:
        rows, kinds = val_sets[name]
        val_sets[f"{name}_shuf"] = (shuffle_videos(rows, valid, cfg["seed"] + 3), kinds)

    model_config = {"frame_dim": data.frames.shape[2], "text_dim": data.full.shape[1],
                    "d_model": cfg["d_model"], "layers": cfg["layers"],
                    "heads": cfg["heads"], "dropout": cfg["dropout"]}
    model = OrderAwareScorer(**model_config).to(device)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=cfg.get("wandb_project", "inthon2026"),
                         name=args.run_name or out.name, dir=str(out),
                         config={**cfg, **model_config, "train_episodes": len(train),
                                 "valid_episodes": len(valid)})
    debias = cfg.get("debias_weight", 0.0) > 0
    bias = TextBias(data.full.shape[1]).to(device) if debias else None
    params = list(model.parameters()) + (list(bias.parameters()) if debias else [])
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * ((len(train) + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["learning_rate"], total_steps=steps,
                                                pct_start=0.1)
    ce = nn.CrossEntropyLoss()
    rng = random.Random(cfg["seed"])
    best, stale, history = -1.0, 0, []
    print(f"device {device}; train {len(train)}; valid {len(valid)}", flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = train[:]
        rng.shuffle(perm)
        loss_sum = 0.0
        for s in range(0, len(perm), cfg["batch_size"]):
            idx = perm[s:s + cfg["batch_size"]]
            target0 = torch.zeros(len(idx), dtype=torch.long, device=device)
            rows = train_rows(data, idx, train, cfg, rng)
            inputs = data.batch(rows, frame_views(data, idx, cfg, rng))
            logits = model(*inputs).view(len(rows), -1)
            if debias:
                # Product of experts: the main scorer sees the text-only bias as a fixed
                # offset, so it gains nothing from re-learning text artifacts.
                b = bias(inputs[2], inputs[4]).view(len(rows), -1)
                loss = ce(logits + b.detach(), target0) + cfg["debias_weight"] * ce(b, target0)
            else:
                loss = ce(logits, target0)
            if cfg.get("t2v_weight", 0.0) > 0:
                t2v = score(model, data, text_to_video_rows(data, idx, train,
                                                            cfg.get("t2v_videos", 7), rng))
                loss = loss + cfg["t2v_weight"] * ce(t2v, target0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            if run:
                run.log({"train/step_loss": loss.item(), "train/lr": sched.get_last_lr()[0]})
        metrics = {"epoch": epoch, "train_loss": round(loss_sum / len(train), 4)}
        for name, (rows, kinds) in val_sets.items():
            res = evaluate(model, data, rows, kinds, name)
            if name.endswith("_shuf"):  # only the score matters for the shuffled-video check
                res = {f"{name}_lb": res[f"{name}_lb"]}
            metrics.update({k: round(v, 4) if isinstance(v, float) else v
                            for k, v in res.items()})
        for name in ["full", "short"]:
            # Video-grounded part of the score: what is lost when the video is wrong.
            metrics[f"{name}_gap"] = round(metrics[f"{name}_lb"] - metrics[f"{name}_shuf_lb"], 4)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        # Select on the mean of both views: test descriptions may be short or long.
        target = (metrics["full_lb"] + metrics["short_lb"]) / 2
        if cfg.get("select_metric") == "gap":
            target = (metrics["full_gap"] + metrics["short_gap"]) / 2
        if run:
            # "full_vs_order" -> "full/vs_order" so wandb groups panels by view.
            run.log({"epoch": epoch, "train/loss": metrics["train_loss"], "val/mean_lb": target,
                     **{k.replace("_", "/", 1): v for k, v in metrics.items()
                        if k.startswith(("full_", "short_"))}})
        if target > best:
            best, stale = target, 0
            torch.save({"format_version": 2, "model_config": model_config,
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": cfg, "epoch": epoch, "metrics": metrics}, out / "best.pt")
        else:
            stale += 1
            if stale >= cfg["patience"]:
                print("Early stopping.", flush=True)
                break
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best mean lb score {best:.4f} -> {out / 'best.pt'}", flush=True)
    if run:
        run.summary["best_mean_lb"] = best
        run.finish()


if __name__ == "__main__":
    main()
