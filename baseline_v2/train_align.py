"""Pretrain ClipMatcher: learn to place each clause of a description at a clip of its video.

Signals (none of them solvable from the text alone):
  retrieval  each GT clause must match its own video better than the other videos in the
             batch (clause score per video = soft max over that video's clips)
  order      the clauses of a GT description must peak at increasing times
  action     per-frame action labels from episode.json (via manifest.json frame ids),
             multi-label over 63 classes, so clip vectors encode what is being done

Validation (episodes held out in splits.json):
  order_rate   share of GT descriptions with >= 3 clauses whose clause peaks are strictly
               in time order (frozen SigLIP: 11.7%, random ~16.7%)
  retrieval@1  clause -> video among all validation videos
  action_acc   top-1 action of frames inside an annotated action

  python train_align.py --output-dir outputs/align_pre
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from model_v2 import ClipMatcher
from train_v2 import Data

ROOT = Path(__file__).resolve().parent
N_ACTIONS = 63


def frame_action_labels(data, train_dir, manifest_path):
    """[N, T, 63] multi-hot: which annotated actions each kept frame falls inside."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frame_ids = {f"episode_{e['episode_id']}": [im["frame_id"] for im in e["images"]]
                 for e in manifest["episodes"]}
    n, t = data.frames.shape[:2]
    labels = torch.zeros(n, t, N_ACTIONS)
    for i, ep in enumerate(data.ids):
        actions = json.loads((train_dir / ep / "episode.json").read_text(encoding="utf-8"))
        for k, fid in enumerate(frame_ids[ep]):
            for a in actions["actions"]:
                if a["frame_ids"][0] <= fid <= a["frame_ids"][1]:
                    labels[i, k, a["label_id"]] = 1
    return labels.to(data.device)


def clause_batch(data, idx):
    """GT (group 0) clause features [B,K,D] + mask for the given episodes."""
    tids = torch.tensor([data.groups[i][0]["pos"] for i in idx], device=data.device)
    start, end = data.clause_off[tids], data.clause_off[tids + 1]
    k = int((end - start).max())
    pos = start.unsqueeze(1) + torch.arange(k, device=data.device)
    mask = torch.arange(k, device=data.device).unsqueeze(0) < (end - start).unsqueeze(1)
    feats = data.clause_flat[pos.clamp(max=len(data.clause_flat) - 1)] * mask.unsqueeze(-1)
    return feats.float(), mask


def frame_batch(data, idx):
    idx_t = torch.tensor(idx, device=data.device)
    t = max(data.lengths[i] for i in idx)
    frames = data.frames[idx_t, :t].float()
    mask = torch.arange(t, device=data.device).unsqueeze(0) < torch.tensor(
        [data.lengths[i] for i in idx], device=data.device).unsqueeze(1)
    return frames, mask, idx_t, t


def losses(model, data, idx, labels, cfg):
    frames, fmask, idx_t, t = frame_batch(data, idx)
    clauses, cmask = clause_batch(data, idx)
    clip, raw = model.clips(frames, fmask)
    cv = model.clause_vectors(clauses)
    scale = model.logit_scale.exp()
    # Retrieval: every clause against every video's clips -> [B,K,B,T].
    sim = torch.einsum("ikd,jtd->ikjt", cv, clip) * scale
    sim = sim.masked_fill(~fmask[None, None], -1e4)
    video_score = torch.logsumexp(sim, dim=-1)  # soft max over clips -> [B,K,B]
    logits = video_score[cmask]  # [n_clauses, B]
    target = torch.arange(len(idx), device=data.device).unsqueeze(1).expand(-1, cmask.shape[1])
    l_ret = F.cross_entropy(logits, target[cmask])
    # Order: expected time of each clause in its own video must increase.
    own = sim[torch.arange(len(idx)), :, torch.arange(len(idx))]  # [B,K,T]
    attn = own.softmax(-1)
    rel = torch.arange(t, device=data.device).float().unsqueeze(0) / (
        fmask.sum(1, keepdim=True).float() - 1).clamp_min(1)
    pos = torch.einsum("bkt,bt->bk", attn, rel)
    pair = cmask[:, 1:] & cmask[:, :-1]
    l_order = (F.relu(pos[:, :-1] - pos[:, 1:] + cfg["order_margin"]) * pair).sum() / pair.sum()
    # Action labels per frame (only frames of the batch's real length).
    act_logits = model.action_head(raw)
    l_act = F.binary_cross_entropy_with_logits(act_logits[fmask], labels[idx_t, :t][fmask])
    return l_ret, l_order, l_act


@torch.no_grad()
def evaluate(model, data, valid, labels):
    model.eval()
    order_ok = order_n = ret_ok = ret_n = act_ok = act_n = 0
    clips_all, masks, lens = [], [], []
    for s in range(0, len(valid), 64):
        idx = valid[s:s + 64]
        frames, fmask, idx_t, t = frame_batch(data, idx)
        clip, raw = model.clips(frames, fmask)
        clauses, cmask = clause_batch(data, idx)
        own = torch.einsum("bkd,btd->bkt", model.clause_vectors(clauses), clip)
        own = own.masked_fill(~fmask.unsqueeze(1), -1e4)
        for b, i in enumerate(idx):
            k = int(cmask[b].sum())
            if k >= 3:
                peak = (own[b, :k].argmax(-1).float() / max(data.lengths[i] - 1, 1)).tolist()
                order_ok += all(a < c for a, c in zip(peak, peak[1:]))
                order_n += 1
        lab = labels[idx_t, :t]
        has = (lab.sum(-1) > 0) & fmask
        pred = model.action_head(raw).argmax(-1)
        act_ok += int((lab.gather(-1, pred.unsqueeze(-1)).squeeze(-1)[has] > 0).sum())
        act_n += int(has.sum())
        clips_all.append(F.pad(clip, (0, 0, 0, 20 - t)))
        masks.append(F.pad(fmask, (0, 20 - t)))
    clip_all, mask_all = torch.cat(clips_all), torch.cat(masks)
    for s in range(0, len(valid), 64):
        idx = valid[s:s + 64]
        clauses, cmask = clause_batch(data, idx)
        sim = torch.einsum("ikd,jtd->ikjt", model.clause_vectors(clauses), clip_all)
        sim = sim.masked_fill(~mask_all[None, None], -1e4).amax(-1)  # [B,K,V]
        best = sim.argmax(-1)
        target = torch.arange(s, s + len(idx), device=data.device).unsqueeze(1)
        ret_ok += int(((best == target) & cmask).sum())
        ret_n += int(cmask.sum())
    model.train()
    return {"order_rate": order_ok / order_n, "retrieval@1": ret_ok / ret_n,
            "action_acc": act_ok / max(act_n, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/align_pre")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--texts", default="texts.pt")
    parser.add_argument("--folds", type=int, default=0,
                        help="split train into this many folds (written to folds.json)")
    parser.add_argument("--exclude-fold", type=int,
                        help="train without this fold -> out-of-fold matcher for it")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg.update({"order_margin": 0.05, "w_order": 1.0, "w_act": 0.5, "lr": 3e-4})
    base = args.config.resolve().parent
    cache = base / cfg["cache_dir"]
    train_dir = base / cfg["data_dir"]
    torch.manual_seed(cfg["seed"])
    data = Data(cache / "frames.pt", cache / args.texts, "cuda:0")
    labels = frame_action_labels(data, train_dir, train_dir / "manifest.json")
    index = {e: i for i, e in enumerate(data.ids)}
    split = json.loads((base / "validation/splits.json").read_text(encoding="utf-8"))
    train = [index[e] for e in split["train"]]
    valid = sorted(index[e] for e in split["valid"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "matcher.pt"
    if args.folds:
        # Out-of-fold matchers: the scorer later sees, for each training video, the signal
        # of a matcher that never trained on it, i.e. as noisy as on unseen test videos.
        order = sorted(split["train"])
        random.Random(cfg["seed"] + 7).shuffle(order)
        folds = {e: k % args.folds for k, e in enumerate(order)}
        (args.output_dir / "folds.json").write_text(json.dumps(folds), encoding="utf-8")
        train = [index[e] for e in split["train"] if folds[e] != args.exclude_fold]
        name = f"matcher_fold{args.exclude_fold}.pt"
    model = ClipMatcher(data.frames.shape[2], data.full.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.05)
    rng = random.Random(cfg["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("before training:", json.dumps({k: round(v, 4) for k, v in
                                          evaluate(model, data, valid, labels).items()}))
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        sums = [0.0, 0.0, 0.0]
        for s in range(0, len(train), 64):
            l_ret, l_order, l_act = losses(model, data, train[s:s + 64], labels, cfg)
            loss = l_ret + cfg["w_order"] * l_order + cfg["w_act"] * l_act
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in enumerate((l_ret, l_order, l_act)):
                sums[k] += v.item()
        steps = (len(train) + 63) // 64
        metrics = {"epoch": epoch, "l_ret": round(sums[0] / steps, 4),
                   "l_order": round(sums[1] / steps, 4), "l_act": round(sums[2] / steps, 4),
                   **{k: round(v, 4) for k, v in evaluate(model, data, valid, labels).items()}}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        target = metrics["order_rate"] + metrics["retrieval@1"]
        if target > best:
            best = target
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": {"frame_dim": data.frames.shape[2],
                                   "text_dim": data.full.shape[1]},
                        "epoch": epoch, "metrics": metrics}, args.output_dir / name)
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
