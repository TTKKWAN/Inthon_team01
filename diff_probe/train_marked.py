"""Train a marked-token video verifier and validate it on the local 5-choice sets.

Frozen encoders, small trained head:
  video  cached per-frame features [DINOv2 CLS ; DINOv2 mean patch ; SigLIP] (2304-d,
         baseline_v2/cache/frames.pt) -> linear + frame-order embedding -> transformer
  text   frozen MPNet token states of the WHOLE choice + a learned "marked" embedding on
         the words that differ from the other choices (marks.py)
  head   text tokens self-attend and cross-attend to the frame tokens; a per-token score
         is averaged over the marked tokens only, so a one-word edit in a 50-word
         description is not diluted, and an added clause is not favoured for its length

Losses
  choice : listwise CE over the 5 choices of an episode (1 positive, 4 rule negatives)
  video  : the positive text against [own video, own video reversed, 3 other videos];
           the text is identical across these, so only the frames can solve it

Rule negatives come from baseline_v2/data/negatives.json (code-generated). Validation
uses diff_probe/valsets.json, built from the held-out validation episodes.

    python diff_probe/train_marked.py --out diff_probe/runs/marked_v1
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from centrality import score_span  # noqa: E402
from marks import mark_spans  # noqa: E402
from sim_valsets import merge_edits  # noqa: E402

RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]
KEEP = ["antonym", "spatial", "attribute", "order", "append_plausible", "drop_corrupt"]
FALLBACK = ["object", "append_unrelated", "replace_clause"]
MARKS = {"mode": "slots"}  # "slots": same places marked in every choice; "ref": old scheme
EVAL_SETS = ["short/H1_disjoint", "short/H2_samespot", "short/H6_fact_a_b_ab_c",
             "short/H7_fact_a_b_ab_ac", "full/H1_disjoint", "full/H5_mix"]


def is_bad(pos, neg):
    """baseline_v2 antonym rule reads the direction 'left' as the verb 'leave'."""
    pw, nw = set(pos.lower().split()), set(neg.lower().replace(".", " ").split())
    return "left" in pos.lower() and bool({"entered", "entering", "enters", "enter"} & (nw - pw))


# ---- data ---------------------------------------------------------------------------


class Frames:
    def __init__(self, path, max_frames):
        d = torch.load(path)
        self.index = {e: i for i, e in enumerate(d["episode_ids"])}
        self.frames, self.offsets = d["frames"], d["frame_offsets"]
        self.max_frames = max_frames

    def get(self, episode):
        i = self.index[episode]
        f = self.frames[self.offsets[i]:self.offsets[i + 1]]
        if len(f) > self.max_frames:
            idx = [round(k * (len(f) - 1) / (self.max_frames - 1)) for k in range(self.max_frames)]
            f = f[idx]
        return f


def make_group(rng, pos, negs, p_combo):
    """4 negatives: prefer KEEP types, sometimes combine two edits (factorial structure)."""
    pool = [t for k, t in negs if k in KEEP and not is_bad(pos, t)]
    extra = [t for k, t in negs if k in FALLBACK and not is_bad(pos, t)]
    rng.shuffle(pool)
    rng.shuffle(extra)
    chosen = []
    if len(pool) >= 2 and rng.random() < p_combo:
        a, b = pool[0], pool[1]
        m = merge_edits(pos, a, b)
        if m and m not in (pos, a, b):
            chosen = [a, b, m]
    for t in pool + extra:
        if len(chosen) == 4:
            break
        if t not in chosen and t != pos:
            chosen.append(t)
    return chosen if len(chosen) == 4 else None


class TrainData:
    def __init__(self, negatives, episodes, frames, seed):
        self.items = [(ep, g["pos"], g["negs"]) for ep in episodes
                      for g in negatives[ep] if ep in frames.index]
        self.frames = frames
        self.rng = random.Random(seed)

    def batches(self, batch_size, p_combo):
        order = list(range(len(self.items)))
        self.rng.shuffle(order)
        for s in range(0, len(order) - batch_size + 1, batch_size):
            batch = []
            for i in order[s:s + batch_size]:
                ep, pos, negs = self.items[i]
                group = make_group(self.rng, pos, negs, p_combo)
                if group is None:
                    continue
                texts = [pos] + group
                perm = list(range(5))
                self.rng.shuffle(perm)
                batch.append({"episode": ep, "texts": [texts[p] for p in perm],
                              "label": perm.index(0), "frames": self.frames.get(ep)})
            if len(batch) >= 2:
                yield batch


# ---- model --------------------------------------------------------------------------


class TextEncoder:
    """Frozen MPNet token states with a 0/1 mark per token."""

    def __init__(self, path, device):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=torch.float16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, texts, spans):
        b = self.tok(texts, return_offsets_mapping=True, return_tensors="pt", padding=True,
                     truncation=True, max_length=256)
        offsets = b.pop("offset_mapping")
        marks = torch.zeros(offsets.shape[:2], dtype=torch.long)
        for n, sp in enumerate(spans):
            for s, e in sp:
                hit = (offsets[n, :, 1] > s) & (offsets[n, :, 0] < e) & (offsets[n, :, 1] > offsets[n, :, 0])
                marks[n][hit] = 1
        b = b.to(self.device)
        h = self.model(**b).last_hidden_state.float()
        return h, b["attention_mask"].bool(), marks.to(self.device)


class MarkedVerifier(nn.Module):
    def __init__(self, frame_dim=2304, text_dim=768, d=256, layers=2, heads=4,
                 max_frames=32, dropout=0.1):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.frame_pos = nn.Embedding(max_frames, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.frame_enc = nn.TransformerEncoder(enc, layers)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.mark_emb = nn.Embedding(2, d)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.text_dec = nn.TransformerDecoder(dec, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def encode_video(self, frames, mask):
        t = torch.arange(frames.shape[1], device=frames.device)
        x = self.frame_proj(frames) + self.frame_pos(t)[None]
        return self.frame_enc(x, src_key_padding_mask=~mask)

    def score(self, text_h, text_mask, marks, mem, mem_mask):
        x = self.text_proj(text_h) + self.mark_emb(marks)
        h = self.text_dec(x, mem, tgt_key_padding_mask=~text_mask, memory_key_padding_mask=~mem_mask)
        s = self.out(h).squeeze(-1)
        w = marks.float() * text_mask.float()
        empty = w.sum(1) == 0
        w[empty] = text_mask[empty].float()  # no marked token: fall back to all tokens
        return (s * w).sum(1) / w.sum(1)


def pad_frames(list_of_frames, device, reverse=False):
    t = max(len(f) for f in list_of_frames)
    x = torch.zeros(len(list_of_frames), t, list_of_frames[0].shape[1])
    m = torch.zeros(len(list_of_frames), t, dtype=torch.bool)
    for i, f in enumerate(list_of_frames):
        f = f.flip(0) if reverse else f
        x[i, :len(f)] = f.float()
        m[i, :len(f)] = True
    return x.to(device), m.to(device)


# ---- train / eval -------------------------------------------------------------------


def step_loss(model, text_enc, batch, device, n_other=3, w_video=1.0):
    b = len(batch)
    texts = [t for s in batch for t in s["texts"]]
    spans = [sp for s in batch for sp in mark_spans(s["texts"], MARKS["mode"])]
    h, tm, mk = text_enc(texts, spans)
    frames = [s["frames"] for s in batch]
    fx, fm = pad_frames(frames, device)
    rx, rm = pad_frames(frames, device, reverse=True)
    mem = model.encode_video(torch.cat([fx, rx]), torch.cat([fm, rm]))
    mmask = torch.cat([fm, rm])

    vid = torch.arange(b, device=device).repeat_interleave(5)
    choice_scores = model.score(h, tm, mk, mem[vid], mmask[vid]).view(b, 5)
    labels = torch.tensor([s["label"] for s in batch], device=device)
    loss_choice = F.cross_entropy(choice_scores, labels)

    pos_rows = torch.arange(b, device=device) * 5 + labels
    vids, rows = [], []
    for i in range(b):
        others = random.sample([j for j in range(b) if j != i], min(n_other, b - 1))
        cand = [i, b + i] + others  # own, own reversed, other episodes
        vids += cand
        rows += [pos_rows[i].item()] * len(cand)
    vids = torch.tensor(vids, device=device)
    rows = torch.tensor(rows, device=device)
    k = 2 + min(n_other, b - 1)
    video_scores = model.score(h[rows], tm[rows], mk[rows], mem[vids], mmask[vids]).view(b, k)
    loss_video = F.cross_entropy(video_scores, torch.zeros(b, dtype=torch.long, device=device))
    acc = (choice_scores.argmax(1) == labels).float().mean().item()
    return loss_choice + w_video * loss_video, loss_choice.item(), loss_video.item(), acc


@torch.no_grad()
def evaluate(model, text_enc, rows, frames, device, condition="normal", seed=0, batch=64):
    """condition: normal | reversed | other (frames of a different validation episode)."""
    model.eval()
    rng = random.Random(seed)
    episodes = [r["episode"] for r in rows]
    total, top1, total_cent, n = 0.0, 0, 0.0, 0
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        texts, spans, labels, fr, cents = [], [], [], [], []
        for k, r in enumerate(chunk):
            ch = list(r["choices"])
            rng.shuffle(ch)
            t = [c["text"] for c in ch]
            texts += t
            spans += mark_spans(t, MARKS["mode"])
            labels.append(next(i for i, c in enumerate(ch) if c["type"] == "pos"))
            cents.append(score_span(t))
            ep = r["episode"]
            if condition == "other":
                ep = episodes[(s + k + len(rows) // 2) % len(rows)]
            fr.append(frames.get(ep))
        h, tm, mk = text_enc(texts, spans)
        fx, fm = pad_frames(fr, device, reverse=(condition == "reversed"))
        mem = model.encode_video(fx, fm)
        vid = torch.arange(len(chunk), device=device).repeat_interleave(5)
        sc = model.score(h, tm, mk, mem[vid], fm[vid]).view(len(chunk), 5).tolist()
        for row_scores, lab, cent in zip(sc, labels, cents):
            order = sorted(range(5), key=lambda i: -row_scores[i])
            total += RANK_SCORE[order.index(lab)]
            top1 += order[0] == lab
            corder = sorted(range(5), key=lambda i: -cent[i])
            total_cent += RANK_SCORE[corder.index(lab)]
            n += 1
    model.train()
    return {"score": round(total / n, 4), "top1": round(top1 / n, 4),
            "centrality": round(total_cent / n, 4), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--negatives", type=Path, default=ROOT / "baseline_v2/data/negatives.json")
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--valsets", type=Path, default=HERE / "valsets.json")
    ap.add_argument("--text-model", default=str(HERE / "models/mpnet"))
    ap.add_argument("--out", type=Path, default=HERE / "runs/marked_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--p-combo", type=float, default=0.3)
    ap.add_argument("--w-video", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--marks", choices=["slots", "ref"], default="slots")
    ap.add_argument("--eval-limit", type=int, default=0, help="debug: episodes per val set")
    args = ap.parse_args()

    MARKS["mode"] = args.marks
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    frames = Frames(args.frames, args.max_frames)
    negatives = json.loads(args.negatives.read_text())["episodes"]
    splits = json.loads(args.splits.read_text())
    valsets = json.loads(args.valsets.read_text())["sets"]
    if args.eval_limit:
        valsets = {k: v[:args.eval_limit] for k, v in valsets.items()}
    data = TrainData(negatives, splits["train"], frames, args.seed)
    text_enc = TextEncoder(args.text_model, device)
    model = MarkedVerifier(max_frames=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(data.items) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1))))
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    print(json.dumps({"train_items": len(data.items), "steps_per_epoch": steps_per_epoch,
                      "total_steps": total_steps, "params": sum(p.numel() for p in model.parameters())}),
          flush=True)

    log, best, step = [], -1.0, 0
    sel_sets = ["short/H2_samespot", "short/H6_fact_a_b_ab_c", "short/H7_fact_a_b_ab_ac"]
    for epoch in range(1, args.epochs + 1):
        t0, agg = time.time(), [0.0, 0.0, 0.0, 0]
        for batch in data.batches(args.batch_size, args.p_combo):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                loss, lc, lv, acc = step_loss(model, text_enc, batch, device, w_video=args.w_video)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg = [agg[0] + lc, agg[1] + lv, agg[2] + acc, agg[3] + 1]
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val = {name: evaluate(model, text_enc, valsets[name], frames, device) for name in sel_sets}
        sel = sum(v["score"] for v in val.values()) / len(val)
        rec = {"epoch": epoch, "step": step, "loss_choice": round(agg[0] / agg[3], 4),
               "loss_video": round(agg[1] / agg[3], 4), "train_acc": round(agg[2] / agg[3], 4),
               "val": {k: v["score"] for k, v in val.items()}, "val_mean": round(sel, 4),
               "seconds": round(time.time() - t0, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args) | {"out": str(args.out)},
                    "epoch": epoch, "step": step}, args.out / f"epoch{epoch:02d}.pt")
        if sel > best:
            best = sel
            torch.save({"model": model.state_dict(), "epoch": epoch, "step": step},
                       args.out / "best.pt")
        if args.max_steps and step >= args.max_steps:
            break

    model.load_state_dict(torch.load(args.out / "best.pt")["model"])
    final = {}
    for name in EVAL_SETS:
        final[name] = {c: evaluate(model, text_enc, valsets[name], frames, device, c)
                       for c in ("normal", "reversed", "other")}
        print(name, json.dumps(final[name]), flush=True)
    (args.out / "results.json").write_text(json.dumps({"log": log, "final": final}, indent=1))


if __name__ == "__main__":
    main()
