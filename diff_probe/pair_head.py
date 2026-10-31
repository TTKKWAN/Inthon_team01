"""A small video-conditioned judge for GT vs one-edit negatives, trained on Qwen edits.

SigLIP zero-shot (siglip_pairs.py) separates object/attribute edits (~0.75) but not action
direction, order or added actions (~0.5-0.6): it sees one frame at a time. This head reads the
frames in order and scores each clause of a description against them.

  frames  : cached [DINOv2 CLS ; DINOv2 mean patch ; SigLIP image] (frozen), temporal transformer
  clauses : SigLIP text embedding per clause (frozen), clause-order embedding
  score   : clauses cross-attend to frames; mean over clauses of <q(clause), video summary>

Text reaches the score only through the frames, and training adds a penalty that makes GT and
negative score the same on ANOTHER episode's video, so a wording shortcut is punished:
  loss = softplus(s_neg - s_pos)                     own video, GT above the negative
       + 0.5 * softplus(s_pos_other - s_pos)          GT matches its own video better
       + mu * (s_pos_other - s_neg_other)^2           no preference without the right video
  + rev * softplus(s_pos(reversed video) - s_pos)    --rev: same text, frames played backwards
  + concat * [two clips joined A+B and B+A, texts "A, B" and "B, A"; each text above the other
             on its own video]                       --concat: event order (Test of Time, CVPR'23)
The reversed and joined terms compare videos (or texts that are both correct for some video), so
they cannot be solved from wording alone.

Data: Qwen3-4B single edits of the GT (HF SANGchu-03/INThon_Augmented qwen_negatives_from_gt,
see qwen_valsets.py), train episodes of baseline_v2/validation/splits.json; validation episodes
are only evaluated. Each epoch is saved.

    python diff_probe/pair_head.py --hf validator/hf/qwen_negatives_from_gt.json --out diff_probe/runs/pair_head
    python diff_probe/pair_head.py --hf ... --rev 1 --concat 1 --out diff_probe/runs/pair_head_rc
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diff_probe"))
from siglip_pairs import SiglipText, split_clauses  # noqa: E402

MAX_CLAUSES = 8


class FrameBank:
    def __init__(self, path):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        self.index = {e: i for i, e in enumerate(cache["episode_ids"])}
        self.frames, self.offsets = cache["frames"], cache["frame_offsets"]
        self.tmax = int((self.offsets[1:] - self.offsets[:-1]).max())

    def batch(self, episodes, device):
        x = torch.zeros(len(episodes), self.tmax, self.frames.shape[1])
        mask = torch.ones(len(episodes), self.tmax, dtype=torch.bool)  # True = padding
        for b, e in enumerate(episodes):
            i = self.index[e]
            f = self.frames[self.offsets[i]:self.offsets[i + 1]].float()
            x[b, :len(f)], mask[b, :len(f)] = f, False
        return x.to(device), mask.to(device)


class ClauseBank:
    """SigLIP embedding of every clause of every text, encoded once."""

    def __init__(self, texts, siglip_dir, device):
        self.clauses = {t: split_clauses(t)[:MAX_CLAUSES] for t in texts}
        flat = sorted({c for cl in self.clauses.values() for c in cl})
        enc = SiglipText(siglip_dir, device)
        emb = enc.encode(flat)
        self.emb = dict(zip(flat, emb))
        del enc

    def batch(self, texts, device):
        x = torch.zeros(len(texts), MAX_CLAUSES, 768)
        mask = torch.ones(len(texts), MAX_CLAUSES, dtype=torch.bool)
        for b, t in enumerate(texts):
            cl = self.clauses[t]
            x[b, :len(cl)] = torch.stack([self.emb[c] for c in cl])
            mask[b, :len(cl)] = False
        return x.to(device), mask.to(device)


def reverse_frames(x, mask):
    """Play each episode's valid frames backwards (padding stays at the end)."""
    out = x.clone()
    for b in range(x.shape[0]):
        n = int((~mask[b]).sum())
        out[b, :n] = x[b, :n].flip(0)
    return out


def join_frames(x, mask, y, ymask):
    """[x valid frames ; y valid frames] per row, padded to x.T + y.T."""
    out = x.new_zeros(x.shape[0], x.shape[1] + y.shape[1], x.shape[2])
    om = torch.ones(out.shape[:2], dtype=torch.bool, device=x.device)
    for b in range(x.shape[0]):
        n, m = int((~mask[b]).sum()), int((~ymask[b]).sum())
        out[b, :n], out[b, n:n + m] = x[b, :n], y[b, :m]
        om[b, :n + m] = False
    return out, om


def join_clauses(c, cmask, d, dmask):
    """Clauses of text A followed by those of text B, capped at MAX_CLAUSES."""
    out, om = torch.zeros_like(c), torch.ones_like(cmask)
    for b in range(c.shape[0]):
        rows = torch.cat([c[b, ~cmask[b]], d[b, ~dmask[b]]])[:MAX_CLAUSES]
        out[b, :len(rows)], om[b, :len(rows)] = rows, False
    return out, om


class PairHead(nn.Module):
    def __init__(self, frame_dim=2304, text_dim=768, d=256, layers=2, heads=4, tmax=64, dropout=0.1):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.frame_pos = nn.Parameter(torch.zeros(tmax, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.clause_pos = nn.Parameter(torch.zeros(MAX_CLAUSES, d))
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.query = nn.Linear(d, d)
        nn.init.normal_(self.frame_pos, std=0.02)
        nn.init.normal_(self.clause_pos, std=0.02)

    def encode_video(self, frames, fmask):
        h = self.frame_proj(frames) + self.frame_pos[:frames.shape[1]]
        return self.temporal(h, src_key_padding_mask=fmask)

    def forward(self, video, fmask, clauses, cmask):
        """video [B,T,d] (encode_video), clauses [B,K,768] -> score [B]."""
        q = self.text_proj(clauses) + self.clause_pos[:clauses.shape[1]]
        summary, _ = self.cross(q, video, video, key_padding_mask=fmask)
        per = (self.query(q) * summary).sum(-1) / q.shape[-1] ** 0.5
        keep = (~cmask).float()
        return (per * keep).sum(1) / keep.sum(1).clamp_min(1)


def load_groups(hf_path, splits_path):
    episodes = json.loads(Path(hf_path).read_text())["episodes"]
    splits = json.loads(Path(splits_path).read_text())
    out = {}
    for part in ("train", "valid"):
        out[part] = [{"episode": e, "view": "full" if g_i == 0 else "short", "pos": g["pos"],
                      "negs": [(k, t) for k, t in g["negs"] if t != g["pos"]]}
                     for e in splits[part] for g_i, g in enumerate(episodes.get(e, []))]
        out[part] = [g for g in out[part] if g["negs"]]
    return out


@torch.no_grad()
def evaluate(model, groups, frames, clauses, device, other):
    """Pairwise accuracy per (video, view, kind) and per half (even / odd episode id)."""
    model.eval()
    hits = defaultdict(list)
    pairs = [(g, k, t) for g in groups for k, t in g["negs"]]
    for s in range(0, len(pairs), 256):
        chunk = pairs[s:s + 256]
        for video in ("real", "other"):
            eps = [g["episode"] if video == "real" else other[g["episode"]] for g, _, _ in chunk]
            fr, fm = frames.batch(eps, device)
            v = model.encode_video(fr, fm)
            sp = model(v, fm, *clauses.batch([g["pos"] for g, _, _ in chunk], device))
            sn = model(v, fm, *clauses.batch([t for _, _, t in chunk], device))
            for (g, k, _), a, b in zip(chunk, sp.tolist(), sn.tolist()):
                hit = 1.0 if a > b else 0.5 if a == b else 0.0
                half = "A" if int(g["episode"].split("_")[1]) % 2 == 0 else "B"
                for kind in (k, "all"):
                    hits[f"{video}/{g['view']}/{kind}/{half}"].append(hit)
    model.train()
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(hits.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True, help="qwen_negatives_from_gt.json")
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--siglip", type=Path, default=ROOT / "baseline_v2/models/siglip")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--rev", type=float, default=0.0, help="weight of the reversed-video loss")
    ap.add_argument("--concat", type=float, default=0.0, help="weight of the joined-clips order loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exclude", type=Path,
                    help="json with 'ref_eps' (rerank_eval.py --out): keep these Train episodes out of "
                         "training so the text-prior references are videos the head never saw")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    groups = load_groups(args.hf, args.splits)
    if args.exclude:
        held = set(json.loads(args.exclude.read_text())["ref_eps"])
        groups["train"] = [g for g in groups["train"] if g["episode"] not in held]
    texts = {t for part in groups.values() for g in part for t in [g["pos"]] + [x for _, x in g["negs"]]}
    frames = FrameBank(args.frames)
    clauses = ClauseBank(texts, args.siglip, args.device)
    valid_eps = sorted({g["episode"] for g in groups["valid"]}, key=lambda e: int(e.split("_")[1]))
    other = {e: valid_eps[(n + 1) % len(valid_eps)] for n, e in enumerate(valid_eps)}
    print(f"train groups {len(groups['train'])}, valid groups {len(groups['valid'])}, "
          f"texts {len(texts)}, tmax {frames.tmax}", flush=True)

    tmax = 2 * frames.tmax if args.concat else frames.tmax
    model = PairHead(tmax=tmax).to(args.device)
    short_train = [g for g in groups["train"] if g["view"] == "short"]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(groups["train"]) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    log = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(groups["train"])
        total = 0.0
        for s in range(0, len(groups["train"]) - args.batch + 1, args.batch):
            batch = groups["train"][s:s + args.batch]
            eps = [g["episode"] for g in batch]
            fr, fm = frames.batch(eps, args.device)
            v = model.encode_video(fr, fm)
            v_other, fm_other = v.roll(1, 0), fm.roll(1, 0)  # another episode's video
            pos = clauses.batch([g["pos"] for g in batch], args.device)
            neg = clauses.batch([random.choice(g["negs"])[1] for g in batch], args.device)
            sp, sn = model(v, fm, *pos), model(v, fm, *neg)
            spo, sno = model(v_other, fm_other, *pos), model(v_other, fm_other, *neg)
            loss = (F.softplus(sn - sp).mean() + 0.5 * F.softplus(spo - sp).mean()
                    + args.mu * ((spo - sno) ** 2).mean())
            if args.rev:
                v_rev = model.encode_video(reverse_frames(fr, fm), fm)
                loss = loss + args.rev * F.softplus(model(v_rev, fm, *pos) - sp).mean()
            if args.concat:
                pair = random.sample(short_train, args.batch)
                fa, fam = frames.batch([g["episode"] for g in pair], args.device)
                fb, fbm = fa.roll(1, 0), fam.roll(1, 0)
                ta = clauses.batch([g["pos"] for g in pair], args.device)
                tb = (ta[0].roll(1, 0), ta[1].roll(1, 0))
                v_ab = model.encode_video(*join_frames(fa, fam, fb, fbm))
                v_ba = model.encode_video(*join_frames(fb, fbm, fa, fam))
                m_ab, m_ba = join_frames(fa, fam, fb, fbm)[1], join_frames(fb, fbm, fa, fam)[1]
                t_ab, t_ba = join_clauses(*ta, *tb), join_clauses(*tb, *ta)
                loss = loss + args.concat * (
                    F.softplus(model(v_ab, m_ab, *t_ba) - model(v_ab, m_ab, *t_ab)).mean()
                    + F.softplus(model(v_ba, m_ba, *t_ab) - model(v_ba, m_ba, *t_ba)).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        res = evaluate(model, groups["valid"], frames, clauses, args.device, other)
        log.append({"epoch": epoch, "loss": total, "valid": res})
        torch.save({"state_dict": model.state_dict(), "tmax": tmax, "epoch": epoch},
                   args.out / f"epoch{epoch:02d}.pt")
        line = " ".join(f"{v}/{view}={res[f'{v}/{view}/all/A']:.3f}"
                        for v in ("real", "other") for view in ("full", "short"))
        print(f"epoch {epoch:2d} loss {total:.1f}  half A: {line}", flush=True)
    (args.out / "log.json").write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
