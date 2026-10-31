"""pair_head.py with spatial input: each frame also gives its DINOv2 4x4 patch grid.

Per frame: 1 global token (cached [DINOv2 CLS ; DINOv2 mean patch ; SigLIP image]) + 16 grid
tokens (baseline_v2/cache/patches4.pt), each grid token also gets its change from the previous
frame (token_t - token_{t-1}), so "where something appeared or disappeared" is visible.
Frame-position and grid-position embeddings; a transformer over all T*17 tokens; clauses
cross-attend to them. Losses, data, splits and evaluation are those of pair_head.py (no reversed
or joined-clip terms).

    python diff_probe/patch_head.py --hf diff_probe/runs/qwen_negatives_from_gt.json --out diff_probe/runs/patch_head
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
from pair_head import MAX_CLAUSES, ClauseBank, FrameBank, load_groups  # noqa: E402

GRID_TOKENS = 16


class PatchBank:
    def __init__(self, path, frames):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        assert cache["episode_ids"] == list(frames.index), "patch cache episode order differs"
        self.patches, self.offsets, self.index = cache["patches"], cache["frame_offsets"], frames.index
        self.tmax = frames.tmax

    def batch(self, episodes, device):
        x = torch.zeros(len(episodes), self.tmax, GRID_TOKENS, self.patches.shape[-1])
        for b, e in enumerate(episodes):
            i = self.index[e]
            p = self.patches[self.offsets[i]:self.offsets[i + 1]].float()
            x[b, :len(p)] = p
        return x.to(device)


class PatchHead(nn.Module):
    def __init__(self, frame_dim=2304, patch_dim=768, text_dim=768, d=256, layers=2, heads=4,
                 tmax=64, dropout=0.1):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.patch_proj = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, d))
        self.delta_proj = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, d))
        self.frame_pos = nn.Parameter(torch.zeros(tmax, d))
        self.slot_pos = nn.Parameter(torch.zeros(1 + GRID_TOKENS, d))  # 0 = global, 1.. = grid
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.clause_pos = nn.Parameter(torch.zeros(MAX_CLAUSES, d))
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.query = nn.Linear(d, d)
        for p in (self.frame_pos, self.slot_pos, self.clause_pos):
            nn.init.normal_(p, std=0.02)

    def encode_video(self, frames, fmask, patches):
        """frames [B,T,2304], fmask [B,T] (True = pad), patches [B,T,16,768] -> tokens, token mask."""
        B, T = fmask.shape
        delta = patches - torch.cat([patches[:, :1], patches[:, :-1]], 1)  # 0 at t = 0
        grid = self.patch_proj(patches) + self.delta_proj(delta)            # [B,T,16,d]
        tokens = torch.cat([self.frame_proj(frames)[:, :, None], grid], 2)  # [B,T,17,d]
        tokens = tokens + self.frame_pos[:T, None] + self.slot_pos[None, None]
        tmask = fmask[:, :, None].expand(B, T, 1 + GRID_TOKENS).reshape(B, -1)
        h = self.encoder(tokens.reshape(B, T * (1 + GRID_TOKENS), -1), src_key_padding_mask=tmask)
        return h, tmask

    def forward(self, video, vmask, clauses, cmask):
        q = self.text_proj(clauses) + self.clause_pos[:clauses.shape[1]]
        summary, _ = self.cross(q, video, video, key_padding_mask=vmask)
        per = (self.query(q) * summary).sum(-1) / q.shape[-1] ** 0.5
        keep = (~cmask).float()
        return (per * keep).sum(1) / keep.sum(1).clamp_min(1)


def video_of(model, frames, patches, episodes, device):
    fr, fm = frames.batch(episodes, device)
    return model.encode_video(fr, fm, patches.batch(episodes, device))


@torch.no_grad()
def evaluate(model, groups, frames, patches, clauses, device, other):
    model.eval()
    hits = defaultdict(list)
    pairs = [(g, k, t) for g in groups for k, t in g["negs"]]
    for s in range(0, len(pairs), 128):
        chunk = pairs[s:s + 128]
        for video in ("real", "other"):
            eps = [g["episode"] if video == "real" else other[g["episode"]] for g, _, _ in chunk]
            v, vm = video_of(model, frames, patches, eps, device)
            sp = model(v, vm, *clauses.batch([g["pos"] for g, _, _ in chunk], device))
            sn = model(v, vm, *clauses.batch([t for _, _, t in chunk], device))
            for (g, k, _), a, b in zip(chunk, sp.tolist(), sn.tolist()):
                hit = 1.0 if a > b else 0.5 if a == b else 0.0
                half = "A" if int(g["episode"].split("_")[1]) % 2 == 0 else "B"
                for kind in (k, "all"):
                    hits[f"{video}/{g['view']}/{kind}/{half}"].append(hit)
    model.train()
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(hits.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--patches", type=Path, default=ROOT / "baseline_v2/cache/patches4.pt")
    ap.add_argument("--siglip", type=Path, default=ROOT / "baseline_v2/models/siglip")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mu", type=float, default=1.0)
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
    patches = PatchBank(args.patches, frames)
    clauses = ClauseBank(texts, args.siglip, args.device)
    valid_eps = sorted({g["episode"] for g in groups["valid"]}, key=lambda e: int(e.split("_")[1]))
    other = {e: valid_eps[(n + 1) % len(valid_eps)] for n, e in enumerate(valid_eps)}
    print(f"train groups {len(groups['train'])}, valid groups {len(groups['valid'])}, tmax {frames.tmax}", flush=True)

    model = PatchHead(tmax=frames.tmax).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(groups["train"]) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    log = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(groups["train"])
        total = 0.0
        for s in range(0, len(groups["train"]) - args.batch + 1, args.batch):
            batch = groups["train"][s:s + args.batch]
            v, vm = video_of(model, frames, patches, [g["episode"] for g in batch], args.device)
            v_other, vm_other = v.roll(1, 0), vm.roll(1, 0)
            pos = clauses.batch([g["pos"] for g in batch], args.device)
            neg = clauses.batch([random.choice(g["negs"])[1] for g in batch], args.device)
            sp, sn = model(v, vm, *pos), model(v, vm, *neg)
            spo, sno = model(v_other, vm_other, *pos), model(v_other, vm_other, *neg)
            loss = (F.softplus(sn - sp).mean() + 0.5 * F.softplus(spo - sp).mean()
                    + args.mu * ((spo - sno) ** 2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        res = evaluate(model, groups["valid"], frames, patches, clauses, args.device, other)
        log.append({"epoch": epoch, "loss": total, "valid": res})
        torch.save({"state_dict": model.state_dict(), "tmax": frames.tmax, "epoch": epoch},
                   args.out / f"epoch{epoch:02d}.pt")
        line = " ".join(f"{v}/{view}={res[f'{v}/{view}/all/A']:.3f}"
                        for v in ("real", "other") for view in ("full", "short"))
        print(f"epoch {epoch:2d} loss {total:.1f}  half A: {line}", flush=True)
    (args.out / "log.json").write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
