"""Score the shipped video-judge submissions on TemporalBench 5-choice cells. EVALUATION ONLY.

Fixed configs only (no λ is chosen on this data):
  span          λ = (0, 0)
  shipped       λ = (0.3, 0.3)   video_judge_patch as submitted
  siglip_only   λ = (0.3, 0)
  head_only     λ = (0, 0.3)
plus the SigLIP-only zip (video_judge_siglip, l2 = 0, its own refs) as siglip_zip.

Submission.score is split into a text part (span base, clauses, the prior over the 64 reference
videos) and a video part (align, head). Each video is encoded once per submission and cached;
the text part is computed once per row and shared by the real video and the other-video
control (frames of the next video of the same source dataset). The first --check rows are
compared against real Submission.predict calls of both zips; any mismatch is reported.

    python external_eval/temporalbench/score.py --patch diff_probe/runs/vjp_unzip \
        --siglip-sub diff_probe/runs/vjs_unzip --out external_eval/temporalbench/results/score.json
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "validator"))
from eval_subs import RANK_SCORE, load_images, shuffled  # noqa: E402

CELLS = ["single", "ab_ab_c", "ab_ab_ac", "share4"]
CONFIGS = {"span": (0, 0), "shipped": (0.3, 0.3), "siglip_only": (0.3, 0), "head_only": (0, 0.3)}


def load_module(folder, name):
    """Import folder/submission.py under its own name so two submissions can coexist."""
    spec = importlib.util.spec_from_file_location(name, folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    spec.loader.exec_module(module)
    sys.path.remove(str(folder))
    for dep in ("centrality", "hyuwooma_bytes", "judge", "model_v2"):  # keep each zip's own copies
        sys.modules.pop(dep, None)
    return module


def partners(rows):
    """Other-video control: the next video (sorted) of the same source dataset."""
    by_ds = defaultdict(list)
    for ep, ds in sorted({(r["episode"], r["dataset"]) for rows_ in rows.values() for r in rows_}):
        by_ds[ds].append(ep)
    return {ep: eps[(i + 1) % len(eps)] for eps in by_ds.values() for i, ep in enumerate(eps)}


class Judge:
    """One submission's score() split into a cached video part and a per-row text part."""

    def __init__(self, mod, sub, patch, episodes):
        self.mod, self.sub, self.patch, self.episodes = mod, sub, patch, episodes
        self.videos = {}

    @torch.inference_mode()
    def video(self, ep):
        if ep not in self.videos:
            s, images = self.sub, load_images(self.episodes / ep)
            if self.patch:
                frames, grids = s._encode(images)
            else:
                frames, grids = s.enc.encode_frames(images).to(s.device), None
            sig = F.normalize(frames[:, -self.mod.SIGLIP_DIM:], dim=-1)
            if len(frames) > s.tmax:
                keep = torch.linspace(0, len(frames) - 1, s.tmax).round().long()
                frames, grids = frames[keep], grids[keep] if grids is not None else None
            fmask = torch.zeros(1, len(frames), dtype=torch.bool, device=s.device)
            if self.patch:
                video, vmask = s._video(frames[None], fmask, grids[None])
            else:
                video, vmask = s.head.encode_video(frames[None], fmask), fmask
            self.videos[ep] = (sig, video, vmask)
        return self.videos[ep]

    @torch.inference_mode()
    def text(self, texts):
        s, m = self.sub, self.mod
        span = m.score_span(texts)
        byte = m._mean_scores([t.encode("utf-8") for t in texts], m.normalized_byte_distance)
        base = torch.tensor([a + 1e-3 * b for a, b in zip(span, byte)])
        cl, cm = s._clauses(texts)
        n, R = len(texts), s.ref_video.shape[0]
        head_prior = None if not self.patch else torch.stack([
            s.head(s.ref_video, s.ref_vmask if self.patch else s.ref_mask, cl[i:i + 1].expand(R, -1, -1),
                   cm[i:i + 1].expand(R, -1)).mean() for i in range(n)])
        align_prior = []
        for i in range(n):
            ref = torch.einsum("kd,rtd->rkt", cl[i, ~cm[i]], s.ref_sig).masked_fill(s.ref_mask[:, None, :], -1e4)
            align_prior.append(m.batched_align(ref).mean())
        return base, cl, cm, head_prior, torch.stack(align_prior)

    @torch.inference_mode()
    def z(self, ep, text):
        """z(align - prior), z(head - prior) for this video, as in Submission.score."""
        _, cl, cm, head_prior, align_prior = text
        sig, video, vmask = self.video(ep)
        n, zs = len(cl), self.mod.zscore
        align = torch.stack([self.mod.batched_align((cl[i, ~cm[i]] @ sig.T)[None])[0] for i in range(n)])
        if head_prior is None:  # SigLIP-only zip: l2 = 0, the head never counts
            return zs(align - align_prior).float().cpu(), None
        head = self.sub.head(video.expand(n, -1, -1), vmask.expand(n, -1), cl, cm)
        return zs(align - align_prior).float().cpu(), zs(head - head_prior).float().cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", type=Path, required=True, help="unzipped video_judge_patch")
    ap.add_argument("--siglip-sub", type=Path, required=True, help="unzipped video_judge_siglip")
    ap.add_argument("--sets", type=Path, default=HERE / "sets")
    ap.add_argument("--episodes", type=Path, default=HERE / "episodes")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", type=int, default=50)
    ap.add_argument("--limit", type=int, help="rows per cell (smoke test)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = {c: json.loads((args.sets / f"{c}.json").read_text(encoding="utf-8"))[:args.limit] for c in CELLS}
    other = partners(rows)
    judges = {}
    for key, folder, patch in (("p", args.patch, True), ("s", args.siglip_sub, False)):
        mod = load_module(folder.resolve(), f"sub_{key}")
        judges[key] = Judge(mod, mod.Submission(str(folder.resolve()), args.device), patch, args.episodes)
    P, S = judges["p"], judges["s"]

    out_rows, mismatch, checked = [], {"shipped": 0, "siglip_zip": 0}, 0
    t0 = time.time()
    flat = sorted(((c, r) for c in CELLS for r in rows[c]), key=lambda cr: cr[1]["episode"])
    for n, (cell, row) in enumerate(flat):
        choices, truth = shuffled(row, args.seed)
        plain = [{"id": c["id"], "text": c["text"]} for c in choices]
        texts = [c["text"] for c in choices]
        kinds = {c["text"]: c.get("kind") for c in row["choices"]}
        rec = {"cell": cell, "episode": row["episode"], "dataset": row["dataset"],
               "shared_kind": row["shared_kind"], "ids": [c["id"] for c in choices], "truth": truth,
               "kinds": [kinds[t] for t in texts]}
        pt, st = P.text(texts), S.text(texts)
        rec["base"] = pt[0].tolist()
        for pre, ep in (("", row["episode"]), ("ctrl_", other[row["episode"]])):
            za, zh = P.z(ep, pt)
            rec[pre + "z_align"], rec[pre + "z_head"] = za.tolist(), zh.tolist()
            sa, _ = S.z(ep, st)
            sig_score = (st[0] + S.sub.l1 * sa).tolist()  # siglip zip: l2 = 0
            rec[pre + "siglip_zip_order"] = [rec["ids"][i] for i in sorted(range(len(texts)),
                                                                           key=lambda i: -sig_score[i])]
        if checked < args.check:  # the split must reproduce both shipped submissions
            imgs = load_images(args.episodes / row["episode"])
            mismatch["shipped"] += P.sub.predict(imgs, plain) != order_of(rec, CONFIGS["shipped"])
            mismatch["siglip_zip"] += S.sub.predict(imgs, plain) != rec["siglip_zip_order"]
            checked += 1
        out_rows.append(rec)
        if n % 500 == 0:
            print(f"{n}/{len(flat)} {time.time() - t0:.0f}s", flush=True)
    print(f"check vs Submission.predict: {mismatch} mismatches in {checked} rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": out_rows, "check": {"rows": checked, **mismatch}}) + "\n",
                        encoding="utf-8")
    print(f"{len(out_rows)} rows, {time.time() - t0:.0f}s -> {args.out}")


def order_of(rec, lam, ctrl=False):
    pre = "ctrl_" if ctrl else ""
    s = [b + lam[0] * a + lam[1] * h for b, a, h in zip(rec["base"], rec[pre + "z_align"], rec[pre + "z_head"])]
    return [rec["ids"][i] for i in sorted(range(len(s)), key=lambda i: -s[i])]


def rank_score(order, truth):
    return RANK_SCORE[order.index(truth)]


if __name__ == "__main__":
    main()
