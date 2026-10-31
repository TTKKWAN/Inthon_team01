"""Video judges with the text prior removed, alone (pairwise) and added to span centrality.

A judge can prefer a choice from its wording alone (siglip_pairs.py / pair_head.py score well
above 0.5 with another episode's video). Remove that part: for each text, subtract its mean
score over R fixed reference videos (Train episodes, chosen before evaluation and shipped with
a submission), so only how much better it fits THIS video than videos in general is left.

  align_db : SigLIP monotonic clause alignment (zero-shot) minus its reference mean
  weak_db  : SigLIP weakest clause, min over clauses of max over frames (an action that never
             happens drags it down), minus its reference mean
  head_db  : pair_head.py checkpoint score minus its reference mean

1. Pairwise accuracy on GT vs one Qwen edit (validation episodes), per kind, real / other video.
2. score = span + l1 * z(align_db) + l2 * z(head_db) on the calibrated cells; (l1, l2) picked
   on half A (even episode ids), reported on half B, weighted by the cell weights.

    python validator/judges/rerank_eval.py --head diff_probe/runs/pair_head/epoch09.pt \
        --hf diff_probe/runs/qwen_negatives_from_gt.json --span-dir diff_probe/runs/span_sub
"""

import argparse
import importlib.util
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diff_probe"))
sys.path.insert(0, str(ROOT / "validator"))
from eval_subs import RANK_SCORE, shuffled  # noqa: E402
from pair_head import ClauseBank, FrameBank, PairHead, load_groups  # noqa: E402
from siglip_pairs import SIGLIP_DIM  # noqa: E402

CELLS = ["single", "ab_ab_c", "ab_ab_ac", "share4", "para_negs"]
L1 = [0, 0.1, 0.2, 0.3, 0.5]
L2 = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def batched_align(sim):
    """sim [R, K, T] -> [R] monotonic clause-to-frame alignment, mean over clauses."""
    best = sim[:, 0]
    for k in range(1, sim.shape[1]):
        best = torch.cummax(best, dim=1).values + sim[:, k]
    return best.max(1).values / sim.shape[1]


class Judges:
    def __init__(self, head_path, frames, clauses, ref_eps, device):
        ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        self.model = PairHead(tmax=ckpt["tmax"]).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.frames, self.clauses, self.device = frames, clauses, device
        self.ref_eps = ref_eps
        with torch.no_grad():
            fr, self.ref_mask = frames.batch(ref_eps, device)
            self.ref_video = self.model.encode_video(fr, self.ref_mask)
            self.ref_sig = F.normalize(fr[..., -SIGLIP_DIM:], dim=-1)  # [R, T, 768]
        self.prior = {}

    def _sig(self, ep):
        fr, fm = self.frames.batch([ep], self.device)
        return F.normalize(fr[0, ~fm[0], -SIGLIP_DIM:], dim=-1)

    @torch.no_grad()
    def _priors(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.prior]
        R = len(self.ref_eps)
        for t in todo:
            cl, cm = self.clauses.batch([t], self.device)
            k = int((~cm[0]).sum())
            head = self.model(self.ref_video, self.ref_mask, cl.expand(R, -1, -1), cm.expand(R, -1)).mean()
            sim = torch.einsum("kd,rtd->rkt", cl[0, :k], self.ref_sig)
            sim = sim.masked_fill(self.ref_mask[:, None, :], -1e4)
            weak = sim.max(2).values.min(1).values.mean()
            self.prior[t] = (batched_align(sim).mean().item(), head.item(), weak.item())

    @torch.no_grad()
    def score(self, texts, ep):
        """{'align_db': [..], 'head_db': [..]} for texts against episode ep's video."""
        self._priors(texts)
        fr, fm = self.frames.batch([ep], self.device)
        video = self.model.encode_video(fr, fm)
        cl, cm = self.clauses.batch(texts, self.device)
        n = len(texts)
        head = self.model(video.expand(n, -1, -1), fm.expand(n, -1), cl, cm).tolist()
        sig = self._sig(ep)
        align, weak = [], []
        for i in range(n):
            k = int((~cm[i]).sum())
            sim = cl[i, :k] @ sig.T
            align.append(batched_align(sim[None]).item())
            weak.append(sim.max(1).values.min().item())
        return {"align_db": [a - self.prior[t][0] for a, t in zip(align, texts)],
                "head_db": [h - self.prior[t][1] for h, t in zip(head, texts)],
                "weak_db": [w - self.prior[t][2] for w, t in zip(weak, texts)],
                "align": align, "head": head}


def load_span(folder):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("span_sub", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Submission(str(folder), "cpu")


def zscore(x):
    x = torch.tensor(x)
    return ((x - x.mean()) / (x.std() + 1e-6)).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--siglip", type=Path, default=ROOT / "baseline_v2/models/siglip")
    ap.add_argument("--span-dir", type=Path, default=ROOT / "diff_probe/submissions/span_centrality")
    ap.add_argument("--refs", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    groups = load_groups(args.hf, args.splits)
    cal = ROOT / "validator/calibrated"
    cells = {v: {c: json.loads((cal / f"{v}_{c}.json").read_text()) for c in CELLS} for v in ("full", "short")}
    weights = {v: json.loads((cal / f"weights_{v}.json").read_text())["weights"] for v in ("full", "short")}
    texts = {t for g in groups["valid"] for t in [g["pos"]] + [x for _, x in g["negs"]]}
    texts |= {c["text"] for v in cells.values() for rows in v.values() for r in rows for c in r["choices"]}
    frames = FrameBank(args.frames)
    clauses = ClauseBank(texts, args.siglip, args.device)
    train_eps = sorted({g["episode"] for g in groups["train"]}, key=lambda e: int(e.split("_")[1]))
    ref_eps = random.Random(0).sample(train_eps, args.refs)
    judges = Judges(args.head, frames, clauses, ref_eps, args.device)
    valid_eps = sorted({g["episode"] for g in groups["valid"]}, key=lambda e: int(e.split("_")[1]))
    other = {e: valid_eps[(n + 1) % len(valid_eps)] for n, e in enumerate(valid_eps)}
    report = {"ref_eps": ref_eps}

    # 1. pairwise accuracy
    hits = defaultdict(list)
    for g in groups["valid"]:
        texts_g = [g["pos"]] + [t for _, t in g["negs"]]
        for video in ("real", "other"):
            s = judges.score(texts_g, g["episode"] if video == "real" else other[g["episode"]])
            for j in ("align", "align_db", "weak_db", "head", "head_db"):
                for (kind, _), sn in zip(g["negs"], s[j][1:]):
                    hit = 1.0 if s[j][0] > sn else 0.5 if s[j][0] == sn else 0.0
                    for k in (kind, "all"):
                        hits[(video, g["view"], k, j)].append(hit)
    pair = {"/".join(k): round(sum(v) / len(v), 4) for k, v in hits.items()}
    report["pairwise"] = pair
    for view in ("full", "short"):
        print(f"\n[{view}] pairwise accuracy, real (other video)")
        print(f"{'kind':14s}" + "".join(f"{j:>16s}" for j in ("align", "align_db", "weak_db", "head", "head_db")))
        for k in ("llm_object", "llm_attribute", "llm_add", "llm_action", "llm_order", "all"):
            print(f"{k:14s}" + "".join(f"{pair[f'real/{view}/{k}/{j}']:.3f} ({pair[f'other/{view}/{k}/{j}']:.3f})".rjust(16)
                                       for j in ("align", "align_db", "weak_db", "head", "head_db")))

    # 2. rerank on calibrated cells
    span = load_span(args.span_dir)
    report["rerank"] = {}
    for view in ("full", "short"):
        items = {c: [] for c in CELLS}
        for c, rows in cells[view].items():
            for row in rows:
                ch, truth = shuffled(row, 0)
                t = [x["text"] for x in ch]
                it = {"half": int(row["episode"].split("_")[1]) % 2,
                      "truth": [x["id"] for x in ch].index(truth),
                      "span": span.score(None, [{"id": x["id"], "text": x["text"]} for x in ch])}
                for video, ep in (("real", row["episode"]), ("other", other[row["episode"]])):
                    s = judges.score(t, ep)
                    it[video] = (zscore(s["align_db"]), zscore(s["head_db"]))
                items[c].append(it)

        def weighted(l1, l2, video, half):
            per = {}
            for c in CELLS:
                got = [it for it in items[c] if it["half"] == half]
                tot = 0.0
                for it in got:
                    za, zh = it[video]
                    final = [s + l1 * a + l2 * h for s, a, h in zip(it["span"], za, zh)]
                    order = sorted(range(5), key=lambda i: -final[i])
                    tot += RANK_SCORE[order.index(it["truth"])]
                per[c] = tot / len(got)
            return sum(weights[view][c] * per[c] for c in CELLS), per

        rows_out = []
        for l1, l2 in itertools.product(L1, L2):
            a, _ = weighted(l1, l2, "real", 0)
            b, cells_b = weighted(l1, l2, "real", 1)
            o, _ = weighted(l1, l2, "other", 1)
            rows_out.append({"l1": l1, "l2": l2, "A": a, "B": b, "other_B": o, "B_cells": cells_b})
        report["rerank"][view] = rows_out
        print(f"\n[{view}] span + l1*z(align_db) + l2*z(head_db): half A picks, half B reports")
        show = [r for r in rows_out if (r["l1"], r["l2"]) in {(0, 0), (0.3, 0), (0, 0.3), (0, 0.5)}]
        show.append(max(rows_out, key=lambda r: r["A"]))
        for r in show:
            cb = " ".join(f"{c} {r['B_cells'][c]:.3f}" for c in CELLS)
            print(f"  l1 {r['l1']:.2f} l2 {r['l2']:.2f}  A {r['A']:.3f}  B {r['B']:.3f}  other-B {r['other_B']:.3f}   {cb}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
