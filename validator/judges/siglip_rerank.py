"""Span centrality + SigLIP clause alignment, scored on the calibrated validator cells.

score(choice) = span centrality + lam * z(align), where align is the monotonic clause-to-frame
SigLIP similarity of siglip_pairs.py and z standardizes it over this episode's 5 choices (only
this episode's inputs). lam is picked on half A of the episodes (even ids) and reported on
half B (odd ids), each cell weighted by validator/calibrated/weights_<view>.json. The same run
with another episode's video shows how much of any gain comes from the video.

    python validator/judges/siglip_rerank.py --view full --out diff_probe/runs/siglip_rerank_full.json
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diff_probe"))
sys.path.insert(0, str(ROOT / "validator"))
from eval_subs import RANK_SCORE, shuffled  # noqa: E402
from siglip_pairs import SIGLIP_DIM, SiglipText, monotonic_align, split_clauses  # noqa: E402

CELLS = ["single", "ab_ab_c", "ab_ab_ac", "share4", "para_negs"]
LAMS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def load_span(folder):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("span_sub", folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Submission(str(folder), "cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["full", "short"], default="full")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--siglip", type=Path, default=ROOT / "baseline_v2/models/siglip")
    ap.add_argument("--span-dir", type=Path, default=ROOT / "diff_probe/submissions/span_centrality")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cal = ROOT / "validator/calibrated"
    weights = json.loads((cal / f"weights_{args.view}.json").read_text())["weights"]
    cells = {c: json.loads((cal / f"{args.view}_{c}.json").read_text()) for c in CELLS}
    span = load_span(args.span_dir)
    cache = torch.load(args.frames, map_location="cpu", weights_only=False)
    index = {e: i for i, e in enumerate(cache["episode_ids"])}
    offsets = cache["frame_offsets"]
    episodes = sorted({r["episode"] for rows in cells.values() for r in rows},
                      key=lambda e: int(e.split("_")[1]))
    other = {e: episodes[(n + 1) % len(episodes)] for n, e in enumerate(episodes)}
    text_enc = SiglipText(args.siglip, args.device)

    def frames_of(ep):
        i = index[ep]
        return F.normalize(cache["frames"][offsets[i]:offsets[i + 1], -SIGLIP_DIM:].float(), dim=-1)

    # per (cell, episode): truth index, span scores, align z-scores for real / other video
    items = {c: [] for c in CELLS}
    for c, rows in cells.items():
        for row in rows:
            choices, truth = shuffled(row, 0)
            texts = [x["text"] for x in choices]
            s = span.score(None, [{"id": x["id"], "text": x["text"]} for x in choices])
            z = {}
            for video, ep in (("real", row["episode"]), ("other", other[row["episode"]])):
                fr = frames_of(ep)
                a = torch.tensor([monotonic_align(text_enc.encode(split_clauses(t)) @ fr.T) for t in texts])
                z[video] = ((a - a.mean()) / (a.std() + 1e-6)).tolist()
            items[c].append({"episode": row["episode"], "truth": [x["id"] for x in choices].index(truth),
                             "span": s, "z": z})
        print(c, len(items[c]), flush=True)

    def score(c, lam, video, half):
        got = [it for it in items[c] if int(it["episode"].split("_")[1]) % 2 == half]
        total = 0.0
        for it in got:
            final = [s + lam * z for s, z in zip(it["span"], it["z"][video])]
            order = sorted(range(5), key=lambda i: -final[i])
            total += RANK_SCORE[order.index(it["truth"])]
        return total / len(got)

    result = {}
    for video in ("real", "other"):
        for lam in LAMS:
            per = {h: {c: score(c, lam, video, h) for c in CELLS} for h in (0, 1)}
            result[f"{video}/{lam}"] = {
                "A": sum(weights[c] * per[0][c] for c in CELLS),
                "B": sum(weights[c] * per[1][c] for c in CELLS),
                "B_cells": per[1]}
    best = max(LAMS, key=lambda l: result[f"real/{l}"]["A"])
    print(f"\nview={args.view}  weighted validator score (half A picks lam, half B reports)")
    print(f"{'lam':>5s} {'real A':>7s} {'real B':>7s} {'other B':>8s}   real-B cells")
    for lam in LAMS:
        r, o = result[f"real/{lam}"], result[f"other/{lam}"]
        cells_b = " ".join(f"{c} {r['B_cells'][c]:.3f}" for c in CELLS)
        print(f"{lam:5.2f} {r['A']:7.3f} {r['B']:7.3f} {o['B']:8.3f}   {cells_b}" + ("  <- picked" if lam == best else ""))
    if args.out:
        args.out.write_text(json.dumps({"picked_lam": best, "results": result}, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
