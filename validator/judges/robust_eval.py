"""Do the video judges survive a different negative generator? Fixed configs, no selection.

Scores span + l1*z(align_db) + l2*z(head_db) on Claude-written evaluation-only sets
(validator/sets/claude_short.json: 4 single edits of the first sentence; claude_ab_ac.json:
{a, b, a+b, a+c} merged from them; never used for training) next to the Qwen cells with the
same structure (validator/calibrated/short_single.json, short_ab_ab_ac.json). The heads were
trained on Qwen negatives, so a gap between the two columns is how much they rely on Qwen wording.

    python validator/judges/robust_eval.py --base diff_probe/runs/pair_head_x/epoch14.pt \
        --patch diff_probe/runs/patch_head_x/epoch11.pt --hf ... --span-dir diff_probe/runs/span_sub
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diff_probe"))
import rerank_eval as R  # noqa: E402
from pair_head import ClauseBank, FrameBank, load_groups  # noqa: E402
from rerank_eval_patch import PatchJudges  # noqa: E402

SETS = {"claude_short": "validator/sets/claude_short.json",
        "qwen_short_single": "validator/calibrated/short_single.json",
        "claude_ab_ac": "validator/sets/claude_ab_ac.json",
        "qwen_short_ab_ab_ac": "validator/calibrated/short_ab_ab_ac.json"}
# (name, head, l1, l2, debiased): debiased=False uses the raw judge scores (no reference-prior
# subtraction), the ablation of the text-prior removal.
CONFIGS = [("span", None, 0, 0, True), ("span+SigLIP", None, 0.3, 0, True),
           ("span+SigLIP+base", "base", 0.3, 0.3, True), ("span+base", "base", 0, 0.3, True),
           ("span+SigLIP+patch", "patch", 0.3, 0.3, True), ("span+patch", "patch", 0, 0.3, True),
           ("raw span+SigLIP", None, 0.3, 0, False), ("raw span+SigLIP+patch", "patch", 0.3, 0.3, False),
           ("raw span+patch", "patch", 0, 0.3, False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--patch", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--siglip", type=Path, default=ROOT / "baseline_v2/models/siglip")
    ap.add_argument("--span-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sets = {k: json.loads((ROOT / v).read_text()) for k, v in SETS.items()}
    groups = load_groups(args.hf, args.splits)
    texts = {c["text"] for rows in sets.values() for r in rows for c in r["choices"]}
    frames = FrameBank(args.frames)
    clauses = ClauseBank(texts, args.siglip, args.device)
    train_eps = sorted({g["episode"] for g in groups["train"]}, key=lambda e: int(e.split("_")[1]))
    ref_eps = random.Random(0).sample(train_eps, 64)  # same references as rerank_eval.py
    judges = {"base": R.Judges(args.base, frames, clauses, ref_eps, args.device),
              "patch": PatchJudges(args.patch, frames, clauses, ref_eps, args.device)}
    valid_eps = sorted({g["episode"] for g in groups["valid"]}, key=lambda e: int(e.split("_")[1]))
    other = {e: valid_eps[(n + 1) % len(valid_eps)] for n, e in enumerate(valid_eps)}
    span = R.load_span(args.span_dir)

    result = {}
    for name, rows in sets.items():
        scores = {(c[0], v): 0.0 for c in CONFIGS for v in ("real", "other")}
        for row in rows:
            ch, truth = R.shuffled(row, 0)
            t = [x["text"] for x in ch]
            ti = [x["id"] for x in ch].index(truth)
            sp = span.score(None, [{"id": x["id"], "text": x["text"]} for x in ch])
            for video, ep in (("real", row["episode"]), ("other", other[row["episode"]])):
                z = {k: j.score(t, ep) for k, j in judges.items()}
                for cname, head, l1, l2, db in CONFIGS:
                    za = R.zscore(z["base"]["align_db" if db else "align"])
                    zh = R.zscore(z[head]["head_db" if db else "head"]) if head else [0.0] * len(t)
                    final = [s + l1 * a + l2 * h for s, a, h in zip(sp, za, zh)]
                    order = sorted(range(len(t)), key=lambda i: -final[i])
                    scores[(cname, video)] += R.RANK_SCORE[order.index(ti)] / len(rows)
        result[name] = {f"{c}/{v}": round(s, 4) for (c, v), s in scores.items()}
        print(name, len(rows), flush=True)

    print(f"\n{'config':24s}" + "".join(f"{k:>22s}" for k in SETS))
    for cname, *_ in CONFIGS:
        print(f"{cname:24s}" + "".join(
            f"{result[k][cname + '/real']:.3f} ({result[k][cname + '/other']:.3f})".rjust(22) for k in SETS))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
