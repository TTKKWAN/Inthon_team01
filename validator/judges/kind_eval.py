"""Calibrated-cell score split by the kind of the SHARED edit (the one span cannot resolve).

span loses when one edit is shared by several negatives; the video judge then has to tell GT
from that edit. LB: span 0.5661, span + 0.3 z(SigLIP) + 0.3 z(patch head) 0.4952. If test's
shared edits are mostly action/order (TemporalBench-like), the rows whose shared edit is
llm_action/llm_order should show the same drop; object/attribute/add rows should not.

Shared edit: ab_ab_ac -> a (in 3 negatives); ab_ab_c -> a and b (2 each, both must be temporal);
share4 -> a (choice type 'a', kind looked up in the Qwen bank). Half B only, plus other-video.
Per-row judge z-scores are saved so other weightings can be tried without a GPU.

    python validator/judges/kind_eval.py --head diff_probe/runs/patch_head_x/epoch11.pt \
        --hf diff_probe/runs/qwen_negatives_from_gt.json --span-dir diff_probe/runs/span_sub \
        --exclude diff_probe/runs/ref_eps.json --out diff_probe/runs/kind_eval.json
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diff_probe"))
sys.path.insert(0, str(ROOT / "validator"))
sys.path.insert(0, str(Path(__file__).parent))
import rerank_eval as R  # noqa: E402
from eval_subs import RANK_SCORE, shuffled  # noqa: E402
from pair_head import ClauseBank, FrameBank, load_groups  # noqa: E402
from rerank_eval_patch import PatchJudges  # noqa: E402

CELLS = ["ab_ab_c", "ab_ab_ac", "share4"]
TEMPORAL = {"llm_action", "llm_order"}
LAMBDAS = [(0, 0), (0.3, 0), (0, 0.3), (0.3, 0.3)]


def shared_kinds(cell, row, bank_kind):
    types = [c["type"] for c in row["choices"][1:]]
    if cell == "share4":
        a = next(c["text"] for c in row["choices"] if c["type"] == "a")
        return {bank_kind.get((row["episode"], a), "unknown")}
    count = Counter(k for t in types for k in t.removeprefix("fact:").split("+"))
    top = max(count.values())
    return {k for k, n in count.items() if n == top}


def group_of(kinds):
    return "temporal" if kinds <= TEMPORAL else "other_kind" if not kinds & TEMPORAL else "mixed"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--siglip", type=Path, default=ROOT / "baseline_v2/models/siglip")
    ap.add_argument("--span-dir", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, help="json with ref_eps (refs used by the shipped judge)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    groups = load_groups(args.hf, args.splits)
    bank_kind = {(g["episode"], t): k for g in groups["valid"] for k, t in g["negs"]}
    cal = ROOT / "validator/calibrated"
    cells = {v: {c: json.loads((cal / f"{v}_{c}.json").read_text()) for c in CELLS} for v in ("full", "short")}
    weights = {v: json.loads((cal / f"weights_{v}.json").read_text())["weights"] for v in ("full", "short")}
    texts = {c["text"] for v in cells.values() for rows in v.values() for r in rows for c in r["choices"]}
    frames = FrameBank(args.frames)
    clauses = ClauseBank(texts, args.siglip, args.device)
    if args.exclude:
        ref_eps = json.loads(args.exclude.read_text())["ref_eps"]
    else:
        train_eps = sorted({g["episode"] for g in groups["train"]}, key=lambda e: int(e.split("_")[1]))
        ref_eps = random.Random(0).sample(train_eps, 64)
    judges = PatchJudges(args.head, frames, clauses, ref_eps, args.device)
    valid_eps = sorted({g["episode"] for g in groups["valid"]}, key=lambda e: int(e.split("_")[1]))
    other = {e: valid_eps[(n + 1) % len(valid_eps)] for n, e in enumerate(valid_eps)}
    span = R.load_span(args.span_dir)

    rows_out, report = [], {}
    for view in ("full", "short"):
        for cell in CELLS:
            for row in cells[view][cell]:
                if int(row["episode"].split("_")[1]) % 2 == 0:
                    continue  # half B only
                ch, truth = shuffled(row, 0)
                t = [x["text"] for x in ch]
                it = {"view": view, "cell": cell, "episode": row["episode"],
                      "group": group_of(shared_kinds(cell, row, bank_kind)),
                      "truth": [x["id"] for x in ch].index(truth),
                      "span": span.score(None, [{"id": x["id"], "text": x["text"]} for x in ch])}
                for video, ep in (("real", row["episode"]), ("other", other[row["episode"]])):
                    s = judges.score(t, ep)
                    it[video] = (R.zscore(s["align_db"]), R.zscore(s["head_db"]))
                rows_out.append(it)
        print(f"[{view}] scored", flush=True)

    def score(items, l1, l2, video):
        tot = 0.0
        for it in items:
            za, zh = it[video]
            final = [s + l1 * a + l2 * h for s, a, h in zip(it["span"], za, zh)]
            order = sorted(range(5), key=lambda i: -final[i])
            tot += RANK_SCORE[order.index(it["truth"])]
        return tot / max(len(items), 1)

    for view in ("full", "short"):
        report[view] = {}
        print(f"\n[{view}] half B, cell-weighted; (n per cell)  real / other video")
        for grp in ("all", "temporal", "other_kind", "mixed"):
            items = {c: [it for it in rows_out if it["view"] == view and it["cell"] == c
                         and (grp == "all" or it["group"] == grp)] for c in CELLS}
            wsum = sum(weights[view][c] for c in CELLS if items[c])
            res = {}
            for l1, l2 in LAMBDAS:
                for video in ("real", "other"):
                    res[f"{l1}/{l2}/{video}"] = sum(weights[view][c] * score(items[c], l1, l2, video)
                                                    for c in CELLS if items[c]) / wsum
            res["n"] = {c: len(items[c]) for c in CELLS}
            report[view][grp] = res
            line = "  ".join(f"({l1},{l2}) {res[f'{l1}/{l2}/real']:.3f}/{res[f'{l1}/{l2}/other']:.3f}"
                             for l1, l2 in LAMBDAS)
            print(f"  {grp:10s} n={res['n']}  {line}")
    args.out.write_text(json.dumps({"report": report, "rows": rows_out}) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
