"""A validation mix whose negative structure matches the test, calibrated on two text-only LB scores.

The two model-free submissions see only the choice texts, so their test scores depend on how the
negatives are built, not on the video domain:
    byte centrality (hyuwooma/)                         LB 0.520
    span centrality (diff_probe/submissions/span_centrality)  LB 0.566
Each cell below is a 5-choice set from the team's Qwen augmentation (validation episodes only)
with one negative structure. We search mixing weights whose expected byte and span scores fall
within --tol of the LB and report their range. The validator is the cell sets themselves plus
the best-fitting weights: score a model on every cell with eval_subs.py, then take the weighted
mean (--combine). Whole cells keep every episode, so the estimate has less sampling noise than a
single sampled set.

    single     GT + 4 single Qwen edits              (diff_probe/valsets_qwen.json A_single)
    ab_ab_c    GT + {a, b, a+b, c}                   (fact_ab_ab_c)
    ab_ab_ac   GT + {a, b, a+b, a+c}                 (fact_ab_ab_ac)
    share4     GT + {a, a+b, a+c, a+d}               (validator/sets/qwen_share3_*)
    para_negs  GT + 4 edits of Qwen paraphrases      (para_negs)

    python validator/calibrate.py --view full --out validator/calibrated
    python validator/eval_subs.py --sets validator/calibrated/full_*.json --sub NAME=DIR --out res.json
    python validator/calibrate.py --combine res.json validator/calibrated/weights_full.json

Word centrality is never submitted; its expected score is printed as a prediction to check later.
"""

import argparse
import importlib.util
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validator"))
from eval_subs import RANK_SCORE, shuffled  # noqa: E402

LB = {"byte": 0.520, "span": 0.566}
CELLS = ["single", "ab_ab_c", "ab_ab_ac", "share4", "para_negs"]
QWEN_KIND = {"single": "A_single", "ab_ab_c": "fact_ab_ab_c", "ab_ab_ac": "fact_ab_ab_ac",
             "para_negs": "para_negs"}


def load_cells(view):
    qwen = json.loads((ROOT / "diff_probe/valsets_qwen.json").read_text())["sets"]
    cells = {c: qwen[f"{view}/{k}"] for c, k in QWEN_KIND.items()}
    cells["share4"] = json.loads((ROOT / f"validator/sets/qwen_share3_{view}.json").read_text())
    return cells


def load_predict(folder, name):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Submission(str(folder), "cpu").predict


def word_predict(choices):
    sys.path.insert(0, str(ROOT / "diff_probe"))
    from centrality import score_word
    scores = score_word([c["text"] for c in choices])
    order = sorted(range(len(choices)), key=lambda i: -scores[i])
    return [choices[i]["id"] for i in order]


def episode_scores(cells, seed):
    """{cell: {episode: {method: rank score}}} with choices shuffled as eval_subs does."""
    methods = {"byte": load_predict(ROOT / "hyuwooma", "sub_byte"),
               "span": load_predict(ROOT / "diff_probe/submissions/span_centrality", "sub_span"),
               "word": lambda images, choices: word_predict(choices)}
    out = {}
    for cell, rows in cells.items():
        out[cell] = {}
        for row in rows:
            choices, truth = shuffled(row, seed)
            plain = [{"id": c["id"], "text": c["text"]} for c in choices]
            out[cell][row["episode"]] = {m: RANK_SCORE[p(None, plain).index(truth)]
                                         for m, p in methods.items()}
    return out


def feasible_mixes(means, step, tol):
    n = round(1 / step)
    found = []
    for parts in itertools.product(range(n + 1), repeat=len(CELLS) - 1):
        if sum(parts) > n:
            continue
        w = dict(zip(CELLS, [p / n for p in parts] + [(n - sum(parts)) / n]))
        expect = {m: sum(w[c] * means[c][m] for c in CELLS) for m in ("byte", "span", "word")}
        if all(abs(expect[m] - LB[m]) <= tol for m in LB):
            found.append((w, expect))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["full", "short"], default="full")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=0.02, help="about one public-split standard error")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="directory for the cell sets and weights_<view>.json")
    ap.add_argument("--combine", nargs=2, metavar=("RESULTS", "WEIGHTS"),
                    help="print the weighted LB estimate for an eval_subs.py --out file")
    args = ap.parse_args()
    if args.combine:
        combine(*args.combine)
        return

    cells = load_cells(args.view)
    scores = episode_scores(cells, args.seed)
    means = {c: {m: statistics.fmean(s[m] for s in scores[c].values()) for m in ("byte", "span", "word")}
             for c in CELLS}
    print(f"view={args.view}")
    for c in CELLS:
        print(f"  {c:10s} n={len(scores[c]):3d}  " + "  ".join(f"{m} {means[c][m]:.3f}" for m in means[c]))

    found = feasible_mixes(means, args.step, args.tol)
    print(f"{len(found)} feasible mixes (step {args.step}, tol {args.tol})")
    if not found:
        return
    for c in CELLS:
        ws = [w[c] for w, _ in found]
        print(f"  {c:10s} weight {min(ws):.2f}-{max(ws):.2f}  mean {statistics.fmean(ws):.2f}")
    words = [e["word"] for _, e in found]
    print(f"  predicted word-centrality LB {min(words):.3f}-{max(words):.3f}")

    centroid = {c: statistics.fmean(w[c] for w, _ in found) for c in CELLS}
    # Best LB fit first; among equal fits, the mix nearest the centroid of all feasible mixes.
    w, expect = min(found, key=lambda f: (round(sum(abs(f[1][m] - LB[m]) for m in LB), 3),
                                          sum((f[0][c] - centroid[c]) ** 2 for c in CELLS)))
    print("chosen  " + "  ".join(f"{c} {w[c]:.2f}" for c in CELLS))
    print("  expected " + "  ".join(f"{m} {v:.3f}" for m, v in expect.items())
          + f"   (LB byte {LB['byte']}, span {LB['span']})")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for c in CELLS:
            (args.out / f"{args.view}_{c}.json").write_text(
                json.dumps(cells[c], ensure_ascii=False, indent=1), encoding="utf-8")
        (args.out / f"weights_{args.view}.json").write_text(json.dumps(
            {"view": args.view, "weights": w, "expected": expect, "lb": LB, "seed": args.seed,
             "feasible": len(found), "tol": args.tol, "step": args.step}, indent=1) + "\n")
        print("wrote", args.out)


def combine(results_path, weights_path):
    """Weighted LB estimate from an eval_subs.py --out file run on the cell sets."""
    spec = json.loads(Path(weights_path).read_text())
    view, w = spec["view"], spec["weights"]
    for name, by_set in json.loads(Path(results_path).read_text()).items():
        got = {c: by_set.get(f"{view}_{c}", {}).get("score") for c in CELLS}
        missing = [c for c, v in got.items() if v is None and w[c] > 0]
        est = None if missing else sum(w[c] * got[c] for c in CELLS if w[c] > 0)
        cells = "  ".join(f"{c} {got[c]:.3f}" for c in CELLS if got[c] is not None)
        print(f"{name:16s} LB estimate {est if est is None else round(est, 3)}   {cells}"
              + (f"   missing {missing}" if missing else ""))

if __name__ == "__main__":
    main()
