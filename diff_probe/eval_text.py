"""Score model-free centrality rankers on local 5-choice sets with the competition metric.

Choices are shuffled per episode (fixed seed) so ties cannot favour the positive, which
is stored first. Ties are broken by that shuffled order, like hyuwooma's stable sort.

    python diff_probe/eval_text.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from centrality import SCORERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]


def load_sets():
    sets = json.loads((ROOT / "diff_probe/valsets.json").read_text())["sets"]
    sets = {"rule_val_full": json.loads(
        (ROOT / "baseline_v2/validation/val_full.json").read_text()), **sets}
    return sets


def evaluate(rows, scorer, seed=0):
    rng = random.Random(seed)
    total, top1, tied = 0.0, 0, 0
    for row in rows:
        choices = list(row["choices"])
        rng.shuffle(choices)
        scores = scorer([c["text"] for c in choices])
        order = sorted(range(len(choices)), key=lambda i: -scores[i])
        rank = next(r for r, i in enumerate(order) if choices[i]["type"] == "pos")
        total += RANK_SCORE[rank]
        top1 += rank == 0
        best = max(scores)
        tied += sum(abs(s - best) < 1e-12 for s in scores) > 1
    n = len(rows)
    return total / n, top1 / n, tied / n


def main():
    sets = load_sets()
    print(f"{'set':16s} " + "  ".join(f"{s:>20s}" for s in SCORERS))
    print(f"{'':16s} " + "  ".join(f"{'score/top1/tie':>20s}" for _ in SCORERS))
    for name, rows in sets.items():
        cells = []
        for sname, scorer in SCORERS.items():
            s, t, tie = evaluate(rows, scorer)
            cells.append(f"{s:.3f}/{t:.2f}/{tie:.2f}".rjust(20))
        print(f"{name:16s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
