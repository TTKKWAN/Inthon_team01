"""For each pair of cells (one-edit-per-negative S, shared-edit H) find mixing weights p
(score = (1-p)*S + p*H) at which the submissions rank like the public LB.

Reads validator/results*/NAME.json (eval_subs.py output, copied from the server) and
validator/results/centrality.json.

    python validator/mix_order.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LB = {"starter": 0.350, "marked_v1": 0.413, "minimal": 0.426, "v2int8": 0.439, "central_byte": 0.52}
PAIRS = [("para_same_full", "para_share3_full"), ("para_same_short", "para_share3_short"),
         ("qwen_full", "qwen_share3_full"), ("qwen_short", "qwen_share3_short"),
         ("claude_short", "claude_ab_ac")]


def load():
    scores = {}
    for d in sorted(HERE.glob("results*")):
        for f in d.glob("*.json"):
            for model, sets in json.loads(f.read_text()).items():
                for s, r in sets.items():
                    scores.setdefault(model, {})[s] = r["score"]
    return scores


def main():
    scores = load()
    models = [m for m in LB if m in scores]
    strict = sorted(models, key=LB.get)
    for s_cell, h_cell in PAIRS:
        if not all(s_cell in scores[m] and h_cell in scores[m] for m in models):
            print(f"{s_cell} + {h_cell}: missing results")
            continue
        ok_strict, ok_loose, best = [], [], None
        for k in range(101):
            p = k / 100
            mix = {m: (1 - p) * scores[m][s_cell] + p * scores[m][h_cell] for m in models}
            order = sorted(models, key=mix.get)
            if order == strict:
                ok_strict.append(p)
            mid = [m for m in models if m not in ("starter", "central_byte")]
            if order[0] == "starter" and order[-1] == "central_byte":
                ok_loose.append(p)
            err = sum((mix[m] - LB[m]) ** 2 for m in models) ** 0.5
            if best is None or err < best[0]:
                best = (err, p, mix)
        rng = lambda ps: f"{min(ps):.2f}-{max(ps):.2f} ({len(ps)} steps)" if ps else "none"
        err, p, mix = best
        print(f"{s_cell} + {h_cell}: strict {rng(ok_strict)} | starter low & centrality top {rng(ok_loose)}")
        print(f"   closest to LB values at p={p:.2f} (rmse {err / len(models) ** 0.5:.3f}): "
              + "  ".join(f"{m}={mix[m]:.3f}" for m in strict))


if __name__ == "__main__":
    main()
