"""Tables from score.py output (CPU only). EVALUATION ONLY: fixed configs, nothing is tuned.

  1. 5-choice score per cell and config, with the other-video control, and the calibrated mix
     (validator/calibrated/weights_full.json: ab_ab_c 0.35, ab_ab_ac 0.55, share4 0.10)
  2. ab_ab_ac / share4 split by the kind of the shared edit a
  3. pairwise accuracy (GT vs one single-edit negative) of each debiased judge, by edit category
  4. the shipped config by source dataset

    python external_eval/temporalbench/report.py external_eval/temporalbench/results/score.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from score import CELLS, CONFIGS, order_of, rank_score

ROOT = Path(__file__).resolve().parents[2]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def cell_scores(rows, key=lambda r: r["cell"]):
    """{group: {config: (score, control score)}} plus the SigLIP zip run as-is."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = out[key(r)]
        for name, lam in CONFIGS.items():
            g[name].append(rank_score(order_of(r, lam), r["truth"]))
            g["ctrl_" + name].append(rank_score(order_of(r, lam, ctrl=True), r["truth"]))
        if "siglip_zip_order" in r:
            g["siglip_zip"].append(rank_score(r["siglip_zip_order"], r["truth"]))
            g["ctrl_siglip_zip"].append(rank_score(r["ctrl_siglip_zip_order"], r["truth"]))
    return {k: {c: (mean(v), len(v)) for c, v in g.items()} for k, g in out.items()}


def print_table(title, table, order=None):
    names = list(CONFIGS) + (["siglip_zip"] if any("siglip_zip" in t for t in table.values()) else [])
    print(f"\n## {title}\n")
    print("| group | n | " + " | ".join(names) + " | " + " | ".join(f"ctrl {n}" for n in names[1:]) + " |")
    print("|---" * (2 + 2 * len(names) - 1) + "|")
    for k in order or sorted(table):
        if k not in table:
            continue
        t = table[k]
        cells = [f"{t[n][0]:.3f}" for n in names] + [f"{t['ctrl_' + n][0]:.3f}" for n in names[1:]]
        print(f"| {k} | {t['span'][1]} | " + " | ".join(cells) + " |")


def pairwise(rows):
    """GT vs each negative in the `single` cell: P(judge_db(GT) > judge_db(neg)), ties 0.5."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in (r for r in rows if r["cell"] == "single"):
        t = r["ids"].index(r["truth"])
        for i, kind in enumerate(r["kinds"]):
            if i == t:
                continue
            for judge in ("z_align", "z_head", "ctrl_z_align", "ctrl_z_head"):
                d = r[judge][t] - r[judge][i]
                acc[kind][judge].append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
                acc["ALL"][judge].append(acc[kind][judge][-1])
    print("\n## Pairwise accuracy, single cell (GT vs one TemporalBench negative)\n")
    print("| edit category | pairs | SigLIP_db | head_db | ctrl SigLIP_db | ctrl head_db |")
    print("|---|---|---|---|---|---|")
    for kind in sorted(acc, key=lambda k: (k == "ALL", k)):
        a = acc[kind]
        print(f"| {kind} | {len(a['z_align'])} | " + " | ".join(
            f"{mean(a[j]):.3f}" for j in ("z_align", "z_head", "ctrl_z_align", "ctrl_z_head")) + " |")


def main():
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = data["rows"]
    print(f"rows {len(rows)}; shipped-config check {data['check']}")

    by_cell = cell_scores(rows)
    w = json.loads((ROOT / "validator/calibrated/weights_full.json").read_text())["weights"]
    mix = {}
    for name in by_cell["ab_ab_c"]:
        mix[name] = (sum(w[c] * by_cell[c][name][0] for c in CELLS if w.get(c)), by_cell["ab_ab_ac"][name][1])
    by_cell["calibrated_mix"] = mix
    print_table("5-choice score by cell (ctrl = frames from another video)", by_cell, CELLS + ["calibrated_mix"])

    for cell in ("ab_ab_ac", "share4"):
        print_table(f"{cell} by kind of the shared edit a",
                    cell_scores([r for r in rows if r["cell"] == cell], key=lambda r: r["shared_kind"]))
    pairwise(rows)
    print_table("calibrated cells (ab_ab_c, ab_ab_ac, share4 pooled) by source dataset",
                cell_scores([r for r in rows if r["cell"] != "single"], key=lambda r: r["dataset"]))


if __name__ == "__main__":
    main()
