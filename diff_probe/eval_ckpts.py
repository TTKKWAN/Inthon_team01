"""Pick an epoch on one half of the validation episodes, report on the other half.

Every set in valsets.json is built from the same 626 validation episodes, so choosing a
checkpoint on some sets and reporting on others still reuses the same videos. Here the
episodes are split in two (alternating, by episode number): half A picks the epoch by the
mean score over ALL structure sets, half B reports that epoch under normal / reversed /
other-video frames.

    python diff_probe/eval_ckpts.py --run diff_probe/runs/marked_v1
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from train_marked import Frames, MarkedVerifier, TextEncoder, evaluate  # noqa: E402

SETS = ["short/H1_disjoint", "short/H2_samespot", "short/H3_multi", "short/H4_chain",
        "short/H5_mix", "short/H6_fact_a_b_ab_c", "short/H7_fact_a_b_ab_ac", "full/H5_mix"]


def split_halves(valsets):
    eps = sorted({r["episode"] for rows in valsets.values() for r in rows},
                 key=lambda e: int(e.split("_")[1]))
    a = set(eps[0::2])
    return ({k: [r for r in v if r["episode"] in a] for k, v in valsets.items()},
            {k: [r for r in v if r["episode"] not in a] for k, v in valsets.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "runs/marked_v1")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--valsets", type=Path, default=HERE / "valsets.json")
    ap.add_argument("--text-model", default=str(HERE / "models/mpnet"))
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    valsets = json.loads(args.valsets.read_text())["sets"]
    valsets = {k: valsets[k] for k in SETS}
    half_a, half_b = split_halves(valsets)
    frames = Frames(args.frames, args.max_frames)
    text_enc = TextEncoder(args.text_model, args.device)
    model = MarkedVerifier(max_frames=32).to(args.device)

    ckpts = sorted(args.run.glob("epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    table = []
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=args.device)["model"])
        scores = {k: evaluate(model, text_enc, rows, frames, args.device)["score"]
                  for k, rows in half_a.items()}
        mean = sum(scores.values()) / len(scores)
        table.append({"ckpt": ck.name, "mean_A": round(mean, 4), "A": scores})
        print(json.dumps(table[-1]), flush=True)

    best = max(table, key=lambda r: r["mean_A"])
    model.load_state_dict(torch.load(args.run / best["ckpt"], map_location=args.device)["model"])
    report = {}
    for k, rows in half_b.items():
        report[k] = {c: evaluate(model, text_enc, rows, frames, args.device, c)
                     for c in ("normal", "reversed", "other")}
        print("B", k, json.dumps(report[k]), flush=True)
    out = {"selected": best["ckpt"], "selection_half_A": table, "report_half_B": report}
    (args.run / "eval_ckpts.json").write_text(json.dumps(out, indent=1))
    print("selected", best["ckpt"], best["mean_A"])


if __name__ == "__main__":
    main()
