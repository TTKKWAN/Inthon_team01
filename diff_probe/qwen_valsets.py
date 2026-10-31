"""5-choice validation sets from the team's Qwen augmentation (fluent negatives, no rule artifacts).

Input: the HF dataset SANGchu-03/INThon_Augmented @ cbb24e162e35128742da011eb41aa289b007ee09 (made by baseline_v2/llm_negatives.py and
llm_paraphrase.py on origin/Sangjun, Qwen3-4B-Instruct-2507 fp16, greedy):
  qwen/qwen_negatives_from_gt.json          A' = GT edited in one way
  qwen/qwen_paraphrases_and_negatives.json  B = paraphrase of the GT, B' = B edited in one way
Only the 626 validation episodes are used. Output has the valsets.json format.

  A_single       GT + 4 A'                       (every edit in one negative: centrality wins)
  para_negs      GT + 4 B'                       (negatives reworded: GT is the odd one out)
  fact_ab_ab_c   GT + {a, b, a+b, c} from A'     (edits shared by 2 negatives)
  fact_ab_ab_ac  GT + {a, b, a+b, a+c} from A'   (edit a in 3 negatives: centrality flips)

    python diff_probe/qwen_valsets.py --hf-dir <download dir> --out diff_probe/valsets_qwen.json
"""

import argparse
import json
import random
from pathlib import Path

from sim_valsets import merge_edits

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
KINDS = ["A_single", "para_negs", "fact_ab_ab_c", "fact_ab_ab_ac"]


def build(kind, gt, a_negs, b_negs, rng):
    if kind == "A_single":
        return [("pos", gt)] + [(t, x) for t, x in rng.sample(a_negs, 4)] if len(a_negs) >= 4 else None
    if kind == "para_negs":
        return [("pos", gt)] + [(f"para:{t}", x) for t, x in rng.sample(b_negs, 4)] if len(b_negs) >= 4 else None
    if len(a_negs) < 3:
        return None
    (ka, a), (kb, b), (kc, c) = rng.sample(a_negs, 3)
    ab, ac = merge_edits(gt, a, b), merge_edits(gt, a, c)
    if kind == "fact_ab_ab_c":
        negs = [(f"fact:{ka}", a), (f"fact:{kb}", b), (f"fact:{ka}+{kb}", ab), (f"fact:{kc}", c)]
    else:
        negs = [(f"fact:{ka}", a), (f"fact:{kb}", b), (f"fact:{ka}+{kb}", ab), (f"fact:{ka}+{kc}", ac)]
    if any(x is None for _, x in negs) or len({x for _, x in negs} | {gt}) < 5:
        return None
    return [("pos", gt)] + negs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "valsets_qwen.json")
    args = ap.parse_args()
    valid = sorted(json.loads(args.splits.read_text())["valid"], key=lambda e: int(e.split("_")[1]))
    a_bank = json.loads((args.hf_dir / "qwen/qwen_negatives_from_gt.json").read_text())["episodes"]
    b_bank = json.loads((args.hf_dir / "qwen/qwen_paraphrases_and_negatives.json").read_text())["episodes"]
    sets = {}
    for kind in KINDS:
        rng = random.Random(SEED)
        for gi, length in ((0, "full"), (1, "short")):
            rows = []
            for ep in valid:
                ga, gb = a_bank[ep][gi], b_bank[ep][gi]
                gt = ga["pos"]
                assert gb["pos"] == gt, ep
                a_negs = [tuple(x) for x in ga["negs"] if x[1] != gt]
                b_negs = [tuple(x) for x in gb["negs"] if x[1] != gt]
                ch = build(kind, gt, a_negs, b_negs, rng)
                if ch:
                    rows.append({"episode": ep, "choices": [{"type": t, "text": x} for t, x in ch]})
            sets[f"{length}/{kind}"] = rows
            print(f"{length}/{kind}", len(rows))
    args.out.write_text(json.dumps({"seed": SEED, "source": "SANGchu-03/INThon_Augmented", "sets": sets}))


if __name__ == "__main__":
    main()
