"""Positive paraphrases (same meaning, other words) + negatives made from them, by the same
local LLM as llm_negatives.py.

If only negatives are LLM-written, "LLM wording = wrong" becomes a new text shortcut. Here the
positives get the same treatment: each positive is rewritten (a) freely and (b) in the
present progressive ("a person is lifting ..."), keeping objects, actions, order, colors,
counts and positions. A rewrite is dropped unless every color / number / position word and
every Train object word of the original survives. Then the five edit types of
llm_negatives.py are applied to the surviving paraphrases, so LLM-worded positives and
LLM-worded negatives come in pairs.

  python llm_paraphrase.py --out cache/llm_paraphrases.json --first-sentence
"""

import argparse
import json
import re
import time
from pathlib import Path

from llm_negatives import (EDITS, MODEL_ID, MODEL_REVISION, TASK, clean, generate, load_gt,
                           load_model)
from negatives import COLORS, NUMBERS, mine_objects, segment, sentences

ROOT = Path(__file__).resolve().parent
PARA_TASK = ("Original description:\n{text}\n\n{how} Keep every object, every action, the "
             "order of events, and all colors, sizes, counts and positions exactly as they "
             "are; do not add, drop or change any detail. Output only the rewritten "
             "description.")
STYLES = {
    "para_rewrite": "Rewrite the description using different words and sentence structure "
                    "but exactly the same meaning.",
    "para_progressive": "Rewrite the description in the present progressive tense (for "
                        "example 'a person is lifting ...').",
}
KEEP_WORDS = set(COLORS) | set(NUMBERS) | {"one", "left", "right", "top", "bottom", "front",
                                           "behind", "above", "below", "under", "inside"}


def preserved(original, rewrite, objects):
    """Every color / number / position word and Train object word must survive."""
    words = set(re.findall(r"[a-z]+", rewrite.lower()))
    stems = words | {w[:-1] for w in words if w.endswith("s")} | {w + "s" for w in words}
    needed = {w for w in re.findall(r"[a-z]+", original.lower())
              if w in KEEP_WORDS or w in objects}
    return needed <= stems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models/qwen3-4b-instruct-2507")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "../data/Train")
    parser.add_argument("--first-sentence", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    gts = load_gt(args.data_dir)
    objects = set(mine_objects(list(gts.values())))
    episodes = list(gts)[:args.limit] if args.limit else list(gts)
    positives = []  # (episode, group index, text)
    for ep in episodes:
        positives.append((ep, 0, gts[ep]))
        if args.first_sentence and len(sentences(gts[ep])) >= 2:
            positives.append((ep, 1, sentences(gts[ep])[0]))
    tok, model = load_model(args.model_dir)
    t0 = time.time()

    # 1) paraphrase every positive in each style
    jobs = [(p, style) for p in positives for style in STYLES]
    raw = generate(tok, model, [PARA_TASK.format(text=p[2], how=STYLES[s]) for p, s in jobs],
                   [p[2] for p, _ in jobs], args.batch_size)
    out = {ep: [] for ep in episodes}
    for ep, gi, text in positives:
        while len(out[ep]) <= gi:
            out[ep].append({"pos": None, "pos_variants": [], "negs": []})
        out[ep][gi]["pos"] = text
    variants, dropped = [], 0
    for ((ep, gi, text), style), r in zip(jobs, raw):
        v = clean(r, text)
        if v and preserved(text, v, objects):
            out[ep][gi]["pos_variants"].append([style, v])
            variants.append((ep, gi, v))
        else:
            dropped += 1
    print(f"paraphrases kept {len(variants)}/{len(jobs)} ({dropped} dropped by checks)",
          flush=True)

    # 2) the same five edits, applied to the paraphrases
    neg_jobs = [(ep, gi, kind, v) for ep, gi, v in variants for kind in EDITS
                if not (kind == "llm_order" and len(segment(v)[0]) < 2)]
    raw = generate(tok, model, [TASK.format(text=v, how=EDITS[k]) for _, _, k, v in neg_jobs],
                   [v for _, _, _, v in neg_jobs], args.batch_size)
    kept = 0
    for (ep, gi, kind, v), r in zip(neg_jobs, raw):
        neg = clean(r, v)
        if neg:
            out[ep][gi]["negs"].append([kind, neg])
            kept += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION,
                                    "dtype": "float16", "decoding": "greedy",
                                    "styles": STYLES, "edits": EDITS, "episodes": out},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: {len(variants)} paraphrases, {kept}/{len(neg_jobs)} negatives from them, "
          f"{time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
