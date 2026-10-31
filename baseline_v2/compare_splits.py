"""Compare the clause_split prototype with model_v2.split_clauses (no training involved).

Reports, for Train descriptions: pieces per description, words per piece, pieces that still
hold two or more verbs, pieces without any verb, word coverage (what the split dropped) and the
longest SigLIP token count. For the shared validation set: how much of each negative ends up
in pieces that differ from the positive's pieces (lower = the edit is better isolated).

  python compare_splits.py --data-dir ../data/Train --val validation/val_full.json \
      --siglip-tokenizer models/siglip/tokenizer.json [--save-lexicon data/verb_lexicon.json]
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from clause_split import WORD, Segmenter, mine_lexicon  # noqa: E402
from model_v2 import split_clauses as old_split  # noqa: E402


def load_descriptions(data_dir):
    folders = sorted((p for p in Path(data_dir).glob("episode_*") if p.is_dir()),
                     key=lambda p: int(p.name.split("_")[1]))
    return [(f / "description.txt").read_text(encoding="utf-8").strip() for f in folders]


def train_stats(name, split, texts, seg, tokenizer):
    pieces = [split(t) for t in texts]
    flat = [p for ps in pieces for p in ps]
    counts = [len(ps) for ps in pieces]
    words = [len(p.split()) for p in flat]
    verbs = [sum(seg.finite(w) for w in WORD.findall(p)) for p in flat]
    no_verb = sum(not seg.has_verb(p) for p in flat)
    missing = Counter()
    total = 0
    for text, ps in zip(texts, pieces):
        orig = Counter(w.lower() for w in WORD.findall(text))
        got = Counter(w.lower() for p in ps for w in WORD.findall(p))
        missing += orig - got
        total += sum(orig.values())
    print(f"[{name}] pieces/description: mean {statistics.mean(counts):.2f}, "
          f"histogram {dict(sorted(Counter(counts).items()))}")
    print(f"[{name}] words/piece: median {statistics.median(words)}, "
          f"p90 {statistics.quantiles(words, n=10)[-1]:.0f}, max {max(words)}")
    print(f"[{name}] pieces with >= 2 verb forms: {sum(v >= 2 for v in verbs) / len(flat):.1%}; "
          f"pieces with no verb: {no_verb / len(flat):.1%}")
    print(f"[{name}] word coverage: {1 - sum(missing.values()) / total:.2%}; "
          f"most dropped: {missing.most_common(12)}")
    if tokenizer:
        lengths = [len(tokenizer.encode(p.lower()).ids) for p in flat]
        print(f"[{name}] SigLIP tokens/piece: median {statistics.median(lengths)}, max {max(lengths)}, "
              f"> 64: {sum(n > 64 for n in lengths)}")


def localization(val, split):
    """Pieces are compared as lowercase word sequences, so case / punctuation changes from
    re-joining clauses do not count as edits."""
    def norm(text):
        return [" ".join(w.lower() for w in WORD.findall(p)) for p in split(text)]

    by_type, order_reordered = defaultdict(list), []
    for q in val:
        pos = norm(q["choices"][0]["text"])
        for choice in q["choices"][1:]:
            if choice["type"] == "random":
                continue
            neg = norm(choice["text"])
            left, changed = Counter(pos), []
            for piece in neg:
                if left[piece]:
                    left[piece] -= 1
                else:
                    changed.append(piece)
            share = sum(len(p.split()) for p in changed) / max(1, sum(len(p.split()) for p in neg))
            by_type[choice["type"]].append((share, len(changed)))
            if choice["type"] == "order":
                order_reordered.append(Counter(pos) == Counter(neg) and pos != neg)
    return by_type, order_reordered


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val", type=Path, help="validation/val_full.json")
    parser.add_argument("--siglip-tokenizer", type=Path, help="SigLIP tokenizer.json")
    parser.add_argument("--save-lexicon", type=Path)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    texts = load_descriptions(args.data_dir)
    lexicon = mine_lexicon(texts)
    print(f"lexicon: {len(lexicon['bases'])} verb bases, {len(lexicon['nounish'])} noun-like forms")
    if args.save_lexicon:
        args.save_lexicon.parent.mkdir(parents=True, exist_ok=True)
        args.save_lexicon.write_text(json.dumps(lexicon, indent=1), encoding="utf-8")
    seg = Segmenter(lexicon)
    tokenizer = None
    if args.siglip_tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.siglip_tokenizer))

    splitters = {"old": old_split, "new": seg.texts}
    for name, split in splitters.items():
        train_stats(name, split, texts, seg, tokenizer)
    pieces = [p for t in texts for p in seg.split(t)]
    print(f"[new] state pieces: {sum(p.kind == 'state' for p in pieces)}; "
          f"rules: {dict(Counter(p.rule for p in pieces).most_common())}")

    if args.val:
        val = json.loads(args.val.read_text(encoding="utf-8"))
        results = {name: localization(val, split) for name, split in splitters.items()}
        print(f"\nvalidation {args.val.name}: share of negative words in pieces that differ from "
              f"the positive (lower = edit isolated) / pieces changed")
        for kind in sorted(results["old"][0]):
            row = []
            for name in splitters:
                vals = results[name][0][kind]
                row.append(f"{name} {statistics.mean(v[0] for v in vals):.3f} / "
                           f"{statistics.mean(v[1] for v in vals):.2f}")
            print(f"  {kind:17s} n={len(results['old'][0][kind]):4d}  " + "  |  ".join(row))
        for name in splitters:
            reordered = results[name][1]
            print(f"  order negatives that are a pure reordering of the positive's pieces "
                  f"[{name}]: {sum(reordered) / max(1, len(reordered)):.1%}")

    rng = random.Random(args.seed)
    for text in rng.sample(texts, args.examples):
        print("\n" + "=" * 100 + "\n" + text)
        print("-- old:")
        for p in old_split(text):
            print(f"   | {p}")
        print("-- new:")
        for p in seg.split(text):
            print(f"   | t={p.t} {p.kind:5s} {p.rule:11s} {p.text}")


if __name__ == "__main__":
    main()
