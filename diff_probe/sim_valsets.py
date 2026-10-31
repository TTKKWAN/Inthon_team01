"""Local 5-choice sets that differ only in HOW the 4 negatives relate to the positive.

Built from the 626 validation episodes with the rule generator in baseline_v2/negatives.py,
so every sentence is a code edit of a Train description. Each set tests one guess about the
hidden Test generator; comparing their centrality scores with the leaderboard (byte = 0.52)
tells which guesses are consistent with Test.

  H1_disjoint  4 single edits of the positive, each of a different type
  H2_samespot  2-3 negatives replace the SAME word of the positive, the rest are single edits
  H3_multi     every negative carries 2 independent single edits
  H4_chain     2 negatives are single edits, 2 are a further edit of one of those negatives
  H5_mix       per episode, one of H1..H4 at random

    python diff_probe/sim_valsets.py --out diff_probe/valsets.json
"""

import argparse
import difflib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline_v2"))
import negatives as ng  # noqa: E402

SEED = 7
SINGLE = ["antonym", "spatial", "attribute", "object", "order", "append_plausible",
          "append_unrelated", "replace_clause"]
WORD_SWAPS = ["antonym", "spatial", "attribute", "object"]


def single(maker, text, kinds, seen):
    for _ in range(30):
        kind = maker.rng.choice(kinds)
        out = getattr(maker, kind)(text)
        if out and out.strip().lower() not in seen:
            seen.add(out.strip().lower())
            return kind, out.strip()
    return None


def changed_word_index(a, b):
    """Index of the single replaced word, or None if the edit is not one word."""
    wa, wb = a.split(), b.split()
    ops = [o for o in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
           if o[0] != "equal"]
    if len(ops) == 1 and ops[0][0] == "replace" and ops[0][2] - ops[0][1] == 1:
        return ops[0][1]
    return None


def h1(maker, pos, seen):
    kinds = maker.rng.sample(SINGLE, len(SINGLE))
    out = []
    for kind in kinds:
        if len(out) == 4:
            break
        r = single(maker, pos, [kind], seen)
        if r:
            out.append(r)
    return out


def h2(maker, pos, seen):
    # Collect many one-word replacements, group them by the word they replace.
    groups = {}
    for _ in range(60):
        r = single(maker, pos, WORD_SWAPS, set(seen))
        if not r:
            continue
        idx = changed_word_index(pos, r[1])
        if idx is not None:
            groups.setdefault(idx, {})[r[1].lower()] = r
    shared = [g for g in groups.values() if len(g) >= 2]
    if not shared:
        return None
    group = list(maker.rng.choice(shared).values())
    k = min(len(group), maker.rng.choice([2, 3]))
    out = maker.rng.sample(group, k)
    for _, t in out:
        seen.add(t.lower())
    while len(out) < 4:
        r = single(maker, pos, SINGLE, seen)
        if not r:
            break
        out.append(r)
    return [(f"same:{kind}", t) for kind, t in out[:k]] + out[k:]


def h3(maker, pos, seen):
    out = []
    for _ in range(12):
        if len(out) == 4:
            break
        a = single(maker, pos, SINGLE, set(seen))
        if not a:
            continue
        b = single(maker, a[1], WORD_SWAPS + ["order"], set(seen))
        if b and b[1].lower() not in seen:
            seen.add(b[1].lower())
            out.append((f"{a[0]}+{b[0]}", b[1]))
    return out


def h4(maker, pos, seen):
    base = [single(maker, pos, SINGLE, seen) for _ in range(2)]
    base = [b for b in base if b]
    out = list(base)
    for b in base:
        r = single(maker, b[1], WORD_SWAPS + ["order"], seen)
        if r:
            out.append((f"chain:{b[0]}>{r[0]}", r[1]))
    return out


def merge_edits(pos, a, b):
    """Apply the word-level edits of both a and b (each an edit of pos) to pos.
    Returns None when the two edits touch overlapping words."""
    wp = pos.split()
    ops = []
    for other in (a, b):
        wo = other.split()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wp, wo, autojunk=False).get_opcodes():
            if tag != "equal":
                ops.append((i1, i2, wo[j1:j2]))
    ops.sort(key=lambda o: (o[0], o[1]))
    for (s1, e1, _), (s2, e2, _) in zip(ops, ops[1:]):
        if s2 < e1 or (s2 == s1 and e1 == s1):  # overlap, or two inserts at one point
            return None
    out, cursor = [], 0
    for s, e, new in ops:
        out += wp[cursor:s] + new
        cursor = e
    return " ".join(out + wp[cursor:])


def factorial(maker, pos, seen, pattern):
    """Negatives are combinations of a few single edits, e.g. {a}, {b}, {a+b}, {c}."""
    for _ in range(10):
        edits = []
        for _ in range(12):
            if len(edits) == 3:
                break
            r = single(maker, pos, SINGLE, set(seen) | {e[1].lower() for e in edits})
            if r:
                edits.append(r)
        if len(edits) < 3:
            return None
        (ka, a), (kb, b), (kc, c) = edits
        ab = merge_edits(pos, a, b)
        ac = merge_edits(pos, a, c)
        table = {"a": (ka, a), "b": (kb, b), "c": (kc, c),
                 "ab": (f"{ka}+{kb}", ab), "ac": (f"{ka}+{kc}", ac)}
        out = [table[k] for k in pattern]
        if all(t for _, t in out) and len({t.lower() for _, t in out}) == 4:
            seen.update(t.lower() for _, t in out)
            return [(f"fact:{k}", t) for k, t in out]
    return None


BUILDERS = {"H1_disjoint": h1, "H2_samespot": h2, "H3_multi": h3, "H4_chain": h4,
            "H6_fact_a_b_ab_c": lambda m, p, s: factorial(m, p, s, ["a", "b", "ab", "c"]),
            "H7_fact_a_b_ab_ac": lambda m, p, s: factorial(m, p, s, ["a", "b", "ab", "ac"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=ROOT / "Train" if (ROOT / "Train").exists() else ROOT / "data/Train")
    ap.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    ap.add_argument("--out", type=Path, default=ROOT / "diff_probe/valsets.json")
    args = ap.parse_args()

    folders = sorted((p for p in args.data_dir.glob("episode_*") if p.is_dir()),
                     key=lambda p: int(p.name.split("_")[1]))
    texts = {f.name: (f / "description.txt").read_text(encoding="utf-8").strip()
             for f in folders}
    valid = json.loads(args.splits.read_text())["valid"]
    maker = ng.Maker(texts, ng.mine_objects(list(texts.values())), random.Random(SEED))

    sets = {}
    # Test descriptions are one sentence (competition slide), so also build every set
    # from the first sentence of each Train description.
    for length in ("full", "short"):
        for name in list(BUILDERS) + ["H5_mix"]:
            sets[f"{length}/{name}"] = []
        for ep in valid:
            pos = texts[ep] if length == "full" else ng.sentences(texts[ep])[0]
            maker.current = ep
            for name, fn in BUILDERS.items():
                negs = fn(maker, pos, {pos.lower()})
                if negs and len(negs) == 4:
                    sets[f"{length}/{name}"].append(
                        {"episode": ep, "choices": [{"type": "pos", "text": pos}] +
                         [{"type": k, "text": t} for k, t in negs]})
            pick = maker.rng.choice(list(BUILDERS))
            negs = BUILDERS[pick](maker, pos, {pos.lower()})
            if negs and len(negs) == 4:
                sets[f"{length}/H5_mix"].append(
                    {"episode": ep, "source": pick, "choices":
                     [{"type": "pos", "text": pos}] + [{"type": k, "text": t} for k, t in negs]})
    args.out.write_text(json.dumps({"seed": SEED, "sets": sets}, indent=1))
    for name, rows in sets.items():
        print(f"{name:12s} {len(rows)} episodes")


if __name__ == "__main__":
    main()
