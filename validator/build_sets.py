"""Collect 5-choice validation sets (626 valid episodes) into validator/sets/ in one format.

  claude_short : first-sentence GT + 4 negatives written by Claude (validator/parts/part_*.json).
                 VALIDATION ONLY, the user allowed Claude-written data here. Never train on it.
  qwen_full/short : GT + 4 of the Qwen3-4B negatives in team01's bank_qval
                 (huggingface SANGchu-03/INThon_Augmented, merged/bank_qval.json), seed-sampled.
  af_full      : baseline_v2/validation/val_af_full.json (rule negatives a text model found GT-like).

    python validator/build_sets.py
"""

import difflib
import json
import sys
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 2026


def write(name, rows):
    (HERE / "sets").mkdir(exist_ok=True)
    (HERE / f"sets/{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(name, len(rows))


def share_all(base, singles, rng, merge_edits):
    """[a, a+b, a+c, a+d] from single edits of base, or None if no a merges with three others."""
    singles = list(dict.fromkeys(singles))
    rng.shuffle(singles)
    for a in singles:
        merged = [m for m in (merge_edits(base, a, x) for x in singles if x != a) if m]
        merged = [m for m in dict.fromkeys(merged) if m not in (a, base)]
        if len(merged) >= 3:
            return [a] + rng.sample(merged, 3)
    return None


def main():
    valid = set(json.loads((ROOT / "baseline_v2/validation/splits.json").read_text())["valid"])

    rows = []
    for p in sorted((HERE / "parts").glob("part_*.json")):
        items = json.loads(p.read_text())
        inputs = json.loads((HERE / f"inputs/in_{p.stem.split('_')[1]}.json").read_text())
        if [x.get("episode") for x in items] != [x["episode"] for x in inputs]:
            print("skip unfinished", p.name)
            continue
        assert all(x["pos"] == i["pos"] and len(set(x["negs"])) == 4 and x["pos"] not in x["negs"]
                   for x, i in zip(items, inputs))
        rows += [{"episode": x["episode"],
                  "choices": [{"type": "pos", "text": x["pos"]}]
                  + [{"type": "claude", "text": t} for t in x["negs"]]} for x in items]
    if rows:
        assert all(r["episode"] in valid for r in rows)
        rows = sorted(rows, key=lambda r: int(r["episode"].split("_")[1]))
        write("claude_short", rows)
        # claude_ab_ac: negatives {a, b, a+b, a+c} from Claude's single edits a, b, c, merged by
        # rule code. Edits that share an edit with the majority, so centrality cannot just vote.
        sys.path.insert(0, str(ROOT / "diff_probe"))
        from sim_valsets import merge_edits
        rng = random.Random(SEED + 20)
        combo = []
        for r in rows:
            pos, singles = r["choices"][0]["text"], [c["text"] for c in r["choices"][1:]]
            for _ in range(6):
                a, b, c = rng.sample(singles, 3)
                negs = [a, b, merge_edits(pos, a, b), merge_edits(pos, a, c)]
                if None not in negs and len(set(negs + [pos])) == 5:
                    combo.append({"episode": r["episode"], "choices": [r["choices"][0]] + [
                        {"type": t, "text": x} for t, x in zip(["a", "b", "ab", "ac"], negs)]})
                    break
        write("claude_ab_ac", combo)

    # para_negs (structure from session inthon-0e): the positive is the untouched GT, the four
    # negatives are Qwen edits of Qwen paraphrases of the GT (B' in qwen_paraphrases_and_negatives).
    para = HERE / "hf/qwen_paraphrases_and_negatives.json"
    if para.exists():
        episodes = json.loads(para.read_text())["episodes"]
        for g, view in [(0, "full"), (1, "short")]:
            rng = random.Random(SEED + 10 + g)
            rows = []
            for ep in sorted(valid, key=lambda e: int(e.split("_")[1])):
                group = episodes[ep][g] if g < len(episodes[ep]) else None
                negs = [n for n in group["negs"] if n[1] != group["pos"]] if group else []
                if len(negs) < 4:
                    continue
                rows.append({"episode": ep, "choices": [{"type": "pos", "text": group["pos"]}]
                             + [{"type": kind, "text": t} for kind, t in rng.sample(negs, 4)]})
            write(f"para_negs_{view}", rows)

        # The positive is itself a Qwen paraphrase, so it no longer stands out by Train wording.
        #   para_same   : 4 Qwen edits of that same paraphrase (each edit in one negative)
        #   para_share3 : {a, a+b, a+c, a+d} from those edits (edit a shared by all four)
        same, share3 = {"full": [], "short": []}, {"full": [], "short": []}
        for g, view in [(0, "full"), (1, "short")]:
            rng = random.Random(SEED + 30 + g)
            for ep in sorted(valid, key=lambda e: int(e.split("_")[1])):
                if g >= len(episodes[ep]):
                    continue
                group = episodes[ep][g]
                paras = [t for _, t in group["pos_variants"] if t != group["pos"]]
                if not paras:
                    continue
                base = rng.choice(paras)
                # B' edits do not say which paraphrase they came from; take the closest one.
                own = [(kind, t) for kind, t in group["negs"] if t != base and max(
                    paras, key=lambda p: difflib.SequenceMatcher(None, p.split(), t.split()).ratio()) == base]
                head = [{"type": "pos", "text": base}]
                if len(own) >= 4:
                    same[view].append({"episode": ep, "choices": head + [
                        {"type": kind, "text": t} for kind, t in rng.sample(own, 4)]})
                negs = share_all(base, [t for _, t in own], rng, merge_edits)
                if negs:
                    share3[view].append({"episode": ep, "choices": head + [
                        {"type": t, "text": x} for t, x in zip(["a", "ab", "ac", "ad"], negs)]})
            write(f"para_same_{view}", same[view])
            write(f"para_share3_{view}", share3[view])

    gt_edits = HERE / "hf/qwen_negatives_from_gt.json"
    if gt_edits.exists():
        episodes = json.loads(gt_edits.read_text())["episodes"]
        for g, view in [(0, "full"), (1, "short")]:
            rng = random.Random(SEED + 40 + g)
            rows = []
            for ep in sorted(valid, key=lambda e: int(e.split("_")[1])):
                if g >= len(episodes[ep]):
                    continue
                group = episodes[ep][g]
                negs = share_all(group["pos"], [t for _, t in group["negs"] if t != group["pos"]],
                                 rng, merge_edits)
                if negs:
                    rows.append({"episode": ep, "choices": [{"type": "pos", "text": group["pos"]}] + [
                        {"type": t, "text": x} for t, x in zip(["a", "ab", "ac", "ad"], negs)]})
            write(f"qwen_share3_{view}", rows)

    bank = HERE / "hf/bank_qval.json"
    if bank.exists():
        episodes = json.loads(bank.read_text())["episodes"]
        for g, view in [(0, "full"), (1, "short")]:
            rng = random.Random(SEED + g)
            rows = []
            for ep in sorted(episodes, key=lambda e: int(e.split("_")[1])):
                groups = episodes[ep]
                if ep not in valid or g >= len(groups) or len(groups[g]["negs"]) < 4:
                    continue
                negs = rng.sample(groups[g]["negs"], 4)
                rows.append({"episode": ep, "choices": [{"type": "pos", "text": groups[g]["pos"]}]
                             + [{"type": kind, "text": t} for kind, t in negs]})
            write(f"qwen_{view}", rows)

    af = json.loads((ROOT / "baseline_v2/validation/val_af_full.json").read_text())
    write("af_full", af)


if __name__ == "__main__":
    main()
