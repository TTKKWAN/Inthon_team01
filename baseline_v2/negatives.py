"""Rule-based hard-negative descriptions built from each Train ground-truth description.

Allowed by rules 3-3 (rule-based augmentation): every negative is a deterministic
string edit of the provided Train text. No generative model, no teacher scores.
Vocabularies (objects, colors seen) are mined from the Train descriptions; the only
hand-written tables are generic English word pairs (antonym verbs, spatial words).

Negative types
  antonym          one action verb -> its opposite (opens -> closes, buttons -> unbuttons)
  spatial          left<->right, above<->below, in front of<->behind, ...
  attribute        one color or count word -> another
  object           one object noun -> another object seen in Train
  order            two event clauses swapped (or all reversed)
  append_unrelated GT + a sentence from another episode
  append_plausible GT + a templated extra action on an object of this scene
  replace_clause   one event clause -> a clause from another episode (same length class)
  drop_corrupt     one clause dropped, then an antonym/spatial/attribute/object edit
Random other-episode descriptions are added at training time, not stored here.

Each episode yields groups: the full GT, and (if it has >= 2 sentences) a short
positive = its first sentence, to cover shorter test descriptions. Negatives of a
group are always edits of that group's positive.

  python negatives.py --data-dir ../inthon2026_train/Train --out cache/negatives.json
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 2026
PER_TYPE = {"antonym": 2, "spatial": 1, "attribute": 1, "object": 2, "order": 2,
            "append_unrelated": 1, "append_plausible": 2, "replace_clause": 1,
            "drop_corrupt": 1}

# ---- generic word tables ---------------------------------------------------------
VERB_PAIRS = [
    ("open", "close"), ("lift", "lower"), ("raise", "lower"), ("pick up", "put down"),
    ("pick up", "set down"), ("put on", "take off"), ("button", "unbutton"),
    ("zip", "unzip"), ("tie", "untie"), ("wrap", "unwrap"), ("fold", "unfold"),
    ("pack", "unpack"), ("plug", "unplug"), ("lock", "unlock"), ("screw", "unscrew"),
    ("cover", "uncover"), ("fill", "empty"), ("push", "pull"), ("insert", "remove"),
    ("add", "remove"), ("turn on", "turn off"), ("stack", "unstack"), ("roll", "unroll"),
    ("attach", "detach"), ("sit down", "stand up"), ("cap", "uncap"), ("load", "unload"),
    ("tighten", "loosen"), ("connect", "disconnect"), ("show", "hide"), ("enter", "leave"),
    ("place", "remove"), ("take out", "put in"), ("appear", "disappear"),
]
IRREGULAR = {
    "put": ("puts", "putting", "put"), "set": ("sets", "setting", "set"),
    "sit": ("sits", "sitting", "sat"), "stand": ("stands", "standing", "stood"),
    "take": ("takes", "taking", "took"), "leave": ("leaves", "leaving", "left"),
    "hide": ("hides", "hiding", "hid"), "zip": ("zips", "zipping", "zipped"),
    "unzip": ("unzips", "unzipping", "unzipped"), "cap": ("caps", "capping", "capped"),
    "uncap": ("uncaps", "uncapping", "uncapped"), "plug": ("plugs", "plugging", "plugged"),
    "unplug": ("unplugs", "unplugging", "unplugged"), "empty": ("empties", "emptying", "emptied"),
    "show": ("shows", "showing", "showed"),
}
SPATIAL_PAIRS = [("left", "right"), ("above", "below"), ("in front of", "behind"),
                 ("top", "bottom"), ("upper", "lower"), ("inside", "outside"),
                 ("into", "out of"), ("toward", "away from"), ("front", "back"),
                 ("near", "far"), ("under", "over"), ("onto", "off")]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
          "black", "white", "gray", "silver", "gold", "beige"]
# "one" is left out: it is mostly a pronoun ("an oval one").
NUMBERS = ["two", "three", "four", "five", "six"]
COUNT_WORDS = {"once": "twice", "twice": "once", "single": "pair of"}
PLAUSIBLE_VERBS = ["pick up", "put down", "push", "open", "close", "lift", "turn over",
                   "shake", "wipe", "drop", "sit down on", "throw", "tap", "rotate", "squeeze"]
STOP = set("""a an the this that these those it its their his her them they he she one two
three four five six other another each both all some several same first second last next
end side edge top bottom middle front back left right view camera hand hands person man woman
table surface time way part pair piece row set rest out up down again away around apart aside
below above closer close open closed place action area angle arrangement center""".split())
DETERMINERS = r"(?:a|an|the|its|their|his|her)"
ADJ_HINT = re.compile(r"^(?:small|large|big|long|short|tall|thin|thick|flat|round|dark|light|"
                      r"pale|empty|metal|wooden|plastic|glass|paper|kitchen|white|clear|"
                      r"square|little|single|open|closed|full|new|old|striped|patterned|"
                      r"[a-z]+(?:ed|y|al|ic|ar|en|ful|less|ish|ous))$")

# ---- helpers ---------------------------------------------------------------------


def inflect(verb):
    """base -> [base, 3sg, -ing, past] for a (possibly phrasal) verb."""
    head, _, tail = verb.partition(" ")
    if head in IRREGULAR:
        s, ing, past = IRREGULAR[head]
    else:
        s = (head + "es" if re.search(r"(s|sh|ch|x|z|o)$", head)
             else head[:-1] + "ies" if re.search(r"[^aeiou]y$", head) else head + "s")
        ing = (head[:-1] + "ing" if head.endswith("e") and not head.endswith("ee")
               else head + "ing")
        past = head + "d" if head.endswith("e") else head + "ed"
    tail = f" {tail}" if tail else ""
    return [head + tail, s + tail, ing + tail, past + tail]


def build_verb_map():
    """surface form -> list of opposite surface forms (same inflection)."""
    mapping = {}
    for a, b in VERB_PAIRS:
        for fa, fb in zip(inflect(a), inflect(b)):
            mapping.setdefault(fa, []).append(fb)
            mapping.setdefault(fb, []).append(fa)
    return mapping


VERB_MAP = build_verb_map()
# Base forms collide with nouns/adjectives (a "close" call, "lower" shelf), so a bare
# base form is only swapped right after a subject-like word.
BASE_FORMS = {inflect(v)[0] for pair in VERB_PAIRS for v in pair}
BASE_CONTEXT = re.compile(r"\b(?:they|then|and|to|he|she|person|hand|hands)\s+$", re.I)


def match_case(src, new):
    return new[:1].upper() + new[1:] if src[:1].isupper() else new


def word_spans(text, phrase):
    return [m.span() for m in re.finditer(rf"\b{re.escape(phrase)}\b", text, re.I)]


def replace_span(text, span, new):
    a, b = span
    return text[:a] + match_case(text[a:b], new) + text[b:]


def swap_from_table(text, table, rng, context_ok=None):
    """Replace one occurrence of a word/phrase with one of its alternatives."""
    hits = []
    for src, alts in table.items():
        for span in word_spans(text, src):
            if context_ok and not context_ok(text, span, src):
                continue
            hits.append((span, alts))
    if not hits:
        return None
    # Prefer the longest match at a position (phrasal verbs before single words).
    span, alts = rng.choice(hits)
    return replace_span(text, span, rng.choice(alts))


def verb_context_ok(text, span, src):
    if src not in BASE_FORMS:
        return True
    return bool(BASE_CONTEXT.search(text[:span[0]]))


def pair_table(pairs):
    table = {}
    for a, b in pairs:
        table.setdefault(a, []).append(b)
        table.setdefault(b, []).append(a)
    return table


SPATIAL_TABLE = pair_table(SPATIAL_PAIRS)

# ---- clause segmentation (lossless, so clauses can be moved and re-joined) --------

_SEP = re.compile(
    r"((?<=[.!?])\s+|,\s*(?:and\s+)?then\s+|;\s*|\s+and\s+then\s+"
    r"|,\s*(?:before|after which|followed by)\s+)", re.I)
# Only boundaries between full events; "followed by"/"before" often join noun phrases.
_EVENT_SEP = re.compile(r"((?<=[.!?])\s+|,\s*(?:and\s+)?then\s+|;\s*|\s+and\s+then\s+)", re.I)
_MARKER = re.compile(r"^(?:finally|next|then|after that|first|lastly),?\s+", re.I)


def segment(text, sep=_SEP):
    """-> clauses, separators with text == c0 + s0 + c1 + ... + cn (+ final punct)."""
    text = text.strip()
    final = text[-1] if text[-1:] in ".!?" else ""
    body = text[:-1] if final else text
    parts = sep.split(body)
    clauses, seps = parts[0::2], parts[1::2]
    # Move each sentence-final '.' into the separator so clauses carry no end punctuation.
    for i, c in enumerate(clauses[:-1]):
        if c[-1:] in ".!?":
            clauses[i], seps[i] = c[:-1], c[-1] + seps[i]
    return clauses, seps, final


def join(clauses, seps, final):
    out = ""
    for i, c in enumerate(clauses):
        sentence_start = i == 0 or seps[i - 1][:1] in ".!?"
        c = c.strip()
        c = c[:1].upper() + c[1:] if sentence_start else (
            c[:1].lower() + c[1:] if not c.startswith("I ") else c)
        out += c + (seps[i] if i < len(seps) else "")
    return out + final


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


# ---- vocabulary mined from Train --------------------------------------------------


def mine_objects(texts, min_count=8):
    """Head nouns: det + (<= 2 adjective-like words) + noun, followed by a preposition or
    punctuation. Words that also show up in verb position are dropped."""
    phrase = re.compile(
        rf"\b{DETERMINERS}\s+((?:[a-z-]+\s+){{0,2}})([a-z]+)"
        r"(?=\s+(?:on|in|into|from|to|with|and|onto|beside|behind|near|at|of|under|over|"
        r"toward|across|inside|against|by)\b|[,.;])", re.I)
    counts = Counter()
    for t in texts:
        for m in phrase.finditer(t):
            if all(w in COLORS or ADJ_HINT.match(w) for w in m.group(1).lower().split()):
                counts[m.group(2).lower()] += 1
    verbish = Counter(m.group(1).lower() for t in texts for m in re.finditer(
        r"\b(?:they|person|hand|then|and)\s+([a-z]+)", t, re.I))
    return sorted(w for w, c in counts.items()
                  if c >= min_count and verbish[w] < c / 4 and w not in STOP
                  and w not in COLORS and w not in NUMBERS and w not in VERB_MAP
                  and len(w) > 2 and not w.endswith("ly"))


# ---- negative makers --------------------------------------------------------------


class Maker:
    def __init__(self, texts, objects, rng):
        """texts: {episode: description}. Pools remember their episode so that an
        appended / substituted piece never comes from the episode being edited."""
        self.rng = rng
        self.objects = objects
        self.object_set = set(objects)
        self.current = None
        self.pool_sentences = [(e, s) for e, t in texts.items() for s in sentences(t)[1:]
                               if re.match(r"(They|Then|Finally|Next|After|The person)\b", s)]
        self.pool_clauses = [(e, c.strip()) for e, t in texts.items()
                             for c in segment(t)[0] if len(c.split()) >= 4]

    def foreign(self, pool):
        while True:
            episode, text = self.rng.choice(pool)
            if episode != self.current:
                return text

    def objects_in(self, text):
        return [w for w in re.findall(r"[a-z]+", text.lower()) if w in self.object_set]

    def antonym(self, text):
        return swap_from_table(text, VERB_MAP, self.rng, verb_context_ok)

    def spatial(self, text):
        return swap_from_table(text, SPATIAL_TABLE, self.rng)

    def attribute(self, text):
        table = {c: [x for x in COLORS if x != c] for c in COLORS}
        table.update({n: [x for x in NUMBERS if x != n] for n in NUMBERS})
        table.update({k: [v] for k, v in COUNT_WORDS.items()})
        return swap_from_table(text, table, self.rng)

    def object(self, text):
        present = sorted(set(self.objects_in(text)))
        if not present:
            return None
        src = self.rng.choice(present)
        # Keep plurality so the edit does not break agreement (a cheap giveaway).
        plural = src.endswith("s") and not src.endswith("ss")
        new = self.rng.choice([o for o in self.objects if o not in present
                               and (o.endswith("s") and not o.endswith("ss")) == plural])
        spans = word_spans(text, src)
        # Replace every mention so the edit is consistent within the description.
        for span in reversed(spans):
            text = replace_span(text, span, new)
        return text

    def order(self, text):
        clauses, seps, final = segment(text, _EVENT_SEP)
        if len(clauses) < 2:
            return None
        # Discourse markers would give the swap away ("Finally, ..." first), so strip
        # them and put "Finally," back on whatever ends up last.
        had_final = bool(_MARKER.match(clauses[-1]))
        clauses = [_MARKER.sub("", c) for c in clauses]
        if len(clauses) > 2 and self.rng.random() < 0.3:
            clauses = clauses[::-1]
        else:
            i = self.rng.randrange(len(clauses) - 1)
            clauses[i], clauses[i + 1] = clauses[i + 1], clauses[i]
        if had_final:
            clauses[-1] = "Finally, " + clauses[-1][:1].lower() + clauses[-1][1:]
        return join(clauses, seps, final)

    def append_unrelated(self, text):
        return text.rstrip() + " " + self.foreign(self.pool_sentences)

    def append_plausible(self, text):
        present = sorted(set(self.objects_in(text))) or self.objects
        verb, obj = self.rng.choice(PLAUSIBLE_VERBS), self.rng.choice(present)
        if f"{verb} the {obj}" in text.lower():
            return None
        if self.rng.random() < 0.5:
            return text.rstrip() + " " + self.rng.choice(
                [f"Then they {verb} the {obj}.", f"Finally, they {verb} the {obj}.",
                 f"They then {verb} the {obj}."])
        # Agree with the subject of the last sentence ("they lift" vs "a person lifts").
        last = sentences(text)[-1]
        form = verb if re.search(r"\bthey\b", last, re.I) else inflect(verb)[1]
        return text.rstrip().rstrip(".") + f", and then {form} the {obj}."

    def replace_clause(self, text):
        clauses, seps, final = segment(text)
        if len(clauses) < 2:  # would just be a random other description
            return None
        i = self.rng.randrange(len(clauses))
        clauses[i] = self.foreign(self.pool_clauses)
        return join(clauses, seps, final)

    def drop_corrupt(self, text):
        clauses, seps, final = segment(text)
        if len(clauses) < 3:
            return None
        i = self.rng.randrange(1, len(clauses))
        del clauses[i]
        del seps[i - 1]
        edit = self.rng.choice([self.antonym, self.spatial, self.attribute, self.object])
        return edit(join(clauses, seps, final))

    def make(self, positive, scale=1):
        negatives, seen = [], {positive.strip().lower()}
        for kind, count in PER_TYPE.items():
            count *= scale
            fn = getattr(self, kind)
            made = 0
            for _ in range(count * 4):
                if made == count:
                    break
                neg = fn(positive)
                if neg and neg.strip().lower() not in seen:
                    seen.add(neg.strip().lower())
                    negatives.append([kind, neg.strip()])
                    made += 1
        return negatives


def build(data_dir, seed=SEED, scale=1):
    folders = sorted((p for p in Path(data_dir).glob("episode_*") if p.is_dir()),
                     key=lambda p: int(p.name.split("_")[1]))
    texts = {f.name: (f / "description.txt").read_text(encoding="utf-8").strip()
             for f in folders}
    objects = mine_objects(list(texts.values()))
    maker = Maker(texts, objects, random.Random(seed))
    out = {}
    for name, gt in texts.items():
        maker.current = name
        groups = [{"pos": gt, "negs": maker.make(gt, scale)}]
        sents = sentences(gt)
        if len(sents) >= 2:
            groups.append({"pos": sents[0], "negs": maker.make(sents[0], scale)})
        out[name] = groups
    return out, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scale", type=int, default=1,
                        help="multiply PER_TYPE counts (bigger bank to sample from)")
    args = parser.parse_args()
    out, objects = build(args.data_dir, args.seed, args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seed": args.seed, "scale": args.scale,
                                    "objects": objects, "episodes": out},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = Counter(k for groups in out.values() for g in groups for k, _ in g["negs"])
    n_groups = sum(len(g) for g in out.values())
    print(f"{len(out)} episodes, {n_groups} groups, {sum(kinds.values())} negatives, "
          f"{len(objects)} object words")
    print(dict(kinds))


if __name__ == "__main__":
    main()
