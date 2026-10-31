"""Rule-based event segmentation of descriptions (prototype, not wired into model_v2 yet).

Each description is cut into pieces that each describe one event a viewer can check in the
frames, put in time order, and static background ("while the cup stays on the table") is kept
apart as a "state" piece. Only `re` and a verb lexicon mined from the Train descriptions are
used, so this runs in the grading environment (no spaCy / NLTK).

Steps per sentence:
  1. drop a leading discourse marker (Then, Next, Finally, After that, ...)
  2. sentence-initial "After / Once / Before X, Y" -> X, Y  (Before: Y, X)
  3. explicit connectors: ", then", "and then", ";", ", after which", "before -ing",
     ", before", "until", "followed by", "while", ", <verb>ing <object>", and mid-sentence
     "after -ing" (that event happened first, so it is moved in front)
  4. verb coordination ", (and) VERB" / "and VERB": only between two verb phrases, when the
     verb agrees with the subject; noun-like verbs ("caps", "turn") also need an object or
     particle right after them
  5. "VERB A, B, and C one after another / one at a time / in turn" -> one piece per item
  6. every event piece gets the sentence subject back, "-ing" pieces get a finite verb, and
     "followed by NP" reuses the verb of the previous piece
The lexicon is data mined from Train by `mine_lexicon`; ship it as JSON with a submission.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MIN_WORDS = 3
MAX_PIECES = 12

SENTENCE = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
MARKER = re.compile(r"^(?:then|next|finally|first|lastly|after that|afterwards?|eventually)\b,?\s*",
                    re.I)
LEAD = re.compile(r"^(?P<conj>after|once|as soon as|before)\s+(?P<sub>[^,]+?),\s*(?P<main>.+)$",
                  re.I | re.S)
CONNECTOR = re.compile(
    r"(?P<then>,\s*(?:and\s+)?then\s+|\s+and\s+then\s+)"
    r"|(?P<semi>\s*;\s*)"
    r"|(?P<afterwhich>,\s*after which\s+)"
    r"|(?P<before>,\s*before\s+|\s+before\s+(?=[a-z]+ing\b))"
    r"|(?P<until>,?\s+until\s+)"
    r"|(?P<after>,?\s+after\s+(?=[a-z]+ing\b))"
    r"|(?P<followed>,?\s*followed by\s+)"
    r"|(?P<while>,?\s+while\s+)"
    r"|(?P<participle>,\s+(?=[a-z]+ing\s))",
    re.I)
COORD = re.compile(r"(?:,\s+and\s+|,\s+|\s+and\s+)(?=([A-Za-z]+)\b)")
# ", and they ..." / " and the cup ...": a new clause with its own subject.
CLAUSE = re.compile(r"(?:,\s+and\s+|\s+and\s+|,\s+but\s+)(?=(?:they|he|she|it|the|a|an|this|that|its|"
                    r"their|his|her|each|both|one|two)\b)", re.I)
LIST_MARK = re.compile(r",?\s*\b(?:one after (?:another|the other)|one at a time|one by one|in turn)\b",
                       re.I)

DETERMINERS = set("""a an the this that these those its their his her each both all some another
other one two three four five six""".split())
PARTICLES = set("up down out back over away off aside around in on".split())
# Words that may follow a coordinated verb (object start, particle or preposition).
FOLLOW = DETERMINERS | PARTICLES | set("""it them into onto to toward towards from across through
behind under above below near beside between with at by together apart forward again along
inside outside""".split())
ADVERBS = set("""next then also again still now first finally briefly carefully slowly quickly
gently just once twice repeatedly immediately lastly""".split())
NOT_VERB = DETERMINERS | ADVERBS | set("""it them they he she we and or but with on in into onto
from by of at to up down out back over away around off together above below near behind beside
under is are was were be been being has have had do does did can will would should may might as
so very not no only even almost nearly still across after apart before between forward toward
towards through along against inside outside without within beneath upon past beyond instead
upright flat""".split())
AUX = {"is", "are", "was", "were"}
# Irregular past participles for passive groups ("is laid", "are shown").
PARTICIPLES = set("""laid held shown drawn put set left taken given brought thrown spun hung kept
made seen worn torn cut built stuck caught struck split""".split())
# A subject noun phrase starts with one of these ("a person", "they", "each card").
SUBJECT_START = DETERMINERS | {"they", "he", "she", "it", "we", "someone", "people"}
STATE_VERBS = set("""remain remains remaining stay stays staying rest rests resting lie lies lying
sit sits sitting stand stands standing""".split())
SUBJECT_HINT = re.compile(r"\b(?:they|he|she|person|man|woman|hand|hands|someone|then)\s+([a-z]+)\b",
                          re.I)
# More verb contexts: infinitives / coordinated verbs followed by an object ("to lift the",
# "and connects the").
OBJECT_HINT = re.compile(r"\b(?:to|and)\s+([a-z]+)\s+(?:the|a|an|it|them|its|their|his|her|both|each)\b",
                         re.I)


def third_person(base):
    if re.search(r"(?:s|sh|ch|x|z|o)$", base):
        return base + "es"
    if re.search(r"[^aeiou]y$", base):
        return base[:-1] + "ies"
    return base + "s"


def to_base(word):
    if word.endswith("ies") and len(word) > 4 and word not in {"lies", "ties", "dies", "unties"}:
        return word[:-3] + "y"
    if re.search(r"(?:ss|sh|ch|x|z|o)es$", word):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def mine_lexicon(texts, min_count=3, noun_ratio=0.3):
    """Verb bases = words right after a subject-like word, seen >= min_count times.
    'nounish' = forms that follow a determiner in >= noun_ratio of their uses."""
    seen = Counter()
    for text in texts:
        for m in list(SUBJECT_HINT.finditer(text)) + list(OBJECT_HINT.finditer(text)):
            word = m.group(1).lower()
            if word not in NOT_VERB and not word.endswith("ly"):
                seen[word] += 1
    # Present-tense verbs only: "-ed" / "-ing" words after a subject are participles
    # ("a person seated"), and short leftovers like "as" -> "a" are not verbs.
    bases = sorted({b for b in (to_base(w) for w, c in seen.items()
                                if c >= min_count and not w.endswith(("ed", "ing"))
                                and w not in PARTICIPLES)
                    if len(b) >= 3 and b not in NOT_VERB})
    forms = set(bases) | {third_person(b) for b in bases}
    total, after_det = Counter(), Counter()
    for text in texts:
        tokens = [t.lower() for t in WORD.findall(text)]
        for prev, tok in zip([""] + tokens, tokens):
            if tok in forms:
                total[tok] += 1
                after_det[tok] += prev in DETERMINERS
    nounish = sorted(f for f in forms if total[f] and after_det[f] / total[f] >= noun_ratio)
    return {"bases": bases, "nounish": nounish}


@dataclass
class Piece:
    text: str
    kind: str = "event"   # "event" | "state"
    t: int | None = None  # time rank; equal ranks are simultaneous; None for states
    rule: str = "start"   # rule that cut this piece off


class Segmenter:
    def __init__(self, lexicon, min_words=MIN_WORDS, max_pieces=MAX_PIECES):
        self.bases = set(lexicon["bases"])
        self.third = {third_person(b): b for b in self.bases}
        self.nounish = set(lexicon["nounish"])
        self.min_words = min_words
        self.max_pieces = max_pieces

    @classmethod
    def from_json(cls, path, **kwargs):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    # ---- word classes -------------------------------------------------------------
    def finite(self, word):
        word = word.lower()
        return word in self.bases or word in self.third

    def gerund_base(self, word):
        word = word.lower()
        if not word.endswith("ing") or len(word) < 5:
            return None
        stem = word[:-3]
        candidates = [stem, stem + "e"]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
        if stem.endswith("y"):
            candidates.append(stem[:-1] + "ie")
        return next((c for c in candidates if c in self.bases), None)

    def has_verb(self, text):
        return any(self.finite(w) or self.gerund_base(w) or w.lower() in AUX
                   for w in WORD.findall(text))

    def is_state(self, text):
        """Static background ("the cup stays on the table", "the other is empty"). Judged on the
        verb after the subject, so an opener like "Standing by the sofa, ..." does not count."""
        _, _, aux, span = self.find_subject(text)
        if span:
            if aux:            # "is lifting" / "is placed": something happens
                return False
            verb = text[span[0]:span[1]].lower()
            rest = WORD.findall(text[span[1]:])
            nxt = rest[0].lower() if rest else ""
            return (verb in STATE_VERBS and nxt not in DETERMINERS
                    and nxt not in {"it", "them", "up", "down"})
        text = re.sub(r"^[A-Za-z]+ing\b[^,]*,\s*", "", text)   # drop a participle opener
        words = [w.lower() for w in WORD.findall(text)]
        for i, w in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else ""
            if w in STATE_VERBS:
                return nxt not in DETERMINERS and nxt not in {"it", "them", "up", "down"}
            if w in AUX:   # "the other is empty", but not "is lifting" / "is placed"
                return not (nxt.endswith(("ing", "ed")) or nxt in PARTICIPLES)
            if self.finite(w):
                return False
        return False

    def find_subject(self, text):
        """-> (subject, plural, aux, verb_span) from the first verb that has a short noun phrase
        right before it (back to the previous comma)."""
        words = list(WORD.finditer(text))
        for i, m in enumerate(words[1:], start=1):
            w = m.group(0).lower()
            j = i + 1   # skip adverbs between an auxiliary and its participle ("is then tilted")
            while j < len(words) and words[j].group(0).lower() in ADVERBS:
                j += 1
            nxt = words[j].group(0).lower() if j < len(words) else ""
            aux = None
            if w in AUX and (nxt.endswith(("ing", "ed")) or nxt in PARTICIPLES):
                aux = w   # progressive "is lifting" or passive "is laid"
            elif not self.finite(w):
                continue
            start = text.rfind(",", 0, m.start()) + 1
            tokens = text[start:m.start()].split()
            while tokens and tokens[0].lower() in ADVERBS | {"and", "but"}:
                tokens.pop(0)
            while tokens and (tokens[-1].lower() in ADVERBS or tokens[-1].lower().endswith("ly")):
                tokens.pop()
            if not 1 <= len(tokens) <= 6:
                continue
            first, last = tokens[0].lower(), tokens[-1].lower()
            if (first not in SUBJECT_START or last in DETERMINERS or last in FOLLOW
                    or last in {"and", "or", "but"}):
                continue
            plural = (last in {"they", "we"} or
                      (last.endswith("s") and not last.endswith("ss") and first not in {"a", "an", "one"}))
            span = (m.start(), words[j].end() if aux else m.end())
            return " ".join(tokens), plural, aux, span
        return None, None, None, None

    def own_subject(self, text):
        """find_subject, but only when the subject starts the text (a clause of its own)."""
        found = self.find_subject(text)
        if found[0] and text.strip().lower().startswith(found[0].lower()):
            return found
        return None

    def agrees(self, word, plural):
        word = word.lower()
        if plural is None:
            return self.finite(word)
        return word in self.bases if plural else word in self.third

    # ---- cutting --------------------------------------------------------------------
    def connector_split(self, text, rule):
        """-> [(segment, rule)] cut at explicit connectors."""
        out, last, last_rule = [], 0, rule
        for m in CONNECTOR.finditer(text):
            name = m.lastgroup
            if name == "participle":
                nxt = WORD.findall(text[m.end():])[:2]
                if len(nxt) < 2 or not self.gerund_base(nxt[0]) or nxt[1].lower() not in FOLLOW:
                    continue
            left, right = text[last:m.start()], text[m.end():]
            if len(left.split()) < 2 or len(right.split()) < 1:
                continue
            out.append((left, last_rule))
            last, last_rule = m.end(), name
        out.append((text[last:], last_rule))
        return out

    def clause_split(self, text, rule):
        """Cut before ", and <subject> <verb>" so each clause keeps its own subject."""
        out, last, last_rule = [], 0, rule
        for m in CLAUSE.finditer(text):
            found = self.own_subject(text[m.end():])
            if not found or found[3][0] > 40:
                continue
            left = text[last:m.start()]
            if len(left.split()) < self.min_words or not self.has_verb(left):
                continue
            out.append((left, last_rule))
            last, last_rule = m.end(), "clause"
        out.append((text[last:], last_rule))
        return out

    def coord_split(self, text, rule, plural, aux):
        out, last, last_rule = [], 0, rule
        for m in COORD.finditer(text):
            word = m.group(1)
            if not (self.agrees(word, plural) or (aux and self.gerund_base(word))):
                continue
            after = WORD.findall(text[m.end() + len(word):])
            nxt = after[0].lower() if after else None
            if nxt in {"and", "or", "but"} or (nxt and self.finite(nxt)):
                continue
            if word.lower() in self.nounish and nxt not in FOLLOW:
                continue
            left = text[last:m.start()]
            if len(left.split()) < self.min_words or not self.has_verb(left):
                continue
            out.append((left, last_rule))
            last, last_rule = m.end(), "coord"
        out.append((text[last:], last_rule))
        return out

    def expand_list(self, text, verb_span):
        mark = LIST_MARK.search(text)
        if not mark:
            return None
        body = (text[:mark.start()] + text[mark.end():]).strip(" ,")
        if verb_span and verb_span[1] <= mark.start():
            vend = verb_span[1]
        else:
            verbs = [m for m in WORD.finditer(body) if self.finite(m.group(0))]
            if not verbs:
                return None
            vend = verbs[0].end()
        rest = WORD.match(body[vend:].lstrip())
        if rest and rest.group(0).lower() in PARTICLES:
            vend = body.index(rest.group(0), vend) + len(rest.group(0))
        head, obj = body[:vend].strip(), body[vend:].strip()
        items = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", obj) if x.strip()]
        if len(items) < 2 or any(not 1 <= len(x.split()) <= 6 or x.split()[0].lower() not in DETERMINERS
                                 for x in items):
            return None
        return [f"{head} {x}" for x in items]

    # ---- rebuilding pieces ------------------------------------------------------------
    def restore(self, text, rule, subject, plural, aux, prev_event):
        t = MARKER.sub("", text.strip(" ,.;"))
        t = re.sub(r"^(?:and|but)\s+", "", t, flags=re.I)
        words = t.split()
        if not words:
            return ""
        first = words[0].lower()
        if rule == "followed" and first in DETERMINERS and prev_event:
            verbs = [m for m in WORD.finditer(prev_event) if self.finite(m.group(0))]
            if verbs:
                v = verbs[0]
                vend = v.end()
                nxt = WORD.match(prev_event[vend:].lstrip())
                if nxt and nxt.group(0).lower() in PARTICLES:
                    vend = prev_event.index(nxt.group(0), vend) + len(nxt.group(0))
                head, tail = prev_event[:vend].strip(), prev_event[vend:].strip()
                if tail:   # "display X, followed by Y" -> "display Y"
                    t = f"{head} {t}"
                else:      # "X returns, followed by Y" -> "Y returns"
                    t = f"{t} {prev_event[v.start():vend].strip()}"
        elif subject and not t.lower().startswith(subject.lower()):
            subj = subject[:1].lower() + subject[1:]
            opening = rule in {"start", "lead_main", "clause"}
            participle = first.endswith(("ing", "ed")) or first in PARTICIPLES
            if self.finite(first):
                t = f"{subj} {t}"
            elif opening:
                pass   # "Seated beside the cup, a person ..." keeps its own opening
            elif aux and participle:
                t = f"{subj} {aux} {t}"
            elif self.gerund_base(first):
                base = self.gerund_base(first)
                t = " ".join([subj, base if plural else third_person(base)] + words[1:])
        return t[:1].upper() + t[1:]

    def sentence_pieces(self, sentence, fallback=None):
        """-> ([(text, kind, relation, rule)], subject used), relation in seq | same | before_prev.
        `fallback` is the previous sentence's subject, for sentences that open with a verb."""
        s = MARKER.sub("", sentence.strip().rstrip(".!? "))
        lead = LEAD.match(s)
        if lead:
            main = lead["main"]
            conj = lead["conj"].lower()
            units = ([(main, "start"), (lead["sub"], "lead_before")] if conj == "before"
                     else [(lead["sub"], "lead"), (main, "lead_main")])
        else:
            main, units = s, [(s, "start")]
        sentence_subject = self.find_subject(main)
        if not sentence_subject[0] and fallback:
            sentence_subject = fallback

        segments = []   # (text, rule, (subject, plural, aux, span))
        for unit, rule in units:
            for seg, seg_rule in self.connector_split(unit, rule):
                if seg_rule == "while":
                    segments.append((seg, seg_rule, sentence_subject))
                    continue
                for clause, clause_rule in self.clause_split(seg, seg_rule):
                    subj = self.own_subject(clause) or sentence_subject
                    for part, part_rule in self.coord_split(clause, clause_rule, subj[1], subj[2]):
                        found = self.find_subject(part)
                        items = self.expand_list(part, found[3] if found[0] else None)
                        if items:
                            segments.append((items[0], part_rule, subj))
                            segments.extend((x, "list", subj) for x in items[1:])
                        else:
                            segments.append((part, part_rule, subj))

        out, prev_event = [], None
        for seg, rule, (subject, plural, aux, _) in segments:
            if rule == "while" and self.is_state(seg):
                out.append((seg.strip(" ,.;"), "state", None, rule))
                continue
            text = self.restore(seg, rule, subject, plural, aux, prev_event)
            if not text:
                continue
            if len(text.split()) < 2 and out:   # stray word: glue back to the previous piece
                prev = out[-1]
                out[-1] = (f"{prev[0]} {text}", prev[1], prev[2], prev[3])
                continue
            kind = "state" if self.is_state(text) else "event"
            relation = ("same" if rule in {"while", "participle"} else
                        "before_prev" if rule == "after" else "seq")
            out.append((text, kind, relation, rule))
            if kind == "event":
                prev_event = text
        return out, sentence_subject

    def split(self, text):
        groups, states, subject = [], [], None
        for sentence in SENTENCE.split(text.strip()):
            pieces, used = self.sentence_pieces(sentence, subject)
            if used and used[0]:
                subject = used
            for piece_text, kind, relation, rule in pieces:
                piece = Piece(piece_text, kind, None, rule)
                if kind == "state":
                    states.append(piece)
                elif relation == "same" and groups:
                    groups[-1].append(piece)
                elif relation == "before_prev" and groups:
                    groups.insert(len(groups) - 1, [piece])
                else:
                    groups.append([piece])
        events = []
        for t, group in enumerate(groups):
            for piece in group:
                piece.t = t
                events.append(piece)
        pieces = events + states
        if len(pieces) > self.max_pieces:
            tail = pieces[self.max_pieces - 1:]
            pieces = pieces[:self.max_pieces - 1] + [
                Piece(" ".join(p.text for p in tail), tail[0].kind, tail[0].t, "merged")]
        return pieces or [Piece(text.strip(), "event", 0)]

    def texts(self, text):
        return [p.text for p in self.split(text)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mine the verb lexicon used by Segmenter.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, help="splits.json: mine from its train IDs only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folders = [p for p in args.data_dir.glob("episode_*") if p.is_dir()]
    if args.splits:
        keep = set(json.loads(args.splits.read_text(encoding="utf-8"))["train"])
        folders = [p for p in folders if p.name in keep]
    texts = [(p / "description.txt").read_text(encoding="utf-8").strip() for p in sorted(folders)]
    lexicon = mine_lexicon(texts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lexicon, indent=1), encoding="utf-8")
    print(f"{len(texts)} descriptions -> {len(lexicon['bases'])} verb bases, "
          f"{len(lexicon['nounish'])} noun-like forms -> {args.out}")


if __name__ == "__main__":
    main()
