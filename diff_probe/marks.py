"""Mark, per choice, the words that make it differ from the other choices of its episode.

Rule-based (difflib on words), runs inside a submission. Every choice is aligned to the
reference (the most central choice). A replaced or inserted word is marked in the choice
that has it; for an insertion or deletion the word just before the gap is marked as the
anchor, so a choice that lacks an added clause still has a marked word ("shirt." vs
"shirt and then sits down on it."). The reference gets the union of its marked words.

    spans = mark_spans(texts)   # per choice: list of (char_start, char_end) of marked words
"""

import difflib
import re

from centrality import score_span

_WORD = re.compile(r"\S+")
_KEY = re.compile(r"[^a-z0-9']+")


def _key(w):
    return _KEY.sub("", w.lower()) or w


def _words(text):
    return [(m.start(), m.end(), _key(m.group())) for m in _WORD.finditer(text)]


def _mark_pair(ref, other):
    """Word indices to mark in ref and in other for one alignment."""
    mr, mo = set(), set()
    ops = difflib.SequenceMatcher(None, [w[2] for w in ref], [w[2] for w in other],
                                  autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        mr.update(range(i1, i2))
        mo.update(range(j1, j2))
        if i1 == i2:  # other has extra words: anchor on the ref word before the gap
            mr.add(max(i1 - 1, 0))
        if j1 == j2:  # other lacks words: anchor on its word before the gap
            mo.add(max(j1 - 1, 0))
    return {i for i in mr if i < len(ref)}, {j for j in mo if j < len(other)}


def mark_spans(texts):
    words = [_words(t) for t in texts]
    scores = score_span(texts)
    ref = max(range(len(texts)), key=lambda i: scores[i])
    marks = [set() for _ in texts]
    for i in range(len(texts)):
        if i == ref:
            continue
        mr, mo = _mark_pair(words[ref], words[i])
        marks[ref] |= mr
        marks[i] |= mo
    return [[(words[i][k][0], words[i][k][1]) for k in sorted(marks[i])]
            for i in range(len(texts))]
