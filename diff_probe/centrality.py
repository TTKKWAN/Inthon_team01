"""Model-free centrality scores over one episode's choices.

Every scorer sees only the texts of a single episode, so no test statistics are shared.
Higher score = closer to the other choices = more likely the unedited description.

  byte  : byte Levenshtein / longer length, mean over others (= hyuwooma 0.52 probe)
  word  : word Levenshtein / longer length, mean over others
  span  : number of edited word spans between two choices, mean over others.
          A block that moved (same words deleted in one place, inserted in another)
          counts as one edit.
"""

import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hyuwooma"))
from submission import normalized_byte_distance  # noqa: E402

TOKEN = re.compile(r"[a-z0-9']+")


def words(text):
    return TOKEN.findall(text.lower())


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def edit_spans(a, b):
    """Non-equal opcodes between token lists a and b, as (tag, a_slice, b_slice)."""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return [(tag, a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag != "equal"]


def span_count(a, b):
    spans = edit_spans(a, b)
    deleted = [tuple(x) for tag, x, _ in spans if tag in ("delete", "replace") and x]
    inserted = [tuple(y) for tag, _, y in spans if tag in ("insert", "replace") and y]
    moves = 0
    for block in deleted:
        if block in inserted:
            inserted.remove(block)
            moves += 1
    return len(spans) - moves


def _mean_scores(items, dist):
    n = len(items)
    rows = [[] for _ in items]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(items[i], items[j])
            rows[i].append(d)
            rows[j].append(d)
    return [-sum(r) / max(len(r), 1) for r in rows]


def score_byte(texts):
    return _mean_scores([t.encode("utf-8") for t in texts], normalized_byte_distance)


def score_word(texts):
    toks = [words(t) for t in texts]
    return _mean_scores(toks, lambda a, b: levenshtein(a, b) / max(len(a), len(b), 1))


def score_span(texts):
    return _mean_scores([words(t) for t in texts], span_count)


SCORERS = {"byte": score_byte, "word": score_word, "span": score_span}
