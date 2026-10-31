"""Find where one episode's choices differ, and cut a short window around each place.

Rule-based only (difflib on whitespace tokens), so it can run inside a submission.

    out = find_spots(texts, context=4)
    out["spots"][k]["variants"]        -> distinct window texts at place k
    out["spots"][k]["choice_variant"]  -> for each choice, its index into variants

Steps
  1. reference = the choice with the fewest edited spans to the others (centrality)
  2. every choice is aligned to the reference; each non-equal opcode is an edit
     interval [start, end) in reference tokens. An insertion at k belongs to the
     words before it ("... his shirt" + "and then sits down on it").
  3. a choice whose edits contain a moved block (order swap) gets one interval that
     covers all of its edits, so the swap is judged as a whole
  4. intervals are widened by `context` words on each side without crossing a sentence
     end ("clause": up to , ; and sentence ends; "full": the whole text); overlapping
     windows are merged
  5. each choice's window text is cut from that choice's own tokens via the alignment
"""

import difflib
import re

from centrality import score_span

_KEY = re.compile(r"[^a-z0-9']+")
_SENT_END = (".", "!", "?")
_CLAUSE_END = (",", ";", ":") + _SENT_END


def _key(tok):
    return _KEY.sub("", tok.lower()) or tok


def _opcodes(ref, other):
    a, b = [_key(t) for t in ref], [_key(t) for t in other]
    return difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def _edits(ref, other, ops):
    edits = [(i1, i2) for tag, i1, i2, _, _ in ops if tag != "equal"]
    deleted = [tuple(_key(t) for t in ref[i1:i2]) for tag, i1, i2, _, _ in ops
               if tag in ("delete", "replace")]
    inserted = [tuple(_key(t) for t in other[j1:j2]) for tag, _, _, j1, j2 in ops
                if tag in ("insert", "replace")]
    if len(edits) > 1 and any(d and d in inserted for d in deleted):
        edits = [(min(e[0] for e in edits), max(e[1] for e in edits))]
    return edits


def _widen(ref, start, end, context):
    if start == end:  # insertion: anchor on the word before it
        start, end = (start - 1, start) if start > 0 else (0, 1)
    if context == "full":
        return 0, len(ref)
    stops = _CLAUSE_END if context == "clause" else _SENT_END
    budget = 10 ** 6 if context == "clause" else int(context)
    s = start
    while s > 0 and start - s < budget and not ref[s - 1].endswith(stops):
        s -= 1
    e = end
    while e < len(ref) and e - end < budget and not ref[e - 1].endswith(stops):
        e += 1
    return s, e


def _boundary_map(ops):
    """ref boundary position -> choice position, taking insertions at a boundary along
    with the words before it (so they land in the window that ends there)."""
    m = {}
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1 + 1):
                m.setdefault(i1 + k, j1 + k)
        else:
            m.setdefault(i1, j1)
            m[i2] = j2
    for tag, i1, i2, j1, j2 in ops:
        if tag == "insert":
            m[i1] = j2
    return m


def _left_map(ops):
    """ref boundary position -> choice position BEFORE any insertion at that position."""
    m = {}
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1 + 1):
                m.setdefault(i1 + k, j1 + k)
        else:
            m.setdefault(i1, j1)
            m.setdefault(i2, j2)
    for tag, i1, _, j1, _ in ops:
        if tag == "insert":
            m[i1] = j1
    return m


def find_atoms(texts, context=4):
    """One spot per edited place (no merging of nearby edits). Context words come from
    the reference, so each question isolates a single difference."""
    scores = score_span(texts)
    ref_idx = max(range(len(texts)), key=lambda i: scores[i])
    ref = texts[ref_idx].split()
    toks = [t.split() for t in texts]
    aligned = [_opcodes(ref, t) for t in toks]

    # An insertion takes the word before it along ("shirt." vs "shirt and then sits ...").
    intervals = sorted(((s - 1, s) if s == e and s > 0 else (s, max(e, 1)) if s == e else (s, e))
                       for t, ops in zip(toks, aligned) for s, e in _edits(ref, t, ops))
    atoms = []
    for s, e in intervals:
        if atoms and (s < atoms[-1][1] or s == atoms[-1][0]):
            atoms[-1] = (atoms[-1][0], max(atoms[-1][1], e))
        else:
            atoms.append((s, e))

    lefts = [_left_map(ops) for ops in aligned]
    rights = [_boundary_map(ops) for ops in aligned]
    spots = []
    for s, e in atoms:
        ws, we = _widen(ref, s, e, context)
        readings, choice_variant = [], []
        for t, lm, rm in zip(toks, lefts, rights):
            middle = t[lm[s]:rm[e]]
            text = " ".join(ref[ws:s] + middle + ref[e:we])
            if text not in readings:
                readings.append(text)
            choice_variant.append(readings.index(text))
        if len(readings) > 1:
            spots.append({"span": [s, e], "variants": readings, "choice_variant": choice_variant})
    return {"reference": ref_idx, "centrality": scores, "spots": spots}


def find_joint(texts, context=4):
    """One question per episode: each choice is shown only by its windows at the places
    where the choices differ, joined with ' … ' (merged windows, in text order)."""
    out = find_spots(texts, context, mode="merged")
    spots = out["spots"]
    if not spots:
        return out
    rendered = [" … ".join(s["variants"][s["choice_variant"][i]] for s in spots)
                for i in range(len(texts))]
    readings, choice_variant = [], []
    for r in rendered:
        if r not in readings:
            readings.append(r)
        choice_variant.append(readings.index(r))
    out["spots"] = [{"span": [spots[0]["span"][0], spots[-1]["span"][1]],
                     "variants": readings, "choice_variant": choice_variant}]
    return out


def find_spots(texts, context=4, mode="merged"):
    if mode == "atoms":
        return find_atoms(texts, context)
    if mode == "joint":
        return find_joint(texts, context)
    scores = score_span(texts)
    ref_idx = max(range(len(texts)), key=lambda i: scores[i])
    ref = texts[ref_idx].split()
    toks = [t.split() for t in texts]
    aligned = [_opcodes(ref, t) for t in toks]

    windows = sorted(_widen(ref, s, e, context)
                     for t, ops in zip(toks, aligned) for s, e in _edits(ref, t, ops))
    merged = []
    for s, e in windows:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    maps = [_boundary_map(ops) for ops in aligned]
    spots = []
    for s, e in merged:
        readings, choice_variant = [], []
        for t, m in zip(toks, maps):
            left = 0 if s == 0 else m[s]
            text = " ".join(t[left:m[e]])
            if text not in readings:
                readings.append(text)
            choice_variant.append(readings.index(text))
        if len(readings) > 1:
            spots.append({"span": [s, e], "variants": readings, "choice_variant": choice_variant})
    return {"reference": ref_idx, "centrality": scores, "spots": spots}
