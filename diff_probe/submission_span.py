"""Model-free API v2: rank choices by word-span centrality, byte centrality breaks ties.

span: number of edited word places between two choices (a moved block counts once),
averaged over the other choices; fewer = closer to the rest = more likely the unedited
description. Ties (common, the counts are small integers) are broken by hyuwooma's
normalized byte Levenshtein. Images are unused; only this episode's texts are read.
"""

from centrality import _mean_scores, score_span
from hyuwooma_bytes import normalized_byte_distance


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        pass

    def score(self, images, choices):
        texts = [c["text"] for c in choices]
        if len(texts) < 2:
            return [1.0] * len(texts)
        span = score_span(texts)
        byte = _mean_scores([t.encode("utf-8") for t in texts], normalized_byte_distance)
        return [s + 1e-3 * b for s, b in zip(span, byte)]  # byte distance is in [0, 1]

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda i: -scores[i])
        return [choices[i]["id"] for i in order]
