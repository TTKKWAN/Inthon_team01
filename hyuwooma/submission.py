"""Model-free API v2 probe: rank text by mean UTF-8 byte edit similarity.

For candidates i != j, d(i,j) is exact byte Levenshtein distance divided by
max(len(utf8_i), len(utf8_j), 1). Candidate i scores 1 - mean_j d(i,j).
Text is used verbatim: no Unicode normalization, case folding or whitespace
trimming. These are heuristic preference scores, not calibrated probabilities.

Images, artifact_dir and device are unused. No state or data from other episodes
is retained. The heuristic makes no assumption about the hidden Test generator.
"""

from math import fsum


def levenshtein_bytes(left: bytes, right: bytes) -> int:
    """Exact unit-cost byte insertions, deletions and substitutions.

    A bit-vector dynamic program stores column differences in arbitrary-width
    Python integers. Common prefix/suffix removal preserves the exact distance;
    it does not affect the original byte lengths used for normalization.
    """
    if not isinstance(left, bytes) or not isinstance(right, bytes):
        raise TypeError("levenshtein_bytes expects two bytes objects")
    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    start = 0
    left_end, right_end = len(left), len(right)
    while start < left_end and left[start] == right[start]:
        start += 1
    while left_end > start and left[left_end - 1] == right[right_end - 1]:
        left_end -= 1
        right_end -= 1
    left, right = left[start:left_end], right[start:right_end]
    if not left:
        return len(right)

    width = len(left)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    matches = [0] * 256
    for index, byte in enumerate(left):
        matches[byte] |= 1 << index
    positive, negative = mask, 0
    distance = width
    for byte in right:
        equal = matches[byte]
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        distance += bool(positive_horizontal & high_bit) - bool(negative_horizontal & high_bit)
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = (positive_horizontal & vertical) & mask
    return distance


def normalized_byte_distance(left: bytes, right: bytes) -> float:
    """Normalize by original byte lengths, including common prefix/suffix bytes."""
    return levenshtein_bytes(left, right) / max(len(left), len(right), 1)


class Submission:
    __slots__ = ()

    def __init__(self, artifact_dir: str, device: str):
        pass

    def score(self, images, choices) -> list[float]:
        """One [0, 1] score per choice; empty input -> [], singleton -> [1.0]."""
        texts = [choice["text"].encode("utf-8") for choice in choices]
        count = len(texts)
        if count < 2:
            return [1.0] * count
        distances = [[] for _ in texts]
        for left_index in range(count):
            for right_index in range(left_index + 1, count):
                distance = normalized_byte_distance(texts[left_index], texts[right_index])
                distances[left_index].append(distance)
                distances[right_index].append(distance)
        # fsum avoids input-order-dependent accumulation error for near ties.
        return [1.0 - fsum(row) / (count - 1) for row in distances]

    def predict(self, images, choices) -> list[str]:
        """Return every supplied ID in descending score order, preserving ties."""
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
