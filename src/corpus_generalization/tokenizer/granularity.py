"""Tokens per word as a function of the merge budget, and learning a tokenizer to a granularity.

A merge budget means nothing across corpora: the same count leaves a small corpus at whole words
and a large one in pieces. Tokens per word is what the diagnosis depends on, so the budget is read
off it (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`, step 1).
"""

from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from corpus_generalization.tokenizer.bpe import (
    DEFAULT_MERGE_BUDGET,
    ByteLevelBPETokenizer,
    Merge,
    MergeProgress,
    _encode_bytes,
    _learn_merges,
    _training_pieces,
)

# Where the 22 labeled cuts held in five domains (TinyStories, FineWeb, Wikipedia, PubMed, ECtHR)
# on 14.4 MB pools, three draws each: 14 of 15 pools once a dropped word no granularity of the pool
# reads absent is left out (12 of 15 without), FineWeb's third failing 1.35-1.90 on a word with 68%
# of its uses in three documents. 1.30 failed in FineWeb's second, 1.80 in two of legal's three. At
# 57.6 MB the range widens to 1.35-2.00 (`experiments/granularity_robustness`).
GRANULARITY_RANGE = (1.35, 1.70)
# Any granularity in the range gives the same verdicts. 1.50 is inside it with margin on both
# sides, and it is where steps 1 and 2 checked the test's level and the class settings in depth.
GRANULARITY_TARGET = 1.50


def tokens_per_word_by_budget(
    documents: Sequence[str], merges: Sequence[Merge], *, recognize_markers: bool = True
) -> list[float]:
    """Tokens per word of `documents` at every budget from 0 to `len(merges)`, in one pass.

    Encoding applies the lowest-ranked merge present, and a merge's new token only enters pairs
    learned after it, so the ranks applied never decrease: encoding at budget k is the full
    encoding stopped before the first rank >= k. Each merge's saving is counted once per piece
    type, weighted by its frequency. Words are whitespace-separated, as `diagnose` counts them."""
    words = sum(len(document.split()) for document in documents)
    if not words:
        raise ValueError("Documents must contain words.")
    types = Counter(_training_pieces(documents, recognize_markers, ()))
    ranks = {pair: rank for rank, (pair, _) in enumerate(merges)}
    ids = {rank: new for rank, (_, new) in enumerate(merges)}
    saved = [0] * (len(merges) + 1)
    total = 0
    for piece, frequency in types.items():
        tokens = _encode_bytes(piece)
        total += frequency * len(tokens)
        while len(tokens) > 1:
            found = [
                (rank, pair) for pair in pairwise(tokens) if (rank := ranks.get(pair)) is not None
            ]
            if not found:
                break
            rank, pair = min(found)
            merged: list[int] = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                    merged.append(ids[rank])
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            saved[rank + 1] += frequency * (len(tokens) - len(merged))
            tokens = merged
    curve, remaining = [], total
    for saving in saved:
        remaining -= saving
        curve.append(remaining / words)
    return curve


def learn_to_granularity(
    documents: Sequence[str],
    target: float = GRANULARITY_TARGET,
    *,
    progress: MergeProgress | None = None,
) -> tuple[ByteLevelBPETokenizer, float]:
    """The fewest merges that bring `documents` to `target` tokens per word or below, and the
    tokens per word they reach. A corpus whose merges run out first keeps all of them.

    Learning resumes in doubling rounds, so at most twice the merges needed are learned; a resumed
    run learns exactly what one run to the larger budget would (`_learn_merges`)."""
    merges: tuple[Merge, ...] = ()
    budget = DEFAULT_MERGE_BUDGET
    while True:
        merges = _learn_merges(
            documents, budget, progress=progress, recognize_markers=True, initial_merges=merges
        )
        curve = tokens_per_word_by_budget(documents, merges)
        reached = next((k for k, value in enumerate(curve) if value <= target), None)
        if reached is not None or len(merges) < budget:
            count = len(merges) if reached is None else reached
            return ByteLevelBPETokenizer.from_merges(merges[:count]), curve[count]
        budget *= 2
