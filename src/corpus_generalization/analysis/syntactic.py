"""Syntactic sufficiency: whether each syntactic pattern of a knowledge domain is seen with enough
different fillers that a model learns the pattern rather than a word. See
`docs/2026-09-28-syntactic-sufficiency-design.md`.

A pattern is a head followed by a token of a grammatical class, so the classes come first. Each
token is described by the distributions of its left and right neighbors, and tokens are clustered
hierarchically on them (Ward). Only a coarse cut is used: finer cuts start grouping tokens by what
they mean (things you pour against things you chop), which would bring semantics into a measure
of syntax. Without a fixed number of classes, the cut is where the dendrogram jumps most — on the
recipe grammar the merge that would join two real categories costs 3.5 times the last merge
inside one, and the cut there recovers the grammar's categories exactly.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from corpus_generalization.analysis.markov_chain import CoverageCurve

# Neighbor features are limited to the most frequent tokens, the rest pooled in one column, and at
# most `MAX_CLASSIFIED_TOKENS` tokens are clustered: Ward keeps a dense distance matrix, quadratic
# in the tokens clustered.
CONTEXT_TOKENS = 1000
MAX_CLASSIFIED_TOKENS = 4000
# Merges below this fraction of the root merge's height count as noise when looking for the cut.
NOISE_FLOOR = 0.05
WORD_FINAL_SHARE = 0.9


def word_like_tokens(
    sequences: Sequence[Sequence[int]],
    decode: Callable[[int], str],
    boundary_ids: Collection[int] = (),
) -> frozenset[int]:
    """Keep space-prefixed alphabetic tokens that usually end the word they start.

    The occurrence check separates whole words from word-initial BPE pieces whose
    decoded text alone looks like a word.
    """
    texts = {token: decode(token) for sequence in sequences for token in sequence}
    candidates = {token for token, text in texts.items() if re.fullmatch(r" [A-Za-z]+", text)}
    boundaries = set(boundary_ids)
    occurrences: Counter[int] = Counter()
    endings: Counter[int] = Counter()
    for sequence in sequences:
        for position, token in enumerate(sequence):
            if token not in candidates:
                continue
            occurrences[token] += 1
            if position + 1 == len(sequence) or sequence[position + 1] in boundaries:
                endings[token] += 1
            else:
                following = texts[sequence[position + 1]]
                if following and (
                    following[0].isspace() or not (following[0].isalpha() or following[0].isdigit())
                ):
                    endings[token] += 1
    return frozenset(
        token for token, count in occurrences.items() if endings[token] / count >= WORD_FINAL_SHARE
    )


@dataclass(frozen=True)
class TokenClasses:
    """`classes` maps each classified token to its class (numbered by smallest member token).
    `merge_heights[k - 1]` is the Ward height of the merge that leaves `k` classes."""

    classes: dict[int, int]
    n_classes: int
    merge_heights: tuple[float, ...]


def _neighbor_features(
    sequences: Sequence[Sequence[int]], tokens: Sequence[int], context: Sequence[int]
) -> np.ndarray:
    """Per token, the square roots of its left and its right neighbor distributions over the
    `context` tokens, a pooled column for every other token and one for the sequence boundary —
    so Euclidean distance between rows is the Hellinger distance on each side."""
    row = {token: index for index, token in enumerate(tokens)}
    column = {token: index for index, token in enumerate(context)}
    other, boundary = len(context), len(context) + 1
    width = len(context) + 2
    counts = np.zeros((len(tokens), 2 * width))
    for sequence in sequences:
        for position, token in enumerate(sequence):
            if token not in row:
                continue
            left = sequence[position - 1] if position > 0 else None
            right = sequence[position + 1] if position + 1 < len(sequence) else None
            for side, neighbor in ((0, left), (width, right)):
                index = boundary if neighbor is None else column.get(neighbor, other)
                counts[row[token], side + index] += 1
    for half in (slice(0, width), slice(width, 2 * width)):
        counts[:, half] /= counts[:, half].sum(axis=1, keepdims=True)
    return np.sqrt(counts)


def induce_classes(
    sequences: Sequence[Sequence[int]],
    n_classes: int | None = None,
    min_documents: int = 2,
    max_classes: int = 64,
    *,
    min_classes: int = 8,
    candidates: Collection[int] | None = None,
) -> TokenClasses:
    """Cluster the tokens found in at least `min_documents` sequences into coarse classes; at the
    largest jump in merge height from `max(2, min_classes)` to `max_classes` classes when
    `n_classes` is `None`. If no candidate fits, use the largest valid cut, or one class for a
    single token. `min_classes` has no effect on a fixed `n_classes`. When `candidates` is
    given, only its tokens are clustered; context columns still use the whole corpus."""
    documents: Counter[int] = Counter()
    frequency: Counter[int] = Counter()
    for sequence in sequences:
        documents.update(set(sequence))
        frequency.update(sequence)
    allowed = set(candidates) if candidates is not None else None
    eligible = [
        token
        for token in frequency
        if documents[token] >= min_documents and (allowed is None or token in allowed)
    ]
    tokens = sorted(
        sorted(eligible, key=lambda token: (-frequency[token], token))[:MAX_CLASSIFIED_TOKENS]
    )
    if n_classes is None and len(tokens) == 1:
        return TokenClasses(classes={tokens[0]: 0}, n_classes=1, merge_heights=())
    context = [token for token, _ in frequency.most_common(CONTEXT_TOKENS)]
    features = _neighbor_features(sequences, tokens, context)

    tree = AgglomerativeClustering(n_clusters=None, distance_threshold=0.0, linkage="ward")
    tree.fit(features)
    # distances_ ascend, merge i leaving len(tokens) - 1 - i clusters: reversed, index k - 1
    # leaves k.
    heights = tuple(float(height) for height in tree.distances_[::-1])

    if n_classes is None:
        # Near the leaves, tokens with almost identical neighbors merge at heights close to zero,
        # and any ratio over them is noise: on the most restricted recipe variant it picked 64
        # classes, nearly all singletons. The floor keeps the jump a jump at the tree's own scale.
        floor = NOISE_FLOOR * heights[0]
        # On 500 BPE-framed recipe documents, k=2 mixed 95 tokens across grammar categories;
        # a floor of eight recovered the coarse categories.
        cuts = range(max(2, min_classes), min(max_classes, len(heights)) + 1)
        n_classes = max(
            cuts,
            key=lambda k: heights[k - 2] / max(heights[k - 1], floor),
            default=min(max_classes, len(heights)),
        )

    labels = AgglomerativeClustering(n_clusters=n_classes, linkage="ward").fit_predict(features)
    members: dict[int, list[int]] = defaultdict(list)
    for token, label in zip(tokens, labels, strict=True):
        members[int(label)].append(token)
    numbered = sorted(members.values(), key=min)
    return TokenClasses(
        classes={token: number for number, group in enumerate(numbered) for token in group},
        n_classes=n_classes,
        merge_heights=heights,
    )


@dataclass(frozen=True)
class Pattern:
    """A head followed by a token of `filler_class`. Occurrences are counted in distinct contexts
    (the token before the head, the token after the filler), so exact copies add nothing."""

    head: tuple[int, ...]
    filler_class: int
    trunk: int
    fillers: int
    class_size: int
    singletons: int
    documents: int
    expected_fillers: float = 0.0

    @property
    def diversity(self) -> float:
        """Fillers relative to free draws from the class at this head's traffic.

        Coverage penalizes heads with fewer occurrences; the chance expectation makes
        heads of different frequencies comparable, accounting for class member frequencies.
        """
        return self.fillers / self.expected_fillers if self.expected_fillers else 0.0

    @property
    def coverage(self) -> float:
        """The share of the class's tokens seen after this head."""
        return self.fillers / self.class_size

    @property
    def missing_mass(self) -> float:
        """Good-Turing: the probability that the next use of the pattern brings a filler it was
        never seen with, from the fillers seen in a single context."""
        return self.singletons / self.trunk


type _PatternKey = tuple[tuple[int, ...], int]


def _pattern_counts(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int,
    heads: Collection[int] | None,
) -> tuple[dict[_PatternKey, dict[int, int]], dict[_PatternKey, int]]:
    """Share context counting so coverage and diversity read exactly the same slots."""
    class_size = Counter(classes.classes.values())
    allowed_heads = set(heads) if heads is not None else None
    contexts: dict[tuple[tuple[int, ...], int], dict[int, set[tuple[int | None, int | None]]]]
    contexts = defaultdict(lambda: defaultdict(set))
    documents: dict[tuple[tuple[int, ...], int], set[int]] = defaultdict(set)
    for document, sequence in enumerate(sequences):
        for start in range(len(sequence) - head_length):
            filler = sequence[start + head_length]
            filler_class = classes.classes.get(filler)
            if filler_class is None or class_size[filler_class] < 2:
                continue
            head = tuple(sequence[start : start + head_length])
            if allowed_heads is not None and any(token not in allowed_heads for token in head):
                continue
            key = (head, filler_class)
            end = start + head_length + 1
            context = (
                sequence[start - 1] if start > 0 else None,
                sequence[end] if end < len(sequence) else None,
            )
            contexts[key][filler].add(context)
            documents[key].add(document)
    return (
        {
            key: {filler: len(seen) for filler, seen in group.items()}
            for key, group in contexts.items()
        },
        {key: len(seen) for key, seen in documents.items()},
    )


def syntactic_patterns(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int = 1,
    *,
    heads: Collection[int] | None = None,
) -> list[Pattern]:
    """Every (head of `head_length` tokens, class) seen in `sequences`, for the classes of two or
    more tokens: a class of one token is a fixed word, with no slot to generalize.
    If `heads` is given, every token of the head must belong to it. Expected fillers are
    the sum of `1 - (1 - p_y) ** trunk`, with class shares from all corpus occurrences:
    unlike coverage, this compares heads at different traffic against what free draws
    from their class would show.
    """
    class_size = Counter(classes.classes.values())
    frequency = Counter(token for sequence in sequences for token in sequence)
    class_counts: dict[int, list[int]] = defaultdict(list)
    for token, label in classes.classes.items():
        class_counts[label].append(frequency[token])
    shares = {
        label: [count / total for count in counts]
        for label, counts in class_counts.items()
        if (total := sum(counts)) > 0
    }
    counts, documents = _pattern_counts(sequences, classes, head_length, heads)
    return [
        Pattern(
            head=head,
            filler_class=filler_class,
            trunk=(trunk := sum(by_filler.values())),
            fillers=len(by_filler),
            class_size=class_size[filler_class],
            singletons=sum(count == 1 for count in by_filler.values()),
            documents=documents[(head, filler_class)],
            expected_fillers=sum(1 - (1 - share) ** trunk for share in shares[filler_class]),
        )
        for (head, filler_class), by_filler in sorted(counts.items())
    ]


@dataclass(frozen=True)
class ReferenceCoverage:
    """Reference filler mass covered by a target pattern, relative to its traffic."""

    head: tuple[int, ...]
    filler_class: int
    trunk: int
    covered_mass: float
    expected_mass: float

    @property
    def ratio(self) -> float:
        """Covered mass relative to the mass expected from the target's trunk."""
        return self.covered_mass / self.expected_mass if self.expected_mass else 0.0


def reference_coverage(
    target: Sequence[Sequence[int]],
    reference: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int = 1,
    *,
    heads: Collection[int] | None = None,
) -> list[ReferenceCoverage]:
    """Measure which fillers a shared slot shows against the reference domain's use.

    A topic-narrow corpus keeps its slots open but fills them with other words;
    diversity cannot see that mismatch. A fixed construction stays near one because
    it is fixed in the reference too.

    For each pattern present in both corpora, reference filler shares p_y use distinct
    contexts. With target trunk n and target filler set S, covered mass is the sum of
    p_y over S; expected mass is the sum of p_y * (1 - (1 - p_y) ** n). Target fillers
    absent from the reference contribute zero. Patterns use the same classes, head
    length and head filter as syntactic_patterns and are sorted by (head, filler_class).
    """
    target_counts, _ = _pattern_counts(target, classes, head_length, heads)
    reference_counts, _ = _pattern_counts(reference, classes, head_length, heads)
    readings = []
    for head, filler_class in sorted(target_counts.keys() & reference_counts.keys()):
        key = (head, filler_class)
        target_fillers = target_counts[key]
        trunk = sum(target_fillers.values())
        reference_fillers = reference_counts[key]
        total = sum(reference_fillers.values())
        shares = {filler: count / total for filler, count in reference_fillers.items()}
        readings.append(
            ReferenceCoverage(
                head=head,
                filler_class=filler_class,
                trunk=trunk,
                covered_mass=sum(
                    share for filler, share in shares.items() if filler in target_fillers
                ),
                expected_mass=sum(share * (1 - (1 - share) ** trunk) for share in shares.values()),
            )
        )
    return readings


Verdict = Literal["sufficient", "under-shown", "fixed", "unread"]


@dataclass(frozen=True)
class Diagnosis:
    """A target pattern's sufficiency relative to the same slot in a reference corpus."""

    pattern: Pattern
    reference: Pattern | None
    verdict: Verdict


def reference_diagnosis(
    target: Sequence[Pattern], reference: Sequence[Pattern], low: float = 0.5
) -> list[Diagnosis]:
    """Diagnose target patterns in order, matching `(head, filler_class)`.

    Both must come from the same tokenizer and classes. A reference distinguishes an
    under-shown slot from a construction the domain itself restricts.
    """
    lookup = {(pattern.head, pattern.filler_class): pattern for pattern in reference}
    diagnoses = []
    for pattern in target:
        matched = lookup.get((pattern.head, pattern.filler_class))
        verdict: Verdict
        if pattern.diversity >= low:
            verdict = "sufficient"
        elif matched is None:
            verdict = "unread"
        elif matched.diversity >= low:
            verdict = "under-shown"
        else:
            verdict = "fixed"
        diagnoses.append(Diagnosis(pattern, matched, verdict))
    return diagnoses


def split_domain(
    patterns: Sequence[Pattern], min_documents: int = 3, min_share: float = 0.01
) -> tuple[list[Pattern], list[Pattern]]:
    """`(domain, outliers)`: a pattern belongs to the domain when it occurs in at least
    `min_documents` documents and holds at least `min_share` of its head's occurrences — the rest
    are text the domain does not repeat, or strays off a head."""
    head_trunk: Counter[tuple[int, ...]] = Counter()
    for pattern in patterns:
        head_trunk[pattern.head] += pattern.trunk
    domain: list[Pattern] = []
    outliers: list[Pattern] = []
    for pattern in patterns:
        share = pattern.trunk / head_trunk[pattern.head]
        is_domain = pattern.documents >= min_documents and share >= min_share
        (domain if is_domain else outliers).append(pattern)
    return domain, outliers


def sufficiency_curve(patterns: Sequence[Pattern]) -> CoverageCurve:
    """For each coverage level reached, the share of the patterns' occurrences (trunk-weighted)
    in patterns covering at least that much of their class. The axis is coverage itself, already
    on `[0, 1]`, so the area is taken on it rather than on the curve's own maximum, and equals the
    trunk-weighted mean coverage."""
    total = sum(pattern.trunk for pattern in patterns)
    if total == 0:
        return CoverageCurve((), (), 0.0)
    levels = sorted({pattern.coverage for pattern in patterns})
    shares = tuple(
        sum(pattern.trunk for pattern in patterns if pattern.coverage >= level) / total
        for level in levels
    )
    auc = sum(pattern.trunk * pattern.coverage for pattern in patterns) / total
    return CoverageCurve(tuple(levels), shares, auc)
