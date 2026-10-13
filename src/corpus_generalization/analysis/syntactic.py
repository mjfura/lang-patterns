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

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import linkage

from corpus_generalization.analysis.markov_chain import CoverageCurve

# Neighbor features are limited to the most frequent tokens, the rest pooled in one column. Every
# eligible token is clustered: on 14.3 MB of FineWeb at 11,563 merges a cap of 4,000 left 13% of
# running tokens without a class, and all 11,636 took 173 s and 1.25 GB
# (`docs/2026-10-11-settings-from-data-design.md`).
CONTEXT_TOKENS = 1000
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
    # Each distinct token decoded once: decoding every occurrence took 38 of a 41-minute
    # diagnosis of 1,000 FineWeb documents against 4,000.
    texts = {token: decode(token) for token in {t for sequence in sequences for t in sequence}}
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
    sequences: Sequence[Sequence[int]],
    tokens: Sequence[int],
    context: Sequence[int],
) -> np.ndarray:
    """Per token, the square roots of its left and its right neighbor distributions over the
    `context` tokens, a pooled column for every other token and one for the sequence boundary —
    so Euclidean distance between rows is the Hellinger distance on each side."""
    other, boundary = len(context), len(context) + 1
    width = len(context) + 2
    counts = np.zeros((len(tokens), 2 * width))
    # Vectorized: the per-token loop took most of a 90-second clustering of 4,600 FineWeb
    # documents.
    flat = np.fromiter(chain.from_iterable(sequences), dtype=np.int64)
    if flat.size:
        lengths = np.fromiter((len(sequence) for sequence in sequences), dtype=np.int64)
        ends = np.cumsum(lengths)[lengths > 0]
        firsts = ends - lengths[lengths > 0]
        size = max(int(flat.max()), max(tokens, default=0), max(context, default=0)) + 1
        row_of = np.full(size, -1, dtype=np.int64)
        row_of[np.asarray(tokens, dtype=np.int64)] = np.arange(len(tokens))
        column_of = np.full(size, other, dtype=np.int64)
        column_of[np.asarray(context, dtype=np.int64)] = np.arange(len(context))
        left = column_of[np.roll(flat, 1)]
        left[firsts] = boundary
        right = column_of[np.roll(flat, -1)]
        right[ends - 1] = boundary
        rows = row_of[flat]
        kept = rows >= 0
        cells = counts.size
        counts += np.bincount(rows[kept] * 2 * width + left[kept], minlength=cells).reshape(
            counts.shape
        )
        counts += np.bincount(
            rows[kept] * 2 * width + width + right[kept], minlength=cells
        ).reshape(counts.shape)
    for half in (slice(0, width), slice(width, 2 * width)):
        counts[:, half] /= counts[:, half].sum(axis=1, keepdims=True)
    return np.sqrt(counts)


@dataclass(frozen=True, eq=False)
class ClassTree:
    """The Ward tree over `tokens`, built once: every cut, coarse or fine, is read from it.
    `merges` is SciPy's linkage, whose tree and heights are sklearn's Ward tree."""

    tokens: tuple[int, ...]
    merges: np.ndarray

    @property
    def merge_heights(self) -> tuple[float, ...]:
        """`merge_heights[k - 1]`: the height of the merge that leaves `k` classes."""
        return tuple(float(height) for height in self.merges[::-1, 2])

    def cut(
        self, n_classes: int | None = None, max_classes: int = 64, *, min_classes: int = 8
    ) -> TokenClasses:
        """`n_classes` classes, or at the largest jump in merge height from
        `max(2, min_classes)` to `max_classes` classes when it is `None`. If no candidate fits,
        the largest valid cut, or one class for a single token. `min_classes` has no effect on a
        fixed `n_classes`."""
        heights = self.merge_heights
        if len(self.tokens) == 1:
            return TokenClasses(classes={self.tokens[0]: 0}, n_classes=1, merge_heights=())
        if n_classes is None:
            # Near the leaves, tokens with almost identical neighbors merge at heights close to
            # zero, and any ratio over them is noise: on the most restricted recipe variant it
            # picked 64 classes, nearly all singletons. The floor keeps the jump a jump at the
            # tree's own scale.
            floor = NOISE_FLOOR * heights[0]
            # On 500 BPE-framed recipe documents, k=2 mixed 95 tokens across grammar categories;
            # a floor of eight recovered the coarse categories.
            cuts = range(max(2, min_classes), min(max_classes, len(heights)) + 1)
            n_classes = max(
                cuts,
                key=lambda k: heights[k - 2] / max(heights[k - 1], floor),
                default=min(max_classes, len(heights)),
            )
        if not 1 <= n_classes <= len(self.tokens):
            raise ValueError("n_classes must be between 1 and the number of tokens.")
        size = len(self.tokens)
        parent = list(range(2 * size - 1))

        def root(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for index, (left, right, _, _) in enumerate(self.merges[: size - n_classes]):
            parent[root(int(left))] = parent[root(int(right))] = size + index
        members: dict[int, list[int]] = defaultdict(list)
        for leaf, token in enumerate(self.tokens):
            members[root(leaf)].append(token)
        numbered = sorted(members.values(), key=min)
        return TokenClasses(
            classes={token: number for number, group in enumerate(numbered) for token in group},
            n_classes=n_classes,
            merge_heights=heights,
        )


def class_tree(
    sequences: Sequence[Sequence[int]],
    min_documents: int = 2,
    *,
    candidates: Collection[int] | None = None,
) -> ClassTree:
    """The Ward tree of the tokens found in at least `min_documents` sequences, on their
    neighbor distributions. When `candidates` is given, only its tokens are clustered; context
    columns still use the whole corpus."""
    documents: Counter[int] = Counter()
    frequency: Counter[int] = Counter()
    for sequence in sequences:
        documents.update(set(sequence))
        frequency.update(sequence)
    allowed = set(candidates) if candidates is not None else None
    tokens = sorted(
        token
        for token in frequency
        if documents[token] >= min_documents and (allowed is None or token in allowed)
    )
    if not tokens:
        raise ValueError("No token is found in enough documents to classify.")
    if len(tokens) == 1:
        return ClassTree(tuple(tokens), np.zeros((0, 4)))
    context = [token for token, _ in frequency.most_common(CONTEXT_TOKENS)]
    return ClassTree(
        tuple(tokens),
        linkage(_neighbor_features(sequences, tokens, context), method="ward"),
    )


def induce_classes(
    sequences: Sequence[Sequence[int]],
    n_classes: int | None = None,
    min_documents: int = 2,
    max_classes: int = 64,
    *,
    min_classes: int = 8,
    candidates: Collection[int] | None = None,
) -> TokenClasses:
    """Cluster the tokens found in at least `min_documents` sequences into coarse classes: one
    cut of `class_tree`, see `ClassTree.cut`."""
    tree = class_tree(sequences, min_documents, candidates=candidates)
    return tree.cut(n_classes, max_classes, min_classes=min_classes)


@dataclass(frozen=True)
class Pattern:
    """A head followed by a token of `filler_class`. Occurrences are counted in distinct contexts
    (the token before the head, the token after the filler), so exact copies add nothing.
    `lexical` patterns are those whose filler continues the head's last word: spelling, not
    syntax."""

    head: tuple[int, ...]
    filler_class: int
    trunk: int
    fillers: int
    class_size: int
    singletons: int
    documents: int
    expected_fillers: float = 0.0
    lexical: bool = False

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


type _PatternKey = tuple[tuple[int, ...], int, bool]


def _joint_codes(*columns: np.ndarray) -> np.ndarray:
    """One integer per row, equal exactly when the rows are equal. Each column is folded in as a
    mixed-radix digit; the codes so far are ranked first only when the product could overflow."""
    codes = np.zeros(len(columns[0]), dtype=np.int64)
    for column in columns:
        shifted = column - column.min() if len(column) else column
        radix = int(shifted.max(initial=0)) + 1
        if (int(codes.max(initial=0)) + 1) * radix >= 2**62:
            codes = np.unique(codes, return_inverse=True)[1].astype(np.int64)
        codes = codes * radix + shifted
    return codes


@dataclass(frozen=True)
class _Slots:
    """Every position a pattern of `head_length` reads, as parallel arrays: the head's tokens,
    the filler, whether it continues a word, the tokens around the pattern (-1 at an edge), and
    the sequence."""

    head: tuple[np.ndarray, ...]
    filler: np.ndarray
    lexical: np.ndarray
    before: np.ndarray
    after: np.ndarray
    document: np.ndarray
    class_of: np.ndarray


def _slots(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int,
    heads: Collection[int] | None,
    word_starts: Sequence[Sequence[bool]] | None,
) -> _Slots | None:
    """Vectorized: a loop over every position took 18 of the 28 seconds of one noise draw of
    2,220 FineWeb documents once every token could be a head. `None` when nothing is read."""
    flat = np.fromiter(chain.from_iterable(sequences), dtype=np.int64)
    if not flat.size:
        return None
    lengths = np.fromiter((len(sequence) for sequence in sequences), dtype=np.int64)
    document_of = np.repeat(np.arange(len(sequences)), lengths)
    position = np.arange(flat.size) - (np.cumsum(lengths) - lengths)[document_of]
    remaining = lengths[document_of] - position
    size = max(int(flat.max()), max(classes.classes, default=0)) + 1
    class_size = Counter(classes.classes.values())
    class_of = np.full(size, -1, dtype=np.int64)
    for token, label in classes.classes.items():
        if class_size[label] >= 2:
            class_of[token] = label
    first = np.flatnonzero(remaining > head_length)
    filler = first + head_length
    kept = class_of[flat[filler]] >= 0
    if heads is not None:
        allowed = np.zeros(size, dtype=bool)
        allowed[[token for token in heads if token < size]] = True
        for offset in range(head_length):
            kept &= allowed[flat[first + offset]]
    first, filler = first[kept], filler[kept]
    if not len(first):
        return None
    if word_starts is not None:
        opens = np.fromiter(chain.from_iterable(word_starts), dtype=bool, count=flat.size)
        lexical = (~opens[filler]).astype(np.int64)
    else:
        lexical = np.zeros(len(filler), dtype=np.int64)
    return _Slots(
        head=tuple(flat[first + offset] for offset in range(head_length)),
        filler=flat[filler],
        lexical=lexical,
        before=np.where(position[first] > 0, flat[first - 1], -1),
        after=np.where(remaining[filler] > 1, flat[np.minimum(filler + 1, flat.size - 1)], -1),
        document=document_of[filler],
        class_of=class_of,
    )


def _representatives(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row index per distinct code, and how many rows share it. Any row of a code serves:
    rows with equal codes are equal. An unstable sort halves `np.unique`'s time here."""
    order = np.argsort(codes, kind="quicksort")
    ordered = codes[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    return order[starts], np.diff(np.r_[starts, len(ordered)])


def _keys(slots: _Slots, rows: np.ndarray) -> list[tuple[_PatternKey, int]]:
    """The pattern key and filler of each of `rows`; built from lists, since indexing numpy
    scalars one at a time cost 20 seconds over 9 million rows."""
    heads = zip(*(column[rows].tolist() for column in slots.head), strict=True)
    fillers = slots.filler[rows]
    return [
        ((head, label, bool(lexical)), filler)
        for head, label, lexical, filler in zip(
            heads,
            slots.class_of[fillers].tolist(),
            slots.lexical[rows].tolist(),
            fillers.tolist(),
            strict=True,
        )
    ]


def _filler_contexts(slots: _Slots) -> tuple[np.ndarray, np.ndarray]:
    """One row per (head, lexical, filler) and its count of distinct contexts."""
    # A filler's class follows from the filler, so (head, lexical, filler) fixes the key.
    slot = _joint_codes(*slots.head, slots.lexical, slots.filler)
    seen, _ = _representatives(_joint_codes(slot, slots.before, slots.after))
    rows, contexts = _representatives(slot[seen])
    return seen[rows], contexts


def _pattern_documents(slots: _Slots, pattern: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row per pattern code and the number of sequences it is seen in."""
    seen, _ = _representatives(_joint_codes(pattern, slots.document))
    rows, spread = _representatives(pattern[seen])
    return seen[rows], spread


def _pattern_counts(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int,
    heads: Collection[int] | None,
    word_starts: Sequence[Sequence[bool]] | None = None,
) -> tuple[dict[_PatternKey, dict[int, int]], dict[_PatternKey, int]]:
    """Share context counting so coverage and diversity read exactly the same slots. A key is
    (head, filler class, lexical); without `word_starts` every filler starts a word. A filler's
    count is its distinct contexts: the token before the head and the token after the filler,
    or none at a sequence's edge."""
    slots = _slots(sequences, classes, head_length, heads, word_starts)
    if slots is None:
        return {}, {}
    rows, contexts = _filler_contexts(slots)
    counts: dict[_PatternKey, dict[int, int]] = defaultdict(dict)
    for (key, filler), count in zip(_keys(slots, rows), contexts.tolist(), strict=True):
        counts[key][filler] = count
    pattern = _joint_codes(*slots.head, slots.lexical, slots.class_of[slots.filler])
    rows, spread = _pattern_documents(slots, pattern)
    documents = {
        key: count for (key, _), count in zip(_keys(slots, rows), spread.tolist(), strict=True)
    }
    return dict(counts), documents


def filler_occurrences(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int,
    word_starts: Sequence[Sequence[bool]] | None = None,
) -> dict[_PatternKey, Counter[int]]:
    """Per pattern key, every occurrence of each filler, copies included: what a report shows as
    a slot's most frequent fillers."""
    slots = _slots(sequences, classes, head_length, None, word_starts)
    if slots is None:
        return {}
    rows, occurrences = _representatives(_joint_codes(*slots.head, slots.lexical, slots.filler))
    counts: dict[_PatternKey, Counter[int]] = defaultdict(Counter)
    for (key, filler), count in zip(_keys(slots, rows), occurrences.tolist(), strict=True):
        counts[key][filler] = count
    return dict(counts)


def pattern_filler_counts(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int,
    word_starts: Sequence[Sequence[bool]] | None = None,
) -> dict[_PatternKey, dict[int, int]]:
    """Per pattern key, each filler's count of distinct contexts: the counts its diversity is
    read from."""
    return _pattern_counts(sequences, classes, head_length, None, word_starts)[0]


def syntactic_patterns(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_length: int = 1,
    *,
    heads: Collection[int] | None = None,
    word_starts: Sequence[Sequence[bool]] | None = None,
) -> list[Pattern]:
    """Every (head of `head_length` tokens, class) seen in `sequences`, for the classes of two or
    more tokens: a class of one token is a fixed word, with no slot to generalize.
    If `heads` is given, every token of the head must belong to it. `word_starts`, parallel to
    `sequences`, marks the tokens that open a pre-token; a filler that does not is a lexical
    pattern, read apart from the syntactic one of the same head and class. Expected fillers are
    the sum of `1 - (1 - p_y) ** trunk`, with class shares from all corpus occurrences:
    unlike coverage, this compares heads at different traffic against what free draws
    from their class would show.
    """
    class_size = Counter(classes.classes.values())
    flat = np.fromiter(chain.from_iterable(sequences), dtype=np.int64)
    frequency = np.bincount(flat, minlength=max(classes.classes, default=0) + 1).tolist()
    class_counts: dict[int, list[int]] = defaultdict(list)
    for token, label in classes.classes.items():
        class_counts[label].append(frequency[token])
    shares = {
        label: [count / total for count in counts]
        for label, counts in class_counts.items()
        if (total := sum(counts)) > 0
    }
    slots = _slots(sequences, classes, head_length, heads, word_starts)
    if slots is None:
        return []
    # Totals per pattern in arrays: a dict of fillers per pattern cost 28 of 109 seconds.
    rows, contexts = _filler_contexts(slots)
    pattern = _joint_codes(*slots.head, slots.lexical, slots.class_of[slots.filler])
    order = np.argsort(pattern[rows], kind="quicksort")
    grouped = pattern[rows][order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    ordered = contexts[order]
    trunks = np.add.reduceat(ordered, starts).tolist()
    filler_counts = np.diff(np.r_[starts, len(ordered)]).tolist()
    singletons = np.add.reduceat((ordered == 1).astype(np.int64), starts).tolist()
    document_rows, spread = _pattern_documents(slots, pattern)
    spread_of = dict(zip(pattern[document_rows].tolist(), spread.tolist(), strict=True))
    documents = [spread_of[code] for code in grouped[starts].tolist()]
    # One sum per (class, trunk): with every token classified a class has hundreds of members,
    # and summing them per pattern was 22 of a 35-second read of 2,220 FineWeb documents.
    expected: dict[tuple[int, int], float] = {}

    def expected_fillers(filler_class: int, trunk: int) -> float:
        key = (filler_class, trunk)
        if key not in expected:
            expected[key] = sum(1 - (1 - share) ** trunk for share in shares[filler_class])
        return expected[key]

    patterns = [
        Pattern(
            head=head,
            filler_class=filler_class,
            trunk=trunk,
            fillers=fillers,
            class_size=class_size[filler_class],
            singletons=single,
            documents=spread_count,
            expected_fillers=expected_fillers(filler_class, trunk),
            lexical=lexical,
        )
        for ((head, filler_class, lexical), _), trunk, fillers, single, spread_count in zip(
            _keys(slots, rows[order[starts]]),
            trunks,
            filler_counts,
            singletons,
            documents,
            strict=True,
        )
    ]
    return sorted(patterns, key=lambda p: (p.head, p.filler_class, p.lexical))


def filler_vocabulary(classes: TokenClasses) -> frozenset[int]:
    """The classified tokens that can fill a slot: a class of one token is a fixed word."""
    sizes = Counter(classes.classes.values())
    return frozenset(token for token, label in classes.classes.items() if sizes[label] >= 2)


def class_fillers(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    head_classes: TokenClasses,
    heads: Collection[int] | None = None,
    word_starts: Sequence[Sequence[bool]] | None = None,
) -> dict[tuple[int, bool], Counter[int]]:
    """Per (head class, lexical), the fillers after any of its one-token heads, in distinct
    contexts.

    A model learns syntax at the class level: `experiments/class_patterns` removed the verb
    after `she` and `they` and their syntax held while `he`, `we` and `I` kept it.
    """
    slots = _slots(sequences, classes, 1, heads, word_starts)
    if slots is None:
        return {}
    rows, contexts = _filler_contexts(slots)
    head_class_of = np.full(len(slots.class_of), -1, dtype=np.int64)
    for token, label in head_classes.classes.items():
        if token < len(head_class_of):
            head_class_of[token] = label
    labels = head_class_of[slots.head[0][rows]]
    classified = labels >= 0
    rows, contexts, labels = rows[classified], contexts[classified], labels[classified]
    if not len(rows):
        return {}
    lexical, fillers = slots.lexical[rows], slots.filler[rows]
    order = np.argsort(_joint_codes(labels, lexical, fillers), kind="quicksort")
    grouped = _joint_codes(labels, lexical, fillers)[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    first = order[starts]
    pooled: dict[tuple[int, bool], Counter[int]] = defaultdict(Counter)
    for label, kind, filler, total in zip(
        labels[first].tolist(),
        lexical[first].tolist(),
        fillers[first].tolist(),
        np.add.reduceat(contexts[order], starts).tolist(),
        strict=True,
    ):
        pooled[(label, bool(kind))][filler] = total
    return dict(pooled)


def head_filler_matrices(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    heads: Sequence[int],
    fillers: Sequence[int],
    word_starts: Sequence[Sequence[bool]] | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Per one-token head (rows, in `heads` order) and filler (columns, in `fillers` order), the
    filler's distinct contexts after the head: syntactic transitions, then lexical ones."""
    shape = (len(heads), len(fillers))
    slots = _slots(sequences, classes, 1, None, word_starts)
    if slots is None:
        return sparse.csr_matrix(shape), sparse.csr_matrix(shape)
    rows, contexts = _filler_contexts(slots)
    size = len(slots.class_of)
    row_of = np.full(max(size, max(heads, default=0) + 1), -1, dtype=np.int64)
    row_of[np.asarray(heads, dtype=np.int64)] = np.arange(len(heads))
    column_of = np.full(len(row_of), -1, dtype=np.int64)
    column_of[np.asarray(fillers, dtype=np.int64)] = np.arange(len(fillers))
    head_rows = row_of[slots.head[0][rows]]
    columns = column_of[slots.filler[rows]]
    kept = (head_rows >= 0) & (columns >= 0)
    lexical = slots.lexical[rows].astype(bool)

    def matrix(kind: np.ndarray) -> sparse.csr_matrix:
        chosen = kept & kind
        return sparse.csr_matrix(
            (contexts[chosen], (head_rows[chosen], columns[chosen])), shape=shape, dtype=np.float64
        )

    return matrix(~lexical), matrix(lexical)


def tree_membership(tree: ClassTree) -> sparse.csr_matrix:
    """Rows are the tree's internal nodes in merge order, columns its leaves (`tree.tokens`): 1
    where the leaf lies under the node."""
    size = len(tree.tokens)
    members: list[list[int]] = [[leaf] for leaf in range(size)]
    rows: list[int] = []
    columns: list[int] = []
    for node, (left, right, _, _) in enumerate(tree.merges):
        joined = members[int(left)] + members[int(right)]
        members.append(joined)
        rows.extend([node] * len(joined))
        columns.extend(joined)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(max(size - 1, 0), size))


def tree_parents(tree: ClassTree) -> np.ndarray:
    """Each internal node's parent, as an internal-node index; -1 at the root."""
    size = len(tree.tokens)
    parent = np.full(max(size - 1, 0), -1, dtype=np.int64)
    for node, (left, right, _, _) in enumerate(tree.merges):
        for child in (int(left), int(right)):
            if child >= size:
                parent[child - size] = node
    return parent


def node_divergences(
    reference: sparse.csr_matrix, corpus: sparse.csr_matrix, unigram: np.ndarray
) -> np.ndarray:
    """`filler_divergence` for every row at once: KL(R || S) in nats per row, S with Witten-Bell
    backoff to `unigram` (one entry per column). Rows with no reference mass read 0."""
    reference = sparse.csr_matrix(reference)
    reference.eliminate_zeros()
    reference.sort_indices()
    corpus = sparse.csr_matrix(corpus)
    corpus.eliminate_zeros()
    corpus.sort_indices()
    width = reference.shape[1]
    totals = np.asarray(reference.sum(axis=1)).ravel()
    seen = np.asarray(corpus.sum(axis=1)).ravel()
    distinct = np.diff(corpus.indptr).astype(np.float64)
    rows = np.repeat(np.arange(reference.shape[0]), np.diff(reference.indptr))
    columns = reference.indices.astype(np.int64)
    counts = reference.data
    # Both matrices' entries sorted by (row, column): a search gathers the corpus's counts at the
    # reference's entries, which sparse fancy indexing did 20 times slower.
    keys = np.repeat(np.arange(corpus.shape[0]), np.diff(corpus.indptr)) * width + corpus.indices
    wanted = rows * width + columns
    found = np.minimum(np.searchsorted(keys, wanted), max(len(keys) - 1, 0))
    shown = (
        np.where(keys[found] == wanted, corpus.data[found], 0.0)
        if len(keys)
        else np.zeros(len(wanted))
    )
    backoff = unigram[columns]
    smoothed = np.where(
        seen[rows] > 0,
        (shown + distinct[rows] * backoff) / np.maximum(seen[rows] + distinct[rows], 1.0),
        backoff,
    )
    share = counts / totals[rows]
    values = np.zeros(reference.shape[0])
    np.add.at(values, rows, share * np.log(share / smoothed))
    return values


def unigram_vector(sequences: Sequence[Sequence[int]], fillers: Sequence[int]) -> np.ndarray:
    """`backoff_unigram` as a vector in `fillers` order."""
    flat = np.fromiter(chain.from_iterable(sequences), dtype=np.int64)
    columns = np.asarray(fillers, dtype=np.int64)
    counts = np.bincount(flat, minlength=int(columns.max(initial=0)) + 1)[columns]
    return (counts + 1.0) / (counts.sum() + len(fillers))


def backoff_unigram(
    sequences: Sequence[Sequence[int]], vocabulary: Collection[int]
) -> dict[int, float]:
    """Every occurrence of each filler, plus one per filler of the vocabulary: without the
    added count a filler absent from the corpus would make the divergence infinite."""
    counts = Counter(token for sequence in sequences for token in sequence if token in vocabulary)
    total = sum(counts.values()) + len(vocabulary)
    return {token: (counts[token] + 1) / total for token in vocabulary}


def filler_divergence(
    reference: Mapping[int, int], corpus: Mapping[int, int], unigram: Mapping[int, float]
) -> float:
    """KL(R || S) in nats: R the reference's fillers, S the corpus's with Witten-Bell backoff to
    `unigram`. A model that learned S loses this much on R's text (`experiments/divergence`)."""
    total = sum(reference.values())
    if total == 0:
        raise ValueError("The reference has no fillers for this slot")
    n, distinct = sum(corpus.values()), len(corpus)
    value = 0.0
    for filler, count in reference.items():
        if count == 0:
            continue
        backoff = unigram[filler]
        smoothed = (corpus.get(filler, 0) + distinct * backoff) / (n + distinct) if n else backoff
        share = count / total
        value += share * math.log(share / smoothed)
    return value


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
    for key in sorted(target_counts.keys() & reference_counts.keys()):
        head, filler_class, _ = key
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
