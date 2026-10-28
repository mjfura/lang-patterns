"""A weak pattern by the corpus's own evidence: a head, or a class of heads, against its sibling.

Without a reference nothing says how many fillers the domain uses after a head. The head tree
does say how many its sibling shows: had the parent node's occurrences of a filler class fallen
to its two children at random, each would show some number of distinct fillers. A child showing
far fewer is weak in the corpus. Whether the domain shares that weakness is what a reference
settles (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`, step 4).

A head of two or more tokens is placed by its first token: ` chop the` is compared with
` cut the` wherever ` chop` and ` cut` fall in the head tree.

Only children of a node inside one coarse class are tested: above the class cut a random split
mixes categories (`,`/`the`/`to` against `.`/`:`), and on FineWeb every such test read as weak.
Against the whole coarse class rather than the sibling, and with a test of how often a head takes
each filler class, FineWeb's control read about 2,000 weak heads and 14,000 weak classes that its
two halves did not share; against the sibling, 264 of its weak tests were shared by both halves,
of 332 and 397 (`experiments/saturation`).
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from corpus_generalization.analysis.syntactic import ClassTree

type PatternKey = tuple[tuple[int, ...], int, bool]


@dataclass(frozen=True)
class SiblingTest:
    """One child of a head-tree node, for one (head suffix, filler class, lexical). `node` is a
    leaf index below `len(tree.tokens)`, else that plus the internal node's merge index. `null`
    holds the distinct fillers of random shares of the parent's occurrences at the child's
    trunk."""

    node: int
    suffix: tuple[int, ...]
    filler_class: int
    lexical: bool
    trunk: int
    fillers: int
    null: np.ndarray
    child_counts: Mapping[int, int]
    parent_counts: Mapping[int, int]

    @property
    def mean(self) -> float:
        return float(self.null.mean())

    @property
    def sd(self) -> float:
        """The splits' spread, with the shown count's rounding: it moves in steps of one."""
        return float(np.sqrt(self.null.var(ddof=1) + 1 / 12))

    @property
    def t(self) -> float:
        return (self.mean - self.fillers) / self.sd

    @property
    def missing(self) -> float:
        """Fillers the child would show as a random share of its parent, and does not."""
        return self.mean - self.fillers


def sibling_tests(
    counts: Mapping[PatternKey, Mapping[int, int]],
    tree: ClassTree,
    n_classes: int,
    draws: int,
    rng: np.random.Generator,
) -> list[SiblingTest]:
    """Every testable child, with `draws` random splits each. `counts` holds each pattern's
    fillers in distinct contexts; heads whose first token is not in the tree are skipped."""
    size = len(tree.tokens)
    leaf_of = {token: leaf for leaf, token in enumerate(tree.tokens)}
    parent = np.full(2 * size - 1, -1, dtype=np.int64)
    children: dict[int, tuple[int, int]] = {}
    for index, (left, right, _, _) in enumerate(tree.merges):
        parent[int(left)] = parent[int(right)] = size + index
        children[size + index] = (int(left), int(right))
    inside = size - n_classes
    groups: dict[tuple[tuple[int, ...], int, bool], list[tuple[int, Mapping[int, int]]]]
    groups = defaultdict(list)
    for (head, filler_class, lexical), fillers in counts.items():
        if head[0] in leaf_of and fillers:
            groups[(head[1:], filler_class, lexical)].append((leaf_of[head[0]], fillers))
    tests: list[SiblingTest] = []
    for (suffix, filler_class, lexical), members in groups.items():
        # Every node holding a member, with its pooled fillers, up to the top of its class.
        pooled: dict[int, dict[int, int]] = {}
        for leaf, fillers in members:
            node = leaf
            pooled[node] = dict(fillers)
            while parent[node] >= 0 and parent[node] - size < inside:
                node = int(parent[node])
                into = pooled.setdefault(node, {})
                for filler, count in fillers.items():
                    into[filler] = into.get(filler, 0) + count
        for node in [node for node in pooled if node >= size]:
            pair = children[node]
            # A child with none of the class leaves its sibling all of the parent: no split.
            if not all(child in pooled for child in pair):
                continue
            bag_counts = pooled[node]
            bag = np.fromiter(bag_counts.values(), dtype=np.int64)
            for child in pair:
                child_counts = pooled[child]
                trunk = sum(child_counts.values())
                split = rng.multivariate_hypergeometric(bag, trunk, size=draws, method="marginals")
                tests.append(
                    SiblingTest(
                        child,
                        suffix,
                        filler_class,
                        lexical,
                        trunk,
                        len(child_counts),
                        (split > 0).sum(axis=1),
                        child_counts,
                        bag_counts,
                    )
                )
    return tests


def sibling_bar(tests: list[SiblingTest], rank: int) -> tuple[float | None, np.ndarray]:
    """The bar of one family of tests, and which tests it reads. Each split is a draw: the bar is
    the `rank`-th largest of the draws' largest t against the other draws. A test is read only
    if a child showing a single filler could cross it (Tarone): untestable ones only raise the
    bar for the rest."""
    readable = np.ones(len(tests), dtype=bool)
    if not tests:
        return None, readable
    null = np.stack([test.null for test in tests], axis=1).astype(np.float64)
    draws = null.shape[0]
    total, squares = null.sum(axis=0), (null**2).sum(axis=0)
    # Each draw against the other draws, from sums: a loop over draws was the cost at 101.
    others_mean = (total - null) / (draws - 1)
    others_var = ((squares - null**2) - (draws - 1) * others_mean**2) / (draws - 2)
    drop = (others_mean - null) / np.sqrt(np.maximum(others_var, 0) + 1 / 12)
    mean = total / draws
    sd = np.sqrt(np.maximum((squares - draws * mean**2) / (draws - 1), 0) + 1 / 12)
    bar: float | None = None
    for _ in range(len(tests)):
        if not readable.any():
            return None, readable
        bar = float(np.sort(drop[:, readable].max(axis=1))[::-1][rank - 1])
        now = (mean - 1) / sd > bar
        if (now == readable).all():
            break
        readable = now
    return bar, readable
