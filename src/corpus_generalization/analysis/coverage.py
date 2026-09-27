"""Whether a corpus has drawn enough samples to have plausibly seen every pattern its domain has
to offer, not just the ones easiest to reach.

Two independent questions, because they need different information to answer:

- If the domain's full enumeration is known (a hand-built combinatorial corpus like `states`,
  where every (pronoun, adverb) pair is a pattern and there are exactly 21 of them), the classic
  coupon-collector tail bound says how many draws it takes, sampling uniformly at random, to have
  covered every pattern with probability at least `1 - epsilon`: `patterns * (ln(patterns) +
  ln(1/epsilon))`. This session's own experiment is the worked example: `states_train.txt`
  originally left 3 of 21 (pronoun, adverb) pairs with zero coverage, and 82 draws fell short of
  the ~74-draw bound at epsilon=0.05 only because those 82 were not spread uniformly across all 21
  — they were concentrated on 18 of them. `missing`, computed directly against the declared
  universe, is what actually caught that: an aggregate probability bound tells you the coverage is
  risky, only the enumeration tells you which patterns to fix.
- If no enumeration is declared — the ordinary case for a real corpus, where nobody hand-lists
  every possible pattern — `good_turing_missing_mass` estimates the same thing without one: the
  fraction of observed draws that were singletons (seen exactly once) is the classic Good-Turing
  estimate of the probability that the *next* draw would be a pattern never seen before. High
  singleton fraction late in a corpus is the saturation-curve flattening the same information some
  other way: a corpus still finding new patterns at every draw estimates its own unseen mass as
  large; one that has stopped estimates it as small.

Both need only counts the corpus supplies about itself — an n-gram frequency table built from the
project's own tokenizer, or, for a hand-built domain, whatever enumeration the builder already
knows — never an external model or reference corpus.
"""

import math
from collections import Counter
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class Encodable(Protocol):
    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]: ...


class NonPositivePatternCountError(ValueError):
    def __init__(self, patterns: int) -> None:
        super().__init__(f"patterns must be positive, got {patterns}.")
        self.patterns = patterns


class EpsilonOutOfRangeError(ValueError):
    def __init__(self, epsilon: float) -> None:
        super().__init__(f"epsilon must be strictly between 0 and 1, got {epsilon}.")
        self.epsilon = epsilon


def coupon_collector_bound(patterns: int, epsilon: float = 0.05) -> float:
    """How many uniform-at-random draws, over `patterns` distinct outcomes, it takes for the
    probability that at least one is still unseen to fall to `epsilon` or below.

    `P(some pattern unseen after k draws) <= patterns * (1 - 1/patterns)^k`, and bounding the
    right side by `epsilon` and solving for `k` gives `k >= patterns * ln(patterns / epsilon)` --
    the same thing written as `patterns * (ln(patterns) + ln(1/epsilon))` here, which is the
    number this docstring's callers actually want to read two terms of: "the size of the domain"
    and "the safety margin", rather than one combined logarithm.
    """
    if patterns <= 0:
        raise NonPositivePatternCountError(patterns)
    if not 0 < epsilon < 1:
        raise EpsilonOutOfRangeError(epsilon)
    return patterns * (math.log(patterns) + math.log(1 / epsilon))


@dataclass(frozen=True)
class PatternCoverage:
    """`counts.values()` boiled down to what a builder needs to know before training on it.

    `universe_size`/`missing`/`required_draws` are all `None` together when `pattern_coverage` was
    given no enumeration: there is no "sufficient" to compute without knowing how large the domain
    is, and reporting a `sufficient=False` a reader could not act on (no `missing` list to fix)
    would be worse than reporting nothing.
    """

    draws: int
    observed: int
    singleton_patterns: int
    good_turing_missing_mass: float
    universe_size: int | None
    missing: tuple[Hashable, ...]
    required_draws: float | None
    epsilon: float

    @property
    def sufficient(self) -> bool | None:
        return None if self.required_draws is None else self.draws >= self.required_draws


def pattern_coverage[Pattern: Hashable](
    counts: Mapping[Pattern, int],
    universe: Collection[Pattern] | None = None,
    epsilon: float = 0.05,
) -> PatternCoverage:
    draws = sum(counts.values())
    singleton_patterns = sum(1 for count in counts.values() if count == 1)
    good_turing = singleton_patterns / draws if draws else 0.0
    if universe is None:
        return PatternCoverage(
            draws, len(counts), singleton_patterns, good_turing, None, (), None, epsilon
        )
    missing = tuple(pattern for pattern in universe if counts.get(pattern, 0) == 0)
    required = coupon_collector_bound(len(universe), epsilon)
    return PatternCoverage(
        draws,
        len(counts),
        singleton_patterns,
        good_turing,
        len(universe),
        missing,
        required,
        epsilon,
    )


def ngram_counts(
    documents: Sequence[str], tokenizer: Encodable, order: int = 1
) -> Counter[tuple[int, ...]]:
    """Every `order`-length run of token ids across `documents`, pooled corpus-wide -- the
    self-contained default pattern for `pattern_coverage` when a corpus has no hand-declared
    enumeration: n-grams read off the project's own tokenizer, never an externally defined
    vocabulary. Position-blind for the same reason `analysis.markov_chain` is: an n-gram
    that opens one document and sits mid-sentence in another is the same pattern either way.
    """
    counts: Counter[tuple[int, ...]] = Counter()
    for document in documents:
        sequence = tokenizer.encode(document, recognize_markers=True)
        for position in range(len(sequence) - order + 1):
            counts[tuple(sequence[position : position + order])] += 1
    return counts
