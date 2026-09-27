"""`analysis.coverage`: coupon-collector coverage and Good-Turing missing mass, both
computed only from the corpus's own pattern counts -- no external model or reference corpus.
"""

import math
from collections import Counter

import pytest

from corpus_generalization.analysis.coverage import (
    EpsilonOutOfRangeError,
    NonPositivePatternCountError,
    coupon_collector_bound,
    ngram_counts,
    pattern_coverage,
)


class WordIdTokenizer:
    """One id per distinct whitespace-separated word, assigned in first-seen order."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        return [self._ids.setdefault(word, len(self._ids)) for word in text.split()]


def test_coupon_collector_bound_matches_the_textbook_formula() -> None:
    assert coupon_collector_bound(21, epsilon=0.05) == pytest.approx(
        21 * (math.log(21) + math.log(1 / 0.05))
    )


def test_coupon_collector_bound_grows_with_more_patterns() -> None:
    assert coupon_collector_bound(105, epsilon=0.05) > coupon_collector_bound(21, epsilon=0.05)


def test_coupon_collector_bound_grows_as_epsilon_tightens() -> None:
    assert coupon_collector_bound(21, epsilon=0.01) > coupon_collector_bound(21, epsilon=0.1)


def test_coupon_collector_bound_rejects_a_non_positive_pattern_count() -> None:
    with pytest.raises(NonPositivePatternCountError):
        coupon_collector_bound(0)


def test_coupon_collector_bound_rejects_epsilon_outside_zero_one() -> None:
    with pytest.raises(EpsilonOutOfRangeError):
        coupon_collector_bound(21, epsilon=1.5)
    with pytest.raises(EpsilonOutOfRangeError):
        coupon_collector_bound(21, epsilon=0.0)


def test_pattern_coverage_finds_exactly_the_patterns_missing_from_a_declared_universe() -> None:
    """The `states` split this session actually built by hand: 21 (pronoun, adverb) pairs, 3 of
    them entirely absent from the counts -- `missing` should name exactly those three."""
    universe = {(pronoun, adverb) for pronoun in range(7) for adverb in range(3)}
    counts = Counter({pair: 5 for pair in universe if pair not in {(0, 2), (3, 0), (4, 1)}})

    coverage = pattern_coverage(counts, universe=universe)

    assert coverage.universe_size == 21
    assert set(coverage.missing) == {(0, 2), (3, 0), (4, 1)}
    assert coverage.observed == 18


def test_pattern_coverage_sufficient_reflects_the_coupon_collector_bound() -> None:
    universe = set(range(21))
    plenty = Counter(dict.fromkeys(universe, 100))
    scarce = Counter(dict.fromkeys(list(universe)[:5], 1))

    assert pattern_coverage(plenty, universe=universe).sufficient is True
    assert pattern_coverage(scarce, universe=universe).sufficient is False


def test_pattern_coverage_without_a_universe_reports_no_bound_and_no_missing_list() -> None:
    coverage = pattern_coverage(Counter({"a": 3, "b": 1}))

    assert coverage.universe_size is None
    assert coverage.required_draws is None
    assert coverage.sufficient is None
    assert coverage.missing == ()
    assert coverage.observed == 2


def test_good_turing_missing_mass_is_the_singleton_fraction_of_all_draws() -> None:
    # "a" seen 3 times, "b" and "c" seen once each: 2 singleton draws out of 5 total.
    coverage = pattern_coverage(Counter({"a": 3, "b": 1, "c": 1}))

    assert coverage.draws == 5
    assert coverage.singleton_patterns == 2
    assert coverage.good_turing_missing_mass == pytest.approx(2 / 5)


def test_good_turing_missing_mass_is_zero_when_every_pattern_repeats() -> None:
    coverage = pattern_coverage(Counter({"a": 3, "b": 2}))

    assert coverage.good_turing_missing_mass == 0.0


def test_ngram_counts_pools_a_transition_wherever_it_occurs_not_only_at_a_document_start() -> None:
    """Mirrors `analysis.markov_chain`'s own test: "I am" opens the first document but
    sits mid-sentence in the second, and should still count as the same order-2 pattern."""
    documents = ("I am happy", "she said I am ready")

    counts = ngram_counts(documents, WordIdTokenizer(), order=2)

    tokenizer = WordIdTokenizer()
    i_id = tokenizer.encode("I")[0]
    am_id = tokenizer.encode("am")[0]
    assert counts[(i_id, am_id)] == 2


def test_ngram_counts_respects_order() -> None:
    counts = ngram_counts(("a b c",), WordIdTokenizer(), order=2)

    assert sum(counts.values()) == 2  # (a,b) and (b,c)
    assert all(len(pattern) == 2 for pattern in counts)
