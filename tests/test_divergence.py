"""The divergence D on small counts, without reading any corpus or result."""

import importlib.util
import math
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from corpus_generalization.analysis.syntactic import TokenClasses


@pytest.fixture
def experiment() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments/divergence/divergence.py"
    spec = importlib.util.spec_from_file_location("divergence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UNIFORM = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}


def test_divergence_is_small_when_the_corpus_matches_the_reference(
    experiment: ModuleType,
) -> None:
    counts = {1: 400, 2: 300, 3: 200, 4: 100}
    assert experiment.divergence(counts, counts, UNIFORM) < 0.01


def test_divergence_grows_as_mass_moves_onto_one_filler(experiment: ModuleType) -> None:
    reference = {1: 250, 2: 250, 3: 250, 4: 250}
    mild = {1: 400, 2: 200, 3: 200, 4: 200}
    strong = {1: 900, 2: 34, 3: 33, 4: 33}
    values = [experiment.divergence(reference, c, UNIFORM) for c in (reference, mild, strong)]
    assert values == sorted(values)
    assert values[2] > 1.0


def test_dropping_rare_fillers_moves_little_mass(experiment: ModuleType) -> None:
    reference = {1: 960, 2: 30, 3: 6, 4: 4}
    without_rare = {1: 960, 2: 30}
    planted = {1: 990, 2: 5, 3: 3, 4: 2}
    assert experiment.divergence(reference, without_rare, UNIFORM) < experiment.divergence(
        reference, planted, UNIFORM
    )


def test_missing_fillers_and_empty_corpus_stay_finite(experiment: ModuleType) -> None:
    reference = {1: 5, 2: 5}
    assert math.isfinite(experiment.divergence(reference, {1: 10}, UNIFORM))
    # With no corpus counts, S is the backoff itself.
    expected = sum(0.5 * math.log(0.5 / 0.25) for _ in range(2))
    assert experiment.divergence(reference, {}, UNIFORM) == pytest.approx(expected)


def test_backoff_unigram_adds_one_per_vocabulary_filler(experiment: ModuleType) -> None:
    unigram = experiment.backoff_unigram([[1, 1, 9], [2]], frozenset({1, 2, 3}))
    assert unigram == {1: 3 / 6, 2: 2 / 6, 3: 1 / 6}


def test_head_fillers_pool_patterns_and_ignore_fixed_words(experiment: ModuleType) -> None:
    classes = TokenClasses({10: 0, 11: 0, 20: 1, 21: 1, 30: 2}, 3, (1.0, 1.0))
    sequences = [[5, 10, 7], [5, 20, 7], [5, 30, 7], [6, 5, 21, 8], [5, 10, 7]]
    fillers = experiment.head_fillers(sequences, classes, {5})
    # The second "5 10 7" repeats a context and adds nothing; 30 is a class of one token.
    assert fillers == {5: Counter({10: 1, 20: 1, 21: 1})}
    assert experiment.fillers_vocabulary(classes) == frozenset({10, 11, 20, 21})


def test_origin_fit_recovers_a_proportional_relation(experiment: ModuleType) -> None:
    slope, r2, residuals = experiment.origin_fit([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0])
    assert slope == pytest.approx(-1.0)
    assert r2 == pytest.approx(1.0)
    assert residuals == pytest.approx([0.0, 0.0, 0.0])


def test_random_cut_reaches_the_target_bytes_and_is_seeded(experiment: ModuleType) -> None:
    sizes = [10, 20, 30, 40, 50]
    cut = experiment.random_cut(sizes, 45, 3)
    assert sum(sizes[i] for i in cut) >= 45
    assert sum(sizes[i] for i in cut[:-1]) < 45
    assert cut == experiment.random_cut(sizes, 45, 3)


def test_score_weights_excess_by_reference_share(experiment: ModuleType) -> None:
    shares = {"a": 0.75, "b": 0.25}
    assert experiment.weighted(shares, {"a": 0.2, "b": -0.4}) == pytest.approx(0.05)
