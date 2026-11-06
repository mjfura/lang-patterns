"""The class-purity experiment's measures: word tags, purity, best node per tag and the rank of the
chosen cut. Pure functions, exercised without spaCy or a corpus."""

import importlib.util
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy import sparse


@pytest.fixture(scope="module")
def cp() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments/class_purity/class_purity.py"
    assert path.exists(), "class-purity experiment has not been implemented"
    spec = importlib.util.spec_from_file_location("class_purity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_word_is_tagged_by_a_clear_majority_of_enough_occurrences(cp: ModuleType) -> None:
    assert cp.word_tag(Counter({"NOUN": 19, "VERB": 1})) == "NOUN"
    assert cp.word_tag(Counter({"NOUN": 18, "VERB": 2})) == "NOUN"  # 90% of 20
    assert cp.word_tag(Counter({"NOUN": 17, "VERB": 3})) is None  # 85%
    assert cp.word_tag(Counter({"DET": 40})) == "DET"
    assert cp.word_tag(Counter({"NOUN": 10})) is None


def test_purity_counts_each_class_majority(cp: ModuleType) -> None:
    classes = np.array([0, 0, 0, 1, 1])
    tags = np.array(["NOUN", "NOUN", "VERB", "VERB", "VERB"])
    assert cp.purity(classes, tags) == pytest.approx(4 / 5)
    assert cp.purity(np.arange(5), tags) == 1.0


def test_best_node_is_the_closest_subtree_to_a_tag(cp: ModuleType) -> None:
    # Leaves 0..3; nodes: {0,1}, {2,3}, {0,1,2,3}. Leaf 3 is not evaluated.
    membership = sparse.csr_matrix(
        np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=float)
    )
    leaves = np.array([0, 1, 2])
    tags = np.array(["NOUN", "NOUN", "VERB"])
    f1, node = cp.best_node(membership, leaves, tags, "NOUN")
    assert (f1, node) == (1.0, 0)
    f1, node = cp.best_node(membership, leaves, tags, "VERB")
    assert (f1, node) == (1.0, 1)


def test_the_rank_of_the_chosen_cut_counts_strictly_better_cuts(cp: ModuleType) -> None:
    by_k = {8: 0.2, 9: 0.5, 10: 0.4, 11: 0.5}
    assert cp.rank_of(by_k, 10) == 3
    assert cp.rank_of(by_k, 9) == 1
