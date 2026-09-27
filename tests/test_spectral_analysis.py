"""`analysis.spectral`: Vendi Score and algebraic connectivity, each buildable from the
project's own tokenizer alone or from a general sentence encoder -- verified here with hand-built
vectors and a tiny word tokenizer, no download and nothing an encoder's opinion could move.
"""

from collections.abc import Sequence

import numpy as np
import pytest

from corpus_generalization.analysis.spectral import (
    Matrix,
    algebraic_connectivity,
    connected_components_count,
    diversity_report,
    lexical_similarity_matrix,
    semantic_similarity_matrix,
    similarity_adjacency,
    transition_adjacency_matrix,
    vendi_score,
)


class WordIdTokenizer:
    """One id per distinct whitespace-separated word, assigned in first-seen order. `bos_id`/
    `eos_id` are negative so they never collide with a real word's id, which always starts at 0."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    @property
    def bos_id(self) -> int:
        return -1

    @property
    def eos_id(self) -> int:
        return -2

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        return [self._ids.setdefault(word, len(self._ids)) for word in text.split()]


class HandBuiltEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts: Sequence[str]) -> Matrix:
        return np.array([self._vectors[text] for text in texts])


def test_vendi_score_of_n_identical_items_is_one() -> None:
    """Every item collapses to the same one: there is exactly one effective item, however many
    copies of it there are."""
    similarity = np.ones((5, 5))

    assert vendi_score(similarity) == pytest.approx(1.0, abs=1e-6)


def test_vendi_score_of_n_orthogonal_items_is_n() -> None:
    """Nothing overlaps with anything else: n items, n effective items."""
    similarity = np.eye(4)

    assert vendi_score(similarity) == pytest.approx(4.0, abs=1e-6)


def test_vendi_score_of_two_identical_pairs_is_between_one_and_the_item_count() -> None:
    """Two clusters of two identical items each: more effective variety than a single cluster,
    less than four fully independent items."""
    similarity = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    score = vendi_score(similarity)

    assert 1.0 < score < 4.0
    assert score == pytest.approx(2.0, abs=1e-6)


def test_vendi_score_of_an_empty_matrix_is_zero() -> None:
    assert vendi_score(np.zeros((0, 0))) == 0.0


def test_algebraic_connectivity_is_zero_for_two_disconnected_components() -> None:
    """Two separate edges, node 0-1 and node 2-3, sharing no connection at all."""
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    assert algebraic_connectivity(adjacency) == pytest.approx(0.0, abs=1e-9)


def test_algebraic_connectivity_of_a_complete_graph_equals_its_node_count() -> None:
    """The textbook closed form for K_n's Laplacian spectrum: 0 once, n exactly n-1 times."""
    n = 5
    adjacency = np.ones((n, n)) - np.eye(n)

    assert algebraic_connectivity(adjacency) == pytest.approx(float(n), abs=1e-9)


def test_algebraic_connectivity_is_zero_below_two_nodes() -> None:
    assert algebraic_connectivity(np.zeros((0, 0))) == 0.0
    assert algebraic_connectivity(np.zeros((1, 1))) == 0.0


def test_connected_components_count_matches_the_number_of_disconnected_pieces() -> None:
    """The exact, discrete complement to `algebraic_connectivity`'s continuous reading: three
    mutually disconnected pairs of nodes should count as exactly three components, not just
    report a Fiedler value near 0."""
    one_piece = np.ones((4, 4)) - np.eye(4)
    two_pieces = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    three_pieces = np.zeros((6, 6))
    three_pieces[0, 1] = three_pieces[1, 0] = 1.0
    three_pieces[2, 3] = three_pieces[3, 2] = 1.0
    three_pieces[4, 5] = three_pieces[5, 4] = 1.0

    assert connected_components_count(one_piece) == 1
    assert connected_components_count(two_pieces) == 2
    assert connected_components_count(three_pieces) == 3


def test_connected_components_count_is_zero_for_an_empty_matrix() -> None:
    assert connected_components_count(np.zeros((0, 0))) == 0


def test_lexical_similarity_matrix_gives_identical_documents_similarity_one() -> None:
    matrix = lexical_similarity_matrix(("a b c", "a b c"), WordIdTokenizer())

    assert matrix[0, 1] == pytest.approx(1.0)


def test_lexical_similarity_matrix_gives_disjoint_vocabulary_similarity_zero() -> None:
    matrix = lexical_similarity_matrix(("a b", "c d"), WordIdTokenizer())

    assert matrix[0, 1] == pytest.approx(0.0)


def test_semantic_similarity_matrix_reads_cosine_similarity_off_the_encoder() -> None:
    encoder = HandBuiltEncoder({"x": [1.0, 0.0], "y": [0.0, 1.0], "z": [1.0, 0.0]})

    matrix = semantic_similarity_matrix(("x", "y", "z"), encoder)

    assert matrix[0, 1] == pytest.approx(0.0)
    assert matrix[0, 2] == pytest.approx(1.0)


def test_transition_adjacency_matrix_is_symmetric_and_weighted_by_transition_count() -> None:
    documents = ("a b", "a b", "b c")

    nodes, adjacency = transition_adjacency_matrix(documents, WordIdTokenizer(), order=1)

    tokenizer = WordIdTokenizer()
    a, b, _c = (tokenizer.encode(word)[0] for word in ("a", "b", "c"))
    index = {node: position for position, node in enumerate(nodes)}
    # "a" -> "b" observed twice, symmetrized onto both (a, b) and (b, a).
    assert adjacency[index[(a,)], index[(b,)]] == pytest.approx(2.0)
    assert adjacency[index[(b,)], index[(a,)]] == pytest.approx(2.0)
    assert np.allclose(adjacency, adjacency.T)


def test_similarity_adjacency_zeroes_the_diagonal_and_clips_negative_similarity() -> None:
    similarity = np.array([[1.0, -0.5], [-0.5, 1.0]])

    adjacency = similarity_adjacency(similarity)

    assert np.allclose(np.diag(adjacency), 0.0)
    assert adjacency[0, 1] == 0.0


def test_diversity_report_reads_lexical_and_semantic_readings_side_by_side() -> None:
    units = ("a b", "a b", "c d")
    # Same two clusters the lexical similarity would already find on its own -- a case where the
    # two readings should agree, not a case designed to make them diverge.
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    report = diversity_report(units, WordIdTokenizer(), embeddings)

    assert report.units == 3
    assert report.vendi_lexical == pytest.approx(report.vendi_semantic, abs=1e-6)
    assert report.algebraic_connectivity_lexical == pytest.approx(0.0, abs=1e-9)
    assert report.algebraic_connectivity_semantic == pytest.approx(0.0, abs=1e-9)
    # Two documents share no vocabulary at all with the third: 2 components, both readings.
    assert report.components_lexical == 2
    assert report.components_semantic == 2


def test_diversity_report_reads_zero_semantic_scores_when_handed_no_embeddings() -> None:
    """What a corpus too small to group at all is handed by `domains.py`'s own pipeline."""
    units = ("a b", "c d")

    report = diversity_report(units, WordIdTokenizer(), np.zeros((2, 0)))

    assert report.vendi_semantic == 0.0
    assert report.algebraic_connectivity_semantic == 0.0
    assert report.components_semantic == 0
    assert report.vendi_lexical > 0.0
