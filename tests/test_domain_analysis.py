"""`analysis.domains`: one domain or several, from a static encoder's geometry.

The embeddings here are built by hand rather than by a real encoder, so what is asserted is the
grouping and the verdict, with no download and nothing a model's opinion could move.
"""

import numpy as np
from numpy.typing import NDArray

from corpus_generalization.analysis.domains import analysis_units, domain_structure

TURN = (
    "<|im_start|>user\nWhat is a circle?\n<|im_end|>\n"
    "<|im_start|>assistant\nIt is round.\n<|im_end|>"
)


def unit_rows(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    result: NDArray[np.float64] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return result


def labels(count: int) -> list[str]:
    return [f"unit {index}" for index in range(count)]


def test_turn_markers_and_roles_are_stripped_before_a_document_is_encoded() -> None:
    assert analysis_units([TURN], "document") == ["What is a circle? It is round."]


def test_sentences_split_a_document_into_units() -> None:
    assert analysis_units([TURN], "sentence") == ["What is a circle?", "It is round."]


def test_two_separated_clouds_read_as_two_domains() -> None:
    rng = np.random.default_rng(0)
    shapes = rng.normal([1.0, 0.0, 0.0], 0.05, size=(8, 3))
    animals = rng.normal([0.0, 1.0, 0.0], 0.05, size=(8, 3))
    units = labels(16)

    structure = domain_structure(units, unit_rows(np.vstack([shapes, animals])), max_clusters=4)

    assert structure.domains == 2
    assert len(set(structure.labels[:8])) == 1
    assert structure.labels[0] != structure.labels[8]
    assert sorted(domain.size for domain in structure.groups) == [8, 8]
    assert {domain.representative for domain in structure.groups} & set(units[:8])
    assert {domain.representative for domain in structure.groups} & set(units[8:])
    assert structure.centroid_distances[0][1] > 0.5
    assert not structure.weak


def test_one_cloud_around_a_shared_direction_reads_as_one_domain() -> None:
    """Texts of one domain share a direction and scatter around it — which is why random directions
    in many dimensions would not do here: their halves' centroids come out nearly orthogonal."""
    rng = np.random.default_rng(0)
    shared = np.zeros(50)
    shared[0] = 1.0
    units = labels(30)

    structure = domain_structure(units, unit_rows(shared + 0.3 * rng.normal(size=(30, 50))))

    assert structure.domains == 1
    assert structure.separations[2] < structure.threshold
    assert structure.groups[0].size == 30


def test_tight_groups_close_together_are_one_domain_however_high_their_silhouette() -> None:
    """The decision the calibration made: two tidy groups a short angle apart score a high
    Silhouette, and are still one domain, because it is the distance between them that decides."""
    rng = np.random.default_rng(0)
    left = rng.normal([1.0, 0.3, 0.0], 0.01, size=(8, 3))
    right = rng.normal([1.0, -0.3, 0.0], 0.01, size=(8, 3))

    structure = domain_structure(labels(16), unit_rows(np.vstack([left, right])), max_clusters=3)

    assert structure.silhouettes[2] > 0.5
    assert structure.domains == 1


def test_too_few_units_cannot_be_grouped() -> None:
    structure = domain_structure(labels(2), np.eye(2))

    assert structure.domains == 0
    assert structure.silhouettes == {}


def test_fewer_than_ten_units_are_undetermined_however_clearly_they_part() -> None:
    """Six units in two exact directions would score a perfect 1.0 at k=2, and it would still be
    three units per domain: too few to call anything a domain."""
    embeddings = unit_rows(np.array([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3))

    structure = domain_structure(labels(6), embeddings, max_clusters=3)

    assert structure.domains == 0


def test_k_is_never_more_than_one_group_per_five_units() -> None:
    """Twelve points scattered in many directions: uncapped, the search could split them into up
    to eleven groups and score the finest split best; capped, it tries k=2 and nothing else."""
    rng = np.random.default_rng(0)

    structure = domain_structure(labels(12), unit_rows(rng.normal(size=(12, 50))), max_clusters=5)

    assert list(structure.silhouettes) == [2]


def test_a_domain_found_from_fewer_than_five_units_is_marked_weak() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal([1.0, 0.0, 0.0], 0.05, size=(10, 3))
    rare = rng.normal([0.0, 1.0, 0.0], 0.05, size=(2, 3))

    structure = domain_structure(labels(12), unit_rows(np.vstack([common, rare])))

    assert structure.domains == 2
    assert sorted(domain.size for domain in structure.groups) == [2, 10]
    assert structure.weak
