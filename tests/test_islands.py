"""`analysis.markov_chain.island_decomposition`: at every threshold the red curve sweeps, every
node of the full graph is either on the mainland (the largest component of the edges that reach
the threshold), on an island (any other component of 2+ nodes) or isolated — see
`docs/2026-09-27-islands-design.md`."""

import networkx as nx
import pytest

from corpus_generalization.analysis.markov_chain import (
    IslandLevel,
    build_transition_graph,
    island_decomposition,
    island_members,
    red_cluster_coverage_curve,
)
from tests.test_markov_chain_analysis import WordIdTokenizer

Node = tuple[int, ...]


def _graph(edges: list[tuple[int, int, int]]) -> "nx.DiGraph[Node]":
    graph: nx.DiGraph[Node] = nx.DiGraph()
    for source, target, count in edges:
        graph.add_edge((source,), (target,), count=count)
    return graph


def _cycle(nodes: tuple[int, ...], count: int) -> list[tuple[int, int, int]]:
    return [(source, nodes[(i + 1) % len(nodes)], count) for i, source in enumerate(nodes)]


def test_two_triangles_and_a_weak_node_split_exactly_at_every_threshold() -> None:
    """Triangle 0-1-2 seen 3 times per edge, triangle 3-4-5 seen twice, node 6 linked once."""
    graph = _graph([*_cycle((0, 1, 2), 3), *_cycle((3, 4, 5), 2), (6, 0, 1)])

    levels = island_decomposition(graph)

    assert levels == (
        IslandLevel(1.0, 4 / 7, 3 / 7, 0.0, island_count=1, largest_island=3),
        IslandLevel(2.0, 3 / 7, 3 / 7, 1 / 7, island_count=1, largest_island=3),
        IslandLevel(3.0, 3 / 7, 0.0, 4 / 7, island_count=0, largest_island=0),
    )


@pytest.mark.parametrize(
    "documents",
    [
        ("a b c a b c a b c", "d e", "f g", "h i", "j k"),
        ("a b a b a c", "b a b a", "c a c a", "x y x y", "p q"),
        ("the cat sat", "the cat ran", "a dog sat", "a dog ran", "the dog", "z z z"),
    ],
)
def test_mainland_is_the_red_curve_and_the_three_classes_sum_to_one(
    documents: tuple[str, ...],
) -> None:
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1, include_boundaries=True)

    levels = island_decomposition(graph)
    red = red_cluster_coverage_curve(graph)

    assert tuple(level.threshold for level in levels) == red.thresholds
    assert tuple(level.mainland for level in levels) == pytest.approx(red.coverages)
    for level in levels:
        assert level.mainland + level.islands + level.isolated == pytest.approx(1.0)


def test_a_tie_for_largest_gives_one_mainland_and_one_island() -> None:
    graph = _graph([(0, 1, 1), (2, 3, 1)])

    (level,) = island_decomposition(graph)

    assert level == IslandLevel(1.0, 0.5, 0.5, 0.0, island_count=1, largest_island=2)


def test_a_node_whose_only_surviving_edge_is_a_self_loop_is_isolated() -> None:
    graph = _graph([(0, 1, 2), (2, 2, 2), (3, 0, 1)])

    levels = island_decomposition(graph)

    assert levels[-1] == IslandLevel(2.0, 2 / 4, 0.0, 2 / 4, island_count=0, largest_island=0)


def test_a_lone_self_loop_is_the_mainland_as_the_red_curve_counts_it() -> None:
    graph = _graph([(0, 0, 1)])

    (level,) = island_decomposition(graph)

    assert level == IslandLevel(1.0, 1.0, 0.0, 0.0, island_count=0, largest_island=0)


def test_a_graph_with_no_edges_has_no_levels() -> None:
    graph: nx.DiGraph[Node] = nx.DiGraph()
    graph.add_node((0,))

    assert island_decomposition(graph) == ()


def test_island_members_are_every_non_mainland_component_largest_first() -> None:
    graph = _graph(
        [
            *_cycle((0, 1, 2), 2),
            (2, 3, 2),  # mainland: 0 1 2 3
            (4, 5, 2),
            (5, 6, 2),  # island: 4 5 6
            (7, 8, 3),  # island: 7 8
            (9, 0, 1),  # below threshold: isolated
            (10, 10, 2),  # self-loop only: isolated
        ]
    )

    members = island_members(graph, threshold=2)

    assert members == (((4,), (5,), (6,)), ((7,), (8,)))


def test_island_members_are_empty_when_nothing_reaches_the_threshold() -> None:
    graph = _graph([(0, 1, 1)])

    assert island_members(graph, threshold=2) == ()
