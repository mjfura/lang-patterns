"""`analysis.markov_chain`: a corpus's token transitions as a Markov chain, position-blind
(a transition counts wherever it occurs, not only at the start of a document — the fix for the
"I am" example: it counts the same whether it opens a sentence or sits in the middle of one).
"""

import math
from typing import cast

import networkx as nx
import pytest

from corpus_generalization.analysis.markov_chain import (
    CoverageCurve,
    build_transition_graph,
    coverage_curve_percentiles,
    document_cluster_coverage_curve,
    document_core_coverage_curve,
    document_reciprocal_cluster_coverage_curve,
    document_reciprocal_core_coverage_curve,
    gini_coefficient,
    markov_chain,
    markov_chain_sweep,
    reciprocal_cluster_coverage,
    reciprocal_cluster_coverage_curve,
    reciprocal_core_coverage_curve,
    reciprocal_core_numbers,
    reciprocal_edge_weights,
    red_cluster_coverage,
    red_cluster_coverage_curve,
    to_log_curve,
    weighted_core_coverage_curve,
    weighted_core_numbers,
    weighted_core_samples,
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


def test_a_transition_counts_wherever_it_occurs_not_only_at_a_document_start() -> None:
    """ "I am" opens D1 but sits mid-sentence in D2 — order 1 should still treat them as the same
    transition, unlike a start-anchored walk."""
    documents = ("I am happy", "she said I am ready")

    stats = markov_chain(documents, WordIdTokenizer(), order=1)

    # "I" -> "am" observed twice (D1 at position 0, D2 at position 2): a real, well-supported state
    assert stats.well_supported_fraction > 0.0
    assert stats.states > 0


def test_states_and_transitions_are_counted_correctly() -> None:
    documents = ("a b c", "a b d")
    tokenizer = WordIdTokenizer()

    stats = markov_chain(documents, tokenizer, order=1)

    # states with an outgoing edge: a, b (both source "a"->"b" and "b"->{c,d})
    # "c" and "d" are sinks (no outgoing edge) but still nodes
    assert stats.states == 4  # a, b, c, d
    assert stats.transitions == 3  # a->b, b->c, b->d


def test_well_supported_fraction_flags_states_seen_only_once() -> None:
    documents = ("a b c", "a b d", "x y")
    tokenizer = WordIdTokenizer()

    stats = markov_chain(documents, tokenizer, order=1)

    # source states: a (->b, seen twice: well supported), b (->c,->d, seen twice total: well
    # supported), x (->y, seen once: NOT well supported)
    assert stats.well_supported_fraction == pytest.approx(2 / 3)


def test_disjoint_vocabularies_show_up_as_separate_weakly_connected_components() -> None:
    """Two documents that share no tokens at all: the transition graph should fracture into (at
    least) two pieces, even though strict strong connectivity would never hold for any real
    corpus (a sentence's last token is always a dead end)."""
    documents = ("a b c", "x y z")

    stats = markov_chain(documents, WordIdTokenizer(), order=1)

    assert stats.components >= 2
    assert stats.largest_component_fraction < 1.0


def test_a_single_cohesive_vocabulary_is_one_weakly_connected_component() -> None:
    documents = ("a b c", "a b d", "b c a")

    stats = markov_chain(documents, WordIdTokenizer(), order=1)

    assert stats.components == 1
    assert stats.largest_component_fraction == pytest.approx(1.0)


def test_entropy_matches_a_pagerank_weighted_recomputation() -> None:
    """Not hand-solving PageRank's fixed point by hand (its damping factor makes that impractical)
    — instead, recomputing the same weighted-entropy formula independently over the same graph, to
    check the wiring rather than reimplementing the algorithm."""
    documents = ("a b c", "a b d", "a b c", "b c a")
    tokenizer = WordIdTokenizer()

    stats = markov_chain(documents, tokenizer, order=1)

    graph: nx.DiGraph[tuple[int, ...]] = nx.DiGraph()
    sequences = [tokenizer.encode(d) for d in documents]
    from collections import Counter, defaultdict

    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for seq in sequences:
        for i in range(len(seq) - 1):
            counts[(seq[i],)][seq[i + 1]] += 1
    for state, next_counts in counts.items():
        total = sum(next_counts.values())
        for token, count in next_counts.items():
            graph.add_edge(state, (*state[1:], token), weight=count / total, count=count)

    pi = nx.pagerank(graph, weight="weight")
    expected = 0.0
    for state in counts:
        weights = [c / sum(counts[state].values()) for c in counts[state].values()]
        h = -sum(w * math.log2(w) for w in weights if w > 0)
        expected += pi.get(state, 0.0) * h

    assert stats.entropy_bits == pytest.approx(expected)
    assert stats.perplexity == pytest.approx(2**expected)


def test_an_empty_corpus_of_sequences_too_short_for_the_order_gives_no_states() -> None:
    stats = markov_chain(("a",), WordIdTokenizer(), order=1)

    assert stats.states == 0
    assert stats.entropy_bits == 0.0
    assert stats.perplexity == 1.0
    assert stats.weakly_connected is True


def test_sweep_returns_one_result_per_order_in_order() -> None:
    documents = ("a b c d", "a b c e")

    results = markov_chain_sweep(documents, WordIdTokenizer(), orders=(1, 2))

    assert [r.order for r in results] == [1, 2]
    assert results[1].states <= results[0].states * results[0].states


def test_gini_coefficient_is_zero_when_every_value_is_equal() -> None:
    assert gini_coefficient([4, 4, 4, 4]) == pytest.approx(0.0)


def test_gini_coefficient_matches_a_hand_worked_case() -> None:
    # verified by hand: mean-absolute-difference form gives 48 / (2 * 16 * 3) = 0.5
    assert gini_coefficient([1, 1, 1, 9]) == pytest.approx(0.5)


def test_gini_coefficient_of_no_values_is_zero() -> None:
    assert gini_coefficient([]) == 0.0


def test_markov_chain_reports_a_gini_coefficient_over_transition_counts() -> None:
    # "a" always leads to "b" (count 3); "x" leads to "y" or "z" once each: unequal frequencies
    documents = ("a b", "a b", "a b", "x y", "x z")

    stats = markov_chain(documents, WordIdTokenizer(), order=1)

    assert stats.gini_coefficient > 0.0


def test_red_cluster_coverage_is_low_when_the_well_supported_part_is_a_small_pocket() -> None:
    """A well-supported triangle (a<->b<->c, each transition seen many times) plus a long tail of
    once-seen edges elsewhere: the well-supported neighborhood should cover only a fraction of the
    whole graph's nodes."""
    documents = (
        "a b c a b c a b c",  # a->b, b->c, c->a each seen 3 times
        "d e",
        "f g",
        "h i",
        "j k",
    )
    tokenizer = WordIdTokenizer()
    graph = build_transition_graph(documents, tokenizer, order=1)

    coverage = red_cluster_coverage(graph, threshold=2)

    assert 0.0 < coverage < 0.5


def test_red_cluster_coverage_is_high_when_the_whole_graph_is_well_supported() -> None:
    documents = ("a b c a b c", "b c a b c a")
    tokenizer = WordIdTokenizer()
    graph = build_transition_graph(documents, tokenizer, order=1)

    coverage = red_cluster_coverage(graph, threshold=2)

    assert coverage == pytest.approx(1.0)


def test_red_cluster_coverage_is_zero_when_no_edge_meets_the_threshold() -> None:
    documents = ("a b", "c d")
    tokenizer = WordIdTokenizer()
    graph = build_transition_graph(documents, tokenizer, order=1)

    coverage = red_cluster_coverage(graph, threshold=2)

    assert coverage == 0.0


def test_markov_chain_reports_red_cluster_coverage() -> None:
    documents = ("a b c a b c a b c", "d e", "f g")

    stats = markov_chain(documents, WordIdTokenizer(), order=1)

    assert 0.0 < stats.red_cluster_coverage < 1.0


def test_red_cluster_coverage_curve_matches_the_single_threshold_reading_at_every_point() -> None:
    documents = ("a b c a b c a b c", "d e", "f g", "h i", "j k")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    curve = red_cluster_coverage_curve(graph)

    assert curve.thresholds == (1.0, 2.0, 3.0)
    for threshold, coverage in zip(curve.thresholds, curve.coverages, strict=True):
        assert coverage == pytest.approx(red_cluster_coverage(graph, int(threshold)))


def test_reciprocal_cluster_coverage_curve_matches_single_threshold_reading() -> None:
    documents = ("a b a b a c", "b a b a", "c a c a")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    curve = reciprocal_cluster_coverage_curve(graph)

    expected = tuple(
        reciprocal_cluster_coverage(graph, threshold) for threshold in curve.thresholds
    )
    assert curve.coverages == pytest.approx(expected)


def test_document_cluster_curves_keep_float_thresholds_after_sweep() -> None:
    documents = ("a b a b", "b a c a", "c a c a")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    curve = document_reciprocal_cluster_coverage_curve(graph)
    expected = tuple(
        reciprocal_cluster_coverage(graph, threshold, "effective_document_support")
        for threshold in curve.thresholds
    )

    assert curve.coverages == pytest.approx(expected)


def test_red_cluster_coverage_curve_is_empty_for_a_graph_with_no_edges() -> None:
    graph = build_transition_graph(("a",), WordIdTokenizer(), order=1)

    curve = red_cluster_coverage_curve(graph)

    assert curve.thresholds == ()
    assert curve.coverages == ()
    assert curve.auc == 0.0


def test_red_cluster_coverage_curve_auc_is_one_for_a_uniformly_well_supported_graph() -> None:
    """A single edge, repeated: coverage is 1.0 at every threshold up to its own count, so the
    curve is flat and its area under a [0, 1]-normalized threshold axis is exactly 1.0."""
    documents = ("a b",) * 5
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    curve = red_cluster_coverage_curve(graph)

    assert curve.auc == pytest.approx(1.0)


def test_red_cluster_coverage_curve_auc_is_low_when_the_well_supported_part_is_a_small_pocket() -> (
    None
):
    documents = ("a b c a b c a b c", "d e", "f g", "h i", "j k")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    curve = red_cluster_coverage_curve(graph)

    assert curve.auc < 0.5


def test_weighted_core_numbers_of_a_uniform_pair_is_the_shared_edge_weight() -> None:
    """Two nodes, one edge repeated 5 times: both peel out together at the same strength, since
    removing either first still leaves the other holding exactly the weight of the edge between
    them -- the running-floor rule assigns both the same core number."""
    documents = ("a b",) * 5
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    core = weighted_core_numbers(graph)

    assert set(core.values()) == {5.0}


def test_weighted_core_numbers_matches_a_hand_solved_symmetric_triangle() -> None:
    """ "a b c" walked three full times, wrapping back to "a" at the end (ten tokens, not nine):
    a->b, b->c and c->a each observed exactly 3 times. Every node's total incident weight (its own
    two edges, counted from both ends) is 2*3=6, and peeling a fully symmetric triangle always
    leaves the last node holding the first node's running floor -- all three end up at the same
    core number."""
    documents = ("a b c a b c a b c a",)
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    core = weighted_core_numbers(graph)

    assert set(core.values()) == {6.0}


def test_weighted_core_and_red_cluster_curves_can_disagree_at_their_lowest_level() -> None:
    """A well-supported triangle plus several once-seen, mutually disconnected pairs: red-cluster
    coverage already reads low at its lowest threshold, because it asks a *global* connectivity
    question and the graph is fragmented from the start; weighted-core coverage reads 1.0 at its
    lowest level, because it asks a *local* one -- every node has some strength of its own, even
    one that belongs to no one else's neighborhood. Exactly the disagreement the module docstring
    says the two curves can show.
    """
    documents = ("a b c a b c a b c", "d e", "f g", "h i", "j k")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    red_cluster = red_cluster_coverage_curve(graph)
    weighted_core = weighted_core_coverage_curve(graph)

    assert red_cluster.coverages[0] < 0.5
    assert weighted_core.coverages[0] == pytest.approx(1.0)
    assert weighted_core.auc > red_cluster.auc


def test_weighted_core_coverage_curve_is_empty_for_a_graph_with_no_edges() -> None:
    graph = build_transition_graph(("a",), WordIdTokenizer(), order=1)

    curve = weighted_core_coverage_curve(graph)

    assert curve.thresholds == ()
    assert curve.auc == 0.0


def test_reciprocal_edge_weights_is_the_harmonic_mean_of_both_directions() -> None:
    # a->b seen 3 times, b->a seen 2 times: genuine but unequal reciprocity.
    documents = ("a b a b a b",)
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    weights = reciprocal_edge_weights(graph)

    (pair, weight) = next(iter(weights.items()))
    assert set(pair) == {(0,), (1,)}
    assert weight == pytest.approx(2 * 3 * 2 / (3 + 2))  # 2.4


def test_reciprocal_edge_weights_is_zero_for_a_purely_one_directional_transition() -> None:
    """c->d occurs, d->c never does: no matter how often c->d repeats, one direction alone is
    not reciprocity."""
    documents = ("c d",) * 10
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    weights = reciprocal_edge_weights(graph)

    assert set(weights.values()) == {0.0}


def test_reciprocal_edge_weights_excludes_self_loops() -> None:
    """A token that immediately repeats itself (e.g. real prose: "very very excited") produces a
    self-loop edge. As an unordered pair it has no "other side" to reciprocate with, so it must be
    dropped rather than crash the frozenset-pair unpacking every other reader of this dict relies
    on."""
    documents = ("a a b",)
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    weights = reciprocal_edge_weights(graph)

    assert all(len(pair) == 2 for pair in weights)
    assert reciprocal_cluster_coverage_curve(graph) == CoverageCurve((), (), 0.0)


def test_reciprocal_cluster_coverage_curve_is_empty_for_a_purely_sequential_corpus() -> None:
    """ "a b c": every transition runs one way only and nothing loops back. No pair is ever
    reciprocal, so reciprocal-cluster has nothing to sweep; and peeling removes a (no input) and c
    (no output), which leaves b with neither, so reciprocal-core is empty too -- the expected
    reading for a template sentence's own grammar, not a bug to work around."""
    graph = build_transition_graph(("a b c",), WordIdTokenizer(), order=1)

    assert reciprocal_cluster_coverage_curve(graph).thresholds == ()
    assert reciprocal_core_coverage_curve(graph).thresholds == ()


def test_reciprocal_core_counts_a_cycle_with_no_reciprocal_pair() -> None:
    """a->b->c->a: no transition is ever observed backwards, so reciprocal-cluster (pairwise
    reciprocity) is empty -- yet every node keeps both an input and an output from the others, so
    reciprocal-core covers all of them. reciprocal-core measures flow through cycles, not A<->B."""
    graph = build_transition_graph(("a b c a b c a",), WordIdTokenizer(), order=1)

    assert reciprocal_cluster_coverage_curve(graph).thresholds == ()
    assert reciprocal_core_coverage_curve(graph).coverages == pytest.approx((1.0,))


def test_reciprocal_cluster_coverage_ignores_a_one_directional_bridge() -> None:
    """a<->b genuinely reciprocal (seen both ways); c->d only ever seen forwards. The reciprocal
    red-cluster reading should cover only the a/b pair, never crediting c/d for its one-way
    count."""
    documents = ("a b a b a b", "c d")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    coverage = reciprocal_cluster_coverage(graph, threshold=0.1)

    assert coverage == pytest.approx(0.5)  # 2 of 4 nodes: a and b, not c and d


def test_reciprocal_core_numbers_scores_a_one_directional_node_at_zero() -> None:
    documents = ("a b a b a b", "c d")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    core = reciprocal_core_numbers(graph)

    tokenizer = WordIdTokenizer()
    a, b, c, d = (tokenizer.encode(word)[0] for word in ("a", "b", "c", "d"))
    assert core[(a,)] == pytest.approx(2.4)
    assert core[(b,)] == pytest.approx(2.4)
    assert core[(c,)] == 0.0
    assert core[(d,)] == 0.0


def test_reciprocal_core_coverage_curve_is_empty_for_a_graph_with_no_edges() -> None:
    graph = build_transition_graph(("a",), WordIdTokenizer(), order=1)

    curve = reciprocal_core_coverage_curve(graph)

    assert curve.thresholds == ()
    assert curve.auc == 0.0


def test_build_transition_graph_tracks_document_support_separately_from_count() -> None:
    """ "a b" repeats 3 times, but all inside ONE document -- document_support stays 1 no matter
    how high count climbs, and effective_document_support is 1.0 too (entropy of a single outcome
    is 0). "c d" occurs only once per document, but in 3 SEPARATE documents -- same final count,
    document_support 3, and effective_document_support also 3.0 since the split across those 3
    documents (1 each) is perfectly even."""
    documents = ("a b a b a b", "c d", "c d", "c d")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    a_to_b = graph[(0,)][(1,)]
    c_to_d = graph[(2,)][(3,)]
    assert a_to_b["count"] == 3
    assert a_to_b["document_support"] == 1
    assert a_to_b["effective_document_support"] == pytest.approx(1.0)
    assert c_to_d["count"] == 3
    assert c_to_d["document_support"] == 3
    assert c_to_d["effective_document_support"] == pytest.approx(3.0)


def test_effective_document_support_is_discounted_by_an_uneven_split() -> None:
    """20 occurrences split 15/5 between two documents reads document_support=2 -- identical to 2
    occurrences split 1/1 between the same two documents -- even though the first is far more
    concentrated. effective_document_support tells them apart: strictly between 1 (all in one
    document) and 2 (a perfectly even split) for the uneven case, exactly 2 for the even one."""
    uneven_documents = (" ".join(["e f"] * 15), " ".join(["e f"] * 5))
    even_documents = ("g h", "g h")
    uneven_graph = build_transition_graph(uneven_documents, WordIdTokenizer(), order=1)
    even_graph = build_transition_graph(even_documents, WordIdTokenizer(), order=1)

    e_to_f = uneven_graph[(0,)][(1,)]
    g_to_h = even_graph[(0,)][(1,)]
    assert e_to_f["count"] == 20
    assert e_to_f["document_support"] == 2
    assert e_to_f["effective_document_support"] == pytest.approx(1.7547653506033232)
    assert g_to_h["count"] == 2
    assert g_to_h["document_support"] == 2
    assert g_to_h["effective_document_support"] == pytest.approx(2.0)


def test_document_cluster_coverage_curve_discounts_a_single_documents_own_repetition() -> None:
    """A 4-node cycle (p->q->r->s->p) repeated 5 times, but entirely within ONE document, versus a
    weaker pair (m->n) attested once each in 5 SEPARATE documents. Raw count favors the cycle
    (count 4-5 per edge) over the pair (count 5 total, about the same) and red-cluster coverage
    stays flat at its starting value all the way to the highest threshold, since both survive it.
    Once weighted by effective_document_support instead, the cycle's edges are all concentrated in
    one document (effective_document_support=1.0, the minimum possible) while the pair's is spread
    perfectly evenly across its 5 documents (effective_document_support=5.0, matching its raw
    document_support exactly) -- document-cluster's AUC comes out lower than red-cluster's on the
    identical graph, and its curve has only two levels (1.0 and 5.0) instead of red's, because
    those are the only two effective_document_support values any edge actually reaches."""
    chain_document = " ".join(["p q r s"] * 5)
    documents = (chain_document, "m n", "m n", "m n", "m n", "m n")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    red = red_cluster_coverage_curve(graph)
    document = document_cluster_coverage_curve(graph)

    assert red.coverages == pytest.approx((2 / 3, 2 / 3))
    assert red.auc == pytest.approx(2 / 3)
    assert document.thresholds == pytest.approx((1.0, 5.0))
    assert document.coverages == pytest.approx((2 / 3, 1 / 3))
    assert document.auc == pytest.approx(0.4)
    assert document.auc < red.auc


def test_document_core_coverage_curve_discounts_the_same_repetition_locally() -> None:
    """The local (per-node) reading of the same graph and the same story: weighted-core still
    credits the single-document cycle's nodes at a high level (9, the two-edge total of count on
    either side), while document-core caps every one of that cycle's nodes at level 2 (each node's
    own two edges both concentrated at effective_document_support=1.0), letting the well-distributed
    pair's nodes (level ~5, their edge's effective_document_support) matter proportionally more."""
    chain_document = " ".join(["p q r s"] * 5)
    documents = (chain_document, "m n", "m n", "m n", "m n", "m n")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    weighted_core = weighted_core_coverage_curve(graph)
    document_core = document_core_coverage_curve(graph)

    assert weighted_core.thresholds == (5.0, 9.0)
    assert document_core.thresholds == pytest.approx((2.0, 5.0))
    assert document_core.auc < weighted_core.auc


def test_document_reciprocal_curves_ask_for_both_directions_each_widely_attested() -> None:
    """Neither the cycle nor the pair above has a genuinely reciprocal EDGE (every step is one
    directional token to the next, never mirrored back the same way at the pair level), so
    document-reciprocal-cluster is empty exactly like its count-based counterpart would be for a
    purely sequential graph. document-reciprocal-core still finds something, though: every node of
    the 4-cycle has both an incoming and an outgoing edge (from its two neighbors in the cycle), so
    it scores above zero there, while the pair's two nodes -- m only ever a source, n only ever a
    target -- score exactly 0 and are excluded."""
    chain_document = " ".join(["p q r s"] * 5)
    documents = (chain_document, "m n", "m n", "m n", "m n", "m n")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    cluster = document_reciprocal_cluster_coverage_curve(graph)
    core = document_reciprocal_core_coverage_curve(graph)

    assert cluster.thresholds == ()
    assert core.thresholds == (1.0,)
    assert core.coverages == pytest.approx((4 / 6,))


def test_include_boundaries_frames_every_document_with_bos_and_eos() -> None:
    """`<bos>` opens every document and `<eos>` closes it, exactly like
    `packing.framed_documents` frames a document before pre-training packing -- off
    by default, so every existing caller is unaffected."""
    tokenizer = WordIdTokenizer()
    documents = ("a b c", "a b d")

    without = build_transition_graph(documents, tokenizer, order=1)
    with_boundaries = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)

    assert (tokenizer.bos_id,) not in without.nodes
    assert (tokenizer.bos_id,) in with_boundaries.nodes
    assert (tokenizer.eos_id,) in with_boundaries.nodes
    assert with_boundaries[(tokenizer.bos_id,)][(0,)]["count"] == 2  # both documents start "a"


def test_bos_out_degree_totals_the_corpus_document_count() -> None:
    """`<bos>` occurs exactly once per document, so summed over every edge leaving it, `count`
    equals the corpus's own document count -- a fixed ceiling every other edge's intensity can be
    read against, independent of vocabulary or repetition. Symmetrically for `<eos>`'s in-degree."""
    tokenizer = WordIdTokenizer()
    documents = ("a b c", "a b d", "x y")

    graph = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)

    bos_out_total = sum(
        data["count"] for _, _, data in graph.out_edges((tokenizer.bos_id,), data=True)
    )
    eos_in_total = sum(
        data["count"] for _, _, data in graph.in_edges((tokenizer.eos_id,), data=True)
    )
    assert bos_out_total == len(documents)
    assert eos_in_total == len(documents)


def test_bos_edges_never_show_the_single_document_repetition_pattern() -> None:
    """`<bos>` cannot occur twice within the same document, so no edge leaving it can ever be the
    "one document repeated many times" case `document_support`/`effective_document_support` exist
    to catch -- `count`, `document_support` and `effective_document_support` are forced equal on
    every one of `<bos>`'s outgoing edges."""
    tokenizer = WordIdTokenizer()
    documents = ("a b", "a b", "a c")

    graph = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)

    for _, _, data in graph.out_edges((tokenizer.bos_id,), data=True):
        assert data["count"] == data["document_support"] == data["effective_document_support"]


def test_markov_chain_leaves_boundaries_out_so_disjoint_vocabularies_still_show_up_separately() -> (
    None
):
    """`markov_chain` never passes `include_boundaries=True` internally: `<bos>`/`<eos>` are the
    same two nodes in every document, so including them would bridge any set of documents into one
    weakly connected component even when they share no real vocabulary at all -- exactly the
    regression this guards against for `components`/`weakly_connected`/
    `largest_component_fraction`."""
    documents = ("a b c", "x y z")

    stats = markov_chain(documents, WordIdTokenizer(), order=1)

    assert stats.components == 2
    assert stats.largest_component_fraction < 1.0


def test_to_log_curve_keeps_the_thresholds_and_coverages_unchanged() -> None:
    """`to_log_curve` recomputes only `auc` -- the actual measured points are the graph analysis's
    own answer and must survive untouched, or the log picture and the linear one would silently
    disagree about what was measured, not just about how the axis reads it."""
    curve = CoverageCurve(thresholds=(1.0, 10.0, 100.0), coverages=(1.0, 0.5, 0.1), auc=0.145)

    log_curve = to_log_curve(curve)

    assert log_curve.thresholds == curve.thresholds
    assert log_curve.coverages == curve.coverages


def test_to_log_curve_weighs_low_thresholds_more_heavily_than_the_linear_auc() -> None:
    """A curve whose highest threshold is a wide outlier (100 against 1 and 10) spends almost its
    entire linear-normalized width on the gap leading up to that outlier, so the linear `auc` ends
    up dominated by the LOWEST coverage value (0.1) even though the curve holds a much higher
    coverage (0.5) across most of its actually-observed thresholds. The log-normalized `auc` gives
    the 10x jump from 1 to 10 the same visual weight as the 10x jump from 10 to 100, so it is not
    swamped by the outlier the same way -- exactly the corpus-with-a-huge-outlier-count case the
    log-scale color and curve siblings in `analysis.markov_graph` exist for."""
    curve = CoverageCurve(thresholds=(1.0, 10.0, 100.0), coverages=(1.0, 0.5, 0.1), auc=0.145)

    log_curve = to_log_curve(curve)

    assert log_curve.auc == pytest.approx(0.3)
    assert log_curve.auc > curve.auc


def test_to_log_curve_falls_back_to_the_flat_value_when_every_threshold_is_at_most_one() -> None:
    """`maximum <= 1` has no ratio to take a log of -- `weighted_core_coverage_curve` can
    legitimately produce this (a node peeled at level exactly 0, alongside others at level 1) --
    the same degenerate case `_area_under_curve` itself falls back to `coverages[0]` for."""
    curve = CoverageCurve(thresholds=(0.0, 1.0), coverages=(1.0, 0.4), auc=0.4)

    log_curve = to_log_curve(curve)

    assert log_curve.auc == curve.coverages[0]


def test_to_log_curve_of_an_empty_curve_is_still_empty() -> None:
    curve = CoverageCurve((), (), 0.0)

    assert to_log_curve(curve) == CoverageCurve((), (), 0.0)


# The linear plot's x axis is `threshold / 100`: a target is a position on that axis, never a
# coverage -- 0.5 means threshold 50 here, although coverage there is 0.2, not 0.5.
AXIS_CURVE = CoverageCurve(
    thresholds=(1.0, 20.0, 50.0, 80.0, 100.0), coverages=(1.0, 0.6, 0.2, 0.1, 0.05), auc=0.0
)


def test_percentiles_are_positions_on_the_linear_threshold_axis() -> None:
    entries = coverage_curve_percentiles(AXIS_CURVE, targets=(1.0, 0.5, 0.3, 0.01))

    assert [entry["threshold"] for entry in entries] == [100.0, 50.0, 30.0, 1.0]


def test_a_percentile_reports_the_coverage_the_step_curve_holds_at_its_threshold() -> None:
    """Coverage at `thresholds[i]` holds down to just above `thresholds[i-1]`, so 30 -- between
    20 and 50 -- reads 50's coverage."""
    entries = coverage_curve_percentiles(AXIS_CURVE, targets=(1.0, 0.5, 0.3, 0.01))

    assert [entry["actual_coverage"] for entry in entries] == [0.05, 0.2, 0.2, 1.0]


def test_weighted_core_samples_sit_around_the_axis_threshold_and_list_their_token_ids() -> None:
    core: dict[tuple[int, ...], float] = {(token,): float(token) for token in range(1, 101)}
    curve = CoverageCurve(thresholds=(1.0, 100.0), coverages=(1.0, 0.01), auc=0.0)

    (sample,) = weighted_core_samples(core, curve, sample_fraction=0.05, targets=(0.5,))

    assert sample["threshold"] == 50.0
    assert sample["token_ids"] == [48, 49, 50, 51, 52]
    nodes = cast(list[dict[str, object]], sample["nodes"])
    assert [node["id"] for node in nodes] == [(48,), (49,), (50,), (51,), (52,)]
