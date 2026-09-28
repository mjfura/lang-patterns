"""`context_support`: a transition credited by the distinct (token before, token after) contexts
it occurs in, so repeated text adds no support — see `docs/2026-09-27-context-support-design.md`."""

from corpus_generalization.analysis.markov_chain import (
    build_transition_graph,
    red_cluster_coverage_curve,
)
from tests.test_markov_chain_analysis import WordIdTokenizer


def _support(documents: tuple[str, ...], source: str, target: str) -> int:
    tokenizer = WordIdTokenizer()
    graph = build_transition_graph(documents, tokenizer, order=1)
    ids = {word: tokenizer.encode(word)[0] for word in (source, target)}
    return int(graph.edges[(ids[source],), (ids[target],)]["context_support"])


def test_a_duplicated_document_doubles_count_and_leaves_context_support_unchanged() -> None:
    documents = ("the cat sat on the mat", "a dog sat on a rug")
    once = build_transition_graph(documents, WordIdTokenizer(), order=1)
    twice = build_transition_graph(documents * 2, WordIdTokenizer(), order=1)

    assert set(once.edges) == set(twice.edges)
    for edge in once.edges:
        assert twice.edges[edge]["count"] == 2 * once.edges[edge]["count"]
        assert twice.edges[edge]["context_support"] == once.edges[edge]["context_support"]


def test_the_same_transition_in_new_surroundings_adds_one() -> None:
    assert _support(("x a b y",), "a", "b") == 1
    assert _support(("x a b y", "z a b w"), "a", "b") == 2
    assert _support(("x a b y", "z a b w", "x a b y"), "a", "b") == 2


def test_context_support_is_between_one_and_count() -> None:
    documents = ("a b a b c a b", "b a c c a", "a b", "c a b a b a")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1, include_boundaries=True)

    for *_, data in graph.edges(data=True):
        assert 1 <= data["context_support"] <= data["count"]


def test_a_missing_neighbor_is_its_own_context_and_boundaries_supply_one() -> None:
    """Without boundaries, "a b" at the start of a document has no left neighbor, which differs
    from the "c" before it elsewhere; framed, both documents put `<bos>`/"c" and `<eos>` there."""
    documents = ("a b", "c a b")
    tokenizer = WordIdTokenizer()
    bare = build_transition_graph(documents, tokenizer, order=1)
    framed = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    a, b = (tokenizer.encode("a")[0],), (tokenizer.encode("b")[0],)

    assert bare.edges[a, b]["context_support"] == 2
    assert framed.edges[a, b]["context_support"] == 2
    assert framed.edges[(tokenizer.bos_id,), a]["context_support"] == 1


def test_at_order_two_the_context_surrounds_the_whole_window() -> None:
    tokenizer = WordIdTokenizer()
    documents = ("x a b c y", "z a b c w")
    edge = tuple((tokenizer.encode(word)[0],) for word in ("a", "b", "c"))
    state, next_state = edge[0] + edge[1], edge[1] + edge[2]

    def support(corpus: tuple[str, ...]) -> int:
        graph = build_transition_graph(corpus, tokenizer, order=2)
        return int(graph.edges[state, next_state]["context_support"])

    assert support(documents) == 2
    assert support((*documents, "x a b c y")) == 2
    assert support((*documents, "x a b c w")) == 3


def test_the_red_curve_over_context_support_ignores_duplicated_documents() -> None:
    documents = ("the cat sat on the mat", "the dog sat on the rug", "a cat ran")
    once = build_transition_graph(documents, WordIdTokenizer(), order=1, include_boundaries=True)
    twice = build_transition_graph(
        documents * 2, WordIdTokenizer(), order=1, include_boundaries=True
    )

    assert red_cluster_coverage_curve(once, "context_support") == red_cluster_coverage_curve(
        twice, "context_support"
    )
