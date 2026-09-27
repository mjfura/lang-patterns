"""`analysis.neo4j_export`: pushing the order-1 token graph into Neo4j, batched and tagged
by `run_label`, with a fake driver/session standing in for a real Neo4j instance."""

from typing import cast

import networkx as nx

from corpus_generalization.analysis.markov_graph import (
    log_edge_colors,
    log_normalized_edge_values,
    token_graph,
)
from corpus_generalization.analysis.neo4j_export import export_graph


class WordIdTokenizer:
    """Matches `tests/test_markov_graph.py`'s own fixture tokenizer."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    @property
    def bos_id(self) -> int:
        return -1

    @property
    def eos_id(self) -> int:
        return -2

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            token_id = self._ids.setdefault(word, len(self._ids))
            self._words[token_id] = word
            ids.append(token_id)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._words[token_id] for token_id in token_ids)


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object] | None]] = []
        self.closed = False

    def run(self, query: str, parameters: dict[str, object] | None = None) -> None:
        self.calls.append((query, parameters))

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.closed = True


class FakeDriver:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []
        self.closed = False

    def session(self) -> FakeSession:
        session = FakeSession()
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


def _small_graph() -> "nx.DiGraph[int]":
    documents = ("a b c", "a b d")
    return token_graph(documents, WordIdTokenizer())


def _rows(call: dict[str, object], key: str) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], call[key])


def _calls_with(session: FakeSession, key: str) -> list[dict[str, object]]:
    return [
        parameters
        for _, parameters in session.calls
        if parameters is not None and key in parameters
    ]


def test_export_graph_uses_one_session_for_the_whole_export() -> None:
    driver = FakeDriver()

    export_graph(_small_graph(), "run1", driver)

    assert len(driver.sessions) == 1
    assert driver.sessions[0].closed


def test_export_graph_creates_the_uniqueness_constraint_first() -> None:
    driver = FakeDriver()

    export_graph(_small_graph(), "run1", driver)

    first_query, first_parameters = driver.sessions[0].calls[0]
    assert "CREATE CONSTRAINT" in first_query
    assert "token_id" in first_query
    assert first_parameters is None


def test_export_graph_writes_every_node_with_its_label_and_run_tag() -> None:
    graph = _small_graph()
    driver = FakeDriver()

    export_graph(graph, "run1", driver)

    node_calls = _calls_with(driver.sessions[0], "nodes")
    written = {row["token_id"]: row["label"] for call in node_calls for row in _rows(call, "nodes")}
    expected = {node: data["label"] for node, data in graph.nodes(data=True)}
    assert written == expected
    for call in node_calls:
        assert call["run"] == "run1"


def test_export_graph_writes_every_edge_with_count_log_value_and_color() -> None:
    graph = _small_graph()
    driver = FakeDriver()
    colors = log_edge_colors(graph)
    values = log_normalized_edge_values(graph)

    export_graph(graph, "run1", driver)

    edge_calls = _calls_with(driver.sessions[0], "edges")
    written = {
        (row["source"], row["target"]): row for call in edge_calls for row in _rows(call, "edges")
    }
    for source, target, data in graph.edges(data=True):
        row = written[(source, target)]
        assert row["count"] == data["count"]
        assert row["log_value"] == values[(source, target)]
        assert row["color"] == colors[(source, target)]
    for call in edge_calls:
        assert call["run"] == "run1"


def test_export_graph_batches_large_graphs_into_multiple_unwind_calls() -> None:
    """Ten documents each contributing at least one distinct edge -- more than `batch_size=3` --
    must split into multiple `UNWIND` calls rather than one giant query."""
    documents = tuple(f"word{i} word{i + 1}" for i in range(10))
    graph = token_graph(documents, WordIdTokenizer())
    driver = FakeDriver()

    export_graph(graph, "run1", driver, batch_size=3)

    edge_calls = _calls_with(driver.sessions[0], "edges")
    assert len(edge_calls) > 1
    for call in edge_calls:
        assert len(_rows(call, "edges")) <= 3
    total_edges = sum(len(_rows(call, "edges")) for call in edge_calls)
    assert total_edges == graph.number_of_edges()
