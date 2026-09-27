"""Pushing a corpus's order-1 token transition graph (`analysis.markov_graph.token_graph`)
into Neo4j, in place of the old pyvis-based `graph.html`: vis.js's browser-side physics simulation
stopped being usable well before a real corpus's vocabulary grew past a few hundred nodes, where
Neo4j Browser/Bloom's server-side querying and rendering still holds up.

Every token becomes a `:Token` node (`token_id`, `label`); every observed transition becomes a
`:TRANSITION` relationship carrying THREE numbers rather than one: `count` (the raw, un-normalized
intensity), `log_value` (`markov_graph.log_normalized_edge_values`'s `[0, 1]` log-normalized
reading of that same intensity), and `color` (`markov_graph.log_edge_colors`'s hex string, derived
FROM `log_value`, not from `count` directly) -- keeping `count` alongside the derived pair means a
Neo4j query can still recover the true magnitude a purely visual reading of `color` would obscure
(see `markov_graph`'s module docstring for why the log scale exists at all).

Every node and relationship this module writes carries a `run` property (`run_label`, typically
"`<model>_<snapshot timestamp>`") and every write is scoped to it (`MERGE`d only against other
data sharing the same `run`) -- multiple analyses can share one Neo4j instance without colliding or
silently overwriting each other's graph, and nothing already loaded is ever deleted by a later run.

`Neo4jDriver`/`Neo4jSession` are narrow protocols matching just the shape of the real `neo4j`
package's `Driver`/`Session` this module actually calls (`.session()`, `.run(query, parameters=)`,
context-manager `Session`) -- so a test can inject a lightweight fake without a real Neo4j server
running, the same role `Encoder`/`HuggingFaceLoader` already play elsewhere in this package for
their own external dependencies.
"""

from collections.abc import Iterator, Sequence
from typing import Protocol

import networkx as nx

from corpus_generalization.analysis.markov_graph import log_edge_colors, log_normalized_edge_values

_CONSTRAINT_QUERY = (
    "CREATE CONSTRAINT token_run_id IF NOT EXISTS "
    "FOR (t:Token) REQUIRE (t.run, t.token_id) IS UNIQUE"
)
_NODE_QUERY = (
    "UNWIND $nodes AS node "
    "MERGE (t:Token {run: $run, token_id: node.token_id}) "
    "SET t.label = node.label"
)
_EDGE_QUERY = (
    "UNWIND $edges AS edge "
    "MATCH (a:Token {run: $run, token_id: edge.source}) "
    "MATCH (b:Token {run: $run, token_id: edge.target}) "
    "MERGE (a)-[r:TRANSITION {run: $run}]->(b) "
    "SET r.count = edge.count, r.log_value = edge.log_value, r.color = edge.color"
)
DEFAULT_BATCH_SIZE = 2000


class Neo4jSession(Protocol):
    def run(self, query: str, parameters: dict[str, object] | None = None) -> object: ...
    def __enter__(self) -> "Neo4jSession": ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...


class Neo4jDriver(Protocol):
    def session(self) -> Neo4jSession: ...
    def close(self) -> None: ...


def _chunks[T](items: Sequence[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def export_graph(
    graph: "nx.DiGraph[int]",
    run_label: str,
    driver: Neo4jDriver,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Writes every node and edge of `graph` (e.g. `markov_graph.token_graph`'s output, `<bos>`/
    `<eos>` included) into Neo4j under `run_label`, batched at `batch_size` rows per `UNWIND` so a
    large vocabulary is loaded in a bounded number of round trips rather than one write per node
    or edge. Nodes are written (and the uniqueness constraint created) before edges, since an edge
    write `MATCH`es both endpoints and does nothing if either is missing."""
    colors = log_edge_colors(graph)
    values = log_normalized_edge_values(graph)
    nodes = [{"token_id": node, "label": data["label"]} for node, data in graph.nodes(data=True)]
    edges = [
        {
            "source": source,
            "target": target,
            "count": int(data["count"]),
            "log_value": values[(source, target)],
            "color": colors[(source, target)],
        }
        for source, target, data in graph.edges(data=True)
    ]
    with driver.session() as session:
        session.run(_CONSTRAINT_QUERY)
        for batch in _chunks(nodes, batch_size):
            session.run(_NODE_QUERY, parameters={"run": run_label, "nodes": batch})
        for batch in _chunks(edges, batch_size):
            session.run(_EDGE_QUERY, parameters={"run": run_label, "edges": batch})
