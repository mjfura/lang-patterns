"""A corpus's tokens as a Markov chain: a directed graph of context states and the transitions
observed between them, pooled across the whole corpus rather than anchored to where in a document
they happen to sit.

State = the last `order` tokens (order 1 = a single token, the textbook case: an N x N transition
matrix over the vocabulary). Building states this way rather than requiring a document's own
prefix from position 0 (`analysis.branching`'s approach) makes a transition position-blind:
"I" followed by "am" counts as the same transition whether it opens a sentence or sits in the
middle of one — the property `analysis.branching` cannot give, since its trie only ever
compares prefixes anchored at a document's start.

Two deliberate departures from the textbook recipe, both forced by working with a real, finite,
short-document corpus rather than an idealized infinite chain:

- The stationary distribution is computed with PageRank rather than solving `pi @ P = pi`
  directly. A document's last token is always a dead end (nothing continues after a full stop), so
  the transition matrix is never a clean row-stochastic matrix over a single ergodic chain — it has
  dangling states with no outgoing edges. PageRank's damping is the standard, well-tested fix for
  exactly this pathology (it exists because the web graph has the same problem: pages with no
  outlinks), and it degrades gracefully to something the literal equation cannot survive at all.
- "Irreducible" is checked as WEAK connectivity (treating edge direction as irrelevant), not strong
  connectivity. Strong connectivity requires reaching every state from every other, which never
  holds for any real text corpus — sentence-final tokens are dead ends, so strongly connected
  components would report "fragmented" for even the most pure single-domain corpus. Weak
  connectivity asks the question that is actually useful for spotting a fractured corpus: is there
  a group of tokens that never appears anywhere near another group, in either direction?

Two more numbers characterize the graph as a whole, distinct questions from everything above:

- `gini_coefficient`, over every edge's raw transition count, is a distribution question: are
  transition frequencies roughly even across the graph, or do a handful of edges hog most of the
  observed mass while the rest are barely-seen? It says nothing about WHERE those heavy edges sit.
- `red_cluster_coverage` is the topology question `gini_coefficient` cannot answer: restrict the
  graph to only well-supported edges (`count >= WELL_SUPPORTED_THRESHOLD`, the same bar
  `well_supported_fraction` uses) and find its largest weakly connected component — then divide by
  the FULL graph's node count, not the filtered one's. High coverage means the well-supported edges
  span nearly the whole vocabulary as one connected piece — a broadly learnable corpus. Low coverage
  means the well-supported edges form a small, tight pocket while most of the graph sits outside
  it, connected only by once-seen edges — one good neighborhood in an otherwise sparse corpus.

`red_cluster_coverage` fixes its threshold at `WELL_SUPPORTED_THRESHOLD` (2) and reports one
number, which hides exactly the question it was built to answer: is there a neighborhood held
together by genuinely high-intensity, oft-repeated transitions, or does the reading only clear the
bar because a couple of once-or-twice-seen edges happen to bridge otherwise-disconnected pieces?
Two curves, both against the same idea of "how much intensity do I demand before I stop counting
an edge", answer that:

- `red_cluster_coverage_curve` is `red_cluster_coverage` itself, swept over every threshold from 1
  to the graph's own highest observed count instead of fixed at one. A corpus with a real
  high-intensity core stays near its starting coverage well past `WELL_SUPPORTED_THRESHOLD` before
  collapsing; one that only scrapes past the fixed threshold on a couple of bridging edges falls
  off almost immediately.
- `weighted_core_coverage_curve` asks the same question a different way: the weighted (generalized)
  k-core, peeling away whichever remaining node has the least *total* incident transition count at
  each step and assigning it that count as its core number — the direct generalization of a plain
  k-core (which only ever counts how many neighbors a node has, never how often any single
  transition to them was actually observed) to a weighted graph. Coverage at level k is the
  fraction of nodes whose core number is at least k. Where the first curve asks a global
  connectivity question (is the strong-edge subgraph still one piece), this one asks a local,
  per-node one (is this node itself embedded in a mutually reinforcing dense neighborhood) — the
  two can disagree, and the disagreement is itself informative.

`island_decomposition` completes the red curve rather than adding a fourth question: at each of
its thresholds, the nodes outside the largest component are split into islands (other components
of 2+ nodes — neighborhoods that do repeat, but out of contact with the mainland at that level)
and isolated nodes (no edge reaching the threshold), so the three shares sum to 1. It replaces
reading island size off the gap between the red and weighted-core curves, which is ill-defined:
their x axes are an edge's count and a node's total weight, each normalized by its own maximum.
`island_members` names the tokens of each island at one threshold, which says where a corpus
needs more text.

Both curves collapse to one comparable number, `auc`: the area under the curve with its threshold
axis normalized to `[0, 1]` by its own maximum, so corpora with different count ranges are
comparable. High `auc` means high-intensity cohesion survives well past the lowest threshold; low
`auc` means coverage falls apart the moment more than minimal repetition is demanded.

A third pair, `reciprocal_cluster_coverage_curve` and `reciprocal_core_coverage_curve`, looks at
direction, which `red_cluster_coverage_curve`/`weighted_core_coverage_curve` never do: they treat a
token that only ever opens sentences (many outgoing transitions, none incoming) the same as one
with real traffic through it, as long as their totals match. Both score with the *harmonic mean*
(`2ab/(a+b)`) rather than a sum, so a one-sided value scores exactly 0 however large it is — but
the two apply it to different things, and so answer different questions:

- `reciprocal_cluster_coverage_curve` is pairwise reciprocity: an unordered pair {A, B} scores the
  harmonic mean of `A->B` and `B->A`, so only transitions observed in BOTH directions count. A
  directed cycle `a->b->c->a` has no such pair and reads empty.
- `reciprocal_core_coverage_curve` is flow persistence, not pairwise reciprocity: a node scores the
  harmonic mean of its remaining total incoming and outgoing weight, and peeling removes the nodes
  where either side runs out. What survives is the part of the graph where every token both
  receives and emits transitions from other survivors — tokens on directed cycles, the paths text
  keeps coming back through. The same `a->b->c->a` cycle reads full coverage here with no mirrored
  pair at all (`test_reciprocal_core_counts_a_cycle_with_no_reciprocal_pair`), while a chain
  `a->b->c` reads empty: peeling removes `a` (no input) and `c` (no output), which leaves `b` with
  neither. This is a scalarized, flow-based relative of the directed-network "D-core" (a node's
  core number generalized to a `(k_in, k_out)` pair), kept to one number per node so it stays a
  single curve comparable to the others.

A purely sequential backbone (a template sentence's own grammar, which never runs backwards or
loops) reads empty on both by construction — the finding, not a bug to work around.

All three pairs above are built from `count`: how many times a transition was observed, pooled
across the whole corpus with no regard for which document any occurrence came from. That pooling
hides a real failure mode: a transition a single long document repeats fifty times internally and
one that appears once each in five different documents both read as `count=50` and `count=5`
respectively regardless of which document did the repeating — `count` cannot tell "well attested
across the corpus" from "one document said this a lot", and a model trained on the corpus sees the
former as one training signal repeated, not five independent confirmations of a rule. Every
function above therefore takes a `weight_key` parameter (default `"count"`, so existing callers are
unaffected) naming which edge attribute to read as intensity. Passing
`"effective_document_support"` instead — `build_transition_graph`'s other edge attribute — reruns
every one of the six underlying functions against a different question, and
`document_cluster_coverage_curve`, `document_core_coverage_curve`,
`document_reciprocal_cluster_coverage_curve` and `document_reciprocal_core_coverage_curve` are
exactly that: the same four questions (global cluster / local core, raw / reciprocal), asked of
document dispersion rather than raw repetition.

A first attempt at "document dispersion" (counting DISTINCT documents that contributed at least one
occurrence, capped at 1 per document) is too coarse to be useful on its own: it cannot tell "20
occurrences split 15/5 between two documents" from "2 occurrences, one in each of the same two
documents" — both read as 2, even though the first is far more concentrated than the second.
`effective_document_support` fixes this by asking, for a transition's `count` occurrences spread as
`c_1, c_2, ..., c_k` across the `k` documents it appears in at all, the perplexity of that split:
`p_i = c_i / count`, `entropy = -sum(p_i * log2(p_i))`, `effective_document_support = 2**entropy` —
the same "effective number of outcomes" idea `MarkovChainStats.perplexity` already applies to a
state's own continuation distribution, applied here to a transition's own distribution across
documents instead. It is 1.0 whenever everything landed in a single document, no matter how large
`count` is (entropy of one outcome is 0); it equals the plain distinct-document count exactly when
the split is perfectly even across every document it touches (maximum entropy for `k` outcomes is
`log2(k)`); and it sits strictly between the two for every uneven split in between —
`effective_document_support <= document_support` always, so it never credits a transition with more
spread than it genuinely has. A corpus where every transition's evidence is spread evenly across
many different documents reads high on this second family regardless of how high or low `count`
itself runs; a corpus where the same few documents are doing all the work — even a corpus with
excellent `count`-based coverage — reads low here, and that gap between the two families is itself
the finding: high `count`-based AUC with low `effective_document_support`-based AUC on the same
graph means the apparent cohesion is real but concentrated in too few documents, or too unevenly
split among them, to trust as a generalizable pattern.

`build_transition_graph`'s `include_boundaries` (default `False`, so every existing caller is
unaffected) frames every document as `<bos>` + its tokens + `<eos>` before extracting transitions
-- the same framing `packing.framed_documents` already applies before pre-training
packing -- so the two special tokens become real nodes with real edges rather than being invisible
to every metric above. This matters specifically because `<bos>` occurs EXACTLY once per document,
by construction: summed over every edge leaving it, `count` equals the corpus's own document count,
giving every intensity number in this module a fixed, corpus-level ceiling to be read against
instead of only against other edges' counts (and symmetrically for `<eos>` and every edge entering
it). `document_support`/`effective_document_support` on a `<bos>`-leaving or `<eos>`-entering edge
also collapse together automatically: `<bos>` cannot occur twice in the same document, so its
outgoing edges can never show the "one document repeated many times" pattern the rest of this
module exists to catch.

This deliberately does NOT flow into `markov_chain`'s own `components`/`weakly_connected`/
`largest_component_fraction` (which always call `build_transition_graph` with
`include_boundaries=False`): `<bos>` and `<eos>` are the same two nodes in every document, so
including them turns any set of documents -- even two that share no vocabulary at all -- into one
weakly connected component through that shared pair, destroying exactly the "do these documents
touch each other at all" signal those three fields exist to report. The coverage curves are far
less sensitive to this, since a threshold sweep only lets the `<bos>`/`<eos>` bridge dominate the
lowest few levels of the curve before its own count is exceeded and it drops out like any other
edge -- but the single, unswept components reading has no such threshold to recover behind.
"""

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from corpus_generalization.packing import FramingTokenizer, framed_documents

WELL_SUPPORTED_THRESHOLD = 2
WEIGHTED_CORE_SAMPLE_TARGETS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01)
WEIGHTED_CORE_SAMPLE_FRACTION = 0.05


@dataclass(frozen=True)
class MarkovChainStats:
    order: int
    states: int
    transitions: int
    entropy_bits: float
    perplexity: float
    well_supported_fraction: float
    components: int
    weakly_connected: bool
    largest_component_fraction: float
    gini_coefficient: float
    red_cluster_coverage: float


def markov_chain(
    documents: Sequence[str], tokenizer: FramingTokenizer, order: int = 1
) -> MarkovChainStats:
    graph = build_transition_graph(documents, tokenizer, order)
    n = graph.number_of_nodes()
    if n == 0:
        return MarkovChainStats(order, 0, 0, 0.0, 1.0, 0.0, 0, True, 0.0, 0.0, 0.0)

    source_states = [state for state in graph.nodes if graph.out_degree(state) > 0]
    edge_data = {
        state: [data for _, _, data in graph.out_edges(state, data=True)] for state in source_states
    }
    totals = {
        state: sum(int(data["count"]) for data in edges) for state, edges in edge_data.items()
    }
    well_supported_fraction = (
        sum(1 for total in totals.values() if total >= WELL_SUPPORTED_THRESHOLD)
        / len(source_states)
        if source_states
        else 0.0
    )

    stationary = nx.pagerank(graph, weight="weight")
    entropy = 0.0
    for state in source_states:
        weights = [float(data["weight"]) for data in edge_data[state]]
        local_entropy = -sum(weight * math.log2(weight) for weight in weights if weight > 0)
        entropy += stationary.get(state, 0.0) * local_entropy

    components = list(nx.weakly_connected_components(graph))
    largest = max((len(component) for component in components), default=0)

    return MarkovChainStats(
        order=order,
        states=n,
        transitions=graph.number_of_edges(),
        entropy_bits=entropy,
        perplexity=2**entropy,
        well_supported_fraction=well_supported_fraction,
        components=len(components),
        weakly_connected=len(components) == 1,
        largest_component_fraction=largest / n,
        gini_coefficient=gini_coefficient(
            [int(data["count"]) for _, _, data in graph.edges(data=True)]
        ),
        red_cluster_coverage=red_cluster_coverage(graph, WELL_SUPPORTED_THRESHOLD),
    )


def markov_chain_sweep(
    documents: Sequence[str], tokenizer: FramingTokenizer, orders: Sequence[int] = (1, 2, 3)
) -> tuple[MarkovChainStats, ...]:
    return tuple(markov_chain(documents, tokenizer, order) for order in orders)


def _document_perplexity(per_document_counts: Sequence[int]) -> float:
    """The effective number of documents `per_document_counts` (one transition's occurrence count
    in each document it appears in at all) is spread across: `2**entropy` of the normalized split,
    the same "effective number of outcomes" idea `MarkovChainStats.perplexity` already applies to a
    state's own continuation distribution. 1.0 when everything is in one document, exactly
    `len(per_document_counts)` when the split is perfectly even, strictly between the two
    otherwise -- see the module docstring for the full derivation and motivating example."""
    total = sum(per_document_counts)
    if total == 0:
        return 0.0
    proportions = [count / total for count in per_document_counts]
    entropy = -sum(p * math.log2(p) for p in proportions if p > 0)
    return 2**entropy


def build_transition_graph(
    documents: Sequence[str],
    tokenizer: FramingTokenizer,
    order: int,
    *,
    include_boundaries: bool = False,
) -> "nx.DiGraph[tuple[int, ...]]":
    """Every state (the last `order` tokens) as a node, an edge to the state reached by sliding the
    window forward one token, weighted by observed probability and carrying three counts: the raw
    `count`, pooled across every document (a transition's position within its document is never
    part of what state it belongs to); `document_support`, the number of DISTINCT documents that
    contributed at least one occurrence of it — capped at 1 per document no matter how many times
    that document repeats the transition internally; and `effective_document_support`, the same
    idea made sensitive to HOW that count is split between those documents rather than just how
    many there are (see `_document_perplexity` and the module docstring). All three can diverge: a
    corpus with a long, self-repetitive document can drive `count` arbitrarily high while both
    document measures stay low; a transition split unevenly across several documents reads a full
    `document_support` but a discounted `effective_document_support`.

    A fourth, `context_support`, is the number of distinct contexts the transition occurs in: the
    token just before the state's window and the token just after the transition, `None` where the
    sequence ends first. A copied document, or a document repeating itself, repeats contexts
    already seen, so it raises `count` and both document measures (each copy is a separate
    document) but adds no `context_support`; new text that uses the transition among different
    neighbors does. `experiments/contamination` is why it exists: exact duplicates raised every
    `count`- and document-based curve as much as genuinely new documents did. It lies between 1 and
    `count`. See `docs/2026-09-27-context-support-design.md`.

    `include_boundaries=True` frames every document as `<bos>` + its tokens + `<eos>` first (see
    the module docstring for why, and for the one caller -- `markov_chain`'s own `components`
    family -- that deliberately leaves it `False`)."""
    counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    document_occurrences: dict[tuple[tuple[int, ...], int], Counter[int]] = defaultdict(Counter)
    contexts: dict[tuple[tuple[int, ...], int], set[tuple[int | None, int | None]]] = defaultdict(
        set
    )
    sequences = (
        framed_documents(documents, tokenizer, recognize_markers=True)
        if include_boundaries
        else [tokenizer.encode(document, recognize_markers=True) for document in documents]
    )
    for document_index, sequence in enumerate(sequences):
        for i in range(len(sequence) - order):
            state = tuple(sequence[i : i + order])
            token = sequence[i + order]
            counts[state][token] += 1
            document_occurrences[(state, token)][document_index] += 1
            after = i + order + 1
            contexts[(state, token)].add(
                (
                    sequence[i - 1] if i > 0 else None,
                    sequence[after] if after < len(sequence) else None,
                )
            )

    graph: nx.DiGraph[tuple[int, ...]] = nx.DiGraph()
    for state, next_counts in counts.items():
        total = sum(next_counts.values())
        for token, count in next_counts.items():
            next_state = (*state[1:], token)
            per_document = document_occurrences[(state, token)]
            graph.add_edge(
                state,
                next_state,
                weight=count / total,
                count=count,
                document_support=len(per_document),
                effective_document_support=_document_perplexity(list(per_document.values())),
                context_support=len(contexts[(state, token)]),
            )
    return graph


def gini_coefficient(values: Sequence[float]) -> float:
    """The Gini coefficient of `values` (here, every edge's raw transition count): 0 when every
    value is equal, approaching 1 when a few values dominate the rest. Uses the sorted-values form
    `(sum((2*i - n - 1) * x_i) for i = 1..n) / (n * sum(x))`, equivalent to but cheaper than the
    textbook double sum over every pair."""
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0.0
    ordered = sorted(values)
    weighted = sum((2 * (index + 1) - n - 1) * value for index, value in enumerate(ordered))
    return weighted / (n * total)


def red_cluster_coverage(
    graph: "nx.DiGraph[tuple[int, ...]]", threshold: float, weight_key: str = "count"
) -> float:
    """The fraction of the FULL graph's nodes covered by the largest weakly connected component of
    the subgraph restricted to edges with `weight_key >= threshold` — how much of the whole
    vocabulary sits inside one well-supported, connected neighborhood, versus scattered outside it.
    `weight_key` (default `"count"`, the raw pooled transition count) can also be
    `"effective_document_support"` — see the module docstring for what that reruns this same
    question against."""
    total_nodes = graph.number_of_nodes()
    if total_nodes == 0:
        return 0.0
    red_edges = [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data[weight_key] >= threshold
    ]
    if not red_edges:
        return 0.0
    red_graph: nx.DiGraph[tuple[int, ...]] = nx.DiGraph()
    red_graph.add_edges_from(red_edges)
    largest = max(len(component) for component in nx.weakly_connected_components(red_graph))
    return largest / total_nodes


@dataclass(frozen=True)
class CoverageCurve:
    """A coverage-vs-threshold curve and its area, threshold-normalized to `[0, 1]` by its own
    maximum so curves from corpora with different count ranges are directly comparable. Empty
    (`thresholds == ()`) for a graph with no edges to sweep at all."""

    thresholds: tuple[float, ...]
    coverages: tuple[float, ...]
    auc: float


def _coverage_curve(thresholds: Sequence[float], coverages: Sequence[float]) -> CoverageCurve:
    auc = _area_under_curve(thresholds, coverages)
    return CoverageCurve(tuple(thresholds), tuple(coverages), auc)


def _component_coverage_curve(
    total_nodes: int,
    edges: Sequence[tuple[tuple[int, ...], tuple[int, ...], float]],
) -> CoverageCurve:
    """Find largest-component coverage for every edge-weight threshold in one descending sweep."""
    if total_nodes == 0 or not edges:
        return CoverageCurve((), (), 0.0)

    parent: dict[tuple[int, ...], tuple[int, ...]] = {}
    size: dict[tuple[int, ...], int] = {}
    largest = 0

    def find(node: tuple[int, ...]) -> tuple[int, ...]:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(source: tuple[int, ...], target: tuple[int, ...]) -> None:
        nonlocal largest
        if source not in parent:
            parent[source] = source
            size[source] = 1
        if target not in parent:
            parent[target] = target
            size[target] = 1
        source_root = find(source)
        target_root = find(target)
        if source_root == target_root:
            largest = max(largest, size[source_root])
            return
        if size[source_root] < size[target_root]:
            source_root, target_root = target_root, source_root
        parent[target_root] = source_root
        size[source_root] += size[target_root]
        largest = max(largest, size[source_root])

    thresholds: list[float] = []
    coverages: list[float] = []
    ordered_edges = sorted(edges, key=lambda edge: edge[2], reverse=True)
    index = 0
    while index < len(ordered_edges):
        threshold = ordered_edges[index][2]
        while index < len(ordered_edges) and ordered_edges[index][2] == threshold:
            source, target, _ = ordered_edges[index]
            union(source, target)
            index += 1
        thresholds.append(threshold)
        coverages.append(largest / total_nodes)

    thresholds.reverse()
    coverages.reverse()
    return _coverage_curve(thresholds, coverages)


def _area_under_curve(thresholds: Sequence[float], coverages: Sequence[float]) -> float:
    """The area under a *step* function, not a trapezoid: `coverages[i]` is "the fraction of
    things at or above threshold `thresholds[i]`", which can only ever drop exactly at an observed
    threshold, never between two of them. Between `thresholds[i]` and `thresholds[i+1]` nothing
    has a level in that gap by construction (`thresholds` lists every level anything actually
    reached), so the true value there is `coverages[i]`, held flat, not a straight line drawn
    between the two surrounding points.

    Getting this wrong matters most exactly where `weighted_core_coverage_curve` differs most from
    `red_cluster_coverage_curve`: peeling tends to produce a few widely-spaced plateaus rather than
    one point per integer the way a threshold swept densely does, and a straight line drawn across
    a wide gap between two plateaus overstates the area under what is actually a sharp drop
    followed by a long flat stretch at the lower value.

    Threshold-normalized to `[0, 1]` by its own maximum, with `coverages[0]` implicitly covering
    `[0, thresholds[0]/maximum]` too: nothing is excluded below the lowest threshold actually
    observed, so the curve is flat from 0 up to there, not undefined.
    """
    if not thresholds:
        return 0.0
    maximum = thresholds[-1]
    if maximum == 0:
        return coverages[0]
    xs = [0.0, *(value / maximum for value in thresholds)]
    area = 0.0
    for index in range(len(coverages)):
        area += coverages[index] * (xs[index + 1] - xs[index])
    return area


def _log_area_under_curve(thresholds: Sequence[float], coverages: Sequence[float]) -> float:
    """`_area_under_curve`'s question, but with the threshold axis normalized by
    `log(threshold) / log(maximum)` instead of `threshold / maximum` -- see
    `analysis.markov_graph`'s module docstring for why this exists. `threshold = 1` (not
    `0`) is this axis's zero point, since `log(0)` is undefined and every `weight_key` this module
    sweeps (`count`, `effective_document_support`) is always >= 1 for a node or edge that actually
    carries one -- except a weighted core number, which can be exactly 0 for a node peeled off
    with no non-self-loop edges at all; that 0 is itself the weakest possible level, so it clamps
    to 1 (`log(1) == 0`) and lands at the same zero point rather than needing a separate case. A
    curve whose `maximum` is itself `<= 1` has no ratio to take a log of, so it falls back to the
    same flat-value answer `_area_under_curve` gives for its own degenerate `maximum == 0` case.
    """
    if not thresholds:
        return 0.0
    maximum = thresholds[-1]
    if maximum <= 1:
        return coverages[0]
    log_maximum = math.log(maximum)
    xs = [0.0, *(math.log(max(value, 1.0)) / log_maximum for value in thresholds)]
    area = 0.0
    for index in range(len(coverages)):
        area += coverages[index] * (xs[index + 1] - xs[index])
    return area


def to_log_curve(curve: CoverageCurve) -> CoverageCurve:
    """The same measured `thresholds`/`coverages` as `curve` -- no graph analysis is redone -- but
    `auc` recomputed with the threshold axis log-normalized (`_log_area_under_curve`) instead of
    linearly. A linear-normalized `auc` is dominated by whichever gap between two consecutive
    observed thresholds is widest in raw terms; on a corpus whose transition counts are Zipfian
    (the common case for natural-language token statistics), that is almost always the gap right
    below the single highest count, so `auc` ends up scored mostly by what happens at that one
    extreme rather than by how coverage holds up across the many low-frequency cuts that make up
    most of a real corpus. This is not a fix for `auc` -- a second, independently meaningful
    reading of the exact same curve, answering "does coverage survive across ORDERS OF MAGNITUDE
    of threshold" rather than "across how much of the raw threshold range" -- so it is exposed as
    a conversion a caller opts into (`analysis.markov_graph.draw_coverage_curves_log` and
    its document-support sibling), not a change to what `red_cluster_coverage_curve` and the rest
    already return.
    """
    return CoverageCurve(
        curve.thresholds, curve.coverages, _log_area_under_curve(curve.thresholds, curve.coverages)
    )


def red_cluster_coverage_curve(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> CoverageCurve:
    """`red_cluster_coverage`, swept over every distinct `weight_key` value actually observed on an
    edge, from lowest to highest, instead of fixed at `WELL_SUPPORTED_THRESHOLD`. Coverage can only
    change exactly at one of these levels (see `_area_under_curve`), so sweeping any finer between
    them -- every integer in between, say -- would only repeat the value already found at the level
    below; sweeping the observed levels is both sufficient for an integer-valued `weight_key` like
    the default `"count"` and correct for one that is not, like `"effective_document_support"`,
    where an integer range would silently truncate every level to its floor and could collapse
    several genuinely different levels onto one. See the module docstring for what the resulting
    curve and its `auc` say that the single fixed-threshold reading cannot."""
    if graph.number_of_nodes() == 0:
        return CoverageCurve((), (), 0.0)
    edges = [
        (source, target, float(data[weight_key])) for source, target, data in graph.edges(data=True)
    ]
    return _component_coverage_curve(graph.number_of_nodes(), edges)


@dataclass(frozen=True)
class IslandLevel:
    """One threshold of `island_decomposition`: fractions of the FULL graph's nodes, summing to 1.
    `largest_island` is in nodes, 0 when there is no island."""

    threshold: float
    mainland: float
    islands: float
    isolated: float
    island_count: int
    largest_island: int


def island_decomposition(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> tuple[IslandLevel, ...]:
    """At every threshold `red_cluster_coverage_curve` sweeps, split the full graph's nodes into the
    mainland (the largest weakly connected component of the edges reaching the threshold — so
    `mainland` IS the red curve), islands (every other component of 2+ nodes) and isolated (the
    rest: nodes with no edge reaching the threshold, or only a self-loop). The red curve alone
    reports the first share and hides where the other two sit; islands are neighborhoods that do
    repeat but not in contact with the mainland at that level, isolated tokens are those whose
    transitions do not repeat enough to be learned at all. Ascending thresholds; empty for a graph
    with no edges. See `docs/2026-09-27-islands-design.md`."""
    total_nodes = graph.number_of_nodes()
    edges = sorted(
        (
            (source, target, float(data[weight_key]))
            for source, target, data in graph.edges(data=True)
        ),
        key=lambda edge: edge[2],
        reverse=True,
    )
    if total_nodes == 0 or not edges:
        return ()

    parent: dict[tuple[int, ...], tuple[int, ...]] = {}
    size: dict[tuple[int, ...], int] = {}
    # How many components have each size; with the running totals below, every threshold is read
    # in O(distinct sizes) instead of rescanning components.
    sizes: Counter[int] = Counter()
    multi_node_components = 0
    multi_node_nodes = 0

    def find(node: tuple[int, ...]) -> tuple[int, ...]:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            parent[node], node = root, parent[node]
        return root

    def add(node: tuple[int, ...]) -> None:
        if node not in parent:
            parent[node] = node
            size[node] = 1
            sizes[1] += 1

    def remove_size(value: int) -> None:
        nonlocal multi_node_components, multi_node_nodes
        sizes[value] -= 1
        if sizes[value] == 0:
            del sizes[value]
        if value >= 2:
            multi_node_components -= 1
            multi_node_nodes -= value

    def union(source: tuple[int, ...], target: tuple[int, ...]) -> None:
        nonlocal multi_node_components, multi_node_nodes
        add(source)
        add(target)
        source_root, target_root = find(source), find(target)
        if source_root == target_root:
            return
        if size[source_root] < size[target_root]:
            source_root, target_root = target_root, source_root
        remove_size(size[source_root])
        remove_size(size[target_root])
        parent[target_root] = source_root
        size[source_root] += size[target_root]
        merged = size[source_root]
        sizes[merged] += 1
        multi_node_components += 1
        multi_node_nodes += merged

    levels: list[IslandLevel] = []
    index = 0
    while index < len(edges):
        threshold = edges[index][2]
        while index < len(edges) and edges[index][2] == threshold:
            union(edges[index][0], edges[index][1])
            index += 1
        largest = max(sizes)
        if largest >= 2:
            island_nodes = multi_node_nodes - largest
            island_count = multi_node_components - 1
            island_sizes = [value for value in sizes if value >= 2 and value < largest]
            largest_island = largest if sizes[largest] >= 2 else max(island_sizes, default=0)
        else:
            island_nodes = island_count = largest_island = 0
        levels.append(
            IslandLevel(
                threshold,
                largest / total_nodes,
                island_nodes / total_nodes,
                (total_nodes - largest - island_nodes) / total_nodes,
                island_count,
                largest_island,
            )
        )
    levels.reverse()
    return tuple(levels)


def island_members(
    graph: "nx.DiGraph[tuple[int, ...]]", threshold: float, weight_key: str = "count"
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """The islands of `island_decomposition` at one threshold, each as its sorted nodes, largest
    first: the actionable reading, which tokens form each neighborhood cut off from the mainland.
    Ties (for the mainland and in the order) break on the sorted nodes, so the result is
    deterministic."""
    kept: nx.DiGraph[tuple[int, ...]] = nx.DiGraph()
    kept.add_edges_from(
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data[weight_key] >= threshold
    )
    components = sorted(
        (tuple(sorted(component)) for component in nx.weakly_connected_components(kept)),
        key=lambda component: (-len(component), component),
    )
    return tuple(component for component in components[1:] if len(component) >= 2)


def weighted_core_numbers(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> dict[tuple[int, ...], float]:
    """Every node's weighted core number: the generalized k-core, found by repeatedly peeling away
    whichever remaining node has the least total incident `weight_key` value (summed over both
    directions, since a directed A->B and B->A pair should reinforce each other's cohesion, not
    compete for it) and assigning it that running-maximum minimum as its core number — the
    textbook k-core peeling rule, generalized from counting neighbors to summing how much of
    `weight_key` each transition to them carried. `weight_key` defaults to `"count"` (raw pooled
    transition count); see the module docstring for what `"effective_document_support"` asks
    instead.

    Isolated nodes (no edges at all) get core number 0 rather than being left out, so every node
    `build_transition_graph` produced has an entry here.
    """
    neighbor_weights: dict[tuple[int, ...], dict[tuple[int, ...], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for source, target, data in graph.edges(data=True):
        weight = float(data[weight_key])
        if source == target:
            continue
        neighbor_weights[source][target] += weight
        neighbor_weights[target][source] += weight

    strength = {node: sum(neighbor_weights[node].values()) for node in graph.nodes}
    remaining = set(graph.nodes)
    core: dict[tuple[int, ...], float] = {}
    running_floor = 0.0
    while remaining:
        node = min(remaining, key=lambda candidate: strength[candidate])
        running_floor = max(running_floor, strength[node])
        core[node] = running_floor
        remaining.discard(node)
        for neighbor, weight in neighbor_weights[node].items():
            if neighbor in remaining:
                strength[neighbor] -= weight
    return core


def weighted_core_coverage_curve(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> CoverageCurve:
    """The k-core analogue of `red_cluster_coverage_curve`: coverage at level `k` is the fraction
    of nodes whose weighted core number (`weighted_core_numbers`) is at least `k`, over every
    distinct level the graph actually reaches. `weight_key` is passed straight through to
    `weighted_core_numbers`."""
    core = weighted_core_numbers(graph, weight_key)
    total = len(core)
    if total == 0:
        return CoverageCurve((), (), 0.0)
    levels = sorted(set(core.values()))
    coverages = [sum(1 for value in core.values() if value >= level) / total for level in levels]
    return _coverage_curve(levels, coverages)


def weighted_core_samples(
    core: dict[tuple[int, ...], float],
    curve: CoverageCurve,
    sample_fraction: float = WEIGHTED_CORE_SAMPLE_FRACTION,
    targets: Sequence[float] = WEIGHTED_CORE_SAMPLE_TARGETS,
) -> list[dict[str, object]]:
    """Select a small, deterministic neighborhood of nodes around positions on the linear axis.

    Each target is a position on the linear coverage plot's x axis, `threshold / maximum`, so its
    threshold is `target * maximum` (see `_axis_threshold`); nodes are then ordered by distance to
    that threshold, above or below it, and the closest ``sample_fraction`` of the full graph is
    retained. Ties are ordered by core number and node id. ``token_ids`` flattens the retained
    nodes' ids in the same order as ``nodes``.
    """
    if not core or not curve.thresholds or sample_fraction <= 0:
        return []
    sample_count = max(1, round(len(core) * sample_fraction))
    nodes = tuple(core)
    samples: list[dict[str, object]] = []
    for target in targets:
        threshold, coverage = _axis_threshold(curve, target)
        ordered = sorted(
            nodes,
            key=lambda node: (abs(core[node] - threshold), core[node], node),
        )
        selected = sorted(ordered[:sample_count], key=lambda node: (core[node], node))
        samples.append(
            {
                "target_fraction": target,
                "target_percent": target * 100,
                "threshold": threshold,
                "actual_coverage": coverage,
                "sample_count": len(selected),
                "sample_fraction": len(selected) / len(core),
                "token_ids": [token for node in selected for token in node],
                "nodes": [
                    {
                        "id": node,
                        "core_number": core[node],
                        "distance_to_threshold": abs(core[node] - threshold),
                    }
                    for node in selected
                ],
            }
        )
    return samples


def coverage_curve_percentiles(
    curve: CoverageCurve,
    targets: Sequence[float] = WEIGHTED_CORE_SAMPLE_TARGETS,
) -> list[dict[str, float]]:
    """The threshold at each target position on the linear plot's x axis, and the coverage the
    curve holds there -- see `_axis_threshold`."""
    if not curve.thresholds:
        return []
    entries: list[dict[str, float]] = []
    for target in targets:
        threshold, coverage = _axis_threshold(curve, target)
        entries.append(
            {
                "target_fraction": target,
                "target_percent": target * 100,
                "threshold": threshold,
                "actual_coverage": coverage,
            }
        )
    return entries


def _axis_threshold(curve: CoverageCurve, target: float) -> tuple[float, float]:
    """`target` read as a position on the linear plot's x axis, `threshold / maximum` (see
    `markov_graph._step_points`): the threshold `target * maximum`, and the coverage the step
    curve holds there -- `coverages[i]` covers every threshold above `thresholds[i-1]` up to
    `thresholds[i]`. A target, not a coverage: on a skewed graph half the maximum count can leave
    two nodes connected, where half the coverage would sit at a small fraction of the maximum."""
    threshold = target * curve.thresholds[-1]
    index = min(bisect_left(curve.thresholds, threshold), len(curve.thresholds) - 1)
    return threshold, curve.coverages[index]


def _harmonic_mean(a: float, b: float) -> float:
    """0 if either is 0 -- a one-directional value has no "other side" to average with, and the
    textbook harmonic-mean formula would divide by `a + b` and silently return the nonzero one, as
    if a single direction were itself evidence of reciprocity."""
    return 0.0 if a == 0.0 or b == 0.0 else 2 * a * b / (a + b)


def reciprocal_edge_weights(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> dict[frozenset[tuple[int, ...]], float]:
    """Every unordered pair of nodes with a directed edge between them in either direction, scored
    by the harmonic mean of the forward and backward `weight_key` value — 0 whenever only one
    direction was ever observed, however large its value. The edge-level reading of the same idea
    `reciprocal_core_numbers` applies node by node: real interaction runs both ways, and a purely
    sequential transition (the grammar of a template sentence, which never runs backwards) is not
    that, no matter how often it occurs. `weight_key` defaults to `"count"`; see the module
    docstring for what `"effective_document_support"` asks instead — both directions each
    genuinely spread across many distinct documents, not merely observed many times somewhere or
    concentrated in a few.
    """
    directed = {
        (source, target): float(data[weight_key])
        for source, target, data in graph.edges(data=True)
        # A self-loop (a token that transitions to itself) has no "other side" to reciprocate
        # with -- (source, source) as a frozenset collapses to one element, not the pair every
        # other reader of this dict expects, so it is excluded here rather than mishandled later.
        if source != target
    }
    pairs: dict[frozenset[tuple[int, ...]], float] = {}
    for source, target in directed:
        key = frozenset((source, target))
        if key in pairs:
            continue
        forward = directed.get((source, target), 0.0)
        backward = directed.get((target, source), 0.0)
        pairs[key] = _harmonic_mean(forward, backward)
    return pairs


def reciprocal_cluster_coverage(
    graph: "nx.DiGraph[tuple[int, ...]]", threshold: float, weight_key: str = "count"
) -> float:
    """`red_cluster_coverage`'s question, asked of `reciprocal_edge_weights` instead of each
    directed edge's own raw value: how much of the vocabulary sits in one connected neighborhood
    of pairs that interact *both* ways at intensity `threshold` or more. A pair with a strong
    value in only one direction contributes nothing, by construction of `reciprocal_edge_weights`.
    `weight_key` is passed straight through to it.
    """
    total_nodes = graph.number_of_nodes()
    if total_nodes == 0:
        return 0.0
    kept = [
        pair
        for pair, weight in reciprocal_edge_weights(graph, weight_key).items()
        if weight >= threshold
    ]
    if not kept:
        return 0.0
    reciprocal_graph: nx.Graph[tuple[int, ...]] = nx.Graph()
    for pair in kept:
        source, target = pair
        reciprocal_graph.add_edge(source, target)
    largest = max(len(component) for component in nx.connected_components(reciprocal_graph))
    return largest / total_nodes


def reciprocal_cluster_coverage_curve(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> CoverageCurve:
    """`red_cluster_coverage_curve`'s sweep, over `reciprocal_edge_weights`' levels instead of raw
    values. Empty whenever the graph has no genuinely bidirectional pair at all — a purely
    sequential corpus (see the module docstring) is expected to give exactly this, not a bug to
    work around. `weight_key` is passed straight through to `reciprocal_edge_weights`."""
    if graph.number_of_nodes() == 0:
        return CoverageCurve((), (), 0.0)
    edges = [
        (source, target, weight)
        for pair, weight in reciprocal_edge_weights(graph, weight_key).items()
        if weight > 0
        for source, target in [tuple(pair)]
    ]
    return _component_coverage_curve(graph.number_of_nodes(), edges)


def reciprocal_core_numbers(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> dict[tuple[int, ...], float]:
    """`weighted_core_numbers`' peeling rule, but each node's current strength is the harmonic mean
    of its remaining total incoming and outgoing `weight_key` value rather than their sum: a node
    whose edges point only one way — the token that only ever opens sentences, or only ever closes
    them — scores 0 here even if its one-directional total is large. Because both totals are
    recomputed as nodes are peeled, this measures flow persistence rather than pairwise
    reciprocity: a node keeps a nonzero core number only while other survivors still feed it and
    are fed by it, which a directed cycle does without any transition ever running backwards (see
    the module docstring, and for what `weight_key="effective_document_support"` asks instead of
    the default `"count"`).

    Isolated nodes, and nodes left with edges in only one direction once peeling reaches them, get
    core number 0 rather than being left out, so every node `build_transition_graph` produced has
    an entry here.
    """
    out_neighbors: dict[tuple[int, ...], dict[tuple[int, ...], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    in_neighbors: dict[tuple[int, ...], dict[tuple[int, ...], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for source, target, data in graph.edges(data=True):
        weight = float(data[weight_key])
        if source == target:
            continue
        out_neighbors[source][target] += weight
        in_neighbors[target][source] += weight

    in_total = {node: sum(in_neighbors[node].values()) for node in graph.nodes}
    out_total = {node: sum(out_neighbors[node].values()) for node in graph.nodes}

    def strength(node: tuple[int, ...]) -> float:
        return _harmonic_mean(in_total[node], out_total[node])

    remaining = set(graph.nodes)
    core: dict[tuple[int, ...], float] = {}
    running_floor = 0.0
    while remaining:
        node = min(remaining, key=strength)
        running_floor = max(running_floor, strength(node))
        core[node] = running_floor
        remaining.discard(node)
        # Removing `node` drops its incoming edges from its in-neighbors' outgoing total, and its
        # outgoing edges from its out-neighbors' incoming total.
        for neighbor, weight in in_neighbors[node].items():
            if neighbor in remaining:
                out_total[neighbor] -= weight
        for neighbor, weight in out_neighbors[node].items():
            if neighbor in remaining:
                in_total[neighbor] -= weight
    return core


def reciprocal_core_coverage_curve(
    graph: "nx.DiGraph[tuple[int, ...]]", weight_key: str = "count"
) -> CoverageCurve:
    """`weighted_core_coverage_curve`'s question, over `reciprocal_core_numbers` instead of
    `weighted_core_numbers`. Empty whenever peeling leaves no node that both receives and emits
    transitions — expected, not a bug, for a purely sequential corpus with no cycles; see the
    module docstring.
    `weight_key` is passed straight through to `reciprocal_core_numbers`."""
    core = reciprocal_core_numbers(graph, weight_key)
    total = len(core)
    if total == 0:
        return CoverageCurve((), (), 0.0)
    levels = sorted({value for value in core.values() if value > 0})
    if not levels:
        return CoverageCurve((), (), 0.0)
    coverages = [sum(1 for value in core.values() if value >= level) / total for level in levels]
    return _coverage_curve(levels, coverages)


def document_cluster_coverage_curve(graph: "nx.DiGraph[tuple[int, ...]]") -> CoverageCurve:
    """`red_cluster_coverage_curve`, thresholded by `effective_document_support` (the perplexity of
    how a transition's occurrences split across the documents it appears in — see
    `_document_perplexity` and the module docstring) instead of raw pooled `count`. A transition a
    single document repeats many times over never counts as well-supported here on the strength of
    that one document alone, and one split unevenly across several documents is discounted toward
    however concentrated it actually is, not given full credit for merely touching all of them."""
    return red_cluster_coverage_curve(graph, weight_key="effective_document_support")


def document_core_coverage_curve(graph: "nx.DiGraph[tuple[int, ...]]") -> CoverageCurve:
    """`weighted_core_coverage_curve`'s local, per-node reading of the same
    `effective_document_support` question `document_cluster_coverage_curve` asks globally."""
    return weighted_core_coverage_curve(graph, weight_key="effective_document_support")


def document_reciprocal_cluster_coverage_curve(
    graph: "nx.DiGraph[tuple[int, ...]]",
) -> CoverageCurve:
    """`reciprocal_cluster_coverage_curve`'s question — how much of the vocabulary sits in one
    connected neighborhood of genuinely bidirectional pairs — asked of `effective_document_support`
    on each side of the harmonic mean instead of raw `count`: a pair only scores well here if BOTH
    its forward and backward direction are each spread genuinely widely and evenly across distinct
    documents, not merely observed many times somewhere or concentrated in a few."""
    return reciprocal_cluster_coverage_curve(graph, weight_key="effective_document_support")


def document_reciprocal_core_coverage_curve(
    graph: "nx.DiGraph[tuple[int, ...]]",
) -> CoverageCurve:
    """`reciprocal_core_coverage_curve`'s flow-persistence reading, over
    `effective_document_support` instead of raw `count`: the cycles text keeps coming back
    through, counting only evidence spread across distinct documents."""
    return reciprocal_core_coverage_curve(graph, weight_key="effective_document_support")
