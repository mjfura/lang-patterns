"""Visual export of a corpus's order-1 token transition graph: static images (networkx +
matplotlib), showing only the tokens the corpus actually uses and the transitions observed between
them. An interactive, browser-rendered page used to be exported here too (pyvis/vis.js) but its
client-side physics simulation stopped being usable well before a real corpus's vocabulary grew
past a few hundred nodes; `analysis.neo4j_export` replaces it for that case, pushing the
same graph into Neo4j instead, where the interactivity is server-backed rather than a browser
running a full-graph physics layout.

Every edge is colored by how often that transition was observed in the corpus, linearly
interpolated from `LOW_FREQUENCY_COLOR` (the rarest transitions) to `HIGH_FREQUENCY_COLOR` (the
most frequent) — so the handful of well-supported transitions stand out in red against a corpus
that is mostly pale, once-seen edges, the same imbalance `well_supported_fraction` reports as a
number. `draw_static_log`'s log-normalized sibling (`_log_position`/`log_edge_colors`) exists
because that linear interpolation collapses against a Zipfian corpus -- a handful of high-count
outliers push every merely-low count to the same near-`LOW_FREQUENCY_COLOR` shade -- see
`_log_frequency_color`'s own docstring; `analysis.neo4j_export` reuses `log_edge_colors`
and `log_normalized_edge_values` directly rather than recomputing the same normalization a third
way.

A chosen document's own walk can still be highlighted on the static image — since order-1 nodes are
individual tokens, a document's sequence is directly a walk over consecutive edges of this graph,
the reason for building this at order 1 rather than a higher one, where a node would be a tuple and
a document's route far less legible. Color is already spoken for by frequency, so a highlighted
path is instead drawn as a distinctly thicker line (and its nodes in a distinct color, since nodes
carry no frequency of their own).

Every document is framed with `<bos>`/`<eos>` (`build_transition_graph`'s `include_boundaries=True`
— see its module docstring) before this graph is built, so the two special tokens appear as real
nodes with real edges rather than being invisible to the picture: `<bos>` is where every document's
own walk starts and `<eos>` where it ends, and `<bos>`'s total out-degree (summed count over every
edge leaving it) is exactly the corpus's document count, a fixed scale every other node's intensity
can be read against.
"""

import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from corpus_generalization.analysis.markov_chain import (
    CoverageCurve,
    build_transition_graph,
    to_log_curve,
)
from corpus_generalization.packing import framed_documents

DEFAULT_NODE_COLOR = "#4C72B0"
HIGHLIGHT_NODE_COLOR = "#D62728"
LOW_FREQUENCY_COLOR = "#D9D9D9"
HIGH_FREQUENCY_COLOR = "#B30000"


class DecodableTokenizer(Protocol):
    @property
    def bos_id(self) -> int: ...
    @property
    def eos_id(self) -> int: ...
    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...


def _label(token: int, tokenizer: DecodableTokenizer) -> str:
    """`decode` drops every special token (see `tokenizer`), so `<bos>`/`<eos>` need their
    own readable label rather than falling through to `decode`'s empty string."""
    if token == tokenizer.bos_id:
        return "<bos>"
    if token == tokenizer.eos_id:
        return "<eos>"
    return tokenizer.decode([token]) or repr(token)


def token_graph(documents: Sequence[str], tokenizer: DecodableTokenizer) -> "nx.DiGraph[int]":
    """The order-1 transition graph, nodes keyed by token id and labeled with the text each token
    decodes to (`<bos>`/`<eos>` included — see the module docstring)."""
    order1 = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    graph: nx.DiGraph[int] = nx.DiGraph()
    for (token,) in order1.nodes:
        graph.add_node(token, label=_label(token, tokenizer))
    for (source,), (target,), data in order1.edges(data=True):
        graph.add_edge(source, target, weight=data["weight"], count=data["count"])
    return graph


def document_path(document: str, tokenizer: DecodableTokenizer) -> list[int]:
    """A document's own token sequence, `<bos>`/`<eos>` framed the same way `token_graph` frames
    every document — the walk to highlight on the graph."""
    (sequence,) = framed_documents([document], tokenizer, recognize_markers=True)
    return sequence


def _highlighted_edges(path: Sequence[int]) -> set[tuple[int, int]]:
    return set(pairwise(path))


def _frequency_color(count: int, minimum: int, maximum: int) -> str:
    """Linear interpolation, in RGB, from `LOW_FREQUENCY_COLOR` at `minimum` to
    `HIGH_FREQUENCY_COLOR` at `maximum`. Every edge the same frequency (`minimum == maximum`) has
    nothing to contrast against, so it lands exactly halfway rather than dividing by zero."""
    span = maximum - minimum
    position = 0.5 if span == 0 else (count - minimum) / span
    low = tuple(int(LOW_FREQUENCY_COLOR[index : index + 2], 16) for index in (1, 3, 5))
    high = tuple(int(HIGH_FREQUENCY_COLOR[index : index + 2], 16) for index in (1, 3, 5))
    rgb = (round(low[channel] + position * (high[channel] - low[channel])) for channel in range(3))
    return "#" + "".join(f"{value:02X}" for value in rgb)


def _edge_colors(graph: "nx.DiGraph[int]") -> dict[tuple[int, int], str]:
    counts = [int(data["count"]) for _, _, data in graph.edges(data=True)]
    minimum, maximum = (min(counts), max(counts)) if counts else (0, 0)
    return {
        (source, target): _frequency_color(int(data["count"]), minimum, maximum)
        for source, target, data in graph.edges(data=True)
    }


def _log_position(count: int, minimum: int, maximum: int) -> float:
    """The `[0, 1]` position `_log_frequency_color` interpolates a color from -- factored out so
    `log_normalized_edge_values` (a raw number `analysis.neo4j_export` stores as a
    relationship property) and `_log_frequency_color` (a color derived FROM that same number)
    always agree, rather than risking two independent, driftable computations of it. Real corpora
    tend to have Zipfian transition counts -- a handful of very frequent transitions and a long
    tail of once- or twice-seen ones -- and a LINEAR position against that shape puts almost every
    edge within a hair of 0, since a count of 1 and a count of 50 both sit near 0 on a scale that
    runs up to an outlier of 5000. Log-space spreads the low end back out (1 vs 50 are five orders
    of magnitude apart in relative terms even though both are "small") at the cost of compressing
    the high end (500 and 5000 look nearly as similar as they are far apart in absolute count) --
    the reverse trade-off from the linear version, useful for spotting which nominally-rare
    transitions are weaker or stronger than the rest of the rare crowd. `minimum` is clamped to at
    least 1 so `log(minimum)` is always defined (an edge's `count` is always >= 1 by construction,
    but the clamp keeps this correct even if that invariant loosens)."""
    minimum = max(minimum, 1)
    span = math.log(maximum) - math.log(minimum) if maximum > minimum else 0.0
    return 0.5 if span == 0 else (math.log(max(count, 1)) - math.log(minimum)) / span


def _log_frequency_color(count: int, minimum: int, maximum: int) -> str:
    """`_frequency_color`'s log-normalized sibling: interpolates from `LOW_FREQUENCY_COLOR` to
    `HIGH_FREQUENCY_COLOR` over `_log_position` instead of a linear position -- see that function's
    docstring for why."""
    position = _log_position(count, minimum, maximum)
    low = tuple(int(LOW_FREQUENCY_COLOR[index : index + 2], 16) for index in (1, 3, 5))
    high = tuple(int(HIGH_FREQUENCY_COLOR[index : index + 2], 16) for index in (1, 3, 5))
    rgb = (round(low[channel] + position * (high[channel] - low[channel])) for channel in range(3))
    return "#" + "".join(f"{value:02X}" for value in rgb)


def log_edge_colors(graph: "nx.DiGraph[int]") -> dict[tuple[int, int], str]:
    """Every edge's `_log_frequency_color`, keyed by `(source, target)` -- public (unlike
    `_edge_colors`, its linear counterpart) because `analysis.neo4j_export` reuses it
    directly, so a Neo4j relationship's color always matches the log-scale PNGs' own palette
    rather than a second, independently-maintained one."""
    counts = [int(data["count"]) for _, _, data in graph.edges(data=True)]
    minimum, maximum = (min(counts), max(counts)) if counts else (0, 0)
    return {
        (source, target): _log_frequency_color(int(data["count"]), minimum, maximum)
        for source, target, data in graph.edges(data=True)
    }


def log_normalized_edge_values(graph: "nx.DiGraph[int]") -> dict[tuple[int, int], float]:
    """Every edge's `_log_position` in `[0, 1]` directly, the same number `log_edge_colors` turns
    into a color -- for a caller (`analysis.neo4j_export`) that wants the number itself
    stored alongside the color, not just the color."""
    counts = [int(data["count"]) for _, _, data in graph.edges(data=True)]
    minimum, maximum = (min(counts), max(counts)) if counts else (0, 0)
    return {
        (source, target): _log_position(int(data["count"]), minimum, maximum)
        for source, target, data in graph.edges(data=True)
    }


def _layout(graph: "nx.DiGraph[int]") -> dict[int, NDArray[np.float64]]:
    """Spring layout, seeded so it reproduces exactly -- shared by every drawing function so a
    node sits at the same spot whether it appears in the full graph or a smaller view derived
    from it (`draw_*_evolution`'s frames), making frame-to-frame comparison about what changed,
    not about where anything moved."""
    node_count = max(graph.number_of_nodes(), 1)
    layout: dict[int, NDArray[np.float64]] = nx.spring_layout(
        graph, seed=0, k=1.5 / node_count**0.5
    )
    return layout


def draw_static(graph: "nx.DiGraph[int]", output: Path, highlight: Sequence[int] = ()) -> None:
    """A static PNG, spring-laid-out. Edges are colored by observed transition frequency
    (`LOW_FREQUENCY_COLOR` to `HIGH_FREQUENCY_COLOR`); `highlight`'s document walk, if given, is
    drawn as a distinctly thicker line instead, since color is already taken."""
    highlighted_nodes = set(highlight)
    highlighted_edges = _highlighted_edges(highlight)
    positions = _layout(graph)
    labels = {node: data["label"] for node, data in graph.nodes(data=True)}
    edge_colors_by_pair = _edge_colors(graph)
    node_colors = [
        HIGHLIGHT_NODE_COLOR if node in highlighted_nodes else DEFAULT_NODE_COLOR
        for node in graph.nodes
    ]
    edge_colors = [edge_colors_by_pair[edge] for edge in graph.edges]
    edge_widths = [2.5 if edge in highlighted_edges else 0.6 for edge in graph.edges]

    figure, axis = plt.subplots(figsize=(14, 14))
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=350, ax=axis)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=axis)
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        connectionstyle="arc3,rad=0.08",
        node_size=350,
        ax=axis,
    )
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def draw_static_log(graph: "nx.DiGraph[int]", output: Path, highlight: Sequence[int] = ()) -> None:
    """`draw_static`'s log-normalized sibling: same picture, but edge color interpolates on
    `log(count)` instead of `count` (see `_log_frequency_color`) so a graph with one or two
    outlier-frequency transitions doesn't collapse the rest of the picture to a uniform pale gray.
    A separate export, not a flag on `draw_static`, so the linear reading it already produces is
    never disturbed by this one."""
    highlighted_nodes = set(highlight)
    highlighted_edges = _highlighted_edges(highlight)
    positions = _layout(graph)
    labels = {node: data["label"] for node, data in graph.nodes(data=True)}
    edge_colors_by_pair = log_edge_colors(graph)
    node_colors = [
        HIGHLIGHT_NODE_COLOR if node in highlighted_nodes else DEFAULT_NODE_COLOR
        for node in graph.nodes
    ]
    edge_colors = [edge_colors_by_pair[edge] for edge in graph.edges]
    edge_widths = [2.5 if edge in highlighted_edges else 0.6 for edge in graph.edges]

    figure, axis = plt.subplots(figsize=(14, 14))
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=350, ax=axis)
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=axis)
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        connectionstyle="arc3,rad=0.08",
        node_size=350,
        ax=axis,
    )
    axis.set_title("edge color: log(count), normalized to the graph's own range")
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _step_digits(steps: int) -> int:
    return max(len(str(max(steps - 1, 0))), 1)


def draw_red_cluster_evolution(
    graph: "nx.DiGraph[int]", curve: CoverageCurve, output_dir: Path
) -> None:
    """One static PNG per threshold in `curve` (pass `red_cluster_coverage_curve`'s own curve,
    computed on the SAME graph, so thresholds and node/edge counts correspond) — only the edges
    surviving `count >= threshold` are drawn, and the nodes among them that fall in the resulting
    largest weakly connected component (the metric's own answer at that threshold) are highlighted;
    a node touched by no surviving edge is dropped from the frame entirely rather than drawn
    isolated. Node positions are computed once on the full graph (`_layout`) and reused unchanged
    across every frame, so a node never moves between steps — only which edges and which nodes are
    still there changes, exactly what the curve itself is measuring."""
    positions = _layout(graph)
    labels = {node: data["label"] for node, data in graph.nodes(data=True)}
    edge_colors_by_pair = _edge_colors(graph)
    output_dir.mkdir(parents=True, exist_ok=True)
    digits = _step_digits(len(curve.thresholds))
    for index, threshold in enumerate(curve.thresholds):
        subgraph: nx.DiGraph[int] = nx.DiGraph()
        subgraph.add_edges_from(
            (source, target)
            for source, target, data in graph.edges(data=True)
            if data["count"] >= threshold
        )
        if subgraph.number_of_nodes() == 0:
            continue
        largest = max(nx.weakly_connected_components(subgraph), key=len)
        node_colors = [
            HIGHLIGHT_NODE_COLOR if node in largest else DEFAULT_NODE_COLOR
            for node in subgraph.nodes
        ]
        edge_colors = [edge_colors_by_pair[edge] for edge in subgraph.edges]
        sub_positions = {node: positions[node] for node in subgraph.nodes}
        sub_labels = {node: labels[node] for node in subgraph.nodes}

        figure, axis = plt.subplots(figsize=(10, 10))
        nx.draw_networkx_nodes(
            subgraph, sub_positions, node_color=node_colors, node_size=350, ax=axis
        )
        nx.draw_networkx_labels(subgraph, sub_positions, labels=sub_labels, font_size=7, ax=axis)
        nx.draw_networkx_edges(
            subgraph,
            sub_positions,
            edge_color=edge_colors,
            arrows=True,
            connectionstyle="arc3,rad=0.08",
            node_size=350,
            ax=axis,
        )
        axis.set_title(
            f"red-cluster — threshold >= {threshold:g} (coverage {curve.coverages[index]:.1%})"
        )
        axis.axis("off")
        figure.savefig(
            output_dir / f"step_{index:0{digits}d}_threshold_{threshold:g}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def draw_weighted_core_evolution(
    graph: "nx.DiGraph[int]",
    core_numbers: dict[int, float],
    curve: CoverageCurve,
    output_dir: Path,
) -> None:
    """One static PNG per level in `curve` (pass `weighted_core_coverage_curve`'s own curve,
    `core_numbers` from `weighted_core_numbers` on the SAME graph) — only the nodes whose weighted
    core number is >= that level are drawn, connected by whichever of their original edges runs
    between two surviving nodes. Unlike `draw_red_cluster_evolution`, no edge is itself
    thresholded: a node's core number is a property peeled from the whole graph, not a single
    edge's own weight, so an edge that individually looks weak can still connect two nodes that
    are each, on their own, well embedded enough to survive this level. Positions are the same
    fixed layout `draw_red_cluster_evolution` uses, so the two evolutions are visually comparable
    frame for frame."""
    positions = _layout(graph)
    labels = {node: data["label"] for node, data in graph.nodes(data=True)}
    edge_colors_by_pair = _edge_colors(graph)
    output_dir.mkdir(parents=True, exist_ok=True)
    digits = _step_digits(len(curve.thresholds))
    for index, level in enumerate(curve.thresholds):
        surviving = {node for node, value in core_numbers.items() if value >= level}
        subgraph = graph.subgraph(surviving)
        if subgraph.number_of_nodes() == 0:
            continue
        edge_colors = [edge_colors_by_pair[edge] for edge in subgraph.edges]
        sub_positions = {node: positions[node] for node in subgraph.nodes}
        sub_labels = {node: labels[node] for node in subgraph.nodes}

        figure, axis = plt.subplots(figsize=(10, 10))
        nx.draw_networkx_nodes(
            subgraph, sub_positions, node_color=HIGHLIGHT_NODE_COLOR, node_size=350, ax=axis
        )
        nx.draw_networkx_labels(subgraph, sub_positions, labels=sub_labels, font_size=7, ax=axis)
        nx.draw_networkx_edges(
            subgraph,
            sub_positions,
            edge_color=edge_colors,
            arrows=True,
            connectionstyle="arc3,rad=0.08",
            node_size=350,
            ax=axis,
        )
        axis.set_title(
            f"weighted-core — level >= {level:g} (coverage {curve.coverages[index]:.1%})"
        )
        axis.axis("off")
        figure.savefig(
            output_dir / f"step_{index:0{digits}d}_level_{level:g}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def draw_red_cluster_evolution_log(
    graph: "nx.DiGraph[int]", curve: CoverageCurve, output_dir: Path
) -> None:
    """`draw_red_cluster_evolution`'s log-normalized sibling: identical frames (same `curve`, same
    threshold-by-threshold subgraphs, same highlighted component), edge color log-scaled per frame
    the way `draw_static_log` does for the whole graph -- see `_log_frequency_color`. A separate
    output directory, not a flag on `draw_red_cluster_evolution`, so its own frames are
    untouched."""
    positions = _layout(graph)
    labels = {node: data["label"] for node, data in graph.nodes(data=True)}
    edge_colors_by_pair = log_edge_colors(graph)
    output_dir.mkdir(parents=True, exist_ok=True)
    digits = _step_digits(len(curve.thresholds))
    for index, threshold in enumerate(curve.thresholds):
        subgraph: nx.DiGraph[int] = nx.DiGraph()
        subgraph.add_edges_from(
            (source, target)
            for source, target, data in graph.edges(data=True)
            if data["count"] >= threshold
        )
        if subgraph.number_of_nodes() == 0:
            continue
        largest = max(nx.weakly_connected_components(subgraph), key=len)
        node_colors = [
            HIGHLIGHT_NODE_COLOR if node in largest else DEFAULT_NODE_COLOR
            for node in subgraph.nodes
        ]
        edge_colors = [edge_colors_by_pair[edge] for edge in subgraph.edges]
        sub_positions = {node: positions[node] for node in subgraph.nodes}
        sub_labels = {node: labels[node] for node in subgraph.nodes}

        figure, axis = plt.subplots(figsize=(10, 10))
        nx.draw_networkx_nodes(
            subgraph, sub_positions, node_color=node_colors, node_size=350, ax=axis
        )
        nx.draw_networkx_labels(subgraph, sub_positions, labels=sub_labels, font_size=7, ax=axis)
        nx.draw_networkx_edges(
            subgraph,
            sub_positions,
            edge_color=edge_colors,
            arrows=True,
            connectionstyle="arc3,rad=0.08",
            node_size=350,
            ax=axis,
        )
        axis.set_title(
            f"red-cluster — threshold >= {threshold:g} (coverage {curve.coverages[index]:.1%})"
        )
        axis.axis("off")
        figure.savefig(
            output_dir / f"step_{index:0{digits}d}_threshold_{threshold:g}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def draw_weighted_core_evolution_log(
    graph: "nx.DiGraph[int]",
    core_numbers: dict[int, float],
    curve: CoverageCurve,
    output_dir: Path,
) -> None:
    """`draw_weighted_core_evolution`'s log-normalized sibling -- see
    `draw_red_cluster_evolution_log`. `core_numbers` and `curve` are unchanged (peeling and
    coverage are unaffected by how the picture colors an edge); only `edge_colors_by_pair` reads
    the log-scaled palette."""
    positions = _layout(graph)
    labels = {node: data["label"] for node, data in graph.nodes(data=True)}
    edge_colors_by_pair = log_edge_colors(graph)
    output_dir.mkdir(parents=True, exist_ok=True)
    digits = _step_digits(len(curve.thresholds))
    for index, level in enumerate(curve.thresholds):
        surviving = {node for node, value in core_numbers.items() if value >= level}
        subgraph = graph.subgraph(surviving)
        if subgraph.number_of_nodes() == 0:
            continue
        edge_colors = [edge_colors_by_pair[edge] for edge in subgraph.edges]
        sub_positions = {node: positions[node] for node in subgraph.nodes}
        sub_labels = {node: labels[node] for node in subgraph.nodes}

        figure, axis = plt.subplots(figsize=(10, 10))
        nx.draw_networkx_nodes(
            subgraph, sub_positions, node_color=HIGHLIGHT_NODE_COLOR, node_size=350, ax=axis
        )
        nx.draw_networkx_labels(subgraph, sub_positions, labels=sub_labels, font_size=7, ax=axis)
        nx.draw_networkx_edges(
            subgraph,
            sub_positions,
            edge_color=edge_colors,
            arrows=True,
            connectionstyle="arc3,rad=0.08",
            node_size=350,
            ax=axis,
        )
        axis.set_title(
            f"weighted-core — level >= {level:g} (coverage {curve.coverages[index]:.1%})"
        )
        axis.axis("off")
        figure.savefig(
            output_dir / f"step_{index:0{digits}d}_level_{level:g}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def _step_points(curve: CoverageCurve) -> tuple[list[float], list[float]]:
    """The `(x, y)` points `_draw_curves` plots with `drawstyle="steps-pre"`, threshold normalized
    to `[0, 1]` by the curve's own maximum. `coverages[i]` IS the coverage measured EXACTLY at
    `thresholds[i]` (`red_cluster_coverage`/`weighted_core_numbers` say so directly, and
    `_area_under_curve` relies on the same fact) -- so every marker at `x = thresholds[i]/maximum`
    must sit at `y = coverages[i]`, its own true value, not a neighbor's. `steps-pre` is what makes
    that same array of points integrate to the right area too: it holds each segment's height at
    its RIGHT point (`(x[i-1], x[i]]` takes `y[i]`), matching how coverage actually behaves here --
    unchanged up to and including the threshold that produced it, dropping only strictly after. The
    leading `(0, coverages[0])` point stands in for "at or below the lowest observed threshold",
    where coverage is already at its least restrictive value.

    Pairing this same point layout with `steps-post` instead (an earlier version of this function
    did, reasoning from the AUC integral's own index arithmetic rather than checking what value
    sits at each x) is subtly wrong in a way that doesn't show up in the area at all, only in where
    each drop visually lands: `steps-post` reads a segment's height off its LEFT point, so it
    draws every threshold's marker one step early, showing the value that will apply only once you
    go PAST that threshold. Both give the identical, correct `auc`, since a single point has zero
    width and can't move the integral -- only `steps-pre` also puts every dot where it belongs.
    Callers must not call this with an empty curve (`thresholds == ()`)."""
    maximum = curve.thresholds[-1]
    x = [0.0, *(threshold / maximum for threshold in curve.thresholds)]
    y = [curve.coverages[0], *curve.coverages]
    return x, y


def _draw_curves(
    curves: Sequence[tuple[str, CoverageCurve, str, str]], title: str, output: Path
) -> None:
    """Shared plotting for `draw_coverage_curves` and `draw_document_coverage_curves`: any number
    of `(label, curve, color, linestyle)` lines, threshold normalized to `[0, 1]` by each curve's
    own maximum so curves with different raw ranges are directly comparable. Each line's label
    carries its `auc`, the single number `analysis.markov_chain`'s module docstring
    describes -- so the picture and the table it appears beside always agree on that number
    without a reader having to compute it by eye off the curve.

    A curve with no points at all (`thresholds == ()`, an empty or edgeless graph, or -- expected
    for a reciprocal curve on a purely sequential corpus, or a document-support curve when nothing
    is genuinely spread across more than one document -- no pair reaching a nonzero level at all)
    is skipped rather than drawn as a flat line at 0, which would claim a measurement that was
    never taken.
    """
    figure, axis = plt.subplots(figsize=(8, 5))
    drew_a_curve = False
    for label, curve, color, linestyle in curves:
        if not curve.thresholds:
            continue
        x, y = _step_points(curve)
        axis.plot(
            x,
            y,
            marker="o",
            markersize=3,
            color=color,
            linestyle=linestyle,
            # Coverage only ever drops exactly at an observed threshold and is flat in between
            # (see `_area_under_curve`) -- a step function, not a smooth line between the points.
            # "steps-pre" (not "steps-post" -- see `_step_points`) is what keeps each marker's
            # x-position showing that same threshold's own true value.
            drawstyle="steps-pre",
            label=f"{label} (AUC={curve.auc:.3f})",
        )
        drew_a_curve = True
    axis.set_xlabel("threshold, normalized to its own maximum")
    axis.set_ylabel("coverage")
    axis.set_ylim(-0.02, 1.05)
    axis.set_title(title)
    if drew_a_curve:
        axis.legend(fontsize=8, loc="lower left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def draw_coverage_curves(
    red_cluster: CoverageCurve,
    weighted_core: CoverageCurve,
    reciprocal_cluster: CoverageCurve,
    reciprocal_core: CoverageCurve,
    order: int,
    output: Path,
) -> None:
    """The raw-`count` family of four coverage curves for one Markov order on one axis: how much
    intensity (red-cluster/weighted-core, solid), and how much of it is direction-aware
    (dashed): transitions observed both ways (reciprocal-cluster) and tokens that keep both
    receiving and emitting flow, i.e. directed cycles (reciprocal-core). See `_draw_curves` and
    `analysis.markov_chain`'s module docstring for the shared plotting rules and what each
    curve and its `auc` mean.
    """
    _draw_curves(
        [
            ("red-cluster coverage", red_cluster, "tab:red", "-"),
            ("weighted-core coverage", weighted_core, "tab:blue", "-"),
            ("reciprocal-cluster coverage", reciprocal_cluster, "tab:orange", "--"),
            ("reciprocal-core coverage", reciprocal_core, "tab:green", "--"),
        ],
        f"Coverage curves — order {order}",
        output,
    )


def draw_document_coverage_curves(
    document_cluster: CoverageCurve,
    document_core: CoverageCurve,
    document_reciprocal_cluster: CoverageCurve,
    document_reciprocal_core: CoverageCurve,
    order: int,
    output: Path,
) -> None:
    """The `effective_document_support` family of the same four curves, on its own axis rather
    than mixed into `draw_coverage_curves`'s: how much of a transition's evidence is spread evenly
    across DISTINCT documents (document-cluster/document-core, solid) versus concentrated in a few
    -- or unevenly split even among the documents it does touch -- and the same two
    direction-aware readings of that spread (document-reciprocal-cluster: both ways;
    document-reciprocal-core: flow through cycles; dashed). Kept as a
    separate picture because it answers a different question from the raw-`count` family, not a
    refinement of it -- a high-`count` curve and a low-`effective_document_support` curve on the
    same graph both describe real, simultaneously true properties, and overlaying all eight lines
    on one axis would obscure that rather than clarify it. See
    `analysis.markov_chain`'s module docstring for the full derivation.
    """
    _draw_curves(
        [
            ("document-cluster coverage", document_cluster, "tab:purple", "-"),
            ("document-core coverage", document_core, "tab:cyan", "-"),
            (
                "document-reciprocal-cluster coverage",
                document_reciprocal_cluster,
                "tab:brown",
                "--",
            ),
            ("document-reciprocal-core coverage", document_reciprocal_core, "tab:olive", "--"),
        ],
        f"Document-support coverage curves — order {order}",
        output,
    )


def _step_points_log(curve: CoverageCurve) -> tuple[list[float], list[float]]:
    """`_step_points`'s log-normalized sibling, for a curve already converted with
    `analysis.markov_chain.to_log_curve` -- `y` is unchanged (the measured coverages
    themselves never depend on how the x-axis is scaled), only `x` reads
    `log(threshold) / log(maximum)` instead of `threshold / maximum`, matching exactly the
    normalization `_log_area_under_curve` used to compute that curve's own `auc` -- so this must
    stay paired with a log-converted curve, or the picture's shape and the AUC in its label will
    disagree, the same failure mode `_step_points`' own docstring documents for the linear case.
    Threshold `0` (a weighted core number with no non-self-loop edges) clamps to `1` for the same
    reason `_log_area_under_curve` does. Callers must not call this with an empty curve."""
    maximum = curve.thresholds[-1]
    log_maximum = math.log(maximum) if maximum > 1 else 1.0
    x = [0.0, *(math.log(max(threshold, 1.0)) / log_maximum for threshold in curve.thresholds)]
    y = [curve.coverages[0], *curve.coverages]
    return x, y


def _draw_curves_log(
    curves: Sequence[tuple[str, CoverageCurve, str, str]], title: str, output: Path
) -> None:
    """`_draw_curves`'s log-normalized sibling: identical plotting mechanics, but points come from
    `_step_points_log` instead of `_step_points`, so callers must pass curves already converted
    with `analysis.markov_chain.to_log_curve` (see that function, and
    `analysis.markov_chain._log_area_under_curve`, for what this trades against the linear
    reading `_draw_curves` gives). A separate function, not a flag on `_draw_curves`, so the
    linear picture it already produces for the 13 analyses run so far is never touched."""
    figure, axis = plt.subplots(figsize=(8, 5))
    drew_a_curve = False
    for label, curve, color, linestyle in curves:
        if not curve.thresholds:
            continue
        x, y = _step_points_log(curve)
        axis.plot(
            x,
            y,
            marker="o",
            markersize=3,
            color=color,
            linestyle=linestyle,
            drawstyle="steps-pre",
            label=f"{label} (AUC={curve.auc:.3f})",
        )
        drew_a_curve = True
    axis.set_xlabel("log(threshold), normalized to its own maximum (0 clamped to threshold=1)")
    axis.set_ylabel("coverage")
    axis.set_ylim(-0.02, 1.05)
    axis.set_title(title)
    if drew_a_curve:
        axis.legend(fontsize=8, loc="lower left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def draw_coverage_curves_log(
    red_cluster: CoverageCurve,
    weighted_core: CoverageCurve,
    reciprocal_cluster: CoverageCurve,
    reciprocal_core: CoverageCurve,
    order: int,
    output: Path,
) -> None:
    """`draw_coverage_curves`'s log-normalized sibling: same four curves' own measured points,
    each converted with `to_log_curve` so both the plotted x-axis and the `AUC` in the legend read
    log(threshold) instead of threshold -- see `analysis.markov_chain.to_log_curve` for
    why this is a second, independent reading of the curve rather than a correction to the first.
    Callers pass the same plain curves `draw_coverage_curves` takes; the conversion happens here."""
    _draw_curves_log(
        [
            ("red-cluster coverage", to_log_curve(red_cluster), "tab:red", "-"),
            ("weighted-core coverage", to_log_curve(weighted_core), "tab:blue", "-"),
            ("reciprocal-cluster coverage", to_log_curve(reciprocal_cluster), "tab:orange", "--"),
            ("reciprocal-core coverage", to_log_curve(reciprocal_core), "tab:green", "--"),
        ],
        f"Coverage curves — order {order} (log-normalized)",
        output,
    )


def draw_document_coverage_curves_log(
    document_cluster: CoverageCurve,
    document_core: CoverageCurve,
    document_reciprocal_cluster: CoverageCurve,
    document_reciprocal_core: CoverageCurve,
    order: int,
    output: Path,
) -> None:
    """`draw_document_coverage_curves`'s log-normalized sibling -- see `draw_coverage_curves_log`
    and `analysis.markov_chain.to_log_curve`."""
    _draw_curves_log(
        [
            ("document-cluster coverage", to_log_curve(document_cluster), "tab:purple", "-"),
            ("document-core coverage", to_log_curve(document_core), "tab:cyan", "-"),
            (
                "document-reciprocal-cluster coverage",
                to_log_curve(document_reciprocal_cluster),
                "tab:brown",
                "--",
            ),
            (
                "document-reciprocal-core coverage",
                to_log_curve(document_reciprocal_core),
                "tab:olive",
                "--",
            ),
        ],
        f"Document-support coverage curves — order {order} (log-normalized)",
        output,
    )
