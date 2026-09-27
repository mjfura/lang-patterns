"""`analysis.markov_graph`: exporting a corpus's order-1 token transition graph as static
images, with an optional document walk highlighted. The interactive-page export moved to
`analysis.neo4j_export` (see `tests/test_neo4j_export.py`).
"""

from pathlib import Path

import pytest

from corpus_generalization.analysis.markov_chain import (
    CoverageCurve,
    build_transition_graph,
    document_cluster_coverage_curve,
    document_core_coverage_curve,
    document_reciprocal_cluster_coverage_curve,
    document_reciprocal_core_coverage_curve,
    reciprocal_cluster_coverage_curve,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    to_log_curve,
    weighted_core_coverage_curve,
    weighted_core_numbers,
)
from corpus_generalization.analysis.markov_graph import (
    HIGH_FREQUENCY_COLOR,
    LOW_FREQUENCY_COLOR,
    _frequency_color,
    _log_frequency_color,
    _step_points,
    _step_points_log,
    document_path,
    draw_coverage_curves,
    draw_coverage_curves_log,
    draw_document_coverage_curves,
    draw_document_coverage_curves_log,
    draw_red_cluster_evolution,
    draw_red_cluster_evolution_log,
    draw_static,
    draw_static_log,
    draw_weighted_core_evolution,
    draw_weighted_core_evolution_log,
    log_edge_colors,
    log_normalized_edge_values,
    token_graph,
)


class WordIdTokenizer:
    """One id per distinct whitespace-separated word; `decode` reverses the mapping, joined by
    spaces, close enough to a real tokenizer's round trip for these tests. `bos_id`/`eos_id` are
    negative so they never collide with a real word's id, which always starts at 0."""

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


def test_token_graph_has_a_node_per_distinct_token_with_a_readable_label() -> None:
    """`<bos>`/`<eos>` are included too (every document is framed before the graph is built) with
    their own readable labels, not the empty string `decode` would otherwise give them."""
    documents = ("a b c", "a b d")
    tokenizer = WordIdTokenizer()

    graph = token_graph(documents, tokenizer)

    labels = {data["label"] for _, data in graph.nodes(data=True)}
    assert labels == {"a", "b", "c", "d", "<bos>", "<eos>"}


def test_token_graph_edges_carry_the_observed_weight_and_count() -> None:
    documents = ("a b", "a b")

    graph = token_graph(documents, WordIdTokenizer())

    edge = next(iter(graph.edges(data=True)))
    assert edge[2]["count"] == 2
    assert edge[2]["weight"] == 1.0


def test_document_path_is_the_documents_own_token_sequence() -> None:
    """Framed the same way `token_graph` frames every document, so the walk starts at `<bos>` and
    ends at `<eos>` -- both graph and highlighted path agree on where a document's edges begin and
    end."""
    tokenizer = WordIdTokenizer()

    path = document_path("a b c", tokenizer)

    # first-seen order: a=0, b=1, c=2; bos_id=-1, eos_id=-2
    assert path == [-1, 0, 1, 2, -2]


def test_frequency_color_is_pale_gray_at_the_low_end_and_intense_red_at_the_high_end() -> None:
    assert _frequency_color(1, minimum=1, maximum=5) == LOW_FREQUENCY_COLOR
    assert _frequency_color(5, minimum=1, maximum=5) == HIGH_FREQUENCY_COLOR


def test_frequency_color_interpolates_in_between() -> None:
    low = _frequency_color(1, minimum=1, maximum=5)
    middle = _frequency_color(3, minimum=1, maximum=5)
    high = _frequency_color(5, minimum=1, maximum=5)

    assert low != middle != high != low


def test_frequency_color_with_no_variance_is_still_defined() -> None:
    """Every edge equally frequent: nothing to contrast, so this must not divide by zero."""
    color = _frequency_color(4, minimum=4, maximum=4)

    assert color.startswith("#") and len(color) == 7


def test_token_graph_gives_more_frequent_edges_a_more_intense_color() -> None:
    """`<bos>`/`<eos>` bring their own edges (`<bos>`->a: count 4, every document starts with
    "a") into the same min/max range the color interpolation reads off the whole graph -- real,
    not a test artifact, so this checks the two content edges specifically rather than the color
    count overall."""
    documents = ("a b", "a b", "a b", "a c")
    tokenizer = WordIdTokenizer()

    graph = token_graph(documents, tokenizer)

    a, b, c = tokenizer.encode("a")[0], tokenizer.encode("b")[0], tokenizer.encode("c")[0]
    counts = {(u, v): data["count"] for u, v, data in graph.edges(data=True)}
    minimum, maximum = min(counts.values()), max(counts.values())
    colors = {edge: _frequency_color(count, minimum, maximum) for edge, count in counts.items()}
    assert colors[(a, b)] != colors[(a, c)]  # a->b (count 3) and a->c (count 1) differ


def test_draw_static_writes_a_nonempty_image(tmp_path: Path) -> None:
    documents = ("a b c", "a b d")
    graph = token_graph(documents, WordIdTokenizer())
    output = tmp_path / "graph.png"

    draw_static(graph, output)

    assert output.is_file()
    assert output.stat().st_size > 0


def _step_function_area(x: list[float], y: list[float]) -> float:
    """The area a reader's eye integrates off a `steps-pre` plot: each segment's height is its
    RIGHT point's y, i.e. `(x[i-1], x[i]]` takes `y[i]` -- independent of, and so a genuine check
    against, `_area_under_curve`'s own arithmetic."""
    return sum(y[index] * (x[index] - x[index - 1]) for index in range(1, len(x)))


def test_step_points_area_matches_the_curves_own_auc() -> None:
    """`_step_points` + `steps-pre` must integrate to exactly the curve's own `auc`, for a curve
    with more than one point (need at least one segment for there to be anything to get wrong)."""
    curve = CoverageCurve(thresholds=(2.0, 5.0, 10.0), coverages=(1.0, 0.6, 0.2), auc=0.48)

    x, y = _step_points(curve)

    assert _step_function_area(x, y) == pytest.approx(curve.auc)


def test_step_points_area_matches_auc_for_every_real_coverage_curve() -> None:
    """Same check as above, but end to end: every curve the pipeline actually produces off a real
    graph (not a hand-built `CoverageCurve`) must still draw to exactly its own `auc`."""
    documents = ("a b c a b c a b c a", "d e", "f g")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    for curve in (
        red_cluster_coverage_curve(graph),
        weighted_core_coverage_curve(graph),
    ):
        x, y = _step_points(curve)
        assert _step_function_area(x, y) == pytest.approx(curve.auc)


def test_step_points_places_each_marker_at_its_own_thresholds_true_value() -> None:
    """A regression test for a real bug: two different point layouts both integrate to the
    correct `auc` (a single point has zero width, so it can't move the area either way), but only
    one of them puts each marker where it visually belongs. Pairing the original point layout with
    `steps-post` instead of `steps-pre` drew every threshold's dot one step early -- showing the
    value that applies only once you go PAST that threshold, not the value the threshold's own
    `coverages[i]` actually reports for it (`red_cluster_coverage(graph, thresholds[i])` says so
    directly). `_step_points` must place `y[k]` at `x[k]` (`k=1..len(thresholds)`, since `x[0]=0`
    is the artificial "at or below the minimum" anchor) equal to `coverages[k-1]`, its own true
    reading, not a neighboring threshold's."""
    curve = CoverageCurve(thresholds=(2.0, 5.0, 10.0), coverages=(1.0, 0.6, 0.2), auc=0.48)

    x, y = _step_points(curve)

    for index, threshold in enumerate(curve.thresholds):
        marker_index = index + 1  # x[0] is the leading (0, coverages[0]) anchor
        assert x[marker_index] == pytest.approx(threshold / curve.thresholds[-1])
        assert y[marker_index] == curve.coverages[index]


def test_draw_coverage_curves_writes_a_nonempty_image(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves.png"

    draw_coverage_curves(
        red_cluster_coverage_curve(graph),
        weighted_core_coverage_curve(graph),
        reciprocal_cluster_coverage_curve(graph),
        reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_draw_coverage_curves_tolerates_an_edgeless_graph(tmp_path: Path) -> None:
    """None of the four curves has any points to plot; the call must still produce a file, not
    raise."""
    graph = build_transition_graph(("a",), WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves.png"

    draw_coverage_curves(
        red_cluster_coverage_curve(graph),
        weighted_core_coverage_curve(graph),
        reciprocal_cluster_coverage_curve(graph),
        reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()


def test_draw_coverage_curves_tolerates_a_purely_sequential_graph(tmp_path: Path) -> None:
    """A purely sequential corpus (no transition ever reversed) leaves both reciprocal curves
    empty while red-cluster and weighted-core still have real points -- the call must draw the
    two that exist and skip the two that don't, not fail on the mix."""
    graph = build_transition_graph(("a b c",), WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves.png"

    draw_coverage_curves(
        red_cluster_coverage_curve(graph),
        weighted_core_coverage_curve(graph),
        reciprocal_cluster_coverage_curve(graph),
        reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_draw_document_coverage_curves_writes_a_nonempty_image(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves_documents.png"

    draw_document_coverage_curves(
        document_cluster_coverage_curve(graph),
        document_core_coverage_curve(graph),
        document_reciprocal_cluster_coverage_curve(graph),
        document_reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_draw_document_coverage_curves_tolerates_an_edgeless_graph(tmp_path: Path) -> None:
    """None of the four document-support curves has any points to plot; the call must still
    produce a file, not raise."""
    graph = build_transition_graph(("a",), WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves_documents.png"

    draw_document_coverage_curves(
        document_cluster_coverage_curve(graph),
        document_core_coverage_curve(graph),
        document_reciprocal_cluster_coverage_curve(graph),
        document_reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()


def test_draw_red_cluster_evolution_writes_one_png_per_threshold(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    tokenizer = WordIdTokenizer()
    graph = token_graph(documents, tokenizer)
    order1 = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    curve = red_cluster_coverage_curve(order1)
    output_dir = tmp_path / "red_cluster"

    draw_red_cluster_evolution(graph, curve, output_dir)

    files = sorted(output_dir.iterdir())
    assert len(files) == len(curve.thresholds)
    assert all(file.stat().st_size > 0 for file in files)


def test_draw_weighted_core_evolution_writes_one_png_per_level(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    tokenizer = WordIdTokenizer()
    graph = token_graph(documents, tokenizer)
    order1 = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    core_by_token = {token: value for (token,), value in weighted_core_numbers(order1).items()}
    curve = weighted_core_coverage_curve(order1)
    output_dir = tmp_path / "weighted_core"

    draw_weighted_core_evolution(graph, core_by_token, curve, output_dir)

    files = sorted(output_dir.iterdir())
    assert len(files) == len(curve.thresholds)
    assert all(file.stat().st_size > 0 for file in files)


def test_draw_evolutions_tolerate_an_edgeless_graph(tmp_path: Path) -> None:
    """No document at all -- `<bos>`/`<eos>` framing means even a single one-word document already
    has an edge, so this is the one case with truly no threshold/level to sweep
    (`curve.thresholds == ()`); the call must still succeed and simply write nothing, not raise."""
    tokenizer = WordIdTokenizer()
    graph = token_graph((), tokenizer)
    order1 = build_transition_graph((), tokenizer, order=1, include_boundaries=True)
    red_curve = red_cluster_coverage_curve(order1)
    core_by_token = {token: value for (token,), value in weighted_core_numbers(order1).items()}
    core_curve = weighted_core_coverage_curve(order1)

    draw_red_cluster_evolution(graph, red_curve, tmp_path / "red_cluster")
    draw_weighted_core_evolution(graph, core_by_token, core_curve, tmp_path / "weighted_core")

    assert list((tmp_path / "red_cluster").iterdir()) == []
    assert list((tmp_path / "weighted_core").iterdir()) == []


def test_layout_is_deterministic_so_positions_never_drift_between_frames() -> None:
    """`draw_*_evolution` calls `_layout` once per export and reuses that one dict for every
    frame, so within a single call a node cannot drift by construction; this instead guards the
    seed itself -- two independent layouts of the same graph must still agree, or a future change
    to `_layout` could silently make separate exports (e.g. re-running the analysis) place the
    same node differently."""
    from corpus_generalization.analysis.markov_graph import _layout

    documents = ("a b c a b c a b c a", "d e", "f g")
    graph = token_graph(documents, WordIdTokenizer())

    first = _layout(graph)
    second = _layout(graph)

    assert set(first) == set(graph.nodes)
    for node in graph.nodes:
        assert tuple(first[node]) == tuple(second[node])


def test_draw_static_marks_the_highlighted_path_distinctly(tmp_path: Path) -> None:
    """Edge color is reserved for frequency, so a highlighted path has to show up some other way
    -- width, and a distinct node color -- rather than color; the rendered picture must differ
    from the one drawn without a highlight."""
    documents = ("a b c", "a b d")
    tokenizer = WordIdTokenizer()
    graph = token_graph(documents, tokenizer)
    highlighted = document_path("a b c", tokenizer)
    output_plain = tmp_path / "plain.png"
    output_highlighted = tmp_path / "highlighted.png"

    draw_static(graph, output_plain)
    draw_static(graph, output_highlighted, highlight=highlighted)

    assert output_plain.read_bytes() != output_highlighted.read_bytes()


def test_log_frequency_color_is_pale_gray_at_the_low_end_and_intense_red_at_the_high_end() -> None:
    assert _log_frequency_color(1, minimum=1, maximum=5) == LOW_FREQUENCY_COLOR
    assert _log_frequency_color(5, minimum=1, maximum=5) == HIGH_FREQUENCY_COLOR


def test_log_frequency_color_with_no_variance_is_still_defined() -> None:
    color = _log_frequency_color(4, minimum=4, maximum=4)

    assert color.startswith("#") and len(color) == 7


def test_log_frequency_color_keeps_low_counts_distinguishable_against_a_huge_outlier_maximum() -> (
    None
):
    """The problem the log-normalized palette exists to fix: against a maximum of 1000, the linear
    interpolation puts a count of 2 exactly AT `LOW_FREQUENCY_COLOR` (`(2-1)/(1000-1)` rounds to
    nothing), the same shade every other low-but-nonzero count also gets -- the whole graph but for
    the outlier reads as one uniform pale gray. The log version spreads that same low end back out,
    so a count of 2 already reads as visibly more intense than `LOW_FREQUENCY_COLOR`."""
    assert _frequency_color(2, minimum=1, maximum=1000) == LOW_FREQUENCY_COLOR
    assert _log_frequency_color(2, minimum=1, maximum=1000) != LOW_FREQUENCY_COLOR


def test_draw_static_log_writes_a_nonempty_image(tmp_path: Path) -> None:
    documents = ("a b c", "a b d")
    graph = token_graph(documents, WordIdTokenizer())
    output = tmp_path / "graph_log.png"

    draw_static_log(graph, output)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_log_edge_colors_matches_log_frequency_color_per_edge() -> None:
    """`log_edge_colors` (public, reused by `analysis.neo4j_export`) must compute exactly
    what `_log_frequency_color` would per edge, off the same graph-wide min/max -- no independent
    or drifted logic."""
    documents = ("a b", "a b", "a b", "a c")
    graph = token_graph(documents, WordIdTokenizer())

    colors = log_edge_colors(graph)

    counts = {(u, v): data["count"] for u, v, data in graph.edges(data=True)}
    minimum, maximum = min(counts.values()), max(counts.values())
    for edge, count in counts.items():
        assert colors[edge] == _log_frequency_color(count, minimum, maximum)


def test_log_normalized_edge_values_are_the_same_position_the_color_was_derived_from() -> None:
    """`log_normalized_edge_values` and `log_edge_colors` must agree edge for edge: the color is
    built FROM this exact number, not a second, independently-normalized one -- the property
    `analysis.neo4j_export` relies on to store `log_value` and `color` consistently."""
    documents = ("a b", "a b", "a b", "a c")
    tokenizer = WordIdTokenizer()
    graph = token_graph(documents, tokenizer)

    values = log_normalized_edge_values(graph)
    colors = log_edge_colors(graph)

    assert set(values) == set(colors)
    for value in values.values():
        assert 0.0 <= value <= 1.0
    a, b, c = tokenizer.encode("a")[0], tokenizer.encode("b")[0], tokenizer.encode("c")[0]
    # a->b (count 3) is more log-intense than a->c (count 1): a higher normalized position.
    assert values[(a, b)] > values[(a, c)]


def _step_function_area_log(x: list[float], y: list[float]) -> float:
    return sum(y[index] * (x[index] - x[index - 1]) for index in range(1, len(x)))


def test_step_points_log_area_matches_the_log_curves_own_auc() -> None:
    """`_step_points_log` + `steps-pre` must integrate to exactly the log-converted curve's own
    `auc` -- the log-scale analogue of `test_step_points_area_matches_the_curves_own_auc`."""
    curve = to_log_curve(
        CoverageCurve(thresholds=(2.0, 5.0, 10.0), coverages=(1.0, 0.6, 0.2), auc=0.48)
    )

    x, y = _step_points_log(curve)

    assert _step_function_area_log(x, y) == pytest.approx(curve.auc)


def test_step_points_log_area_matches_auc_for_every_real_coverage_curve() -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)

    for curve in (
        to_log_curve(red_cluster_coverage_curve(graph)),
        to_log_curve(weighted_core_coverage_curve(graph)),
    ):
        x, y = _step_points_log(curve)
        assert _step_function_area_log(x, y) == pytest.approx(curve.auc)


def test_draw_coverage_curves_log_writes_a_nonempty_image(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves_log.png"

    draw_coverage_curves_log(
        red_cluster_coverage_curve(graph),
        weighted_core_coverage_curve(graph),
        reciprocal_cluster_coverage_curve(graph),
        reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_draw_coverage_curves_log_tolerates_an_edgeless_graph(tmp_path: Path) -> None:
    graph = build_transition_graph(("a",), WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves_log.png"

    draw_coverage_curves_log(
        red_cluster_coverage_curve(graph),
        weighted_core_coverage_curve(graph),
        reciprocal_cluster_coverage_curve(graph),
        reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()


def test_draw_document_coverage_curves_log_writes_a_nonempty_image(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    graph = build_transition_graph(documents, WordIdTokenizer(), order=1)
    output = tmp_path / "coverage_curves_documents_log.png"

    draw_document_coverage_curves_log(
        document_cluster_coverage_curve(graph),
        document_core_coverage_curve(graph),
        document_reciprocal_cluster_coverage_curve(graph),
        document_reciprocal_core_coverage_curve(graph),
        order=1,
        output=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_draw_red_cluster_evolution_log_writes_one_png_per_threshold(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    tokenizer = WordIdTokenizer()
    graph = token_graph(documents, tokenizer)
    order1 = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    curve = red_cluster_coverage_curve(order1)
    output_dir = tmp_path / "red_cluster_log"

    draw_red_cluster_evolution_log(graph, curve, output_dir)

    files = sorted(output_dir.iterdir())
    assert len(files) == len(curve.thresholds)
    assert all(file.stat().st_size > 0 for file in files)


def test_draw_weighted_core_evolution_log_writes_one_png_per_level(tmp_path: Path) -> None:
    documents = ("a b c a b c a b c a", "d e", "f g")
    tokenizer = WordIdTokenizer()
    graph = token_graph(documents, tokenizer)
    order1 = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    core_by_token = {token: value for (token,), value in weighted_core_numbers(order1).items()}
    curve = weighted_core_coverage_curve(order1)
    output_dir = tmp_path / "weighted_core_log"

    draw_weighted_core_evolution_log(graph, core_by_token, curve, output_dir)

    files = sorted(output_dir.iterdir())
    assert len(files) == len(curve.thresholds)
    assert all(file.stat().st_size > 0 for file in files)


def test_draw_evolutions_log_tolerate_an_edgeless_graph(tmp_path: Path) -> None:
    tokenizer = WordIdTokenizer()
    graph = token_graph((), tokenizer)
    order1 = build_transition_graph((), tokenizer, order=1, include_boundaries=True)
    red_curve = red_cluster_coverage_curve(order1)
    core_by_token = {token: value for (token,), value in weighted_core_numbers(order1).items()}
    core_curve = weighted_core_coverage_curve(order1)

    draw_red_cluster_evolution_log(graph, red_curve, tmp_path / "red_cluster_log")
    draw_weighted_core_evolution_log(
        graph, core_by_token, core_curve, tmp_path / "weighted_core_log"
    )

    assert list((tmp_path / "red_cluster_log").iterdir()) == []
    assert list((tmp_path / "weighted_core_log").iterdir()) == []
