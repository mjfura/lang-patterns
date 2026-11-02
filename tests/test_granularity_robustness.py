"""The robustness experiment's rules: variants, the pool layout, the clustering switch, workers and
judging. Pure functions, exercised without downloading or reading a corpus."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def rb() -> ModuleType:
    path = (
        Path(__file__).parents[1] / "experiments/granularity_robustness/granularity_robustness.py"
    )
    assert path.exists(), "granularity-robustness experiment has not been implemented"
    spec = importlib.util.spec_from_file_location("granularity_robustness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variant_names(rb: ModuleType) -> None:
    assert rb.Variant("wikipedia", 0.25, 1).name == "wikipedia@0.25#1"
    assert rb.Variant("tinystories", 4.0, 0).name == "tinystories@4#0"
    assert rb.parse_variant("pubmed@0.0625#2") == rb.Variant("pubmed", 0.0625, 2)
    assert len(rb.variants()) == 50 and len(rb.new_variants()) == 47
    assert sum(len(v) for v in rb.STEPS.values()) == 47
    assert len(rb.STEPS["small"]) == 30 and len(rb.STEPS["4x"]) == 5


@pytest.mark.parametrize("name", ["legal@4#1", "fineweb@2#0", "news@1#0", "tinystories@1#3", "x"])
def test_parse_variant_rejects(rb: ModuleType, name: str) -> None:
    with pytest.raises(ValueError):
        rb.parse_variant(name)


def test_layout_slices_are_nested_and_disjoint(rb: ModuleType) -> None:
    sizes = [10] * 100  # unit 40 -> 4 documents per 1x pool
    cut = rb.layout(sizes, 40, first=[0, 1, 2, 3], rest_from=4)
    assert cut[(1.0, 0)] == [0, 1, 2, 3]
    assert cut[(4.0, 0)] == list(range(16))
    assert cut[(1.0, 1)] == [16, 17, 18, 19] and cut[(1.0, 2)] == [20, 21, 22, 23]
    assert cut[(0.25, 1)] == [16] and cut[(0.0625, 2)] == [20]


def test_layout_fineweb_first_pool_is_not_a_prefix(rb: ModuleType) -> None:
    cut = rb.layout([10] * 100, 40, first=[1, 3, 5, 7], rest_from=10)
    assert cut[(4.0, 0)][:4] == [1, 3, 5, 7] and cut[(4.0, 0)][4] == 10


def test_layout_drops_pool_2_when_short(rb: ModuleType) -> None:
    cut = rb.layout([10] * 22, 40, first=[0, 1, 2, 3], rest_from=4)
    assert (1.0, 1) in cut and not any(k[1] == 2 for k in cut)
    with pytest.raises(ValueError):
        rb.layout([10] * 18, 40, first=[0, 1, 2, 3], rest_from=4)


def test_scipy_rule_bounds_the_measured_peaks(rb: ModuleType) -> None:
    assert rb.scipy_peak_bytes(11_459) >= 1.38e9 and rb.scipy_peak_bytes(15_893) >= 1.76e9
    assert rb.uses_scipy(57_944) and rb.uses_scipy(75_000) and not rb.uses_scipy(76_000)


def test_workers_for(rb: ModuleType) -> None:
    assert rb.workers_for(4 * 10**9, 2 * 10**9) == 9
    assert rb.workers_for(10 * 10**9, 5 * 10**9) == 7
    assert rb.workers_for(47 * 10**9, 5 * 10**9) == 1


def test_workers_from_follow_each_readings_own_tree(rb: ModuleType) -> None:
    gb = 10**9
    # After wikipedia@1#1 at 1.30 (38.5 GB parent), a 2.50 reading whose tree projects to 2 GB
    # still gets many workers: its parent peak follows its own tree, not the step's largest.
    seen = [(int(38.5 * gb), int(38.5 * gb), int(5.9 * gb))]
    assert rb.workers_from(2 * gb, seen) == 7
    # The largest overshoot of a measured parent over its projection is kept as a margin.
    seen.append((23 * gb, int(25.1 * gb), 5 * gb))
    assert rb.workers_from(23 * gb, seen) == 3
    assert rb.workers_from(46 * gb, seen) == 1


def test_legal_document_and_pubmed_chunk(rb: ModuleType) -> None:
    assert (
        rb.legal_document(["1. The applicant ", "", "2. He was"]) == "1. The applicant\n2. He was"
    )
    assert rb.pubmed_chunk(998) == "chunk/pubmed23n0998.jsonl"


def test_agrees_skips_unreachable(rb: ModuleType) -> None:
    base = {"1.30": True, "1.50": True, "2.20": False}
    assert rb.agrees({"1.30": None, "1.50": True, "2.20": False}, base)
    assert not rb.agrees({"1.30": None, "1.50": True, "2.20": True}, base)


def test_first_failure_above(rb: ModuleType) -> None:
    holds = {"1.50": False, "2.00": True, "2.20": True, "2.50": False}
    assert rb.first_failure_above(holds) == "2.50"
    assert rb.first_failure_above({"2.20": True, "2.50": None}) is None


def test_resolvable_leaves_out_words_missed_at_every_granularity(rb: ModuleType) -> None:
    classes = {"they": True, "make": True}
    judged = {
        # `oil` is missed wherever it is one token; `art` is found at 2.00 only.
        "1.50": {"classes_found": classes, "dropped_found": {"oil": False, "art": False}},
        "2.00": {"classes_found": classes, "dropped_found": {"oil": False, "art": True}},
        "2.50": {"classes_found": {"they": True, "make": False}, "dropped_found": {"art": True}},
    }
    resolvable = rb.resolvable(judged)
    assert {t: v["holds"] for t, v in resolvable.items()} == {
        "1.50": False,
        "2.00": True,
        "2.50": False,
    }
    assert resolvable["1.50"]["unresolved_drops"] == ["oil"]
    assert judged["2.00"] == {
        "classes_found": classes,
        "dropped_found": {"oil": False, "art": True},
    }


def test_clustering_restores_linkage(rb: ModuleType) -> None:
    from corpus_generalization.analysis import syntactic

    # `linkage` is SciPy's, imported into `syntactic`, not part of its own interface.
    original = getattr(syntactic, "linkage")  # noqa: B009
    with rb.clustering(80_000) as tree:
        assert tree == "vector" and getattr(syntactic, "linkage") is not original  # noqa: B009
    assert getattr(syntactic, "linkage") is original  # noqa: B009
    with rb.clustering(1_000) as tree:
        assert tree == "scipy" and getattr(syntactic, "linkage") is original  # noqa: B009


def test_prepare_refuses_done(rb: ModuleType) -> None:
    with pytest.raises(SystemExit):
        rb.prepare("fineweb@1#0")


def test_run_order(rb: ModuleType) -> None:
    order = rb.reading_order("small")
    assert order[0] == (rb.STEPS["small"][0], "2.50")
    assert order[-1] == (rb.STEPS["small"][-1], "1.30")
    assert len(order) == 30 * 11


def test_predictions_on_a_fixture(rb: ModuleType) -> None:
    def variant(corpus: str, holds_150: bool) -> dict[str, object]:
        return {"corpus": corpus, "size": 0.25, "pool": 0, "holds": {"1.50": holds_150}}

    summary = {"variants": {"a@0.25#0": variant("a", True), "b@0.25#0": variant("b", True)}}
    assert rb.predictions(summary)["1"] == "HELD"
    assert rb.predictions(summary)["4"] == "unreadable"
    summary["variants"]["b@0.25#0"] = variant("b", False)
    assert rb.predictions(summary)["1"] == "FAILED"


def test_predictions_ignore_variants_not_read(rb: ModuleType) -> None:
    empty = {"corpus": "x", "size": 4.0, "pool": 0, "read": 0, "holds": {}, "agrees": True}
    assert rb.predictions({"variants": {"x@4#0": empty}})["4"] == "unreadable"


def test_predictions_8_and_9_read_the_preparation(rb: ModuleType) -> None:
    pubmed = {"corpus": "pubmed", "size": 1.0, "pool": 0, "read": 0, "holds": {}}
    pubmed |= {"merges_at_1.50": 15_066, "replaced": False}
    result = rb.predictions({"variants": {"pubmed@1#0": pubmed}})
    assert result["8"] == "FAILED" and result["9"] == "HELD"
