"""`corpus_generalization.analyze`: the analysis wired end to end, with a fake encoder."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from corpus_generalization import analyze as script
from corpus_generalization import corpora

CORPUS = "toy-corpus"


class KeywordEncoder:
    """Two fixed directions, chosen by one word, so the grouping is decided by the text alone."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        self.calls += 1
        return np.array([[1.0, 0.0] if "shape" in text else [0.0, 1.0] for text in texts])


@pytest.fixture(autouse=True)
def analysis_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    directory = tmp_path / "analysis"
    monkeypatch.setattr(script, "ANALYSIS_DIR", directory)
    return directory


def configure(tmp_path: Path, documents: Sequence[str], **settings: object) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"{document}\n" for document in documents), encoding="utf-8")
    folder = corpora.CORPORA_CONFIG_DIR / CORPUS
    folder.mkdir(parents=True, exist_ok=True)
    (folder / script.CONFIG_FILENAME).write_text(
        json.dumps({"files": [str(corpus)], "merge_budget": 20, **settings})
    )


def configure_without_files(**settings: object) -> None:
    """For the `huggingface_dataset`-only configs: `configure` always writes a local `files`
    entry, which is exactly what these tests need to NOT have."""
    folder = corpora.CORPORA_CONFIG_DIR / CORPUS
    folder.mkdir(parents=True, exist_ok=True)
    (folder / script.CONFIG_FILENAME).write_text(json.dumps({"merge_budget": 20, **settings}))


class FakeHuggingFaceLoader:
    """Records every call instead of touching the network -- what `huggingface_loader=` in
    `script.main` exists for, the same role `encoder=` already plays for
    `SentenceTransformerEncoder`."""

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, str, str, int | None, str | None]] = []

    def __call__(
        self,
        name: str,
        split: str,
        text_field: str,
        limit: int | None,
        config_name: str | None = None,
    ) -> list[str]:
        self.calls.append((name, split, text_field, limit, config_name))
        return self.documents


class FakeNeo4jSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def run(self, query: str, parameters: dict[str, object] | None = None) -> None:
        self.calls.append((query, parameters))

    def __enter__(self) -> "FakeNeo4jSession":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class FakeNeo4jDriver:
    """Records every session opened instead of connecting to a real Neo4j instance -- what
    `neo4j_driver=` in `script.main` exists for, the same role `huggingface_loader=` already plays
    for `datasets`."""

    def __init__(self) -> None:
        self.sessions: list[FakeNeo4jSession] = []
        self.closed = False

    def session(self) -> FakeNeo4jSession:
        session = FakeNeo4jSession()
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


def test_the_analysis_reports_tokenization_and_two_domains_and_writes_its_report(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        use_semantic=True,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    domains = json.loads((snapshot / "domains.json").read_text())
    assert exit_code == 0
    assert "2 domains" in printed
    assert "entropy" in printed
    assert domains["domains"] == 2
    assert sorted(group["size"] for group in domains["groups"]) == [5, 5]
    assert (snapshot / "tokens.tsv").read_text().startswith("id\tkind\ttoken")
    assert "## Domains" in (snapshot / "report.md").read_text()


def test_the_analysis_reports_cohesion_and_writes_it(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    cohesion = json.loads((snapshot / "cohesion.json").read_text())
    assert exit_code == 0
    assert "CDCI" in printed
    assert "tokenizer fertility" in printed
    assert cohesion["documents"] == 10
    assert len(cohesion["per_document"]) == 10
    first = cohesion["per_document"][0]
    assert {"text", "tau", "ipbe", "fertility", "steps"} <= first.keys()
    assert "## Cohesion" in (snapshot / "report.md").read_text()


def test_the_analysis_reports_kl_divergence_and_writes_it(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    divergence = json.loads((snapshot / "divergence.json").read_text())
    assert exit_code == 0
    assert "KL divergence" in printed
    assert divergence["documents"] == 10
    assert len(divergence["per_document"]) == 10
    assert {"text", "kl_divergence"} <= divergence["per_document"][0].keys()
    assert "## KL divergence" in (snapshot / "report.md").read_text()


def test_the_analysis_reports_the_markov_chain_and_writes_the_graph(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        markov_orders=[1, 2],
        highlight_document="a shape number 0",
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    markov_stats = json.loads((snapshot / "markov_chain.json").read_text())
    assert exit_code == 0
    assert "markov order 1" in printed
    assert "markov order 2" in printed
    assert [entry["order"] for entry in markov_stats] == [1, 2]
    assert {"states", "entropy_bits", "well_supported_fraction", "components"} <= markov_stats[
        0
    ].keys()
    assert "## Markov chain" in (snapshot / "report.md").read_text()
    assert (snapshot / "graph.png").stat().st_size > 0
    assert (snapshot / "graph_log.png").stat().st_size > 0
    assert not (snapshot / "graph.html").exists()


def test_the_analysis_reports_coverage_curves_and_writes_a_png_per_order(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        markov_orders=[1, 2],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    markov_stats = json.loads((snapshot / "markov_chain.json").read_text())
    assert exit_code == 0
    assert "coverage curve AUC, order 1" in printed
    assert "coverage curve AUC, order 2" in printed
    assert "reciprocal-cluster" in printed
    assert "reciprocal-core" in printed
    for entry in markov_stats:
        for key in (
            "red_cluster_coverage_curve",
            "weighted_core_coverage_curve",
            "reciprocal_cluster_coverage_curve",
            "reciprocal_core_coverage_curve",
        ):
            assert {"thresholds", "coverages", "auc"} <= entry[key].keys()
    assert (snapshot / "coverage_curves_order1.png").stat().st_size > 0
    assert (snapshot / "coverage_curves_order2.png").stat().st_size > 0
    report = (snapshot / "report.md").read_text()
    assert "coverage curve AUC" in report
    assert "reciprocal-cluster" in report


def test_the_analysis_reports_document_support_curves_and_writes_a_png_per_order(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        markov_orders=[1, 2],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    markov_stats = json.loads((snapshot / "markov_chain.json").read_text())
    assert exit_code == 0
    assert "document-support curve AUC, order 1" in printed
    assert "document-support curve AUC, order 2" in printed
    assert "document-cluster" in printed
    assert "document-reciprocal-core" in printed
    for entry in markov_stats:
        for key in (
            "document_cluster_coverage_curve",
            "document_core_coverage_curve",
            "document_reciprocal_cluster_coverage_curve",
            "document_reciprocal_core_coverage_curve",
        ):
            assert {"thresholds", "coverages", "auc"} <= entry[key].keys()
    assert (snapshot / "coverage_curves_documents_order1.png").stat().st_size > 0
    assert (snapshot / "coverage_curves_documents_order2.png").stat().st_size > 0
    report = (snapshot / "report.md").read_text()
    assert "## Document support" in report
    assert "document-support curve AUC" in report


def test_the_analysis_writes_linear_weighted_core_samples(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        markov_orders=[1],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    capsys.readouterr()
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    samples = json.loads((snapshot / "weighted_core_samples.json").read_text())
    assert exit_code == 0
    assert samples["metric"] == "weighted_core_number"
    assert samples["scale"] == "linear"
    assert samples["total_nodes"] > 0
    assert samples["sample_fraction"] == pytest.approx(0.05)
    assert [sample["target_percent"] for sample in samples["samples"]] == [
        100.0,
        75.0,
        50.0,
        25.0,
        10.0,
        5.0,
        1.0,
    ]
    assert all(
        {"id", "label", "core_number", "distance_to_threshold"} <= node.keys()
        for sample in samples["samples"]
        for node in sample["nodes"]
    )
    assert all(
        sample["token_ids"] == [token for node in sample["nodes"] for token in node["id"]]
        for sample in samples["samples"]
    )


def test_the_analysis_writes_linear_red_cluster_thresholds(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        markov_orders=[1],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    capsys.readouterr()
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    thresholds = json.loads((snapshot / "red_cluster_thresholds.json").read_text())
    markov = json.loads((snapshot / "markov_chain.json").read_text())[0]
    curve = markov["red_cluster_coverage_curve"]
    assert exit_code == 0
    assert thresholds["metric"] == "red_cluster_coverage"
    assert thresholds["scale"] == "linear"
    assert thresholds["order"] == 1
    assert thresholds["weight_key"] == "count"
    assert len(thresholds["thresholds"]) == 7
    assert all(
        entry["threshold"] == pytest.approx(entry["target_fraction"] * curve["thresholds"][-1])
        and entry["actual_coverage"] in curve["coverages"]
        for entry in thresholds["thresholds"]
    )


def test_the_analysis_reports_pattern_coverage_and_writes_it(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    coverage = json.loads((snapshot / "coverage.json").read_text())
    assert exit_code == 0
    assert "Good-Turing missing mass" in printed
    assert coverage["draws"] > 0
    assert coverage["observed_patterns"] > 0
    assert coverage["universe_size"] is None  # no declared enumeration for this corpus
    assert coverage["missing"] == []
    assert "## Pattern coverage" in (snapshot / "report.md").read_text()


def test_the_analysis_reports_spectral_diversity_and_writes_it(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        use_semantic=True,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    spectral = json.loads((snapshot / "spectral.json").read_text())
    assert exit_code == 0
    assert "Vendi Score" in printed
    assert "algebraic connectivity" in printed
    assert spectral["units"] == 10
    assert spectral["vendi_semantic"] > 1.0  # two well-separated clusters, not one blob
    assert spectral["vendi_lexical"] > 1.0
    # Semantic embeddings are exactly orthogonal between the two topics (KeywordEncoder): two
    # disconnected components. Lexically every document still shares the word "number": one.
    assert spectral["components_lexical"] == 1
    assert spectral["components_semantic"] == 2
    assert "connected components" in printed
    assert "## Diversity & cohesion (spectral)" in (snapshot / "report.md").read_text()


def test_too_few_documents_are_reported_undetermined_without_loading_an_encoder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(tmp_path, ["a round shape", "a big dog"], use_semantic=True)
    encoder = KeywordEncoder()

    exit_code = script.main(["--corpus", CORPUS], encoder=encoder)

    assert exit_code == 0
    assert "undetermined" in capsys.readouterr().out
    assert encoder.calls == 0


def test_use_semantic_defaults_to_off_and_never_loads_the_encoder(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plenty of units to group (unlike the too-few-documents case above), but `use_semantic`
    left unset: the encoder must still never be called, and every embeddings-derived number --
    not just the domain split, the `_semantic` half of diversity too -- must be absent rather
    than reported as a misleading zero."""
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
    )
    encoder = KeywordEncoder()

    exit_code = script.main(["--corpus", CORPUS], encoder=encoder)

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    domains = json.loads((snapshot / "domains.json").read_text())
    spectral = json.loads((snapshot / "spectral.json").read_text())
    assert exit_code == 0
    assert encoder.calls == 0
    assert domains["use_semantic"] is False
    assert domains["domains"] == 0
    assert "skipped (use_semantic=false)" in printed
    assert "Vendi Score (lexical)" in printed
    assert "vendi_semantic" not in spectral
    assert "components_semantic" not in spectral
    assert "algebraic_connectivity_semantic" not in spectral


def test_show_evolution_defaults_to_off_and_writes_no_evolution_folder(
    tmp_path: Path, analysis_dir: Path
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    assert exit_code == 0
    assert not (snapshot / "graph_evolution_order1").exists()


def test_show_evolution_true_writes_one_png_per_cut_point(
    tmp_path: Path, analysis_dir: Path
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        show_evolution=True,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    red_files = list((snapshot / "graph_evolution_order1" / "red_cluster").iterdir())
    core_files = list((snapshot / "graph_evolution_order1" / "weighted_core").iterdir())
    assert exit_code == 0
    assert red_files and all(file.stat().st_size > 0 for file in red_files)
    assert core_files and all(file.stat().st_size > 0 for file in core_files)


def test_compute_coverage_curves_false_skips_the_curves_and_their_pngs(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        compute_coverage_curves=False,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    markov_stats = json.loads((snapshot / "markov_chain.json").read_text())
    assert exit_code == 0
    assert "skipped (compute_coverage_curves=false)" in printed
    for entry in markov_stats:
        assert entry["red_cluster_coverage_curve"] is None
        assert entry["weighted_core_coverage_curve"] is None
        assert entry["reciprocal_cluster_coverage_curve"] is None
        assert entry["reciprocal_core_coverage_curve"] is None
    assert not (snapshot / "coverage_curves_order1.png").exists()
    assert not (snapshot / "coverage_curves_order1_log.png").exists()


def test_compute_document_coverage_curves_false_skips_the_curves_and_their_pngs(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        compute_document_coverage_curves=False,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    markov_stats = json.loads((snapshot / "markov_chain.json").read_text())
    assert exit_code == 0
    assert "skipped (compute_document_coverage_curves=false)" in printed
    for entry in markov_stats:
        assert entry["document_cluster_coverage_curve"] is None
        assert entry["document_core_coverage_curve"] is None
        assert entry["document_reciprocal_cluster_coverage_curve"] is None
        assert entry["document_reciprocal_core_coverage_curve"] is None
    assert not (snapshot / "coverage_curves_documents_order1.png").exists()
    assert not (snapshot / "coverage_curves_documents_order1_log.png").exists()
    # The count-based family is a separate flag -- unaffected.
    assert (snapshot / "coverage_curves_order1.png").stat().st_size > 0


def test_compute_diversity_false_skips_the_whole_diversity_report(
    tmp_path: Path, analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        use_semantic=True,
        compute_diversity=False,
    )
    encoder = KeywordEncoder()

    exit_code = script.main(["--corpus", CORPUS], encoder=encoder)

    printed = capsys.readouterr().out
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    spectral = json.loads((snapshot / "spectral.json").read_text())
    assert exit_code == 0
    assert "skipped (compute_diversity=false)" in printed
    assert spectral == {"computed": False}
    # `use_semantic` is unaffected -- the domain split still runs and still uses the encoder.
    assert encoder.calls > 0
    assert "## Diversity & cohesion (spectral)" in (snapshot / "report.md").read_text()


def test_draw_token_graph_false_skips_the_graph_pngs_but_not_evolution(
    tmp_path: Path, analysis_dir: Path
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        draw_token_graph=False,
        show_evolution=True,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    assert exit_code == 0
    assert not (snapshot / "graph.png").exists()
    assert not (snapshot / "graph.html").exists()
    assert not (snapshot / "graph_log.png").exists()
    assert not (snapshot / "graph_log.html").exists()
    # `show_evolution` draws from smaller per-threshold subgraphs, not the full rendering above.
    red_files = list((snapshot / "graph_evolution_order1" / "red_cluster").iterdir())
    assert red_files and all(file.stat().st_size > 0 for file in red_files)


def test_compute_coverage_curves_false_also_skips_evolution_even_when_show_evolution_is_on(
    tmp_path: Path, analysis_dir: Path
) -> None:
    """Evolution draws from `coverage_curves[1]` (the red-cluster/weighted-core curves) -- with
    nothing to draw evolution FROM, `show_evolution=true` alone cannot produce it."""
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        show_evolution=True,
        compute_coverage_curves=False,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    assert exit_code == 0
    assert not (snapshot / "graph_evolution_order1").exists()


def test_a_corpus_without_an_analysis_config_names_the_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    assert exit_code == 1
    assert script.CONFIG_FILENAME in capsys.readouterr().err


def test_an_unknown_unit_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    configure(tmp_path, ["a round shape"], unit="paragraph")

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    assert exit_code == 1
    assert "unit" in capsys.readouterr().err


def test_the_analysis_needs_a_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = script.main([], encoder=KeywordEncoder())

    assert exit_code == 1
    assert "--corpus" in capsys.readouterr().err


def test_the_analysis_needs_a_corpus_or_a_config(capsys: pytest.CaptureFixture[str]) -> None:
    script.main([], encoder=KeywordEncoder())

    assert "--config" in capsys.readouterr().err


def test_a_config_file_from_another_project_is_analyzed_and_named_by_its_folder(
    tmp_path: Path, analysis_dir: Path
) -> None:
    """What lets another project run the analysis on its own configs: a config outside
    `configs/corpora/`, whose folder names the report."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a round shape\nan animal walks\n", encoding="utf-8")
    config = tmp_path / "models" / "other-model" / script.CONFIG_FILENAME
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"files": [str(corpus)], "merge_budget": 20}))

    exit_code = script.main(["--config", str(config)], encoder=KeywordEncoder())

    (snapshot,) = (analysis_dir / "other-model").iterdir()
    assert exit_code == 0
    assert (snapshot / "report.md").is_file()


def test_output_chooses_where_the_dated_reports_go(tmp_path: Path, analysis_dir: Path) -> None:
    configure(tmp_path, ["a round shape", "an animal walks"])
    output = tmp_path / "elsewhere"

    exit_code = script.main(["--corpus", CORPUS, "--output", str(output)], encoder=KeywordEncoder())

    (snapshot,) = output.iterdir()
    assert exit_code == 0
    assert (snapshot / "report.md").is_file()
    assert not analysis_dir.exists()


def test_a_missing_config_file_is_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "nowhere" / script.CONFIG_FILENAME

    exit_code = script.main(["--config", str(missing)], encoder=KeywordEncoder())

    assert exit_code == 1
    assert str(missing) in capsys.readouterr().err


def test_corpus_and_config_cannot_both_be_given(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        script.main(["--corpus", CORPUS, "--config", str(tmp_path / "x.json")])


def test_the_analysis_needs_files_or_a_huggingface_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_without_files()

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    assert exit_code == 1
    assert "huggingface_dataset" in capsys.readouterr().err


def test_huggingface_dataset_supplies_documents_without_any_local_files(
    analysis_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_without_files(huggingface_dataset="roneneldan/TinyStories")
    loader = FakeHuggingFaceLoader(
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)]
    )

    exit_code = script.main(
        ["--corpus", CORPUS], encoder=KeywordEncoder(), huggingface_loader=loader
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert loader.calls == [("roneneldan/TinyStories", "train", "text", 2000, None)]
    assert "documents" in printed
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    cohesion = json.loads((snapshot / "cohesion.json").read_text())
    assert cohesion["documents"] == 10
    assert "roneneldan/TinyStories" in (snapshot / "report.md").read_text()


def test_huggingface_dataset_combines_with_local_files(analysis_dir: Path, tmp_path: Path) -> None:
    configure(
        tmp_path,
        ["a local shape", "a local animal"],
        huggingface_dataset="roneneldan/TinyStories",
        huggingface_split="validation",
        huggingface_text_field="story",
        huggingface_limit=3,
    )
    loader = FakeHuggingFaceLoader(["a hub shape", "a hub animal", "another hub shape"])

    exit_code = script.main(
        ["--corpus", CORPUS], encoder=KeywordEncoder(), huggingface_loader=loader
    )

    assert exit_code == 0
    assert loader.calls == [("roneneldan/TinyStories", "validation", "story", 3, None)]
    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    cohesion = json.loads((snapshot / "cohesion.json").read_text())
    assert cohesion["documents"] == 5  # 2 local + 3 from the (fake) hub


def test_huggingface_limit_null_means_every_row() -> None:
    """A file that sets `huggingface_limit` to `null` (JSON) asks for the whole split -- unlike
    leaving the key out, which falls back to the 2000-row safety default -- and that `None` must
    reach the loader as `None`, not silently become the default because `null` looks absent."""
    configure_without_files(huggingface_dataset="roneneldan/TinyStories", huggingface_limit=None)
    loader = FakeHuggingFaceLoader(["a story"])

    exit_code = script.main(
        ["--corpus", CORPUS], encoder=KeywordEncoder(), huggingface_loader=loader
    )

    assert exit_code == 0
    assert loader.calls == [("roneneldan/TinyStories", "train", "text", None, None)]


def test_export_neo4j_defaults_to_off_and_never_touches_a_driver(
    tmp_path: Path, analysis_dir: Path
) -> None:
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    assert exit_code == 0
    assert "Not pushed to Neo4j (`export_neo4j=false`)." in (snapshot / "report.md").read_text()


def test_export_neo4j_true_pushes_the_graph_through_the_injected_driver(
    tmp_path: Path, analysis_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "irrelevant-here-driver-is-injected")
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        export_neo4j=True,
    )
    driver = FakeNeo4jDriver()

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder(), neo4j_driver=driver)

    (snapshot,) = (analysis_dir / CORPUS).iterdir()
    assert exit_code == 0
    assert driver.closed
    (session,) = driver.sessions
    node_calls = [p for _, p in session.calls if p is not None and "nodes" in p]
    edge_calls = [p for _, p in session.calls if p is not None and "edges" in p]
    assert node_calls and edge_calls
    run_label = f"{CORPUS}_{snapshot.name}"
    assert all(call["run"] == run_label for call in node_calls)
    assert all(call["run"] == run_label for call in edge_calls)
    assert f'Pushed to Neo4j under `run="{run_label}"`' in (snapshot / "report.md").read_text()


def test_export_neo4j_true_without_neo4j_password_fails_fast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    configure(
        tmp_path,
        [f"a shape number {index}" for index in range(5)]
        + [f"an animal number {index}" for index in range(5)],
        export_neo4j=True,
    )

    exit_code = script.main(["--corpus", CORPUS], encoder=KeywordEncoder())

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    assert exit_code == 1
    assert "NEO4J_PASSWORD" in printed
