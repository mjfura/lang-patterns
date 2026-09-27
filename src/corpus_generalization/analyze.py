"""Analyze a corpus before training: how it tokenizes, and whether it is one domain or several.

    corpus-analyze --corpus NAME                  # configs/corpora/NAME/tokenizer_analysis.json
    corpus-analyze --config PATH [--output DIR]   # any tokenizer_analysis.json

Installed as a command, it runs in whichever project calls it: `--config` names that project's
own config file, the `files` it lists resolve against the working directory, and the report goes
under `--output` (default `analysis/NAME/`, NAME being the config's folder). The config reads:

    files                 the raw text files to analyze (required unless huggingface_dataset is
                          set; documents from both are combined when both are given)
    huggingface_dataset   a Hugging Face Hub dataset repo id (e.g. "roneneldan/TinyStories") to
                          pull documents from instead of, or alongside, files (default: none)
    huggingface_config    optional dataset configuration/subset passed to Hugging Face as `name`
    huggingface_split     which split of huggingface_dataset to read (default "train")
    huggingface_text_field the column of huggingface_dataset holding each row's document text
                          (default "text" -- TinyStories' own column name; a different dataset
                          may need this overridden)
    huggingface_limit     the most rows to pull from huggingface_dataset (default 2000): several
                          Hub datasets run into the millions of rows, and merge learning is the
                          slowest step in this whole pipeline (see _progress_bar/_stage below) --
                          unbounded by default would make a first run open-ended rather than sized
                          to what's actually being asked. Set to `null` for every row in the split
                          (fetched with a plain, non-streaming download rather than
                          `huggingface_documents`' streaming `.take(limit)` path, since there is no
                          longer a limit to stream up to)
    tokenizer             "reserved" (default) or "byte_level"
    merge_budget          merges the tokenizer may learn (default 1000)
    unit                  "document" (default) or "sentence": what the encoder sees as one point
    encoder               a sentence-transformers model (default all-MiniLM-L6-v2)
    max_clusters          the largest k tried (default 5)
    centroid_distance     how far apart (cosine) every pair of group centroids has to be for the
                          groups to count as separate domains (default 0.8, measured: see
                          corpus_generalization.analysis.domains)
    seed                  k-means' seed (default 0)
    markov_orders         context lengths to sweep for the Markov-chain view (default [1] -- higher
                          orders fragment this project's small combinatorial corpora into
                          near-unique states, where every curve collapses to "one document's own
                          exact wording," not a property of the corpus)
    highlight_document    exact text of one document to trace on the token graph (default: none)
    use_semantic          load the sentence-transformers encoder and compute every metric built on
                          it -- the "## Domains" section and the `_semantic` half of "## Diversity &
                          cohesion" (default false: neither is computed or reported at all, not even
                          as a zero, since a corpus large enough to make `show_evolution` worth
                          disabling is exactly the kind where downloading and running an encoder
                          over every unit could dominate the whole run)
    show_evolution        write `graph_evolution_order1/{red_cluster,weighted_core}/`, one PNG per
                          threshold/level of those two curves (default false: a small corpus has few
                          cut points, but nothing bounds that count as the corpus grows -- this can
                          become thousands of files)
    compute_coverage_curves     sweep and draw the four `count`-based coverage curves (red-cluster,
                          weighted-core, reciprocal-cluster, reciprocal-core) and their AUCs
                          (default true). On a large corpus this is one of the slower stages (it
                          sweeps every distinct observed transition count); false skips computing
                          it, drawing its PNGs (linear and log-normalized), and the graph evolution
                          PNGs (`show_evolution` has nothing to draw evolution frames FROM without
                          it), and `markov_chain.json`'s four `*_coverage_curve` fields become
                          `null` for that order rather than a real curve.
    compute_document_coverage_curves  the same sweep and draw, for the four
                          `effective_document_support`-based curves instead (default true) -- kept
                          as its own flag because it is a second, independently expensive pass over
                          the graph (per-document dispersion, not just raw count), not a cheap
                          side effect of the flag above.
    compute_diversity     compute the "## Diversity & cohesion (spectral)" section -- Vendi Score,
                          connected components, algebraic connectivity, `lexical` reading included
                          (default true). The `lexical` reading is an O(units^2) pairwise
                          similarity pass with no encoder involved, so it costs real time even with
                          `use_semantic=false`; false skips it entirely (not reported, not zero) and
                          `spectral.json` records `"computed": false` instead of any numbers.
    draw_token_graph      render `graph.png` (and its log-normalized sibling `graph_log.png`) --
                          the full order-1 token graph as a static image (default true). On a large
                          vocabulary this rendering itself takes real time independent of
                          everything above; false skips it (the report's "Order-1 token graph" line
                          says so), while leaving `show_evolution`'s per-cut-point frames
                          unaffected, since those are drawn from smaller per-threshold subgraphs,
                          not this one full rendering.
    export_neo4j          push the order-1 token graph into Neo4j instead of (there is no longer a
                          browser-rendered interactive page -- see `analysis.neo4j_export`
                          for why) -- default false. Requires a running Neo4j instance
                          (`docker-compose.yaml` at the repo root) and the `NEO4J_PASSWORD`
                          environment variable set to its password; the command exits with an error
                          before doing any work if `export_neo4j=true` and that variable is unset.
    neo4j_uri             the Bolt URI to connect to (default "bolt://localhost:7687", matching
                          `docker-compose.yaml`'s own default port)
    neo4j_user             the Neo4j username (default "neo4j"); the password is never read from
                          this file -- see `export_neo4j` above

Nothing is written outside the report: the tokenizer is learned in memory from `files`, and the
report goes to `analysis/NAME/<date>/` — `report.md`, `tokens.tsv` (every learned merge and every
token the corpus uses, with counts), `domains.json` (scores, verdict, every unit's group),
`cohesion.json` (branching depth, IPBE and tokenizer fertility for every document, and the corpus's
CDCI), `divergence.json` (leave-one-out token-distribution KL divergence for every document, the
Llama 3 style corpus-filtering signal — see `analysis.kl_divergence` for why it is included
only as a benchmark against `cohesion.json`, not as a replacement for it), `markov_chain.json`
(transition entropy, well-supported-state fraction, weak-connectivity, the Gini coefficient of
transition frequency, the well-supported "red cluster" coverage, and all eight coverage curves'
full points and AUC per order — four built from raw transition `count`, four more from
`effective_document_support` (the perplexity of how a transition's occurrences split across the
documents it appears in — 1.0 if concentrated in a single document no matter how many times it was
repeated there, the plain distinct-document count only if spread perfectly evenly across every one
of them, discounted in between) — see `analysis.markov_chain` for why a transition counts
wherever it occurs, not only at a document's start, why "irreducible" is checked as weak rather
than strong connectivity, what the four `count`-based coverage curves say that the single
fixed-threshold red-cluster reading cannot, and why `effective_document_support` is a different
question from `count`, not a refinement of it), the order-1 token graph itself as a static image,
`graph.png`, edges colored by transition frequency (`export_neo4j=true` pushes the same graph
into Neo4j instead -- see that config key above), `coverage_curves_order<N>.png` per swept order
(the four `count`-based curves plotted together), `islands_order<N>.png` (the red curve's
thresholds again, each splitting the vocabulary into mainland, islands and isolated tokens --
the same numbers, plus each island's tokens at count >= 2, are `markov_chain.json`'s `islands`;
see `analysis.markov_chain.island_decomposition`), `coverage_curves_documents_order<N>.png` (the
four `effective_document_support`-based curves, on their own axis) and, for order 1 only,
`graph_evolution_order1/red_cluster/` and `graph_evolution_order1/weighted_core/` (one PNG per
threshold/level of those two curves, at the same fixed node positions across every frame, so the
graph's own shrinking as the cutoff rises is visible step by step — see
`analysis.markov_graph`), `coverage.json`
(Good-Turing missing mass and, when the corpus
declares a pattern universe, the coupon-collector bound and exactly which patterns are still
missing — see `analysis.coverage`) and `spectral.json` (Vendi Score and algebraic
connectivity, each computed twice: self-contained from the project's own tokenizer, and again from
the same sentence-encoder embeddings the domain split already used — see
`analysis.spectral`). The first run downloads the encoder. See `analysis` for what
each number means.

Every one of the graph and coverage-curve artifacts above also gets a log-normalized sibling,
written alongside it rather than replacing it: `graph_log.png` (edge color interpolated on
`log(count)` instead of `count`, so a corpus with a few outlier-frequency transitions doesn't
collapse the rest of the picture to a uniform pale gray), `graph_evolution_order1_log/` (same
frames as `graph_evolution_order1/`, log-normalized edge color), and
`coverage_curves_order<N>_log.png`/`coverage_curves_documents_order<N>_log.png` (the same measured
curves, threshold axis and `AUC` both read `log(threshold)` instead of `threshold` -- a second,
independent summary of how coverage survives across ORDERS OF MAGNITUDE of threshold, not a
correction to the linear one). See `analysis.markov_graph`'s `_log_frequency_color`
and `analysis.markov_chain.to_log_curve` for why both exist and what each trades away.

`export_neo4j=true` pushes the SAME order-1 graph `graph.png` draws into Neo4j: `:Token` nodes and
`:TRANSITION` relationships, each carrying `count` (raw intensity), `log_value` (the same
log-normalized `[0, 1]` reading `graph_log.png`'s color comes from) and `color` (that color's own
hex string, computed from `log_value`, matching `graph_log.png` rather than `graph.png`) -- so a
Neo4j query or Bloom/Browser view reads the same log-scaled story the static PNG does, without
vis.js's browser-side physics simulation, which stops being usable well before a real corpus's
vocabulary size. Every node/relationship is tagged with a `run` property
(`"<corpus>_<snapshot timestamp>"`) so multiple analyses can share one Neo4j instance without
colliding; see `analysis.neo4j_export` for the full schema and `docker-compose.yaml` at
the repo root for standing up a local instance.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from dotenv import load_dotenv

from corpus_generalization.analysis.branching import (
    CorpusCohesion,
    DocumentBranching,
    branching_entropy,
)
from corpus_generalization.analysis.coverage import PatternCoverage, ngram_counts, pattern_coverage
from corpus_generalization.analysis.domains import (
    DEFAULT_CENTROID_DISTANCE,
    DEFAULT_ENCODER,
    DEFAULT_MAX_CLUSTERS,
    MIN_UNITS,
    MIN_UNITS_PER_DOMAIN,
    DomainStructure,
    Encoder,
    SentenceTransformerEncoder,
    Unit,
    analysis_units,
    domain_structure,
)
from corpus_generalization.analysis.kl_divergence import (
    CorpusDivergence,
    DocumentDivergence,
    token_distribution_divergence,
)
from corpus_generalization.analysis.markov_chain import (
    WELL_SUPPORTED_THRESHOLD,
    CoverageCurve,
    IslandLevel,
    MarkovChainStats,
    build_transition_graph,
    coverage_curve_percentiles,
    document_cluster_coverage_curve,
    document_core_coverage_curve,
    document_reciprocal_cluster_coverage_curve,
    document_reciprocal_core_coverage_curve,
    island_decomposition,
    island_members,
    markov_chain_sweep,
    reciprocal_cluster_coverage_curve,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    weighted_core_coverage_curve,
    weighted_core_numbers,
    weighted_core_samples,
)
from corpus_generalization.analysis.markov_graph import (
    document_path,
    draw_coverage_curves,
    draw_coverage_curves_log,
    draw_document_coverage_curves,
    draw_document_coverage_curves_log,
    draw_islands,
    draw_red_cluster_evolution,
    draw_red_cluster_evolution_log,
    draw_static,
    draw_static_log,
    draw_weighted_core_evolution,
    draw_weighted_core_evolution_log,
    token_graph,
)
from corpus_generalization.analysis.neo4j_export import Neo4jDriver, export_graph
from corpus_generalization.analysis.spectral import DiversityReport, diversity_report
from corpus_generalization.analysis.tokenization import (
    TokenizationStats,
    learned_tokenizer,
    tokenization_stats,
)
from corpus_generalization.config_file import (
    ConfigFileError,
    ConfigSchema,
    ListOf,
    Nullable,
    load_config,
)
from corpus_generalization.corpora import (
    CorpusRequiredError,
    InvalidCorpusNameError,
    TokenizerKind,
    corpus_config_dir,
)
from corpus_generalization.sources import (
    EmptyDataFileError,
    HuggingFaceLoader,
    MissingDataFileError,
    huggingface_documents,
    read_documents_from,
)

CONFIG_FILENAME = "tokenizer_analysis.json"
CONFIG_SCHEMA: ConfigSchema = {
    "files": ListOf(str),
    "huggingface_dataset": str,
    "huggingface_config": str,
    "huggingface_split": str,
    "huggingface_text_field": str,
    "huggingface_limit": Nullable(int),
    "tokenizer": str,
    "merge_budget": int,
    "vocab_domain": ListOf(str),
    "unit": str,
    "encoder": str,
    "max_clusters": int,
    "centroid_distance": float,
    "seed": int,
    "markov_orders": ListOf(int),
    "highlight_document": str,
    "use_semantic": bool,
    "show_evolution": bool,
    "compute_coverage_curves": bool,
    "compute_document_coverage_curves": bool,
    "compute_diversity": bool,
    "draw_token_graph": bool,
    "export_neo4j": bool,
    "neo4j_uri": str,
    "neo4j_user": str,
}
DEFAULT_MERGE_BUDGET = 1000
DEFAULT_HUGGINGFACE_SPLIT = "train"
DEFAULT_HUGGINGFACE_TEXT_FIELD = "text"
DEFAULT_HUGGINGFACE_LIMIT = 2000
DEFAULT_MARKOV_ORDERS = (1,)
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
NEO4J_PASSWORD_ENV_VAR = "NEO4J_PASSWORD"
# Relative, so both resolve against the working directory: installed as a command, the analysis
# reads and writes inside whichever project runs it, not inside this package.
ANALYSIS_DIR = Path("analysis")
load_dotenv(Path(".env"))
# One Markov order's four coverage curves, in the fixed order `draw_coverage_curves` and every
# reader of this dict below expects them: red-cluster, weighted-core, reciprocal-cluster,
# reciprocal-core.
CoverageCurves = tuple[CoverageCurve, CoverageCurve, CoverageCurve, CoverageCurve]
# The `document_support` analogue of `CoverageCurves`, same fixed order: document-cluster,
# document-core, document-reciprocal-cluster, document-reciprocal-core.
DocumentCoverageCurves = tuple[CoverageCurve, CoverageCurve, CoverageCurve, CoverageCurve]
# One Markov order's island decomposition (every threshold) and the islands' labeled members at
# `WELL_SUPPORTED_THRESHOLD`, largest first.
Islands = tuple[tuple[IslandLevel, ...], list[dict[str, object]]]
ISLANDS_SHOWN = 10
ISLAND_MIN_SHOWN_SIZE = 3
TOP_TOKENS_SHOWN = 20
WEAKEST_DOCUMENTS_SHOWN = 20
PROGRESS_LINE_WIDTH = 64


class AnalysisConfigError(ValueError):
    def __init__(self, path: Path, problem: str) -> None:
        super().__init__(f"{path}: {problem}.")
        self.path = path


@dataclass(frozen=True)
class _Settings:
    files: list[Path]
    huggingface_dataset: str | None
    huggingface_config: str | None
    huggingface_split: str
    huggingface_text_field: str
    huggingface_limit: int | None
    tokenizer: TokenizerKind
    merge_budget: int
    vocab_domain: tuple[str, ...]
    unit: Unit
    encoder: str
    max_clusters: int
    centroid_distance: float
    seed: int
    markov_orders: tuple[int, ...]
    highlight_document: str | None
    use_semantic: bool
    show_evolution: bool
    compute_coverage_curves: bool
    compute_document_coverage_curves: bool
    compute_diversity: bool
    draw_token_graph: bool
    export_neo4j: bool
    neo4j_uri: str
    neo4j_user: str


def _settings(path: Path) -> _Settings:
    if not path.is_file():
        raise AnalysisConfigError(path, "does not exist; it names the files to analyze")
    config = load_config(path, CONFIG_SCHEMA)
    files = config.get("files")
    huggingface_dataset = config.get("huggingface_dataset")
    if not files and not huggingface_dataset:
        raise AnalysisConfigError(
            path, '"files" has to list at least one text file, or "huggingface_dataset" a dataset'
        )
    tokenizer: TokenizerKind
    match config.get("tokenizer", "reserved"):
        case "reserved":
            tokenizer = "reserved"
        case "byte_level":
            tokenizer = "byte_level"
        case other:
            raise AnalysisConfigError(path, f'"tokenizer" is {other!r}; use reserved or byte_level')
    unit: Unit
    match config.get("unit", "document"):
        case "document":
            unit = "document"
        case "sentence":
            unit = "sentence"
        case other:
            raise AnalysisConfigError(path, f'"unit" is {other!r}; use document or sentence')
    return _Settings(
        files=[Path(file) for file in (files or [])],
        huggingface_dataset=huggingface_dataset,
        huggingface_config=config.get("huggingface_config"),
        huggingface_split=config.get("huggingface_split", DEFAULT_HUGGINGFACE_SPLIT),
        huggingface_text_field=config.get("huggingface_text_field", DEFAULT_HUGGINGFACE_TEXT_FIELD),
        huggingface_limit=config.get("huggingface_limit", DEFAULT_HUGGINGFACE_LIMIT),
        tokenizer=tokenizer,
        merge_budget=config.get("merge_budget", DEFAULT_MERGE_BUDGET),
        vocab_domain=tuple(config.get("vocab_domain", [])),
        unit=unit,
        encoder=config.get("encoder", DEFAULT_ENCODER),
        max_clusters=config.get("max_clusters", DEFAULT_MAX_CLUSTERS),
        centroid_distance=config.get("centroid_distance", DEFAULT_CENTROID_DISTANCE),
        seed=config.get("seed", 0),
        markov_orders=tuple(config.get("markov_orders", DEFAULT_MARKOV_ORDERS)),
        highlight_document=config.get("highlight_document"),
        use_semantic=config.get("use_semantic", False),
        show_evolution=config.get("show_evolution", False),
        compute_coverage_curves=config.get("compute_coverage_curves", True),
        compute_document_coverage_curves=config.get("compute_document_coverage_curves", True),
        compute_diversity=config.get("compute_diversity", True),
        draw_token_graph=config.get("draw_token_graph", True),
        export_neo4j=config.get("export_neo4j", False),
        neo4j_uri=config.get("neo4j_uri", DEFAULT_NEO4J_URI),
        neo4j_user=config.get("neo4j_user", DEFAULT_NEO4J_USER),
    )


def _progress_bar(unit: str) -> Callable[[int, int | None], None]:
    """A single-line bar for learning merges -- the one step in this pipeline slow enough on a
    large corpus to need one."""

    def render(done: int, total: int | None) -> None:
        line = f"{unit} {done}" if not total else f"{unit} {done}/{total}"
        end = "\n" if total and done >= total else ""
        print(f"\r{line:<{PROGRESS_LINE_WIDTH}}", end=end, file=sys.stderr, flush=True)

    return render


@contextmanager
def _stage(label: str) -> Iterator[None]:
    """Announces `label` as it starts and how long it took once it's done, to stderr -- so the
    rest of the pipeline (no natural per-item progress to report, unlike merge learning above)
    still says which stage is the slow one on a large corpus, rather than leaving that to be
    guessed. Kept off stdout (the data channel a test's `capsys...out` or another program parsing
    the printed report reads) on purpose: timing is a diagnostic, not part of the report."""
    start = time.perf_counter()
    print(f"  {label}...", file=sys.stderr, flush=True)
    yield
    print(f"  {label}: {time.perf_counter() - start:.1f}s", file=sys.stderr, flush=True)


def _neo4j_driver(settings: _Settings) -> Neo4jDriver:
    """A real `neo4j.GraphDatabase` driver, connected to `settings.neo4j_uri` as
    `settings.neo4j_user` with the password from `NEO4J_PASSWORD` (checked present by `main`
    before this is ever called). Imports `neo4j` lazily, the same reason
    `sources.huggingface_documents` imports `datasets` inside its own
    function: most runs never touch it."""
    from neo4j import GraphDatabase

    driver: Neo4jDriver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, os.environ[NEO4J_PASSWORD_ENV_VAR])
    )
    return driver


def main(
    argv: Sequence[str] | None = None,
    encoder: Encoder | None = None,
    huggingface_loader: HuggingFaceLoader | None = None,
    neo4j_driver: Neo4jDriver | None = None,
) -> int:
    """`encoder`, given, replaces the configured sentence-transformers model,
    `huggingface_loader` replaces `sources.huggingface_documents`, and
    `neo4j_driver` replaces the driver `_neo4j_driver` would otherwise construct from
    `settings`/`NEO4J_PASSWORD` -- what lets a test run without downloading anything or
    connecting to a real Neo4j instance."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--corpus", help="the corpus under configs/corpora/ whose analysis to run")
    source.add_argument(
        "--config", type=Path, help=f"any {CONFIG_FILENAME}; its folder names the report"
    )
    parser.add_argument(
        "--output", type=Path, help="where the dated report folders go (default analysis/NAME/)"
    )
    args = parser.parse_args(argv)
    try:
        if args.config is not None:
            config_path: Path = args.config
            name = config_path.resolve().parent.name
        elif args.corpus is not None:
            config_path = corpus_config_dir(args.corpus) / CONFIG_FILENAME
            name = args.corpus
        else:
            raise CorpusRequiredError("corpus-analyze")
        settings = _settings(config_path)
        if (
            settings.export_neo4j
            and neo4j_driver is None
            and NEO4J_PASSWORD_ENV_VAR not in os.environ
        ):
            raise AnalysisConfigError(
                config_path,
                f'"export_neo4j" is true but the {NEO4J_PASSWORD_ENV_VAR} environment variable '
                "is not set",
            )
        with _stage("reading documents"):
            documents = list(read_documents_from(settings.files))
            if settings.huggingface_dataset is not None:
                load: HuggingFaceLoader = huggingface_loader or huggingface_documents
                documents += load(
                    settings.huggingface_dataset,
                    settings.huggingface_split,
                    settings.huggingface_text_field,
                    settings.huggingface_limit,
                    settings.huggingface_config,
                )
    except (
        AnalysisConfigError,
        ConfigFileError,
        CorpusRequiredError,
        EmptyDataFileError,
        InvalidCorpusNameError,
        MissingDataFileError,
    ) as unusable:
        print(unusable, file=sys.stderr)
        return 1

    run_start = time.perf_counter()
    print(f"  {len(documents):,} documents from {_source_description(settings)}", file=sys.stderr)
    with _stage("learning tokenizer"):
        tokenizer = learned_tokenizer(
            documents,
            settings.tokenizer,
            settings.merge_budget,
            _progress_bar("merge"),
            domain_tokens=settings.vocab_domain,
        )
    with _stage("tokenization stats"):
        stats = tokenization_stats(documents, tokenizer, settings.merge_budget)
    with _stage("cohesion (branching depth)"):
        cohesion = branching_entropy(documents, tokenizer)
    with _stage("KL divergence (leave-one-out)"):
        divergence = token_distribution_divergence(documents, tokenizer)
    with _stage("Markov chain stats"):
        markov_stats = markov_chain_sweep(documents, tokenizer, settings.markov_orders)
    with _stage("transition graphs (bos/eos framed)"):
        order_graphs = {
            order: build_transition_graph(documents, tokenizer, order, include_boundaries=True)
            for order in settings.markov_orders
        }
    coverage_curves: dict[int, CoverageCurves] = {}
    islands: dict[int, Islands] = {}
    if settings.compute_coverage_curves:
        with _stage("count-based coverage curves (red/blue/reciprocal)"):
            coverage_curves = {
                order: (
                    red_cluster_coverage_curve(graph),
                    weighted_core_coverage_curve(graph),
                    reciprocal_cluster_coverage_curve(graph),
                    reciprocal_core_coverage_curve(graph),
                )
                for order, graph in order_graphs.items()
            }
        with _stage("island decomposition"):
            islands = {
                order: (
                    island_decomposition(graph),
                    _labeled_islands(
                        island_members(graph, WELL_SUPPORTED_THRESHOLD),
                        cast(_AnalysisTokenizer, tokenizer),
                    ),
                )
                for order, graph in order_graphs.items()
            }
    document_coverage_curves: dict[int, DocumentCoverageCurves] = {}
    if settings.compute_document_coverage_curves:
        with _stage("document-support coverage curves"):
            document_coverage_curves = {
                order: (
                    document_cluster_coverage_curve(graph),
                    document_core_coverage_curve(graph),
                    document_reciprocal_cluster_coverage_curve(graph),
                    document_reciprocal_core_coverage_curve(graph),
                )
                for order, graph in order_graphs.items()
            }
    weighted_core_sample_data: list[dict[str, object]] = []
    red_cluster_percentile_data: list[dict[str, float]] = []
    if settings.compute_coverage_curves and 1 in order_graphs:
        core = weighted_core_numbers(order_graphs[1])
        weighted_core_sample_data = weighted_core_samples(core, coverage_curves[1][1])
        red_cluster_percentile_data = coverage_curve_percentiles(coverage_curves[1][0])
        for sample in weighted_core_sample_data:
            nodes = cast(list[dict[str, object]], sample["nodes"])
            sample["nodes"] = [
                {
                    **node,
                    "label": _node_label(
                        cast(tuple[int, ...], node["id"]), cast(_AnalysisTokenizer, tokenizer)
                    ),
                }
                for node in nodes
            ]
    with _stage("pattern coverage"):
        coverage = pattern_coverage(ngram_counts(documents, tokenizer, order=1))
    units = analysis_units(documents, settings.unit)
    # Too few units to group are never encoded, so the encoder is not loaded — or downloaded — for
    # an answer that cannot be given; `use_semantic=False` is the same withholding, chosen rather
    # than forced, and applies even when there are plenty of units.
    if settings.use_semantic:
        with _stage("domain structure (sentence-transformers)"):
            chosen_encoder = (
                encoder if encoder is not None else SentenceTransformerEncoder(settings.encoder)
            )
            embeddings = (
                chosen_encoder.encode(units)
                if len(units) >= MIN_UNITS
                else np.zeros((len(units), 0))
            )
            structure = domain_structure(
                units, embeddings, settings.max_clusters, settings.centroid_distance, settings.seed
            )
    else:
        embeddings = np.zeros((len(units), 0))
        # `domain_structure` itself only ever sees a zero-width `embeddings` when it has already
        # decided (via `MIN_UNITS`) not to cluster at all; skipping straight to that same "nothing
        # to report" shape here, rather than calling it with enough units to attempt k-means on
        # zero-width vectors, is what keeps this an untried combination instead of an untested one.
        structure = DomainStructure(
            len(units), {}, {}, settings.centroid_distance, 0, (0,) * len(units), (), ()
        )
    if settings.compute_diversity:
        with _stage("diversity report (Vendi/connectivity)"):
            diversity = diversity_report(units, tokenizer, embeddings)
    else:
        # The `lexical` reading is an O(units^2) pairwise pass with no encoder involved -- real
        # cost even with `use_semantic=false` -- so `compute_diversity=false` withholds it
        # entirely (see `_diversity_rows`), the same "not even as a zero" choice `use_semantic`
        # already makes for the `_semantic` half.
        diversity = DiversityReport(len(units), 0.0, 0.0, 0.0, 0.0, 0, 0)
    output: Path = args.output if args.output is not None else ANALYSIS_DIR / name
    snapshot = output / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with _stage("writing report (JSON + report.md)"):
        _write_report(
            snapshot,
            name,
            settings,
            stats,
            structure,
            cohesion,
            divergence,
            markov_stats,
            coverage,
            diversity,
            coverage_curves,
            document_coverage_curves,
            islands,
            weighted_core_sample_data,
            red_cluster_percentile_data,
            order_graphs[1].number_of_nodes() if 1 in order_graphs else 0,
            units,
        )
    highlight = (
        document_path(settings.highlight_document, tokenizer)
        if settings.highlight_document is not None
        else []
    )
    # The evolution frames below are drawn from `graph` too, so it still has to be built whenever
    # they -- or the Neo4j export -- will run even if `draw_token_graph` itself is off, but not
    # otherwise, since building it (unlike drawing it) is not free on a large vocabulary.
    needs_graph = (
        settings.draw_token_graph
        or settings.export_neo4j
        or (settings.show_evolution and settings.compute_coverage_curves)
    )
    if needs_graph:
        graph = token_graph(documents, tokenizer)
    if settings.draw_token_graph:
        with _stage("drawing token graph (PNG)"):
            draw_static(graph, snapshot / "graph.png", highlight=highlight)
        with _stage("drawing token graph (PNG) -- log-normalized edge color"):
            draw_static_log(graph, snapshot / "graph_log.png", highlight=highlight)
    if settings.export_neo4j:
        with _stage("exporting token graph to Neo4j"):
            driver = neo4j_driver if neo4j_driver is not None else _neo4j_driver(settings)
            try:
                export_graph(graph, f"{name}_{snapshot.name}", driver)
            finally:
                driver.close()
    if settings.show_evolution and settings.compute_coverage_curves and 1 in order_graphs:
        with _stage("drawing graph evolution PNGs"):
            core_by_token = {
                token: value for (token,), value in weighted_core_numbers(order_graphs[1]).items()
            }
            draw_red_cluster_evolution(
                graph, coverage_curves[1][0], snapshot / "graph_evolution_order1" / "red_cluster"
            )
            draw_weighted_core_evolution(
                graph,
                core_by_token,
                coverage_curves[1][1],
                snapshot / "graph_evolution_order1" / "weighted_core",
            )
        with _stage("drawing graph evolution PNGs -- log-normalized edge color"):
            draw_red_cluster_evolution_log(
                graph,
                coverage_curves[1][0],
                snapshot / "graph_evolution_order1_log" / "red_cluster",
            )
            draw_weighted_core_evolution_log(
                graph,
                core_by_token,
                coverage_curves[1][1],
                snapshot / "graph_evolution_order1_log" / "weighted_core",
            )
    with _stage("drawing coverage-curve PNGs"):
        for order, (
            red_cluster,
            weighted_core,
            reciprocal_cluster,
            reciprocal_core,
        ) in coverage_curves.items():
            draw_coverage_curves(
                red_cluster,
                weighted_core,
                reciprocal_cluster,
                reciprocal_core,
                order,
                snapshot / f"coverage_curves_order{order}.png",
            )
        for order, (
            document_cluster,
            document_core,
            document_reciprocal_cluster,
            document_reciprocal_core,
        ) in document_coverage_curves.items():
            draw_document_coverage_curves(
                document_cluster,
                document_core,
                document_reciprocal_cluster,
                document_reciprocal_core,
                order,
                snapshot / f"coverage_curves_documents_order{order}.png",
            )
    with _stage("drawing island PNGs"):
        for order, (levels, _) in islands.items():
            draw_islands(levels, order, snapshot / f"islands_order{order}.png")
    with _stage("drawing coverage-curve PNGs -- log-normalized threshold axis"):
        for order, (
            red_cluster,
            weighted_core,
            reciprocal_cluster,
            reciprocal_core,
        ) in coverage_curves.items():
            draw_coverage_curves_log(
                red_cluster,
                weighted_core,
                reciprocal_cluster,
                reciprocal_core,
                order,
                snapshot / f"coverage_curves_order{order}_log.png",
            )
        for order, (
            document_cluster,
            document_core,
            document_reciprocal_cluster,
            document_reciprocal_core,
        ) in document_coverage_curves.items():
            draw_document_coverage_curves_log(
                document_cluster,
                document_core,
                document_reciprocal_cluster,
                document_reciprocal_core,
                order,
                snapshot / f"coverage_curves_documents_order{order}_log.png",
            )
    print(f"  total: {time.perf_counter() - run_start:.1f}s", file=sys.stderr)
    _print_summary(
        name,
        settings,
        stats,
        structure,
        cohesion,
        divergence,
        markov_stats,
        coverage,
        diversity,
        coverage_curves,
        document_coverage_curves,
        islands,
        snapshot,
    )
    return 0


def _verdict(structure: DomainStructure, use_semantic: bool) -> str:
    if not use_semantic:
        return "skipped (use_semantic=false)"
    if structure.domains == 0:
        return f"undetermined: {structure.units} units, telling domains apart needs {MIN_UNITS}"
    if structure.domains == 1:
        closest = structure.separations.get(2)
        if closest is None:
            return "1 domain (no grouping of two or more separated at all)"
        return (
            f"1 domain (the two groups of k=2 are {closest:.3f} apart, under {structure.threshold})"
        )
    weak = (
        f" — weak: a domain has fewer than {MIN_UNITS_PER_DOMAIN} units" if structure.weak else ""
    )
    return (
        f"{structure.domains} domains (every pair of centroids at least {structure.threshold} "
        f"apart; closest {structure.separations[structure.domains]:.3f}){weak}"
    )


def _tokenization_rows(stats: TokenizationStats) -> list[tuple[str, str]]:
    return [
        ("documents", f"{stats.documents:,}"),
        ("words", f"{stats.words:,}"),
        ("bytes", f"{stats.bytes:,}"),
        ("tokens", f"{stats.tokens:,}"),
        ("merges learned", f"{stats.merges:,} of {stats.merge_budget:,}"),
        (
            "vocabulary used",
            f"{stats.distinct_tokens:,} of {stats.vocab_size:,} ({stats.vocabulary_used:.1%})",
        ),
        ("tokens per word", f"{stats.tokens_per_word:.3f}"),
        ("bytes per token", f"{stats.bytes_per_token:.3f}"),
        (
            "entropy",
            f"{stats.entropy_bits:.3f} bits/token (normalized {stats.normalized_entropy:.3f})",
        ),
        ("perplexity", f"{stats.perplexity:.3f}"),
        ("bits per byte", f"{stats.bits_per_byte:.3f}"),
        ("tokens seen once", f"{stats.singletons:,}"),
    ]


def _domain_rows(settings: _Settings, structure: DomainStructure) -> list[tuple[str, str]]:
    def per_k(values: dict[int, float]) -> str:
        return ", ".join(f"k={k} {value:.3f}" for k, value in values.items()) or "none"

    if not settings.use_semantic:
        return [
            ("units", f"{structure.units:,} {settings.unit}s"),
            ("verdict", _verdict(structure, settings.use_semantic)),
        ]
    rows = [
        ("units", f"{structure.units:,} {settings.unit}s"),
        ("encoder", settings.encoder),
        ("closest centroids", per_k(structure.separations)),
        ("silhouette", per_k(structure.silhouettes)),
        ("verdict", _verdict(structure, settings.use_semantic)),
    ]
    rows += [
        (f"  domain {index + 1}", f"{domain.size} units, e.g. {_clip(domain.representative)}")
        for index, domain in enumerate(structure.groups)
    ]
    return rows


def _cohesion_rows(cohesion: CorpusCohesion) -> list[tuple[str, str]]:
    return [
        ("CDCI", f"{cohesion.cdci:.3f} bits (mean IPBE over {cohesion.documents:,} documents)"),
        (
            "tokenizer fertility",
            f"{cohesion.mean_fertility:.3f} ± {cohesion.fertility_stdev:.3f} tokens/word",
        ),
    ]


def _weakest_documents(cohesion: CorpusCohesion, limit: int) -> list[DocumentBranching]:
    """The documents least supported by the rest of the corpus: lowest branching depth first, ties
    broken by IPBE — the first place to look for an out-of-domain intruder."""
    return sorted(cohesion.per_document, key=lambda doc: (doc.tau, doc.ipbe))[:limit]


def _divergence_rows(divergence: CorpusDivergence) -> list[tuple[str, str]]:
    return [
        (
            "KL divergence (Llama 3 style, leave-one-out)",
            f"{divergence.mean_kl_divergence:.3f} ± {divergence.kl_divergence_stdev:.3f} bits",
        ),
    ]


def _highest_divergence_documents(
    divergence: CorpusDivergence, limit: int
) -> list[DocumentDivergence]:
    """The documents whose own token frequencies look least like the rest of the corpus's — the
    Llama-3-style benchmark's own candidates for an out-of-domain document, to compare against
    `_weakest_documents`'s (branching-depth) picks."""
    return sorted(divergence.per_document, key=lambda doc: -doc.kl_divergence)[:limit]


def _coverage_rows(coverage: PatternCoverage) -> list[tuple[str, str]]:
    rows = [
        ("distinct order-1 patterns seen", f"{coverage.observed:,} over {coverage.draws:,} draws"),
        (
            "Good-Turing missing mass",
            f"{coverage.good_turing_missing_mass:.1%} "
            f"({coverage.singleton_patterns:,} patterns seen exactly once)",
        ),
    ]
    if coverage.universe_size is not None:
        rows.append(
            (
                "coupon-collector coverage",
                f"{coverage.observed}/{coverage.universe_size} patterns, "
                f"{coverage.draws:,} draws vs. {coverage.required_draws:.1f} required "
                f"(epsilon={coverage.epsilon}) — "
                f"{'sufficient' if coverage.sufficient else 'insufficient'}",
            )
        )
    return rows


def _diversity_rows(
    diversity: DiversityReport, use_semantic: bool, compute_diversity: bool
) -> list[tuple[str, str]]:
    if not compute_diversity:
        return [("Vendi Score / connectivity", "skipped (compute_diversity=false)")]
    if not use_semantic:
        return [
            ("Vendi Score (lexical)", f"{diversity.vendi_lexical:.2f} of {diversity.units} units"),
            (
                "connected components (lexical)",
                f"{diversity.components_lexical} — 1 means one cohesive corpus, 2+ means it "
                "splits into that many disconnected groups",
            ),
            (
                "algebraic connectivity (lexical)",
                f"{diversity.algebraic_connectivity_lexical:.3f} — only meaningful once "
                "components above reads 1: how richly connected that one piece is, not "
                "comparable across corpora of different size",
            ),
        ]
    return [
        (
            "Vendi Score (lexical / semantic)",
            f"{diversity.vendi_lexical:.2f} / {diversity.vendi_semantic:.2f} "
            f"of {diversity.units} units",
        ),
        (
            "connected components (lexical / semantic)",
            f"{diversity.components_lexical} / {diversity.components_semantic} "
            f"— 1 means one cohesive corpus, 2+ means it splits into that many disconnected "
            "groups",
        ),
        (
            "algebraic connectivity (lexical / semantic)",
            f"{diversity.algebraic_connectivity_lexical:.3f} / "
            f"{diversity.algebraic_connectivity_semantic:.3f} "
            "— only meaningful once components above reads 1: how richly connected that one "
            "piece is, not comparable across corpora of different size",
        ),
    ]


def _markov_rows(markov_stats: Sequence[MarkovChainStats]) -> list[tuple[str, str]]:
    return [
        (
            f"markov order {stats.order}",
            f"{stats.states:,} states, {stats.entropy_bits:.3f} bits "
            f"(perplexity {stats.perplexity:.3f}), "
            f"{stats.well_supported_fraction:.1%} well-supported, "
            f"{stats.components} weakly-connected component(s) "
            f"({stats.largest_component_fraction:.1%} in the largest), "
            f"gini {stats.gini_coefficient:.3f}, "
            f"red-cluster coverage {stats.red_cluster_coverage:.1%}",
        )
        for stats in markov_stats
    ]


def _curve_json(curve: CoverageCurve) -> dict[str, object]:
    return {"thresholds": curve.thresholds, "coverages": curve.coverages, "auc": curve.auc}


def _coverage_curve_rows(
    coverage_curves: dict[int, CoverageCurves], compute_coverage_curves: bool
) -> list[tuple[str, str]]:
    if not compute_coverage_curves:
        return [("coverage curve AUC", "skipped (compute_coverage_curves=false)")]
    return [
        (
            f"coverage curve AUC, order {order}",
            f"red-cluster {red_cluster.auc:.3f}, weighted-core {weighted_core.auc:.3f}, "
            f"reciprocal-cluster {reciprocal_cluster.auc:.3f}, "
            f"reciprocal-core {reciprocal_core.auc:.3f}",
        )
        for order, (red_cluster, weighted_core, reciprocal_cluster, reciprocal_core) in (
            coverage_curves.items()
        )
    ]


def _level_at(levels: Sequence[IslandLevel], threshold: float) -> IslandLevel | None:
    """The decomposition at `threshold`: the level of the lowest observed threshold at or above
    it, since nothing changes between two observed levels. `None` when no edge reaches it."""
    return next((level for level in levels if level.threshold >= threshold), None)


def _island_rows(
    islands: dict[int, Islands], compute_coverage_curves: bool
) -> list[tuple[str, str]]:
    if not compute_coverage_curves:
        return [("islands", "skipped (compute_coverage_curves=false)")]
    rows = []
    for order, (levels, _) in islands.items():
        level = _level_at(levels, WELL_SUPPORTED_THRESHOLD)
        label = f"islands at count >= {WELL_SUPPORTED_THRESHOLD}, order {order}"
        if level is None:
            rows.append((label, f"no transition is seen {WELL_SUPPORTED_THRESHOLD}+ times"))
            continue
        rows.append(
            (
                label,
                f"mainland {level.mainland:.1%}, islands {level.islands:.1%} "
                f"({level.island_count} island(s), largest {level.largest_island} tokens), "
                f"isolated {level.isolated:.1%}",
            )
        )
    return rows


def _island_member_lines(islands: dict[int, Islands]) -> list[str]:
    lines = []
    for order, (_, members) in islands.items():
        shown = [
            island for island in members if cast(int, island["size"]) >= ISLAND_MIN_SHOWN_SIZE
        ][:ISLANDS_SHOWN]
        if not shown:
            continue
        lines += [
            f"Largest islands at count >= {WELL_SUPPORTED_THRESHOLD}, order {order} "
            f"(of {ISLAND_MIN_SHOWN_SIZE}+ tokens; every island is in `markov_chain.json`):",
            "",
            "| tokens | members |",
            "|---|---|",
            *(
                f"| {island['size']} | "
                + _clip(
                    " · ".join(
                        str(node["label"])
                        for node in cast(list[dict[str, object]], island["nodes"])
                    ),
                    100,
                )
                + " |"
                for island in shown
            ),
            "",
        ]
    return lines


def _islands_json(order_islands: Islands) -> dict[str, object]:
    levels, members = order_islands
    return {
        "levels": [
            {
                "threshold": level.threshold,
                "mainland": level.mainland,
                "islands": level.islands,
                "isolated": level.isolated,
                "island_count": level.island_count,
                "largest_island": level.largest_island,
            }
            for level in levels
        ],
        "members_threshold": WELL_SUPPORTED_THRESHOLD,
        "members": members,
    }


def _labeled_islands(
    members: Sequence[Sequence[tuple[int, ...]]], tokenizer: "_AnalysisTokenizer"
) -> list[dict[str, object]]:
    return [
        {
            "size": len(island),
            "nodes": [{"id": list(node), "label": _node_label(node, tokenizer)} for node in island],
        }
        for island in members
    ]


def _document_coverage_curve_rows(
    document_coverage_curves: dict[int, DocumentCoverageCurves],
    compute_document_coverage_curves: bool,
) -> list[tuple[str, str]]:
    if not compute_document_coverage_curves:
        return [
            (
                "document-support curve AUC",
                "skipped (compute_document_coverage_curves=false)",
            )
        ]
    return [
        (
            f"document-support curve AUC, order {order}",
            f"document-cluster {document_cluster.auc:.3f}, "
            f"document-core {document_core.auc:.3f}, "
            f"document-reciprocal-cluster {document_reciprocal_cluster.auc:.3f}, "
            f"document-reciprocal-core {document_reciprocal_core.auc:.3f}",
        )
        for order, (
            document_cluster,
            document_core,
            document_reciprocal_cluster,
            document_reciprocal_core,
        ) in document_coverage_curves.items()
    ]


def _print_summary(
    name: str,
    settings: _Settings,
    stats: TokenizationStats,
    structure: DomainStructure,
    cohesion: CorpusCohesion,
    divergence: CorpusDivergence,
    markov_stats: Sequence[MarkovChainStats],
    coverage: PatternCoverage,
    diversity: DiversityReport,
    coverage_curves: dict[int, CoverageCurves],
    document_coverage_curves: dict[int, DocumentCoverageCurves],
    islands: dict[int, Islands],
    snapshot: Path,
) -> None:
    rows = [
        ("corpus", name),
        ("files", _source_description(settings)),
        ("tokenizer", f"{settings.tokenizer}, learned in memory"),
        *_tokenization_rows(stats),
        *_domain_rows(settings, structure),
        *_cohesion_rows(cohesion),
        *_divergence_rows(divergence),
        *_markov_rows(markov_stats),
        *_coverage_curve_rows(coverage_curves, settings.compute_coverage_curves),
        *_island_rows(islands, settings.compute_coverage_curves),
        *_document_coverage_curve_rows(
            document_coverage_curves, settings.compute_document_coverage_curves
        ),
        *_coverage_rows(coverage),
        *_diversity_rows(diversity, settings.use_semantic, settings.compute_diversity),
        ("report", str(snapshot)),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")


def _markov_chain_json_entry(
    chain: MarkovChainStats,
    coverage_curves: dict[int, CoverageCurves],
    document_coverage_curves: dict[int, DocumentCoverageCurves],
    islands: dict[int, Islands],
) -> dict[str, object]:
    """One order's row for `markov_chain.json`. `coverage_curves`/`document_coverage_curves` are
    `.get()`, not indexed directly: `compute_coverage_curves=false`/
    `compute_document_coverage_curves=false` leave the corresponding dict without an entry for
    this order at all, which must read as `null` here -- "not computed" -- not the empty-curve
    `{"thresholds": [], "coverages": [], "auc": 0.0}` a graph that genuinely has no reciprocal
    pairs also produces; conflating the two would make a skipped stage indistinguishable from a
    real, negative finding."""
    order_curves = coverage_curves.get(chain.order)
    order_document_curves = document_coverage_curves.get(chain.order)
    order_islands = islands.get(chain.order)
    return {
        "order": chain.order,
        "states": chain.states,
        "transitions": chain.transitions,
        "entropy_bits": chain.entropy_bits,
        "perplexity": chain.perplexity,
        "well_supported_fraction": chain.well_supported_fraction,
        "components": chain.components,
        "weakly_connected": chain.weakly_connected,
        "largest_component_fraction": chain.largest_component_fraction,
        "gini_coefficient": chain.gini_coefficient,
        "red_cluster_coverage": chain.red_cluster_coverage,
        "red_cluster_coverage_curve": _curve_json(order_curves[0]) if order_curves else None,
        "weighted_core_coverage_curve": _curve_json(order_curves[1]) if order_curves else None,
        "reciprocal_cluster_coverage_curve": (
            _curve_json(order_curves[2]) if order_curves else None
        ),
        "reciprocal_core_coverage_curve": _curve_json(order_curves[3]) if order_curves else None,
        "islands": _islands_json(order_islands) if order_islands else None,
        "document_cluster_coverage_curve": (
            _curve_json(order_document_curves[0]) if order_document_curves else None
        ),
        "document_core_coverage_curve": (
            _curve_json(order_document_curves[1]) if order_document_curves else None
        ),
        "document_reciprocal_cluster_coverage_curve": (
            _curve_json(order_document_curves[2]) if order_document_curves else None
        ),
        "document_reciprocal_core_coverage_curve": (
            _curve_json(order_document_curves[3]) if order_document_curves else None
        ),
    }


class _AnalysisTokenizer(Protocol):
    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    def decode(self, token_ids: list[int]) -> str: ...


def _node_label(node: tuple[int, ...], tokenizer: _AnalysisTokenizer) -> str:
    if node == (tokenizer.bos_id,):
        return "<bos>"
    if node == (tokenizer.eos_id,):
        return "<eos>"
    return tokenizer.decode(list(node))


def _write_report(
    snapshot: Path,
    name: str,
    settings: _Settings,
    stats: TokenizationStats,
    structure: DomainStructure,
    cohesion: CorpusCohesion,
    divergence: CorpusDivergence,
    markov_stats: Sequence[MarkovChainStats],
    coverage: PatternCoverage,
    diversity: DiversityReport,
    coverage_curves: dict[int, CoverageCurves],
    document_coverage_curves: dict[int, DocumentCoverageCurves],
    islands: dict[int, Islands],
    weighted_core_sample_data: Sequence[dict[str, object]],
    red_cluster_percentile_data: Sequence[dict[str, float]],
    weighted_core_total_nodes: int,
    units: Sequence[str],
) -> None:
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "tokens.tsv").write_text(
        "id\tkind\ttoken\tparts\tcount\tfrequency\n"
        + "".join(
            f"{row.token_id}\t{row.kind}\t{row.text!r}\t"
            f"{'' if row.parts is None else f'{row.parts[0]}+{row.parts[1]}'}\t"
            f"{row.count}\t{row.count / stats.tokens if stats.tokens else 0.0:.6f}\n"
            for row in stats.rows
        ),
        encoding="utf-8",
    )
    (snapshot / "domains.json").write_text(
        json.dumps(
            {
                "use_semantic": settings.use_semantic,
                "unit": settings.unit,
                "encoder": settings.encoder,
                "units": structure.units,
                "closest_centroid_distance": {
                    str(k): distance for k, distance in structure.separations.items()
                },
                "silhouette": {str(k): score for k, score in structure.silhouettes.items()},
                "centroid_distance_threshold": structure.threshold,
                "domains": structure.domains,
                "weak": structure.weak,
                "groups": [
                    {"size": domain.size, "representative": domain.representative}
                    for domain in structure.groups
                ],
                "centroid_distances": structure.centroid_distances,
                "assignments": [
                    {"domain": label + 1, "text": text}
                    for label, text in zip(structure.labels, units, strict=True)
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot / "cohesion.json").write_text(
        json.dumps(
            {
                "documents": cohesion.documents,
                "cdci": cohesion.cdci,
                "mean_fertility": cohesion.mean_fertility,
                "fertility_stdev": cohesion.fertility_stdev,
                "per_document": [
                    {
                        "text": doc.text,
                        "tau": doc.tau,
                        "ipbe": doc.ipbe,
                        "fertility": doc.fertility,
                        "steps": [
                            {
                                "position": step.position,
                                "support": step.support,
                                "entropy_bits": step.entropy_bits,
                            }
                            for step in doc.steps
                        ],
                    }
                    for doc in cohesion.per_document
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot / "divergence.json").write_text(
        json.dumps(
            {
                "documents": divergence.documents,
                "mean_kl_divergence": divergence.mean_kl_divergence,
                "kl_divergence_stdev": divergence.kl_divergence_stdev,
                "per_document": [
                    {"text": doc.text, "kl_divergence": doc.kl_divergence}
                    for doc in divergence.per_document
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot / "markov_chain.json").write_text(
        json.dumps(
            [
                _markov_chain_json_entry(chain, coverage_curves, document_coverage_curves, islands)
                for chain in markov_stats
            ],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot / "weighted_core_samples.json").write_text(
        json.dumps(
            {
                "metric": "weighted_core_number",
                "scale": "linear",
                "total_nodes": weighted_core_total_nodes,
                "sample_fraction": 0.05,
                "samples": weighted_core_sample_data,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot / "red_cluster_thresholds.json").write_text(
        json.dumps(
            {
                "metric": "red_cluster_coverage",
                "scale": "linear",
                "order": 1,
                "weight_key": "count",
                "thresholds": list(red_cluster_percentile_data),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot / "coverage.json").write_text(
        json.dumps(
            {
                "draws": coverage.draws,
                "observed_patterns": coverage.observed,
                "singleton_patterns": coverage.singleton_patterns,
                "good_turing_missing_mass": coverage.good_turing_missing_mass,
                "universe_size": coverage.universe_size,
                "required_draws": coverage.required_draws,
                "epsilon": coverage.epsilon,
                "sufficient": coverage.sufficient,
                "missing": [
                    list(pattern) if isinstance(pattern, tuple) else pattern
                    for pattern in coverage.missing
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot / "spectral.json").write_text(
        json.dumps(
            {
                "computed": settings.compute_diversity,
                **(
                    {
                        "use_semantic": settings.use_semantic,
                        "units": diversity.units,
                        "vendi_lexical": diversity.vendi_lexical,
                        "components_lexical": diversity.components_lexical,
                        "algebraic_connectivity_lexical": diversity.algebraic_connectivity_lexical,
                        **(
                            {
                                "vendi_semantic": diversity.vendi_semantic,
                                "components_semantic": diversity.components_semantic,
                                "algebraic_connectivity_semantic": (
                                    diversity.algebraic_connectivity_semantic
                                ),
                            }
                            if settings.use_semantic
                            else {}
                        ),
                    }
                    if settings.compute_diversity
                    else {}
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    top = sorted((row for row in stats.rows if row.count), key=lambda row: -row.count)
    weakest = _weakest_documents(cohesion, WEAKEST_DOCUMENTS_SHOWN)
    most_divergent = _highest_divergence_documents(divergence, WEAKEST_DOCUMENTS_SHOWN)
    lines = [
        f"# Corpus analysis — {name}",
        "",
        f"Files: {_source_description(settings)}  ",
        f"Tokenizer: {settings.tokenizer}, merge budget {settings.merge_budget:,}, "
        "learned in memory",
        "",
        "## Tokenization",
        "",
        "| metric | value |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in _tokenization_rows(stats)),
        "",
        f"## Most frequent tokens (top {TOP_TOKENS_SHOWN})",
        "",
        "| id | kind | token | count |",
        "|---|---|---|---|",
        *(
            f"| {row.token_id} | {row.kind} | `{row.text!r}` | {row.count:,} |"
            for row in top[:TOP_TOKENS_SHOWN]
        ),
        "",
        "## Domains",
        "",
        "| metric | value |",
        "|---|---|",
        *(f"| {label.strip()} | {value} |" for label, value in _domain_rows(settings, structure)),
        "",
        "## Cohesion",
        "",
        "| metric | value |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in _cohesion_rows(cohesion)),
        "",
        f"Least-supported documents (bottom {WEAKEST_DOCUMENTS_SHOWN} by branching depth "
        "`tau`, ties broken by `IPBE`) — the first place to look for an out-of-domain document:",
        "",
        "| tau | IPBE (bits) | fertility | document |",
        "|---|---|---|---|",
        *(
            f"| {doc.tau} | {doc.ipbe:.3f} | {doc.fertility:.3f} | {_clip(doc.text)} |"
            for doc in weakest
        ),
        "",
        "## KL divergence (Llama 3 style benchmark)",
        "",
        "Token-distribution KL divergence, leave-one-out per document — the corpus-filtering "
        "signal reported in the Llama 3 paper (section 3.1.1), included here as a benchmark "
        "against the branching-depth metrics above, not as a replacement for them: it compares "
        "each document's own token frequencies against the rest of the corpus's, with no regard "
        "for token order.",
        "",
        "| metric | value |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in _divergence_rows(divergence)),
        "",
        f"Highest-divergence documents (top {WEAKEST_DOCUMENTS_SHOWN}) — this benchmark's own "
        "picks for an out-of-domain document, to compare against the branching-depth picks above:",
        "",
        "| KL divergence (bits) | document |",
        "|---|---|",
        *(f"| {doc.kl_divergence:.3f} | {_clip(doc.text)} |" for doc in most_divergent),
        "",
        "## Markov chain",
        "",
        "A transition counts wherever it occurs in the corpus, not only at a document's start "
        '(the fix for e.g. "I am", which should count the same whether it opens a sentence or '
        "sits in the middle of one — a limitation of the branching-depth metrics above). "
        "`well-supported` is the fraction of states seen at least twice; below it, a state's "
        "entropy of 0 is more likely sample-size sparsity than a real rule. Connectivity is "
        "checked as WEAK, not strong: a document's last token is always a dead end, so strong "
        "connectivity would report every corpus as fragmented. `gini` measures how evenly "
        "transition frequencies are spread across every edge (0 = even, 1 = a few edges dominate); "
        "`red-cluster coverage` is the fraction of the whole vocabulary covered by the largest "
        "connected neighborhood of well-supported edges alone — a low value with a high `gini` "
        "means one well-supported pocket surrounded by mostly once-seen edges, not a broadly "
        "learnable corpus.",
        "",
        "| order | value |",
        "|---|---|",
        *(
            f"| {label.replace('markov order ', '')} | {value} |"
            for label, value in _markov_rows(markov_stats)
        ),
        "",
        "`red-cluster coverage` fixes its threshold at 2 occurrences, which hides whether the "
        "well-supported neighborhood it reports is held together by genuinely high-intensity, "
        "oft-repeated transitions or only just clears the bar. `coverage_curves_order<N>.png` "
        "sweeps that threshold from 1 to the highest observed count instead, alongside the "
        "weighted (generalized) k-core coverage curve — the same question asked node-by-node, "
        "peeling away whichever node has "
        "the least total incident transition count at each step, rather than as a single global "
        "connectivity check. The two can disagree: a graph that is globally fragmented from the "
        "start reads low red-cluster coverage even at its lowest threshold, while weighted-core "
        "coverage reads 1.0 there, since every node has *some* strength of its own even when it "
        "belongs to no one else's neighborhood. A third, dashed pair looks at direction, scoring "
        "with the harmonic mean rather than the sum so a one-sided value scores exactly 0: "
        "`reciprocal-cluster` keeps only transitions observed in *both* directions (A→B and "
        "B→A), while `reciprocal-core` is flow persistence — peeling away tokens that stop "
        "either receiving or emitting transitions, it keeps the directed cycles text keeps coming "
        "back through, with no transition needing to run backwards (a cycle a→b→c→a reads full "
        "`reciprocal-core` and empty `reciprocal-cluster`). A purely sequential corpus (a "
        "template sentence's own grammar, which never runs backwards or loops) reads empty on "
        "both by construction — that is the finding, not a gap to fill. Every curve's `auc` "
        "(area under it, threshold normalized to `[0, 1]`) is "
        "the single number both this table and the picture read off — high `auc` means that "
        "curve's kind of cohesion survives well past the weakest threshold, low `auc` means it "
        "collapses the moment more is demanded — see `analysis.markov_chain` for the "
        "full derivation.",
        "",
        "Unlike the table above (and `components`/`weakly_connected` in it), every curve here and "
        "the token graph below are built with `<bos>`/`<eos>` included as real nodes: `<bos>`'s "
        "total out-degree equals the corpus's own document count, a fixed ceiling every other "
        "edge's intensity can be read against. They are deliberately left out of `components`/"
        "`weakly_connected` instead — `<bos>` and `<eos>` are the same two nodes in every "
        "document, so including them there would turn any set of documents, even ones sharing no "
        "vocabulary at all, into one artificially connected piece; see "
        "`analysis.markov_chain` for the full reasoning.",
        "",
        "| order | coverage curve AUC "
        "(red-cluster / weighted-core / reciprocal-cluster / reciprocal-core) |",
        "|---|---|",
        *(
            f"| {label.replace('coverage curve AUC, order ', '')} | {value} |"
            for label, value in _coverage_curve_rows(
                coverage_curves, settings.compute_coverage_curves
            )
        ),
        "",
        "### Islands",
        "",
        "At each threshold of the red curve, every token is on the mainland (the largest "
        "connected piece of the transitions seen at least that often — the red curve itself), on "
        "an island (any other piece of 2+ tokens: a neighborhood that repeats, but out of contact "
        "with the mainland) or isolated (no transition seen that often: not repeated enough to be "
        "learned). The three shares sum to 100%; every threshold is in `markov_chain.json` "
        "(`islands`) and in `islands_order<N>.png`. The islands listed below are the tokens that "
        "need more text connecting them to the rest of the corpus.",
        "",
        "| reading | value |",
        "|---|---|",
        *(
            f"| {label} | {value} |"
            for label, value in _island_rows(islands, settings.compute_coverage_curves)
        ),
        "",
        *_island_member_lines(islands),
        (
            "Order-1 token graph: `graph.png` (static), plus its log-normalized edge-color "
            "sibling `graph_log.png`"
            + (
                f', with the walk of "{_clip(settings.highlight_document)}" highlighted.'
                if settings.highlight_document
                else "."
            )
            if settings.draw_token_graph
            else "Order-1 token graph: skipped (`draw_token_graph=false`)."
        ),
        (
            f'Pushed to Neo4j under `run="{name}_{snapshot.name}"` (`export_neo4j=true`).'
            if settings.export_neo4j
            else "Not pushed to Neo4j (`export_neo4j=false`)."
        ),
        "",
        "## Document support",
        "",
        "Every curve above is built from `count`: how many times a transition was observed, "
        "pooled across the whole corpus with no regard for which document any occurrence came "
        "from. That hides a real failure mode: a transition a single long document repeats fifty "
        "times internally, and one that appears once each in five different documents, both read "
        "as `count=50` and `count=5` respectively regardless of which document did the "
        "repeating — a model trained on the corpus sees the former as one training signal "
        "repeated, not five independent confirmations of a rule. Counting DISTINCT documents "
        "alone is not enough to fix this either: it cannot tell 20 occurrences split 15/5 between "
        "two documents from 2 occurrences split 1/1 between the same two, even though the first "
        "is far more concentrated. `effective_document_support` is the same four questions "
        "(global cluster / local core, raw / reciprocal) asked of the perplexity of how a "
        "transition's occurrences split across the documents it appears in instead — 1.0 "
        "whenever everything landed in one document no matter how many times it was repeated "
        "there, the plain distinct-document count only when the split is perfectly even across "
        "every one of them, discounted in between; the graph-edge analogue of a "
        "dispersion-adjusted frequency. A corpus where every transition's evidence is spread "
        "evenly across many different documents reads high here regardless of how high or low "
        "`count`-based coverage runs; a corpus where a handful of documents are doing all the "
        "work, or where the split across documents is lopsided, reads low here even with "
        "excellent `count`-based coverage, and that gap between the two families is itself the "
        "finding — high `count`-based AUC with low `effective_document_support`-based AUC on the "
        "same graph means the apparent cohesion is real but concentrated in too few documents, "
        "or too unevenly split among them, to trust as a generalizable pattern. See "
        "`analysis.markov_chain` for the full derivation.",
        "",
        "| order | document-support curve AUC "
        "(document-cluster / document-core / document-reciprocal-cluster / "
        "document-reciprocal-core) |",
        "|---|---|",
        *(
            f"| {label.replace('document-support curve AUC, order ', '')} | {value} |"
            for label, value in _document_coverage_curve_rows(
                document_coverage_curves, settings.compute_document_coverage_curves
            )
        ),
        "",
        "`coverage_curves_documents_order<N>.png` plots all four, same layout and reading as "
        "`coverage_curves_order<N>.png` above, on its own axis rather than mixed into that "
        "picture — a high-`count` curve and a low-`document_support` curve on the same graph both "
        "describe real, simultaneously true properties, and overlaying all eight lines on one "
        "axis would obscure that rather than clarify it.",
        "",
        "## Pattern coverage",
        "",
        "Coupon-collector coverage over order-1 patterns (single tokens), read only off this "
        "corpus's own counts — no external model or reference corpus. Good-Turing missing mass "
        "is the fraction of draws that were singletons (seen exactly once), the classic estimate "
        "of the probability the *next* draw is a pattern never seen before; it needs no declared "
        "enumeration of the domain. The coupon-collector bound itself — how many draws it takes "
        "to have covered a *known* number of patterns with probability `1 - epsilon` — is only "
        "reported when this corpus declares one (a hand-built combinatorial domain, not this "
        "one); see `analysis.coverage` for both.",
        "",
        "| metric | value |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in _coverage_rows(coverage)),
        "",
        "## Diversity & cohesion (spectral)",
        "",
        "Three readings of the same unit-similarity graph, each computed twice — `lexical` from "
        "nothing but this corpus's own tokenizer (a bag-of-token-id cosine similarity — no "
        "external artifact), `semantic` from the same sentence-encoder embeddings the domain "
        "split above already computed. Vendi Score is the effective number of independent, "
        "non-redundant units (n identical units score 1, n pairwise-unrelated units score n). "
        'Connected components is the exact, discrete answer to "is this one cohesive corpus, '
        'or does it split into several" — 1 means one piece, 2 or 3 means exactly that many '
        "disconnected groups, read directly off the Laplacian's zero-eigenvalue count, not "
        "eyeballed. Algebraic connectivity (the Fiedler value) is the *continuous* follow-up "
        "once components reads 1: how richly connected that one piece is — but it has no fixed "
        "scale to compare across corpora of different size, so read it only as a within-corpus, "
        "before/after signal, never against some universal threshold. Agreement between lexical "
        "and semantic on all three is a second, independent confirmation; a semantic-only split "
        "into more components is documents sharing no vocabulary at all yet meaning nearly the "
        "same thing (or the reverse) — see `analysis.spectral`."
        + (
            " This whole section is skipped (`compute_diversity=false`): not even the "
            "self-contained `lexical` reading, which costs real time on its own (an O(units^2) "
            "pairwise pass) independent of `semantic`/`use_semantic`."
            if not settings.compute_diversity
            else ""
            if settings.use_semantic
            else " `semantic` is not computed here (`use_semantic=false`): only the "
            "self-contained `lexical` reading below."
        ),
        "",
        "| metric | value |",
        "|---|---|",
        *(
            f"| {label} | {value} |"
            for label, value in _diversity_rows(
                diversity, settings.use_semantic, settings.compute_diversity
            )
        ),
        "",
        "Every token is in `tokens.tsv`; every unit's domain is in `domains.json`; every "
        "document's full branching walk is in `cohesion.json`; every document's KL divergence is "
        "in `divergence.json`; every order's full stats, including all eight coverage curves' "
        "every point, are in `markov_chain.json`; the full coverage numbers are in "
        "`coverage.json`; the full Vendi/connectivity numbers are in `spectral.json`.",
        "",
    ]
    (snapshot / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _clip(text: str, width: int = 60) -> str:
    return repr(text if len(text) <= width else f"{text[: width - 3]}...")


def _source_description(settings: _Settings) -> str:
    files = ", ".join(str(file) for file in settings.files) or "(none)"
    if settings.huggingface_dataset is None:
        return files
    limit = "all" if settings.huggingface_limit is None else f"first {settings.huggingface_limit:,}"
    dataset = (
        f"{settings.huggingface_dataset} (split={settings.huggingface_split}, "
        f"field={settings.huggingface_text_field}, {limit} rows)"
    )
    return dataset if not settings.files else f"{files} + {dataset}"


if __name__ == "__main__":
    sys.exit(main())
