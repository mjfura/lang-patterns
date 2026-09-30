"""Explore the experiments and try every trained model by hand.

    make ui          # streamlit run src/corpus_generalization/ui/app.py

Runs are read from `runs/` (or the `CORPUS_RUNS_DIR` environment variable). Loading and generation
go through `model.generate`, the same functions `corpus-generate` uses.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import fields
from fnmatch import fnmatch
from pathlib import Path

import streamlit as st

from corpus_generalization.diagnose import VERDICTS, DiagnosisReport, diagnose
from corpus_generalization.model.generate import LoadedRun, generate, load_run
from corpus_generalization.model.train import TrainingSettings
from corpus_generalization.ui.datasets import datasets
from corpus_generalization.ui.experiments import EXPERIMENTS, OVERVIEW, Experiment

RUNS_DIR_ENV_VAR = "CORPUS_RUNS_DIR"


def runs(root: Path) -> list[Path]:
    """Every finished run under `root`, newest first."""
    finished = [path.parent for path in root.rglob("model.pt")]
    return sorted(finished, key=lambda path: (path / "model.pt").stat().st_mtime, reverse=True)


@st.cache_resource
def _load(path: str) -> LoadedRun:
    return load_run(Path(path))


def _source(sources: Mapping[str, object], run: Path, experiment: Experiment | None) -> str:
    if sources.get("huggingface_dataset"):
        return str(sources["huggingface_dataset"])
    files = sources.get("files", [])
    if isinstance(files, list) and files:
        return ", ".join(str(file) for file in files)
    if experiment is not None:
        for pattern, description in experiment.corpus.items():
            if fnmatch(run.name, pattern):
                return description
    return (
        ", ".join(
            f"{key}: {value}"
            for key, value in sources.items()
            if key not in {"files", "huggingface_dataset"}
        )
        or "(unknown)"
    )


def _experiments_by_run(root: Path, available: list[Path]) -> dict[Path, Experiment]:
    finished = set(available)
    owners: dict[Path, Experiment] = {}
    for experiment in EXPERIMENTS:
        for pattern in experiment.models:
            for path in root.glob(pattern):
                if path in finished:
                    owners.setdefault(path, experiment)
    return owners


def _overview(owners: dict[Path, Experiment]) -> None:
    st.title("Overview")
    st.markdown(OVERVIEW)
    rows = []
    for experiment in EXPERIMENTS:
        count = sum(owner.key == experiment.key for owner in owners.values())
        rows.append(
            {
                "Phase": experiment.phase,
                "Title": experiment.title,
                "Predictions": experiment.predictions,
                "Models trained": str(count) if count else "corpus readings only",
            }
        )
    st.dataframe(rows, hide_index=True)


def _model_row(run: Path, root: Path, experiment: Experiment) -> dict[str, object]:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    training_keys = {field.name for field in fields(TrainingSettings)}
    sources = {key: value for key, value in config.items() if key not in training_keys}
    return {
        "Run": run.relative_to(root).as_posix(),
        "Corpus": _source(sources, run, experiment),
        "Parameters": metrics["parameters"],
        "Layers x width": f"{config['n_layer']} x {config['d_model']}",
        "Best held-out loss": metrics["best_heldout_loss"],
        "Best step": metrics["best_step"],
    }


def _experiments(root: Path, available: list[Path], owners: dict[Path, Experiment]) -> None:
    st.title("Experiments")
    by_key = {experiment.key: experiment for experiment in EXPERIMENTS}
    chosen = st.selectbox(
        "Experiment",
        list(by_key),
        format_func=lambda key: f"{by_key[key].phase} — {by_key[key].title}",
        key="experiment",
    )
    experiment = by_key[chosen]
    for label, text in (
        ("Question", experiment.question),
        ("Design", experiment.design),
        ("Finding", experiment.finding),
        ("Predictions", experiment.predictions),
    ):
        st.markdown(f"**{label}**")
        st.markdown(text)
    st.subheader("Models trained")
    matching = [run for run in available if owners.get(run) == experiment]
    if matching:
        st.dataframe([_model_row(run, root, experiment) for run in matching], hide_index=True)
    else:
        st.markdown("No model: this experiment reads corpora only.")
    with st.expander("Full write-up"):
        readme = Path.cwd() / experiment.readme
        if experiment.readme and readme.is_file():
            st.markdown(readme.read_text(encoding="utf-8"))
        else:
            st.warning(f"Full write-up not found: {experiment.readme or '(no README specified)'}")


def _models(root: Path, available: list[Path], owners: dict[Path, Experiment]) -> None:
    st.title("Corpus models")
    if not available:
        st.info(f"No trained runs in `{root}/` yet. Train one with `corpus-train --config ...`.")
        return
    chosen = st.sidebar.selectbox(
        "Run",
        available,
        format_func=lambda path: path.relative_to(root).as_posix(),
        key="run",
    )
    loaded = _load(str(chosen))
    metrics, settings = loaded.metrics, loaded.settings

    st.subheader(f"{chosen.parent.name} — {chosen.name}")
    experiment = owners.get(chosen)
    if experiment is not None:
        st.caption(f"Experiment: {experiment.title} — {experiment.phase}")
    columns = st.columns(4)
    columns[0].metric("Best held-out loss", f"{metrics['best_heldout_loss']:.4f}")
    columns[1].metric("Train loss there", f"{metrics['best_train_loss']:.4f}")
    columns[2].metric("Gap", f"{metrics['gap_at_best']:.4f}")
    columns[3].metric("Best step", f"{metrics['best_step']:,} / {settings.steps:,}")
    st.caption(
        f"Corpus: {_source(loaded.sources, chosen, experiment)} · "
        f"{metrics['documents']['train']:,} train / "
        f"{metrics['documents']['heldout']:,} held-out documents · "
        f"{metrics['tokens']['train']:,} train tokens · vocabulary {metrics['vocab_size']:,} "
        f"({settings.merge_budget:,} merges) · {metrics['parameters']:,} parameters "
        f"({settings.n_layer} layers, {settings.n_head} heads, width {settings.d_model}, "
        f"context {settings.context})"
    )
    evaluations = metrics["evaluations"]
    st.line_chart(
        {
            "step": [evaluation["step"] for evaluation in evaluations],
            "train": [evaluation["train_loss"] for evaluation in evaluations],
            "held-out": [evaluation["heldout_loss"] for evaluation in evaluations],
        },
        x="step",
        y=["train", "held-out"],
        x_label="step",
        y_label="loss",
    )

    st.subheader("Generate")
    prompt = st.text_area("Prompt", value="The", key="prompt")
    controls = st.columns(5)
    max_tokens = controls[0].number_input("Max tokens", 1, 1000, 60, key="max_tokens")
    temperature = controls[1].slider("Temperature", 0.0, 2.0, 0.8, 0.05, key="temperature")
    top_k = controls[2].number_input("Top-k (0 = off)", 0, 10_000, 40, key="top_k")
    samples = controls[3].number_input("Samples", 1, 10, 3, key="samples")
    seed = controls[4].number_input("Seed", 0, 1_000_000, 0, key="seed")
    if st.button("Generate", key="generate"):
        for index in range(int(samples)):
            text = generate(
                loaded.model,
                loaded.tokenizer,
                prompt,
                int(max_tokens),
                float(temperature),
                int(top_k) or None,
                int(seed) + index,
            )
            st.markdown(f"**{index + 1}.** {prompt}{text}")


@st.cache_data(show_spinner=False)
def _diagnose_cached(
    corpus_key: str,
    reference_key: str | None,
    limit: int,
    reference_limit: int,
    merges: int,
    head_lengths: tuple[int, ...],
    min_trunk: int,
) -> DiagnosisReport:
    catalog = {dataset.key: dataset for dataset in datasets()}
    st.write("Loading corpus documents")
    documents = catalog[corpus_key].load(limit or None)
    reference = None
    if reference_key is not None:
        st.write("Loading reference documents")
        reference = catalog[reference_key].load(reference_limit or None)
    return diagnose(
        documents,
        reference=reference,
        merge_budget=merges,
        head_lengths=head_lengths,
        min_trunk=min_trunk,
        progress=lambda message: st.write(message),
    )


def _diagnosis_output(report: DiagnosisReport) -> None:
    st.text(report.summary())
    for warning in report.warnings:
        st.warning(warning)
    for column, (verdict, count) in zip(
        st.columns(len(VERDICTS)), report.counts.items(), strict=True
    ):
        column.metric(verdict, count)
    st.markdown(
        "**How to read this**\n\n"
        "This report is per pattern, not per corpus. **under-shown** means the domain uses "
        "more fillers here than the corpus shows. A model trained on it is expected to tie "
        "the head to the fillers shown; add more text where the head takes other fillers "
        "of its class. **fixed** means the language's or domain's own construction, not a "
        "defect. **restricted** (no reference) is a candidate to be judged. "
        "**low = 0.5** is permissive: the recipe model generalized from diversity ~0.8. "
        "Calibrate the threshold per model family and size."
    )
    verdicts = st.multiselect(
        "Verdicts",
        VERDICTS,
        default=["under-shown", "restricted", "fixed"],
        key="diagnose_verdicts",
    )
    rows = [
        {
            "head": row.head,
            "head length": row.head_length,
            "verdict": row.verdict,
            "diversity": row.diversity,
            "reference diversity": row.reference_diversity,
            "margin mean": row.margin_mean,
            "trunk": row.trunk,
            "fillers": row.fillers,
            "top fillers": ", ".join(token.strip() for token in row.top_fillers),
            "class's top tokens": ", ".join(token.strip() for token in row.top_tokens),
        }
        for row in report.rows
        if row.verdict in verdicts
    ]
    st.dataframe(rows, hide_index=True)
    with st.expander("Classes"):
        st.dataframe(
            [
                {
                    "label": group.label,
                    "size": len(group.tokens),
                    "top tokens": ", ".join(token.strip() for token in group.top_tokens),
                }
                for group in report.classes
            ],
            hide_index=True,
        )
    st.download_button(
        "Download report as JSON",
        json.dumps(report.to_json(), indent=2),
        file_name="diagnosis.json",
        mime="application/json",
        key="diagnose_download",
    )


def _diagnose() -> None:
    st.title("Diagnose")
    catalog = {dataset.label: dataset for dataset in datasets()}
    chosen = st.selectbox("Corpus", list(catalog), key="diagnose_corpus")
    corpus = catalog[chosen]
    st.caption(corpus.description)
    reference_label = st.selectbox("Reference", ["None", *catalog], key="diagnose_reference")
    reference = catalog.get(reference_label)
    if reference is not None:
        st.caption(reference.description)
    limit = st.number_input(
        "Documents per dataset (0 = all)",
        min_value=0,
        value=1000,
        key="diagnose_limit",
        help="The reference uses its separate limit below.",
    )
    reference_limit = st.number_input(
        "Reference documents (0 = all)",
        min_value=0,
        value=4000,
        key="diagnose_reference_limit",
        help="Overrides the document limit for the reference.",
    )
    if (
        st.session_state.get("diagnose_merge_corpus") != corpus.key
        or "diagnose_merges" not in st.session_state
    ):
        st.session_state["diagnose_merges"] = corpus.merges
        st.session_state["diagnose_merge_corpus"] = corpus.key
    merges = st.number_input("Merge budget", min_value=0, key="diagnose_merges")
    lengths = st.multiselect("Head lengths", [1, 2], default=[1, 2], key="diagnose_head_lengths")
    min_trunk = st.number_input(
        "Minimum trunk",
        min_value=1,
        value=20,
        key="diagnose_min_trunk",
        help="Lower it for small synthetic corpora, e.g. 5 for recipes.",
    )
    st.caption(
        "Real sources with large merge budgets take minutes. Suggested demo: "
        '"Recipes: each verb with 1 ingredient" against '
        '"Recipes: every verb with all its ingredients (other documents)" with minimum trunk 5, '
        'or "FineWeb with a planted gap" against "Real text: fineweb".'
    )
    inputs = (
        corpus.key,
        reference.key if reference else None,
        int(limit),
        int(reference_limit),
        int(merges),
        tuple(sorted(lengths)),
        int(min_trunk),
    )
    if not lengths:
        st.warning("Choose at least one head length.")
    if st.button("Run diagnosis", key="diagnose_run", disabled=not lengths):
        with st.status("Running diagnosis", expanded=True) as status:
            try:
                report = _diagnose_cached(*inputs)
            except (OSError, ValueError) as error:
                status.update(label="Diagnosis failed", state="error")
                st.error(str(error))
            else:
                st.session_state["diagnose_report"] = report
                st.session_state["diagnose_report_inputs"] = inputs
                status.update(label="Diagnosis complete", state="complete")
    saved_report = st.session_state.get("diagnose_report")
    if isinstance(saved_report, DiagnosisReport):
        if st.session_state.get("diagnose_report_inputs") != inputs:
            st.info("Showing the last completed report. Run diagnosis to apply these inputs.")
        _diagnosis_output(saved_report)


def main() -> None:
    st.set_page_config(page_title="Corpus models", layout="wide")
    root = Path(os.environ.get(RUNS_DIR_ENV_VAR, "runs"))
    available = runs(root)
    owners = _experiments_by_run(root, available)
    page = st.sidebar.radio("Page", ("Overview", "Experiments", "Diagnose", "Models"), key="page")
    if page == "Overview":
        _overview(owners)
    elif page == "Experiments":
        _experiments(root, available, owners)
    elif page == "Diagnose":
        _diagnose()
    else:
        _models(root, available, owners)


main()
