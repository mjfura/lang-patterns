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

from corpus_generalization.model.generate import LoadedRun, generate, load_run
from corpus_generalization.model.train import TrainingSettings
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


def main() -> None:
    st.set_page_config(page_title="Corpus models", layout="wide")
    root = Path(os.environ.get(RUNS_DIR_ENV_VAR, "runs"))
    available = runs(root)
    owners = _experiments_by_run(root, available)
    page = st.sidebar.radio("Page", ("Overview", "Experiments", "Models"), key="page")
    if page == "Overview":
        _overview(owners)
    elif page == "Experiments":
        _experiments(root, available, owners)
    else:
        _models(root, available, owners)


main()
