"""Try every trained model by hand: pick a run, read its numbers, generate from a prompt.

    make ui          # streamlit run src/corpus_generalization/ui/app.py

Runs are read from `runs/` (or the `CORPUS_RUNS_DIR` environment variable). Loading and generation
go through `model.generate`, the same functions `corpus-generate` uses.
"""

import os
from pathlib import Path

import streamlit as st

from corpus_generalization.model.generate import LoadedRun, generate, load_run

RUNS_DIR_ENV_VAR = "CORPUS_RUNS_DIR"


def runs(root: Path) -> list[Path]:
    """Every finished run under `root`, newest first."""
    finished = [path.parent for path in root.glob("*/*/model.pt")]
    return sorted(finished, key=lambda path: path.name, reverse=True)


@st.cache_resource
def _load(path: str) -> LoadedRun:
    return load_run(Path(path))


def _source(loaded: LoadedRun) -> str:
    if loaded.sources.get("huggingface_dataset"):
        return str(loaded.sources["huggingface_dataset"])
    return ", ".join(str(file) for file in loaded.sources.get("files", [])) or "(unknown)"


def main() -> None:
    st.set_page_config(page_title="Corpus models", layout="wide")
    st.title("Corpus models")
    root = Path(os.environ.get(RUNS_DIR_ENV_VAR, "runs"))
    available = runs(root)
    if not available:
        st.info(f"No trained runs in `{root}/` yet. Train one with `corpus-train --config ...`.")
        return
    chosen = st.sidebar.selectbox(
        "Run",
        available,
        format_func=lambda path: f"{path.parent.name} / {path.name}",
        key="run",
    )
    loaded = _load(str(chosen))
    metrics, settings = loaded.metrics, loaded.settings

    st.subheader(f"{chosen.parent.name} — {chosen.name}")
    columns = st.columns(4)
    columns[0].metric("Best held-out loss", f"{metrics['best_heldout_loss']:.4f}")
    columns[1].metric("Train loss there", f"{metrics['best_train_loss']:.4f}")
    columns[2].metric("Gap", f"{metrics['gap_at_best']:.4f}")
    columns[3].metric("Best step", f"{metrics['best_step']:,} / {settings.steps:,}")
    st.caption(
        f"Corpus: {_source(loaded)} · {metrics['documents']['train']:,} train / "
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


main()
