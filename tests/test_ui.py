"""`ui/app.py`: the Streamlit page lists the trained runs and generates from the chosen one."""

import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from corpus_generalization.model.train import TrainingSettings, train
from corpus_generalization.ui.experiments import EXPERIMENTS, OVERVIEW

# Run as a script by Streamlit, never imported: importing it would render the page.
APP = Path(__file__).resolve().parents[1] / "src" / "corpus_generalization" / "ui" / "app.py"
DOCUMENTS = tuple(f"the cat number {index} sat on the mat" for index in range(20))
TINY = TrainingSettings(
    merge_budget=10,
    n_layer=1,
    n_head=1,
    d_model=16,
    context=16,
    dropout=0.0,
    batch_size=4,
    steps=5,
    warmup_steps=1,
    eval_every=5,
    eval_batches=1,
)


def test_the_page_says_so_when_there_are_no_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))

    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.radio(key="page").set_value("Models").run()

    assert not page.exception
    assert "No trained runs" in page.info[0].value


def test_the_page_shows_a_run_and_generates_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train(DOCUMENTS, TINY, tmp_path / "tiny" / "2026-01-01_00-00-00", {"files": ["corpus.txt"]})
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))

    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.radio(key="page").set_value("Models").run()
    page.text_area(key="prompt").input("the cat")
    page.number_input(key="samples").set_value(2)
    page.button(key="generate").click().run()

    assert not page.exception
    assert page.selectbox(key="run").value == tmp_path / "tiny" / "2026-01-01_00-00-00"
    generated = [block.value for block in page.markdown if block.value.startswith("**")]
    assert len(generated) == 2
    assert all("the cat" in text for text in generated)


def test_overview_lists_every_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))
    page = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not page.exception
    assert page.radio(key="page").value == "Overview"
    assert OVERVIEW.strip() in [block.value for block in page.markdown]
    table = page.dataframe[0].value
    assert table["Title"].tolist() == [experiment.title for experiment in EXPERIMENTS]
    assert table["Models trained"].tolist() == ["corpus readings only"] * len(EXPERIMENTS)


def test_nested_recipe_run_has_experiment_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "syntactic_sufficiency" / "small" / "1_seed0"
    train(DOCUMENTS, TINY, run, {"synthetic": "recipes fillers=1"})
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))
    recipe = next(experiment for experiment in EXPERIMENTS if experiment.key == "recipes")
    page = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not page.exception
    table = page.dataframe[0].value
    assert table.loc[table["Title"] == recipe.title, "Models trained"].item() == "1"
    page.radio(key="page").set_value("Experiments").run()
    page.selectbox(key="experiment").select(recipe.key).run()

    assert not page.exception
    table = page.dataframe[0].value
    metrics = json.loads((run / "metrics.json").read_text())
    assert table.to_dict("records") == [
        {
            "Run": "syntactic_sufficiency/small/1_seed0",
            "Corpus": recipe.corpus["*"],
            "Parameters": metrics["parameters"],
            "Layers x width": "1 x 16",
            "Best held-out loss": metrics["best_heldout_loss"],
            "Best step": metrics["best_step"],
        }
    ]
    page.radio(key="page").set_value("Models").run()

    assert not page.exception
    assert page.selectbox(key="run").options == ["syntactic_sufficiency/small/1_seed0"]
    captions = " ".join(caption.value for caption in page.caption)
    assert recipe.corpus["*"] in captions
    assert recipe.title in captions and recipe.phase in captions
    assert "(unknown)" not in captions


def test_unknown_corpus_stays_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train(DOCUMENTS, TINY, tmp_path / "unlisted" / "run", {})
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))
    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.radio(key="page").set_value("Models").run()

    assert not page.exception
    assert any("Corpus: (unknown)" in caption.value for caption in page.caption)


def test_experiment_tables_do_not_load_weights_and_handle_missing_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "planted_model" / "k2_seed0"
    run.mkdir(parents=True)
    (run / "model.pt").write_text("Weights must not be read by this page.")
    (run / "config.json").write_text(
        json.dumps(
            {
                "planted_model": "k2",
                "n_layer": 2,
                "d_model": 32,
            }
        )
    )
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "parameters": 1234,
                "best_heldout_loss": 1.5,
                "best_step": 5,
            }
        )
    )
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.radio(key="page").set_value("Experiments").run()
    assert not page.exception
    assert any("No model: this experiment reads corpora only." in b.value for b in page.markdown)
    page.selectbox(key="experiment").select("planted_model").run()

    assert not page.exception
    assert page.dataframe[0].value["Corpus"].tolist() == [
        "FineWeb slices 0-6, five heads kept to 2 fillers"
    ]
    assert any("not found" in warning.value for warning in page.warning)


def test_models_are_sorted_by_checkpoint_time_at_any_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = tmp_path / "z" / "last"
    newer = tmp_path / "a" / "deep" / "first"
    train(DOCUMENTS, TINY, older, {"files": ["old.txt"]})
    train(DOCUMENTS, TINY, newer, {"synthetic": "custom recipe"})
    os.utime(older / "model.pt", (100, 100))
    os.utime(newer / "model.pt", (200, 200))
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))
    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.radio(key="page").set_value("Models").run()

    assert not page.exception
    assert page.selectbox(key="run").options == ["a/deep/first", "z/last"]
    assert any("Corpus: synthetic: custom recipe" in caption.value for caption in page.caption)


def _diagnose_page() -> AppTest:
    page = AppTest.from_file(str(APP), default_timeout=180).run()
    page.radio(key="page").set_value("Diagnose").run()
    return page


def _run_recipe_diagnosis(reference: str) -> AppTest:
    page = _diagnose_page()
    page.selectbox(key="diagnose_corpus").select("Recipes: each verb with 1 ingredient").run()
    page.selectbox(key="diagnose_reference").select(reference)
    # The corpus is smaller than its reference, so that random reference samples vary.
    page.number_input(key="diagnose_limit").set_value(1000)
    page.number_input(key="diagnose_reference_limit").set_value(2000)
    page.number_input(key="diagnose_merges").set_value(1000)
    page.button(key="diagnose_run").click().run()
    return page


def test_diagnose_page_renders() -> None:
    page = _diagnose_page()

    assert not page.exception
    assert page.selectbox(key="diagnose_reference").value == "None"
    assert page.number_input(key="diagnose_limit").value == 1000
    assert page.number_input(key="diagnose_reference_limit").value == 4000
    assert page.number_input(key="diagnose_merges").value == 1000
    assert page.number_input(key="diagnose_min_trunk").value == 0
    assert page.multiselect(key="diagnose_head_lengths").value == [1, 2]


def test_diagnose_recipe_gap_is_under_shown() -> None:
    page = _run_recipe_diagnosis("Recipes: every verb with all its ingredients (other documents)")

    assert not page.exception
    table = page.dataframe[0].value
    cut = table.loc[table["head"].str.replace(" ", "") == "cutthe"]
    assert not cut.empty
    assert set(cut["verdict"]) == {"under-shown"}
    assert page.status[0].state == "complete"
    assert any("Learning tokenizer" in block.value for block in page.markdown)
    page.multiselect(key="diagnose_verdicts").set_value(["under-shown"]).run()
    assert not page.exception
    assert set(page.dataframe[0].value["verdict"]) == {"under-shown"}
    page.button(key="diagnose_run").click().run()
    assert not page.exception
    assert page.dataframe[0].value.equals(
        table.loc[table["verdict"] == "under-shown"].reset_index(drop=True)
    )


def test_diagnose_without_reference_has_weak_rows() -> None:
    page = _run_recipe_diagnosis("None")

    assert not page.exception
    assert "weak" in set(page.dataframe[0].value["verdict"])
    assert all(value is None for value in page.dataframe[0].value["reference diversity"])
