"""`ui/app.py`: the Streamlit page lists the trained runs and generates from the chosen one."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from corpus_generalization.model.train import TrainingSettings, train

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

    assert not page.exception
    assert "No trained runs" in page.info[0].value


def test_the_page_shows_a_run_and_generates_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train(DOCUMENTS, TINY, tmp_path / "tiny" / "2026-01-01_00-00-00", {"files": ["corpus.txt"]})
    monkeypatch.setenv("CORPUS_RUNS_DIR", str(tmp_path))

    page = AppTest.from_file(str(APP), default_timeout=60).run()
    page.text_area(key="prompt").input("the cat")
    page.number_input(key="samples").set_value(2)
    page.button(key="generate").click().run()

    assert not page.exception
    assert page.selectbox(key="run").value == tmp_path / "tiny" / "2026-01-01_00-00-00"
    generated = [block.value for block in page.markdown if block.value.startswith("**")]
    assert len(generated) == 2
    assert all("the cat" in text for text in generated)
