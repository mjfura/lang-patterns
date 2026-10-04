"""The item construction of the syntactic evaluation, on small counts without spaCy or models."""

import importlib.util
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def experiment() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments/syntactic_eval/syntactic_eval.py"
    spec = importlib.util.spec_from_file_location("syntactic_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_word_classes_keep_frequent_unambiguous_open_class_words(experiment: ModuleType) -> None:
    tags = {
        "garlic": Counter({"NOUN": 30}),
        "cut": Counter({"VERB": 15, "NOUN": 15}),
        "the": Counter({"DET": 100}),
        "rare": Counter({"ADJ": 5}),
        "quickly": Counter({"ADV": 19, "ADJ": 1}),
    }
    assert experiment.word_classes(tags) == {"garlic": "NOUN", "quickly": "ADV"}


def test_slot_class_needs_an_open_class_with_enough_share(experiment: ModuleType) -> None:
    assert experiment.slot_class(Counter({"VERB": 4, "PRON": 3, "PART": 3})) == "VERB"
    assert experiment.slot_class(Counter({"DET": 9, "NOUN": 1})) is None
    assert experiment.slot_class(Counter()) is None


def test_pairs_match_frequency_without_reuse_and_are_seeded(experiment: ModuleType) -> None:
    frequency = Counter({"swim": 100, "run": 1000, "table": 110, "chair": 900, "blue": 20})
    pairs = experiment.pair_by_frequency(["swim", "run"], ["table", "chair", "blue"], frequency, 0)
    assert sorted(pairs) == [("run", "chair"), ("swim", "table")]
    assert pairs == experiment.pair_by_frequency(
        ["swim", "run"], ["table", "chair", "blue"], frequency, 0
    )


def test_follower_pattern_matches_whole_words_after_the_head(experiment: ModuleType) -> None:
    pattern = experiment.follower_pattern(["people", "she"])
    text = "Many people swim. Townspeople run. she said, people, and she can"
    assert [m.groups() for m in pattern.finditer(text)] == [
        ("people", "swim"),
        ("she", "said"),
        ("she", "can"),
    ]


def test_contexts_end_at_the_head_and_are_capped(
    experiment: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = ["the people swim and people run and people walk"]
    monkeypatch.setattr(experiment, "CONTEXTS", 2)
    found = experiment.contexts(documents, ["people"])
    assert found == {"people": ["the people", "the people swim and people"]}
