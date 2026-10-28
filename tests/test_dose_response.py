"""Small fixtures exercise planting and reporting without preparing or training a corpus."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def experiment() -> ModuleType:
    path = Path(__file__).parents[1] / "experiments/dose_response/dose_response.py"
    assert path.exists(), "dose-response experiment has not been implemented"
    spec = importlib.util.spec_from_file_location("dose_response", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_planting_keeps_original_hits_and_punctuation(experiment: ModuleType) -> None:
    documents = ["alpha beta outside?! alpha kept. alpha can't. alpha 123."]
    kept = {"alpha": ["kept", "spare"], "beta": ["safe", "extra"]}
    assert experiment.plant_partial(documents, kept, {"alpha": 1.0, "beta": 1.0}, 0) == [
        "alpha kept safe?! alpha kept. alpha can't. alpha 123."
    ]
    assert experiment.plant_partial(documents, kept, {"alpha": 0.0, "beta": 0.0}, 0) == documents


def test_partial_planting_is_nested_and_repeatable(experiment: ModuleType) -> None:
    documents = ["head outside." for _ in range(200)]
    kept = {"head": ["kept", "spare"]}
    low = experiment.plant_partial(documents, kept, {"head": 0.25}, 7)
    high = experiment.plant_partial(documents, kept, {"head": 0.75}, 7)
    assert low == experiment.plant_partial(documents, kept, {"head": 0.25}, 7)
    low_hits = {i for i, text in enumerate(low) if text == "head kept."}
    high_hits = {i for i, text in enumerate(high) if text == "head kept."}
    assert 0 < len(low_hits) < len(high_hits) < 200
    assert low_hits < high_hits
    assert low != experiment.plant_partial(documents, kept, {"head": 0.25}, 8)


@pytest.mark.parametrize(
    ("drops", "expected"),
    [
        ([-1.2, -0.9, -0.6, -0.2, -0.1, 0.0, 0.0], 0.375),
        ([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0, 0.0], 0.4),
        ([0.0] * 7, None),
        ([-0.5] * 7, None),
        ([-0.8, 0.0, -0.8, 0.0, -0.8, 0.0, 0.0], 0.5625),
        ([-0.8, -0.8, -0.8, -0.8, -0.8, -0.2, -0.4], 0.65),
    ],
)
def test_threshold_uses_first_crossing_from_top(
    experiment: ModuleType, drops: list[float], expected: float | None
) -> None:
    result = experiment.crossing(drops)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.fixture
def results(experiment: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(experiment, "RESULTS", tmp_path)
    planted = {f"head{i}": ["kept", "spare"] for i in range(21)}
    comparison = {f"comparison{i}": ["kept", "spare"] for i in range(5)}
    calibration = {
        assignment: {
            head: {
                "target": ((i + shift) % 7 + 1) / 10,
                "q": 0.5,
                "diversity": ((i + shift) % 7 + 1) / 10,
                "trunk": 100,
            }
            for i, head in enumerate(planted)
        }
        for assignment, shift in (("A", 0), ("B", 3))
    }
    (tmp_path / "prepare.json").write_text(
        json.dumps(
            {
                "plant_seed": 0,
                "planted": planted,
                "comparison": comparison,
                "calibration": calibration,
                "indicator_diversity": {},
            }
        )
    )
    for seed in (0, 1):
        for variant in ("control", "A", "B"):
            occurrences = []
            for head in [*planted, *comparison]:
                drop = 0.0
                if variant != "control" and head in planted:
                    target = calibration[variant][head]["target"]
                    drop = {
                        0.1: -1.2,
                        0.2: -0.9,
                        0.3: -0.6,
                        0.4: -0.2,
                        0.5: -0.1,
                        0.6: 0.0,
                        0.7: 0.0,
                    }[target]
                for word in ("kept", "outside"):
                    occurrences.append({"head": head, "word": word, "log_probability": -3.0 + drop})
            (tmp_path / f"{variant}_seed{seed}.json").write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "best_step": 100,
                        "best_heldout_loss": 2.0,
                        "eval_bits_per_byte": 2.0,
                        "run": "fixture",
                        "reused_run": variant == "control",
                        "occurrences": occurrences,
                    }
                )
            )
    return tmp_path


def test_summary_predictions_and_missing_readings(experiment: ModuleType, results: Path) -> None:
    experiment.summarize()
    summary = json.loads((results / "summary.json").read_text())
    assert summary["predictions"] == {f"P{i}": "HELD" for i in range(1, 7)}
    assert summary["crossing_diversity"] == pytest.approx(0.375)
    assert summary["readings"]["A_seed0"]["drops"]["head0"] == pytest.approx(-1.2)
    (results / "B_seed1.json").unlink()
    experiment.summarize()
    summary = json.loads((results / "summary.json").read_text())
    assert set(summary["predictions"].values()) == {"unreadable"}


def test_summary_rejects_misaligned_occurrences(experiment: ModuleType, results: Path) -> None:
    path = results / "A_seed0.json"
    run = json.loads(path.read_text())
    run["occurrences"][0]["word"] = "wrong"
    path.write_text(json.dumps(run))
    with pytest.raises(AssertionError, match="occurrences"):
        experiment.summarize()


def test_summary_reports_failed_predictions(experiment: ModuleType, results: Path) -> None:
    prepared = json.loads((results / "prepare.json").read_text())
    for assignment, offset in (("A", 0.4), ("B", 0.9)):
        for seed in (0, 1):
            path = results / f"{assignment}_seed{seed}.json"
            run = json.loads(path.read_text())
            for occurrence in run["occurrences"]:
                head = occurrence["head"]
                if head in prepared["planted"]:
                    target = prepared["calibration"][assignment][head]["target"]
                    occurrence["log_probability"] = -3.0 - target - offset
                elif head in ("comparison0", "comparison1"):
                    occurrence["log_probability"] = -3.6
            path.write_text(json.dumps(run))
    experiment.summarize()
    summary = json.loads((results / "summary.json").read_text())
    assert summary["predictions"] == {f"P{i}": "FAILED" for i in range(1, 7)}
    assert summary["crossing"] == "no crossing"
    assert summary["readings"]["A_seed0"]["spearman"] == pytest.approx(-1.0)
    assert summary["assignment_differences"]["0"]["0.1"] == pytest.approx(0.5)


def test_summary_averages_all_occurrences(experiment: ModuleType, results: Path) -> None:
    path = results / "A_seed0.json"
    run = json.loads(path.read_text())
    for occurrence in run["occurrences"]:
        if occurrence["head"] == "head0":
            occurrence["log_probability"] = -1.0 if occurrence["word"] == "kept" else -7.0
    path.write_text(json.dumps(run))
    experiment.summarize()
    summary = json.loads((results / "summary.json").read_text())
    assert summary["readings"]["A_seed0"]["drops"]["head0"] == pytest.approx(-1.0)
    assert summary["readings"]["A_seed0"]["occurrences"]["head0"] == 2


def test_empty_occurrences_are_unreadable(experiment: ModuleType, results: Path) -> None:
    for path in results.glob("*_seed*.json"):
        run = json.loads(path.read_text())
        run["occurrences"] = []
        path.write_text(json.dumps(run))
    experiment.summarize()
    summary = json.loads((results / "summary.json").read_text())
    assert set(summary["predictions"].values()) == {"unreadable"}
