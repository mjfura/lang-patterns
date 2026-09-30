"""End-to-end diagnosis of planted recipe restrictions and command output."""

import json
from pathlib import Path

import pytest

from corpus_generalization.diagnose import DiagnosisReport, diagnose, main
from corpus_generalization.synthetic.recipes import LEXICON, recipe_documents, verb_pairing


def _recipes(fillers: int | None, count: int = 2000, seed: int = 0) -> list[str]:
    pairing = verb_pairing(LEXICON, fillers, 2, seed)
    return recipe_documents(LEXICON, pairing, count, seed)


def _verb_verdicts(report: DiagnosisReport) -> dict[str, str]:
    noun_classes = {group.label for group in report.classes if " tomato" in group.tokens}
    return {
        row.head.strip().removesuffix(" the"): row.verdict
        for row in report.rows
        if row.head.strip() in {f"{verb} the" for verb in LEXICON.verbs}
        and row.filler_class in noun_classes
    }


def test_reference_separates_restricted_and_diverse_recipes() -> None:
    reference = _recipes(None)
    # Templated grammar has few distinct contexts; the default trunk of 20 is for real text.
    restricted = diagnose(_recipes(1), reference, merge_budget=1000, min_trunk=5)
    assert _verb_verdicts(restricted) == dict.fromkeys(LEXICON.verbs, "under-shown")
    diverse = diagnose(_recipes(None, seed=1), reference, merge_budget=1000, min_trunk=5)
    assert {row.head_length for row in diverse.rows} == {1, 2}
    assert all(row.top_fillers for row in diverse.rows)
    assert len(_verb_verdicts(diverse)) == len(LEXICON.verbs)
    assert all(row.verdict != "under-shown" for row in diverse.rows)
    assert all(
        row.verdict == "sufficient" for row in diverse.rows if row.diversity >= diverse.settings.low
    )


def test_without_reference_marks_verb_slots_restricted() -> None:
    report = diagnose(_recipes(1), merge_budget=1000, min_trunk=5)
    assert _verb_verdicts(report) == dict.fromkeys(LEXICON.verbs, "restricted")
    assert all(row.reference_diversity is None for row in report.rows)


def test_small_corpus_warns_and_reports_progress() -> None:
    messages: list[str] = []
    report = diagnose(
        _recipes(None, 300), merge_budget=1000, progress=messages.append, head_lengths=(2,)
    )
    assert report.settings.head_lengths == (2,)
    assert {row.head_length for row in report.rows} == {2}
    assert any("1,000 documents" in warning for warning in report.warnings)
    assert report.documents == 300
    assert report.words > 0
    assert report.tokens_per_word > 0
    assert len(messages) >= 5
    assert json.loads(json.dumps(report.to_json()))["documents"] == 300
    assert all(len(group.top_tokens) <= 12 for group in report.classes)
    assert all(len(row.top_fillers) <= 5 for row in report.rows)


@pytest.mark.parametrize("head_lengths", [None, [1], [2]])
def test_command_writes_reports_and_reports_file_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], head_lengths: list[int] | None
) -> None:
    source = tmp_path / "recipes.txt"
    source.write_text("\n<|endoftext|>\n".join(_recipes(None, 30)), encoding="utf-8")
    output = tmp_path / "diagnosis"
    args = ["--files", str(source), "--merges", "1000", "--output", str(output)]
    if head_lengths is not None:
        args.extend(["--head-lengths", *(str(length) for length in head_lengths)])
    assert main(args) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert report["documents"] == 30
    assert report["settings"]["head_lengths"] == (head_lengths or [1, 2])
    assert all(row["head_length"] in (head_lengths or [1, 2]) for row in report["rows"])
    assert "Warnings" in markdown and "Classes" in markdown
    assert "30 documents" in capsys.readouterr().out
    missing = tmp_path / "missing.txt"
    assert main(["--files", str(missing)]) == 2
    assert str(missing) in capsys.readouterr().err
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    assert main(["--files", str(empty)]) == 2
    assert str(empty) in capsys.readouterr().err
