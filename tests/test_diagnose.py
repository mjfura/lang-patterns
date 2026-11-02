"""End-to-end diagnosis of planted recipe restrictions and command output."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta

from corpus_generalization.diagnose import (
    ALPHA,
    RATE_MARGIN,
    RATE_SPREAD,
    DiagnosisReport,
    _draws_for,
    _drop_maxima,
    _rank,
    _readable_trunk,
    diagnose,
    main,
    prepare_reference,
)
from corpus_generalization.synthetic.recipes import LEXICON, recipe_documents, verb_pairing
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer


def _recipes(fillers: int | None, count: int = 2000, seed: int = 0) -> list[str]:
    pairing = verb_pairing(LEXICON, fillers, 2, seed)
    return recipe_documents(LEXICON, pairing, count, seed)


def _verb_verdicts(report: DiagnosisReport) -> dict[str, str]:
    noun_classes = {group.label for group in report.classes if " tomato" in group.tokens}
    # Mid-sentence heads only: a document's first verb has no leading space, and is its own token.
    return {
        row.head.strip().removesuffix(" the"): row.verdict
        for row in report.rows
        if row.head in {f" {verb} the" for verb in LEXICON.verbs}
        and row.filler_class in noun_classes
    }


def test_reference_separates_restricted_and_diverse_recipes() -> None:
    reference = _recipes(None)
    # A corpus smaller than its reference, so that the draws vary. The draws set the trunk read.
    restricted = diagnose(_recipes(1, 1000), reference, merge_budget=1000)
    assert _verb_verdicts(restricted) == dict.fromkeys(LEXICON.verbs, "under-shown")
    diverse = diagnose(_recipes(None, 1000, seed=1), reference, merge_budget=1000)
    assert {row.head_length for row in diverse.rows} == {1, 2}
    assert all(row.top_fillers for row in diverse.rows)
    assert set(_verb_verdicts(diverse).values()) == {"within-noise"}
    assert restricted.settings.min_trunk >= 2 and restricted.settings.class_min_trunk is not None
    assert "alpha = 0.06 against 101 draws" in restricted.markdown()
    # The bar is the furthest any of 16 draws strays: a random corpus passes it somewhere about
    # once in 17, here at one of its patterns.
    assert sum(row.verdict == "under-shown" for row in diverse.rows) <= 1
    assert diverse.head_critical_t is not None


def test_patterns_no_corpus_could_flag_are_not_read() -> None:
    # Rows are 16 draws. Rare patterns (trunk 3) sometimes drop to one filler in one draw and set
    # a bar no single filler of theirs can pass; read from there, they would also hide the
    # medium patterns (trunk 10) behind it. Common ones (trunk 50) are flaggable either way.
    rng = np.random.default_rng(0)
    draws = 16
    rare = np.ones((draws, 50))
    rare[np.arange(50) % draws, np.arange(50)] = 1 / 3
    medium = 1 + rng.normal(0, 0.15, (draws, 50))
    common = 1 + rng.normal(0, 0.01, (draws, 50))
    diversity = np.hstack([rare, medium, common])
    expected = np.hstack([np.full((draws, 50), value) for value in (3.0, 8.0, 40.0)])
    trunks = np.hstack([np.full((draws, 50), value) for value in (3, 10, 50)])

    assert _readable_trunk(diversity, expected, trunks, draws - 2, rank=1) == 10


def test_the_draw_count_holds_the_flag_rate_near_alpha_for_any_draw_set() -> None:
    draws = _draws_for(ALPHA)
    for count, holds in ((draws, True), (draws - 1, False)):
        rank = _rank(ALPHA, count)
        low, high = beta.ppf((RATE_SPREAD, 1 - RATE_SPREAD), rank, count + 1 - rank)
        within = low >= ALPHA * (1 - RATE_MARGIN) and high <= ALPHA * (1 + RATE_MARGIN)
        assert within == holds


def test_drop_maxima_matches_a_loop_over_draws() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(1, 0.1, (12, 40))
    noise[rng.random(noise.shape) < 0.2] = np.nan
    expected = rng.uniform(2, 30, noise.shape)
    maxima = []
    for draw in range(len(noise)):
        rest = np.delete(noise, draw, axis=0)
        usable = ((~np.isnan(rest)).sum(axis=0) >= 9) & ~np.isnan(noise[draw])
        mean = np.nanmean(rest[:, usable], axis=0)
        sd = np.nanstd(rest[:, usable], axis=0, ddof=1)
        spread = np.sqrt(sd**2 + (1 / expected[draw, usable]) ** 2 / 12)
        maxima.append(((mean - noise[draw, usable]) / spread).max())

    assert np.allclose(_drop_maxima(noise, 10, expected), maxima)


def test_a_corpus_as_large_as_its_reference_reads_no_verdict() -> None:
    reference = _recipes(None)
    report = diagnose(_recipes(1), reference, merge_budget=1000, min_trunk=5)
    assert {row.verdict for row in report.rows} == {"unread"}
    assert report.head_critical_t is None and report.class_critical_t is None


def test_without_reference_verbs_narrower_than_their_siblings_are_weak() -> None:
    # Each restricted verb takes its own one or two ingredients: a random share of its class's
    # occurrences would show more. Verbs sharing their ingredients cannot tell each other apart.
    restricted = _verb_verdicts(diagnose(_recipes(1), merge_budget=1000, min_trunk=5))
    assert sum(verdict == "weak" for verdict in restricted.values()) >= 10
    assert set(restricted.values()) <= {"weak", "within-noise", "unread"}
    diverse = diagnose(_recipes(None), merge_budget=1000, min_trunk=5)
    assert "weak" not in _verb_verdicts(diverse).values()
    assert all(row.reference_diversity is None for row in diverse.rows)
    assert "random splits" in diverse.markdown()


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


def test_a_domain_under_the_measured_size_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    recipes = _recipes(None, 300)
    small = diagnose(recipes, merge_budget=1000)
    assert any("under 14.4 MB" in warning for warning in small.warnings)
    monkeypatch.setattr(
        "corpus_generalization.diagnose.MIN_DOMAIN_BYTES",
        sum(len(document.encode("utf-8")) for document in recipes),
    )
    assert not any("MB" in warning for warning in diagnose(recipes, merge_budget=1000).warnings)


@pytest.mark.parametrize("head_lengths", [None, [1], [2]])
def test_command_writes_reports_and_reports_file_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], head_lengths: list[int] | None
) -> None:
    source = tmp_path / "recipes.txt"
    source.write_text("\n<|endoftext|>\n".join(_recipes(None, 30)), encoding="utf-8")
    output = tmp_path / "diagnosis"
    args = ["--files", str(source), "--output", str(output)]
    if head_lengths is not None:
        args.extend(["--head-lengths", *(str(length) for length in head_lengths)])
    assert main(args) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert report["documents"] == 30
    # Unset, the budget is the fewest merges that reach 1.50 tokens per word on the corpus.
    assert report["settings"]["merge_budget"] == 151
    assert not any("tokens per word" in warning for warning in report["warnings"])
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


def test_the_command_refuses_a_budget_outside_the_granularity_range_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "recipes.txt"
    source.write_text("\n<|endoftext|>\n".join(_recipes(None, 30)), encoding="utf-8")
    output = tmp_path / "diagnosis"
    # The recipes run out of merges at 269, near whole words: below the range.
    args = ["--files", str(source), "--merges", "1000", "--output", str(output)]
    assert main(args) == 2
    assert "1.01 tokens per word with 269 merges" in capsys.readouterr().err
    assert not output.exists()
    assert main([*args, "--force"]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert any("tokens per word, outside 1.35 to 1.70" in w for w in report["warnings"])


def test_a_prepared_reference_reads_like_raw_documents() -> None:
    reference = _recipes(None)
    raw = diagnose(_recipes(1), reference, merge_budget=1000, min_trunk=5)
    reading = prepare_reference(reference, 1000)
    prepared = diagnose(_recipes(1), reading, min_trunk=5)
    assert prepared.to_json() == raw.to_json()


def test_a_cut_of_a_longer_merge_run_reads_like_learning_at_the_budget() -> None:
    reference = _recipes(None)
    # The recipe grammar stops learning at 290 merges: cut below that, or the test is empty.
    longer = ByteLevelBPETokenizer.from_document_sequence(reference, recognize_markers=True)
    assert len(longer.merges) > 200
    cut = ByteLevelBPETokenizer.from_merges(longer.merges[:200])
    reading = prepare_reference(reference, 200, tokenizer=cut)
    raw = diagnose(_recipes(1), reference, merge_budget=200, min_trunk=5)
    assert diagnose(_recipes(1), reading, min_trunk=5).to_json() == raw.to_json()


def test_settings_that_contradict_a_prepared_reference_are_refused() -> None:
    reading = prepare_reference(_recipes(None), 1000)
    with pytest.raises(ValueError, match="merge_budget"):
        diagnose(_recipes(1), reading, merge_budget=2000, min_trunk=5)


def test_punctuation_and_word_pieces_are_read_and_lexical_patterns_kept_apart() -> None:
    # Few merges leave ingredients split into pieces, so word-internal transitions are common.
    report = diagnose(_recipes(None, 500), merge_budget=120, min_trunk=5, head_lengths=(1,))
    syntactic = [row for row in report.rows if not row.lexical]
    lexical = [row for row in report.rows if row.lexical]

    assert lexical
    assert any(row.head == " ." for row in syntactic)
    # A word's last piece leads into the punctuation that follows the word.
    assert any(not row.head.startswith(" ") and " ." in row.top_tokens for row in syntactic)
    assert sum(report.counts.values()) == len(syntactic)
    assert sum(report.lexical_counts.values()) == len(lexical)
    markdown = report.markdown()
    assert "## Flagged syntactic patterns" in markdown
    assert "## Flagged lexical patterns" in markdown
