"""Class patterns in the diagnosis: a head class's fillers, pooled over its heads."""

import math
import random

from corpus_generalization.analysis.syntactic import filler_divergence
from corpus_generalization.diagnose import DiagnosisReport, diagnose

SUBJECTS = ("he", "she", "we", "they", "you", "i")
VERBS = [
    "eat",
    "cook",
    "wash",
    "cut",
    "take",
    "see",
    "find",
    "like",
    "make",
    "bring",
    "hold",
    "keep",
    "move",
    "open",
    "paint",
    "push",
    "pull",
    "read",
    "sell",
    "show",
    "throw",
    "turn",
    "use",
    "want",
    "watch",
    "fix",
    "carry",
    "clean",
    "drop",
    "fill",
]
NOUNS = [
    "apple",
    "bread",
    "chair",
    "door",
    "egg",
    "fork",
    "glass",
    "hat",
    "jar",
    "key",
    "lamp",
    "map",
    "nail",
    "orange",
    "pan",
    "pen",
    "plate",
    "rope",
    "shoe",
    "spoon",
    "table",
    "towel",
    "vase",
    "wheel",
    "box",
    "cup",
    "bowl",
    "book",
    "coat",
    "bag",
]
ADJECTIVES = ["red", "big", "old", "new", "small", "clean", "soft", "hard", "warm", "cold"]
# Distinct contexts make a head's slot readable; a template's fixed neighbors would count once.
OPENERS = ["then", "now", "so", "and", "but", "also", "still", "often", "again", "first"]


def _documents(
    count: int, seed: int, restricted: tuple[str, ...], subjects: tuple[str, ...] = SUBJECTS
) -> list[str]:
    """Subject-verb-object text; `restricted` subjects only ever take the first two verbs."""
    rng = random.Random(seed)
    documents = []
    for _ in range(count):
        sentences = []
        for _ in range(6):
            subject = rng.choice(subjects)
            verbs = VERBS[:2] if subject in restricted else VERBS
            sentences.append(
                f"{rng.choice(OPENERS)} {subject} {rng.choice(verbs)} the {rng.choice(NOUNS)} ."
            )
            sentences.append(f"the {rng.choice(NOUNS)} is {rng.choice(ADJECTIVES)} .")
        documents.append(" ".join(sentences))
    return documents


def _reported_heads(report: DiagnosisReport) -> list[set[str]]:
    return [{head.strip() for head in row.heads} for row in report.class_rows if not row.lexical]


REFERENCE = _documents(3000, 0, ())


def test_a_class_that_loses_its_slot_is_under_shown() -> None:
    report = diagnose(_documents(1000, 1, SUBJECTS), REFERENCE, merge_budget=1000)
    assert any({"he", "she", "we", "they"} <= heads for heads in _reported_heads(report))
    assert all(row.verdict == "under-shown" for row in report.class_rows)
    assert all(row.t > (report.class_critical_t or 0) for row in report.class_rows)


def test_two_restricted_heads_leave_their_class_pattern_standing() -> None:
    report = diagnose(_documents(1000, 1, ("she", "they")), REFERENCE, merge_budget=1000)
    assert not any(heads & set(SUBJECTS) for heads in _reported_heads(report))
    restricted = {
        row.head.strip(): row.diversity
        for row in report.rows
        if row.head.strip() in {"she", "they"} and row.head_length == 1
    }
    assert restricted and all(value < 0.5 for value in restricted.values())


def test_a_pattern_the_draws_show_and_the_corpus_lacks_is_absent() -> None:
    without_she = tuple(subject for subject in SUBJECTS if subject != "she")
    report = diagnose(_documents(1000, 1, (), without_she), REFERENCE, merge_budget=1000)
    absent = {row.head.strip() for row in report.rows if row.verdict == "absent"}
    assert "she" in absent
    assert not absent & set(without_she)


def test_a_random_sample_of_the_reference_has_no_absent_rows() -> None:
    report = diagnose(_documents(1000, 2, ()), REFERENCE, merge_budget=1000)
    assert [row.head for row in report.rows if row.verdict == "absent"] == []
    assert report.class_rows == ()
    assert report.class_nodes_read > 0


def test_without_a_reference_a_class_lacking_its_slot_is_not_seen() -> None:
    # Every subject loses the verbs past the first two: no sibling shows more, and only a
    # reference sees what the class lacks (`test_a_class_that_loses_its_slot_is_under_shown`).
    report = diagnose(_documents(1000, 1, SUBJECTS), merge_budget=1000)
    assert "weak" not in {row.verdict for row in report.rows}
    assert "a reference from the same domain does both" in report.markdown()


def test_without_a_reference_a_corpus_with_no_gap_has_no_weak_pattern() -> None:
    report = diagnose(_documents(1000, 1, ()), merge_budget=1000)
    assert report.class_rows == ()
    assert not {"weak", "absent"} & {row.verdict for row in report.rows}
    assert report.class_nodes_read > 0


def test_filler_divergence_tracks_moved_mass() -> None:
    uniform = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
    reference = {1: 250, 2: 250, 3: 250, 4: 250}
    assert filler_divergence(reference, reference, uniform) < 0.01
    moved = filler_divergence(reference, {1: 900, 2: 34, 3: 33, 4: 33}, uniform)
    assert moved > 1.0
    assert math.isfinite(filler_divergence(reference, {1: 10}, uniform))
