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


def _documents(count: int, seed: int, restricted: tuple[str, ...]) -> list[str]:
    """Subject-verb-object text; `restricted` subjects only ever take the first two verbs."""
    rng = random.Random(seed)
    documents = []
    for _ in range(count):
        sentences = []
        for _ in range(6):
            subject = rng.choice(SUBJECTS)
            verbs = VERBS[:2] if subject in restricted else VERBS
            sentences.append(
                f"{rng.choice(OPENERS)} {subject} {rng.choice(verbs)} the {rng.choice(NOUNS)} ."
            )
            sentences.append(f"the {rng.choice(NOUNS)} is {rng.choice(ADJECTIVES)} .")
        documents.append(" ".join(sentences))
    return documents


def _subject_row_verdict(report: DiagnosisReport) -> str:
    rows = [row for row in report.class_rows if " she" in row.heads or "she" in row.heads]
    assert len(rows) == 1, [row.heads for row in report.class_rows]
    assert {" he", " we", " they"} <= set(rows[0].heads)
    return rows[0].verdict


REFERENCE = _documents(3000, 0, ())


def test_a_class_that_loses_its_slot_is_under_shown() -> None:
    report = diagnose(_documents(1000, 1, SUBJECTS), REFERENCE, merge_budget=1000, head_classes=8)
    assert _subject_row_verdict(report) == "under-shown"


def test_two_restricted_heads_leave_their_class_pattern_standing() -> None:
    report = diagnose(
        _documents(1000, 1, ("she", "they")), REFERENCE, merge_budget=1000, head_classes=8
    )
    assert _subject_row_verdict(report) != "under-shown"
    restricted = {
        row.head.strip(): row.diversity
        for row in report.rows
        if row.head.strip() in {"she", "they"} and row.head_length == 1
    }
    assert restricted and all(value < 0.5 for value in restricted.values())


def test_without_a_reference_no_class_pattern_is_read() -> None:
    report = diagnose(_documents(1000, 1, ()), merge_budget=1000, head_classes=8)
    assert report.class_rows == ()
    assert "need a reference" in report.markdown()


def test_filler_divergence_tracks_moved_mass() -> None:
    uniform = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
    reference = {1: 250, 2: 250, 3: 250, 4: 250}
    assert filler_divergence(reference, reference, uniform) < 0.01
    moved = filler_divergence(reference, {1: 900, 2: 34, 3: 33, 4: 33}, uniform)
    assert moved > 1.0
    assert math.isfinite(filler_divergence(reference, {1: 10}, uniform))
