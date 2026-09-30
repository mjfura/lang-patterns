"""`analysis.syntactic` patterns, domain split and sufficiency curve — see
`docs/2026-09-28-syntactic-sufficiency-design.md`."""

from dataclasses import replace
from itertools import pairwise

import pytest

from corpus_generalization.analysis import syntactic
from corpus_generalization.analysis.syntactic import (
    Pattern,
    TokenClasses,
    induce_classes,
    split_domain,
    sufficiency_curve,
    syntactic_patterns,
)
from corpus_generalization.synthetic.recipes import (
    LEXICON,
    foreign_documents,
    recipe_documents,
    verb_pairing,
)


def _encode(documents: list[str]) -> tuple[list[list[int]], dict[str, int]]:
    ids: dict[str, int] = {}
    sequences = [[ids.setdefault(word, len(ids)) for word in doc.split()] for doc in documents]
    return sequences, ids


def _recipes(fillers_per_verb: int | None, count: int = 2000) -> list[str]:
    pairing = verb_pairing(LEXICON, fillers_per_verb, reserved_per_verb=2, seed=0)
    return recipe_documents(LEXICON, pairing, count=count, seed=0)


def _noun_slot_coverage(documents: list[str]) -> float:
    """Mean coverage of the patterns `<verb> the` → the class of the singular ingredients."""
    sequences, ids = _encode(documents)
    classes = induce_classes(sequences)
    nouns = classes.classes[ids["tomato"]]
    verbs = {ids[verb] for verb in LEXICON.verbs}
    coverages = [
        pattern.coverage
        for pattern in syntactic_patterns(sequences, classes, head_length=2)
        if pattern.filler_class == nouns and pattern.head[0] in verbs
    ]
    assert len(coverages) == len(LEXICON.verbs)
    return sum(coverages) / len(coverages)


def _classes(**classes: list[int]) -> TokenClasses:
    mapping = {token: number for number, tokens in enumerate(classes.values()) for token in tokens}
    return TokenClasses(classes=mapping, n_classes=len(classes), merge_heights=())


def test_verbs_seen_with_every_ingredient_cover_their_noun_slot() -> None:
    assert _noun_slot_coverage(_recipes(None)) > 0.9


def test_verbs_seen_with_one_or_two_ingredients_do_not() -> None:
    assert _noun_slot_coverage(_recipes(2)) < 0.2
    assert _noun_slot_coverage(_recipes(1)) < 0.2


def test_readings_count_distinct_contexts() -> None:
    classes = _classes(heads=[0, 5, 6, 7], nouns=[10, 11, 12, 13, 14])
    sequences = [[5, 0, 10, 6], [5, 0, 11, 6], [7, 0, 11, 6], [5, 0, 11, 6]]

    (pattern,) = [p for p in syntactic_patterns(sequences, classes) if p.head == (0,)]

    assert pattern.trunk == 3  # the fourth sequence repeats the second's context
    assert pattern.fillers == 2
    assert pattern.singletons == 1  # 10 in one context, 11 in two
    assert pattern.documents == 4
    assert pattern.coverage == pytest.approx(2 / 5)
    assert pattern.missing_mass == pytest.approx(1 / 3)


def test_classes_of_one_token_hold_no_pattern() -> None:
    classes = _classes(head=[0], fixed=[1], nouns=[2, 3])
    patterns = syntactic_patterns([[0, 1, 0, 2, 0, 3]], classes)

    assert {pattern.filler_class for pattern in patterns} == {2}


def test_exact_copies_change_nothing_but_the_document_count() -> None:
    documents = _recipes(None, count=300)
    sequences, _ = _encode(documents)
    copied, _ = _encode([*documents, *documents[:100]])
    classes = induce_classes(sequences)

    original = syntactic_patterns(sequences, classes, head_length=2)
    with_copies = syntactic_patterns(copied, classes, head_length=2)

    def readings(patterns: list[Pattern]) -> list[tuple[object, ...]]:
        return [(p.head, p.filler_class, p.trunk, p.fillers, p.singletons) for p in patterns]

    assert readings(original) == readings(with_copies)


def test_a_few_foreign_documents_are_outliers() -> None:
    documents = [*_recipes(None, count=500), *foreign_documents(count=2, seed=0)]
    sequences, ids = _encode(documents)
    classes = induce_classes(sequences)
    foreign = {ids[word] for word in ("bank", "market", "fund", "company", "investor", "broker")}

    domain, outliers = split_domain(syntactic_patterns(sequences, classes))

    assert not any(set(pattern.head) & foreign for pattern in domain)
    assert any(set(pattern.head) & foreign for pattern in outliers)


def test_the_sufficiency_area_is_the_weighted_mean_coverage() -> None:
    sequences, _ = _encode(_recipes(None, count=300))
    domain, _ = split_domain(syntactic_patterns(sequences, induce_classes(sequences), 2))

    curve = sufficiency_curve(domain)

    total = sum(pattern.trunk for pattern in domain)
    assert curve.auc == pytest.approx(sum(p.trunk * p.coverage for p in domain) / total)
    assert curve.coverages[0] == pytest.approx(1.0)
    assert all(a >= b for a, b in pairwise(curve.coverages))


def test_restricting_the_verbs_lowers_the_sufficiency_area_only_a_little() -> None:
    """Only the verbs' noun slots are restricted; every other pattern of the corpus is untouched
    and keeps the corpus-wide area high. The restriction shows pattern by pattern."""

    def area(documents: list[str]) -> float:
        sequences, _ = _encode(documents)
        patterns = syntactic_patterns(sequences, induce_classes(sequences), head_length=2)
        return sufficiency_curve(split_domain(patterns)[0]).auc

    diverse, restricted = area(_recipes(None)), area(_recipes(1))
    assert restricted < diverse < restricted + 0.1


def test_word_like_tokens_require_whole_words() -> None:
    words = {0: " model", 1: " is", 2: " al", 3: "gebra", 4: ".", 5: "ing"}
    sequences = [[0, 1, 4], [2, 3, 4], [5, 4]]

    assert syntactic.word_like_tokens(sequences, words.__getitem__) == frozenset({0, 1})


@pytest.mark.parametrize(("final_count", "expected"), [(19, {0}), (18, {0}), (17, set())])
def test_word_like_tokens_use_all_occurrences(final_count: int, expected: set[int]) -> None:
    words = {0: " word", 1: ".", 2: "ing"}
    sequences = [[0, 1]] * final_count + [[0, 2]] * (20 - final_count)

    assert syntactic.word_like_tokens(sequences, words.__getitem__) == frozenset(expected)


@pytest.mark.parametrize("following", [[], [1], [2], [3], [4]])
def test_word_like_tokens_recognize_word_endings(following: list[int]) -> None:
    words = {0: " word", 1: "EOS", 2: "\nnext", 3: ".", 4: "\tmore"}

    assert syntactic.word_like_tokens([[0, *following]], words.__getitem__, {1}) == {0}


@pytest.mark.parametrize("following", ["suffix", "7", "é", ""])
def test_word_like_tokens_reject_continuations(following: str) -> None:
    words = {0: " word", 1: following}

    assert syntactic.word_like_tokens([[0, 1]], words.__getitem__) == frozenset()


def test_heads_filter_every_token_of_the_head() -> None:
    sequences = [[0, 1, 10], [0, 2, 11], [2, 1, 10]]
    classes = _classes(nouns=[10, 11])

    patterns = syntactic_patterns(sequences, classes, 2, heads={0, 1})

    assert [pattern.head for pattern in patterns] == [(0, 1)]
    assert syntactic_patterns(sequences, classes, 2, heads=set()) == []


def test_expected_fillers_use_class_occurrences_and_distinct_trunk() -> None:
    classes = _classes(nouns=[10, 11, 12])
    sequences = [[0, 10], [0, 10], [0, 10], [0, 11]]

    (pattern,) = syntactic_patterns(sequences, classes)

    assert pattern.trunk == 2
    # Shares 3/4 and 1/4 give (1 - 1/16) + (1 - 9/16); unseen 12 adds zero.
    assert pattern.expected_fillers == pytest.approx(1.375)
    assert pattern.diversity == pytest.approx(2 / 1.375)
    assert replace(pattern, expected_fillers=0.0).diversity == 0.0


def test_recipe_noun_slot_diversity_separates_variants() -> None:
    means = []
    for fillers_per_verb in (None, 1):
        sequences, ids = _encode(_recipes(fillers_per_verb))
        classes = induce_classes(sequences)
        nouns = classes.classes[ids["tomato"]]
        heads = {(ids[verb], ids["the"]) for verb in LEXICON.verbs}
        diversities = [
            pattern.diversity
            for pattern in syntactic_patterns(sequences, classes, head_length=2)
            if pattern.filler_class == nouns and pattern.head in heads
        ]
        assert len(diversities) == len(LEXICON.verbs)
        means.append(sum(diversities) / len(diversities))
    diverse, restricted = means
    print(f"Recipe mean diversities: diverse={diverse:.12f}, restricted={restricted:.12f}")
    assert diverse >= 0.8
    assert restricted <= 0.2


def test_reference_diagnosis_matches_heads_and_classes_in_target_order() -> None:
    base = Pattern((0,), 0, 10, 1, 10, 0, 3)
    target = [
        replace(base, head=(0,), expected_fillers=2.0),
        replace(base, head=(1,), expected_fillers=10.0),
        replace(base, head=(2,), expected_fillers=10.0),
        replace(base, head=(3,), expected_fillers=10.0),
    ]
    reference = [
        replace(target[3], filler_class=1, expected_fillers=1.0),
        replace(target[2], expected_fillers=10.0),
        replace(target[1], expected_fillers=2.0),
    ]

    result = syntactic.reference_diagnosis(target, reference)

    assert [diagnosis.pattern for diagnosis in result] == target
    assert [diagnosis.reference for diagnosis in result] == [None, reference[2], reference[1], None]
    assert [diagnosis.verdict for diagnosis in result] == [
        "sufficient",
        "under-shown",
        "fixed",
        "unread",
    ]
    assert syntactic.reference_diagnosis(target[:1], [], low=0.6)[0].verdict == "unread"


def test_reference_coverage_uses_pattern_context_shares_and_target_trunk() -> None:
    classes = _classes(nouns=[10, 11, 12])
    reference = [[5, 0, 10, 6], [7, 0, 10, 6], [8, 0, 10, 6], [5, 0, 11, 6]]
    # Repetitions and another head must not change this pattern's reference shares.
    reference += [reference[-1]] * 5 + [[9, 11]] * 10
    target = [[5, 0, 10, 6], [7, 0, 10, 6]]

    (reading,) = syntactic.reference_coverage(target, reference, classes)

    assert reading.head == (0,)
    assert reading.filler_class == 0
    assert reading.trunk == 2
    assert reading.covered_mass == pytest.approx(0.75)
    expected = 0.75 * (1 - 0.25**2) + 0.25 * (1 - 0.75**2)
    assert expected == 0.8125
    assert reading.expected_mass == pytest.approx(expected)
    assert reading.ratio == pytest.approx(0.75 / 0.8125)
    assert replace(reading, expected_mass=0.0).ratio == 0.0


def test_reference_coverage_identical_corpora_cover_all_mass() -> None:
    classes = _classes(nouns=[10, 11, 12])
    sequences = [[2, 10], [2, 11], [0, 10], [0, 12]]

    readings = syntactic.reference_coverage(sequences, sequences, classes)

    assert [reading.head for reading in readings] == [(0,), (2,)]
    assert all(reading.covered_mass == pytest.approx(1.0) for reading in readings)
    assert all(reading.ratio >= 1.0 for reading in readings)
    assert all(reading.trunk == 2 for reading in readings)


def test_reference_coverage_exact_target_copies_change_no_readings() -> None:
    classes = _classes(nouns=[10, 11])
    reference = [[5, 0, 10, 6], [7, 0, 11, 6]]
    target = [[5, 0, 10, 6], [7, 0, 10, 6]]

    original = syntactic.reference_coverage(target, reference, classes)
    copied = syntactic.reference_coverage([*target, *target, target[0]], reference, classes)

    assert len(original) == 1
    assert copied == original


def test_reference_coverage_omits_patterns_absent_from_reference() -> None:
    classes = _classes(nouns=[10, 11], verbs=[20, 21])
    target = [[0, 10], [1, 10], [0, 20]]
    reference = [[0, 11], [2, 10]]

    readings = syntactic.reference_coverage(target, reference, classes)

    assert [(reading.head, reading.filler_class) for reading in readings] == [((0,), 0)]
    assert readings[0].covered_mass == 0.0
    assert readings[0].expected_mass == 1.0
    assert readings[0].ratio == 0.0
    assert syntactic.reference_coverage(target, [], classes) == []
    assert syntactic.reference_coverage([], reference, classes) == []


def test_reference_coverage_filters_heads_and_filler_classes() -> None:
    classes = _classes(nouns=[10, 11], fixed=[12], verbs=[20, 21])
    sequences = [[0, 1, 20], [0, 1, 10], [0, 2, 11], [2, 1, 10], [0, 1, 12], [0, 1, 99]]

    readings = syntactic.reference_coverage(sequences, sequences, classes, 2, heads={0, 1})

    assert [(reading.head, reading.filler_class) for reading in readings] == [
        ((0, 1), 0),
        ((0, 1), 2),
    ]
    assert all(reading.ratio == 1.0 for reading in readings)
    assert syntactic.reference_coverage(sequences, sequences, classes, 2, heads=set()) == []


def test_recipe_reference_coverage_separates_restricted_and_diverse_targets() -> None:
    reference_documents = _recipes(None)
    restricted_documents = _recipes(1)
    pairing = verb_pairing(LEXICON, None, reserved_per_verb=2, seed=0)
    diverse_documents = recipe_documents(LEXICON, pairing, count=2000, seed=1)
    sequences, ids = _encode([*reference_documents, *restricted_documents, *diverse_documents])
    reference, restricted, diverse = sequences[:2000], sequences[2000:4000], sequences[4000:]
    classes = induce_classes(reference)
    nouns = classes.classes[ids["tomato"]]
    heads = {(ids[verb], ids["the"]) for verb in LEXICON.verbs}
    means = []
    for target in (restricted, diverse):
        ratios = [
            reading.ratio
            for reading in syntactic.reference_coverage(target, reference, classes, head_length=2)
            if reading.filler_class == nouns and reading.head in heads
        ]
        assert len(ratios) == len(LEXICON.verbs) == 14
        means.append(sum(ratios) / len(ratios))
    restricted_mean, diverse_mean = means
    print(
        f"Recipe mean reference coverage ratios: restricted={restricted_mean:.12f}, "
        f"diverse={diverse_mean:.12f}"
    )
    assert restricted_mean <= 0.3
    assert diverse_mean >= 0.9
