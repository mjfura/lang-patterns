"""`analysis.syntactic` patterns, domain split and sufficiency curve — see
`docs/2026-09-28-syntactic-sufficiency-design.md`."""

from itertools import pairwise

import pytest

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
