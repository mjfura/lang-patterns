"""`synthetic.recipes`: the recipe grammar of `docs/2026-09-28-syntactic-sufficiency-design.md`."""

from collections import defaultdict

import pytest

from corpus_generalization.synthetic.recipes import (
    LEXICON,
    foreign_documents,
    minimal_pairs,
    recipe_documents,
    verb_pairing,
)

SINGULAR = {ingredient.plural: ingredient.singular for ingredient in LEXICON.ingredients} | {
    ingredient.singular: ingredient.singular for ingredient in LEXICON.ingredients
}
PLURALS = {ingredient.plural for ingredient in LEXICON.ingredients}


def _sentences(documents: list[str]) -> list[list[str]]:
    sentences: list[list[str]] = []
    for document in documents:
        assert document.endswith(" .")
        sentences.extend(part.split() for part in document[: -len(" .")].split(" . "))
    return sentences


def _actions(documents: list[str]) -> dict[str, set[str]]:
    """Each verb's ingredients (singular form) over every action sentence."""
    seen: dict[str, set[str]] = defaultdict(set)
    for words in _sentences(documents):
        if words[0] in LEXICON.verbs:
            seen[words[0]].update(SINGULAR[word] for word in words if word in SINGULAR)
    return seen


def test_documents_follow_the_four_rules() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=None, reserved_per_verb=2, seed=0)
    documents = recipe_documents(LEXICON, pairing, count=200, seed=0)

    for words in _sentences(documents):
        if words[0] in LEXICON.verbs:
            assert words[1] == "the"
            noun_at = 3 if words[2] in LEXICON.adjectives else 2
            assert words[noun_at] in SINGULAR
            rest = words[noun_at + 1 :]
            assert (
                rest == []
                or (len(rest) == 1 and rest[0] in LEXICON.adverbs)
                or (len(rest) == 3 and (rest[0], rest[2]) in LEXICON.places and rest[1] == "the")
            )
        else:
            assert words[0] == "the" and words[1] in SINGULAR
            assert words[2] == ("are" if words[1] in PLURALS else "is")
            assert words[3] in LEXICON.adjectives and len(words) == 4


def test_every_document_has_four_to_ten_sentences() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=None, reserved_per_verb=2, seed=0)
    for document in recipe_documents(LEXICON, pairing, count=100, seed=1):
        assert 4 <= document.count(" .") <= 10


def test_reserved_combinations_never_occur() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=None, reserved_per_verb=2, seed=0)
    documents = recipe_documents(LEXICON, pairing, count=500, seed=0)

    for verb, ingredients in _actions(documents).items():
        for ingredient in ingredients:
            assert (verb, ingredient) not in pairing.reserved_actions
    for words in _sentences(documents):
        if words[0] == "the":
            assert (SINGULAR[words[1]], words[3]) not in pairing.reserved_states


def test_diverse_verbs_meet_every_ingredient_not_reserved() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=None, reserved_per_verb=2, seed=0)
    documents = recipe_documents(LEXICON, pairing, count=2000, seed=0)

    every = {ingredient.singular for ingredient in LEXICON.ingredients}
    for verb, ingredients in _actions(documents).items():
        reserved = {noun for other, noun in pairing.reserved_actions if other == verb}
        assert ingredients == every - reserved


def test_restricted_verbs_meet_only_their_few_ingredients() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=2, reserved_per_verb=2, seed=0)
    documents = recipe_documents(LEXICON, pairing, count=2000, seed=0)

    for verb, ingredients in _actions(documents).items():
        assert ingredients == set(pairing.allowed[verb])
        assert len(ingredients) == 2


def test_every_ingredient_occurs_in_the_restricted_variant() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=1, reserved_per_verb=2, seed=0)
    documents = recipe_documents(LEXICON, pairing, count=500, seed=0)

    words = {word for document in documents for word in document.split()}
    for ingredient in LEXICON.ingredients:
        assert ingredient.singular in words or ingredient.plural in words


def test_variants_of_a_seed_reserve_the_same_combinations() -> None:
    diverse = verb_pairing(LEXICON, fillers_per_verb=None, reserved_per_verb=2, seed=3)
    restricted = verb_pairing(LEXICON, fillers_per_verb=1, reserved_per_verb=2, seed=3)

    assert diverse.reserved_actions == restricted.reserved_actions
    assert diverse.reserved_states == restricted.reserved_states
    assert len(diverse.reserved_actions) == 2 * len(LEXICON.verbs)
    assert len(diverse.reserved_states) == len(LEXICON.ingredients)


def test_more_fillers_than_the_unreserved_ingredients_is_rejected() -> None:
    too_many = len(LEXICON.ingredients) - 1
    with pytest.raises(ValueError, match="fillers_per_verb"):
        verb_pairing(LEXICON, fillers_per_verb=too_many, reserved_per_verb=2, seed=0)


def test_generation_is_deterministic_per_seed() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=None, reserved_per_verb=2, seed=0)

    first = recipe_documents(LEXICON, pairing, count=20, seed=5)
    assert first == recipe_documents(LEXICON, pairing, count=20, seed=5)
    assert first != recipe_documents(LEXICON, pairing, count=20, seed=6)


def test_minimal_pairs_test_reserved_combinations_by_one_word() -> None:
    pairing = verb_pairing(LEXICON, fillers_per_verb=None, reserved_per_verb=2, seed=0)
    pairs = minimal_pairs(LEXICON, pairing, seed=0)

    verb_pairs = [pair for pair in pairs if pair.kind == "verb"]
    agreement_pairs = [pair for pair in pairs if pair.kind == "agreement"]
    assert len(verb_pairs) == len(pairing.reserved_actions)
    assert len(agreement_pairs) == 2 * len(pairing.reserved_states)

    for pair in pairs:
        good, bad = pair.good.split(), pair.bad.split()
        assert len(good) == len(bad)
        assert sum(a != b for a, b in zip(good, bad, strict=True)) == 1
    for pair in verb_pairs:
        verb, _, noun, _ = pair.good.split()
        assert (verb, noun) in pairing.reserved_actions
        assert pair.bad.split()[2] in LEXICON.adverbs
    for pair in agreement_pairs:
        _, noun, verb, adjective = pair.good.split()[:4]
        assert (SINGULAR[noun], adjective) in pairing.reserved_states
        assert verb == ("are" if noun in PLURALS else "is")
        assert pair.bad.split()[2] == ("is" if verb == "are" else "are")


def test_foreign_documents_share_no_content_word_with_recipes() -> None:
    recipe_words = (
        set(LEXICON.verbs)
        | set(SINGULAR)
        | set(LEXICON.adjectives)
        | set(LEXICON.adverbs)
        | {tool for _, tool in LEXICON.places}
    )
    documents = foreign_documents(count=50, seed=0)

    assert len(documents) == 50
    words = {word for document in documents for word in document.split()}
    assert not words & recipe_words
