"""`analysis.syntactic.induce_classes`: coarse syntactic classes from neighbor distributions — see
`docs/2026-09-28-syntactic-sufficiency-design.md`."""

import random
from collections import Counter
from itertools import pairwise

from sklearn.metrics import v_measure_score

from corpus_generalization.analysis.syntactic import induce_classes
from corpus_generalization.synthetic.recipes import LEXICON, recipe_documents, verb_pairing


def _truth() -> dict[str, str]:
    """The grammar's categories, singular and plural nouns apart and `is` and `are` apart: the
    agreement level a clean cut should reach."""
    truth = dict.fromkeys(LEXICON.verbs, "verb")
    truth |= {ingredient.singular: "singular" for ingredient in LEXICON.ingredients}
    truth |= {ingredient.plural: "plural" for ingredient in LEXICON.ingredients}
    truth |= dict.fromkeys(LEXICON.adjectives, "adjective")
    truth |= dict.fromkeys(LEXICON.adverbs, "adverb")
    truth |= {preposition: "preposition" for preposition, _ in LEXICON.places}
    truth |= {tool: "tool" for _, tool in LEXICON.places}
    return truth | {word: word for word in ("the", "is", "are", ".")}


def _encode(documents: list[str]) -> tuple[list[list[int]], dict[int, str]]:
    ids: dict[str, int] = {}
    sequences = [[ids.setdefault(word, len(ids)) for word in doc.split()] for doc in documents]
    return sequences, {index: word for word, index in ids.items()}


def _recipes(fillers_per_verb: int | None, count: int = 2000) -> list[str]:
    pairing = verb_pairing(LEXICON, fillers_per_verb, reserved_per_verb=2, seed=0)
    return recipe_documents(LEXICON, pairing, count=count, seed=0)


def _v_measure(classes: dict[int, int], words: dict[int, str]) -> float:
    truth = _truth()
    tokens = sorted(classes)
    return float(
        v_measure_score([truth[words[token]] for token in tokens], [classes[t] for t in tokens])
    )


def test_the_diverse_grammar_is_cut_at_its_eleven_categories() -> None:
    sequences, words = _encode(_recipes(None))

    result = induce_classes(sequences)

    assert result.n_classes == 11
    assert _v_measure(result.classes, words) == 1.0


def test_a_restricted_grammar_is_not_shattered_into_single_tokens() -> None:
    sequences, _ = _encode(_recipes(1))

    result = induce_classes(sequences)

    assert result.n_classes <= 16
    assert max(Counter(result.classes.values()).values()) >= 8


def test_a_fixed_number_of_classes_is_honored() -> None:
    sequences, _ = _encode(_recipes(None, count=300))

    result = induce_classes(sequences, n_classes=5)

    assert result.n_classes == 5
    assert len(set(result.classes.values())) == 5


def test_tokens_in_too_few_documents_are_left_unclassified() -> None:
    documents = [*_recipes(None, count=300), "cut the saffron ."]
    sequences, words = _encode(documents)

    result = induce_classes(sequences, min_documents=2)

    classified = {words[token] for token in result.classes}
    assert "saffron" not in classified
    assert "cut" in classified


def test_shuffled_words_lose_the_categories() -> None:
    rng = random.Random(0)
    shuffled = []
    for document in _recipes(None):
        tokens = document.split()
        rng.shuffle(tokens)
        shuffled.append(" ".join(tokens))
    sequences, words = _encode(shuffled)

    result = induce_classes(sequences, n_classes=11)

    assert _v_measure(result.classes, words) < 0.5


def test_merge_heights_do_not_decrease_toward_fewer_classes() -> None:
    sequences, _ = _encode(_recipes(None, count=300))

    heights = induce_classes(sequences).merge_heights

    assert all(a >= b for a, b in pairwise(heights))


def test_classes_are_deterministic() -> None:
    sequences, _ = _encode(_recipes(None, count=300))

    assert induce_classes(sequences) == induce_classes(sequences)
