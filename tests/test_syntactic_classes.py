"""`analysis.syntactic.induce_classes`: coarse syntactic classes from neighbor distributions — see
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md`."""

import random
from collections import Counter
from itertools import pairwise

import numpy as np
import pytest
from sklearn.metrics import v_measure_score

from corpus_generalization.analysis.syntactic import _neighbor_features, class_tree, induce_classes
from corpus_generalization.packing import framed_documents
from corpus_generalization.synthetic.recipes import LEXICON, recipe_documents, verb_pairing
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer


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


def test_small_bpe_grammar_keeps_ingredients_separate_from_determiners() -> None:
    documents = _recipes(None, count=500)
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        documents, max_merges=1000, recognize_markers=True
    )
    sequences = framed_documents(documents, tokenizer, True)
    tomato = tokenizer.encode(" tomato", recognize_markers=True)
    determiner = tokenizer.encode(" the", recognize_markers=True)
    assert len(tomato) == len(determiner) == 1

    result = induce_classes(sequences)

    assert result.n_classes >= 8
    assert result.classes[tomato[0]] != result.classes[determiner[0]]


@pytest.mark.parametrize(
    ("sequences", "max_classes", "expected_classes"),
    [([[0, 1, 2], [2, 1, 0]], 64, 2), ([[0, 1, 2], [2, 1, 0]], 1, 1), ([[0], [0]], 64, 1)],
)
def test_min_classes_above_token_count_uses_largest_valid_cut(
    sequences: list[list[int]], max_classes: int, expected_classes: int
) -> None:
    result = induce_classes(sequences, min_classes=100, max_classes=max_classes)

    assert result.n_classes == expected_classes
    assert len(set(result.classes.values())) == expected_classes
    assert set(result.classes) == {token for sequence in sequences for token in sequence}


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


def test_candidates_restrict_members_but_keep_whole_corpus_neighbors() -> None:
    sequences = [[0, 10, 1, 0, 11, 1, 2, 12, 3, 2, 13, 3]] * 2 + [[14]]

    result = induce_classes(sequences, n_classes=2, candidates={10, 11, 12, 13, 14})

    assert set(result.classes) == {10, 11, 12, 13}
    assert result.classes[10] == result.classes[11]
    assert result.classes[12] == result.classes[13]
    assert result.classes[10] != result.classes[12]


def test_every_cut_of_one_tree_refines_the_coarser_ones() -> None:
    sequences, _ = _encode(_recipes(None, count=300))
    tree = class_tree(sequences)

    coarse, fine = tree.cut(), tree.cut(20)

    assert coarse == induce_classes(sequences)
    assert fine.n_classes == 20
    for label in set(fine.classes.values()):
        members = [token for token, fine_label in fine.classes.items() if fine_label == label]
        assert len({coarse.classes[token] for token in members}) == 1


def test_neighbor_features_are_the_square_roots_of_direct_neighbor_shares() -> None:
    sequences = [[1, 2, 3], [2, 2], [], [3]]

    features = _neighbor_features(sequences, [2, 3], [2])

    # Columns per side: token 2, every other token, the sequence boundary.
    left_of_2, right_of_2 = [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]
    left_of_3, right_of_3 = [1 / 2, 0.0, 1 / 2], [0.0, 0.0, 1.0]
    expected = np.sqrt([left_of_2 + right_of_2, left_of_3 + right_of_3])
    assert np.allclose(features, expected)
