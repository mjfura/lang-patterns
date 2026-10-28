"""`analysis.information`: held-out information gain, in bits per byte — see
`docs/superpowers/specs/2026-09-28-heldout-information-design.md`."""

import random
import string

import pytest

from corpus_generalization.analysis.information import (
    fit_ngram,
    heldout_information,
    probability,
    split_halves,
)

ALPHABET = string.ascii_letters + string.digits + "+/"  # 64 one-byte symbols: 6 bits each


def _grammar_documents(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    adjectives, nouns = ("big", "small", "red", "old"), ("cat", "dog", "bird", "fox", "cow")
    verbs = ("sees", "chases", "likes", "follows")

    def sentence() -> str:
        return (
            f"the {rng.choice(adjectives)} {rng.choice(nouns)} {rng.choice(verbs)} "
            f"the {rng.choice(nouns)} ."
        )

    return [" ".join(sentence() for _ in range(8)) for _ in range(count)]


def test_every_conditional_distribution_sums_to_one() -> None:
    sequences = [[0, 5, 6, 7, 5, 6, 8, 1], [0, 6, 7, 7, 5, 1]]
    model = fit_ngram(sequences, order=2, vocab_size=10)

    for context in ((), (5,), (5, 6), (6, 7), (9, 9)):
        total = sum(probability(model, context, token) for token in range(10))
        assert total == pytest.approx(1.0)


def test_random_symbols_read_six_bits_per_byte_and_gain_nothing() -> None:
    rng = random.Random(0)
    documents = ["".join(rng.choice(ALPHABET) for _ in range(3000)) for _ in range(40)]

    information = heldout_information(documents, merge_budget=0, seed=0)

    assert information.heldout_bpb[0] == pytest.approx(6.0, abs=0.1)
    for order in (1, 2, 3):
        assert information.gain[order] < 0.05


def test_a_deterministic_cycle_is_almost_free_to_predict_given_context() -> None:
    documents = [" ".join(["a b c"] * (200 + index)) for index in range(20)]

    information = heldout_information(documents, merge_budget=0, seed=0)

    assert information.heldout_bpb[2] < 0.1
    assert information.gain[2] > 1.0


def test_shuffling_the_words_removes_most_of_the_gain() -> None:
    documents = _grammar_documents(60, seed=0)
    rng = random.Random(1)
    shuffled = []
    for document in documents:
        words = document.split()
        rng.shuffle(words)
        shuffled.append(" ".join(words))

    ordered = heldout_information(documents, merge_budget=50, seed=0)
    disordered = heldout_information(shuffled, merge_budget=50, seed=0)

    assert ordered.gain[2] > disordered.gain[2] + 0.3


def test_the_split_is_seeded() -> None:
    documents = _grammar_documents(30, seed=2)

    first = heldout_information(documents, merge_budget=20, orders=(0, 1), seed=0)

    assert heldout_information(documents, 20, orders=(0, 1), seed=0) == first
    assert heldout_information(documents, 20, orders=(0, 1), seed=1) != first
    assert first.documents == {"fit": 15, "heldout": 15}


def test_every_copy_of_a_document_falls_in_the_same_half() -> None:
    """A copy on each side would let the held-out half be predicted from text the fit half
    memorized (`experiments/information`: 50% duplicates read 1.72 bits per byte, the clean
    control 2.23)."""
    documents = _grammar_documents(20, seed=3)
    duplicated = documents + documents[:10] * 3

    information = heldout_information(duplicated, merge_budget=20, orders=(0, 1), seed=0)
    fit, heldout = split_halves(duplicated, seed=0)

    assert not set(fit) & set(heldout)
    assert information.documents == {"fit": len(fit), "heldout": len(heldout)}


def test_a_corpus_needs_two_distinct_texts() -> None:
    with pytest.raises(ValueError, match="distinct"):
        heldout_information(["same text"] * 5, merge_budget=0)
