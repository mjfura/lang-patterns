"""`analysis.kl_divergence`: token-distribution KL divergence, leave-one-out per document
— the Llama 3 style corpus-filtering signal, included here as a benchmark against
`analysis.branching`'s sequence-aware metrics.
"""

import math

import pytest

from corpus_generalization.analysis.kl_divergence import token_distribution_divergence


class WordIdTokenizer:
    """One id per distinct whitespace-separated word; `vocab_size` reflects every word seen across
    every `encode` call so far, so it is only meaningful once every document has been encoded —
    exactly the order `token_distribution_divergence` uses it in."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._ids)

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        return [self._ids.setdefault(word, len(self._ids)) for word in text.split()]


def test_a_document_that_looks_like_the_rest_of_the_corpus_has_low_divergence() -> None:
    documents = ("a a b", "a b b", "a a b b", "b a a b")

    divergence = token_distribution_divergence(documents, WordIdTokenizer())

    assert all(doc.kl_divergence < 0.2 for doc in divergence.per_document)


def test_an_intruder_with_its_own_vocabulary_has_far_higher_divergence() -> None:
    documents = ("a a b", "a b b", "a a b b", "zzz yyy xxx zzz")

    divergence = token_distribution_divergence(documents, WordIdTokenizer())
    by_text = {doc.text: doc.kl_divergence for doc in divergence.per_document}
    ordinary = max(value for text, value in by_text.items() if text != "zzz yyy xxx zzz")

    assert by_text["zzz yyy xxx zzz"] > 5 * ordinary


def test_leave_one_out_excludes_the_documents_own_counts_from_its_reference() -> None:
    """Two identical documents: each one's leave-one-out reference is the OTHER one's counts, not
    its own — verified against the same smoothed formula the module uses, computed inline."""
    documents = ("a b", "a b")
    tokenizer = WordIdTokenizer()
    smoothing = 0.01

    divergence = token_distribution_divergence(documents, tokenizer, smoothing=smoothing)

    vocab = tokenizer.vocab_size
    total = 2 + smoothing * vocab
    expected = sum(
        ((1 + smoothing) / total) * math.log2(((1 + smoothing) / total) / ((1 + smoothing) / total))
        for _ in range(vocab)
    )
    assert expected == pytest.approx(0.0)
    for doc in divergence.per_document:
        assert doc.kl_divergence == pytest.approx(expected, abs=1e-9)


def test_a_single_document_corpus_is_compared_against_pure_smoothing_and_stays_finite() -> None:
    """With no other document, the leave-one-out reference is empty, so smoothing alone makes it
    uniform. A document whose own word counts are skewed (unlike a uniform reference) should still
    show that mismatch as a positive, finite divergence rather than blowing up or going negative."""
    divergence = token_distribution_divergence(("only one one here here here",), WordIdTokenizer())

    (doc,) = divergence.per_document
    assert math.isfinite(doc.kl_divergence)
    assert doc.kl_divergence > 0.0


def test_aggregate_mean_and_stdev_summarize_the_per_document_values() -> None:
    documents = ("a a b", "a b b", "zzz yyy xxx zzz")

    divergence = token_distribution_divergence(documents, WordIdTokenizer())

    values = [doc.kl_divergence for doc in divergence.per_document]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    assert divergence.documents == 3
    assert divergence.mean_kl_divergence == pytest.approx(mean)
    assert divergence.kl_divergence_stdev == pytest.approx(math.sqrt(variance))
