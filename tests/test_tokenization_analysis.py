"""`analysis.tokenization`: how a corpus tokenizes under a tokenizer learned from it."""

import math

import pytest

from corpus_generalization.analysis.tokenization import learned_tokenizer, tokenization_stats
from corpus_generalization.tokenizer.reserved_bpe import FIRST_MERGE_ID

DOCUMENTS = ("the cat sat on the mat", "the cat ran to the mat")


def test_the_counts_add_up_and_the_entropy_is_that_of_the_token_distribution() -> None:
    tokenizer = learned_tokenizer(DOCUMENTS, "reserved", merge_budget=10)

    stats = tokenization_stats(DOCUMENTS, tokenizer, merge_budget=10)

    counts = [row.count for row in stats.rows if row.count]
    expected = -sum((count / stats.tokens) * math.log2(count / stats.tokens) for count in counts)
    assert (stats.documents, stats.words) == (2, 12)
    assert stats.bytes == sum(len(document.encode()) for document in DOCUMENTS)
    assert stats.tokens == sum(counts)
    assert stats.distinct_tokens == len(counts)
    assert stats.entropy_bits == pytest.approx(expected)
    assert stats.perplexity == pytest.approx(2**expected)
    assert stats.bits_per_byte == pytest.approx(expected * stats.tokens / stats.bytes)


def test_a_reserved_tokenizer_counts_its_whole_budget_in_the_vocabulary() -> None:
    tokenizer = learned_tokenizer(DOCUMENTS, "reserved", merge_budget=10)

    stats = tokenization_stats(DOCUMENTS, tokenizer, merge_budget=10)

    assert stats.vocab_size == FIRST_MERGE_ID + 10


def test_a_byte_level_tokenizer_counts_only_the_merges_it_learned() -> None:
    tokenizer = learned_tokenizer(DOCUMENTS, "byte_level", merge_budget=10)

    stats = tokenization_stats(DOCUMENTS, tokenizer, merge_budget=10)

    assert stats.vocab_size == FIRST_MERGE_ID + stats.merges


def test_every_learned_merge_is_listed_with_what_it_joins_even_when_nothing_uses_it_alone() -> None:
    """A merge a later merge absorbs never appears on its own in the encoded corpus, but it is
    still one of the tokens the tokenizer learned, so the list keeps it — at a count of zero."""
    tokenizer = learned_tokenizer(DOCUMENTS, "reserved", merge_budget=10)

    stats = tokenization_stats(DOCUMENTS, tokenizer, merge_budget=10)

    by_id = {row.token_id: row for row in stats.rows}
    assert [row.token_id for row in stats.rows if row.kind == "merge"] == [
        new_id for _, new_id in tokenizer.merges
    ]
    for (left, right), new_id in tokenizer.merges:
        assert by_id[new_id].parts == (left, right)
        assert by_id[new_id].text == by_id[left].text + by_id[right].text


def test_one_token_repeated_carries_no_entropy() -> None:
    documents = ("a", "a")
    tokenizer = learned_tokenizer(documents, "reserved", merge_budget=10)

    stats = tokenization_stats(documents, tokenizer, merge_budget=10)

    assert stats.entropy_bits == 0.0
    assert stats.normalized_entropy == 0.0
    assert stats.singletons == 0
