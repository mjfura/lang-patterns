"""`analysis.branching`: prefix-trie branching entropy as a corpus-intrinsic cohesion
signal — tau (branching depth), IPBE (integrated prefix branching entropy) and CDCI (corpus domain
cohesion index), plus tokenizer fertility.

A tiny word-id tokenizer stands in for a real BPE tokenizer: `branching_entropy` only needs
`encode` to return token ids, and hand-verifying entropy math is far easier against fixed,
predictable ids than against whatever a learned BPE happens to merge.
"""

import math

import pytest

from corpus_generalization.analysis.branching import branching_entropy


class WordIdTokenizer:
    """One id per distinct whitespace-separated word, assigned in first-seen order."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            ids.append(self._ids.setdefault(word, len(self._ids)))
        return ids


def h(*counts: int) -> float:
    """Shannon entropy, in bits, of the distribution `counts` sums to — the same formula the
    module computes, used here to derive expected values instead of hardcoding decimals."""
    total = sum(counts)
    return -sum((count / total) * math.log2(count / total) for count in counts)


def test_two_documents_sharing_a_full_prefix_branch_all_the_way_through() -> None:
    """Identical documents: every position has both documents as company, so support never drops
    below 2 until the sequence ends, and every step carries zero entropy (only one continuation is
    ever observed)."""
    documents = ("a b c", "a b c")

    cohesion = branching_entropy(documents, WordIdTokenizer())

    for doc in cohesion.per_document:
        assert doc.tau == 2  # last of 3 positions (0, 1, 2)
        assert doc.ipbe == 0.0  # only one continuation ever observed at each step
        assert doc.fertility == 1.0  # one token per word, word-id tokenizer


def test_branching_depth_and_ipbe_match_a_hand_worked_corpus() -> None:
    """Two documents share 'I say' before diverging (good evidence of a shared skeleton); a third
    shares only the empty prefix (the trivial, always-true support of the whole corpus) before
    peeling off on its very first token."""
    documents = (
        "I say hello to you",
        "I say good morning to us",
        "You say hi to me",
    )

    cohesion = branching_entropy(documents, WordIdTokenizer())
    by_text = {doc.text: doc for doc in cohesion.per_document}

    # step 0: all 3 documents share the empty prefix -> entropy of {I: 2, You: 1} over 3
    # step 1 (after "I"): both "I"-documents share it -> entropy of {say: 2} over 2 -> 0 bits
    # step 2 (after "I say"): still both -> entropy of {hello: 1, good: 1} over 2 -> 1 bit
    # step 3 onward: each has the corpus to itself
    expected_shared_prefix_entropy = h(2, 1) + h(2) + h(1, 1)
    assert by_text["I say hello to you"].tau == 2
    assert by_text["I say hello to you"].ipbe == pytest.approx(expected_shared_prefix_entropy)
    assert by_text["I say good morning to us"].tau == 2
    assert by_text["I say good morning to us"].ipbe == pytest.approx(expected_shared_prefix_entropy)

    # "You say hi to me" only ever has company at the trivial empty-prefix step
    assert by_text["You say hi to me"].tau == 0
    assert by_text["You say hi to me"].ipbe == pytest.approx(h(2, 1))

    assert cohesion.cdci == pytest.approx((2 * expected_shared_prefix_entropy + h(2, 1)) / 3)


def test_an_intruder_collapses_to_zero_company_almost_immediately() -> None:
    """An out-of-domain document sharing no vocabulary with the rest of the corpus: its first token
    is already unique, so tau is the lowest it can be (only the trivial step 0 has company) and
    IPBE is just that one trivial term."""
    documents = (
        "I say hello to you",
        "I say good morning to us",
        "The pyramid stands alone",
    )

    cohesion = branching_entropy(documents, WordIdTokenizer())
    intruder = next(doc for doc in cohesion.per_document if doc.text == "The pyramid stands alone")

    assert intruder.tau == 0
    assert intruder.ipbe == pytest.approx(h(2, 1))


def test_fertility_is_tokens_per_word() -> None:
    class CharTokenizer:
        def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
            return [ord(char) for char in text if not char.isspace()]

    cohesion = branching_entropy(("ab cd",), CharTokenizer())

    assert cohesion.per_document[0].fertility == pytest.approx(4 / 2)


def test_a_single_document_corpus_has_no_company_and_so_no_branching() -> None:
    cohesion = branching_entropy(("only one here",), WordIdTokenizer())

    (doc,) = cohesion.per_document
    assert doc.tau == 0
    assert doc.ipbe == 0.0
    assert cohesion.cdci == 0.0


def test_steps_record_support_and_entropy_at_every_position() -> None:
    cohesion = branching_entropy(("a b", "a c"), WordIdTokenizer())

    doc = next(doc for doc in cohesion.per_document if doc.text == "a b")
    assert [step.support for step in doc.steps] == [2, 2]
    assert doc.steps[0].entropy_bits == pytest.approx(0.0)  # both share "a"
    assert doc.steps[1].entropy_bits == pytest.approx(1.0)  # split evenly into b/c
