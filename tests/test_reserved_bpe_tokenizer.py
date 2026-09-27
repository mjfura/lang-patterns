"""`ReservedBPETokenizer`: a byte-level BPE whose vocab_size is fixed before it learns anything."""

import pytest

from corpus_generalization.tokenizer.base import (
    BOS_ID,
    EOS_ID,
    IM_START_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    Tokenizer,
)
from corpus_generalization.tokenizer.bpe import BASE_TOKEN_ID, BYTE_VOCABULARY_SIZE, _learn_merges
from corpus_generalization.tokenizer.reserved_bpe import (
    FIRST_MERGE_ID,
    InvalidCapacityError,
    MergesExceedCapacityError,
    ReservedBPETokenizer,
    ReservedCapacityFullError,
)

CAPACITY = 1000
CORPUS = (
    "the cat sat on the mat",
    "the cat sat on the hat",
    "the dog sat on the log",
    "a cat and a dog sat on a mat",
    "def add(a, b):\n    return a + b",
    "def sub(a, b):\n    return a - b",
)
# ` zebra` twice, in text `CORPUS` never had, is what guarantees extending has a pair to learn
# whatever the first run already filled.
GROWN_CORPUS = (*CORPUS, "the zebra sat on the mat", "a zebra and a cat")
# How many merges `CORPUS` supports before the frequency rule stops it: measured, not chosen.
SUPPORTED = len(_learn_merges(CORPUS))


@pytest.fixture(scope="module")
def tokenizer() -> ReservedBPETokenizer:
    return ReservedBPETokenizer.trained(CORPUS, CAPACITY)


def test_the_corpus_supports_a_run_worth_cutting_and_stays_under_capacity() -> None:
    assert 4 <= SUPPORTED < CAPACITY


def test_merges_start_right_after_the_specials_and_the_bytes() -> None:
    assert FIRST_MERGE_ID == len(SPECIAL_TOKENS) + BYTE_VOCABULARY_SIZE == 261


def test_specials_and_bytes_keep_the_ids_the_byte_level_tokenizer_gives_them(
    tokenizer: ReservedBPETokenizer,
) -> None:
    assert (tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id) == (BOS_ID, EOS_ID, PAD_ID)
    assert tokenizer.encode("\x00") == [BASE_TOKEN_ID]


def test_vocab_size_counts_the_whole_capacity_filled_or_not(
    tokenizer: ReservedBPETokenizer,
) -> None:
    assert tokenizer.vocab_size == FIRST_MERGE_ID + CAPACITY
    assert tokenizer.filled + tokenizer.free == CAPACITY


def test_a_small_corpus_stops_at_the_frequency_rule_below_capacity(
    tokenizer: ReservedBPETokenizer,
) -> None:
    assert tokenizer.filled == SUPPORTED
    assert tokenizer.free == CAPACITY - SUPPORTED


def test_capacity_stops_learning_when_the_corpus_supports_more() -> None:
    capped = ReservedBPETokenizer.trained(CORPUS, 3)

    assert (capped.filled, capped.free, capped.vocab_size) == (3, 0, FIRST_MERGE_ID + 3)


def test_training_a_prefix_then_extending_learns_what_one_uninterrupted_run_learns() -> None:
    uninterrupted = ReservedBPETokenizer.trained(CORPUS, SUPPORTED)
    prefix = uninterrupted.merges[: SUPPORTED // 2]

    extended = ReservedBPETokenizer(prefix, SUPPORTED, recognize_markers=False).extended(CORPUS)

    assert extended.merges == uninterrupted.merges


def test_extending_on_a_grown_corpus_keeps_every_id_and_the_vocab_size(
    tokenizer: ReservedBPETokenizer,
) -> None:
    grown = tokenizer.extended(GROWN_CORPUS)

    assert grown.merges[: tokenizer.filled] == tokenizer.merges
    assert grown.filled > tokenizer.filled
    assert (grown.vocab_size, grown.capacity) == (tokenizer.vocab_size, tokenizer.capacity)


def test_extending_leaves_the_tokenizer_it_was_called_on_untouched(
    tokenizer: ReservedBPETokenizer,
) -> None:
    before = tokenizer.merges

    tokenizer.extended(GROWN_CORPUS)

    assert tokenizer.merges == before


def test_extending_stops_at_capacity() -> None:
    empty = ReservedBPETokenizer((), 2, recognize_markers=False)

    assert empty.extended(CORPUS).filled == 2


def test_extending_with_every_slot_filled_raises() -> None:
    full = ReservedBPETokenizer.trained(CORPUS, 3)

    with pytest.raises(ReservedCapacityFullError):
        full.extended(GROWN_CORPUS)


@pytest.mark.parametrize("recognize_markers", [True, False])
def test_extending_learns_under_the_marker_setting_the_tokenizer_was_trained_with(
    recognize_markers: bool,
) -> None:
    untrained = ReservedBPETokenizer.trained(("x",), CAPACITY, recognize_markers=recognize_markers)

    grown = untrained.extended(("<|im_start|> <|im_start|>",))

    assert grown.recognize_markers is recognize_markers
    # Recognized, the markers are removed before counting and only a lone space is left to merge.
    assert (grown.filled == 0) is recognize_markers


def test_more_merges_than_capacity_is_rejected() -> None:
    merges = ReservedBPETokenizer.trained(CORPUS, 5).merges

    with pytest.raises(MergesExceedCapacityError):
        ReservedBPETokenizer(merges, 4, recognize_markers=False)


@pytest.mark.parametrize("capacity", [0, -1])
def test_a_capacity_below_one_is_rejected(capacity: int) -> None:
    with pytest.raises(InvalidCapacityError):
        ReservedBPETokenizer((), capacity, recognize_markers=False)
    with pytest.raises(InvalidCapacityError):
        ReservedBPETokenizer.trained(CORPUS, capacity)


def test_encode_never_produces_a_free_slot_id(tokenizer: ReservedBPETokenizer) -> None:
    first_free = FIRST_MERGE_ID + tokenizer.filled

    assert all(token < first_free for text in GROWN_CORPUS for token in tokenizer.encode(text))


def test_decode_drops_a_free_slot_id_wherever_it_appears(tokenizer: ReservedBPETokenizer) -> None:
    assert tokenizer.free > 0
    free_id = tokenizer.vocab_size - 1

    assert tokenizer.decode([free_id, *tokenizer.encode("the cat"), free_id]) == "the cat"


def test_decode_rejects_an_id_past_the_vocabulary(tokenizer: ReservedBPETokenizer) -> None:
    with pytest.raises(ValueError, match=str(tokenizer.vocab_size)):
        tokenizer.decode([tokenizer.vocab_size])


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat",
        "def add(a, b):\n    return a + b",
        "naïve café 日本語 🙂",
        "an unseen zebra",
    ],
)
def test_encode_then_decode_returns_the_text(tokenizer: ReservedBPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_markers_are_recognized_only_when_the_caller_opts_in() -> None:
    tokenizer = ReservedBPETokenizer.trained(CORPUS, CAPACITY, recognize_markers=True)

    assert tokenizer.encode("<|im_start|>the cat", recognize_markers=True)[0] == IM_START_ID
    assert IM_START_ID not in tokenizer.encode("<|im_start|>the cat")


def test_it_satisfies_the_tokenizer_protocol(tokenizer: ReservedBPETokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)
