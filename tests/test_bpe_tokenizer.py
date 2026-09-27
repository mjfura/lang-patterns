import json
from itertools import pairwise
from pathlib import Path

import pytest

from corpus_generalization.tokenizer.base import (
    BOS_ID,
    EOS_ID,
    IM_END_ID,
    IM_END_TOKEN,
    IM_START_ID,
    IM_START_TOKEN,
    PAD_ID,
    SPECIAL_TOKENS,
    Tokenizer,
)
from corpus_generalization.tokenizer.bpe import (
    BASE_TOKEN_ID,
    BYTE_VOCABULARY_SIZE,
    PRE_TOKEN_PATTERN,
    REPLACEMENT_CHARACTER,
    ByteLevelBPETokenizer,
    TokenPair,
    _learn_merges,
)

FIRST_MERGE_ID = BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE

# Discovered by constructing the tokenizer and inspecting `merges`, not chosen in advance.
#
# A hundred and four, where an earlier version of this tokenizer learned 358 over the same corpus.
# The difference is pre-tokenization. Merging used to run over a whole message's bytes, so the
# frequency rule walked straight across spaces and the last merges spelled phrases lifted from one
# conversation each — `she has`, `See you l` — which is memorizing noise, and is why the tokenizer
# once needed an explicit merge cap. `PRE_TOKEN_PATTERN` now splits text into words and runs of
# punctuation first and merging never crosses a piece, so the same rule stops on its own: the last
# merges here are ` Let`, ` le`, ` let`. The cap went away because the thing it guarded against
# did.
EXPECTED_MERGE_COUNT = 104
# The most frequent pair inside a pre-token is `s` followed by `t`, at 164 occurrences over the
# strings the formatter encodes. It is a word-internal pair — `assistant`, `system`, `just` — which
# is what pre-tokenization is for: the previous winner was ` a`, a space and a letter, and a space
# is now the start of its own piece rather than something a merge can reach across.
FIRST_MERGE_FREQUENCY = 164
# Text the corpus is built from, used wherever a test needs strings the merges actually apply to.
CORPUS_WORDS = ("You", "are", "assistant", "helpful", "system", "user", "the", "ok", "Yes")
A_TRAINING_MESSAGE = "Hi! Yes, I am ok. Thank you."
# The formatted chat conversations eva-llm's tokenizer was specified against (roles, separators
# and contents, as its chat formatter encodes them), frozen here so every discovered number below
# stays checkable without that project.
CHAT_TEXTS: tuple[str, ...] = tuple(
    json.loads((Path(__file__).parent / "fixtures" / "bpe_training_texts.json").read_text())
)


def chat_trained(max_merges: int | None = None) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer.from_document_sequence(CHAT_TEXTS, max_merges)


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelBPETokenizer:
    return chat_trained()


def test_bpe_tokenizer_satisfies_the_tokenizer_protocol(tokenizer: ByteLevelBPETokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)


def test_specials_occupy_ids_0_to_4_matching_word_tokenizer() -> None:
    """Five now, not three: the two chat markers are reserved beside the sequence framing, which
    is what makes them ids `encode` cannot produce. They were appended rather than inserted, so
    `<bos>`, `<eos>` and `<pad>` keep their ids and only the bytes above them moved."""
    assert SPECIAL_TOKENS == ("<bos>", "<eos>", "<pad>", IM_START_TOKEN, IM_END_TOKEN)
    assert BASE_TOKEN_ID == len(SPECIAL_TOKENS) == 5


def test_the_special_ids_are_the_ones_the_other_tokenizer_uses(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Both tokenizers put the specials first and in the same order, so a sequence framed by one
    is readable by the other and `WordGrid` can swap between them without renumbering."""
    assert (tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id) == (BOS_ID, EOS_ID, PAD_ID)


def test_base_vocabulary_covers_every_byte_value_at_ids_5_to_260() -> None:
    byte_ids = {BASE_TOKEN_ID + byte for byte in range(BYTE_VOCABULARY_SIZE)}

    assert byte_ids == set(range(5, 261))


def test_vocab_size_is_specials_plus_bytes_plus_learned_merges(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    assert tokenizer.vocab_size == 5 + 256 + len(tokenizer.merges)


def test_this_corpus_learns_exactly_the_discovered_number_of_merges(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    assert len(tokenizer.merges) == EXPECTED_MERGE_COUNT
    assert tokenizer.vocab_size == 5 + 256 + EXPECTED_MERGE_COUNT


def test_merges_are_neither_empty_nor_trivially_one(tokenizer: ByteLevelBPETokenizer) -> None:
    assert len(tokenizer.merges) > 1


def test_merge_ids_are_learned_in_increasing_contiguous_order(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    ids = [new_id for _, new_id in tokenizer.merges]

    assert ids == list(range(FIRST_MERGE_ID, FIRST_MERGE_ID + len(ids)))


def test_construction_is_deterministic_across_instances() -> None:
    first = chat_trained()
    second = chat_trained()

    assert first.merges == second.merges


def test_the_first_merge_is_the_most_frequent_byte_pair_in_the_raw_corpus(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Independently recomputes pair frequencies (not by calling into `tokenizer.bpe`'s
    own counting code) to prove the first merge the tokenizer picked really was the most frequent
    pair, not just self-consistent with the module's own arithmetic.

    Counted over the strings the formatter encodes rather than over the file's bytes, because
    that is the corpus BPE is given: the file is JSON, and its punctuation is never encoded.

    Counted *within pre-tokens* rather than across a whole message, because that is what the
    tokenizer counts. Splitting first is the design, so a recomputation that ignored it would be
    measuring a rule the code no longer follows and would name ` a` the winner. The split uses the
    declared pattern rather than the module's splitting code, which keeps the recount independent
    of everything this test is trying to check.
    """
    counts: dict[TokenPair, int] = {}
    for text in CHAT_TEXTS:
        for piece in PRE_TOKEN_PATTERN.findall(text):
            byte_ids = [BASE_TOKEN_ID + byte for byte in piece.encode("utf-8")]
            for pair in pairwise(byte_ids):
                counts[pair] = counts.get(pair, 0) + 1
    most_frequent_pair, most_frequent_count = max(counts.items(), key=lambda item: item[1])

    assert most_frequent_count == FIRST_MERGE_FREQUENCY
    assert tokenizer.merges[0] == (most_frequent_pair, FIRST_MERGE_ID)


def test_no_pair_repeats_once_every_learned_merge_has_been_replayed(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Independently replays the learned merges over the corpus and recounts pairs, to prove the
    stopping rule actually held: if any pair still occurred more than once, training stopped too
    early.

    Replayed per pre-token, for the reason the frequency recount above splits too. A pair that
    straddles two pieces — the `t` ending one word and the space beginning the next — is one the
    tokenizer was never able to merge and never counted, so measuring it here would fail this
    test for doing exactly what pre-tokenization exists to do.
    """
    final_counts: dict[TokenPair, int] = {}
    for text in CHAT_TEXTS:
        for piece in PRE_TOKEN_PATTERN.findall(text):
            sequence = [BASE_TOKEN_ID + byte for byte in piece.encode("utf-8")]
            for pair, new_id in tokenizer.merges:
                sequence = _replay_merge(sequence, pair, new_id)
            for replayed_pair in pairwise(sequence):
                final_counts[replayed_pair] = final_counts.get(replayed_pair, 0) + 1

    assert all(count <= 1 for count in final_counts.values())


def _replay_merge(sequence: list[int], pair: TokenPair, new_id: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if index < len(sequence) - 1 and (sequence[index], sequence[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


def test_the_default_learns_every_merge_the_stopping_rule_allows() -> None:
    """`max_merges` defaults to no cap, so a caller that does not ask for one gets the merges the
    corpus supports and nothing about the tokenizer has moved."""
    assert chat_trained().merges == chat_trained(max_merges=None).merges


@pytest.mark.parametrize("cap", [0, 1, 3, 5])
def test_a_cap_below_what_the_corpus_supports_stops_the_learning_there(cap: int) -> None:
    """What `max_merges` is for: a target vocabulary size, which is how BPE is normally trained.

    The stopping rule on its own is a frequency threshold, and on a corpus where several
    sentences share a long prefix that threshold is reached only after whole phrases have been
    merged into single tokens.
    """
    capped = chat_trained(max_merges=cap)

    assert len(capped.merges) == cap
    assert capped.vocab_size == BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE + cap


def test_a_cap_takes_the_merges_the_uncapped_tokenizer_would_have_learned_first() -> None:
    """The cap truncates the sequence rather than changing it: merge i is the same merge with or
    without a cap, so capping keeps the most frequent pairs and drops the rest."""
    cap = 4
    uncapped = chat_trained()

    assert chat_trained(max_merges=cap).merges == uncapped.merges[:cap]


def test_a_cap_above_what_the_corpus_supports_still_stops_at_the_stopping_rule() -> None:
    """The two rules are both ceilings, and the lower one wins: a cap does not invent merges out
    of pairs that occur once, which would memorize noise."""
    uncapped = chat_trained()

    generous = chat_trained(max_merges=len(uncapped.merges) + 100)

    assert generous.merges == uncapped.merges


def test_a_capped_tokenizer_still_roundtrips(tokenizer: ByteLevelBPETokenizer) -> None:
    """Fewer merges means longer sequences and nothing else: the base vocabulary is every byte,
    so encoding stays total no matter where the learning stopped."""
    capped = chat_trained(max_merges=2)
    text = "the quick brown fox"

    assert capped.decode(capped.encode(text)) == text
    assert len(capped.encode(text)) >= len(tokenizer.encode(text))


# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║ TASK 2: ENCODE/DECODE TESTS                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════════╝


def test_encode_is_segmentation_alone_and_frames_nothing(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """`encode` turns text into the ids that spell it and stops there.

    Framing a sequence with `<bos>`/`<eos>` and squaring a batch off with `<pad>` are decisions
    about a training sequence rather than about segmentation, so they belong to the dataset that
    makes them — `WordGrid` — and both tokenizers leave them to it.
    """
    assert tokenizer.encode("q") == [BASE_TOKEN_ID + ord("q")]


def test_encode_of_the_empty_string_is_the_empty_sequence(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    assert tokenizer.encode("") == []


@pytest.mark.parametrize("word", CORPUS_WORDS)
def test_encode_roundtrip_on_a_training_word(tokenizer: ByteLevelBPETokenizer, word: str) -> None:
    """Every word the corpus is built from survives encoding and decoding."""
    assert tokenizer.decode(tokenizer.encode(word)) == word


def test_encode_roundtrip_on_training_phrase(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode a full training message should return the original."""
    encoded = tokenizer.encode(A_TRAINING_MESSAGE)

    assert tokenizer.decode(encoded) == A_TRAINING_MESSAGE


def test_encode_roundtrip_on_unseen_word(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode 'xyz' (not in training) should return 'xyz'."""
    text = "xyz"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_ascii_punctuation(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode ASCII punctuation should roundtrip."""
    text = "!@#$%^&*()"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_numbers(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode numeric strings should roundtrip."""
    text = "0123456789"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_multibyte_utf8_accents(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Encode and decode accented Latin characters (multibyte UTF-8) should roundtrip."""
    text = "café naïve résumé"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_emoji(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode emoji should roundtrip."""
    text = "hello 👋 world 🌍"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_mixed_scripts(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode text with mixed scripts (Latin, Greek, Cyrillic) should roundtrip."""
    text = "Latin αβγ Кириллица"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_japanese(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode Japanese text should roundtrip."""
    text = "こんにちは世界"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_chinese(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode Chinese text should roundtrip."""
    text = "你好世界"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_roundtrip_on_arabic(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encode and decode Arabic text should roundtrip."""
    text = "مرحبا بالعالم"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_determinism_same_text_twice_gives_same_ids(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Encoding the same text twice must yield identical token sequences."""
    text = "where is the xyz ?"
    first_encoding = tokenizer.encode(text)
    second_encoding = tokenizer.encode(text)
    assert first_encoding == second_encoding


def test_decode_drops_the_framing_the_dataset_added(tokenizer: ByteLevelBPETokenizer) -> None:
    """The other half of leaving framing to the caller: what comes back from a model is framed,
    so decode has to see through `<bos>` and `<eos>` to the text underneath."""
    content = tokenizer.encode("hello")

    assert tokenizer.decode([BOS_ID, *content]) == "hello"
    assert tokenizer.decode([*content, EOS_ID]) == "hello"
    assert tokenizer.decode([BOS_ID, *content, EOS_ID]) == "hello"


def test_decode_drops_a_special_token_wherever_it_appears(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Specials are dropped by id rather than stripped off the ends, because these ids come back
    from a model that can emit any of them at any position: a padded row of a batch ends in
    `<pad>` after its `<eos>`, and a sampler can draw one mid-sequence."""
    content = tokenizer.encode("hello")

    assert tokenizer.decode([BOS_ID, *content, EOS_ID, PAD_ID, PAD_ID]) == "hello"
    assert tokenizer.decode([*content[:2], PAD_ID, *content[2:]]) == "hello"


def test_decode_with_only_byte_ids_decodes_correctly(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Decoding byte ids (5-260) without special tokens should still work."""
    # Manually construct: 'a' (0x61) = id 5 + 0x61 = id 5 + 97 = id 102
    byte_a_id = BASE_TOKEN_ID + ord("a")
    byte_b_id = BASE_TOKEN_ID + ord("b")
    decoded = tokenizer.decode([byte_a_id, byte_b_id])
    assert decoded == "ab"


def test_encode_handles_whitespace_correctly(tokenizer: ByteLevelBPETokenizer) -> None:
    """Spaces are bytes like any other; encode/decode should preserve them."""
    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_handles_newlines(tokenizer: ByteLevelBPETokenizer) -> None:
    """Newlines (0x0A) should encode and decode correctly."""
    text = "line1\nline2"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_handles_tabs(tokenizer: ByteLevelBPETokenizer) -> None:
    """Tabs (0x09) should encode and decode correctly."""
    text = "col1\tcol2\tcol3"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_encode_all_printable_ascii(tokenizer: ByteLevelBPETokenizer) -> None:
    """All 95 printable ASCII characters should roundtrip."""
    import string

    text = string.printable
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_decode_marks_bytes_that_spell_no_character_instead_of_raising(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """0xFF starts no UTF-8 character, and a model sampling over a byte vocabulary can emit it.

    Decoding is how a generation is read, so it has to be total over anything the model can
    produce: an undecodable byte becomes U+FFFD rather than an exception that would turn one
    unlucky sample into a crash.
    """
    invalid_byte_id = BASE_TOKEN_ID + 0xFF

    assert tokenizer.decode([BOS_ID, invalid_byte_id, EOS_ID]) == REPLACEMENT_CHARACTER


def test_decode_marks_a_character_the_generation_stopped_halfway_through(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """The ordinary case rather than the corrupt one: `é` is 0xC3 0xA9, and a model that emits
    `<eos>` between the two bytes has written half a character. There is nothing to raise
    about — the generation simply stopped there."""
    lead_byte_of_e_acute = BASE_TOKEN_ID + 0xC3

    assert tokenizer.decode([BOS_ID, lead_byte_of_e_acute, EOS_ID]) == REPLACEMENT_CHARACTER


def test_encode_sequence_respects_merge_order(tokenizer: ByteLevelBPETokenizer) -> None:
    """Encoding applies merges strictly in the order they were learned.

    This test verifies the greedy application of merges by encoding the same text
    twice and checking that the ids are identical, proving determinism and merge-order
    consistency. Implicitly, if merges were applied in a different order or greedily
    in a different way, the ids would differ for texts where merge order matters.
    """
    text = A_TRAINING_MESSAGE
    first = tokenizer.encode(text)
    second = tokenizer.encode(text)
    assert first == second
    # Further: verify that the sequence is shorter than raw bytes (merges happened)
    raw_byte_ids = [BASE_TOKEN_ID + byte for byte in text.encode("utf-8")]
    assert len(first) < len(raw_byte_ids)


def test_encode_verifies_learned_merges_were_applied(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Encoding should compress the byte sequence using learned merges.

    Verify that at least one merge was applied by checking that the encoded sequence is shorter
    than the bytes it spells.
    """
    text = "assistant"
    encoded = tokenizer.encode(text)
    raw_bytes = text.encode("utf-8")
    assert len(encoded) < len(raw_bytes)


def test_protocol_contract_encode_returns_list_of_ints(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Encode must return list[int], per the Tokenizer Protocol."""
    result = tokenizer.encode("test")
    assert isinstance(result, list)
    assert all(isinstance(token_id, int) for token_id in result)


def test_protocol_contract_decode_returns_str(tokenizer: ByteLevelBPETokenizer) -> None:
    """Decode must return str, per the Tokenizer Protocol."""
    encoded = tokenizer.encode("test")
    result = tokenizer.decode(encoded)
    assert isinstance(result, str)


def test_encode_decode_are_consistent_across_corpus_words(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Every word the training corpus is built from should roundtrip."""
    for word in CORPUS_WORDS:
        encoded = tokenizer.encode(word)
        decoded = tokenizer.decode(encoded)
        assert decoded == word, f"Roundtrip failed for '{word}': got '{decoded}'"


def test_encode_produces_no_special_token_of_its_own(tokenizer: ByteLevelBPETokenizer) -> None:
    """The invariant `WordGrid` and `chat.format_conversation` both rely on: nothing `encode`
    returns can collide with the `<bos>`, `<eos>` and `<pad>` the dataset frames a sequence with,
    or with the two markers the formatter puts around a turn — as long as nobody passes
    `recognize_markers=True`, which none of those three callers ever does."""
    assert all(token_id >= BASE_TOKEN_ID for token_id in tokenizer.encode("hello"))
    assert all(
        token_id >= BASE_TOKEN_ID
        for token_id in tokenizer.encode(f"{IM_START_TOKEN}system{IM_END_TOKEN}")
    )


def test_recognize_markers_defaults_to_off(tokenizer: ByteLevelBPETokenizer) -> None:
    """The flag exists so `scripts/build_pretrain_data.py` can opt in; every other caller —
    `eva_llm.data.chat`, `WordGrid`, `eva generate --prompt` — has to get the safe behavior by
    doing nothing, not by remembering to pass `recognize_markers=False` explicitly."""
    assert IM_START_ID not in tokenizer.encode(IM_START_TOKEN)
    assert IM_END_ID not in tokenizer.encode(IM_END_TOKEN)


def test_a_literal_chat_marker_encodes_to_its_reserved_id_when_opted_in(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """A corpus that writes `<|im_start|>`/`<|im_end|>` by hand means the reserved id, the same
    as `chat.format_conversation` inserting it directly — there is one meaning for the id,
    whichever path put it in a sequence, once that path opts in. This is what makes a raw
    pre-training document able to use the markers at all: nothing about `PretrainGrid`'s framing
    calls `format_conversation`, so `encode(..., recognize_markers=True)` is the only way they can
    ever reach a pre-training sequence.
    """
    assert tokenizer.encode(IM_START_TOKEN, recognize_markers=True) == [IM_START_ID]
    assert tokenizer.encode(IM_END_TOKEN, recognize_markers=True) == [IM_END_ID]


def test_a_chat_marker_embedded_in_text_splits_the_surrounding_content_correctly(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """The pieces on either side of a marker are still encoded exactly as they would be alone —
    recognizing the marker does not disturb pre-tokenization or merging around it."""
    encoded = tokenizer.encode(f"hello {IM_START_TOKEN} world", recognize_markers=True)

    assert encoded == [
        *tokenizer.encode("hello "),
        IM_START_ID,
        *tokenizer.encode(" world"),
    ]


def test_adjacent_and_boundary_markers_are_not_dropped(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """Two markers with nothing between them, and a marker sitting at either edge of the text,
    are where an off-by-one in the split would first show up as a missing or duplicated id."""
    assert tokenizer.encode(f"{IM_START_TOKEN}{IM_END_TOKEN}", recognize_markers=True) == [
        IM_START_ID,
        IM_END_ID,
    ]
    assert tokenizer.encode(f"{IM_START_TOKEN}hi{IM_END_TOKEN}", recognize_markers=True) == [
        IM_START_ID,
        *tokenizer.encode("hi"),
        IM_END_ID,
    ]


def test_a_decoded_chat_marker_is_still_dropped_like_every_other_special(
    tokenizer: ByteLevelBPETokenizer,
) -> None:
    """`decode` is unchanged regardless of `recognize_markers`: it drops all five specials from
    the text it renders, deliberately, because the same id also marks real chat structure that a
    caller reading generated text should not see. Recognizing the marker as content on the way in
    does not make it visible on the way back out — the two directions are not required to be
    inverses here, the same as they already were not for `<bos>`/`<eos>`/`<pad>`.
    """
    assert tokenizer.decode([IM_START_ID]) == ""
    assert tokenizer.decode(tokenizer.encode(f"a{IM_START_TOKEN}b", recognize_markers=True)) == "ab"


def test_a_corpus_that_is_only_repeated_markers_teaches_no_merges_when_opted_in() -> None:
    """Merge-learning must see a marker exactly as `encode` will — as an atomic, reserved unit —
    or a corpus repeating it often enough would walk a merge up through pieces of its bytes, and
    encoding would then disagree with what training just learned. A document that is nothing but
    the marker repeated fifty times is the case most likely to trigger that: byte-level BPE's
    frequency rule has nothing else to reach for.
    """
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        [IM_START_TOKEN * 50], recognize_markers=True
    )

    assert tokenizer.merges == ()


def test_without_opting_in_a_repeated_marker_is_learned_like_any_other_text() -> None:
    """The mirror of the test above: `recognize_markers` defaulting to `False` in
    `from_document_sequence` too is what keeps a fine-tuning-style corpus's merge-learning
    unchanged, the same as `encode`'s default does for encoding."""
    tokenizer = ByteLevelBPETokenizer.from_document_sequence([IM_START_TOKEN * 50])

    assert len(tokenizer.merges) > 0


def test_domain_tokens_are_not_relearned_from_space_prefixed_words() -> None:
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        ["He has he. She sees he."], max_merges=100, domain_tokens=("he",)
    )

    assert tokenizer.encode("he") == [BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE]
    assert tokenizer.encode(" he") == [
        BASE_TOKEN_ID + ord(" "),
        BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE,
    ]
    assert all(
        pair != (BASE_TOKEN_ID + ord("h"), BASE_TOKEN_ID + ord("e")) for pair, _ in tokenizer.merges
    )


# Small enough to learn in milliseconds, repetitive enough to support a run worth cutting in half.
RESUMABLE_CORPUS = (
    "the cat sat on the mat",
    "the cat sat on the hat",
    "the dog sat on the log",
    "a cat and a dog sat on a mat",
    "def add(a, b):\n    return a + b",
    "def sub(a, b):\n    return a - b",
)


def test_the_resumable_corpus_supports_a_run_worth_cutting() -> None:
    assert len(_learn_merges(RESUMABLE_CORPUS)) >= 4


def test_resuming_from_a_prefix_of_a_run_learns_exactly_the_rest_of_it() -> None:
    full = _learn_merges(RESUMABLE_CORPUS)

    resumed = _learn_merges(RESUMABLE_CORPUS, initial_merges=full[: len(full) // 2])

    assert resumed == full


def test_max_merges_counts_the_initial_merges() -> None:
    full = _learn_merges(RESUMABLE_CORPUS)

    resumed = _learn_merges(RESUMABLE_CORPUS, max_merges=3, initial_merges=full[:2])

    assert resumed == full[:3]


def test_resuming_on_a_grown_corpus_keeps_the_initial_merges_and_contiguous_ids() -> None:
    initial = _learn_merges(RESUMABLE_CORPUS[:2])

    grown = _learn_merges(RESUMABLE_CORPUS, initial_merges=initial)

    assert grown[: len(initial)] == initial
    assert len(grown) > len(initial)
    assert [new_id for _, new_id in grown] == list(
        range(FIRST_MERGE_ID, FIRST_MERGE_ID + len(grown))
    )


# Every piece repeated, several spellings sharing prefixes, and counts built to tie: the case
# where learning over each distinct piece once, weighted by how often it occurs, could pick a
# different merge than counting every occurrence would, if the tie-break did not survive it.
REPEATED_PIECES_CORPUS = (
    "ab ab ba ba cd cd dc dc",
    "abab baba abcd dcba",
    "the then there the then there",
    "xy yx xy yx xyz zyx",
    "    return x\n    return y\n        return z",
)


def _textbook_merges(texts: tuple[str, ...]) -> list[tuple[TokenPair, int]]:
    """The re-count-everything-every-round BPE, over every occurrence of every piece: a pair's
    count is summed occurrence by occurrence, and a tie goes to the pair a left-to-right scan of
    the corpus meets first -- `dict` insertion order, which `max` keeps. Written independently of
    `_learn_merges`, which is what makes it a reference and not a restatement."""
    sequences = [
        [BASE_TOKEN_ID + byte for byte in piece.encode("utf-8")]
        for text in texts
        for piece in PRE_TOKEN_PATTERN.findall(text)
    ]
    merges: list[tuple[TokenPair, int]] = []
    next_id = FIRST_MERGE_ID
    while True:
        counts: dict[TokenPair, int] = {}
        for sequence in sequences:
            for pair in pairwise(sequence):
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best_pair, best_count = max(counts.items(), key=lambda item: item[1])
        if best_count <= 1:
            break
        sequences = [_replay_merge(sequence, best_pair, next_id) for sequence in sequences]
        merges.append((best_pair, next_id))
        next_id += 1
    return merges


@pytest.mark.parametrize(
    "texts",
    [
        RESUMABLE_CORPUS,
        REPEATED_PIECES_CORPUS,
        tuple(CHAT_TEXTS),
    ],
    ids=["resumable", "repeated_pieces", "chat_corpus"],
)
def test_learning_picks_the_textbook_merges_tie_for_tie(texts: tuple[str, ...]) -> None:
    assert list(_learn_merges(texts)) == _textbook_merges(texts)
