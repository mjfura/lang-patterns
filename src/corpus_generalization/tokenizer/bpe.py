import re
from collections.abc import Callable, Iterator, Sequence
from itertools import pairwise

from corpus_generalization.tokenizer.base import (
    BOS_ID,
    EOS_ID,
    IM_END_ID,
    IM_END_TOKEN,
    IM_START_ID,
    IM_START_TOKEN,
    PAD_ID,
    SPECIAL_TOKENS,
)

BASE_TOKEN_ID = len(SPECIAL_TOKENS)
BYTE_VOCABULARY_SIZE = 256
REPLACEMENT_CHARACTER = "\ufffd"
# The two chat markers, recognized as literal text by `encode`/`_learn_merges` when their caller
# opts in with `recognize_markers=True` \u2014 a pre-training document can then write
# `<|im_start|>`/`<|im_end|>` directly and mean the reserved id, the same as `eva_llm.data.chat`
# emitting it between turns.
#
# Opt-in rather than universal is what keeps the id reserved for its original purpose
# everywhere else. `eva_llm.data.chat`, `WordGrid` and `eva generate --prompt` never pass the
# flag, so a chat message or a generation prompt whose *content* happens to spell one of these
# strings still encodes to ordinary bytes \u2014 exactly as before this existed. Recognizing the
# markers everywhere `encode` runs would let that content collapse to the same id a real turn
# boundary uses, indistinguishably, which is the forgery
# `tests/test_grid.py::test_a_prompt_spelling_a_marker_cannot_open_a_turn_of_its_own` exists to
# catch: a prompt cannot write itself a system turn no `WordGrid` conversation gave it.
_MARKER_IDS = {IM_START_TOKEN: IM_START_ID, IM_END_TOKEN: IM_END_ID}
_MARKER_PATTERN = re.compile("(" + "|".join(re.escape(marker) for marker in _MARKER_IDS) + ")")
# Where a merge is not allowed to reach across. Text is cut into these pieces before BPE sees it,
# and pairs are only ever counted and merged inside one — so no token can spell "the cat", because
# the boundary between the two words is not there to cross.
#
# This replaces a cap on how long a merge could get. That cap was this project's own invention and
# it treated the symptom: it measured a merge after proposing it and threw away the long ones,
# which meant it also threw away legitimate long units and still let 30% of the vocabulary spell
# multi-word fragments. Splitting first is what every byte-level BPE since GPT-2 does instead, and
# it makes the problem structurally impossible rather than detectable.
#
# The branches, in order: contractions; a run of two or more spaces; a run of newlines; an
# optional leading space with letters, with digits, with punctuation, or with underscores; a run
# of whitespace; and a final catch-all.
#
# The two branches for whitespace runs come first and exist for source code, where indentation is
# syntax: without them GPT-2's rule splits four leading spaces into three plus one, gluing the
# last onto the word. They cost nothing on prose, where a document is one line and single spaces
# never match either.
#
# The last branch is what guarantees the pieces rejoin into the original text — a character
# matching no branch would be dropped, and `encode` would silently lose it.
PRE_TOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r"|[ ]{2,}"
    r"|[\r\n]+"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?[^\s\w]+"
    r"| ?_+"
    r"|\s+(?!\S)"
    r"|\s+"
    r"|."
)  # How many merges `dynamic_max_merges` will learn at most. A budget, not a discovery: nothing
# bounds a run except the frequency rule, and that walks for thousands of merges on a corpus of
# any size. Every real BPE picks its vocabulary size outright — GPT-2 at 50257, Llama 3 at 128k,
# Qwen3.5 at 248k — for the same reason: it is a decision about the model, not a property of the
# text. The number below is a starting point for corpora of this project's size.
DEFAULT_MERGE_BUDGET = 1024

# Called with how many merges have been learned and the ceiling they are working towards,
# after each one. Learning a vocabulary reads the whole corpus once per merge, so on a corpus
# of any size it is minutes of silence unless something says otherwise.
type MergeProgress = Callable[[int, int | None], None]

type TokenPair = tuple[int, int]
type Merge = tuple[TokenPair, int]


class ByteLevelBPETokenizer:
    """Byte-level BPE, learned from raw text documents (`from_document_sequence`) or restored from
    merges learned earlier (`from_merges`).

    The base vocabulary is every possible byte value, so any UTF-8 string decomposes into known
    tokens and no `<unk>` token exists.

    `max_merges` is the target vocabulary size real BPE is trained against, expressed as the
    number of merges above the base vocabulary. It defaults to no cap, so a caller that does not
    ask for one gets whatever the frequency stopping rule allows.
    """

    _domain_tokens: tuple[str, ...]
    _merges: tuple[Merge, ...]
    _ranks: dict[TokenPair, int]

    @staticmethod
    def from_document_sequence(
        documents: Sequence[str],
        max_merges: int | None = None,
        progress: MergeProgress | None = None,
        *,
        recognize_markers: bool = False,
        domain_tokens: Sequence[str] = (),
    ) -> "ByteLevelBPETokenizer":
        """Create a tokenizer trained on raw text documents.

        Each document is a separate text string, and no JSON parsing or chat formatting is
        applied. Used for pre-training on plain text corpora where each line is one document.

        `recognize_markers` must agree with whatever `encode` will later be called with on this
        same corpus — see its docstring. A mismatch is not merely inconsistent: with it `True`
        here and `False` at encoding time (or the reverse), a document that repeats a literal
        `<|im_start|>`/`<|im_end|>` could walk a merge up through the marker's own bytes during
        learning while encoding later treats it as the atomic reserved id, or the other way
        around, and the two would segment the same corpus differently.
        """
        tokenizer = ByteLevelBPETokenizer.__new__(ByteLevelBPETokenizer)
        tokenizer._domain_tokens = _validate_domain_tokens(domain_tokens)
        tokenizer._merges = _learn_merges(
            documents,
            max_merges,
            progress=progress,
            recognize_markers=recognize_markers,
            domain_tokens=tokenizer._domain_tokens,
        )
        tokenizer._ranks = _merge_ranks(tokenizer._merges)
        return tokenizer

    @staticmethod
    def from_merges(
        merges: Sequence[Merge], domain_tokens: Sequence[str] = ()
    ) -> "ByteLevelBPETokenizer":
        """A tokenizer restored from merges learned earlier, without seeing a corpus at all.

        This is what makes a vocabulary an artifact rather than a derivation. Learning merges
        reads the whole corpus once per merge, so deriving them at import cost minutes on a
        corpus of any size; worse, it made the meaning of every token id a function of whatever
        the corpus happened to be that day. Restored from a file, an id means what it meant when
        the file was written, which is the only footing a checkpoint can stand on.
        """
        tokenizer = ByteLevelBPETokenizer.__new__(ByteLevelBPETokenizer)
        tokenizer._domain_tokens = _validate_domain_tokens(domain_tokens)
        tokenizer._merges = tuple(merges)
        tokenizer._ranks = _merge_ranks(tokenizer._merges)
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return self._merge_start + len(self._merges)

    @property
    def domain_tokens(self) -> tuple[str, ...]:
        return self._domain_tokens

    @property
    def _merge_start(self) -> int:
        return BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE + len(self._domain_tokens)

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def merges(self) -> tuple[Merge, ...]:
        """Learned merges in the order they were learned: for each, the pair of ids it replaces
        and the id it produces. Applying them in this order is how encode reproduces training."""
        return self._merges

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        """Encode text to the token ids that spell it: UTF-8 bytes, then the learned merges.

        No `<bos>`, `<eos>` or `<pad>` is added. Framing a sequence is the dataset's decision,
        not the segmenter's, so `WordGrid` makes it for both tokenizers in one place.

        `recognize_markers`, given, treats a literal `<|im_start|>`/`<|im_end|>` in `text` as its
        reserved id instead of ordinary bytes — what `scripts/build_pretrain_data.py` asks for so
        a pre-training document can write these markers directly, the same as
        `eva_llm.data.chat.format_conversation` inserting one between turns.

        Left `False`, which every caller reachable from a prompt is — `eva_llm.data.chat`,
        `WordGrid`, and `eva generate --prompt` alike — encoding a marker's characters produces
        only bytes, never the reserved id. That is what keeps a user's message from forging a
        turn: `<|im_start|>system` typed into a chat prompt is data describing those characters,
        not a way to write a turn boundary the formatter did not put there. Recognizing markers
        everywhere `encode` runs would close that gap the moment anyone builds a sequence from
        text a user chose, which pre-training documents are not but a chat prompt always is —
        so this is opt-in, and only the pre-training path opts in.
        """
        pieces = _split_on_domain_tokens(text, self._domain_tokens)
        if not recognize_markers:
            return [token for piece in pieces for token in self._encode_piece(piece)]
        ids: list[int] = []
        for piece in _split_on_markers(text):
            marker_id = _MARKER_IDS.get(piece)
            if marker_id is not None:
                ids.append(marker_id)
            else:
                ids.extend(
                    token
                    for sub in _split_on_domain_tokens(piece, self._domain_tokens)
                    for token in self._encode_piece(sub)
                )
        return ids

    def _encode_piece(self, piece: str) -> list[int]:
        domain_id = self._domain_tokens.index(piece) if piece in self._domain_tokens else None
        if domain_id is not None:
            return [BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE + domain_id]
        return [token for sub in _pre_tokens(piece) for token in self._merged(_encode_bytes(sub))]

    def _merged(self, tokens: list[int]) -> list[int]:
        """One pre-token piece with this tokenizer's merges applied — see `_reduced`."""
        return _reduced(tokens, self._ranks, self._merge_start)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text: drop specials, expand merges, decode the bytes.

        Total over anything a model can emit, by design. Specials are dropped by id wherever
        they appear rather than stripped off the ends, because a padded row carries `<pad>`
        after its `<eos>` and a sampler can draw one anywhere. Bytes that spell no character are
        marked with U+FFFD for the same reason: over a byte vocabulary, a generation that stops
        halfway through a multibyte character is ordinary, and reading a generation must not be
        able to raise.
        """
        merge_by_id = {new_id: pair for pair, new_id in self._merges}
        text_parts: list[str] = []
        byte_values: list[int] = []
        for token_id in token_ids:
            if BASE_TOKEN_ID <= token_id < BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE:
                byte_values.append(token_id - BASE_TOKEN_ID)
            elif BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE <= token_id < self._merge_start:
                if byte_values:
                    text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
                    byte_values.clear()
                index = token_id - BASE_TOKEN_ID - BYTE_VOCABULARY_SIZE
                text_parts.append(self._domain_tokens[index])
            elif token_id >= self._merge_start:
                byte_values.extend(_expand_token_id(token_id, merge_by_id))
        if byte_values:
            text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
        return "".join(text_parts)


def dynamic_max_merges(texts: Sequence[str], budget: int = DEFAULT_MERGE_BUDGET) -> int:
    """How many merges this corpus supports before one would start spelling a multi-word
    fragment rather than a reusable sub-word piece — a `max_merges` derived from the corpus
    instead of a number fixed in code, which a corpus that later grows or shrinks makes wrong.

    A bigger, more varied corpus legitimately supports far more merges than a small one: the
    frequency rule already requires a pair to recur, but on a corpus with a handful of near-
    identical strings — five near-identical sentences, a system prompt repeated across every
    conversation — that recurrence can come from copies of nearly the same text rather than
    structure repeating across genuinely different ones. What tells the two apart is not how
    often a pair recurs but how long the merged result gets: a real sub-word piece stays short,
    a memorized phrase spans several words.

    So: learn merges, passing over any whose combined length would reach `MEMORIZATION_LENGTH`
    — see that constant for where the number comes from and what a document-relative version of
    it got wrong.

    `budget` is what actually ends the run on a corpus of any size. The length rule used to end
    it, and that made this function report a *smaller* vocabulary the more text it was given:
    the first long frequent phrase stopped every merge after it, so the eight-year corpus and
    its first year alone both stopped at 151 merges. Now that the rule skips rather than stops,
    the only thing left to end a run is the frequency rule, which is far away — so the ceiling
    is stated instead of discovered. What stays derived is how many of that budget the corpus
    can actually fill: a small corpus still runs out of recurring pairs early and returns less.
    """
    return len(_learn_merges(texts, max_merges=budget))


def _expand_token_id(token_id: int, merge_by_id: dict[int, tuple[int, int]]) -> list[int]:
    """Recursively expand a token id to its underlying bytes.

    A byte id expands to itself; a merge id expands to the union of its two constituent token
    ids. Both ranges start at `BASE_TOKEN_ID`, which moves whenever a special token is added.
    """
    if token_id in merge_by_id:
        # This is a learned merge: recursively expand both parts of the pair.
        id1, id2 = merge_by_id[token_id]
        return _expand_token_id(id1, merge_by_id) + _expand_token_id(id2, merge_by_id)
    elif BASE_TOKEN_ID <= token_id < BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE:
        # This is a byte id: return it as-is.
        return [token_id - BASE_TOKEN_ID]
    else:
        # Specials are already gone, so what is left is an id past this tokenizer's vocabulary:
        # it spells no bytes at all, and guessing which ones would invent text.
        raise ValueError(f"Token id {token_id} is invalid in content (not a byte or known merge).")


def _learn_merges(
    texts: Sequence[str],
    max_merges: int | None = None,
    progress: MergeProgress | None = None,
    recognize_markers: bool = False,
    initial_merges: Sequence[Merge] = (),
    domain_tokens: Sequence[str] = (),
) -> tuple[Merge, ...]:
    """Standard BPE: repeatedly merge the corpus's most frequent adjacent pair into a new token.

    Two ceilings, and learning stops at whichever is reached first.

    `max_merges` is the target vocabulary size, which is how every real BPE is trained: the size
    of a vocabulary is a decision about the model, not something read off the corpus.

    The frequency rule is the second. A pair seen exactly once cannot represent a pattern this
    corpus repeats, so merging it would memorize noise rather than learn structure.

    What used to be a third rule — a cap on how long a merge could get — is gone, replaced by
    `PRE_TOKEN_PATTERN`. Counting happens inside pre-token pieces, so a pair never spans a word
    boundary and no merge can spell a phrase. The old cap measured merges after proposing them
    and discarded the long ones, which threw away legitimate long units and still left a third of
    the vocabulary spelling multi-word fragments.

    Each piece is counted on its own, and so is each text: no pair spans two of either. That
    matches how they are encoded — `encode` splits by the same pattern and merges within each
    piece — which is what keeps training and encoding from disagreeing about segmentation.

    `recognize_markers` must agree with whatever `encode` will be called with on this corpus
    afterward — see its docstring for why, and `ByteLevelBPETokenizer.from_document_sequence` for
    the consequence of the two disagreeing. Given `True`, a literal `<|im_start|>`/`<|im_end|>` in
    a text contributes nothing here: `_split_on_markers` removes it before pre-tokenization runs,
    so a corpus that writes either marker many times over cannot walk a merge up through pieces of
    it — they stay the two reserved ids exactly, never a learned token spelling half of one.

    `initial_merges`, given, resumes a run instead of starting one: every piece is first reduced
    by them, in the same rank order `encode` applies, and learning continues from the id after the
    last of them, returning them followed by whatever it learns. `max_merges` counts them. On the
    corpus they came from, resuming from any prefix of a run learns exactly the rest of that run —
    which is what lets `ReservedBPETokenizer.extended` fill free slots without renumbering an id.

    Incremental, not the textbook re-scan-everything-every-round BPE: a merge changes only the
    pieces that contained the pair just merged, and only their own neighbouring pairs, so
    `_pair_index`/`_replace_pair` touch just that -- not every piece in the corpus -- on every
    round after the first. See `_most_frequent_pair` for why this still picks the exact merge,
    tie for tie, the discarded from-scratch re-scan would have: on this project's own training
    corpus 65 of 104 rounds have a tie for the top count, so a faster approach that broke ties
    differently would silently learn a different vocabulary, not just learn the same one sooner.

    Learned over each distinct piece once, its pairs counted as many times as the piece occurs:
    every occurrence of a piece is rewritten identically by every merge, so one copy stands for
    all of them. On the Python corpus that is 63,642 distinct pieces for 4,936,320 occurrences.
    """
    domain_tokens = _validate_domain_tokens(domain_tokens)
    merge_start = BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE + len(domain_tokens)
    ranks = _merge_ranks(initial_merges)
    words: dict[str, int] = {}
    sequences: list[list[int]] = []
    frequencies: list[int] = []
    origins: list[list[int]] = []
    next_origin = 0
    for piece in _training_pieces(texts, recognize_markers, domain_tokens):
        word_index = words.get(piece)
        if word_index is None:
            word_index = words[piece] = len(sequences)
            sequence = _encode_bytes(piece)
            if initial_merges:
                sequence = _reduced(sequence, ranks, merge_start)
            sequences.append(sequence)
            frequencies.append(0)
            origins.append(list(range(next_origin, next_origin + len(sequence))))
        frequencies[word_index] += 1
        next_origin += len(sequences[word_index])
    merges: list[Merge] = list(initial_merges)
    next_id = merge_start + len(merges)
    pair_counts, pair_locations = _pair_index(sequences, frequencies)
    while max_merges is None or len(merges) < max_merges:
        best_pair = _most_frequent_pair(pair_counts, pair_locations, origins)
        if best_pair is None or pair_counts[best_pair] <= 1:
            break
        _replace_pair(
            sequences, frequencies, origins, pair_counts, pair_locations, best_pair, next_id
        )
        merges.append((best_pair, next_id))
        next_id += 1
        if progress is not None:
            progress(len(merges), max_merges)
    return tuple(merges)


def _training_pieces(
    texts: Sequence[str], recognize_markers: bool, domain_tokens: Sequence[str]
) -> Iterator[str]:
    """Every piece `_learn_merges` counts, in corpus order, one per occurrence: what `encode`
    would merge within, less the markers and domain tokens that encode to a reserved id instead."""
    for text in texts:
        segments = _split_on_markers(text) if recognize_markers else [text]
        for segment in segments:
            if segment in _MARKER_IDS and recognize_markers:
                continue
            for piece in _pre_tokens(segment):
                for domain_piece in _split_on_domain_tokens(piece, domain_tokens):
                    if domain_piece not in domain_tokens:
                        yield domain_piece


def _split_on_markers(text: str) -> list[str]:
    """`text` cut at every literal `<|im_start|>`/`<|im_end|>`, the markers kept as pieces of the
    result in their own right rather than dropped — a caller tells a marker piece from a content
    piece by looking it up in `_MARKER_IDS`, which is exactly what `encode` and `_learn_merges`
    each do with what this returns, for opposite reasons: one to emit the reserved id, the other
    to skip the piece entirely.

    `re.split` with a capturing group is what makes this one call rather than a hand-rolled scan:
    it returns the text between matches and the matches themselves, interleaved in order, with
    empty strings exactly where two markers are adjacent or one sits at either end of `text`.
    """
    return [piece for piece in _MARKER_PATTERN.split(text) if piece]


def _split_on_domain_tokens(text: str, domain_tokens: Sequence[str]) -> list[str]:
    if not domain_tokens:
        return [text]
    alternatives = "|".join(
        re.escape(token) for token in sorted(domain_tokens, key=len, reverse=True)
    )
    pattern = re.compile(r"(?<![\w'])((?:" + alternatives + r"))(?![\w'])")
    return [piece for piece in pattern.split(text) if piece]


def _validate_domain_tokens(domain_tokens: Sequence[str]) -> tuple[str, ...]:
    tokens = tuple(domain_tokens)
    if len(set(tokens)) != len(tokens) or any(not token for token in tokens):
        raise ValueError("domain_tokens must contain unique, non-empty strings")
    return tokens


def _pre_tokens(text: str) -> list[str]:
    """`text` cut into the pieces a merge may not reach across.

    The pieces rejoin into exactly the text they came from — `PRE_TOKEN_PATTERN`'s last branch
    exists to guarantee it — so splitting here loses nothing that encoding has to put back.
    """
    return PRE_TOKEN_PATTERN.findall(text)


def _merge_ranks(merges: Sequence[Merge]) -> dict[TokenPair, int]:
    """Each merge's position in the order it was learned, which is the order `encode` applies
    them in. Kept beside the merges because encoding consults it for every pair of every piece,
    and rebuilding it per call would cost more than the merging it directs."""
    return {pair: rank for rank, (pair, _) in enumerate(merges)}


def _reduced(tokens: list[int], ranks: dict[TokenPair, int], merge_start: int) -> list[int]:
    """One pre-token piece with the merges `ranks` describes applied, earliest-learned first.

    Repeatedly takes the lowest-ranked pair present rather than sweeping every merge over the
    sequence in order. The two agree — applying merges in rank order is what both do — but a
    sweep costs one pass per merge whatever the text, which on a thousand-merge vocabulary
    meant a thousand passes over a five-byte word. This costs one pass per merge actually
    made, and a piece that short admits at most a handful.

    A module function rather than a method because `_learn_merges` needs it too, to resume a run
    from merges it did not learn itself.
    """
    while len(tokens) > 1:
        ranked = [
            (rank, pair) for pair in pairwise(tokens) if (rank := ranks.get(pair)) is not None
        ]
        if not ranked:
            break
        rank, pair = min(ranked)
        tokens = _apply_merge(tokens, pair, merge_start + rank)
    return tokens


def _encode_bytes(text: str) -> list[int]:
    return [BASE_TOKEN_ID + byte for byte in text.encode("utf-8")]


type PairLocations = dict[TokenPair, set[tuple[int, int]]]


def _pair_index(
    sequences: Sequence[Sequence[int]], frequencies: Sequence[int]
) -> tuple[dict[TokenPair, int], PairLocations]:
    """Every adjacent pair's count -- each distinct piece's pairs weighted by how often it
    occurs -- and exact `(sequence_index, position)` locations, in one pass: the only full scan
    of the distinct pieces this module's training loop ever does. Every round after this one
    updates both structures incrementally instead of rebuilding them (see `_replace_pair`)."""
    counts: dict[TokenPair, int] = {}
    locations: PairLocations = {}
    for sequence_index, (sequence, frequency) in enumerate(
        zip(sequences, frequencies, strict=True)
    ):
        for position, pair in enumerate(pairwise(sequence)):
            counts[pair] = counts.get(pair, 0) + frequency
            locations.setdefault(pair, set()).add((sequence_index, position))
    return counts, locations


def _most_frequent_pair(
    pair_counts: dict[TokenPair, int],
    pair_locations: PairLocations,
    origins: Sequence[Sequence[int]],
) -> TokenPair | None:
    """The pair with the highest count, breaking ties by which one a fresh left-to-right scan of
    the corpus would reach first -- what the discarded from-scratch re-scan's `pair_counts`
    insertion order used to give for free, now read directly off `origins` instead. Two different
    pair types can never tie on origin: an origin belongs to exactly one symbol, which is the
    left half of at most one current pair, so this total order has no further ties to break --
    unlike the count, which routinely does (65 of 104 rounds, training this project's own
    tokenizer). The min is only ever taken over pairs already tied on count, never over every
    known pair, so this stays cheap even though it is a scan.

    A symbol's origin is its rank in a left-to-right flattening of the whole corpus, every
    occurrence of every piece included; a merged symbol keeps its left half's. Each distinct
    piece carries the origins of its first occurrence only, which is enough: every occurrence is
    rewritten identically, so a pair's earliest occurrence anywhere is at the same position of
    some piece's first occurrence.
    """
    if not pair_counts:
        return None
    best_count = max(pair_counts.values())
    tied = [pair for pair, count in pair_counts.items() if count == best_count]
    if len(tied) == 1:
        return tied[0]

    def first_occurrence(pair: TokenPair) -> int:
        return min(
            origins[sequence_index][position] for sequence_index, position in pair_locations[pair]
        )

    return min(tied, key=first_occurrence)


def _reindex_sequence(
    sequence: Sequence[int],
    sequence_index: int,
    pair_counts: dict[TokenPair, int],
    pair_locations: PairLocations,
    delta: int,
) -> None:
    """Adds (`delta` positive) or removes (`delta` negative) every one of `sequence`'s own
    adjacent pairs, `abs(delta)` occurrences each, from `pair_counts`/`pair_locations` -- the two
    calls in `_replace_pair` that keep both in step with a sequence a merge just rewrote, without
    touching any sequence the merge left alone."""
    for position, pair in enumerate(pairwise(sequence)):
        count = pair_counts.get(pair, 0) + delta
        if count <= 0:
            del pair_counts[pair]
        else:
            pair_counts[pair] = count
        location = (sequence_index, position)
        if delta > 0:
            pair_locations.setdefault(pair, set()).add(location)
        else:
            locations = pair_locations[pair]
            locations.discard(location)
            if not locations:
                del pair_locations[pair]


def _merge_sequence(
    sequence: Sequence[int], sequence_origins: Sequence[int], pair: TokenPair, new_id: int
) -> tuple[list[int], list[int]]:
    """`_apply_merge`'s own left-to-right, no-overlap merge sweep, carrying each symbol's origin
    along beside it -- a merged symbol inherits its LEFT half's origin, since that is still the
    leftmost byte it was ever built from."""
    merged: list[int] = []
    merged_origins: list[int] = []
    index = 0
    while index < len(sequence):
        if index < len(sequence) - 1 and (sequence[index], sequence[index + 1]) == pair:
            merged.append(new_id)
            merged_origins.append(sequence_origins[index])
            index += 2
        else:
            merged.append(sequence[index])
            merged_origins.append(sequence_origins[index])
            index += 1
    return merged, merged_origins


def _replace_pair(
    sequences: list[list[int]],
    frequencies: Sequence[int],
    origins: list[list[int]],
    pair_counts: dict[TokenPair, int],
    pair_locations: PairLocations,
    pair: TokenPair,
    new_id: int,
) -> None:
    """Every occurrence of `pair` replaced by `new_id`, and `pair_counts`/`pair_locations` kept
    exactly in step -- touching only the sequences that actually contained `pair` (typically a
    handful of short pre-token pieces, not the whole corpus), which is the entire reason this
    training loop no longer re-scans every sequence on every round.

    Each affected sequence has ALL of its own pairs removed before the merge and ALL of them
    added back after, not just the ones touching `pair` directly: a pre-token piece can hold
    `pair` more than once (`_merge_sequence` merges every non-overlapping occurrence in one
    sweep), and the neighbours on either side of each occurrence gain a new pair with `new_id` and
    lose whichever pair they held with the old symbols -- both are exactly the pairs a full
    remove-then-add already accounts for, without having to reason about each one individually.
    """
    affected = sorted({sequence_index for sequence_index, _ in pair_locations[pair]})
    for sequence_index in affected:
        old_sequence = sequences[sequence_index]
        frequency = frequencies[sequence_index]
        _reindex_sequence(
            old_sequence, sequence_index, pair_counts, pair_locations, delta=-frequency
        )
        new_sequence, new_origins = _merge_sequence(
            old_sequence, origins[sequence_index], pair, new_id
        )
        sequences[sequence_index] = new_sequence
        origins[sequence_index] = new_origins
        _reindex_sequence(
            new_sequence, sequence_index, pair_counts, pair_locations, delta=frequency
        )


def _apply_merge(sequence: Sequence[int], pair: TokenPair, new_id: int) -> list[int]:
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
