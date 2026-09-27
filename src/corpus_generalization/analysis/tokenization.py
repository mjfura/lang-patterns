"""How a corpus tokenizes under a tokenizer learned from it, measured before anything trains on it.

The tokenizer is learned exactly as the pipeline learns one — the same classes, markers recognized —
and kept in memory: an analysis never writes a model's tokenizer. What is measured is each
document's content tokens, without the `<bos>`/`<eos>` framing training adds, since the question is
what the text itself costs in tokens.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from corpus_generalization.corpora import TokenizerKind
from corpus_generalization.tokenizer.base import SPECIAL_TOKENS
from corpus_generalization.tokenizer.bpe import (
    BASE_TOKEN_ID,
    BYTE_VOCABULARY_SIZE,
    ByteLevelBPETokenizer,
    Merge,
    MergeProgress,
)
from corpus_generalization.tokenizer.reserved_bpe import ReservedBPETokenizer

FIRST_MERGE_ID = BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE


class AnalyzedTokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def merges(self) -> tuple[Merge, ...]: ...

    # Unused by this module's own functions (see the module docstring: token counting here is
    # deliberately unframed) -- present so the one tokenizer `learned_tokenizer` returns also
    # satisfies `packing.FramingTokenizer` for callers elsewhere in the same analysis
    # pass (`analysis.markov_chain.build_transition_graph`'s `include_boundaries=True`)
    # without a second, separately-learned tokenizer instance.
    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def domain_tokens(self) -> tuple[str, ...]: ...

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class TokenRow:
    """One token: every learned merge, whether or not the corpus still uses it once later merges
    absorb it, every token a merge is built from — a byte that only ever appears inside merges
    included — and every special or byte the corpus does use."""

    token_id: int
    kind: str
    text: str
    parts: tuple[int, int] | None
    count: int


@dataclass(frozen=True)
class TokenizationStats:
    documents: int
    words: int
    bytes: int
    tokens: int
    merges: int
    merge_budget: int
    vocab_size: int
    distinct_tokens: int
    singletons: int
    entropy_bits: float
    rows: tuple[TokenRow, ...]

    @property
    def vocabulary_used(self) -> float:
        return self.distinct_tokens / self.vocab_size if self.vocab_size else 0.0

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / self.words if self.words else 0.0

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / self.tokens if self.tokens else 0.0

    @property
    def normalized_entropy(self) -> float:
        """Entropy over the most it could be for the tokens the corpus uses: 1 is every token
        equally common, 0 is one token repeated."""
        ceiling = math.log2(self.distinct_tokens) if self.distinct_tokens > 1 else 0.0
        return self.entropy_bits / ceiling if ceiling else 0.0

    @property
    def perplexity(self) -> float:
        return float(2**self.entropy_bits)

    @property
    def bits_per_byte(self) -> float:
        """What the unigram distribution costs per byte of text, which — unlike bits per token —
        stays comparable across tokenizers that cut the same text into different numbers of
        tokens."""
        return self.entropy_bits * self.tokens / self.bytes if self.bytes else 0.0


def learned_tokenizer(
    documents: Sequence[str],
    kind: TokenizerKind,
    merge_budget: int,
    progress: MergeProgress | None = None,
    domain_tokens: Sequence[str] = (),
) -> AnalyzedTokenizer:
    """The tokenizer the pipeline would learn from `documents`: a reserved one with `merge_budget`
    slots, or a byte-level one capped at `merge_budget` merges. `progress`, forwarded as-is to
    whichever tokenizer learns the merges, is how a caller reports on the one step here slow
    enough to need it -- a corpus large enough to make that worth showing is also large enough
    that every other step in this module is comparatively instant."""
    if kind == "reserved":
        return ReservedBPETokenizer.trained(
            documents, merge_budget, progress, recognize_markers=True
        )
    return ByteLevelBPETokenizer.from_document_sequence(
        documents,
        max_merges=merge_budget,
        progress=progress,
        recognize_markers=True,
        domain_tokens=domain_tokens,
    )


def tokenization_stats(
    documents: Sequence[str], tokenizer: AnalyzedTokenizer, merge_budget: int
) -> TokenizationStats:
    counts: Counter[int] = Counter()
    for document in documents:
        counts.update(tokenizer.encode(document, recognize_markers=True))
    tokens = sum(counts.values())
    entropy = -sum((count / tokens) * math.log2(count / tokens) for count in counts.values())
    parts = {new_id: pair for pair, new_id in tokenizer.merges}
    pieces = {piece for pair in parts.values() for piece in pair}
    rows = tuple(
        TokenRow(
            token_id,
            _kind(token_id, len(tokenizer.domain_tokens)),
            _text(token_id, parts, tokenizer.domain_tokens),
            parts.get(token_id),
            counts.get(token_id, 0),
        )
        for token_id in sorted(set(counts) | set(parts) | pieces)
    )
    return TokenizationStats(
        documents=len(documents),
        words=sum(len(document.split()) for document in documents),
        bytes=sum(len(document.encode("utf-8")) for document in documents),
        tokens=tokens,
        merges=len(parts),
        merge_budget=merge_budget,
        vocab_size=tokenizer.vocab_size,
        distinct_tokens=len(counts),
        singletons=sum(count == 1 for count in counts.values()),
        entropy_bits=entropy if tokens else 0.0,
        rows=rows,
    )


def _kind(token_id: int, domain_count: int = 0) -> str:
    if token_id < BASE_TOKEN_ID:
        return "special"
    return (
        "byte"
        if token_id < BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE
        else ("domain" if token_id < FIRST_MERGE_ID + domain_count else "merge")
    )


def _text(
    token_id: int, parts: Mapping[int, tuple[int, int]], domain_tokens: Sequence[str] = ()
) -> str:
    """A token as the text it spells, with bytes that are not whole characters shown as escapes
    rather than replacement marks."""
    if token_id < BASE_TOKEN_ID:
        return SPECIAL_TOKENS[token_id]
    domain_start = BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE
    if domain_start <= token_id < domain_start + len(domain_tokens):
        return domain_tokens[token_id - domain_start]
    return _bytes(token_id, parts, len(domain_tokens)).decode("utf-8", errors="backslashreplace")


def _bytes(token_id: int, parts: Mapping[int, tuple[int, int]], domain_count: int = 0) -> bytes:
    if token_id < BASE_TOKEN_ID:
        return b""
    if token_id < BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE:
        return bytes([token_id - BASE_TOKEN_ID])
    left, right = parts[token_id]
    return _bytes(left, parts, domain_count) + _bytes(right, parts, domain_count)
