"""Framing a document as `<bos>` + its tokens + `<eos>`, the way a pre-training corpus is framed
before packing — which is what makes the two boundary tokens real nodes of a transition graph."""

from collections.abc import Sequence
from typing import Protocol


class FramingTokenizer(Protocol):
    """What framing needs from a tokenizer — both `ByteLevelBPETokenizer` and
    `ReservedBPETokenizer` provide it, with the same `recognize_markers` contract."""

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]: ...


def framed_documents(
    documents: Sequence[str], tokenizer: FramingTokenizer, recognize_markers: bool
) -> list[list[int]]:
    """Every document as `<bos>` + tokens + `<eos>`. `recognize_markers` has to be the same
    setting the vocabulary was learned under — see `ByteLevelBPETokenizer.encode`."""
    bos_id, eos_id = tokenizer.bos_id, tokenizer.eos_id
    return [
        [bos_id, *tokenizer.encode(document, recognize_markers=recognize_markers), eos_id]
        for document in documents
    ]


class PreTokenizingTokenizer(FramingTokenizer, Protocol):
    """A framing tokenizer that also reports where its pre-tokens start."""

    def encode_pre_tokens(
        self, text: str, *, recognize_markers: bool = False
    ) -> list[list[int]]: ...


def framed_word_starts(
    documents: Sequence[str], tokenizer: PreTokenizingTokenizer, recognize_markers: bool
) -> tuple[list[list[int]], list[list[bool]]]:
    """`framed_documents`, with a parallel flag per token: whether it opens a pre-token, which no
    merge crosses. `<bos>` and `<eos>` open one, so a sequence's end is a syntactic transition."""
    sequences: list[list[int]] = []
    starts: list[list[bool]] = []
    for document in documents:
        ids, flags = [tokenizer.bos_id], [True]
        for word in tokenizer.encode_pre_tokens(document, recognize_markers=recognize_markers):
            ids.extend(word)
            flags.append(True)
            flags.extend([False] * (len(word) - 1))
        ids.append(tokenizer.eos_id)
        flags.append(True)
        sequences.append(ids)
        starts.append(flags)
    return sequences, starts
