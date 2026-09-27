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
