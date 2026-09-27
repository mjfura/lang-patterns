"""A byte-level BPE whose `vocab_size` is decided before a single merge is learned.

`ByteLevelBPETokenizer.vocab_size` is specials + bytes + however many merges its corpus supported,
so every merge a grown corpus adds is one more embedding row, and a checkpoint trained against the
old count stops loading. Here the merge space is a fixed `capacity` — the merge budget — and
`vocab_size` counts all of it from the start, filled or not. Training fills a prefix of that space;
`extended` fills more of it from a grown corpus without renumbering anything already there. The
model's dimensions never move, and an id keeps meaning the bytes it meant when it was assigned.

Segmentation is not reimplemented: encoding delegates to a `ByteLevelBPETokenizer` built from the
same merges, and learning runs the same `_learn_merges`, so the two cannot disagree about how a
text is cut.
"""

from collections.abc import Sequence

from corpus_generalization.tokenizer.bpe import (
    BASE_TOKEN_ID,
    BYTE_VOCABULARY_SIZE,
    ByteLevelBPETokenizer,
    Merge,
    MergeProgress,
    _learn_merges,
)

FIRST_MERGE_ID = BASE_TOKEN_ID + BYTE_VOCABULARY_SIZE


class InvalidCapacityError(ValueError):
    def __init__(self, capacity: int) -> None:
        super().__init__(f"capacity must be at least 1 merge slot, got {capacity}")
        self.capacity = capacity


class MergesExceedCapacityError(ValueError):
    def __init__(self, merges: int, capacity: int) -> None:
        super().__init__(f"{merges} merges do not fit in a capacity of {capacity}")
        self.merges = merges
        self.capacity = capacity


class ReservedCapacityFullError(ValueError):
    def __init__(self, capacity: int) -> None:
        super().__init__(
            f"All {capacity} merge slots are already filled. Learning another token would change "
            f"vocab_size, and with it the embedding shape of every checkpoint trained against "
            f"this tokenizer — a larger capacity means training a new tokenizer and a new model."
        )
        self.capacity = capacity


class ReservedBPETokenizer:
    """Byte-level BPE over a merge space of fixed `capacity`, filled in order and never renumbered.

    Ids `0-4` are the specials, `5-260` the bytes, and `261` onward the merges: the first `filled`
    of them learned, the remaining `free` reserved. `encode` only ever produces learned ids.
    """

    def __init__(self, merges: Sequence[Merge], capacity: int, *, recognize_markers: bool) -> None:
        if capacity < 1:
            raise InvalidCapacityError(capacity)
        if len(merges) > capacity:
            raise MergesExceedCapacityError(len(merges), capacity)
        self._segmenter = ByteLevelBPETokenizer.from_merges(merges)
        self._capacity = capacity
        self._recognize_markers = recognize_markers

    @staticmethod
    def trained(
        documents: Sequence[str],
        capacity: int,
        progress: MergeProgress | None = None,
        *,
        recognize_markers: bool = False,
    ) -> "ReservedBPETokenizer":
        """A tokenizer learned from `documents`, stopping at `capacity` merges or earlier, when
        no pair recurs — which on a small corpus is what ends it, long before the budget does."""
        if capacity < 1:
            raise InvalidCapacityError(capacity)
        merges = _learn_merges(
            documents, max_merges=capacity, progress=progress, recognize_markers=recognize_markers
        )
        return ReservedBPETokenizer(merges, capacity, recognize_markers=recognize_markers)

    def extended(
        self, documents: Sequence[str], progress: MergeProgress | None = None
    ) -> "ReservedBPETokenizer":
        """A new tokenizer with this one's merges frozen and free slots filled from `documents`.

        `documents` should be the whole grown corpus, not only what was added: a merge is worth a
        slot when its pair recurs across the corpus the model trains on, and counting only the
        new text would spend slots on pairs that happen to repeat inside it.
        """
        if self.free == 0:
            raise ReservedCapacityFullError(self._capacity)
        merges = _learn_merges(
            documents,
            max_merges=self._capacity,
            progress=progress,
            recognize_markers=self._recognize_markers,
            initial_merges=self.merges,
        )
        return ReservedBPETokenizer(
            merges, self._capacity, recognize_markers=self._recognize_markers
        )

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + self._capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def filled(self) -> int:
        return len(self._segmenter.merges)

    @property
    def free(self) -> int:
        return self._capacity - self.filled

    @property
    def merges(self) -> tuple[Merge, ...]:
        return self._segmenter.merges

    @property
    def recognize_markers(self) -> bool:
        """The marker setting merges were learned under. `extended` reuses it, and whoever encodes
        the training corpus has to pass the same value to `encode` — see
        `ByteLevelBPETokenizer.from_document_sequence` for what a mismatch does."""
        return self._recognize_markers

    @property
    def domain_tokens(self) -> tuple[str, ...]:
        return ()

    @property
    def bos_id(self) -> int:
        return self._segmenter.bos_id

    @property
    def eos_id(self) -> int:
        return self._segmenter.eos_id

    @property
    def pad_id(self) -> int:
        return self._segmenter.pad_id

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        """`ByteLevelBPETokenizer.encode`, flag and all. It stays per call and off by default for
        the reason that docstring gives: text a user chose must not be able to forge a turn."""
        return self._segmenter.encode(text, recognize_markers=recognize_markers)

    def decode(self, token_ids: list[int]) -> str:
        """`ByteLevelBPETokenizer.decode`, with free-slot ids dropped like specials.

        A model trained after an extension has embedding rows for every slot, and one sampled
        before those rows learned anything can land on an id no merge fills yet. Reading a
        generation must not raise over that. An id at or past `vocab_size` is a different thing —
        no model built for this tokenizer can emit it — and still raises.
        """
        for token_id in token_ids:
            if token_id >= self.vocab_size:
                raise ValueError(
                    f"Token id {token_id} is past this tokenizer's vocab_size {self.vocab_size}."
                )
        first_free = FIRST_MERGE_ID + self.filled
        return self._segmenter.decode([token for token in token_ids if token < first_free])
