"""Training data: documents split into train and held-out, framed and packed into token streams.

The split is by document, so no held-out document is ever seen in training, and the tokenizer is
learned on the train split alone (`model.train`): the held-out loss then measures how well what
was learned carries to unseen text of the same corpus, with nothing of that text leaked in.
"""

import math
import random
from collections.abc import Sequence

import torch

from corpus_generalization.packing import FramingTokenizer, framed_documents


def split_documents(
    documents: Sequence[str], heldout_fraction: float, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(train, heldout)`: a seeded shuffle of `documents`, `heldout_fraction` of them (at least
    one) held out."""
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    heldout_count = max(1, math.floor(len(documents) * heldout_fraction))
    heldout = tuple(documents[index] for index in order[:heldout_count])
    train = tuple(documents[index] for index in order[heldout_count:])
    return train, heldout


def token_stream(documents: Sequence[str], tokenizer: FramingTokenizer) -> torch.Tensor:
    """Every document as `<bos>` + tokens + `<eos>`, concatenated into one 1-D id tensor."""
    ids = [token for document in framed_documents(documents, tokenizer, True) for token in document]
    return torch.tensor(ids, dtype=torch.long)


def random_batch(
    stream: torch.Tensor, batch_size: int, context: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """`batch_size` random windows of `stream`: inputs of `context` tokens and, as targets, the
    same windows shifted one token forward."""
    if len(stream) <= context:
        raise ValueError(f"a stream of {len(stream)} tokens is too short for context {context}")
    starts = torch.randint(0, len(stream) - context, (batch_size,), generator=generator)
    windows = torch.stack([stream[start : start + context + 1] for start in starts.tolist()])
    return windows[:, :-1], windows[:, 1:]
