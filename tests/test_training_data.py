"""`model.data`: a document-level train/held-out split and the token streams batches come from."""

import torch

from corpus_generalization.model.data import random_batch, split_documents, token_stream
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

DOCUMENTS = tuple(f"document number {index} says something" for index in range(40))


def test_the_split_is_disjoint_complete_and_seeded() -> None:
    train, heldout = split_documents(DOCUMENTS, heldout_fraction=0.25, seed=3)

    assert len(heldout) == 10
    assert sorted(train + heldout) == sorted(DOCUMENTS)
    assert not set(train) & set(heldout)
    assert split_documents(DOCUMENTS, 0.25, seed=3) == (train, heldout)
    assert split_documents(DOCUMENTS, 0.25, seed=4) != (train, heldout)


def test_at_least_one_document_is_held_out() -> None:
    _, heldout = split_documents(DOCUMENTS[:3], heldout_fraction=0.1, seed=0)

    assert len(heldout) == 1


def test_the_stream_frames_every_document_with_bos_and_eos() -> None:
    tokenizer = ByteLevelBPETokenizer.from_merges(())

    stream = token_stream(("ab", "c"), tokenizer)

    assert stream.tolist() == [
        tokenizer.bos_id,
        *tokenizer.encode("ab"),
        tokenizer.eos_id,
        tokenizer.bos_id,
        *tokenizer.encode("c"),
        tokenizer.eos_id,
    ]


def test_batch_targets_are_the_inputs_shifted_by_one() -> None:
    stream = torch.arange(100)

    inputs, targets = random_batch(stream, batch_size=4, context=8, generator=torch.Generator())

    assert inputs.shape == targets.shape == (4, 8)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert torch.equal(targets[:, -1], inputs[:, -1] + 1)
