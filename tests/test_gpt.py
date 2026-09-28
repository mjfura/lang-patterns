"""`model.gpt`: the fixed decoder-only transformer every corpus is trained on."""

import torch

from corpus_generalization.model.gpt import GPT, GPTConfig

CONFIG = GPTConfig(vocab_size=50, context=16, n_layer=2, n_head=2, d_model=32, dropout=0.0)


def test_logits_have_one_distribution_per_position() -> None:
    model = GPT(CONFIG)
    tokens = torch.randint(0, CONFIG.vocab_size, (3, 10))

    assert model(tokens).shape == (3, 10, CONFIG.vocab_size)


def test_a_later_token_never_changes_an_earlier_positions_logits() -> None:
    torch.manual_seed(0)
    model = GPT(CONFIG).eval()
    tokens = torch.randint(0, CONFIG.vocab_size, (1, 12))
    changed = tokens.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % CONFIG.vocab_size

    with torch.no_grad():
        before, after = model(tokens), model(changed)

    assert torch.allclose(before[0, :8], after[0, :8], atol=1e-6)
    assert not torch.allclose(before[0, 8:], after[0, 8:])


def test_the_output_head_is_tied_to_the_token_embedding() -> None:
    model = GPT(CONFIG)

    assert model.head.weight is model.token_embedding.weight


def test_a_sequence_longer_than_the_context_is_refused() -> None:
    model = GPT(CONFIG)
    tokens = torch.zeros((1, CONFIG.context + 1), dtype=torch.long)

    try:
        model(tokens)
    except ValueError as error:
        assert "context" in str(error)
    else:
        raise AssertionError("expected a ValueError")
