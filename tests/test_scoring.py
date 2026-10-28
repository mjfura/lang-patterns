"""`model.scoring`: log probabilities of continuations under a trained model."""

import math

import pytest
import torch

from corpus_generalization.model.gpt import GPT, GPTConfig
from corpus_generalization.model.scoring import continuation_log_probability
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

TEXTS = ["cut the tomato . fry the onion .", "peel the tomato . cut the onion ."] * 5


def _model_and_tokenizer() -> tuple[GPT, ByteLevelBPETokenizer]:
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        TEXTS, max_merges=200, recognize_markers=True
    )
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(vocab_size=tokenizer.vocab_size, context=32, n_layer=1, n_head=2, d_model=16)
    )
    model.eval()
    return model, tokenizer


def _manual(model: GPT, tokens: list[int], start: int) -> float:
    with torch.no_grad():
        logits = model(torch.tensor([tokens]))[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    return sum(float(log_probs[i - 1, tokens[i]]) for i in range(start, len(tokens)))


def test_it_is_the_sum_of_the_continuation_tokens_log_probabilities() -> None:
    model, tokenizer = _model_and_tokenizer()

    score = continuation_log_probability(model, tokenizer, "cut the", " tomato .")

    prefix = [tokenizer.bos_id, *tokenizer.encode("cut the", recognize_markers=True)]
    full = [tokenizer.bos_id, *tokenizer.encode("cut the tomato .", recognize_markers=True)]
    assert score == pytest.approx(_manual(model, full, len(prefix)))


def test_an_empty_prefix_scores_from_the_document_start() -> None:
    model, tokenizer = _model_and_tokenizer()

    score = continuation_log_probability(model, tokenizer, "", "cut the tomato .")

    full = [tokenizer.bos_id, *tokenizer.encode("cut the tomato .", recognize_markers=True)]
    assert score == pytest.approx(_manual(model, full, 1))


def test_single_token_continuations_form_a_distribution() -> None:
    model, tokenizer = _model_and_tokenizer()
    words = [" tomato", " onion", " the", " .", " cut", " fry"]
    assert all(len(tokenizer.encode(word, recognize_markers=True)) == 1 for word in words)

    total = sum(
        math.exp(continuation_log_probability(model, tokenizer, "cut the", word)) for word in words
    )

    assert 0 < total <= 1.0 + 1e-6


def test_a_continuation_that_changes_the_prefix_tokens_is_rejected() -> None:
    model, tokenizer = _model_and_tokenizer()

    with pytest.raises(ValueError, match="prefix"):
        continuation_log_probability(model, tokenizer, "cut the tom", "ato")
