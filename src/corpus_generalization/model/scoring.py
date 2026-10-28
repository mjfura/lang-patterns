"""What a trained model thinks of a continuation: the log probability of its tokens after a prefix,
read from the document start (`<bos>`), which is how the syntactic tests of
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` query a model."""

import torch

from corpus_generalization.model.generate import LanguageModel
from corpus_generalization.packing import FramingTokenizer


def continuation_log_probability(
    model: LanguageModel, tokenizer: FramingTokenizer, prefix: str, continuation: str
) -> float:
    """Natural-log probability of `continuation`'s tokens given `<bos>` + `prefix`. The prefix
    must tokenize the same alone as inside the whole text (start the continuation with a space),
    or the continuation's tokens would not be the ones the model is asked about."""
    prefix_ids = [tokenizer.bos_id, *tokenizer.encode(prefix, recognize_markers=True)]
    full_ids = [tokenizer.bos_id, *tokenizer.encode(prefix + continuation, recognize_markers=True)]
    if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) == len(prefix_ids):
        raise ValueError(f"{continuation!r} changes how the prefix {prefix!r} tokenizes")
    if len(full_ids) > model.context:
        raise ValueError(f"{len(full_ids)} tokens exceed the model's context of {model.context}")
    with torch.no_grad():
        logits = model(torch.tensor([full_ids]))[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    return sum(
        float(log_probs[position - 1, full_ids[position]])
        for position in range(len(prefix_ids), len(full_ids))
    )
