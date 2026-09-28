"""Held-out information gain: how many bits per byte a token's preceding context saves on text the
estimate never saw — the core learnability indicator. See
`docs/2026-09-28-heldout-information-design.md`.

The graph curves of `analysis.markov_chain` measure the topology of the transitions: whether they
stay connected, dense or cyclic as more repetition is demanded. A language model learns something
else, the conditional probabilities, and it generalizes when what it estimated on some text
predicts other text. `experiments/contamination` showed the difference: shuffled text raised the
red and blue curves (random word order adds connections) while carrying no order a model could
learn. This module measures predictability directly: smoothed n-gram models are fitted on one half
of the documents and scored on the other.

Cross-entropy is per byte of text, not per token, so it does not depend on how many bytes a token
spans; `experiments/granularity` found that the same merge budget gives very different tokens
across corpora.
"""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from corpus_generalization.packing import framed_documents
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

Context = tuple[int, ...]


@dataclass(frozen=True)
class NGramModel:
    """Counts for every context length `0..order`: for each context, how often each token
    followed it, plus the two totals Witten-Bell reads (occurrences, distinct continuations)."""

    order: int
    vocab_size: int
    continuations: tuple[dict[Context, Counter[int]], ...]
    totals: tuple[dict[Context, int], ...]
    types: tuple[dict[Context, int], ...]


def fit_ngram(sequences: Sequence[Sequence[int]], order: int, vocab_size: int) -> NGramModel:
    """Counts from framed sequences: every token after the first is predicted, from each context
    of up to `order` preceding tokens that the sequence has."""
    continuations: list[dict[Context, Counter[int]]] = [
        defaultdict(Counter) for _ in range(order + 1)
    ]
    for sequence in sequences:
        for position in range(1, len(sequence)):
            token = sequence[position]
            for length in range(min(order, position) + 1):
                continuations[length][tuple(sequence[position - length : position])][token] += 1
    return NGramModel(
        order=order,
        vocab_size=vocab_size,
        continuations=tuple(dict(level) for level in continuations),
        totals=tuple(
            {context: sum(counts.values()) for context, counts in level.items()}
            for level in continuations
        ),
        types=tuple(
            {context: len(counts) for context, counts in level.items()} for level in continuations
        ),
    )


def probability(model: NGramModel, context: Context, token: int) -> float:
    """Witten-Bell interpolation from a uniform distribution over the vocabulary up to the longest
    available context: `p = (c(h, w) + T(h) * p_lower) / (c(h) + T(h))`, with `T(h)` the number of
    distinct tokens seen after `h`; an unseen context leaves `p_lower` as it is."""
    value = 1.0 / model.vocab_size
    for length in range(min(model.order, len(context)) + 1):
        history = context[len(context) - length :] if length else ()
        total = model.totals[length].get(history)
        if not total:
            continue
        seen = model.continuations[length][history].get(token, 0)
        distinct = model.types[length][history]
        value = (seen + distinct * value) / (total + distinct)
    return value


def cross_entropy_bits(model: NGramModel, sequences: Sequence[Sequence[int]]) -> float:
    """Total `-log2 p` over every predicted token of `sequences` (all but each first)."""
    bits = 0.0
    for sequence in sequences:
        for position in range(1, len(sequence)):
            context = tuple(sequence[max(0, position - model.order) : position])
            bits -= math.log2(probability(model, context, sequence[position]))
    return bits


@dataclass(frozen=True)
class HeldoutInformation:
    orders: tuple[int, ...]
    heldout_bpb: dict[int, float]
    fit_bpb: dict[int, float]
    gain: dict[int, float]
    gap: dict[int, float]
    documents: dict[str, int]
    tokens: dict[str, int]
    bytes: dict[str, int]
    vocab_size: int


def split_halves(documents: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    """`(fit, heldout)`: the distinct texts shuffled with `seed` and halved, every copy of a text
    kept with it. Copies on both sides would let the held-out half be predicted from text the fit
    half memorized: `experiments/information` measured FineWeb with 50% exact duplicates at 1.72
    bits per byte against its clean control's 2.23 when copies were split independently."""
    copies = Counter(documents)
    distinct = list(copies)
    if len(distinct) < 2:
        raise ValueError(f"{len(distinct)} distinct text(s): two halves need at least two")
    random.Random(seed).shuffle(distinct)
    middle = len(distinct) // 2
    return (
        [text for text in distinct[:middle] for _ in range(copies[text])],
        [text for text in distinct[middle:] for _ in range(copies[text])],
    )


def heldout_information(
    documents: Sequence[str],
    merge_budget: int,
    orders: Sequence[int] = (0, 1, 2, 3),
    seed: int = 0,
) -> HeldoutInformation:
    """Fit on one seeded half of `documents` (the tokenizer too, with `merge_budget` merges),
    score both halves, and report every order's bits per byte, gain over order 0 on the held-out
    half, and held-out minus fit gap."""
    fit_documents, heldout_documents = split_halves(documents, seed)
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        fit_documents, max_merges=merge_budget, recognize_markers=True
    )
    halves = {
        "fit": framed_documents(fit_documents, tokenizer, recognize_markers=True),
        "heldout": framed_documents(heldout_documents, tokenizer, recognize_markers=True),
    }
    size = {
        "fit": sum(len(document.encode()) for document in fit_documents),
        "heldout": sum(len(document.encode()) for document in heldout_documents),
    }
    model = fit_ngram(halves["fit"], max(orders), tokenizer.vocab_size)
    heldout_bpb, fit_bpb = {}, {}
    for order in orders:
        at_order = NGramModel(
            order, model.vocab_size, model.continuations, model.totals, model.types
        )
        heldout_bpb[order] = cross_entropy_bits(at_order, halves["heldout"]) / size["heldout"]
        fit_bpb[order] = cross_entropy_bits(at_order, halves["fit"]) / size["fit"]
    base = heldout_bpb[min(orders)]
    return HeldoutInformation(
        orders=tuple(orders),
        heldout_bpb=heldout_bpb,
        fit_bpb=fit_bpb,
        gain={order: base - heldout_bpb[order] for order in orders},
        gap={order: heldout_bpb[order] - fit_bpb[order] for order in orders},
        documents={"fit": len(fit_documents), "heldout": len(heldout_documents)},
        tokens={name: sum(len(sequence) for sequence in half) for name, half in halves.items()},
        bytes=size,
        vocab_size=tokenizer.vocab_size,
    )
