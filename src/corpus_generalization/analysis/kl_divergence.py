"""Token-distribution KL divergence, leave-one-out per document — the coarse, order-blind sibling
of `analysis.branching`'s sequence-aware metrics, included here as a benchmark rather than
a replacement. It is the technique Meta reports using to filter Llama 3's pre-training web crawl
("The Llama 3 Herd of Models", section 3.1.1): a document's own unigram token-frequency
distribution compared, by Kullback-Leibler divergence, against the rest of the corpus's — a signal
that catches a document whose token usage looks nothing like the rest (their example: Base64,
network logs, memory dumps), without ever looking at token order. See `analysis.branching`
for what that costs.

"The rest of the corpus", not "the whole corpus": at Meta's scale (a web-scale crawl) one document
does not move the aggregate distribution it is graded against, so grading it against the whole
corpus including itself makes no practical difference. At this project's scale — dozens of short
documents — a single document is a measurable fraction of the total, so including it in its own
reference distribution would quietly flatter it. Leaving it out (the reference each document is
compared against is the corpus minus that document) is the standard leave-one-out correction, and
it costs nothing at this scale.

Both distributions are Lidstone-smoothed — a small additive pseudocount spread over the tokenizer's
whole vocabulary, not just the tokens a document happens to use — so that a token appearing nowhere
else in the corpus (exactly the case this metric exists to flag) gives a finite, large divergence
instead of an undefined one (`log(p / 0)`). The default pseudocount is far smaller than textbook
add-one smoothing: these documents run a handful of tokens against a vocabulary in the hundreds,
and add-one would swamp whatever real signal is there.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

DEFAULT_SMOOTHING = 1e-3


class VocabEncodable(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class DocumentDivergence:
    text: str
    kl_divergence: float


@dataclass(frozen=True)
class CorpusDivergence:
    documents: int
    mean_kl_divergence: float
    kl_divergence_stdev: float
    per_document: tuple[DocumentDivergence, ...]


def token_distribution_divergence(
    documents: Sequence[str], tokenizer: VocabEncodable, smoothing: float = DEFAULT_SMOOTHING
) -> CorpusDivergence:
    per_document_counts = [
        Counter(tokenizer.encode(document, recognize_markers=True)) for document in documents
    ]
    vocab_size = tokenizer.vocab_size
    total_counts: Counter[int] = Counter()
    for counts in per_document_counts:
        total_counts.update(counts)

    per_document = [
        DocumentDivergence(
            document,
            _kl_divergence(counts, total_counts - counts, vocab_size, smoothing),
        )
        for document, counts in zip(documents, per_document_counts, strict=True)
    ]

    values = [doc.kl_divergence for doc in per_document]
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    return CorpusDivergence(
        documents=len(documents),
        mean_kl_divergence=mean,
        kl_divergence_stdev=math.sqrt(variance),
        per_document=tuple(per_document),
    )


def _kl_divergence(
    document_counts: Counter[int],
    reference_counts: Counter[int],
    vocab_size: int,
    smoothing: float,
) -> float:
    """`document_counts` over `reference_counts`, both Lidstone-smoothed over the full `vocab_size`
    — `reference_counts` is already the leave-one-out corpus (this document's counts subtracted
    out), computed once by the caller so every document reuses the same corpus-wide totals."""
    document_total = sum(document_counts.values()) + smoothing * vocab_size
    reference_total = sum(reference_counts.values()) + smoothing * vocab_size
    divergence = 0.0
    for token in range(vocab_size):
        p = (document_counts.get(token, 0) + smoothing) / document_total
        q = (reference_counts.get(token, 0) + smoothing) / reference_total
        divergence += p * math.log2(p / q)
    # KL divergence is never negative (Gibbs' inequality); a result under 0 here is floating-point
    # rounding on a true value of 0, not a real signal.
    return max(0.0, divergence)
