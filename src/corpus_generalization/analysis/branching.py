"""How much shared, repeated sequential structure a corpus offers before each document becomes
unique to itself.

Every document's own token sequence is walked position by position against a trie built from the
whole corpus. At each position, `support` is how many documents — this one included — share that
exact prefix, and `entropy_bits` is the Shannon entropy of what token they diverge into next: zero
when the prefix has only one document behind it (there is nothing left to disagree with, not
evidence of a firm rule), non-zero when several documents still share the prefix and split.

`tau` (branching depth) is the last position with at least two documents in support — position 0
always qualifies for any corpus of two or more documents, since the empty prefix trivially matches
everyone, so `tau == 0` is the floor: a document whose very first token already has no company.
`ipbe` (integrated prefix branching entropy) sums the entropy over every position up to `tau`; since
support is non-increasing in position (a longer prefix can only keep or lose company, never gain
it), summing to `tau` already excludes every position where the sum would be a trivial zero.

`cdci` (corpus domain cohesion index) is the mean `ipbe` over the corpus: high when documents
typically share several real, multi-document-supported branching points before becoming unique
(reusable sub-structure a small model can learn to compose), low when most documents peel off
almost immediately (a bag of one-off examples with nothing to generalize from). It is not
comparable across corpora of different size or document length — more documents mechanically gives
more chances for any prefix to have company — so it is only meaningful as a signal about one
corpus, or about the same corpus before and after a change.

Tokenizer fertility (`tokens / words`) is reported alongside as a second, independent signal: a
document whose vocabulary the tokenizer never learned merges for (because it barely repeats)
tokenizes close to one token per character instead of close to one per word.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


class Encodable(Protocol):
    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class PrefixStep:
    position: int
    support: int
    entropy_bits: float


@dataclass(frozen=True)
class DocumentBranching:
    text: str
    tau: int
    ipbe: float
    fertility: float
    steps: tuple[PrefixStep, ...]


@dataclass(frozen=True)
class CorpusCohesion:
    documents: int
    cdci: float
    mean_fertility: float
    fertility_stdev: float
    per_document: tuple[DocumentBranching, ...]


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    visits: int = 0


def _insert(root: _TrieNode, sequence: Sequence[int]) -> None:
    node = root
    for token in sequence:
        child = node.children.get(token)
        if child is None:
            child = _TrieNode()
            node.children[token] = child
        child.visits += 1
        node = child


def _entropy(counts: Iterable[int], total: int) -> float:
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts if count)


def branching_entropy(documents: Sequence[str], tokenizer: Encodable) -> CorpusCohesion:
    sequences = [tuple(tokenizer.encode(doc, recognize_markers=True)) for doc in documents]
    root = _TrieNode()
    for sequence in sequences:
        _insert(root, sequence)

    per_document = []
    for document, sequence in zip(documents, sequences, strict=True):
        node = root
        steps = []
        tau = 0
        ipbe = 0.0
        for position, token in enumerate(sequence):
            child_visits = [child.visits for child in node.children.values()]
            support = sum(child_visits)
            entropy = _entropy(child_visits, support)
            steps.append(PrefixStep(position, support, entropy))
            if support >= 2:
                tau = position
                ipbe += entropy
            node = node.children[token]
        words = len(document.split())
        fertility = len(sequence) / words if words else 0.0
        per_document.append(DocumentBranching(document, tau, ipbe, fertility, tuple(steps)))

    fertilities = [doc.fertility for doc in per_document]
    mean_fertility = sum(fertilities) / len(fertilities) if fertilities else 0.0
    variance = (
        sum((value - mean_fertility) ** 2 for value in fertilities) / len(fertilities)
        if fertilities
        else 0.0
    )
    ipbe_values = [doc.ipbe for doc in per_document]
    return CorpusCohesion(
        documents=len(documents),
        cdci=sum(ipbe_values) / len(ipbe_values) if ipbe_values else 0.0,
        mean_fertility=mean_fertility,
        fertility_stdev=math.sqrt(variance),
        per_document=tuple(per_document),
    )
