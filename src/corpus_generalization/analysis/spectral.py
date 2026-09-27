"""Two spectral readings of a corpus, each computable two ways: from the corpus's own tokenizer
alone, or from a general-purpose sentence encoder's geometry -- the same split
`analysis.domains` already draws for its own question.

Vendi Score is the effective number of independent items a set of texts contains: build an n x n
similarity (kernel) matrix over them, take the von Neumann entropy of its eigenvalues (divided by
n so they sum to 1, like a probability distribution), and exponentiate. n identical texts score 1
("one effective item" -- everything collapses to it); n pairwise-orthogonal texts score n ("n
effective items" -- nothing is redundant). It answers a question `analysis.domains`'s
k-means split cannot: not "how many clusters" but "how much real, non-redundant variety", which a
corpus of many small, tight clusters and a corpus of one broad one can report identically for.

Algebraic connectivity (the Fiedler value, `lambda_2` of the graph Laplacian) is the spectral
answer to the same fragmentation question `markov_chain.py`'s weak-connectivity check answers
combinatorially: it is exactly 0 when a graph splits into two or more disconnected pieces, and
larger the more evenly and richly connected the whole graph is otherwise. Unlike weak connectivity,
it is not just yes/no -- a corpus that is technically one component but held together by one thin
bridge reads close to 0 too.

But `lambda_2` alone only ever answers "is this graph in one piece" (exactly 0) or "not" (anything
else) -- it has no fixed scale a reader can compare across corpora the way `domains.py`'s cosine
threshold works, because it grows with both how many units there are and how strongly they are
connected on average. Telling "2 groups" from "3 groups" needs the *count* of eigenvalues sitting
at 0 before the spectrum jumps to its first real value (the graph Laplacian's zero eigenvalue has
multiplicity exactly equal to the number of connected components -- verified in this session by
building corpora with 1, 2 and 3 deliberately disconnected pieces and checking both the spectrum
and networkx's own component count agreed every time). `connected_components_count` reads off that
exact integer directly, cheaply, with no eigengap to eyeball -- `algebraic_connectivity` is then
the *continuous* follow-up question once the discrete one says "1": not just "connected" but how
richly.

Two substrates feed both:

- `lexical_similarity_matrix` / `transition_adjacency_matrix`, built only from the project's own
  tokenizer (a bag-of-token-id cosine matrix, and `markov_chain.py`'s own transition graph
  symmetrized) -- no external artifact, so a corpus can be certified self-contained and cohesive
  purely from what it teaches its own vocabulary, before any encoder ever sees it.
- `semantic_similarity_matrix`, built from a general sentence encoder's embeddings (the same
  `Encoder` protocol and default model `analysis.domains` uses) -- a second, independent
  opinion informed by everything that encoder has seen beyond this corpus, useful for cross-
  checking the self-contained reading rather than replacing it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from corpus_generalization.analysis.markov_chain import build_transition_graph
from corpus_generalization.packing import FramingTokenizer

type Matrix = NDArray[np.float64]

# Eigenvalues at or below this are treated as exactly zero: `eigvalsh` on a matrix that is only
# PSD up to floating-point rounding can return a tiny negative value where the true eigenvalue is
# 0, and `log` of a negative number is what that would otherwise produce.
_ZERO_EIGENVALUE_TOLERANCE = 1e-12


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Matrix: ...


class Encodable(Protocol):
    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]: ...


def vendi_score(similarity: Matrix) -> float:
    """The effective number of independent items `similarity` (an n x n, symmetric, positive
    semi-definite kernel with 1s on the diagonal) describes.

    `exp(-sum(lambda_i * log(lambda_i)))` over the eigenvalues of `similarity / n`: dividing by n
    first is what makes them sum to 1, so the sum is a Shannon entropy rather than an unscaled
    number with no fixed ceiling.
    """
    n = similarity.shape[0]
    if n == 0:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(similarity / n)
    positive = eigenvalues[eigenvalues > _ZERO_EIGENVALUE_TOLERANCE]
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(np.exp(entropy))


def connected_components_count(adjacency: Matrix) -> int:
    """The exact number of disconnected pieces `adjacency`'s graph splits into -- the discrete
    complement to `algebraic_connectivity`'s continuous reading, and the one that actually answers
    "how many groups", not just "one piece or more than one".

    Zero-weight entries are never read as edges (`nx.from_numpy_array` skips them), so a unit
    with similarity 0 to everything else is correctly its own component rather than silently
    folded into whichever one happens to be node 0's.
    """
    n = adjacency.shape[0]
    if n == 0:
        return 0
    return nx.number_connected_components(nx.from_numpy_array(adjacency))


def algebraic_connectivity(adjacency: Matrix) -> float:
    """The Fiedler value: the second-smallest eigenvalue of the graph Laplacian `D - A` of a
    symmetric, non-negative weighted adjacency matrix `A` (`D` its diagonal degree matrix).

    0 for fewer than 2 nodes -- there is nothing left to disconnect. The smallest eigenvalue of any
    graph Laplacian is always exactly 0 (the all-ones vector), which is why the answer to "is this
    graph in one piece" is the *second* smallest rather than the smallest.
    """
    n = adjacency.shape[0]
    if n < 2:
        return 0.0
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return float(eigenvalues[1])


def lexical_similarity_matrix(documents: Sequence[str], tokenizer: Encodable) -> Matrix:
    """Cosine similarity between documents' own bag-of-token-id count vectors.

    Built purely from ids the project's own tokenizer assigns -- no external encoder, no
    pretrained weights, nothing that has ever seen text this corpus did not supply itself.
    """
    counts_per_document = [_token_counts(document, tokenizer) for document in documents]
    vocabulary = sorted({token_id for counts in counts_per_document for token_id in counts})
    column = {token_id: index for index, token_id in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)))
    for row, counts in enumerate(counts_per_document):
        for token_id, count in counts.items():
            matrix[row, column[token_id]] = count
    return cosine_similarity_matrix(matrix)


def semantic_similarity_matrix(documents: Sequence[str], encoder: Encoder) -> Matrix:
    """Cosine similarity between documents' embeddings under a general-purpose sentence encoder --
    the second opinion `lexical_similarity_matrix` has no way to give on its own, since it never
    reads anything the tokenizer does not already assign an id to.

    Embeds `documents` itself; a caller that already has embeddings on hand (`domains.py`'s report
    pipeline always does, for its own k-means split) should call `cosine_similarity_matrix`
    directly on them instead of paying for a second encoder pass here.
    """
    return cosine_similarity_matrix(np.asarray(encoder.encode(documents), dtype=np.float64))


def transition_adjacency_matrix(
    documents: Sequence[str], tokenizer: FramingTokenizer, order: int = 1
) -> tuple[tuple[tuple[int, ...], ...], Matrix]:
    """The self-contained substrate for `algebraic_connectivity`: `analysis.markov_chain`'s
    own transition graph, symmetrized (edge direction dropped, the same "weak connectivity" reading
    that module already uses -- a document's last token being a dead end should not by itself
    count as fragmentation) and weighted by observed transition count.

    Returns the graph's nodes alongside the matrix, in the order the matrix's rows/columns use,
    since a Fiedler value alone cannot say which states an unusually thin connection sits between.
    """
    graph = build_transition_graph(documents, tokenizer, order)
    nodes = tuple(graph.nodes)
    index = {node: position for position, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)))
    for source, target, data in graph.edges(data=True):
        weight = float(data["count"])
        source_index, target_index = index[source], index[target]
        adjacency[source_index, target_index] += weight
        adjacency[target_index, source_index] += weight
    return nodes, adjacency


def similarity_adjacency(similarity: Matrix) -> Matrix:
    """`similarity` turned into a graph adjacency matrix for `algebraic_connectivity`: the diagonal
    (every document's similarity to itself) zeroed, since a self-loop contributes nothing to
    whether two *different* documents are connected, and any negative cosine similarity clipped to
    0, since a graph edge's weight cannot meaningfully be negative."""
    adjacency = np.clip(similarity, 0.0, None).copy()
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


@dataclass(frozen=True)
class DiversityReport:
    """Vendi Score and algebraic connectivity for the same units, read two ways side by side:
    self-contained (`_lexical`, only the project's own tokenizer) and informed by a general
    sentence encoder's embeddings (`_semantic`). Agreement between the two is an independent
    second confirmation neither reading gives alone; disagreement says which one is picking up
    something the other structurally cannot -- a semantic-only drop in connectivity, for
    instance, is two documents that share no vocabulary at all yet mean nearly the same thing.
    """

    units: int
    vendi_lexical: float
    vendi_semantic: float
    algebraic_connectivity_lexical: float
    algebraic_connectivity_semantic: float
    components_lexical: int
    components_semantic: int


def diversity_report(
    units: Sequence[str], tokenizer: Encodable, embeddings: Matrix
) -> DiversityReport:
    """`embeddings`, already computed for `units` by a caller (`domains.py`'s report pipeline
    always has them, for its own k-means split), are read directly rather than re-derived --
    paying for a second encoder pass here would be pure waste. Empty (`embeddings.shape[1] == 0`,
    what a corpus too small to group at all is handed), the semantic readings are 0.0/0 rather than
    the result of running spectral math on a matrix with nothing in it.
    """
    lexical_similarity = lexical_similarity_matrix(units, tokenizer)
    lexical_adjacency = similarity_adjacency(lexical_similarity)
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        vendi_semantic, connectivity_semantic, components_semantic = 0.0, 0.0, 0
    else:
        semantic_similarity = cosine_similarity_matrix(embeddings)
        semantic_adjacency = similarity_adjacency(semantic_similarity)
        vendi_semantic = vendi_score(semantic_similarity)
        connectivity_semantic = algebraic_connectivity(semantic_adjacency)
        components_semantic = connected_components_count(semantic_adjacency)
    return DiversityReport(
        units=len(units),
        vendi_lexical=vendi_score(lexical_similarity),
        vendi_semantic=vendi_semantic,
        algebraic_connectivity_lexical=algebraic_connectivity(lexical_adjacency),
        algebraic_connectivity_semantic=connectivity_semantic,
        components_lexical=connected_components_count(lexical_adjacency),
        components_semantic=components_semantic,
    )


def cosine_similarity_matrix(vectors: Matrix) -> Matrix:
    """Every pairwise cosine similarity between `vectors`' rows, clipped to `[-1, 1]` against
    floating-point overshoot. Public so a caller who already has vectors on hand -- embeddings a
    domain-structure pass already computed, most often -- never has to re-derive them just to
    reach `vendi_score`/`algebraic_connectivity`."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)
    similarity: Matrix = np.clip(unit @ unit.T, -1.0, 1.0)
    return similarity


def _token_counts(document: str, tokenizer: Encodable) -> dict[int, int]:
    counts: dict[int, int] = {}
    for token_id in tokenizer.encode(document, recognize_markers=True):
        counts[token_id] = counts.get(token_id, 0) + 1
    return counts
