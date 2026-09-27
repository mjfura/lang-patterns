"""Whether a corpus is one domain of knowledge or several, read off a static text encoder.

Each unit of text — a document or a sentence — becomes a dense vector from a frozen, general
encoder that has never seen this project, so the geometry says something about the text rather than
about a model trained on it. K-means groups the vectors for k = 2 up to a ceiling.

The verdict is the distance between groups, not how tidy they are. The corpus is k domains for the
largest k whose every pair of group centroids is at least `centroid_distance` apart (cosine), and
one domain when not even the two groups of k = 2 are. The Silhouette Score is still measured and
reported, and was the first criterion tried; it cannot be the verdict. Its value follows how
compact the texts inside each domain are, so no threshold on it separates one domain from two:
measured with all-MiniLM-L6-v2 at k = 2, eva-learner's fourteen geometry documents — one topic —
scored 0.312, while real two-domain mixes of 40 documents scored as low as 0.172 (Python
documentation with child speech) and 0.219 (Python Q&A with child speech). k-means had split every
one of those mixes exactly along its two sources (adjusted Rand index 1.00); only the Silhouette's
verdict was wrong.

The centroid distance separated every case. Over five random samples per condition, single domains
reached at most 0.761 (fourteen Python Q&A documents) and otherwise stayed at or under 0.597, one
topic in two registers (Python Q&A with Python documentation) at most 0.580; two domains started at
0.816 (geometry with Python Q&A), and five Python documents slipped into the geometry corpus
reached 0.854 or more. 0.8 sits in that gap — a narrow one, measured on English text with this one
encoder, and only for k = 2: another encoder or language needs the experiment repeated.

k is capped at one group per `MIN_UNITS_PER_DOMAIN` units. Uncapped, the same twelve geometry
documents were split into five groups, one of them a single document: a grouping of one or two
points is always tidy, so an uncapped search rewards splitting a small corpus as finely as allowed.
"""

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score

from corpus_generalization.tokenizer.base import IM_END_TOKEN, IM_START_TOKEN

# The ChatML roles a chat-formatted corpus writes after `<|im_start|>`.
ROLES = ("system", "user", "assistant")

type Unit = Literal["document", "sentence"]
type Embeddings = NDArray[np.float64]

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_CLUSTERS = 5
# See the module docstring for the measurements this sits between.
DEFAULT_CENTROID_DISTANCE = 0.8
# The fewest units a domain is allowed to be found from: k is capped at one group per this many
# units, and a verdict with a smaller group is reported as weak. A rule of thumb rather than a
# measurement: a grouping of a handful of points moves a long way when one point moves.
MIN_UNITS_PER_DOMAIN = 5
# Two domains of `MIN_UNITS_PER_DOMAIN` each: below this there is no k worth trying.
MIN_UNITS = 2 * MIN_UNITS_PER_DOMAIN

# A turn opening with its role, or a turn closing. Left in, the encoder would group texts by chat
# format — every document carries the same markers — before it grouped them by what they say.
_TURN_MARKER = re.compile(
    rf"{re.escape(IM_START_TOKEN)}(?:{'|'.join(ROLES)})?|{re.escape(IM_END_TOKEN)}"
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Embeddings: ...


class SentenceTransformerEncoder:
    """A frozen `sentence-transformers` model, loaded on first use — the first use ever downloads
    it into the Hugging Face cache — returning unit-length embeddings."""

    def __init__(self, name: str = DEFAULT_ENCODER) -> None:
        self.name = name
        self._model: Any = None

    def encode(self, texts: Sequence[str]) -> Embeddings:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float64)


@dataclass(frozen=True)
class Domain:
    size: int
    # The unit closest to the domain's centroid: the text that best stands for the group.
    representative: str


@dataclass(frozen=True)
class DomainStructure:
    """What the grouping found for every k tried — the Silhouette Score and the distance between
    its two closest centroids — the verdict, and the partition behind it. `domains` is 0 when there
    were too few units to group at all."""

    units: int
    silhouettes: dict[int, float]
    separations: dict[int, float]
    threshold: float
    domains: int
    labels: tuple[int, ...]
    groups: tuple[Domain, ...]
    centroid_distances: tuple[tuple[float, ...], ...]

    @property
    def weak(self) -> bool:
        """Several domains, one of them found from fewer than `MIN_UNITS_PER_DOMAIN` units."""
        return self.domains > 1 and any(
            domain.size < MIN_UNITS_PER_DOMAIN for domain in self.groups
        )


def analysis_units(documents: Sequence[str], unit: Unit) -> list[str]:
    """Each document, or each sentence of each document, with turn markers removed and whitespace
    collapsed; units left empty are dropped."""
    units: list[str] = []
    for document in documents:
        cleaned = _TURN_MARKER.sub("\n", document)
        pieces = [cleaned] if unit == "document" else _SENTENCE_BREAK.split(cleaned)
        units.extend(text for piece in pieces if (text := " ".join(piece.split())))
    return units


def domain_structure(
    units: Sequence[str],
    embeddings: Embeddings,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    threshold: float = DEFAULT_CENTROID_DISTANCE,
    seed: int = 0,
) -> DomainStructure:
    if len(units) < MIN_UNITS:
        return DomainStructure(len(units), {}, {}, threshold, 0, (0,) * len(units), (), ())
    silhouettes: dict[int, float] = {}
    separations: dict[int, float] = {}
    partitions: dict[int, NDArray[np.int32]] = {}
    for k in range(2, min(max_clusters, len(units) // MIN_UNITS_PER_DOMAIN) + 1):
        with warnings.catch_warnings():
            # Fewer distinct units than k: k-means says so and returns fewer groups, skipped below.
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)
        if len(set(labels.tolist())) < 2:
            continue
        silhouettes[k] = float(silhouette_score(embeddings, labels, metric="cosine"))
        separations[k] = _closest_pair(_centroids(embeddings, labels))
        partitions[k] = labels
    separated = [k for k, distance in separations.items() if distance >= threshold]
    chosen = partitions[max(separated)] if separated else np.zeros(len(units), dtype=np.int32)
    return _structure(units, embeddings, silhouettes, separations, threshold, chosen)


def _structure(
    units: Sequence[str],
    embeddings: Embeddings,
    silhouettes: dict[int, float],
    separations: dict[int, float],
    threshold: float,
    labels: NDArray[np.int32],
) -> DomainStructure:
    centroids = _centroids(embeddings, labels)
    groups = []
    for index, group in enumerate(sorted(set(labels.tolist()))):
        members = np.flatnonzero(labels == group)
        nearest = members[int(np.argmax(_unit(embeddings[members]) @ centroids[index]))]
        groups.append(Domain(len(members), units[int(nearest)]))
    distances = 1.0 - centroids @ centroids.T
    return DomainStructure(
        units=len(units),
        silhouettes=silhouettes,
        separations=separations,
        threshold=threshold,
        domains=len(groups),
        labels=tuple(int(label) for label in labels),
        groups=tuple(groups),
        centroid_distances=tuple(tuple(float(value) for value in row) for row in distances),
    )


def _centroids(embeddings: Embeddings, labels: NDArray[np.int32]) -> Embeddings:
    """Each group's mean direction, one unit-length row per group in label order."""
    means = np.stack(
        [embeddings[labels == group].mean(axis=0) for group in sorted(set(labels.tolist()))]
    )
    return _unit(means)


def _closest_pair(centroids: Embeddings) -> float:
    """The cosine distance between the two centroids nearest each other: a grouping is only as
    separated as its least separated pair."""
    return min(
        float(1.0 - centroids[left] @ centroids[right])
        for left, right in combinations(range(len(centroids)), 2)
    )


def _unit(vectors: Embeddings) -> Embeddings:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    result: Embeddings = vectors / np.where(norms == 0, 1.0, norms)
    return result
