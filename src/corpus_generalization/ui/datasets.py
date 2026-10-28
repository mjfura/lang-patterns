"""The corpora the Streamlit diagnosis page offers, as corpus or reference: the recipe grammar's
variants, the hand-built `data/kid_learning` corpora, the real sources the experiments cached, and
FineWeb with the planted gap of `experiments/planted_model`. Cached sources appear only when their
cache file exists (the experiments' downloads are gitignored).
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from corpus_generalization.sources import read_documents
from corpus_generalization.synthetic.recipes import LEXICON, recipe_documents, verb_pairing

KID_LEARNING = Path("data/kid_learning")
CACHES = {
    "fineweb": Path("experiments/merge_sweep/fineweb_documents.json"),
    **{
        name: Path(f"experiments/clean_domains/{name}_documents.json")
        for name in (
            "wikipedia",
            "fineweb-edu",
            "c4",
            "arxiv",
            "pubmed",
            "case-law",
            "multi-nli",
        )
    },
}
# Merges that give each cached source about 1.5 tokens per word (experiments/granularity); case
# law never reaches it.
SOURCE_MERGES = {
    "arxiv": 3001,
    "pubmed": 3931,
    "multi-nli": 5291,
    "c4": 8696,
    "fineweb-edu": 10802,
    "fineweb": 11563,
    "wikipedia": 13674,
    "case-law": 32000,
}
# The five FineWeb heads of experiments/planted_model and the two fillers each is kept to.
PLANTED = {
    "people": ["who", "to"],
    "she": ["was", "said"],
    "But": ["the", "it"],
    "these": ["things", "days"],
    "only": ["a", "to"],
}
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    description: str
    merges: int  # a suggested merge budget: every word one token, or about 1.5 tokens per word
    load: Callable[[int | None], list[str]]  # the first `limit` documents (all when None)


def _limited(documents: list[str], limit: int | None) -> list[str]:
    return documents if limit is None else documents[:limit]


def _recipes(fillers: int | None, seed: int) -> Callable[[int | None], list[str]]:
    def load(limit: int | None) -> list[str]:
        pairing = verb_pairing(LEXICON, fillers, reserved_per_verb=2, seed=0)
        return recipe_documents(LEXICON, pairing, count=limit or 2000, seed=seed)

    return load


def _file(path: Path) -> Callable[[int | None], list[str]]:
    return lambda limit: _limited(list(read_documents(path)), limit)


def _cache(path: Path) -> Callable[[int | None], list[str]]:
    return lambda limit: _limited(json.loads(path.read_text()), limit)


def plant(documents: list[str], heads: dict[str, list[str]]) -> list[str]:
    """Replace every word after a head that is not one of its kept fillers with the first kept
    filler (trailing punctuation kept): the head keeps its traffic and loses its variety, as in
    `experiments/planted_gap` (replace mode)."""
    planted = []
    for document in documents:
        sentences = []
        for sentence in SENTENCE.split(document):
            words = sentence.split()
            hits = [
                (index + 1, word)
                for index, (word, following) in enumerate(pairwise(words))
                if word in heads
                and re.fullmatch(r"[A-Za-z]+", core := re.sub(r"[^A-Za-z]+$", "", following))
                and core not in heads[word]
            ]
            for index, head in hits:
                tail = re.search(r"[^A-Za-z]*$", words[index])
                words[index] = heads[head][0] + (tail.group() if tail else "")
            sentences.append(" ".join(words))
        planted.append(" ".join(sentences))
    return planted


def datasets() -> list[Dataset]:
    """Every dataset available on this machine, recipes first."""
    available = [
        Dataset(
            "recipes-all",
            "Recipes: every verb with all its ingredients",
            "Synthetic recipe grammar, 2000 documents: the diverse variant (seed 0).",
            1000,
            _recipes(None, 0),
        ),
        Dataset(
            "recipes-all-seed1",
            "Recipes: every verb with all its ingredients (other documents)",
            "The diverse variant again, other documents (seed 1): a reference for the others.",
            1000,
            _recipes(None, 1),
        ),
        *(
            Dataset(
                f"recipes-{k}",
                f"Recipes: each verb with {k} ingredient{'s' if k > 1 else ''}",
                f"Synthetic recipe grammar, 2000 documents, each verb seen with {k} of its 28 "
                "ingredients: a known syntactic gap.",
                1000,
                _recipes(k, 0),
            )
            for k in (16, 4, 1)
        ),
    ]
    if (fineweb := CACHES["fineweb"]).is_file():
        available.append(
            Dataset(
                "fineweb-planted",
                "FineWeb with a planted gap (people, she, But, these, only)",
                "FineWeb documents where five words are kept to two next words each "
                "(experiments/planted_model): a known gap in real text. Read it against FineWeb.",
                SOURCE_MERGES["fineweb"],
                lambda limit: plant(_limited(json.loads(fineweb.read_text()), limit), PLANTED),
            )
        )
    for name, path in CACHES.items():
        if path.is_file():
            available.append(
                Dataset(
                    name,
                    f"Real text: {name}",
                    f"The {name} documents the experiments cached ({path}).",
                    SOURCE_MERGES[name],
                    _cache(path),
                )
            )
    for path in sorted(KID_LEARNING.glob("*.txt")):
        available.append(
            Dataset(
                f"kid-{path.stem}",
                f"Kid learning: {path.stem}",
                f"Hand-built corpus {path} (generated with Gemini; small).",
                1000,
                _file(path),
            )
        )
    return available
