"""Contamination: does the indicator tell data that adds domain patterns from data that only adds
volume?

Base: the first 2000 FineWeb documents (the same cache `experiments/merge_sweep` downloads). To it
we add, at 10% / 30% / 50% of the base's bytes, one kind of data:

- `control`: further FineWeb documents (2000 onward), what an engineer expects to help;
- `duplicate`: exact copies of base documents, sampled without replacement;
- `python`: Python source files (`codeparrot/codeparrot-clean-valid`), a domain far from the base;
- `shuffled`: the `control` documents with their words shuffled, the same vocabulary and word
  frequencies with the order destroyed.

Every kind adds the same bytes at a given dose, so the variants are compared at equal size. Each
variant learns its own tokenizer, like a real retraining, and is scored across the merge sweep's
budgets.

`contamination.py <kind> [seed]`: seed 0 (the default) keeps the cached order, so it reproduces the
first run; any other seed shuffles the FineWeb documents before splitting base from the rest, and
the Python files, so both what forms the base and what is added change.
"""

import json
import random
import sys
import time
from pathlib import Path

from corpus_generalization.analysis.markov_chain import (
    build_transition_graph,
    document_cluster_coverage_curve,
    reciprocal_cluster_coverage_curve,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    to_log_curve,
    weighted_core_coverage_curve,
)
from corpus_generalization.analysis.tokenization import learned_tokenizer, tokenization_stats
from corpus_generalization.sources import huggingface_documents

OUT = Path(__file__).parent
FINEWEB_CACHE = OUT.parent / "merge_sweep" / "fineweb_documents.json"
PYTHON_CACHE = OUT / "python_documents.json"
BASE_DOCUMENTS = 2000
PYTHON_FILES = 800
DOSES = (0.1, 0.3, 0.5)
BUDGETS = (0, 16, 64, 250, 1000, 4000, 16000)
CURVES = {
    "red": red_cluster_coverage_curve,
    "blue": weighted_core_coverage_curve,
    "orange": reciprocal_cluster_coverage_curve,
    "green": reciprocal_core_coverage_curve,
    "document_red": document_cluster_coverage_curve,
}


def size(documents: list[str]) -> int:
    return sum(len(document.encode()) for document in documents)


def up_to(pool: list[str], target: int) -> list[str]:
    """Whole documents from `pool`, in order, until they reach `target` bytes."""
    taken: list[str] = []
    for document in pool:
        if size(taken) >= target:
            break
        taken.append(document)
    return taken


def python_documents() -> list[str]:
    if PYTHON_CACHE.exists():
        return list(json.loads(PYTHON_CACHE.read_text()))
    documents = huggingface_documents(
        "codeparrot/codeparrot-clean-valid", "train", "content", PYTHON_FILES
    )
    PYTHON_CACHE.write_text(json.dumps(documents))
    return documents


def shuffled(document: str, rng: random.Random) -> str:
    words = document.split()
    rng.shuffle(words)
    return " ".join(words)


def seeded(documents: list[str], seed: int) -> list[str]:
    return documents if seed == 0 else random.Random(seed).sample(documents, len(documents))


def addition(kind: str, base: list[str], rest: list[str], target: int, seed: int) -> list[str]:
    if kind == "control":
        return up_to(rest, target)
    if kind == "duplicate":
        return up_to(random.Random(seed).sample(base, len(base)), target)
    if kind == "python":
        return up_to(seeded(python_documents(), seed), target)
    if kind == "shuffled":
        rng = random.Random(seed)
        return [shuffled(document, rng) for document in up_to(rest, target)]
    raise ValueError(kind)


def score(documents: list[str], budget: int) -> dict[str, object]:
    tokenizer = learned_tokenizer(documents, "byte_level", budget)
    stats = tokenization_stats(documents, tokenizer, budget)
    graph = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    row: dict[str, object] = {
        "merge_budget": budget,
        "merges": stats.merges,
        "tokens": stats.tokens,
        "bytes_per_token": stats.bytes_per_token,
        "nodes": graph.number_of_nodes(),
    }
    for label, curve_of in CURVES.items():
        row[f"{label}_log"] = to_log_curve(curve_of(graph)).auc
    return row


def main(kind: str, seed: int) -> None:
    fineweb = seeded(list(json.loads(FINEWEB_CACHE.read_text())), seed)
    base, rest = fineweb[:BASE_DOCUMENTS], fineweb[BASE_DOCUMENTS:]
    rows = []
    for dose in (0.0,) if kind == "base" else DOSES:
        added = [] if kind == "base" else addition(kind, base, rest, int(dose * size(base)), seed)
        documents = base + added
        for budget in BUDGETS:
            start = time.time()
            row = {
                "kind": kind,
                "seed": seed,
                "dose": dose,
                "added_bytes": size(added),
                "bytes": size(documents),
                **score(documents, budget),
                "seconds": 0.0,
            }
            row["seconds"] = round(time.time() - start, 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    suffix = "" if seed == 0 else f"_seed{seed}"
    (OUT / f"{kind}{suffix}_results.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
