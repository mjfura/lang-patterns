"""Clean domains: where does clean text of different domains sit on the two single-corpus
readings `experiments/contamination` found?

The readings: the gap between red read over `count` and over `context_support` at 16000 merges
(redundancy; 0.008 for clean FineWeb at any size), and green over `context_support` at 4000
merges (order). If clean text of every domain lands near the same values, they can be given
thresholds; if not, a threshold depends on the domain.

Every corpus is cut into two samples of `SAMPLE_BYTES` (the contamination base's size), taken as
whole documents in order: the first sample and the one after it, so the pair shows how much a
reading moves between two slices of the same source.

    clean_domains.py download <name>   # fetch and cache the documents, then exit
    clean_domains.py score <name>      # score both samples; writes <name>_results.json
    clean_domains.py dedup <name>      # the first sample again, exact duplicate documents
                                       # removed first; writes <name>_dedup_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

from corpus_generalization.analysis.markov_chain import (
    build_transition_graph,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    to_log_curve,
    weighted_core_coverage_curve,
)
from corpus_generalization.analysis.tokenization import learned_tokenizer, tokenization_stats
from corpus_generalization.sources import huggingface_documents

OUT = Path(__file__).parent
SAMPLE_BYTES = 6_495_298  # the 2000 FineWeb documents `experiments/contamination` uses as base
SAMPLES = 2
BUDGETS = (4000, 16000)
# name: (dataset, config, split, text field, rows to request: enough for two samples)
CORPORA: dict[str, tuple[str, str | None, str, str, int]] = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "default", "train", "text", 5000),
    "c4": ("allenai/c4", "en", "train", "text", 12000),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train", "text", 1500),
    "multi-nli": ("nyu-mll/multi_nli", None, "train", "premise", 150000),
    "pubmed": ("ccdv/pubmed-summarization", "document", "train", "abstract", 14000),
    "arxiv": ("ccdv/arxiv-summarization", "document", "train", "abstract", 11000),
    "case-law": ("HFforLegal/case-law", None, "us", "document", 3000),
}
FINEWEB_CACHE = OUT.parent / "merge_sweep" / "fineweb_documents.json"


def size(documents: list[str]) -> int:
    return sum(len(document.encode()) for document in documents)


def cache_of(name: str) -> Path:
    return FINEWEB_CACHE if name == "fineweb" else OUT / f"{name}_documents.json"


def download(name: str) -> None:
    dataset, config, split, field, rows = CORPORA[name]
    documents = huggingface_documents(dataset, split, field, rows, config)
    cache_of(name).write_text(json.dumps(documents))
    print(f"{name}: {len(documents)} documents, {size(documents):,} bytes", flush=True)
    # A streamed dataset can leave threads that keep the interpreter from exiting.
    os._exit(0)


def samples(documents: list[str]) -> list[list[str]]:
    cut: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for document in documents:
        current.append(document)
        current_size += len(document.encode())
        if current_size >= SAMPLE_BYTES:
            cut.append(current)
            current, current_size = [], 0
            if len(cut) == SAMPLES:
                break
    return cut


def deduplicated(documents: list[str]) -> list[str]:
    return list(dict.fromkeys(documents))


def score(name: str, dedup: bool = False) -> None:
    documents = list(json.loads(cache_of(name).read_text()))
    if dedup:
        documents = deduplicated(documents)
    cut = samples(documents)[: 1 if dedup else SAMPLES]
    if len(cut) < (1 if dedup else SAMPLES):
        raise SystemExit(f"{name}: {size(documents):,} bytes cached, too few for the samples")
    rows = []
    for index, sample in enumerate(cut):
        for budget in BUDGETS:
            start = time.time()
            tokenizer = learned_tokenizer(sample, "byte_level", budget)
            stats = tokenization_stats(sample, tokenizer, budget)
            graph = build_transition_graph(sample, tokenizer, order=1, include_boundaries=True)
            row: dict[str, float | int | str] = {
                "corpus": name,
                "sample": index,
                "documents": len(sample),
                "bytes": size(sample),
                "merge_budget": budget,
                "tokens": stats.tokens,
                "bytes_per_token": stats.bytes_per_token,
            }
            aucs = {
                f"{prefix}{label}_log": to_log_curve(curve_of(graph, weight_key)).auc
                for weight_key, prefix in (("count", ""), ("context_support", "context_"))
                for label, curve_of in (
                    ("red", red_cluster_coverage_curve),
                    ("blue", weighted_core_coverage_curve),
                    ("green", reciprocal_core_coverage_curve),
                )
            }
            row.update(aucs)
            row["red_gap"] = aucs["red_log"] - aucs["context_red_log"]
            row["seconds"] = round(time.time() - start, 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
    suffix = "_dedup" if dedup else ""
    (OUT / f"{name}{suffix}_results.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    commands = {"download": download, "score": score, "dedup": lambda name: score(name, True)}
    commands[sys.argv[1]](sys.argv[2])
