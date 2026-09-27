"""Subsampling test: do the coverage-curve AUCs measure structure or corpus size?

Tokenizer fixed (learned once on the first 5000 docs, as the original analysis did), documents
subsampled at several sizes and seeds; each sample's order-1 graph (bos/eos framed) scored with
the four curves' linear and log AUC.
"""

import json
import random
import sys
import time
from pathlib import Path

from corpus_generalization.analysis.markov_chain import (
    build_transition_graph,
    reciprocal_cluster_coverage_curve,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    to_log_curve,
    weighted_core_coverage_curve,
)
from corpus_generalization.analysis.tokenization import learned_tokenizer
from corpus_generalization.sources import huggingface_documents

OUT = Path(__file__).parent
DATASETS = {
    "wikipedia": ("wikimedia/wikipedia", "20231101.en"),
    "fineweb": ("HuggingFaceFW/fineweb", None),
}
TOTAL = 20000
TOKENIZER_DOCS = 5000
SIZES = (1250, 2500, 5000, 10000, 20000)
SEEDS = (0, 1, 2)
CURVES = {
    "red": red_cluster_coverage_curve,
    "blue": weighted_core_coverage_curve,
    "orange": reciprocal_cluster_coverage_curve,
    "green": reciprocal_core_coverage_curve,
}


class Cached:
    """Documents are passed as their index; encode returns the ids encoded once up front."""

    def __init__(self, ids: list[list[int]], bos: int, eos: int) -> None:
        self.ids, self.bos_id, self.eos_id = ids, bos, eos

    def encode(self, text: str, *, recognize_markers: bool = False) -> list[int]:
        return self.ids[int(text)]


def encoded(name: str) -> Cached:
    cache = OUT / f"{name}_ids.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        return Cached(saved["ids"], saved["bos"], saved["eos"])
    dataset, config = DATASETS[name]
    start = time.time()
    documents = huggingface_documents(dataset, "train", "text", TOTAL, config)
    print(f"{name}: {len(documents)} docs loaded in {time.time() - start:.0f}s", flush=True)
    tokenizer = learned_tokenizer(documents[:TOKENIZER_DOCS], "byte_level", 3000)
    print(f"{name}: tokenizer learned at {time.time() - start:.0f}s", flush=True)
    ids = [tokenizer.encode(d, recognize_markers=True) for d in documents]
    print(f"{name}: encoded at {time.time() - start:.0f}s", flush=True)
    cached = Cached(ids, tokenizer.bos_id, tokenizer.eos_id)
    cache.write_text(json.dumps({"ids": ids, "bos": cached.bos_id, "eos": cached.eos_id}))
    return cached


def main(name: str) -> None:
    cached = encoded(name)
    rows = []
    for size in SIZES:
        for seed in SEEDS if size < len(cached.ids) else (0,):
            sample = random.Random(seed).sample(range(len(cached.ids)), size)
            graph = build_transition_graph(
                [str(i) for i in sample], cached, order=1, include_boundaries=True
            )
            row = {
                "size": size,
                "seed": seed,
                "nodes": graph.number_of_nodes(),
                "tokens": sum(len(cached.ids[i]) for i in sample),
            }
            for label, curve_of in CURVES.items():
                curve = curve_of(graph)
                row[f"{label}_linear"] = curve.auc
                row[f"{label}_log"] = to_log_curve(curve).auc
                row[f"{label}_max"] = curve.thresholds[-1] if curve.thresholds else 0.0
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUT / f"{name}_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
