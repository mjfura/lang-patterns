"""Merge sweep: how do the curves and islands move with the tokenizer's merge budget?

For each corpus, a byte-level tokenizer is learned on the whole corpus at every budget in
`BUDGETS`, and its order-1 `<bos>`/`<eos>`-framed graph is scored: the four curves' linear and log
AUC and the island decomposition at count >= 2, next to the token count and bytes per token, which
change with the budget and have to be read alongside (log AUC grows with token count; see
`experiments/subsampling`).
"""

import json
import sys
import time
from pathlib import Path

from corpus_generalization.analysis.markov_chain import (
    WELL_SUPPORTED_THRESHOLD,
    build_transition_graph,
    island_decomposition,
    reciprocal_cluster_coverage_curve,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    to_log_curve,
    weighted_core_coverage_curve,
)
from corpus_generalization.analysis.tokenization import learned_tokenizer, tokenization_stats
from corpus_generalization.sources import huggingface_documents, read_documents_from

OUT = Path(__file__).parent
BUDGETS = (0, 16, 64, 250, 1000, 4000, 16000)
CORPORA = {
    "actions": ("files", ["data/kid_learning/actions_train.txt"]),
    "states-x-space": (
        "files",
        ["data/kid_learning/states_train.txt", "data/kid_learning/space_astronomy_train.txt"],
    ),
    # The same 5000 documents analysis-15-fineweb reads.
    "fineweb": ("huggingface", ["HuggingFaceFW/fineweb"]),
}
FINEWEB_DOCUMENTS = 5000
CURVES = {
    "red": red_cluster_coverage_curve,
    "blue": weighted_core_coverage_curve,
    "orange": reciprocal_cluster_coverage_curve,
    "green": reciprocal_core_coverage_curve,
}


def documents_of(name: str) -> list[str]:
    kind, sources = CORPORA[name]
    if kind == "files":
        return list(read_documents_from([Path(source) for source in sources]))
    cache = OUT / f"{name}_documents.json"
    if cache.exists():
        return list(json.loads(cache.read_text()))
    documents = huggingface_documents(sources[0], "train", "text", FINEWEB_DOCUMENTS, None)
    cache.write_text(json.dumps(documents))
    return documents


def main(name: str) -> None:
    documents = documents_of(name)
    rows = []
    for budget in BUDGETS:
        start = time.time()
        tokenizer = learned_tokenizer(documents, "byte_level", budget)
        stats = tokenization_stats(documents, tokenizer, budget)
        graph = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
        row: dict[str, object] = {
            "merge_budget": budget,
            "merges": stats.merges,
            "distinct_tokens": stats.distinct_tokens,
            "tokens": stats.tokens,
            "bytes_per_token": stats.bytes_per_token,
            "nodes": graph.number_of_nodes(),
        }
        for label, curve_of in CURVES.items():
            curve = curve_of(graph)
            row[f"{label}_linear"] = curve.auc
            row[f"{label}_log"] = to_log_curve(curve).auc
        red_max = max((float(count) for *_, count in graph.edges(data="count")), default=0.0)
        row["red_max"] = red_max
        at_threshold = next(
            (
                level
                for level in island_decomposition(graph)
                if level.threshold >= WELL_SUPPORTED_THRESHOLD
            ),
            None,
        )
        for field in ("mainland", "islands", "isolated", "island_count", "largest_island"):
            row[field] = getattr(at_threshold, field) if at_threshold else 0
        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / f"{name}_results.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
