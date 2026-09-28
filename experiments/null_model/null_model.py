"""Null model: score a corpus against itself with the word order destroyed.

`experiments/clean_domains` found that neither single-corpus reading has one value for clean text
of every source. The null of a corpus is the same documents with their words shuffled inside each
document (the `shuffled` contamination's transformation, applied to the whole corpus): same
vocabulary, word frequencies and document lengths, no order. Each reading is then taken as the
corpus's distance from its own null:

- order: green over `context_support` at 4000 merges, corpus minus null;
- redundancy: the red gap (red over `count` minus over `context_support`) at 16000 merges, corpus
  minus null — the null has the same document lengths, which `clean_domains` suggests set the
  gap's baseline.

Only the nulls are scored here; the corpora's own readings come from `experiments/contamination`
and `experiments/clean_domains`, whose documents are rebuilt the same way.

    null_model.py clean <source>                   # first clean_domains sample
    null_model.py contamination <kind> <dose>      # seed 0; kind `base` takes dose 0
    null_model.py summarize                        # print the distances
"""

import json
import random
import sys
import time
from pathlib import Path

from corpus_generalization.analysis.markov_chain import (
    build_transition_graph,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    to_log_curve,
)
from corpus_generalization.analysis.tokenization import learned_tokenizer

OUT = Path(__file__).parent
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "contamination"), str(EXPERIMENTS / "clean_domains")]

import clean_domains  # noqa: E402
import contamination  # noqa: E402

BUDGETS = (4000, 16000)


def null(documents: list[str]) -> list[str]:
    rng = random.Random(0)
    return [contamination.shuffled(document, rng) for document in documents]


def readings(documents: list[str], budget: int) -> dict[str, float]:
    tokenizer = learned_tokenizer(documents, "byte_level", budget)
    graph = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
    return {
        "green_log": to_log_curve(reciprocal_core_coverage_curve(graph)).auc,
        "context_green_log": to_log_curve(
            reciprocal_core_coverage_curve(graph, "context_support")
        ).auc,
        "red_log": to_log_curve(red_cluster_coverage_curve(graph)).auc,
        "context_red_log": to_log_curve(red_cluster_coverage_curve(graph, "context_support")).auc,
    }


def clean_documents(source: str) -> list[str]:
    documents = list(json.loads(clean_domains.cache_of(source).read_text()))
    return clean_domains.samples(documents)[0]


def contamination_documents(kind: str, dose: float) -> list[str]:
    fineweb = list(json.loads(contamination.FINEWEB_CACHE.read_text()))
    base = fineweb[: contamination.BASE_DOCUMENTS]
    rest = fineweb[contamination.BASE_DOCUMENTS :]
    if kind == "base":
        return base
    return base + contamination.addition(kind, base, rest, int(dose * contamination.size(base)), 0)


def score(name: str, documents: list[str]) -> None:
    rows = []
    shuffled = null(documents)
    for budget in BUDGETS:
        start = time.time()
        row: dict[str, object] = {"target": name, "merge_budget": budget}
        row.update(readings(shuffled, budget))
        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / f"{name}_null_results.json").write_text(json.dumps(rows, indent=1) + "\n")


def own(path: Path, budget: int, **match: object) -> dict[str, float]:
    return next(
        row
        for row in json.loads(path.read_text())
        if row["merge_budget"] == budget and all(row.get(k) == v for k, v in match.items())
    )


def distances(target: str, corpus_row: dict[int, dict[str, float]]) -> tuple[float, float]:
    nulls = {
        row["merge_budget"]: row
        for row in json.loads((OUT / f"{target}_null_results.json").read_text())
    }
    order = corpus_row[4000]["context_green_log"] - nulls[4000]["context_green_log"]
    gap = corpus_row[16000]["red_log"] - corpus_row[16000]["context_red_log"]
    null_gap = nulls[16000]["red_log"] - nulls[16000]["context_red_log"]
    return order, gap - null_gap


def summarize() -> None:
    print(f"{'target':24} {'order distance':>15} {'redundancy distance':>20}")
    for source in (*clean_domains.CORPORA, "fineweb"):
        target = f"clean-{source}"
        if not (OUT / f"{target}_null_results.json").exists():
            continue
        path = clean_domains.OUT / f"{source}_results.json"
        corpus = {budget: own(path, budget, sample=0) for budget in BUDGETS}
        print(
            f"{target:24} {distances(target, corpus)[0]:15.3f} {distances(target, corpus)[1]:20.3f}"
        )
    for kind, doses in (
        ("base", (0.0,)),
        *((k, contamination.DOSES) for k in ("control", "duplicate", "python", "shuffled")),
    ):
        for dose in doses:
            target = f"contamination-{kind}-{dose}"
            if not (OUT / f"{target}_null_results.json").exists():
                continue
            path = contamination.OUT / f"{kind}_results.json"
            corpus = {budget: own(path, budget, dose=dose) for budget in BUDGETS}
            order, redundancy = distances(target, corpus)
            print(f"{target:24} {order:15.3f} {redundancy:20.3f}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "clean":
        score(f"clean-{sys.argv[2]}", clean_documents(sys.argv[2]))
    elif command == "contamination":
        kind, dose = sys.argv[2], float(sys.argv[3])
        score(f"contamination-{kind}-{dose}", contamination_documents(kind, dose))
    else:
        summarize()
