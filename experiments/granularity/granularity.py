"""Granularity: compare corpora at equal token granularity, not at equal merge budgets.

Every earlier experiment scored all corpora at the same budgets (4000 and 16000 merges). The same
budget does not give the same tokens everywhere: one corpus's merges reach whole words sooner than
another's. Here each clean source of `experiments/clean_domains` (sample 0), and its word-shuffled
null (`experiments/null_model`), is swept over `BUDGETS` and records, next to the readings, how
close its tokens are to words:

- `tokens_per_word`: tokens over whitespace-separated words; 1.0 when every word is one token;
- `whole_word_fraction`: of the merges learned since the previous budget, the share whose token
  (leading space stripped) is a whole alphabetic word of the sample — whether the tokenizer is
  still building pieces or already spelling words.

BPE merges are a prefix of any longer run on the same corpus (checked: the first k merges of a
run and a run capped at k give the same merges and ids), so each version learns its merges once at
the largest budget and is cut to each smaller one.

    granularity.py score <source> [null]    # writes <source>[_null]_results.json
    granularity.py summarize                # readings at matched tokens_per_word
"""

import json
import math
import re
import sys
import time
from itertools import pairwise
from pathlib import Path

from corpus_generalization.analysis.markov_chain import (
    build_transition_graph,
    reciprocal_core_coverage_curve,
    red_cluster_coverage_curve,
    to_log_curve,
)
from corpus_generalization.analysis.tokenization import learned_tokenizer, tokenization_stats
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "clean_domains"), str(EXPERIMENTS / "null_model")]

import clean_domains  # noqa: E402
import null_model  # noqa: E402

BUDGETS = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000)
SOURCES = ("fineweb", *clean_domains.CORPORA)
TARGETS = (1.5, 1.3, 1.2)
WORD = re.compile(r"[A-Za-z]+")


def score(source: str, null: bool) -> None:
    documents = null_model.clean_documents(source)
    if null:
        documents = null_model.null(documents)
    words = set(WORD.findall(" ".join(documents)))
    start = time.time()
    full = learned_tokenizer(documents, "byte_level", BUDGETS[-1])
    print(f"{source}: {len(full.merges)} merges in {time.time() - start:.0f}s", flush=True)
    rows = []
    previous = 0
    for budget in BUDGETS:
        start = time.time()
        merges = full.merges[:budget]
        tokenizer = ByteLevelBPETokenizer.from_merges(merges)
        new_tokens = [tokenizer.decode([new_id]).strip() for _, new_id in merges[previous:]]
        stats = tokenization_stats(documents, tokenizer, budget)
        graph = build_transition_graph(documents, tokenizer, order=1, include_boundaries=True)
        row: dict[str, object] = {
            "source": source,
            "null": null,
            "merge_budget": budget,
            "merges": len(merges),
            "tokens_per_word": stats.tokens_per_word,
            "bytes_per_token": stats.bytes_per_token,
            "whole_word_fraction": (
                sum(1 for token in new_tokens if WORD.fullmatch(token) and token in words)
                / len(new_tokens)
                if new_tokens
                else 0.0
            ),
            "green_log": to_log_curve(reciprocal_core_coverage_curve(graph)).auc,
            "context_green_log": to_log_curve(
                reciprocal_core_coverage_curve(graph, "context_support")
            ).auc,
            "red_log": to_log_curve(red_cluster_coverage_curve(graph)).auc,
            "context_red_log": to_log_curve(
                red_cluster_coverage_curve(graph, "context_support")
            ).auc,
        }
        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)
        previous = len(merges)
        print(json.dumps(row), flush=True)
    suffix = "_null" if null else ""
    (OUT / f"{source}{suffix}_results.json").write_text(json.dumps(rows, indent=1) + "\n")


Row = dict[str, float]


def _bracket(rows: list[Row], target: float) -> tuple[Row, Row, float] | None:
    """The two budgets whose `tokens_per_word` bracket `target`, and how far between them it lies
    (0 at the lower budget); `None` if the sweep never reaches `target`."""
    for low, high in pairwise(rows):
        if low["tokens_per_word"] >= target >= high["tokens_per_word"]:
            span = low["tokens_per_word"] - high["tokens_per_word"]
            return low, high, (low["tokens_per_word"] - target) / span if span else 0.0
    return None


def _at(low: Row, high: Row, weight: float, key: str) -> float:
    return low[key] + weight * (high[key] - low[key])


def readings(rows: list[Row], target: float) -> tuple[float, float, float] | None:
    """Merges (interpolated in log), green over `context_support` and the red gap at
    `tokens_per_word == target`, each linear between the two bracketing budgets."""
    bracket = _bracket(rows, target)
    if bracket is None:
        return None
    low, high, weight = bracket
    log_low, log_high = math.log(low["merges"]), math.log(high["merges"])
    merges = math.exp(log_low + weight * (log_high - log_low))
    gap = _at(low, high, weight, "red_log") - _at(low, high, weight, "context_red_log")
    return merges, _at(low, high, weight, "context_green_log"), gap


def summarize() -> None:
    for target in TARGETS:
        print(f"== tokens_per_word = {target}")
        print(f"{'source':12} {'merges':>7} {'order':>7} {'redund':>7} {'ctx green':>9} {'gap':>7}")
        for source in SOURCES:
            paths = (OUT / f"{source}_results.json", OUT / f"{source}_null_results.json")
            if not all(path.exists() for path in paths):
                continue
            corpus, null = (readings(json.loads(path.read_text()), target) for path in paths)
            if corpus is None or null is None:
                print(f"{source:12} not reached within {BUDGETS[-1]} merges")
                continue
            merges, green, gap = corpus
            print(
                f"{source:12} {merges:7.0f} {green - null[1]:7.3f} {gap - null[2]:7.3f} "
                f"{green:9.3f} {gap:7.3f}"
            )


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "null")
    else:
        summarize()
