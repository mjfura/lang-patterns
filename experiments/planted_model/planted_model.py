"""Does a model trained on a gapped corpus fail, on unseen text of the domain, exactly where the
indicator flags? "Closing the loop on natural text" in
`docs/2026-09-28-syntactic-sufficiency-design.md`.

    planted_model.py prepare                 choose the heads, read the indicator on each corpus
    planted_model.py train <variant> <seed>  train on a corpus and score slice 7
    planted_model.py summarize

Variants: `control`, `k2`, `k16`. FineWeb's first 6.5 MB sample (`experiments/clean_domains`) in
8 slices: 0-6 train, 7 evaluates. Runs go to `runs/planted_model/<variant>_seed<seed>/`
(gitignored); results to `results/` here. `THREADS` sets torch's thread count.
"""

import json
import math
import os
import re
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import torch

from corpus_generalization.analysis.syntactic import (
    induce_classes,
    syntactic_patterns,
    word_like_tokens,
)
from corpus_generalization.model.generate import LoadedRun, load_run
from corpus_generalization.model.scoring import continuation_log_probability
from corpus_generalization.model.train import TrainingSettings, train
from corpus_generalization.packing import framed_documents
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "planted_model"
sys.path[:0] = [
    str(OUT.parent / "planted_gap"),
    str(OUT.parent / "natural_sufficiency"),
    str(OUT.parent / "clean_domains"),
]

import clean_domains  # noqa: E402
import natural  # noqa: E402
import planted  # noqa: E402

SOURCE = "fineweb"
VARIANTS = {"control": None, "k2": 2, "k16": 16}
SEEDS = (0, 1)
HEADS = 5
EVAL_SLICE = 7
MAX_OCCURRENCES = 200  # per head in the evaluation slice
PREFIX_CHARS = 500
SETTINGS = TrainingSettings(
    merge_budget=natural.MERGES[SOURCE],
    heldout_fraction=0.05,
    n_layer=2,
    n_head=4,
    d_model=128,
    context=128,
    steps=3000,
    learning_rate=1e-3,
    eval_every=100,
)


def slices() -> list[list[str]]:
    documents = clean_domains.samples(json.loads(clean_domains.cache_of(SOURCE).read_text()))[0]
    size = len(documents) // planted.SLICES
    return [documents[i * size : (i + 1) * size] for i in range(planted.SLICES)]


def training_documents(
    parts: list[list[str]], kept: dict[str, list[str]], k: int | None
) -> list[str]:
    documents = [document for part in parts[:EVAL_SLICE] for document in part]
    if k is None:
        return documents
    return planted.plant(
        documents, {head: fillers[:k] for head, fillers in kept.items()}, "replace"
    )[0]


def prepare() -> None:
    """Choose the planted and the comparison heads on the whole sample, and read the indicator on
    each training corpus with the sample's tokenizer and classes."""
    parts = slices()
    documents = [document for part in parts for document in part]
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        documents, max_merges=SETTINGS.merge_budget, recognize_markers=True
    )

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    reference = framed_documents(documents, tokenizer, True)
    words = word_like_tokens(reference, spell, boundary_ids={tokenizer.bos_id, tokenizer.eos_id})
    classes = induce_classes(reference, candidates=words)
    reference_patterns = {
        (p.head, p.filler_class): p for p in syntactic_patterns(reference, classes, heads=words)
    }
    chosen, kept = planted.choose_heads(reference, classes, reference_patterns, spell, HEADS)
    comparison, comparison_kept = planted.choose_heads(
        reference, classes, reference_patterns, spell, HEADS, skip=HEADS
    )

    indicator = {}
    for variant, k in VARIANTS.items():
        corpus = framed_documents(training_documents(parts, kept, k), tokenizer, True)
        read = {
            (p.head, p.filler_class): p for p in syntactic_patterns(corpus, classes, heads=words)
        }
        indicator[variant] = {
            spell(p.head[0]).strip(): read[(p.head, p.filler_class)].diversity
            if (p.head, p.filler_class) in read
            else None
            for p in chosen + comparison
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "heads.json").write_text(
        json.dumps(
            {
                "planted": {
                    spell(p.head[0]).strip(): kept[spell(p.head[0]).strip()] for p in chosen
                },
                "comparison": {
                    spell(p.head[0]).strip(): comparison_kept[spell(p.head[0]).strip()]
                    for p in comparison
                },
                "indicator_diversity": indicator,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(indicator, indent=2))


def heldout_bits_per_byte(loaded: LoadedRun, documents: list[str]) -> float:
    """Cross-entropy of the evaluation slice in bits per byte, over non-overlapping windows of
    the model's context: per byte so models with different tokenizers compare."""
    stream = [t for doc in framed_documents(documents, loaded.tokenizer, True) for t in doc]
    context = loaded.model.context
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(stream) - 1, context):
            window = torch.tensor([stream[start : start + context + 1]])
            if window.shape[1] < 2:
                break
            log_probs = torch.log_softmax(loaded.model(window[:, :-1]), dim=-1)
            total -= float(log_probs[0].gather(1, window[0, 1:, None]).sum())
    size = sum(len(document.encode()) for document in documents)
    return total / math.log(2) / size


def occurrence_scores(
    loaded: LoadedRun, documents: list[str], heads: list[str]
) -> list[dict[str, object]]:
    """For each head followed by a word in `documents` (up to `MAX_OCCURRENCES` per head): the
    model's log probability of that word given the text before it, from `<bos>`."""
    rows: list[dict[str, object]] = []
    counts = dict.fromkeys(heads, 0)
    pattern = re.compile(
        r"(?<![A-Za-z])(" + "|".join(map(re.escape, heads)) + r") ([A-Za-z]+)(?![A-Za-z])"
    )
    for document in documents:
        for match in pattern.finditer(document):
            head, word = match.group(1), match.group(2)
            if counts[head] >= MAX_OCCURRENCES:
                continue
            start = max(0, match.end(1) - PREFIX_CHARS)
            prefix = document[start : match.end(1)]
            if start > 0:
                prefix = prefix[prefix.find(" ") + 1 :]
            while True:
                try:
                    value = continuation_log_probability(
                        loaded.model, loaded.tokenizer, prefix, " " + word
                    )
                    break
                except ValueError as error:
                    if "context" not in str(error) or " " not in prefix:
                        value = None
                        break
                    prefix = prefix[prefix.find(" ") + 1 :]
            if value is None:
                continue
            counts[head] += 1
            rows.append({"head": head, "word": word, "log_probability": value})
    return rows


def train_variant(variant: str, seed: int) -> None:
    if threads := int(os.environ.get("THREADS", "0")):
        torch.set_num_threads(threads)
    heads = json.loads((RESULTS / "heads.json").read_text())
    parts = slices()
    documents = training_documents(parts, heads["planted"], VARIANTS[variant])
    output = RUNS / f"{variant}_seed{seed}"
    train(documents, replace(SETTINGS, seed=seed), output, sources={"planted_model": variant})
    loaded = load_run(output)
    evaluation = parts[EVAL_SLICE]
    results = {
        "variant": variant,
        "seed": seed,
        "best_step": loaded.metrics["best_step"],
        "best_heldout_loss": loaded.metrics["best_heldout_loss"],
        "eval_bits_per_byte": heldout_bits_per_byte(loaded, evaluation),
        "occurrences": occurrence_scores(
            loaded, evaluation, [*heads["planted"], *heads["comparison"]]
        ),
    }
    (RESULTS / f"{variant}_seed{seed}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(variant, seed, results["eval_bits_per_byte"], len(results["occurrences"]))


def summarize() -> None:
    heads = json.loads((RESULTS / "heads.json").read_text())
    print("indicator diversity in each training corpus:")
    for variant, reading in heads["indicator_diversity"].items():
        print(
            f"  {variant:>7}: "
            + ", ".join(f"{h} {v:.2f}" if v is not None else f"{h} -" for h, v in reading.items())
        )
    for seed in SEEDS:
        runs = {
            v: json.loads((RESULTS / f"{v}_seed{seed}.json").read_text())
            for v in VARIANTS
            if (RESULTS / f"{v}_seed{seed}.json").exists()
        }
        if "control" not in runs:
            continue
        print(
            f"\nseed {seed}: eval bits per byte "
            + ", ".join(f"{v} {r['eval_bits_per_byte']:.4f}" for v, r in runs.items())
        )
        control = runs["control"]["occurrences"]
        for variant, k in VARIANTS.items():
            if k is None or variant not in runs:
                continue
            gapped = runs[variant]["occurrences"]
            print(f"  {variant}: mean log p(word), planted minus control; outside / inside kept")
            for group in ("planted", "comparison"):
                for head, fillers in heads[group].items():
                    kept = set(fillers[:k])
                    rows = [
                        (c, g) for c, g in zip(control, gapped, strict=True) if c["head"] == head
                    ]
                    outside = [
                        g["log_probability"] - c["log_probability"]
                        for c, g in rows
                        if c["word"] not in kept
                    ]
                    inside = [
                        g["log_probability"] - c["log_probability"]
                        for c, g in rows
                        if c["word"] in kept
                    ]
                    print(
                        f"    {group:>10} {head!r:>10}: outside {_mean(outside):+.2f} "
                        f"({len(outside)}), inside {_mean(inside):+.2f} ({len(inside)})"
                    )


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["prepare"]:
            prepare()
        case ["train", variant, seed] if variant in VARIANTS:
            train_variant(variant, int(seed))
        case ["summarize"]:
            summarize()
        case _:
            sys.exit(__doc__)
