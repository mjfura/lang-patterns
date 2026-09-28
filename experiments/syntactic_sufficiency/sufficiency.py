"""Does a verb's coverage of its noun slot predict whether the model ties the verb to the nouns it
was shown? The first trained test of `docs/2026-09-28-syntactic-sufficiency-design.md`, whose
predictions were written before this ran.

    sufficiency.py run <variant> [<size> <seed>]   train on one variant and score it
    sufficiency.py jobs                             the grid's runs without results, one per line
    sufficiency.py recoverage                       recompute coverage for existing results
    sufficiency.py summarize                        tables and rank correlations

Variants: `fillers_per_verb` in 1, 2, 4, 8, 16, 20, 24 and `all`; sizes `small` and `medium`;
seeds 0-2 (a seed draws the reserved combinations, the fillers, the documents and the
initialization). Runs go to `runs/syntactic_sufficiency/<size>/<variant>_seed<seed>/`
(gitignored); results to `results/<size>/<variant>_seed<seed>.json` here. `THREADS` sets torch's
thread count, so several runs can train at once:

    poetry run python sufficiency.py jobs | xargs -P 4 -L 1 env THREADS=2 poetry run python \
        sufficiency.py run
"""

import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from scipy.stats import spearmanr

from corpus_generalization.analysis.information import heldout_information
from corpus_generalization.analysis.syntactic import induce_classes, syntactic_patterns
from corpus_generalization.model.data import split_documents
from corpus_generalization.model.generate import load_run
from corpus_generalization.model.scoring import continuation_log_probability
from corpus_generalization.model.train import TrainingSettings, train
from corpus_generalization.packing import framed_documents
from corpus_generalization.synthetic.recipes import (
    LEXICON,
    minimal_pairs,
    recipe_documents,
    verb_pairing,
)
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
RUNS = Path("runs") / "syntactic_sufficiency"
VARIANTS: dict[str, int | None] = {
    "1": 1, "2": 2, "4": 4, "8": 8, "16": 16, "20": 20, "24": 24, "all": None,
}  # fmt: skip
SIZES = {"small": {"n_layer": 2, "d_model": 128}, "medium": {"n_layer": 4, "d_model": 256}}
SEEDS = (0, 1, 2)
DOCUMENTS = 2000
BASE = TrainingSettings(
    merge_budget=1000, n_head=4, context=128, steps=3000, learning_rate=1e-3, eval_every=100
)
SINGULAR = {i.singular: i.singular for i in LEXICON.ingredients} | {
    i.plural: i.singular for i in LEXICON.ingredients
}
NOUNS = [ingredient.singular for ingredient in LEXICON.ingredients]


def seen_with_verb(documents: list[str]) -> dict[str, set[str]]:
    """Each verb's ingredients (singular) over every training sentence the verb opens."""
    seen: dict[str, set[str]] = {verb: set() for verb in LEXICON.verbs}
    for document in documents:
        for sentence in document.split(" . "):
            words = sentence.rstrip(" .").split()
            if words and words[0] in seen:
                seen[words[0]].update(SINGULAR[word] for word in words if word in SINGULAR)
    return seen


def result_path(variant: str, size: str, seed: int) -> Path:
    return OUT / "results" / size / f"{variant}_seed{seed}.json"


def verb_coverage(
    train_documents: list[str], tokenizer: ByteLevelBPETokenizer
) -> tuple[dict[str, tuple[float, int]], int]:
    """Each verb's `<verb> the` -> ingredient pattern, on the model's own tokens: `(coverage,
    class size)`, and the number of classes. When the classes split the ingredients (in restricted
    variants, those some verb uses apart from those seen only in state sentences), the verb's slot
    is the ingredient class it is seen with most."""
    sequences = framed_documents(train_documents, tokenizer, True)
    classes = induce_classes(sequences)

    def token(word: str) -> int:
        (only,) = tokenizer.encode(word, recognize_markers=True)
        return only

    noun_classes = {classes.classes.get(token(" " + noun)) for noun in NOUNS} - {None}
    patterns = [
        pattern
        for pattern in syntactic_patterns(sequences, classes, head_length=2)
        if pattern.filler_class in noun_classes and pattern.head[1] == token(" the")
    ]
    coverage = {}
    for verb in LEXICON.verbs:
        slots = [pattern for pattern in patterns if pattern.head[0] == token(" " + verb)]
        slot = max(slots, key=lambda pattern: pattern.trunk)
        coverage[verb] = (slot.coverage, slot.class_size)
    return coverage, classes.n_classes


def recoverage() -> None:
    """Recompute every existing result's coverage from its run's tokenizer and documents."""
    for size in SIZES:
        for seed in SEEDS:
            for variant in VARIANTS:
                path = result_path(variant, size, seed)
                if not path.exists():
                    continue
                pairing = verb_pairing(LEXICON, VARIANTS[variant], reserved_per_verb=2, seed=seed)
                documents = recipe_documents(LEXICON, pairing, count=DOCUMENTS, seed=seed)
                train_documents, _ = split_documents(documents, BASE.heldout_fraction, seed)
                tokenizer = load_run(RUNS / size / f"{variant}_seed{seed}").tokenizer
                coverage, n_classes = verb_coverage(list(train_documents), tokenizer)
                results = json.loads(path.read_text())
                for verb, (value, class_size) in coverage.items():
                    results["verbs"][verb]["coverage"] = value
                    results["verbs"][verb]["class_size"] = class_size
                results["mean_coverage"] = statistics.mean(v for v, _ in coverage.values())
                results["n_classes"] = n_classes
                path.write_text(json.dumps(results, indent=2) + "\n")


def run(variant: str, size: str, seed: int) -> None:
    if threads := int(os.environ.get("THREADS", "0")):
        torch.set_num_threads(threads)
    start = time.time()
    settings = replace(BASE, seed=seed, **SIZES[size])
    pairing = verb_pairing(LEXICON, VARIANTS[variant], reserved_per_verb=2, seed=seed)
    documents = recipe_documents(LEXICON, pairing, count=DOCUMENTS, seed=seed)
    output = RUNS / size / f"{variant}_seed{seed}"
    train(documents, settings, output, sources={"synthetic": f"recipes fillers={variant}"})
    loaded = load_run(output)
    model, tokenizer = loaded.model, loaded.tokenizer
    train_documents, _ = split_documents(documents, settings.heldout_fraction, seed)

    coverage, n_classes = verb_coverage(list(train_documents), tokenizer)

    # Model: the unseen ingredients' share after `<bos><verb> the`, against their share of the
    # ingredients.
    seen = seen_with_verb(list(train_documents))
    verbs = {}
    for verb in LEXICON.verbs:
        probabilities = {
            noun: continuation_log_probability(model, tokenizer, f"{verb} the", f" {noun}")
            for noun in NOUNS
        }
        unseen = [noun for noun in NOUNS if noun not in seen[verb]]
        total = sum(math.exp(value) for value in probabilities.values())
        unseen_share = sum(math.exp(probabilities[noun]) for noun in unseen) / total
        verbs[verb] = {
            "coverage": coverage[verb][0],
            "class_size": coverage[verb][1],
            "seen": len(seen[verb]),
            "unseen": len(unseen),
            "noun_probability": total,
            "ratio": unseen_share / (len(unseen) / len(NOUNS)) if unseen else None,
        }

    pairs = minimal_pairs(LEXICON, pairing, seed=seed)
    accuracy = {
        kind: statistics.mean(
            continuation_log_probability(model, tokenizer, "", pair.good)
            > continuation_log_probability(model, tokenizer, "", pair.bad)
            for pair in pairs
            if pair.kind == kind
        )
        for kind in ("verb", "agreement")
    }
    information = heldout_information(documents, merge_budget=settings.merge_budget, seed=seed)

    results = {
        "variant": variant,
        "size": size,
        "seed": seed,
        "parameters": loaded.metrics["parameters"],
        "n_classes": n_classes,
        "verbs": verbs,
        "mean_coverage": statistics.mean(v["coverage"] for v in verbs.values()),
        "mean_ratio": statistics.mean(v["ratio"] for v in verbs.values() if v["ratio"] is not None),
        "minimal_pair_accuracy": accuracy,
        "best_heldout_loss": loaded.metrics["best_heldout_loss"],
        "best_train_loss": loaded.metrics["best_train_loss"],
        "best_step": loaded.metrics["best_step"],
        "ngram_predictability_bpb": min(information.heldout_bpb.values()),
        "seconds": round(time.time() - start, 1),
    }
    path = result_path(variant, size, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "verbs"}, indent=2))


def jobs() -> None:
    for size in SIZES:
        for seed in SEEDS:
            for variant in VARIANTS:
                if not result_path(variant, size, seed).exists():
                    print(variant, size, seed)


def _span(values: list[float]) -> str:
    return f"{statistics.mean(values):6.3f} [{min(values):.3f}, {max(values):.3f}]"


def summarize() -> None:
    for size in SIZES:
        results = [
            json.loads(result_path(variant, size, seed).read_text())
            for seed in SEEDS
            for variant in VARIANTS
            if result_path(variant, size, seed).exists()
        ]
        if not results:
            continue
        print(f"\n{size} ({results[0]['parameters']:,} parameters), mean [min, max] over seeds")
        print("variant  seeds  coverage                ratio                   verb pairs"
              "              held-out loss")  # fmt: skip
        for variant in VARIANTS:
            rows = [r for r in results if r["variant"] == variant]
            if not rows:
                continue
            print(
                f"{variant:>7}  {len(rows):5d}  {_span([r['mean_coverage'] for r in rows])}  "
                f"{_span([r['mean_ratio'] for r in rows])}  "
                f"{_span([r['minimal_pair_accuracy']['verb'] for r in rows])}  "
                f"{_span([r['best_heldout_loss'] for r in rows])}"
            )
        for seed in SEEDS:
            rows = [r for r in results if r["seed"] == seed]
            pairs = [
                (verb["coverage"], verb["ratio"])
                for r in rows
                for verb in r["verbs"].values()
                if verb["ratio"] is not None
            ]
            if len(rows) < 2:
                continue
            per_verb = spearmanr(*zip(*pairs, strict=True)).statistic
            reverse = spearmanr(
                [r["best_heldout_loss"] for r in rows], [r["mean_ratio"] for r in rows]
            ).statistic
            print(
                f"seed {seed}: {len(rows)} variants, per-verb Spearman(coverage, ratio) "
                f"{per_verb:.3f} over {len(pairs)} verbs; "
                f"Spearman(held-out loss, ratio) across variants {reverse:.3f}"
            )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", variant] if variant in VARIANTS:
            run(variant, "small", 0)
        case ["run", variant, size, seed] if variant in VARIANTS and size in SIZES:
            run(variant, size, int(seed))
        case ["jobs"]:
            jobs()
        case ["recoverage"]:
            recoverage()
        case ["summarize"]:
            summarize()
        case _:
            sys.exit(__doc__)
