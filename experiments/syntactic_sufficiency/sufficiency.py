"""Does a verb's coverage of its noun slot predict whether the model ties the verb to the nouns it
was shown? The first trained test of `docs/2026-09-28-syntactic-sufficiency-design.md`, whose
predictions were written before this ran.

    sufficiency.py run <variant>     train on one variant and score it
    sufficiency.py summarize         the table and the per-verb rank correlation

Variants: `fillers_per_verb` in 1, 2, 4, 8, 16 and `all`. Runs go to
`runs/syntactic_sufficiency/<variant>/` (gitignored); results to `<variant>_results.json` here.
`THREADS` sets torch's thread count, so several variants can train at once.
"""

import json
import math
import os
import statistics
import sys
import time
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

OUT = Path(__file__).parent
RUNS = Path("runs") / "syntactic_sufficiency"
VARIANTS: dict[str, int | None] = {"1": 1, "2": 2, "4": 4, "8": 8, "16": 16, "all": None}
DOCUMENTS = 2000
SEED = 0
SETTINGS = TrainingSettings(
    merge_budget=1000,
    n_layer=2,
    n_head=4,
    d_model=128,
    context=128,
    steps=3000,
    learning_rate=1e-3,
    eval_every=100,
    seed=SEED,
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


def run(variant: str) -> None:
    if threads := int(os.environ.get("THREADS", "0")):
        torch.set_num_threads(threads)
    start = time.time()
    pairing = verb_pairing(LEXICON, VARIANTS[variant], reserved_per_verb=2, seed=SEED)
    documents = recipe_documents(LEXICON, pairing, count=DOCUMENTS, seed=SEED)
    output = RUNS / variant
    train(documents, SETTINGS, output, sources={"synthetic": f"recipes fillers={variant}"})
    loaded = load_run(output)
    model, tokenizer = loaded.model, loaded.tokenizer
    train_documents, _ = split_documents(documents, SETTINGS.heldout_fraction, SEED)

    # Corpus: each verb's `<verb> the` -> singular-ingredient pattern, on the model's own tokens.
    sequences = framed_documents(train_documents, tokenizer, True)
    classes = induce_classes(sequences)

    def token(word: str) -> int:
        (only,) = tokenizer.encode(word, recognize_markers=True)
        return only

    noun_class = classes.classes[token(" tomato")]
    patterns = {
        pattern.head: pattern
        for pattern in syntactic_patterns(sequences, classes, head_length=2)
        if pattern.filler_class == noun_class
    }

    # Model: the unseen ingredients' share after `<bos><verb> the`, against their share of the
    # ingredients.
    seen = seen_with_verb(list(train_documents))
    verbs = {}
    for verb in LEXICON.verbs:
        pattern = patterns.get((token(" " + verb), token(" the")))
        probabilities = {
            noun: continuation_log_probability(model, tokenizer, f"{verb} the", f" {noun}")
            for noun in NOUNS
        }
        unseen = [noun for noun in NOUNS if noun not in seen[verb]]
        total = sum(math.exp(value) for value in probabilities.values())
        unseen_share = sum(math.exp(probabilities[noun]) for noun in unseen) / total
        verbs[verb] = {
            "coverage": pattern.coverage if pattern else 0.0,
            "class_size": pattern.class_size if pattern else None,
            "seen": len(seen[verb]),
            "unseen": len(unseen),
            "noun_probability": total,
            "ratio": unseen_share / (len(unseen) / len(NOUNS)) if unseen else None,
        }

    pairs = minimal_pairs(LEXICON, pairing, seed=SEED)
    accuracy = {
        kind: statistics.mean(
            continuation_log_probability(model, tokenizer, "", pair.good)
            > continuation_log_probability(model, tokenizer, "", pair.bad)
            for pair in pairs
            if pair.kind == kind
        )
        for kind in ("verb", "agreement")
    }
    information = heldout_information(documents, merge_budget=SETTINGS.merge_budget, seed=SEED)

    results = {
        "variant": variant,
        "n_classes": classes.n_classes,
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
    (OUT / f"{variant}_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "verbs"}, indent=2))


def summarize() -> None:
    header = "variant  coverage  ratio  pairs(verb/agr)  heldout  train  step  ngram-bpb  classes"
    print(header)
    coverages, ratios = [], []
    for variant in VARIANTS:
        path = OUT / f"{variant}_results.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        acc = r["minimal_pair_accuracy"]
        print(
            f"{variant:>7}  {r['mean_coverage']:8.3f}  {r['mean_ratio']:5.3f}  "
            f"{acc['verb']:6.3f}/{acc['agreement']:5.3f}  {r['best_heldout_loss']:7.3f}  "
            f"{r['best_train_loss']:5.3f}  {r['best_step']:4d}  "
            f"{r['ngram_predictability_bpb']:9.3f}  {r['n_classes']:7d}"
        )
        for verb in r["verbs"].values():
            if verb["ratio"] is not None:
                coverages.append(verb["coverage"])
                ratios.append(verb["ratio"])
    if coverages:
        rho = spearmanr(coverages, ratios).statistic
        print(f"per-verb Spearman(coverage, ratio) over {len(ratios)} verbs: {rho:.3f}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", variant] if variant in VARIANTS:
            run(variant)
        case ["summarize"]:
            summarize()
        case _:
            sys.exit(__doc__)
