"""Corpus or model? Attributing a syntactic failure across three model configurations.

"Corpus or model? Attributing a syntactic failure" in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` records the design and
predictions before training.

    model_sweep.py train <corpus> <config>   config: medium | long (small runs already exist)
    model_sweep.py items                     fine-tagged items for pronouns, modals, possessives
    model_sweep.py summarize                 G per class, corpus and configuration; predictions

Runs go to `runs/model_sweep/` (gitignored), results to `results/` here. Train two runs at a
time with `THREADS=3`. No command runs on import.
"""

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "model_sweep"
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "class_patterns")]

import class_patterns as patterns  # noqa: E402
import planted_model  # noqa: E402  (on the path through class_patterns)
import replication  # noqa: E402
from class_patterns import divergence, natural_gap, subset_selection, syntactic_eval  # noqa: E402

CORPORA = ("full", "r10", "control", "class_cut", "poss_control", "poss_class_cut")
CONFIGS = {
    "small": {},
    "medium": {"n_layer": 4, "d_model": 256, "learning_rate": 5e-4},
    "long": {"steps": 9000},
}
SEED = 0
CLASSES = {
    "pronoun": ("I", "you", "he", "she", "we", "they"),
    "modal": replication.MODALS,
    "possessive": replication.POSSESSIVES,
}
PAIRS = {
    "pronoun": ("control", "class_cut"),
    "modal": ("control", "class_cut"),
    "possessive": ("poss_control", "poss_class_cut"),
}


def documents(corpus: str) -> list[str]:
    if corpus in ("full", "r10"):
        _, pool = natural_gap.split()
        return pool if corpus == "full" else subset_selection.selected(pool, corpus)
    return patterns.corpus(corpus)


def run_path(corpus: str, config: str) -> Path:
    if config == "small":
        folder = subset_selection.RUNS if corpus in ("full", "r10") else patterns.RUNS
        return folder / f"{corpus}_seed{SEED}"
    return RUNS / f"{corpus}_{config}_seed{SEED}"


def train_corpus(corpus: str, config: str) -> None:
    settings = replace(planted_model.SETTINGS, **CONFIGS[config])
    subset_selection.train_and_score(
        documents(corpus),
        f"{corpus}_{config}",
        SEED,
        run_path(corpus, config),
        RESULTS,
        "model_sweep",
        settings,
    )


def valid_and_contrast(
    kind: str, classes: dict[str, str], forms: dict[str, str]
) -> tuple[set[str], set[str]]:
    """Pronouns: past-tense verbs, grammatical after every pronoun, against nouns and
    adjectives. Modals and possessives as in `replication`."""
    if kind != "pronoun":
        return replication.valid_and_contrast(kind, classes, forms)
    valid = {w for w, c in classes.items() if c == "VERB" and forms.get(w) == "VBD"}
    contrast = {w for w, c in classes.items() if c in ("NOUN", "ADJ")}
    return valid, contrast


def items() -> None:
    evaluation, pool = natural_gap.split()
    classes, forms, frequency = replication.fine_tags(pool)
    heads = tuple(h for members in CLASSES.values() for h in members)
    followers: dict[str, set[str]] = {head: set() for head in heads}
    for document in pool:
        for match in syntactic_eval.follower_pattern(heads).finditer(document):
            followers[match.group(1)].add(match.group(2).lower())
    runs = [
        syntactic_eval.load_run(run_path(corpus, config))
        for corpus in CORPORA
        for config in CONFIGS
    ]
    tokenizable = {
        w for w in classes if all(syntactic_eval.single_token(r, w) is not None for r in runs)
    }
    prefixes = syntactic_eval.contexts(evaluation, heads)
    rows = {}
    for kind, members in CLASSES.items():
        valid, contrast = valid_and_contrast(kind, classes, forms)
        for head in members:
            unseen = tokenizable - followers[head]
            pairs = syntactic_eval.pair_by_frequency(
                sorted(valid & unseen), sorted(contrast & unseen), frequency, seed=0
            )
            rows[head] = {"kind": kind, "pairs": pairs, "contexts": prefixes[head]}
            print(f"{kind} {head}: {len(pairs)} pairs, {len(prefixes[head])} contexts; {pairs[:3]}")
    divergence.write_json(RESULTS / "items.json", rows)


def summarize() -> None:
    item_rows = json.loads((RESULTS / "items.json").read_text())
    g: dict[str, dict[str, dict[str, float]]] = {}
    for corpus in CORPORA:
        g[corpus] = {}
        for config in CONFIGS:
            divergence.progress(f"Scoring {corpus} {config}")
            run = syntactic_eval.load_run(run_path(corpus, config))
            g[corpus][config] = {
                head: float(syntactic_eval.head_reading(run, item)["g"])
                for head, item in item_rows.items()
            }

    def class_g(corpus: str, config: str, kind: str) -> float:
        return statistics.mean(g[corpus][config][h] for h in CLASSES[kind])

    def best(corpus: str, kind: str) -> float:
        return max(class_g(corpus, config, kind) for config in CONFIGS)

    print("class G per corpus (small / medium / long)")
    table: dict[str, dict[str, dict[str, float]]] = {}
    for kind in CLASSES:
        table[kind] = {}
        for corpus in CORPORA:
            values = {config: class_g(corpus, config, kind) for config in CONFIGS}
            table[kind][corpus] = values
            print(
                f"  {kind:10} {corpus:15} "
                + " / ".join(f"{values[c]:.3f}" for c in CONFIGS)
                + f"  best {max(values.values()):.3f}"
            )
    held = {"1": True, "2": True, "3": True}
    for kind, (control, cut) in PAIRS.items():
        full = best("full", kind)
        held["1"] &= best(cut, kind) <= full - 0.10
        held["2"] &= best(control, kind) >= full - 0.05
        gaps = {c: class_g(control, c, kind) - class_g(cut, c, kind) for c in CONFIGS}
        held["3"] &= all(gap >= 0.05 for gap in gaps.values())
        print(
            f"{kind}: best F {full:.3f}, best control {best(control, kind):.3f}, best cut "
            f"{best(cut, kind):.3f}; control - cut per config "
            + ", ".join(f"{c} {v:+.3f}" for c, v in gaps.items())
            + f"; best r10 {best('r10', kind):.3f}"
        )
    predictions = {k: syntactic_eval.status(v) for k, v in held.items()}
    predictions["4"] = "exploratory"
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(
        RESULTS / "summary.json", {"g": g, "class_g": table, "predictions": predictions}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("corpus", choices=CORPORA)
    training.add_argument("config", choices=("medium", "long"))
    commands.add_parser("items")
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "train":
            train_corpus(args.corpus, args.config)
        case "items":
            items()
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
