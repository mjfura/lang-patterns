"""Scale: more data and longer training, on one CPU night.

"Scale: more data and longer training" in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` records the design and
predictions before training.

    scale.py prepare          BigF (FineWeb's first 20,000 documents minus the 400 evaluation
                              documents), big_class_cut and big_control
    scale.py train <name>     name: bigf | big_class_cut | big_control (about 5 hours each)
    scale.py items            G items for pronouns and modals, BigF as the reference
    scale.py summarize        BLiMP and G; the three predictions

`runs/scale/fineweb_20000.json` holds the streamed documents (gitignored, like the corpora and
runs under `runs/scale/`). Train detached (nohup): a run outlives a two-hour background task.
No command runs on import.
"""

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "scale"
CORPORA = RUNS / "corpora"
DOWNLOAD = RUNS / "fineweb_20000.json"
EXPERIMENTS = OUT.parent
sys.path[:0] = [
    str(EXPERIMENTS / "blimp"),
    str(EXPERIMENTS / "model_sweep"),
    str(EXPERIMENTS / "class_patterns"),
]

import blimp  # noqa: E402
import class_patterns  # noqa: E402
import model_sweep  # noqa: E402
import planted_model  # noqa: E402  (on the path through class_patterns)
import replication  # noqa: E402
from class_patterns import divergence, natural_gap, subset_selection, syntactic_eval  # noqa: E402

NAMES = ("bigf", "big_class_cut", "big_control")
SETTINGS = replace(
    planted_model.SETTINGS,
    n_layer=4,
    d_model=256,
    learning_rate=5e-4,
    steps=30_000,
    eval_every=500,
)
CLASSES = {"pronoun": model_sweep.CLASSES["pronoun"], "modal": model_sweep.CLASSES["modal"]}


def big_f() -> list[str]:
    evaluation, _ = natural_gap.split()
    held = set(evaluation)
    return [d for d in json.loads(DOWNLOAD.read_text()) if d not in held]


def corpus(name: str) -> list[str]:
    return big_f() if name == "bigf" else json.loads((CORPORA / f"{name}.json").read_text())


def prepare() -> None:
    documents = big_f()
    segmented = class_patterns.segment(documents)
    every = {
        (d, i) for d, sentences in enumerate(segmented) for i, (_, ts) in enumerate(sentences) if ts
    }
    class_cut = class_patterns.assemble(segmented, every)
    target = class_patterns.size(class_cut)
    corpora = {
        "big_class_cut": class_cut,
        "big_control": class_patterns.assemble(
            segmented, class_patterns.match_bytes(segmented, set(), target, 0)
        ),
    }
    CORPORA.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "bigf": {"documents": len(documents), "bytes": class_patterns.size(documents)},
        "sentences": sum(len(s) for s in segmented),
        "target_sentences": len(every),
    }
    for name, texts in corpora.items():
        (CORPORA / f"{name}.json").write_text(json.dumps(texts))
        record[name] = {"documents": len(texts), "bytes": class_patterns.size(texts)}
    for name, row in record.items():
        print(name, row)
    divergence.write_json(RESULTS / "corpora.json", record)


def run_path(name: str) -> Path:
    return RUNS / f"{name}_seed0"


def train_corpus(name: str) -> None:
    subset_selection.train_and_score(
        corpus(name), name, 0, run_path(name), RESULTS, "scale", SETTINGS
    )


def items() -> None:
    evaluation, _ = natural_gap.split()
    reference = big_f()
    classes, forms, frequency = replication.fine_tags(reference)
    heads = tuple(h for members in CLASSES.values() for h in members)
    followers: dict[str, set[str]] = {head: set() for head in heads}
    for document in reference:
        for match in syntactic_eval.follower_pattern(heads).finditer(document):
            followers[match.group(1)].add(match.group(2).lower())
    runs = [syntactic_eval.load_run(run_path(name)) for name in NAMES]
    tokenizable = {
        w for w in classes if all(syntactic_eval.single_token(r, w) is not None for r in runs)
    }
    prefixes = syntactic_eval.contexts(evaluation, heads)
    rows = {}
    for kind, members in CLASSES.items():
        valid, contrast = model_sweep.valid_and_contrast(kind, classes, forms)
        for head in members:
            unseen = tokenizable - followers[head]
            pairs = syntactic_eval.pair_by_frequency(
                sorted(valid & unseen), sorted(contrast & unseen), frequency, seed=0
            )
            rows[head] = {"kind": kind, "pairs": pairs, "contexts": prefixes[head]}
            print(f"{kind} {head}: {len(pairs)} pairs, {len(prefixes[head])} contexts")
    divergence.write_json(RESULTS / "items.json", rows)


def summarize() -> None:
    item_rows = json.loads((RESULTS / "items.json").read_text())
    pairs = blimp.pairs()
    good = [p["good"] for p in pairs]
    bad = [p["bad"] for p in pairs]
    g: dict[str, dict[str, float]] = {}
    blimp_overall: dict[str, float] = {}
    for name in NAMES:
        divergence.progress(f"Scoring {name}")
        run = syntactic_eval.load_run(run_path(name))
        g[name] = {
            kind: statistics.mean(
                float(syntactic_eval.head_reading(run, item_rows[h])["g"]) for h in members
            )
            for kind, members in CLASSES.items()
        }
        correct = [
            x > y
            for x, y in zip(
                blimp.log_probabilities(run, good), blimp.log_probabilities(run, bad), strict=True
            )
        ]
        by_paradigm: dict[str, list[bool]] = {}
        for pair, ok in zip(pairs, correct, strict=True):
            by_paradigm.setdefault(pair["paradigm"], []).append(ok)
        blimp_overall[name] = statistics.mean(sum(v) / len(v) for v in by_paradigm.values())
        metrics = run.metrics
        print(
            f"{name}: BLiMP {blimp_overall[name]:.4f}; G {g[name]}; best step "
            f"{metrics['best_step']}, held-out loss {metrics['best_heldout_loss']:.4f}"
        )
    held = {
        "1": blimp_overall["bigf"] >= 0.60,
        "2": all(g["big_control"][k] - g["big_class_cut"][k] >= 0.10 for k in CLASSES),
        "3": all(g["big_control"][k] >= g["bigf"][k] - 0.05 for k in CLASSES),
    }
    predictions = {k: "HELD" if v else "FAILED" for k, v in held.items()}
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(
        RESULTS / "summary.json", {"g": g, "blimp": blimp_overall, "predictions": predictions}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    training = commands.add_parser("train")
    training.add_argument("name", choices=NAMES)
    commands.add_parser("items")
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare()
        case "train":
            train_corpus(args.name)
        case "items":
            items()
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
