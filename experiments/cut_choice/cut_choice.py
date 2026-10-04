"""Choosing among cuts of one size: does the score order the models?

"Choosing among cuts of one size" in `docs/2026-09-28-syntactic-sufficiency-design.md` records
the design and predictions before running.

    cut_choice.py prepare              select the seven new 25% cuts of F
    cut_choice.py train <name> <seed>  train and score one new cut, as subset_selection does
    cut_choice.py score                the divergence score and E per head for all ten cuts
    cut_choice.py summarize            check the five predictions

The three cuts of `subset_selection` (r25, wiki25, self25) keep their models; drops are read
against its `full` models on its evaluation occurrences. Runs go to `runs/cut_choice/`
(gitignored), results to `results/` here. Train two runs at a time with `THREADS=3`.
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "cut_choice"
EXPERIMENTS = OUT.parent
sys.path[:0] = [
    str(EXPERIMENTS / "divergence"),
    str(EXPERIMENTS / "subset_selection"),
    str(EXPERIMENTS / "natural_gap"),
    str(EXPERIMENTS / "planted_model"),
    str(EXPERIMENTS / "planted_gap"),
    str(EXPERIMENTS / "natural_sufficiency"),
    str(EXPERIMENTS / "clean_domains"),
]

import clean_domains  # noqa: E402
import divergence  # noqa: E402
import natural_gap  # noqa: E402
import subset_selection  # noqa: E402

NEW = ("r25b", "r25c", "selfhigh25", "wikihigh25", "short25", "long25", "mix25")
OLD = ("r25", "wiki25", "self25")
CUTS = (*OLD, *NEW)
RANDOM = ("r25", "r25b", "r25c")
SEEDS = (0, 1)
FRACTION = 0.25


def prepare() -> None:
    _, pool = natural_gap.split()
    sizes = [len(document.encode("utf-8")) for document in pool]
    target = FRACTION * sum(sizes)
    cuts: dict[str, list[int]] = {}
    for name, seed in (("r25b", 203), ("r25c", 204)):
        order = list(range(len(pool)))
        random.Random(seed).shuffle(order)
        cuts[name] = subset_selection.fill(sizes, order, target)
    eligible = {
        i: tokens for i, doc in enumerate(pool) if len(tokens := subset_selection.words(doc)) >= 50
    }
    by_source = {
        "self": subset_selection.Bigram(pool),
        "wiki": subset_selection.Bigram(
            json.loads(clean_domains.cache_of("wikipedia").read_text())
        ),
    }
    ascending: dict[str, list[int]] = {}
    for source, model in by_source.items():
        perplexities = {i: model.perplexity(tokens) for i, tokens in eligible.items()}
        ascending[source] = sorted(eligible, key=lambda i: (perplexities[i], i))
        cuts[f"{source}high25"] = subset_selection.fill(
            sizes, sorted(eligible, key=lambda i: (-perplexities[i], i)), target
        )
    cuts["short25"] = subset_selection.fill(
        sizes, sorted(range(len(pool)), key=lambda i: (sizes[i], i)), target
    )
    cuts["long25"] = subset_selection.fill(
        sizes, sorted(range(len(pool)), key=lambda i: (-sizes[i], i)), target
    )
    half = subset_selection.fill(sizes, ascending["self"], target / 2)
    taken = set(half)
    rest = [i for i in range(len(pool)) if i not in taken]
    random.Random(205).shuffle(rest)
    cuts["mix25"] = half + subset_selection.fill(sizes, rest, target - sum(sizes[i] for i in half))
    rows = {
        name: {
            "indices": indices,
            "documents": len(indices),
            "bytes": sum(sizes[i] for i in indices),
        }
        for name, indices in cuts.items()
    }
    subset_selection.write_json(RESULTS / "cuts.json", rows)
    for name, row in rows.items():
        print(f"{name}: {row['documents']} documents, {row['bytes']:,} bytes")


def indices() -> dict[str, list[int]]:
    new = json.loads((RESULTS / "cuts.json").read_text())
    old = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    return {name: (new if name in NEW else old)[name]["indices"] for name in CUTS}


def train_cut(name: str, seed: int) -> None:
    _, pool = natural_gap.split()
    documents = [pool[i] for i in indices()[name]]
    subset_selection.train_and_score(
        documents, name, seed, RUNS / f"{name}_seed{seed}", RESULTS, "cut_choice"
    )


def score() -> None:
    reading = divergence.read_full()
    baseline = divergence.load_baseline()
    means, spread = baseline.means["r25"], baseline.spread["r25"]
    rows: dict[str, object] = {}
    for name, chosen in indices().items():
        divergence.progress(f"Reading {name}")
        heads = reading.read([reading.full[i] for i in chosen])
        excess = {h: heads[h]["divergence"] - means[h] for h in baseline.shares}
        value = divergence.weighted(baseline.shares, excess)
        rows[name] = {
            "score": value,
            "z": value / spread,
            "documents": len(chosen),
            "excess": excess,
        }
        print(f"{name:11} score {value:+.4f}  z {value / spread:+.2f}  documents {len(chosen)}")
    subset_selection.write_json(RESULTS / "scores.json", rows)


def run_result(name: str, seed: int) -> dict[str, object]:
    folder = RESULTS if name in NEW else subset_selection.RESULTS
    result: dict[str, object] = json.loads((folder / f"{name}_seed{seed}.json").read_text())
    return result


def spearman(xs: list[float], ys: list[float]) -> float:
    return float(spearmanr(xs, ys).statistic)


def status(holds: bool) -> str:
    return "HELD" if holds else "FAILED"


def summarize() -> None:
    scores = json.loads((RESULTS / "scores.json").read_text())
    report: dict[str, object] = {}
    checks: dict[str, list[bool]] = {str(i): [] for i in range(1, 5)}
    for seed in SEEDS:
        runs = {name: run_result(name, seed) for name in CUTS}
        full = run_result("full", seed)
        bpb = {name: float(runs[name]["eval_bits_per_byte"]) for name in CUTS}  # type: ignore[arg-type]
        drops = {
            name: subset_selection.drops(runs[name]["occurrences"], full["occurrences"])  # type: ignore[arg-type]
            for name in CUTS
        }
        print(f"\nseed {seed}")
        for name in sorted(CUTS, key=lambda n: scores[n]["score"]):
            print(
                f"  {name:11} score {scores[name]['score']:+.4f}  z {scores[name]['z']:+6.2f}  "
                f"documents {scores[name]['documents']:5}  bits per byte {bpb[name]:.4f}"
            )
        values = [scores[name]["score"] for name in CUTS]
        losses = [bpb[name] for name in CUTS]
        rho = spearman(values, losses)
        checks["1"].append(rho >= 0.5)
        dose = ("r25", "mix25", "self25")
        checks["2"].append(
            scores["r25"]["score"] < scores["mix25"]["score"] < scores["self25"]["score"]
            and bpb["r25"] < bpb["mix25"] < bpb["self25"]
        )
        span = max(bpb[n] for n in RANDOM) - min(bpb[n] for n in RANDOM)
        checks["3"].append(all(abs(scores[n]["z"]) <= 2 for n in ("r25b", "r25c")) and span <= 0.02)
        random_drop = {
            h: statistics.mean(drops[n][h] for n in RANDOM)
            for h in set.intersection(*(set(drops[n]) for n in RANDOM))
        }
        per_head: dict[str, dict[str, float | int]] = {}
        for name in NEW:
            if name in RANDOM or scores[name]["z"] < 3:
                continue
            heads = sorted(h for h in drops[name].keys() & random_drop.keys())
            r = spearman(
                [scores[name]["excess"][h] for h in heads],
                [drops[name][h] - random_drop[h] for h in heads],
            )
            per_head[name] = {"heads": len(heads), "spearman": r}
            checks["4"].append(r <= -0.2)
        log_documents = [math.log(scores[name]["documents"]) for name in CUTS]
        rho_documents = spearman(log_documents, losses)
        slope, intercept = np.polyfit(log_documents, losses, 1)
        left = [y - (slope * x + intercept) for x, y in zip(log_documents, losses, strict=True)]
        rho_left = spearman(values, left)
        print(
            f"  P1 Spearman(score, bpb) {rho:.3f}; P2 dose {[bpb[n] for n in dose]}; "
            f"P3 random span {span:.4f}\n  P4 {per_head}\n"
            f"  P5 Spearman(log documents, bpb) {rho_documents:.3f}, "
            f"Spearman(score, bpb left after documents) {rho_left:.3f}"
        )
        report[str(seed)] = {
            "bits_per_byte": bpb,
            "spearman_score_bpb": rho,
            "random_span": span,
            "per_head": per_head,
            "spearman_log_documents_bpb": rho_documents,
            "spearman_score_residual": rho_left,
        }
    predictions = {
        key: status(all(values)) if values else "unreadable" for key, values in checks.items()
    }
    predictions["5"] = "exploratory"
    print(f"\nPredictions: {json.dumps(predictions)}")
    subset_selection.write_json(
        RESULTS / "summary.json",
        {
            "scores": {n: {k: v for k, v in scores[n].items() if k != "excess"} for n in CUTS},
            "seeds": report,
            "predictions": predictions,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    training = commands.add_parser("train")
    training.add_argument("name", choices=NEW)
    training.add_argument("seed", type=int)
    commands.add_parser("score")
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare()
        case "train":
            train_cut(args.name, args.seed)
        case "score":
            score()
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
