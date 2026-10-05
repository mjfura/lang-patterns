"""Choose a training subset against the full corpus as reference.

"Choosing a training subset: the full corpus as reference" in
`docs/2026-09-28-syntactic-sufficiency-design.md` records the predictions before running.

    subset_selection.py prepare                 split and select all six corpora by bytes
    subset_selection.py diagnose <name>         name: r50 | r25 | r10 | wiki25 | self25
    subset_selection.py train <name> <seed>     also accepts full; requires all five diagnoses
    subset_selection.py summarize              compare every complete six-model seed

FineWeb and Wikipedia caches come from `experiments/clean_domains`. The evaluation split and
model settings match `natural_gap` and `planted_model`. Runs go to `runs/subset_selection/`
(gitignored); results to `results/` here, created at runtime. Train seeds 0 and 1, two runs at a
time with `THREADS=3`; `THREADS` sets torch's thread count. No command runs on import.
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

import torch
from scipy.stats import spearmanr

from corpus_generalization.diagnose import VERDICTS, diagnose
from corpus_generalization.model.generate import LoadedRun, load_run
from corpus_generalization.model.train import TrainingSettings, train

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "subset_selection"
sys.path[:0] = [
    str(OUT.parent / "natural_gap"),
    str(OUT.parent / "planted_model"),
    str(OUT.parent / "planted_gap"),
    str(OUT.parent / "natural_sufficiency"),
    str(OUT.parent / "clean_domains"),
]

import clean_domains  # noqa: E402
import natural_gap  # noqa: E402
import planted_model  # noqa: E402

SUBSETS = ("r50", "r25", "r10", "wiki25", "self25")
NAMES = ("full", *SUBSETS)
FILTERED = ("wiki25", "self25")
SEEDS = (0, 1)
MIN_OCCURRENCES = 20
MAX_OCCURRENCES = 50


class Subset(TypedDict):
    indices: list[int]
    documents: int
    bytes: int


class FilteredSubset(Subset):
    mean_perplexity_chosen: float
    mean_perplexity_eligible: float


class Row(TypedDict):
    head: str
    head_length: int
    trunk: int
    diversity: float
    margin_mean: float | None
    margin_sd: float | None
    verdict: str


class Diagnosis(TypedDict):
    rows: list[Row]


class Occurrence(TypedDict):
    head: str
    word: str
    position: list[int]
    log_probability: float


class Run(TypedDict):
    corpus: str
    seed: int
    best_step: int
    eval_bits_per_byte: float
    occurrences: list[Occurrence]


@dataclass
class Head:
    flagged: bool = False
    z: float | None = None
    trunk: int = 0


class Bigram:
    def __init__(self, documents: list[str]) -> None:
        self.words: Counter[str] = Counter()
        self.contexts: Counter[str] = Counter()
        self.pairs: Counter[tuple[str, str]] = Counter()
        for document in documents:
            previous = "<s>"
            for word in words(document):
                self.words[word] += 1
                self.contexts[previous] += 1
                self.pairs[previous, word] += 1
                previous = word
        self.denominator = sum(self.words.values()) + len(self.words) + 1

    def perplexity(self, tokens: list[str]) -> float:
        if not tokens:
            raise ValueError("Perplexity requires at least one word.")
        previous, total = "<s>", 0.0
        for word in tokens:
            unigram = (self.words[word] + 1) / self.denominator
            count = self.contexts[previous]
            probability = (
                0.7 * self.pairs[previous, word] / count + 0.3 * unigram if count else unigram
            )
            total -= math.log(probability)
            previous = word
        return math.exp(total / len(tokens))


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fill(sizes: list[int], order: list[int], target: float) -> list[int]:
    chosen, size = [], 0
    for index in order:
        if size >= target:
            break
        chosen.append(index)
        size += sizes[index]
    if size < target:
        raise ValueError(f"Eligible documents contain only {size} bytes; need {target:g}.")
    return chosen


def prepare() -> None:
    _, pool = natural_gap.split()
    sizes = [len(document.encode("utf-8")) for document in pool]
    total = sum(sizes)
    if not total:
        raise ValueError("The training pool is empty.")
    chosen = {"full": list(range(len(pool)))}
    for name, fraction, seed in (("r50", 0.5, 200), ("r25", 0.25, 201), ("r10", 0.1, 202)):
        order = list(range(len(pool)))
        random.Random(seed).shuffle(order)
        chosen[name] = fill(sizes, order, fraction * total)
    subsets: dict[str, Subset] = {
        name: {
            "indices": indices,
            "documents": len(indices),
            "bytes": sum(sizes[i] for i in indices),
        }
        for name, indices in chosen.items()
    }
    eligible = {i: tokens for i, doc in enumerate(pool) if len(tokens := words(doc)) >= 50}
    for name in FILTERED:
        documents = (
            json.loads(clean_domains.cache_of("wikipedia").read_text())
            if name == "wiki25"
            else pool
        )
        model = Bigram(documents)
        perplexities = {i: model.perplexity(tokens) for i, tokens in eligible.items()}
        order = sorted(eligible, key=lambda i: (perplexities[i], i))
        indices = fill(sizes, order, 0.25 * total)
        filtered: FilteredSubset = {
            "indices": indices,
            "documents": len(indices),
            "bytes": sum(sizes[i] for i in indices),
            "mean_perplexity_chosen": statistics.mean(perplexities[i] for i in indices),
            "mean_perplexity_eligible": statistics.mean(perplexities.values()),
        }
        subsets[name] = filtered
    write_json(RESULTS / "subsets.json", subsets)
    for name, subset in subsets.items():
        detail = ""
        if name in FILTERED:
            filtered = cast(FilteredSubset, subset)
            detail = (
                f"; mean perplexity chosen {filtered['mean_perplexity_chosen']:.3f}"
                f", eligible {filtered['mean_perplexity_eligible']:.3f}"
            )
        print(f"{name}: {subset['documents']} documents, {subset['bytes']:,} bytes{detail}")


def selected(pool: list[str], name: str) -> list[str]:
    subsets: dict[str, Subset] = json.loads((RESULTS / "subsets.json").read_text())
    return [pool[i] for i in subsets[name]["indices"]]


def progress(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def diagnose_subset(name: str) -> None:
    _, pool = natural_gap.split()
    report = diagnose(selected(pool, name), reference=pool, merge_budget=11563, progress=progress)
    write_json(RESULTS / f"diagnosis_{name}.json", report.to_json())
    print(report.summary())


def diagnoses() -> dict[str, Diagnosis]:
    paths = {name: RESULTS / f"diagnosis_{name}.json" for name in SUBSETS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run diagnose for all five subsets first; missing: {', '.join(missing)}"
        )
    return {name: json.loads(path.read_text()) for name, path in paths.items()}


def head_readings(report: Diagnosis) -> dict[str, Head]:
    heads: dict[str, Head] = {}
    for row in report["rows"]:
        head = row["head"].strip()
        if row["head_length"] != 1 or re.fullmatch(r"[A-Za-z]+", head) is None:
            continue
        reading = heads.setdefault(head, Head())
        reading.flagged |= row["verdict"] == "under-shown"
        mean, sd = row["margin_mean"], row["margin_sd"]
        if mean is not None and sd is not None and sd > 0:
            z = (row["diversity"] - mean) / sd
            if reading.z is None or z < reading.z:
                reading.z, reading.trunk = z, row["trunk"]
    return heads


def occurrence_positions(
    documents: list[str], heads: list[str]
) -> list[tuple[str, str, list[int]]]:
    if not heads:
        return []
    pattern = re.compile(
        r"(?<![A-Za-z])(" + "|".join(map(re.escape, heads)) + r") ([A-Za-z]+)(?![A-Za-z])"
    )
    counts: Counter[str] = Counter()
    positions = []
    for index, document in enumerate(documents):
        for match in pattern.finditer(document):
            head, word = match.group(1), match.group(2)
            if counts[head] < MAX_OCCURRENCES:
                positions.append((head, word, [index, match.start()]))
                counts[head] += 1
    return positions


def check_scoreable(
    loaded: LoadedRun,
    documents: list[str],
    positions: list[tuple[str, str, list[int]]],
) -> None:
    # A skipped match could be replaced by an identical head/word after the cap. Check the
    # scorer's tokenization and prefix-shortening rules before assigning physical positions.
    for head, word, (index, offset) in positions:
        end = offset + len(head)
        start = max(0, end - planted_model.PREFIX_CHARS)
        prefix = documents[index][start:end]
        if start > 0:
            prefix = prefix[prefix.find(" ") + 1 :]
        while True:
            prefix_ids = [
                loaded.tokenizer.bos_id,
                *loaded.tokenizer.encode(prefix, recognize_markers=True),
            ]
            full_ids = [
                loaded.tokenizer.bos_id,
                *loaded.tokenizer.encode(prefix + " " + word, recognize_markers=True),
            ]
            if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) == len(prefix_ids):
                raise ValueError(
                    f"Unscorable token boundary at evaluation position {[index, offset]}."
                )
            if len(full_ids) <= loaded.model.context:
                break
            if " " not in prefix:
                raise ValueError(f"Unscorable context at evaluation position {[index, offset]}.")
            prefix = prefix[prefix.find(" ") + 1 :]


def train_corpus(name: str, seed: int) -> None:
    _, pool = natural_gap.split()
    train_and_score(
        selected(pool, name), name, seed, RUNS / f"{name}_seed{seed}", RESULTS, "subset_selection"
    )


def train_and_score(
    documents: list[str],
    name: str,
    seed: int,
    output: Path,
    results: Path,
    source: str,
    settings: TrainingSettings | None = None,
) -> None:
    """Train on `documents` and score this experiment's evaluation occurrences, so that other
    experiments' cuts read drops against the same `full` models."""
    reports = diagnoses()
    heads = sorted({head for report in reports.values() for head in head_readings(report)})
    if threads := int(os.environ.get("THREADS", "0")):
        torch.set_num_threads(threads)
    evaluation, _ = natural_gap.split()
    train(
        documents,
        replace(settings or planted_model.SETTINGS, seed=seed),
        output,
        sources={source: name},
    )
    loaded = load_run(output)
    planted_model.MAX_OCCURRENCES = MAX_OCCURRENCES
    positions = occurrence_positions(evaluation, heads)
    check_scoreable(loaded, evaluation, positions)
    scores = planted_model.occurrence_scores(loaded, evaluation, heads) if heads else []
    assert len(scores) == len(positions), "The scorer skipped evaluation occurrences."
    occurrences: list[Occurrence] = []
    for row, (head, word, position) in zip(scores, positions, strict=True):
        assert (row["head"], row["word"]) == (head, word), "Evaluation occurrence order changed."
        occurrences.append(
            {
                "head": head,
                "word": word,
                "position": position,
                "log_probability": float(cast(float, row["log_probability"])),
            }
        )
    result = {
        "corpus": name,
        "seed": seed,
        "best_step": loaded.metrics["best_step"],
        "eval_bits_per_byte": planted_model.heldout_bits_per_byte(loaded, evaluation),
        "occurrences": occurrences,
    }
    write_json(results / f"{name}_seed{seed}.json", result)
    print(name, seed, result["eval_bits_per_byte"], len(occurrences))


def drops(rows: list[Occurrence], full: list[Occurrence]) -> dict[str, float]:
    assert len(rows) == len(full), "Models scored different numbers of occurrences."
    by_head: dict[str, list[float]] = defaultdict(list)
    for row, reference in zip(rows, full, strict=True):
        assert (row["head"], row["word"], row["position"]) == (
            reference["head"],
            reference["word"],
            reference["position"],
        ), "Models scored different occurrence sequences."
        by_head[row["head"]].append(row["log_probability"] - reference["log_probability"])
    return {
        head: statistics.mean(values)
        for head, values in by_head.items()
        if len(values) >= MIN_OCCURRENCES
    }


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rho = float(spearmanr(xs, ys).statistic)
    return rho if math.isfinite(rho) else None


def status(holds: bool | None) -> str:
    return "unreadable" if holds is None else "HELD" if holds else "FAILED"


def both_seeds(values: dict[int, bool | None]) -> str:
    required = [values.get(seed) for seed in SEEDS]
    return status(None if any(value is None for value in required) else all(required))


def summarize() -> None:
    reports = diagnoses()
    readings = {name: head_readings(report) for name, report in reports.items()}
    diagnosis_summary: dict[str, object] = {}
    shares: dict[str, float | None] = {}
    under_shown: dict[str, int] = {}
    for name, report in reports.items():
        counts = Counter(row["verdict"] for row in report["rows"])
        rows = len(report["rows"])
        shares[name] = counts["under-shown"] / rows if rows else None
        under_shown[name] = counts["under-shown"]
        diagnosis_summary[name] = {
            "rows_read": rows,
            "counts": {v: counts[v] for v in VERDICTS},
            "under_shown_share": shares[name],
        }
        print(f"{name}: {json.dumps(diagnosis_summary[name])}")
    p1 = {
        name: status(share <= 0.01 if (share := shares[name]) is not None else None)
        for name in ("r50", "r25", "r10")
    }
    p2 = {name: status(under_shown[name] >= max(10, 3 * under_shown["r25"])) for name in FILTERED}
    print(f"Prediction 1: {p1}; shares {shares}; threshold <= 0.01")
    print(f"Prediction 2: {p2}; under-shown {under_shown}; threshold >= max(10, 3 * r25)")
    candidates = sorted(
        {
            int(match.group(1))
            for path in RESULTS.glob("full_seed*.json")
            if (match := re.fullmatch(r"full_seed(-?\d+)\.json", path.name))
        }
    )
    seeds = [
        seed
        for seed in candidates
        if all((RESULTS / f"{name}_seed{seed}.json").is_file() for name in NAMES)
    ]
    p3: dict[int, bool | None] = {}
    p4: dict[str, dict[int, bool | None]] = {name: {} for name in FILTERED}
    p5: dict[int, bool | None] = {}
    per_seed: dict[str, object] = {}
    for seed in seeds:
        runs: dict[str, Run] = {
            name: json.loads((RESULTS / f"{name}_seed{seed}.json").read_text()) for name in NAMES
        }
        bpb = {name: run["eval_bits_per_byte"] for name, run in runs.items()}
        per_head = {
            name: drops(runs[name]["occurrences"], runs["full"]["occurrences"]) for name in SUBSETS
        }
        p3[seed] = bpb["full"] < bpb["r50"] < bpb["r25"] < bpb["r10"] and all(
            bpb[name] > bpb["r25"] for name in FILTERED
        )
        print(f"Prediction 3 seed {seed}: {status(p3[seed])}; bits per byte {bpb}")
        for head in sorted({head for values in per_head.values() for head in values}):
            print(
                f"seed {seed} head {head}: drop "
                + json.dumps(
                    {name: values[head] for name, values in per_head.items() if head in values}
                )
            )
        gaps: dict[str, object] = {}
        for name in FILTERED:
            eligible = sorted(set(readings[name]) & per_head[name].keys() & per_head["r25"].keys())
            flagged = [head for head in eligible if readings[name][head].flagged]
            unflagged = [head for head in eligible if not readings[name][head].flagged]
            gap_filtered = gap_random = difference = None
            if len(flagged) >= 5 and unflagged:
                gap_filtered = statistics.mean(
                    per_head[name][h] for h in flagged
                ) - statistics.mean(per_head[name][h] for h in unflagged)
                gap_random = statistics.mean(per_head["r25"][h] for h in flagged) - statistics.mean(
                    per_head["r25"][h] for h in unflagged
                )
                difference = gap_filtered - gap_random
            p4[name][seed] = difference <= -0.3 if difference is not None else None
            gaps[name] = {
                "status": status(p4[name][seed]),
                "flagged_heads": flagged,
                "unflagged_heads": unflagged,
                "gap_filtered": gap_filtered,
                "gap_r25": gap_random,
                "difference": difference,
            }
            print(
                f"Prediction 4 {name} seed {seed}: {status(p4[name][seed])}; "
                f"flagged {len(flagged)}, unflagged {len(unflagged)}, "
                f"gap(filtered) {gap_filtered}, gap(r25) {gap_random}, difference {difference}"
            )
        heads = sorted(
            h for h in per_head["r10"] if h in readings["r10"] and readings["r10"][h].z is not None
        )
        values = [per_head["r10"][h] for h in heads]
        rho = correlation([cast(float, readings["r10"][h].z) for h in heads], values)
        trunk_rho = correlation([math.log(readings["r10"][h].trunk) for h in heads], values)
        p5[seed] = rho >= 0.15 if rho is not None else None
        print(
            f"Prediction 5 seed {seed}: {status(p5[seed])}; heads {len(heads)}, "
            f"Spearman(z, drop) {rho}, Spearman(log trunk, drop) {trunk_rho}"
        )
        per_seed[str(seed)] = {
            "eval_bits_per_byte": bpb,
            "drops": per_head,
            "occurrence_counts": dict(Counter(row["head"] for row in runs["full"]["occurrences"])),
            "prediction_3": status(p3[seed]),
            "prediction_4": gaps,
            "prediction_5": {
                "status": status(p5[seed]),
                "heads": heads,
                "spearman_z_drop": rho,
                "spearman_log_trunk_drop": trunk_rho,
            },
        }
    predictions = {
        "1": p1,
        "2": p2,
        "3": both_seeds(p3),
        "4": {name: both_seeds(p4[name]) for name in FILTERED},
        "5": both_seeds(p5),
    }
    print(f"Predictions (both seeds {SEEDS} required for 3-5): {json.dumps(predictions)}")
    write_json(
        RESULTS / "summary.json",
        {
            "diagnoses": diagnosis_summary,
            "heads": {
                name: {
                    h: {"flagged": row.flagged, "z": row.z, "trunk": row.trunk}
                    for h, row in heads.items()
                }
                for name, heads in readings.items()
            },
            "required_seeds": SEEDS,
            "complete_seeds": seeds,
            "seeds": per_seed,
            "predictions": predictions,
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Select all six corpora by bytes")
    diagnosis = commands.add_parser("diagnose", help="Diagnose one subset against the full pool")
    diagnosis.add_argument("name", choices=SUBSETS)
    training = commands.add_parser("train", help="Train and score one corpus and seed")
    training.add_argument("name", choices=NAMES)
    training.add_argument("seed", type=int)
    commands.add_parser("summarize", help="Compare complete six-model seeds")
    args = parser.parse_args(argv)
    match args.command:
        case "prepare":
            prepare()
        case "diagnose":
            diagnose_subset(args.name)
        case "train":
            train_corpus(args.name, args.seed)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
