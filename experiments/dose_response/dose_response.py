"""Dose and response: how far must diversity fall before a model pays?

See the section of that name in docs/2026-09-28-syntactic-sufficiency-design.md.
FineWeb slices 0-6 train; slice 7 evaluates. Controls reuse planted_model runs.

    dose_response.py prepare
    dose_response.py train <A|B> <seed>
    dose_response.py rescore-control <seed>
    dose_response.py summarize

THREADS sets torch's thread count. Results live beside this script in results/.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from dataclasses import replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import TypedDict, cast

import torch
from scipy.stats import spearmanr

from corpus_generalization.analysis.syntactic import (
    Pattern,
    induce_classes,
    syntactic_patterns,
    word_like_tokens,
)
from corpus_generalization.model.generate import load_run
from corpus_generalization.model.train import train
from corpus_generalization.packing import framed_documents
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "dose_response"
sys.path[:0] = [str(OUT.parent / "planted_model"), str(OUT.parent / "planted_gap")]

import planted  # noqa: E402
import planted_model  # noqa: E402

LEVELS = tuple(i / 10 for i in range(1, 8))
ASSIGNMENTS = {"A": 0, "B": 3}
SEEDS = (0, 1)
PLANT_SEED = 0
TOLERANCE = 0.03
ITERATIONS = 12
# The 6.5 MB sample has 17 heads that pass planted_gap's rule; the design asked for 26 (revised
# before any run): two planted heads per level, and the remaining three as comparison heads.
PLANTED = 14
COMPARISON = 3
Key = tuple[tuple[int, ...], int]


class Calibration(TypedDict):
    target: float
    q: float
    diversity: float | None
    trunk: int


class Prepared(TypedDict):
    plant_seed: int
    planted: dict[str, list[str]]
    comparison: dict[str, list[str]]
    calibration: dict[str, dict[str, Calibration]]
    indicator_diversity: dict[str, dict[str, float | None]]


class Occurrence(TypedDict):
    head: str
    word: str
    log_probability: float


class RunResult(TypedDict):
    variant: str
    seed: int
    run: str
    reused_run: bool
    best_step: int
    best_heldout_loss: float
    eval_bits_per_byte: float
    occurrences: list[Occurrence]


def progress(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plant_partial(
    documents: list[str], kept: dict[str, list[str]], q: dict[str, float], seed: int
) -> list[str]:
    """Replace eligible followers using q-independent draws at original word positions."""
    if any(not fillers for fillers in kept.values()):
        raise ValueError("Each planted head needs kept fillers")
    if any(head not in q or not 0 <= q[head] <= 1 for head in kept):
        raise ValueError("Every planted head needs a replacement probability in [0, 1]")
    result = []
    for document_index, document in enumerate(documents):
        sentences = []
        for sentence_index, sentence in enumerate(planted.SENTENCE.split(document)):
            words = sentence.split()
            # A replaced follower can itself be the next head: hits use the original words.
            hits = [
                (index + 1, word)
                for index, (word, following) in enumerate(pairwise(words))
                if word in kept
                and re.fullmatch(r"[A-Za-z]+", core := re.sub(r"[^A-Za-z]+$", "", following))
                and core not in kept[word]
            ]
            for index, head in hits:
                coordinates = f"{seed}:{document_index}:{sentence_index}:{index}".encode()
                digest = hashlib.sha256(coordinates).digest()
                # 53 bits keep the float strictly below one, including the largest digest.
                draw = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
                if draw < q[head]:
                    tail = re.search(r"[^A-Za-z]*$", words[index])
                    words[index] = kept[head][0] + (tail.group() if tail else "")
            sentences.append(" ".join(words) if hits else sentence)
        result.append(" ".join(sentences))
    return result


def prepare() -> None:
    """Calibrate both assignments with the whole sample's tokenizer and classes."""
    progress("Loading slices and learning the sample tokenizer")
    parts = planted_model.slices()
    documents = [document for part in parts for document in part]
    training = [document for part in parts[: planted_model.EVAL_SLICE] for document in part]
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        documents, max_merges=planted_model.SETTINGS.merge_budget, recognize_markers=True
    )

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    progress("Reading whole-sample classes and patterns")
    reference = framed_documents(documents, tokenizer, True)
    words = word_like_tokens(reference, spell, boundary_ids={tokenizer.bos_id, tokenizer.eos_id})
    classes = induce_classes(reference, candidates=words)

    def read(texts: list[str]) -> dict[Key, Pattern]:
        corpus = framed_documents(texts, tokenizer, True)
        return {
            (p.head, p.filler_class): p for p in syntactic_patterns(corpus, classes, heads=words)
        }

    reference_patterns = {
        (p.head, p.filler_class): p for p in syntactic_patterns(reference, classes, heads=words)
    }
    chosen, fillers = planted.choose_heads(reference, classes, reference_patterns, spell, PLANTED)
    comparison, comparison_kept = planted.choose_heads(
        reference, classes, reference_patterns, spell, COMPARISON, skip=PLANTED
    )
    if len(chosen) != PLANTED or len(comparison) != COMPARISON:
        raise ValueError(
            f"Need {PLANTED + COMPARISON} qualifying heads ({PLANTED} planted + {COMPARISON} "
            f"comparison); "
            f"only {len(chosen) + len(comparison)} qualify"
        )
    kept = {spell(p.head[0]).strip(): fillers[spell(p.head[0]).strip()][:2] for p in chosen}
    keys = {spell(p.head[0]).strip(): (p.head, p.filler_class) for p in chosen + comparison}
    if len(keys) != PLANTED + COMPARISON:
        raise ValueError("The selected heads must have distinct word spellings")

    def diversity(reading: dict[Key, Pattern]) -> dict[str, float | None]:
        return {
            head: reading[key].diversity if key in reading else None for head, key in keys.items()
        }

    progress("Reading the unplanted training corpus")
    indicator = {"control": diversity(read(training))}
    calibration: dict[str, dict[str, Calibration]] = {}
    for assignment, shift in ASSIGNMENTS.items():
        targets = {head: LEVELS[(i + shift) % len(LEVELS)] for i, head in enumerate(kept)}
        lo, hi = dict.fromkeys(kept, 0.0), dict.fromkeys(kept, 1.0)
        rows: dict[str, Calibration] = {}
        for iteration in range(1, ITERATIONS + 1):
            probabilities = {head: (lo[head] + hi[head]) / 2 for head in kept}
            progress(f"{assignment} iteration {iteration}/{ITERATIONS}: planting and reading")
            reading = read(plant_partial(training, kept, probabilities, PLANT_SEED))
            rows = {}
            for head in kept:
                pattern = reading.get(keys[head])
                achieved = pattern.diversity if pattern else None
                rows[head] = {
                    "target": targets[head],
                    "q": probabilities[head],
                    "diversity": achieved,
                    "trunk": pattern.trunk if pattern else 0,
                }
                if achieved is not None and achieved > targets[head]:
                    lo[head] = probabilities[head]
                else:
                    hi[head] = probabilities[head]
            matched = sum(
                row["diversity"] is not None and abs(row["diversity"] - row["target"]) <= TOLERANCE
                for row in rows.values()
            )
            progress(f"{assignment} iteration {iteration}: {matched}/{PLANTED} within {TOLERANCE}")
            if matched == PLANTED:
                break
        calibration[assignment] = rows
        indicator[assignment] = diversity(reading)
        print(f"\nAssignment {assignment}\nhead             target       q  diversity   trunk")
        for head, row in rows.items():
            print(
                f"{head:16} {row['target']:6.2f} {row['q']:7.4f} "
                f"{display(row['diversity']):>10} {row['trunk']:7}"
            )
            if row["diversity"] is None or abs(row["diversity"] - row["target"]) > TOLERANCE:
                print(f"WARNING: {assignment} {head} missed target by more than {TOLERANCE}")
        print("Comparison diversity (unplanted -> planted):")
        for head in comparison_kept:
            print(
                f"  {head}: {display(indicator['control'][head])} -> "
                f"{display(indicator[assignment][head])}"
            )
    prepared: Prepared = {
        "plant_seed": PLANT_SEED,
        "planted": kept,
        "comparison": comparison_kept,
        "calibration": calibration,
        "indicator_diversity": indicator,
    }
    write_json(RESULTS / "prepare.json", prepared)


def load_prepared() -> Prepared:
    return cast(Prepared, json.loads((RESULTS / "prepare.json").read_text()))


def set_threads() -> None:
    if threads := int(os.environ.get("THREADS", "0")):
        torch.set_num_threads(threads)


def score_run(
    variant: str,
    seed: int,
    output: Path,
    evaluation: list[str],
    prepared: Prepared,
    *,
    reused: bool,
) -> None:
    loaded = load_run(output)
    results = {
        "variant": variant,
        "seed": seed,
        "run": str(output),
        "reused_run": reused,
        "best_step": loaded.metrics["best_step"],
        "best_heldout_loss": loaded.metrics["best_heldout_loss"],
        "eval_bits_per_byte": planted_model.heldout_bits_per_byte(loaded, evaluation),
        "occurrences": planted_model.occurrence_scores(
            loaded, evaluation, [*prepared["planted"], *prepared["comparison"]]
        ),
    }
    write_json(RESULTS / f"{variant}_seed{seed}.json", results)
    print(
        f"{variant} seed {seed}: {results['eval_bits_per_byte']} bits per byte; "
        f"{'reused run' if reused else 'run'} {output}"
    )


def train_assignment(assignment: str, seed: int) -> None:
    set_threads()
    prepared = load_prepared()
    parts = planted_model.slices()
    documents = [document for part in parts[: planted_model.EVAL_SLICE] for document in part]
    probabilities = {head: row["q"] for head, row in prepared["calibration"][assignment].items()}
    documents = plant_partial(documents, prepared["planted"], probabilities, prepared["plant_seed"])
    output = RUNS / f"{assignment}_seed{seed}"
    train(
        documents,
        replace(planted_model.SETTINGS, seed=seed),
        output,
        sources={"dose_response": assignment},
    )
    score_run(assignment, seed, output, parts[planted_model.EVAL_SLICE], prepared, reused=False)


def rescore_control(seed: int) -> None:
    set_threads()
    prepared = load_prepared()
    output = planted_model.RUNS / f"control_seed{seed}"
    progress(f"Rescoring reused run {output}")
    score_run(
        "control",
        seed,
        output,
        planted_model.slices()[planted_model.EVAL_SLICE],
        prepared,
        reused=True,
    )


def mean(values: list[float | None]) -> float | None:
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return None
    return statistics.mean(cast(list[float], values))


def display(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "unreadable"


def crossing(drops: list[float]) -> float | None:
    """First intersection with -0.3 when traversing target levels from 0.7 down."""
    points = list(zip(LEVELS, drops, strict=True))[::-1]
    for (high, high_drop), (low, low_drop) in pairwise(points):
        if high_drop == -0.3:
            return high
        if (high_drop + 0.3) * (low_drop + 0.3) <= 0:
            return high + (-0.3 - high_drop) * (low - high) / (low_drop - high_drop)
    return LEVELS[0] if drops[0] == -0.3 else None


def verdict(checks: list[bool | None]) -> str:
    if not checks or None in checks:
        return "unreadable"
    return "HELD" if all(checks) else "FAILED"


def summarize() -> None:
    prepared = load_prepared()
    heads = [*prepared["planted"], *prepared["comparison"]]
    checks: dict[str, list[bool | None]] = {f"P{i}": [] for i in range(1, 7)}
    readings: dict[str, object] = {}
    all_drops: dict[str, dict[str, float | None]] = {}
    counts: dict[str, dict[str, int]] = {}
    levels: dict[str, list[float | None]] = {}
    eval_bpb: dict[str, float] = {}
    for seed in SEEDS:
        runs = {
            variant: cast(RunResult, json.loads(path.read_text()))
            for variant in ("control", *ASSIGNMENTS)
            if (path := RESULTS / f"{variant}_seed{seed}.json").exists()
        }
        for variant, run in runs.items():
            key = f"{variant}_seed{seed}"
            eval_bpb[key] = run["eval_bits_per_byte"]
            print(f"{key}: eval bits per byte {run['eval_bits_per_byte']:.4f}")
        for assignment in ASSIGNMENTS:
            key = f"{assignment}_seed{seed}"
            if "control" not in runs or assignment not in runs:
                print(f"{key}: unreadable (requires control and assignment results)")
                for prediction in ("P1", "P2", "P3", "P5"):
                    checks[prediction].append(None)
                continue
            control, other = runs["control"]["occurrences"], runs[assignment]["occurrences"]
            # Skipped occurrences would silently pair different words and corrupt every drop.
            assert [(o["head"], o["word"]) for o in other] == [
                (o["head"], o["word"]) for o in control
            ], f"{key}: occurrences do not match control"
            differences: dict[str, list[float | None]] = {head: [] for head in heads}
            for baseline, changed in zip(control, other, strict=True):
                differences[baseline["head"]].append(
                    changed["log_probability"] - baseline["log_probability"]
                )
            drops = {head: mean(values) for head, values in differences.items()}
            all_drops[key] = drops
            counts[key] = {head: len(values) for head, values in differences.items()}
            calibration = prepared["calibration"][assignment]
            per_level = [
                mean([drops[h] for h, row in calibration.items() if row["target"] == level])
                for level in LEVELS
            ]
            levels[key] = per_level
            diversities = [row["diversity"] for row in calibration.values()]
            planted_drops = [drops[head] for head in calibration]
            rho = None
            if (
                mean(diversities) is not None
                and mean(planted_drops) is not None
                and len(set(diversities)) > 1
                and len(set(planted_drops)) > 1
            ):
                correlation = float(spearmanr(diversities, planted_drops).statistic)
                rho = correlation if math.isfinite(correlation) else None
            strong = per_level[0]
            mild = mean([drops[h] for h, row in calibration.items() if row["target"] >= 0.5])
            comparison = {h: drops[h] for h in prepared["comparison"]}
            exceptions = [
                h for h, value in comparison.items() if value is not None and abs(value) > 0.3
            ]
            checks["P1"].append(rho >= 0.6 if rho is not None else None)
            checks["P2"].append(strong <= -1.0 if strong is not None else None)
            checks["P3"].append(mild >= -0.3 if mild is not None else None)
            checks["P5"].append(
                len(exceptions) <= 1 if mean(list(comparison.values())) is not None else None
            )
            readings[key] = {
                "drops": drops,
                "occurrences": counts[key],
                "spearman": rho,
                "strong_mean_drop": strong,
                "mild_mean_drop": mild,
                "comparison_exceptions": exceptions,
                "level_mean_drops": dict(zip(map(str, LEVELS), per_level, strict=True)),
            }
            print(
                f"{key}: P1 rho={display(rho)}; P2 strong={display(strong)}; "
                f"P3 mild={display(mild)}; P5 exceptions={exceptions}"
            )
            print(
                "  level mean drops: "
                + ", ".join(
                    f"{level:.1f}: {display(value)}"
                    for level, value in zip(LEVELS, per_level, strict=True)
                )
            )
    averaged = [
        mean([levels.get(f"{a}_seed{s}", [None] * 7)[i] for a in ASSIGNMENTS for s in SEEDS])
        for i in range(7)
    ]
    threshold = crossing(cast(list[float], averaged)) if mean(averaged) is not None else None
    checks["P4"] = [
        0.2 <= threshold <= 0.5
        if threshold is not None
        else False
        if mean(averaged) is not None
        else None
    ]
    assignment_differences: dict[str, dict[str, float | None]] = {}
    for seed in SEEDS:
        delta: dict[str, float | None] = {}
        for i, level in enumerate(LEVELS):
            a = levels.get(f"A_seed{seed}", [None] * 7)[i]
            b = levels.get(f"B_seed{seed}", [None] * 7)[i]
            value = abs(a - b) if a is not None and b is not None else None
            delta[str(level)] = value
            checks["P6"].append(value <= 0.3 if value is not None else None)
        assignment_differences[str(seed)] = delta
        print(
            f"P6 seed {seed}: " + ", ".join(f"{level}: {display(v)}" for level, v in delta.items())
        )
    print("\nhead             assignment target diversity      q  occurrences/drop per seed")
    head_rows: list[dict[str, object]] = []
    for head in heads:
        for assignment in ASSIGNMENTS:
            row = prepared["calibration"][assignment].get(head)
            achieved = (
                row["diversity"]
                if row
                else prepared["indicator_diversity"].get(assignment, {}).get(head)
            )
            target, probability = (row["target"], row["q"]) if row else (None, None)
            by_seed = {
                str(seed): {
                    "occurrences": counts.get(f"{assignment}_seed{seed}", {}).get(head),
                    "drop": all_drops.get(f"{assignment}_seed{seed}", {}).get(head),
                }
                for seed in SEEDS
            }
            head_rows.append(
                {
                    "head": head,
                    "assignment": assignment,
                    "target": target,
                    "diversity": achieved,
                    "q": probability,
                    "seeds": by_seed,
                }
            )
            seed_text = "; ".join(
                f"seed {seed}: {values['occurrences']} / {display(values['drop'])}"
                for seed, values in by_seed.items()
            )
            print(
                f"{head:16} {assignment:>10} {display(target):>6} {display(achieved):>9} "
                f"{display(probability):>6}  {seed_text}"
            )
    predictions = {prediction: verdict(values) for prediction, values in checks.items()}
    print(
        "\nP4 level mean drops: "
        + ", ".join(
            f"{level:.1f}: {display(value)}" for level, value in zip(LEVELS, averaged, strict=True)
        )
    )
    crossing_text = (
        display(threshold)
        if threshold is not None
        else ("no crossing" if mean(averaged) is not None else "unreadable")
    )
    print(f"P4 crossing diversity: {crossing_text}")
    for prediction, status in predictions.items():
        print(f"{prediction}: {status}")
    write_json(
        RESULTS / "summary.json",
        {
            "readings": readings,
            "heads": head_rows,
            "eval_bits_per_byte": eval_bpb,
            "level_mean_drops": dict(zip(map(str, LEVELS), averaged, strict=True)),
            "crossing_diversity": threshold,
            "crossing": crossing_text,
            "assignment_differences": assignment_differences,
            "predictions": predictions,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Calibrate both assignments; no model training")
    training = commands.add_parser("train", help="Train and score a calibrated assignment")
    training.add_argument("assignment", choices=tuple(ASSIGNMENTS))
    training.add_argument("seed", type=int, choices=SEEDS)
    control = commands.add_parser("rescore-control", help="Score a reused planted_model control")
    control.add_argument("seed", type=int, choices=SEEDS)
    commands.add_parser("summarize", help="Report predictions from saved results")
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare()
        case "train":
            train_assignment(args.assignment, args.seed)
        case "rescore-control":
            rescore_control(args.seed)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
