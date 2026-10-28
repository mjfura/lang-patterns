"""A divergence as the effect size: re-read dose_response and subset_selection with D.

"A divergence as the effect size" in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` records the definition and the
predictions before computing. No model is trained: the drops are read from the two experiments'
results, and only the corpus reading changes.

    divergence.py dose        D per planted head and assignment (re-plants from prepare.json)
    divergence.py subsets     D per head for each subset of subset_selection, against F
    divergence.py summarize   check the five predictions
    divergence.py baseline    D for 8 random cuts of F at each of r50, r25 and r10's bytes
    divergence.py summarize-baseline   excess over the draws, one score per cut

Results go to `results/` here. No command runs on import.
"""

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from scipy.stats import spearmanr

from corpus_generalization.analysis.syntactic import (
    TokenClasses,
    _pattern_counts,
    induce_classes,
    word_like_tokens,
)
from corpus_generalization.diagnose import prepare_reference
from corpus_generalization.packing import framed_documents
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
RESULTS = OUT / "results"
EXPERIMENTS = OUT.parent
sys.path[:0] = [
    str(EXPERIMENTS / "dose_response"),
    str(EXPERIMENTS / "subset_selection"),
    str(EXPERIMENTS / "natural_gap"),
    str(EXPERIMENTS / "planted_model"),
    str(EXPERIMENTS / "planted_gap"),
    str(EXPERIMENTS / "natural_sufficiency"),
    str(EXPERIMENTS / "clean_domains"),
]

SEEDS = (0, 1)
SUBSETS = ("r50", "r25", "r10", "wiki25", "self25")
EXPLORATORY = ("r10", "self25")
MATERIAL_DROP = -0.3
BASELINE_SIZES = ("r50", "r25", "r10")
DRAWS = 8


def progress(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def head_fillers(
    sequences: Sequence[Sequence[int]],
    classes: TokenClasses,
    heads: Collection[int],
    word_starts: Sequence[Sequence[bool]] | None = None,
) -> dict[int, Counter[int]]:
    """Per one-token head, its fillers over all its syntactic patterns, counted in distinct
    contexts. The tool's own counting, so that D reads exactly the slots diversity reads; a
    lexical transition continues the head's word and is spelling, not a slot."""
    counts, _ = _pattern_counts(sequences, classes, 1, heads, word_starts)
    per_head: dict[int, Counter[int]] = defaultdict(Counter)
    for (head, _, lexical), by_filler in counts.items():
        if not lexical:
            per_head[head[0]].update(by_filler)
    return dict(per_head)


def backoff_unigram(
    sequences: Sequence[Sequence[int]], vocabulary: Collection[int]
) -> dict[int, float]:
    """Every occurrence of each filler in the corpus, plus one per filler of the vocabulary:
    without the added count a filler absent from a subset would make D infinite."""
    counts = Counter(token for sequence in sequences for token in sequence if token in vocabulary)
    total = sum(counts.values()) + len(vocabulary)
    return {token: (counts[token] + 1) / total for token in vocabulary}


def fillers_vocabulary(classes: TokenClasses) -> frozenset[int]:
    """The classified tokens that can fill a slot: classes of one token are fixed words."""
    sizes = Counter(classes.classes.values())
    return frozenset(token for token, label in classes.classes.items() if sizes[label] >= 2)


def divergence(
    reference: Mapping[int, int], corpus: Mapping[int, int], unigram: Mapping[int, float]
) -> float:
    """KL(R || S) in nats: R the reference's maximum-likelihood fillers, S the corpus's with
    Witten-Bell backoff to `unigram`, lambda = n / (n + T)."""
    total = sum(reference.values())
    if total == 0:
        raise ValueError("The reference has no fillers for this head")
    n, distinct = sum(corpus.values()), len(corpus)
    value = 0.0
    for filler, count in reference.items():
        if count == 0:
            continue
        backoff = unigram[filler]
        smoothed = (corpus.get(filler, 0) + distinct * backoff) / (n + distinct) if n else backoff
        r = count / total
        value += r * math.log(r / smoothed)
    return value


def dose() -> None:
    import dose_response
    import planted_model

    prepared = dose_response.load_prepared()
    progress("Loading slices and learning the sample tokenizer, as dose_response.prepare")
    parts = planted_model.slices()
    documents = [document for part in parts for document in part]
    training = [document for part in parts[: planted_model.EVAL_SLICE] for document in part]
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        documents, max_merges=planted_model.SETTINGS.merge_budget, recognize_markers=True
    )

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    progress("Reading whole-sample classes")
    sample = framed_documents(documents, tokenizer, True)
    words = word_like_tokens(sample, spell, boundary_ids={tokenizer.bos_id, tokenizer.eos_id})
    classes = induce_classes(sample, candidates=words)
    vocabulary = fillers_vocabulary(classes)
    tokens = {text.strip(): token for token in words if (text := spell(token))}
    heads = [*prepared["planted"], *prepared["comparison"]]
    progress("Reading the unplanted training corpus")
    control = framed_documents(training, tokenizer, True)
    reference = head_fillers(control, classes, words)
    rows: dict[str, dict[str, dict[str, float | int]]] = {}
    for assignment, calibration in prepared["calibration"].items():
        progress(f"Re-planting assignment {assignment}")
        probabilities = {head: row["q"] for head, row in calibration.items()}
        planted_documents = dose_response.plant_partial(
            training, prepared["planted"], probabilities, prepared["plant_seed"]
        )
        corpus = framed_documents(planted_documents, tokenizer, True)
        unigram = backoff_unigram(corpus, vocabulary)
        fillers = head_fillers(corpus, classes, words)
        rows[assignment] = {}
        for head in heads:
            token = tokens[head]
            value = divergence(reference[token], fillers.get(token, Counter()), unigram)
            rows[assignment][head] = {
                "divergence": value,
                "n": sum(fillers.get(token, Counter()).values()),
            }
            print(f"{assignment} {head:10} D = {value:.4f}")
    write_json(RESULTS / "dose.json", rows)


@dataclass(frozen=True)
class FullReading:
    """F read as `diagnose(subset, reference=F)` reads it; `full[i]` frames `pool[i]`."""

    pool: list[str]
    tokenizer: ByteLevelBPETokenizer
    full: Sequence[list[int]]
    word_starts: Sequence[list[bool]]
    classes: TokenClasses
    words: frozenset[int]
    vocabulary: frozenset[int]
    tokens: dict[str, int]
    reference: dict[int, Counter[int]]

    def read(self, chosen: Sequence[int]) -> dict[str, dict[str, float | int]]:
        """D and count per alphabetic head read in F, over the pool documents `chosen`."""
        corpus = [self.full[i] for i in chosen]
        unigram = backoff_unigram(corpus, self.vocabulary)
        fillers = head_fillers(
            corpus, self.classes, self.words, [self.word_starts[i] for i in chosen]
        )
        return {
            head: {
                "divergence": divergence(
                    self.reference[token], fillers.get(token, Counter()), unigram
                ),
                "n": sum(fillers.get(token, Counter()).values()),
            }
            for head, token in sorted(self.tokens.items())
            if head.isascii() and head.isalpha() and self.reference.get(token)
        }


def read_full() -> FullReading:
    """F as the tool reads it now: every token classified, learned to 1.50 tokens per word
    (11,381 merges). The first reading, of word-like tokens at 11,563 merges, is in
    `results/word_like/`. Heads are the space-prefixed alphabetic tokens, the words the models'
    drops are read at."""
    import natural_gap

    _, pool = natural_gap.split()
    reading = prepare_reference(pool, head_lengths=(1,), progress=progress)
    classes = reading.classes
    words = frozenset(classes.classes)
    tokens = {
        text[1:]: token
        for token in words
        if (text := reading.spell(token)).startswith(" ")
        and text[1:].isascii()
        and text[1:].isalpha()
    }
    return FullReading(
        pool=list(pool),
        tokenizer=reading.tokenizer,
        full=reading.documents,
        word_starts=reading.word_starts,
        classes=classes,
        words=words,
        vocabulary=fillers_vocabulary(classes),
        tokens=tokens,
        reference=head_fillers(reading.documents, classes, words, reading.word_starts),
    )


def subsets() -> None:
    import subset_selection

    reading = read_full()
    cuts = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    rows: dict[str, object] = {"n_classes": reading.classes.n_classes}
    for name in SUBSETS:
        progress(f"Reading {name}")
        rows[name] = reading.read(cuts[name]["indices"])
    write_json(RESULTS / "subsets.json", rows)


def random_cut(sizes: Sequence[int], target: int, seed: int) -> list[int]:
    """Documents in seeded shuffled order until the cut reaches `target` bytes."""
    order = list(range(len(sizes)))
    random.Random(seed).shuffle(order)
    chosen, total = [], 0
    for index in order:
        if total >= target:
            break
        chosen.append(index)
        total += sizes[index]
    return chosen


def baseline() -> None:
    import subset_selection

    reading = read_full()
    sizes = [len(document.encode("utf-8")) for document in reading.pool]
    cuts = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    rows: dict[str, list[dict[str, dict[str, float | int]]]] = {}
    for size in BASELINE_SIZES:
        rows[size] = []
        for seed in range(DRAWS):
            progress(f"Reading draw {seed} at the size of {size}")
            chosen = random_cut(sizes, cuts[size]["bytes"], seed)
            rows[size].append(reading.read(chosen))
    weights = {
        head: sum(reading.reference[token].values())
        for head, token in reading.tokens.items()
        if head.isascii() and head.isalpha() and reading.reference.get(token)
    }
    write_json(RESULTS / "baseline.json", {"draws": rows, "reference_counts": weights})


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    return float(spearmanr(xs, ys).statistic)


def origin_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, list[float]]:
    """Slope through the origin, R² against the mean, residuals."""
    x, y = np.asarray(xs), np.asarray(ys)
    slope = float(x @ y / (x @ x))
    residuals = y - slope * x
    r2 = 1 - float(residuals @ residuals) / float(((y - y.mean()) ** 2).sum())
    return slope, r2, residuals.tolist()


def line_residuals(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    slope, intercept = np.polyfit(xs, ys, 1)
    return (np.asarray(ys) - (slope * np.asarray(xs) + intercept)).tolist()


def status(holds: bool) -> str:
    return "HELD" if holds else "FAILED"


def summarize() -> None:
    dose_rows = json.loads((RESULTS / "dose.json").read_text())
    subset_rows = json.loads((RESULTS / "subsets.json").read_text())
    dose_summary = json.loads((EXPERIMENTS / "dose_response/results/summary.json").read_text())
    subset_summary = json.loads((EXPERIMENTS / "subset_selection/results/summary.json").read_text())
    planted = [row for row in dose_summary["heads"] if row["target"] is not None]
    points = [
        {
            "head": row["head"],
            "assignment": row["assignment"],
            "seed": seed,
            "diversity": row["diversity"],
            "divergence": dose_rows[row["assignment"]][row["head"]]["divergence"],
            "drop": row["seeds"][str(seed)]["drop"],
        }
        for row in planted
        for seed in SEEDS
    ]
    print("head        asg  diversity        D   drop s0   drop s1")
    for row in planted:
        d = dose_rows[row["assignment"]][row["head"]]["divergence"]
        drops = [row["seeds"][str(seed)]["drop"] for seed in SEEDS]
        print(
            f"{row['head']:10} {row['assignment']:>4} {row['diversity']:10.3f} {d:8.3f} "
            f"{drops[0]:9.3f} {drops[1]:9.3f}"
        )

    p1: dict[str, object] = {}
    for seed in SEEDS:
        chosen = [p for p in points if p["seed"] == seed]
        drops = [p["drop"] for p in chosen]
        rho_d = correlation([p["divergence"] for p in chosen], drops)
        rho_div = correlation([p["diversity"] for p in chosen], drops)
        p1[str(seed)] = {
            "spearman_divergence": rho_d,
            "spearman_diversity": rho_div,
            "holds": abs(rho_d) - abs(rho_div) >= 0.05,
        }
        print(f"P1 seed {seed}: Spearman D {rho_d:.3f}, diversity {rho_div:.3f}")

    divergences = [p["divergence"] for p in points]
    all_drops = [p["drop"] for p in points]
    slope, r2, residuals = origin_fit(divergences, all_drops)
    p2 = -2 <= slope <= -0.5 and r2 >= 0.6
    print(f"P2: slope {slope:.3f}, R² {r2:.3f} over {len(points)} readings")

    diversity_residuals = line_residuals([p["diversity"] for p in points], all_drops)
    sd_d, sd_div = statistics.pstdev(residuals), statistics.pstdev(diversity_residuals)
    p3 = sd_d <= 2 / 3 * sd_div
    print(f"P3: residual sd on D {sd_d:.3f}, on diversity {sd_div:.3f}, ratio {sd_d / sd_div:.3f}")

    kappa = MATERIAL_DROP / slope
    self25 = subset_rows["self25"]
    p4: dict[str, object] = {}
    for seed in SEEDS:
        flagged = subset_summary["seeds"][str(seed)]["prediction_4"]["self25"]["flagged_heads"]
        above = [head for head in flagged if self25[head]["divergence"] > kappa]
        p4[str(seed)] = {
            "flagged": {head: self25[head]["divergence"] for head in flagged},
            "above_kappa": above,
            "holds": len(above) <= 1,
        }
        print(
            f"P4 seed {seed}: kappa {kappa:.3f}; flagged D "
            + ", ".join(f"{head} {self25[head]['divergence']:.3f}" for head in flagged)
        )

    p5: dict[str, object] = {}
    for name in EXPLORATORY:
        for seed in SEEDS:
            drops = subset_summary["seeds"][str(seed)]["drops"][name]
            heads = sorted(head for head in drops if head in subset_rows[name])
            ys = [drops[head] for head in heads]
            rho_d = correlation([subset_rows[name][h]["divergence"] for h in heads], ys)
            rho_n = correlation([math.log(max(subset_rows[name][h]["n"], 1)) for h in heads], ys)
            p5[f"{name}_seed{seed}"] = {
                "heads": len(heads),
                "spearman_divergence": rho_d,
                "spearman_log_trunk": rho_n,
                "holds": rho_d <= -0.2 and abs(rho_d) - abs(rho_n) >= 0.05,
            }
            print(
                f"P5 {name} seed {seed}: {len(heads)} heads, Spearman D {rho_d:.3f}, "
                f"log trunk {rho_n:.3f}"
            )
    for name in SUBSETS:
        values = [row["divergence"] for row in subset_rows[name].values()]
        print(f"{name}: median D {statistics.median(values):.4f} over {len(values)} heads")

    predictions = {
        "1": status(all(cast(dict[str, bool], v)["holds"] for v in p1.values())),
        "2": status(p2),
        "3": status(p3),
        "4": status(all(cast(dict[str, bool], v)["holds"] for v in p4.values())),
        "5": status(all(cast(dict[str, bool], v)["holds"] for v in p5.values())),
    }
    print(f"Predictions: {json.dumps(predictions)}")
    write_json(
        RESULTS / "summary.json",
        {
            "points": points,
            "prediction_1": p1,
            "prediction_2": {"slope": slope, "r2": r2},
            "prediction_3": {"residual_sd_divergence": sd_d, "residual_sd_diversity": sd_div},
            "kappa": kappa,
            "prediction_4": p4,
            "prediction_5": p5,
            "predictions": predictions,
        },
    )


SIZE_OF = {"r50": "r50", "r25": "r25", "r10": "r10", "wiki25": "r25", "self25": "r25"}


def weighted(shares: Mapping[str, float], values: Mapping[str, float]) -> float:
    return sum(share * values[head] for head, share in shares.items())


@dataclass(frozen=True)
class Baseline:
    """Per size of cut: the draws' mean D per head and the sd of their leave-one-out scores."""

    shares: dict[str, float]
    means: dict[str, dict[str, float]]
    spread: dict[str, float]


def load_baseline() -> Baseline:
    stored = json.loads((RESULTS / "baseline.json").read_text())
    counts: dict[str, int] = stored["reference_counts"]
    total = sum(counts.values())
    shares = {head: count / total for head, count in counts.items()}
    draws: dict[str, list[dict[str, float]]] = {
        size: [{h: row["divergence"] for h, row in draw.items()} for draw in rows]
        for size, rows in stored["draws"].items()
    }
    means = {
        size: {h: statistics.mean(draw[h] for draw in rows) for h in shares}
        for size, rows in draws.items()
    }
    spread: dict[str, float] = {}
    for size, rows in draws.items():
        scores = []
        for j, draw in enumerate(rows):
            others = [other for k, other in enumerate(rows) if k != j]
            rest = {h: statistics.mean(o[h] for o in others) for h in shares}
            scores.append(weighted(shares, {h: draw[h] - rest[h] for h in shares}))
        spread[size] = statistics.stdev(scores)
    return Baseline(shares, means, spread)


def summarize_baseline() -> None:
    subset_rows = json.loads((RESULTS / "subsets.json").read_text())
    subset_summary = json.loads((EXPERIMENTS / "subset_selection/results/summary.json").read_text())
    loaded = load_baseline()
    shares, means, spread = loaded.shares, loaded.means, loaded.spread
    excess = {
        name: {h: subset_rows[name][h]["divergence"] - means[SIZE_OF[name]][h] for h in shares}
        for name in SUBSETS
    }
    scores = {name: weighted(shares, excess[name]) for name in SUBSETS}
    z = {name: scores[name] / spread[SIZE_OF[name]] for name in SUBSETS}
    for name in SUBSETS:
        print(
            f"{name:7} score {scores[name]:+.4f}  z {z[name]:+.2f}  "
            f"(draw sd {spread[SIZE_OF[name]]:.4f})"
        )
    p1 = all(abs(z[name]) <= 2 for name in BASELINE_SIZES)
    p2 = all(z[name] >= 3 for name in ("wiki25", "self25"))
    p3 = scores["r25"] < scores["self25"] < scores["wiki25"]

    p4: dict[str, object] = {}
    p5: dict[str, object] = {}
    for seed in SEEDS:
        drops = subset_summary["seeds"][str(seed)]["drops"]
        for name in ("self25", "wiki25"):
            heads = sorted(h for h in drops[name].keys() & drops["r25"].keys() if h in shares)
            rho = correlation(
                [excess[name][h] for h in heads], [drops[name][h] - drops["r25"][h] for h in heads]
            )
            p4[f"{name}_seed{seed}"] = {"heads": len(heads), "spearman": rho, "holds": rho <= -0.2}
            print(f"P4 {name} seed {seed}: {len(heads)} heads, Spearman(E, drop - r25) {rho:.3f}")
        heads = sorted(h for h in drops["r10"] if h in shares)
        ys = [drops["r10"][h] for h in heads]
        rho_e = correlation([excess["r10"][h] for h in heads], ys)
        rho_d = correlation([subset_rows["r10"][h]["divergence"] for h in heads], ys)
        p5[str(seed)] = {
            "heads": len(heads),
            "spearman_excess": rho_e,
            "spearman_divergence": rho_d,
        }
        print(f"P5 r10 seed {seed}: Spearman E {rho_e:.3f}, D {rho_d:.3f}")

    predictions = {
        "1": status(p1),
        "2": status(p2),
        "3": status(p3),
        "4": status(all(cast(dict[str, bool], v)["holds"] for v in p4.values())),
        "5": "exploratory",
    }
    print(f"Predictions: {json.dumps(predictions)}")
    write_json(
        RESULTS / "baseline_summary.json",
        {
            "scores": scores,
            "z": z,
            "draw_score_sd": spread,
            "prediction_4": p4,
            "prediction_5": p5,
            "predictions": predictions,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = {
        "dose": dose,
        "subsets": subsets,
        "summarize": summarize,
        "baseline": baseline,
        "summarize-baseline": summarize_baseline,
    }
    parser.add_argument("command", choices=tuple(commands))
    commands[parser.parse_args().command]()


if __name__ == "__main__":
    main()
