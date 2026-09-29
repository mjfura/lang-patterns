"""A natural gap: does the indicator flag, in a topic-narrow corpus, the heads where a model
trained on it then does worse on general text than one trained on a random corpus of the same
size? "A natural gap" in `docs/2026-09-28-syntactic-sufficiency-design.md`.

    natural_gap.py prepare              split, cluster, read the indicator on both corpora
    natural_gap.py train <corpus> <seed>   corpus: narrow | random
    natural_gap.py summarize

FineWeb's cached documents (`experiments/clean_domains`). Runs go to `runs/natural_gap/`
(gitignored); results to `results/` here. `THREADS` sets torch's thread count.
"""

import json
import os
import random
import statistics
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

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
RUNS = Path("runs") / "natural_gap"
sys.path[:0] = [
    str(OUT.parent / "planted_model"),
    str(OUT.parent / "planted_gap"),
    str(OUT.parent / "natural_sufficiency"),
    str(OUT.parent / "clean_domains"),
]

import clean_domains  # noqa: E402
import planted_model  # noqa: E402

EVAL_DOCUMENTS = 400
CORPUS_BYTES = 3_500_000
CLUSTERS = 12
GENERAL_CLUSTERS = 2
MARGIN_DRAWS = 8
MIN_TRUNK = 20
LOW = 0.5
SIGMAS = 3
OCCURRENCES = 50  # per head in the evaluation text
SEEDS = (0, 1)
Key = tuple[tuple[int, ...], int]


def split() -> tuple[list[str], list[str]]:
    """`(evaluation, pool)`: a seeded draw of `EVAL_DOCUMENTS` documents, and the rest."""
    documents = json.loads(clean_domains.cache_of("fineweb").read_text())
    order = list(range(len(documents)))
    random.Random(0).shuffle(order)
    evaluation = [documents[i] for i in sorted(order[:EVAL_DOCUMENTS])]
    pool = [documents[i] for i in sorted(order[EVAL_DOCUMENTS:])]
    return evaluation, pool


def fill(pool: list[str], indices: list[int]) -> list[int]:
    """The leading indices whose documents first reach `CORPUS_BYTES`."""
    chosen, size = [], 0
    for index in indices:
        if size >= CORPUS_BYTES:
            break
        chosen.append(index)
        size += len(pool[index].encode())
    return chosen


def corpora(pool: list[str]) -> dict[str, list[int]]:
    """The narrow corpus (whole topic clusters, largest first), the random one, and the margin
    draws, as indices into the pool."""
    vectors = TfidfVectorizer(max_features=20000, stop_words="english", min_df=3).fit_transform(
        pool
    )
    labels = KMeans(n_clusters=CLUSTERS, n_init=4, random_state=0).fit_predict(vectors)
    # The two clusters with most documents are registers, not topics (news: "new, said, year";
    # conversational blogs: "just, like, don, really"); the narrow corpus takes the topical
    # clusters with most text instead.
    registers = {label for label, _ in Counter(labels.tolist()).most_common(GENERAL_CLUSTERS)}
    by_bytes = sorted(
        (label for label in set(labels.tolist()) if label not in registers),
        key=lambda label: (
            -sum(len(pool[i].encode()) for i in range(len(pool)) if labels[i] == label)
        ),
    )
    narrow_order = [i for label in by_bytes for i in range(len(pool)) if labels[i] == label]
    result = {"narrow": fill(pool, narrow_order), "clusters": labels.tolist()}
    for draw in range(MARGIN_DRAWS + 1):
        order = list(range(len(pool)))
        random.Random(100 + draw).shuffle(order)
        result["random" if draw == 0 else f"margin{draw}"] = fill(pool, order)
    return result


def prepare() -> None:
    _, pool = split()
    chosen = corpora(pool)
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        pool, max_merges=planted_model.SETTINGS.merge_budget, recognize_markers=True
    )

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    reference = framed_documents(pool, tokenizer, True)
    words = word_like_tokens(reference, spell, boundary_ids={tokenizer.bos_id, tokenizer.eos_id})
    classes = induce_classes(reference, candidates=words)

    def read(indices: list[int]) -> dict[Key, Pattern]:
        sequences = [reference[i] for i in indices]
        return {
            (p.head, p.filler_class): p for p in syntactic_patterns(sequences, classes, heads=words)
        }

    reference_patterns = read(list(range(len(pool))))
    margins = [read(chosen[f"margin{d}"]) for d in range(1, MARGIN_DRAWS + 1)]

    def noise(key: Key) -> tuple[float, float] | None:
        values = [m[key].diversity for m in margins if key in m and m[key].trunk >= MIN_TRUNK]
        if len(values) < MARGIN_DRAWS - 2:
            return None
        return statistics.mean(values), statistics.stdev(values)

    readings = {}
    for name in ("narrow", "random"):
        patterns = read(chosen[name])
        heads: dict[str, dict[str, object]] = {}
        for key, pattern in patterns.items():
            stats, ref = noise(key), reference_patterns.get(key)
            if pattern.trunk < MIN_TRUNK or stats is None or ref is None or stats[1] == 0:
                continue
            z = (pattern.diversity - stats[0]) / stats[1]
            flagged = (
                pattern.diversity < LOW
                and pattern.diversity < stats[0] - SIGMAS * stats[1]
                and ref.diversity >= LOW
            )
            head = spell(key[0][0]).strip()
            row = heads.setdefault(head, {"z": z, "flagged": False, "patterns": []})
            row["z"] = min(float(row["z"]), z)  # type: ignore[arg-type]
            row["flagged"] = bool(row["flagged"]) or flagged
            row["patterns"].append(  # type: ignore[attr-defined]
                {
                    "class": key[1],
                    "trunk": pattern.trunk,
                    "diversity": pattern.diversity,
                    "margin_mean": stats[0],
                    "margin_sd": stats[1],
                    "reference_diversity": ref.diversity,
                    "flagged": flagged,
                }
            )
        readings[name] = heads
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "prepare.json").write_text(
        json.dumps(
            {
                "corpora": {k: v for k, v in chosen.items() if k != "clusters"},
                "narrow_clusters": sorted({chosen["clusters"][i] for i in chosen["narrow"]}),
                "cluster_sizes": Counter(chosen["clusters"]).most_common(),
                "readings": readings,
            },
            indent=2,
        )
        + "\n"
    )
    for name, heads in readings.items():
        flagged = sorted(h for h, row in heads.items() if row["flagged"])
        print(f"{name}: {len(heads)} heads read, {len(flagged)} flagged: {flagged}")


def train_corpus(name: str, seed: int) -> None:
    if threads := int(os.environ.get("THREADS", "0")):
        torch.set_num_threads(threads)
    prepared = json.loads((RESULTS / "prepare.json").read_text())
    evaluation, pool = split()
    documents = [pool[i] for i in prepared["corpora"][name]]
    output = RUNS / f"{name}_seed{seed}"
    train(
        documents,
        replace(planted_model.SETTINGS, seed=seed),
        output,
        sources={"natural_gap": name},
    )
    loaded = load_run(output)
    heads = sorted(prepared["readings"]["narrow"])
    planted_model.MAX_OCCURRENCES = OCCURRENCES
    results = {
        "corpus": name,
        "seed": seed,
        "best_step": loaded.metrics["best_step"],
        "eval_bits_per_byte": planted_model.heldout_bits_per_byte(loaded, evaluation),
        "occurrences": planted_model.occurrence_scores(loaded, evaluation, heads),
    }
    (RESULTS / f"{name}_seed{seed}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(name, seed, results["eval_bits_per_byte"], len(results["occurrences"]))


def summarize() -> None:
    prepared = json.loads((RESULTS / "prepare.json").read_text())
    heads = prepared["readings"]["narrow"]
    for name, reading in prepared["readings"].items():
        flagged = sorted(h for h, row in reading.items() if row["flagged"])
        print(f"{name}: {len(reading)} heads read, {len(flagged)} flagged: {flagged}")
    for seed in SEEDS:
        paths = {n: RESULTS / f"{n}_seed{seed}.json" for n in ("narrow", "random")}
        if not all(p.exists() for p in paths.values()):
            continue
        runs = {n: json.loads(p.read_text()) for n, p in paths.items()}
        # Every run scores the same occurrences in the same order (the evaluation text and the
        # regex are fixed, and no occurrence was skipped); checked here, since a skip in one run
        # would misalign the pairs.
        pairs = list(zip(runs["narrow"]["occurrences"], runs["random"]["occurrences"], strict=True))
        assert all((a["head"], a["word"]) == (b["head"], b["word"]) for a, b in pairs)
        drops: dict[str, list[float]] = {}
        for row, other in pairs:
            drops.setdefault(row["head"], []).append(
                row["log_probability"] - other["log_probability"]
            )
        per_head = {h: statistics.mean(v) for h, v in drops.items() if h in heads}
        flagged = [d for h, d in per_head.items() if heads[h]["flagged"]]
        unflagged = [d for h, d in per_head.items() if not heads[h]["flagged"]]
        rho = spearmanr(
            [heads[h]["z"] for h in per_head], [per_head[h] for h in per_head]
        ).statistic
        print(
            f"seed {seed}: eval bits per byte narrow {runs['narrow']['eval_bits_per_byte']:.4f} "
            f"random {runs['random']['eval_bits_per_byte']:.4f}; mean drop flagged "
            f"{statistics.mean(flagged) if flagged else float('nan'):+.3f} ({len(flagged)} heads), "
            f"unflagged {statistics.mean(unflagged):+.3f} ({len(unflagged)} heads); "
            f"Spearman(z, drop) {rho:.3f}"
        )
        worst = sorted(per_head.items(), key=lambda item: item[1])[:10]
        print(
            "  largest drops: "
            + ", ".join(f"{h} {d:+.2f}{'*' if heads[h]['flagged'] else ''}" for h, d in worst)
        )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["prepare"]:
            prepare()
        case ["train", ("narrow" | "random") as name, seed]:
            train_corpus(name, int(seed))
        case ["summarize"]:
            summarize()
        case _:
            sys.exit(__doc__)
