"""Do existing corpus metrics see what the indicator sees?

"Do existing corpus metrics see what the indicator sees?" in
`docs/2026-09-28-syntactic-sufficiency-design.md` records the design and predictions before
computing. Corpus side only.

    baselines.py compute      six baselines on the class cuts, head cuts and controls
    baselines.py summarize    the two predictions

No command runs on import.
"""

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
RESULTS = OUT / "results"
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "class_patterns")]

import class_patterns as base  # noqa: E402
import clean_domains  # noqa: E402  (on the path through class_patterns)
from class_patterns import divergence, natural_gap, subset_selection  # noqa: E402

CLASS_CUTS = ("class_cut", "poss_class_cut", "num_class_cut", "verb_class_cut")
HEAD_CUTS = ("head_cut", "poss_head_cut", "num_head_cut", "verb_head_cut")
CONTROLS = (
    "control",
    "poss_control",
    "num_control",
    "verb_control",
    "ctrl_a",
    "ctrl_b",
    "ctrl_c",
    "ctrl_d",
)
WINDOW = 1000
DISTINCT_WORDS = 500_000
VENDI_DOCUMENTS = 1000
FLAG_SD = 3.0


def shuffled_words(documents: list[str]) -> list[str]:
    order = list(range(len(documents)))
    random.Random(0).shuffle(order)
    return [w for i in order for w in subset_selection.words(documents[i])]


def mean_perplexity(model: "subset_selection.Bigram", documents: list[str]) -> float:
    return statistics.mean(
        model.perplexity(tokens) for d in documents if (tokens := subset_selection.words(d))
    )


def unigram_kl(reference: Counter[str], words: list[str]) -> float:
    """KL(F || corpus) over F's words, the corpus add-one smoothed."""
    counts = Counter(words)
    total_r = sum(reference.values())
    total_c = len(words) + len(reference)
    return sum(
        (n / total_r) * math.log((n / total_r) / ((counts[w] + 1) / total_c))
        for w, n in reference.items()
    )


def mattr(words: list[str], window: int = WINDOW) -> float:
    """Moving-average type-token ratio: the mean distinct share over every full window."""
    counts: Counter[str] = Counter(words[:window])
    ratios = [len(counts) / window]
    for i in range(window, len(words)):
        counts[words[i]] += 1
        old = words[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        ratios.append(len(counts) / window)
    return statistics.mean(ratios)


def distinct_2(words: list[str]) -> float:
    head = words[:DISTINCT_WORDS]
    pairs = list(pairwise(head))
    return len(set(pairs)) / len(pairs)


def vendi(documents: list[str], encoder: object) -> float:
    """exp of the eigenvalue entropy of the cosine similarity matrix over n, n documents."""
    chosen = random.Random(0).sample(documents, min(VENDI_DOCUMENTS, len(documents)))
    embeddings = np.asarray(
        encoder.encode(chosen, normalize_embeddings=True, batch_size=64),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(embeddings @ embeddings.T / len(chosen))
    positive = eigenvalues[eigenvalues > 1e-12]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def compute() -> None:
    from sentence_transformers import SentenceTransformer

    _, pool = natural_gap.split()
    own = subset_selection.Bigram(pool)
    wiki = subset_selection.Bigram(json.loads(clean_domains.cache_of("wikipedia").read_text()))
    reference = Counter(w for d in pool for w in subset_selection.words(d))
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    rows: dict[str, dict[str, float]] = {}
    for name in (*CLASS_CUTS, *HEAD_CUTS, *CONTROLS):
        divergence.progress(f"Reading {name}")
        documents = base.corpus(name)
        words = shuffled_words(documents)
        rows[name] = {
            "perplexity_f_bigram": mean_perplexity(own, documents),
            "perplexity_wiki_bigram": mean_perplexity(wiki, documents),
            "unigram_kl_from_f": unigram_kl(reference, words),
            "mattr": mattr(words),
            "distinct_2": distinct_2(words),
            "vendi": vendi(documents, encoder),
        }
        print(name, {k: round(v, 4) for k, v in rows[name].items()})
    divergence.write_json(RESULTS / "baselines.json", rows)


def summarize() -> None:
    rows = json.loads((RESULTS / "baselines.json").read_text())
    metrics = list(rows[CONTROLS[0]])
    held = {"1": True, "2": True}
    report: dict[str, object] = {}
    for metric in metrics:
        values = [rows[c][metric] for c in CONTROLS]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        z_class = {c: (rows[c][metric] - mean) / sd for c in CLASS_CUTS}
        z_pair = {
            c: (rows[c][metric] - rows[h][metric]) / sd
            for c, h in zip(CLASS_CUTS, HEAD_CUTS, strict=True)
        }
        flagged = [c for c, z in z_class.items() if abs(z) > FLAG_SD]
        separated = [c for c, z in z_pair.items() if abs(z) > FLAG_SD]
        held["1"] &= len(flagged) <= 1
        held["2"] &= len(separated) <= 1
        report[metric] = {
            "control_mean": mean,
            "control_sd": sd,
            "z_class_cut": z_class,
            "z_class_minus_head": z_pair,
            "flagged": flagged,
            "separated": separated,
        }
        print(
            f"{metric:24} class cuts z "
            + ", ".join(
                f"{c.removesuffix('_class_cut') or 'class'} {z:+.1f}" for c, z in z_class.items()
            )
            + " | class - head z "
            + ", ".join(f"{z:+.1f}" for z in z_pair.values())
        )
    predictions = {k: "HELD" if v else "FAILED" for k, v in held.items()}
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(RESULTS / "summary.json", {"metrics": report, "predictions": predictions})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("compute", "summarize"))
    {"compute": compute, "summarize": summarize}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
