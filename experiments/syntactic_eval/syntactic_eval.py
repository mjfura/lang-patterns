"""A syntactic evaluation on natural text: does a model prefer, after a head, a word of the head's
slot class it never saw there over a word of another class?

"A syntactic evaluation on natural text" in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` records the design and
predictions before any model is scored. No model is trained: the families are the models of
`dose_response` (plantings) and of `subset_selection` and `cut_choice` (cuts).

    syntactic_eval.py prepare <plantings|cuts>   tag the reference with spaCy, build the items
    syntactic_eval.py score <plantings|cuts>     G per model and head
    syntactic_eval.py summarize                  check the six predictions

Classes come from spaCy's `en_core_web_sm`, never from the indicator. Results go to `results/`
here. No command runs on import.
"""

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from scipy.stats import spearmanr

from corpus_generalization.model.generate import LoadedRun, load_run

OUT = Path(__file__).parent
RESULTS = OUT / "results"
EXPERIMENTS = OUT.parent
sys.path[:0] = [
    str(EXPERIMENTS / name)
    for name in (
        "cut_choice",
        "divergence",
        "dose_response",
        "subset_selection",
        "natural_gap",
        "planted_model",
        "planted_gap",
        "natural_sufficiency",
        "clean_domains",
    )
]

FAMILIES = ("plantings", "cuts")
OPEN_CLASSES = ("NOUN", "VERB", "ADJ", "ADV")
SEEDS = (0, 1)
WORD_SHARE = 0.9
WORD_OCCURRENCES = 20
SLOT_SHARE = 0.2
PAIRS = 20
CONTEXTS = 20
MIN_EVAL_OCCURRENCES = 20
PREFIX_CHARS = 500
SUBSET_NAMES = ("full", "r50", "r25", "r10", "wiki25", "self25")
RANDOM_CUTS = ("r25", "r25b", "r25c")
SELF_LADDER = ("r25", "self_m25", "mix25", "self_m75", "self25")
WORD = re.compile(r"[a-z]+")


def progress(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@dataclass(frozen=True)
class Family:
    reference: list[str]
    evaluation: list[str]
    heads: list[str]
    runs: dict[str, Path]


def family(name: str) -> Family:
    if name == "plantings":
        import dose_response
        import planted_model

        parts = planted_model.slices()
        prepared = dose_response.load_prepared()
        runs = {
            f"{variant}_seed{seed}": (
                planted_model.RUNS / f"control_seed{seed}"
                if variant == "control"
                else dose_response.RUNS / f"{variant}_seed{seed}"
            )
            for variant in ("control", "A", "B")
            for seed in SEEDS
        }
        return Family(
            reference=[doc for part in parts[: planted_model.EVAL_SLICE] for doc in part],
            evaluation=parts[planted_model.EVAL_SLICE],
            heads=[*prepared["planted"], *prepared["comparison"]],
            runs=runs,
        )
    import cut_choice
    import natural_gap
    import subset_selection

    evaluation, pool = natural_gap.split()
    summary = json.loads((subset_selection.RESULTS / "summary.json").read_text())
    counts: dict[str, int] = summary["seeds"]["0"]["occurrence_counts"]
    runs = {
        f"{cut}_seed{seed}": (subset_selection.RUNS if cut in SUBSET_NAMES else cut_choice.RUNS)
        / f"{cut}_seed{seed}"
        for cut in (*SUBSET_NAMES, *cut_choice.NEW, *cut_choice.LADDER_NEW)
        for seed in SEEDS
    }
    return Family(
        reference=pool,
        evaluation=evaluation,
        heads=sorted(h for h, n in counts.items() if n >= MIN_EVAL_OCCURRENCES),
        runs=runs,
    )


def follower_pattern(heads: Iterable[str]) -> re.Pattern[str]:
    """A head followed by a word, as `planted_model.occurrence_scores` matches it."""
    alternatives = "|".join(map(re.escape, sorted(heads, key=len, reverse=True)))
    return re.compile(r"(?<![A-Za-z])(" + alternatives + r") ([A-Za-z]+)(?![A-Za-z])")


def tag_reference(
    documents: Sequence[str], heads: Sequence[str]
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    """Per lowercase word, its spaCy universal tags; per head, the tags of the words after it."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    head_set = set(heads)
    word_tags: dict[str, Counter[str]] = defaultdict(Counter)
    slot_tags: dict[str, Counter[str]] = defaultdict(Counter)
    for index, doc in enumerate(nlp.pipe(documents, batch_size=64)):
        if index % 500 == 0:
            progress(f"Tagged {index} of {len(documents)} documents")
        for position, token in enumerate(doc):
            text = token.text.lower()
            if WORD.fullmatch(text):
                word_tags[text][token.pos_] += 1
            if token.text in head_set and position + 1 < len(doc):
                following = doc[position + 1]
                if following.text.isalpha() and token.whitespace_:
                    slot_tags[token.text][following.pos_] += 1
    return dict(word_tags), dict(slot_tags)


def word_classes(word_tags: dict[str, Counter[str]]) -> dict[str, str]:
    classes = {}
    for word, tags in word_tags.items():
        total = sum(tags.values())
        tag, count = tags.most_common(1)[0]
        if total >= WORD_OCCURRENCES and tag in OPEN_CLASSES and count / total >= WORD_SHARE:
            classes[word] = tag
    return classes


def slot_class(tags: Counter[str]) -> str | None:
    """The most frequent open-class tag after the head, if it covers `SLOT_SHARE` of followers."""
    total = sum(tags.values())
    open_tags = [(count, tag) for tag, count in tags.items() if tag in OPEN_CLASSES]
    if not total or not open_tags:
        return None
    count, tag = max(open_tags)
    return tag if count / total >= SLOT_SHARE else None


def pair_by_frequency(
    valid: Sequence[str], contrast: Sequence[str], frequency: Counter[str], seed: int
) -> list[tuple[str, str]]:
    """Up to `PAIRS` valid words, drawn with `seed`, each with the unused contrast word nearest in
    log frequency, so that frequency does not decide the preference."""
    drawn = sorted(valid)
    random.Random(seed).shuffle(drawn)
    available = sorted(contrast)
    pairs = []
    for word in drawn[:PAIRS]:
        if not available:
            break
        target = math.log(frequency[word])
        match = min(available, key=lambda c: (abs(math.log(frequency[c]) - target), c))
        available.remove(match)
        pairs.append((word, match))
    return pairs


def contexts(documents: Sequence[str], heads: Sequence[str]) -> dict[str, list[str]]:
    """Up to `CONTEXTS` evaluation prefixes ending at each head, cut to whole words."""
    found: dict[str, list[str]] = {head: [] for head in heads}
    pattern = follower_pattern(heads)
    for document in documents:
        for match in pattern.finditer(document):
            head = match.group(1)
            if len(found[head]) >= CONTEXTS:
                continue
            start = max(0, match.end(1) - PREFIX_CHARS)
            prefix = document[start : match.end(1)]
            if start > 0:
                prefix = prefix[prefix.find(" ") + 1 :]
            found[head].append(prefix)
    return found


def single_token(run: LoadedRun, word: str) -> int | None:
    ids = run.tokenizer.encode(" " + word, recognize_markers=True)
    return ids[0] if len(ids) == 1 else None


def prepare(name: str) -> None:
    chosen = family(name)
    progress(f"{name}: tagging {len(chosen.reference)} reference documents")
    word_tags, slot_tags = tag_reference(chosen.reference, chosen.heads)
    classes = word_classes(word_tags)
    frequency = Counter({w: sum(t.values()) for w, t in word_tags.items() if w in classes})
    followers: dict[str, set[str]] = defaultdict(set)
    for document in chosen.reference:
        for match in follower_pattern(chosen.heads).finditer(document):
            followers[match.group(1)].add(match.group(2).lower())
    progress("Loading every model's tokenizer")
    runs = [load_run(path) for path in chosen.runs.values()]
    tokenizable = {w for w in classes if all(single_token(run, w) is not None for run in runs)}
    prefixes = contexts(chosen.evaluation, chosen.heads)
    items: dict[str, object] = {}
    skipped: Counter[str] = Counter()
    for head in chosen.heads:
        target = slot_class(slot_tags.get(head, Counter()))
        if target is None:
            skipped["no open slot class"] += 1
            continue
        unseen = [w for w in sorted(tokenizable) if w not in followers[head]]
        valid = [w for w in unseen if classes[w] == target]
        contrast = [w for w in unseen if classes[w] != target]
        pairs = pair_by_frequency(valid, contrast, frequency, seed=0)
        if not pairs or not prefixes[head]:
            skipped["no pairs or contexts"] += 1
            continue
        items[head] = {
            "slot_class": target,
            "slot_share": slot_tags[head][target] / sum(slot_tags[head].values()),
            "pairs": pairs,
            "contexts": prefixes[head],
        }
    pair_counts = [len(cast(dict[str, list[object]], item)["pairs"]) for item in items.values()]
    context_counts = [
        len(cast(dict[str, list[object]], item)["contexts"]) for item in items.values()
    ]
    print(
        f"{name}: {len(classes)} classed words, {len(tokenizable)} single-token in all "
        f"{len(runs)} models; {len(items)} of {len(chosen.heads)} heads kept; skipped "
        f"{dict(skipped)}; pairs per head median {statistics.median(pair_counts)}, "
        f"contexts per head median {statistics.median(context_counts)}; slot classes "
        f"{dict(Counter(cast(dict[str, str], i)['slot_class'] for i in items.values()))}"
    )
    write_json(RESULTS / f"items_{name}.json", {"heads": items, "word_classes": classes})


def next_token_log_probs(run: LoadedRun, prefix: str) -> torch.Tensor | None:
    """The model's next-token log probabilities after `<bos>` + `prefix`, dropping leading words
    until the prefix fits the context with room for one more token."""
    while True:
        ids = [run.tokenizer.bos_id, *run.tokenizer.encode(prefix, recognize_markers=True)]
        if len(ids) < run.model.context:
            break
        if " " not in prefix:
            return None
        prefix = prefix[prefix.find(" ") + 1 :]
    with torch.no_grad():
        logits = run.model(torch.tensor([ids]))[0, -1]
    return torch.log_softmax(logits, dim=-1)


def head_reading(run: LoadedRun, item: dict[str, object]) -> dict[str, float | int]:
    """G: the share of (context, pair) comparisons the valid word wins; also the mean margin."""
    pairs = cast(list[list[str]], item["pairs"])
    # Items are built single-token for their own family; a model outside it skips the rest.
    tokens = [(single_token(run, valid), single_token(run, contrast)) for valid, contrast in pairs]
    ids = [
        (valid, contrast)
        for valid, contrast in tokens
        if valid is not None and contrast is not None
    ]
    wins, margins = 0, []
    for prefix in cast(list[str], item["contexts"]):
        log_probs = next_token_log_probs(run, prefix)
        if log_probs is None:
            continue
        for valid, contrast in ids:
            margin = float(log_probs[valid] - log_probs[contrast])
            margins.append(margin)
            wins += margin > 0
    return {
        "g": wins / len(margins),
        "margin": statistics.mean(margins),
        "comparisons": len(margins),
    }


def score(name: str) -> None:
    chosen = family(name)
    items = json.loads((RESULTS / f"items_{name}.json").read_text())["heads"]
    rows: dict[str, dict[str, dict[str, float | int]]] = {}
    for model, path in chosen.runs.items():
        progress(f"Scoring {model}")
        run = load_run(path)
        rows[model] = {head: head_reading(run, item) for head, item in items.items()}
        print(f"{model}: mean G {statistics.mean(r['g'] for r in rows[model].values()):.4f}")
    write_json(RESULTS / f"scores_{name}.json", rows)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return float(spearmanr(xs, ys).statistic)


def status(holds: bool) -> str:
    return "HELD" if holds else "FAILED"


def mean_g(readings: dict[str, dict[str, float | int]]) -> float:
    return statistics.mean(float(r["g"]) for r in readings.values())


def summarize_plantings(checks: dict[str, list[bool]]) -> dict[str, object]:
    import dose_response

    prepared = dose_response.load_prepared()
    scores = json.loads((RESULTS / "scores_plantings.json").read_text())
    divergences = json.loads((EXPERIMENTS / "divergence/results/dose.json").read_text())
    report: dict[str, object] = {}
    for seed in SEEDS:
        control = scores[f"control_seed{seed}"]
        control_mean = mean_g(control)
        checks["1"].append(control_mean >= 0.70)
        strong, comparison, d_values, deltas = [], [], [], []
        for assignment in ("A", "B"):
            planted = scores[f"{assignment}_seed{seed}"]
            for head in control.keys() & planted.keys():
                delta = float(planted[head]["g"]) - float(control[head]["g"])
                row = prepared["calibration"][assignment].get(head)
                if row is None:
                    comparison.append(abs(delta))
                    continue
                if row["target"] <= 0.2:
                    strong.append(delta)
                d_values.append(divergences[assignment][head]["divergence"])
                deltas.append(delta)
        strong_mean = statistics.mean(strong) if strong else math.nan
        comparison_mean = statistics.mean(comparison) if comparison else math.nan
        checks["2"].append(strong_mean <= -0.10 and comparison_mean <= 0.05)
        rho = spearman(d_values, deltas)
        checks["3"].append(rho <= -0.5)
        print(
            f"plantings seed {seed}: control mean G {control_mean:.3f} over {len(control)} heads; "
            f"strong-dose mean delta {strong_mean:+.3f} ({len(strong)} readings); comparison "
            f"mean |delta| {comparison_mean:.3f} ({len(comparison)}); Spearman(D, delta) "
            f"{rho:.3f} over {len(deltas)} readings"
        )
        for assignment in ("A", "B"):
            planted = scores[f"{assignment}_seed{seed}"]
            for head in sorted(control.keys() & planted.keys()):
                target = prepared["calibration"][assignment].get(head, {}).get("target")
                print(
                    f"    {assignment} {head:10} target {target}  G control "
                    f"{float(control[head]['g']):.3f}  planted {float(planted[head]['g']):.3f}"
                )
        report[str(seed)] = {
            "control_mean_g": control_mean,
            "strong_mean_delta": strong_mean,
            "comparison_mean_abs_delta": comparison_mean,
            "spearman_divergence_delta": rho,
        }
    return report


def summarize_cuts(checks: dict[str, list[bool]]) -> dict[str, object]:
    import cut_choice

    scores = json.loads((RESULTS / "scores_cuts.json").read_text())
    cut_scores = json.loads((cut_choice.RESULTS / "scores.json").read_text())
    report: dict[str, object] = {}
    means = {
        cut: {seed: mean_g(scores[f"{cut}_seed{seed}"]) for seed in SEEDS} for cut in cut_scores
    }
    for cut in sorted(cut_scores, key=lambda c: cut_scores[c]["score"]):
        print(
            f"  {cut:13} score {cut_scores[cut]['score']:+.4f}  z {cut_scores[cut]['z']:+6.2f}  "
            f"documents {cut_scores[cut]['documents']:5}  mean G "
            + " / ".join(f"{means[cut][seed]:.4f}" for seed in SEEDS)
        )
    for cut in ("full", "r50", "r10"):
        print(
            f"  {cut:13} mean G "
            + " / ".join(f"{mean_g(scores[f'{cut}_seed{s}']):.4f}" for s in SEEDS)
        )
    ladder = [statistics.mean(means[c].values()) for c in SELF_LADDER]
    rho_ladder = spearman([cut_scores[c]["score"] for c in SELF_LADDER], ladder)
    checks["4"].append(rho_ladder <= -0.9)
    print(f"P4 self ladder Spearman(score, mean G) {rho_ladder:.3f}")
    ten = cut_choice.CUTS
    per_head: dict[str, object] = {}
    for seed in SEEDS:
        values = [means[c][seed] for c in ten]
        rho_score = spearman([cut_scores[c]["score"] for c in ten], values)
        rho_documents = spearman([math.log(cut_scores[c]["documents"]) for c in ten], values)
        checks["5"].append(rho_score <= -0.6 and abs(rho_documents) < abs(rho_score))
        print(
            f"P5 seed {seed}: Spearman(score, mean G) {rho_score:.3f}, "
            f"Spearman(log documents, mean G) {rho_documents:.3f}"
        )
        random_g = {
            h: statistics.mean(float(scores[f"{c}_seed{seed}"][h]["g"]) for c in RANDOM_CUTS)
            for h in scores[f"r25_seed{seed}"]
        }
        for cut, row in cut_scores.items():
            if cut in RANDOM_CUTS or row["z"] < 3 or row["score"] < 0.05:
                continue
            heads = sorted(h for h in random_g if h in row["excess"])
            rho = spearman(
                [row["excess"][h] for h in heads],
                [float(scores[f"{cut}_seed{seed}"][h]["g"]) - random_g[h] for h in heads],
            )
            per_head[f"{cut}_seed{seed}"] = {"heads": len(heads), "spearman": rho}
            checks["6"].append(rho <= -0.2)
            print(f"P6 {cut} seed {seed}: Spearman(E, G - random G) {rho:.3f} over {len(heads)}")
        report[str(seed)] = {"spearman_score": rho_score, "spearman_log_documents": rho_documents}
    report["mean_g"] = means
    report["self_ladder_spearman"] = rho_ladder
    report["per_head"] = per_head
    return report


def summarize() -> None:
    checks: dict[str, list[bool]] = {str(i): [] for i in range(1, 7)}
    print("Plantings")
    plantings = summarize_plantings(checks)
    print("\nCuts")
    cuts = summarize_cuts(checks)
    predictions = {k: status(all(v)) if v else "unreadable" for k, v in checks.items()}
    print(f"\nPredictions: {json.dumps(predictions)}")
    write_json(
        RESULTS / "summary.json",
        {"plantings": plantings, "cuts": cuts, "predictions": predictions},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "score"):
        sub = commands.add_parser(command)
        sub.add_argument("family", choices=FAMILIES)
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare(args.family)
        case "score":
            score(args.family)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
