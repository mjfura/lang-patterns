"""Head patterns and class patterns: at which level does a model learn syntax?

"Head patterns and class patterns" in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` records the design and
predictions before running.

    class_patterns.py prepare              segment and tag r50, build class_cut, head_cut, control
    class_patterns.py train <name> <seed>  train and score one corpus, as subset_selection does
    class_patterns.py items                syntactic items for the 14 heads, as syntactic_eval
    class_patterns.py summarize            head- and class-level D, G per head, the four predictions

Corpora go to `runs/class_patterns/corpora/` (gitignored, rebuilt by `prepare`); results to
`results/` here. No command runs on import.
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from corpus_generalization.packing import framed_documents

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "class_patterns"
CORPORA = RUNS / "corpora"
EXPERIMENTS = OUT.parent
sys.path[:0] = [
    str(EXPERIMENTS / name)
    for name in (
        "syntactic_eval",
        "divergence",
        "subset_selection",
        "natural_gap",
        "planted_model",
        "planted_gap",
        "natural_sufficiency",
        "clean_domains",
    )
]

import divergence  # noqa: E402
import natural_gap  # noqa: E402
import subset_selection  # noqa: E402
import syntactic_eval  # noqa: E402

CLASSES = {
    "pronoun": ("I", "you", "he", "she", "we", "they"),
    "modal": ("can", "could", "will", "would", "should", "may", "might", "must"),
}
HEADS = tuple(head for heads in CLASSES.values() for head in heads)
REMOVED = ("she", "they", "can", "will")
CORPUS_NAMES = ("class_cut", "head_cut", "control")
SEEDS = (0, 1)


def head_of(text: str, heads: Sequence[str] = HEADS) -> str | None:
    """The listed head a token spells: `I` by case, the others in any case."""
    if text == "I":
        return "I" if "I" in heads else None
    lowered = text.lower()
    return lowered if lowered in heads and lowered != "i" else None


def segment(
    documents: Sequence[str], heads: Sequence[str] = HEADS, follower: str = "VERB"
) -> list[list[tuple[str, frozenset[str]]]]:
    """Per document, its sentences with the listed heads that a `follower` token follows."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.enable_pipe("senter")
    result = []
    for index, doc in enumerate(nlp.pipe(documents, batch_size=64)):
        if index % 500 == 0:
            divergence.progress(f"Segmented {index} of {len(documents)} documents")
        sentences = []
        for sentence in doc.sents:
            targets = {
                head
                for token in sentence[:-1]
                if (head := head_of(token.text, heads)) is not None
                and token.nbor().pos_ == follower
            }
            sentences.append((sentence.text_with_ws, frozenset(targets)))
        result.append(sentences)
    return result


def assemble(
    segmented: Sequence[Sequence[tuple[str, frozenset[str]]]], removed: set[tuple[int, int]]
) -> list[str]:
    documents = []
    for d, sentences in enumerate(segmented):
        text = "".join(s for i, (s, _) in enumerate(sentences) if (d, i) not in removed).strip()
        if text:
            documents.append(text)
    return documents


def size(documents: Sequence[str]) -> int:
    return sum(len(document.encode("utf-8")) for document in documents)


def match_bytes(
    segmented: Sequence[Sequence[tuple[str, frozenset[str]]]],
    removed: set[tuple[int, int]],
    target: int,
    seed: int,
) -> set[tuple[int, int]]:
    """Add random non-target sentences to `removed` until the corpus has at most `target` bytes."""
    candidates = [
        (d, i)
        for d, sentences in enumerate(segmented)
        for i, (_, targets) in enumerate(sentences)
        if not targets and (d, i) not in removed
    ]
    random.Random(seed).shuffle(candidates)
    removed = set(removed)
    current = size(assemble(segmented, removed))
    for d, i in candidates:
        if current <= target:
            break
        removed.add((d, i))
        current -= len(segmented[d][i][0].encode("utf-8"))
    return removed


def prepare() -> None:
    _, pool = natural_gap.split()
    subsets = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    documents = [pool[i] for i in subsets["r50"]["indices"]]
    segmented = segment(documents)
    every = {
        (d, i)
        for d, sentences in enumerate(segmented)
        for i, (_, targets) in enumerate(sentences)
        if targets
    }
    partial = {
        (d, i)
        for d, sentences in enumerate(segmented)
        for i, (_, targets) in enumerate(sentences)
        if targets & set(REMOVED)
    }
    class_cut = assemble(segmented, every)
    target = size(class_cut)
    corpora = {
        "class_cut": class_cut,
        "head_cut": assemble(segmented, match_bytes(segmented, partial, target, 0)),
        "control": assemble(segmented, match_bytes(segmented, set(), target, 0)),
    }
    per_head = Counter(head for sentences in segmented for _, ts in sentences for head in ts)
    sentences_total = sum(len(sentences) for sentences in segmented)
    print(
        f"r50: {len(documents)} documents, {size(documents):,} bytes, {sentences_total} "
        f"sentences; target sentences {len(every)} (all), {len(partial)} (removed heads)"
    )
    print(f"target sentences per head: {dict(per_head.most_common())}")
    CORPORA.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, corpus in corpora.items():
        (CORPORA / f"{name}.json").write_text(json.dumps(corpus))
        summary[name] = {"documents": len(corpus), "bytes": size(corpus)}
        print(f"{name}: {len(corpus)} documents, {size(corpus):,} bytes")
    divergence.write_json(
        RESULTS / "corpora.json",
        {
            "r50_bytes": size(documents),
            "sentences": sentences_total,
            "target_sentences": len(every),
            "removed_head_sentences": len(partial),
            "target_sentences_per_head": dict(per_head),
            "corpora": summary,
        },
    )


def corpus(name: str) -> list[str]:
    return cast(list[str], json.loads((CORPORA / f"{name}.json").read_text()))


def train_corpus(name: str, seed: int) -> None:
    subset_selection.train_and_score(
        corpus(name), name, seed, RUNS / f"{name}_seed{seed}", RESULTS, "class_patterns"
    )


def runs() -> dict[str, Path]:
    return {f"{n}_seed{s}": RUNS / f"{n}_seed{s}" for n in CORPUS_NAMES for s in SEEDS}


def items() -> None:
    """Items for the 14 heads, built as syntactic_eval builds them, with the slot class fixed to
    VERB: the slot these cuts remove."""
    evaluation, pool = natural_gap.split()
    word_tags, _ = syntactic_eval.tag_reference(pool, HEADS)
    classes = syntactic_eval.word_classes(word_tags)
    frequency = Counter({w: sum(t.values()) for w, t in word_tags.items() if w in classes})
    followers: dict[str, set[str]] = {head: set() for head in HEADS}
    pattern = syntactic_eval.follower_pattern(HEADS)
    for document in pool:
        for match in pattern.finditer(document):
            followers[match.group(1)].add(match.group(2).lower())
    loaded = [syntactic_eval.load_run(path) for path in runs().values()]
    tokenizable = sorted(
        w for w in classes if all(syntactic_eval.single_token(r, w) is not None for r in loaded)
    )
    prefixes = syntactic_eval.contexts(evaluation, HEADS)
    rows = {}
    for head in HEADS:
        unseen = [w for w in tokenizable if w not in followers[head]]
        pairs = syntactic_eval.pair_by_frequency(
            [w for w in unseen if classes[w] == "VERB"],
            [w for w in unseen if classes[w] != "VERB"],
            frequency,
            seed=0,
        )
        rows[head] = {"slot_class": "VERB", "pairs": pairs, "contexts": prefixes[head]}
        print(f"{head}: {len(pairs)} pairs, {len(prefixes[head])} contexts")
    divergence.write_json(RESULTS / "items.json", rows)


def class_reading(
    reading: divergence.FullReading, documents: Sequence[Sequence[int]]
) -> tuple[dict[str, float], dict[str, float]]:
    """Head-level D for the 14 heads, and class-level D with each class's heads pooled."""
    unigram = divergence.backoff_unigram(documents, reading.vocabulary)
    fillers = divergence.head_fillers(documents, reading.classes, reading.words)
    heads = {
        head: divergence.divergence(
            reading.reference[reading.tokens[head]],
            fillers.get(reading.tokens[head], Counter()),
            unigram,
        )
        for head in HEADS
    }
    pooled = {}
    for name, members in CLASSES.items():
        reference: Counter[int] = Counter()
        observed: Counter[int] = Counter()
        for head in members:
            reference.update(reading.reference[reading.tokens[head]])
            observed.update(fillers.get(reading.tokens[head], Counter()))
        pooled[name] = divergence.divergence(reference, observed, unigram)
    return heads, pooled


def summarize() -> None:
    reading = divergence.read_full()
    head_d: dict[str, dict[str, float]] = {}
    class_d: dict[str, dict[str, float]] = {}
    for name in CORPUS_NAMES:
        divergence.progress(f"Reading {name}")
        head_d[name], class_d[name] = class_reading(
            reading, framed_documents(corpus(name), reading.tokenizer, True)
        )
    delta_head = {
        name: {h: head_d[name][h] - head_d["control"][h] for h in HEADS} for name in CORPUS_NAMES
    }
    delta_class = {
        name: {c: class_d[name][c] - class_d["control"][c] for c in CLASSES}
        for name in CORPUS_NAMES
    }
    siblings = [h for h in HEADS if h not in REMOVED]
    p1 = (
        all(delta_head["head_cut"][h] >= 1 for h in REMOVED)
        and all(delta_head["class_cut"][h] >= 1 for h in HEADS)
        and all(delta_head["head_cut"][h] < 0.2 for h in siblings)
        and all(delta_class["head_cut"][c] < delta_class["class_cut"][c] / 2 for c in CLASSES)
    )
    print("head-level delta D (head_cut / class_cut):")
    for h in HEADS:
        print(f"  {h:7} {delta_head['head_cut'][h]:+.3f} / {delta_head['class_cut'][h]:+.3f}")
    print(
        f"class-level delta D: head_cut {delta_class['head_cut']}, "
        f"class_cut {delta_class['class_cut']}"
    )

    item_rows = json.loads((RESULTS / "items.json").read_text())
    others = json.loads((EXPERIMENTS / "syntactic_eval/results/items_cuts.json").read_text())
    other_items = {h: i for h, i in others["heads"].items() if head_of(h) is None}
    g: dict[str, dict[str, float]] = {}
    g_other: dict[str, dict[str, float]] = {}
    for model, path in runs().items():
        divergence.progress(f"Scoring {model}")
        run = syntactic_eval.load_run(path)
        g[model] = {
            h: float(syntactic_eval.head_reading(run, i)["g"]) for h, i in item_rows.items()
        }
        g_other[model] = {
            h: float(syntactic_eval.head_reading(run, i)["g"]) for h, i in other_items.items()
        }

    def mean_over_seeds(scores: dict[str, dict[str, float]], name: str, head: str) -> float:
        return statistics.mean(scores[f"{name}_seed{s}"][head] for s in SEEDS)

    delta_g = {
        name: {h: mean_over_seeds(g, name, h) - mean_over_seeds(g, "control", h) for h in HEADS}
        for name in ("class_cut", "head_cut")
    }
    print("G per head, mean of seeds (control / head_cut / class_cut):")
    for h in HEADS:
        print(
            f"  {h:7} {mean_over_seeds(g, 'control', h):.3f} / "
            f"{mean_over_seeds(g, 'head_cut', h):.3f} / {mean_over_seeds(g, 'class_cut', h):.3f}"
        )
    class_mean = statistics.mean(delta_g["class_cut"].values())
    removed_class = statistics.mean(delta_g["class_cut"][h] for h in REMOVED)
    removed_head = statistics.mean(delta_g["head_cut"][h] for h in REMOVED)
    siblings_head = statistics.mean(delta_g["head_cut"][h] for h in siblings)
    p2 = class_mean <= -0.10
    p3 = abs(removed_head) < abs(removed_class) / 2
    other_shift = {
        name: statistics.mean(
            abs(mean_over_seeds(g_other, name, h) - mean_over_seeds(g_other, "control", h))
            for h in other_items
        )
        for name in ("class_cut", "head_cut")
    }
    p4 = all(value <= 0.03 for value in other_shift.values())
    print(
        f"P2 class_cut mean delta G {class_mean:+.3f}\n"
        f"P3 removed heads: head_cut {removed_head:+.3f}, class_cut {removed_class:+.3f}; "
        f"siblings in head_cut {siblings_head:+.3f}\n"
        f"P4 other heads mean |delta G| {other_shift}"
    )
    predictions = {
        "1": syntactic_eval.status(p1),
        "2": syntactic_eval.status(p2),
        "3": syntactic_eval.status(p3),
        "4": syntactic_eval.status(p4),
    }
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(
        RESULTS / "summary.json",
        {
            "head_divergence": head_d,
            "class_divergence": class_d,
            "delta_head_divergence": delta_head,
            "delta_class_divergence": delta_class,
            "g": g,
            "delta_g": delta_g,
            "other_heads_mean_abs_delta_g": other_shift,
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
    training.add_argument("name", choices=CORPUS_NAMES)
    training.add_argument("seed", type=int)
    commands.add_parser("items")
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare()
        case "train":
            train_corpus(args.name, args.seed)
        case "items":
            items()
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
