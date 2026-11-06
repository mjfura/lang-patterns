"""Do the Ward classes match parts of speech?

`docs/superpowers/specs/2026-11-05-class-purity-design.md` records the design and predictions,
written before running. Corpus side only: no model is trained.

Each domain's pool #1 at 1x (14.4 MB) from `experiments/granularity_robustness`, tokenized at its
1.50 budget, clustered and cut as `diagnose` does. spaCy tags the same pool; the cut and the tree
are compared with the tags of whole-word tokens. No command runs on import.

    uv run python experiments/class_purity/class_purity.py read <corpus>
    uv run python experiments/class_purity/class_purity.py summarize
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

OUT = Path(__file__).parent
RESULTS = OUT / "results"
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "granularity_robustness"), str(EXPERIMENTS / "syntactic_eval")]

import granularity_robustness as robustness  # noqa: E402
from syntactic_eval import WORD_OCCURRENCES, WORD_SHARE  # noqa: E402

from corpus_generalization.analysis import syntactic  # noqa: E402
from corpus_generalization.packing import framed_word_starts  # noqa: E402
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402

CORPORA = robustness.CORPORA
POOL = 1
TARGET = "1.50"
CUT_PERMUTATIONS = 1000
TREE_PERMUTATIONS = 200
# The tags prediction 3 names; any tag with enough tokens is read.
NAMED_TAGS = ("NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON")
CUTS = range(8, 65)


def progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def word_tag(counts: Counter[str]) -> str | None:
    """A word's majority universal tag, when it holds `WORD_SHARE` of at least
    `WORD_OCCURRENCES` occurrences."""
    total = sum(counts.values())
    if total < WORD_OCCURRENCES:
        return None
    tag, count = counts.most_common(1)[0]
    return tag if count / total >= WORD_SHARE else None


def purity(classes: np.ndarray, tags: np.ndarray) -> float:
    """The share of items whose class's majority tag is their own."""
    pairs = Counter(zip(classes.tolist(), tags.tolist(), strict=True))
    best: dict[int, int] = defaultdict(int)
    for (label, _), count in pairs.items():
        best[label] = max(best[label], count)
    return sum(best.values()) / len(tags)


def best_node(
    membership: sparse.csr_matrix, leaves: np.ndarray, tags: np.ndarray, tag: str
) -> tuple[float, int]:
    """The internal node whose evaluated leaves best match `tag`'s (F1), and that F1.
    `leaves` are the evaluated leaves' columns, `tags` their tags."""
    evaluated = membership[:, leaves]
    hits = np.asarray(evaluated @ (tags == tag).astype(float)).ravel()
    sizes = np.asarray(evaluated.sum(axis=1)).ravel()
    f1 = 2 * hits / (sizes + (tags == tag).sum())
    node = int(np.argmax(f1))
    return float(f1[node]), node


def rank_of(by_k: dict[int, float], chosen: int) -> int:
    """1 + the number of cuts strictly better than the chosen one."""
    return 1 + sum(value > by_k[chosen] for value in by_k.values())


def tag_pool(pool: Sequence[str]) -> dict[str, Counter[str]]:
    """Per word after whitespace, case kept, its spaCy universal tags."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.max_length = max(len(document) for document in pool) + 1
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for index, doc in enumerate(nlp.pipe(pool, batch_size=32, n_process=4)):
        if index % 1000 == 0:
            progress(f"Tagged {index} of {len(pool)} documents")
        for position in range(1, len(doc)):
            token = doc[position]
            if doc[position - 1].whitespace_ and token.text.isalpha():
                counts[token.text][token.pos_] += 1
    return counts


def read(corpus: str) -> None:
    name = f"{corpus}@1#{POOL}"
    variant = robustness.parse_variant(name)
    pool = robustness.pool_of(variant)
    record = robustness.prepared(name)
    assert record is not None, f"{name} has not been prepared"
    budget = record["budgets"][TARGET]
    tokenizer = ByteLevelBPETokenizer.from_merges(robustness.merges_of(name)[:budget])
    progress(f"{name}: {len(pool)} documents, {budget} merges")
    framed, _ = framed_word_starts(pool, tokenizer, True)
    clustered = robustness.domains.clustered_tokens(pool, tokenizer)
    with robustness.clustering(clustered):
        tree = syntactic.class_tree(framed)
    classes = tree.cut()
    progress(f"{name}: {len(tree.tokens)} tokens, {classes.n_classes} classes")

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    word_like = syntactic.word_like_tokens(framed, spell, (tokenizer.bos_id, tokenizer.eos_id))
    tags_of = {word: tag for word, counts in tag_pool(pool).items() if (tag := word_tag(counts))}
    frequency = Counter(token for sequence in framed for token in sequence)
    leaves, tags = [], []
    for leaf, token in enumerate(tree.tokens):
        text = spell(token)
        if token in word_like and (tag := tags_of.get(text[1:])) is not None:
            leaves.append(leaf)
            tags.append(tag)
    leaf_array, tag_array = np.array(leaves), np.array(tags)
    labels = np.array([classes.classes[tree.tokens[leaf]] for leaf in leaves])
    rng = np.random.default_rng(0)

    cut_purity = purity(labels, tag_array)
    cut_ari = float(adjusted_rand_score(tag_array, labels))
    null_purity, null_ari = [], []
    for _ in range(CUT_PERMUTATIONS):
        shuffled = rng.permutation(tag_array)
        null_purity.append(purity(labels, shuffled))
        null_ari.append(float(adjusted_rand_score(shuffled, labels)))

    membership = syntactic.tree_membership(tree)
    tag_sizes = Counter(tags)
    per_tag: dict[str, Any] = {}
    nulls = [rng.permutation(tag_array) for _ in range(TREE_PERMUTATIONS)]
    for tag, size in tag_sizes.most_common():
        if size < WORD_OCCURRENCES:
            continue
        f1, node = best_node(membership, leaf_array, tag_array, tag)
        node_leaves = membership[node].indices
        node_evaluated = np.isin(leaf_array, node_leaves)
        per_tag[tag] = {
            "tokens": size,
            "best_f1": f1,
            "node_leaves": len(node_leaves),
            "node_evaluated": int(node_evaluated.sum()),
            "node_hits": int((node_evaluated & (tag_array == tag)).sum()),
            "null_max_f1": max(best_node(membership, leaf_array, n, tag)[0] for n in nulls),
            "examples": [spell(tree.tokens[leaf]) for leaf in leaf_array[tag_array == tag][:12]],
        }
    ari_by_k = {}
    for k in CUTS:
        if k <= len(tree.tokens):
            cut = tree.cut(k)
            ari_by_k[k] = float(
                adjusted_rand_score(tag_array, [cut.classes[tree.tokens[leaf]] for leaf in leaves])
            )
    composition = {}
    for label in range(classes.n_classes):
        members = [
            leaf for leaf, token in enumerate(tree.tokens) if classes.classes[token] == label
        ]
        in_class = labels == label
        composition[label] = {
            "tokens": len(members),
            "evaluated": int(in_class.sum()),
            "tags": dict(Counter(tag_array[in_class].tolist()).most_common()),
            "top": [
                spell(tree.tokens[leaf])
                for leaf in sorted(members, key=lambda m: -frequency[tree.tokens[m]])[:12]
            ],
        }
    running = sum(frequency[tree.tokens[leaf]] for leaf in leaves)
    total = sum(count for token, count in frequency.items() if token > 4)
    result = {
        "pool": name,
        "merges": budget,
        "tokens": len(tree.tokens),
        "n_classes": classes.n_classes,
        "evaluated": len(leaves),
        "running_share": running / total,
        "tag_sizes": dict(tag_sizes.most_common()),
        "purity": cut_purity,
        "null_purity": {"mean": float(np.mean(null_purity)), "max": float(np.max(null_purity))},
        "ari": cut_ari,
        "null_ari": {"mean": float(np.mean(null_ari)), "max": float(np.max(null_ari))},
        "per_tag": per_tag,
        "ari_by_k": ari_by_k,
        "chosen_rank": rank_of(ari_by_k, classes.n_classes),
        "best_k": max(ari_by_k, key=lambda k: ari_by_k[k]),
        "composition": composition,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{corpus}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    progress(f"{name}: purity {cut_purity:.3f}, ARI {cut_ari:.3f}")


def summarize() -> None:
    results = {corpus: json.loads((RESULTS / f"{corpus}.json").read_text()) for corpus in CORPORA}
    quarter = len(CUTS) / 4
    named = {
        tag: sum(
            tag in r["per_tag"] and r["per_tag"][tag]["best_f1"] > r["per_tag"][tag]["null_max_f1"]
            for r in results.values()
        )
        for tag in NAMED_TAGS
    }
    predictions = {
        "1": all(r["purity"] > r["null_purity"]["max"] for r in results.values()),
        "2": all(r["ari"] > r["null_ari"]["max"] for r in results.values()),
        "3": all(count >= 4 for count in named.values()),
        "4": sum(r["chosen_rank"] <= quarter for r in results.values()) >= 3,
    }
    summary = {
        "domains": {
            corpus: {
                key: r[key]
                for key in (
                    "n_classes",
                    "evaluated",
                    "running_share",
                    "purity",
                    "null_purity",
                    "ari",
                    "null_ari",
                    "chosen_rank",
                    "best_k",
                )
            }
            | {"best_ari": r["ari_by_k"][str(r["best_k"])]}
            | {"best_f1": {tag: v["best_f1"] for tag, v in r["per_tag"].items()}}
            for corpus, r in results.items()
        },
        "named_tags_beating_null": named,
        "predictions": predictions,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("read").add_argument("corpus", choices=CORPORA)
    commands.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "read":
        read(args.corpus)
    else:
        summarize()


if __name__ == "__main__":
    main()
