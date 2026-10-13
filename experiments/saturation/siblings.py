"""Step 4, second approach: a weak pattern by the corpus's own evidence, against its siblings.

Every child of a head-tree node (a head, or a class of heads) is compared with its parent, per
filler class: had the parent's occurrences of that class fallen to its children at random, how
many distinct fillers would this child show? The expectation is exact (rarefaction without
replacement, Hurlbert 1971). A child showing far fewer than it would is weak in the corpus, even
though the corpus has no reference to say whether the domain shares that weakness.

    siblings.py read <name>      every child's fillers, and random splits of its parent's
    siblings.py summarize        weak children per corpus, and the pronoun heads and node
    siblings.py tool <name>      corpus-diagnose itself on one corpus, without a reference

Corpora are those of `saturation.py`. No command runs on import.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).parent
RESULTS = OUT / "results" / "siblings"
sys.path[:0] = [str(OUT)]

import saturation  # noqa: E402

from corpus_generalization.analysis.syntactic import (  # noqa: E402
    head_filler_matrices,
)
from corpus_generalization.diagnose import (  # noqa: E402
    ALPHA,
    _bar,
    _draws_for,
    _drop_maxima,
    _rank,
    _with_step,
    diagnose,
    prepare_reference,
)
from corpus_generalization.packing import framed_word_starts  # noqa: E402

PRONOUNS = tuple(f" {head}" for head in saturation.ANCHORS["pronoun"])
VERB = " know"
SEED = 0


def read(name: str) -> None:
    # Control's halves are read on control's tokenizer, tree and classes, so that each test of one
    # half is the same test in the other: a weak child that is the corpus's own shows in both.
    structure = "control" if name.startswith("half_") else name
    reading = prepare_reference(
        saturation.corpus(structure), head_lengths=(1,), progress=saturation.cp.divergence.progress
    )
    nodes = reading.nodes
    assert nodes is not None
    tree = nodes.tree
    size = len(tree.tokens)
    ids, starts = reading.documents, reading.word_starts
    if structure != name:
        ids, starts = framed_word_starts(saturation.corpus(name), reading.tokenizer, True)
    leaves = head_filler_matrices(ids, reading.classes, tree.tokens, nodes.fillers, starts)[
        0
    ].tocsr()
    internal = (nodes.membership @ leaves).tocsr()
    labels = np.asarray([reading.classes.classes[token] for token in nodes.fillers])
    spelled = [reading.spell(token) for token in tree.tokens]

    def row(node: int) -> np.ndarray:
        matrix = leaves if node < size else internal
        return np.asarray(matrix[node if node < size else node - size].todense()).ravel()

    members = [set(row_.indices) for row_ in nodes.membership]
    pronoun_leaves = {leaf for leaf, text in enumerate(spelled) if text in PRONOUNS}
    holding = [n for n, leaves_ in enumerate(members) if len(leaves_ & pronoun_leaves) >= 4]
    pronoun_node = size + min(holding, key=lambda n: len(members[n])) if holding else -1
    verb_class = reading.classes.classes.get(
        next((t for t in nodes.fillers if reading.spell(t) == VERB), -1), -1
    )
    rng = np.random.default_rng(SEED)
    draws = _draws_for(ALPHA)
    shown_all: list[int] = []
    trunk_all: list[float] = []
    leaf_all: list[int] = []
    child_all: list[str] = []
    class_all: list[int] = []

    def label_of(child: int) -> str:
        if child < size:
            return spelled[child]
        heads = sorted(members[child - size], key=lambda leaf: -leaves[leaf].sum())
        return "|".join(spelled[leaf] for leaf in heads[:6]) + f" ({len(heads)})"

    null: list[np.ndarray] = []
    targets: dict[str, int] = {}
    # Only children of a node inside one class: above the class cut, a random split of the parent
    # mixes categories (`,`/`the`/`to` against `.`/`:`), and every such test read as weak.
    inside = size - reading.classes.n_classes
    keys: list[str] = []
    for parent, (left, right, _, _) in enumerate(tree.merges[:inside]):
        parent_row = row(size + parent)
        children = (int(left), int(right))
        child_rows = [row(child) for child in children]
        for label in np.unique(labels[parent_row > 0]):
            columns = (labels == label) & (parent_row > 0)
            bag = parent_row[columns]
            trunks = [round(float(child[columns].sum())) for child in child_rows]
            if min(trunks) < 1:
                continue
            for child, child_row, trunk in zip(children, child_rows, trunks, strict=True):
                # The exact null: `trunk` of the parent's occurrences, without replacement.
                # Independent presences, tried first, overstated the spread: ` they` showing 3
                # verbs where 26.5 were due could not have been flagged even with one.
                split = rng.multivariate_hypergeometric(
                    bag.astype(np.int64), trunk, size=draws, method="marginals"
                )
                null.append((split > 0).sum(axis=1))
                shown_all.append(int((child_row[columns] > 0).sum()))
                trunk_all.append(float(trunk))
                leaf_all.append(int(child < size))
                child_all.append(label_of(child))
                keys.append(f"{parent}:{child}:{label}")
                class_all.append(int(label))
                name_ = (
                    spelled[child]
                    if child < size and child in pronoun_leaves
                    else ("pronoun node" if child == pronoun_node else None)
                )
                if name_ is not None and label == verb_class:
                    targets[name_] = len(shown_all) - 1
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS / f"{name}.npz",
        shown=np.asarray(shown_all),
        trunk=np.asarray(trunk_all),
        leaf=np.asarray(leaf_all),
        child=np.asarray(child_all),
        filler_class=np.asarray(class_all),
        key=np.asarray(keys),
        null=np.asarray(null, dtype=np.int32),
    )
    top = {
        int(label): [
            reading.spell(token)
            for token in sorted(
                (t for t in nodes.fillers if reading.classes.classes[t] == label),
                key=lambda t: -reading.frequency[t],
            )[:6]
        ]
        for label in np.unique(labels)
    }
    (RESULTS / f"{name}.json").write_text(json.dumps({"targets": targets, "classes": top}))


def verdicts(name: str) -> dict[str, Any]:
    """Weak children: fewer fillers than the bar of random splits allows, read only where a
    single filler could be flagged (Tarone), as `diagnose` reads its draws."""
    data = np.load(RESULTS / f"{name}.npz")
    shown = data["shown"].astype(np.float64)
    null = data["null"].astype(np.float64).T  # draws x tests
    rank = _rank(ALPHA, null.shape[0])
    mean, sd = null.mean(axis=0), _with_step(null.std(axis=0, ddof=1), np.ones(null.shape[1]))
    readable = np.ones(len(shown), dtype=bool)
    for _ in range(10):
        bar = _bar(_drop_maxima(null[:, readable], null.shape[0], np.ones(readable.sum())), rank)
        # A child showing a single filler is the furthest one can fall.
        now = (mean - 1) / sd > bar
        if (now == readable).all():
            break
        readable = now
    t = (mean - shown) / sd
    weak = readable & (t > bar)
    meta = json.loads((RESULTS / f"{name}.json").read_text())
    targets = meta["targets"]
    order = np.argsort(-np.where(weak, t, -np.inf))[: int(weak.sum())]
    examples = [
        (
            str(data["child"][index]),
            meta["classes"][str(int(data["filler_class"][index]))][:4],
            int(shown[index]),
            round(float(mean[index]), 1),
            float(data["trunk"][index]),
        )
        for index in order
    ]
    return {
        "tests": len(shown),
        "readable": int(readable.sum()),
        "bar": round(bar, 2),
        "weak": int(weak.sum()),
        "weak_leaves": int((weak & (data["leaf"] == 1)).sum()),
        "examples": examples[:15],
        "weak_tests": sorted(str(data["key"][index]) for index in order),
        "weak_keys": sorted(
            f"{data['child'][index]} -> {meta['classes'][str(int(data['filler_class'][index]))][0]}"
            for index in order
        ),
        "targets": {
            label: {
                "shown": int(shown[index]),
                "expected": round(float(mean[index]), 1),
                "trunk": float(data["trunk"][index]),
                "t": round(float(t[index]), 2),
                "readable": bool(readable[index]),
                "weak": bool(weak[index]),
            }
            for label, index in sorted(targets.items())
        },
    }


def summarize() -> None:
    summary = {path.stem: verdicts(path.stem) for path in sorted(RESULTS.glob("*.npz"))}
    for name, values in summary.items():
        print(name, json.dumps(values))
    saturation.cp.divergence.write_json(RESULTS.parent / "siblings_summary.json", summary)


def tool(name: str) -> None:
    """`corpus-diagnose` itself, without a reference, at heads of one token."""
    started = time.time()
    report = diagnose(
        saturation.corpus(name), head_lengths=(1,), progress=saturation.cp.divergence.progress
    )
    flagged = [row for row in report.rows if row.verdict in ("weak", "absent")]
    pronouns = {
        f"{row.head}|{row.verdict}|{row.top_tokens[:3]}|{row.trunk}"
        for row in flagged
        if row.head in PRONOUNS and not row.lexical
    }
    saturation.cp.divergence.write_json(
        RESULTS.parent / "tool" / f"{name}.json",
        {
            "seconds": round(time.time() - started),
            "counts": report.counts,
            "lexical_counts": report.lexical_counts,
            "bars": [report.head_critical_t, report.absent_critical_t, report.class_critical_t],
            "class_rows": len(report.class_rows),
            "class_nodes_read": report.class_nodes_read,
            "pronoun_rows": sorted(pronouns),
            "pronoun_class_rows": [
                [
                    list(row.heads[:6]),
                    list(row.top_fillers[:3]),
                    row.trunk,
                    round(row.margin_mean, 1),
                    round(row.t, 1),
                ]
                for row in report.class_rows
                if {head.strip() for head in row.heads} & {"they", "she", "he", "we"}
                and not row.lexical
            ],
            "flags": sorted(
                f"{row.head} -> {row.top_tokens[0] if row.top_tokens else ''} {row.verdict}"
                for row in flagged
                if not row.lexical
            ),
            "examples": [
                [row.head, row.verdict, list(row.top_tokens[:3]), row.trunk, row.fillers]
                for row in flagged
                if not row.lexical
            ][:25],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("read")
    one.add_argument("name")
    run = commands.add_parser("tool")
    run.add_argument("name")
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "read":
            read(args.name)
        case "tool":
            tool(args.name)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
