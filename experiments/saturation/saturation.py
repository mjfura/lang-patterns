"""Step 4 of the settings-from-data path: class patterns without a reference, by saturation.

Without a reference a corpus can still say whether it has shown a class pattern enough times to
pin down its variety. The slope of a slot's filler-accumulation curve at the full corpus is,
in expectation, its Good-Turing missing mass: fillers seen once over the slot's trunk. A slot
still discovering fillers is under-shown by the corpus's own evidence. A slot that was removed
outright is not: it shows few fillers, again and again, like a restriction of the domain.

The cuts thin the pronoun -> VERB pattern of `experiments/class_patterns` instead of removing it:
`thin_<p>` keeps a share p of its target sentences, at the bytes of `class_cut`, which keeps none.
Each corpus is read on its own, at 1.50 tokens per word, with no reference. Corpus side only.

    saturation.py segment              segment and tag r50 once, into runs/saturation/
    saturation.py read <name>          missing mass at every head-tree node of one corpus
    saturation.py summarize            the pronoun and modal nodes across corpora

No command runs on import.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "saturation"
sys.path[:0] = [str(OUT.parent / "class_patterns")]

import class_patterns as cp  # noqa: E402

from corpus_generalization.analysis.syntactic import head_filler_matrices  # noqa: E402
from corpus_generalization.diagnose import prepare_reference  # noqa: E402

SHARES = (0.05, 0.1, 0.25, 0.5)
ANCHORS = {"pronoun": cp.CLASSES["pronoun"], "modal": cp.CLASSES["modal"]}
SEED = 0


def segment() -> None:
    _, pool = cp.natural_gap.split()
    subsets = json.loads((cp.subset_selection.RESULTS / "subsets.json").read_text())
    documents = [pool[i] for i in subsets["r50"]["indices"]]
    segmented = cp.segment(documents)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "segmented.json").write_text(
        json.dumps([[[text, sorted(targets)] for text, targets in doc] for doc in segmented])
    )


def _segmented() -> list[list[tuple[str, frozenset[str]]]]:
    raw = json.loads((RUNS / "segmented.json").read_text())
    return [[(text, frozenset(targets)) for text, targets in doc] for doc in raw]


def corpus(name: str) -> list[str]:
    """`thin_<p>` and `half_a`/`half_b` (control's documents split at random), built here; any
    other name is a corpus of `experiments/class_patterns`."""
    if name.startswith("half_"):
        documents = cp.corpus("control")
        order = list(range(len(documents)))
        random.Random(SEED).shuffle(order)
        half = order[: len(order) // 2] if name == "half_a" else order[len(order) // 2 :]
        return [documents[index] for index in sorted(half)]
    if not name.startswith("thin_"):
        return cp.corpus(name)
    share = float(name.removeprefix("thin_"))
    segmented = _segmented()
    pronouns = set(ANCHORS["pronoun"])
    targets = [
        (d, i)
        for d, sentences in enumerate(segmented)
        for i, (_, heads) in enumerate(sentences)
        if heads
    ]
    # Only pronoun sentences are thinned; the modal ones are removed as in `class_cut`, so the
    # modal node is the removed-pattern case in every corpus.
    thinned = [(d, i) for d, i in targets if segmented[d][i][1] & pronouns]
    kept = set(random.Random(SEED).sample(thinned, round(share * len(thinned))))
    kept = {(d, i) for d, i in kept if not segmented[d][i][1] - pronouns}
    removed = set(targets) - kept
    target = cp.size(cp.corpus("class_cut"))
    return cp.assemble(segmented, cp.match_bytes(segmented, removed, target, SEED))


def _row(counts: np.ndarray) -> dict[str, float]:
    """Trunk, distinct fillers and fillers seen once, of one slot's filler counts."""
    values = counts[counts > 0]
    return {
        "trunk": float(values.sum()),
        "fillers": len(values),
        "singletons": int((values == 1).sum()),
    }


def read(name: str) -> None:
    documents = corpus(name)
    reading = prepare_reference(documents, head_lengths=(1,), progress=cp.divergence.progress)
    nodes = reading.nodes
    assert nodes is not None
    counts = nodes.reference[0].tocsr()
    trunk = np.asarray(counts.sum(axis=1)).ravel()
    distinct = np.diff(counts.indptr)
    singletons = np.add.reduceat(
        np.r_[(counts.data == 1).astype(np.int64), 0], counts.indptr[:-1]
    ) * (distinct > 0)
    # The anchors as heads, pooled: the same space-led tokens in every corpus. The tree's own
    # node for them moves between corpora once their contexts change.
    leaves = head_filler_matrices(
        reading.documents, reading.classes, nodes.tree.tokens, nodes.fillers, reading.word_starts
    )[0].tocsr()
    spelled = [reading.spell(token) for token in nodes.tree.tokens]
    anchors = {}
    for label, heads in ANCHORS.items():
        rows = [leaf for leaf, text in enumerate(spelled) if text in {f" {h}" for h in heads}]
        pooled = leaves[rows].sum(axis=0).A1
        shown = np.flatnonzero(pooled)
        anchors[label] = (
            {"heads": sorted(spelled[row] for row in rows)}
            | _row(pooled)
            | {
                "row": {
                    reading.spell(nodes.fillers[column]): [
                        float(pooled[column]),
                        reading.classes.classes[nodes.fillers[column]],
                    ]
                    for column in shown
                }
            }
        )
    read_nodes = trunk >= 2
    cp.divergence.write_json(
        RESULTS / f"{name}.json",
        {
            "name": name,
            "documents": len(documents),
            "bytes": cp.size(documents),
            "merge_budget": reading.merge_budget,
            "tokens_per_word": reading.tokens_per_word,
            "anchors": anchors,
            "nodes": {
                "trunk": trunk[read_nodes].tolist(),
                "fillers": distinct[read_nodes].tolist(),
                "singletons": singletons[read_nodes].tolist(),
            },
        },
    )


def summarize() -> None:
    summary: dict[str, Any] = {}
    for path in sorted(RESULTS.glob("*.json")):
        if path.stem == "summary":
            continue
        data = json.loads(path.read_text())
        trunk = np.asarray(data["nodes"]["trunk"])
        mass = np.asarray(data["nodes"]["singletons"]) / trunk
        row: dict[str, Any] = {"tokens_per_word": round(data["tokens_per_word"], 3)}
        for label, anchor in cast(dict[str, dict[str, Any]], data["anchors"]).items():
            missing = anchor["singletons"] / anchor["trunk"]
            # Its standing among nodes of similar trunk (within a factor of two): missing mass
            # falls with trunk everywhere, so only a like-sized node is a fair comparison.
            similar = (trunk >= anchor["trunk"] / 2) & (trunk <= anchor["trunk"] * 2)
            row[label] = {
                "heads": len(anchor["heads"]),
                "trunk": anchor["trunk"],
                "fillers": anchor["fillers"],
                "missing_mass": round(missing, 4),
                "share_of_similar_nodes_below": round(float((mass[similar] < missing).mean()), 3),
                "similar_nodes": int(similar.sum()),
            }
        row["missing_mass_quantiles"] = {
            str(q): round(float(np.quantile(mass, q)), 4) for q in (0.5, 0.9, 0.99)
        }
        summary[path.stem] = row
        print(path.stem, json.dumps(row))
    cp.divergence.write_json(RESULTS / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("segment")
    one = commands.add_parser("read")
    one.add_argument("name")
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "segment":
            segment()
        case "read":
            read(args.name)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
