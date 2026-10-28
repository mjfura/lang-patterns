"""Held-out information gain on the corpora already built: does it separate what the graph curves
could not, and is it comparable across domains and merge budgets?

Targets: the contamination variants (seed 0, `experiments/contamination`) and the clean sources'
first samples (`experiments/clean_domains`), each at `BUDGETS` merges. See
`docs/superpowers/specs/2026-09-28-heldout-information-design.md` for the measure.

    information.py contamination <kind> <dose> [null]
    information.py clean <source> [null]

`null` scores the target's word-shuffled null (`experiments/null_model`) instead: the order
signal is how much better the corpus is predicted than its own null.
    information.py summarize
"""

import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

from corpus_generalization.analysis.information import heldout_information

OUT = Path(__file__).parent
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "null_model")]

import null_model  # noqa: E402

BUDGETS = (1000, 4000, 16000)
KINDS = ("control", "duplicate", "python", "shuffled")
SOURCES = ("fineweb", "fineweb-edu", "c4", "wikipedia", "multi-nli", "pubmed", "arxiv", "case-law")


def null(documents: list[str]) -> list[str]:
    """The word-shuffled null, with every copy of a text shuffled the same way, so the null keeps
    the corpus's duplicates (`null_model.null` shuffles each copy differently, which turns copies
    into distinct texts and gives the null more distinct data than the corpus has)."""
    shuffled = []
    for document in documents:
        seed = int.from_bytes(hashlib.sha256(document.encode()).digest()[:8], "big")
        words = document.split()
        random.Random(seed).shuffle(words)
        shuffled.append(" ".join(words))
    return shuffled


def score(name: str, documents: list[str]) -> None:
    rows = []
    for budget in BUDGETS:
        start = time.time()
        information = heldout_information(documents, budget)
        row = {"target": name, "merge_budget": budget, **asdict(information)}
        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / f"{name}_results.json").write_text(json.dumps(rows, indent=1) + "\n")


def _rows(name: str) -> dict[int, dict[str, object]] | None:
    path = OUT / f"{name}_results.json"
    if not path.exists():
        return None
    return {row["merge_budget"]: row for row in json.loads(path.read_text())}


def summarize() -> None:
    def line(name: str, rows: dict[int, dict[str, object]]) -> str:
        cells = []
        for budget in BUDGETS:
            row = rows[budget]
            gain, gap, bpb = row["gain"], row["gap"], row["heldout_bpb"]
            cells.append(
                f"{bpb['0']:.3f} {gain['1']:.3f} {gain['2']:.3f} {gain['3']:.3f} g3={gap['3']:.3f}"
            )
        return f"{name:28} " + " | ".join(cells)

    print(f"{'target':28} per budget {BUDGETS}: bpb0 gain1 gain2 gain3 gap3")
    for target in (
        "contamination-base-0.0",
        *(f"contamination-{k}-{d}" for k in KINDS for d in (0.1, 0.3, 0.5)),
        *(f"clean-{s}" for s in SOURCES),
    ):
        rows = _rows(target)
        if rows:
            print(line(target, rows))


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["contamination", kind, dose, *flags]:
            name = f"contamination-{kind}-{float(dose)}"
            documents = null_model.contamination_documents(kind, float(dose))
        case ["clean", source, *flags]:
            name, documents = f"clean-{source}", null_model.clean_documents(source)
        case _:
            summarize()
            sys.exit(0)
    if flags == ["null"]:
        name, documents = f"{name}-null", null(documents)
    score(name, documents)
