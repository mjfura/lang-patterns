"""`absent` at a quarter of the draws' trunk, on fresh cuts.

"`absent` at a quarter of the draws' trunk" in `docs/2026-09-28-syntactic-sufficiency-design.md`
records the predictions before running. The m/4 reading filters the m/2 rule's rows, which
record the corpus trunk t and the draws' mean m. Corpus side only.

    absent_refine.py prepare            ctrl_c, ctrl_d, drop4b from r50; r25d, r25e from F
    absent_refine.py diagnose <name>    corpus-diagnose on one cut against F
    absent_refine.py summarize          the three predictions

No command runs on import.
"""

import argparse
import json
import random
import re
from typing import Any

import absent_check
import class_patterns as base
import replication
from class_patterns import divergence, natural_gap, subset_selection, syntactic_eval

WORDS = ("although", "within", "various", "whose")
CONTROLS = ("ctrl_c", "ctrl_d", "r25d", "r25e")
NAMES = (*CONTROLS, "drop4b")


def prepare() -> None:
    _, pool = natural_gap.split()
    subsets = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    documents = [pool[i] for i in subsets["r50"]["indices"]]
    segmented = base.segment(documents, (), "VERB")
    corpora = {
        name: base.assemble(segmented, base.match_bytes(segmented, set(), target, seed))
        for name, (target, seed) in {"ctrl_c": (4_000_000, 3), "ctrl_d": (6_000_000, 4)}.items()
    }
    pattern = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)
    dropped = {
        (d, i)
        for d, sentences in enumerate(segmented)
        for i, (text, _) in enumerate(sentences)
        if pattern.search(text)
    }
    corpora["drop4b"] = base.assemble(segmented, dropped)
    sizes = [len(d.encode("utf-8")) for d in pool]
    for name, seed in (("r25d", 206), ("r25e", 207)):
        order = list(range(len(pool)))
        random.Random(seed).shuffle(order)
        corpora[name] = [pool[i] for i in subset_selection.fill(sizes, order, 0.25 * sum(sizes))]
    for name, corpus in corpora.items():
        (base.CORPORA / f"{name}.json").write_text(json.dumps(corpus))
        print(f"{name}: {len(corpus)} documents, {base.size(corpus):,} bytes")
    print(f"drop4b removed {len(dropped)} sentences")


def at_quarter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["trunk"] <= r["draw_trunk_mean"] / 4]


def summarize() -> None:
    def absent(name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = json.loads((base.RESULTS / f"tool_{name}.json").read_text())[
            "absent"
        ]
        return rows

    held = {"1": True, "2": True, "3": True}
    for name in CONTROLS:
        rows = absent(name)
        quarter = at_quarter(rows)
        held["1"] &= len(quarter) <= 2
        print(f"{name:8} absent rows: m/2 {len(rows)}, m/4 {len(quarter)}")
    absent_check.WORDS = WORDS
    expected = absent_check.qualifying("drop4b")
    rows = absent("drop4b")
    quarter = at_quarter(rows)
    flagged = {f"{r['head']}|{r['filler_class']}" for r in quarter}
    missed = sorted(set(expected) - flagged)
    others = [r for r in quarter if not any(w in r["head"].lower() for w in WORDS)]
    held["2"] &= not missed
    held["3"] &= len(others) <= 5
    print(
        f"drop4b: m/2 {len(rows)}, m/4 {len(quarter)} absent rows; {len(expected)} qualifying, "
        f"missed {missed}; m/4 rows outside the words {len(others)}: "
        f"{[r['head'] for r in others][:10]}"
    )
    predictions = {k: syntactic_eval.status(v) for k, v in held.items()}
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(
        base.RESULTS / "absent_refine_summary.json",
        {
            "controls": {
                n: {"m2": len(absent(n)), "m4": len(at_quarter(absent(n)))} for n in CONTROLS
            },
            "drop4b": {
                "m2": len(rows),
                "m4": len(quarter),
                "missed": missed,
                "outside": len(others),
            },
            "predictions": predictions,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    tool = commands.add_parser("diagnose")
    tool.add_argument("name", choices=NAMES)
    commands.add_parser("summarize")
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare()
        case "diagnose":
            replication.tool_diagnose(args.name)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
