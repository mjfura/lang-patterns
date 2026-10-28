"""The tool, second pass: absent patterns and short heads in a class, on fresh cuts.

"The tool, second pass" in `docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md`
records the design and predictions before running. Corpus side only: no model is trained.

    tool_check.py prepare            numbers and base-verb cuts from r50, classes from the tool
    tool_check.py diagnose <name>    corpus-diagnose on one cut against F
    tool_check.py summarize          the four predictions over the fresh and earlier cuts

No command runs on import.
"""

import argparse
import json
from collections import Counter
from typing import Any

import class_patterns as base
import replication
from class_patterns import divergence, natural_gap, subset_selection, syntactic_eval

from corpus_generalization.analysis.syntactic import induce_classes

FRESH = {
    "num": {"anchor": "two", "follower": "NOUN", "removed": ("five", "six")},
    "verb": {"anchor": "make", "follower": "DET", "removed": ("make", "get")},
}
FRESH_NAMES = tuple(f"{k}_{c}" for k in FRESH for c in ("class_cut", "head_cut", "control"))
EARLIER_CONTROLS = ("control", "poss_control")
ALL_NAMES = (*FRESH_NAMES, *replication.TOOL_CORPORA)


def class_members(anchor: str) -> tuple[str, ...]:
    """The heads of the tool's 64-class cut on F that share a class with `anchor`."""
    reading = divergence.read_full()
    tree = induce_classes(reading.full, n_classes=64, candidates=reading.words)
    label = tree.classes[reading.tokens[anchor]]
    return tuple(
        sorted(
            text
            for token, cls in tree.classes.items()
            if cls == label and (text := reading.tokenizer.decode([token]).strip()).isalpha()
        )
    )


def prepare() -> None:
    _, pool = natural_gap.split()
    subsets = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    documents = [pool[i] for i in subsets["r50"]["indices"]]
    record: dict[str, object] = {}
    for kind, spec in FRESH.items():
        members = class_members(str(spec["anchor"]))
        lowered = tuple(dict.fromkeys(m.lower() for m in members))
        print(f"{kind}: {len(lowered)} heads {lowered}")
        segmented = base.segment(documents, lowered, str(spec["follower"]))
        every = {(d, i) for d, ss in enumerate(segmented) for i, (_, ts) in enumerate(ss) if ts}
        removed = set(spec["removed"])
        partial = {
            (d, i) for d, ss in enumerate(segmented) for i, (_, ts) in enumerate(ss) if ts & removed
        }
        class_cut = base.assemble(segmented, every)
        target = base.size(class_cut)
        corpora = {
            f"{kind}_class_cut": class_cut,
            f"{kind}_head_cut": base.assemble(
                segmented, base.match_bytes(segmented, partial, target, 0)
            ),
            f"{kind}_control": base.assemble(
                segmented, base.match_bytes(segmented, set(), target, 0)
            ),
        }
        for name, corpus in corpora.items():
            (base.CORPORA / f"{name}.json").write_text(json.dumps(corpus))
            print(f"{name}: {len(corpus)} documents, {base.size(corpus):,} bytes")
        record[kind] = {
            "heads": lowered,
            "target_sentences": len(every),
            "removed_head_sentences": len(partial),
            "bytes": base.size(class_cut),
        }
        per_head = Counter(h for ss in segmented for _, ts in ss for h in ts)
        print(
            f"{kind}: {len(every)} target sentences, {len(partial)} of the removed heads; "
            f"per head {dict(per_head.most_common())}"
        )
    divergence.write_json(base.RESULTS / "tool_check_corpora.json", record)


type Row = dict[str, Any]


def class_row(rows: list[Row], head: str) -> Row | None:
    return next((r for r in rows if head in [h.strip() for h in r["heads"]]), None)


def short_heads(row: Row | None) -> set[str]:
    return {h.strip() for h in row["short_heads"]} if row else set()


def summarize() -> None:
    def load(name: str) -> dict[str, list[Row]]:
        return json.loads((base.RESULTS / f"tool_{name}.json").read_text())

    held = {"1": True, "2": True, "3": True, "4": True}
    report: dict[str, object] = {}
    for name in ALL_NAMES:
        data = load(name)
        flagged = [r for r in data["class_rows"] if r["verdict"] == "under-shown"]
        absent = data.get("absent", [])
        print(
            f"{name:16} class rows under-shown: "
            f"{[[h.strip() for h in r['heads']][:5] for r in flagged]}; absent head rows "
            f"{len(absent)}"
        )
        report[name] = {"flagged": flagged, "absent": absent}
        if name.endswith("control"):
            held["1"] &= not flagged and len(absent) <= 2
    for kind, spec in FRESH.items():
        anchor = str(spec["anchor"])
        cut = class_row(load(f"{kind}_class_cut")["class_rows"], anchor)
        head = class_row(load(f"{kind}_head_cut")["class_rows"], anchor)
        short = short_heads(head)
        held["2"] &= cut is not None and cut["verdict"] == "under-shown"
        held["3"] &= (
            head is not None and head["verdict"] != "under-shown" and set(spec["removed"]) <= short
        )
        for label, row in (("class cut", cut), ("head cut", head)):
            if row:
                print(
                    f"{kind} {label}: {row['verdict']}, D {row['divergence']:.3f}, excess "
                    f"{row['excess']:+.3f}, short heads "
                    f"{sorted(short_heads(row))}"
                )
    poss = class_row(load("poss_class_cut")["class_rows"], "his")
    poss_head = class_row(load("poss_head_cut")["class_rows"], "his")
    poss_short = short_heads(poss)
    poss_head_short = short_heads(poss_head)
    held["4"] &= len(poss_short & set(replication.POSSESSIVES)) >= 5
    held["4"] &= {"his", "their"} <= poss_head_short
    print(f"poss class cut short heads: {sorted(poss_short)}")
    print(f"poss head cut short heads: {sorted(poss_head_short)}")
    predictions = {k: syntactic_eval.status(v) for k, v in held.items()}
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(
        base.RESULTS / "tool_check_summary.json", {"cuts": report, "predictions": predictions}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    tool = commands.add_parser("diagnose")
    tool.add_argument("name", choices=ALL_NAMES)
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
