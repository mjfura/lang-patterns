"""The `absent` rule against the draws' trunk noise, on cuts never read with `absent`.

"The `absent` rule against the draws' trunk noise" in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` records the design and
predictions before the rule changed. Corpus side only: no model is trained.

    absent_check.py prepare            ctrl_a, ctrl_b and drop4 from r50; r25b and r25c
    absent_check.py diagnose <name>    corpus-diagnose on one cut against F
    absent_check.py summarize          the three predictions, and the earlier cuts for the record

No command runs on import.
"""

import argparse
import json
import random
import re
import statistics
from collections import defaultdict

import class_patterns as base
import replication
import tool_check
from class_patterns import divergence, natural_gap, subset_selection, syntactic_eval

from corpus_generalization.analysis.syntactic import syntactic_patterns

WORDS = ("using", "making", "however", "several")
CONTROLS = ("ctrl_a", "ctrl_b", "r25b", "r25c")
FRESH = (*CONTROLS, "drop4")
EARLIER = tool_check.ALL_NAMES
MIN_TRUNK = 20
DRAWS = 8


def prepare() -> None:
    _, pool = natural_gap.split()
    subsets = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    documents = [pool[i] for i in subsets["r50"]["indices"]]
    segmented = base.segment(documents, (), "VERB")
    verbs = json.loads((base.RESULTS / "corpora.json").read_text())["corpora"]
    possessives = json.loads((base.RESULTS / "possessive_corpora.json").read_text())["corpora"]
    # Fresh seeds at the sizes of the earlier cuts: 1 at class_cut's, 2 at poss_class_cut's.
    targets = {
        "ctrl_a": (verbs["class_cut"]["bytes"], 1),
        "ctrl_b": (possessives["poss_class_cut"]["bytes"], 2),
    }
    corpora = {
        name: base.assemble(segmented, base.match_bytes(segmented, set(), target, seed))
        for name, (target, seed) in targets.items()
    }
    pattern = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)
    dropped = {
        (d, i)
        for d, sentences in enumerate(segmented)
        for i, (text, _) in enumerate(sentences)
        if pattern.search(text)
    }
    corpora["drop4"] = base.assemble(segmented, dropped)
    cuts = json.loads((base.EXPERIMENTS / "cut_choice/results/cuts.json").read_text())
    for name in ("r25b", "r25c"):
        corpora[name] = [pool[i] for i in cuts[name]["indices"]]
    for name, corpus in corpora.items():
        (base.CORPORA / f"{name}.json").write_text(json.dumps(corpus))
        print(f"{name}: {len(corpus)} documents, {base.size(corpus):,} bytes")
    print(f"drop4 removed {len(dropped)} sentences")


def qualifying(name: str) -> dict[str, float]:
    """The one-token head patterns of the four words whose mean trunk over the tool's own draws
    is at least 2·`MIN_TRUNK`: the draws are replicated exactly (seed 0, whole documents to the
    corpus's bytes)."""
    reading = divergence.read_full()
    corpus_bytes = base.size(base.corpus(name))
    sizes = [len(d.encode("utf-8")) for d in reading.pool]
    rng = random.Random(0)
    trunks: dict[tuple[tuple[int, ...], int], list[int]] = defaultdict(list)
    wanted = {reading.tokens[w] for w in WORDS}
    for _ in range(DRAWS):
        order = list(range(len(reading.full)))
        rng.shuffle(order)
        sampled, size = [], 0
        for index in order:
            sampled.append(reading.full[index])
            size += sizes[index]
            if size >= corpus_bytes:
                break
        for p in syntactic_patterns(sampled, reading.classes, head_length=1, heads=wanted):
            trunks[(p.head, p.filler_class)].append(p.trunk)
    result = {}
    for (head, filler_class), observed in trunks.items():
        mean = statistics.mean(observed + [0] * (DRAWS - len(observed)))
        if mean >= 2 * MIN_TRUNK:
            result[f"{reading.tokenizer.decode(list(head))}|{filler_class}"] = mean
    return result


def summarize() -> None:
    def absent(name: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = json.loads(
            (base.RESULTS / f"tool_{name}.json").read_text()
        )["absent"]
        return rows

    held = {"1": True, "2": True, "3": True}
    for name in CONTROLS:
        count = len(absent(name))
        held["1"] &= count <= 2
        print(f"{name:8} absent rows {count}")
    rows = absent("drop4")
    flagged = {f"{r['head']}|{r['filler_class']}" for r in rows}
    expected = qualifying("drop4")
    missed = sorted(set(expected) - flagged)
    held["2"] &= not missed
    others = [r for r in rows if not any(w in str(r["head"]).lower() for w in WORDS)]
    held["3"] &= len(others) <= 5
    print(
        f"drop4: {len(rows)} absent rows; {len(expected)} qualifying patterns of the four words, "
        f"missed {missed}; absent rows outside the words {len(others)}: "
        f"{[r['head'] for r in others][:10]}"
    )
    print("earlier cuts, for the record:")
    record = {}
    for name in EARLIER:
        count = len(absent(name))
        record[name] = count
        print(f"  {name:16} absent rows {count}")
    predictions = {k: syntactic_eval.status(v) for k, v in held.items()}
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(
        base.RESULTS / "absent_check_summary.json",
        {
            "controls": {name: len(absent(name)) for name in CONTROLS},
            "drop4": {
                "absent": rows,
                "qualifying": expected,
                "missed": missed,
                "outside_words": len(others),
            },
            "earlier": record,
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
    tool.add_argument("name", choices=(*FRESH, *EARLIER))
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
