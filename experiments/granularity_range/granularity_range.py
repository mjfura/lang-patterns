"""Step 1 of the settings-from-data path: over which tokens-per-word range do the verdicts hold?

"Step 1: the tokens-per-word range" in
`docs/superpowers/specs/2026-10-11-settings-from-data-design.md` records the design and predictions
before running. Corpus side only: no model is trained.

    granularity_range.py budgets             the merge budget of each target granularity
    granularity_range.py diagnose <target>   read F at that budget, diagnose the 22 labeled cuts
        [--min-trunk M]                      ... reading patterns from trunk M (default: set by
                                             the draws), into `results/min_trunk/` when given
    granularity_range.py summarize           detection, false flags and stability per granularity
    granularity_range.py exchangeable <n>    diagnose n random whole-document cuts of F at 1.50

The tokenizer is learned once on F, the FineWeb pool, at `MAX_MERGES`
(`runs/granularity_range/merges.json`); every granularity is a cut of it, which is exactly a
tokenizer learned at that budget. No command runs on import.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "granularity_range"
MERGES = RUNS / "merges.json"
sys.path[:0] = [str(OUT.parent / "class_patterns")]

import class_patterns as base  # noqa: E402
from class_patterns import divergence, natural_gap  # noqa: E402

from corpus_generalization.diagnose import (  # noqa: E402
    DiagnosisReport,
    DomainReading,
    diagnose,
    prepare_reference,
)
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer, Merge  # noqa: E402
from corpus_generalization.tokenizer.granularity import tokens_per_word_by_budget  # noqa: E402

MAX_MERGES = 60000
TARGETS = (1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.80)
# Head classes that hold a removed class's heads, read by one of its members.
ANCHORS = {"class_cut": ("they", "can"), "poss_class_cut": ("his",), "num_class_cut": ("two",)}
ANCHORS["verb_class_cut"] = ("make",)
HEAD_CUTS = ("head_cut", "poss_head_cut", "num_head_cut", "verb_head_cut")
CONTROLS = ("control", "poss_control", "num_control", "verb_control")
FRESH_CONTROLS = ("ctrl_a", "ctrl_b", "ctrl_c", "ctrl_d", "r25b", "r25c", "r25d", "r25e")
DROPS = {
    "drop4": ("using", "making", "however", "several"),
    "drop4b": ("although", "within", "various", "whose"),
}
CUTS = (*ANCHORS, *HEAD_CUTS, *CONTROLS, *FRESH_CONTROLS, *DROPS)
WORKERS = int(os.environ.get("WORKERS", "4"))

_reading: DomainReading | None = None
_min_trunk: int | None = None
_vary_draws = False


def merges() -> tuple[Merge, ...]:
    return tuple(((left, right), new) for (left, right), new in json.loads(MERGES.read_text()))


def budgets() -> None:
    _, pool = natural_gap.split()
    learned = merges()
    curve = tokens_per_word_by_budget(pool, learned)
    chosen = {}
    for target in TARGETS:
        budget = next((k for k, value in enumerate(curve) if value <= target), None)
        chosen[f"{target:.2f}"] = budget
        shown = "beyond the learned merges" if budget is None else f"{budget:,} merges"
        print(f"{target:.2f} tokens/word: {shown}")
    divergence.write_json(
        RESULTS / "budgets.json",
        {
            "learned_merges": len(learned),
            "budgets": chosen,
            "curve": {str(k): round(curve[k], 4) for k in range(0, len(curve), 500)},
        },
    )


def _start_worker(reading: DomainReading, min_trunk: int | None, vary_draws: bool = False) -> None:
    global _reading, _min_trunk, _vary_draws
    _reading, _min_trunk, _vary_draws = reading, min_trunk, vary_draws


def _diagnose_cut(name: str) -> tuple[str, dict[str, Any], float]:
    assert _reading is not None
    started = time.time()
    report = diagnose(base.corpus(name), _reading, min_trunk=_min_trunk)
    return name, reading_of(report), time.time() - started


def reading_of(report: DiagnosisReport) -> dict[str, Any]:
    """What `summarize` reads from one report, plus every syntactic row's verdict for the
    stability reading."""
    return {
        "tokens_per_word": report.tokens_per_word,
        "counts": report.counts,
        "class_nodes_read": report.class_nodes_read,
        "min_trunk": report.settings.min_trunk,
        "class_min_trunk": report.settings.class_min_trunk,
        "class_critical_t": report.class_critical_t,
        "head_critical_t": report.head_critical_t,
        "absent_critical_t": report.absent_critical_t,
        "lexical_counts": report.lexical_counts,
        "class_rows": [
            {
                "heads": list(row.heads),
                "size": row.size,
                "t": row.t,
                "excess": row.excess,
                "lexical": row.lexical,
            }
            for row in report.class_rows
        ],
        "absent": [
            {"head": row.head, "filler_class": row.filler_class, "lexical": row.lexical}
            for row in report.rows
            if row.verdict == "absent"
        ],
        "rows": [
            [row.head, row.filler_class, row.verdict]
            for row in report.rows
            if not row.lexical and row.head_length == 1
        ],
    }


def diagnose_target(target: str, min_trunk: int | None = None) -> None:
    budget = json.loads((RESULTS / "budgets.json").read_text())["budgets"][target]
    if budget is None:
        raise SystemExit(f"{target} tokens/word needs more than the learned merges.")
    _, pool = natural_gap.split()
    tokenizer = ByteLevelBPETokenizer.from_merges(merges()[:budget])
    started = time.time()
    reading = prepare_reference(
        pool,
        budget,
        tokenizer=tokenizer,
        head_lengths=(1,),
        progress=divergence.progress,
    )
    reference_seconds = time.time() - started
    transitions = sum(len(starts) - 1 for starts in reading.word_starts)
    lexical = sum(not flag for starts in reading.word_starts for flag in starts)
    words = sum(len(document.split()) for document in pool)
    tokens = sum(len(sequence) - 2 for sequence in reading.documents)
    classes = {str(group.label): list(group.top_tokens) for group in reading.groups}
    members = {str(group.label): list(group.tokens) for group in reading.groups}
    cuts: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    workers = ProcessPoolExecutor(WORKERS, initializer=_start_worker, initargs=(reading, min_trunk))
    with workers as pool_:
        for name, result, elapsed in pool_.map(_diagnose_cut, CUTS):
            divergence.progress(f"{name}: {result['counts']} in {elapsed:.0f}s")
            cuts[name], seconds[name] = result, elapsed
    out = (
        RESULTS / f"target_{target}.json"
        if min_trunk is None
        else RESULTS / "min_trunk" / f"target_{target}_m{min_trunk}.json"
    )
    divergence.write_json(
        out,
        {
            "target": target,
            "min_trunk": min_trunk,
            "merge_budget": budget,
            "pool_tokens_per_word": tokens / words,
            "lexical_share": lexical / transitions,
            "classes": classes,
            "class_members": members,
            "reference_seconds": reference_seconds,
            "cut_seconds": seconds,
            "cuts": cuts,
        },
    )


def _class_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The reported syntactic class rows: every one is under-shown."""
    return [
        {"heads": [h.strip() for h in r["heads"]][:6], "size": r["size"], "t": round(r["t"], 1)}
        for r in rows
        if not r["lexical"]
    ]


def _class_map(left: dict[str, list[str]], right: dict[str, list[str]]) -> dict[int, int]:
    """Each class of one granularity to the class of the other sharing most of its members."""
    owner = {token: int(label) for label, tokens in right.items() for token in tokens}
    mapping = {}
    for label, tokens in left.items():
        votes = Counter(owner[token] for token in tokens if token in owner)
        if votes:
            mapping[int(label)] = votes.most_common(1)[0][0]
    return mapping


def _agreement(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, int]:
    """Share of syntactic one-token patterns read at both granularities with the same verdict,
    over every cut; a pattern is its head's text and its filler class mapped by members."""
    mapping = _class_map(left["class_members"], right["class_members"])
    same = total = 0
    for name in CUTS:
        verdicts = {(h, c): v for h, c, v in right["cuts"][name]["rows"]}
        for head, label, verdict in left["cuts"][name]["rows"]:
            other = verdicts.get((head, mapping.get(label, -1)))
            if other is not None:
                total += 1
                same += other == verdict
    return (same / total if total else float("nan")), total


def _flag_agreement(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, int]:
    """Flagged (under-shown or absent) syntactic one-token patterns read at both granularities:
    how many keep their flag at the other, of how many are flagged at either."""
    mapping = _class_map(left["class_members"], right["class_members"])
    kept = either = 0
    for name in CUTS:
        verdicts = {(h, c): v for h, c, v in right["cuts"][name]["rows"]}
        for head, label, verdict in left["cuts"][name]["rows"]:
            other = verdicts.get((head, mapping.get(label, -1)))
            flags = {verdict, other} & {"under-shown", "absent"}
            if other is not None and flags:
                either += 1
                kept += verdict == other
    return kept, either


def _summary(data: dict[str, Any]) -> dict[str, Any]:
    """Detection, false flags and what is read, for one reading of the 22 cuts."""
    cuts = data["cuts"]
    detection = {
        name: [
            row for row in _class_rows(cuts[name]["class_rows"]) if set(anchors) & set(row["heads"])
        ]
        for name, anchors in ANCHORS.items()
    }
    false_flags = {
        name: _class_rows(cuts[name]["class_rows"]) for name in (*CONTROLS, *FRESH_CONTROLS)
    }
    head_cut_rows = {name: _class_rows(cuts[name]["class_rows"]) for name in HEAD_CUTS}
    head_flags = {
        name: sum(verdict == "under-shown" for _, _, verdict in cuts[name]["rows"]) for name in CUTS
    }
    absent = {
        name: len([r for r in cuts[name]["absent"] if not r["lexical"]])
        for name in (*FRESH_CONTROLS, *DROPS)
    }
    dropped = {
        name: sum(
            any(word == r["head"].strip() for r in cuts[name]["absent"] if not r["lexical"])
            for word in words
        )
        for name, words in DROPS.items()
    }
    return {
        "merge_budget": data["merge_budget"],
        "pool_tokens_per_word": round(data["pool_tokens_per_word"], 3),
        "lexical_share": round(data["lexical_share"], 3),
        "classes": len(data["classes"]),
        "detection": detection,
        "false_flags": false_flags,
        "head_cut_rows": head_cut_rows,
        "head_flags": head_flags,
        "absent_rows": absent,
        "dropped_words_found": dropped,
        "unread": statistics.mean(cuts[name]["counts"]["unread"] for name in CUTS),
        "read": statistics.mean(
            cuts[name]["counts"]["within-noise"] + cuts[name]["counts"]["under-shown"]
            for name in CUTS
        ),
        "class_nodes_read": cuts[CUTS[0]]["class_nodes_read"],
        "bars": {
            bar: statistics.median(value for name in CUTS if (value := cuts[name].get(bar)))
            if any(cuts[name].get(bar) for name in CUTS)
            else None
            for bar in ("class_critical_t", "head_critical_t", "absent_critical_t")
        },
        "reference_seconds": round(data["reference_seconds"]),
        "mean_cut_seconds": round(statistics.mean(data["cut_seconds"].values())),
    }


def summarize() -> None:
    readings = {
        target: json.loads(path.read_text())
        for target in (f"{t:.2f}" for t in TARGETS)
        if (path := RESULTS / f"target_{target}.json").exists()
    }
    summary = {target: _summary(data) for target, data in readings.items()}
    for target, values in summary.items():
        print(f"{target}: {json.dumps(values)}")
    ordered = sorted(readings)
    ordered = sorted(readings)
    stability = {}
    for left, right in pairwise(ordered):
        share, total = _agreement(readings[left], readings[right])
        kept, either = _flag_agreement(readings[left], readings[right])
        stability[f"{left}-{right}"] = {
            "agreement": share,
            "patterns": total,
            "flags_kept": kept,
            "flagged_at_either": either,
        }
        print(
            f"{left} vs {right}: {share:.3f} of {total} patterns agree; "
            f"{kept} of {either} flagged patterns keep their flag"
        )
    divergence.write_json(RESULTS / "summary.json", {"targets": summary, "stability": stability})


def _random_cut(seed: int) -> list[str]:
    """Whole documents of F in random order, to the bytes of `control`: a corpus exchangeable with
    the draws, which `control` (sentences of many documents) is not."""
    _, pool = natural_gap.split()
    target = sum(len(document.encode("utf-8")) for document in base.corpus("control"))
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    cut, size = [], 0
    for index in order:
        if size >= target:
            break
        cut.append(pool[index])
        size += len(pool[index].encode("utf-8"))
    return cut


def _diagnose_random(seed: int) -> dict[str, Any]:
    assert _reading is not None
    report = diagnose(
        _random_cut(seed), _reading, min_trunk=_min_trunk, seed=seed if _vary_draws else 0
    )
    return {
        "seed": seed,
        "min_trunk": report.settings.min_trunk,
        "head_critical_t": report.head_critical_t,
        "class_critical_t": report.class_critical_t,
        "class_rows": len([row for row in report.class_rows if not row.lexical]),
        "under_shown": [
            {
                "head": row.head,
                "filler_class": row.filler_class,
                "trunk": row.trunk,
                "fillers": row.fillers,
                "diversity": row.diversity,
                "margin_mean": row.margin_mean,
                "margin_sd": row.margin_sd,
                "top_fillers": list(row.top_fillers),
            }
            for row in report.rows
            if row.verdict == "under-shown" and not row.lexical
        ],
        "absent": len([row for row in report.rows if row.verdict == "absent" and not row.lexical]),
    }


def exchangeable(count: int, min_trunk: int | None = None, vary_draws: bool = False) -> None:
    """How often a corpus exchangeable with the draws is flagged: the bar is built for about 1 in
    (draws + 1), over random draws. `vary_draws` draws anew for each cut; otherwise every cut
    meets the same draws (seed 0), as every diagnosis does."""
    budget = json.loads((RESULTS / "budgets.json").read_text())["budgets"]["1.50"]
    _, pool = natural_gap.split()
    reading = prepare_reference(
        pool,
        budget,
        tokenizer=ByteLevelBPETokenizer.from_merges(merges()[:budget]),
        head_lengths=(1,),
        progress=divergence.progress,
    )
    workers = ProcessPoolExecutor(
        WORKERS, initializer=_start_worker, initargs=(reading, min_trunk, vary_draws)
    )
    results = []
    with workers as pool_:
        for result in pool_.map(_diagnose_random, range(1000, 1000 + count)):
            divergence.progress(
                f"seed {result['seed']}: {len(result['under_shown'])} under-shown, "
                f"{result['class_rows']} class rows, {result['absent']} absent"
            )
            results.append(result)
    suffix = ("" if min_trunk is None else f"_m{min_trunk}") + ("_varied" if vary_draws else "")
    divergence.write_json(RESULTS / "exchangeable" / f"cuts_1.50{suffix}.json", results)
    flagged = sum(bool(result["under_shown"]) for result in results)
    print(f"{flagged} of {count} cuts with a head flag")


def summarize_min_trunk() -> None:
    """The same summary at 1.50 tokens per word for every fixed min_trunk, and derived."""
    paths = {
        int(path.stem.rsplit("_m", 1)[1]): path
        for path in (RESULTS / "min_trunk").glob("target_1.50_m*.json")
    }
    summary = {str(m): _summary(json.loads(paths[m].read_text())) for m in sorted(paths)}
    if (derived := RESULTS / "target_1.50.json").exists():
        summary["derived"] = _summary(json.loads(derived.read_text()))
    for m, values in summary.items():
        print(f"min_trunk {m}: {json.dumps(values)}")
    divergence.write_json(RESULTS / "min_trunk" / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("budgets")
    run = commands.add_parser("diagnose")
    run.add_argument("target", choices=[f"{t:.2f}" for t in TARGETS])
    run.add_argument("--min-trunk", type=int)
    commands.add_parser("summarize")
    commands.add_parser("summarize-min-trunk")
    random_cuts = commands.add_parser("exchangeable")
    random_cuts.add_argument("count", type=int)
    random_cuts.add_argument("--min-trunk", type=int)
    random_cuts.add_argument("--vary-draws", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    match args.command:
        case "budgets":
            budgets()
        case "diagnose":
            diagnose_target(args.target, args.min_trunk)
        case "summarize":
            summarize()
        case "summarize-min-trunk":
            summarize_min_trunk()
        case "exchangeable":
            exchangeable(args.count, args.min_trunk, args.vary_draws)


if __name__ == "__main__":
    main()
