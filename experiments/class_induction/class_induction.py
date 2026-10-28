"""Step 2 of the settings-from-data path: do the class-induction constants change the verdicts?

Each constant is read at several values on F at 1.50 tokens per word, with the 22 labeled cuts of
`experiments/granularity_range`. Corpus side only: no model is trained.

- `CONTEXT_TOKENS`: neighbor features over the most frequent tokens, the rest pooled.
- The coarse cut's class count. `min_classes`, `max_classes` and `NOISE_FLOOR` only choose
  where the Ward tree is cut, so forcing the count covers all three.

    class_induction.py context <n>    diagnose the cuts with n context tokens
    class_induction.py classes <k>    diagnose the cuts with the tree cut at k classes
    class_induction.py summarize      detection, false flags and agreement with the default

The default (1,000 context tokens, which cuts at 12 classes) is
`experiments/granularity_range/results/target_1.50.json`. No command runs on import, except
the forced cut, which every process installs from `CLASS_CUT`.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent
RESULTS = OUT / "results"
sys.path[:0] = [str(OUT.parent / "granularity_range")]

import granularity_range as g  # noqa: E402

from corpus_generalization.analysis import syntactic  # noqa: E402
from corpus_generalization.analysis.syntactic import ClassTree, TokenClasses  # noqa: E402
from corpus_generalization.diagnose import prepare_reference  # noqa: E402
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402

TARGET = "1.50"
BASELINE = 1000


def _force_cut(count: int) -> None:
    """Cut every tree at `count` classes. Installed at import from `CLASS_CUT`, so that worker
    processes, which import this module afresh, cut the same way as the parent."""
    default = ClassTree.cut

    def cut(
        self: ClassTree,
        n_classes: int | None = None,
        max_classes: int = 64,
        *,
        min_classes: int = 8,
    ) -> TokenClasses:
        if n_classes is None:
            n_classes = min(count, len(self.tokens))
        return default(self, n_classes, max_classes, min_classes=min_classes)

    ClassTree.cut = cut  # type: ignore[method-assign]


if "CLASS_CUT" in os.environ:
    _force_cut(int(os.environ["CLASS_CUT"]))


def context(count: int, name: str | None = None) -> None:
    budget = json.loads((g.RESULTS / "budgets.json").read_text())["budgets"][TARGET]
    _, pool = g.natural_gap.split()
    tokenizer = ByteLevelBPETokenizer.from_merges(g.merges()[:budget])
    # `class_tree` reads the constant when called; the reference is read here, in this process.
    syntactic.CONTEXT_TOKENS = count
    started = time.time()
    reading = prepare_reference(
        pool, budget, tokenizer=tokenizer, head_lengths=(1,), progress=g.divergence.progress
    )
    reference_seconds = time.time() - started
    frequency = Counter(token for sequence in reading.documents for token in sequence)
    covered = sum(n for _, n in frequency.most_common(count)) / sum(frequency.values())
    cuts: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    workers = ProcessPoolExecutor(g.WORKERS, initializer=g._start_worker, initargs=(reading, None))
    with workers as pool_:
        for cut, result, elapsed in pool_.map(g._diagnose_cut, g.CUTS):
            g.divergence.progress(f"{cut}: {result['counts']} in {elapsed:.0f}s")
            cuts[cut], seconds[cut] = result, elapsed
    g.divergence.write_json(
        RESULTS / (name or f"context_{count}.json"),
        {
            "target": TARGET,
            "context_tokens": count,
            "class_cut": int(os.environ["CLASS_CUT"]) if "CLASS_CUT" in os.environ else None,
            "context_share": covered,
            "merge_budget": budget,
            "pool_tokens_per_word": 0.0,
            "lexical_share": 0.0,
            "classes": {str(group.label): list(group.top_tokens) for group in reading.groups},
            "class_members": {str(group.label): list(group.tokens) for group in reading.groups},
            "reference_seconds": reference_seconds,
            "cut_seconds": seconds,
            "cuts": cuts,
        },
    )


def classes(count: int) -> None:
    os.environ["CLASS_CUT"] = str(count)
    _force_cut(count)
    context(BASELINE, f"classes_{count}.json")


def summarize() -> None:
    baseline = json.loads((g.RESULTS / f"target_{TARGET}.json").read_text())
    readings = {"default": baseline} | {
        path.stem: json.loads(path.read_text())
        for prefix in ("context", "classes")
        for path in sorted(
            RESULTS.glob(f"{prefix}_*.json"), key=lambda path: int(path.stem.split("_")[1])
        )
    }
    summary: dict[str, Any] = {}
    for count, data in readings.items():
        values = g._summary(data)
        share, total = g._agreement(data, baseline)
        kept, either = g._flag_agreement(data, baseline)
        summary[str(count)] = {
            "context_share": data.get("context_share"),
            "classes": values["classes"],
            "detection": values["detection"],
            "false_flags": values["false_flags"],
            "head_flags": values["head_flags"],
            "absent_rows": values["absent_rows"],
            "dropped_words_found": values["dropped_words_found"],
            "read": values["read"],
            "bars": values["bars"],
            "agreement_with_baseline": share,
            "patterns_compared": total,
            "flags_kept": kept,
            "flagged_at_either": either,
        }
        print(f"{count}: {json.dumps(summary[str(count)])}")
    g.divergence.write_json(RESULTS / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("context")
    run.add_argument("count", type=int)
    cut = commands.add_parser("classes")
    cut.add_argument("count", type=int)
    commands.add_parser("summarize")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    match args.command:
        case "context":
            context(args.count)
        case "classes":
            classes(args.count)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
