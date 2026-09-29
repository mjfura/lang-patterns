"""A planted gap in natural text: does the reference diagnosis find a restriction put into real
text on purpose, and only it, once its verdict needs a margin over sampling noise? See "A planted
gap in natural text" in `docs/2026-09-28-syntactic-sufficiency-design.md`.

    planted.py run <source> <remove|replace|sweep>
    planted.py show <source> <remove|replace|sweep>

`sweep` plants the replacement restriction at every `k` of `SWEEP` fillers kept in turn.

The source's first 6.5 MB sample (`experiments/clean_domains`) is the reference and is cut into
`SLICES` disjoint slices of documents, the targets. Results: `<source>_results.json` here.
"""

import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

from corpus_generalization.analysis.syntactic import (
    Pattern,
    TokenClasses,
    induce_classes,
    syntactic_patterns,
    word_like_tokens,
)
from corpus_generalization.packing import framed_documents
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
sys.path[:0] = [str(OUT.parent / "natural_sufficiency"), str(OUT.parent / "clean_domains")]

import clean_domains  # noqa: E402
import natural  # noqa: E402

SLICES = 8
PLANTED_SLICE = 0
HEADS = 5
FREQUENT_EXCLUDED = 100  # the most frequent tokens are not planted: removing them guts the slice
TRUNK_RANGE = (200, 2000)
MIN_DIVERSITY = 0.8
KEPT_FILLERS = 2
SWEEP = (2, 4, 8, 16, 32)  # fillers kept per head in the sensitivity sweep
MIN_TRUNK = 20
LOW = 0.5
SIGMAS = 3
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def plant(documents: list[str], heads: dict[str, list[str]], mode: str) -> tuple[list[str], int]:
    """Plant the restriction: every word after a head outside its kept fillers either removes
    its sentence (`remove`) or is replaced by the head's most frequent kept filler, trailing
    punctuation kept (`replace`). Returns the planted documents and how many sentences were
    removed or words replaced."""
    changed = 0
    planted = []
    for document in documents:
        kept = []
        for sentence in SENTENCE.split(document):
            words = sentence.split()
            # (position, head) taken before any replacement: a replaced word may itself be the
            # next hit's head.
            hits = [
                (index + 1, word)
                for index, (word, following) in enumerate(pairwise(words))
                if word in heads
                and re.fullmatch(r"[A-Za-z]+", core := re.sub(r"[^A-Za-z]+$", "", following))
                and core not in heads[word]
            ]
            if not hits:
                kept.append(sentence)
            elif mode == "remove":
                changed += 1
            else:
                for index, head in hits:
                    tail = re.search(r"[^A-Za-z]*$", words[index])
                    words[index] = heads[head][0] + (tail.group() if tail else "")
                    changed += 1
                kept.append(" ".join(words))
        planted.append(" ".join(kept))
    return planted, changed


def choose_heads(
    reference: list[list[int]],
    classes: TokenClasses,
    reference_patterns: dict[tuple[tuple[int, ...], int], Pattern],
    spell: Callable[[int], str],
    count: int,
    skip: int = 0,
) -> tuple[list[Pattern], dict[str, list[str]]]:
    """The heads to plant: each head's largest-trunk pattern, within `TRUNK_RANGE`, not among the
    most frequent tokens, with diversity at least `MIN_DIVERSITY`; ranked by trunk, `count` of
    them after the first `skip`. With each head's reference fillers in that pattern's class, most
    frequent first, up to `max(SWEEP)`."""
    frequency = Counter(token for sequence in reference for token in sequence)
    frequent = {token for token, _ in frequency.most_common(FREQUENT_EXCLUDED)}
    best: dict[int, Pattern] = {}
    for pattern in reference_patterns.values():
        head = pattern.head[0]
        if head in frequent or not TRUNK_RANGE[0] <= pattern.trunk <= TRUNK_RANGE[1]:
            continue
        if head not in best or pattern.trunk > best[head].trunk:
            best[head] = pattern
    chosen = sorted(
        (p for p in best.values() if p.diversity >= MIN_DIVERSITY), key=lambda p: -p.trunk
    )[skip : skip + count]
    kept: dict[str, list[str]] = {}
    for pattern in chosen:
        followers: Counter[int] = Counter(
            sequence[i + 1]
            for sequence in reference
            for i in range(len(sequence) - 1)
            if sequence[i] == pattern.head[0]
            and classes.classes.get(sequence[i + 1]) == pattern.filler_class
        )
        kept[spell(pattern.head[0]).strip()] = [
            spell(token).strip() for token, _ in followers.most_common(max(SWEEP))
        ]
    return chosen, kept


def run(source: str, mode: str) -> None:
    documents = clean_domains.samples(json.loads(clean_domains.cache_of(source).read_text()))[0]
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        documents, max_merges=natural.MERGES[source], recognize_markers=True
    )

    def encode(texts: list[str]) -> list[list[int]]:
        return framed_documents(texts, tokenizer, True)

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    reference = encode(documents)
    words = word_like_tokens(reference, spell, boundary_ids={tokenizer.bos_id, tokenizer.eos_id})
    classes = induce_classes(reference, candidates=words)

    def read(sequences: list[list[int]]) -> dict[tuple[tuple[int, ...], int], Pattern]:
        return {
            (p.head, p.filler_class): p for p in syntactic_patterns(sequences, classes, heads=words)
        }

    reference_patterns = read(reference)

    chosen, kept = choose_heads(reference, classes, reference_patterns, spell, HEADS)

    size = len(documents) // SLICES
    slices = [documents[i * size : (i + 1) * size] for i in range(SLICES)]
    readings = [read(encode(part)) for part in slices]

    def noise(key: tuple[tuple[int, ...], int]) -> tuple[float, float] | None:
        values = [r[key].diversity for r in readings if key in r and r[key].trunk >= MIN_TRUNK]
        if len(values) < SLICES - 2:
            return None
        return statistics.mean(values), statistics.stdev(values)

    def under_shown(pattern: Pattern, key: tuple[tuple[int, ...], int]) -> bool:
        stats, ref = noise(key), reference_patterns.get(key)
        return (
            stats is not None
            and ref is not None
            and pattern.diversity < LOW
            and pattern.diversity < stats[0] - SIGMAS * stats[1]
            and ref.diversity >= LOW
        )

    def planted_heads(
        planted: dict[tuple[tuple[int, ...], int], Pattern], k: int
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for pattern in chosen:
            key = (pattern.head, pattern.filler_class)
            stats = noise(key)
            target = planted.get(key)
            rows.append(
                {
                    "head": spell(pattern.head[0]).strip(),
                    "kept": sorted(kept[spell(pattern.head[0]).strip()][:k]),
                    "reference_trunk": pattern.trunk,
                    "reference_diversity": pattern.diversity,
                    "slice_diversities": [
                        round(r[key].diversity, 3) if key in r else None for r in readings
                    ],
                    "slice_mean": stats[0] if stats else None,
                    "slice_sd": stats[1] if stats else None,
                    "planted_trunk": target.trunk if target else 0,
                    "planted_diversity": target.diversity if target else None,
                    "under_shown": bool(target and under_shown(target, key)),
                }
            )
        return rows

    chosen_keys = {(p.head, p.filler_class) for p in chosen}

    def false_alarms(reading: dict[tuple[tuple[int, ...], int], Pattern]) -> tuple[int, int, int]:
        """Among the non-planted patterns read: how many fall below their margin, and how many
        of those are called under-shown."""
        read_keys = [
            key
            for key, p in reading.items()
            if key not in chosen_keys and p.trunk >= MIN_TRUNK and noise(key) is not None
        ]
        below = [
            key
            for key in read_keys
            if reading[key].diversity < noise(key)[0] - SIGMAS * noise(key)[1]  # type: ignore[index]
        ]
        return len(read_keys), len(below), sum(under_shown(reading[key], key) for key in below)

    def planting(k: int, how: str) -> tuple[dict[tuple[tuple[int, ...], int], Pattern], int]:
        restricted = {head: fillers[:k] for head, fillers in kept.items()}
        documents_planted, changed = plant(slices[PLANTED_SLICE], restricted, how)
        return read(encode(documents_planted)), changed

    results: dict[str, object] = {
        "source": source,
        "slices": SLICES,
        "slice_documents": size,
        "mode": mode,
        "others_unplanted_slice": false_alarms(readings[PLANTED_SLICE]),
    }
    if mode == "sweep":
        sweep = {}
        for k in SWEEP:
            planted, changed = planting(k, "replace")
            sweep[str(k)] = {
                "changed": changed,
                "heads": planted_heads(planted, k),
                "others_planted_slice": false_alarms(planted),
            }
        results["sweep"] = sweep
    else:
        planted, changed = planting(KEPT_FILLERS, mode)
        results |= {
            "changed": changed,
            "heads": planted_heads(planted, KEPT_FILLERS),
            "others_planted_slice": false_alarms(planted),
        }
    (OUT / f"{source}_{mode}_results.json").write_text(json.dumps(results, indent=2) + "\n")
    show(source, mode)


def show(source: str, mode: str) -> None:
    r = json.loads((OUT / f"{source}_{mode}_results.json").read_text())
    if mode == "sweep":
        total, below, _ = r["others_unplanted_slice"]
        print(
            f"== {source}, sweep: {r['slices']} slices of {r['slice_documents']} documents; "
            f"unplanted slice: {below} of {total} other patterns below their margin"
        )
        for k, reading in r["sweep"].items():
            heads = reading["heads"]
            found = sum(h["under_shown"] for h in heads)
            total, below, _ = reading["others_planted_slice"]
            diversities = ", ".join(
                f"{h['head']} {h['planted_diversity']:.2f}"
                if h["planted_diversity"] is not None
                else f"{h['head']} -"
                for h in heads
            )
            print(
                f"  k={k:>2}: found {found}/{len(heads)} ({diversities}); "
                f"others below margin {below}/{total}"
            )
        return
    print(
        f"== {source}, {mode}: {r['slices']} slices of {r['slice_documents']} documents; "
        f"{r['changed']} {'sentences removed' if mode == 'remove' else 'words replaced'} "
        f"in slice {PLANTED_SLICE}"
    )
    for h in r["heads"]:
        print(
            f"  {h['head']!r:>14} kept {h['kept']}: reference {h['reference_diversity']:.3f} "
            f"({h['reference_trunk']}), slices {h['slice_diversities']} "
            f"mean {h['slice_mean']:.3f} sd {h['slice_sd']:.3f} -> planted "
            f"{h['planted_diversity']} ({h['planted_trunk']}), under-shown {h['under_shown']}"
        )
    for name in ("others_planted_slice", "others_unplanted_slice"):
        total, below, called = r[name]
        print(
            f"  {name}: {below} of {total} other patterns below their margin "
            f"({below / total:.3f}), {called} called under-shown"
        )


if __name__ == "__main__":
    match sys.argv[1:]:
        case [("run" | "show") as command, source, ("remove" | "replace" | "sweep") as mode] if (
            source in natural.MERGES
        ):
            run(source, mode) if command == "run" else show(source, mode)
        case _:
            sys.exit(__doc__)
