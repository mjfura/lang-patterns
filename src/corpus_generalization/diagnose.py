"""Per-pattern syntactic diagnosis; see docs/2026-09-30-corpus-diagnose-design.md.

Read one- and two-token word-like heads by default: a determiner separates a verb from its
object in "cut the tomato". Select lengths with diagnose(head_lengths=...) or the command's
--head-lengths 1 2; corpus, reference, and noise draws use the same lengths.
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from corpus_generalization.analysis.syntactic import (
    Pattern,
    TokenClasses,
    backoff_unigram,
    class_fillers,
    filler_divergence,
    filler_vocabulary,
    induce_classes,
    syntactic_patterns,
    word_like_tokens,
)
from corpus_generalization.packing import framed_documents
from corpus_generalization.sources import (
    EmptyDataFileError,
    MissingDataFileError,
    read_documents_from,
)
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

type Verdict = Literal[
    "under-shown", "absent", "restricted", "fixed", "within-noise", "sufficient", "unread"
]
type PatternKey = tuple[tuple[int, ...], int]
VERDICTS: tuple[Verdict, ...] = (
    "under-shown",
    "absent",
    "restricted",
    "fixed",
    "within-noise",
    "sufficient",
    "unread",
)


@dataclass(frozen=True)
class DiagnosisClass:
    label: int
    tokens: tuple[str, ...]
    top_tokens: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosisRow:
    head: str
    head_length: int
    filler_class: int
    top_tokens: tuple[str, ...]
    trunk: int
    fillers: int
    diversity: float
    reference_diversity: float | None
    margin_mean: float | None
    margin_sd: float | None
    verdict: Verdict
    top_fillers: tuple[str, ...]
    draw_trunk_mean: float | None = None
    draw_trunk_sd: float | None = None


@dataclass(frozen=True)
class DiagnosisClassRow:
    """A class pattern: a head class's fillers, after any of its one-token heads."""

    head_class: int
    heads: tuple[str, ...]
    trunk: int
    reference_trunk: int
    divergence: float
    margin_mean: float | None
    margin_sd: float | None
    excess: float | None
    verdict: Verdict
    top_fillers: tuple[str, ...]
    reference_fillers: tuple[str, ...]
    short_heads: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosisSettings:
    merge_budget: int
    draws: int
    min_trunk: int
    low: float
    sigmas: float
    seed: int
    head_lengths: tuple[int, ...]
    head_classes: int = 64
    class_floor: float = 0.5


@dataclass(frozen=True)
class DiagnosisReport:
    documents: int
    words: int
    tokens_per_word: float
    merge_budget: int
    classes: tuple[DiagnosisClass, ...]
    rows: tuple[DiagnosisRow, ...]
    warnings: tuple[str, ...]
    settings: DiagnosisSettings
    class_rows: tuple[DiagnosisClassRow, ...] = ()

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(row.verdict for row in self.rows)
        return {verdict: counts[verdict] for verdict in VERDICTS}

    def to_json(self) -> dict[str, object]:
        """Return content directly serializable by json.dumps."""
        return {**asdict(self), "n_classes": self.n_classes, "counts": self.counts}

    def summary(self) -> str:
        counts = ", ".join(f"{count} {verdict}" for verdict, count in self.counts.items())
        return (
            f"{self.documents:,} documents, {self.words:,} words, "
            f"{self.tokens_per_word:.2f} tokens/word, {self.merge_budget:,} merge budget\n"
            f"{self.n_classes} classes, {len(self.rows)} read patterns: {counts}"
        )

    def markdown(self) -> str:
        lines = ["# Syntactic-sufficiency diagnosis", "", self.summary(), "", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in self.warnings)
        if not self.warnings:
            lines.append("None.")
        lines.extend(
            [
                "",
                "Syntactic sufficiency does not establish coverage outside the corpus's topics.",
                "The diversity threshold is permissive and needs calibration "
                "per model family and size.",
                "",
                "## Classes",
                "",
                "| Class | Most frequent tokens |",
                "| --- | --- |",
            ]
        )
        for group in self.classes:
            lines.append(f"| {group.label} | {_tokens(group.top_tokens)} |")
        lines.extend(
            [
                "",
                "## Flagged patterns",
                "",
                "Under-shown slots need text with other fillers of their class. "
                "Restricted slots are",
                "candidates for inspection; fixed slots are domain constructions, not defects.",
                "",
                "| Head | Head length | Class | Class tokens | Trunk | Fillers | Diversity | "
                "Reference | Margin mean | Margin sd | Verdict | Corpus fillers |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in self.rows:
            if row.verdict == "sufficient":
                continue
            numbers = " | ".join(
                "—" if value is None else f"{value:.3f}"
                for value in (
                    row.diversity,
                    row.reference_diversity,
                    row.margin_mean,
                    row.margin_sd,
                )
            )
            lines.append(
                f"| {_tokens((row.head,))} | {row.head_length} | {row.filler_class} | "
                f"{_tokens(row.top_tokens)} | "
                f"{row.trunk} | {row.fillers} | {numbers} | {row.verdict} | "
                f"{_tokens(row.top_fillers)} |"
            )
        lines.extend(
            [
                "",
                "## Class patterns",
                "",
                "A head class's fillers after any of its heads. A model learns syntax here: "
                "an under-shown class pattern is where it is expected to lose the slot's class.",
                "",
            ]
        )
        if not self.class_rows:
            lines.append("Not read: class patterns need a reference.")
        else:
            lines.extend(
                [
                    "| Head class | Heads | Trunk | D | Margin mean | Margin sd | Excess | "
                    "Verdict | Short heads | Corpus fillers | Reference fillers |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for crow in self.class_rows:
                numbers = " | ".join(
                    "—" if value is None else f"{value:.3f}"
                    for value in (crow.divergence, crow.margin_mean, crow.margin_sd, crow.excess)
                )
                lines.append(
                    f"| {crow.head_class} | {_tokens(crow.heads)} | {crow.trunk} | {numbers} | "
                    f"{crow.verdict} | {_tokens(crow.short_heads)} | {_tokens(crow.top_fillers)} | "
                    f"{_tokens(crow.reference_fillers)} |"
                )
        lines.extend(
            [
                "",
                "## Settings",
                "",
                "```json",
                json.dumps(asdict(self.settings), indent=2),
                "```",
                "",
            ]
        )
        return "\n".join(lines)


def _tokens(tokens: tuple[str, ...]) -> str:
    return ", ".join(token.strip() for token in tokens)


def diagnose(
    documents: Sequence[str],
    reference: Sequence[str] | None = None,
    merge_budget: int = 4000,
    draws: int = 8,
    min_trunk: int = 20,
    low: float = 0.5,
    sigmas: float = 3.0,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
    head_lengths: Sequence[int] = (1, 2),
    head_classes: int = 64,
    class_floor: float = 0.5,
) -> DiagnosisReport:
    """Read corpus slots against shared domain classes and optional size-matched noise."""
    if not documents or (reference is not None and not reference):
        raise ValueError("Corpus and reference must contain documents.")
    if merge_budget < 0 or draws < 2 or min_trunk < 1 or low < 0 or sigmas < 0:
        raise ValueError("Require merges >= 0, draws >= 2, min_trunk >= 1, low and sigmas >= 0.")
    if head_classes < 1 or class_floor < 0:
        raise ValueError("Require head_classes >= 1 and class_floor >= 0.")

    lengths = tuple(dict.fromkeys(head_lengths))
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("Head lengths must be positive and nonempty.")

    def step(message: str) -> None:
        if progress is not None:
            progress(message)

    domain = documents if reference is None else reference
    step("Learning tokenizer")
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        domain, max_merges=merge_budget, recognize_markers=True
    )
    step("Framing documents")
    corpus_ids = framed_documents(documents, tokenizer, True)
    domain_ids = corpus_ids if reference is None else framed_documents(domain, tokenizer, True)

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    step("Finding word-like tokens")
    heads = word_like_tokens(domain_ids, spell, {tokenizer.bos_id, tokenizer.eos_id})
    step("Inducing classes")
    document_frequency = Counter(token for sequence in domain_ids for token in set(sequence))
    # Ward requires an eligible token; tiny or fragmented corpora can have none.
    classes = (
        induce_classes(domain_ids, candidates=heads)
        if any(document_frequency[token] >= 2 for token in heads)
        else TokenClasses({}, 0, ())
    )
    frequency = Counter(token for sequence in domain_ids for token in sequence)
    groups: dict[int, list[int]] = defaultdict(list)
    for token, label in classes.classes.items():
        groups[label].append(token)
    class_rows = tuple(
        DiagnosisClass(
            label,
            tuple(spell(token) for token in sorted(tokens)),
            tuple(spell(token) for token in sorted(tokens, key=lambda t: (-frequency[t], t))[:12]),
        )
        for label, tokens in sorted(groups.items())
    )
    top_tokens = {group.label: group.top_tokens for group in class_rows}

    def read_patterns(sequences: Sequence[Sequence[int]]) -> list[Pattern]:
        return [
            pattern
            for length in lengths
            for pattern in syntactic_patterns(sequences, classes, head_length=length, heads=heads)
        ]

    step("Reading corpus patterns")
    patterns = read_patterns(corpus_ids)
    references: dict[PatternKey, float] = {}
    margins: dict[PatternKey, list[float]] = defaultdict(list)
    # Every draw's trunk per pattern; draws that miss a pattern are its zeros.
    draw_trunks: dict[PatternKey, list[int]] = defaultdict(list)
    corpus_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    domain_sizes = [len(doc.encode("utf-8")) for doc in domain]
    head_tree: TokenClasses | None = None
    vocabulary = filler_vocabulary(classes)
    reference_fillers: dict[int, Counter[int]] = {}
    class_margins: dict[int, list[float]] = defaultdict(list)

    def read_class_divergence(
        sequences: Sequence[Sequence[int]], into: dict[int, list[float]]
    ) -> None:
        assert head_tree is not None
        unigram = backoff_unigram(sequences, vocabulary)
        fillers = class_fillers(sequences, classes, head_tree, heads)
        for label, counts in reference_fillers.items():
            into[label].append(filler_divergence(counts, fillers.get(label, Counter()), unigram))

    if reference is not None and len(classes.classes) >= 2:
        step("Inducing head classes")
        # The coarse cut pools pronouns and modals with prepositions; at 64 on FineWeb the
        # same tree separates pronouns, modals and possessives.
        head_tree = induce_classes(
            domain_ids, n_classes=min(head_classes, len(classes.classes)), candidates=heads
        )
        reference_fillers = {
            label: counts
            for label, counts in class_fillers(domain_ids, classes, head_tree, heads).items()
            if sum(counts.values()) >= min_trunk
        }
    if reference is not None:
        step("Reading reference patterns")
        references = {
            (pattern.head, pattern.filler_class): pattern.diversity
            for pattern in read_patterns(domain_ids)
        }
        rng = random.Random(seed)
        for draw in range(draws):
            step(f"Reading reference draw {draw + 1}/{draws}")
            order = list(range(len(domain_ids)))
            rng.shuffle(order)
            sampled: list[list[int]] = []
            size = 0
            # Whole documents preserve contexts; the last may overshoot the byte budget.
            for index in order:
                sampled.append(domain_ids[index])
                size += domain_sizes[index]
                if size >= corpus_bytes:
                    break
            for pattern in read_patterns(sampled):
                key = (pattern.head, pattern.filler_class)
                draw_trunks[key].append(pattern.trunk)
                if pattern.trunk >= min_trunk:
                    margins[key].append(pattern.diversity)
            if head_tree is not None:
                read_class_divergence(sampled, class_margins)

    step("Building report and warnings")
    filler_counts: dict[PatternKey, Counter[int]] = defaultdict(Counter)
    for sequence in corpus_ids:
        for length in lengths:
            for start in range(len(sequence) - length):
                head = tuple(sequence[start : start + length])
                filler = sequence[start + length]
                filler_class = classes.classes.get(filler)
                if all(token in heads for token in head) and filler_class is not None:
                    filler_counts[(head, filler_class)][filler] += 1
    rows = []
    for pattern in patterns:
        if pattern.trunk < min_trunk:
            continue
        key = (pattern.head, pattern.filler_class)
        ref = references.get(key)
        values = margins.get(key, [])
        mean = statistics.mean(values) if values else None
        sd = statistics.stdev(values) if len(values) >= 2 else None
        verdict: Verdict
        if pattern.diversity >= low:
            verdict = "sufficient"
        elif reference is None:
            verdict = "restricted"
        elif ref is None or len(values) < draws - 2:
            verdict = "unread"
        elif ref is not None and ref < low:
            verdict = "fixed"
        elif mean is None or sd is None:
            verdict = "unread"
        elif pattern.diversity < mean - sigmas * sd:
            verdict = "under-shown"
        else:
            verdict = "within-noise"
        rows.append(
            DiagnosisRow(
                tokenizer.decode(list(pattern.head)),
                len(pattern.head),
                pattern.filler_class,
                top_tokens[pattern.filler_class],
                pattern.trunk,
                pattern.fillers,
                pattern.diversity,
                ref,
                mean,
                sd,
                verdict,
                tuple(spell(token) for token, _ in filler_counts[key].most_common(5)),
            )
        )
    # A pattern random cuts of this size show and the corpus does not would have no row: a
    # removed slot (`experiments/class_patterns`) left nothing to flag. Against the draws'
    # trunk noise: a fixed `min_trunk` alone flagged 79-114 patterns on random FineWeb cuts.
    by_key = {(p.head, p.filler_class): p for p in patterns}
    for key, observed in draw_trunks.items():
        corpus_pattern = by_key.get(key)
        trunk = corpus_pattern.trunk if corpus_pattern else 0
        if trunk >= min_trunk:
            continue
        trunks = observed + [0] * (draws - len(observed))
        trunk_mean, trunk_sd = statistics.mean(trunks), statistics.stdev(trunks)
        if (
            trunk_mean < min_trunk
            or trunk >= trunk_mean - sigmas * trunk_sd
            or trunk > trunk_mean / 2
        ):
            continue
        values = margins.get(key, [])
        head, filler_class = key
        rows.append(
            DiagnosisRow(
                tokenizer.decode(list(head)),
                len(head),
                filler_class,
                top_tokens[filler_class],
                corpus_pattern.trunk if corpus_pattern else 0,
                corpus_pattern.fillers if corpus_pattern else 0,
                corpus_pattern.diversity if corpus_pattern else 0.0,
                references.get(key),
                statistics.mean(values) if values else None,
                statistics.stdev(values) if len(values) >= 2 else None,
                "absent",
                tuple(spell(token) for token, _ in filler_counts[key].most_common(5)),
                trunk_mean,
                trunk_sd,
            )
        )
    rows.sort(key=lambda row: (VERDICTS.index(row.verdict), -row.trunk))
    class_pattern_rows: list[DiagnosisClassRow] = []
    if head_tree is not None:
        step("Reading class patterns")
        corpus_divergence: dict[int, list[float]] = defaultdict(list)
        read_class_divergence(corpus_ids, corpus_divergence)
        corpus_fillers = class_fillers(corpus_ids, classes, head_tree, heads)
        head_frequency = Counter(token for sequence in domain_ids for token in sequence)
        members: dict[int, list[int]] = defaultdict(list)
        for token, label in head_tree.classes.items():
            members[label].append(token)
        short = {
            row.head
            for row in rows
            if row.head_length == 1 and row.verdict in ("under-shown", "absent")
        }
        for label, counts in reference_fillers.items():
            value = corpus_divergence[label][0]
            values = class_margins.get(label, [])
            mean = statistics.mean(values) if values else None
            sd = statistics.stdev(values) if len(values) >= 2 else None
            excess = value - mean if mean is not None else None
            class_verdict: Verdict
            if mean is None or sd is None or len(values) < draws - 2:
                class_verdict = "unread"
            elif value > mean + sigmas * sd and excess is not None and excess >= class_floor:
                class_verdict = "under-shown"
            else:
                class_verdict = "within-noise"
            shown = corpus_fillers.get(label, Counter())
            class_pattern_rows.append(
                DiagnosisClassRow(
                    label,
                    tuple(
                        spell(token)
                        for token in sorted(members[label], key=lambda t: (-head_frequency[t], t))[
                            :12
                        ]
                    ),
                    sum(shown.values()),
                    sum(counts.values()),
                    value,
                    mean,
                    sd,
                    excess,
                    class_verdict,
                    tuple(spell(token) for token, _ in shown.most_common(5)),
                    tuple(spell(token) for token, _ in counts.most_common(5)),
                    tuple(
                        spell(token)
                        for token in sorted(members[label], key=lambda t: (-head_frequency[t], t))
                        if spell(token) in short
                    ),
                )
            )
        class_pattern_rows.sort(
            key=lambda row: (VERDICTS.index(row.verdict), -(row.excess or 0.0), -row.trunk)
        )
    words = sum(len(doc.split()) for doc in documents)
    tokens_per_word = sum(len(sequence) - 2 for sequence in corpus_ids) / words if words else 0.0
    warnings = []
    if len(documents) < 1000 or words < 50000:
        warnings.append(
            "Corpus has fewer than 1,000 documents or 50,000 words; "
            "classes and correlations may be unreliable."
        )
    if reference is not None and corpus_bytes > sum(domain_sizes):
        warnings.append(
            "Corpus is larger than its reference in bytes; draws use the full reference."
        )
    if tokens_per_word > 1.6:
        warnings.append(
            "Tokenizer leaves more than 1.6 tokens per word; word pieces fill the classes."
        )
    return DiagnosisReport(
        len(documents),
        words,
        tokens_per_word,
        merge_budget,
        class_rows,
        tuple(rows),
        tuple(warnings),
        DiagnosisSettings(
            merge_budget, draws, min_trunk, low, sigmas, seed, lengths, head_classes, class_floor
        ),
        tuple(class_pattern_rows),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", nargs="+", type=Path)
    parser.add_argument("--merges", type=int, default=4000)
    parser.add_argument(
        "--head-lengths",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Word-like tokens per head (default: 1 2)",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.merges < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--merges must be nonnegative and --limit must be positive")

    if any(length < 1 for length in args.head_lengths):
        parser.error("--head-lengths must be positive")

    def read(paths: list[Path]) -> tuple[str, ...]:
        return tuple(doc for path in paths for doc in read_documents_from([path])[: args.limit])

    try:
        documents = read(args.files)
        reference = read(args.reference) if args.reference is not None else None
    except (MissingDataFileError, EmptyDataFileError) as error:
        print(error, file=sys.stderr)
        return 2
    report = diagnose(
        documents, reference, merge_budget=args.merges, head_lengths=args.head_lengths
    )
    output = args.output or Path("diagnoses") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(report.markdown(), encoding="utf-8")
    print(report.summary())
    flagged = sum(row.verdict == "under-shown" for row in report.class_rows)
    if report.class_rows:
        print(f"{len(report.class_rows)} class patterns read, {flagged} under-shown")
    for warning in report.warnings:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
