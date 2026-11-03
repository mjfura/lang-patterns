"""Per-pattern syntactic diagnosis; see docs/superpowers/specs/2026-09-30-corpus-diagnose-design.md.

Read one- and two-token heads by default: a determiner separates a verb from its object in "cut
the tomato". Select lengths with diagnose(head_lengths=...) or the command's --head-lengths 1 2;
corpus, reference, and noise draws use the same lengths.

Every token can be a head or a filler. A filler that continues the head's last word makes a
lexical pattern, reported apart from the syntactic ones
(`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`).
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from warnings import catch_warnings, simplefilter

import numpy as np
from scipy import sparse
from scipy.stats import beta

from corpus_generalization.analysis.siblings import SiblingTest, sibling_bar, sibling_tests
from corpus_generalization.analysis.syntactic import (
    ClassTree,
    Pattern,
    TokenClasses,
    class_tree,
    filler_occurrences,
    filler_vocabulary,
    head_filler_matrices,
    node_divergences,
    pattern_filler_counts,
    syntactic_patterns,
    tree_membership,
    tree_parents,
    unigram_vector,
)
from corpus_generalization.packing import framed_word_starts
from corpus_generalization.sources import (
    EmptyDataFileError,
    MissingDataFileError,
    read_documents_from,
)
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer
from corpus_generalization.tokenizer.granularity import (
    GRANULARITY_RANGE,
    GRANULARITY_TARGET,
    learn_to_granularity,
    tokens_per_word_by_budget,
)

type Verdict = Literal["under-shown", "absent", "weak", "fixed", "within-noise", "rare", "unread"]
type PatternKey = tuple[tuple[int, ...], int, bool]
VERDICTS: tuple[Verdict, ...] = (
    "under-shown",
    "absent",
    "weak",
    "fixed",
    "within-noise",
    "rare",
    "unread",
)


# The verdicts a report lists; the rest are counted only.
FLAGGED = ("under-shown", "absent", "weak")
# One occurrence shows one filler, and its diversity is 1 by definition: two is the least trunk
# whose diversity says anything.
READABLE = 2
# A random corpus's chance of any flag, over draw sets: the test's level, a tolerance for false
# alarms rather than a property of the data. 1/17 was implicit in 16 draws.
ALPHA = 0.06
# How far that chance may drift with one fixed set of draws, and for what share of sets:
# `_draws_for` reads the draw count from them.
RATE_MARGIN = 0.5
RATE_SPREAD = 0.1
# The smallest size measured at which 1.50 held on nearly every draw of five corpora (14 of 15 pools
# at 14.4 MB with unresolvable dropped words left out, 12 of 15 without; its misses there are
# dropped words, never classes); at 3.6 MB Wikipedia and legal missed removed classes, and neither
# NOISE_FLOOR nor a size-matched CONTEXT_TOKENS recovered them
# (`experiments/granularity_robustness`).
MIN_DOMAIN_BYTES = 14_370_440


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
    lexical: bool = False


@dataclass(frozen=True)
class DiagnosisClassRow:
    """A class pattern: the fillers after any one-token head under one node of the head tree.
    `heads` are its most frequent heads, of `size`; `t` is its excess over the draws' sd."""

    node: int
    heads: tuple[str, ...]
    size: int
    trunk: int
    reference_trunk: int
    divergence: float
    margin_mean: float
    margin_sd: float
    excess: float
    t: float
    verdict: Verdict
    top_fillers: tuple[str, ...]
    reference_fillers: tuple[str, ...]
    short_heads: tuple[str, ...] = ()
    lexical: bool = False
    # Without a reference a class row is one filler class's slot, read against its sibling.
    filler_class: int | None = None


@dataclass(frozen=True)
class DiagnosisSettings:
    merge_budget: int
    draws: int
    min_trunk: int
    seed: int
    head_lengths: tuple[int, ...]
    class_min_trunk: float | None = None
    alpha: float = ALPHA


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
    class_nodes_read: int = 0
    class_critical_t: float | None = None
    head_critical_t: float | None = None
    absent_critical_t: float | None = None
    has_reference: bool = True

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def counts(self) -> dict[str, int]:
        """Verdicts of the syntactic patterns: the diagnosis."""
        counts = Counter(row.verdict for row in self.rows if not row.lexical)
        return {verdict: counts[verdict] for verdict in VERDICTS}

    @property
    def lexical_counts(self) -> dict[str, int]:
        counts = Counter(row.verdict for row in self.rows if row.lexical)
        return {verdict: counts[verdict] for verdict in VERDICTS}

    def to_json(self) -> dict[str, object]:
        """Return content directly serializable by json.dumps."""
        return {
            **asdict(self),
            "n_classes": self.n_classes,
            "counts": self.counts,
            "lexical_counts": self.lexical_counts,
        }

    def summary(self) -> str:
        lines = [
            f"{self.documents:,} documents, {self.words:,} words, "
            f"{self.tokens_per_word:.2f} tokens/word, {self.merge_budget:,} merge budget",
            f"{self.n_classes} classes",
        ]
        for name, counts, lexical in (
            ("syntactic", self.counts, False),
            ("lexical", self.lexical_counts, True),
        ):
            read = sum(row.lexical == lexical for row in self.rows)
            listed = ", ".join(f"{count} {verdict}" for verdict, count in counts.items())
            lines.append(f"{read} read {name} patterns: {listed}")
        return "\n".join(lines)

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
                *self._level(),
                "",
                "## Classes",
                "",
                "| Class | Most frequent tokens |",
                "| --- | --- |",
            ]
        )
        for group in self.classes:
            lines.append(f"| {group.label} | {_tokens(group.top_tokens)} |")
        for lexical in (False, True):
            lines.extend(self._pattern_section(lexical))
        for lexical in (False, True):
            lines.extend(self._class_section(lexical))
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

    def _level(self) -> list[str]:
        if self.head_critical_t is None and self.class_critical_t is None:
            return []
        if not self.has_reference:
            return [
                "",
                f"Without a reference, each family of verdicts (head rows, class rows) is read "
                f"at alpha = {self.settings.alpha:g} against {self.settings.draws} random splits "
                "of each head-tree node's occurrences between its children: a head holding a "
                "random share of them is flagged in a family with probability about alpha. A "
                "weak pattern may be the domain's own, and a pattern the corpus lacks, or a "
                "class shown weakly as a whole, is not seen: a reference from the same domain "
                "does both (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`).",
            ]
        return [
            "",
            f"Each family of verdicts (head rows, absent, class rows) is read at alpha = "
            f"{self.settings.alpha:g} against {self.settings.draws} draws: a random sample of "
            "the reference this size is flagged in a family with probability about alpha. "
            "Alpha is not split among the families; on 48 random cuts of FineWeb, 2 drew any "
            "flag (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`).",
        ]

    def _pattern_section(self, lexical: bool) -> list[str]:
        title = "## Flagged lexical patterns" if lexical else "## Flagged syntactic patterns"
        lines = ["", title, ""]
        lines.extend(
            [
                "The filler continues the head's last word: how words are spelled, not how they "
                "combine. Read apart from the diagnosis.",
                "",
            ]
            if lexical
            else [
                "Under-shown slots need text with other fillers of their class; absent ones, "
                "text that uses them. Weak slots (no reference) show fewer fillers than a "
                "random share of their sibling's occurrences would; a reference from the same "
                "domain tells whether the domain shares that. Most fillers missing first.",
                "",
            ]
        )
        lines.extend(
            [
                "| Head | Head length | Class | Class tokens | Trunk | Fillers | Diversity | "
                "Reference | Margin mean | Margin sd | Verdict | Corpus fillers |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in self.rows:
            if row.verdict not in FLAGGED or row.lexical != lexical:
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
        return lines

    def _class_section(self, lexical: bool) -> list[str]:
        lines = [
            "",
            "## Lexical class patterns" if lexical else "## Syntactic class patterns",
            "",
            "The fillers after any head under one node of the head tree. A model learns syntax "
            "here: an under-shown class pattern is where it is expected to lose the slot's class. "
            "Every node is read; of each branch, the node furthest above its draws is listed.",
            "",
        ]
        if not self.has_reference:
            return lines + self._sibling_class_rows(lexical)
        if self.class_critical_t is None:
            lines.append("Not read: class patterns need a reference.")
            return lines
        lines.append(
            f"{self.class_nodes_read} class patterns read; under-shown above t = "
            f"{self.class_critical_t:.1f}, the furthest any draw strays from the others.",
        )
        lines.append("")
        rows = [crow for crow in self.class_rows if crow.lexical == lexical]
        lines.extend(
            [
                "| Node | Heads | Size | Trunk | D | Margin mean | Margin sd | Excess | t | "
                "Verdict | Short heads | Corpus fillers | Reference fillers |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for crow in rows:
            numbers = " | ".join(
                f"{value:.3f}"
                for value in (crow.divergence, crow.margin_mean, crow.margin_sd, crow.excess)
            )
            lines.append(
                f"| {crow.node} | {_tokens(crow.heads)} | {crow.size} | {crow.trunk} | {numbers} | "
                f"{crow.t:.1f} | "
                f"{crow.verdict} | {_tokens(crow.short_heads)} | {_tokens(crow.top_fillers)} | "
                f"{_tokens(crow.reference_fillers)} |"
            )
        return lines

    def _sibling_class_rows(self, lexical: bool) -> list[str]:
        if self.class_critical_t is None:
            return ["Not read: no class of heads had siblings to compare with."]
        lines = [
            f"{self.class_nodes_read} class patterns read against their sibling; weak above "
            f"t = {self.class_critical_t:.1f}. Expected is the fillers a random share of its "
            "parent's occurrences would show; most fillers missing first.",
            "",
            "| Node | Heads | Size | Class | Trunk | Fillers | Expected | sd | Missing | t | "
            "Corpus fillers | Parent fillers |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for crow in self.class_rows:
            if crow.lexical != lexical:
                continue
            lines.append(
                f"| {crow.node} | {_tokens(crow.heads)} | {crow.size} | {crow.filler_class} | "
                f"{crow.trunk} | {crow.divergence:.0f} | {crow.margin_mean:.1f} | "
                f"{crow.margin_sd:.1f} | {crow.excess:.1f} | {crow.t:.1f} | "
                f"{_tokens(crow.top_fillers)} | {_tokens(crow.reference_fillers)} |"
            )
        return lines


def _tokens(tokens: tuple[str, ...]) -> str:
    return ", ".join(token.strip() for token in tokens)


@dataclass(frozen=True, eq=False)
class ClassNodes:
    """The reference's class patterns at every internal node of the head tree, syntactic and
    lexical: `reference[kind]` holds node rows over `fillers` columns. `read[kind]` marks the
    nodes whose reference trunk reaches `READABLE`; `diagnose` reads those of them a corpus
    could flag."""

    tree: ClassTree
    membership: sparse.csr_matrix
    parents: np.ndarray
    fillers: tuple[int, ...]
    reference: tuple[sparse.csr_matrix, sparse.csr_matrix]
    read: tuple[np.ndarray, np.ndarray]

    def divergences(
        self,
        sequences: Sequence[Sequence[int]],
        classes: TokenClasses,
        word_starts: Sequence[Sequence[bool]],
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[sparse.csr_matrix, sparse.csr_matrix]]:
        """D per read node of each kind for `sequences`, and their node filler counts."""
        counts = tuple(
            self.membership @ matrix
            for matrix in head_filler_matrices(
                sequences, classes, self.tree.tokens, self.fillers, word_starts
            )
        )
        unigram = unigram_vector(sequences, self.fillers)
        values = tuple(
            node_divergences(
                self.reference[kind][self.read[kind]], counts[kind][self.read[kind]], unigram
            )
            for kind in (0, 1)
        )
        return (values[0], values[1]), (counts[0], counts[1])

    def single_filler(self, kind: int, trunks: np.ndarray, unigram: np.ndarray) -> np.ndarray:
        """D per read node of a corpus that shows the node's `trunks` occurrences all with the
        reference's most frequent filler: the furthest a restricted class pattern can stray."""
        reference = self.reference[kind][self.read[kind]]
        top = np.asarray(reference.argmax(axis=1)).ravel()
        shown = sparse.csr_matrix(
            (np.maximum(np.rint(trunks), 1.0), (np.arange(len(top)), top)), shape=reference.shape
        )
        return node_divergences(reference, shown, unigram)


def _t(excess: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Excess in units of sd; with no spread, any excess is infinitely far."""
    out = np.where(excess > 0, np.inf, 0.0)
    np.divide(excess, sd, out=out, where=sd > 0)
    return out


def _draws_for(alpha: float) -> int:
    """The fewest draws whose bar holds a random corpus's chance of a flag within half to one and
    a half `alpha`, for 80% of draw sets. That chance, for the `_rank`-th largest of n
    exchangeable draws' maxima, is Beta(rank, n + 1 - rank) over draw sets: with 16 draws and
    alpha 1/17 it ran from 0.4% to 8% across sets, and one fixed set flagged 6 of 48 random
    cuts (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`)."""
    draws = 3
    while True:
        rank = _rank(alpha, draws, strict=False)
        low, high = beta.ppf((RATE_SPREAD, 1 - RATE_SPREAD), rank, draws + 1 - rank)
        if rank >= 1 and low >= alpha * (1 - RATE_MARGIN) and high <= alpha * (1 + RATE_MARGIN):
            return draws
        draws += 1


def _rank(alpha: float, draws: int, *, strict: bool = True) -> int:
    """Which largest draw maximum is the bar: a random corpus passes the k-th largest of n
    exchangeable maxima with probability k / (n + 1)."""
    rank = round(alpha * (draws + 1))
    if strict and rank < 1:
        raise ValueError(f"alpha {alpha} needs at least {round(1 / alpha - 1)} draws.")
    return rank


def _bar(maxima: np.ndarray, rank: int) -> float:
    """The `rank`-th largest of the draws' maxima."""
    return float(np.sort(maxima)[::-1][rank - 1])


def _leave_one_out(noise: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every draw (row) and column, the other draws' count, mean and sd, NaN read as
    missing. Sums rather than a loop over draws: at 104 draws the loop took minutes."""
    shown = ~np.isnan(noise)
    values = np.where(shown, noise, 0.0)
    counts = shown.sum(axis=0) - shown
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (values.sum(axis=0) - values) / counts
        squares = (values**2).sum(axis=0) - values**2
        variance = np.maximum(squares - counts * mean**2, 0.0) / (counts - 1)
    return counts, mean, np.sqrt(variance)


def _null_maxima(noises: Sequence[np.ndarray]) -> np.ndarray:
    """The largest t each draw reaches, at any node, against the other draws (rows are draws,
    columns nodes)."""
    maxima = np.full(len(noises[0]) if noises else 0, -np.inf)
    for noise in noises:
        if noise.shape[1]:
            _, mean, sd = _leave_one_out(noise)
            maxima = np.maximum(maxima, _t(noise - mean, sd).max(axis=1))
    return maxima


def _with_step(sd: np.ndarray, step: np.ndarray) -> np.ndarray:
    """The draws' spread, with the rounding of the value tested, which moves in steps of
    `step`."""
    spread: np.ndarray = np.sqrt(sd**2 + np.nan_to_num(step) ** 2 / 12)
    return spread


def _drop_maxima(noise: np.ndarray, least: int, expected: np.ndarray) -> np.ndarray:
    """The largest drop of each draw's diversity below the other draws' mean, in their spread,
    over the patterns at least `least` draws read (NaN where a draw does not). A draw's value
    moves in steps of one filler, 1 / its `expected` fillers."""
    counts, mean, sd = _leave_one_out(noise)
    usable = (counts >= max(least - 1, 2)) & ~np.isnan(noise)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _t(np.where(usable, mean - noise, 0.0), _with_step(sd, 1 / expected))
    maxima: np.ndarray = np.where(usable, t, -np.inf).max(axis=1, initial=-np.inf)
    return maxima


def _absent_maxima(trunks: np.ndarray, min_trunk: int) -> np.ndarray:
    """`_drop_maxima` for trunks, over the patterns an absent row could be: the other draws read
    them (mean >= `min_trunk`) and this draw does not (trunk < `min_trunk`)."""
    _, mean, sd = _leave_one_out(trunks)
    usable = (mean >= min_trunk) & (trunks < min_trunk)
    t = _t(np.where(usable, mean - trunks, 0.0), _with_step(sd, np.ones_like(sd)))
    maxima: np.ndarray = np.where(usable, t, -np.inf).max(axis=1, initial=-np.inf)
    return maxima


def _readable_trunk(
    diversity: np.ndarray, expected: np.ndarray, trunks: np.ndarray, least: int, rank: int
) -> int:
    """The trunk from which patterns are read: the one that leaves the most patterns a corpus
    could flag, by showing a single filler where the draws show their mean. A pattern too rare
    for that can never be flagged, yet its noise raises the bar for every other (Tarone's
    criterion for discrete tests, 1990). Below 20, rare patterns raised the class bar from 52 to
    93 and the head bar from 6.6 to 8.4; above it, fewer patterns were read
    (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`)."""
    best, most = READABLE, -1
    for trunk in np.unique(trunks[trunks >= READABLE]):
        shown = trunks >= trunk
        reads = shown.sum(axis=0) >= least
        if int(reads.sum()) <= most:
            break
        values = np.where(shown, diversity, np.nan)
        with catch_warnings():
            simplefilter("ignore", RuntimeWarning)
            step = 1 / np.nanmean(np.where(shown, expected, np.nan), axis=0)
        bar = _bar(_drop_maxima(values, least, expected), rank)
        mean = np.nanmean(values[:, reads], axis=0)
        spread = _with_step(np.nanstd(values[:, reads], axis=0, ddof=1), step[reads])
        flaggable = int((_t(mean - step[reads], spread) > bar).sum())
        if flaggable > most:
            best, most = int(trunk), flaggable
    return best


def _readable_nodes(
    noises: Sequence[np.ndarray],
    trunks: Sequence[np.ndarray],
    extremes: Sequence[np.ndarray],
    rank: int,
) -> float:
    """`_readable_trunk` for class patterns: the draws' mean node trunk from which nodes are
    read, the one that leaves the most nodes whose `extremes` (D of a single filler) would
    exceed the draws' bar."""
    best, most = float(READABLE), -1
    for trunk in np.unique(np.concatenate([values[values >= READABLE] for values in trunks])):
        chosen = [values >= trunk for values in trunks]
        if sum(int(read.sum()) for read in chosen) <= most:
            break
        bar = _bar(
            _null_maxima([noise[:, read] for noise, read in zip(noises, chosen, strict=True)]),
            rank,
        )
        flaggable = sum(
            int(
                (
                    _t(
                        extreme[read] - noise[:, read].mean(axis=0),
                        noise[:, read].std(axis=0, ddof=1),
                    )
                    > bar
                ).sum()
            )
            for noise, read, extreme in zip(noises, chosen, extremes, strict=True)
        )
        if flaggable > most:
            best, most = float(trunk), flaggable
    return best


def _branch_maxima(values: np.ndarray, flagged: np.ndarray, parents: np.ndarray) -> list[int]:
    """The flagged nodes whose value exceeds that of every flagged ancestor and descendant:
    of each branch, the node where the class pattern is furthest from its noise. A parent's index
    is always above its children's (merge order), so one pass each way suffices."""
    marked = np.where(flagged, values, -np.inf)
    above = np.full(len(values), -np.inf)
    for node in range(len(values) - 1, -1, -1):
        parent = parents[node]
        if parent >= 0:
            above[node] = max(above[parent], marked[parent])
    below = np.full(len(values), -np.inf)
    for node in range(len(values)):
        parent = parents[node]
        if parent >= 0:
            below[parent] = max(below[parent], marked[node], below[node])
    return [
        int(node)
        for node in np.flatnonzero(flagged)
        if marked[node] > above[node] and marked[node] > below[node]
    ]


@dataclass(frozen=True, eq=False)
class DomainReading:
    """What `diagnose` reads from the domain alone: the reference, or the corpus itself when
    there is none. A reference reading is reused across corpora: one reference served 22 cuts in
    `experiments/granularity_range`, and learning its tokenizer was 40% of each run."""

    tokenizer: ByteLevelBPETokenizer
    merge_budget: int
    head_lengths: tuple[int, ...]
    is_reference: bool
    tokens_per_word: float
    documents: tuple[list[int], ...]
    word_starts: tuple[list[bool], ...]
    sizes: tuple[int, ...]
    classes: TokenClasses
    groups: tuple[DiagnosisClass, ...]
    frequency: Counter[int]
    nodes: "ClassNodes | None"
    patterns: dict[PatternKey, float]
    tree: ClassTree | None = None
    # Decoding rebuilds the merge table on every call: 33 of 164 seconds of one diagnosis.
    spelled: dict[tuple[int, ...], str] = field(default_factory=dict)

    def spell(self, token: int) -> str:
        return self.spell_all((token,))

    def spell_all(self, tokens: tuple[int, ...]) -> str:
        if tokens not in self.spelled:
            self.spelled[tokens] = self.tokenizer.decode(list(tokens))
        return self.spelled[tokens]

    def read_patterns(
        self, sequences: Sequence[Sequence[int]], word_starts: Sequence[Sequence[bool]]
    ) -> list[Pattern]:
        return [
            pattern
            for length in self.head_lengths
            for pattern in syntactic_patterns(
                sequences, self.classes, head_length=length, word_starts=word_starts
            )
        ]


def _head_lengths(head_lengths: Sequence[int]) -> tuple[int, ...]:
    lengths = tuple(dict.fromkeys(head_lengths))
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("Head lengths must be positive and nonempty.")
    return lengths


def _read_domain(
    domain: Sequence[str],
    tokenizer: ByteLevelBPETokenizer,
    merge_budget: int,
    lengths: tuple[int, ...],
    is_reference: bool,
    step: Callable[[str], None],
) -> DomainReading:
    step("Framing domain documents")
    framed, starts = framed_word_starts(domain, tokenizer, True)
    domain_ids, word_starts = tuple(framed), tuple(starts)

    def spell(token: int) -> str:
        return tokenizer.decode([token])

    step("Inducing classes")
    document_frequency = Counter(token for sequence in domain_ids for token in set(sequence))
    # Ward requires an eligible token; tiny or fragmented corpora can have none.
    tree = (
        class_tree(domain_ids) if any(count >= 2 for count in document_frequency.values()) else None
    )
    classes = tree.cut() if tree is not None else TokenClasses({}, 0, ())
    frequency = Counter(token for sequence in domain_ids for token in sequence)
    members: dict[int, list[int]] = defaultdict(list)
    for token, label in classes.classes.items():
        members[label].append(token)
    groups = tuple(
        DiagnosisClass(
            label,
            tuple(spell(token) for token in sorted(tokens)),
            tuple(spell(token) for token in sorted(tokens, key=lambda t: (-frequency[t], t))[:12]),
        )
        for label, tokens in sorted(members.items())
    )
    nodes: ClassNodes | None = None
    fillers = tuple(sorted(filler_vocabulary(classes)))
    if is_reference and tree is not None and len(tree.tokens) >= 2 and fillers:
        step("Reading class patterns of the reference")
        # Every node rather than one cut: at 64 head classes over every token, the pronouns and
        # possessives shared a 90-token class with `the` and `a`, and their removal moved it less
        # than noise did elsewhere
        # (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`).
        membership = tree_membership(tree)
        reference = tuple(
            membership @ matrix
            for matrix in head_filler_matrices(
                domain_ids, classes, tree.tokens, fillers, word_starts
            )
        )
        nodes = ClassNodes(
            tree,
            membership,
            tree_parents(tree),
            fillers,
            (reference[0], reference[1]),
            tuple(  # type: ignore[arg-type]
                np.asarray(matrix.sum(axis=1)).ravel() >= READABLE for matrix in reference
            ),
        )
    words = sum(len(document.split()) for document in domain)
    reading = DomainReading(
        tokenizer,
        merge_budget,
        lengths,
        is_reference,
        sum(len(sequence) - 2 for sequence in domain_ids) / words if words else 0.0,
        domain_ids,
        word_starts,
        tuple(len(doc.encode("utf-8")) for doc in domain),
        classes,
        groups,
        frequency,
        nodes,
        {},
        tree,
    )
    if is_reference:
        step("Reading reference patterns")
        reading.patterns.update(
            ((pattern.head, pattern.filler_class, pattern.lexical), pattern.diversity)
            for pattern in reading.read_patterns(domain_ids, word_starts)
        )
    return reading


def _domain_tokenizer(
    domain: Sequence[str],
    merge_budget: int | None,
    tokenizer: ByteLevelBPETokenizer | None,
    step: Callable[[str], None],
) -> tuple[ByteLevelBPETokenizer, int]:
    """The domain's tokenizer and merge budget: `tokenizer` as given, learned at `merge_budget`,
    or learned to `GRANULARITY_TARGET` tokens per word when neither is given."""
    if tokenizer is not None:
        return tokenizer, len(tokenizer.merges) if merge_budget is None else merge_budget
    if merge_budget is None:
        step(f"Learning tokenizer to {GRANULARITY_TARGET:.2f} tokens per word")
        tokenizer, _ = learn_to_granularity(domain)
        return tokenizer, len(tokenizer.merges)
    step("Learning tokenizer")
    learned = ByteLevelBPETokenizer.from_document_sequence(
        domain, max_merges=merge_budget, recognize_markers=True
    )
    return learned, merge_budget


def prepare_reference(
    reference: Sequence[str],
    merge_budget: int | None = None,
    *,
    tokenizer: ByteLevelBPETokenizer | None = None,
    head_lengths: Sequence[int] = (1, 2),
    progress: Callable[[str], None] | None = None,
) -> DomainReading:
    """Read a reference once, for `diagnose(corpus, reading)` on any number of corpora.

    `merge_budget` unset, the tokenizer is learned to `GRANULARITY_TARGET` tokens per word
    (`learn_to_granularity`), or is `tokenizer`'s own merges. `tokenizer`, given, replaces
    learning one: the first `merge_budget` merges of a longer run on the same reference are
    exactly the merges a run at `merge_budget` learns."""
    if not reference:
        raise ValueError("Corpus and reference must contain documents.")
    if merge_budget is not None and merge_budget < 0:
        raise ValueError("Require merges >= 0.")
    # Fewer is legitimate: learning stops early once no pair repeats.
    if tokenizer is not None and merge_budget is not None and len(tokenizer.merges) > merge_budget:
        raise ValueError("The tokenizer has more merges than merge_budget.")

    def step(message: str) -> None:
        if progress is not None:
            progress(message)

    tokenizer, merge_budget = _domain_tokenizer(reference, merge_budget, tokenizer, step)
    return _read_domain(
        reference,
        tokenizer,
        merge_budget,
        _head_lengths(head_lengths),
        True,
        step,
    )


def diagnose(
    documents: Sequence[str],
    reference: Sequence[str] | DomainReading | None = None,
    merge_budget: int | None = None,
    draws: int | None = None,
    min_trunk: int | None = None,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
    head_lengths: Sequence[int] | None = None,
    alpha: float = ALPHA,
    *,
    tokenizer: ByteLevelBPETokenizer | None = None,
) -> DiagnosisReport:
    """Read corpus slots against shared domain classes and optional size-matched noise.

    `reference` is documents, or a `prepare_reference` reading of them; a reading fixes the merge
    budget and head lengths, and any of them given here must agree. Unset, the domain (the
    reference, or the corpus without one) is tokenized to `GRANULARITY_TARGET` tokens per word,
    and heads are one and two tokens long. `tokenizer`, given, is the domain's tokenizer.

    `min_trunk` unset, a reference's draws set the least trunk read, of patterns and of class
    patterns (`_readable_trunk`); without one, the sibling tests read only what a single filler
    could flag. Given, it fixes both: a class pattern's trunk is then the draws' mean.

    Without a reference, a pattern is `weak` when its head shows fewer fillers than a random
    share of its parent's occurrences in the head tree would, past the same kind of bar over
    random splits (`analysis/siblings.py`); `rare` when it occurs fewer times than any slot the
    test can read; `unread` when its head has no sibling to compare with, or one too much alike
    to tell. A pattern the corpus lacks altogether, or a whole class shown weakly, needs a
    reference to be seen.

    A class pattern is under-shown when its t (excess over the draws, in their sd) exceeds the
    bar: of the largest t each draw reaches, at any class pattern, against the other draws, the
    k-th largest, k = alpha * (draws + 1). A random corpus passes it with probability `alpha`;
    `draws` unset is the count that keeps that chance near `alpha` for any one set of draws
    (`_draws_for`). Head rows and `absent` are read the same way. A fixed 3 sd with a floor of
    0.5 nats missed every removed class; a normal-theory bar flagged rare word pieces in random
    cuts, whose spread across draws has long tails
    (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`)."""
    if not documents or (isinstance(reference, Sequence) and not reference):
        raise ValueError("Corpus and reference must contain documents.")
    if (draws is not None and draws < 3) or (min_trunk is not None and min_trunk < 1):
        raise ValueError("Require draws >= 3 and min_trunk >= 1.")
    if not 0 < alpha < 1:
        raise ValueError("Require 0 < alpha < 1.")
    draws = _draws_for(alpha) if draws is None else draws
    rank = _rank(alpha, draws)

    def step(message: str) -> None:
        if progress is not None:
            progress(message)

    domain: DomainReading
    if isinstance(reference, DomainReading):
        given = {
            "merge_budget": (merge_budget, reference.merge_budget),
            "tokenizer": (
                None if tokenizer is None else tokenizer.merges,
                reference.tokenizer.merges,
            ),
            "head_lengths": (
                None if head_lengths is None else _head_lengths(head_lengths),
                reference.head_lengths,
            ),
        }
        conflicts = [name for name, (value, read) in given.items() if value not in (None, read)]
        if conflicts:
            raise ValueError(f"Settings differ from the reference reading: {', '.join(conflicts)}.")
        domain = reference
    else:
        if merge_budget is not None and merge_budget < 0:
            raise ValueError("Require merges >= 0.")
        if reference is None:
            learned, budget = _domain_tokenizer(documents, merge_budget, tokenizer, step)
            domain = _read_domain(
                documents,
                learned,
                budget,
                _head_lengths((1, 2) if head_lengths is None else head_lengths),
                False,
                step,
            )
        else:
            domain = prepare_reference(
                reference,
                merge_budget,
                tokenizer=tokenizer,
                head_lengths=(1, 2) if head_lengths is None else head_lengths,
                progress=progress,
            )
    has_reference = domain.is_reference
    tokenizer = domain.tokenizer
    lengths = domain.head_lengths
    classes = domain.classes
    nodes = domain.nodes
    references = domain.patterns
    spell = domain.spell
    read_patterns = domain.read_patterns
    domain_ids = domain.documents
    domain_sizes = domain.sizes
    top_tokens = {group.label: group.top_tokens for group in domain.groups}

    step("Framing documents")
    corpus_ids, corpus_starts = (
        (list(domain_ids), list(domain.word_starts))
        if not has_reference
        else framed_word_starts(documents, tokenizer, True)
    )
    step("Reading corpus patterns")
    patterns = read_patterns(corpus_ids, corpus_starts)
    # Per draw, the columns of its patterns and their trunks, diversities and expected fillers:
    # arrays, as a hundred draws of dicts held gigabytes.
    column: dict[PatternKey, int] = {}
    drawn_columns: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    corpus_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    # Per kind (syntactic, lexical): every draw's D and trunk at every read node.
    class_draws: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
    class_trunks: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])

    if has_reference:
        rng = random.Random(seed)
        for draw in range(draws):
            step(f"Reading reference draw {draw + 1}/{draws}")
            order = list(range(len(domain_ids)))
            rng.shuffle(order)
            sampled: list[list[int]] = []
            sampled_starts: list[list[bool]] = []
            size = 0
            # Whole documents preserve contexts; the last may overshoot the byte budget.
            for index in order:
                sampled.append(domain_ids[index])
                sampled_starts.append(domain.word_starts[index])
                size += domain_sizes[index]
                if size >= corpus_bytes:
                    break
            drawn_patterns = read_patterns(sampled, sampled_starts)
            drawn_columns.append(
                (
                    np.array(
                        [
                            column.setdefault((p.head, p.filler_class, p.lexical), len(column))
                            for p in drawn_patterns
                        ],
                        dtype=np.int64,
                    ),
                    np.array([p.trunk for p in drawn_patterns], dtype=np.float64),
                    np.array([p.diversity for p in drawn_patterns]),
                    np.array([p.expected_fillers for p in drawn_patterns]),
                )
            )
            if nodes is not None:
                drawn, drawn_counts = nodes.divergences(sampled, classes, sampled_starts)
                for kind in (0, 1):
                    class_draws[kind].append(drawn[kind])
                    class_trunks[kind].append(
                        np.asarray(drawn_counts[kind][nodes.read[kind]].sum(axis=1)).ravel()
                    )

    step("Building report and warnings")
    filler_counts: dict[PatternKey, Counter[int]] = defaultdict(Counter)
    for length in lengths:
        filler_counts.update(filler_occurrences(corpus_ids, classes, length, corpus_starts))
    # Rows are draws, columns patterns: diversity where a draw reads the pattern, else NaN; trunk,
    # 0 where a draw misses it.
    keys = list(column)
    diversity_noise = np.full((len(drawn_columns), len(keys)), np.nan)
    expected_noise = np.full((len(drawn_columns), len(keys)), np.nan)
    trunk_noise = np.zeros((len(drawn_columns), len(keys)))
    for draw, (columns, trunks, diversities, expected) in enumerate(drawn_columns):
        trunk_noise[draw, columns] = trunks
        enough = trunks >= READABLE
        diversity_noise[draw, columns[enough]] = diversities[enough]
        expected_noise[draw, columns[enough]] = expected[enough]
    # A corpus as large as its reference draws the whole reference every time: the draws do not
    # vary, and no bar can be read from them.
    varied = has_reference and corpus_bytes < sum(domain_sizes)
    head_trunk = (
        min_trunk
        if min_trunk is not None
        else _readable_trunk(diversity_noise, expected_noise, trunk_noise, draws - 2, rank)
        if keys and varied
        else READABLE
    )
    below = trunk_noise < head_trunk
    diversity_noise[below] = np.nan
    expected_noise[below] = np.nan
    # Diversity moves in steps of one filler (1 / expected fillers) and a trunk in steps of one:
    # a spread below the tested value's step is rounding, read as its variance step**2 / 12. The
    # step is the tested value's own: a corpus showing ` angels` 4 times, where the draws showed
    # it more often, moves in steps of 0.26 against their 0.04, and flagged a control.
    head_bar = (
        _bar(_drop_maxima(diversity_noise, draws - 2, expected_noise), rank)
        if keys and varied
        else None
    )
    absent_bar = _bar(_absent_maxima(trunk_noise, head_trunk), rank) if keys and varied else None
    # Without a reference each head is read against its sibling in the corpus's head tree:
    # random splits of their parent's occurrences are its draws (`analysis/siblings.py`).
    sibling: dict[PatternKey, tuple[SiblingTest, bool]] = {}
    node_tests: list[SiblingTest] = []
    node_read = np.zeros(0, dtype=bool)
    node_bar: float | None = None
    least_read: int | None = None
    if not has_reference and domain.tree is not None:
        step("Splitting each node's occurrences between its children")
        tree = domain.tree
        generator = np.random.default_rng(seed)
        tests = [
            test
            for length in lengths
            for test in sibling_tests(
                pattern_filler_counts(corpus_ids, classes, length, corpus_starts),
                tree,
                classes.n_classes,
                draws,
                generator,
            )
        ]
        leaf_count = len(tree.tokens)
        leaf_tests = [test for test in tests if test.node < leaf_count]
        node_tests = [test for test in tests if test.node >= leaf_count]
        # Two families, as with a reference: head rows and class rows.
        head_bar, leaf_read = sibling_bar(leaf_tests, rank)
        node_bar, node_read = sibling_bar(node_tests, rank)
        # `rare` is a slot shown fewer times than any slot the test can read; one that is shown
        # often but cannot be flagged has a sibling too much alike to tell (recipe verbs sharing
        # their only ingredient), and is `unread`.
        least_read = min(
            (test.trunk for test, read in zip(leaf_tests, leaf_read, strict=True) if read),
            default=None,
        )
        for test, read in zip(leaf_tests, leaf_read, strict=True):
            head = (tree.tokens[test.node], *test.suffix)
            sibling[(head, test.filler_class, test.lexical)] = (test, bool(read))
    rows = []
    for pattern in patterns:
        if pattern.trunk < head_trunk:
            continue
        key = (pattern.head, pattern.filler_class, pattern.lexical)
        ref = references.get(key)
        position = column.get(key)
        values = (
            diversity_noise[:, position][~np.isnan(diversity_noise[:, position])]
            if position is not None
            else np.zeros(0)
        )
        mean = float(values.mean()) if len(values) else None
        sd = float(values.std(ddof=1)) if len(values) >= 2 else None
        verdict: Verdict
        if not has_reference:
            found = sibling.get(key)
            if found is None:
                verdict = "unread"
            else:
                test, read = found
                # In diversity units, as the draws are with a reference.
                mean = test.mean / pattern.expected_fillers
                sd = test.sd / pattern.expected_fillers
                verdict = (
                    ("weak" if head_bar is not None and test.t > head_bar else "within-noise")
                    if read
                    else "rare"
                    if least_read is None or pattern.trunk < least_read
                    else "unread"
                )
        elif (
            head_bar is None or ref is None or len(values) < draws - 2 or mean is None or sd is None
        ):
            verdict = "unread"
        elif (
            _t(
                np.array([mean - pattern.diversity]),
                _with_step(np.array([sd]), np.array([1 / pattern.expected_fillers])),
            )[0]
            > head_bar
        ):
            verdict = "under-shown"
        else:
            verdict = "within-noise"
        rows.append(
            DiagnosisRow(
                domain.spell_all(pattern.head),
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
                lexical=pattern.lexical,
            )
        )
    # A pattern random cuts of this size show and the corpus does not would have no row: a
    # removed slot (`experiments/class_patterns`) left nothing to flag. Its trunk is read against
    # the draws' own bar, as diversity is.
    by_key = {(p.head, p.filler_class, p.lexical): p for p in patterns}
    for key in keys if absent_bar is not None else ():
        corpus_pattern = by_key.get(key)
        trunk = corpus_pattern.trunk if corpus_pattern else 0
        trunks = trunk_noise[:, column[key]]
        trunk_mean, trunk_sd = float(trunks.mean()), float(trunks.std(ddof=1))
        if trunk >= head_trunk or trunk_mean < head_trunk:
            continue
        spread = _with_step(np.array([trunk_sd]), np.ones(1))
        if _t(np.array([trunk_mean - trunk]), spread)[0] <= absent_bar:
            continue
        values = diversity_noise[:, column[key]]
        values = values[~np.isnan(values)]
        head, filler_class, lexical = key
        rows.append(
            DiagnosisRow(
                domain.spell_all(head),
                len(head),
                filler_class,
                top_tokens[filler_class],
                trunk,
                corpus_pattern.fillers if corpus_pattern else 0,
                corpus_pattern.diversity if corpus_pattern else 0.0,
                references.get(key),
                float(values.mean()) if len(values) else None,
                float(values.std(ddof=1)) if len(values) >= 2 else None,
                "absent",
                tuple(spell(token) for token, _ in filler_counts[key].most_common(5)),
                trunk_mean,
                trunk_sd,
                lexical,
            )
        )
    rows.sort(
        key=lambda row: (row.lexical, VERDICTS.index(row.verdict), -_missing(row), -row.trunk)
    )
    class_pattern_rows: list[DiagnosisClassRow] = []
    class_nodes_read = 0
    critical: float | None = None
    class_trunk: float | None = None
    if nodes is not None:
        step("Reading class patterns")
        class_values, class_shown = nodes.divergences(corpus_ids, classes, corpus_starts)
        kinds = [kind for kind in (0, 1) if nodes.read[kind].any()]
        # Positions among each kind's read nodes that are tested.
        readable = [np.ones(int(read.sum()), dtype=bool) for read in nodes.read]
        if kinds and varied:
            noises = [np.vstack(class_draws[kind]) for kind in kinds]
            trunk_means = [np.vstack(class_trunks[kind]).mean(axis=0) for kind in kinds]
            if min_trunk is None:
                unigram = unigram_vector(corpus_ids, nodes.fillers)
                class_trunk = _readable_nodes(
                    noises,
                    trunk_means,
                    [
                        nodes.single_filler(kind, trunks, unigram)
                        for kind, trunks in zip(kinds, trunk_means, strict=True)
                    ],
                    rank,
                )
            else:
                class_trunk = float(min_trunk)
            for kind, trunks in zip(kinds, trunk_means, strict=True):
                readable[kind] = trunks >= class_trunk
            critical = _bar(
                _null_maxima(
                    [
                        noise[:, readable[kind]]
                        for kind, noise in zip(kinds, noises, strict=True)
                        if readable[kind].any()
                    ]
                ),
                rank,
            )
        class_nodes_read = int(sum(read.sum() for read in readable))
        short = {
            (row.head, row.lexical)
            for row in rows
            if row.head_length == 1 and row.verdict in ("under-shown", "absent")
        }
        head_frequency = domain.frequency
        leaves = nodes.membership.tolil().rows
        for kind, lexical in ((0, False), (1, True)):
            read_nodes = np.flatnonzero(nodes.read[kind])[readable[kind]]
            if critical is None or not len(read_nodes):
                continue
            noise = np.vstack(class_draws[kind])[:, readable[kind]]
            noise_mean, noise_sd = noise.mean(axis=0), noise.std(axis=0, ddof=1)
            excess = class_values[kind][readable[kind]] - noise_mean
            t = _t(excess, noise_sd)
            score = np.full(len(nodes.parents), -np.inf)
            score[read_nodes] = t
            flagged = score > critical
            for node in _branch_maxima(score, flagged, nodes.parents):
                index = int(np.searchsorted(read_nodes, node))
                heads = sorted(
                    (nodes.tree.tokens[leaf] for leaf in leaves[node]),
                    key=lambda token: (-head_frequency[token], token),
                )
                corpus_row = class_shown[kind].getrow(node)
                reference_row = nodes.reference[kind].getrow(node)

                def top(row: sparse.csr_matrix) -> tuple[str, ...]:
                    order = np.argsort(-row.data, kind="stable")[:5]
                    return tuple(spell(nodes.fillers[row.indices[i]]) for i in order)

                class_pattern_rows.append(
                    DiagnosisClassRow(
                        node,
                        tuple(spell(token) for token in heads[:12]),
                        len(heads),
                        int(corpus_row.sum()),
                        int(reference_row.sum()),
                        float(class_values[kind][readable[kind]][index]),
                        float(noise_mean[index]),
                        float(noise_sd[index]),
                        float(excess[index]),
                        float(t[index]),
                        "under-shown",
                        top(corpus_row),
                        top(reference_row),
                        tuple(spell(token) for token in heads if (spell(token), lexical) in short),
                        lexical,
                    )
                )
        class_pattern_rows.sort(key=lambda row: (row.lexical, -row.t))
    if not has_reference and domain.tree is not None:
        critical = node_bar
        class_nodes_read = int(node_read.sum())
        leaf_count = len(domain.tree.tokens)
        members = tree_membership(domain.tree).tolil().rows
        head_frequency = domain.frequency

        def top_of(counts: Mapping[int, int]) -> tuple[str, ...]:
            return tuple(spell(token) for token, _ in Counter(counts).most_common(5))

        for test, read in zip(node_tests, node_read, strict=True):
            if not read or critical is None or test.t <= critical:
                continue
            heads = sorted(
                (domain.tree.tokens[leaf] for leaf in members[test.node - leaf_count]),
                key=lambda token: (-head_frequency[token], token),
            )
            class_pattern_rows.append(
                DiagnosisClassRow(
                    test.node - leaf_count,
                    tuple(domain.spell_all((token, *test.suffix)) for token in heads[:12]),
                    len(heads),
                    test.trunk,
                    sum(test.parent_counts.values()),
                    float(test.fillers),
                    test.mean,
                    test.sd,
                    test.missing,
                    test.t,
                    "weak",
                    top_of(test.child_counts),
                    top_of(test.parent_counts),
                    (),
                    test.lexical,
                    test.filler_class,
                )
            )
        class_pattern_rows.sort(key=lambda row: (row.lexical, -row.excess))
    words = sum(len(doc.split()) for doc in documents)
    tokens_per_word = sum(len(sequence) - 2 for sequence in corpus_ids) / words if words else 0.0
    warnings = []
    if len(documents) < 1000 or words < 50000:
        warnings.append(
            "Corpus has fewer than 1,000 documents or 50,000 words; "
            "classes and correlations may be unreliable."
        )
    if has_reference and corpus_bytes > sum(domain_sizes):
        warnings.append(
            "Corpus is larger than its reference in bytes; draws use the full reference."
        )
    if sum(domain_sizes) < MIN_DOMAIN_BYTES:
        warnings.append(
            f"The {'reference' if has_reference else 'corpus'} holds "
            f"{sum(domain_sizes) / 1e6:.1f} MB, under {MIN_DOMAIN_BYTES / 1e6:.1f} MB, the "
            "smallest size at which the verdicts held across five domains; classes and absent "
            "verdicts may be unreliable."
        )
    low_edge, high_edge = GRANULARITY_RANGE
    if not low_edge <= domain.tokens_per_word <= high_edge:
        warnings.append(
            f"The {'reference' if has_reference else 'corpus'} reads at "
            f"{domain.tokens_per_word:.2f} tokens per word, outside {low_edge:.2f} to "
            f"{high_edge:.2f}, where the verdicts were shown to hold; leave the merge budget "
            "unset to tokenize at "
            f"{GRANULARITY_TARGET:.2f}."
        )
    return DiagnosisReport(
        len(documents),
        words,
        tokens_per_word,
        domain.merge_budget,
        domain.groups,
        tuple(rows),
        tuple(warnings),
        DiagnosisSettings(
            domain.merge_budget,
            draws,
            head_trunk,
            seed,
            lengths,
            class_trunk,
            alpha,
        ),
        tuple(class_pattern_rows),
        class_nodes_read,
        critical,
        head_bar,
        absent_bar,
        has_reference,
    )


def _missing(row: DiagnosisRow) -> float:
    """Fillers a row's slot shows fewer than its draws, or its siblings, would: what a model
    trained on the corpus does not see there."""
    if row.margin_mean is None or not row.diversity:
        return 0.0
    return row.margin_mean * row.fillers / row.diversity - row.fillers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", nargs="+", type=Path)
    parser.add_argument(
        "--merges",
        type=int,
        help=(
            "Merge budget of the domain's tokenizer (default: the fewest merges that reach "
            f"{GRANULARITY_TARGET:.2f} tokens per word)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            f"Diagnose even when the domain reads outside {GRANULARITY_RANGE[0]:.2f} to "
            f"{GRANULARITY_RANGE[1]:.2f} tokens per word"
        ),
    )
    parser.add_argument(
        "--head-lengths",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Tokens per head (default: 1 2)",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (args.merges is not None and args.merges < 0) or (args.limit is not None and args.limit < 1):
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
    domain = documents if reference is None else reference
    tokenizer, _ = _domain_tokenizer(domain, args.merges, None, print)
    # Checked before the diagnosis, which takes minutes to hours on a real corpus: the verdicts were
    # shown to hold only within the range
    # (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`).
    granularity = tokens_per_word_by_budget(domain, tokenizer.merges)[-1]
    low_edge, high_edge = GRANULARITY_RANGE
    if not low_edge <= granularity <= high_edge and not args.force:
        print(
            f"The {'reference' if reference is not None else 'corpus'} reads at "
            f"{granularity:.2f} tokens per word with {len(tokenizer.merges):,} merges, outside "
            f"{low_edge:.2f} to {high_edge:.2f}, where the verdicts were shown to hold. Leave "
            "--merges unset to tokenize at "
            f"{GRANULARITY_TARGET:.2f}, or pass --force to diagnose anyway.",
            file=sys.stderr,
        )
        return 2
    report = diagnose(
        documents,
        reference,
        merge_budget=args.merges,
        head_lengths=args.head_lengths,
        tokenizer=tokenizer,
    )
    output = args.output or Path("diagnoses") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(report.markdown(), encoding="utf-8")
    print(report.summary())
    syntactic_rows = [row for row in report.class_rows if not row.lexical]
    flagged = sum(row.verdict == "under-shown" for row in syntactic_rows)
    if syntactic_rows:
        print(f"{len(syntactic_rows)} syntactic class patterns read, {flagged} under-shown")
    for warning in report.warnings:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
