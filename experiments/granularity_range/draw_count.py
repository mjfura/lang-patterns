"""How many draws make the head bar stable? Its false-flag rate given one set of draws.

The bar is built so that a corpus exchangeable with the draws passes it about once in
(draws + 1), over random draws. With one fixed set of 16 draws, 6 of 48 random document cuts were
flagged and 0 of 48 with fresh draws for each (`README.md`): the rate a diagnosis meets depends on
its draws. Here one pool of independent draws of F, each the bytes of `control`, serves both
sides: for a set S of n draws, the held-out draws play random corpora against S's bar.

    draw_count.py pool <size>    read <size> draws of F at 1.50 tokens per word
    draw_count.py rates          bar and conditional false-flag rate for n = 16, 32, 64, 128

Head rows only: class rows flagged no random cut. No command runs on import.
"""

import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path[:0] = [str(Path(__file__).parent)]

import granularity_range as g  # noqa: E402

from corpus_generalization.diagnose import (  # noqa: E402
    READABLE,
    DomainReading,
    _drop_bar,
    _readable_trunk,
    _t,
    _with_step,
    prepare_reference,
)
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402

POOL = g.RESULTS / "draw_count" / "pool.npz"
SIZES = (16, 32, 64, 128)
SETS = 20
HELD_OUT = 40

_reading: DomainReading | None = None
_target = 0


def _start(reading: DomainReading, target: int) -> None:
    global _reading, _target
    _reading, _target = reading, target


def _draw(seed: int) -> dict[Any, tuple[int, float, float]]:
    """Every readable pattern of one draw, sampled as `diagnose` samples."""
    assert _reading is not None
    order = list(range(len(_reading.documents)))
    random.Random(seed).shuffle(order)
    sequences, starts, size = [], [], 0
    for index in order:
        sequences.append(_reading.documents[index])
        starts.append(_reading.word_starts[index])
        size += _reading.sizes[index]
        if size >= _target:
            break
    return {
        (p.head, p.filler_class, p.lexical): (p.trunk, p.diversity, p.expected_fillers)
        for p in _reading.read_patterns(sequences, starts)
        if p.trunk >= READABLE
    }


def pool(size: int) -> None:
    budget = json.loads((g.RESULTS / "budgets.json").read_text())["budgets"]["1.50"]
    _, documents = g.natural_gap.split()
    reading = prepare_reference(
        documents,
        budget,
        tokenizer=ByteLevelBPETokenizer.from_merges(g.merges()[:budget]),
        head_lengths=(1,),
        progress=g.divergence.progress,
    )
    target = sum(len(document.encode("utf-8")) for document in g.base.corpus("control"))
    with ProcessPoolExecutor(g.WORKERS, initializer=_start, initargs=(reading, target)) as workers:
        draws = []
        for seed, drawn in enumerate(workers.map(_draw, range(size))):
            draws.append(drawn)
            g.divergence.progress(f"draw {seed}: {len(drawn)} patterns")
    keys = sorted({key for drawn in draws for key in drawn})
    column = {key: index for index, key in enumerate(keys)}
    trunks = np.zeros((size, len(keys)))
    diversity = np.full((size, len(keys)), np.nan)
    expected = np.full((size, len(keys)), np.nan)
    for row, drawn in enumerate(draws):
        for key, (trunk, value, fillers) in drawn.items():
            trunks[row, column[key]] = trunk
            diversity[row, column[key]] = value
            expected[row, column[key]] = fillers
    lexical = np.array([key[2] for key in keys])
    POOL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        POOL, trunks=trunks, diversity=diversity, expected=expected, lexical=lexical
    )


def _bar(
    trunks: np.ndarray, diversity: np.ndarray, expected: np.ndarray
) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    """The trunk and bar `diagnose` derives from these draws, and the patterns it reads."""
    least = len(trunks) - 2
    trunk = _readable_trunk(diversity, expected, trunks, least)
    shown = trunks >= trunk
    values = np.where(shown, diversity, np.nan)
    bar = _drop_bar(values, least, np.where(shown, expected, np.nan))
    reads = shown.sum(axis=0) >= least
    mean = np.full(len(reads), np.nan)
    sd = np.full(len(reads), np.nan)
    mean[reads] = np.nanmean(values[:, reads], axis=0)
    sd[reads] = np.nanstd(values[:, reads], axis=0, ddof=1)
    return trunk, bar, reads, mean, sd


def _flagged(
    reading: tuple[int, float, np.ndarray, np.ndarray, np.ndarray],
    trunks: np.ndarray,
    diversity: np.ndarray,
    expected: np.ndarray,
) -> bool:
    """Whether one draw, read as a corpus, passes the bar at a syntactic pattern."""
    trunk, bar, reads, mean, sd = reading
    tested = reads & (trunks >= trunk)
    t = _t(mean[tested] - diversity[tested], _with_step(sd[tested], 1 / expected[tested]))
    return bool((t > bar).any())


def rates() -> None:
    data = np.load(POOL)
    syntactic = ~data["lexical"]
    trunks = data["trunks"][:, syntactic]
    diversity = data["diversity"][:, syntactic]
    expected = data["expected"][:, syntactic]
    rng = np.random.default_rng(0)
    summary: dict[str, Any] = {}
    for n in SIZES:
        if n + HELD_OUT > len(trunks):
            continue
        bars, set_rates = [], []
        for _ in range(SETS):
            order = rng.permutation(len(trunks))
            chosen, held_out = order[:n], order[n : n + HELD_OUT]
            reading = _bar(trunks[chosen], diversity[chosen], expected[chosen])
            flags = [
                _flagged(reading, trunks[test], diversity[test], expected[test])
                for test in held_out
            ]
            bars.append(reading[1])
            set_rates.append(sum(flags) / len(flags))
        summary[str(n)] = {
            "bars": bars,
            "rates": set_rates,
            "bar_mean": statistics.mean(bars),
            "bar_sd": statistics.stdev(bars),
            "rate_mean": statistics.mean(set_rates),
            "rate_min": min(set_rates),
            "rate_max": max(set_rates),
            "design_rate": 1 / (n + 1),
        }
        print(
            f"{n} draws: bar {statistics.mean(bars):.2f} ± {statistics.stdev(bars):.2f}, "
            f"rate {statistics.mean(set_rates):.3f} (sets {min(set_rates):.3f} to "
            f"{max(set_rates):.3f}), design {1 / (n + 1):.3f}"
        )
    g.divergence.write_json(g.RESULTS / "draw_count" / "rates.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    drawn = commands.add_parser("pool")
    drawn.add_argument("size", type=int)
    commands.add_parser("rates")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    match args.command:
        case "pool":
            pool(args.size)
        case "rates":
            rates()


if __name__ == "__main__":
    main()
