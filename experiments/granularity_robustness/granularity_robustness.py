"""The tokens-per-word range across corpus size, pool draw and two knowledge domains.

`docs/superpowers/specs/2026-10-28-granularity-robustness-design.md` records the design and
predictions before running; `docs/superpowers/plans/2026-10-28-granularity-robustness.md` the
steps. Corpus side only: no model is trained.

A variant is `<corpus>@<size>#<pool>`: `wikipedia@0.25#1` is the first quarter of Wikipedia's
pool #1, `tinystories@4#0` TinyStories' 4x pool. Every variant is read at the eleven granularities
of `experiments/granularity_domains`, with the same 22 cuts, gate and pass rule. Pools, merges and
trees go to `runs/granularity_robustness/<variant>/` (gitignored), results to
`results/<variant>/`. No command runs on import.
"""

import argparse
import json
import os
import random
import re
import resource
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "granularity_robustness"
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "granularity_domains")]

import granularity_domains as domains  # noqa: E402

from corpus_generalization.analysis import syntactic  # noqa: E402
from corpus_generalization.sources import huggingface_documents  # noqa: E402
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer, Merge  # noqa: E402
from corpus_generalization.tokenizer.granularity import (  # noqa: E402
    learn_to_granularity,
    tokens_per_word_by_budget,
)

base = domains.base
natural_gap = domains.natural_gap
subset_selection = domains.subset_selection

CORPORA = ("tinystories", "fineweb", "wikipedia", "pubmed", "legal")
SIZES = (0.0625, 0.25, 1.0, 4.0)
POOLS = (0, 1, 2)
TARGETS = tuple(f"{t:.2f}" for t in domains.TARGETS)
# Read in `experiments/granularity_domains`; their verdicts are taken from its summary.
DONE = frozenset({"tinystories@1#0", "fineweb@1#0", "wikipedia@1#0"})
# Three quarters of the machine's 64 GB: a memory-pressure kill takes the whole machine down.
MEMORY_CEILING = 48 * 10**9
# 18 cores, and every worker runs BLAS on OMP_NUM_THREADS=2, as in `granularity_domains`.
MAX_WORKERS = 9
# Above SciPy's two measured peaks (`experiments/ward_memory`: 1.38 GB at 11,459 tokens, 1.76 GB
# at 15,893) when added to two copies of the condensed table.
SCIPY_OVERHEAD = 2 * 10**9
CUTS = 22
# Rows streamed per general corpus: enough for 6 units (the 4x pool and two 1x slices after it).
ROWS = {"tinystories": 100_000, "fineweb": 30_000, "wikipedia": 10_000}
# The caches earlier experiments read, written as prefixes of the streams above when absent.
LEGACY = {
    "tinystories": (domains.TINYSTORIES, 25_000),
    "fineweb": (natural_gap.clean_domains.cache_of("fineweb"), 5_000),
    "wikipedia": (domains.WIKIPEDIA, 1_500),
}
PUBMED_LAST_CHUNK = 1000
VARIANT = re.compile(r"^([a-z]+)@([0-9.]+)#([0-9]+)$")


@dataclass(frozen=True)
class Variant:
    corpus: str
    size: float
    pool: int

    @property
    def name(self) -> str:
        return f"{self.corpus}@{self.size:g}#{self.pool}"


def parse_variant(name: str) -> Variant:
    match = VARIANT.match(name)
    if match is None:
        raise ValueError(f"{name!r} is not <corpus>@<size>#<pool>, e.g. wikipedia@0.25#1.")
    corpus, size, pool = match[1], float(match[2]), int(match[3])
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus {corpus!r}; one of {', '.join(CORPORA)}.")
    if size not in SIZES:
        raise ValueError(f"Unknown size {match[2]}; one of {', '.join(f'{s:g}' for s in SIZES)}.")
    if pool not in POOLS:
        raise ValueError(f"Pool {pool} is outside 0-2.")
    if size == 4.0 and pool != 0:
        raise ValueError("4x is read in pool 0 only.")
    return Variant(corpus, size, pool)


def variants() -> list[Variant]:
    return [
        Variant(corpus, size, pool)
        for corpus in CORPORA
        for size in SIZES
        for pool in POOLS
        if size < 4.0 or pool == 0
    ]


def new_variants() -> list[Variant]:
    return [v for v in variants() if v.name not in DONE]


# The spec's order 4-7: the cheapest readings first.
STEPS: dict[str, list[str]] = {
    "small": [Variant(c, s, p).name for c in CORPORA for s in (0.0625, 0.25) for p in POOLS],
    "pools": [Variant(c, 1.0, p).name for c in CORPORA for p in (1, 2)],
    "domains": [Variant(c, 1.0, 0).name for c in ("pubmed", "legal")],
    "4x": [Variant(c, 4.0, 0).name for c in CORPORA],
}


def layout(
    sizes: Sequence[int], unit: int, first: Sequence[int], rest_from: int
) -> dict[tuple[float, int], list[int]]:
    """Indices into the cache for every `(size, pool)`. Pool #0 at 1x is `first`; its 4x pool
    continues from `rest_from`; pools #1 and #2 are the next slices after the 4x pool; a smaller
    size is the leading documents of its pool at 1x. A pool #2 the cache cannot fill is left out;
    any earlier shortfall raises `ValueError`."""
    fill = subset_selection.fill
    rest = list(range(rest_from, len(sizes)))
    cut: dict[tuple[float, int], list[int]] = {(1.0, 0): list(first)}
    cut[(4.0, 0)] = fill(list(sizes), [*first, *rest], 4 * unit)
    start = max(cut[(4.0, 0)]) + 1
    for pool in (1, 2):
        try:
            cut[(1.0, pool)] = fill(list(sizes), list(range(start, len(sizes))), unit)
        except ValueError:
            if pool == 2:
                break
            raise
        start = cut[(1.0, pool)][-1] + 1
    for pool in POOLS:
        if (1.0, pool) in cut:
            for size in SIZES[:2]:
                cut[(size, pool)] = fill(list(sizes), cut[(1.0, pool)], size * unit)
    return cut


def scipy_peak_bytes(tokens: int) -> int:
    """SciPy's projected peak: two copies of the condensed table, plus the reading around it."""
    return 8 * tokens * (tokens - 1) + SCIPY_OVERHEAD


def uses_scipy(tokens: int) -> bool:
    return scipy_peak_bytes(tokens) <= MEMORY_CEILING


def _vector_linkage(features: np.ndarray, method: str) -> np.ndarray:
    import fastcluster

    return np.asarray(fastcluster.linkage_vector(features, method=method))


@contextmanager
def clustering(tokens: int) -> Iterator[str]:
    """SciPy's Ward where its table fits, `linkage_vector` above: the same tree
    (`experiments/ward_memory`), without the table."""
    if uses_scipy(tokens):
        yield "scipy"
        return
    original = syntactic.linkage
    syntactic.linkage = _vector_linkage  # type: ignore[assignment]
    try:
        yield "vector"
    finally:
        syntactic.linkage = original


def workers_for(parent_peak: int, child_peak: int) -> int:
    """The most workers whose peaks, with the parent's, stay under the memory ceiling."""
    return max(1, min(MAX_WORKERS, CUTS, (MEMORY_CEILING - parent_peak) // child_peak))


def workers_from(projected: int, seen: Sequence[tuple[int, int, int]]) -> int:
    """Workers for a reading whose SciPy tree projects to `projected` bytes, from the step's
    readings so far as (projected, parent peak, child peak). The parent's peak follows its own tree
    (measured within 10% of the projection), so it is the projection plus the largest overshoot
    seen; the step's largest parent peak pinned every later reading to one worker."""
    overshoot = max(0, *(parent - plan for plan, parent, _ in seen))
    return workers_for(projected + overshoot, max(child for *_, child in seen))


def legal_document(paragraphs: Sequence[str]) -> str:
    """One ECtHR case: its fact paragraphs, one per line."""
    return "\n".join(p.strip() for p in paragraphs if p.strip())


def pubmed_chunk(number: int) -> str:
    return f"chunk/pubmed23n{number:04d}.jsonl"


def agrees(variant: dict[str, bool | None], base: dict[str, bool]) -> bool:
    """Every granularity the variant reached gives the base pool's verdict."""
    return all(base.get(t) == holds for t, holds in variant.items() if holds is not None)


def first_failure_above(holds: dict[str, bool | None], floor: str = "2.00") -> str | None:
    return next((t for t in sorted(holds) if t > floor and holds[t] is False), None)


@cache
def unit() -> int:
    """F's bytes: the 1x pool every corpus is sized to."""
    return base.size(natural_gap.split()[1])


def cache_of(corpus: str) -> Path:
    return OUT / f"{corpus}_documents.json"


def _stream(corpus: str) -> list[str]:
    rows = ROWS[corpus]
    if corpus == "tinystories":
        return huggingface_documents("roneneldan/TinyStories", "train", "text", rows)
    if corpus == "fineweb":
        return huggingface_documents("HuggingFaceFW/fineweb", "train", "text", rows)
    return huggingface_documents("wikimedia/wikipedia", "train", "text", rows, "20231101.en")


def _pubmed(need: int) -> list[str]:
    """The fewest highest-numbered chunks whose abstracts reach `need` bytes, in ascending order."""
    chunks: dict[int, list[str]] = {}
    number = PUBMED_LAST_CHUNK
    while sum(base.size(c) for c in chunks.values()) < need:
        chunks[number] = huggingface_documents(
            "MedRAG/pubmed", "train", "content", None, None, [pubmed_chunk(number)]
        )
        print(f"pubmed: chunk {number}, {base.size(chunks[number]):,} bytes", flush=True)
        number -= 1
    return [d for n in sorted(chunks) for d in chunks[n]]


def _legal() -> list[str]:
    from datasets import load_dataset

    documents = []
    for split in ("train", "validation", "test"):
        rows = load_dataset("coastalcph/lex_glue", "ecthr_a", split=split)
        documents += [d for row in rows if (d := legal_document(row["text"]))]
    return documents


def download(corpus: str) -> None:
    if corpus in ROWS:
        documents = _stream(corpus)
        legacy, rows = LEGACY[corpus]
        if not legacy.exists():
            legacy.write_text(json.dumps(documents[:rows]))
            print(f"{corpus}: wrote the earlier cache {legacy} ({rows:,} documents)", flush=True)
    elif corpus == "pubmed":
        documents = _pubmed(6 * unit())
    else:
        documents = _legal()
    cache_of(corpus).write_text(json.dumps(documents))
    size = base.size(documents)
    print(
        f"{corpus}: {len(documents):,} documents, {size:,} bytes, {size / unit():.2f} units",
        flush=True,
    )
    # A streamed dataset can leave threads that keep the interpreter from exiting.
    os._exit(0)


@cache
def documents_of(corpus: str) -> list[str]:
    return list(json.loads(cache_of(corpus).read_text()))


@cache
def cut_of(corpus: str) -> dict[tuple[float, int], list[int]]:
    documents = documents_of(corpus)
    sizes = [len(d.encode("utf-8")) for d in documents]
    if corpus == "fineweb":
        # F is `natural_gap.split()`'s pool: the leading 5,000 documents shuffled with seed 0,
        # the first 400 kept out for evaluation, the rest in cache order.
        order = list(range(LEGACY["fineweb"][1]))
        random.Random(0).shuffle(order)
        first, rest_from = sorted(order[natural_gap.EVAL_DOCUMENTS :]), len(order)
    else:
        first = subset_selection.fill(sizes, list(range(len(sizes))), unit())
        rest_from = len(first)
    return layout(sizes, unit(), first, rest_from)


def pool_of(variant: Variant) -> list[str]:
    cut = cut_of(variant.corpus)
    if (variant.size, variant.pool) not in cut:
        raise SystemExit(f"{variant.name}: the corpus has too little text; pool #2 is dropped.")
    documents = documents_of(variant.corpus)
    return [documents[i] for i in cut[(variant.size, variant.pool)]]


def budgets_of(pool: Sequence[str]) -> dict[str, int | None]:
    """The merges at which `pool` first reaches each granularity, learned down to 1.30."""
    tokenizer, _ = learn_to_granularity(pool, min(domains.TARGETS))
    curve = tokens_per_word_by_budget(pool, tokenizer.merges)
    return {t: next((k for k, v in enumerate(curve) if v <= float(t)), None) for t in TARGETS}


def verify(corpora: Sequence[str]) -> None:
    """Pool #0 at 1x is the pool `experiments/granularity_domains` read: F itself for FineWeb,
    and every corpus reaches each granularity at the merges recorded there."""
    recorded = json.loads((domains.RESULTS / "summary.json").read_text())["domains"]
    failed = False
    for corpus in corpora:
        pool = pool_of(Variant(corpus, 1.0, 0))
        check: dict[str, object] = {"documents": len(pool), "bytes": base.size(pool)}
        if corpus == "fineweb":
            check["is_f"] = pool == natural_gap.split()[1]
            failed |= not check["is_f"]
        budgets = budgets_of(pool)
        expected = {t: v["merge_budget"] for t, v in recorded[corpus]["targets"].items()}
        check |= {"budgets": budgets, "recorded": expected, "match": budgets == expected}
        failed |= budgets != expected
        domains.write(RESULTS / f"verify_{corpus}.json", check)
        print(f"{corpus}: {'MATCH' if check['match'] else 'MISMATCH'} {json.dumps(check)}")
    if failed:
        raise SystemExit("Pool #0 does not reproduce; stop for a decision.")


def prepared(name: str) -> dict[str, Any] | None:
    path = RESULTS / name / "prepared.json"
    return cast(dict[str, Any], json.loads(path.read_text())) if path.exists() else None


def merges_of(name: str) -> tuple[Merge, ...]:
    raw = json.loads((RUNS / name / "merges.json").read_text())
    return tuple(((left, right), new) for (left, right), new in raw)


def prepare(name: str) -> None:
    """`granularity_domains`' preparation of one variant's pool, its trees at 1.50 on
    `linkage_vector` where SciPy's table would not fit, then the clustered tokens and the tree
    of every reachable granularity."""
    if name in DONE:
        raise SystemExit(f"{name} was read in experiments/granularity_domains.")
    variant = parse_variant(name)
    pool = pool_of(variant)
    trees: dict[str, str] = {}

    def at_150(tokenizer: ByteLevelBPETokenizer) -> AbstractContextManager[str]:
        return clustering(domains.clustered_tokens(pool, tokenizer))

    domains.prepare_pool(pool, RUNS / name, RESULTS / name, before_reading=at_150)
    record = prepared(name)
    assert record is not None
    merges = merges_of(name)
    tokens: dict[str, int] = {}
    for target, budget in record["budgets"].items():
        if budget is not None:
            tokenizer = ByteLevelBPETokenizer.from_merges(merges[:budget])
            tokens[target] = domains.clustered_tokens(pool, tokenizer)
            trees[target] = "scipy" if uses_scipy(tokens[target]) else "vector"
    record |= {
        "pool": {"corpus": variant.corpus, "size": variant.size, "pool": variant.pool},
        "clustered_tokens": tokens,
        "tree": trees,
        "projected_scipy_bytes": {t: scipy_peak_bytes(n) for t, n in tokens.items()},
    }
    domains.write(RESULTS / name / "prepared.json", record)


def expand(names: Sequence[str]) -> list[str]:
    """Variant names, with a step's name standing for its variants."""
    return [v for name in names for v in (STEPS.get(name) or [parse_variant(name).name])]


def prepare_all(names: Sequence[str], jobs: int) -> None:
    """Each variant in its own process, `jobs` at a time, skipping those already prepared."""
    todo = [n for n in expand(names) if prepared(n) is None or "tree" not in cast(Any, prepared(n))]
    logs = RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> tuple[str, int]:
        with (logs / f"prepare_{name}.log").open("w") as log:
            command = [sys.executable, __file__, "prepare", name]
            return name, subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(jobs) as running:
        for name, code in running.map(one, todo):
            print(f"PREPARED {name} exit {code}", flush=True)


def costs() -> None:
    rows = []
    for variant in new_variants():
        record = prepared(variant.name)
        if record is None or "tree" not in record:
            continue
        for target in TARGETS:
            rows.append(
                {
                    "variant": variant.name,
                    "target": target,
                    "merges": record["budgets"][target],
                    "clustered_tokens": record["clustered_tokens"].get(target),
                    "tree": record["tree"].get(target),
                    "projected_scipy_gb": round(
                        record["projected_scipy_bytes"].get(target, 0) / 1e9, 1
                    ),
                    "reachable": record["budgets"][target] is not None,
                }
            )
    totals = {
        step: {
            tree: sum(r["variant"] in names and r["tree"] == tree for r in rows)
            for tree in ("scipy", "vector", None)
        }
        for step, names in STEPS.items()
    }
    domains.write(RESULTS / "costs.json", {"rows": rows, "totals": totals})
    for r in rows:
        print(
            f"{r['variant']:<22} {r['target']}  merges {r['merges'] or '-':>7}  "
            f"tokens {r['clustered_tokens'] or '-':>7}  {r['tree'] or 'unreachable':<11} "
            f"{r['projected_scipy_gb']:>5} GB"
        )
    print(json.dumps(totals, indent=1))


def reading_order(step: str) -> list[tuple[str, str]]:
    """Every reading of a step, each variant from 2.50 down: the cheap trees first."""
    return [(name, target) for name in STEPS[step] for target in reversed(TARGETS)]


def target_path(name: str, target: str) -> Path:
    return RESULTS / name / f"target_{target}.json"


def diagnose(name: str, target: str, workers: int) -> None:
    """One reading, in this process, so that its peaks are its own (`ru_maxrss` is a lifetime
    peak, in bytes on macOS)."""
    record = prepared(name)
    if record is None:
        raise SystemExit(f"{name} is not prepared.")
    budget = record["budgets"][target]
    started = time.time()
    with clustering(record["clustered_tokens"].get(target, 0)) as tree:
        data = domains.diagnose_pool(
            pool_of(parse_variant(name)),
            merges_of(name),
            budget,
            target,
            RUNS / name / "corpora",
            target_path(name, target),
            workers,
        )
    data |= {
        "tree": tree if budget is not None else None,
        "workers": workers,
        "peak_parent_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_child_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "seconds": time.time() - started,
    }
    domains.write(target_path(name, target), data)


def _diagnose_alone(name: str, target: str, workers: int) -> dict[str, Any]:
    logs = RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, __file__, "diagnose", name, target, "--workers", str(workers)]
    with (logs / f"diagnose_{name}_{target}.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        print(f"FAILED {name} {target} exit {code}", flush=True)
        raise SystemExit(code)
    return cast(dict[str, Any], json.loads(target_path(name, target).read_text()))


def calibrate() -> None:
    name, target = "wikipedia@1#1", "1.50"
    path = target_path(name, target)
    data = json.loads(path.read_text()) if path.exists() else _diagnose_alone(name, target, 2)
    parent, child = data["peak_parent_bytes"], data["peak_child_bytes"]
    record = {
        "reading": f"{name} {target}",
        "seconds": data["seconds"],
        "peak_parent_bytes": parent,
        "peak_child_bytes": child,
        "workers": workers_for(parent, child),
    }
    domains.write(RESULTS / "calibration.json", record)
    print(f"CALIBRATED {json.dumps(record)}", flush=True)


def run(step: str) -> None:
    """A step's readings, one process each. Workers come from each reading's projected tree and the
    step's peaks so far; before its first reading, from the calibration, except at 4x, which starts
    from two."""
    calibration = RESULTS / "calibration.json"
    peaks: list[tuple[int, int, int]] = []
    for name, target in reading_order(step):
        path = target_path(name, target)
        record = prepared(name) or {}
        projected = int(record.get("projected_scipy_bytes", {}).get(target, 0))
        if path.exists():
            data = json.loads(path.read_text())
        else:
            if peaks:
                workers = workers_from(projected, peaks)
            elif step != "4x" and calibration.exists():
                workers = json.loads(calibration.read_text())["workers"]
            else:
                workers = 2
            data = _diagnose_alone(name, target, workers)
            print(
                f"READ {name} {target} {data['status']} {data['seconds']:.0f}s "
                f"workers {workers} tree {data['tree']}",
                flush=True,
            )
        if data["status"] == "read":
            peaks.append((projected, data["peak_parent_bytes"], data["peak_child_bytes"]))
    print(f"STEP DONE {step}", flush=True)


# The grid above 2.00, where the first failure is located; "none" is past its end.
EDGE_STEPS = ("2.00", "2.20", "2.50", "none")


def _failed_at(judged: dict[str, Any] | None) -> list[str]:
    """What a failing granularity missed: the classes not found, and `drops` for a dropped word."""
    if judged is None:
        return []
    missed = [c for c, found in judged["classes_found"].items() if not found]
    return [*missed, "drops"] if not all(judged["dropped_found"].values()) else missed


def _bases(resolvable_drops: bool = False) -> dict[str, dict[str, Any]]:
    """Pool #0 at 1x per corpus: the earlier experiment's summary for the three general corpora,
    this experiment's readings for the two knowledge domains (`summarize` fills those in)."""
    recorded = json.loads((domains.RESULTS / "summary.json").read_text())["domains"]
    bases = {}
    for corpus in ("tinystories", "fineweb", "wikipedia"):
        judged = recorded[corpus]["targets"]
        judged = resolvable(judged) if resolvable_drops else judged
        bases[corpus] = {"holds": {t: v["holds"] for t, v in judged.items()}, "judged": judged}
    return bases


def resolvable(judged: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """`judged` with each granularity's verdict read only on the dropped words some granularity
    of the same pool finds. A drop removes every sentence holding the word, so a word the tool
    misses is one whose total absence its draws cannot resolve; missed at every granularity, that
    is the pool's (at 1x: `oil`, 243 of its 304 uses in one article), not a granularity's."""
    found = {w for v in judged.values() for w, seen in v["dropped_found"].items() if seen}
    unresolved = sorted({w for v in judged.values() for w in v["dropped_found"]} - found)
    return {
        t: v
        | {
            "holds": all(v["classes_found"].values())
            and all(seen for w, seen in v["dropped_found"].items() if w in found),
            "unresolved_drops": unresolved,
        }
        for t, v in judged.items()
    }


def judge_variant(name: str, *, resolvable_drops: bool = False) -> dict[str, Any] | None:
    record = prepared(name)
    if record is None:
        return None
    variant = parse_variant(name)
    data = {t: json.loads(p.read_text()) for t in TARGETS if (p := target_path(name, t)).exists()}
    read = {t: d for t, d in data.items() if d["status"] == "read"}
    judged = domains.judge_all(read, record, merges_of(name)) if read else {}
    if resolvable_drops:
        judged = resolvable(judged)
    holds: dict[str, bool | None] = {t: judged[t]["holds"] for t in judged}
    holds |= {t: None for t, d in data.items() if d["status"] == "unreachable"}
    pooled = domains.pooled_false_flags(judged) if judged else {}
    best = domains.longest_run(judged) if judged else []
    first = first_failure_above(holds)
    return {
        "corpus": variant.corpus,
        "size": variant.size,
        "pool": variant.pool,
        "read": len(data),
        "holds": dict(sorted(holds.items())),
        "measured_range": [best[0], best[-1]] if best else None,
        "pooled_false_flags": pooled,
        "range_valid": bool(pooled) and all(f["passes"] for f in pooled.values()),
        "first_failure": first,
        "failed_at_first": _failed_at(judged.get(first)) if first else [],
        "trees": {t: d.get("tree") for t, d in data.items() if d["status"] == "read"},
        "merges_at_1.50": record["budgets"]["1.50"],
        "replaced": any(slot.get("replaced") for slot in record["slots"].values()),
        "dropped_not_tokens": {t: v["dropped_not_tokens"] for t, v in judged.items()},
        "unresolved_drops": next(iter(judged.values())).get("unresolved_drops", [])
        if judged
        else [],
        "judged": judged,
    }


def summarize(resolvable_drops: bool = False) -> None:
    """`results/summary.json` under the recorded rule; with `resolvable_drops`,
    `results/summary_resolvable.json`, where a dropped word no granularity finds is left out."""
    variants = {
        v.name: j
        for v in new_variants()
        if (j := judge_variant(v.name, resolvable_drops=resolvable_drops)) is not None
    }
    bases = _bases(resolvable_drops)
    for corpus in ("pubmed", "legal"):
        if (own := variants.get(f"{corpus}@1#0")) is not None:
            bases[corpus] = {"holds": own["holds"], "judged": own["judged"]}
    for values in variants.values():
        base = bases.get(values["corpus"])
        if base is None or (values["size"], values["pool"]) == (1.0, 0):
            continue
        holds = values["holds"]
        values["agrees"] = agrees(holds, base["holds"])
        values["agrees_to_2.00"] = agrees(
            {t: h for t, h in holds.items() if t <= "2.00"}, base["holds"]
        )
        values["base_first_failure"] = (bf := first_failure_above(base["holds"]))
        values["base_failed_at_first"] = _failed_at(base["judged"].get(bf)) if bf else []
    summary = {
        "variants": {n: {k: v for k, v in j.items() if k != "judged"} for n, j in variants.items()},
        "base": {c: b["holds"] for c, b in bases.items()},
    }
    summary["predictions"] = predictions(summary)
    domains.write(
        RESULTS / ("summary_resolvable.json" if resolvable_drops else "summary.json"), summary
    )
    for name, values in summary["variants"].items():
        print(
            f"{name:<22} read {values['read']:>2}  range {values['measured_range']}  "
            f"valid {values['range_valid']}  agrees {values.get('agrees')}"
        )
    print(json.dumps(summary["predictions"]))


def predictions(summary: dict[str, Any]) -> dict[str, str]:
    """The spec's twelve predictions, each on the variants read so far."""

    def status(values: list[bool]) -> str:
        return "unreadable" if not values else ("HELD" if all(values) else "FAILED")

    # A variant with no reading yet says nothing, and `agrees` over no reading is vacuously true.
    variants = [v for v in summary["variants"].values() if v.get("read", 1) > 0]
    sized = [v for v in variants if v["size"] != 1.0]
    drawn = [v for v in variants if v["size"] == 1.0 and v["pool"] != 0]
    small_or_drawn = [v for v in variants if v["size"] < 1.0 or v["pool"] != 0]
    # Predictions 8 and 9 read the preparation, which exists before any reading.
    domain = {
        v["corpus"]: v for v in summary["variants"].values() if (v["size"], v["pool"]) == (1.0, 0)
    }

    def step(edge: str | None) -> int:
        return EDGE_STEPS.index(edge or "none")

    def complete(v: dict[str, Any]) -> bool:
        return v.get("read") == len(TARGETS)

    return {
        "1": status([v["holds"]["1.50"] is True for v in sized if "1.50" in v["holds"]]),
        "2": status([v["agrees_to_2.00"] for v in sized if "agrees_to_2.00" in v]),
        "3": status(
            [
                abs(step(v["first_failure"]) - step(v["base_first_failure"])) <= 1
                for v in sized
                if complete(v) and "base_first_failure" in v
            ]
        ),
        "4": status([v["agrees"] for v in variants if v["size"] == 4.0 and "agrees" in v]),
        "5": status([v["range_valid"] for v in sized if complete(v)]),
        "6": status([v["holds"]["1.50"] is True for v in small_or_drawn if "1.50" in v["holds"]]),
        "7": status(
            [
                v["first_failure"] == v["base_first_failure"]
                and v["failed_at_first"] == v["base_failed_at_first"]
                for v in drawn
                if complete(v) and "base_first_failure" in v
            ]
        ),
        "8": status(
            [
                *([domain["pubmed"]["merges_at_1.50"] > 16_022] if "pubmed" in domain else []),
                *(
                    [1_223 < domain["legal"]["merges_at_1.50"] < 16_022]
                    if "legal" in domain
                    else []
                ),
            ]
        ),
        "9": status([not domain[c]["replaced"] for c in ("pubmed", "legal") if c in domain]),
        "10": status(
            [
                domain[c]["range_valid"]
                and domain[c]["measured_range"] is not None
                and domain[c]["measured_range"][0] <= "1.30"
                and domain[c]["measured_range"][1] >= "2.00"
                for c in ("pubmed", "legal")
                if c in domain and complete(domain[c])
            ]
        ),
        "11": status(
            [
                domain[c]["range_valid"]
                for c in ("pubmed", "legal")
                if c in domain and complete(domain[c])
            ]
        ),
        "12": status(
            [
                tree != "vector" or (v["size"] == 4.0 and t <= "1.50")
                for v in variants
                for t, tree in v.get("trees", {}).items()
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download").add_argument("corpus", choices=CORPORA)
    prep = commands.add_parser("prepare")
    prep.add_argument("names", nargs="+", help="variants or steps (small, pools, domains, 4x)")
    prep.add_argument("--jobs", type=int, default=1)
    commands.add_parser("costs")
    one = commands.add_parser("diagnose")
    one.add_argument("name")
    one.add_argument("target", choices=TARGETS)
    one.add_argument("--workers", type=int, default=2)
    commands.add_parser("calibrate")
    commands.add_parser("run").add_argument("step", choices=list(STEPS))
    commands.add_parser("summarize")
    commands.add_parser("verify").add_argument(
        "corpora", nargs="+", choices=("tinystories", "fineweb", "wikipedia")
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    match args.command:
        case "download":
            download(args.corpus)
        case "verify":
            verify(args.corpora)
        case "prepare" if len(args.names) == 1 and args.names[0] not in STEPS:
            prepare(args.names[0])
        case "prepare":
            prepare_all(args.names, args.jobs)
        case "costs":
            costs()
        case "diagnose":
            diagnose(args.name, args.target, args.workers)
        case "calibrate":
            calibrate()
        case "run":
            run(args.step)
        case "summarize":
            summarize()
            print("--- dropped words no granularity finds left out ---")
            summarize(resolvable_drops=True)


if __name__ == "__main__":
    main()
