"""The class tree without the n² table: does `fastcluster.linkage_vector` give SciPy's tree?

`docs/superpowers/specs/2026-10-27-memory-light-ward-design.md` records the design and predictions
before running. Corpus side only: no model is trained.

    ward_memory.py features <domain> <t>        the clustered tokens and their features
    ward_memory.py link <domain> <t> <method>   one linkage in this process: tree, time, peak RSS
    ward_memory.py compare <domain> <t>         levels 1 and 2: merges, heights, nodes, coarse cut
    ward_memory.py verdicts <domain>            level 3: the 22 cuts at 1.50 with linkage_vector
    ward_memory.py summarize                    the five predictions
    ward_memory.py sweep <domain> <t>           the 22 cuts at a granularity SciPy cannot read,
                                                on the tree `link ... vector` saved

`<method>` is `scipy` or `vector`. Each `link` runs alone so its peak memory is its own. Features
and trees go to `runs/ward_memory/` (gitignored), results to `results/`. No command runs on
import.
"""

import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "ward_memory"
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "granularity_domains")]

import granularity_domains as domains  # noqa: E402

from corpus_generalization.analysis import syntactic  # noqa: E402
from corpus_generalization.analysis.syntactic import ClassTree  # noqa: E402
from corpus_generalization.packing import framed_word_starts  # noqa: E402
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer  # noqa: E402

RELATIVE = 1e-9


def _vector_linkage(features: np.ndarray, method: str) -> np.ndarray:
    import fastcluster

    return np.asarray(fastcluster.linkage_vector(features, method=method))


def tokenizer_of(domain: str, target: str) -> ByteLevelBPETokenizer:
    budget = domains.budget_of(domain, target)
    if budget is None:
        raise SystemExit(f"{domain} does not reach {target} tokens per word.")
    return ByteLevelBPETokenizer.from_merges(domains.merges_of(domain)[:budget])


def features(domain: str, target: str) -> None:
    """Exactly what `class_tree` clusters for the domain's reading (`_read_domain`)."""
    pool = domains.pool_of(domain)
    framed, _ = framed_word_starts(pool, tokenizer_of(domain, target), True)
    sequences = tuple(framed)
    documents: Counter[int] = Counter()
    frequency: Counter[int] = Counter()
    for sequence in sequences:
        documents.update(set(sequence))
        frequency.update(sequence)
    tokens = sorted(token for token in frequency if documents[token] >= 2)
    context = [token for token, _ in frequency.most_common(syntactic.CONTEXT_TOKENS)]
    matrix = syntactic._neighbor_features(sequences, tokens, context)
    out = RUNS / f"{domain}_{target}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features.npy", matrix)
    np.save(out / "tokens.npy", np.asarray(tokens, dtype=np.int64))
    print(f"{domain} {target}: {len(tokens):,} tokens, features {matrix.shape}", flush=True)


def link(domain: str, target: str, method: str) -> None:
    out = RUNS / f"{domain}_{target}"
    matrix = np.load(out / "features.npy")
    # ru_maxrss is in bytes on macOS: the peak before and after the linkage, in this process.
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.time()
    if method == "scipy":
        merges = syntactic.linkage(matrix, method="ward")
    else:
        merges = _vector_linkage(matrix, "ward")
    seconds = time.time() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    np.save(out / f"merges_{method}.npy", merges)
    record = {
        "tokens": int(matrix.shape[0]),
        "features_bytes": int(matrix.nbytes),
        "seconds": seconds,
        "peak_bytes": int(peak),
        "peak_before_linkage_bytes": int(before),
    }
    domains.write(RESULTS / f"link_{domain}_{target}_{method}.json", record)
    print(f"{domain} {target} {method}: {seconds:.0f}s, peak {peak / 1e9:.2f} GB", flush=True)


def signatures(merges: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per merge, a label-free name of the cluster it forms: a sum of random 64-bit leaf keys and
    the cluster's size. Two trees form the same cluster when the names match; a collision is a
    2^-64 event."""
    keys = np.random.default_rng(0).integers(0, 2**63, size=size, dtype=np.uint64)
    names = np.zeros(2 * size - 1, dtype=np.uint64)
    names[:size] = keys
    counts = np.ones(2 * size - 1, dtype=np.int64)
    for index, (left, right, _, _) in enumerate(merges):
        node = size + index
        with np.errstate(over="ignore"):
            names[node] = names[int(left)] + names[int(right)]
        counts[node] = counts[int(left)] + counts[int(right)]
    return names[size:], counts[size:]


def compare(domain: str, target: str) -> None:
    out = RUNS / f"{domain}_{target}"
    tokens = tuple(int(t) for t in np.load(out / "tokens.npy"))
    size = len(tokens)
    trees = {m: np.load(out / f"merges_{m}.npy") for m in ("scipy", "vector")}
    names = {m: signatures(trees[m], size) for m in trees}
    same_step = [
        (int(a), int(ca)) == (int(b), int(cb))
        for a, ca, b, cb in zip(*names["scipy"], *names["vector"], strict=True)
    ]
    differing = [i for i, same in enumerate(same_step) if not same]
    heights = {m: trees[m][:, 2] for m in trees}
    scale = np.maximum(np.abs(heights["scipy"]), 1e-300)
    height_gap = np.abs(heights["scipy"] - heights["vector"]) / scale
    near_ties = []
    for i in differing[:50]:
        later = heights["scipy"][i + 1] if i + 1 < size - 1 else heights["scipy"][i]
        near_ties.append(
            {
                "step": i,
                "height": float(heights["scipy"][i]),
                "gap_to_next": float((later - heights["scipy"][i]) / scale[i]),
            }
        )
    node_sets = {m: set(zip(*(x.tolist() for x in names[m]), strict=True)) for m in names}
    cuts = {m: ClassTree(tokens, trees[m]).cut() for m in trees}
    partitions = {
        m: sorted(
            sorted(t for t, c in cuts[m].classes.items() if c == label)
            for label in set(cuts[m].classes.values())
        )
        for m in cuts
    }
    record: dict[str, Any] = {
        "tokens": size,
        "level1_steps_differing": len(differing),
        "level1_first_differing": near_ties,
        "level1_max_relative_height_gap": float(height_gap.max()),
        "level1_holds": not differing and float(height_gap.max()) <= RELATIVE,
        "level2_nodes_only_scipy": len(node_sets["scipy"] - node_sets["vector"]),
        "level2_nodes_only_vector": len(node_sets["vector"] - node_sets["scipy"]),
        "level2_coarse_classes": {m: cuts[m].n_classes for m in cuts},
        "level2_same_coarse_partition": partitions["scipy"] == partitions["vector"],
    }
    record["level2_holds"] = (
        record["level2_same_coarse_partition"]
        and record["level2_nodes_only_scipy"] == 0
        and record["level2_nodes_only_vector"] == 0
    )
    domains.write(RESULTS / f"compare_{domain}_{target}.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "level1_first_differing"}))


def verdicts(domain: str) -> None:
    """The 22 cuts at 1.50 against a reading whose tree is `linkage_vector`'s, beside the SciPy
    reading of `experiments/granularity_domains`."""
    syntactic.linkage = _vector_linkage  # type: ignore[assignment]
    domains.diagnose_target(domain, "1.50", RESULTS / "vector")
    scipy_path = (
        domains.step1.RESULTS / "target_1.50.json"
        if domain == "fineweb"
        else OUT.parent / "granularity_domains" / "results" / domain / "target_1.50.json"
    )
    left = json.loads(scipy_path.read_text())
    right = json.loads((RESULTS / "vector" / domain / "target_1.50.json").read_text())
    differences: dict[str, Any] = {}
    for name in domains.cut_names():
        a, b = left["cuts"][name], right["cuts"][name]
        rows_a = {(h, c): v for h, c, v in a["rows"]}
        rows_b = {(h, c): v for h, c, v in b["rows"]}
        changed = [
            [h, c, rows_a.get((h, c)), rows_b.get((h, c))]
            for h, c in rows_a.keys() | rows_b.keys()
            if rows_a.get((h, c)) != rows_b.get((h, c))
        ]

        def class_view(cut: dict[str, Any]) -> list[tuple[tuple[str, ...], int]]:
            return sorted((tuple(r["heads"]), r["size"]) for r in cut["class_rows"])

        def absent_view(cut: dict[str, Any]) -> list[tuple[str, int]]:
            return sorted((r["head"], r["filler_class"]) for r in cut["absent"])

        differences[name] = {
            "rows_changed": len(changed),
            "examples": changed[:10],
            "same_class_rows": class_view(a) == class_view(b),
            "same_absent": absent_view(a) == absent_view(b),
            "same_counts": a["counts"] == b["counts"],
        }
    holds = all(
        d["rows_changed"] == 0 and d["same_class_rows"] and d["same_absent"]
        for d in differences.values()
    )
    domains.write(RESULTS / f"verdicts_{domain}.json", {"holds": holds, "cuts": differences})
    print(f"{domain}: level 3 {'holds' if holds else 'FAILS'}")
    for name, d in differences.items():
        if d["rows_changed"] or not d["same_class_rows"] or not d["same_absent"]:
            print(f"  {name}: {json.dumps(d)}")


def sweep(domain: str, target: str) -> None:
    """Complete `experiments/granularity_domains` where the SciPy table did not fit: the reading
    clusters with the `linkage_vector` tree already saved for these exact features (hours to
    rebuild), and its result goes where the sweep's other readings are."""
    saved = np.load(RUNS / f"{domain}_{target}" / "merges_vector.npy")

    def saved_tree(features: np.ndarray, method: str) -> np.ndarray:
        if method != "ward" or features.shape[0] != len(saved) + 1:
            raise ValueError("The saved tree is not for these features.")
        return saved

    syntactic.linkage = saved_tree  # type: ignore[assignment]
    domains.diagnose_target(domain, target, memory_rule=False)


def summarize() -> None:
    def load(path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text()) if path.exists() else None

    pools = [(d, "1.50") for d in ("tinystories", "fineweb", "wikipedia")]
    large = [("fineweb", "1.30"), ("wikipedia", "1.35"), ("wikipedia", "1.30")]
    compares = {f"{d} {t}": load(RESULTS / f"compare_{d}_{t}.json") for d, t in pools}
    links = {
        f"{d} {t} {m}": load(RESULTS / f"link_{d}_{t}_{m}.json")
        for d, t in pools + large
        for m in ("scipy", "vector")
    }
    level3 = {d: load(RESULTS / f"verdicts_{d}.json") for d in ("tinystories", "fineweb")}

    def status(value: bool | None) -> str:
        return "unreadable" if value is None else ("HELD" if value else "FAILED")

    read = [c for c in compares.values() if c]
    ratios = [
        links[f"{d} {t} vector"]["seconds"] / links[f"{d} {t} scipy"]["seconds"]
        for d, t in pools
        if links.get(f"{d} {t} vector") and links.get(f"{d} {t} scipy")
    ]
    big = [links.get(f"{d} {t} vector") for d, t in (("fineweb", "1.30"), ("wikipedia", "1.30"))]
    predictions = {
        "1": status(all(c["level1_holds"] for c in read) if len(read) == len(pools) else None),
        "2": status(all(c["level2_holds"] for c in read) if len(read) == len(pools) else None),
        "3": status(
            all(v["holds"] for v in level3.values() if v) if all(level3.values()) else None
        ),
        "4": status(all(b["peak_bytes"] < 4e9 for b in big if b) if all(big) else None),
        "5": status(max(ratios) <= 5 if len(ratios) == len(pools) else None),
    }
    domains.write(
        RESULTS / "summary.json",
        {
            "compare": compares,
            "links": links,
            "level3": {d: (v["holds"] if v else None) for d, v in level3.items()},
            "time_ratios": ratios,
            "predictions": predictions,
        },
    )
    print(json.dumps(predictions))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("features", "compare"):
        command = commands.add_parser(name)
        command.add_argument("domain", choices=domains.DOMAINS)
        command.add_argument("target")
    linking = commands.add_parser("link")
    linking.add_argument("domain", choices=domains.DOMAINS)
    linking.add_argument("target")
    linking.add_argument("method", choices=("scipy", "vector"))
    verdict = commands.add_parser("verdicts")
    verdict.add_argument("domain", choices=domains.DOMAINS)
    commands.add_parser("summarize")
    completing = commands.add_parser("sweep")
    completing.add_argument("domain", choices=domains.DOMAINS)
    completing.add_argument("target")
    args = parser.parse_args()
    match args.command:
        case "features":
            features(args.domain, args.target)
        case "link":
            link(args.domain, args.target, args.method)
        case "compare":
            compare(args.domain, args.target)
        case "verdicts":
            verdicts(args.domain)
        case "summarize":
            summarize()
        case "sweep":
            sweep(args.domain, args.target)


if __name__ == "__main__":
    main()
