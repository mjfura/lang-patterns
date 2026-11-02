"""The tokens-per-word range across domains, and its upper edge.

`docs/superpowers/specs/2026-10-25-granularity-across-domains-design.md` records the design and
predictions before running. Corpus side only: no model is trained.

    granularity_domains.py download                 cache TinyStories next to this script
    granularity_domains.py prepare <domain>         pool, merges, the 22 labeled cuts, the gate
    granularity_domains.py diagnose <domain> <t>    read the pool at t tokens/word, diagnose cuts
    granularity_domains.py summarize                the pass rule and the six predictions

Domains are `tinystories`, `wikipedia` and `fineweb`. FineWeb keeps step 1's pool, merges, cuts
and readings at 1.35-1.80 (`experiments/granularity_range`); only the targets above 1.80 are read
here. Pools, merges and cuts go to `runs/granularity_domains/<domain>/` (gitignored), results to
`results/<domain>/`. No command runs on import.
"""

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

OUT = Path(__file__).parent
RESULTS = OUT / "results"
RUNS = Path("runs") / "granularity_domains"
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "granularity_range"), str(EXPERIMENTS / "class_patterns")]

import class_patterns as base  # noqa: E402
import granularity_range as step1  # noqa: E402
from class_patterns import divergence, natural_gap, subset_selection  # noqa: E402

from corpus_generalization.analysis.syntactic import induce_classes  # noqa: E402
from corpus_generalization.diagnose import DomainReading, diagnose, prepare_reference  # noqa: E402
from corpus_generalization.sources import huggingface_documents  # noqa: E402
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer, Merge  # noqa: E402
from corpus_generalization.tokenizer.granularity import (  # noqa: E402
    learn_to_granularity,
    tokens_per_word_by_budget,
)

DOMAINS = ("tinystories", "wikipedia", "fineweb")
TARGETS = (1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.20, 2.50)
FINEWEB_TARGETS = (1.90, 2.00, 2.20, 2.50)
STEP1_TARGETS = (1.35, 1.40, 1.50, 1.60, 1.70, 1.80)
TINYSTORIES = OUT / "tinystories_documents.json"
TINYSTORIES_ROWS = 25_000
WIKIPEDIA = EXPERIMENTS / "clean_domains" / "wikipedia_documents.json"
WORKERS = int(os.environ.get("WORKERS", "2"))
ALPHA = 0.06

PRONOUNS = ("I", "you", "he", "she", "we", "they")
MODALS = ("can", "could", "will", "would", "should", "may", "might", "must")
# The four slots of step 1, in its names, and the two fixed replacements in the declared order.
# `heads` None: the members of the anchor's class in the domain's 64-class cut.
SLOTS: dict[str, dict[str, Any]] = {
    "": {
        "heads": PRONOUNS + MODALS,
        "follower": "VERB",
        "removed": ("she", "they", "can", "will"),
        "anchors": ("they", "can"),
    },
    "poss_": {
        "heads": ("my", "your", "his", "our", "their", "its"),
        "follower": "NOUN",
        "removed": ("his", "their"),
        "anchors": ("his",),
    },
    "num_": {
        "heads": None,
        "anchor": "two",
        "follower": "NOUN",
        "removed": ("five", "six"),
        "anchors": ("two",),
    },
    "verb_": {
        "heads": None,
        "anchor": "make",
        "follower": "DET",
        "removed": ("make", "get"),
        "anchors": ("make",),
    },
}
REPLACEMENTS: tuple[dict[str, Any], ...] = (
    {
        "heads": ("in", "on", "at", "with", "from", "for"),
        "follower": "DET",
        "removed": ("on", "with"),
        "anchors": ("in",),
    },
    {
        "heads": ("the", "a", "an", "this", "that"),
        "follower": "NOUN",
        "removed": ("this", "that"),
        "anchors": ("the",),
    },
)
SENTENCE_CONTROLS = ("ctrl_a", "ctrl_b", "ctrl_c", "ctrl_d")
DOCUMENT_CONTROLS = ("r25b", "r25c", "r25d", "r25e")
FRESH_CONTROLS = (*SENTENCE_CONTROLS, *DOCUMENT_CONTROLS)
FINEWEB_DROPS = (
    ("using", "making", "however", "several"),
    ("although", "within", "various", "whose"),
)
WORD = re.compile(r"[a-z]+")

_reading: DomainReading | None = None
_corpora: Path | None = None


def write(path: Path, data: object) -> None:
    divergence.write_json(path, data)


def pool_of(domain: str) -> list[str]:
    """The domain's documents in cache order until they match F's bytes; FineWeb's is F."""
    _, f = natural_gap.split()
    if domain == "fineweb":
        return f
    source = TINYSTORIES if domain == "tinystories" else WIKIPEDIA
    documents = cast(list[str], json.loads(source.read_text()))
    sizes = [len(d.encode("utf-8")) for d in documents]
    chosen = subset_selection.fill(sizes, list(range(len(documents))), base.size(f))
    return [documents[i] for i in chosen]


def merges_path(domain: str) -> Path:
    return step1.MERGES if domain == "fineweb" else RUNS / domain / "merges.json"


def merges_of(domain: str) -> tuple[Merge, ...]:
    raw = json.loads(merges_path(domain).read_text())
    return tuple(((left, right), new) for (left, right), new in raw)


def corpora_dir(domain: str) -> Path:
    return base.CORPORA if domain == "fineweb" else RUNS / domain / "corpora"


def cut_names(drops: Sequence[str] = ("drop4", "drop4b")) -> tuple[str, ...]:
    slots = tuple(f"{p}class_cut" for p in SLOTS)
    heads = tuple(f"{p}head_cut" for p in SLOTS)
    controls = tuple(f"{p}control" for p in SLOTS)
    return (*slots, *heads, *controls, *FRESH_CONTROLS, *drops)


def download() -> None:
    documents = huggingface_documents("roneneldan/TinyStories", "train", "text", TINYSTORIES_ROWS)
    TINYSTORIES.write_text(json.dumps(documents))
    print(f"tinystories: {len(documents)} documents, {base.size(documents):,} bytes", flush=True)
    # A streamed dataset can leave threads that keep the interpreter from exiting.
    os._exit(0)


def anchor_class(reading: DomainReading, anchor: str) -> tuple[str, ...]:
    """The lowercased alphabetic members of `anchor`'s class in the domain's 64-class cut, as
    `tool_check.class_members` reads them on F."""
    words = frozenset(reading.classes.classes)
    tokens = {
        text[1:]: token
        for token in words
        if (text := reading.spell(token)).startswith(" ")
        and text[1:].isascii()
        and text[1:].isalpha()
    }
    if anchor not in tokens:
        return ()
    tree = induce_classes(reading.documents, n_classes=64, candidates=words)
    label = tree.classes[tokens[anchor]]
    members = (
        text
        for token, cls in tree.classes.items()
        if cls == label and (text := reading.spell(token).strip()).isalpha()
    )
    return tuple(dict.fromkeys(sorted(m.lower() for m in members)))


def segment_all(
    documents: Sequence[str], specs: dict[str, dict[str, Any]]
) -> list[list[tuple[str, dict[str, frozenset[str]]]]]:
    """Per document, its sentences with, for every spec, the listed heads its follower follows:
    one spaCy pass for every slot, where step 1 ran one per slot."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.enable_pipe("senter")
    result = []
    for index, doc in enumerate(nlp.pipe(documents, batch_size=64)):
        if index % 500 == 0:
            divergence.progress(f"Segmented {index} of {len(documents)} documents")
        sentences = []
        for sentence in doc.sents:
            targets: dict[str, set[str]] = {key: set() for key in specs}
            for token in sentence[:-1]:
                follower = token.nbor().pos_
                for key, spec in specs.items():
                    if follower == spec["follower"]:
                        head = base.head_of(token.text, spec["heads"])
                        if head is not None:
                            targets[key].add(head)
            sentences.append(
                (sentence.text_with_ws, {k: frozenset(v) for k, v in targets.items() if v})
            )
        result.append(sentences)
    return result


def view(
    segmented: Sequence[Sequence[tuple[str, dict[str, frozenset[str]]]]], key: str
) -> list[list[tuple[str, frozenset[str]]]]:
    """One spec's targets in the shape `class_patterns` builds cuts from."""
    return [[(text, t.get(key, frozenset())) for text, t in sentences] for sentences in segmented]


def fineweb_gate_share() -> float:
    """The smallest share of r50's sentences a FineWeb class cut removed (numbers: 1,972 of
    62,025), and was found at every granularity: the least evidence a cut is known to test."""
    results = base.RESULTS
    sentences = json.loads((results / "corpora.json").read_text())["sentences"]
    counts = [
        json.loads((results / "corpora.json").read_text())["target_sentences"],
        json.loads((results / "possessive_corpora.json").read_text())["target_sentences"],
        *(
            v["target_sentences"]
            for v in json.loads((results / "tool_check_corpora.json").read_text()).values()
        ),
    ]
    return float(min(counts) / sentences)


def word_ranks(documents: Sequence[str]) -> dict[str, int]:
    counts = Counter(w for d in documents for w in WORD.findall(d.lower()))
    return {w: rank for rank, (w, _) in enumerate(counts.most_common(), start=1)}


def drop_words(pool: Sequence[str], heads: set[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Eight words in the frequency-rank band FineWeb's eight hold in F, not a head of any cut,
    sampled with seed 0."""
    _, f = natural_gap.split()
    f_ranks = word_ranks(f)
    band = [f_ranks[w] for words in FINEWEB_DROPS for w in words]
    low, high = min(band), max(band)
    candidates = sorted(
        w for w, rank in word_ranks(pool).items() if low <= rank <= high and w not in heads
    )
    chosen = random.Random(0).sample(candidates, 8)
    return tuple(chosen[:4]), tuple(chosen[4:])


def prepare(domain: str) -> None:
    if domain == "fineweb":
        raise SystemExit("FineWeb keeps step 1's pool, merges and cuts.")
    prepare_pool(pool_of(domain), RUNS / domain, RESULTS / domain)


def prepare_pool(
    pool: list[str],
    out: Path,
    results: Path,
    *,
    before_reading: Callable[[ByteLevelBPETokenizer], AbstractContextManager[object]] | None = None,
) -> None:
    """The pool's merges and 22 labeled cuts under `out`, and its record in `results`.
    `before_reading`, given the tokenizer at 1.50, wraps the class trees built there."""
    (out / "corpora").mkdir(parents=True, exist_ok=True)
    words = sum(len(d.split()) for d in pool)
    divergence.progress(
        f"{out.name}: {len(pool)} documents, {base.size(pool):,} bytes, {words:,} words"
    )
    started = time.time()
    tokenizer, reached = learn_to_granularity(pool, min(TARGETS))
    learned = tokenizer.merges
    (out / "merges.json").write_text(json.dumps(learned))
    curve = tokens_per_word_by_budget(pool, learned)
    budgets = {
        f"{t:.2f}": next((k for k, value in enumerate(curve) if value <= t), None) for t in TARGETS
    }
    divergence.progress(
        f"{len(learned):,} merges to {reached:.3f} tokens/word in "
        f"{time.time() - started:.0f}s; budgets {budgets}"
    )
    at_150 = budgets["1.50"]
    if at_150 is None:
        raise SystemExit(f"{out.name} does not reach 1.50 tokens per word.")
    at_tokenizer = ByteLevelBPETokenizer.from_merges(learned[:at_150])
    with before_reading(at_tokenizer) if before_reading else nullcontext():
        reading = prepare_reference(
            pool, at_150, tokenizer=at_tokenizer, head_lengths=(1,), progress=divergence.progress
        )
        specs: dict[str, dict[str, Any]] = {}
        for prefix, slot in SLOTS.items():
            heads = slot["heads"] or anchor_class(reading, slot["anchor"])
            specs[prefix] = {**slot, "heads": tuple(heads)}
    for index, replacement in enumerate(REPLACEMENTS):
        specs[f"rep{index}"] = replacement
    half_order = list(range(len(pool)))
    random.Random(0).shuffle(half_order)
    half = [pool[i] for i in sorted(half_order[: len(pool) // 2])]
    segmented = segment_all(half, specs)
    sentences = sum(len(s) for s in segmented)
    share = fineweb_gate_share()
    record: dict[str, Any] = {
        "documents": len(pool),
        "bytes": base.size(pool),
        "words": words,
        "learned_merges": len(learned),
        "lowest_reached": reached,
        "budgets": budgets,
        "curve": {str(k): round(curve[k], 4) for k in range(0, len(curve), 250)},
        "half_documents": len(half),
        "half_bytes": base.size(half),
        "half_sentences": sentences,
        "gate_share": share,
        "slots": {},
    }
    spare = [f"rep{i}" for i in range(len(REPLACEMENTS))]
    corpora: dict[str, list[str]] = {}
    for prefix in SLOTS:
        key = prefix
        while True:
            targets = view(segmented, key)
            every = {(d, i) for d, ss in enumerate(targets) for i, (_, ts) in enumerate(ss) if ts}
            if len(every) >= share * sentences or not spare:
                break
            divergence.progress(
                f"slot {prefix or 'pron_modal'}: {len(every)} target sentences "
                f"under the gate ({share * sentences:.0f}); replaced"
            )
            record["slots"].setdefault(prefix or "pron_modal", {}).setdefault("failed", []).append(
                {"heads": list(specs[key]["heads"]), "target_sentences": len(every)}
            )
            key = spare.pop(0)
        spec = specs[key]
        per_head = Counter(h for ss in targets for _, ts in ss for h in ts)
        removed = [h for h in spec["removed"] if h in spec["heads"]]
        for head, _ in per_head.most_common():
            if len(removed) >= len(spec["removed"]):
                break
            if head not in removed and head not in spec["anchors"]:
                removed.append(head)
        partial = {
            (d, i)
            for d, ss in enumerate(targets)
            for i, (_, ts) in enumerate(ss)
            if ts & set(removed)
        }
        class_cut = base.assemble(targets, every)
        target = base.size(class_cut)
        corpora[f"{prefix}class_cut"] = class_cut
        corpora[f"{prefix}head_cut"] = base.assemble(
            targets, base.match_bytes(targets, partial, target, 0)
        )
        corpora[f"{prefix}control"] = base.assemble(
            targets, base.match_bytes(targets, set(), target, 0)
        )
        record["slots"].setdefault(prefix or "pron_modal", {}).update(
            {
                "heads": list(spec["heads"]),
                "follower": spec["follower"],
                "anchors": list(spec["anchors"]),
                "removed": removed,
                "target_sentences": len(every),
                "removed_head_sentences": len(partial),
                "per_head": dict(per_head.most_common()),
                "replaced": key != prefix,
            }
        )
    plain = view(segmented, "no targets")
    half_bytes = base.size(half)
    r50_bytes = json.loads((base.RESULTS / "corpora.json").read_text())["r50_bytes"]
    # Step 1's sizes: ctrl_a and ctrl_b at two class cuts' bytes, ctrl_c and ctrl_d at 4.0 and
    # 6.0 MB of a 7.19 MB r50, kept here as shares of the half pool.
    sizes = {
        "ctrl_a": (base.size(corpora["class_cut"]), 1),
        "ctrl_b": (base.size(corpora["poss_class_cut"]), 2),
        "ctrl_c": (round(half_bytes * 4_000_000 / r50_bytes), 3),
        "ctrl_d": (round(half_bytes * 6_000_000 / r50_bytes), 4),
    }
    for name, (target, seed) in sizes.items():
        corpora[name] = base.assemble(plain, base.match_bytes(plain, set(), target, seed))
    pool_sizes = [len(d.encode("utf-8")) for d in pool]
    for name, seed in zip(("r25b", "r25c", "r25d", "r25e"), range(204, 208), strict=True):
        order = list(range(len(pool)))
        random.Random(seed).shuffle(order)
        corpora[name] = [
            pool[i] for i in subset_selection.fill(pool_sizes, order, 0.25 * sum(pool_sizes))
        ]
    heads = {h.lower() for spec in specs.values() for h in spec["heads"]}
    drops = drop_words(pool, heads)
    record["drops"] = {"drop4": list(drops[0]), "drop4b": list(drops[1])}
    for name, words_ in zip(("drop4", "drop4b"), drops, strict=True):
        pattern = re.compile(r"\b(" + "|".join(words_) + r")\b", re.IGNORECASE)
        dropped = {
            (d, i)
            for d, ss in enumerate(plain)
            for i, (text, _) in enumerate(ss)
            if pattern.search(text)
        }
        corpora[name] = base.assemble(plain, dropped)
        record["drops"][f"{name}_sentences"] = len(dropped)
    record["corpora"] = {}
    for name, corpus in corpora.items():
        (out / "corpora" / f"{name}.json").write_text(json.dumps(corpus))
        record["corpora"][name] = {"documents": len(corpus), "bytes": base.size(corpus)}
        print(f"{name}: {len(corpus)} documents, {base.size(corpus):,} bytes", flush=True)
    write(results / "prepared.json", record)


def fineweb_record() -> dict[str, Any]:
    """Step 1's cuts in this experiment's terms."""
    tool = json.loads((base.RESULTS / "tool_check_corpora.json").read_text())
    heads = {"num_": tool["num"]["heads"], "verb_": tool["verb"]["heads"]}
    return {
        "slots": {
            (prefix or "pron_modal"): {
                "anchors": list(slot["anchors"]),
                "heads": list(slot["heads"] or heads[prefix]),
            }
            for prefix, slot in SLOTS.items()
        },
        "drops": {"drop4": list(FINEWEB_DROPS[0]), "drop4b": list(FINEWEB_DROPS[1])},
    }


def record_of(domain: str) -> dict[str, Any]:
    if domain == "fineweb":
        return fineweb_record()
    return cast(dict[str, Any], json.loads((RESULTS / domain / "prepared.json").read_text()))


def budget_of(domain: str, target: str) -> int | None:
    if domain == "fineweb":
        curve = tokens_per_word_by_budget(pool_of("fineweb"), merges_of("fineweb"))
        return next((k for k, value in enumerate(curve) if value <= float(target)), None)
    return cast(int | None, record_of(domain)["budgets"][target])


def clustered_tokens(pool: Sequence[str], tokenizer: ByteLevelBPETokenizer) -> int:
    """Tokens in at least two documents: what the class tree clusters, the reading's memory."""
    seen: Counter[int] = Counter()
    for document in pool:
        seen.update(set(tokenizer.encode(document)))
    return sum(count >= 2 for count in seen.values())


def memory_limit() -> int:
    """The most clustered tokens a reading has fit in 16 GB: F at 1.35 tokens per word. F at
    1.30 (45,748 tokens) did not fit; a reading above the limit is recorded, not run, since a
    memory-pressure kill takes the machine down with it."""
    path = RESULTS / "fineweb_limit.json"
    if path.exists():
        return cast(int, json.loads(path.read_text())["tokens"])
    budget = budget_of("fineweb", "1.35")
    assert budget is not None
    tokens = clustered_tokens(
        pool_of("fineweb"), ByteLevelBPETokenizer.from_merges(merges_of("fineweb")[:budget])
    )
    write(path, {"tokens": tokens, "budget": budget})
    return tokens


def _start(reading: DomainReading, corpora: Path) -> None:
    global _reading, _corpora
    _reading, _corpora = reading, corpora


def _diagnose_cut(name: str) -> tuple[str, dict[str, Any], float]:
    assert _reading is not None and _corpora is not None
    started = time.time()
    corpus = json.loads((_corpora / f"{name}.json").read_text())
    report = diagnose(corpus, _reading)
    return name, step1.reading_of(report), time.time() - started


def diagnose_target(
    domain: str, target: str, results: Path | None = None, *, memory_rule: bool = True
) -> None:
    """`memory_rule` False reads past the SciPy table's limit, for a tree that does not need
    the table (`experiments/ward_memory`)."""
    out = (results or RESULTS) / domain / f"target_{target}.json"
    budget = budget_of(domain, target)
    pool = pool_of(domain)
    if budget is not None and memory_rule:
        tokenizer = ByteLevelBPETokenizer.from_merges(merges_of(domain)[:budget])
        tokens, limit = clustered_tokens(pool, tokenizer), memory_limit()
        if tokens > limit:
            write(
                out,
                {
                    "target": target,
                    "status": "too large",
                    "merge_budget": budget,
                    "clustered_tokens": tokens,
                    "limit": limit,
                },
            )
            print(f"{domain} {target}: {tokens:,} tokens to cluster, above {limit:,}; not run")
            return
    diagnose_pool(pool, merges_of(domain), budget, target, corpora_dir(domain), out)


def diagnose_pool(
    pool: list[str],
    merges: Sequence[Merge],
    budget: int | None,
    target: str,
    corpora: Path,
    out: Path,
    workers: int = WORKERS,
) -> dict[str, Any]:
    """The 22 cuts in `corpora` read against `pool` at `budget` merges, written to `out`."""
    if budget is None:
        record: dict[str, Any] = {"target": target, "status": "unreachable"}
        write(out, record)
        print(f"{out.parent.name} {target}: unreachable, its merges run out first")
        return record
    tokenizer = ByteLevelBPETokenizer.from_merges(merges[:budget])
    tokens = clustered_tokens(pool, tokenizer)
    started = time.time()
    reading = prepare_reference(
        pool, budget, tokenizer=tokenizer, head_lengths=(1,), progress=divergence.progress
    )
    reference_seconds = time.time() - started
    transitions = sum(len(starts) - 1 for starts in reading.word_starts)
    lexical = sum(not flag for starts in reading.word_starts for flag in starts)
    words = sum(len(document.split()) for document in pool)
    pool_tokens = sum(len(sequence) - 2 for sequence in reading.documents)
    cuts: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    executor = ProcessPoolExecutor(workers, initializer=_start, initargs=(reading, corpora))
    with executor as running:
        for name, result, elapsed in running.map(_diagnose_cut, cut_names()):
            divergence.progress(
                f"{out.parent.name} {target} {name}: {result['counts']} in {elapsed:.0f}s"
            )
            cuts[name], seconds[name] = result, elapsed
    record = {
        "target": target,
        "status": "read",
        "merge_budget": budget,
        "clustered_tokens": tokens,
        "pool_tokens_per_word": pool_tokens / words,
        "lexical_share": lexical / transitions,
        "classes": {str(g.label): list(g.top_tokens) for g in reading.groups},
        "class_members": {str(g.label): list(g.tokens) for g in reading.groups},
        "reference_seconds": reference_seconds,
        "cut_seconds": seconds,
        "cuts": cuts,
    }
    write(out, record)
    return record


def readings(domain: str) -> dict[str, dict[str, Any]]:
    paths = {f"{t:.2f}": RESULTS / domain / f"target_{t:.2f}.json" for t in TARGETS}
    if domain == "fineweb":
        paths |= {f"{t:.2f}": step1.RESULTS / f"target_{t:.2f}.json" for t in STEP1_TARGETS}
    return {t: json.loads(p.read_text()) for t, p in sorted(paths.items()) if p.exists()}


def _syntactic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not r["lexical"]]


def groups(slot: str, values: dict[str, Any]) -> list[list[str]]:
    """The removed classes of a slot, each led by its anchor: pronouns and modals are two, as in
    step 1's five classes."""
    if slot == "pron_modal":
        return [["they", *(h.lower() for h in PRONOUNS)], ["can", *MODALS]]
    heads = [h.lower() for h in values["heads"]]
    return [[values["anchors"][0], *heads]]


def judge(data: dict[str, Any], record: dict[str, Any], single: set[str]) -> dict[str, Any]:
    """Step 1's readings of one granularity, and the per-granularity part of the pass rule.
    `single`: the dropped words that are one token at this granularity."""
    cuts = data["cuts"]
    found = {}
    for slot, values in record["slots"].items():
        prefix = "" if slot == "pron_modal" else slot
        rows = _syntactic(cuts[f"{prefix}class_cut"]["class_rows"])
        flagged = [{h.strip().lower() for h in r["heads"]} for r in rows]
        # The removed class is found when a flagged class row holds one of its heads. Step 1 asked
        # for the anchor itself, but TinyStories' tree puts `two` in a large adjective class, where
        # it need not be among a row's 12 listed heads (amended before any diagnosis was read).
        for group in groups(slot, values):
            found[group[0]] = any(set(group) & heads for heads in flagged)
    dropped, unseen = {}, []
    for name in ("drop4", "drop4b"):
        absent = {r["head"].strip().lower() for r in _syntactic(cuts[name]["absent"])}
        for word in record["drops"][name]:
            # A word the tokenizer splits is not a unit the tool can report missing: it is left out
            # of the rule and counted (the user's choice, after TinyStories at 1.50).
            if word in single:
                dropped[word] = word in absent
            else:
                unseen.append(word)
    # Matched controls remove only sentences without the pattern, so where the pattern is common
    # (46% of TinyStories' sentences) they concentrate it; false flags are read on the eight random
    # controls only (the user's choice, after TinyStories at 1.50).
    controls = list(FRESH_CONTROLS)
    return {
        "merge_budget": data["merge_budget"],
        "pool_tokens_per_word": round(data["pool_tokens_per_word"], 3),
        "lexical_share": round(data["lexical_share"], 3),
        "classes_found": found,
        "dropped_found": dropped,
        "dropped_not_tokens": unseen,
        "holds": all(found.values()) and all(dropped.values()),
        # Sentence cuts orphan closing quotes in dialogue (TinyStories), a real change the class
        # rows flag (`."`, `!"`); class false flags are read on whole-document cuts only (the
        # user's choice, after TinyStories' sweep). The sentence controls' flags are kept.
        "control_class_flags": [c for c in DOCUMENT_CONTROLS if _syntactic(cuts[c]["class_rows"])],
        "sentence_control_class_flags": [
            c for c in SENTENCE_CONTROLS if _syntactic(cuts[c]["class_rows"])
        ],
        "matched_control_class_flags": [
            f"{p}control" for p in SLOTS if _syntactic(cuts[f"{p}control"]["class_rows"])
        ],
        "control_head_flags": [
            c for c in controls if any(v == "under-shown" for _, _, v in cuts[c]["rows"])
        ],
        "fresh_absent_flags": [c for c in FRESH_CONTROLS if _syntactic(cuts[c]["absent"])],
        "head_flags_class_cut_head_cut": [
            sum(v == "under-shown" for _, _, v in cuts[f"{p}{kind}"]["rows"])
            for kind in ("class_cut", "head_cut")
            for p in SLOTS
        ],
        "mean_cut_seconds": round(statistics.mean(data["cut_seconds"].values())),
    }


def one_token(merges: Sequence[Merge], words: Sequence[str]) -> set[str]:
    """The words that are one token after a space, lowercase or capitalized, at these merges."""
    tokenizer = ByteLevelBPETokenizer.from_merges(merges)
    return {
        w
        for w in words
        if any(len(tokenizer.encode(f" {form}")) == 1 for form in (w, w.capitalize()))
    }


def swaps(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, int]:
    """Patterns read (under-shown or within noise) at both granularities, and how many change
    between the two: step 1's swap count, over every cut."""
    mapping = step1._class_map(left["class_members"], right["class_members"])
    read = {"under-shown", "within-noise"}
    swapped = total = 0
    for name in cut_names():
        verdicts = {(h, c): v for h, c, v in right["cuts"][name]["rows"]}
        for head, label, verdict in left["cuts"][name]["rows"]:
            other = verdicts.get((head, mapping.get(label, -1)))
            if verdict in read and other in read:
                total += 1
                swapped += verdict != other
    return swapped, total


def judge_all(
    read: dict[str, dict[str, Any]], record: dict[str, Any], merges: Sequence[Merge]
) -> dict[str, dict[str, Any]]:
    """`judge` at every granularity read, each with the dropped words that are one token there."""
    words = [w for name in ("drop4", "drop4b") for w in record["drops"][name]]
    return {
        t: judge(d, record, one_token(merges[: d["merge_budget"]], words)) for t, d in read.items()
    }


def pooled_false_flags(per_target: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per family, the distinct controls flagged over the granularities read, against 2 x alpha."""
    families = {
        "class rows": "control_class_flags",
        "head rows": "control_head_flags",
        "absent": "fresh_absent_flags",
    }
    readings_per = {"class rows": 4, "head rows": 8, "absent": 8}
    # The same controls are read at every granularity, so a flag that recurs on one control
    # is that corpus's, not the granularity's: it counts once (amended before any new domain
    # was diagnosed: FineWeb's r25c and r25e flag the same word at several granularities).
    return {
        family: {
            "flagged": (flagged := sorted({c for v in per_target.values() for c in v[key]})),
            "flags_by_target": {t: len(v[key]) for t, v in per_target.items()},
            "readings": (n := readings_per[family] * len(per_target)),
            "rate": len(flagged) / n if n else None,
            "passes": n > 0 and len(flagged) / n <= 2 * ALPHA,
        }
        for family, key in families.items()
    }


def longest_run(per_target: dict[str, dict[str, Any]]) -> list[str]:
    """The longest run of consecutive granularities that hold."""
    best: list[str] = []
    run: list[str] = []
    for target in sorted(per_target):
        run = [*run, target] if per_target[target]["holds"] else []
        if len(run) > len(best):
            best = run
    return best


def summarize() -> None:
    summary: dict[str, Any] = {}
    for domain in DOMAINS:
        record = (
            record_of(domain)
            if (domain == "fineweb" or (RESULTS / domain / "prepared.json").exists())
            else None
        )
        if record is None:
            continue
        data = readings(domain)
        read = {t: d for t, d in data.items() if d.get("status", "read") == "read"}
        per_target = judge_all(read, record, merges_of(domain))
        not_read = {t: d["status"] for t, d in data.items() if t not in read}
        pooled = pooled_false_flags(per_target)
        ordered = sorted(per_target)
        best = longest_run(per_target)
        stability = {}
        for left, right in pairwise(ordered):
            swapped, total = swaps(read[left], read[right])
            stability[f"{left}-{right}"] = {"swapped": swapped, "patterns": total}
        summary[domain] = {
            "targets": per_target,
            "not_read": not_read,
            "pooled_false_flags": pooled,
            "measured_range": [best[0], best[-1]] if best else None,
            "range_valid": all(p["passes"] for p in pooled.values()),
            "stability": stability,
            "merges_at_1.50": per_target.get("1.50", {}).get("merge_budget"),
            "clustered_at_1.30": data.get("1.30", {}).get("clustered_tokens"),
        }
        print(f"{domain}: range {summary[domain]['measured_range']}, pooled {json.dumps(pooled)}")
        for target, values in per_target.items():
            print(f"  {target}: {json.dumps(values)}")
        for target, status in not_read.items():
            print(f"  {target}: {status}")
    write(RESULTS / "summary.json", {"domains": summary, "predictions": predictions(summary)})


def _inside(summary: dict[str, Any], domain: str, target: str) -> bool:
    values = summary.get(domain)
    if not values or not values["measured_range"] or not values["range_valid"]:
        return False
    low, high = values["measured_range"]
    return bool(low <= target <= high)


def predictions(summary: dict[str, Any]) -> dict[str, str]:
    def status(value: bool | None) -> str:
        return "unreadable" if value is None else ("HELD" if value else "FAILED")

    new = [d for d in ("tinystories", "wikipedia") if d in summary]
    complete = len(new) == 2
    tiny = summary.get("tinystories", {})
    wiki = summary.get("wikipedia", {})
    first_failure = []
    for values in summary.values():
        for target in sorted(values["targets"]):
            if target > "1.80" and not values["targets"][target]["holds"]:
                v = values["targets"][target]
                first_failure.append(
                    not all(v["dropped_found"].values()) and all(v["classes_found"].values())
                )
                break
    swaps = [
        s["swapped"] / s["patterns"]
        for values in summary.values()
        for s in values["stability"].values()
        if s["patterns"]
    ]
    return {
        "1": status(all(_inside(summary, d, "1.50") for d in new) if complete else None),
        "2": status(
            (tiny["merges_at_1.50"] < 3000 and "1.30" in tiny["targets"])
            if tiny.get("merges_at_1.50")
            else None
        ),
        # "Does not fit" is SciPy's table: judged on the tokens to cluster, since 1.30 was later
        # read on a tree built without the table (`experiments/ward_memory`).
        "3": status(
            (wiki["merges_at_1.50"] > 11381 and wiki["clustered_at_1.30"] > memory_limit())
            if wiki.get("merges_at_1.50") and wiki.get("clustered_at_1.30")
            else None
        ),
        "4": status((bool(first_failure) and all(first_failure)) if len(summary) == 3 else None),
        "5": status(all(v["range_valid"] for v in summary.values()) if complete else None),
        "6": status(max(swaps) < 0.001 if swaps else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    prep = commands.add_parser("prepare")
    prep.add_argument("domain", choices=DOMAINS)
    run = commands.add_parser("diagnose")
    run.add_argument("domain", choices=DOMAINS)
    run.add_argument("target", choices=[f"{t:.2f}" for t in TARGETS])
    commands.add_parser("summarize")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    match args.command:
        case "download":
            download()
        case "prepare":
            prepare(args.domain)
        case "diagnose":
            diagnose_target(args.domain, args.target)
        case "summarize":
            summarize()


if __name__ == "__main__":
    main()
