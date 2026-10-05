"""Replication of the class-level result on modals and possessives, with grammatical contrasts.

"Replication on two more classes" in `docs/2026-09-28-syntactic-sufficiency-design.md` records
the design and predictions before scoring or training.

    replication.py prepare              build poss_class_cut, poss_head_cut, poss_control from r50
    replication.py train <name> <seed>  train and score one possessive corpus
    replication.py items                fine-tagged items for modals and possessives
    replication.py summarize            corpus-side D, G per head, the four predictions per class
    replication.py diagnose <name>      corpus-diagnose on one cut against F (class rows)
    replication.py summarize-tool       the tool's class rows on the six cuts

The modal reading uses the models of `class_patterns.py`; the possessive reading its own.
Corpora and runs share `class_patterns`'s folders. No command runs on import.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import class_patterns as base
from class_patterns import divergence, natural_gap, subset_selection, syntactic_eval

from corpus_generalization.packing import framed_documents

MODALS = base.CLASSES["modal"]
POSSESSIVES = ("my", "your", "his", "our", "their", "its")
REMOVED = {"modal": ("can", "will"), "possessive": ("his", "their")}
MEMBERS = {"modal": MODALS, "possessive": POSSESSIVES}
CORPORA = {
    "modal": {"class": "class_cut", "head": "head_cut", "control": "control"},
    "possessive": {"class": "poss_class_cut", "head": "poss_head_cut", "control": "poss_control"},
}
POSSESSIVE_NAMES = tuple(CORPORA["possessive"].values())
SEEDS = base.SEEDS


def prepare() -> None:
    _, pool = natural_gap.split()
    subsets = json.loads((subset_selection.RESULTS / "subsets.json").read_text())
    documents = [pool[i] for i in subsets["r50"]["indices"]]
    segmented = base.segment(documents, POSSESSIVES, "NOUN")
    every = {
        (d, i) for d, sentences in enumerate(segmented) for i, (_, ts) in enumerate(sentences) if ts
    }
    partial = {
        (d, i)
        for d, sentences in enumerate(segmented)
        for i, (_, ts) in enumerate(sentences)
        if ts & set(REMOVED["possessive"])
    }
    class_cut = base.assemble(segmented, every)
    target = base.size(class_cut)
    corpora = {
        "poss_class_cut": class_cut,
        "poss_head_cut": base.assemble(segmented, base.match_bytes(segmented, partial, target, 0)),
        "poss_control": base.assemble(segmented, base.match_bytes(segmented, set(), target, 0)),
    }
    per_head = Counter(head for sentences in segmented for _, ts in sentences for head in ts)
    total = sum(len(sentences) for sentences in segmented)
    print(
        f"r50: {total} sentences; target sentences {len(every)} (all), "
        f"{len(partial)} (removed heads); per head {dict(per_head.most_common())}"
    )
    base.CORPORA.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, corpus in corpora.items():
        (base.CORPORA / f"{name}.json").write_text(json.dumps(corpus))
        summary[name] = {"documents": len(corpus), "bytes": base.size(corpus)}
        print(f"{name}: {len(corpus)} documents, {base.size(corpus):,} bytes")
    divergence.write_json(
        base.RESULTS / "possessive_corpora.json",
        {
            "sentences": total,
            "target_sentences": len(every),
            "removed_head_sentences": len(partial),
            "target_sentences_per_head": dict(per_head),
            "corpora": summary,
        },
    )


def train_corpus(name: str, seed: int) -> None:
    subset_selection.train_and_score(
        base.corpus(name),
        name,
        seed,
        base.RUNS / f"{name}_seed{seed}",
        base.RESULTS,
        "class_patterns",
    )


def runs() -> dict[str, Path]:
    names = (*base.CORPUS_NAMES, *POSSESSIVE_NAMES)
    return {f"{n}_seed{s}": base.RUNS / f"{n}_seed{s}" for n in names for s in SEEDS}


def fine_tags(
    documents: Sequence[str],
) -> tuple[dict[str, str], dict[str, str], Counter[str]]:
    """Per lowercase word: its open coarse class (≥ 90%, ≥ 20 occurrences, as syntactic_eval),
    and the majority fine tag among its VERB occurrences."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    coarse: dict[str, Counter[str]] = defaultdict(Counter)
    fine: dict[str, Counter[str]] = defaultdict(Counter)
    for index, doc in enumerate(nlp.pipe(documents, batch_size=64)):
        if index % 500 == 0:
            divergence.progress(f"Tagged {index} of {len(documents)} documents")
        for token in doc:
            text = token.text.lower()
            if syntactic_eval.WORD.fullmatch(text):
                coarse[text][token.pos_] += 1
                if token.pos_ == "VERB":
                    fine[text][token.tag_] += 1
    classes = syntactic_eval.word_classes(coarse)
    verb_forms = {w: fine[w].most_common(1)[0][0] for w in classes if classes[w] == "VERB"}
    frequency = Counter({w: sum(coarse[w].values()) for w in classes})
    return classes, verb_forms, frequency


def valid_and_contrast(
    kind: str, classes: dict[str, str], forms: dict[str, str]
) -> tuple[set[str], set[str]]:
    if kind == "modal":
        valid = {w for w, c in classes.items() if c == "VERB" and forms.get(w) == "VB"}
        contrast = {w for w, c in classes.items() if c in ("NOUN", "ADJ")}
    else:
        valid = {w for w, c in classes.items() if c == "NOUN"}
        contrast = {
            w
            for w, c in classes.items()
            if c == "ADV" or (c == "VERB" and forms.get(w) in ("VBD", "VBZ"))
        }
    return valid, contrast


def items() -> None:
    evaluation, pool = natural_gap.split()
    classes, forms, frequency = fine_tags(pool)
    heads = (*MODALS, *POSSESSIVES)
    followers: dict[str, set[str]] = {head: set() for head in heads}
    for document in pool:
        for match in syntactic_eval.follower_pattern(heads).finditer(document):
            followers[match.group(1)].add(match.group(2).lower())
    loaded = [syntactic_eval.load_run(path) for path in runs().values()]
    tokenizable = {
        w for w in classes if all(syntactic_eval.single_token(r, w) is not None for r in loaded)
    }
    prefixes = syntactic_eval.contexts(evaluation, heads)
    rows = {}
    for kind, members in MEMBERS.items():
        valid, contrast = valid_and_contrast(kind, classes, forms)
        for head in members:
            unseen = tokenizable - followers[head]
            pairs = syntactic_eval.pair_by_frequency(
                sorted(valid & unseen), sorted(contrast & unseen), frequency, seed=0
            )
            rows[head] = {"kind": kind, "pairs": pairs, "contexts": prefixes[head]}
            print(f"{kind} {head}: {len(pairs)} pairs, {len(prefixes[head])} contexts; {pairs[:3]}")
    divergence.write_json(base.RESULTS / "replication_items.json", rows)


def corpus_side(reading: divergence.FullReading) -> dict[str, object]:
    """Possessives: head-level and class-level D per corpus, minus the control."""
    head_d: dict[str, dict[str, float]] = {}
    class_d: dict[str, float] = {}
    for name in POSSESSIVE_NAMES:
        documents = framed_documents(base.corpus(name), reading.tokenizer, True)
        unigram = divergence.backoff_unigram(documents, reading.vocabulary)
        fillers = divergence.head_fillers(documents, reading.classes, reading.words)
        reference: Counter[int] = Counter()
        observed: Counter[int] = Counter()
        head_d[name] = {}
        for head in POSSESSIVES:
            token = reading.tokens[head]
            head_d[name][head] = divergence.divergence(
                reading.reference[token], fillers.get(token, Counter()), unigram
            )
            reference.update(reading.reference[token])
            observed.update(fillers.get(token, Counter()))
        class_d[name] = divergence.divergence(reference, observed, unigram)
    names = CORPORA["possessive"]
    delta_head = {
        cut: {h: head_d[names[cut]][h] - head_d[names["control"]][h] for h in POSSESSIVES}
        for cut in ("class", "head")
    }
    delta_class = {
        cut: class_d[names[cut]] - class_d[names["control"]] for cut in ("class", "head")
    }
    siblings = [h for h in POSSESSIVES if h not in REMOVED["possessive"]]
    holds = (
        all(delta_head["head"][h] >= 1 for h in REMOVED["possessive"])
        and all(delta_head["class"][h] >= 1 for h in POSSESSIVES)
        and all(delta_head["head"][h] < 0.2 for h in siblings)
        and delta_class["head"] < delta_class["class"] / 2
    )
    print("possessive head-level delta D (head cut / class cut):")
    for h in POSSESSIVES:
        print(f"  {h:6} {delta_head['head'][h]:+.3f} / {delta_head['class'][h]:+.3f}")
    print(f"possessive class-level delta D: {delta_class}")
    return {"delta_head": delta_head, "delta_class": delta_class, "holds": holds}


def summarize() -> None:
    reading = divergence.read_full()
    side = corpus_side(reading)
    item_rows = json.loads((base.RESULTS / "replication_items.json").read_text())
    cut_items = json.loads(
        (base.EXPERIMENTS / "syntactic_eval/results/items_cuts.json").read_text()
    )
    listed = {h.lower() for h in (*base.HEADS, *POSSESSIVES)}
    other_items = {h: i for h, i in cut_items["heads"].items() if h.lower() not in listed}
    g: dict[str, dict[str, float]] = {}
    g_other: dict[str, dict[str, float]] = {}
    for model, path in runs().items():
        divergence.progress(f"Scoring {model}")
        run = syntactic_eval.load_run(path)
        g[model] = {
            h: float(syntactic_eval.head_reading(run, i)["g"]) for h, i in item_rows.items()
        }
        g_other[model] = {
            h: float(syntactic_eval.head_reading(run, i)["g"]) for h, i in other_items.items()
        }

    def mean_g(scores: dict[str, dict[str, float]], corpus: str, head: str) -> float:
        return statistics.mean(scores[f"{corpus}_seed{s}"][head] for s in SEEDS)

    report: dict[str, object] = {"possessive_corpus_side": side}
    predictions: dict[str, str] = {"1 possessive": syntactic_eval.status(bool(side["holds"]))}
    for kind, members in MEMBERS.items():
        names = CORPORA[kind]
        delta = {
            cut: {h: mean_g(g, names[cut], h) - mean_g(g, names["control"], h) for h in members}
            for cut in ("class", "head")
        }
        print(f"\n{kind}: G per head, mean of seeds (control / head cut / class cut)")
        for h in members:
            print(
                f"  {h:7} {mean_g(g, names['control'], h):.3f} / {mean_g(g, names['head'], h):.3f}"
                f" / {mean_g(g, names['class'], h):.3f}"
            )
        class_mean = statistics.mean(delta["class"].values())
        removed_head = statistics.mean(delta["head"][h] for h in REMOVED[kind])
        removed_class = statistics.mean(delta["class"][h] for h in REMOVED[kind])
        siblings_head = statistics.mean(delta["head"][h] for h in members if h not in REMOVED[kind])
        shift = {
            cut: statistics.mean(
                mean_g(g_other, names[cut], h) - mean_g(g_other, names["control"], h)
                for h in other_items
            )
            for cut in ("class", "head")
        }
        print(
            f"P2 class cut mean delta G {class_mean:+.3f}\n"
            f"P3 removed heads: head cut {removed_head:+.3f}, class cut {removed_class:+.3f}; "
            f"siblings in head cut {siblings_head:+.3f}\n"
            f"P4 other heads mean signed delta G {shift}"
        )
        predictions[f"2 {kind}"] = syntactic_eval.status(class_mean <= -0.10)
        predictions[f"3 {kind}"] = syntactic_eval.status(abs(removed_head) < abs(removed_class) / 2)
        predictions[f"4 {kind}"] = syntactic_eval.status(
            all(abs(v) <= 0.02 for v in shift.values())
        )
        report[kind] = {
            "delta_g": delta,
            "class_mean_delta_g": class_mean,
            "removed_head_cut": removed_head,
            "removed_class_cut": removed_class,
            "siblings_head_cut": siblings_head,
            "other_heads_signed_shift": shift,
        }
    print(f"\nPredictions: {json.dumps(predictions)}")
    report["g"] = g
    report["predictions"] = predictions
    divergence.write_json(base.RESULTS / "replication_summary.json", report)


TOOL_CORPORA = (*base.CORPUS_NAMES, *POSSESSIVE_NAMES)
TOOL_HEADS = {"pronoun": "she", "modal": "can", "possessive": "his"}
TOOL_EXPECTED = {"class_cut": ("pronoun", "modal"), "poss_class_cut": ("possessive",)}


def tool_diagnose(name: str) -> None:
    """`corpus-diagnose` on one cut against F, at F's merge budget; one-token heads only, which
    are all its class rows read."""
    from corpus_generalization.diagnose import diagnose

    _, pool = natural_gap.split()
    report = diagnose(
        base.corpus(name),
        pool,
        merge_budget=divergence.SUBSET_MERGES,
        head_lengths=(1,),
        progress=divergence.progress,
    )
    payload = report.to_json()
    divergence.write_json(
        base.RESULTS / f"tool_{name}.json",
        {"class_rows": payload["class_rows"], "counts": payload["counts"]},
    )
    print(report.summary())


def summarize_tool() -> None:
    held: dict[str, bool] = {"1": True, "2": True}
    rows_out: dict[str, object] = {}
    for name in TOOL_CORPORA:
        rows = json.loads((base.RESULTS / f"tool_{name}.json").read_text())["class_rows"]
        found = {}
        for kind, head in TOOL_HEADS.items():
            row = next((r for r in rows if head in [h.strip() for h in r["heads"]]), None)
            found[kind] = row
            verdict = row["verdict"] if row else "absent"
            expected = kind in TOOL_EXPECTED.get(name, ())
            if expected:
                held["1"] &= verdict == "under-shown"
            else:
                held["2"] &= verdict != "under-shown"
            detail = (
                f"D {row['divergence']:.3f}, excess {row['excess']:+.3f}, heads "
                f"{[h.strip() for h in row['heads']][:8]}"
                if row
                else ""
            )
            print(f"{name:15} {kind:10} {verdict:13} {detail}")
        flagged = [
            [h.strip() for h in r["heads"]][:6] for r in rows if r["verdict"] == "under-shown"
        ]
        print(f"{name:15} all under-shown class rows: {flagged}")
        rows_out[name] = found
    predictions = {k: syntactic_eval.status(v) for k, v in held.items()}
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(
        base.RESULTS / "tool_summary.json", {"rows": rows_out, "predictions": predictions}
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    training = commands.add_parser("train")
    training.add_argument("name", choices=POSSESSIVE_NAMES)
    training.add_argument("seed", type=int)
    commands.add_parser("items")
    commands.add_parser("summarize")
    tool = commands.add_parser("diagnose")
    tool.add_argument("name", choices=TOOL_CORPORA)
    commands.add_parser("summarize-tool")
    args = parser.parse_args()
    match args.command:
        case "prepare":
            prepare()
        case "train":
            train_corpus(args.name, args.seed)
        case "items":
            items()
        case "summarize":
            summarize()
        case "diagnose":
            tool_diagnose(args.name)
        case "summarize-tool":
            summarize_tool()


if __name__ == "__main__":
    main()
