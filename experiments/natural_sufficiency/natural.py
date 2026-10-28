"""Syntactic sufficiency on natural text: do the coarse classes look grammatical, and are the
patterns flagged as poorly covered interpretable? The exploratory reading planned in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` ("First reading on natural
text").

    natural.py run <source>     classes and one-token-head patterns of the source's first sample
    natural.py show <source>    print the reading

Sources and their first 6.5 MB samples come from `experiments/clean_domains` (its cache must be
there); each is tokenized at the merge budget that gives it about 1.5 tokens per word
(`experiments/granularity`). Results: `<source>_results.json` here.
"""

import json
import statistics
import sys
import time
from pathlib import Path

from corpus_generalization.analysis.syntactic import (
    induce_classes,
    split_domain,
    sufficiency_curve,
    syntactic_patterns,
)
from corpus_generalization.packing import framed_documents
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
sys.path[:0] = [str(OUT.parent / "clean_domains")]

import clean_domains  # noqa: E402

# Merges for about 1.5 tokens per word, from experiments/granularity; case law never reaches it
# (1.56 at 32,000).
MERGES = {
    "arxiv": 3001,
    "pubmed": 3931,
    "multi-nli": 5291,
    "c4": 8696,
    "fineweb-edu": 10802,
    "fineweb": 11563,
    "wikipedia": 13674,
    "case-law": 32000,
}
LISTED = 25
# A head counts as frequent for the low-coverage list when its pattern's trunk is at least this
# share of the largest domain trunk: rare heads have low coverage simply because they are rare.
FREQUENT_SHARE = 0.01


def run(source: str) -> None:
    start = time.time()
    documents = clean_domains.samples(json.loads(clean_domains.cache_of(source).read_text()))[0]
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        documents, max_merges=MERGES[source], recognize_markers=True
    )
    sequences = framed_documents(documents, tokenizer, True)
    words = sum(len(document.split()) for document in documents)
    tokens = sum(len(sequence) - 2 for sequence in sequences)
    classes = induce_classes(sequences)

    frequency: dict[int, int] = {}
    for sequence in sequences:
        for token in sequence:
            frequency[token] = frequency.get(token, 0) + 1
    members: dict[int, list[int]] = {}
    for token, number in classes.classes.items():
        members.setdefault(number, []).append(token)

    def spell(token: int) -> str:
        return tokenizer.decode([token]) or f"<{token}>"

    class_listing = [
        {
            "class": number,
            "size": len(group),
            "top": [spell(t) for t in sorted(group, key=lambda t: -frequency[t])[:15]],
        }
        for number, group in sorted(members.items(), key=lambda item: -len(item[1]))
    ]

    patterns = syntactic_patterns(sequences, classes, head_length=1)
    domain, outliers = split_domain(patterns)
    curve = sufficiency_curve(domain)
    largest = max(pattern.trunk for pattern in domain)

    def row(pattern: object) -> dict[str, object]:
        assert hasattr(pattern, "head")
        return {
            "head": spell(pattern.head[0]),
            "class": pattern.filler_class,
            "trunk": pattern.trunk,
            "fillers": pattern.fillers,
            "class_size": pattern.class_size,
            "coverage": round(pattern.coverage, 4),
            "missing_mass": round(pattern.missing_mass, 4),
            "documents": pattern.documents,
        }

    frequent = [pattern for pattern in domain if pattern.trunk >= FREQUENT_SHARE * largest]
    results = {
        "source": source,
        "merges": MERGES[source],
        "documents": len(documents),
        "tokens_per_word": tokens / words,
        "classified_tokens": len(classes.classes),
        "n_classes": classes.n_classes,
        "classes": class_listing,
        "patterns": len(patterns),
        "domain_patterns": len(domain),
        "outlier_patterns": len(outliers),
        "outlier_occurrence_share": sum(p.trunk for p in outliers) / sum(p.trunk for p in patterns),
        "sufficiency_auc": curve.auc,
        "coverage_quartiles": statistics.quantiles([p.coverage for p in domain], n=4),
        "most_travelled": [row(p) for p in sorted(domain, key=lambda p: -p.trunk)[:LISTED]],
        "frequent_lowest_coverage": [
            row(p) for p in sorted(frequent, key=lambda p: p.coverage)[:LISTED]
        ],
        "seconds": round(time.time() - start, 1),
    }
    (OUT / f"{source}_results.json").write_text(json.dumps(results, indent=2) + "\n")
    show(source)


def show(source: str) -> None:
    r = json.loads((OUT / f"{source}_results.json").read_text())
    print(
        f"\n== {source}: {r['documents']} documents, {r['tokens_per_word']:.2f} tokens/word, "
        f"{r['classified_tokens']} classified tokens in {r['n_classes']} classes, "
        f"{r['domain_patterns']} domain / {r['outlier_patterns']} outlier patterns "
        f"(outliers hold {r['outlier_occurrence_share']:.3f} of occurrences), "
        f"sufficiency area {r['sufficiency_auc']:.3f}, coverage quartiles "
        + ", ".join(f"{q:.3f}" for q in r["coverage_quartiles"])
    )
    for c in r["classes"]:
        print(f"  class {c['class']:>2} ({c['size']:>4}): {' | '.join(c['top'])}")
    for title in ("most_travelled", "frequent_lowest_coverage"):
        print(f"  {title}:")
        for p in r[title]:
            print(
                f"    {p['head']!r:>16} -> class {p['class']:>2}  trunk {p['trunk']:>7}  "
                f"fillers {p['fillers']:>4}/{p['class_size']:<4}  coverage {p['coverage']:.3f}  "
                f"missing {p['missing_mass']:.3f}"
            )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", source] if source in MERGES:
            run(source)
        case ["show", source] if source in MERGES:
            show(source)
        case _:
            sys.exit(__doc__)
