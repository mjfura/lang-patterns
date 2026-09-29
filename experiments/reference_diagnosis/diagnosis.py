"""Fixed construction or missing data? Each pattern of a target corpus read against a reference
corpus of the same domain — predictions 1 to 4 of "Word-like tokens, diversity against chance, and a
reference corpus" in `docs/2026-09-28-syntactic-sufficiency-design.md`.

    diagnosis.py recipes            restricted recipe variants against the diverse one
    diagnosis.py natural <source>   the first 1/8 of a clean source's sample against all of it
    diagnosis.py show <name>        print a result

One tokenizer, one set of word-like tokens and one set of classes, all learned on the reference,
read both corpora. Results: `<name>_results.json` here.
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from corpus_generalization.analysis.syntactic import (
    Pattern,
    induce_classes,
    reference_diagnosis,
    split_domain,
    syntactic_patterns,
    word_like_tokens,
)
from corpus_generalization.packing import framed_documents
from corpus_generalization.synthetic.recipes import LEXICON, recipe_documents, verb_pairing
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

OUT = Path(__file__).parent
sys.path[:0] = [str(OUT.parent / "natural_sufficiency"), str(OUT.parent / "clean_domains")]

import clean_domains  # noqa: E402
import natural  # noqa: E402

TARGET_SHARE = 8  # the target is the first 1/8 of the reference's documents
MIN_TRUNK = 20  # target patterns with fewer occurrences are not read: too few to judge
LISTED = 30
CONSTRUCTIONS = ("due", "able", "pursuant", "such", "didn", "doesn", "don")


def read(
    reference_documents: list[str], target_documents: list[str], merges: int, head_length: int
) -> tuple[ByteLevelBPETokenizer, list[Pattern], list[Pattern], int]:
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        reference_documents, max_merges=merges, recognize_markers=True
    )
    reference = framed_documents(reference_documents, tokenizer, True)
    target = framed_documents(target_documents, tokenizer, True)
    words = word_like_tokens(
        reference,
        lambda token: tokenizer.decode([token]),
        boundary_ids={tokenizer.bos_id, tokenizer.eos_id},
    )
    classes = induce_classes(reference, candidates=words)
    reference_patterns = syntactic_patterns(reference, classes, head_length, heads=words)
    target_patterns = syntactic_patterns(target, classes, head_length, heads=words)
    return tokenizer, reference_patterns, target_patterns, classes.n_classes


def recipes() -> None:
    """Prediction 1 (diversity per variant) and 2 (restricted verbs are under-shown)."""
    results = {}
    diverse = recipe_documents(LEXICON, verb_pairing(LEXICON, None, 2, seed=0), 2000, seed=0)
    for fillers in (1, 2, 4, 8, 16, 24):
        restricted = recipe_documents(
            LEXICON, verb_pairing(LEXICON, fillers, 2, seed=0), 2000, seed=0
        )
        tokenizer, reference, target, n_classes = read(diverse, restricted, 1000, head_length=2)
        verbs = {tokenizer.encode(" " + verb, recognize_markers=True)[0] for verb in LEXICON.verbs}
        the = tokenizer.encode(" the", recognize_markers=True)[0]
        slots = [p for p in target if p.head[0] in verbs and p.head[1] == the]
        noun_class = Counter(p.filler_class for p in slots).most_common(1)[0][0]
        slots = [p for p in slots if p.filler_class == noun_class]
        diagnoses = reference_diagnosis(slots, reference)
        results[str(fillers)] = {
            "n_classes": n_classes,
            "target_diversity": statistics.mean(p.diversity for p in slots),
            "reference_diversity": statistics.mean(
                d.reference.diversity for d in diagnoses if d.reference is not None
            ),
            "verdicts": Counter(d.verdict for d in diagnoses),
        }
    (OUT / "recipes_results.json").write_text(json.dumps(results, indent=2) + "\n")
    show("recipes")


def natural_source(source: str) -> None:
    """Predictions 3 (named constructions are fixed) and 4 (fixed outnumber under-shown)."""
    documents = clean_domains.samples(json.loads(clean_domains.cache_of(source).read_text()))[0]
    target_documents = documents[: len(documents) // TARGET_SHARE]
    tokenizer, reference, target, n_classes = read(
        documents, target_documents, natural.MERGES[source], head_length=1
    )
    domain, _ = split_domain(target)
    readable = [p for p in domain if p.trunk >= MIN_TRUNK]
    diagnoses = reference_diagnosis(readable, reference)

    def row(diagnosis: object) -> dict[str, object]:
        assert hasattr(diagnosis, "pattern")
        pattern, ref = diagnosis.pattern, diagnosis.reference
        return {
            "head": tokenizer.decode(list(pattern.head)),
            "class": pattern.filler_class,
            "trunk": pattern.trunk,
            "fillers": pattern.fillers,
            "diversity": round(pattern.diversity, 3),
            "reference_trunk": ref.trunk if ref else None,
            "reference_diversity": round(ref.diversity, 3) if ref else None,
            "verdict": diagnosis.verdict,
        }

    by_verdict = {
        verdict: sorted(
            (d for d in diagnoses if d.verdict == verdict), key=lambda d: -d.pattern.trunk
        )
        for verdict in ("under-shown", "fixed", "unread")
    }
    results = {
        "source": source,
        "reference_documents": len(documents),
        "target_documents": len(target_documents),
        "n_classes": n_classes,
        "readable_patterns": len(readable),
        "verdicts": Counter(d.verdict for d in diagnoses),
        "constructions": [
            row(d)
            for d in diagnoses
            if tokenizer.decode(list(d.pattern.head)).strip() in CONSTRUCTIONS
        ],
        "examples": {
            verdict: [row(d) for d in rows[:LISTED]] for verdict, rows in by_verdict.items()
        },
    }
    (OUT / f"{source}_results.json").write_text(json.dumps(results, indent=2) + "\n")
    show(source)


def show(name: str) -> None:
    r = json.loads((OUT / f"{name}_results.json").read_text())
    if name == "recipes":
        for fillers, reading in r.items():
            print(
                f"fillers {fillers:>2}: {reading['n_classes']} classes, verb noun-slot diversity "
                f"target {reading['target_diversity']:.3f} reference "
                f"{reading['reference_diversity']:.3f}, verdicts {reading['verdicts']}"
            )
        return
    print(
        f"== {name}: target {r['target_documents']} of {r['reference_documents']} documents, "
        f"{r['n_classes']} classes, {r['readable_patterns']} readable patterns, "
        f"verdicts {r['verdicts']}"
    )
    print("  named constructions:")
    for p in r["constructions"]:
        print(f"    {p}")
    for verdict, rows in r["examples"].items():
        print(f"  {verdict}:")
        for p in rows[:15]:
            print(
                f"    {p['head']!r:>14} -> class {p['class']:>2}  trunk {p['trunk']:>5}  "
                f"fillers {p['fillers']:>4}  diversity {p['diversity']:.3f}  reference "
                f"{p['reference_diversity']}  ({p['reference_trunk']})"
            )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["recipes"]:
            recipes()
        case ["natural", source] if source in natural.MERGES:
            natural_source(source)
        case ["show", name]:
            show(name)
        case _:
            sys.exit(__doc__)
