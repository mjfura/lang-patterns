# Syntactic sufficiency on natural text: a first, exploratory reading

The reading planned in `docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` ("First reading on
natural text"), whose two failure criteria were written before this ran. The eight clean sources
of `experiments/clean_domains`, first 6.5 MB sample each, tokenized at the merge budget that gives
about 1.5 tokens per word (`experiments/granularity`); classes by `induce_classes`, patterns with
one-token heads, domain split and sufficiency area. No model, no ground truth.

`poetry run python experiments/natural_sufficiency/natural.py run <source>` (under a minute and
about 600 MB each), `... show <source>` prints the classes and the pattern lists. Results:
`<source>_results.json`.

## Result (2026-09-28)

| Source | Classes | Domain / outlier patterns | Outliers' share of occurrences | Sufficiency area | Coverage quartiles |
|---|---|---|---|---|---|
| arXiv | 8 | 12,182 / 2,985 | 0.007 | 0.298 | 0.010, 0.023, 0.062 |
| PubMed | 8 | 14,690 / 4,574 | 0.010 | 0.277 | 0.006, 0.014, 0.039 |
| MultiNLI | 11 | 21,193 / 8,642 | 0.016 | 0.294 | 0.007, 0.016, 0.045 |
| C4 | 9 | 27,140 / 13,421 | 0.023 | 0.266 | 0.006, 0.011, 0.023 |
| FineWeb-Edu | 11 | 29,061 / 21,381 | 0.037 | 0.273 | 0.006, 0.012, 0.028 |
| FineWeb | 9 | 31,687 / 18,790 | 0.031 | 0.260 | 0.005, 0.009, 0.020 |
| Wikipedia | 19 | 35,875 / 37,958 | 0.075 | 0.295 | 0.007, 0.015, 0.037 |
| Case law | 12 | 32,144 / 50,928 | 0.094 | 0.297 | 0.006, 0.008, 0.016 |

**Criterion 1 — the classes look grammatical: partly met.** No class groups by topic. Several are
clean grammatical categories: Wikipedia's determiners and possessives (the, a, an, his, their, its,
other, this, all, most, some, many, her), its singular nouns (time, state, film, number, world…)
apart from its plural nouns (States, people, languages, species…), its adjectives and numerals
(first, more, American, two, new…); participles in arXiv (observed, used, found, shown) and case law
(filed, stated, held, provided, denied); nouns in every source (patients, study, treatment in
PubMed; case, order, law, evidence in case law). But three kinds of class are not categories:

- punctuation lumped with function words (arXiv: the , . of and a in to is for we that);
- word pieces: word-initial fragments (" s", " b", " al", " sp", " qu") and suffixes (ing, ed, ly,
  ation), the parts of words the tokenizer did not merge at 1.5 tokens per word;
- markup and notation (arXiv's xmath, @, xcite; digits).

arXiv and PubMed were cut at 8 classes, the floor `min_classes` sets: their merge heights show no
jump above it, so the floor, not the data, chose the cut. Wikipedia, with 19, reads the most like a
part-of-speech inventory.

**Criterion 2 — the low-coverage patterns are interpretable: failed as written.** Among the
frequent heads with the lowest coverage, debris dominates: `@`, `]`, `|`, digits, line breaks,
word fragments (" pro", " tra", " exp"). Among the word-like ones, what is flagged is readable, but
it is not what the indicator is meant to find:

| Source | Head → fillers seen (occurrences) |
|---|---|
| arXiv | due → 5 (590), non → 5 (822), black → 11 (278) |
| case law | pursuant → 7 (433), v → 9 (2,050), Haw → 7 (634) |
| C4 | able → 9 (414), such → 19 (565), during → 24 (384) |
| MultiNLI | didn → 1 (426), doesn → 1 (273), don → 2 (704) |

These are **fixed constructions**, not under-covered patterns: "due to", "able to", "pursuant to",
"such as", "black hole", "didn't", "Haw." in citations. Their slot is filled by one word by the
grammar of the language or the conventions of the domain, and the corpus shows it thousands of
times (missing mass near 0). On the recipe grammar a verb tied to one ingredient looked exactly
like this, and there the tie was the defect.

## What this means

- **Coverage alone cannot tell a fixed construction from a pattern the corpus under-shows.** Both
  read as a frequent head with few fillers and no missing mass. From the corpus alone they are the
  same observation; which one it is depends on whether the domain allows more fillers in that slot,
  which the corpus does not say. On the synthetic grammar we knew; on natural text the indicator
  can list the candidates, and deciding needs something outside the corpus (the domain's own
  grammar, a reader, or a larger corpus of the same domain).
- **Raw coverage is not comparable between heads.** Median coverage is 0.008–0.023 in every source:
  a head followed by a class of 500 tokens covers little of it unless it is a function word. The
  0.8 at which the recipe model generalized does not transfer. Coverage needs a reference for what
  a head with that many occurrences would cover if its fillers were not restricted (e.g. the
  fillers expected from as many draws of the class's own distribution).
- **At 1.5 tokens per word, a large part of the graph is word pieces**, which fill whole classes and
  most of the low-coverage list. Whether to read patterns over word-like tokens only, or at a
  larger merge budget, is a decision about the nodes the paper counts.
