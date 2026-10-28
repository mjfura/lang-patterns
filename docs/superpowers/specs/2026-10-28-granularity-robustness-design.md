# The tokens-per-word range: corpus size, pool draw and a knowledge domain

**Status:** draft, waiting for the user's review. No plan yet.

Proposed to the user on 2026-10-28. Design and predictions, written before running.

## Why

`experiments/granularity_domains` measured where the verdicts hold, from 1.30 to 2.50 tokens per
word, in three domains:

| Domain | Measured range | First failure above |
|---|---|---|
| TinyStories | 1.30–2.00 | 2.20: the modals' class |
| FineWeb | 1.30–2.50 | none |
| Wikipedia | 1.30–2.20 | 2.50: the base verbs' class |

The intersection is 1.30–2.00, and 1.50 is inside every range. That supports one claim: **1.50 is
a safe default for English text of three kinds, at one size.** It does not yet support a range as
a general rule. Three gaps remain:

1. **One size.** Every pool is 14.4 MB, F's size. Tokens per word was chosen so that the
   granularity would not depend on corpus size, but that was never tested. A domain corpus is
   often a few MB, and the tool is meant for any size.
2. **One draw.** Each domain was read on one pool: one set of documents, one tree, one set of
   cuts. The upper edge (TinyStories fails at 2.20) comes from that single draw. Another draw
   could move it.
3. **No knowledge domain.** TinyStories, FineWeb and Wikipedia are general text. The paper is
   about knowledge-domain corpora, which have their own terms, abbreviations and numbers.

Another language is out of scope, as before: it changes the pre-tokenizer's word boundaries and
the morphology together. It is reported as a limitation.

The question: **does the range hold across corpus size, across pool draws, and in a knowledge
domain?**

## What stays the same

Everything in `docs/superpowers/specs/2026-10-25-granularity-across-domains-design.md` and its amendments, unless
stated here:

- the 22 labeled cuts, built by the same rules and seeds from each pool;
- the feasibility gate and its replacements;
- the pass rule: (a) every removed class flagged by a class row, (b) every dropped word that is
  one token read `absent`;
- pooled false flags at most 2 × alpha (12%):
  - a recurring flag on one control counts once;
  - class rows are read on the four whole-document controls, head rows and `absent` on all eight
    random controls;
- two workers.

**One change: the clustering.** Where the class tree's pairwise table fits (at most 29,274
clustered tokens, F at 1.35), SciPy builds the tree, as before. Above that, `linkage_vector`
builds it, which `experiments/ward_memory` showed gives the same verdicts. Nothing is skipped
for memory any more. The tree each reading used is recorded.

## The new domain: PubMed abstracts

- **Source:** `MedRAG/pubmed`, the PubMed abstracts that MedRAG uses, one JSONL file per PubMed
  baseline chunk.
- **Field:** `content`, the abstract alone. The title is left out: titles of older articles are
  bracketed translations.
- **Chunks:** the highest-numbered chunks, which hold the most recent articles, read in file
  order. The low-numbered chunks hold articles from the 1970s.

Biomedical abstracts are a knowledge domain in the paper's sense:

- a technical vocabulary;
- abbreviations, units and numbers;
- a fixed register.

They are also expected to need the most merges of the four corpora, so they test the fine end
too.

**The likely weak spot is the gate.** Abstracts use `we` and modals (`can`, `may`), but few
`she`/`he`, and few base verbs like `make`. A slot that fails the gate is replaced in the declared
order (prepositions, then determiners), as before, and the replacement is recorded.

## Part A: corpus size

- **Sizes:**
  - **¼ (3.6 MB) in all four corpora.** A small corpus is the common case for a knowledge domain.
  - **4× (57.6 MB) in TinyStories only.** It is the one corpus where 4× is cheap: few merges, few
    clustered tokens. In the others, 4× needs `linkage_vector` even at 1.50, and hours per cell.
- **Pools are nested.** The ¼ pool is the first quarter of the 1× pool, in the same order. The 4×
  pool starts with the 1× pool and continues in cache order. So only the size changes, not the
  documents' kind.
- **One tokenizer per size**, learned on that pool. A smaller pool reaches each granularity with
  fewer merges. That is the point of measuring granularity in tokens per word, and it is what
  this part tests.
- **Granularities:** 1.30, 1.50, 2.00, 2.20 and 2.50:
  - 1.30: the finest declared;
  - 1.50: the tool's target;
  - 2.00, 2.20 and 2.50: around the two upper edges found.

  A small pool may run out of merges before 1.30. That reading is recorded as unreachable, not
  as a failure.
- **The 22 cuts are rebuilt from each pool**, with the same seeds.

## Part B: the pool draw

- **Two more pools per domain with an upper edge**: TinyStories and Wikipedia, plus PubMed if
  Part C finds an edge in it.
- **The pools are disjoint, each the size of F.** Pool 0 is the existing one. Pools 1 and 2 are
  the next two 14.4 MB slices of the source in cache order.
- **Everything is rebuilt per pool:**
  - the tokenizer and the tree;
  - the anchors' classes;
  - the 22 cuts, with the same seeds;
  - the dropped words, by the same rank-band rule.

  A new pool changes the documents, so it changes all of these at once. That is the draw a user's
  corpus would make.
- **Granularities:** 1.50, 2.00, 2.20 and 2.50, the target and the upper edge.
- **More data is needed:**
  - **TinyStories:** the cache holds 25,000 stories (22.3 MB). 4× and three disjoint pools need
    57.6 MB, about 65,000 stories.
  - **Wikipedia:** the cache holds 1,500 articles (32.6 MB) and three pools need 43.2 MB. It is
    extended in the same streaming order, so pool 0 is unchanged. This is checked before reading.

## Part C: the knowledge domain at full size

PubMed at 1×, read at all eleven granularities, as the other three were.

## Judging

- **Per variant** (a size or a pool): at each granularity read, does it hold? The verdict is
  compared with the domain's original pool at the same granularity.
- **A variant agrees** when every granularity read gives the same hold or fail as the original.
  An unreachable granularity is left out of the comparison.
- **Pooled false flags are read per variant.** At five granularities that is 20 class-row
  readings and 40 for heads and `absent`, so 12% allows two flagged controls of 20. That is a
  coarse check, and it is reported as such.
- **Also recorded, not part of the pass rule:**
  - merges and clustered tokens per granularity;
  - the tree used (SciPy or `linkage_vector`);
  - dropped words that are one token;
  - swaps between adjacent granularities read.

## Choices that are not measured

These are set by judgment, not by a measurement. They are listed so that they can be challenged:

- **¼ and 4×.** A span of 16 times, limited above by cost. Nothing below ¼ is read.
- **Two more pools.** Three draws per domain is the least that can show whether an edge moves. It
  cannot give the edge's spread.
- **The grid above 2.00** (2.00, 2.20, 2.50) locates an edge only to within one step.
- **PubMed** as the single knowledge domain. The alternative considered was legal text
  (`coastalcph/lex_glue`, ECtHR facts). PubMed was chosen because its vocabulary is the furthest
  from the three general corpora.

## Predictions

**A. Size**

1. At ¼, **1.50 holds in all four corpora.**
2. At ¼, **every corpus agrees with its 1× pool at 1.30–2.00**, where 1.30 is reachable.
3. Above 2.00, **the first failure at ¼ is at most one grid step from the 1× pool's.** A smaller
   pool has fewer occurrences per class, so if the edge moves, it moves down.
4. At 4×, **TinyStories agrees with its 1× pool** at all five granularities.
5. **Pooled false flags pass** in every size variant.

**B. Draw**

6. **1.50 holds in every new pool.**
7. **The first failure is at the same granularity in all three pools** of each domain: 2.20 in
   TinyStories, 2.50 in Wikipedia. It is the same class that fails first (the modals in
   TinyStories, the base verbs in Wikipedia).

**C. Knowledge domain**

8. **PubMed needs more merges than Wikipedia for 1.50** (over 16,022).
9. **All four of PubMed's slots pass the gate**, with no replacement.
10. **PubMed's measured range contains 1.30–2.00.** So the four-corpus intersection stays
    1.30–2.00, and 1.50 stays inside it.
11. **Pooled false flags pass** in PubMed.

## What follows from the outcome

- **Everything holds:** the range 1.30–2.00 is stated for English corpora from 3.6 to 57.6 MB in
  four domains, with 57.6 MB measured in TinyStories only. `GRANULARITY_RANGE` changes to it, with
  a test that pins it. The paper's §5.8 reports the four domains and the sizes, and §7 keeps
  language and sizes below 3.6 MB as limitations.
- **1.50 fails at ¼ somewhere:** the failure is a size limit, not a granularity limit. The tool
  needs a minimum corpus size, designed with the user as a separate question.
- **The edge moves between draws:** the upper end becomes the lowest edge seen, minus one step,
  and the paper reports the edge as varying with the draw.
- **PubMed's range excludes 1.50, or a slot cannot be built:** the path is revised with the user
  before any change to the tool, as in the previous design.

## Code

- `experiments/granularity_robustness/`: one script that reuses `granularity_domains`'s
  functions, adds `pubmed` as a domain, and adds variants named `<domain>@0.25`,
  `<domain>@4` and `<domain>#1`, `<domain>#2`.
  - **Commands:** `download`, `prepare <variant>`, `diagnose <variant> <t>`, `summarize`.
  - **Pools, merges and trees** go to `runs/granularity_robustness/` (gitignored).
  - **Results:** `results/<variant>/`; per-target readings are gitignored, `prepared.json` and
    `summary.json` are kept.
- **Clustering:** the SciPy/`linkage_vector` switch above 29,274 tokens lives in the experiment,
  not in `src/`. Moving it into `class_tree` is the decision still open from
  `experiments/ward_memory`.
- No change to `src/` unless the outcome calls for one, and then with a test first.

## Order and cost

From the measured cell times:

- a TinyStories cell takes about 35 minutes;
- a Wikipedia cell takes 25 minutes at 2.00 and 1 hour at 1.50;
- a `linkage_vector` tree at the fine end takes 1.5–10 hours.

Earlier estimates ran up to three times short, so the cheapest and most informative cells come
first:

1. Download and prepare every pool, and check that the gate passes.
2. Part A at ¼, four corpora × five granularities: about 5 hours.
3. Part B, TinyStories and Wikipedia: about 10 hours.
4. Part C, PubMed from 2.50 down to 1.30. The fine end is the slow part: 20 hours or more.
5. Part A at 4×, TinyStories: about 7 hours.
6. Part B for PubMed, if it has an edge: about 5 hours.

That is about two to three days of compute. Results are summarized after each step, so a step
that fails its predictions can stop the rest for a decision with the user.
