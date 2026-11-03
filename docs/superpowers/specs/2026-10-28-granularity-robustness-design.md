# The tokens-per-word range: corpus size, pool draw and knowledge domains

**Status:** experiment done (2026-11-01), result in
`experiments/granularity_robustness/README.md`. Applied 2026-11-02: `GRANULARITY_RANGE` is
1.35–1.70, `diagnose` warns below 14.4 MB (`MIN_DOMAIN_BYTES`), and the paper's §3.6, §5.8, §7,
§8 and Appendix A (EN and ES) report it. Plan:
`docs/superpowers/plans/2026-10-28-granularity-robustness.md`.

Proposed to the user on 2026-10-28. Design and predictions, written before running.

The first draft read a reduced set of cells: ¼ in four corpora, 4× in TinyStories only, two extra
pools in two domains, five granularities per variant. It was cut down to fit a 16 GB machine, on
which SciPy's Ward table stopped fitting at 29,274 clustered tokens and two workers was the most
that ran safely. The work moved to a 64 GB, 18-core machine, and the user asked for the full
grid. This revision replaces that draft.

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

The question: **does the range hold across corpus size, across pool draws, and in knowledge
domains?**

## The grid

| Axis | Values | Count |
|---|---|---|
| Corpus | TinyStories, FineWeb, Wikipedia, PubMed, legal (ECtHR) | 5 |
| Size and draw | 1/16, ¼ and 1× in pools #0, #1, #2; 4× in pool #0 only | 10 |
| Granularity | 1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.20, 2.50 | 11 |

That is **550 readings**:

- **33 are done:** TinyStories, FineWeb and Wikipedia at 1×, pool #0, in
  `experiments/granularity_domains`. They are not read again.
- **517 are new.**

Every variant is read at all eleven granularities, so each one gets the same pass rule, at the same
strength, as the original pools.

**Rejected:** three draws at 4× (+110 readings). A larger pool varies less from one draw to the
next, so the extra 4× draws are the least informative cells, and they triple the data each corpus
needs. The draw is read where it matters most, at the small sizes.

## What stays the same

Everything in `docs/superpowers/specs/2026-10-25-granularity-across-domains-design.md` and its
amendments, unless stated here:

- the 22 labeled cuts, built by the same rules and seeds from each pool;
- the feasibility gate and its replacements;
- the pass rule: (a) every removed class flagged by a class row, (b) every dropped word that is
  one token read `absent`;
- pooled false flags at most 2 × alpha (12%):
  - a recurring flag on one control counts once;
  - class rows are read on the four whole-document controls, head rows and `absent` on all eight
    random controls.

## What changes with the machine

Neither change can alter a verdict, only the time a reading takes.

- **The clustering.** SciPy builds the class tree wherever its pairwise table fits in memory;
  `linkage_vector` builds it above that. `experiments/ward_memory` showed that both give the same
  verdicts.
  - The 16 GB limit of 29,274 clustered tokens no longer applies.
  - The new limit is a projected peak, set above SciPy's peaks measured in
    `experiments/ward_memory`:
    - 1.38 GB at 11,459 tokens;
    - 1.76 GB at 15,893 tokens.

    The projection is two copies of the condensed table (8n² bytes) plus 2 GB, which bounds both
    measured peaks from above. SciPy is used when it is at most 48 GB, so up to about 75,800
    clustered tokens. Wikipedia at 1.30 (57,944 tokens, about 29 GB projected) is expected to
    fit.
  - The tree each reading used is recorded.
- **Workers.** Diagnosis workers are no longer fixed at two. Their number is set by a calibration
  reading (step 3 of "Order"):
  - workers are added while the machine's peak memory stays below 48 GB, three quarters of 64 GB;
  - the margin is there because a memory-pressure kill takes the whole machine down.

  Workers read cuts in parallel and do not change any reading.

## The two knowledge domains

### PubMed abstracts

- **Source:** `MedRAG/pubmed`, the PubMed abstracts that MedRAG uses. It holds 1,000 JSONL files
  (`chunk/pubmed23n0001.jsonl` to `pubmed23n1000.jsonl`), one per PubMed baseline chunk, about
  87 MB each.
- **Field:** `content`, the abstract alone. The title is left out: titles of older articles are
  bracketed translations.
- **Chunks:** the fewest highest-numbered chunks whose abstracts reach the bytes the grid needs
  (86.4 MB, see "Pools"), read in ascending file order. The high-numbered chunks hold the most
  recent articles; the low-numbered ones hold articles from the 1970s.

### Legal: European Court of Human Rights cases

- **Source:** `coastalcph/lex_glue`, config `ecthr_a`. It holds the facts of 11,000 cases (9,000
  train, 1,000 validation, 1,000 test); `ecthr_b` holds the same cases, so it adds no text.
- **Field:** `text`, a list of the case's fact paragraphs, joined with a newline into one
  document per case.
- **Order:** train, then validation, then test, each in file order.
- **Size is the risk.** The three splits hold about 100 MB in memory, labels included, against the
  86.4 MB the grid needs. This is checked after download. If the text falls short, legal drops
  pool #2, and that is recorded.

### Why these two

Both are knowledge domains in the paper's sense: a technical vocabulary, a fixed register, and
their own conventions.

- **PubMed** has abbreviations, units and numbers, and its vocabulary is the furthest from the
  three general corpora. It is expected to need the most merges, so it tests the fine end.
- **Legal** has long formulaic sentences, case citations and dates. It narrates facts in the third
  person (`the applicant`, `he`, `his`), unlike PubMed's `we`.

Two knowledge domains, so that the claim about domain corpora does not rest on one.

### The gate

Each domain's likely weak spot is the gate, as before:

- abstracts use `we` and modals (`can`, `may`), but few `she`/`he`, and few base verbs like `make`;
- legal facts use few modals.

A slot that fails the gate is replaced in the declared order (prepositions, then determiners), as
before, and the replacement is recorded.

## Pools

Every pool is cut from the corpus's documents in cache order, documents kept whole, until it
reaches its bytes.

- **Pool #0 at 1×** is the existing pool:
  - F for FineWeb (`natural_gap.split()`, 4,600 documents, 14.4 MB);
  - the leading 14.4 MB of the cache for the other corpora.
- **4× (57.6 MB), pool #0 only.** Pool #0 at 1×, followed by the next documents in cache order.
  For FineWeb these are the documents streamed after the 5,000 that F is drawn from. F's 400
  evaluation documents stay out of every pool.
- **Pools #1 and #2 at 1×.** The next two 14.4 MB slices after the 4× pool. They are disjoint from
  each other and from all of pool #0, 4× included.
- **¼ (3.6 MB) and 1/16 (0.9 MB) of each pool.** The leading documents of that pool at 1×. The
  sizes are nested, so only the size changes, not the documents' kind.
- **Each corpus needs 86.4 MB**: 57.6 MB for the 4× pool and two 14.4 MB slices after it.

**Downloads:**

| Corpus | Cached now (lost with the old machine) | Needed |
|---|---|---|
| TinyStories | 25,000 stories, 22.3 MB | about 100,000 stories |
| FineWeb | 5,000 documents | 5,000 plus about 25,000 more, streamed after them |
| Wikipedia | 1,500 articles, 32.6 MB | about 4,000 articles |
| PubMed | none | the highest-numbered chunks, about 86.4 MB of abstracts |
| Legal | none | all three splits of `ecthr_a` |

The three general corpora are streamed in the same order as before, so pool #0 should come out
unchanged. That is checked before any new reading:

- each corpus's tokenizer at 1×, pool #0, must reach every granularity at the merge counts in
  `experiments/granularity_domains/results/summary.json` (for example TinyStories 1,223 merges at
  1.50, FineWeb 11,381, Wikipedia 16,022);
- a mismatch stops the run for a decision with the user.

**Whole documents at small sizes.** A pool at 1/16 holds about:

- 40 Wikipedia articles;
- 100 legal cases;
- 1,000 TinyStories stories.

The 25% document cuts of a pool that small remove a handful of articles, so the size-matched draws
are coarse. The 10-25 design kept articles whole for the same reason (splitting them would need an
arbitrary length), and the false flags on the controls measure whether coarse draws matter. A
corpus that fails at 1/16 is read with this in mind (see "What follows").

**Every new variant is prepared by the 10-25 rules**, FineWeb's included. F at 1× keeps step 1's
cuts, among them its eight hand-picked dropped words. FineWeb's other variants take their dropped
words by the rank-band rule, as TinyStories and Wikipedia did.

**Per pool, everything is rebuilt:**

- the tokenizer and the tree;
- the anchors' classes;
- the 22 cuts, with the same seeds;
- the dropped words, by the same rank-band rule.

A new pool changes the documents, so it changes all of these at once. That is the draw a user's
corpus would make.

**One tokenizer per pool**, learned on it. A smaller pool reaches each granularity with fewer
merges. That is the point of measuring granularity in tokens per word, and it is what the size
axis tests.

A small pool may run out of merges before 1.30. That reading is recorded as unreachable, not as a
failure.

## Judging

- **Per variant** (a size or a pool): at each granularity, does it hold? The verdict is compared
  with the corpus's pool #0 at 1× at the same granularity. For PubMed and legal, pool #0 at 1× is
  itself new: its measured range plays the role the original pools play for the other three.
- **A variant agrees** when every granularity read gives the same hold or fail as pool #0 at 1×.
  An unreachable granularity is left out of the comparison.
- **Pooled false flags are read per variant**, over its eleven granularities: 44 class-row
  readings and 88 for heads and `absent`, as for the original pools.
- **Also recorded, not part of the pass rule:**
  - merges and clustered tokens per granularity;
  - the tree used (SciPy or `linkage_vector`), and its time and peak memory;
  - the gate's replacements;
  - dropped words that are one token;
  - swaps between adjacent granularities.

## Choices that are not measured

These are set by judgment, not by a measurement. They are listed so that they can be challenged:

- **The sizes, 1/16 to 4×.** A span of 64 times. Nothing below 0.9 MB or above 57.6 MB is read.
- **One draw at 4×.** This rests on the argument that a larger pool varies less between draws. It
  is not measured.
- **Three draws at the smaller sizes.** That is the least that can show whether an edge moves. It
  cannot give the edge's spread.
- **The grid above 2.00** (2.00, 2.20, 2.50) locates an edge only to within one step.
- **The knowledge domains:** PubMed and legal (ECtHR), and within them:
  - PubMed's highest-numbered chunks;
  - legal's paragraphs joined with newlines.
- **The 48 GB memory ceiling for workers.** It is an operating margin, and changes no reading.

## Predictions

**A. Size**

1. **1.50 holds in all five corpora at every size**, 1/16 included.
2. **Every size variant agrees with its pool #0 at 1× from 1.30 to 2.00**, where the granularity
   is reachable.
3. **Above 2.00, the first failure is at most one grid step from that of pool #0 at 1×.** A smaller
   pool has fewer occurrences per class, so if the edge moves, it moves down at ¼ and 1/16.
4. **At 4×, every corpus agrees with its pool #0 at 1× at all eleven granularities.**
5. **Pooled false flags pass in every size variant.** 1/16 in Wikipedia and legal is the most
   likely to fail, because their draws are coarse.

**B. Draw**

6. **1.50 holds in every pool**, at every size.
7. **At 1×, the first failure is at the same granularity in all three pools of each corpus:**
   - 2.20 in TinyStories, with the modals' class failing first;
   - 2.50 in Wikipedia, with the base verbs' class;
   - none in FineWeb.

   For PubMed and legal, the edge is wherever pool #0 puts it.

**C. Knowledge domains**

8. **PubMed needs more merges than Wikipedia to reach 1.50** (more than 16,022). **Legal needs
   fewer than Wikipedia and more than TinyStories** (between 1,223 and 16,022): its vocabulary is
   formulaic, but it has names and places.
9. **All four slots pass the gate, with no replacement**, in both PubMed and legal.
10. **The measured range of both PubMed and legal contains 1.30–2.00.** So the intersection over
    the five corpora stays 1.30–2.00, and 1.50 stays inside it.
11. **Pooled false flags pass** in both.

**D. Memory**

12. **Every reading at 1× and below fits SciPy's table on this machine.** Only 4× readings at the
    fine end of FineWeb, Wikipedia, PubMed and legal (at or below 1.50) may need `linkage_vector`.

## What follows from the outcome

- **Everything holds:** the range 1.30–2.00 is stated for English corpora from 0.9 to 57.6 MB, in
  five domains, two of them knowledge domains.
  - `GRANULARITY_RANGE` changes to it, with a test that pins it.
  - The paper's §5.8 reports the domains and the sizes.
  - §7 keeps language, sizes outside 0.9–57.6 MB and a single draw at 4× as limitations.
- **1.50 fails at a small size somewhere:** the failure is a size limit, not a granularity limit.
  - The tool needs a minimum corpus size, set from the smallest size at which every corpus holds.
  - It is designed with the user as a separate question.
  - A failure at 1/16 in Wikipedia or legal alone, with failing false flags, points to coarse
    draws rather than to granularity. It is reported as such.
- **The edge moves between draws:** the upper end becomes the lowest edge seen, minus one step.
  The paper reports the edge as varying with the draw.
- **A knowledge domain's range excludes 1.50, or a slot cannot be built:** the path is revised with
  the user before any change to the tool, as in the previous design.

## Code

- **`experiments/granularity_robustness/`:** one script that reuses `granularity_domains`'s
  functions.
  - It adds `pubmed` and `legal` as corpora.
  - It adds variants named `<corpus>@<size>#<pool>`, for example `wikipedia@0.25#1`,
    `tinystories@4#0` and `pubmed@1#0`.
  - **Commands:**
    - `download <corpus>`;
    - `prepare <variant>`: pool, tokenizer, clustered tokens per granularity, cuts, gate;
    - `diagnose <variant> <t>`;
    - `summarize`.
  - **Pools, merges and trees** go to `runs/granularity_robustness/` (gitignored).
  - **Results** go to `results/<variant>/`. Per-target readings are gitignored; `prepared.json`
    and `summary.json` are kept.
- **`src/corpus_generalization/sources.py`:** `huggingface_documents` takes an optional
  `data_files`, so that PubMed's chunks can be named. It is tested first. Legal's paragraph
  joining stays in the experiment.
- **Clustering:** the SciPy/`linkage_vector` switch lives in the experiment, not in `src/`.
  Moving it into `class_tree` is the decision still open from `experiments/ward_memory`.
- No other change to `src/` unless the outcome calls for one, and then with a test first.

## Order

Earlier time estimates ran up to three times short, so cost is measured before it is planned, and
the cheapest cells come first. Results are summarized after each step. A step whose predictions
fail stops the rest for a decision with the user.

1. **Download** every corpus, and check legal's size.
2. **Prepare every variant:**
   - pools;
   - tokenizers;
   - clustered tokens at each granularity;
   - cuts and gate.

   Check that pool #0 reproduces the recorded merge counts (see "Pools"). The clustered tokens tell
   which readings fit SciPy (prediction 12). With them, the time is estimated, and the estimate is
   shown to the user before the long steps.
3. **Calibrate** on one reading (Wikipedia at 1×, pool #1, 1.50): its time and peak memory set
   the number of workers.
4. **1/16 and ¼** in all five corpora and all three pools: 330 readings, the cheapest.
5. **Pools #1 and #2 at 1×** in all five corpora: 110 readings.
6. **PubMed and legal at 1×, pool #0:** 22 readings, their measured ranges.
7. **4×, pool #0**, in all five corpora: 55 readings, the costliest, from 2.50 down to 1.30.
