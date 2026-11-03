# The tokens-per-word range: corpus size, pool draw and knowledge domains

["The tokens-per-word range: corpus size, pool draw and knowledge domains"](../../docs/superpowers/specs/2026-10-28-granularity-robustness-design.md)
sets out the design and twelve predictions, written before running. Corpus side only: no model is
trained.

[`experiments/granularity_domains`](../granularity_domains/README.md) measured where the verdicts
hold, 1.30 to 2.50 tokens per word, on one 14.4 MB pool of each of three domains. This experiment
asks whether that range survives three things it held fixed:

- **size:** 1/16, 1/4, 1x and 4x of 14,370,440 bytes (0.9 to 57.6 MB);
- **draw:** three disjoint pools per corpus at 1/16, 1/4 and 1x (one at 4x);
- **domain:** two knowledge domains added, PubMed abstracts and ECtHR case facts ("legal").

Five corpora × (3 sizes × 3 pools + 4x pool #0) × 11 granularities = 550 readings. 33 of them
(TinyStories, FineWeb and Wikipedia at 1x pool #0) are `granularity_domains`' own; 517 are new.

    uv run python experiments/granularity_robustness/granularity_robustness.py download <corpus>
    uv run python experiments/granularity_robustness/granularity_robustness.py verify <corpora>
    uv run python experiments/granularity_robustness/granularity_robustness.py prepare <variants or step> [--jobs N]
    uv run python experiments/granularity_robustness/granularity_robustness.py costs
    uv run python experiments/granularity_robustness/granularity_robustness.py calibrate
    uv run python experiments/granularity_robustness/granularity_robustness.py run <small|pools|domains|4x>
    uv run python experiments/granularity_robustness/granularity_robustness.py summarize

A variant is `<corpus>@<size>#<pool>`. Pools, merges and cuts go to `runs/granularity_robustness/`
(gitignored). The per-reading `results/<variant>/target_*.json` (9.7 GB in all) are gitignored and
rebuilt by `run`; the downloaded `*_documents.json` are gitignored too. Each variant's
`prepared.json`, `costs.json`, `calibration.json` and the two summaries are kept. `summarize`
writes `results/summary.json` (the spec's rule) and `results/summary_resolvable.json` (the rule
added on 2026-10-30, below).

## Result (2026-11-01)

A granularity holds when (a) every removed class is flagged by a class row and (b) every dropped
word that is one token at that granularity is read `absent`.

**With a 14.4 MB pool or more, the verdicts hold from 1.35 to 1.70 tokens per word in all five
domains**, on 14 of the 15 pools at 1x with unresolvable drops left out (12 of 15 under the
spec's rule). The exception is FineWeb pool #2, which fails 1.35–1.90 under both rules. At 57.6
MB the range is 1.35 to 2.00. Below 14.4 MB the readings are erratic, and Wikipedia and legal hold almost nowhere at 0.9 MB.

1.50 holds:

| Size | Spec's rule | Unresolvable drops left out |
|---|---|---|
| 1/16 (0.9 MB) | 1 of 15 pools | 3 of 15 |
| 1/4 (3.6 MB) | 6 of 15 | 8 of 15 |
| 1x (14.4 MB) | 12 of 15 | 14 of 15 |
| 4x (57.6 MB) | 5 of 5 | 5 of 5 |

The range's edges at 1x and above:

- **1.35 below:** 1.30 fails in FineWeb pool #1, which misses `pain` there under both rules.
  PubMed's merges and Wikipedia pool #2's run out before 1.30, so it is not reachable there.
- **1.70 above:** 1.80 fails in two of legal's three pools at 1x.
- **The upper edge moves with the draw and the size:** Wikipedia's first failure is 2.50 on pools
  #0 and #1 and none on #2; at 4x, Wikipedia and legal widen to 2.50 and 2.20.

**Every reading, per corpus.** ✓ holds under both rules; ◐ fails the spec's rule and holds with
unresolvable drops left out; ✗ fails both; · the granularity is not reachable (the merges run out
first). 1x pool #0 of TinyStories, FineWeb and Wikipedia is `granularity_domains`' reading.

**TinyStories**

| Size | Pool | 1.30 | 1.35 | 1.40 | 1.50 | 1.60 | 1.70 | 1.80 | 1.90 | 2.00 | 2.20 | 2.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1/16 | #0 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1/16 | #1 | ◐ | ◐ | ◐ | ◐ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 1/16 | #2 | ◐ | ◐ | ◐ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1/4 | #0 | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1/4 | #1 | ◐ | ◐ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1/4 | #2 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1x | #1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1x | #2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 4x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |

**FineWeb**

| Size | Pool | 1.30 | 1.35 | 1.40 | 1.50 | 1.60 | 1.70 | 1.80 | 1.90 | 2.00 | 2.20 | 2.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1/16 | #0 | · | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 1/16 | #1 | · | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/16 | #2 | · | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| 1/4 | #0 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ◐ | ✓ | ✓ |
| 1/4 | #1 | ◐ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ◐ | ✗ | ✗ | ✓ |
| 1/4 | #2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1x | #1 | ✗ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ✓ | ✓ | ✓ | ✓ |
| 1x | #2 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| 4x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Wikipedia**

| Size | Pool | 1.30 | 1.35 | 1.40 | 1.50 | 1.60 | 1.70 | 1.80 | 1.90 | 2.00 | 2.20 | 2.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1/16 | #0 | · | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/16 | #1 | · | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/16 | #2 | · | · | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/4 | #0 | ✗ | ✗ | ◐ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/4 | #1 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/4 | #2 | · | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ✗ | ✗ |
| 1x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1x | #1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 1x | #2 | · | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ✓ | ✓ |
| 4x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**PubMed**

| Size | Pool | 1.30 | 1.35 | 1.40 | 1.50 | 1.60 | 1.70 | 1.80 | 1.90 | 2.00 | 2.20 | 2.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1/16 | #0 | · | · | · | ◐ | ◐ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/16 | #1 | · | ◐ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ | ◐ | ✗ | ✓ |
| 1/16 | #2 | · | · | ✗ | ✗ | ✗ | ◐ | ✗ | ◐ | ✗ | ✗ | ✗ |
| 1/4 | #0 | · | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| 1/4 | #1 | · | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/4 | #2 | · | · | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| 1x | #0 | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1x | #1 | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 1x | #2 | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4x | #0 | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Legal (ECtHR)**

| Size | Pool | 1.30 | 1.35 | 1.40 | 1.50 | 1.60 | 1.70 | 1.80 | 1.90 | 2.00 | 2.20 | 2.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1/16 | #0 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/16 | #1 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/16 | #2 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/4 | #0 | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1/4 | #1 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 1/4 | #2 | ✗ | ✗ | ✗ | ◐ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| 1x | #1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 1x | #2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 4x | #0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

**False flags pass everywhere.** Pooled over each variant's granularities, the controls' flag
rate stays under twice the test's level (0.12) for class rows, head rows and `absent`, in all
50 variants.

### Why the small sizes fail

At 1/16 and 1/4 the reference is the small pool itself, and both parts of the rule weaken:

| Size | Readings that fail | A removed class missed | Dropped words only |
|---|---|---|---|
| 1/16 | 134 of 152 | 86 | 48 |
| 1/4 | 109 of 160 | 68 | 41 |
| 1x (new pools) | 36 of 128 | 14 | 22 |
| 4x | 3 of 54 | 3 | 0 |

The classes most often missed are the base verbs (`make`) and the numbers (`two`). Two fixed
settings of the class tree were checked as causes (2026-11-02, one-off scripts not kept):

- **`NOISE_FLOOR`** never binds: the tree's cut picks the same number of classes with and without
  it, on nine pools from 1/16 to 1x at 1.50.
- **`CONTEXT_TOKENS`**, matched by count to the measurement behind it (the 1000th context token's
  463 uses in a 14.4 MB FineWeb pool), recovered `his` in all three TinyStories readings tried,
  and none of the seven Wikipedia and legal misses: there the context fell to 33–178 tokens, and
  three readings lost another class.

So the settings stay, and the tool warns when its domain holds less than 14.4 MB
(`MIN_DOMAIN_BYTES` in `diagnose.py`).

### Dropped words concentrated in few documents

Pool #0's eight dropped words were FineWeb's, chosen by hand in step 1; the other pools draw eight
from the same frequency-rank band. The words that fail are those whose uses sit in few documents,
whatever the granularity. In FineWeb at 1x (case-folded counts):

| Pool | Word | Uses | Documents | Top document | Top three |
|---|---|---|---|---|---|
| #0 | `using` … `whose` (all eight) | 292–1,132 | 4.9–14.6% | 1–4% | 3–9% |
| #1 | `assessment` | 313 | 1.4% | 59% | 65% |
| #1 | `pain` (missed at 1.30) | 310 | 2.0% | 18% | 30% |
| #2 | `retrieved` | 352 | 0.6% | 43% | 68% |
| #1, #2 | the other thirteen | 308–1,122 | 3.3–11.8% | 2–15% | 5–23% |

The `absent` verdict is read against random whole-document draws: when half of a word's uses sit
in one document, the draws that miss it look like its absence. Wikipedia pool #2's `oil` (243 of
304 uses in one article) is the same case.

**The resolvable rule** (added 2026-10-30, while step 4x ran): a dropped word that no granularity
of the same pool reads `absent` is left out of every granularity's verdict and listed in
`unresolved_drops`; a word found at some granularity still counts. `retrieved` is found at 1.30,
so FineWeb pool #2 fails 1.35–1.90 under both rules.

### Against the predictions

Both rules give the same verdicts.

1. **Failed.** 1.50 does not hold at every size: 1 of 15 pools at 1/16, 6 of 15 at 1/4.
2. **Failed.** From 1.30 to 2.00, no 1/16 variant and one 1/4 variant (FineWeb pool #2) agree
   with pool #0 at 1x.
3. **Failed.** Above 2.00, FineWeb and PubMed have no failure at 1x, but most of their 1/16 and 1/4
   pools fail at 2.20: two steps down.
4. **Failed, upward.** At 4x, Wikipedia (1.30–2.50) and legal (1.30–2.20) widen their 1x range;
   TinyStories, FineWeb and PubMed match it.
5. **Held.** Pooled false flags pass in every size variant.
6. **Failed.** 1.50 fails in FineWeb pools #1 and #2 and Wikipedia pool #2 at 1x under the spec's
   rule (in FineWeb pool #2 under both), each on one concentrated dropped word.
7. **Failed.** The first failure above 2.00 moves with the draw (Wikipedia: 2.50, 2.50, none).
8. **Failed.** PubMed reaches 1.50 with 15,066 merges, fewer than Wikipedia's 16,022. Legal's
   3,194 falls between TinyStories and Wikipedia, as predicted.
9. **Held.** All four slots pass the gate in PubMed and legal, with no replacement.
10. **Failed.** Legal at 1x holds 1.30–1.70, not 1.30–2.00, and PubMed cannot reach 1.30.
11. **Held.** Pooled false flags pass in PubMed and legal.
12. **Held.** Every reading fit SciPy's table on the 64 GB machine; none needed
    `linkage_vector`.

### Costs

| Size | Readings | Wall time | Largest parent peak |
|---|---|---|---|
| 1/16 | 165 | 2.4 h | 2.6 GB |
| 1/4 | 165 | 7.4 h | 8.8 GB |
| 1x (new) | 132 | 30.3 h | 38.5 GB |
| 4x | 55 | 47.7 h | 36.6 GB |

On an 18-core, 64 GB machine, with workers sized by measured peaks under a 48 GB ceiling. The
slowest readings were the 4x trees at 1.30–1.35 (FineWeb 4.4 h, Wikipedia 4.7 h, PubMed 4.0 h at
1.35), built on one core.

### Deviations from the spec

- **2026-10-28.** Wikipedia's download reads 10,000 rows, not 4,000: 4,000 articles gave 4.5
  units of text and 4x needs at least 6.
- **2026-10-29.** Step 1/16–1/4 ran to the end although prediction 1 had already failed: the
  plan stops between steps, and the step was cheap.
- **2026-10-30.** Workers were sized by the step's largest parent peak, which left step 1x on one
  worker for hours; they now follow each reading's own projected tree. Results are unaffected.
- **2026-10-30.** The resolvable rule, above, was added and reported beside the spec's rule. The
  first proposal, recomputing each dropped word's detectability, was vacuous: a drop leaves no use
  of the word, so the tool's own verdict already is that test.
- **2026-11-01.** Step 4x's workers still followed the step's largest child peak (12.4 GB), so
  legal at 4x ran on 3 workers where 5–9 fit. Left to finish (about an hour lost); results are
  unaffected.
- **2026-11-02.** The size check above (`NOISE_FLOOR`, `CONTEXT_TOKENS`), asked for by the user
  after the grid.
