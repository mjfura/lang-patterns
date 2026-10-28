# Choosing a training subset against the full corpus

["Choosing a training subset: the full corpus as reference"](../../docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md#choosing-a-training-subset-the-full-corpus-as-reference)
records the design and five predictions before running. FineWeb's cached documents are split
as in `natural_gap`: 400 random documents evaluate, and the remaining pool is the full training
corpus and every diagnosis's reference. Random cuts retain 50%, 25%, and 10% of its UTF-8 bytes.
Two further 25% cuts retain the lowest word-bigram perplexities: `wiki25` uses cached Wikipedia,
and `self25` uses the pool itself. Only documents with at least 50 words enter these filtered
cuts. Whole documents are kept until the byte target is reached, including the last document.

The shipped diagnosis reads one- and two-token patterns against the full pool, read once, at
the tool's own granularity (1.50 tokens per word, 11,381 merges) and all other defaults. The
first readings, at 11,563 merges with a 3-sigma margin, are in `results/three_sigma/`. The `planted_model` settings train six models per seed, for seeds
0 and 1. Evaluation compares bits per byte and each one-token head's mean log-probability drop
relative to the full model, using up to 50 occurrences and requiring at least 20 for comparisons.

Run from the repository root, with the FineWeb and Wikipedia caches already available:

    poetry run python experiments/subset_selection/subset_selection.py prepare
    poetry run python experiments/subset_selection/subset_selection.py diagnose <name> [<name> ...]
    THREADS=3 poetry run python experiments/subset_selection/subset_selection.py train <name> <seed>
    poetry run python experiments/subset_selection/subset_selection.py summarize

Diagnose each of `r50`, `r25`, `r10`, `wiki25`, and `self25` before training. Train each of those
and `full` with seeds 0 and 1, two runs at a time. For example:

    THREADS=3 poetry run python experiments/subset_selection/subset_selection.py train full 0 &
    THREADS=3 poetry run python experiments/subset_selection/subset_selection.py train r50 0 &
    wait

Runs go to `runs/subset_selection/`; `results/` is created by the script. `subsets.json` records
indices, document and byte counts, and the filtered cuts' document-mean perplexities against
all eligible documents. Each diagnosis and each model has its own JSON file. `summary.json`
records verdict counts and shares, head readings, per-seed losses and drops, and all five
prediction outcomes. Every complete six-model seed is summarized; predictions 3–5 require both
predeclared seeds, 0 and 1. Missing seeds or undefined statistics are reported as `unreadable`.

Rows read means all rows returned by the diagnosis, including the `unread` verdict, matching
its report summary. A head is read when it has a one-token alphabetic row. Prediction 4 uses
exactly the same flagged and unflagged head sets for the filtered and random models; fewer
than five eligible flagged heads, or no unflagged heads, makes the comparison unreadable.
For prediction 5, the context view is Spearman(drop, log trunk), using the trunk of the row
that supplies the head's minimum z (first row on ties). Constant vectors are unreadable.
Occurrence positions are `[evaluation document index, character offset of head]`, reconstructed
from the scorer's matching rule and checked against its returned sequence before saving.
Tokenization is checked first so unscorable matches cannot silently shift those positions.
Preparation fails if eligible documents cannot reach a filtered cut's byte target.

## Result with the settings from the data (2026-10-15)

The diagnoses were rerun with the tool whose settings all come from the data
(`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`); the models and their drops are unchanged.
Reading the pool and the five subsets took about an hour in one process. The five diagnoses
(50–190 MB each, every pattern read) are gitignored and rebuilt by `diagnose`; `summary.json`
keeps their verdict counts.

| Subset | Tokens/word | Rows read | Under-shown | Absent | Class rows flagged |
|---|---|---|---|---|---|
| `r50` | 1.49 | 272,841 | 0 | 0 | 0 |
| `r25` | 1.49 | 153,009 | 0 | 0 | 0 |
| `r10` | 1.52 | 69,883 | 0 | 0 | 0 |
| `wiki25` | 1.39 | 138,537 | 2 | 11 | 4 |
| `self25` | 1.61 | 160,484 | 1 | 4 | 1 |

Rows read now counts every pattern the diagnosis met, most of them `unread` (too thin for the
bar); the old diagnosis returned some 50 times fewer rows, so shares are not comparable, while
counts of flags are.

1. **Held, more strongly.** Random cuts flag nothing at all, rows or classes, where the 3-sigma
   margin flagged 2–8 rows each.
2. **Failed for both.** The filtered cuts are the only ones flagged, but by far fewer rows than
   the predeclared bar of 10 under-shown: `wiki25` flags "," and ":" before determiners and
   labels, `self25` flags ". Our". Their absent rows are a few thin heads ("50", " a day",
   " they should", ", perhaps"). Wikipedia-like filtering shifts `wiki25`'s granularity to 1.39
   tokens per word, inside the measured range but the farthest from the pool.
3. **Held** (models unchanged).
4. **Unreadable for both**: no one-token alphabetic head is flagged, so there is no flagged set.
5. **Held.** Spearman(z, drop) in `r10` is 0.157 / 0.181 over 416 heads, above 0.15, with log
   trunk alone at 0.037 / 0.018: the z of the new bar carries what the trunk does not. Under the
   3-sigma margin it was 0.10 / 0.12 with trunk at 0.10 / 0.11.

So the derived bar removes the chance flags of the 3-sigma margin and keeps the filtered cuts as
the only flagged ones, while the mild restrictions it no longer flags were the ones that did
not cost the model (prediction 4 below). Its z ranks heads by what a cut costs the model better
than their frequency does.

## Result (2026-10-01)

Diagnoses against the full pool, about 4 minutes each:

| Subset | Rows read | Under-shown | Share |
|---|---|---|---|
| `r50` | 5,251 | 8 | 0.15% |
| `r25` | 2,619 | 4 | 0.15% |
| `r10` | 961 | 2 | 0.21% |
| `wiki25` | 3,091 | 9 | 0.29% |
| `self25` | 2,244 | 12 | 0.53% |

Evaluation bits per byte (seed 0 / seed 1): full 2.043 / 2.042, `r50` 2.071 / 2.067, `r25`
2.128 / 2.130, `r10` 2.332 / 2.336, `wiki25` 2.181 / 2.181, `self25` 2.146 / 2.145.

Against the predictions:

1. **Held.** Random cuts flag 0.15–0.21% of rows, the rate a 3-sigma margin gives by chance.
2. **Held for `self25`, failed for `wiki25`.** `self25` flags 12 rows (the bar was 12): the
   prepositions "in", "on", "from", tied to "the" and "a", and "some", "make", "free", "hours",
   "If you", "I had", "will not", "in which", "due to". `wiki25` flags 9. The flags are mild:
   diversity 0.26–0.50 against margins of 0.42–0.76, where the planted gaps read 0.02–0.16.
3. **Held.** Full < `r50` < `r25` < `r10` on both seeds, and both filtered cuts above `r25`
   (`self25` +0.016, `wiki25` +0.052 bits per byte).
4. **Failed for `self25`; unreadable for `wiki25`** (2 flagged heads with 20 evaluation
   occurrences). The `self25` model does not lose more at its 7 flagged heads than at the
   other 394: gap(`self25`) − gap(`r25`) = +0.10 / +0.17 nats, the wrong sign. Only "on" loses
   clearly more under `self25` than under `r25` (−0.84 / −0.98 against −0.16 / −0.53).
5. **Failed.** In `r10`, Spearman(z, drop) is 0.10 / 0.12 over 306 heads, below 0.15. The sign
   is the predicted one on both seeds, but log trunk alone correlates as much (0.10 / 0.11).
   Exploratory, read after the outcome: controlling for log trunk, the partial rank correlation
   is 0.12 / 0.14.

## What this means

The diagnosis behaves as a test should on real cuts of a real corpus. It raises no more alarms
on random cuts than chance allows, and somewhat more on a filter that keeps formulaic text.
But the slots it flags in a natural cut are restricted mildly. On this model and evaluation
text, a mild restriction does not show up as a measurable loss at those heads. A head's drop
over 50 occurrences moves by 0.2–0.5 nats between seeds, larger than any effect a diversity of
0.4 instead of 0.6 could produce. The planted experiments needed diversity near 0.1 for losses
of 1–4 nats.

So a 3-sigma flag is statistically significant without being practically large. For choosing
data, the flag needs an effect size on top of its significance: how far the diversity falls,
not only that it falls beyond noise. This bounds the claim that a subset's diagnosis says where
a cut will cost a model: it holds for strong restrictions (planted), and was not shown for the
mild ones that random and perplexity-filtered cuts of FineWeb produce. Still open: one source,
one model size, and no natural cut produced a strong restriction.
