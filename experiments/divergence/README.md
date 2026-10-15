# A divergence as the effect size

["A divergence as the effect size"](../../docs/2026-09-28-syntactic-sufficiency-design.md#a-divergence-as-the-effect-size)
sets out the definition and five predictions, written before computing. No model is trained:
the drops come from `experiments/dose_response` and `experiments/subset_selection`, and only the
corpus reading changes.

D(h) = KL(R_h ‖ S_h) in nats. R_h is the reference's distribution of h's fillers, over all its
patterns, counted in distinct contexts as the tool counts them. S_h is the corpus's distribution
with Witten–Bell backoff to the corpus's filler unigram (one added count per filler).

- **Dose:** R is the unplanted training corpus, S each assignment's planted corpus, re-made from
  `dose_response/results/prepare.json` with the whole sample's tokenizer and classes.
- **Subsets:** R is the full pool F, S each subset, with F's tokenizer (11,563 merges) and
  classes, as `diagnose(subset, reference=F)` builds them.

Run from the repository root (a few minutes each, no training):

    poetry run python experiments/divergence/divergence.py dose
    poetry run python experiments/divergence/divergence.py subsets
    poetry run python experiments/divergence/divergence.py summarize

Results go to `results/dose.json`, `results/subsets.json` and `results/summary.json`.

## Result (2026-10-03)

| Prediction | Result |
|---|---|
| 1. D orders planted drops ≥ 0.05 better than diversity | FAILED: −0.90 vs 0.90, −0.84 vs 0.87 |
| 2. drop ≈ slope · D, slope in [−2, −0.5], R² ≥ 0.6 | HELD: slope −0.675, R² 0.77 |
| 3. residual sd on D ≤ 2/3 of that on diversity | FAILED: 0.46 vs 0.52 (ratio 0.88) |
| 4. at most 1 of `self25`'s 7 flagged heads has D > κ | FAILED: κ = 0.44, 5 above |
| 5. Spearman(D, drop) ≤ −0.2 in `r10`, `self25`, beating log trunk | HELD: −0.39 to −0.45 vs +0.13 to +0.17 |

D has the drop's units (slope −0.68) and, on natural cuts, ranks where a model loses, which
diversity did not. On plantings it is no better than diversity. It also has a sample-size
floor from the smoothing: unplanted heads read 0.21–0.27, and in `r10` the median D is 2.5
under 20 occurrences against 0.39 over 1000. κ, set on heads with hundreds of occurrences, is
therefore not a cut, and `self25`'s flagged heads read what they read in `r25`. The next reading
is D in excess of size-matched random draws from the reference.

## Against random cuts of the same size (2026-10-03)

    poetry run python experiments/divergence/divergence.py baseline
    poetry run python experiments/divergence/divergence.py summarize-baseline

`baseline` reads D on 8 random cuts of F at each of `r50`, `r25` and `r10`'s bytes
(`results/baseline.json`). E = D minus the draws' mean at the cut's size. One score per cut is
Σ w_h E(h), with w_h the head's share of F's filler occurrences, and z divides it by the sd of
the draws' leave-one-out scores (`results/baseline_summary.json`).

| Prediction | Result |
|---|---|
| 1. random cuts \|z\| ≤ 2 | HELD: −0.38, −0.42, +0.59 |
| 2. filters z ≥ 3 | HELD: `wiki25` +3.87, `self25` +15.0 |
| 3. score orders r25 < self25 < wiki25, as bits per byte | FAILED: self25 scores worst |
| 4. Spearman(E, drop − drop_r25) ≤ −0.2 in both filters, both seeds | HELD: −0.21, −0.22, −0.28, −0.21 |
| 5. r10, E against D (exploratory) | E 0.02 / −0.04 against D −0.44 / −0.45 |

Against random cuts of the same size, the score separates filters from random cuts. Its
per-head excess points, narrowly, at where a filter's model loses more than a random cut's.
Raw D's ranking on random cuts was sample size alone. The score does not order whole models:
`wiki25` costs more than its slots explain, most likely by topic.

## With the tool's current reading (2026-10-15)

`subsets`, `baseline` and `summarize-baseline` were rerun with F read as `corpus-diagnose` reads
it now: every token classified, a tokenizer learned to 1.50 tokens per word (11,381 merges), and
only syntactic transitions counted as a head's fillers. Heads are the space-prefixed alphabetic
tokens, where the models' drops are read. Draws, weights and predictions are unchanged. The first
readings are in `results/word_like/`; `dose` was not rerun. About 10 minutes with `cut_choice`.

| Prediction | First reading | Current reading |
|---|---|---|
| 1. random cuts \|z\| ≤ 2 | −0.38, −0.42, +0.59 | HELD: −0.15, −0.17, +1.47 |
| 2. filters z ≥ 3 | `wiki25` +3.87, `self25` +15.0 | HELD: `wiki25` +9.65, `self25` +18.6 |
| 3. score orders r25 < self25 < wiki25 | FAILED | FAILED: `self25` still scores worst |
| 4. Spearman(E, drop − drop_r25) ≤ −0.2, both filters and seeds | −0.21 to −0.28 | HELD: −0.24 to −0.32 |
| 5. r10, E against D (exploratory) | E 0.02 / −0.04, D −0.44 / −0.45 | E −0.05 / −0.11, D −0.42 / −0.45 |

The reading over every token separates the Wikipedia-like filter far more clearly (z 9.65
against 3.87), and E points at where the filters' models lose a little more strongly.
