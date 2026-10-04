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
