# Choosing among cuts of one size: does the score order the models?

["Choosing among cuts of one size"](../../docs/2026-09-28-syntactic-sufficiency-design.md#choosing-among-cuts-of-one-size-does-the-score-order-the-models)
sets out the design and five predictions, written before running.

Ten cuts of F (FineWeb pool, 4,600 documents, 14.4 MB), all at 25% of its bytes:

- `r25`, `wiki25` and `self25` from `experiments/subset_selection`, with their models.
- Seven new cuts:
  - `r25b`, `r25c`: random, seeds 203 and 204.
  - `selfhigh25`, `wikihigh25`: the highest bigram perplexity, under F and under Wikipedia.
  - `short25`, `long25`: the shortest and the longest documents first.
  - `mix25`: half `self25`'s order, half random.

The score of each cut is that of `experiments/divergence`: D in excess of 8 random cuts of F at
`r25`'s bytes, weighted by F's filler shares. Models are the small model of `subset_selection`,
seeds 0 and 1. The evaluation occurrences are the same, so drops read against its `full`
models.

Run from the repository root:

    poetry run python experiments/cut_choice/cut_choice.py prepare
    poetry run python experiments/cut_choice/cut_choice.py score
    THREADS=3 poetry run python experiments/cut_choice/cut_choice.py train <name> <seed>
    poetry run python experiments/cut_choice/cut_choice.py summarize

Train the seven new cuts on seeds 0 and 1, two runs at a time (about 16 minutes each). Runs go
to `runs/cut_choice/` (gitignored). Cuts, scores, per-run scores and the summary go to
`results/`.

## Result (2026-10-04)

| Prediction | Result |
|---|---|
| 1. Spearman(score, bits per byte) ≥ 0.5, both seeds | FAILED: 0.55 / 0.45 |
| 2. r25 < mix25 < self25 in score and bits per byte | HELD |
| 3. random cuts \|z\| ≤ 2, bits per byte span ≤ 0.02 | HELD: span 0.010 / 0.014 |
| 4. Spearman(E, drop − random drop) ≤ −0.2 on new filters | HELD: −0.23 to −0.38 on all four |
| 5. topic (exploratory) | log documents −0.47 / −0.44; score on the rest 0.50 / 0.49 |

The score reads how far a cut moves F's slots. Prediction 4 replicates on six filters: the
model pays at the heads the score points to. Along one filter's dose, the score orders the
models. Across filters of different kinds, document count, a proxy for topics, matters about
as much. `short25` (3,106 documents) trains the best model despite z 4.2, and `long25` (108)
the worst. The score alone does not pick the best cut, but it carries information that
document count does not.
