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

## Two dose ladders (2026-10-04)

    poetry run python experiments/cut_choice/cut_choice.py prepare-ladders
    poetry run python experiments/cut_choice/cut_choice.py score
    THREADS=3 poetry run python experiments/cut_choice/cut_choice.py train <name> <seed>
    poetry run python experiments/cut_choice/cut_choice.py summarize-ladders

A filter's order (`self`: lowest own-bigram perplexity; `wikihigh`: highest Wikipedia-bigram
perplexity) is mixed with random documents (seed 205, as `mix25`) at doses 0, 25, 50, 75 and
100%. That gives five new cuts, `self_m25`, `self_m75`, `wikihigh_m25`, `wikihigh_m50` and
`wikihigh_m75`, trained on seeds 0 and 1. `prepare-ladders` checks that the construction
reproduces `mix25`. The summary goes to `results/ladders.json`. Keep the machine on power with
the lid open: runs that sleep stall past the two-hour background limit.

| Prediction | Result |
|---|---|
| 1. score rises strictly with the dose | HELD in both ladders |
| 2. Spearman(score, bits per byte) ≥ 0.9 mean, ≥ 0.7 per seed | `self` 0.90 / 0.90 / 0.90; `wikihigh` 0.60: FAILED |
| 4. per-head Spearman ≤ −0.2 on every new cut with z ≥ 3 | FAILED: holds at doses 50–75, near zero at dose 25 |

With documents held (`self`), the models follow the score from dose 25 on, by 0.017 bits per
byte over the ladder. When the dose also adds documents (`wikihigh`, 1,137 → 1,788), the loss
first falls and then rises: more documents win at low doses, the slot loss from a score of
about 0.05.

## With the tool's current reading (2026-10-15)

`score`, `summarize` and `summarize-ladders` were rerun on `experiments/divergence`'s current
reading (every token, 1.50 tokens per word, syntactic transitions). The models are unchanged. The
first scores and summaries are in `results/word_like/`.

| Prediction | First reading | Current reading |
|---|---|---|
| 1. Spearman(score, bits per byte) ≥ 0.5, both seeds | FAILED: 0.55 / 0.45 | HELD: 0.65 / 0.55 |
| 2. r25 < mix25 < self25 in score and bits per byte | HELD | HELD |
| 3. random cuts \|z\| ≤ 2, bits per byte span ≤ 0.02 | HELD | HELD: z −0.17 to +1.26 |
| 4. Spearman(E, drop − random drop) ≤ −0.2 on new filters | HELD: −0.23 to −0.38 | HELD: −0.24 to −0.37 |
| 5. topic (exploratory) | documents −0.47 / −0.44; rest 0.50 / 0.49 | documents −0.47 / −0.44; rest 0.58 / 0.56 |

Ladders: the score still rises strictly with the dose in both ladders. `self` orders the models
at 0.90 and `wikihigh` at 0.60 (prediction 2 fails as before). The per-head reading is −0.19 to
−0.31 at doses 50–75 and is near zero at dose 25 (−0.01 to −0.14), so prediction 4
fails as before. `short25` (z 4.6) still trains the best model and `long25` (z 15.2) the worst.
