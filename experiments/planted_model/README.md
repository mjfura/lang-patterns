# Closing the loop on natural text: the model fails where the indicator flags

"Closing the loop on natural text" in `docs/2026-09-28-syntactic-sufficiency-design.md`, with its
predictions written before this ran. FineWeb's first 6.5 MB sample in 8 slices; slices 0–6 train,
slice 7 (never planted) evaluates. Five heads are chosen as in `experiments/planted_gap` and five
comparison heads by the same rule from the next ranks. Three training corpora: control, and the
replacement restriction planted in all of slices 0–6 with each head kept to its 2 (`k2`) or 16
(`k16`) most frequent fillers. The small model (2 layers, 128 wide), 3000 steps, learning rate
1e-3, BPE at 11,563 merges learned on each corpus, seeds 0 and 1.

    poetry run python experiments/planted_model/planted_model.py prepare
    THREADS=3 poetry run python experiments/planted_model/planted_model.py train <control|k2|k16> <seed>
    poetry run python experiments/planted_model/planted_model.py summarize

`prepare` takes about 20 minutes; each run about 35 minutes, two at a time (six at once ran the
16 GB machine out of memory: each learns its own tokenizer). Runs in `runs/planted_model/`
(gitignored); results in `results/`.

For every occurrence in slice 7 of a head followed by a word (51–110 per head), the model's log
probability of that word given the text before it, read for words outside the kept fillers — what
the gap hid — and inside them; occurrences are joined across models by position.

## Result (2026-09-29)

The indicator on each training corpus (diversity of the planted heads): control 0.80–0.98; `k2`
0.02–0.11; `k16` 0.05–0.16. The comparison heads do not move (0.78–1.11 in all three).

Mean log probability of the word after the head, planted model minus control model, in nats,
for words outside the kept fillers (seed 0 / seed 1):

| Head | `k2` | `k16` |
|---|---|---|
| people | −3.50 / −3.18 | −1.96 / −1.80 |
| she | −2.54 / −2.29 | −1.66 / −2.03 |
| But | −2.33 / −2.17 | −1.12 / −1.64 |
| these | −3.84 / −3.84 | −2.77 / −3.60 |
| only | −4.60 / −4.14 | −3.35 / −3.48 |
| *comparison heads* | −0.21 to +0.14 | −0.51 to +0.18 |

Inside the kept fillers the planted models gain (+0.57 to +1.47 nats with `k2`; −0.44 to +0.21
with `k16`, where the kept set is large). Slice 7 in bits per byte: control 2.092 / 2.089, `k2`
2.122 / 2.123, `k16` 2.121 / 2.117.

Against the predictions:

1. **Held.** Every planted head below 0.5 in both gapped corpora, at least 0.5 in the control.
2. **Held.** With `k2` the true word after a planted head is 2.17–4.60 nats less probable than
   under the control model, on every head and seed — 9 to 100 times less likely.
3. **Held.** With `k16`, 1.12–3.60 nats less, on all five heads and both seeds.
4. **Failed once.** Nineteen of twenty comparison readings stay within 0.3 nats; "now" with `k16`
   on seed 1 reads −0.51 (33 occurrences).
5. **Failed.** The whole-slice loss rises by 0.028–0.035 bits per byte (about 1.5%), above the
   0.02 predicted. The global loss does register the gap — but only against a control trained
   without it, and without saying where; the indicator names the five heads from the corpus
   alone, before training.

What this establishes, for this model and budget: on real text, the heads the indicator flags as
under-shown are exactly where a model trained on that corpus fails to complete unseen text of the
domain — the true word becomes one to two orders of magnitude less likely — while heads it reads
as sufficient behave as in the control. Even a mild restriction (16 fillers kept) costs 1–3.6
nats per occurrence. Still open: one source, one model size, a planted rather than naturally
occurring gap.
