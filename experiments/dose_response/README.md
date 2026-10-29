# Dose and response: how far must diversity fall before a model pays?

["Dose and response: how far must diversity fall before a model pays?"](../../docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md#dose-and-response-how-far-must-diversity-fall-before-a-model-pays)
sets out the design and six predictions, written before running. FineWeb's first 6.5 MB sample
in 8 slices; slices 0-6 train and slice 7, never planted, evaluates. The first 21 qualifying
heads from `planted_gap.choose_heads` are planted; the next 5 are comparison heads.

Each head keeps its two most frequent fillers. Other followers are replaced probabilistically,
with deterministic occurrence draws shared by both assignments and all training seeds. Assignment
A cycles through target diversities 0.1-0.7 in trunk order, three heads per level; B shifts the
levels by three. `prepare` jointly bisects each head's replacement probability, reading the
training corpus with the whole sample's tokenizer and classes, for at most 12 iterations per
assignment or until every head is within 0.03 of its target. Missed targets produce warnings.
It saves the probabilities, achieved diversities, trunks, and comparison readings in
`results/prepare.json`. Preparation is long; timestamped progress accompanies each iteration.

The small model and settings are those of `experiments/planted_model`: 2 layers, 128 wide,
3000 steps, learning rate 1e-3, BPE at 11,563 merges learned on each training corpus, seeds 0
and 1. The two controls reuse `runs/planted_model/control_seed*`; `rescore-control` loads those
existing runs and scores this experiment's heads, without training a new control.

Run from the repository root:

    uv run python experiments/dose_response/dose_response.py prepare
    uv run python experiments/dose_response/dose_response.py rescore-control 0
    uv run python experiments/dose_response/dose_response.py rescore-control 1
    THREADS=3 uv run python experiments/dose_response/dose_response.py train <A|B> <seed>
    uv run python experiments/dose_response/dose_response.py summarize

Train A and B for seeds 0 and 1, two runs at a time at most, as in `planted_model` (each run
learns its own tokenizer). Runs go to `runs/dose_response/<assignment>_seed<seed>/`
(gitignored); scores go to `results/<assignment>_seed<seed>.json`, reused control scores to
`results/control_seed<seed>.json`, and the summary to `results/summary.json`.

A head's drop is the mean log probability under the planted model minus the control, in nats,
over all its scored slice-7 occurrences (up to 200). Kept and non-kept followers both count.
The summary checks that the occurrence sequences match and reports per-head drops, evaluation
bits per byte, and HELD/FAILED/unreadable for each prediction:

1. Spearman(achieved diversity, drop) at least 0.6 in all four readings.
2. Mean drop at target 0.1 at most -1.0 nat in all four readings.
3. Mean drop at targets 0.5-0.7 at least -0.3 nats in all four readings.
4. The first crossing of -0.3, scanning down from 0.7 and linearly interpolating the
   four-reading mean drops against target levels, lies in [0.2, 0.5]. No crossing fails.
5. At most one comparison head exceeds an absolute drop of 0.3 per reading.
6. A and B's mean drops differ by at most 0.3 at every level on each seed.

Missing runs or heads without scored occurrences leave dependent predictions unreadable.
An undefined Spearman correlation is unreadable. No threshold is extrapolated outside the
sampled levels; a curve already below -0.3 throughout has no observed crossing.

## Result (2026-10-01)

Calibration hit every target within 0.03 in both assignments (12 bisection steps at most, about
8 seconds each). The comparison heads did not move: "life" 0.854 → 0.859, "really"
1.155 → 1.165, "system" 0.809 → 0.814. Evaluation bits per byte: control 2.092 / 2.089;
`A` 2.121 / 2.122; `B` 2.121 / 2.120 (seed 0 / seed 1).

Mean drop per level in nats, averaged over the four readings (2 heads per level, 2 assignments,
2 seeds):

| Target diversity | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|
| Mean drop | −3.06 | −1.69 | −1.02 | −0.87 | −0.92 | −0.46 | −0.07 |

Against the predictions:

1. **Held.** Spearman(diversity, drop) 0.92 / 0.89 (`A`) and 0.84 / 0.76 (`B`).
2. **Held.** The heads at 0.1 lose 2.78–3.29 nats.
3. **Failed.** The heads at 0.5–0.7 lose 0.38–0.57 nats, not at most 0.3. Only 0.7 is near free.
4. **Failed.** The mean drop crosses −0.3 nats at diversity 0.64, above the predicted
   [0.2, 0.5].
5. **Held.** One comparison reading out of twelve exceeds 0.3 nats ("life", −0.39, 23
   occurrences).
6. **Failed.** At some levels `A` and `B` differ by up to 0.75 nats: with two heads per level,
   which heads sit at a level matters. The same head is read at two doses, though, and its
   drop always follows its dose (for example "only": −1.29 at 0.48, −3.63 / −3.26 at 0.11).

## What this means

The response is strong and monotone, and it begins early. A model pays half a nat or more per
occurrence wherever the planted diversity is 0.6 or below, against these heads' natural
0.8–0.9. That is consistent with the recipe grammar, where the model generalized only from
about 0.8. The tool's 0.5 cut is too permissive for this kind of restriction.

It does not square with `experiments/subset_selection`, where natural flags at 0.26–0.50 cost
nothing measurable. The two restrictions differ in kind:

- **Planting moves probability mass.** At q = 0.6 it sends 60% of a head's other followers to
  one word.
- **The natural cuts remove rare fillers.** The heads they flagged were also naturally low
  (reference diversity 0.50–0.73), and the cut leaves the distribution of the frequent fillers
  close to the domain's.

Diversity counts distinct fillers. The model's loss follows how far the filler distribution has
moved. The two coincide at the strong end and part at the mild end. So the effect size a flag
needs is a distributional one (a divergence between the corpus's and the reference's fillers for
the slot), not a lower diversity cut. That is untested here.
